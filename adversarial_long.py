"""Long reactive-site simulation with costs and a hyperparameter sweep.

Synthetic only; no network requests. Reuses the reactive ("adversarial")
environment from bandit_environments.py: each configuration's success rate
falls with its recent share of traffic and recovers when it rests. Adds
illustrative per-request costs (A, the best when rested, is the most expensive)
and runs long enough for periodic re-tests to repeat many times.
"""
import argparse
import json
import random
from pathlib import Path

import bandit_environments as be

COST = [4.0, 2.0, 1.0]
SETTINGS = (
    [("round_robin", "Round-robin", {})]
    + [("grid_search", f"Explore then commit ({t}/arm)", {"trials": t}) for t in (10, 30, 100)]
    + [("grid_retune", f"Periodic re-test every {i:,}", {"interval": i}) for i in (100, 250, 1000, 5000)]
    + [("grid_threshold", f"Commit, re-search below {b:.0%}", {"threshold": b}) for b in (0.5, 0.6, 0.7)]
    + [("epsilon_greedy", f"Epsilon-greedy {e}", {"epsilon": e}) for e in (0.05, 0.1, 0.2)]
    + [("eg_decay", f"Epsilon-greedy 0.1, forgetting {d}", {"decay": d}) for d in (0.98, 0.99, 0.995)]
    + [("ucb1", "UCB1", {})]
    + [("ucb_decay", f"UCB, forgetting {d}", {"decay": d}) for d in (0.98, 0.99, 0.995)]
    + [("exp3", f"EXP3 gamma {g}", {"gamma": g}) for g in (0.05, 0.1, 0.2)]
    + [("thompson", "Thompson, no forgetting", {})]
    + [("thompson_decay", f"Thompson, forgetting {d}", {"decay": d}) for d in (0.9, 0.95, 0.98, 0.99, 0.995, 0.999)]
)


class Policy(be.Policy):
    def __init__(self, name, k, n, rng, trials=30, interval=1000, threshold=0.6, block_window=50, **kw):
        super().__init__(name, k, n, rng, **kw)
        self.trials, self.interval = trials, interval
        self.threshold, self.block_window, self.recent = threshold, block_window, []
        self.w_counts, self.w_rewards = [0.0] * k, [0.0] * k

    def select(self, t):
        if self.name in ("eg_decay", "ucb_decay"):
            if t < self.k:
                return t
            means = [self.w_rewards[a] / max(1e-9, self.w_counts[a]) for a in range(self.k)]
            if self.name == "eg_decay":
                if self.rng.random() < self.epsilon:
                    return self.rng.randrange(self.k)
                return max(range(self.k), key=lambda a: means[a])
            total = sum(self.w_counts)
            return max(range(self.k), key=lambda a: means[a]
                       + be.math.sqrt(2 * be.math.log(max(total, 1.0001)) / max(1e-9, self.w_counts[a])))
        if self.name == "grid_threshold" and (
                self.committed is not None and len(self.recent) == self.block_window
                and sum(self.recent) / self.block_window < self.threshold):
            self._restart_search(t)
        if self.name == "grid_retune" and t > 0 and t % self.interval == 0:
            self._restart_search(t)
        if self.name in ("grid_threshold", "grid_retune"):
            name, self.name = self.name, "grid_search"
            arm = super().select(t)
            self.name = name
            return arm
        return super().select(t)

    def _restart_search(self, t):
        self.phase_start, self.committed, self.recent = t, None, []
        self.phase_counts, self.phase_rewards = [0] * self.k, [0.0] * self.k

    def update(self, arm, reward):
        super().update(arm, reward)
        if self.name in ("eg_decay", "ucb_decay"):
            self.w_counts = [c * self.decay for c in self.w_counts]
            self.w_rewards = [r * self.decay for r in self.w_rewards]
            self.w_counts[arm] += 1
            self.w_rewards[arm] += reward
        if self.name == "grid_threshold" and self.committed is not None:
            self.recent = (self.recent + [reward])[-self.block_window:]


def simulate(name, params, n, seed, window):
    outcomes, choices = random.Random(seed), random.Random(seed + 1_000_003)
    env = be.Environment("adversarial", n)
    policy = Policy(name, 3, n, choices, **params)
    wins, cost, roll = 0, 0.0, []
    recent = []
    for t in range(n):
        arm = policy.select(t)
        reward = int(outcomes.random() < env.rates(t)[arm])
        env.observe(arm)
        policy.update(arm, reward)
        wins += reward
        cost += COST[arm]
        recent.append(reward)
        if len(recent) > window:
            recent.pop(0)
        roll.append(sum(recent) / len(recent))
    return wins, cost, roll


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=20000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--window", type=int, default=500)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "adversarial-long")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for name, label, params in SETTINGS:
        rates, per_ok, per_req, curve = [], [], [], [0.0] * args.requests
        for seed in range(args.seeds):
            wins, cost, roll = simulate(name, params, args.requests, seed, args.window)
            rates.append(wins / args.requests)
            per_ok.append(cost / max(1, wins))
            per_req.append(cost / args.requests)
            curve = [c + r / args.seeds for c, r in zip(curve, roll)]
        row = {"policy": name, "label": label, "params": params,
               "mean_success_rate": sum(rates) / len(rates),
               "min_success_rate": min(rates), "max_success_rate": max(rates),
               "cost_per_success": sum(per_ok) / len(per_ok),
               "cost_per_request": sum(per_req) / len(per_req)}
        rows.append(row)
        curves[label] = curve
        print(f"{label:34s} {row['mean_success_rate']:6.1%} {row['cost_per_success']:6.2f} {row['cost_per_request']:6.2f}")
    summary = {"simulation": True, "environment": "adversarial", "requests": args.requests,
               "seeds": args.seeds, "costs": COST, "base_rates": be.BASE,
               "strength": be.ADV_STRENGTH, "usage_memory": be.ADV_MEMORY, "results": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        shown = ["Round-robin", "Explore then commit (30/arm)", "Periodic re-test every 1,000",
                 "Periodic re-test every 100", "Thompson, no forgetting", "Thompson, forgetting 0.99"]
        fig, ax = plt.subplots(figsize=(12, 6))
        for label in shown:
            ax.plot(curves[label], label=label, linewidth=2)
        ax.set_xlabel("Request")
        ax.set_ylabel(f"Success rate (rolling {args.window})")
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(args.output_dir / "rolling.png", dpi=150)


if __name__ == "__main__":
    main()
