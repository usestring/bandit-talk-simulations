"""Offline comparison of request-configuration selection methods.

Every environment is synthetic. No network requests are made. The simulation
core uses only the standard library; --plot and --gif need matplotlib and
pillow.

Three-arm environments (arms A, B, C; base success rates 0.90 / 0.75 / 0.60):
  stationary  fixed success probabilities
  shift       A drops abruptly from 0.90 to 0.40 halfway through
  drift       A decays and C improves gradually over the middle half
  adversarial each arm's success probability falls with its recent share of
              traffic and recovers when it rests (a crude "the site learns
              your most-used config" model, not a real anti-bot system)

Knob environment (36 arms = 3 fetch methods x 4 proxies x 3 header sets):
  The pruned policy drops plain HTTP, datacenter proxies and mismatched
  headers up front, a heuristic a practitioner would apply, leaving 12 arms.
  knobs       success is a logistic function of additive knob effects;
              halfway through, the site starts flagging the TLS-impersonating
              client, so the best combination changes
"""
import argparse
import itertools
import json
import math
import random
from pathlib import Path

BASE = [0.90, 0.75, 0.60]
ADV_STRENGTH = 0.6
ADV_MEMORY = 0.01

FETCH = {"http": -1.5, "tls": 1.2, "browser": 0.6}
PROXY = {"datacenter": -1.0, "resi-us": 1.0, "resi-eu": 0.3, "mobile": 1.2}
HEADERS = {"minimal": -0.5, "consistent": 0.8, "mismatched": -1.5}
FETCH_COST = {"http": 1, "tls": 2, "browser": 20}
PROXY_COST = {"datacenter": 1, "resi-us": 8, "resi-eu": 8, "mobile": 25}
KNOB_INTERCEPT = -0.5
TLS_FLAG_PENALTY = 2.5
KNOB_ARMS = list(itertools.product(FETCH, PROXY, HEADERS))
PRUNED_ARMS = [i for i, (fetch, proxy, headers) in enumerate(KNOB_ARMS)
               if fetch != "http" and proxy != "datacenter" and headers != "mismatched"]

ENVIRONMENTS = ["stationary", "shift", "drift", "adversarial", "knobs"]
TITLES = {
    "stationary": "Stationary: fixed success rates (3 configs)",
    "shift": "Abrupt shift: A drops 0.90 -> 0.40 at the midpoint (3 configs)",
    "drift": "Gradual drift: A decays, C improves (3 configs)",
    "adversarial": "Adversarial: heavily used configs get blocked more (3 configs)",
    "knobs": "36 knob combinations: TLS client gets flagged at the midpoint",
}
POLICIES = ["round_robin", "grid_search", "grid_retune", "random_search",
            "epsilon_greedy", "ucb1", "exp3", "thompson", "thompson_decay", "thompson_pruned"]
CHART_POLICIES = ["round_robin", "grid_search", "grid_retune", "random_search",
                  "ucb1", "exp3", "thompson_decay", "thompson_pruned"]
LABELS = {
    "round_robin": "Round-robin",
    "grid_search": "Grid search, then commit",
    "grid_retune": "Grid search, re-run every 1,000",
    "random_search": "Random search, then commit",
    "epsilon_greedy": "Epsilon-greedy (0.1)",
    "ucb1": "UCB1",
    "exp3": "EXP3 (gamma 0.1)",
    "thompson": "Thompson (no forgetting)",
    "thompson_decay": "Thompson + forgetting (0.995)",
    "thompson_pruned": f"Thompson + forgetting, pruned to {len(PRUNED_ARMS)} arms",
}
COLORS = {"round_robin": "#9aa0a6", "grid_search": "#d93025", "grid_retune": "#f28b82",
          "random_search": "#e37400", "epsilon_greedy": "#f9ab00", "ucb1": "#1a73e8",
          "exp3": "#00897b", "thompson": "#a142f4", "thompson_decay": "#188038",
          "thompson_pruned": "#0b5394"}


class Environment:
    def __init__(self, kind, n):
        self.kind, self.n = kind, n
        self.k = len(KNOB_ARMS) if kind == "knobs" else 3
        self.usage = [1 / 3] * 3

    def rates(self, t):
        if self.kind == "stationary":
            return list(BASE)
        if self.kind == "shift":
            return list(BASE) if t < self.n // 2 else [0.40, 0.75, 0.60]
        if self.kind == "drift":
            start, end = self.n // 4, 3 * self.n // 4
            x = min(1.0, max(0.0, (t - start) / (end - start)))
            return [0.90 - 0.50 * x, 0.75, 0.60 + 0.25 * x]
        if self.kind == "adversarial":
            return [b * (1 - ADV_STRENGTH * u) for b, u in zip(BASE, self.usage)]
        if self.kind == "knobs":
            flagged = t >= self.n // 2
            out = []
            for fetch, proxy, headers in KNOB_ARMS:
                logit = KNOB_INTERCEPT + FETCH[fetch] + PROXY[proxy] + HEADERS[headers]
                if flagged and fetch == "tls":
                    logit -= TLS_FLAG_PENALTY
                out.append(1 / (1 + math.exp(-logit)))
            return out
        raise ValueError(f"Unknown environment: {self.kind}")

    def observe(self, arm):
        if self.kind == "adversarial":
            self.usage = [u * (1 - ADV_MEMORY) + ADV_MEMORY * (a == arm)
                          for a, u in enumerate(self.usage)]

    def cost(self, arm):
        if self.kind != "knobs":
            return 1.0
        fetch, proxy, _ = KNOB_ARMS[arm]
        return FETCH_COST[fetch] + PROXY_COST[proxy]

    def group(self, arm):
        return KNOB_ARMS[arm][0] if self.kind == "knobs" else "ABC"[arm]

    def groups(self):
        return list(FETCH) if self.kind == "knobs" else list("ABC")


class Policy:
    def __init__(self, name, k, n, rng, epsilon=0.1, decay=0.995, gamma=0.1):
        self.name, self.k, self.n, self.rng = name, k, n, rng
        self.epsilon, self.decay, self.gamma = epsilon, decay, gamma
        self.counts, self.rewards = [0] * k, [0.0] * k
        self.alpha, self.beta = [1.0] * k, [1.0] * k
        self.log_w = [0.0] * k
        self.probs = [1 / k] * k
        self.trials = 30 if k <= 3 else 10
        self.candidates = list(range(k))
        if name == "random_search":
            self.candidates = rng.sample(range(k), min(k, 9)) if k > 3 else list(range(k))
        self.allowed = PRUNED_ARMS if name == "thompson_pruned" else range(k)
        self.phase_start, self.committed = 0, None
        self.phase_counts, self.phase_rewards = [0] * k, [0.0] * k

    def select(self, t):
        if self.name == "round_robin":
            return t % self.k
        if self.name in ("grid_search", "grid_retune", "random_search"):
            if self.name == "grid_retune" and t > 0 and t % 1000 == 0:
                self.phase_start, self.committed = t, None
                self.phase_counts, self.phase_rewards = [0] * self.k, [0.0] * self.k
            step = t - self.phase_start
            budget = self.trials * len(self.candidates)
            if step < budget:
                return self.candidates[step % len(self.candidates)]
            if self.committed is None:
                self.committed = max(self.candidates, key=lambda a:
                                     self.phase_rewards[a] / max(1, self.phase_counts[a]))
            return self.committed
        if self.name == "epsilon_greedy":
            if t < self.k:
                return t
            if self.rng.random() < self.epsilon:
                return self.rng.randrange(self.k)
            return max(range(self.k), key=lambda a: self.rewards[a] / self.counts[a])
        if self.name == "ucb1":
            if t < self.k:
                return t
            return max(range(self.k), key=lambda a: self.rewards[a] / self.counts[a]
                       + math.sqrt(2 * math.log(t) / self.counts[a]))
        if self.name == "exp3":
            top = max(self.log_w)
            w = [math.exp(x - top) for x in self.log_w]
            total = sum(w)
            self.probs = [(1 - self.gamma) * x / total + self.gamma / self.k for x in w]
            return self.rng.choices(range(self.k), weights=self.probs)[0]
        if self.name in ("thompson", "thompson_decay", "thompson_pruned"):
            samples = {a: self.rng.betavariate(self.alpha[a], self.beta[a]) for a in self.allowed}
            return max(samples, key=samples.get)
        raise ValueError(f"Unknown policy: {self.name}")

    def update(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.phase_counts[arm] += 1
        self.phase_rewards[arm] += reward
        if self.name == "exp3":
            self.log_w[arm] += self.gamma * (reward / self.probs[arm]) / self.k
        if self.name in ("thompson_decay", "thompson_pruned"):
            self.alpha = [1 + (a - 1) * self.decay for a in self.alpha]
            self.beta = [1 + (b - 1) * self.decay for b in self.beta]
        if self.name in ("thompson", "thompson_decay", "thompson_pruned"):
            self.alpha[arm] += reward
            self.beta[arm] += 1 - reward


def simulate(policy_name, kind, n, seed):
    outcomes, choices = random.Random(seed), random.Random(seed + 1_000_003)
    env = Environment(kind, n)
    policy = Policy(policy_name, env.k, n, choices)
    arms, wins, cost = [], [], 0.0
    for t in range(n):
        arm = policy.select(t)
        rates = env.rates(t)
        reward = int(outcomes.random() < rates[arm])
        env.observe(arm)
        policy.update(arm, reward)
        cost += env.cost(arm)
        arms.append(arm)
        wins.append(reward)
    return env, arms, wins, cost


def rolling(values, window):
    out, total = [], 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def applicable(policy, kind):
    return policy not in ("random_search", "thompson_pruned") or kind == "knobs"


def run(n, seeds, window):
    results = {}
    for kind in ENVIRONMENTS:
        env_result = {}
        for policy in POLICIES:
            if not applicable(policy, kind):
                continue
            mean_roll, share, rates, costs = [0.0] * n, None, [], []
            for seed in seeds:
                env, arms, wins, cost = simulate(policy, kind, n, seed)
                groups = env.groups()
                if share is None:
                    share = [[0.0] * len(groups) for _ in range(n)]
                rates.append(sum(wins) / n)
                costs.append(cost / max(1, sum(wins)))
                for i, r in enumerate(rolling(wins, window)):
                    mean_roll[i] += r / len(seeds)
                for g, name in enumerate(groups):
                    for i, r in enumerate(rolling([int(env.group(a) == name) for a in arms], window)):
                        share[i][g] += r / len(seeds)
            env_result[policy] = {
                "mean_success_rate": sum(rates) / len(rates),
                "min_success_rate": min(rates),
                "max_success_rate": max(rates),
                "mean_cost_per_success": sum(costs) / len(costs),
                "rolling_success": mean_roll,
                "rolling_group_share": share,
            }
        if kind != "adversarial":
            oracle = Environment(kind, n)
            env_result["best_arm_rate"] = [max(oracle.rates(t)) for t in range(n)]
        results[kind] = env_result
    return results


def chart_policies(kind):
    return [p for p in CHART_POLICIES if applicable(p, kind)]


def plot(results, n, window, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    for kind in ENVIRONMENTS:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        if "best_arm_rate" in results[kind]:
            ax.plot(results[kind]["best_arm_rate"], color="black", ls="--", lw=1,
                    label="Best config (oracle)")
        for p in chart_policies(kind):
            r = results[kind][p]
            ax.plot(r["rolling_success"], color=COLORS[p], lw=2,
                    label=f"{LABELS[p]}  {r['mean_success_rate']:.3f}")
        ax.set_title(TITLES[kind])
        ax.set_xlabel("Request")
        ax.set_ylabel(f"Success rate (rolling {window}, mean of seeds)")
        ax.set_ylim(0.2, 1.0)
        ax.set_xlim(0, n)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=7, title="Method  overall success",
                  title_fontsize=8)
        fig.text(0.99, 0.01, "Synthetic simulation, not production data",
                 ha="right", fontsize=7, color="#5f6368")
        fig.tight_layout()
        path = out_dir / f"{kind}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def animate(results, n, window, out_dir, kind, frames=60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=100,
                                 gridspec_kw={"width_ratios": [2, 1]})
    policies = chart_policies(kind)
    lines = {p: ax.plot([], [], color=COLORS[p], lw=2, label=LABELS[p])[0] for p in policies}
    if "best_arm_rate" in results[kind]:
        ax.plot(results[kind]["best_arm_rate"], color="black", ls="--", lw=1,
                label="Best config (oracle)")
    ax.set_xlim(0, n)
    ax.set_ylim(0.2, 1.0)
    ax.set_title(TITLES[kind], fontsize=9)
    ax.set_xlabel("Request")
    ax.set_ylabel(f"Success rate (rolling {window})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=6)
    groups = Environment(kind, n).groups()
    bars = bx.bar(groups, [0] * len(groups), color=["#d93025", "#1a73e8", "#188038"])
    bx.set_ylim(0, 1)
    bx.set_title("Thompson + forgetting: traffic share" +
                 (" by fetch method" if kind == "knobs" else ""), fontsize=9)
    fig.text(0.99, 0.01, "Synthetic simulation, not production data",
             ha="right", fontsize=7, color="#5f6368")
    fig.tight_layout()
    share = results[kind]["thompson_decay"]["rolling_group_share"]

    def update(frame):
        end = max(1, int(n * (frame + 1) / frames))
        for p in policies:
            lines[p].set_data(range(end), results[kind][p]["rolling_success"][:end])
        for bar, value in zip(bars, share[end - 1]):
            bar.set_height(value)
        return list(lines.values()) + list(bars)

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    path = out_dir / f"{kind}.gif"
    anim.save(path, writer=PillowWriter(fps=12))
    plt.close(fig)
    hold_last_frame(path)
    return path


def hold_last_frame(path, seconds=10):
    """Rewrite the final frame's delay so a looping GIF rests on the finished chart."""
    data = bytearray(path.read_bytes())
    pos, last = 13, None
    if data[10] & 0x80:
        pos += 3 << ((data[10] & 7) + 1)
    while data[pos] != 0x3B:
        if data[pos] == 0x21:
            if data[pos + 1] == 0xF9:
                last = pos + 4
            pos += 2
        else:
            flags = data[pos + 9]
            pos += 10 + ((3 << ((flags & 7) + 1)) if flags & 0x80 else 0) + 1
        while data[pos]:
            pos += data[pos] + 1
        pos += 1
    data[last:last + 2] = (seconds * 100).to_bytes(2, "little")
    path.write_bytes(bytes(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "environments")
    parser.add_argument("--plot", action="store_true", help="write a PNG per environment")
    parser.add_argument("--gif", action="store_true", help="write an animated GIF per environment")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run(args.requests, range(args.seeds), args.window)
    print(f"SIMULATION requests={args.requests} seeds=0-{args.seeds - 1} (mean success rate)")
    print(f"{'method':32s}" + "".join(f"{k:>13s}" for k in ENVIRONMENTS) + f"{'knobs cost/ok':>15s}")
    for p in POLICIES:
        cells = "".join(f"{results[k][p]['mean_success_rate']:13.3f}" if p in results[k]
                        else f"{'n/a':>13s}" for k in ENVIRONMENTS)
        print(f"{LABELS[p]:32s}{cells}{results['knobs'][p]['mean_cost_per_success']:15.1f}")
    summary = {
        "simulation": True, "requests": args.requests, "seeds": args.seeds,
        "rolling_window": args.window, "base_rates": BASE,
        "adversarial": {"strength": ADV_STRENGTH, "usage_memory": ADV_MEMORY},
        "knobs": {"intercept": KNOB_INTERCEPT, "fetch": FETCH, "proxy": PROXY,
                  "headers": HEADERS, "tls_flag_penalty": TLS_FLAG_PENALTY,
                  "fetch_cost": FETCH_COST, "proxy_cost": PROXY_COST},
        "results": {k: {p: {m: v for m, v in results[k][p].items()
                            if not m.startswith("rolling")}
                        for p in POLICIES if p in results[k]}
                    for k in ENVIRONMENTS},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.plot:
        for path in plot(results, args.requests, args.window, args.output_dir):
            print(f"Wrote {path}")
    if args.gif:
        for kind in ENVIRONMENTS:
            print(f"Wrote {animate(results, args.requests, args.window, args.output_dir, kind)}")


if __name__ == "__main__":
    main()
