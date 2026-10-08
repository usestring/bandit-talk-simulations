"""Thompson sampling scored on value, not success alone.

Synthetic only; no network requests. Three configurations with a realistic
spread: cheap methods fail far more often than expensive ones.

Every Thompson policy keeps the same Beta(1,1) success estimates and draws a
success rate per configuration. Only the score used to pick among the draws
changes:
  success only     draw
  value            V * draw - cost
  performance      M * V * draw - cost   (successes weighted M times)
Each attempt takes the same time, so time to a success is LATENCY / success
rate; that is a reported measure, not a score (as a score it would rank the
configurations exactly as "success only" does).
"""
import argparse
import json
import random
from pathlib import Path

import bandit_environments as be

ARMS = ["Stealth browser", "TLS client", "Plain HTTP"]
BASE = [0.90, 0.45, 0.15]
COST = [4.0, 1.0, 0.25]
VALUE = 5.0
MULTIPLIER = 5
LATENCY = 5.0
SETTINGS = [
    ("round_robin", "Round-robin", {}),
    ("thompson", "Thompson, success only", {}),
    ("value", f"Thompson, value (success worth ${VALUE:g})", {"weight": VALUE}),
    ("value", f"Thompson, performance ({MULTIPLIER}x success weight)", {"weight": MULTIPLIER * VALUE}),
]
COLORS = ["#9aa0a6", "#a142f4", "#1a73e8", "#188038"]


def simulate(kind, name, params, n, seed, window):
    outcomes, choices = random.Random(seed), random.Random(seed + 1_000_003)
    alpha, beta, usage = [1.0] * 3, [1.0] * 3, [1 / 3] * 3
    wins, spend, picks = 0, 0.0, [0] * 3
    recent, recent_cost, roll, roll_cost = [], [], [], []
    for t in range(n):
        if name == "round_robin":
            arm = t % 3
        else:
            draws = [choices.betavariate(alpha[a], beta[a]) for a in range(3)]
            score = draws if name == "thompson" else [params["weight"] * d - c for d, c in zip(draws, COST)]
            arm = max(range(3), key=score.__getitem__)
        rates = BASE if kind == "stationary" else [b * (1 - be.ADV_STRENGTH * u) for b, u in zip(BASE, usage)]
        ok = int(outcomes.random() < rates[arm])
        usage = [u * (1 - be.ADV_MEMORY) + be.ADV_MEMORY * (a == arm) for a, u in enumerate(usage)]
        alpha[arm] += ok
        beta[arm] += 1 - ok
        wins += ok
        spend += COST[arm]
        picks[arm] += 1
        recent = (recent + [ok])[-window:]
        recent_cost = (recent_cost + [COST[arm]])[-window:]
        roll.append(sum(recent) / len(recent))
        roll_cost.append(sum(recent_cost) / len(recent))
    return wins, spend, picks, roll, roll_cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "value-scoring")
    parser.add_argument("--gif", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"simulation": True, "requests": args.requests, "seeds": args.seeds, "arms": ARMS,
               "base_rates": BASE, "costs": COST, "success_value": VALUE, "multiplier": MULTIPLIER,
               "latency_per_attempt_s": LATENCY, "strength": be.ADV_STRENGTH,
               "usage_memory": be.ADV_MEMORY, "environments": {}}
    for kind in ("stationary", "adversarial"):
        rows, panels = [], [[], []]
        for (name, label, params), color in zip(SETTINGS, COLORS):
            wins = spend = 0
            picks, succ, cost = [0] * 3, [0.0] * args.requests, [0.0] * args.requests
            for seed in range(args.seeds):
                w, s, p, roll, roll_cost = simulate(kind, name, params, args.requests, seed, args.window)
                wins, spend = wins + w, spend + s
                picks = [a + b for a, b in zip(picks, p)]
                succ = [a + b / args.seeds for a, b in zip(succ, roll)]
                cost = [a + b / args.seeds for a, b in zip(cost, roll_cost)]
            total = args.requests * args.seeds
            rate = wins / total
            row = {"policy": name, "label": label, "params": params, "success_rate": rate,
                   "seconds_per_success": LATENCY / rate, "attempts_per_success": 1 / rate,
                   "cost_per_success": spend / wins, "cost_per_request": spend / total,
                   "value_per_request": (VALUE * wins - spend) / total,
                   "traffic_share": dict(zip(ARMS, (x / total for x in picks)))}
            rows.append(row)
            panels[0].append((label, color, [LATENCY / max(r, 1e-3) for r in succ]))
            panels[1].append((label, color, [c / max(r, 1e-3) for c, r in zip(cost, succ)]))
            print(f"{kind:11s} {label:42s} {rate:6.1%} {LATENCY / rate:5.1f}s ${spend / wins:5.2f}/success "
                  + " ".join(f"{a[:5]}:{v:4.0%}" for a, v in row["traffic_share"].items()))
        summary["environments"][kind] = rows
        if args.gif:
            extras = [{}, {}]
            if kind == "adversarial":
                # The same policies on the same site when it does not react, so the gap is what detection costs.
                calm = summary["environments"]["stationary"]
                extras = [{"baselines": [r[key] for r in calm],
                           "baseline_label": "Dashed: same method, site not reacting"}
                          for key in ("seconds_per_success", "cost_per_success")]
            title = "Stable site" if kind == "stationary" else "Reactive site"
            be.animate_lines(args.output_dir / f"{kind}.gif", [
                {"title": f"{title}: time to a success ({LATENCY:g} s per attempt)",
                 "ylabel": f"Seconds per success (rolling {args.window})", "ylim": (0, 30),
                 "series": panels[0], **extras[0]},
                {"title": "Cost per success (illustrative: 4 / 1 / 0.25)",
                 "ylabel": f"Cost per success (rolling {args.window})", "ylim": (0, 12),
                 "series": panels[1], **extras[1]},
            ], args.requests)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
