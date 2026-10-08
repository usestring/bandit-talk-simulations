"""Offline proxy-pool simulation. Run with Python 3, without dependencies."""
import argparse
import csv
import json
import random
from pathlib import Path


def rates_at(t, shift_at):
    return [0.9, 0.75, 0.6] if t < shift_at else [0.4, 0.75, 0.6]


def simulate(policy, n=2000, seed=7, epsilon=0.1, decay=0.995):
    outcomes = random.Random(seed)
    choices = random.Random(seed)
    counts, rewards = [0] * 3, [0] * 3
    alpha, beta = [1.0] * 3, [1.0] * 3
    successes = 0
    regret = 0.0
    trace = []
    shift_at = n // 2
    for t in range(n):
        if policy == "round_robin":
            arm = t % 3
        elif policy == "epsilon_greedy":
            if t < 3:
                arm = t
            elif choices.random() < epsilon:
                arm = choices.randrange(3)
            else:
                arm = max(range(3), key=lambda a: rewards[a] / counts[a])
        elif policy == "thompson":
            samples = [choices.betavariate(alpha[a], beta[a]) for a in range(3)]
            arm = max(range(3), key=lambda a: samples[a])
        else:
            raise ValueError(f"Unknown policy: {policy}")
        rates = rates_at(t, shift_at)
        reward = int(outcomes.random() < rates[arm])
        counts[arm] += 1
        rewards[arm] += reward
        successes += reward
        regret += max(rates) - rates[arm]
        if policy == "thompson":
            alpha = [1 + (a - 1) * decay for a in alpha]
            beta = [1 + (b - 1) * decay for b in beta]
            alpha[arm] += reward
            beta[arm] += 1 - reward
        trace.append({"request": t + 1, "arm": "ABC"[arm], "reward": reward,
                      "phase": "before" if t < shift_at else "after",
                      "pseudo_regret": round(regret, 6)})
    return {"policy": policy, "successes": successes,
            "success_rate": successes / n, "pseudo_regret": regret,
            "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    if args.requests < 6:
        parser.error("--requests must be at least 6")
    results = [simulate(p, args.requests, args.seed) for p in
               ("round_robin", "epsilon_greedy", "thompson")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"SIMULATION requests={args.requests} shift_at={args.requests // 2} seed={args.seed}")
    for r in results:
        print(f"{r['policy']:14s} success={r['success_rate']:.3f} pseudo_regret={r['pseudo_regret']:.1f}")
    with (args.output_dir / "regret.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request"] + [r["policy"] for r in results])
        for i in range(args.requests):
            writer.writerow([i + 1] + [r["trace"][i]["pseudo_regret"] for r in results])
    for r in results:
        with (args.output_dir / f"{r['policy']}_trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(r["trace"][0]))
            writer.writeheader()
            writer.writerows(r["trace"])
    windows = []
    th = results[2]["trace"]
    shift = args.requests // 2
    for start, end in [(0, shift), (shift, min(shift + 100, args.requests)),
                       (max(shift, args.requests - 100), args.requests)]:
        window = th[start:end]
        counts = {arm: sum(row["arm"] == arm for row in window) for arm in "ABC"}
        windows.append({"first_request": start + 1, "last_request": end, "arm_pulls": counts})
        print(f"Thompson requests {start + 1}-{end}: {counts}")
    summary = {"simulation": True, "requests": args.requests, "seed": args.seed,
               "shift_at": shift, "rates_before": rates_at(0, shift),
               "rates_after": rates_at(shift, shift),
               "metric": "Cumulative expected reward gap to the best available arm at each request",
               "results": [{k: v for k, v in r.items() if k != "trace"} for r in results],
               "thompson_windows": windows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved CSV traces and summary to {args.output_dir}")


if __name__ == "__main__":
    main()
