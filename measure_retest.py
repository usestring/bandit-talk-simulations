import hashlib
import json
import random
from pathlib import Path

from bandit_environments import RETEST_THRESHOLD, RETEST_WINDOW, simulate_with_policy

REQUESTS = 2000
SEEDS = range(50, 150)
POLICIES = ("grid_search", "triggered_retest", "thompson", "thompson_decay")


def measure(kind):
    shifts = [random.Random(seed + 9_000_007).randint(400, 1600) for seed in SEEDS]
    results = {}
    for name in POLICIES:
        rates, costs, retests, test_requests = [], [], [], []
        for seed, shift_at in zip(SEEDS, shifts):
            _, policy, _, wins, cost = simulate_with_policy(name, kind, REQUESTS, seed, shift_at)
            rates.append(sum(wins) / REQUESTS)
            costs.append(cost / max(1, sum(wins)))
            retests.append(policy.retests)
            test_requests.append(policy.test_requests)
        results[name] = {
            "mean_success_rate": sum(rates) / len(SEEDS),
            "mean_cost_per_success": sum(costs) / len(SEEDS),
            "mean_retests": sum(retests) / len(SEEDS),
            "runs_with_retests": sum(value > 0 for value in retests),
            "mean_test_requests": sum(test_requests) / len(SEEDS),
            "mean_test_fraction": sum(test_requests) / (len(SEEDS) * REQUESTS),
        }
    return {"shift_at_by_seed": shifts if kind == "shift" else None, "policies": results}


def main():
    receipt = {
        "simulation": True, "requests_per_run": REQUESTS, "seed_range": [50, 149],
        "triggered_retest": {"window": RETEST_WINDOW, "threshold": RETEST_THRESHOLD,
                             "chosen_before_evaluation": True},
        "change_times": "Uniform integers 400..1600 from a separate RNG; never passed to the policy",
        "cost": "One synthetic unit per attempt, including detection delay and all trial traffic",
        "counters": "Re-tests and test requests count explicit grid-search rounds only; zero for a bandit does not mean zero exploration",
        "boundaries": "No parameter search. The floor is illustrative, not a tuned optimum or production SLA. Below-floor best arms and sampling noise can cause repeated tests. Improvements above the floor can go unnoticed.",
        "simulation_sha256": hashlib.sha256(Path(__file__).with_name("bandit_environments.py").read_bytes()).hexdigest(),
        "results": {"random_shift": measure("shift"), "no_change": measure("stationary")},
    }
    output = Path(__file__).parent / "results/environments/retest-check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n")
    for kind, result in receipt["results"].items():
        for name, values in result["policies"].items():
            print(kind, name, json.dumps(values))


if __name__ == "__main__":
    main()
