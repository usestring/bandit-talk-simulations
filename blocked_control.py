"""Compare reactive routing with an always-blocked fourth configuration."""
import hashlib
import json
import random
import statistics
from pathlib import Path

import bandit_environments as be

COST = [4.0, 2.0, 1.0, 1.0]
POLICIES = ["round_robin", "grid_search", "triggered_retest", "ucb1",
            "exp3", "thompson", "thompson_decay"]


class ReactiveControl(be.Environment):
    def __init__(self, n):
        super().__init__("adversarial", n)
        self.k = 4
        self.usage.append(0.0)

    def rates(self, t):
        return super().rates(t) + [0.0]

    def observe(self, arm):
        self.usage = [u * (1 - be.ADV_MEMORY) + be.ADV_MEMORY * (a == arm)
                      for a, u in enumerate(self.usage)]


def simulate(name, n, seed, scenario):
    outcomes = random.Random(seed)
    choices = random.Random(seed + 1_000_003)
    env = be.Environment("adversarial", n) if scenario == "three_arms" else ReactiveControl(n)
    k = 4 if scenario == "blocked_unknown" else 3
    policy = be.Policy(name, k, n, choices)
    policy.trials = 30
    successes = blocked = 0
    spend = 0.0
    for t in range(n):
        arm = policy.select(t)
        ok = int(outcomes.random() < env.rates(t)[arm])
        env.observe(arm)
        policy.update(arm, ok)
        successes += ok
        blocked += arm == 3
        spend += COST[arm]
    return {"success": successes / n, "blocked_share": blocked / n,
            "units_per_success": spend / max(1, successes)}


def main():
    results = {}
    for n, seeds in [(2000, 50), (20000, 20)]:
        rows = {}
        for name in POLICIES:
            rows[name] = {}
            for scenario in ["three_arms", "blocked_unknown", "blocked_pruned"]:
                runs = [simulate(name, n, seed, scenario) for seed in range(seeds)]
                rows[name][scenario] = {
                    key: statistics.mean(run[key] for run in runs) for key in runs[0]}
                assert scenario != "blocked_pruned" or rows[name][scenario] == rows[name]["three_arms"]
            print(n, name, rows[name])
        results[str(n)] = {"seeds": seeds, "requests_per_seed": n, "policies": rows}
    here = Path(__file__).parent
    receipt = {"environment": "Reactive A/B/C plus D that always fails",
               "base_probabilities": be.BASE + [0.0], "attempt_costs": COST,
               "usage_penalty": be.ADV_STRENGTH, "usage_update": be.ADV_MEMORY,
               "initial_usage": [1 / 3] * 3 + [0.0],
               "settings": "Default policies; 30 search trials per arm; re-test <70% over 100 committed outcomes",
               "scenarios": {"three_arms": "Original three-arm reactive site",
                             "blocked_unknown": "D is available; learner observes its failures",
                             "blocked_pruned": "D is known to be blocked and excluded before selection"},
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "simulator_sha256": hashlib.sha256((here / "bandit_environments.py").read_bytes()).hexdigest(),
               "results": results}
    out = here / "results" / "blocked-control"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
