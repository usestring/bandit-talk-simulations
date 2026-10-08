import hashlib
import json
import math
import random
from pathlib import Path

from bandit_environments import (Environment, Policy, POLICIES, PRUNED_ARMS,
                                 applicable, rolling, simulate)

REQUESTS = 2000
SEEDS = range(50)
SECONDS_PER_REQUEST = 5
WINDOW = 100
HOLD = 100
MAX_EXPECTED_GAP = 0.05


def confirmation_request(gaps):
    consecutive = 0
    for request, gap in enumerate(rolling(gaps, WINDOW), 1):
        consecutive = consecutive + 1 if request >= WINDOW and gap <= MAX_EXPECTED_GAP else 0
        if consecutive == HOLD:
            return request
    return None


def censored_quantile(hits, probability):
    ordered = sorted(hit for hit in hits if hit is not None)
    rank = math.ceil(len(hits) * probability)
    return ordered[rank - 1] if len(ordered) >= rank else None


def summarize(hits, budget):
    quantiles = {name: censored_quantile(hits, probability)
                 for name, probability in (("median", 0.5), ("p90", 0.9))}
    return {"runs_converged": sum(hit is not None for hit in hits),
            "runs": len(hits), "budget_requests": budget,
            "budget_seconds": budget * SECONDS_PER_REQUEST,
            **{name + "_requests": hit for name, hit in quantiles.items()},
            **{name + "_seconds": None if hit is None else hit * SECONDS_PER_REQUEST
               for name, hit in quantiles.items()}, "confirmation_requests_by_seed": hits}


def measure(kind, policy_name, shortlist=False):
    phases = ([("initial", 0, REQUESTS)] if kind == "stationary" else
              [("final_plateau", 1500, REQUESTS)] if kind == "drift" else
              [("initial", 0, 1000), ("after_shift", 1000, REQUESTS)])
    hits = {phase: [] for phase, _, _ in phases}
    success_rates = []
    for seed in SEEDS:
        env = Environment(kind, REQUESTS)
        mapping = PRUNED_ARMS if shortlist else list(range(env.k))
        if shortlist:
            policy = Policy(policy_name, len(mapping), REQUESTS, random.Random(seed + 1_000_003))
            outcomes = random.Random(seed)
            arms, wins = [], []
            for t in range(REQUESTS):
                local_arm = policy.select(t)
                arm = mapping[local_arm]
                reward = int(outcomes.random() < env.rates(t)[arm])
                policy.update(local_arm, reward)
                arms.append(arm)
                wins.append(reward)
        else:
            _, arms, wins, _ = simulate(policy_name, kind, REQUESTS, seed)
        success_rates.append(sum(wins) / REQUESTS)
        gaps = []
        for t, arm in enumerate(arms):
            rates = env.rates(t)
            gaps.append(max(rates) - rates[arm])
        for phase, start, end in phases:
            hits[phase].append(confirmation_request(gaps[start:end]))
    return {"mean_success_rate": sum(success_rates) / len(success_rates),
            "phases": {phase: summarize(hits[phase], end - start)
                       for phase, start, end in phases}}


def main():
    results = {kind: {name: measure(kind, name) for name in POLICIES if applicable(name, kind)}
               for kind in ("stationary", "shift", "drift", "knobs")}
    results["knobs_shortlist"] = {name: measure("knobs", name, shortlist=True)
        for name in ("round_robin", "thompson", "thompson_decay", "epsilon_greedy")}
    receipt = {"simulation": True, "requests_per_run": REQUESTS, "seed_range": [0, 49],
        "seconds_per_sequential_request": SECONDS_PER_REQUEST,
        "metric": "Confirmation of rolling-100 mean expected gap <= 0.05 for 100 consecutive full windows within a phase",
        "expected_gap": "Oracle success probability minus chosen-arm success probability before observing its outcome",
        "quantiles": "Nearest rank across all 50 seeds, with non-converged runs right-censored. Null means rank not reached within budget.",
        "timing": "Attempts across a learning run, not retries of one fetch. Sequential service-time equivalent excludes backoff, arrival gaps and concurrency.",
        "adversarial": "Not applicable: traffic changes future rewards and there is no fixed convergence target.",
        "simulation_sha256": hashlib.sha256(Path(__file__).with_name("bandit_environments.py").read_bytes()).hexdigest(),
        "results": results}
    output = Path(__file__).parent / "results/environments/convergence.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    header = {key: value for key, value in receipt.items() if key != "results"}
    lines = [json.dumps(header, indent=2)[:-2] + ",", '  "results": {']
    for kind_index, (kind, policies) in enumerate(results.items()):
        lines.append('    "' + kind + '": {')
        for policy_index, (name, result) in enumerate(policies.items()):
            comma = "," if policy_index < len(policies) - 1 else ""
            lines.append('      "' + name + '": ' + json.dumps(result) + comma)
        lines.append("    }" + ("," if kind_index < len(results) - 1 else ""))
    lines.extend(["  }", "}"])
    output.write_text("\n".join(lines) + "\n")
    print(output)
    for kind, policies in results.items():
        for name, result in policies.items():
            for phase, values in result["phases"].items():
                print(kind, name, phase, values["median_requests"], values["median_seconds"],
                      values["p90_requests"], values["runs_converged"])


if __name__ == "__main__":
    main()
