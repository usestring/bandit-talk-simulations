"""Many-sites simulation: does what we learn on one domain transfer to the next?

Synthetic only; no network requests. Standard library only.

500 domains share the 12 pruned knob configurations from bandit_environments.py
and their per-request costs. Traffic is long-tailed (Zipf), so a few domains
get thousands of requests and most get a handful. A domain's success rate for
a configuration is a logistic function of
  the global knob effects (some configurations are better everywhere)
  + its protection family's effect (4 families, each flags one knob)
  + a domain-specific deviation.
Learners maximize success only; cost is reported afterwards.
"""
import argparse
import json
import math
import random
from pathlib import Path

import bandit_environments as be

ARMS = [be.KNOB_ARMS[i] for i in be.PRUNED_ARMS]
COST = [be.FETCH_COST[f] + be.PROXY_COST[p] for f, p, _ in ARMS]
FAMILIES = [("fetch", "tls"), ("fetch", "browser"), ("proxy", "mobile"), ("proxy", "resi-us")]
FAMILY_PENALTY = 2.0
DOMAIN_SD = 0.6
SETTINGS = (
    [("global", "One shared Thompson, ignores domain", {})]
    + [("per_domain", "Thompson per domain, starts from scratch", {})]
    + [("shared_prior", f"Thompson per domain, global prior (weight {k})", {"k": k}) for k in (2, 5, 20)]
    + [("family_prior", "Thompson per domain, prior from its family (weight 5)", {"k": 5})]
    + [("etc", f"Explore then commit per domain ({t}/config)", {"trials": t}) for t in (3, 10)]
    + [("oracle", "Oracle: best config per domain", {})]
)


def make_domains(n_domains, rng):
    domains = []
    for _ in range(n_domains):
        family = rng.randrange(len(FAMILIES))
        knob, value = FAMILIES[family]
        rates = []
        for fetch, proxy, headers in ARMS:
            z = be.KNOB_INTERCEPT + be.FETCH[fetch] + be.PROXY[proxy] + be.HEADERS[headers]
            if {"fetch": fetch, "proxy": proxy}[knob] == value:
                z -= FAMILY_PENALTY
            rates.append(1 / (1 + math.exp(-(z + rng.gauss(0, DOMAIN_SD)))))
        domains.append((family, rates))
    return domains


def traffic(n_domains, n_requests, rng, s=1.1):
    weights = [1 / (r + 1) ** s for r in range(n_domains)]
    return rng.choices(range(n_domains), weights=weights, k=n_requests)


def simulate(name, params, domains, stream, rng):
    k = len(ARMS)
    wins = [[0] * k for _ in domains]
    losses = [[0] * k for _ in domains]
    g_wins, g_losses = [0] * k, [0] * k
    f_wins = [[0] * k for _ in FAMILIES]
    f_losses = [[0] * k for _ in FAMILIES]
    out = []
    for d in stream:
        family, rates = domains[d]
        w, l = wins[d], losses[d]
        if name == "oracle":
            arm = max(range(k), key=lambda a: rates[a])
        elif name == "global":
            arm = max(range(k), key=lambda a: rng.betavariate(1 + g_wins[a], 1 + g_losses[a]))
        elif name == "etc":
            tried = [w[a] + l[a] for a in range(k)]
            short = [a for a in range(k) if tried[a] < params["trials"]]
            arm = short[0] if short else max(range(k), key=lambda a: w[a] / tried[a])
        else:
            if name == "per_domain":
                pa, pb = [1] * k, [1] * k
            else:
                sw, sl = (f_wins[family], f_losses[family]) if name == "family_prior" else (g_wins, g_losses)
                mean = [(1 + sw[a]) / (2 + sw[a] + sl[a]) for a in range(k)]
                pa = [1 + params["k"] * m for m in mean]
                pb = [1 + params["k"] * (1 - m) for m in mean]
            arm = max(range(k), key=lambda a: rng.betavariate(pa[a] + w[a], pb[a] + l[a]))
        ok = rng.random() < rates[arm]
        if ok:
            w[arm] += 1
            g_wins[arm] += 1
            f_wins[family][arm] += 1
        else:
            l[arm] += 1
            g_losses[arm] += 1
            f_losses[family][arm] += 1
        out.append((d, ok, COST[arm]))
    return out


def summarize(outcomes, keep):
    rows = [(ok, cost) for d, ok, cost in outcomes if keep(d)]
    wins = sum(ok for ok, _ in rows)
    spend = sum(cost for _, cost in rows)
    return {"requests": len(rows), "success_rate": wins / max(1, len(rows)),
            "cost_per_success": spend / max(1, wins), "cost_per_request": spend / max(1, len(rows))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", type=int, default=500)
    parser.add_argument("--requests", type=int, default=30000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--gif", action="store_true", help="animated success and cost per success")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "many-sites")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    groups = ("all", "head", "tail")
    totals = {label: {g: [] for g in groups} for _, label, _ in SETTINGS}
    roll = {label: [0.0] * args.requests for _, label, _ in SETTINGS}
    spend = {label: [0.0] * args.requests for _, label, _ in SETTINGS}
    splits = []
    for seed in range(args.seeds):
        world = random.Random(seed)
        domains = make_domains(args.domains, world)
        stream = traffic(args.domains, args.requests, world)
        counts = [0] * args.domains
        for d in stream:
            counts[d] += 1
        head = {d for d in range(args.domains) if counts[d] >= 100}
        tail = {d for d in range(args.domains) if 0 < counts[d] <= 20}
        splits.append((len(head), len(tail), sum(counts[d] for d in tail) / args.requests))
        keep = {"all": lambda d: True, "head": head.__contains__, "tail": tail.__contains__}
        for name, label, params in SETTINGS:
            outcomes = simulate(name, params, domains, stream, random.Random(seed + 7_000_003))
            for g in groups:
                totals[label][g].append(summarize(outcomes, keep[g]))
            if args.gif:
                for i, r in enumerate(be.rolling([ok for _, ok, _ in outcomes], args.window)):
                    roll[label][i] += r / args.seeds
                for i, r in enumerate(be.rolling([cost for _, _, cost in outcomes], args.window)):
                    spend[label][i] += r / args.seeds
    rows = []
    for name, label, params in SETTINGS:
        row = {"policy": name, "label": label, "params": params}
        for g in groups:
            runs = totals[label][g]
            row[g] = {m: sum(r[m] for r in runs) / len(runs)
                      for m in ("success_rate", "cost_per_success", "cost_per_request")}
        rows.append(row)
        print(f"{label:52s} " + " ".join(
            f"{g}:{row[g]['success_rate']:6.1%} {row[g]['cost_per_success']:6.1f}" for g in groups))
    n_head = sum(h for h, _, _ in splits) / len(splits)
    n_tail = sum(t for _, t, _ in splits) / len(splits)
    tail_traffic = sum(s for _, _, s in splits) / len(splits)
    print(f"head domains (>=100 requests): {n_head:.0f}; tail domains (1-20 requests): {n_tail:.0f}, "
          f"{tail_traffic:.1%} of traffic")
    summary = {"simulation": True, "domains": args.domains, "requests": args.requests, "seeds": args.seeds,
               "arms": [list(a) for a in ARMS], "costs": COST, "families": FAMILIES,
               "family_penalty": FAMILY_PENALTY, "domain_sd": DOMAIN_SD,
               "head_domains": n_head, "tail_domains": n_tail, "tail_traffic_share": tail_traffic,
               "results": rows}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if args.gif:
        shown = {"One shared Thompson, ignores domain": "#9aa0a6",
                 "Thompson per domain, starts from scratch": "#d93025",
                 "Thompson per domain, global prior (weight 5)": "#1a73e8",
                 "Thompson per domain, prior from its family (weight 5)": "#188038",
                 "Explore then commit per domain (3/config)": "#e37400"}
        oracle = "Oracle: best config per domain"
        panels = [
            {"title": f"{args.domains} sites, {args.requests:,} requests x {args.seeds} seeds",
             "ylabel": f"Success rate (rolling {args.window:,})", "ylim": (0.5, 1.0),
             "ref": ("Oracle: best config per site", roll[oracle]),
             "series": [(label, color, roll[label]) for label, color in shown.items()]},
            {"title": "Cost per success (illustrative units)",
             "ylabel": f"Cost per success (rolling {args.window:,})", "ylim": (0, 50),
             "ref": ("Oracle", [c / max(1e-9, r) for c, r in zip(spend[oracle], roll[oracle])]),
             "series": [(label, color, [c / max(1e-9, r) for c, r in zip(spend[label], roll[label])])
                        for label, color in shown.items()]},
        ]
        be.animate_lines(args.output_dir / "learning.gif", panels, args.requests)


if __name__ == "__main__":
    main()
