# Bandit talk simulations

Offline simulations from **The Multi-Armed Bandit and Web Data**, presented at Extract Summit Austin, October 2026.

Compare round-robin, explore-then-commit, periodic re-testing, epsilon-greedy, UCB1, EXP3, and Thompson sampling with and without forgetting. All outcomes and costs are synthetic. The scripts make no network requests.

## Run

Use Python 3.10 or later. The simulation and measurement scripts use only the standard library.

```bash
python3 bandit_demo.py
python3 bandit_environments.py
python3 adversarial_long.py
python3 many_sites.py
python3 measure_convergence.py
python3 -m unittest -v
```

The default runs reproduce the saved JSON receipts in `reference-results/`. Generated files go to `results/`, which Git ignores. The larger runs can take several minutes.

For a quick comparison:

```bash
python3 bandit_environments.py --requests 2000 --seeds 2
python3 adversarial_long.py --requests 2000 --seeds 2
python3 many_sites.py --domains 100 --requests 3000 --seeds 2
```

## Charts and animations

Install the optional plotting dependencies in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-plot.txt
python bandit_environments.py --plot --gif
python adversarial_long.py --plot
```

`bandit_environments.py --plot --gif` writes five PNG charts and five looping GIFs. Every environment chart labels its results as synthetic.

## Experiments

| Script | Question | Default run |
| --- | --- | --- |
| `bandit_demo.py` | How does a small selector adapt when the winning arm changes? | 2,000 decisions, seed 7 |
| `bandit_environments.py` | How do selection methods behave under stable rewards, abrupt shifts, gradual drift, reactive rewards, and a larger configuration space? | 2,000 decisions per policy and environment, 50 seeds |
| `adversarial_long.py` | How do forgetting and re-testing settings behave when traffic changes future rewards? | 20,000 decisions per setting, 20 seeds |
| `many_sites.py` | Does sharing a prior help domains with little traffic? | 500 synthetic domains, 30,000 decisions, 10 seeds |
| `measure_convergence.py` | How long does each policy sustain a small expected reward gap? | 2,000 decisions per run, 50 seeds |

The first four environments use three abstract configurations with base success probabilities 0.90, 0.75, and 0.60. The knob environment uses 36 combinations of fetch method, proxy type, and header set. A predefined shortlist keeps 12 combinations. The many-sites experiment uses that same shortlist.

## Reading the results

- Policies learn from binary success alone. Cost is an invented per-request value reported afterward; the policies do not optimize cost or latency.
- Expected regret uses the simulator's known probabilities. A real selector does not observe those probabilities or the oracle best arm.
- The reactive environment is called `adversarial` in the code. It reduces an arm's success probability with recent traffic share. This toy model does not establish performance against arbitrary adversaries.
- The shortlist retains the best combination before and after the modeled change by construction. Pruning can discard the best arm in another problem.
- Convergence means a rolling mean expected gap of at most 0.05 over 100 decisions, sustained for 100 consecutive full windows within a phase. The earliest confirmation is decision 199. Missing quantiles indicate runs censored at the phase budget.
- Reported convergence seconds assume five seconds per sequential decision. They are a timing conversion, not measured latency or retries within one request.

These experiments illustrate tradeoffs in a fixed model. They do not establish a universal policy ranking or a production improvement.

## License

MIT; see [LICENSE](LICENSE).
