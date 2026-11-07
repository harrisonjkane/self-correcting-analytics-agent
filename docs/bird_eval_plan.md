# BIRD evaluation plan

## Why

The in-repo benchmark is 11 auto-scored questions on a synthetic five-table database, and the Critic's Layer 2 rules were written for that database. "100% on 11 questions" can't distinguish a good Critic from one fitted to its test. BIRD is an external, widely used text-to-SQL benchmark built from real, messy databases (dirty values, ambiguous columns, domain knowledge), so it answers the question that matters:

> Does the Critic improve accuracy on data it was never tuned on, and at what cost?

## What gets measured

| | Baseline | Critic |
|---|---|---|
| SQL generation | 1 attempt | Up to 3 attempts, Critic-reviewed |
| Critic Layer 1 (schema-derived) | — | On |
| Critic Layer 2 (demo business rules) | Off | **Off** |
| RAG correction log | Off | Off |
| Prompt | Identical | Identical |

The only difference between conditions is the Critic. Both run on the same stratified sample.

- **Metric:** execution accuracy (EX), BIRD's standard. A prediction is correct when its result set equals the gold SQL's result set (order-insensitive). Queries time out at 30s.
- **Strictness:** SQL the Critic rejects on all three attempts is scored wrong and never executed, matching what the app does.
- **Also reported:** accuracy by difficulty, average tokens and latency per question, retries, questions the Critic rejected outright, and the paired comparison (questions the Critic **fixed** vs. **broke**). The broken count matters: a Critic that rejects correct SQL has a real cost.
- **Hints:** BIRD's expert `evidence` field is appended to each question as a hint, which is standard practice for BIRD reporting.

## Held-out discipline

Layer 1 has never seen BIRD, so the first full run is held-out by construction. Protect that:

1. Run the headline sample **once** (`--n 150 --seed 42`) before changing anything.
2. If the results suggest Critic changes, iterate on a different sample (`--seed 7`) and keep the original results file as the pre-change number.
3. Report both: the clean first-run number, and any post-change number labelled as such.

## Running it

1. Download BIRD Mini-Dev (500 questions, 11 SQLite databases, roughly 30/50/20 simple/moderate/challenging) from <https://github.com/bird-bench/mini_dev> and unzip it anywhere.
2. Smoke test with 20 questions. The printed summary includes average tokens per question; use it to estimate the cost of the full run before starting it.
   ```bash
   python3 scripts/run_bird_eval.py --bird-dir ~/data/minidev --n 20 --label smoke
   ```
3. Full run:
   ```bash
   python3 scripts/run_bird_eval.py --bird-dir ~/data/minidev --n 150 --seed 42
   ```
   Results stream to `evals/results/bird_n150_seed42.jsonl`. If it stops partway (rate limit, laptop sleep), re-run the same command and it resumes.
4. Commit the `.jsonl` and `.summary.json`, using `git add -f` because results are gitignored by default, so every published number has an artifact behind it.

## How to report it

Add a section to the README with the table below, filled from the summary file. Lead with the paired fixed/broken counts and the cost delta, not just the headline accuracy.

| Condition | EX (n=150) | Simple | Moderate | Challenging | Avg tokens | Avg latency |
|---|---|---|---|---|---|---|
| Baseline | | | | | | |
| Critic | | | | | | |

Critic fixed **X** questions and broke **Y**.

Whatever the result, it's publishable. A Critic that adds +3pp at 3× cost is an honest, interesting finding. So is one that helps on the demo database and not on BIRD, since that would show Layer 2, not Layer 1, carried the original gains.
