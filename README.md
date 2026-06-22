# Self-Correcting Analytics Agent

**Ask questions about your data in plain English. SQL is written, reviewed by a Critic agent, and automatically corrected before you ever see it.**

Live demo: [multi-agent-query-engine.streamlit.app](https://multi-agent-query-engine.streamlit.app)

---

## What it does

Turn a natural language question into a verified, plain-English answer backed by real SQL — without writing a single line of code or query.

Point it at a database, ask a business question, and it autonomously explores the schema, writes SQL, has that SQL reviewed and corrected by a second agent before execution, queries the data, builds a visualization if appropriate, optionally trains a predictive model, and explains the findings in plain language.

The job to be done: bridge the gap between "I have a question about our data" and "I have a trustworthy answer" — without a human analyst or SQL knowledge in the loop.

---

## Why this is different from a standard LLM analytics tool

Most natural language → SQL tools generate a query and run it. If the query is wrong but syntactically valid, there's no way to know — it runs, returns a number, and you have no reason to doubt it.

This system adds a **Critic agent** that reviews generated SQL *before* execution, catching:
- String literal case mismatches against the database's actual stored values (e.g. `'critical'` vs. the database's `'Critical'` — syntactically valid, silently matches zero rows)
- Filter logic that technically runs but answers the wrong question (e.g. `status != 'Cancelled'` when the intent was `status = 'Completed'`)

Combined with a RAG layer that retrieves similar past corrections before each generation, the system improves over time without retraining anything.

The claim is not "100% accuracy always." It's "verified accuracy, with a documented, measured improvement from the unreviewed baseline."

---

## Architecture

```
User question
    ↓
Orchestrator Agent  (decides: chart needed? model needed?)
    ↓
Schema Agent  →  SQL Agent  →  Critic Agent  →  Execute
    ↑                               ↓
    └─── retry (up to 3×) ←── FAIL / PASS ──→ log_correction()
    ↓
Explainer Agent  →  Chart Agent (conditional)  →  Model Agent (conditional)
    ↓
Plain-English answer + chart + model results
```

**Eight specialized agents**, each a separate Claude API call with a focused system prompt:

| Agent | Role |
|---|---|
| Orchestrator | Routes question — does it need a chart? A predictive model? |
| Schema Agent | Introspects the database — tables, columns, actual stored enum values |
| SQL Agent | Generates SQL using formal tool use (not regex extraction) |
| Critic Agent | Reviews SQL before execution — two-layer design (schema-derived + domain rules) |
| Explainer Agent | Converts raw results into plain-English narrative |
| Chart Agent | Generates and executes matplotlib charts, reviewed by Critic before execution |
| Model Agent | Trains sklearn models (regression/classification/clustering) when question implies prediction |
| Correction Log | Embeds past mistakes via Voyage AI, stores in ChromaDB, retrieves similar corrections before generation |

---

## The Critic Agent — two layers, explicitly labeled

A schema-only Critic correctly catches case-sensitivity and basic filter-logic errors, but cannot infer business semantics from a schema alone. Testing confirmed a **90.9% ceiling** for the general-purpose version.

The Critic is structured in two explicitly-labeled layers:

- **Layer 1 (schema-derived, general-purpose):** case-sensitivity checks, filter logic soundness, SQL safety — reusable across any database
- **Layer 2 (domain knowledge, maintained):** business-specific rules that cannot be inferred from schema or question text alone (e.g. "revenue implies `orders.status = 'Completed'`")

This design is documented rather than hidden. Layer 2 exists because Layer 1 has a real, measured limit.

---

## Evaluation — built before the agents

The evaluation benchmark was defined **before any agent code was written**, per the project's build-order principle: measure first, build second.

**16 questions across 5 difficulty tiers**, each with a verified correct answer:

| Tier | Focus | Count |
|---|---|---|
| 1 | Direct lookup | 4 |
| 2 | Aggregation | 4 |
| 3 | Multi-table joins | 3 |
| 4 | Ambiguous / business logic | 3 |
| 5 | Visualization + modeling | 2 |

**Five measured states — all confounds isolated before reporting results:**

| State | Accuracy | Notes |
|---|---|---|
| No Critic (Phase 1 baseline) | 45.5% | Bugs are semantic — valid SQL, wrong answers |
| Critic, accidentally domain-armed | 100% | Caught via git diff audit — unrequested scope change, reverted |
| Critic, schema-only | 90.9% | Honest ceiling — can't infer business semantics from schema alone |
| Hybrid Critic, RAG cleared (CLEAN_run1) | 100% | Domain rule isolated from RAG influence — 2 real retries confirm it works independently |
| Hybrid Critic, RAG warm (run2/3) | 100%, 0 retries | RAG retrieval working — past corrections reduce retries on similar questions |

**Cost of the improvement (not free, worth knowing):**

| Metric | Phase 1 (no Critic) | Phase 2 (avg of 3 runs) | Delta |
|---|---|---|---|
| Accuracy | 45.5% | 100% | +54.5pp |
| Avg latency | 3.33s | 8.64s | +5.31s |
| Total tokens | 5,635 | 32,115 | +26,480 (4.7×) |
| Total retries | 0 | 10 | +10 |

---

## Tech stack

| Layer | Choice |
|---|---|
| LLM | Claude Sonnet (claude-sonnet-4-6) via Anthropic API |
| Database | SQLite (same SQL patterns transfer directly to Snowflake/Postgres) |
| Embeddings | Voyage AI (voyage-3) |
| Vector store | ChromaDB (local persistence) |
| ML | scikit-learn (RandomForest, LogisticRegression, KMeans) |
| Forecasting | statsmodels SARIMAX with `conf_int(alpha)` for confidence intervals |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Database schema

Five tables, realistic relational structure, seeded with `random.seed(42)` for reproducibility:

```
customers      (60 rows)  — segment, region, signup_date, account_owner
products       (18 rows)  — category, unit_price
orders         (285 rows) — customer_id, order_date, status
order_items    (731 rows) — order_id, product_id, quantity, unit_price
support_tickets(170 rows) — customer_id, priority, status, subject
```

Revenue = `SUM(order_items.quantity × order_items.unit_price)` where `orders.status = 'Completed'`.

---

## Setup

```bash
git clone https://github.com/harrisonjkane/self-correcting-analytics-agent
cd self-correcting-analytics-agent

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build the test database
python3 data/build_database.py

# Run locally
streamlit run app.py
```

**Environment variables** — create a `.env` file:

```
ANTHROPIC_API_KEY=sk-ant-...
VOYAGE_API_KEY=pa-...
```

For Streamlit Cloud deployment, add both keys in Settings → Secrets.

---

## Sample questions to try

```
How many customers do we have?
What is our total revenue from completed orders?
Who are our top 5 customers by revenue?
Show me revenue broken down by product category
Which region has the most critical support tickets?
Predict order value based on quantity and product category
Forecast next quarter's revenue with a confidence interval
```

---

## Project structure

```
self-correcting-analytics-agent/
├── app.py                          # Streamlit UI — Multi-Agent Query Engine
├── data/
│   └── build_database.py           # Generates analytics.db (run once)
├── agents/
│   ├── orchestrator_agent.py       # Pipeline coordinator — routes to specialists
│   ├── schema_agent.py             # Database introspection (no LLM calls)
│   ├── sql_agent.py                # SQL generation entry point + thin wrapper
│   ├── critic_agent.py             # Two-layer SQL + chart review (temperature=0)
│   ├── explainer_agent.py          # Plain-English narrative generation
│   ├── chart_agent.py              # Matplotlib generation + Critic-reviewed execution
│   ├── model_agent.py              # sklearn ML + SARIMAX forecasting
│   └── correction_log.py           # RAG — Voyage AI embeddings + ChromaDB
├── evals/
│   ├── benchmark.py                # 16 questions + verified ground truth (built first)
│   └── scoring.py                  # Pipeline-agnostic scoring harness
├── run_phase1_benchmark.py         # Baseline measurement (no Critic)
├── run_phase2_benchmark.py         # Critic-enabled measurement
├── run_phase3_benchmark.py         # RAG-enabled measurement
└── requirements.txt
```

---

## A note on AI-assisted development

This project was built using Claude Code as a primary development tool. One notable event during development: mid-session, the AI coding assistant made an unrequested change — hard-coding a business rule directly into the SQL generation prompt while fixing an unrelated Critic reliability issue. This made benchmark numbers look better than they actually were, for the wrong reason.

The change was caught by auditing the full `git diff` before trusting an improved metric, rather than accepting a better-looking result at face value. The unrequested change was reverted, and the benchmark was re-run against only the approved fix.

This is documented in the project's Technical Analysis and AI-Assisted Development Addendum — not as a criticism of AI coding tools, but as a concrete example of what "critically evaluating AI-generated output" looks like in practice.

---

## What this project demonstrates

- **Evals-first methodology** — benchmark defined before any agent code exists, re-run after every architectural change
- **Measured improvement, not just claimed improvement** — five distinct, isolated states with documented confounds
- **Self-correction via a Critic agent** — semantic error detection, not just syntax checking
- **RAG-over-failure-history** — retrieval conditioned on the system's own past mistakes, with a documented null case (empty log) and working case (seeded log) both on record
- **Honest uncertainty** — the 90.9% general-purpose ceiling is documented, the SARIMAX confidence interval width is explained, the cost tradeoff is reported
- **Production-grade engineering habits in an AI-assisted development context**

---

*Built as a portfolio project demonstrating applied AI engineering. Not a production system — single-user, SQLite rather than Snowflake, not yet evaluated on real enterprise data. The architecture and methodology are production-grade; the scale is not.*
