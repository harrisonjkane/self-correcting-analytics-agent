# AI Analytics Consultant (DataChat v2)

A multi-agent analytics system that connects to a database, explores schema,
writes SQL, builds visualizations, trains models, and explains results —
the production evolution of [DataChat](https://github.com/harrisonjkane/datachat).

**Status:** Phase 0 complete — evaluation benchmark built before any agent code,
per the project design doc's build-order principle.

## Why evals come first

Most agentic AI projects build the agents first and demo them once. This
project defines a fixed 16-question benchmark with verified ground-truth
answers *before* writing any agent code. Every architectural change —
adding the Critic agent, adding RAG over failure history — gets re-scored
against the same benchmark, producing measurable before/after evidence
instead of a one-time demo.

## Project structure

```
ai-analytics-consultant/
├── data/
│   ├── build_database.py    # Generates the test SQLite database
│   └── analytics.db         # 5-table relational schema (generated, gitignored)
├── evals/
│   ├── benchmark.py         # 16 questions, 5 tiers, verified ground truth
│   ├── scoring.py           # Pipeline-agnostic scoring harness
│   └── results/             # Benchmark run outputs (gitignored)
├── agents/                  # Orchestrator, Schema, SQL, Critic, Chart, Model, Explainer
├── tools/                   # Formal tool definitions (Anthropic tool use API)
└── requirements.txt
```

## Database schema

5 tables, realistic relational structure:
- `customers` (60 rows) — segment, region, signup date, account owner
- `products` (18 rows) — category, unit price
- `orders` (285 rows) — customer, date, status
- `order_items` (731 rows) — line items per order
- `support_tickets` (170 rows) — priority, status, subject

Designed specifically to exercise multi-table joins and ambiguous
business-logic questions in the evaluation benchmark.

## Benchmark tiers

| Tier | Focus | Count |
|---|---|---|
| 1 | Direct lookup | 4 |
| 2 | Aggregation | 4 |
| 3 | Multi-table joins | 3 |
| 4 | Ambiguous / business logic | 3 |
| 5 | Visualization + modeling | 2 (placeholder until Chart/Model agents exist) |

## Setup

```bash
git clone <repo-url>
cd ai-analytics-consultant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Build the test database
python3 data/build_database.py

# Smoke-test the scoring harness
python3 evals/scoring.py
```

## Architecture (planned)

```
User question
    ↓
Orchestrator Agent (routes to specialists)
    ↓
Schema Agent → SQL Agent → Critic Agent → Chart Agent → Model Agent
    ↓
Explainer Agent (synthesizes plain-English narrative)
    ↓
Response
```

See the full project design document for agent roles, tool definitions,
the self-correction loop, and the RAG-over-failure-history design.

## License

Private project — not licensed for redistribution.
