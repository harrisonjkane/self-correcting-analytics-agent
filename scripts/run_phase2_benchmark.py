"""
Phase 2 benchmark runner — SQL agent + Critic self-correction loop.

Runs the same Tier 1–3 questions as Phase 1 but through the ask() pipeline,
which adds a Critic review step between SQL generation and execution.

Usage:
    python3 scripts/run_phase2_benchmark.py

Compare results against Phase 1:
    python -c "
    import sys; sys.path.insert(0, 'evals')
    from scoring import compare_runs
    compare_runs('phase1_single_agent_baseline', 'phase2_critic_added')
    "
"""

# Run from anywhere: put the repo root on sys.path and use it as the working dir.
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import anthropic

load_dotenv()

sys.path.insert(0, str(ROOT / "evals"))

from agents.schema_agent import get_schema, get_schema_with_enums
from agents.sql_agent import ask

DB_PATH = str(ROOT / "data" / "analytics.db")


def main():
    if not Path(DB_PATH).exists():
        sys.exit("Database not found. Run `python data/build_database.py` first.")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic(api_key=api_key)
    schema = get_schema(DB_PATH)
    schema_with_enums = get_schema_with_enums(DB_PATH)

    def pipeline_fn(question: str) -> dict:
        return ask(question, schema, schema_with_enums, DB_PATH, client)

    from scoring import run_benchmark
    run_benchmark(pipeline_fn, "phase2_critic_added", tiers=[1, 2, 3])


if __name__ == "__main__":
    main()
