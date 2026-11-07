"""
Phase 3 benchmark runner — SQL agent + Critic + RAG correction history.

Runs the same Tier 1–3 questions as Phases 1 and 2 but with past corrections
retrieved from ChromaDB and injected into the system prompt before first
SQL generation. Primary metric vs Phase 2: retry count (accuracy is already
100% and has no further room to improve).

Usage:
    python3 scripts/populate_correction_log.py   # seed the log first
    python3 scripts/run_phase3_benchmark.py      # then run this

Compare against Phase 2:
    python -c "
    import sys; sys.path.insert(0, 'evals')
    from scoring import compare_runs
    compare_runs('phase2_critic_added', 'phase3_rag_added')
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

    for key in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY"):
        if not os.environ.get(key):
            sys.exit(f"{key} is not set.")

    client = anthropic.Anthropic()
    schema = get_schema(DB_PATH)
    schema_with_enums = get_schema_with_enums(DB_PATH)

    from scoring import run_benchmark
    run_benchmark(
        lambda q: ask(q, schema, schema_with_enums, DB_PATH, client),
        "phase3_rag_added",
        tiers=[1, 2, 3],
    )


if __name__ == "__main__":
    main()
