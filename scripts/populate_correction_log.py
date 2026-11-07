"""
Seeds the correction log by running one full Phase 2-style benchmark pass.

During this run the correction log is empty, so RAG retrieval finds nothing
and the pipeline behaves identically to Phase 2. Every time the Critic rejects
SQL and the next attempt passes, ask() calls log_correction() automatically,
building up real correction records from actual Critic rejections.

After this script finishes, run run_phase3_benchmark.py to test whether those
stored corrections reduce retries on the same question set.

Usage:
    python3 scripts/populate_correction_log.py
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

    print("Seeding correction log (RAG retrieval inactive — log is empty)...")
    run_benchmark(
        lambda q: ask(q, schema, schema_with_enums, DB_PATH, client),
        "phase3_correction_log_seed",
        tiers=[1, 2, 3],
    )

    from agents.correction_log import _chroma_collection
    count = _chroma_collection().count()
    print(f"\nCorrection log seeded: {count} record(s) stored in ChromaDB.")
    print("Run `python3 scripts/run_phase3_benchmark.py` to test RAG-assisted generation.")


if __name__ == "__main__":
    main()
