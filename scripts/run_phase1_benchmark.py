"""
Phase 1 benchmark runner — single-agent SQL baseline.

Builds the database if it doesn't exist, then runs Tier 1–3 questions
through the SQL agent and scores results via the eval harness.

Usage:
    python3 scripts/run_phase1_benchmark.py
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

# evals/ uses bare `from benchmark import BENCHMARK` — add it to path
sys.path.insert(0, str(ROOT / "evals"))

from agents.schema_agent import get_schema
from agents.sql_agent import run_query

DB_PATH = str(ROOT / "data" / "analytics.db")


def _ensure_db():
    if not Path(DB_PATH).exists():
        print("Database not found — building it now...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_database",
            ROOT / "data" / "build_database.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.build()


def main():
    _ensure_db()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.")

    client = anthropic.Anthropic(api_key=api_key)
    schema = get_schema(DB_PATH)

    def pipeline_fn(question: str) -> dict:
        return run_query(question, schema, DB_PATH, client)

    # Import here so evals/ is already in path
    from scoring import run_benchmark
    run_benchmark(pipeline_fn, "phase1_single_agent_baseline", tiers=[1, 2, 3])


if __name__ == "__main__":
    main()
