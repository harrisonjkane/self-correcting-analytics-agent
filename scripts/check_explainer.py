"""
Manual test for the Explainer Agent.

Runs 4 benchmark questions through the full pipeline and prints both the raw
answer and the explanation side by side for quality evaluation.

Questions selected to cover each answer type:
  T1-01  exact_number  (simple count)
  T2-01  exact_number  (revenue — tests dollar formatting)
  T3-01  ranked_list   (top 5 customers)
  T2-02  grouped_values (revenue by segment)

Usage:
    python3 scripts/check_explainer.py
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

sys.path.insert(0, str(ROOT))

from agents.schema_agent import get_schema, get_schema_with_enums
from agents.sql_agent import ask

DB_PATH = str(ROOT / "data" / "analytics.db")

TEST_QUESTIONS = [
    {
        "id": "T1-01",
        "answer_type": "exact_number",
        "question": "How many customers do we have?",
    },
    {
        "id": "T2-01",
        "answer_type": "exact_number",
        "question": "What is our total revenue from completed orders?",
    },
    {
        "id": "T3-01",
        "answer_type": "ranked_list",
        "question": "Who are our top 5 customers by total revenue?",
    },
    {
        "id": "T2-02",
        "answer_type": "grouped_values",
        "question": "What is our revenue broken down by customer segment?",
    },
]

DIVIDER = "─" * 70


def main():
    if not Path(DB_PATH).exists():
        sys.exit("Database not found. Run `python data/build_database.py` first.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set.")

    client = anthropic.Anthropic()
    schema = get_schema(DB_PATH)
    schema_with_enums = get_schema_with_enums(DB_PATH)

    for item in TEST_QUESTIONS:
        print(f"\n{DIVIDER}")
        print(f"[{item['id']}]  {item['answer_type'].upper()}")
        print(f"Question:    {item['question']}")
        print(DIVIDER)

        result = ask(item["question"], schema, schema_with_enums, DB_PATH, client)

        print(f"Raw answer:  {result['answer']}")
        print(f"Explanation: {result['explanation']}")

        if result.get("error"):
            print(f"Error:       {result['error']}")
        if result.get("retries"):
            print(f"Retries:     {result['retries']}")

    print(f"\n{DIVIDER}")


if __name__ == "__main__":
    main()
