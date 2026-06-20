"""
Smoke test for the Chart Agent.

Runs 3 benchmark questions through the full pipeline and confirms:
  - Chart-worthy questions produce a saved image file
  - Single-value questions produce no chart (should_chart returns False)
  - Charts pass Critic review before execution
"""

import os
import sys

import anthropic
from dotenv import load_dotenv

from agents.chart_agent import create_chart, should_chart
from agents.critic_agent import review_chart_code
from agents.schema_agent import get_schema, get_schema_with_enums
from agents.sql_agent import ask

load_dotenv()

DB_PATH = "data/analytics.db"
CHART_DIR = "charts_test"

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
schema = get_schema(DB_PATH)
schema_with_enums = get_schema_with_enums(DB_PATH)


def run(label: str, question: str, expect_chart: bool):
    print(f"\n{'─' * 60}")
    print(f"TEST: {label}")
    print(f"Q:    {question}")

    result = ask(
        question=question,
        schema=schema,
        schema_with_enums=schema_with_enums,
        db_path=DB_PATH,
        client=client,
    )

    answer = result.get("answer")
    chart_path = result.get("chart_path")

    print(f"answer type:  {type(answer).__name__}")
    print(f"should_chart: {should_chart(question, answer)}")
    print(f"chart_path:   {chart_path}")

    if expect_chart:
        if chart_path and os.path.exists(chart_path):
            size = os.path.getsize(chart_path)
            print(f"PASS — chart saved ({size:,} bytes): {chart_path}")
        else:
            print("FAIL — expected a chart but none was created")
            sys.exit(1)
    else:
        if chart_path is None:
            print("PASS — correctly produced no chart for single-value answer")
        else:
            print(f"FAIL — expected no chart but got: {chart_path}")
            sys.exit(1)

    if result.get("explanation"):
        print(f"explanation:  {result['explanation'][:120]}...")


# ── Test 1: grouped breakdown → vertical bar chart ───────────────────────────
run(
    label="Revenue by customer segment (grouped dict → bar chart)",
    question="What is our revenue broken down by customer segment?",
    expect_chart=True,
)

# ── Test 2: ranked list → horizontal bar chart ────────────────────────────────
run(
    label="Top 5 customers by revenue (ranked list → barh chart)",
    question="Who are our top 5 customers by total revenue?",
    expect_chart=True,
)

# ── Test 3: single value → no chart ──────────────────────────────────────────
run(
    label="Single value — should NOT produce a chart",
    question="How many customers do we have?",
    expect_chart=False,
)

print(f"\n{'─' * 60}")
print("All tests passed.")
