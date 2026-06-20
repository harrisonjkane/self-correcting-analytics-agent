"""
Smoke test for the Model Agent.

Tests:
  1. should_model() correctly identifies modeling vs non-modeling questions
  2. End-to-end regression: "predict order value based on quantity and product category"
"""

import os
import sys

import anthropic
from dotenv import load_dotenv

from agents.model_agent import should_model, train_model
from agents.schema_agent import get_schema_with_enums

load_dotenv()

DB_PATH = "data/analytics.db"
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
schema_with_enums = get_schema_with_enums(DB_PATH)

# ── Test 1: should_model() keyword detection ──────────────────────────────────
print("═" * 60)
print("TEST 1 — should_model() keyword detection")
print("═" * 60)

cases = [
    # (question, expected_result)
    ("predict order value based on quantity and product category", True),
    ("forecast next quarter's revenue",                            True),
    ("which customers are likely to churn?",                      True),
    ("cluster customers by purchase behavior",                    True),
    ("classify orders by risk level",                             True),
    # Non-modeling questions — must all return False
    ("how many customers do we have?",                            False),
    ("what is our total revenue from completed orders?",          False),
    ("who are our top 5 customers by revenue?",                   False),
    ("which region has the most critical tickets?",               False),
    ("what is the average order value?",                          False),
    ("is our support team keeping up with demand?",               False),
]

all_passed = True
for question, expected in cases:
    result = should_model(question)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        all_passed = False
    flag = "MODEL" if result else "SKIP "
    print(f"  {status}  [{flag}]  {question[:60]}")

if not all_passed:
    print("\nshould_model() test FAILED")
    sys.exit(1)
print(f"\n  All {len(cases)} keyword detection cases correct.\n")


# ── Test 2: End-to-end regression ────────────────────────────────────────────
print("═" * 60)
print("TEST 2 — End-to-end regression")
print("Q: predict order value based on quantity and product category")
print("═" * 60)

result = train_model(
    question="predict order value based on quantity and product category",
    schema_with_enums=schema_with_enums,
    db_path=DB_PATH,
    client=client,
)

print(f"\ntask_type:   {result['task_type']}")
print(f"model_class: {result['model_class']}")
print(f"sql:         {result['sql'][:120]}...")
print(f"metrics:     {result['metrics']}")
print(f"summary:     {result['summary']}")
print(f"plot_path:   {result['plot_path']}")
print(f"error:       {result['error']}")

if result.get("error"):
    print(f"\nFAIL — train_model() returned an error: {result['error']}")
    sys.exit(1)

if result["task_type"] != "regression":
    print(f"\nFAIL — expected task_type='regression', got '{result['task_type']}'")
    sys.exit(1)

metrics = result["metrics"]
if "r2" not in metrics or "mae" not in metrics:
    print(f"\nFAIL — missing expected regression metrics: {metrics}")
    sys.exit(1)

if result["plot_path"] and not __import__("os").path.exists(result["plot_path"]):
    print(f"\nFAIL — plot_path set but file does not exist: {result['plot_path']}")
    sys.exit(1)

print(f"\nPASS — regression model trained: R²={metrics['r2']}, MAE={metrics['mae']:,.2f}")
if result["plot_path"]:
    size = os.path.getsize(result["plot_path"])
    print(f"PASS — scatter plot saved ({size:,} bytes): {result['plot_path']}")

print("\n" + "═" * 60)
print("All model agent tests passed.")
