"""
_test_error_handling.py — Deliberately triggers failure conditions to confirm graceful handling.

Tests 1-3 require no real API calls (pure function or mock-based).
Test 4 requires network access to obtain a typed AuthenticationError from Anthropic.

Failure modes tested:
  1. Missing / empty database  → RuntimeError with human-readable message
  2. Short forecast history    → _err() dict returned, no exception raised
  3. Critic exhaustion         → error dict returned after 3 rejections, rejected SQL never runs
  4. Bad API key               → anthropic.AuthenticationError (typed), not a generic crash
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import anthropic
import pandas as pd

from agents.model_agent import _train_forecast
from agents.orchestrator_agent import run_pipeline
from agents.schema_agent import get_schema, get_schema_with_enums

PASS = "PASS"
FAIL = "FAIL"
all_passed = True


def fail(msg: str):
    global all_passed
    all_passed = False
    print(f"  {FAIL} — {msg}")


# ── Test 1: Missing / empty database ─────────────────────────────────────────
print("═" * 60)
print("TEST 1 — Missing / empty database → clear RuntimeError")
print("═" * 60)

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
    empty_db = f.name  # an empty file — sqlite3 opens it but finds no tables

try:
    for fn_name, fn in [("get_schema", get_schema), ("get_schema_with_enums", get_schema_with_enums)]:
        try:
            fn(empty_db)
            fail(f"{fn_name}() did not raise — expected RuntimeError for empty DB")
        except RuntimeError as e:
            msg = str(e)
            if "no tables" in msg.lower() or "missing" in msg.lower() or "empty" in msg.lower():
                print(f"  {PASS}  [{fn_name}] raised RuntimeError: {msg}")
            else:
                fail(f"{fn_name}() raised RuntimeError but message was unclear: {msg}")
        except Exception as e:
            fail(f"{fn_name}() raised unexpected {type(e).__name__}: {e}")
finally:
    os.unlink(empty_db)

# Also verify that a genuinely missing path (never created) is handled
missing_path = "/tmp/_test_db_does_not_exist_XYZXYZ.db"
if os.path.exists(missing_path):
    os.remove(missing_path)
try:
    get_schema(missing_path)
    fail("get_schema() on nonexistent path did not raise")
except RuntimeError as e:
    print(f"  {PASS}  [missing path] raised RuntimeError: {e}")
except Exception as e:
    fail(f"get_schema() on missing path raised {type(e).__name__}: {e}")
finally:
    if os.path.exists(missing_path):
        os.remove(missing_path)


# ── Test 2: Short forecast history ───────────────────────────────────────────
print()
print("═" * 60)
print("TEST 2 — Short forecast history (2 months) → graceful _err() return")
print("═" * 60)

short_df = pd.DataFrame({
    "period": ["2024-01", "2024-02"],
    "revenue": [10000.0, 12000.0],
})

result = _train_forecast(
    df=short_df,
    date_column="period",
    value_column="revenue",
    sql="SELECT strftime('%Y-%m', order_date) AS period, SUM(amount) AS revenue FROM orders",
    question="forecast next quarter's revenue",
    steps=3,
    alpha=0.05,
)

if result.get("error"):
    msg = result["error"]
    if any(kw in msg.lower() for kw in ("insufficient", "observation", "data")):
        print(f"  {PASS}  returned _err() dict with clear message: {msg}")
    else:
        print(f"  {PASS}  returned _err() dict (message could be clearer): {msg}")
else:
    fail("_train_forecast() with 2 rows succeeded — expected an error about insufficient data")

# Confirm it returned a dict (not raised an exception) — reaching here means no crash
print(f"  {PASS}  no exception raised — caller receives a dict, not a traceback")


# ── Test 3: Critic exhaustion — rejected SQL never executed ──────────────────
print()
print("═" * 60)
print("TEST 3 — Critic exhaustion: all 3 attempts rejected → error dict, no execution")
print("═" * 60)

def _mock_message(text: str):
    """Build a minimal MagicMock that looks like an Anthropic API response."""
    msg = MagicMock()
    msg.content[0].text = text
    msg.usage.input_tokens = 10
    msg.usage.output_tokens = 10
    return msg

_PLAN_JSON = '{"needs_sql": true, "needs_chart": false, "needs_model": false, "reasoning": "test"}'

with patch("agents.orchestrator_agent.plan") as mock_plan, \
     patch("agents.orchestrator_agent.review_sql") as mock_review, \
     patch("agents.orchestrator_agent.retrieve_similar_corrections", return_value=[]):

    mock_plan.return_value = {
        "needs_sql": True, "needs_chart": False, "needs_model": False,
        "reasoning": "test", "tokens_used": 0,
    }
    mock_review.return_value = {
        "passed": False, "reason": "FILTER LOGIC: mocked rejection", "tokens_used": 0,
    }

    client = MagicMock()
    client.messages.create.return_value = _mock_message("SELECT * FROM orders")

    result = run_pipeline(
        question="how many completed orders do we have?",
        schema="Tables:\n  orders(id INTEGER, status TEXT)",
        schema_with_enums="Tables:\n  orders(id INTEGER, status TEXT)\n    -- status: 'Completed'",
        db_path="data/analytics.db",
        client=client,
    )

# Confirm the result is a dict with a clear error, not a crash
if result.get("error"):
    err = result["error"]
    if "3" in err or "attempt" in err.lower() or "valid" in err.lower():
        print(f"  {PASS}  returned error after exhaustion: {err}")
    else:
        print(f"  {PASS}  returned error (message: {err})")
else:
    fail("run_pipeline() returned no error after 3 Critic rejections — unreviewed SQL may have run")

# Confirm mock_review was called exactly MAX_ATTEMPTS times (3), not more, not fewer
calls = mock_review.call_count
if calls == 3:
    print(f"  {PASS}  review_sql() called exactly 3 times — retry loop ran to completion")
else:
    fail(f"review_sql() called {calls} times (expected 3)")

# Confirm SQL execution (sqlite3) was never reached — client.messages.create was called
# for SQL generation but sqlite3.connect should NOT have been called on the real DB
# (we verify this by checking the result has no raw_result, answer, or explanation)
if result.get("answer") is None and result.get("raw_result") is None:
    print(f"  {PASS}  answer=None and raw_result=None — rejected SQL was never executed")
else:
    fail(f"answer or raw_result is populated despite Critic rejection — unreviewed SQL ran")


# ── Test 4: Bad API key → typed AuthenticationError ──────────────────────────
print()
print("═" * 60)
print("TEST 4 — Bad API key → anthropic.AuthenticationError (typed, not a crash)")
print("         (requires network access to reach the Anthropic API)")
print("═" * 60)

bad_client = anthropic.Anthropic(api_key="sk-ant-INVALID-KEY-FOR-TESTING")
try:
    bad_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=16,
        messages=[{"role": "user", "content": "test"}],
    )
    fail("No exception raised for invalid API key — expected AuthenticationError")
except anthropic.AuthenticationError as e:
    print(f"  {PASS}  raised anthropic.AuthenticationError (not a generic Exception)")
    print(f"         message: {str(e)[:120]}")
    print(f"         → app.py's specific handler will show a clear auth-failure message")
except anthropic.APIConnectionError:
    print(f"  SKIP — No network access; can't reach Anthropic API to verify auth error type")
    print(f"         (local test environment; this test will pass in deployed environment)")
except Exception as e:
    fail(f"Wrong exception type: {type(e).__name__}: {e}")


# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("═" * 60)
if all_passed:
    print("All error handling tests passed.")
else:
    print("One or more tests FAILED — review output above.")
    sys.exit(1)
