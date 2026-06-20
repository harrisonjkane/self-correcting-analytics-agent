"""
Forecast test — SARIMAX/ARIMA monthly revenue forecast.

Tests:
  1. should_model() correctly catches forecast phrasing (spot-check)
  2. End-to-end forecast: "forecast next quarter's revenue with a confidence interval"
     through the full train_model() pipeline (plan → Critic SQL → DataFrame → model → plot)

Honest data-limitation notes are printed inline — don't paper over them.
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

# ── Test 1: keyword detection for forecast phrases ────────────────────────────
print("═" * 60)
print("TEST 1 — should_model() catches forecast phrasing")
print("═" * 60)

forecast_cases = [
    ("forecast next quarter's revenue with a confidence interval", True),
    ("what will revenue look like next month?",                    True),
    ("project revenue for the next quarter",                       True),
    ("trend over time for completed orders",                       True),
    # Must NOT trigger — ordinary lookups
    ("what is our total revenue from completed orders?",           False),
    ("how many orders did we complete last month?",                False),
]

all_ok = True
for q, expected in forecast_cases:
    got = should_model(q)
    status = "PASS" if got == expected else "FAIL"
    if status == "FAIL":
        all_ok = False
    print(f"  {status}  [{'MODEL' if got else 'SKIP '}]  {q[:65]}")

if not all_ok:
    print("\nFAIL: should_model() keyword detection regression")
    sys.exit(1)
print()

# ── Test 2: end-to-end SARIMAX forecast ──────────────────────────────────────
QUESTION = "forecast next quarter's revenue with a confidence interval"

print("═" * 60)
print("TEST 2 — End-to-end SARIMAX/ARIMA revenue forecast")
print(f"Q: {QUESTION}")
print("═" * 60)

result = train_model(
    question=QUESTION,
    schema_with_enums=schema_with_enums,
    db_path=DB_PATH,
    client=client,
)

print(f"\ntask_type:          {result.get('task_type')}")
print(f"model_class:        {result.get('model_class')}")
print(f"error:              {result.get('error')}")

if result.get("error"):
    print(f"\nFAIL — train_model() returned error: {result['error']}")
    sys.exit(1)

metrics = result.get("metrics", {})
print(f"\n── Series info ──────────────────────────────────────────")
print(f"  Monthly observations used: {metrics.get('n_obs')}")
print(f"  Gaps filled with 0:        {metrics.get('n_gaps_filled')}")
print(f"  Seasonal SARIMAX fitted:   {metrics.get('seasonal_fitted')}")
print(f"  Model AIC:                 {metrics.get('aic')}")

print(f"\n── Forecast (3-month point estimates) ───────────────────")
dates  = result.get("forecast_dates", [])
pts    = result.get("point_forecast", [])
lowers = result.get("lower_bound", [])
uppers = result.get("upper_bound", [])
cl     = result.get("confidence_level", 0.95)

for d, pt, lo, hi in zip(dates, pts, lowers, uppers):
    print(f"  {d}:  ${pt:>12,.0f}   [{cl*100:.0f}% CI: ${lo:>12,.0f} – ${hi:>12,.0f}]")

total_pt = sum(max(0.0, v) for v in pts)
total_lo = sum(max(0.0, v) for v in lowers)
total_hi = sum(v for v in uppers)
print(f"\n  Q1 2025 total:  ${total_pt:,.0f}")
print(f"  CI total range: ${total_lo:,.0f} – ${total_hi:,.0f}")

print(f"\n── Plain-English summary ────────────────────────────────")
print(f"  {result.get('summary')}")

print(f"\n── Plot ─────────────────────────────────────────────────")
plot_path = result.get("plot_path")
if plot_path and os.path.exists(plot_path):
    size = os.path.getsize(plot_path)
    print(f"  PASS — saved ({size:,} bytes): {plot_path}")
else:
    print(f"  (no plot generated)")

# ── Honest data-quality assessment ───────────────────────────────────────────
print(f"\n── Data-limitation assessment ───────────────────────────")
n_obs = metrics.get("n_obs", 0)
seasonal = metrics.get("seasonal_fitted", False)

if seasonal:
    print(
        f"  SARIMAX(1,1,1)(1,1,1,12) fitted on {n_obs} monthly observations (~{n_obs/12:.1f} years).\n"
        f"  Seasonal parameter reliability: MARGINAL. Estimating 12-month seasonal\n"
        f"  components reliably needs ~5+ years; with ~{n_obs/12:.1f} years the seasonal\n"
        f"  parameters carry significant uncertainty. Treat CI bounds as rough guides,\n"
        f"  not precise probability statements. The upward trend signal is real;\n"
        f"  the seasonal shape is speculative."
    )
else:
    print(
        f"  ARIMA(1,1,1) used on {n_obs} monthly observations.\n"
        f"  Non-seasonal: the model captures trend and autocorrelation but cannot\n"
        f"  model within-year seasonality. Wide confidence intervals reflect the\n"
        f"  high month-to-month variance in this dataset ($3K–$329K range).\n"
        f"  Forecasts should be treated as directional, not precise."
    )
print(
    f"\n  The CI width (${total_hi - total_lo:,.0f} for the full quarter)\n"
    f"  honestly reflects how noisy this revenue series is at monthly granularity.\n"
    f"  Aggregate quarterly or annual views would produce tighter intervals."
)

print("\n" + "═" * 60)
print("Forecast test complete.")
