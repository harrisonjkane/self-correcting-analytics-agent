"""Deterministic Model Agent logic: routing, forecast params, the conditional clamp."""
import pandas as pd
import pytest

from agents.model_agent import (
    _metric_can_be_negative,
    _parse_forecast_params,
    _train_forecast,
    should_model,
)

REVENUE_SQL = (
    "SELECT strftime('%Y-%m', o.order_date) AS period, "
    "ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
    "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
    "WHERE o.status = 'Completed' GROUP BY 1 ORDER BY 1"
)
DELTA_SQL = (
    "SELECT period, current_rev - prior_rev AS revenue_change "
    "FROM monthly_summary ORDER BY period"
)


@pytest.mark.parametrize(
    "question, expected",
    [
        ("forecast next quarter's revenue with a confidence interval", True),
        ("what will revenue look like next month?", True),
        ("project revenue for the next quarter", True),
        ("trend over time for completed orders", True),
        ("what is our total revenue from completed orders?", False),
        ("how many orders did we complete last month?", False),
    ],
)
def test_should_model(question, expected):
    assert should_model(question) is expected


# Regression test for the forecast clamp bug: lower bounds are clamped to 0 only
# for metrics that cannot go negative (revenue, counts). Deltas keep negatives.
@pytest.mark.parametrize(
    "question, sql, can_be_negative",
    [
        ("forecast next quarter's revenue with a confidence interval", REVENUE_SQL, False),
        ("forecast month-over-month revenue change for next quarter", REVENUE_SQL, True),
        ("project revenue delta over the next 3 months", REVENUE_SQL, True),
        ("forecast the difference in monthly revenue versus last year", REVENUE_SQL, True),
        ("forecast customer count for next quarter", REVENUE_SQL, False),
        ("predict next month's units sold", REVENUE_SQL, False),
        ("forecast revenue change year-over-year", REVENUE_SQL, True),
        ("forecast next quarter revenue", DELTA_SQL, True),  # SQL subtraction alone
    ],
)
def test_metric_can_be_negative(question, sql, can_be_negative):
    assert _metric_can_be_negative(question, sql) is can_be_negative


@pytest.mark.parametrize(
    "question, expected",
    [
        ("forecast next quarter", (3, 0.05)),
        ("forecast next month", (1, 0.05)),
        ("forecast next year with a 90% interval", (12, 0.10)),
        ("forecast 6 month revenue at 99%", (6, 0.01)),
    ],
)
def test_parse_forecast_params(question, expected):
    assert _parse_forecast_params(question) == expected


def test_short_history_fails_gracefully():
    df = pd.DataFrame({"period": ["2024-01", "2024-02"], "revenue": [10000.0, 12000.0]})
    result = _train_forecast(
        df=df, date_column="period", value_column="revenue",
        sql=REVENUE_SQL, question="forecast next quarter's revenue",
    )
    assert result.get("error")


def test_missing_column_fails_gracefully():
    df = pd.DataFrame({"period": ["2024-01"], "revenue": [1.0]})
    result = _train_forecast(
        df=df, date_column="month", value_column="revenue",
        sql=REVENUE_SQL, question="forecast revenue",
    )
    assert "not found" in result["error"]
