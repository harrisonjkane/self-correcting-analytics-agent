"""Model Agent — trains sklearn models on database data for predictive questions."""

import json
import os
import sqlite3
import uuid

import anthropic
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, r2_score, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

from agents.critic_agent import review_sql

MODEL = "claude-sonnet-4-6"

_SKLEARN_MODELS = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier,
    "KMeans": KMeans,
}

_MODELING_KEYWORDS = frozenset({
    "predict", "forecast", "likely to", "classify", "cluster",
    "group customers by behavior", "segment customers by", "regression",
    "churn prediction", "machine learning", "ml model", "identify patterns",
    "propensity", "probability of",
    # forecast-specific
    "next month", "next quarter", "next year", "trend over time",
    "predict future", "confidence interval", "project revenue",
})

_MODEL_PLAN_SYSTEM = """\
You are a machine learning planning agent for a B2B SaaS analytics database.
Given a business question and database schema, produce a concrete ML training plan.

Return ONLY a JSON object — no markdown fences, no explanation — with ALL of these keys
(use null for keys that do not apply to the chosen task type):
{
  "task_type": "regression" | "classification" | "clustering" | "forecast",
  "sql": "<valid SQLite SELECT query — no semicolons, no markdown>",
  "model_class": "LinearRegression" | "RandomForestRegressor" | "LogisticRegression" | "RandomForestClassifier" | "KMeans" | "SARIMAX",
  "target_column": "<column alias for supervised target, null for clustering/forecast>",
  "feature_columns": ["<alias>", ...],
  "date_column": "<period alias for time-series, null for non-forecast tasks>",
  "value_column": "<metric alias for time-series, null for non-forecast tasks>",
  "n_clusters": <int 3-5 for KMeans, null otherwise>
}

Rules for ALL tasks:
- SQL column aliases must exactly match every column name you list in the JSON.
- SQL must be a bare SELECT — no markdown, no semicolons.
- Include WHERE orders.status = 'Completed' for any revenue/order-value computation.

Rules for FORECAST tasks (task_type = "forecast"):
- SQL must return exactly two columns: (date_period, numeric_value) ordered by date_period ASC.
- Use strftime('%Y-%m', date_col) for monthly aggregation, aliased as "period".
- Alias the aggregated value (e.g. SUM of revenue) as "revenue" or another clear name.
- Set model_class = "SARIMAX", target_column = null, feature_columns = [].
- Set date_column to the period alias and value_column to the metric alias.
- Example: SELECT strftime('%Y-%m', o.order_date) AS period,
           ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue
           FROM order_items oi JOIN orders o ON oi.order_id = o.order_id
           WHERE o.status = 'Completed' GROUP BY 1 ORDER BY 1

Rules for REGRESSION/CLASSIFICATION tasks:
- Prefer RandomForest over linear models.
- Use LogisticRegression only for binary targets.

Rules for CLUSTERING:
- Use KMeans, n_clusters 3-5."""


def should_model(question: str) -> bool:
    """Conservative keyword check for predictive intent. Returns False by default."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in _MODELING_KEYWORDS)


def _plan_model(question: str, schema_context: str, client: anthropic.Anthropic) -> dict:
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0,
        system=_MODEL_PLAN_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Schema:\n{schema_context}\n\nQuestion: {question}",
            }
        ],
    )
    raw = resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
    return json.loads(raw)


def _plot_regression(y_true: np.ndarray, y_pred: np.ndarray, chart_dir: str) -> str | None:
    os.makedirs(chart_dir, exist_ok=True)
    path = os.path.join(chart_dir, f"{uuid.uuid4().hex}.png")
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(y_true, y_pred, alpha=0.45, color="#2E75B6", edgecolors="white", linewidth=0.3)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Perfect fit")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs. Predicted")
        ax.legend()
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=150)
    finally:
        plt.close("all")
    return path if os.path.exists(path) else None


def _plot_clusters(X_scaled: np.ndarray, labels: np.ndarray, chart_dir: str) -> str | None:
    os.makedirs(chart_dir, exist_ok=True)
    path = os.path.join(chart_dir, f"{uuid.uuid4().hex}.png")
    try:
        n_components = min(2, X_scaled.shape[1])
        if n_components < 2:
            return None
        X_2d = PCA(n_components=2).fit_transform(X_scaled)
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.colorbar(sc, ax=ax, label="Cluster")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("Cluster Distribution (PCA Projection)")
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=150)
    finally:
        plt.close("all")
    return path if os.path.exists(path) else None


def train_model(
    question: str,
    schema_with_enums: str,
    db_path: str,
    client: anthropic.Anthropic,
) -> dict:
    """
    Plan → Critic-reviewed SQL → DataFrame → sklearn → metrics + plot + summary.
    Returns {"task_type", "metrics", "plot_path", "summary", "sql", "model_class", "error"}.
    """
    # ── Step 1: Plan ─────────────────────────────────────────────────────────
    try:
        plan = _plan_model(question, schema_with_enums, client)
    except Exception as exc:
        return _err(f"model planning failed: {exc}")

    sql = plan.get("sql", "")
    task_type = plan.get("task_type", "regression")
    model_class_name = plan.get("model_class", "RandomForestRegressor")
    feature_columns = plan.get("feature_columns") or []
    target_column = plan.get("target_column")
    n_clusters = plan.get("n_clusters") or 3
    date_column = plan.get("date_column") or "period"
    value_column = plan.get("value_column") or "revenue"

    # ── Step 2: Critic review of training SQL ────────────────────────────────
    review = review_sql(sql, schema_with_enums, client, question=question)
    if not review["passed"]:
        return _err(f"SQL review failed: {review['reason']}", task_type=task_type, sql=sql)

    # ── Step 3: Execute SQL → DataFrame ──────────────────────────────────────
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(sql)
        col_names = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
    except Exception as exc:
        return _err(f"SQL execution failed: {exc}", task_type=task_type, sql=sql)

    df = pd.DataFrame(rows, columns=col_names)
    if df.empty:
        return _err("training query returned no rows", task_type=task_type, sql=sql)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    try:
        if task_type == "forecast":
            steps, alpha = _parse_forecast_params(question)
            return _train_forecast(df, date_column, value_column, sql, question, steps, alpha)

        # For supervised/unsupervised tasks: resolve feature columns
        available = set(df.columns)
        features = [c for c in feature_columns if c in available]
        if not features:
            features = [c for c in df.select_dtypes(include=[np.number]).columns
                        if c != target_column]
        if not features:
            return _err("no usable feature columns found in query result",
                        task_type=task_type, sql=sql)

        if task_type == "regression":
            return _train_regression(df, features, target_column, model_class_name, sql)
        elif task_type == "classification":
            return _train_classification(df, features, target_column, model_class_name, sql)
        else:
            return _train_clustering(df, features, int(n_clusters), sql)
    except Exception as exc:
        return _err(f"model training failed: {exc}", task_type=task_type, sql=sql)


# ── Private training helpers ──────────────────────────────────────────────────

def _train_regression(df, features, target_column, model_class_name, sql):
    if target_column not in df.columns:
        return _err(f"target column '{target_column}' not in query results",
                    task_type="regression", sql=sql)

    X = pd.get_dummies(df[features], drop_first=True).fillna(0).astype(float)
    y = pd.to_numeric(df[target_column], errors="coerce").dropna()
    X = X.loc[y.index]

    if len(X) < 10:
        return _err(f"insufficient data for regression ({len(X)} rows)", task_type="regression", sql=sql)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ModelClass = _SKLEARN_MODELS.get(model_class_name, RandomForestRegressor)
    model = (ModelClass(n_estimators=100, random_state=42)
             if "Forest" in model_class_name else ModelClass())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    metrics = {
        "r2": round(r2, 4),
        "mae": round(mae, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    summary = (
        f"A {model_class_name} trained on {len(X_train)} samples explains "
        f"{r2 * 100:.1f}% of variance in {target_column} "
        f"(R²={r2:.3f}, MAE={mae:,.2f})."
    )
    plot_path = _plot_regression(y_test.values, y_pred, "charts")

    return {
        "task_type": "regression", "metrics": metrics,
        "plot_path": plot_path, "summary": summary,
        "sql": sql, "model_class": model_class_name, "error": None,
    }


def _train_classification(df, features, target_column, model_class_name, sql):
    if target_column not in df.columns:
        return _err(f"target column '{target_column}' not in query results",
                    task_type="classification", sql=sql)

    X = pd.get_dummies(df[features], drop_first=True).fillna(0).astype(float)
    y = df[target_column].astype(str)

    if len(X) < 10:
        return _err(f"insufficient data ({len(X)} rows)", task_type="classification", sql=sql)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ModelClass = _SKLEARN_MODELS.get(model_class_name, RandomForestClassifier)
    if "Forest" in model_class_name:
        model = ModelClass(n_estimators=100, random_state=42)
    elif model_class_name == "LogisticRegression":
        model = ModelClass(max_iter=1000, random_state=42)
    else:
        model = ModelClass()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    metrics = {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    summary = (
        f"A {model_class_name} achieved {acc * 100:.1f}% accuracy and "
        f"{f1:.3f} weighted F1 on {len(X_test)} test samples "
        f"predicting {target_column}."
    )

    return {
        "task_type": "classification", "metrics": metrics,
        "plot_path": None, "summary": summary,
        "sql": sql, "model_class": model_class_name, "error": None,
    }


def _train_clustering(df, features, n_clusters, sql):
    X = pd.get_dummies(df[features], drop_first=True).fillna(0).astype(float)

    if len(X) < n_clusters * 2:
        return _err(f"insufficient data for {n_clusters} clusters ({len(X)} rows)",
                    task_type="clustering", sql=sql)

    X_scaled = StandardScaler().fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = model.fit_predict(X_scaled)
    sil = float(silhouette_score(X_scaled, labels)) if n_clusters > 1 else 0.0

    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil, 4),
        "n_samples": len(df),
    }
    summary = (
        f"KMeans identified {n_clusters} distinct clusters across {len(df)} samples "
        f"with a silhouette score of {sil:.3f} "
        f"({'good separation' if sil > 0.5 else 'moderate separation' if sil > 0.25 else 'weak separation'})."
    )
    plot_path = _plot_clusters(X_scaled, labels, "charts")

    return {
        "task_type": "clustering", "metrics": metrics,
        "plot_path": plot_path, "summary": summary,
        "sql": sql, "model_class": "KMeans", "error": None,
    }


def _parse_forecast_params(question: str) -> tuple[int, float]:
    """Parse (steps, alpha) from question phrasing. Defaults: 3 steps, 0.05 alpha (95% CI)."""
    q = question.lower()
    steps = 3  # default: one quarter
    if "next month" in q:
        steps = 1
    elif "next year" in q or "12 month" in q:
        steps = 12
    elif "6 month" in q or "six month" in q or "half year" in q:
        steps = 6
    # "next quarter" / "q1/q2/q3/q4" stays at default 3

    alpha = 0.05  # default: 95% CI
    if "90%" in q or "90 percent" in q:
        alpha = 0.10
    elif "99%" in q or "99 percent" in q:
        alpha = 0.01

    return steps, alpha


def _plot_forecast(
    series: pd.Series,
    forecast_dates: list[str],
    point_forecast: list[float],
    lower_bound: list[float],
    upper_bound: list[float],
    chart_dir: str,
    confidence_level: float = 0.95,
) -> str | None:
    os.makedirs(chart_dir, exist_ok=True)
    path = os.path.join(chart_dir, f"{uuid.uuid4().hex}.png")
    try:
        n_show = min(18, len(series))
        hist = series.iloc[-n_show:]
        fc_dates = pd.to_datetime([d + "-01" for d in forecast_dates])
        cl_pct = int(confidence_level * 100)

        fig, ax = plt.subplots(figsize=(10, 5))

        # Historical line
        ax.plot(hist.index, hist.values, color="#2E75B6", lw=2, label="Historical")

        # Bridge from last historical point to first forecast point
        ax.plot(
            [hist.index[-1], fc_dates[0]],
            [hist.iloc[-1], point_forecast[0]],
            color="#2E75B6", lw=2, linestyle="--",
        )
        # Forecast line
        ax.plot(fc_dates, point_forecast, color="#2E75B6", lw=2,
                linestyle="--", label="Forecast")
        # Confidence band (forecast portion only)
        ax.fill_between(fc_dates, lower_bound, upper_bound,
                        color="#2E75B6", alpha=0.18, label=f"{cl_pct}% CI")

        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue ($)")
        ax.set_title("Monthly Revenue Forecast")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        ax.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(path, bbox_inches="tight", dpi=150)
    finally:
        plt.close("all")
    return path if os.path.exists(path) else None


def _train_forecast(
    df: pd.DataFrame,
    date_column: str,
    value_column: str,
    sql: str,
    question: str,
    steps: int = 3,
    alpha: float = 0.05,
) -> dict:
    import warnings

    if date_column not in df.columns or value_column not in df.columns:
        missing = [c for c in (date_column, value_column) if c not in df.columns]
        return _err(f"columns {missing} not found in query result (got {list(df.columns)})",
                    task_type="forecast", sql=sql)

    # Build a complete monthly DatetimeIndex, fill any gaps with 0
    df[date_column] = pd.to_datetime(df[date_column].astype(str) + "-01")
    df[value_column] = pd.to_numeric(df[value_column], errors="coerce").fillna(0.0)
    series = df.set_index(date_column)[value_column].sort_index()
    full_idx = pd.date_range(series.index.min(), series.index.max(), freq="MS")
    n_gaps = len(full_idx) - len(series)
    series = series.reindex(full_idx, fill_value=0.0)

    n_obs = len(series)
    confidence_level = 1.0 - alpha

    # ── Model selection ───────────────────────────────────────────────────────
    # Seasonal SARIMAX(1,1,1)(1,1,1,12) needs at least 2 full seasonal cycles
    # plus headroom for differencing: 2×12 + 10 = 34 obs minimum.
    MIN_FOR_SEASONAL = 34
    seasonal_attempted = n_obs >= MIN_FOR_SEASONAL
    model_used = None
    fallback_reason = None

    if seasonal_attempted:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(
                    series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 12),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False, maxiter=300)
            model_used = "SARIMAX(1,1,1)(1,1,1,12)"
        except Exception as exc:
            fallback_reason = f"SARIMAX seasonal fit failed ({exc}); fell back to ARIMA(1,1,1)"
            seasonal_attempted = False

    if not seasonal_attempted:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = SARIMAX(
                    series,
                    order=(1, 1, 1),
                    seasonal_order=(0, 0, 0, 0),
                ).fit(disp=False, maxiter=300)
            model_used = "ARIMA(1,1,1)"
            if fallback_reason is None:
                fallback_reason = (
                    f"non-seasonal ARIMA used: only {n_obs} monthly observations "
                    f"(≥{MIN_FOR_SEASONAL} required for seasonal estimation)"
                )
        except Exception as exc:
            return _err(f"ARIMA fit failed: {exc}", task_type="forecast", sql=sql)

    # ── Forecast ──────────────────────────────────────────────────────────────
    try:
        fc_obj = fit.get_forecast(steps=steps)
        point_mean = fc_obj.predicted_mean
        ci = fc_obj.conf_int(alpha=alpha)
    except Exception as exc:
        return _err(f"forecast generation failed: {exc}", task_type="forecast", sql=sql)

    point_forecast = [round(float(v), 2) for v in point_mean.values]
    lower_bound    = [round(float(v), 2) for v in ci.iloc[:, 0].values]
    upper_bound    = [round(float(v), 2) for v in ci.iloc[:, 1].values]

    last_date = series.index[-1]
    forecast_dates = [
        (last_date + pd.DateOffset(months=i + 1)).strftime("%Y-%m")
        for i in range(steps)
    ]

    # ── Summary ───────────────────────────────────────────────────────────────
    cl_pct = int(confidence_level * 100)
    if steps == 1:
        summary = (
            f"We project {forecast_dates[0]} revenue of approximately "
            f"${point_forecast[0]:,.0f}, with {cl_pct}% confidence the actual "
            f"value will fall between ${lower_bound[0]:,.0f} and "
            f"${upper_bound[0]:,.0f}."
        )
    else:
        total_pt = sum(max(0, v) for v in point_forecast)
        total_lo = sum(max(0, v) for v in lower_bound)
        total_hi = sum(v for v in upper_bound)
        period = (
            f"{forecast_dates[0]} – {forecast_dates[-1]}"
            if steps > 1 else forecast_dates[0]
        )
        summary = (
            f"We project {period} total revenue of approximately ${total_pt:,.0f}, "
            f"with {cl_pct}% confidence the actual value will fall between "
            f"${total_lo:,.0f} and ${total_hi:,.0f}. "
            f"Model: {model_used} fitted on {n_obs} monthly observations."
        )
    if fallback_reason:
        summary += f" Note: {fallback_reason}."

    aic = round(float(fit.aic), 2) if hasattr(fit, "aic") else None

    plot_path = _plot_forecast(
        series, forecast_dates, point_forecast, lower_bound, upper_bound,
        "charts", confidence_level,
    )

    return {
        "task_type": "forecast",
        "point_forecast": point_forecast,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "confidence_level": confidence_level,
        "forecast_dates": forecast_dates,
        "plot_path": plot_path,
        "summary": summary,
        "sql": sql,
        "model_class": model_used,
        "metrics": {
            "n_obs": n_obs,
            "n_gaps_filled": n_gaps,
            "model": model_used,
            "aic": aic,
            "seasonal_fitted": model_used is not None and "SARIMAX" in model_used,
            "steps_ahead": steps,
            "confidence_level": confidence_level,
        },
        "error": None,
    }


def _err(message, task_type=None, sql=None):
    return {
        "task_type": task_type, "metrics": {}, "plot_path": None,
        "summary": message, "sql": sql, "model_class": None, "error": message,
    }
