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
})

_MODEL_PLAN_SYSTEM = """\
You are a machine learning planning agent for a B2B SaaS analytics database.
Given a business question and database schema, produce a concrete ML training plan.

Return ONLY a JSON object — no markdown fences, no explanation:
{
  "task_type": "regression" | "classification" | "clustering",
  "sql": "<valid SQLite SELECT query — no semicolons, no markdown>",
  "model_class": "LinearRegression" | "RandomForestRegressor" | "LogisticRegression" | "RandomForestClassifier" | "KMeans",
  "target_column": "<column alias that will appear in SELECT results, null for clustering>",
  "feature_columns": ["<column alias 1>", "<column alias 2>", ...],
  "n_clusters": <int 3-5 for KMeans, null otherwise>
}

Rules:
- SQL column aliases must exactly match the feature_columns and target_column names you list.
- For "predict order value based on quantity and category": SELECT quantity, category,
  ROUND(quantity * unit_price, 2) AS order_value ... (alias the computed target).
- Prefer RandomForest over linear models (better handles nonlinearity and categoricals).
- Use LogisticRegression only if the target is binary. Use KMeans for "cluster/group" questions.
- Include WHERE orders.status = 'Completed' for any revenue/order-value target.
- SQL must be a bare SELECT — no markdown, no semicolon."""


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

    # Resolve feature_columns to what's actually in the DataFrame
    available = set(df.columns)
    features = [c for c in feature_columns if c in available]
    if not features:
        # Fallback: all numeric columns except target
        features = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    if not features:
        return _err("no usable feature columns found in query result", task_type=task_type, sql=sql)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    try:
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


def _err(message, task_type=None, sql=None):
    return {
        "task_type": task_type, "metrics": {}, "plot_path": None,
        "summary": message, "sql": sql, "model_class": None, "error": message,
    }
