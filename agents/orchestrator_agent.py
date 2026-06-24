"""
Orchestrator Agent — makes pipeline routing decisions explicit and intelligent.

plan()         — calls Claude to decide which agents are needed for a question.
run_pipeline() — executes the full agent sequence based on the plan; drop-in
                 replacement for the orchestration logic previously in ask().
"""

import json
import sqlite3

import anthropic

from agents.chart_agent import create_chart
from agents.critic_agent import review_sql
from agents.correction_log import log_correction, retrieve_similar_corrections
from agents.explainer_agent import explain
from agents.model_agent import should_model, train_model

MODEL = "claude-sonnet-4-6"
MAX_ATTEMPTS = 3  # initial generation + up to 2 corrective regenerations

_PLAN_SYSTEM = """\
You are a routing agent for a B2B SaaS analytics system. Given a business question
and a brief schema summary, decide which downstream agents are needed.

Return ONLY a JSON object — no markdown fences, no explanation:
{"needs_sql": true, "needs_chart": <bool>, "needs_model": <bool>, "reasoning": "<one sentence>"}

needs_sql is always true — every question in this system requires SQL.

needs_chart is true when the question implies comparing multiple categories or
displaying a ranked list visually:
  TRUE signals: "breakdown by", "compare", "by segment", "by region",
  "by category", "top N", "top 5", "show me", "which categories", "how does X
  compare", "distribution", "over time".
  FALSE signals: "how many", "what is the total", "what is the average",
  "which one", "who has the most", "is our", single-value lookups.

needs_model is true ONLY for explicitly predictive, forecasting, or unsupervised-learning questions.
  TRUE signals: "predict", "forecast", "likely to", "classify", "cluster",
  "regression", "machine learning", "segment customers by behavior", "churn prediction",
  "next quarter", "next month", "trend over time", "confidence interval", "project revenue".
  FALSE for all descriptive, aggregation, or lookup questions — the vast majority.

reasoning is one sentence for logging only — never shown to users."""

_SQL_SYSTEM = """\
You are a senior SQL analyst for a B2B SaaS company.
Given a SQLite database schema and a business question, write exactly ONE SQLite
SQL query that answers the question. Return ONLY the raw SQL — no markdown fences,
no explanation, no trailing semicolon."""

_EXTRACT_SYSTEM = """\
You are a data analyst. A SQL query was executed and returned rows of data.
Given the original business question and the raw rows, extract the final answer
and return it as a JSON object with a single key "answer" whose value is:
  - a number (int or float) for count / sum / average questions
  - a string for "which one" questions
  - an ordered list of strings for "top N" / ranking questions
  - a dict mapping group name (string) to value (float) for "breakdown by X" questions
Return ONLY the JSON object. No explanation."""


def plan(question: str, schema_context: str, client: anthropic.Anthropic) -> dict:
    """
    Analyze the question and decide which agents are needed.
    Returns {"needs_sql": bool, "needs_chart": bool, "needs_model": bool,
             "reasoning": str, "tokens_used": int}.
    Fails safe on parse error: needs_chart and needs_model default to False.
    """
    resp = client.messages.create(
        model=MODEL,
        max_tokens=128,
        temperature=0,
        system=_PLAN_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Schema (truncated):\n{schema_context[:400]}\n\nQuestion: {question}",
            }
        ],
    )
    tokens = resp.usage.input_tokens + resp.usage.output_tokens

    try:
        result = json.loads(resp.content[0].text.strip())
        result["tokens_used"] = tokens
        return result
    except (json.JSONDecodeError, AttributeError):
        return {
            "needs_sql": True,
            "needs_chart": False,
            "needs_model": False,
            "reasoning": "parse_error — defaulting to no chart/model",
            "tokens_used": tokens,
        }


def run_pipeline(
    question: str,
    schema: str,
    schema_with_enums: str,
    db_path: str,
    client: anthropic.Anthropic,
    _status=None,
    on_step=None,
) -> dict:
    """
    Full agent pipeline: Plan → RAG → SQL+Critic loop → Execute → Extract → Explain
    → conditionally Chart.

    Returns the same dict shape as the previous ask() so all callers are unaffected.
    _status: optional callable(str) for the simple live progress bar.
    on_step: optional callable(agent, status, detail) for the detailed step log.
    """
    def _step(msg: str) -> None:
        if _status:
            _status(msg)

    def _emit(agent: str, status: str, detail: str) -> None:
        if on_step:
            on_step(agent, status, detail)

    tokens = 0
    retries = 0
    critic_warnings = None

    # ── Step 1: Plan ─────────────────────────────────────────────────────────
    _step("Planning query...")
    _emit("Orchestrator", "running", "")
    plan_result = plan(question, schema, client)
    tokens += plan_result.get("tokens_used", 0)
    needs_chart = plan_result.get("needs_chart", False)
    needs_model = plan_result.get("needs_model", False) or should_model(question)
    _emit("Orchestrator", "done",
          f"Chart: {'Yes' if needs_chart else 'No'} | "
          f"Model: {'Yes' if needs_model else 'No'}")

    # Schema loaded at app startup — emit table count immediately
    _schema_tables = [
        line.strip().split("(")[0]
        for line in schema.splitlines()
        if line.strip() and "(" in line
    ]
    _emit("Schema Agent", "done", f"{len(_schema_tables)} tables loaded")

    # ── Step 2: RAG retrieval ─────────────────────────────────────────────────
    _emit("RAG", "running", "")
    similar_corrections = []
    try:
        similar_corrections = retrieve_similar_corrections(question, "sql", k=3)
    except Exception:
        pass  # RAG failure must never block the main pipeline
    n_rag = len(similar_corrections)
    if n_rag > 0:
        first_reason = similar_corrections[0].get("error_or_reason", "")
        category = first_reason.split(":")[0].strip() if ":" in first_reason else ""
        hint = f" (e.g. {category})" if category else ""
        _emit("RAG", "done",
              f"{n_rag} past correction{'s' if n_rag != 1 else ''} retrieved{hint}")
    else:
        _emit("RAG", "done", "0 past corrections found (cold start)")

    system = _SQL_SYSTEM
    if similar_corrections:
        examples = [
            f'Past question: "{s["question"]}"\n'
            f'Wrong SQL: {s["generated_code"]}\n'
            f'Issue: {s["error_or_reason"]}\n'
            f'Corrected SQL: {s["corrected_code"]}'
            for s in similar_corrections
        ]
        rag_block = (
            "Here are similar past questions where the first attempt was wrong "
            "and how it was corrected:\n\n" + "\n\n".join(examples)
        )
        system = f"{_SQL_SYSTEM}\n\n{rag_block}"

    # ── Step 3: SQL generation + Critic self-correction loop ──────────────────
    messages = [{"role": "user", "content": f"{schema}\n\nQuestion: {question}"}]
    sql = None
    prev_rejected_sql = None
    prev_rejection_reason = None
    review = {"passed": True, "reason": "", "tokens_used": 0}  # safe default before loop

    for attempt in range(MAX_ATTEMPTS):
        if attempt == 0:
            _step("Generating SQL...")
            _emit("SQL Agent", "running", "")
        else:
            _step(f"Revising SQL (attempt {attempt + 1})...")
            # Show the rejection reason from the previous Critic on the retry line
            _emit("SQL Agent", "retry",
                  f"Attempt {attempt + 1} (rejected: {prev_rejection_reason})")

        gen_resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=system,
            messages=messages,
        )
        sql = gen_resp.content[0].text.strip()
        tokens += gen_resp.usage.input_tokens + gen_resp.usage.output_tokens

        if attempt == 0:
            _emit("SQL Agent", "done", "Query generated")

        _step("Reviewing SQL...")
        _emit("Critic", "running", "")
        review = review_sql(sql, schema_with_enums, client, question=question)
        tokens += review.get("tokens_used", 0)

        if review["passed"]:
            logged = False
            if prev_rejected_sql is not None:
                try:
                    log_correction(
                        question, "sql",
                        prev_rejected_sql, prev_rejection_reason, sql,
                    )
                    logged = True
                except Exception:
                    pass
            attempt_str = "first attempt" if attempt == 0 else f"attempt {attempt + 1}"
            suffix = " (1 correction logged to RAG)" if logged else ""
            _emit("Critic", "done", f"Passed on {attempt_str}{suffix}")
            break

        critic_warnings = review["reason"]
        prev_rejected_sql = sql
        prev_rejection_reason = critic_warnings

        if attempt == MAX_ATTEMPTS - 1:
            _emit("Critic", "retry",
                  f"FAILED after {MAX_ATTEMPTS} attempts: {critic_warnings}")
            break

        _emit("Critic", "retry", "Rejected")
        retries += 1
        messages.append({"role": "assistant", "content": sql})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Your SQL has an issue that must be fixed:\n{critic_warnings}\n\n"
                    "Rewrite the SQL to fix this exact issue. "
                    "Return ONLY the corrected SQL."
                ),
            }
        )

    # ── Guard: never execute SQL the Critic rejected on all attempts ─────────
    if not review["passed"]:
        return {
            "answer": None,
            "explanation": None,
            "sql": sql,
            "raw_result": None,
            "tokens_used": tokens,
            "retries": retries,
            "rag_hits": len(similar_corrections),
            "critic_warnings": critic_warnings,
            "error": (
                f"Couldn't generate a valid query after {MAX_ATTEMPTS} attempts "
                f"(last issue: {critic_warnings}). Try rephrasing your question."
            ),
            "chart_bytes": None,
            "model_result": None,
        }

    # ── Step 4: Execute SQL ───────────────────────────────────────────────────
    _step("Executing query...")
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(sql).fetchall()
        conn.close()
    except Exception as exc:
        return {
            "answer": None,
            "explanation": None,
            "sql": sql,
            "raw_result": None,
            "tokens_used": tokens,
            "retries": retries,
            "rag_hits": len(similar_corrections),
            "critic_warnings": critic_warnings,
            "error": str(exc),
            "chart_bytes": None,
            "model_result": None,
        }

    # ── Step 5: Extract typed answer ──────────────────────────────────────────
    _step("Interpreting results...")
    raw_preview = json.dumps(rows[:20], default=str)
    extract_resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=_EXTRACT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\n\nSQL result rows: {raw_preview}",
            }
        ],
    )
    tokens += extract_resp.usage.input_tokens + extract_resp.usage.output_tokens

    try:
        extract_text = extract_resp.content[0].text.strip()
        # Strip accidental markdown fences the model sometimes adds
        if extract_text.startswith("```"):
            extract_text = "\n".join(
                line for line in extract_text.splitlines()
                if not line.startswith("```")
            ).strip()
        answer = json.loads(extract_text).get("answer")
        exec_error = None
    except (json.JSONDecodeError, AttributeError):
        answer = None
        exec_error = "failed to parse answer JSON"

    # ── Step 6: Explain ───────────────────────────────────────────────────────
    explanation = None
    if answer is not None:
        _emit("Explainer", "running", "")
        try:
            explanation = explain(question, sql, answer, client)
            _emit("Explainer", "done", "Narrative ready")
        except Exception:
            _emit("Explainer", "done", "Skipped — generation error")

    # ── Step 7: Chart — gated by plan AND data-shape safety check ────────────
    chart_bytes = None
    if answer is not None and needs_chart:
        _step("Generating chart...")
        try:
            chart_bytes = create_chart(question, answer, "charts", client, raw_result=rows)
        except Exception:
            pass
        if chart_bytes:
            chart_type = "Horizontal bar" if isinstance(answer, list) else "Bar"
            _emit("Chart Agent", "done", f"{chart_type} chart created and Critic-reviewed")

    # ── Step 8: Model — separate path, gated by plan AND keyword safety check ─
    model_result = None
    if needs_model:
        try:
            model_result = train_model(question, schema_with_enums, db_path, client)
        except Exception:
            model_result = None
        if model_result and not model_result.get("error"):
            task = model_result.get("task_type", "")
            metrics = model_result.get("metrics", {})
            if task == "regression":
                r2 = metrics.get("r2", 0)
                detail = f"Regression complete (R²: {r2:.2f})"
            elif task == "classification":
                acc = metrics.get("accuracy", 0)
                detail = f"Classification complete ({acc * 100:.0f}% accuracy)"
            elif task == "clustering":
                n = metrics.get("n_clusters", 0)
                sil = metrics.get("silhouette_score", 0)
                detail = f"Clustering complete ({n} clusters, silhouette: {sil:.2f})"
            elif task == "forecast":
                mdl = metrics.get("model", "") or model_result.get("model_class", "")
                detail = f"Forecast complete ({mdl})"
            else:
                detail = "Complete"
            _emit("Model Agent", "done", detail)

    return {
        "answer": answer,
        "explanation": explanation,
        "sql": sql,
        "raw_result": rows,
        "tokens_used": tokens,
        "retries": retries,
        "rag_hits": len(similar_corrections),
        "critic_warnings": critic_warnings,
        "error": exec_error,
        "chart_bytes": chart_bytes,
        "model_result": model_result,
    }
