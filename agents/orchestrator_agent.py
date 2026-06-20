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
) -> dict:
    """
    Full agent pipeline: Plan → RAG → SQL+Critic loop → Execute → Extract → Explain
    → conditionally Chart.

    Returns the same dict shape as the previous ask() so all callers are unaffected.
    """
    tokens = 0
    retries = 0
    critic_warnings = None

    # ── Step 1: Plan ─────────────────────────────────────────────────────────
    plan_result = plan(question, schema, client)
    tokens += plan_result.get("tokens_used", 0)

    # ── Step 2: RAG retrieval ─────────────────────────────────────────────────
    similar_corrections = []
    try:
        similar_corrections = retrieve_similar_corrections(question, "sql", k=3)
    except Exception:
        pass  # RAG failure must never block the main pipeline

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

    for attempt in range(MAX_ATTEMPTS):
        gen_resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=system,
            messages=messages,
        )
        sql = gen_resp.content[0].text.strip()
        tokens += gen_resp.usage.input_tokens + gen_resp.usage.output_tokens

        review = review_sql(sql, schema_with_enums, client, question=question)
        tokens += review.get("tokens_used", 0)

        if review["passed"]:
            if prev_rejected_sql is not None:
                try:
                    log_correction(
                        question, "sql",
                        prev_rejected_sql, prev_rejection_reason, sql,
                    )
                except Exception:
                    pass
            break

        critic_warnings = review["reason"]
        prev_rejected_sql = sql
        prev_rejection_reason = critic_warnings

        if attempt == MAX_ATTEMPTS - 1:
            break

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

    # ── Step 4: Execute SQL ───────────────────────────────────────────────────
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
            "chart_path": None,
            "model_result": None,
        }

    # ── Step 5: Extract typed answer ──────────────────────────────────────────
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
        answer = json.loads(extract_resp.content[0].text.strip()).get("answer")
        exec_error = None
    except (json.JSONDecodeError, AttributeError):
        answer = None
        exec_error = "failed to parse answer JSON"

    # ── Step 6: Explain ───────────────────────────────────────────────────────
    explanation = None
    if answer is not None:
        try:
            explanation = explain(question, sql, answer, client)
        except Exception:
            pass  # explainer failure must never block the result

    # ── Step 7: Chart — gated by plan AND data-shape safety check ────────────
    chart_path = None
    if answer is not None and plan_result.get("needs_chart", False):
        try:
            # create_chart internally calls should_chart() as a second-layer check:
            # if the orchestrator said "chart" but the data shape can't support one,
            # should_chart() returns False and no chart is generated.
            chart_path = create_chart(question, answer, "charts", client, raw_result=rows)
        except Exception:
            pass  # chart failure must never block the result

    # ── Step 8: Model — separate path, gated by plan AND keyword safety check ─
    model_result = None
    needs_model = plan_result.get("needs_model", False) or should_model(question)
    if needs_model:
        try:
            model_result = train_model(question, schema_with_enums, db_path, client)
        except Exception:
            pass  # model failure must never block the result

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
        "chart_path": chart_path,
        "model_result": model_result,
    }
