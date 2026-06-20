"""
SQL agent — Phase 1 and Phase 2 implementations.

Phase 1: run_query()  — single-pass, no self-correction (self-contained).
Phase 2: ask()        — thin entry point; all routing and pipeline logic
                        lives in orchestrator_agent.run_pipeline().

Both functions return the same dict shape so the eval harness scores them
identically. Only 'retries', 'critic_warnings', and 'chart_path' differ.
"""

import json
import sqlite3

import anthropic

MODEL = "claude-sonnet-4-6"

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


def ask(
    question: str,
    schema: str,
    schema_with_enums: str,
    db_path: str,
    client: anthropic.Anthropic,
) -> dict:
    """
    Phase 2+ entry point — delegates all pipeline logic to the Orchestrator.
    Signature and return shape are unchanged; all callers are unaffected.
    """
    from agents.orchestrator_agent import run_pipeline  # lazy import avoids circular dep
    return run_pipeline(question, schema, schema_with_enums, db_path, client)


def run_query(question: str, schema: str, db_path: str, client: anthropic.Anthropic) -> dict:
    """Phase 1 baseline — single-pass SQL generation, no Critic or self-correction."""
    tokens = 0

    # ── Step 1: generate SQL ─────────────────────────────────────────────────
    gen_resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=_SQL_SYSTEM,
        messages=[
            {"role": "user", "content": f"{schema}\n\nQuestion: {question}"}
        ],
    )
    sql = gen_resp.content[0].text.strip()
    tokens += gen_resp.usage.input_tokens + gen_resp.usage.output_tokens

    # ── Step 2: execute SQL ──────────────────────────────────────────────────
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(sql).fetchall()
        conn.close()
        exec_error = None
    except Exception as exc:
        return {
            "answer": None,
            "sql": sql,
            "error": str(exc),
            "tokens_used": tokens,
            "retries": 0,
        }

    # ── Step 3: extract typed answer ─────────────────────────────────────────
    raw_preview = json.dumps(rows[:20], default=str)
    extract_resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=_EXTRACT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"SQL result rows: {raw_preview}"
                ),
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

    return {
        "answer": answer,
        "sql": sql,
        "raw_result": rows,
        "tokens_used": tokens,
        "retries": 0,
        "error": exec_error,
    }
