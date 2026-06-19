"""
SQL agent — Phase 1 and Phase 2 implementations.

Phase 1: run_query()  — single-pass, no self-correction.
Phase 2: ask()        — generate → Critic review → self-correct loop → execute.

Both functions return the same dict shape so the eval harness scores them
identically. Only 'retries' and 'critic_warnings' differ between phases.
"""

import json
import sqlite3

import anthropic

from agents.critic_agent import review_sql

MODEL = "claude-sonnet-4-6"
MAX_ATTEMPTS = 3  # initial generation + up to 2 corrective regenerations

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
    Phase 2: generate SQL → Critic review → self-correct loop → execute → extract.

    schema            — basic schema for SQL generation (same as Phase 1).
    schema_with_enums — extended schema with stored enum values for the Critic.
    """
    tokens = 0
    retries = 0
    critic_warnings = None

    # Multi-turn message history so each regeneration has full correction context.
    messages = [{"role": "user", "content": f"{schema}\n\nQuestion: {question}"}]

    sql = None
    for attempt in range(MAX_ATTEMPTS):
        # ── Generate (or re-generate) SQL ────────────────────────────────────
        gen_resp = client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=_SQL_SYSTEM,
            messages=messages,
        )
        sql = gen_resp.content[0].text.strip()
        tokens += gen_resp.usage.input_tokens + gen_resp.usage.output_tokens

        # ── Critic review ─────────────────────────────────────────────────────
        review = review_sql(sql, schema_with_enums, client)
        tokens += review.get("tokens_used", 0)

        if review["passed"]:
            break

        critic_warnings = review["reason"]

        if attempt == MAX_ATTEMPTS - 1:
            # All attempts exhausted — execute the last SQL anyway.
            break

        # ── Prepare correction context for the next attempt ───────────────────
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

    # ── Execute SQL ───────────────────────────────────────────────────────────
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
            "retries": retries,
            "critic_warnings": critic_warnings,
        }

    # ── Extract typed answer ──────────────────────────────────────────────────
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

    return {
        "answer": answer,
        "sql": sql,
        "raw_result": rows,
        "tokens_used": tokens,
        "retries": retries,
        "critic_warnings": critic_warnings,
        "error": exec_error,
    }


def run_query(question: str, schema: str, db_path: str, client: anthropic.Anthropic) -> dict:
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
    raw_preview = json.dumps(rows[:20], default=str)  # cap at 20 rows for the prompt
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
        answer_json = json.loads(extract_resp.content[0].text.strip())
        answer = answer_json.get("answer")
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
