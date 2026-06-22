"""
SQL agent — Phase 1 and Phase 2 implementations.

Phase 1: run_query()  — single-pass, no self-correction (self-contained).
Phase 2: ask()        — thin entry point; all routing and pipeline logic
                        lives in orchestrator_agent.run_pipeline().

Both functions return the same dict shape so the eval harness scores them
identically. Only 'retries', 'critic_warnings', and 'chart_path' differ.
"""

import json
import re
import sqlite3

import anthropic
import pandas as pd

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


def _table_names_from_schema(schema: str) -> list[str]:
    """Parse table names out of the schema string returned by get_schema()."""
    names = []
    for line in schema.splitlines():
        m = re.match(r"\s+(\w+)\(", line)
        if m:
            names.append(m.group(1).lower())
    return names


def check_browse_request(question: str, schema: str, db_path: str) -> dict | None:
    """
    Detect table-browsing requests and execute them directly, bypassing the
    full pipeline.  Returns a result dict on match, None if not a browse request.

    Result dict keys:
      browse_df   — pandas DataFrame (on success)
      browse_error— human-readable error string (on table-not-found or DB error)
      sql         — the executed query (on success)
    """
    q = question.strip().lower()
    table_names = _table_names_from_schema(schema)

    limit = 10
    table = None

    # "first N rows of/from X"
    m = re.search(r"first\s+(\d+)\s+rows?\s+(?:of|from)\s+(\w+)", q)
    if m:
        limit = int(m.group(1))
        table = m.group(2)

    # "show me the X table" / "what's in the X table" / "what does the X table look like"
    if table is None:
        m = re.search(r"\b(\w+)\s+table\b", q)
        if m:
            table = m.group(1)

    # "show me some X data"
    if table is None:
        m = re.search(r"some\s+(\w+)\s+data\b", q)
        if m:
            table = m.group(1)

    if table is None:
        return None

    if table not in table_names:
        return {
            "browse_error": (
                f"Table '{table}' not found. "
                f"Available tables: {', '.join(table_names)}."
            )
        }

    sql = f"SELECT * FROM {table} LIMIT {limit}"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(sql)
        cols = [d[0] for d in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        return {
            "browse_df": pd.DataFrame(rows, columns=cols),
            "sql": sql,
        }
    except Exception as exc:
        return {"browse_error": str(exc)}


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
