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
no explanation, no trailing semicolon.

SQLite date handling — date columns (order_date, signup_date, opened_date,
closed_date) are stored as ISO-8601 TEXT ('YYYY-MM-DD'). Use these patterns:
  - Monthly grouping:  strftime('%Y-%m', order_date)
  - Year grouping:     strftime('%Y', order_date)
  - Date filtering:    date(order_date) >= date('2023-01-01')
  - Ordering by date:  ORDER BY order_date  (ISO-8601 sorts lexicographically, so this is correct)
  - Age in days:       julianday('now') - julianday(order_date)
Never compare date columns as arbitrary strings; always use date() or strftime()."""

_EXTRACT_SYSTEM = """\
You are a data analyst. A SQL query was executed and returned rows of data.
Given the original business question and the raw rows, extract the final answer
and return it as a JSON object with a single key "answer" whose value is:
  - a number (int or float) for count / sum / average questions
  - a string for "which one" questions
  - an ordered list of strings for "top N" / ranking questions
  - a dict mapping group name (string) to value (float) for "breakdown by X" questions
Return ONLY the JSON object. No explanation."""


_BROWSE_INTENT = frozenset({
    "show", "print", "display", "list", "give", "fetch",
    "get", "preview", "sample", "see", "rows", "first", "last",
})
# Presence of any of these alongside a table name signals analytics, not browse
_ANALYTICS_OVERRIDE = frozenset({
    "by", "avg", "average", "sum", "total", "count", "rank",
    "top", "bottom", "compare", "trend", "group", "having",
    "predict", "forecast", "breakdown",
})
_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


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
    words = set(re.findall(r"\w+", q))
    table_names = _table_names_from_schema(schema)

    limit = 10
    table = None

    # ── Specific patterns (checked first; unambiguous) ────────────────────────

    # "first N rows of/from [the] X"  — optional "the" before table name
    m = re.search(r"first\s+(\d+)\s+rows?\s+(?:of|from)\s+(?:the\s+)?(\w+)", q)
    if m and m.group(2) not in ("table",):
        limit = int(m.group(1))
        table = m.group(2)

    # "[the] X table"  — also catches unknown-table requests for a helpful error
    if table is None:
        m = re.search(r"\b(\w+)\s+table\b", q)
        if m:
            candidate = m.group(1)
            if candidate not in ("the", "a", "this", "that"):
                if candidate in table_names:
                    table = candidate
                else:
                    return {
                        "browse_error": (
                            f"Table '{candidate}' not found. "
                            f"Available tables: {', '.join(table_names)}."
                        )
                    }

    # "some X data"  — single-word capture; only fires if X is a real table
    if table is None:
        m = re.search(r"some\s+(\w+)\s+data\b", q)
        if m and m.group(1) in table_names:
            table = m.group(1)

    # ── General intent detection ──────────────────────────────────────────────
    # Fire when: browse intent word present, no analytics signals, and a known
    # table name appears in the question (exact for single-word names; prefix
    # match for multi-word names like support_tickets vs "support ticket").
    if table is None:
        has_intent = bool(words & _BROWSE_INTENT)
        has_analytics = bool(words & _ANALYTICS_OVERRIDE)
        if has_intent and not has_analytics:
            for tn in table_names:
                parts = tn.split("_")
                if len(parts) == 1:
                    if tn in words:
                        table = tn
                        break
                else:
                    # All parts must appear; prefix match handles singular/plural
                    if all(
                        any(w == p or p.startswith(w) or w.startswith(p) for w in words)
                        for p in parts
                    ):
                        table = tn
                        break

    if table is None:
        return None

    # ── Extract limit (digit wins; fall back to word number) ──────────────────
    m = re.search(r"\b(\d+)\b", q)
    if m:
        limit = int(m.group(1))
    else:
        for word, num in _WORD_NUMS.items():
            if word in words:
                limit = num
                break

    # ── Execute ───────────────────────────────────────────────────────────────
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
    _status=None,
    on_step=None,
) -> dict:
    """
    Phase 2+ entry point — delegates all pipeline logic to the Orchestrator.
    _status: optional callable(str) for the simple live progress bar.
    on_step: optional callable(agent, status, detail) for the detailed step log.
    """
    from agents.orchestrator_agent import run_pipeline  # lazy import avoids circular dep
    return run_pipeline(
        question, schema, schema_with_enums, db_path, client,
        _status=_status, on_step=on_step,
    )


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
