"""
Critic Agent — reviews generated SQL before execution.

Two-layer design:
  LAYER 1 — Schema-derived checks: general-purpose, reusable across any database.
             Derives correctness entirely from schema enum values and question wording.
  LAYER 2 — Domain knowledge rules: business-specific, requires maintenance as
             business logic changes. Documents assumptions that cannot be inferred
             from schema or question text alone.
"""

import json

import anthropic

MODEL = "claude-sonnet-4-6"

_CRITIC_SYSTEM = """\
You are a SQL quality critic for a SQLite analytics database.
You will receive the original business question, the database schema (including
the exact stored values for categorical columns), and the SQL to review.

Your checks are organized in two layers:

## LAYER 1: SCHEMA-DERIVED CHECKS
## (General-purpose — derived from schema enum values and question wording only.
##  These checks are reusable across any database with the same schema.)

1. STRING LITERAL CASE
   Every string value in WHERE / HAVING / JOIN ... ON clauses must EXACTLY match
   the stored values shown in the schema (SQLite string comparisons are
   case-sensitive by default). If stored priority values are 'Critical', 'High',
   'Low', 'Medium', then WHERE priority = 'critical' is WRONG.

2. FILTER LOGIC
   Read the original question. If the question targets a specific subset of rows
   identified by a categorical column value (e.g. "completed orders", "critical
   tickets", "Hardware products"), the WHERE clause must use direct equality
   (= 'X'), NOT exclusion (!= 'Y' or NOT IN ('Y')), because exclusion silently
   includes every other category value.
   Example: question asks for "completed orders" — WHERE status != 'Cancelled'
   is WRONG because it also includes 'Pending' and 'Refunded' rows.

3. DDL / DML SAFETY
   The query must contain only SELECT statements. Any INSERT, UPDATE, DELETE,
   DROP, ALTER, CREATE, or TRUNCATE is a violation.

## LAYER 2: DOMAIN KNOWLEDGE RULES
## (Business-specific — encodes assumptions that cannot be derived from the
##  schema or question text alone. Must be reviewed and updated whenever the
##  underlying business logic changes.)

4. REVENUE IMPLIES COMPLETED STATUS
   In this B2B SaaS database, "revenue" means recognised revenue only.
   Any query that computes revenue — SUM of order_items quantities × prices,
   average order value, or customer ranking by spend — MUST filter orders by
   status = 'Completed'. Pending, Cancelled, and Refunded orders are not
   recognised revenue. If a revenue query joins orders and order_items without
   WHERE orders.status = 'Completed', it is a violation, even if the question
   does not use the word "completed".

Respond with ONLY a JSON object — no markdown fences, no explanation:
  {"passed": true, "reason": ""}
  {"passed": false, "reason": "<check name>: <specific description of the issue>"}

Report the first failing check you find. Do not list multiple issues."""

_CHART_CRITIC_SYSTEM = """\
You are a code safety and quality reviewer for auto-generated matplotlib chart code.

Check the code against these rules:

1. NO IMPORTS: The code must not contain any import statements. All needed
   libraries (plt, pd) are pre-injected into the execution scope. Any import
   line is an immediate failure.

2. NO UNSAFE OPERATIONS: The code must not call open(), eval(), exec(), or
   access os, subprocess, requests, socket, or any other module or file/network
   operation. The ONLY allowed file operation is plt.savefig(CHART_PATH, ...).

3. SENSIBLE CHART TYPE: Bar charts (ax.bar) for grouped category comparisons.
   Horizontal bar charts (ax.barh) for ranked lists so long labels fit. Pie
   charts and scatter plots are only acceptable if the data shape clearly
   requires them. Flag completely wrong chart types.

4. SAVES CORRECTLY: The code must call plt.savefig(CHART_PATH, ...) and must
   NOT call plt.show().

Respond with ONLY a JSON object — no markdown fences:
  {"passed": true, "reason": ""}
  {"passed": false, "reason": "<check name>: <specific issue>"}

Report the first failing check only."""


def review_sql(
    sql: str,
    schema_context: str,
    client: anthropic.Anthropic,
    question: str = "",
) -> dict:
    """
    Reviews sql against schema_context and the original question.
    question is required for the MISSING FILTER check; always pass it.
    Returns {"passed": bool, "reason": str, "tokens_used": int}.
    """
    question_line = f"Original question: {question}\n\n" if question else ""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        temperature=0,
        system=_CRITIC_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{question_line}"
                    f"Schema (with stored values):\n{schema_context}"
                    f"\n\nSQL to review:\n{sql}"
                ),
            }
        ],
    )
    tokens = resp.usage.input_tokens + resp.usage.output_tokens

    try:
        result = json.loads(resp.content[0].text.strip())
        result["tokens_used"] = tokens
        return result
    except (json.JSONDecodeError, AttributeError):
        return {"passed": True, "reason": "", "tokens_used": tokens, "parse_error": True}


def review_chart_code(
    code: str,
    question: str,
    result_description: str,
    client: anthropic.Anthropic,
) -> dict:
    """
    Reviews auto-generated matplotlib chart code for safety and correctness.
    Returns {"passed": bool, "reason": str, "tokens_used": int}.
    """
    resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        temperature=0,
        system=_CHART_CRITIC_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Data being plotted:\n{result_description}\n\n"
                    f"Chart code to review:\n{code}"
                ),
            }
        ],
    )
    tokens = resp.usage.input_tokens + resp.usage.output_tokens

    try:
        result = json.loads(resp.content[0].text.strip())
        result["tokens_used"] = tokens
        return result
    except (json.JSONDecodeError, AttributeError):
        return {"passed": True, "reason": "", "tokens_used": tokens, "parse_error": True}
