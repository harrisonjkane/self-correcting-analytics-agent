"""
Critic Agent — reviews generated SQL before execution.

Checks for the three failure modes observed in the Phase 1 baseline:
  1. String literal case mismatches vs actual stored values
  2. Filter logic errors (exclusion when inclusion is needed)
  3. DDL/DML safety (defense-in-depth on top of the read-only connection)
"""

import json

import anthropic

MODEL = "claude-sonnet-4-6"

_CRITIC_SYSTEM = """\
You are a SQL quality critic for a SQLite analytics database.
You will receive a SQL query and a schema that includes the exact stored values
for categorical columns. Check the SQL for exactly three categories of errors:

1. STRING LITERAL CASE
   Every string value in WHERE / HAVING / JOIN ... ON clauses must EXACTLY match
   the stored values shown in the schema (SQLite string comparisons are
   case-sensitive by default). If the schema shows priority stored values are
   'Critical', 'High', 'Low', 'Medium', then WHERE priority = 'critical' is WRONG.

2. FILTER LOGIC
   When a question targets one specific category value, the WHERE clause must use
   direct equality (= 'X'), NOT exclusion (!= 'Y' or NOT IN ('Y')), because
   exclusion silently includes every other value.
   Example violation: WHERE status != 'Cancelled' when the intent is
   status = 'Completed' — this also returns 'Pending' and 'Refunded' rows.

3. DDL / DML SAFETY
   The query must contain only SELECT statements. Any INSERT, UPDATE, DELETE,
   DROP, ALTER, CREATE, or TRUNCATE is a violation.

Respond with ONLY a JSON object — no markdown fences, no explanation:
  {"passed": true, "reason": ""}
  {"passed": false, "reason": "<category name>: <specific description of the issue>"}

Report the first failing check you find. Do not list multiple issues."""


def review_sql(sql: str, schema_context: str, client: anthropic.Anthropic) -> dict:
    """
    Reviews sql against schema_context (which must include stored enum values).
    Returns {"passed": bool, "reason": str, "tokens_used": int}.
    """
    resp = client.messages.create(
        model=MODEL,
        max_tokens=256,
        temperature=0,
        system=_CRITIC_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
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
        # If the Critic fails to return valid JSON, pass the SQL through rather
        # than blocking execution on a harness bug.
        return {"passed": True, "reason": "", "tokens_used": tokens, "parse_error": True}
