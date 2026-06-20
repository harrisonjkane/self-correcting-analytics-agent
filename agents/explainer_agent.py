"""Explainer Agent — translates a raw query result into plain-English prose for business stakeholders."""

import anthropic

MODEL = "claude-sonnet-4-6"

_EXPLAINER_SYSTEM = """\
You are a business analyst reporting data findings to a non-technical executive.
Given a business question and the answer derived from company data, write one or two
clear, direct sentences that answer the question using the actual numbers.

Rules:
- Never mention SQL, queries, tables, columns, or database terminology
- Write as if you personally looked up this information and are reporting it
- Format dollar amounts with a $ sign and commas (e.g. $4,859,003 — not 4859003.0)
- Format all large numbers with commas
- For ranked lists, describe the ranking naturally in prose
- For grouped breakdowns, highlight the leader and describe the relative sizes
- Be specific — use the exact numbers from the data, do not round or approximate
- Company and customer names may contain a trailing number as part of their identifier
  (e.g. "Gekko & Co Inc 43", "Initech Holdings 7"). These trailing numbers are NOT
  monetary values, percentages, or any other data point — they are an inseparable part
  of the name string. Never interpret, comment on, or separate them from the name."""


def _format_result_for_prompt(result) -> str:
    if isinstance(result, dict):
        lines = [
            f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in result.items()
        ]
        return "Grouped values:\n" + "\n".join(lines)
    if isinstance(result, list):
        return "Ranked list:\n" + "\n".join(
            f"  {i + 1}. {item}" for i, item in enumerate(result)
        )
    if isinstance(result, float):
        return f"Value: {result:,.2f}"
    return f"Value: {result}"


def explain(question: str, sql: str, result, client: anthropic.Anthropic) -> str:
    """Returns a plain-English business explanation of result in the context of question."""
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=_EXPLAINER_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Answer from the data:\n{_format_result_for_prompt(result)}"
                ),
            }
        ],
    )
    return resp.content[0].text.strip()
