"""Chart Agent — generates matplotlib charts for structured query results."""

import io
import os

import anthropic
import matplotlib
matplotlib.use('Agg')  # headless backend; must precede pyplot import
import matplotlib.pyplot as plt
import pandas as pd

from agents.critic_agent import review_chart_code

MODEL = "claude-sonnet-4-6"

_CHART_SYSTEM = """\
You are a data visualization expert. Given a business question and its answer data,
write matplotlib Python code to create a clear, professional chart.

Strict rules:
- Do NOT write any import statements. All libraries are pre-injected:
    plt  → matplotlib.pyplot
    pd   → pandas
  CHART_PATH is a writable buffer — call plt.savefig(CHART_PATH, ...) to write the chart into it.
- Use ax.bar() for grouped category comparisons and breakdowns.
- Use ax.barh() for ranked lists (top-N) so long labels fit without rotation.
- Never use pie charts or scatter plots.
- Title the chart with a clean version of the business question.
- Label axes clearly. Format dollar amounts on axis ticks with $ and commas
  using: ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
  or ax.xaxis equivalent for horizontal charts.
- Color all bars #2E75B6.
- Call plt.tight_layout() before saving.
- End with exactly: plt.savefig(CHART_PATH, bbox_inches='tight', dpi=150)
- Do NOT call plt.show().
- Return ONLY the raw Python code — no markdown fences, no explanation."""

_SAFE_BUILTINS = {
    "range": range, "len": len, "list": list, "tuple": tuple,
    "dict": dict, "set": set, "zip": zip, "enumerate": enumerate,
    "sorted": sorted, "reversed": reversed, "str": str, "int": int,
    "float": float, "round": round, "min": min, "max": max,
    "sum": sum, "abs": abs, "bool": bool, "isinstance": isinstance,
    "print": print,
}


def should_chart(question: str, result) -> bool:
    """Return True if the result has enough structure to be worth charting."""
    if isinstance(result, dict) and len(result) >= 2:
        return True
    if isinstance(result, list) and len(result) >= 3:
        return True
    return False


def _format_result_for_prompt(result, raw_result=None) -> str:
    if isinstance(result, dict):
        lines = [
            f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}"
            for k, v in result.items()
        ]
        return "Type: grouped breakdown\nData:\n" + "\n".join(lines)

    if isinstance(result, list):
        # If raw_result has (label, value) pairs, use those so the chart has actual values.
        if (
            raw_result
            and isinstance(raw_result[0], (list, tuple))
            and len(raw_result[0]) >= 2
        ):
            lines = [
                f"  {row[0]}: {row[1]:,.2f}" if isinstance(row[1], float)
                else f"  {row[0]}: {row[1]}"
                for row in raw_result
            ]
            return "Type: ranked list with values\nData:\n" + "\n".join(lines)
        else:
            lines = [f"  {i + 1}. {item}" for i, item in enumerate(result)]
            return "Type: ranked list (names only)\nData:\n" + "\n".join(lines)

    return f"Type: single value\nData: {result}"


def create_chart(
    question: str,
    result,
    chart_dir: str,  # retained for call-site compatibility; no longer used
    client: anthropic.Anthropic,
    raw_result=None,
) -> bytes | None:
    """
    Generate, review, and execute a matplotlib chart for result.
    Returns PNG bytes rendered in memory, or None if no chart was produced.
    Renders entirely in memory — no filesystem writes.
    """
    if not should_chart(question, result):
        return None

    buf = io.BytesIO()
    result_description = _format_result_for_prompt(result, raw_result)

    # ── Ask Claude to generate chart code ────────────────────────────────────
    gen_resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=_CHART_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Question: {question}\n\nAnswer data:\n{result_description}",
            }
        ],
    )
    code = gen_resp.content[0].text.strip()

    # Strip accidental markdown fences
    if code.startswith("```"):
        code = "\n".join(
            line for line in code.splitlines() if not line.startswith("```")
        ).strip()

    # ── Critic review before execution ───────────────────────────────────────
    review = review_chart_code(code, question, result_description, client)
    if not review["passed"]:
        return None

    # ── Execute in restricted scope ───────────────────────────────────────────
    scope = {
        "plt": plt,
        "pd": pd,
        "CHART_PATH": buf,
        "__builtins__": _SAFE_BUILTINS,
    }

    try:
        exec(code, scope)  # noqa: S102
    except Exception:
        plt.close("all")
        return None

    plt.close("all")
    buf.seek(0)
    data = buf.read()
    return data if data else None
