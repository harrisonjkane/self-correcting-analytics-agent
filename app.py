"""Multi-Agent Query Engine — Streamlit UI for the full agent pipeline."""

import os
import traceback

import anthropic
import streamlit as st
from dotenv import load_dotenv

from agents.schema_agent import get_schema, get_schema_with_enums, get_sample_rows
from agents.sql_agent import ask, check_browse_request
from evals.benchmark import BENCHMARK

load_dotenv()

DB_PATH = "data/analytics.db"


def _get_api_key(key: str) -> str | None:
    """Check Streamlit Secrets first (for deployed apps), then fall back to env."""
    try:
        val = st.secrets.get(key)
        if val:
            return val
    except Exception:
        pass  # st.secrets raises if secrets.toml is absent
    return os.environ.get(key)


_PIPELINE_STEPS = [
    ("⚙️",  "Planning your question"),
    ("🗄️",  "Reading database schema"),
    ("🔍",  "Checking past corrections"),
    ("✍️",  "Writing SQL query"),
    ("✅",  "Reviewing query"),
    ("📊",  "Building your answer"),
]

EXAMPLE_QUESTIONS = [
    q["question"] for q in BENCHMARK if q["tier"] in (1, 2, 3)
][:6] + ["Show revenue split by product category"]

st.set_page_config(page_title="Multi-Agent Query Engine", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
        .stChatMessage { border-radius: 12px; }
        .block-container { padding-top: 2rem; }
        h1 { color: #1F4E79 !important; }
        a { color: #2E75B6 !important; }
        .stCode code, pre code { color: #1a1a1a !important; }
        [data-testid="stChatMessage"] p,
        [data-testid="stChatMessage"] li,
        [data-testid="stChatMessage"] span { color: #1a1a1a !important; }
        [data-testid="stChatMessageAvatarUser"],
        [data-testid="stChatMessageAvatarAssistant"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ────────────────────────────────────────────────────────
if "ready" not in st.session_state:
    api_key = _get_api_key("ANTHROPIC_API_KEY")
    if not api_key:
        st.session_state.ready = False
        st.session_state.init_error = "ANTHROPIC_API_KEY is not set in .env"
    else:
        try:
            st.session_state.schema = get_schema(DB_PATH)
            st.session_state.schema_with_enums = get_schema_with_enums(DB_PATH)
            st.session_state.sample_rows = get_sample_rows(DB_PATH)
            st.session_state.client = anthropic.Anthropic(api_key=api_key)
            st.session_state.messages = []
            st.session_state.ready = True
        except Exception as exc:
            st.session_state.ready = False
            st.session_state.init_error = str(exc)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Multi-Agent Query Engine")
    st.markdown(
        "Ask questions about your data in plain English. "
        "SQL is generated, reviewed by a Critic agent, and automatically corrected "
        "before you ever see it."
    )

    st.divider()

    st.subheader("Database Schema")
    if st.session_state.get("ready"):
        try:
            for line in st.session_state.schema.splitlines()[1:]:  # skip "Tables:" header
                line = line.strip()
                if not line:
                    continue
                table_part, _, fk_raw = line.partition(" -- FK: ")
                paren = table_part.index("(")
                table_name = table_part[:paren]
                cols_str = table_part[paren + 1 : -1]
                cols = [c.strip() for c in cols_str.split(", ")]
                with st.expander(table_name):
                    for col in cols:
                        name, _, col_type = col.partition(" ")
                        st.markdown(f"**`{name}`** {col_type}")
                    if fk_raw:
                        st.markdown("**Foreign keys:**")
                        for fk in fk_raw.split("; "):
                            st.markdown(f"- `{fk.strip()}`")
                    sample = st.session_state.get("sample_rows", {}).get(table_name)
                    if sample:
                        st.markdown("**Sample row:**")
                        for col_name, val in sample.items():
                            display_val = str(val) if val is not None else "null"
                            st.caption(f"`{col_name}` → {display_val}")
        except Exception:
            st.caption("Schema display error — check application logs.")
    else:
        st.caption("Schema unavailable — check startup error above.")

    st.divider()

    st.subheader("Try asking...")
    for q in EXAMPLE_QUESTIONS:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state["prefilled_question"] = q
            st.rerun()

    st.divider()
    st.toggle("Show pipeline steps", value=False, key="show_steps")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📊 Multi-Agent Query Engine")

if not st.session_state.get("ready"):
    st.error(
        f"Startup failed: {st.session_state.get('init_error', 'Unknown error')}. "
        "Check your `.env` file and database path, then refresh the page."
    )
    st.stop()

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("browse_df") is not None:
            st.dataframe(msg["browse_df"])
        if msg["role"] == "assistant" and msg.get("chart_bytes"):
            st.image(msg["chart_bytes"])
        if msg["role"] == "assistant" and msg.get("sql"):
            with st.expander("View SQL & raw results"):
                st.code(msg["sql"], language="sql")
                if msg.get("raw_result"):
                    st.write(msg["raw_result"][:20])

# User input — sidebar buttons set prefilled_question; consume it before chat_input
prefilled = st.session_state.pop("prefilled_question", None)
question = prefilled or st.chat_input("Ask a question about your data...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # ── Browse shortcut: bypass the full pipeline for table-scan requests ──
        browse = check_browse_request(
            question,
            st.session_state.schema,
            DB_PATH,
        )
        if browse is not None:
            if browse.get("browse_error"):
                st.markdown(browse["browse_error"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": browse["browse_error"]}
                )
            else:
                df = browse["browse_df"]
                reply = f"Here are the first {len(df)} rows of **{browse['sql'].split('FROM')[1].split('LIMIT')[0].strip()}**:"
                st.markdown(reply)
                st.dataframe(df)
                with st.expander("View SQL"):
                    st.code(browse["sql"], language="sql")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "browse_df": df,
                    "sql": browse["sql"],
                })
        else:
            show_steps = st.session_state.get("show_steps", False)
            result = None

            def _render_result(result: dict) -> None:
                if result.get("error"):
                    reply = (
                        "I wasn't able to answer that question — the query didn't execute successfully. "
                        "Try rephrasing it."
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    explanation = result.get("explanation") or str(result.get("answer", "No result."))
                    st.markdown(explanation)
                    if result.get("chart_bytes"):
                        st.image(result["chart_bytes"])
                    if result.get("sql"):
                        with st.expander("View SQL & raw results"):
                            st.code(result["sql"], language="sql")
                            if result.get("raw_result"):
                                st.write(result["raw_result"][:20])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": explanation,
                        "chart_bytes": result.get("chart_bytes"),
                        "sql": result.get("sql"),
                        "raw_result": result.get("raw_result"),
                    })

            if show_steps:
                # ── Detailed step-by-step log ─────────────────────────────────
                steps_placeholder = st.empty()
                step_log: list[tuple[str, str, str]] = []

                def _step_lines() -> list[str]:
                    lines = []
                    for agent, status, detail in step_log:
                        if status == "running":
                            lines.append(f"⏳&nbsp; **{agent}**" + (f" — {detail}" if detail else ""))
                        elif status == "done":
                            lines.append(f"✅&nbsp; **{agent}**" + (f" — {detail}" if detail else ""))
                        elif status == "retry":
                            lines.append(f"🔄&nbsp; **{agent}** — {detail}")
                    return lines

                def _render_steps() -> None:
                    lines = _step_lines()
                    steps_placeholder.markdown(
                        "\n\n".join(lines) if lines else "⏳&nbsp; Starting...",
                    )

                def _on_step(agent: str, status: str, detail: str) -> None:
                    if status in ("done", "retry"):
                        # Replace the most-recent "running" entry for this agent
                        for i in range(len(step_log) - 1, -1, -1):
                            if step_log[i][0] == agent and step_log[i][1] == "running":
                                step_log[i] = (agent, status, detail)
                                _render_steps()
                                return
                    step_log.append((agent, status, detail))
                    _render_steps()

                try:
                    result = ask(
                        question=question,
                        schema=st.session_state.schema,
                        schema_with_enums=st.session_state.schema_with_enums,
                        db_path=DB_PATH,
                        client=st.session_state.client,
                        _status=None,
                        on_step=_on_step,
                    )
                except anthropic.AuthenticationError:
                    steps_placeholder.empty()
                    reply = (
                        "API authentication failed — your ANTHROPIC_API_KEY may be invalid "
                        "or expired. Check your `.env` file or Streamlit Secrets and refresh."
                    )
                    st.error(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()
                except anthropic.RateLimitError:
                    steps_placeholder.empty()
                    reply = (
                        "The Anthropic API rate limit was reached. "
                        "Please wait a moment and try again."
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()
                except anthropic.APIConnectionError:
                    steps_placeholder.empty()
                    reply = (
                        "Could not connect to the Anthropic API — "
                        "check your internet connection and try again."
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()
                except Exception:
                    steps_placeholder.empty()
                    traceback.print_exc()
                    reply = "Something went wrong while answering that question. Please try again."
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()

                # Pipeline done: clear the live placeholder and render the step log
                # as ordinary static content so the answer renders cleanly below it.
                steps_placeholder.empty()
                _final = _step_lines()
                if _final:
                    st.markdown("\n\n".join(_final))

                with st.container():
                    _render_result(result)

            else:
                # ── Simple progress bar (default) ─────────────────────────────
                progress_area = st.empty()

                def _render_progress(active: int, done: set) -> None:
                    lines = []
                    for i, (icon, label) in enumerate(_PIPELINE_STEPS):
                        if i in done:
                            lines.append(f"✅ {label}")
                        elif i == active:
                            lines.append(f"**{icon}&nbsp; {label}...**")
                        else:
                            lines.append(
                                f"<span style='color:#aaa'>{icon}&nbsp; {label}</span>"
                            )
                    progress_area.markdown("\n\n".join(lines), unsafe_allow_html=True)

                def _on_status(msg: str) -> None:
                    if "Planning" in msg:
                        _render_progress(0, set())
                    elif "Generating SQL" in msg or "Revising SQL" in msg:
                        _render_progress(3, {0, 1, 2})
                    elif "Reviewing SQL" in msg:
                        _render_progress(4, {0, 1, 2, 3})
                    elif "Executing" in msg or "Interpreting" in msg or "Generating chart" in msg:
                        _render_progress(5, {0, 1, 2, 3, 4})

                _render_progress(0, set())

                try:
                    result = ask(
                        question=question,
                        schema=st.session_state.schema,
                        schema_with_enums=st.session_state.schema_with_enums,
                        db_path=DB_PATH,
                        client=st.session_state.client,
                        _status=_on_status,
                        on_step=None,
                    )
                except anthropic.AuthenticationError:
                    progress_area.empty()
                    reply = (
                        "API authentication failed — your ANTHROPIC_API_KEY may be invalid "
                        "or expired. Check your `.env` file or Streamlit Secrets and refresh."
                    )
                    st.error(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()
                except anthropic.RateLimitError:
                    progress_area.empty()
                    reply = (
                        "The Anthropic API rate limit was reached. "
                        "Please wait a moment and try again."
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()
                except anthropic.APIConnectionError:
                    progress_area.empty()
                    reply = (
                        "Could not connect to the Anthropic API — "
                        "check your internet connection and try again."
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()
                except Exception:
                    progress_area.empty()
                    traceback.print_exc()
                    reply = "Something went wrong while answering that question. Please try again."
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()

                progress_area.empty()
                _render_result(result)
