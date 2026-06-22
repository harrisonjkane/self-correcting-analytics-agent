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
        st.markdown(f"- {q}")

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
        if msg["role"] == "assistant" and msg.get("chart_path"):
            st.image(msg["chart_path"])
        if msg["role"] == "assistant" and msg.get("sql"):
            with st.expander("View SQL & raw results"):
                st.code(msg["sql"], language="sql")
                if msg.get("raw_result"):
                    st.write(msg["raw_result"][:20])

# User input
if question := st.chat_input("Ask a question about your data..."):
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
            with st.spinner("Analyzing..."):
                try:
                    result = ask(
                        question=question,
                        schema=st.session_state.schema,
                        schema_with_enums=st.session_state.schema_with_enums,
                        db_path=DB_PATH,
                        client=st.session_state.client,
                    )
                except anthropic.AuthenticationError:
                    reply = (
                        "API authentication failed — your ANTHROPIC_API_KEY may be invalid "
                        "or expired. Check your `.env` file or Streamlit Secrets and refresh."
                    )
                    st.error(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()
                except anthropic.RateLimitError:
                    reply = (
                        "The Anthropic API rate limit was reached. "
                        "Please wait a moment and try again."
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()
                except anthropic.APIConnectionError:
                    reply = (
                        "Could not connect to the Anthropic API — "
                        "check your internet connection and try again."
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()
                except Exception:
                    traceback.print_exc()  # emits full stack trace to Streamlit Cloud logs
                    reply = "Something went wrong while answering that question. Please try again."
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    st.stop()

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
                if result.get("chart_path"):
                    st.image(result["chart_path"])
                if result.get("sql"):
                    with st.expander("View SQL & raw results"):
                        st.code(result["sql"], language="sql")
                        if result.get("raw_result"):
                            st.write(result["raw_result"][:20])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": explanation,
                    "chart_path": result.get("chart_path"),
                    "sql": result.get("sql"),
                    "raw_result": result.get("raw_result"),
                })
