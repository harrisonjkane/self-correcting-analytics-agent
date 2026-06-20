"""DataChat — Streamlit UI for the self-correcting analytics agent pipeline."""

import os

import anthropic
import streamlit as st
from dotenv import load_dotenv

from agents.schema_agent import get_schema, get_schema_with_enums
from agents.sql_agent import ask
from evals.benchmark import BENCHMARK

load_dotenv()

DB_PATH = "data/analytics.db"

EXAMPLE_QUESTIONS = [
    q["question"] for q in BENCHMARK if q["tier"] in (1, 2, 3)
][:6]

st.set_page_config(page_title="DataChat", page_icon="📊", layout="wide")

# ── Session state init ────────────────────────────────────────────────────────
if "ready" not in st.session_state:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.session_state.ready = False
        st.session_state.init_error = "ANTHROPIC_API_KEY is not set in .env"
    else:
        try:
            st.session_state.schema = get_schema(DB_PATH)
            st.session_state.schema_with_enums = get_schema_with_enums(DB_PATH)
            st.session_state.client = anthropic.Anthropic(api_key=api_key)
            st.session_state.messages = []
            st.session_state.ready = True
        except Exception as exc:
            st.session_state.ready = False
            st.session_state.init_error = str(exc)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("DataChat")
    st.markdown(
        "Ask plain-English questions about your B2B SaaS business data. "
        "The pipeline writes SQL, reviews it for correctness, self-corrects if needed, "
        "and returns a plain-English answer — no SQL knowledge required."
    )

    st.divider()

    st.subheader("Database Schema")
    if st.session_state.get("ready"):
        st.code(st.session_state.schema, language=None)
    else:
        st.caption("Schema unavailable — check startup error above.")

    st.divider()

    st.subheader("Try asking...")
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- {q}")

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📊 DataChat")

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
        with st.spinner("Analyzing..."):
            try:
                result = ask(
                    question=question,
                    schema=st.session_state.schema,
                    schema_with_enums=st.session_state.schema_with_enums,
                    db_path=DB_PATH,
                    client=st.session_state.client,
                )
            except Exception:
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
            if result.get("sql"):
                with st.expander("View SQL & raw results"):
                    st.code(result["sql"], language="sql")
                    if result.get("raw_result"):
                        st.write(result["raw_result"][:20])
            st.session_state.messages.append({
                "role": "assistant",
                "content": explanation,
                "sql": result.get("sql"),
                "raw_result": result.get("raw_result"),
            })
