"""Headless UI tests via Streamlit AppTest. The agent pipeline is stubbed out — no API calls."""
from unittest.mock import patch

import pytest
from streamlit.testing.v1 import AppTest

from agents.critic_agent import DEMO_DOMAIN_RULES
from agents.data_ingest import load_upload
from conftest import ROOT

FAKE_RESULT = {
    "answer": 60, "explanation": "You have 60 customers.", "sql": "SELECT COUNT(*) FROM customers",
    "raw_result": [(60,)], "chart_bytes": None, "error": None,
}


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.chdir(ROOT)
    with patch("agents.sql_agent.ask", return_value=FAKE_RESULT) as ask:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30)
        at.run()
        assert not at.exception
        yield at, ask


def test_chat_input_survives_example_click(app):
    at, _ = app
    assert len(at.chat_input) == 1
    at.sidebar.button[0].click().run()  # first example question
    assert not at.exception
    # Regression: the input box used to disappear after clicking an example.
    assert len(at.chat_input) == 1


def test_typed_question_uses_demo_rules_and_rag(app):
    at, ask = app
    at.chat_input[0].set_value("How many customers do we have?").run()
    kwargs = ask.call_args.kwargs
    assert kwargs["domain_rules"] == DEMO_DOMAIN_RULES
    assert kwargs["use_rag"] is True
    assert "You have 60 customers." in [m.value for m in at.markdown]


def test_uploaded_data_turns_off_demo_rules_and_rag(app, tmp_path):
    at, ask = app
    path = load_upload("sales.csv", b"region,amount\nWest,10\nEast,20\n", tmp_path)
    at.session_state["db_path"] = path
    at.session_state["is_demo"] = False
    at.run()
    at.chat_input[0].set_value("total amount by region").run()
    kwargs = ask.call_args.kwargs
    assert kwargs["db_path"] == path
    assert kwargs["domain_rules"] is None
    assert kwargs["use_rag"] is False
