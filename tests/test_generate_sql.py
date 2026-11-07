"""The SQL + Critic loop, with the Anthropic client and Critic stubbed out."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import agents.orchestrator_agent as orch
from agents.critic_agent import DEMO_DOMAIN_RULES


class FakeClient:
    """Returns the queued SQL strings in order and records each call."""

    def __init__(self, *sqls):
        self.sqls = list(sqls)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.sqls.pop(0)
        return SimpleNamespace(
            content=[SimpleNamespace(text=text)],
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        )


def _review(*verdicts):
    """Critic stub returning the given pass/fail verdicts in order."""
    seq = [{"passed": v, "reason": "" if v else "FILTER LOGIC: bad", "tokens_used": 3} for v in verdicts]
    return patch.object(orch, "review_sql", side_effect=seq)


def test_no_critic_generates_once_without_review():
    client = FakeClient("SELECT 1")
    with _review() as review:
        out = orch.generate_sql("q", "schema", "enums", client, use_critic=False)
    assert out == {"sql": "SELECT 1", "passed": True, "tokens_used": 15,
                   "retries": 0, "critic_warnings": None}
    review.assert_not_called()


def test_rejection_then_fix_logs_one_correction():
    client = FakeClient("SELECT bad", "SELECT good")
    with _review(False, True), patch.object(orch, "log_correction") as log:
        out = orch.generate_sql("q", "schema", "enums", client)
    assert out["sql"] == "SELECT good" and out["passed"] and out["retries"] == 1
    log.assert_called_once_with("q", "sql", "SELECT bad", "FILTER LOGIC: bad", "SELECT good")
    # The retry prompt carries the Critic's reason back to the SQL agent.
    assert "FILTER LOGIC: bad" in client.calls[1]["messages"][-1]["content"]


def test_corrections_not_logged_when_disabled():
    client = FakeClient("SELECT bad", "SELECT good")
    with _review(False, True), patch.object(orch, "log_correction") as log:
        orch.generate_sql("q", "schema", "enums", client, log_corrections=False)
    log.assert_not_called()


def test_three_rejections_fail_closed():
    client = FakeClient("a", "b", "c")
    with _review(False, False, False):
        out = orch.generate_sql("q", "schema", "enums", client)
    assert out["passed"] is False
    assert len(client.calls) == orch.MAX_ATTEMPTS


@pytest.mark.parametrize("rules", [DEMO_DOMAIN_RULES, None])
def test_domain_rules_reach_the_critic(rules):
    client = FakeClient("SELECT 1")
    with _review(True) as review:
        orch.generate_sql("q", "schema", "enums", client, domain_rules=rules)
    assert review.call_args.kwargs["domain_rules"] is rules


def test_pipeline_never_executes_rejected_sql(db_path):
    client = FakeClient("DELETE FROM customers", "DELETE FROM customers", "DELETE FROM customers")
    with _review(False, False, False), \
         patch.object(orch, "plan", return_value={"needs_chart": False, "needs_model": False}):
        out = orch.run_pipeline("how many customers?", "schema", "enums", db_path, client, use_rag=False)
    assert out["error"] and out["answer"] is None


def test_pipeline_execution_is_read_only(db_path):
    # Even if the Critic wrongly passes a write, the read-only connection blocks it.
    client = FakeClient("DELETE FROM customers")
    with _review(True), \
         patch.object(orch, "plan", return_value={"needs_chart": False, "needs_model": False}):
        out = orch.run_pipeline("q", "schema", "enums", db_path, client, use_rag=False)
    assert "readonly" in out["error"]
    import sqlite3
    assert sqlite3.connect(db_path).execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 60
