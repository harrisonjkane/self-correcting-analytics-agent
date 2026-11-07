"""BIRD harness helpers, run against a tiny fake BIRD layout — no API calls."""
import json
import sqlite3
from unittest.mock import patch

import pytest

from evals.bird import (
    execute, find_bird_files, load_questions, paired_comparison,
    results_match, stratified_sample, summarize,
)


@pytest.fixture
def fake_bird(tmp_path):
    root = tmp_path / "minidev" / "mini_dev_data"
    db_dir = root / "dev_databases" / "shop"
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / "shop.sqlite")
    con.execute('CREATE TABLE "order items" (id INTEGER, "Unit Price" REAL, status TEXT)')
    con.executemany('INSERT INTO "order items" VALUES (?,?,?)',
                    [(1, 10.0, "Done"), (2, 5.0, "Open"), (3, 7.5, "Done")])
    con.commit()
    con.close()
    questions = [
        {"question_id": i, "db_id": "shop", "question": f"q{i}", "evidence": "",
         "SQL": 'SELECT SUM("Unit Price") FROM "order items" WHERE status = \'Done\'',
         "difficulty": d}
        for i, d in enumerate(["simple"] * 3 + ["moderate"] * 5 + ["challenging"] * 2)
    ]
    (root / "mini_dev_sqlite.json").write_text(json.dumps(questions))
    return tmp_path / "minidev"


def test_find_and_load(fake_bird):
    qpath, db_root = find_bird_files(fake_bird)
    assert qpath.name == "mini_dev_sqlite.json"
    assert (db_root / "shop" / "shop.sqlite").exists()
    assert len(load_questions(qpath)) == 10


def test_stratified_sample_keeps_mix_and_is_deterministic(fake_bird):
    qs = load_questions(find_bird_files(fake_bird)[0])
    a = stratified_sample(qs, 5, seed=1)
    assert a == stratified_sample(qs, 5, seed=1)
    assert len(a) == 5
    diffs = [q["difficulty"] for q in a]
    assert diffs.count("moderate") >= 2  # 50% of the set


def test_execute_is_read_only_and_handles_quoted_names(fake_bird):
    db = find_bird_files(fake_bird)[1] / "shop" / "shop.sqlite"
    assert execute(db, 'SELECT SUM("Unit Price") FROM "order items" WHERE status=\'Done\'') == [(17.5,)]
    with pytest.raises(sqlite3.OperationalError):
        execute(db, 'DELETE FROM "order items"')


def test_execute_times_out(fake_bird):
    db = find_bird_files(fake_bird)[1] / "shop" / "shop.sqlite"
    slow = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c"
    with pytest.raises(TimeoutError):
        execute(db, slow, timeout_s=0.2)


def test_results_match_is_order_insensitive():
    assert results_match([(1,), (2,)], [(2,), (1,)])
    assert not results_match([(1,)], [(1,), (2,)])


def test_summary_and_paired_comparison():
    def rec(qid, correct, diff="simple"):
        return {"question_id": qid, "correct": correct, "difficulty": diff, "tokens_used": 100,
                "latency_s": 1.0, "retries": 0, "critic_passed": True, "pred_error": None}
    base = [rec(1, False), rec(2, True), rec(3, True)]
    treat = [rec(1, True), rec(2, True), rec(3, False)]
    assert summarize(base)["accuracy"] == round(2 / 3, 4)
    assert paired_comparison(base, treat) == {"shared_questions": 3, "fixed": 1, "broken": 1}


def test_harness_end_to_end_with_stubbed_llm(fake_bird, tmp_path, monkeypatch):
    """Runs the real script with generate_sql stubbed: baseline wrong, critic right."""
    import runpy
    import sys

    gold = 'SELECT SUM("Unit Price") FROM "order items" WHERE status = \'Done\''
    wrong = 'SELECT SUM("Unit Price") FROM "order items"'

    def fake_generate_sql(question, schema, schema_with_enums, client, *, use_critic, **kw):
        assert kw["domain_rules"] is None and kw["log_corrections"] is False
        assert '"order items"' in schema  # quoted identifiers reach the prompt
        return {"sql": gold if use_critic else wrong, "passed": True,
                "tokens_used": 10, "retries": 0, "critic_warnings": None}

    monkeypatch.setattr(sys, "argv", ["run_bird_eval.py", "--bird-dir", str(fake_bird),
                                      "--n", "4", "--label", "pytest_tmp", "--workers", "1"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    import agents.orchestrator_agent as orch
    monkeypatch.setattr(orch, "generate_sql", fake_generate_sql)
    from conftest import ROOT
    out = ROOT / "evals" / "results" / "bird_pytest_tmp.jsonl"
    try:
        runpy.run_path(str(ROOT / "scripts" / "run_bird_eval.py"), run_name="__main__")
        summary = json.loads(out.with_suffix(".summary.json").read_text())
        assert summary["baseline"]["accuracy"] == 0.0
        assert summary["critic"]["accuracy"] == 1.0
        assert summary["baseline_vs_critic"]["fixed"] == 4
    finally:
        out.unlink(missing_ok=True)
        out.with_suffix(".summary.json").unlink(missing_ok=True)
