"""Benchmark integrity: the ground truth every accuracy number rests on."""
import sqlite3

import pytest

from benchmark import BENCHMARK
from scoring import (
    score_answer,
    score_exact_number,
    score_grouped_values,
    score_ranked_list,
    score_single_value,
)

AUTO_SCORED_TYPES = {"exact_number", "single_value", "ranked_list", "grouped_values"}
AUTO_SCORED = [q for q in BENCHMARK if q["answer_type"] in AUTO_SCORED_TYPES]


def test_benchmark_shape():
    # README reports accuracy over the 11 auto-scored questions of 16 total.
    assert len(BENCHMARK) == 16
    assert len(AUTO_SCORED) == 11
    assert len({q["id"] for q in BENCHMARK}) == 16


@pytest.mark.parametrize("q", AUTO_SCORED, ids=[q["id"] for q in AUTO_SCORED])
def test_ground_truth_reproduces_from_database(q, db_path):
    """Each hand-verified answer must still come out of its ground-truth SQL."""
    rows = sqlite3.connect(db_path).execute(q["ground_truth_sql"]).fetchall()
    kind = q["answer_type"]

    if kind == "exact_number":
        answer = rows[0][0]
    elif kind == "single_value":
        answer = rows[0][0]
    elif kind == "ranked_list":
        answer = [r[0] for r in rows]
    else:  # grouped_values
        answer = {r[0]: r[1] for r in rows}

    assert score_answer(q, {"answer": answer})["correct"] is True


def test_non_auto_scored_questions_are_flagged_for_review():
    for q in BENCHMARK:
        if q["answer_type"] not in AUTO_SCORED_TYPES:
            assert score_answer(q, {"answer": "anything"})["correct"] is None


def test_scorers():
    assert score_exact_number(100.004, 100)
    assert not score_exact_number(101, 100)
    assert not score_exact_number(None, 100)
    assert score_single_value("  west ", "West")
    assert score_ranked_list(["A", "b"], ["a", "B"])
    assert not score_ranked_list(["b", "a"], ["a", "b"])  # order matters
    assert score_grouped_values({"x": 1.0, "y": 2.0}, {"y": 2.0, "x": 1.0})
    assert not score_grouped_values({"x": 1.0}, {"x": 1.0, "y": 2.0})
