"""
Phase 0 — Scoring harness for AI Analytics Consultant.

Runs the benchmark question set through whatever pipeline currently exists
and scores results on four dimensions: correctness, efficiency, latency, cost.

This is intentionally pipeline-agnostic in its interface — run_benchmark()
accepts any callable that takes a question string and returns a result dict.
That means the SAME harness scores the Phase 1 single-agent baseline, the
Phase 2 multi-agent + Critic version, and the Phase 3 RAG-enhanced version,
producing a genuine before/after comparison rather than three different
measurement approaches.
"""

import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path

# Make `benchmark` findable regardless of the caller's working directory.
sys.path.insert(0, str(Path(__file__).parent))
from benchmark import BENCHMARK

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def score_exact_number(actual, expected, tolerance=0.01):
    if actual is None:
        return False
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


def score_single_value(actual, expected):
    if actual is None:
        return False
    return str(actual).strip().lower() == str(expected).strip().lower()


def score_ranked_list(actual, expected):
    """Order-sensitive comparison for top-N style questions."""
    if not actual:
        return False
    actual_norm = [str(a).strip().lower() for a in actual]
    expected_norm = [str(e).strip().lower() for e in expected]
    return actual_norm == expected_norm


def score_grouped_values(actual, expected, tolerance=0.01):
    """Compares a dict of group -> value against ground truth, order-insensitive."""
    if not actual or not isinstance(actual, dict):
        return False
    if set(actual.keys()) != set(expected.keys()):
        return False
    return all(
        abs(float(actual[k]) - float(expected[k])) <= tolerance
        for k in expected
    )


def score_answer(question: dict, agent_output: dict) -> dict:
    """
    Scores a single agent response against a benchmark question's ground truth.
    agent_output is expected to contain: {"answer": ..., "raw_result": ...}
    """
    answer_type = question["answer_type"]
    actual = agent_output.get("answer")

    if answer_type == "reasoning" or answer_type in ("chart", "model"):
        # Tier 4/5 — not auto-scored. Flagged for manual review.
        return {"correct": None, "scoring_method": "manual_review_required"}

    expected = question["ground_truth"]
    tolerance = question.get("tolerance", 0.01)

    if answer_type == "exact_number":
        correct = score_exact_number(actual, expected, tolerance)
    elif answer_type == "single_value":
        correct = score_single_value(actual, expected)
    elif answer_type == "ranked_list":
        correct = score_ranked_list(actual, expected)
    elif answer_type == "grouped_values":
        correct = score_grouped_values(actual, expected, tolerance)
    else:
        correct = None

    return {"correct": correct, "scoring_method": "automated"}


def run_benchmark(pipeline_fn, run_label: str, tiers: list = None) -> dict:
    """
    Runs the full benchmark (or a subset of tiers) through the given pipeline
    function and scores every question.

    pipeline_fn: callable(question: str) -> dict
        Must return at minimum: {"answer": ..., "tokens_used": int, "retries": int}
        Extra keys are preserved in the results log.

    run_label: identifies this run, e.g. "phase1_single_agent_baseline"

    tiers: optional list of tier numbers to run, e.g. [1, 2, 3] to skip
        the not-yet-scoreable Tier 4/5 questions during early phases.

    Returns the full results dict and also writes it to evals/results/.
    """
    questions = BENCHMARK if tiers is None else [q for q in BENCHMARK if q["tier"] in tiers]

    results = {
        "run_label": run_label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question_count": len(questions),
        "questions": [],
    }

    for q in questions:
        start = time.time()
        try:
            output = pipeline_fn(q["question"])
            error = None
        except Exception as e:
            output = {"answer": None, "tokens_used": 0, "retries": 0}
            error = str(e)
        latency = time.time() - start

        scoring = score_answer(q, output)

        results["questions"].append({
            "id": q["id"],
            "tier": q["tier"],
            "question": q["question"],
            "expected": q.get("ground_truth"),
            "actual": output.get("answer"),
            "correct": scoring["correct"],
            "scoring_method": scoring["scoring_method"],
            "latency_seconds": round(latency, 2),
            "tokens_used": output.get("tokens_used", 0),
            "retries": output.get("retries", 0),
            "rag_hits": output.get("rag_hits", 0),
            "error": error,
        })

    # ── Summary stats ────────────────────────────────────────────────────
    scored = [r for r in results["questions"] if r["correct"] is not None]
    correct_count = sum(1 for r in scored if r["correct"])

    results["summary"] = {
        "auto_scoreable_questions": len(scored),
        "correct": correct_count,
        "accuracy": round(correct_count / len(scored), 3) if scored else None,
        "avg_latency_seconds": round(
            sum(r["latency_seconds"] for r in results["questions"]) / len(results["questions"]), 2
        ),
        "total_tokens": sum(r["tokens_used"] for r in results["questions"]),
        "total_retries": sum(r["retries"] for r in results["questions"]),
        "errors": sum(1 for r in results["questions"] if r["error"]),
    }

    # ── Persist ──────────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"{run_label}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n=== Benchmark run: {run_label} ===")
    print(f"Questions: {results['question_count']} ({len(scored)} auto-scoreable)")
    print(f"Accuracy: {results['summary']['accuracy']} "
          f"({correct_count}/{len(scored)})")
    print(f"Avg latency: {results['summary']['avg_latency_seconds']}s")
    print(f"Total tokens: {results['summary']['total_tokens']}")
    print(f"Total retries: {results['summary']['total_retries']}")
    print(f"Errors: {results['summary']['errors']}")
    print(f"Saved to {out_path}")

    return results


def compare_runs(label_a: str, label_b: str):
    """Loads two saved benchmark runs and prints a before/after comparison."""
    with open(RESULTS_DIR / f"{label_a}.json") as f:
        run_a = json.load(f)
    with open(RESULTS_DIR / f"{label_b}.json") as f:
        run_b = json.load(f)

    print(f"\n=== Comparison: {label_a} -> {label_b} ===")
    print(f"{'Metric':<22}{'Before':<15}{'After':<15}{'Delta':<10}")
    print("-" * 62)

    sa, sb = run_a["summary"], run_b["summary"]
    metrics = [
        ("Accuracy", sa["accuracy"], sb["accuracy"]),
        ("Avg latency (s)", sa["avg_latency_seconds"], sb["avg_latency_seconds"]),
        ("Total tokens", sa["total_tokens"], sb["total_tokens"]),
        ("Total retries", sa["total_retries"], sb["total_retries"]),
        ("Errors", sa["errors"], sb["errors"]),
    ]
    for name, before, after in metrics:
        delta = round(after - before, 3) if isinstance(before, (int, float)) and isinstance(after, (int, float)) else "—"
        print(f"{name:<22}{str(before):<15}{str(after):<15}{str(delta):<10}")


if __name__ == "__main__":
    # ── Dummy pipeline for smoke-testing the harness itself ──────────────
    # Replace with the real agent pipeline once Phase 1 exists.
    def dummy_pipeline(question: str) -> dict:
        return {"answer": None, "tokens_used": 0, "retries": 0}

    run_benchmark(dummy_pipeline, "phase0_harness_smoketest", tiers=[1, 2, 3])
