"""
BIRD evaluation: does the Critic improve execution accuracy on an external benchmark?

Runs the same stratified sample of BIRD questions under two conditions:
  baseline  — single SQL generation, no Critic
  critic    — SQL generation + Critic self-correction loop (Layer 1 only)
Demo-specific Layer 2 rules and the RAG correction log are off in both, so the only
difference between conditions is the Critic.

Setup: download BIRD Mini-Dev (https://github.com/bird-bench/mini_dev) and unzip it.

Usage:
    python3 scripts/run_bird_eval.py --bird-dir ~/data/minidev --n 20           # smoke test
    python3 scripts/run_bird_eval.py --bird-dir ~/data/minidev --n 150 --seed 42

Results append to evals/results/bird_<label>.jsonl (re-running resumes where it left
off) and a summary prints at the end. See docs/bird_eval_plan.md.
"""

# Run from anywhere: put the repo root on sys.path and use it as the working dir.
import os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic
from dotenv import load_dotenv

from agents.orchestrator_agent import generate_sql
from agents.schema_agent import get_schema, get_schema_with_enums
from evals.bird import (
    db_file, execute, find_bird_files, load_questions, paired_comparison,
    question_text, results_match, stratified_sample, summarize,
)

# Same instruction as the app's SQL agent, minus the B2B SaaS framing that would
# be wrong for BIRD's domains.
BIRD_SQL_SYSTEM = """\
You are a senior SQL analyst.
Given a SQLite database schema and a question, write exactly ONE SQLite
SQL query that answers the question. Return ONLY the raw SQL — no markdown fences,
no explanation, no trailing semicolon."""

CONDITIONS = {
    "baseline": {"use_critic": False},
    "critic": {"use_critic": True},
}

_schema_cache: dict[str, tuple[str, str]] = {}
_schema_lock = threading.Lock()


def schemas_for(db_path: Path) -> tuple[str, str]:
    key = str(db_path)
    with _schema_lock:
        if key not in _schema_cache:
            _schema_cache[key] = (get_schema(key), get_schema_with_enums(key))
        return _schema_cache[key]


def run_one(q: dict, condition: str, db_root: Path, client, gold_cache: dict) -> dict:
    db = db_file(db_root, q["db_id"])
    schema, schema_with_enums = schemas_for(db)

    start = time.monotonic()
    gen = generate_sql(
        question_text(q), schema, schema_with_enums, client,
        system=BIRD_SQL_SYSTEM, domain_rules=None, log_corrections=False,
        **CONDITIONS[condition],
    )
    latency = time.monotonic() - start

    pred_error = None
    correct = False
    # Matches the app: SQL the Critic rejected on every attempt is never run.
    if gen["passed"]:
        try:
            pred_rows = execute(db, gen["sql"])
            correct = results_match(pred_rows, gold_cache[q["question_id"]])
        except Exception as exc:
            pred_error = f"{type(exc).__name__}: {exc}"

    return {
        "question_id": q["question_id"],
        "db_id": q["db_id"],
        "difficulty": q["difficulty"],
        "condition": condition,
        "sql": gen["sql"],
        "gold_sql": q["SQL"],
        "critic_passed": gen["passed"],
        "critic_warnings": gen["critic_warnings"],
        "retries": gen["retries"],
        "tokens_used": gen["tokens_used"],
        "latency_s": round(latency, 2),
        "pred_error": pred_error,
        "correct": correct,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bird-dir", required=True, help="Folder containing BIRD Mini-Dev or dev data")
    ap.add_argument("--n", type=int, default=150, help="Questions to sample (stratified by difficulty)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--conditions", default="baseline,critic")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--label", default=None, help="Results file label (default: n<N>_seed<SEED>)")
    args = ap.parse_args()

    load_dotenv()
    client = anthropic.Anthropic()

    questions_path, db_root = find_bird_files(Path(args.bird_dir).expanduser())
    sample = stratified_sample(load_questions(questions_path), args.n, args.seed)
    conditions = [c.strip() for c in args.conditions.split(",")]
    label = args.label or f"n{args.n}_seed{args.seed}"
    out_path = ROOT / "evals" / "results" / f"bird_{label}.jsonl"

    done = set()
    records: list[dict] = []
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            r = json.loads(line)
            records.append(r)
            done.add((r["question_id"], r["condition"]))

    # Gold results, computed once per question. Skip any whose gold SQL fails here.
    gold_cache, skipped = {}, []
    for q in sample:
        try:
            gold_cache[q["question_id"]] = execute(db_file(db_root, q["db_id"]), q["SQL"])
        except Exception as exc:
            skipped.append((q["question_id"], str(exc)))
    sample = [q for q in sample if q["question_id"] in gold_cache]

    todo = [(q, c) for c in conditions for q in sample if (q["question_id"], c) not in done]
    print(f"{questions_path.name}: {len(sample)} questions x {len(conditions)} conditions; "
          f"{len(todo)} runs to go -> {out_path.relative_to(ROOT)}")
    if skipped:
        print(f"Skipped {len(skipped)} questions whose gold SQL failed locally: {skipped[:3]}")

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool, out_path.open("a") as f:
        futures = {pool.submit(run_one, q, c, db_root, client, gold_cache): (q, c) for q, c in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            q, c = futures[fut]
            try:
                r = fut.result()
            except anthropic.APIError as exc:
                print(f"  [{i}/{len(todo)}] {c} q{q['question_id']}: API error, will retry on next run ({exc})")
                continue
            with lock:
                f.write(json.dumps(r, default=str) + "\n")
                f.flush()
                records.append(r)
            mark = "✓" if r["correct"] else "✗"
            print(f"  [{i}/{len(todo)}] {c:<8} q{r['question_id']:<5} {r['difficulty']:<12} {mark}")

    ids = {q["question_id"] for q in sample}
    by_cond = {c: [r for r in records if r["condition"] == c and r["question_id"] in ids] for c in conditions}
    summary = {c: summarize(rs) for c, rs in by_cond.items() if rs}
    if "baseline" in by_cond and "critic" in by_cond:
        summary["baseline_vs_critic"] = paired_comparison(by_cond["baseline"], by_cond["critic"])
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
