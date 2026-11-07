"""BIRD benchmark helpers: load questions, sample, execute, and score.

Scoring follows BIRD's execution accuracy (EX): a prediction is correct when the
set of rows it returns equals the set of rows the gold SQL returns.
Works with BIRD Mini-Dev (mini_dev_sqlite.json) or full BIRD dev (dev.json).
"""

import json
import random
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

QUERY_TIMEOUT_S = 30.0


def find_bird_files(bird_dir: str | Path) -> tuple[Path, Path]:
    """Locate the question JSON and the dev_databases folder under bird_dir."""
    bird_dir = Path(bird_dir)
    for name in ("mini_dev_sqlite.json", "dev.json"):
        hits = sorted(bird_dir.rglob(name))
        if hits:
            questions = hits[0]
            break
    else:
        raise FileNotFoundError(f"No mini_dev_sqlite.json or dev.json under {bird_dir}")
    db_roots = sorted(p for p in bird_dir.rglob("dev_databases") if p.is_dir())
    if not db_roots:
        raise FileNotFoundError(f"No dev_databases/ folder under {bird_dir}")
    return questions, db_roots[0]


def load_questions(path: str | Path) -> list[dict]:
    items = json.loads(Path(path).read_text())
    for i, q in enumerate(items):
        q.setdefault("question_id", i)
        q.setdefault("difficulty", "unknown")
        q.setdefault("evidence", "")
    return items


def db_file(db_root: str | Path, db_id: str) -> Path:
    return Path(db_root) / db_id / f"{db_id}.sqlite"


def stratified_sample(questions: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample n questions, keeping the difficulty mix of the full set. Deterministic per seed."""
    if n >= len(questions):
        return list(questions)
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        by_diff[q["difficulty"]].append(q)
    rng = random.Random(seed)
    picked = []
    for diff in sorted(by_diff):
        group = by_diff[diff]
        k = round(n * len(group) / len(questions))
        picked.extend(rng.sample(group, min(k, len(group))))
    # Rounding can leave us a question or two off target; fix up deterministically.
    rest = [q for q in questions if q not in picked]
    rng.shuffle(rest)
    picked = (picked + rest)[:n]
    return sorted(picked, key=lambda q: q["question_id"])


def question_text(q: dict) -> str:
    """BIRD supplies expert 'evidence' (domain hints); standard practice is to include it."""
    evidence = (q.get("evidence") or "").strip()
    return f"{q['question']}\nHint: {evidence}" if evidence else q["question"]


def execute(db_path: str | Path, sql: str, timeout_s: float = QUERY_TIMEOUT_S) -> list[tuple]:
    """Run sql read-only, aborting after timeout_s. Raises on error or timeout."""
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError as exc:
        if time.monotonic() > deadline:
            raise TimeoutError(f"query exceeded {timeout_s:.0f}s") from exc
        raise
    finally:
        conn.close()


def results_match(predicted: list[tuple], gold: list[tuple]) -> bool:
    """BIRD EX: order-insensitive set comparison of result rows."""
    return set(predicted) == set(gold)


def summarize(records: list[dict]) -> dict:
    """Accuracy overall and by difficulty, plus cost/latency, for one condition's records."""
    def acc(rs):
        return round(sum(r["correct"] for r in rs) / len(rs), 4) if rs else None

    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_diff[r["difficulty"]].append(r)
    n = len(records)
    return {
        "n": n,
        "accuracy": acc(records),
        "by_difficulty": {d: {"n": len(rs), "accuracy": acc(rs)} for d, rs in sorted(by_diff.items())},
        "avg_tokens": round(sum(r["tokens_used"] for r in records) / n) if n else None,
        "avg_latency_s": round(sum(r["latency_s"] for r in records) / n, 2) if n else None,
        "total_retries": sum(r["retries"] for r in records),
        "critic_rejected_all": sum(1 for r in records if not r["critic_passed"]),
        "execution_errors": sum(1 for r in records if r["pred_error"]),
    }


def paired_comparison(base: list[dict], treat: list[dict]) -> dict:
    """Per-question flips between two conditions run on the same questions."""
    b = {r["question_id"]: r["correct"] for r in base}
    t = {r["question_id"]: r["correct"] for r in treat}
    shared = b.keys() & t.keys()
    return {
        "shared_questions": len(shared),
        "fixed": sum(1 for q in shared if not b[q] and t[q]),   # wrong -> right
        "broken": sum(1 for q in shared if b[q] and not t[q]),  # right -> wrong
    }
