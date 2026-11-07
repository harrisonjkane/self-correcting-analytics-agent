import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evals"))

DB_PATH = ROOT / "data" / "analytics.db"


@pytest.fixture(scope="session")
def db_path() -> str:
    assert DB_PATH.exists(), "Run `python data/build_database.py` first."
    return str(DB_PATH)
