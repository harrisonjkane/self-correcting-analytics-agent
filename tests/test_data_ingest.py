"""Bring-your-own-data loader."""
import io
import sqlite3

import pandas as pd
import pytest

from agents.data_ingest import IngestError, load_upload, sanitize_identifier
from agents.schema_agent import get_schema


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Order Date (UTC)", "order_date_utc"),
        ("2024 sales", "t_2024_sales"),
        ("  Revenue $ ", "revenue"),
        ("!!!", "col"),
    ],
)
def test_sanitize_identifier(raw, expected):
    assert sanitize_identifier(raw) == expected


def _tables(db_path):
    con = sqlite3.connect(db_path)
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return con, names


def test_csv_becomes_one_clean_table(tmp_path):
    csv = b"Region,Total Sales,Total Sales\nWest,10,1\nEast,20,2\n,,\n"
    path = load_upload("Q3 Report.csv", csv, tmp_path)
    con, names = _tables(path)
    assert names == ["q3_report"]
    cols = [r[1] for r in con.execute("PRAGMA table_info(q3_report)")]
    assert cols == ["region", "total_sales", "total_sales_1"]
    assert con.execute("SELECT COUNT(*) FROM q3_report").fetchone()[0] == 2  # blank row dropped
    assert "q3_report" in get_schema(path)


def test_excel_sheets_become_tables(tmp_path):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as xw:
        pd.DataFrame({"a": [1, 2]}).to_excel(xw, sheet_name="Orders", index=False)
        pd.DataFrame({"b": ["x"]}).to_excel(xw, sheet_name="Customer List", index=False)
    path = load_upload("book.xlsx", buf.getvalue(), tmp_path)
    _, names = _tables(path)
    assert sorted(names) == ["customer_list", "orders"]


def test_sqlite_upload_passes_through(tmp_path, db_path):
    with open(db_path, "rb") as f:
        path = load_upload("analytics.db", f.read(), tmp_path / "out")
    _, names = _tables(path)
    assert "orders" in names


def test_reupload_replaces_previous_data(tmp_path):
    load_upload("a.csv", b"x\n1\n", tmp_path)
    path = load_upload("b.csv", b"y\n2\n", tmp_path)
    _, names = _tables(path)
    assert names == ["b"]


@pytest.mark.parametrize(
    "name, data, message",
    [
        ("notes.txt", b"hello", "Unsupported file type"),
        ("fake.db", b"not a database", "isn't a valid SQLite"),
        ("empty.csv", b"a,b\n", "no data rows"),
    ],
)
def test_bad_uploads_raise_friendly_errors(tmp_path, name, data, message):
    with pytest.raises(IngestError, match=message):
        load_upload(name, data, tmp_path)


def test_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.data_ingest.MAX_UPLOAD_BYTES", 10)
    with pytest.raises(IngestError, match="larger than"):
        load_upload("big.csv", b"a\n" * 20, tmp_path)
