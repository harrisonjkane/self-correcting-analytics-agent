"""Load user-supplied data (CSV, Excel, SQLite) into a SQLite file the pipeline can query.

Every agent already speaks SQLite, so uploads are normalized into one session-local
database file. Table and column names are sanitized to plain snake_case identifiers
so generated SQL never needs quoting.
"""

import io
import re
import sqlite3
from pathlib import Path

import pandas as pd

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".db", ".sqlite", ".sqlite3")
_SQLITE_HEADER = b"SQLite format 3\x00"


class IngestError(ValueError):
    """Raised with a user-facing message when an upload can't be loaded."""


def sanitize_identifier(name: str, fallback: str = "col") -> str:
    """'Order Date (UTC)' -> 'order_date_utc'; '2024 sales' -> 't_2024_sales'."""
    ident = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    if not ident:
        ident = fallback
    if ident[0].isdigit():
        ident = f"t_{ident}"
    return ident


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all").dropna(axis=1, how="all")
    df.columns = _dedupe([sanitize_identifier(c) for c in df.columns])
    return df


def load_upload(filename: str, data: bytes, out_dir: str | Path) -> str:
    """Write the upload into a SQLite database under out_dir and return its path."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError(f"File is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestError(f"Unsupported file type '{suffix}'. Use CSV, Excel (.xlsx), or SQLite.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "uploaded.db"
    db_path.unlink(missing_ok=True)

    if suffix in (".db", ".sqlite", ".sqlite3"):
        if not data.startswith(_SQLITE_HEADER):
            raise IngestError("That file isn't a valid SQLite database.")
        db_path.write_bytes(data)
        _require_tables(db_path)
        return str(db_path)

    if suffix == ".csv":
        try:
            frames = {Path(filename).stem: pd.read_csv(io.BytesIO(data))}
        except Exception as exc:
            raise IngestError(f"Couldn't parse the CSV: {exc}") from exc
    else:  # .xlsx — one table per sheet
        try:
            frames = pd.read_excel(io.BytesIO(data), sheet_name=None)
        except Exception as exc:
            raise IngestError(f"Couldn't read the Excel file: {exc}") from exc

    table_names = _dedupe([sanitize_identifier(n, fallback="data") for n in frames])
    conn = sqlite3.connect(db_path)
    try:
        written = 0
        for table, df in zip(table_names, frames.values()):
            df = _clean_frame(df)
            if df.empty:
                continue
            df.to_sql(table, conn, index=False)
            written += 1
    finally:
        conn.close()

    if not written:
        raise IngestError("The file has no data rows.")
    return str(db_path)


def _require_tables(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        n = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise IngestError("That file isn't a readable SQLite database.") from exc
    finally:
        conn.close()
    if not n:
        raise IngestError("The SQLite file has no tables.")
