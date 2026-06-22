"""Inspects the database schema and returns formatted strings for use in prompts."""

import os
import sqlite3


def _check_db_exists(db_path: str) -> None:
    """Raise before sqlite3.connect() can create a stray empty file."""
    if not os.path.exists(db_path):
        raise RuntimeError(
            f"Database file not found at '{db_path}'. "
            "Make sure the file path is correct and the database has been seeded."
        )


def get_schema(db_path: str) -> str:
    """Basic schema: table definitions + FK notes. Used for SQL generation."""
    _check_db_exists(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    if not tables:
        conn.close()
        raise RuntimeError(
            f"No tables found in database at '{db_path}'. "
            "The file may be empty or corrupted."
        )

    parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)

        cursor.execute(f"PRAGMA foreign_key_list({table})")
        fks = cursor.fetchall()
        fk_notes = ""
        if fks:
            fk_notes = " -- FK: " + "; ".join(
                f"{fk[3]} -> {fk[2]}.{fk[4]}" for fk in fks
            )

        parts.append(f"  {table}({col_defs}){fk_notes}")

    conn.close()
    return "Tables:\n" + "\n".join(parts)


def get_sample_rows(db_path: str) -> dict[str, dict]:
    """Return one sample row per table as {table_name: {col: value, ...}}."""
    _check_db_exists(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    result = {}
    for table in tables:
        cursor.execute(f"SELECT * FROM {table} LIMIT 1")
        cols = [d[0] for d in cursor.description]
        row = cursor.fetchone()
        if row:
            result[table] = dict(zip(cols, row))
    conn.close()
    return result


def get_schema_with_enums(db_path: str) -> str:
    """Extended schema: adds distinct stored values for categorical TEXT columns.

    Used by the Critic so it can verify string literal case and filter logic
    against the actual values in the database.
    """
    _check_db_exists(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    if not tables:
        conn.close()
        raise RuntimeError(
            f"No tables found in database at '{db_path}'. "
            "The file may be empty or corrupted."
        )

    parts = []
    for table in tables:
        cursor.execute(f"PRAGMA table_info({table})")
        cols = cursor.fetchall()
        col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)

        cursor.execute(f"PRAGMA foreign_key_list({table})")
        fks = cursor.fetchall()
        fk_notes = ""
        if fks:
            fk_notes = " -- FK: " + "; ".join(
                f"{fk[3]} -> {fk[2]}.{fk[4]}" for fk in fks
            )

        parts.append(f"  {table}({col_defs}){fk_notes}")

        for col in cols:
            col_name, col_type = col[1], col[2]
            if col_type.upper() != "TEXT":
                continue
            try:
                cursor.execute(
                    f'SELECT DISTINCT "{col_name}" FROM {table} '
                    f'WHERE "{col_name}" IS NOT NULL ORDER BY "{col_name}"'
                )
                distinct = [r[0] for r in cursor.fetchall()]
                if 2 <= len(distinct) <= 15:
                    values_str = ", ".join(f"'{v}'" for v in distinct)
                    parts.append(f"    -- {col_name} stored values: {values_str}")
            except Exception:
                pass

    conn.close()
    return "Tables:\n" + "\n".join(parts)
