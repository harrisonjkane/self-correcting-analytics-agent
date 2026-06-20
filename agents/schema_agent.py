"""Inspects the database schema and returns formatted strings for use in prompts."""

import sqlite3


def get_schema(db_path: str) -> str:
    """Basic schema: table definitions + FK notes. Used for SQL generation."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    if not tables:
        conn.close()
        raise RuntimeError(
            f"No tables found in database at '{db_path}'. "
            "The file may be missing, empty, or corrupted."
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


def get_schema_with_enums(db_path: str) -> str:
    """Extended schema: adds distinct stored values for categorical TEXT columns.

    Used by the Critic so it can verify string literal case and filter logic
    against the actual values in the database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]
    if not tables:
        conn.close()
        raise RuntimeError(
            f"No tables found in database at '{db_path}'. "
            "The file may be missing, empty, or corrupted."
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
