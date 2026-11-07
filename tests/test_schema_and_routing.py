"""Schema introspection and chart routing — no LLM calls."""
import pytest

from agents.chart_agent import should_chart
from agents.schema_agent import get_schema, get_schema_with_enums

TABLES = ["customers", "products", "orders", "order_items", "support_tickets"]


def test_schema_lists_all_tables(db_path):
    schema = get_schema(db_path)
    for table in TABLES:
        assert table in schema


def test_schema_with_enums_exposes_stored_casing(db_path):
    # The Critic's case-sensitivity check depends on seeing real stored values.
    assert "Completed" in get_schema_with_enums(db_path)


def test_missing_db_raises_without_creating_file(tmp_path):
    missing = tmp_path / "nope.db"
    with pytest.raises(RuntimeError):
        get_schema(str(missing))
    assert not missing.exists()


@pytest.mark.parametrize(
    "result, expected",
    [
        ({"SMB": 1.0, "Enterprise": 2.0}, True),
        (["a", "b", "c"], True),
        (["a", "b"], False),
        ({"only": 1.0}, False),
        (60, False),
    ],
)
def test_should_chart(result, expected):
    assert should_chart("any question", result) is expected
