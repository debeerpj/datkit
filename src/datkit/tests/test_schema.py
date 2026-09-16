"""Tests for the schema models and consistency checks."""

import pytest
from pydantic import ValidationError

from datkit.schema import Column, Relationship, Schema, Table, check_schema


def make_column(name: str, type_: str = "varchar") -> Column:
    """Build a Column with minimal ceremony."""
    return Column(name=name, type=type_)


def make_valid_schema() -> Schema:
    """Build a small schema in which every check should pass.

    Two tables joined on a single column, plus a composite-key table joined
    back to the first. Deliberately tiny so a failure is readable.
    """
    basics = Table(
        name="basics",
        source="dbo.basics",
        columns=[make_column("tconst"), make_column("primaryTitle")],
        primary_key=["tconst"],
    )
    ratings = Table(
        name="ratings",
        source="dbo.ratings",
        columns=[make_column("tconst"), make_column("numVotes", "int")],
        primary_key=["tconst"],
    )
    akas = Table(
        name="akas",
        source="dbo.akas",
        columns=[make_column("titleId"), make_column("ordering", "int")],
        primary_key=["titleId", "ordering"],
    )
    ratings_to_basics = Relationship(
        name="ratings_to_basics",
        from_table="ratings",
        from_columns=["tconst"],
        to_table="basics",
        to_columns=["tconst"],
        cardinality="one_to_one",
        coverage="full",
    )
    akas_to_basics = Relationship(
        name="akas_to_basics",
        from_table="akas",
        from_columns=["titleId"],
        to_table="basics",
        to_columns=["tconst"],
        cardinality="many_to_one",
        coverage="full",
    )
    return Schema(
        name="test_schema",
        tables=[basics, ratings, akas],
        relationships=[ratings_to_basics, akas_to_basics],
    )


# --- the happy path -------------------------------------------------------


def test_valid_schema_passes():
    """A coherent schema raises nothing."""
    check_schema(make_valid_schema())


# --- table-level checks ---------------------------------------------------


def test_duplicate_table_name_is_reported():
    """Two tables sharing a name cannot both be indexed."""
    schema = make_valid_schema()
    # Reuse the first table's name on a second, otherwise unrelated table.
    schema.tables.append(Table(name="basics", source="dbo.other", columns=[make_column("x")]))
    with pytest.raises(ValueError, match="Duplicate tables basics"):
        check_schema(schema)


def test_duplicate_column_name_is_reported():
    """Two columns sharing a name within one table."""
    schema = make_valid_schema()
    schema.tables[0].columns.append(make_column("tconst"))
    with pytest.raises(ValueError, match="Duplicate column tconst"):
        check_schema(schema)


def test_primary_key_must_exist_as_a_column():
    """A primary key naming a column the table does not have."""
    schema = make_valid_schema()
    schema.tables[0].primary_key = ["does_not_exist"]
    with pytest.raises(ValueError, match="Primary key does_not_exist"):
        check_schema(schema)


# --- relationship-level checks --------------------------------------------


def test_unknown_from_table_is_reported():
    """A relationship starting at a table that is not defined."""
    schema = make_valid_schema()
    schema.relationships[0].from_table = "nope"
    with pytest.raises(ValueError, match="unknown from_table nope"):
        check_schema(schema)


def test_unknown_to_table_is_reported():
    """A relationship ending at a table that is not defined."""
    schema = make_valid_schema()
    schema.relationships[0].to_table = "nope"
    with pytest.raises(ValueError, match="unknown to_table nope"):
        check_schema(schema)


def test_unknown_from_column_is_reported():
    """A relationship naming a column absent from its from_table."""
    schema = make_valid_schema()
    schema.relationships[0].from_columns = ["not_a_column"]
    with pytest.raises(ValueError, match="unknown column not_a_column"):
        check_schema(schema)


def test_unknown_to_column_is_reported():
    """A relationship naming a column absent from its to_table."""
    schema = make_valid_schema()
    schema.relationships[0].to_columns = ["not_a_column"]
    with pytest.raises(ValueError, match="unknown column not_a_column"):
        check_schema(schema)


def test_mismatched_composite_key_lengths_are_reported():
    """Composite keys pair positionally, so the two sides must be equal length."""
    schema = make_valid_schema()
    # akas has a two-part key; point both parts at a one-column key.
    schema.relationships[1].from_columns = ["titleId", "ordering"]
    with pytest.raises(ValueError, match="mismatched column counts"):
        check_schema(schema)


# --- error collection -----------------------------------------------------


def test_all_errors_are_reported_together():
    """Independent problems appear in one message, not one per run."""
    schema = make_valid_schema()
    schema.tables[0].primary_key = ["does_not_exist"]
    schema.relationships[0].to_columns = ["also_missing"]

    with pytest.raises(ValueError) as excinfo:
        check_schema(schema)

    message = str(excinfo.value)
    assert "does_not_exist" in message
    assert "also_missing" in message


# --- pydantic model validation --------------------------------------------


def test_relationship_requires_at_least_one_from_column():
    """An empty join key is incoherent and should fail at construction."""
    with pytest.raises(ValidationError):
        Relationship(
            name="bad",
            from_table="a",
            from_columns=[],
            to_table="b",
            to_columns=["x"],
            cardinality="many_to_one",
        )


def test_invalid_cardinality_is_rejected():
    """Cardinality is constrained to a fixed set of values."""
    with pytest.raises(ValidationError):
        Relationship(
            name="bad",
            from_table="a",
            from_columns=["x"],
            to_table="b",
            to_columns=["x"],
            cardinality="sideways",
        )
