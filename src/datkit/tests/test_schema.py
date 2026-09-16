"""Tests for the schema models and consistency checks."""

import pytest
from pydantic import ValidationError

from datkit.schema import Column, Relationship, Schema, Table, check_schema, load_schema_from_yaml


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


# --- load_schema_from_yaml ------------------------------------------------
# tmp_path is a pytest built-in fixture: a fresh empty directory per test,
# cleaned up afterwards. Used here so the suite carries no fixture files —
# the tests ship inside the wheel and must not depend on anything outside it.


def write_yaml(tmp_path, text: str):
    """Write a schema YAML file and return its path."""
    path = tmp_path / "schema.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_schema_from_yaml_reads_a_valid_file(tmp_path):
    """A well-formed file becomes a Schema with its tables and columns."""
    path = write_yaml(
        tmp_path,
        """
        name: imdb
        tables:
          - name: title_basics
            source: title.basics.tsv
            primary_key: [tconst]
            columns:
              - name: tconst
                type: varchar
              - name: startYear
                type: int
        """,
    )
    schema = load_schema_from_yaml(path)

    assert schema.name == "imdb"
    assert [t.name for t in schema.tables] == ["title_basics"]
    assert [c.name for c in schema.tables[0].columns] == ["tconst", "startYear"]


def test_load_schema_from_yaml_accepts_a_string_path(tmp_path):
    """The signature allows str as well as Path."""
    path = write_yaml(tmp_path, "name: imdb\ntables: []\n")
    assert load_schema_from_yaml(str(path)).name == "imdb"


def test_load_schema_from_yaml_applies_column_defaults(tmp_path):
    """A column with no type recorded comes back as 'unknown', not blank."""
    path = write_yaml(
        tmp_path,
        """
        name: imdb
        tables:
          - name: t
            source: s
            columns:
              - name: c
        """,
    )
    column = load_schema_from_yaml(path).tables[0].columns[0]
    assert column.type == "unknown"
    assert column.tags == []


def test_load_schema_from_yaml_rejects_an_empty_file(tmp_path):
    """An empty file is refused at load time, not later.

    safe_load returns None for an empty file, which would fail obscurely
    downstream. The loader raises a message naming the file instead.
    """
    path = write_yaml(tmp_path, "")
    with pytest.raises(ValueError, match="empty or invalid"):
        load_schema_from_yaml(path)


def test_load_schema_from_yaml_rejects_a_comments_only_file(tmp_path):
    """Also None from safe_load, and a plausible way to get an empty file."""
    path = write_yaml(tmp_path, "# nothing here yet\n")
    with pytest.raises(ValueError, match="empty or invalid"):
        load_schema_from_yaml(path)


def test_load_schema_from_yaml_rejects_a_schema_that_does_not_validate(tmp_path):
    """Model validation still applies to a file that parses as YAML.

    An empty column name is caught at load time, not at the point the schema
    is eventually used.
    """
    path = write_yaml(
        tmp_path,
        """
        name: imdb
        tables:
          - name: t
            source: s
            columns:
              - name: ""
        """,
    )
    with pytest.raises(ValidationError):
        load_schema_from_yaml(path)


def test_load_schema_from_yaml_raises_when_the_file_is_missing(tmp_path):
    """A missing file surfaces as FileNotFoundError from open()."""
    with pytest.raises(FileNotFoundError):
        load_schema_from_yaml(tmp_path / "not_here.yaml")
