"""Schema definitions for the datkit package."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Column(BaseModel):
    """A single column with its type and tags.

    Type should refer to SQL Sever types for this first version.
    """

    name: str = Field(min_length=1, description="the name of the column")  # not just a string — a non-empty one
    type: str = Field(default="unknown", min_length=1, description="the type of the column as a SQL Server type")
    tags: list[str] = []


class Table(BaseModel):
    """A single table with its columns and primary key."""

    name: str = Field(min_length=1, description="the name of the table")
    source: str = Field(min_length=1, description="the source of the table")
    columns: list[Column] = []
    primary_key: list[str] = []


class Relationship(BaseModel):
    """A single relationship between two tables with its columns."""

    name: str = Field(min_length=1, description="the name of the relationship")
    from_table: str = Field(min_length=1, description="the table that the relationship starts from")
    from_columns: list[str] = Field(
        min_length=1, description="the columns in the from_table that are part of the relationship"
    )  # same syntax checks for at least 1 item
    to_table: str = Field(min_length=1, description="the table that the relationship ends at")
    to_columns: list[str] = Field(
        min_length=1, description="the columns in the to_table that are part of the relationship"
    )
    cardinality: Literal["one_to_one", "one_to_many", "many_to_one", "many_to_many"]
    coverage: Literal["full", "partial"] = "partial"


class Schema(BaseModel):
    """A schema is a collection of tables and relationships."""

    name: str = Field(min_length=1, description="the name of the schema")
    tables: list[Table] = []
    relationships: list[Relationship] = []


def check_schema(schema: Schema) -> None:
    """Check the schema for consistency and validity.

    Raises a ValueError if the schema is invalid.
    """
    # list to hold the errors
    errors: list[str] = []

    # index for column lookup and checks for duplicate table names
    tables_index: dict[str, Table] = {}
    # another index for column names and checks for duplicates in a table
    # also checks primary keys exist in the table
    tables_columns_index: dict[str, set[str]] = {}
    for table in schema.tables:
        if table.name in tables_index:
            errors.append(f"Duplicate tables {table.name}")
        else:
            tables_index[table.name] = table
            tables_columns_index[table.name] = set()
            for column in table.columns:
                if column.name in tables_columns_index.get(table.name, set()):
                    errors.append(f"Duplicate column {column.name} in table {table.name}")
                else:
                    tables_columns_index[table.name].add(column.name)
            for primarykey in table.primary_key:
                if primarykey not in tables_columns_index[table.name]:
                    errors.append(f"Primary key {primarykey} in table {table.name} does not exist as a column")

    # relationship validation checks
    for rel in schema.relationships:
        # table names exist
        if rel.from_table not in tables_index:
            errors.append(f"Relationship {rel.name} refers to unknown from_table {rel.from_table}")
            continue  # skip further checks for this relationship if from_table is unknown
        if rel.to_table not in tables_index:
            errors.append(f"Relationship {rel.name} refers to unknown to_table {rel.to_table}")
            continue  # skip further checks for this relationship if to_table is unknown

        # column names exist in relevant tables
        for column in rel.from_columns:
            if column not in tables_columns_index[rel.from_table]:
                errors.append(f"Relationship {rel.name} refers to unknown column {column} in table {rel.from_table}")

        for column in rel.to_columns:
            if column not in tables_columns_index[rel.to_table]:
                errors.append(f"Relationship {rel.name} refers to unknown column {column} in table {rel.to_table}")

        # from_columns and to_columns must have the same length
        if len(rel.from_columns) != len(rel.to_columns):
            errors.append(f"Relationship {rel.name} has mismatched column counts")

    if errors:
        raise ValueError(f"Schema {schema.name!r} has errors:\n" + "\n".join(errors))


def load_schema_from_yaml(path: str | Path) -> Schema:
    """Load a schema from a YAML file."""
    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"Schema file {path} is empty or invalid YAML.")

    schema = Schema.model_validate(data)

    return schema
