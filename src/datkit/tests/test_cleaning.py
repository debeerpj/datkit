"""Test the column cleaning functions."""

import pandas as pd
from pandas.testing import assert_series_equal

from datkit.cleaning import (
    clean_column,
    clean_dataframe,
    clean_dates,
    clean_numbers,
    missing_count,
    remove_whitespace,
    sample_records,
    to_null,
    uppercase,
)

# Comparing two Series with `==` produces a Series of booleans, and `assert` on
# that raises "truth value is ambiguous". Use assert_series_equal, which also
# checks dtype and index, or compare .tolist() when only the values matter.


# --- remove_whitespace ----------------------------------------------------


def test_remove_whitespace_strips_both_ends():
    """Leading and trailing whitespace goes; internal whitespace stays."""
    result = remove_whitespace(pd.Series([" before", "after ", " both ", "mid dle"]))
    assert_series_equal(result, pd.Series(["before", "after", "both", "mid dle"]))


def test_remove_whitespace_preserves_nulls():
    """A null stays null rather than becoming the string 'nan'."""
    result = remove_whitespace(pd.Series([" a ", None]))
    assert result.tolist()[0] == "a"
    assert pd.isna(result.tolist()[1])


def test_remove_whitespace_leaves_numeric_untouched():
    """A non-string column passes through unchanged."""
    original = pd.Series([1, 2, 3])
    assert_series_equal(remove_whitespace(original), original)


# --- to_null --------------------------------------------------------------


def test_to_null_converts_default_tokens():
    """Each default placeholder becomes NaN."""
    result = to_null(pd.Series([r"\N", "NA", "n/a", "NULL", "-", ""]))
    assert result.isna().all()


def test_to_null_leaves_real_values():
    """Values that are not placeholders survive."""
    result = to_null(pd.Series(["tt0001", "NA", "drama"]))
    assert result.tolist()[0] == "tt0001"
    assert pd.isna(result.tolist()[1])
    assert result.tolist()[2] == "drama"


def test_to_null_accepts_custom_tokens():
    """A caller-supplied token set replaces the defaults."""
    result = to_null(pd.Series(["MISSING", "NA"]), tokens=("MISSING",))
    assert pd.isna(result.tolist()[0])
    # "NA" is not in the custom set, so it is left alone.
    assert result.tolist()[1] == "NA"


def test_to_null_does_not_match_substrings():
    """A value merely containing a token is not nulled."""
    result = to_null(pd.Series(["NAME", "BANANA"]))
    assert result.notna().all()


def test_to_null_leaves_numeric_untouched():
    """A non-string column passes through unchanged."""
    original = pd.Series([1, 2, 3])
    assert_series_equal(to_null(original), original)


def test_to_null_returns_a_new_series():
    """Cleaning does not mutate the caller's data in place."""
    original = pd.Series(["NA", "x"])
    before = original.tolist()
    to_null(original)
    assert original.tolist() == before


# --- clean_numbers --------------------------------------------------------


def test_clean_numbers_strips_currency_and_separators():
    """Currency symbols and thousands separators are removed."""
    result = clean_numbers(pd.Series(["$1,234", "R56", "78%"]))
    assert result.tolist() == [1234, 56, 78]


def test_clean_numbers_returns_numeric_dtype():
    """The result is a real numeric column, not text."""
    result = clean_numbers(pd.Series(["1", "2", "3"]))
    assert pd.api.types.is_numeric_dtype(result)


def test_clean_numbers_refuses_non_numeric_column():
    """A column that would lose data to coercion is returned untouched."""
    original = pd.Series(["drama", "comedy", "documentary"])
    assert_series_equal(clean_numbers(original), original)


def test_clean_numbers_preserves_existing_nulls():
    """Nulls already present do not count as coercion losses."""
    result = clean_numbers(pd.Series(["1", None, "3"]))
    assert result.tolist()[0] == 1
    assert pd.isna(result.tolist()[1])
    assert result.tolist()[2] == 3


def test_clean_numbers_does_not_partially_convert():
    """One unconvertible value leaves the whole column as text."""
    original = pd.Series(["100", "200", "not a number"])
    assert_series_equal(clean_numbers(original), original)


def test_clean_numbers_leaves_numeric_untouched():
    """An already-numeric column passes through unchanged."""
    original = pd.Series([1.5, 2.5])
    assert_series_equal(clean_numbers(original), original)


# --- uppercase ------------------------------------------------------------


def test_uppercase_converts_strings():
    """Plain string columns are uppercased."""
    result = uppercase(pd.Series(["abc", "DeF"]))
    assert result.tolist() == ["ABC", "DEF"]


# Categorical columns are not handled yet — deferred, because data arriving with
# categories already defined is unlikely. When added, note that
# pd.api.types.is_string_dtype returns False for a categorical even when its
# categories are strings, so it needs its own branch, and that
# s.cat.rename_categories(str.upper) is the cheap path (it edits the category
# dictionary rather than every row) but raises ValueError on a collision.


def test_uppercase_leaves_numeric_untouched():
    """A non-string column passes through unchanged."""
    original = pd.Series([1, 2, 3])
    assert_series_equal(uppercase(original), original)


# --- sample_records -------------------------------------------------------


def test_sample_records_returns_all_when_smaller_than_n():
    """A frame shorter than n comes back whole."""
    df = pd.DataFrame({"a": range(5)})
    assert len(sample_records(df, n=10)) == 5


def test_sample_records_caps_at_n():
    """A frame longer than n is reduced to n rows."""
    df = pd.DataFrame({"a": range(100)})
    assert len(sample_records(df, n=10)) == 10


def test_sample_records_is_reproducible():
    """The same seed selects the same rows."""
    df = pd.DataFrame({"a": range(100)})
    first = sample_records(df, n=10, seed=7)
    second = sample_records(df, n=10, seed=7)
    assert first.index.tolist() == second.index.tolist()


def test_sample_records_seed_changes_selection():
    """A different seed selects different rows."""
    df = pd.DataFrame({"a": range(1000)})
    first = sample_records(df, n=10, seed=1)
    second = sample_records(df, n=10, seed=2)
    assert first.index.tolist() != second.index.tolist()


# --- boundaries between functions -----------------------------------------
# Each function matches on exact values, so what it sees depends on what ran
# before it. These pin down that boundary. The test that the pipeline actually
# applies them in the right order belongs on the driver function.


def test_to_null_does_not_strip_whitespace():
    """A padded placeholder is not matched — stripping is a separate step."""
    padded = pd.Series([" NA ", "\tNULL"])

    # to_null matches exact values, so padded tokens pass through.
    assert to_null(padded).notna().all()

    # Once stripped, the same values are recognised.
    assert to_null(remove_whitespace(padded)).isna().all()


def test_clean_numbers_treats_placeholder_text_as_a_loss():
    """Unconverted placeholder text trips the coercion-loss guard."""
    raw = pd.Series(["1", "2", "NA"])

    # "NA" coerces to NaN, so clean_numbers sees a loss and refuses.
    assert not pd.api.types.is_numeric_dtype(clean_numbers(raw))

    # Already-null values are accounted for, so conversion proceeds.
    assert pd.api.types.is_numeric_dtype(clean_numbers(to_null(raw)))


def test_uppercase_changes_which_tokens_match():
    """Casing affects placeholder matching when the token set is narrow."""
    raw = pd.Series(["na"])

    # The default token set carries both cases.
    assert to_null(raw).isna().all()

    # A narrow client-specific set no longer matches once uppercased.
    assert to_null(uppercase(raw), tokens=("na",)).notna().all()


# --- clean_dates ----------------------------------------------------------


def test_clean_dates_converts_iso_dates():
    """A clean ISO date column becomes datetime64."""
    result = clean_dates(pd.Series(["2024-01-01", "2024-02-15", "2024-03-30"]))
    assert pd.api.types.is_datetime64_any_dtype(result)
    assert result.tolist()[0] == pd.Timestamp("2024-01-01")


def test_clean_dates_refuses_free_text():
    """A column of text is returned untouched, not converted to all-NaT."""
    original = pd.Series(["drama", "comedy", "documentary"])
    assert_series_equal(clean_dates(original), original)


def test_clean_dates_refuses_mostly_dates_with_one_bad_value():
    """A single unparseable value vetoes the conversion.

    Strict by design: any growth in the null count means data would be lost.
    """
    original = pd.Series(["2024-01-01"] * 19 + ["garbage"])
    assert_series_equal(clean_dates(original), original)


def test_clean_dates_preserves_existing_nulls():
    """Nulls already present do not count as conversion losses."""
    result = clean_dates(pd.Series(["2024-01-01", None, "2024-03-30"]))
    assert pd.api.types.is_datetime64_any_dtype(result)
    assert pd.isna(result.tolist()[1])


def test_clean_dates_reads_unambiguous_day_first():
    """25/12/2024 can only be 25 December, whichever order is assumed."""
    result = clean_dates(pd.Series(["25/12/2024", "13/01/2025", "30/06/2024"]))
    assert result.tolist()[0] == pd.Timestamp("2024-12-25")
    assert result.tolist()[1] == pd.Timestamp("2025-01-13")


def test_clean_dates_reads_unambiguous_month_first():
    """12/25/2024 can only be 25 December, whichever order is assumed."""
    result = clean_dates(pd.Series(["12/25/2024", "01/13/2025", "06/30/2024"]))
    assert result.tolist()[0] == pd.Timestamp("2024-12-25")
    assert result.tolist()[1] == pd.Timestamp("2025-01-13")


def test_clean_dates_resolves_ambiguous_dates_day_first():
    """03/04/2025 is genuinely ambiguous; the current behaviour is day-first.

    This documents the choice rather than endorsing it — both interpretations
    parse cleanly, and dayfirst wins only because it is tried first.
    """
    result = clean_dates(pd.Series(["03/04/2025", "05/06/2025", "07/08/2025"]))
    assert result.tolist()[0] == pd.Timestamp("2025-04-03")


def test_clean_dates_leaves_numeric_untouched():
    """A non-string column passes through unchanged."""
    original = pd.Series([1, 2, 3])
    assert_series_equal(clean_dates(original), original)


def test_clean_dates_leaves_datetime_untouched():
    """An already-parsed datetime column passes through unchanged."""
    original = pd.to_datetime(pd.Series(["2024-01-01", "2024-02-01"]))
    assert_series_equal(clean_dates(original), original)


def test_clean_dates_returns_a_new_series():
    """Cleaning does not mutate the caller's data in place."""
    original = pd.Series(["2024-01-01", "2024-02-15"])
    before = original.tolist()
    clean_dates(original)
    assert original.tolist() == before


# Known gaps, deliberately not asserted yet:
#   - A year-only column (["2019", "2020"]) converts to 1 January of that year.
#     Needs a minimum-length guard on the sampled values.
#   - A long column with fewer than 10 non-null values raises from
#     .sample(n=10): "Cannot take a larger sample than population".


# --- clean_column (the pipeline) ------------------------------------------
# These are the ordering tests. Each asserts an end-to-end outcome that only
# holds if the steps run in the right sequence — reorder clean_column and one
# of these fails and names the reason.


def test_pipeline_handles_padded_placeholders():
    """A padded placeholder is stripped, nulled, and does not block conversion.

    Requires remove_whitespace before to_null, and to_null before clean_numbers.
    """
    result = clean_column(pd.Series([" NA ", "2", "3"]))
    assert pd.api.types.is_numeric_dtype(result)
    assert pd.isna(result.tolist()[0])
    assert result.tolist()[1] == 2


def test_pipeline_converts_money_text_to_numbers():
    """Currency symbols, separators and padding all clear."""
    result = clean_column(pd.Series(["$1,234", " $56 ", "78"]))
    assert result.tolist() == [1234, 56, 78]


def test_pipeline_converts_iso_dates():
    """A date column reaches clean_dates intact and converts."""
    result = clean_column(pd.Series(["2024-01-01", "2024-02-15"]))
    assert pd.api.types.is_datetime64_any_dtype(result)


def test_pipeline_converts_day_first_dates():
    """Unambiguous day-first dates survive the whole pipeline."""
    result = clean_column(pd.Series(["25/12/2024", "13/01/2025", "30/06/2024"]))
    assert result.tolist()[0] == pd.Timestamp("2024-12-25")


def test_pipeline_converts_dates_written_with_month_names():
    """Uppercasing before date parsing does not break month-name formats."""
    result = clean_column(pd.Series(["12 mar 2024", "05 jun 2024"]))
    assert result.tolist()[0] == pd.Timestamp("2024-03-12")


def test_pipeline_leaves_year_only_columns_numeric():
    """A year column is claimed by clean_numbers, not turned into January dates.

    clean_numbers runs before clean_dates, so numeric-looking text never
    reaches the date parser. This is what protects a year column.
    """
    result = clean_column(pd.Series(["2019", "2020", "2021"]))
    assert pd.api.types.is_numeric_dtype(result)
    assert result.tolist() == [2019, 2020, 2021]


def test_pipeline_does_not_recognise_yyyymmdd_dates():
    """Known gap: compact numeric dates are claimed by clean_numbers.

    20240101 is a real date format in SQL extracts, but clean_numbers converts
    it first, so clean_dates never sees it. Documented, not endorsed.
    """
    result = clean_column(pd.Series(["20240101", "20240215"]))
    assert pd.api.types.is_numeric_dtype(result)


def test_pipeline_does_not_uppercase_free_text():
    """The uppercase step is opt-in, not part of the default pipeline.

    It is lossy — titles, names and descriptions cannot be recovered — and it
    is the step most likely to exhaust memory on a large object column. Call
    it deliberately if you want it.
    """
    result = clean_column(pd.Series(["The Matrix", "Amelie"]))
    assert result.tolist() == ["The Matrix", "Amelie"]


def test_pipeline_is_idempotent():
    """Cleaning an already-cleaned column changes nothing further."""
    for values in (
        [" NA ", "2", "3"],
        ["$1,234", " $56 "],
        ["2024-01-01", "2024-02-15"],
        ["The Matrix", "Amelie"],
    ):
        once = clean_column(pd.Series(values))
        twice = clean_column(once)
        assert_series_equal(once, twice)


def test_pipeline_does_not_mutate_the_input():
    """The caller's Series is left untouched."""
    original = pd.Series([" NA ", "2", "3"])
    before = original.tolist()
    clean_column(original)
    assert original.tolist() == before


# --- clean_dataframe ------------------------------------------------------


def make_messy_frame() -> pd.DataFrame:
    """A small frame exercising each cleaning path once."""
    return pd.DataFrame(
        {
            "amount": ["$1,234", " $56 ", "78"],
            "started": ["2024-01-01", "2024-02-15", "2024-03-30"],
            "title": ["The Matrix", "Amelie", "Se7en"],
            "status": [" NA ", "open", "closed"],
        }
    )


def test_clean_dataframe_cleans_every_column():
    """Each column reaches the dtype its content implies."""
    cleaned, _ = clean_dataframe(make_messy_frame())

    assert pd.api.types.is_numeric_dtype(cleaned["amount"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["started"])
    # uppercase is not in the default pipeline, so titles keep their casing.
    assert cleaned["title"].tolist() == ["The Matrix", "Amelie", "Se7en"]
    assert pd.isna(cleaned["status"].tolist()[0])


def test_clean_dataframe_preserves_shape_and_column_order():
    """Cleaning changes values and dtypes, never the frame's shape."""
    original = make_messy_frame()
    cleaned, _ = clean_dataframe(original)

    assert cleaned.shape == original.shape
    assert cleaned.columns.tolist() == original.columns.tolist()
    assert cleaned.index.tolist() == original.index.tolist()


def test_clean_dataframe_copies_by_default():
    """The caller's frame is untouched unless inplace is requested."""
    original = make_messy_frame()
    before = original["amount"].tolist()

    clean_dataframe(original)

    assert original["amount"].tolist() == before
    # Still text — asserting "not numeric" rather than a specific dtype, since
    # pandas infers object or str depending on version.
    assert not pd.api.types.is_numeric_dtype(original["amount"])


def test_clean_dataframe_inplace_modifies_the_caller():
    """With inplace=True the caller's frame is changed and returned."""
    original = make_messy_frame()

    cleaned, _ = clean_dataframe(original, inplace=True)

    assert cleaned is original
    assert pd.api.types.is_numeric_dtype(original["amount"])


# --- the record -----------------------------------------------------------


def test_record_has_one_row_per_column():
    """Every column is accounted for, in order."""
    frame = make_messy_frame()
    _, record = clean_dataframe(frame)

    assert len(record) == len(frame.columns)
    assert record["column"].tolist() == frame.columns.tolist()


def test_record_reports_dtype_changes():
    """Converted is True exactly where the dtype changed."""
    _, record = clean_dataframe(make_messy_frame())
    by_column = record.set_index("column")

    assert by_column.loc["amount", "converted"]
    assert by_column.loc["started", "converted"]
    # title stays text, so its dtype is unchanged.
    assert not by_column.loc["title", "converted"]


def test_record_reports_nulls_created_by_placeholder_removal():
    """Nulling a placeholder is a real null, and is reported as created."""
    _, record = clean_dataframe(make_messy_frame())
    by_column = record.set_index("column")

    assert by_column.loc["status", "nulls_before"] == 0
    assert by_column.loc["status", "nulls_after"] == 1
    assert by_column.loc["status", "nulls_created"] == 1


def test_record_creates_no_nulls_in_convertible_columns():
    """A guard-protected conversion never loses values."""
    _, record = clean_dataframe(make_messy_frame())
    by_column = record.set_index("column")

    for column in ("amount", "started", "title"):
        assert by_column.loc[column, "nulls_created"] == 0


def test_record_reports_memory_change():
    """bytes_saved is the difference between the before and after sizes."""
    _, record = clean_dataframe(make_messy_frame())

    computed = record["bytes_before"] - record["bytes_after"]
    assert record["bytes_saved"].tolist() == computed.tolist()
    # Text to numeric should shrink the column.
    assert record.set_index("column").loc["amount", "bytes_saved"] > 0


def test_record_is_a_dataframe_with_the_expected_columns():
    """The record's shape is part of the contract other code relies on."""
    _, record = clean_dataframe(make_messy_frame())

    expected = {
        "column",
        "dtype_before",
        "dtype_after",
        "converted",
        "bytes_before",
        "bytes_after",
        "bytes_saved",
        "mb_before",
        "mb_after",
        "mb_saved",
        "nulls_before",
        "nulls_after",
        "nulls_created",
    }
    assert set(record.columns) == expected


# --- edge cases -----------------------------------------------------------


def test_clean_dataframe_handles_an_empty_frame():
    """No columns means no work and an empty record."""
    cleaned, record = clean_dataframe(pd.DataFrame())

    assert cleaned.empty
    assert record.empty


def test_clean_dataframe_is_idempotent():
    """Cleaning an already-clean frame changes nothing further."""
    once, _ = clean_dataframe(make_messy_frame())
    twice, record = clean_dataframe(once)

    for column in once.columns:
        assert_series_equal(once[column], twice[column])
    assert (record["nulls_created"] == 0).all()


# --- backend parity -------------------------------------------------------
# The target environment decides whether strings arrive as object, pandas
# StringDtype, or Arrow-backed. These run once per available backend via the
# `text_dtype` fixture in conftest.py.
#
# The failure they exist to catch: in Arrow, NaN is a valid double distinct
# from null, so `count()` treats it as PRESENT. Under numpy it is MISSING. A
# guard written as `len(s) - s.count()` therefore passes under Arrow and lets
# a failed numeric coercion through, turning every text column into NaN.


def test_missing_count_agrees_across_backends(text_dtype):
    """Nulls are counted the same whatever the string backend."""
    s = pd.Series(["a", None, "c"], dtype=text_dtype)
    assert missing_count(s) == 1


def test_missing_count_treats_nan_as_missing(text_dtype):
    """A coerced NaN counts as missing even where Arrow calls it present."""
    s = pd.Series(["1", "not a number", "3"], dtype=text_dtype)
    coerced = pd.to_numeric(s.astype("string"), errors="coerce")  # dtype follows the input backend
    assert missing_count(coerced) >= 1


def test_id_column_is_not_destroyed(text_dtype):
    """An identifier column survives cleaning on every backend.

    This is the regression test for the Arrow NaN bug: without a portable
    missing count, clean_numbers converted this to an all-NaN float column.
    """
    original = ["tt0000001", "tt0000002", "tt0000003"]
    result = clean_column(pd.Series(original, dtype=text_dtype))

    assert not pd.api.types.is_numeric_dtype(result)
    assert result.tolist() == original


def test_free_text_is_not_destroyed(text_dtype):
    """Descriptive text survives cleaning on every backend."""
    original = ["The Matrix", "Amelie", "Se7en"]
    result = clean_column(pd.Series(original, dtype=text_dtype))

    assert not pd.api.types.is_numeric_dtype(result)
    assert result.tolist() == original


def test_numeric_text_still_converts(text_dtype):
    """The guard does not become so strict that real numbers stop converting."""
    result = clean_column(pd.Series(["1", "2", "3"], dtype=text_dtype))

    assert pd.api.types.is_numeric_dtype(result)
    assert result.tolist() == [1, 2, 3]


def test_placeholders_are_nulled_on_every_backend(text_dtype):
    """Placeholder nulling works the same regardless of representation."""
    result = to_null(pd.Series([r"\N", "NA", "keep"], dtype=text_dtype))

    assert missing_count(result) == 2
    assert result.tolist()[2] == "keep"


def test_whitespace_is_stripped_on_every_backend(text_dtype):
    """Whitespace stripping works the same regardless of representation."""
    result = remove_whitespace(pd.Series([" a ", "b "], dtype=text_dtype))
    assert result.tolist() == ["a", "b"]


def test_dates_convert_on_every_backend(text_dtype):
    """A date column is recognised whatever the string representation."""
    result = clean_column(pd.Series(["2024-01-01", "2024-02-15"], dtype=text_dtype))
    assert pd.api.types.is_datetime64_any_dtype(result)


def test_clean_dataframe_reports_no_false_losses(text_dtype):
    """A mixed frame cleans without the record claiming lost values."""
    frame = pd.DataFrame(
        {
            "tconst": pd.Series(["tt01", "tt02", "tt03"], dtype=text_dtype),
            "title": pd.Series(["The Matrix", "Amelie", "Se7en"], dtype=text_dtype),
            "votes": pd.Series(["10", "20", "30"], dtype=text_dtype),
        }
    )

    cleaned, record = clean_dataframe(frame)

    # Text columns keep their values; the numeric one converts.
    assert cleaned["tconst"].tolist() == ["tt01", "tt02", "tt03"]
    assert cleaned["title"].tolist() == ["The Matrix", "Amelie", "Se7en"]
    assert pd.api.types.is_numeric_dtype(cleaned["votes"])

    # Nothing was lost anywhere.
    assert (record["nulls_created"] == 0).all()
