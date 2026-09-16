"""Test the column cleaning functions."""

import pandas as pd
from pandas.testing import assert_series_equal

from datkit.cleaning import (
    MAX_SPECIALS,
    _guess_format,
    _is_text,
    _to_datetime,
    _to_numeric,
    clean_column,
    clean_column_dtype,
    clean_dataframe,
    clean_dates,
    clean_numbers,
    column_record,
    discover_specials,
    effective_dtype,
    failed_values,
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
        # The sentinel verdict: what the column would be without the values
        # listed in `specials`. Only reproducible alongside them, so the two
        # are recorded together.
        "dtype_effective",
        "specials",
        "specials_count",
        "n_unique",
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


# --- _is_text -------------------------------------------------------------
#
# The guard everything else is gated on, so it gets the most direct tests.
#
# The failure it exists to catch: is_string_dtype is True for ANY object
# column, and object is still the default for text on pandas 2.2.3 — the
# target version. That let an object column of ints reach .str, and an object
# column of mixed types lose its non-string values to coercion. Neither shows
# up on pandas 3 locally, where text arrives as the `str` dtype.


def test_is_text_accepts_strings_in_every_backend(text_dtype):
    """Real text is text whatever the string backend."""
    assert _is_text(pd.Series(["a", "b", None], dtype=text_dtype))


def test_is_text_rejects_object_columns_that_are_not_text():
    """The shapes is_string_dtype gets wrong. Each is True there, False here."""
    not_text = {
        "ints": pd.Series([1, 2, 3], dtype=object),
        "mixed": pd.Series([1, "a", None], dtype=object),
        "bools": pd.Series([True, False], dtype=object),
        "all null": pd.Series([None, None], dtype=object),
        "empty": pd.Series([], dtype=object),
    }
    for label, s in not_text.items():
        # is_string_dtype says True for all of these — that is the bug.
        assert pd.api.types.is_string_dtype(s.dtype), f"{label}: premise changed"
        assert not _is_text(s), f"{label}: should not be treated as text"


def test_is_text_rejects_non_object_dtypes():
    """Numbers, dates and categories are not text."""
    assert not _is_text(pd.Series([1, 2]))
    assert not _is_text(pd.Series([1.5, 2.5]))
    assert not _is_text(pd.to_datetime(pd.Series(["2020-01-01"])))
    assert not _is_text(pd.Series(["a", "b"], dtype="category"))


def test_is_text_ignores_nulls_when_judging():
    """A column of strings is still text with nulls scattered through it."""
    assert _is_text(pd.Series([None, "a", None, "b"], dtype=object))


# --- _guess_format --------------------------------------------------------


def test_guess_format_reads_an_ordinary_date():
    """A plain ISO date gives back its format string."""
    assert _guess_format("2020-03-15", dayfirst=False) == "%Y-%m-%d"


def test_guess_format_retries_lowercase_month_names():
    """The guesser matches month names literally; to_datetime does not.

    'Mar' is in calendar.month_abbr, 'mar' and 'MAR' are not, so the raw
    guesser returns None for them even though the dates parse fine. The retry
    title-cases alphabetic runs so all three spellings agree.
    """
    titled = _guess_format("15-Mar-2020", dayfirst=True)
    assert titled is not None
    assert _guess_format("15-mar-2020", dayfirst=True) == titled
    assert _guess_format("15-MAR-2020", dayfirst=True) == titled


def test_guess_format_returns_none_for_non_dates():
    """Text that is not a date has no format."""
    assert _guess_format("not a date", dayfirst=False) is None


def test_guess_format_dayfirst_misreads_ambiguous_iso_dates():
    """Documents the trap clean_dates works around.

    With dayfirst=True an ISO string whose last field could be a month comes
    back as %Y-%d-%m — year, then DAY, then month. Parsing a column with that
    silently transposes every date in it. clean_dates checks for a %Y prefix
    and switches to the dayfirst=False format because of this.

    Only ambiguous dates are affected, which is what makes it dangerous: a
    sample of dates with days above 12 guesses correctly and the column looks
    fine until a date like 2020-01-02 arrives.
    """
    # Last field is 02 — could be February, so dayfirst reorders it.
    assert _guess_format("2020-01-02", dayfirst=True) == "%Y-%d-%m"

    # Last field is 15 — cannot be a month, so it is read correctly.
    assert _guess_format("2020-03-15", dayfirst=True) == "%Y-%m-%d"


# --- _to_numeric / _to_datetime -------------------------------------------
# Thin wrappers over the raw pandas coercions. They exist so discover_specials
# can see what fails: clean_numbers and clean_dates refuse and hand back the
# original, which discards exactly the evidence being recovered.


def test_to_numeric_nulls_what_will_not_convert():
    """Unconvertible values become null rather than raising."""
    result = _to_numeric(pd.Series(["1", "2", "unknown"]))
    assert result.tolist()[:2] == [1, 2]
    assert pd.isna(result.tolist()[2])


def test_to_datetime_nulls_what_will_not_convert():
    """Same contract for dates."""
    result = _to_datetime(pd.Series(["2020-01-01", "not collected"]))
    assert result.tolist()[0] == pd.Timestamp("2020-01-01")
    assert pd.isna(result.tolist()[1])


def test_to_numeric_differs_from_clean_numbers_on_refusal():
    """The wrapper coerces where clean_numbers refuses. That is the point."""
    raw = pd.Series(["1", "2", "unknown"])

    # clean_numbers protects the data: one loss and the column comes back as-is.
    assert_series_equal(clean_numbers(raw), raw)

    # _to_numeric does not protect it, so the failures are visible.
    assert _to_numeric(raw).isna().sum() == 1


# --- failed_values --------------------------------------------------------


def test_failed_values_reports_what_the_coercion_lost():
    """The distinct original values that became null."""
    raw = pd.Series(["1", "2", "unknown", "3", "REFUSED", "unknown"])
    assert sorted(failed_values(raw, _to_numeric(raw))) == ["REFUSED", "unknown"]


def test_failed_values_is_empty_when_nothing_failed():
    """A clean conversion has no offenders."""
    raw = pd.Series(["1", "2", "3"])
    assert failed_values(raw, _to_numeric(raw)) == []


def test_failed_values_ignores_values_that_were_already_null():
    """A pre-existing null was not lost by the coercion, so it is not blamed."""
    raw = pd.Series(["1", None, "unknown"])
    assert failed_values(raw, _to_numeric(raw)) == ["unknown"]


def test_failed_values_honours_the_limit():
    """Collection stops at the limit rather than building an unbounded list."""
    raw = pd.Series([f"bad_{i}" for i in range(100)])
    assert len(failed_values(raw, _to_numeric(raw), limit=5)) == 5


def test_failed_values_default_limit_exceeds_the_cap_by_one():
    """One over MAX_SPECIALS, so a caller can tell 'at the cap' from 'over it'."""
    raw = pd.Series([f"bad_{i}" for i in range(100)])
    assert len(failed_values(raw, _to_numeric(raw))) == MAX_SPECIALS + 1


# --- discover_specials ----------------------------------------------------


def test_discover_specials_finds_the_blocking_values():
    """A numeric column held back by two tokens proposes exactly those two."""
    s = pd.Series(["1", "2", "unknown", "4", "REFUSED", "6"], dtype=object)
    assert sorted(discover_specials(s)) == ["REFUSED", "unknown"]


def test_discover_specials_finds_date_sentinels():
    """Falls through to the date coercion when the numeric one does not fit."""
    s = pd.Series(["2020-01-01", "2020-02-01", "not collected", "2020-04-01"], dtype=object)
    assert discover_specials(s) == ["not collected"]


def test_discover_specials_rejects_a_column_of_names():
    """The ratio guard rejects a column whose values are all unconvertible.

    Six names are six unconvertible values — under the cap of ten, so without
    the ratio test every one of them would be proposed as a sentinel.
    """
    s = pd.Series(["ann", "bob", "cy", "di", "ed", "fi"], dtype=object)
    assert discover_specials(s) == []


def test_discover_specials_rejects_too_many_distinct_failures():
    """The absolute cap. Scattered unconvertible values are not sentinels."""
    # 20 distinct failures among 200 values: ratio passes, cap does not.
    values = [str(i) for i in range(180)] + [f"bad_{i}" for i in range(20)]
    assert discover_specials(pd.Series(values, dtype=object)) == []


def test_discover_specials_returns_nothing_for_a_clean_column():
    """Nothing is blocking the conversion, so there is nothing to propose."""
    assert discover_specials(pd.Series(["1", "2", "3"], dtype=object)) == []


def test_discover_specials_skips_non_text_columns():
    """An object column of ints is not text.

    Before the gate this reached .str inside remove_whitespace and raised
    AttributeError, because is_string_dtype is True for any object column.
    """
    assert discover_specials(pd.Series([1, 2, 3], dtype=object)) == []
    assert discover_specials(pd.Series([1, 2, 3])) == []


def test_discover_specials_proposes_but_never_acts():
    """The column is not modified, and the sentinels are still in it."""
    s = pd.Series(["1", "2", "unknown"], dtype=object)
    before = s.tolist()
    discover_specials(s)
    assert s.tolist() == before


def test_discover_specials_is_reproducible():
    """The same seed proposes the same candidates."""
    values = [str(i) for i in range(500)] + ["unknown"] * 20
    s = pd.Series(values, dtype=object)
    assert discover_specials(s, sample_n=100, seed=3) == discover_specials(s, sample_n=100, seed=3)


def test_discover_specials_ignores_whitespace_and_null_tokens():
    """Discovery sees what the coercions would see, not the raw column.

    The candidates are found after remove_whitespace and to_null, so a padded
    "N/A" is already a null by then and is not proposed as a sentinel.
    """
    s = pd.Series([" 1", "2 ", " N/A ", "4", "unknown"], dtype=object)
    assert discover_specials(s) == ["unknown"]


# --- effective_dtype ------------------------------------------------------


def test_effective_dtype_reports_what_the_column_would_be():
    """With the sentinels set aside, the column is numeric."""
    s = pd.Series(["1", "2", "unknown", "4"], dtype=object)
    assert effective_dtype(s, ["unknown"]) == "int64"


def test_effective_dtype_with_no_specials_is_the_ordinary_dtype():
    """An empty list makes this the dtype clean_column would produce."""
    s = pd.Series(["1", "2", "3"], dtype=object)
    assert effective_dtype(s, []) == str(clean_column(s).dtype)


def test_effective_dtype_leaves_the_column_alone():
    """A claim about the column, not a change to it."""
    s = pd.Series(["1", "2", "unknown"], dtype=object)
    before = s.tolist()
    effective_dtype(s, ["unknown"])
    assert s.tolist() == before


def test_effective_dtype_when_everything_is_excluded():
    """Excluding every value leaves nothing to infer from, so the dtype stands."""
    s = pd.Series(["unknown", "unknown"], dtype=object)
    assert effective_dtype(s, ["unknown"]) == str(s.dtype)


def test_effective_dtype_drops_rather_than_nulls():
    """Sentinels are removed, not nulled.

    Nulling them would register as a failed coercion inside clean_numbers,
    which would then refuse the column — leaving the effective dtype equal to
    the original and the whole exercise pointless.
    """
    s = pd.Series(["1", "2", "unknown", "4"], dtype=object)
    assert effective_dtype(s, ["unknown"]) != str(s.dtype)


# --- clean_column_dtype ---------------------------------------------------


def test_clean_column_dtype_matches_clean_column():
    """The probe agrees with the thing it is probing."""
    for values in (["1", "2", "3"], ["2020-01-01", "2020-02-01"], ["a", "b"]):
        s = pd.Series(values, dtype=object)
        assert clean_column_dtype(s) == clean_column(s).dtype


def test_clean_column_dtype_does_not_return_the_cleaned_column():
    """It reports a dtype, so it cannot be mistaken for the cleaner."""
    assert not isinstance(clean_column_dtype(pd.Series(["1", "2"])), pd.Series)


def test_clean_column_dtype_skips_non_text_columns():
    """Same gate as clean_column. This raised AttributeError before the fix."""
    s = pd.Series([1, 2, 3], dtype=object)
    assert clean_column_dtype(s) == s.dtype


# --- column_record --------------------------------------------------------
# Split out of clean_dataframe so the record's shape can be tested without
# cleaning a frame to get one.


def make_record(values=("1", "2", "unknown", "4"), specials=("unknown",)):
    """Build one record row from a column and its sentinels."""
    before = pd.Series(list(values), dtype=object)
    after = clean_column(before)
    return column_record(
        name="votes",
        before=before,
        after=after,
        specials=list(specials),
        dtype_effective=effective_dtype(before, list(specials)),
        n_unique=int(before.nunique()),
    )


def test_column_record_has_the_expected_fields():
    """The row's shape is the contract clean_dataframe builds its frame from."""
    assert set(make_record()) == {
        "column",
        "dtype_before",
        "dtype_after",
        "converted",
        "dtype_effective",
        "specials",
        "specials_count",
        "n_unique",
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


def test_column_record_counts_the_sentinel_rows():
    """specials_count is rows, not distinct values."""
    record = make_record(values=("1", "unknown", "unknown", "4"))
    assert record["specials_count"] == 2


def test_column_record_counts_no_sentinels_when_there_are_none():
    """An empty sentinel list counts zero rows, with no special case.

    isin against an empty list is all False, so the count falls out correctly.
    """
    record = make_record(values=("1", "2", "3"), specials=())
    assert record["specials_count"] == 0


def test_column_record_reports_the_effective_dtype_alongside_the_actual():
    """The two verdicts differ exactly when sentinels are holding a column back."""
    held_back = make_record(values=("1", "2", "unknown", "4"))
    assert held_back["dtype_after"] == "object"
    assert held_back["dtype_effective"] == "int64"

    clean = make_record(values=("1", "2", "3"), specials=())
    assert clean["dtype_after"] == clean["dtype_effective"]


def test_column_record_flags_conversion():
    """`converted` compares the dtype before and after, nothing else."""
    assert make_record(values=("1", "2", "3"), specials=())["converted"]
    assert not make_record(values=("a", "b"), specials=())["converted"]


def test_column_record_bytes_saved_is_the_difference():
    """The derived byte fields agree with the measured ones."""
    record = make_record(values=("1", "2", "3"), specials=())
    assert record["bytes_saved"] == record["bytes_before"] - record["bytes_after"]
    assert record["mb_saved"] == round(record["bytes_saved"] / 1024**2, 2)


def test_column_record_nulls_created_is_zero_when_nothing_is_lost():
    """A conversion that keeps every value reports no new nulls."""
    assert make_record(values=("1", "2", "3"), specials=())["nulls_created"] == 0


def test_column_record_n_unique_is_carried_through():
    """Recorded so bin_column can skip a cardinality probe of its own."""
    assert make_record(values=("1", "1", "2"), specials=())["n_unique"] == 2
