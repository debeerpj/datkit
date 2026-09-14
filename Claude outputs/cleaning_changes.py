"""DRAFT — proposed additions to cleaning.py. Not merged, for review.

Nothing here changes what `clean_column` returns. A column that will not
convert still comes back as it arrived, with its sentinels intact, because
nulling them would destroy the information we are trying to surface.

What is added is a second verdict alongside the first: what the column *would*
be if the sentinels were removed. The profiler reads that verdict instead of
deciding a column's type for itself, so there is still one place that decides.

Four new public functions and four new record fields:

    failed_values(s, converted)     which values a coercion turned into nulls
    discover_specials(s)            sentinel candidates, from a sample
    effective_dtype(s, specials)    the dtype once those are removed
    column_record(name, before, after, ...)   the record row, extended

    dtype_effective   dtype ignoring the sentinels; == dtype_after when none
    specials          the values excluded to reach it
    specials_count    rows those values account for
    n_unique          distinct non-null values, which bin_column can reuse
"""

import pandas as pd
from datkit.cleaning import clean_dates, clean_numbers, missing_count, remove_whitespace, to_null

# A column with more distinct unconvertible values than this is not a numeric
# column with sentinels in it — it is text with numbers in it, and no list of
# exclusions will rescue it.
MAX_SPECIALS = 10

# The absolute cap is not enough on its own: a six-row column of names has six
# unconvertible values, which is under the cap, and every one of them would be
# proposed as a sentinel. Sentinels must also be a minority of the column's
# distinct values — that test is scale-free and does not care how often each
# sentinel occurs.
MAX_SPECIAL_RATIO = 0.5

# Discovery only needs to see each sentinel once, so it runs on a sample.
# Hashing 90M values to find "N/A" is the cost this whole toolkit avoids.
DISCOVERY_SAMPLE = 100_000


def failed_values(s: pd.Series, converted: pd.Series, limit: int = MAX_SPECIALS + 1) -> list:
    """The distinct original values a coercion turned into nulls.

    This is the evidence `clean_numbers` and `clean_dates` currently throw away.
    When they refuse a column, the reason is knowable at that moment: the rows
    where the converted series is null but the original was not.

    Parameters
    ----------
    s : pandas.Series
        The column as it arrived.
    converted : pandas.Series
        The same column after a coercion with errors="coerce".
    limit : int, default MAX_SPECIALS + 1
        Stop after this many distinct values. One more than the ceiling, so a
        caller can tell "exactly at the limit" from "more than the limit"
        without collecting an unbounded list.

    Returns
    -------
    list
        Distinct offending values, at most `limit` of them.
    """
    # Null after, not null before. Note this uses isna() rather than
    # missing_count(): we need the mask, not the count, and the Arrow NaN
    # problem does not arise because the "before" side is text.
    lost = converted.isna() & s.notna()
    if not lost.any():
        return []

    return list(pd.unique(s[lost])[:limit])


def discover_specials(
    s: pd.Series,
    sample_n: int | None = DISCOVERY_SAMPLE,
    seed: int = 1,
    max_ratio: float = MAX_SPECIAL_RATIO,
) -> list:
    """Sentinel candidates in a text column: the few values blocking conversion.

    A handful of distinct values that will not coerce are sentinels. Values
    scattered across many distinct forms are not — that column really is text,
    and it comes back as no candidates rather than a long list of noise.

    Two tests, both needed. The absolute cap (MAX_SPECIALS) keeps the list
    short. The ratio test keeps a short *column* from looking like it is all
    sentinels: six names are six unconvertible values, under the cap, and
    without the ratio every one would be proposed.

    Proposes; never acts. The caller decides whether these are sentinels, and
    nothing downstream is nulled or altered on the strength of it.

    Parameters
    ----------
    s : pandas.Series
        Column as it arrived. A non-text column has nothing to discover.
    sample_n : int or None, default DISCOVERY_SAMPLE
        Rows to look at. None looks at everything. A sample can miss a rare
        sentinel; it cannot invent one.
    seed : int, default 1
        Sample seed, so a rerun proposes the same candidates.
    max_ratio : float, default MAX_SPECIAL_RATIO
        Most of the column's distinct values that may be candidates. Above
        this the column is text, not a rescuable column.

    Returns
    -------
    list
        Candidate values, at most MAX_SPECIALS. Empty when the column needs no
        rescuing, or is past rescuing.
    """
    if not pd.api.types.is_string_dtype(s.dtype):
        return []

    sample = s
    if sample_n is not None and len(s) > sample_n:
        sample = s.sample(n=sample_n, random_state=seed)

    # Same preparation clean_column does before the coercions, so the
    # candidates are the values those coercions would actually meet.
    prepared = to_null(remove_whitespace(sample))

    distinct = int(prepared.nunique())
    if distinct == 0:
        return []

    # Explicit loop: try each coercion in the order clean_column applies them
    # and take the first that leaves a short, nameable list of failures.
    #
    # The raw pandas coercions, not clean_numbers/clean_dates: those refuse and
    # hand back the original, which is exactly the information being recovered
    # here. format="mixed" is looser than clean_dates' guessed format, but this
    # only has to identify which values fail, not parse them correctly.
    for convert in (_to_numeric, _to_datetime):
        converted = convert(prepared)

        candidates = failed_values(prepared, converted)
        if not candidates:
            # Nothing was blocking this coercion, so nothing to propose.
            return []

        failed_distinct = int(prepared[converted.isna() & prepared.notna()].nunique())
        if len(candidates) <= MAX_SPECIALS and failed_distinct / distinct <= max_ratio:
            return candidates

    return []


def _to_numeric(s: pd.Series) -> pd.Series:
    """Coerce to number, nulling whatever will not convert."""
    return pd.to_numeric(s, errors="coerce")


def _to_datetime(s: pd.Series) -> pd.Series:
    """Coerce to date, nulling whatever will not convert."""
    return pd.to_datetime(s, errors="coerce", format="mixed")


def effective_dtype(s: pd.Series, specials: list) -> str:
    """The dtype this column would take if `specials` were not in it.

    The column itself is untouched. This is a claim recorded about the column,
    and it is only reproducible alongside the `specials` that produced it —
    record both or neither.

    Parameters
    ----------
    s : pandas.Series
        Column as it arrived.
    specials : list
        Values to disregard. An empty list makes this the ordinary dtype.

    Returns
    -------
    str
        The dtype name, equal to the cleaned dtype when nothing was excluded.
    """
    if not len(specials):
        return str(clean_column_dtype(s))

    # Drop, do not null: a null would be counted as a failed coercion by the
    # guards inside clean_numbers and clean_dates, and we would be back where
    # we started.
    kept = s[~s.isin(specials)]
    if kept.empty:
        return str(s.dtype)

    return str(clean_column_dtype(kept))


def clean_column_dtype(s: pd.Series):
    """The dtype clean_column would produce, without keeping the cleaned column.

    Separate from clean_column so the probe cannot accidentally become the
    thing that cleans. It runs the same steps and reports only the dtype.
    """
    probe = clean_dates(clean_numbers(to_null(remove_whitespace(s))))
    return probe.dtype


def column_record(
    name: str,
    before: pd.Series,
    after: pd.Series,
    specials: list,
    dtype_effective: str,
    n_unique: int,
) -> dict:
    """One record row, with the four new fields alongside the existing ones.

    Split out of clean_dataframe so the record's shape can be tested without
    cleaning a frame to get one.
    """
    dtype_before = str(before.dtype)
    dtype_after = str(after.dtype)

    # index=False measures the values only, so columns are comparable.
    bytes_before = before.memory_usage(deep=True, index=False)
    bytes_after = after.memory_usage(deep=True, index=False)

    nulls_before = missing_count(before)
    nulls_after = missing_count(after)

    # Rows the sentinels account for. isin on an empty list is all False, so
    # this is 0 when nothing was found, with no special case.
    specials_count = int(before.isin(specials).sum())

    return {
        "column": name,
        "dtype_before": dtype_before,
        "dtype_after": dtype_after,
        "converted": dtype_before != dtype_after,
        # What the column would be without the sentinels. Equal to dtype_after
        # when there are none, so a reader can compare the two columns to see
        # at a glance which columns are being held back by a handful of values.
        "dtype_effective": dtype_effective,
        # Recorded together: the dtype alone is not reproducible, because the
        # next reader cannot tell what had to be excluded to reach it.
        "specials": specials,
        "specials_count": specials_count,
        # Carried so bin_column can skip its own cardinality probe.
        "n_unique": n_unique,
        "bytes_before": bytes_before,
        "bytes_after": bytes_after,
        "bytes_saved": bytes_before - bytes_after,
        "mb_before": round(bytes_before / 1024**2, 2),
        "mb_after": round(bytes_after / 1024**2, 2),
        "mb_saved": round((bytes_before - bytes_after) / 1024**2, 2),
        "nulls_before": nulls_before,
        "nulls_after": nulls_after,
        "nulls_created": nulls_after - nulls_before,
    }


def clean_dataframe(
    df: pd.DataFrame,
    inplace: bool = False,
    specials: dict | None = None,
    discover: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean every column, returning the frame and a record of what changed.

    Behaviour is unchanged: columns are cleaned exactly as before, and no
    sentinel is ever nulled or removed from the data. The record gains four
    fields describing what the column would be without its sentinels.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to clean.
    inplace : bool, default False
        If True, modify the caller's frame. If False, work on a copy — safer,
        but doubles peak memory.
    specials : dict or None, default None
        Sentinels the caller already knows, keyed by column name. Takes
        precedence over discovery for those columns, so a second run with a
        corrected list does not get argued with.
    discover : bool, default True
        Propose sentinels for text columns the coercions refused. Costs one
        sampled pass per unconverted text column. Set False to skip it.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The cleaned frame, and one record row per column.
    """
    if not inplace:
        df = df.copy()

    if specials is None:
        specials = {}

    records = []

    for col in df.columns:
        before = df[col]
        after = clean_column(before)

        # The caller's list wins. Discovery only runs where there is something
        # to rescue: a column that already converted needs no sentinels.
        if col in specials:
            found = list(specials[col])
        elif discover and after.dtype == before.dtype:
            found = discover_specials(before)
        else:
            found = []

        records.append(
            column_record(
                name=col,
                before=before,
                after=after,
                specials=found,
                dtype_effective=effective_dtype(before, found),
                # Cheap relative to the cleaning pass, and it saves bin_column
                # a hash of its own later.
                n_unique=int(before.nunique()),
            )
        )

        # Assign back one column at a time so the old one can be freed before
        # the next is built.
        df[col] = after

    return df, pd.DataFrame(records)


def clean_column(s: pd.Series) -> pd.Series:
    """Unchanged — repeated here only so this draft runs standalone."""
    s = remove_whitespace(s)
    s = to_null(s)
    s = clean_numbers(s)
    s = clean_dates(s)
    return s
