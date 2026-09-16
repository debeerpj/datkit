"""Helper functions for cleaning columns in a DataFrame."""

import re

import pandas as pd

"""Still need to add special treatment for category type fields -
update the dictionary of categories instead of the values themselves"""

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


def sample_records(df: pd.DataFrame, n=10000, seed=1) -> pd.DataFrame:
    """Return a sample of records from the DataFrame."""
    if len(df) > n:
        return df.sample(n=n, random_state=seed)
    else:
        return df


def missing_count(s: pd.Series) -> int:
    """Count values that are null or NaN.

    Arrow and numpy disagree about NaN. Under numpy, NaN *is* missing — both
    `isna()` and `count()` treat it that way. Under Arrow, NaN is a valid
    double distinct from null, so `isna()` returns False for it and `count()`
    counts it as present.

    That difference silently breaks any guard written as
    `len(s) - s.count()`: a failed numeric coercion produces NaN, Arrow
    reports no new nulls, and the guard lets the ruined column through.

    Parameters
    ----------
    s : pandas.Series
        Any dtype, any backend.

    Returns
    -------
    int
        Count of values that are absent by either definition.
    """
    # Catches real nulls: None, NaN under numpy, <NA> under Arrow.
    missing = s.isna()

    if pd.api.types.is_float_dtype(s.dtype):
        # Only float columns can hold NaN, so skip this for text and integers.
        #
        # NaN is the one value not equal to itself (IEEE 754), so `s != s` is
        # True exactly where NaN sits — including the NaNs Arrow's isna()
        # reported as present.
        #
        # `|` is element-wise OR on the two boolean Series. Not `or`, which
        # needs a single truth value and raises on a Series.
        missing = missing | (s != s)

    # At a genuine null, `s != s` evaluates to <NA> rather than True, which
    # would propagate into sum(). A null is missing, so fill it with True.
    return int(missing.fillna(True).sum())


def remove_whitespace(s: pd.Series) -> pd.Series:
    """Remove leading and trailing whitespace from string columns in the series."""
    # use is_string_dtype to also include apache arrow data types
    if pd.api.types.is_string_dtype(s.dtype):
        s = s.str.strip()
    return s


def to_null(s: pd.Series, tokens=(r"\N", "NA", "N/A", "na", "n/a", "null", "NULL", "", "-")) -> pd.Series:
    """Convert common placeholcer values with proper null values (NaN) in the series."""
    if pd.api.types.is_string_dtype(s.dtype):
        # mask replaces the full values based on a condition,
        # does not replace substrings like replace and processes faster
        s = s.mask(s.isin(tokens))
    return s


def clean_numbers(s: pd.Series) -> pd.Series:
    """Clean numeric columns in the series."""
    remove = ["$", "R", "%", ","]

    # ordering is important:
    # first check if everything left over will be a number,
    # then start removing characters,
    # then convert to numeric
    if pd.api.types.is_string_dtype(s.dtype):
        r = s
        for char in remove:
            r = r.str.replace(char, "", regex=False)
        r = pd.to_numeric(r, errors="coerce")  # convert to numeric, setting errors to NaN

        # if there are more nulls in the cleaned series than the original,
        # then it might not be a numeric column
        if missing_count(s) < missing_count(r):
            return s  # return the original series if cleaning results in more nulls
        else:
            return r
    else:
        return s  # return the original series if it's not a string type


def uppercase(s: pd.Series) -> pd.Series:
    """Convert string columns in the series to uppercase."""
    if pd.api.types.is_string_dtype(s.dtype):
        return s.str.upper()
    else:
        return s


def _guess_format(v: str, dayfirst: bool) -> str | None:
    """Guess a strftime format, retrying with month names title-cased.

    The guesser matches calendar.month_abbr literally ('Mar'), so 'mar' and
    'MAR' return None even though to_datetime parses them happily.
    """
    fmt = pd.tseries.api.guess_datetime_format(v, dayfirst=dayfirst)
    if fmt is None:
        # Title-case alphabetic runs of 3+ letters only, leaving AM/PM alone.
        fmt = pd.tseries.api.guess_datetime_format(
            re.sub(r"[A-Za-z]{3,}", lambda m: m.group(0).title(), v), dayfirst=dayfirst
        )
    return fmt


def clean_dates(s: pd.Series) -> pd.Series:
    """Clean date columns in the series."""
    # always cater for US and non-US date formats by trying both dayfirst True and False

    if pd.api.types.is_string_dtype(s.dtype):
        non_null = (
            s.dropna()
        )  # dropna() retains original index, so if first item was null non_null[0] will give an error

        # if all null column return original
        if len(non_null) == 0:
            return s

        # test if first value can be converted to datetime, if not return the original series
        dateformat_dayfirsttrue = _guess_format(non_null.iloc[0], dayfirst=True)
        dateformat_dayfirstfalse = _guess_format(non_null.iloc[0], dayfirst=False)
        if dateformat_dayfirsttrue is None and dateformat_dayfirstfalse is None:
            return s

        # take a sample to determine if date format is consistent
        # and if dayfirst=True or dayfirst=False as a lot of dates will work with both and give incorrect results
        if len(non_null) > 100:
            dt = non_null.sample(n=100, random_state=1)
        else:
            dt = non_null

        dateformat_dayfirsttrue = dt.map(lambda v: _guess_format(v, dayfirst=True), na_action="ignore")
        dateformat_dayfirstfalse = dt.map(lambda v: _guess_format(v, dayfirst=False), na_action="ignore")

        # if some non-null values can't be converted by either then return the original
        # or if multiple potential formats are found then return the original

        nonnull_count_dayfirsttrue = len(dateformat_dayfirsttrue.dropna())
        nonnull_count_dayfirstfalse = len(dateformat_dayfirstfalse.dropna())
        unique_count_dayfirsttrue = len(dateformat_dayfirsttrue.dropna().unique())
        unique_count_dayfirstfalse = len(dateformat_dayfirstfalse.dropna().unique())

        if nonnull_count_dayfirsttrue < len(dt) and nonnull_count_dayfirstfalse < len(dt):
            return s
        elif (
            nonnull_count_dayfirsttrue == len(dt)
            and unique_count_dayfirsttrue == 1
            and dateformat_dayfirsttrue.dropna().unique()[0][0:2] == "%Y"
        ):
            # dayfirst=true break dates starting with year
            formatted_dates = pd.to_datetime(s, format=dateformat_dayfirstfalse.dropna().unique()[0], errors="coerce")
        elif nonnull_count_dayfirsttrue == len(dt) and unique_count_dayfirsttrue == 1:
            formatted_dates = pd.to_datetime(s, format=dateformat_dayfirsttrue.dropna().unique()[0], errors="coerce")
        elif nonnull_count_dayfirstfalse == len(dt) and unique_count_dayfirstfalse == 1:
            formatted_dates = pd.to_datetime(s, format=dateformat_dayfirstfalse.dropna().unique()[0], errors="coerce")
        else:
            return s

        # if there are more nulls in the cleaned series than the original,
        # then it might not be a date column
        if missing_count(s) == missing_count(formatted_dates):
            return formatted_dates
        else:
            return s

    else:
        return s


def _is_text(s: pd.Series) -> bool:
    """Whether the column actually holds text, judged on the values.

    The one guard the text pipeline is gated on, in one place because there are
    now three callers and they must not drift apart.

    is_string_dtype is True for *any* object column, and object is still the
    default for text on pandas 2.2.3 — the target version. That let an object
    column of ints reach .str (AttributeError), and an object column of mixed
    types lose its non-string values to coercion.

    infer_dtype inspects the values instead, so it tells object-of-strings apart
    from object-of-ints. It costs a scan, so callers invoke it once per column
    rather than once per step. Inside the gate the helpers' own is_string_dtype
    checks are sufficient, and are what makes clean_dates skip a column
    clean_numbers has already converted to numeric.

    Parameters
    ----------
    s : pandas.Series
        Any dtype, any backend.

    Returns
    -------
    bool
        True when the non-null values are strings.
    """
    return pd.api.types.infer_dtype(s, skipna=True) == "string"


def clean_column(s: pd.Series) -> pd.Series:
    """Clean a column in the DataFrame.

    Parameters
    ----------
    s : pandas.Series
        Column to clean. Columns that do not hold text are returned untouched.

    Returns
    -------
    pandas.Series
        The cleaned column, or the original if it does not hold text.
    """
    if not _is_text(s):
        return s

    s = remove_whitespace(s)
    s = to_null(s)
    s = clean_numbers(s)
    # s = uppercase(s)
    # removed from default excution because it uses a lot of memory
    s = clean_dates(s)
    return s


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
    if not _is_text(s):
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
    if not _is_text(s):
        return s.dtype

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
