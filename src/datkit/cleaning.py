"""Helper functions for cleaning columns in a DataFrame."""

import re

import numpy as np
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


def missing_mask(s: pd.Series) -> pd.Series:
    """Boolean mask of values that are null or NaN, on any backend.

    The mask behind missing_count, split out because a caller needing to know
    WHICH rows are missing must not reach for `s.isna()` instead. Under Arrow
    a failed coercion produces NaN, which is a valid double distinct from
    null, so `isna()` reports False for it and the failed rows vanish.

    Parameters
    ----------
    s : pandas.Series
        Any dtype, any backend.

    Returns
    -------
    pandas.Series
        Boolean, True where the value is absent by either definition.
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
    return missing.fillna(True).astype(bool)


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
    return int(missing_mask(s).sum())


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


# Symbols that decorate a number without changing it. Shared so the
# convertibility check measures the same string clean_numbers would coerce —
# otherwise "1,234" is reported as unconvertible when it plainly is not.
NUMERIC_SYMBOLS = ("$", "R", "%", ",")


def strip_numeric_symbols(s: pd.Series) -> pd.Series:
    """Remove currency and grouping symbols from a text column.

    Parameters
    ----------
    s : pandas.Series
        Text column. A non-text column is returned untouched.

    Returns
    -------
    pandas.Series
        The column with each symbol in NUMERIC_SYMBOLS removed.
    """
    if not pd.api.types.is_string_dtype(s.dtype):
        return s

    # Explicit loop: each pass rewrites the column, so this is real work.
    r = s
    for char in NUMERIC_SYMBOLS:
        r = r.str.replace(char, "", regex=False)
    return r


def clean_numbers(s: pd.Series) -> pd.Series:
    """Clean numeric columns in the series."""
    # ordering is important:
    # first check if everything left over will be a number,
    # then start removing characters,
    # then convert to numeric
    if pd.api.types.is_string_dtype(s.dtype):
        r = strip_numeric_symbols(s)
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
    # Null after, not null before.
    #
    # missing_mask, not isna(). The Arrow NaN problem is on the AFTER side:
    # a failed coercion of an Arrow column produces NaN, which isna() reports
    # as present, so every failure would be invisible and the column would
    # look clean.
    lost = missing_mask(converted) & ~missing_mask(s)
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

        failed_distinct = int(prepared[missing_mask(converted) & ~missing_mask(prepared)].nunique())
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


# Above this share of failures the column is not contaminated — it is text.
# Reporting every title in primaryTitle as "unconvertible" is true and
# useless: it buries the column that has 32 bad rows among the columns that
# were never numbers. Shared by the record and by find_unconvertible so the
# two views cannot disagree about the same column.
CONTAMINATION_MAX_PCT = 50.0

# How many distinct offending values to keep as evidence. A bounded list, so
# the output stays the same size whether the column has 12 bad rows or 12M.
MAX_UNCONVERTIBLE_VALUES = 20

# Rows to look at when deciding which coercion a column is closest to. That
# is a majority question — is this mostly numbers or mostly dates — and a
# sample settles it. Counting the failures is a rare-event question and is
# not sampled; see convertibility.
TARGET_SAMPLE = 10_000

# A sample that turns up almost nothing to convert cannot choose a target, so
# fall back to the whole column rather than decide on a handful of values.
MIN_TARGET_SAMPLE = 100

# Rows drawn per row wanted. Nulls and placeholders do not help the decision,
# so a mostly-empty column needs more raw rows to yield enough real values.
OVERSAMPLE = 5

# Coercions the convertibility check can measure against, in the order
# clean_column would try them.
TARGETS = {"numeric": _to_numeric, "datetime": _to_datetime}


def convertibility(s: pd.Series, target: str | None = None, max_pct: float = CONTAMINATION_MAX_PCT) -> dict:
    """Measure how close a text column is to being wholly convertible.

    The question discover_specials cannot answer. That one asks whether a
    SHORT LIST of values is blocking a conversion, and rejects a column with
    too many distinct offenders. This asks whether a SMALL SHARE OF ROWS is
    blocking it, however many distinct forms they take — the shape that means
    something upstream is wrong rather than that the column has sentinels.

    IMDb's runtimeMinutes is the case in point: 32 bad rows in 12.7M, but 13
    distinct values. Over the sentinel cap, far under any row threshold.

    Deliberately unsampled. Contamination this rare is invisible to a sample —
    32 rows in 12.7M against a 100k sample is a quarter of an expected hit —
    so this is one full pass over the column. That is the cost of finding it
    at all.

    Reports; never judges. No threshold is applied and nothing is flagged:
    the counts are returned and the caller decides what share is tolerable.

    Parameters
    ----------
    s : pandas.Series
        Column as it arrived. A non-text column has nothing to measure.
    target : {"numeric", "datetime"} or None, default None
        Which coercion to measure against. None tries each in turn and keeps
        the best fit, stopping early when one converts everything — nothing
        can beat no failures, so the second pass is not run.

    Returns
    -------
    dict
        target, n_rows, n_null, n_placeholder, n_unconvertible, n_infinite,
        pct_unconvertible, and values — the distinct offenders with counts,
        at most MAX_UNCONVERTIBLE_VALUES of them.

    Raises
    ------
    KeyError
        If target is not one of TARGETS.
    """
    # Choosing the target is a majority question, so it runs on a sample and
    # costs almost nothing. Only the count below needs the whole column.
    if target is None:
        target = best_target(s)

    convert = TARGETS[target]

    empty = {
        "target": target,
        "verdict": "clean",
        "n_rows": len(s),
        "n_null": missing_count(s),
        "n_placeholder": 0,
        "n_unconvertible": 0,
        "n_infinite": 0,
        "pct_unconvertible": 0.0,
        "values": {},
    }
    if not _is_text(s):
        return empty

    stripped = remove_whitespace(s)
    nulled = to_null(stripped)

    # Placeholders are counted separately rather than lumped in with the
    # failures. "30% of this column is \N" is a different problem from "30%
    # of it will not parse", and by the time the coercion runs the
    # placeholders are already null and would be invisible.
    n_null = missing_count(s)
    n_placeholder = missing_count(nulled) - n_null

    # Symbols come off before the coercion, exactly as clean_numbers does it,
    # so "$1,234" is not reported as unconvertible.
    prepared = strip_numeric_symbols(nulled) if target == "numeric" else nulled
    converted = convert(prepared)

    # Failed: null after, not null before. The placeholders are already null
    # on both sides, so they do not land here.
    lost = missing_mask(converted) & ~missing_mask(nulled)
    n_unconvertible = int(lost.sum())

    # Infinities are the opposite failure: 'inf' converts cleanly and then
    # poisons every mean downstream. Counted so it cannot pass unnoticed.
    n_infinite = 0
    if pd.api.types.is_numeric_dtype(converted.dtype):
        n_infinite = int(np.isinf(converted.to_numpy(dtype="float64", na_value=np.nan)).sum())

    # Denominator is rows that had something to convert: total less the nulls
    # and placeholders. A column that is 90% empty should not look 90% clean.
    convertible_rows = len(s) - n_null - n_placeholder
    pct = round(n_unconvertible / convertible_rows * 100, 4) if convertible_rows else 0.0

    # A column that fails almost everywhere was never this kind of column.
    # Saying so, and reporting nothing, beats listing twenty titles as
    # "offending values" in a report meant to surface the odd bad row.
    if pct > max_pct:
        return {
            "target": target,
            "verdict": "text",
            "n_rows": len(s),
            "n_null": n_null,
            "n_placeholder": n_placeholder,
            "n_unconvertible": 0,
            "n_infinite": n_infinite,
            "pct_unconvertible": 0.0,
            "values": {},
        }

    return {
        "target": target,
        "verdict": "contaminated" if n_unconvertible else "clean",
        "n_rows": len(s),
        "n_null": n_null,
        "n_placeholder": n_placeholder,
        "n_unconvertible": n_unconvertible,
        "n_infinite": n_infinite,
        "pct_unconvertible": pct,
        # value_counts on the offenders only, so this is bounded by how many
        # distinct bad values there are, not by the size of the column.
        "values": nulled[lost].value_counts().head(MAX_UNCONVERTIBLE_VALUES).to_dict(),
    }


def best_target(s: pd.Series, sample_n: int | None = TARGET_SAMPLE) -> str:
    """Which coercion this column is closest to satisfying.

    Decided on a sample, deliberately. This is a question about the bulk of
    the column — is it mostly numbers or mostly dates — and a sample answers
    that with room to spare. It is the opposite of the failure count in
    convertibility, which is about a handful of rare values and cannot be
    sampled at all.

    Doing it this way is also what keeps a merely-numeric column off the date
    coercion, which parses value by value and is the expensive one.

    Ties go to the first target tried, so a column that converts perfectly
    both ways is reported as numeric.

    Parameters
    ----------
    s : pandas.Series
        Column as it arrived.
    sample_n : int or None, default TARGET_SAMPLE
        Rows to decide on. None uses the whole column. A sample holding fewer
        than MIN_TARGET_SAMPLE convertible values falls back to the whole
        column, since a column that is nearly all nulls gives a sample almost
        nothing to judge by.

    Returns
    -------
    str
        A key of TARGETS.
    """
    if not _is_text(s):
        return "numeric"

    # Sample the raw column FIRST, then prepare only those rows. Preparing
    # the whole column before sampling costs a full pass of string operations
    # and throws almost all of it away — which defeats the point of sampling.
    candidates = s
    if sample_n is not None and len(s) > sample_n:
        # Oversample, because nulls and placeholders do not count towards the
        # decision and a column can be mostly empty.
        candidates = s.sample(n=min(len(s), sample_n * OVERSAMPLE), random_state=1)

    candidates = to_null(remove_whitespace(candidates)).dropna()

    # Too little survived to judge by, so look at everything rather than
    # decide a whole column on a handful of values.
    if len(candidates) < MIN_TARGET_SAMPLE and sample_n is not None and len(s) > sample_n:
        candidates = to_null(remove_whitespace(s)).dropna()

    if candidates.empty:
        return "numeric"

    if sample_n is not None and len(candidates) > sample_n:
        candidates = candidates.iloc[:sample_n]

    fewest = None
    chosen = "numeric"

    # Explicit loop: each pass coerces the sample, which is real work.
    for name in TARGETS:
        probe = strip_numeric_symbols(candidates) if name == "numeric" else candidates
        failures = int(TARGETS[name](probe).isna().sum())
        if fewest is None or failures < fewest:
            fewest = failures
            chosen = name
        if fewest == 0:
            break  # nothing beats a clean conversion

    return chosen


def find_unconvertible(
    df: pd.DataFrame,
    columns: list | None = None,
    target: str | None = None,
    max_rows: int = 1000,
    max_pct: float = CONTAMINATION_MAX_PCT,
) -> pd.DataFrame:
    """The individual rows a coercion would lose, so they can be looked at.

    The record says a column has 32 bad rows. This says which rows, and what
    is in them — the step before deciding whether they are sentinels, a
    parsing bug, or genuine text.

    Output is capped at max_rows regardless of input size, so a column that is
    half rubbish returns the same size answer as one with a handful of typos.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to inspect. Non-text columns are skipped.
    columns : list or None, default None
        Columns to check. None checks every column.
    target : {"numeric", "datetime"} or None, default None
        Coercion to measure against. None picks each column's best fit.
    max_rows : int, default 1000
        Ceiling on returned rows, taken in frame order.
    max_pct : float, default CONTAMINATION_MAX_PCT
        Skip a column when more than this share of its convertible rows
        fail. A column where most values will not coerce is simply text, and
        listing every row of it buries the columns that are contaminated.
        Tighten it to 5 to see only the near-clean ones.

    Returns
    -------
    pandas.DataFrame
        Columns: index, column, value. Empty when nothing failed.
    """
    if columns is None:
        columns = list(df.columns)

    found = []
    remaining = max_rows

    for col in columns:
        if remaining <= 0:
            break

        s = df[col]
        if not _is_text(s):
            continue

        # One measurement decides the target and the share in the same pass.
        measured = convertibility(s, target=target, max_pct=max_pct)
        column_target = measured["target"]
        if measured["verdict"] != "contaminated":
            continue

        nulled = to_null(remove_whitespace(s))
        prepared = strip_numeric_symbols(nulled) if column_target == "numeric" else nulled
        lost = missing_mask(TARGETS[column_target](prepared)) & ~missing_mask(nulled)

        # head() before building the frame, so a ruined column cannot
        # materialise millions of rows here.
        offenders = s[lost].head(remaining)
        found.append(
            pd.DataFrame(
                {
                    "index": offenders.index,
                    "column": col,
                    "value": offenders.to_numpy(),
                }
            )
        )
        remaining -= len(offenders)

    if not found:
        return pd.DataFrame(columns=["index", "column", "value"])

    return pd.concat(found, ignore_index=True)


# Above this share of distinct values a category costs more than it saves.
# Measured on 2M rows of 20-character strings: at a 0.025 ratio the column
# shrinks 12x, at 0.20 only 3.3x, and at 0.50 just 1.4x — by which point the
# categories array is most of the column and the awkwardness is not repaid.
CATEGORY_MAX_RATIO = 0.05

# An absolute ceiling as well as a ratio. Past roughly this many categories
# the codes are int32 anyway, so the saving stops improving, while the
# categories array keeps growing and every operation carries it around.
CATEGORY_MAX_UNIQUE = 50_000

# Rows to look at when rejecting a column as too varied to be categorical.
# A sample can only ever UNDER-count distinct values, so a sample that looks
# varied proves the column is varied — a sound rejection. A sample that looks
# repetitive proves nothing, and those columns get the full pass.
CATEGORY_SAMPLE = 10_000

# Distinct share within the sample above which the column is rejected without
# a full pass. Deliberately loose: it only has to catch the obviously varied.
CATEGORY_SAMPLE_MAX_RATIO = 0.5


def category_profile(
    s: pd.Series,
    sample_n: int | None = CATEGORY_SAMPLE,
    max_ratio: float = CATEGORY_MAX_RATIO,
    max_unique: int = CATEGORY_MAX_UNIQUE,
) -> dict:
    """Whether a text column is worth holding as a category, and what it saves.

    Two stages, for the same reason best_target has two. Deciding a column is
    too varied is a question about the bulk of it, so a sample settles it and
    the expensive full count is never paid. Getting the categories themselves
    is not: a value occurring twice in 90M rows is invisible to any sample and
    must still become a category, or converting the column would silently drop
    it. So the sample rejects, and whatever survives is counted in full.

    Proposes; never acts. Nothing is converted here — the caller decides,
    because a category changes behaviour well beyond memory: concatenating two
    categoricals with different categories reverts to object, assigning an
    unseen value raises, and groupby(observed=False) emits a row per category
    whether or not it occurs.

    Parameters
    ----------
    s : pandas.Series
        Column as it arrived. A non-text column is not a candidate.
    sample_n : int or None, default CATEGORY_SAMPLE
        Rows for the rejection filter. None skips it and counts in full.
    max_ratio : float, default CATEGORY_MAX_RATIO
        Largest distinct-to-rows share still worth converting.
    max_unique : int, default CATEGORY_MAX_UNIQUE
        Largest number of categories still worth converting.

    Returns
    -------
    dict
        as_category, n_unique, ratio, bytes_before, bytes_after, saving, and
        rejected_by — which test failed, or None. n_unique is -1 when the
        sample rejected the column, since no full count was taken.
    """
    unknown = {
        "as_category": False,
        "n_unique": -1,
        "ratio": None,
        "bytes_before": int(s.memory_usage(deep=True, index=False)),
        "bytes_after": None,
        "saving": None,
        "rejected_by": None,
    }

    if not _is_text(s):
        return {**unknown, "rejected_by": "not text"}

    non_null = s.dropna()
    if non_null.empty:
        return {**unknown, "rejected_by": "empty"}

    # Stage one: reject the obviously varied without counting everything.
    if sample_n is not None and len(non_null) > sample_n:
        sampled = non_null.sample(n=sample_n, random_state=1)
        if sampled.nunique() / len(sampled) > CATEGORY_SAMPLE_MAX_RATIO:
            return {**unknown, "rejected_by": "sample"}

    # Stage two: the exact count, over everything. A category seen twice in
    # 90M rows is still a category, and no sample would have found it.
    n_unique = int(non_null.nunique())
    ratio = n_unique / len(non_null)

    rejected_by = None
    if n_unique > max_unique:
        rejected_by = "max_unique"
    elif ratio > max_ratio:
        rejected_by = "max_ratio"

    bytes_before = int(s.memory_usage(deep=True, index=False))
    profile = {
        "as_category": rejected_by is None,
        "n_unique": n_unique,
        "ratio": round(ratio, 6),
        "bytes_before": bytes_before,
        "bytes_after": None,
        "saving": None,
        "rejected_by": rejected_by,
    }

    # Only measure the converted size for columns actually worth converting;
    # building a category to find out it was a bad idea is the cost being
    # avoided.
    if rejected_by is None:
        bytes_after = int(s.astype("category").memory_usage(deep=True, index=False))
        profile["bytes_after"] = bytes_after
        profile["saving"] = round(bytes_before / bytes_after, 1) if bytes_after else None

    return profile


def to_category(s: pd.Series, **kwargs) -> pd.Series:
    """Convert a column to a category when category_profile says it is worth it.

    Parameters
    ----------
    s : pandas.Series
        Column to consider.
    **kwargs
        Passed to category_profile: sample_n, max_ratio, max_unique.

    Returns
    -------
    pandas.Series
        The column as a category, or unchanged when it is not a candidate.
    """
    if not category_profile(s, **kwargs)["as_category"]:
        return s

    # astype reads the whole column, so rare categories are included by
    # construction — this is the full pass the sample could not replace.
    return s.astype("category")


def column_record(
    name: str,
    before: pd.Series,
    after: pd.Series,
    specials: list,
    dtype_effective: str,
    n_unique: int,
    convertible: dict | None = None,
    category: dict | None = None,
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

    # A measurement, not a verdict: no threshold is applied here. The caller
    # sorts or filters on pct_unconvertible to decide what counts as a
    # problem, because 5% means something different at 20 rows and at 12M.
    if convertible is None:
        convertible = convertibility(before)

    # Also a measurement, not an action. clean_dataframe does not convert on
    # the strength of this: a category changes behaviour beyond memory, so
    # the caller opts in.
    if category is None:
        category = category_profile(before)

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
        # Rows a coercion would lose that are neither null nor placeholder —
        # the "funnies" worth looking at before anything is nulled.
        # clean / contaminated / text — filter the report on this rather than
        # eyeballing which 100% figures are meaningful.
        "unconvertible_verdict": convertible["verdict"],
        "n_unconvertible": convertible["n_unconvertible"],
        "pct_unconvertible": convertible["pct_unconvertible"],
        # Bounded evidence: the distinct offenders with their row counts.
        "unconvertible_values": convertible["values"],
        # Counted apart from the failures. By coercion time these are already
        # null, so without this they would be invisible.
        "n_placeholder": convertible["n_placeholder"],
        # 'inf' coerces cleanly and then poisons every mean downstream.
        "n_infinite": convertible["n_infinite"],
        # Whether holding this column as a category is worth it, and by how
        # much. -1 categories means the sample rejected it without counting.
        "as_category": category["as_category"],
        "n_categories": category["n_unique"],
        "category_ratio": category["ratio"],
        "category_saving": category["saving"],
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
    categorize: bool = False,
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
    categorize : bool, default False
        Convert columns the category profile approves. Off by default: the
        record reports what each column would save, and converting changes
        behaviour beyond memory, so it is opted into rather than out of.

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
                # One call: it picks the target and returns that pass's counts.
                convertible=convertibility(before),
                category=category_profile(before),
                # Cheap relative to the cleaning pass, and it saves bin_column
                # a hash of its own later.
                n_unique=int(before.nunique()),
            )
        )

        # Convert only when asked, and only on the verdict already computed,
        # so the column is not profiled twice.
        if categorize and records[-1]["as_category"]:
            after = after.astype("category")
            records[-1]["dtype_after"] = str(after.dtype)
            records[-1]["converted"] = records[-1]["dtype_before"] != records[-1]["dtype_after"]

        # Assign back one column at a time so the old one can be freed before
        # the next is built.
        df[col] = after

    return df, pd.DataFrame(records)
