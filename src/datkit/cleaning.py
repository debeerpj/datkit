"""Helper functions for cleaning columns in a DataFrame."""

import pandas as pd

"""Still need to add special treatment for category type fields -
update the dictionary of categories instead of the values themselves"""


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


def clean_dates(s: pd.Series) -> pd.Series:
    """Clean date columns in the series."""
    # always cater for US and non-US date formats by trying both dayfirst True and False

    if pd.api.types.is_string_dtype(s.dtype):
        # take a sample to test if the column can be converted to datetime
        non_null = s.dropna()
        if len(non_null) > 10:
            dt = non_null.sample(n=10, random_state=1)
        else:
            dt = non_null

        sample_dt_dayftrue = pd.to_datetime(dt, dayfirst=True, errors="coerce")
        sample_dt_dayffalse = pd.to_datetime(dt, dayfirst=False, errors="coerce")

        # if any of the sample items cannot be converted to datetime, return the original series
        if (sample_dt_dayftrue.isna() & sample_dt_dayffalse.isna()).any():
            return s

        dt_dayftrue = pd.to_datetime(s, dayfirst=True, errors="coerce")
        dt_dayffalse = pd.to_datetime(s, dayfirst=False, errors="coerce")

        # if there are more nulls in the both cleaned series than the original,
        # then it might not be a date column
        if missing_count(s) == missing_count(dt_dayftrue):
            return dt_dayftrue
        elif missing_count(s) == missing_count(dt_dayffalse):
            return dt_dayffalse  # return the original series if cleaning results in more nulls
        else:
            return s

    else:
        return s


def clean_column(s: pd.Series) -> pd.Series:
    """Clean a column in the DataFrame."""
    s = remove_whitespace(s)
    s = to_null(s)
    s = clean_numbers(s)
    # s = uppercase(s)
    # removed from default excution because it uses a lot of memory
    s = clean_dates(s)
    return s


def clean_dataframe(df: pd.DataFrame, inplace: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean every column, returning the frame and a record of what changed.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to clean.
    inplace : bool, default False
        If True, modify the caller's frame. If False, work on a copy — safer,
        but doubles peak memory.

    Returns
    -------
    tuple of (pandas.DataFrame, pandas.DataFrame)
        The cleaned frame, and one record row per column.
    """
    if not inplace:
        df = df.copy()

    records = []

    for col in df.columns:
        before = df[col]

        # Capture these before cleaning — the old column is about to be replaced.
        # index=False measures the values only, so columns are comparable.
        dtype_before = str(before.dtype)
        bytes_before = before.memory_usage(deep=True, index=False)
        nulls_before = missing_count(before)

        after = clean_column(before)

        dtype_after = str(after.dtype)
        bytes_after = after.memory_usage(deep=True, index=False)
        nulls_after = missing_count(after)

        records.append(
            {
                "column": col,
                "dtype_before": dtype_before,
                "dtype_after": dtype_after,
                "converted": dtype_before != dtype_after,
                "bytes_before": bytes_before,
                "bytes_after": bytes_after,
                "bytes_saved": bytes_before - bytes_after,
                "mb_before": round(bytes_before / 1024**2, 2),
                "mb_after": round(bytes_after / 1024**2, 2),
                "mb_saved": round((bytes_before - bytes_after) / 1024**2, 2),
                "nulls_before": nulls_before,
                "nulls_after": nulls_after,
                # Any growth here means values were lost to coercion.
                "nulls_created": nulls_after - nulls_before,
            }
        )

        # Assign back one column at a time so the old one can be freed
        # before the next is built.
        df[col] = after

    return df, pd.DataFrame(records)
