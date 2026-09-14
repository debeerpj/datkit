"""Draft: top-N summary for text and categorical columns."""

import numpy as np
import pandas as pd

HIGH_CARDINALITY = "(high cardinality)"
OTHER = "(other)"


def bin_categorical(
    s: pd.Series,
    top_n: int = 20,
    high_card_ratio: float = 0.5,
    n_unique: int | None = None,
    sample_n: int = 100_000,
    seed: int = 1,
) -> pd.DataFrame:
    """Describe a text or categorical column as its top values, with the rest pooled.

    Same columns as bin_numeric, so a profile can hold both without branching.
    `width` carries the number of distinct values behind a row: 1 for a named
    value, the pooled count for "(other)".

    Cardinality is judged on a sample. A column whose distinct count is a large
    fraction of its rows — a description, a name, a key — has no distribution
    worth showing, so it collapses to a single row rather than hashing every
    value to prove it.

    Parameters
    ----------
    s : pandas.Series
        Text or categorical column. Anything else returns an empty frame.
    top_n : int, default 20
        Values to name individually. The remainder pool into "(other)".
    high_card_ratio : float, default 0.5
        Distinct values as a fraction of non-null rows, above which the column
        is reported as "(high cardinality)" and nothing else. Judged on the
        sample unless `n_unique` is given.
    n_unique : int or None, default None
        Distinct count, if the caller already has one — the cleaning record
        does. Used in place of the sample probe, so the decision is exact.
        Must exclude nulls, as `nunique()` does.
    sample_n : int, default 200_000
        Rows sampled for the cardinality probe. Counts are over everything.
    seed : int, default 1
        Sample seed, so a rerun makes the same decision.

    Returns
    -------
    pandas.DataFrame
        Columns bin, left, right, width, count, pct. `left` and `right` are
        always NA — there is no ordering to record — and are kept only so the
        frame matches bin_numeric. Empty frame if the column is not text or
        categorical, or holds no non-null values.
    """
    columns = ["bin", "left", "right", "width", "count", "pct"]

    # is_string_dtype is False for a categorical even when the categories are
    # strings, so a categorical needs its own test.
    is_text = pd.api.types.is_string_dtype(s.dtype)
    is_categorical = isinstance(s.dtype, pd.CategoricalDtype)
    if not (is_text or is_categorical):
        return pd.DataFrame(columns=columns)

    # Count of values that are present. Nulls are excluded from every figure
    # below, matching nunique() and bin_numeric.
    total = int(s.notna().sum())
    if total == 0:
        return pd.DataFrame(columns=columns)

    distinct = _distinct_estimate(s, total, n_unique, sample_n, seed)

    # A column that is mostly distinct values has no distribution to show, and
    # counting them exactly would mean a hash entry per row. The ratio alone is
    # not enough: on a short column every value looks distinct, and if there are
    # no more values than rows we would name anyway, there is nothing to pool.
    if distinct > top_n and distinct / total > high_card_ratio:
        rows = [
            {
                "bin": HIGH_CARDINALITY,
                # Exact when the caller supplied it, an estimate otherwise.
                "width": float(distinct) if n_unique is not None else np.nan,
                "count": total,
            }
        ]
        return _finish(rows, total, columns)

    # Cardinality is known to be modest, so the counts table is bounded and a
    # full pass is affordable. Counting on the sample instead would make every
    # figure an estimate for no memory saving at this point.
    counts = s.value_counts(dropna=True, sort=True)

    # A categorical lists every declared category, including ones no row uses.
    # An unused category is not part of the distribution.
    counts = counts[counts > 0]

    rows = []

    # Explicit loop: each named value gets its own row, and this is where any
    # per-value detail would go.
    head = counts.head(top_n)
    for value, count in head.items():
        rows.append({"bin": str(value), "width": 1.0, "count": int(count)})

    # Everything past top_n pools into one row carrying how many values it hides.
    pooled = counts.iloc[top_n:]
    if len(pooled):
        rows.append({"bin": OTHER, "width": float(len(pooled)), "count": int(pooled.sum())})

    return _finish(rows, total, columns)


def _distinct_estimate(s: pd.Series, total: int, n_unique: int | None, sample_n: int, seed: int) -> int:
    """Distinct count: the caller's if given, the categories if categorical, else a sample probe.

    A sample undercounts — 200k rows cannot show more than 200k distinct values
    — so this is only ever used to decide whether the column is worth counting
    properly, never reported as a figure.
    """
    if n_unique is not None:
        return n_unique

    # A categorical stores integer codes, so nunique() counts those rather than
    # the values — cheap enough to do exactly, whatever the row count. Note
    # len(s.cat.categories) is the wrong figure: it counts categories that were
    # declared but never used.
    if isinstance(s.dtype, pd.CategoricalDtype):
        return int(s.nunique())

    if total > sample_n:
        sample = s.sample(n=sample_n, random_state=seed)
        # Scale the sample's distinct count back up. Crude, but it only has to
        # separate "a handful of codes" from "one value per row".
        return int(sample.nunique() * (total / sample_n))

    return int(s.nunique())


def _finish(rows: list[dict], total: int, columns: list[str]) -> pd.DataFrame:
    """Add the derived columns and fix the column order."""
    out = pd.DataFrame(rows)
    # No ordering to record for text, but the columns stay so the frame matches
    # bin_numeric and anything downstream can treat the two the same.
    out["left"] = pd.NA
    out["right"] = pd.NA
    out["pct"] = (out["count"] / total * 100).round(2)
    return out[columns]
