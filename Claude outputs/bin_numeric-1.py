"""Draft: clip-then-histogram binning for numeric distributions."""

import numpy as np
import pandas as pd


def bin_numeric(
    s: pd.Series,
    clip: tuple[float, float] = (0.5, 99.5),
    bins: str | int = "doane",
    max_bins: int = 20,
    tail_min_span: float = 0.05,
    discrete_max: int = 20,
    n_unique: int | None = None,
    sample_n: int = 100_000,
    seed: int = 1,
    precision: int = 3,
) -> pd.DataFrame:
    """Describe a numeric column's distribution as bins, or as values if there are few.

    Two modes, one output shape. A column with at most `discrete_max` distinct
    values is reported value by value — a histogram over fewer values than bins
    invents structure, showing empty ranges between values that do not exist.
    Anything wider is binned: edges are chosen from a clipped sample, then
    applied to the whole column.

    Parameters
    ----------
    s : pandas.Series
        Numeric column. Non-numeric returns an empty frame.
    clip : tuple of float, default (0.5, 99.5)
        Percentiles bounding the binned region. Values outside land in a tail
        row instead of stretching the edges. Subject to `tail_min_span`.
    bins : str or int, default 30
        Passed to numpy. A fixed count keeps the record one size whatever the
        row count, and makes two profiles comparable. The string rules are
        available but n-dependent: "fd", "scott" and "rice" scale with
        n**(1/3), so they ask for hundreds of bins on a large table;
        "sturges" and "doane" scale with log2(n) and stay readable.
    max_bins : int, default 60
        Ceiling on bin count, for when a string rule asks for too many.
    tail_min_span : float, default 0.05
        A side is only clipped if the span it excludes is at least this
        fraction of the span kept. Skewed columns usually have a tail on one
        side only; clipping the quiet side carves off a sliver of range into
        a bin narrower than the rest, holding almost nothing, and buys no
        protection for the edges.
    discrete_max : int, default 20
        At most this many distinct values, and each value gets its own row.
        Keep it below `bins`, or the binned mode is drawing empty ranges.
    n_unique : int or None, default None
        Distinct count, if the caller already has one. Saves working it out again. 
        Must exclude nulls. When None it is estimated from the sample 
        and verified before use.
    sample_n : int, default 200_000
        Rows sampled to choose edges. Counts are always over everything.
    seed : int, default 1
        Sample seed, so a rerun gives the same edges.
    precision : int, default 3
        Decimal places in the bin label and width.

    Returns
    -------
    pandas.DataFrame
        Columns bin, min, max, width, count, pct — one row per bin, per
        tail, or per distinct value. Discrete rows carry min == max and
        width == 0. Empty frame if there is nothing numeric to describe.
    """
    columns = ["bin", "bin_min", "bin_max", "width", "count", "pct"]

    if not pd.api.types.is_numeric_dtype(s.dtype):
        return pd.DataFrame(columns=columns)

    # One conversion to a plain float array. Arrow, nullable Int64 and numpy
    # columns all land in the same place, and NaN is then the only null.
    values = s.to_numpy(dtype="float64", na_value=np.nan, copy=False)
    values = values[~np.isnan(values)]

    total = values.size
    if total == 0:
        return pd.DataFrame(columns=columns)

    # Sample once: the edge rules and the cardinality probe both use it.
    if total > sample_n:
        rng = np.random.default_rng(seed)
        sample = rng.choice(values, size=sample_n, replace=False)
    else:
        sample = values

    # A caller that already counted passes it in. Otherwise probe the sample —
    # nunique() over the full column is a hash pass we usually do not need, and
    # a sample that shows few distinct values is verified below before use.
    if n_unique is None:
        looks_discrete = np.unique(sample).size <= discrete_max
    else:
        looks_discrete = n_unique <= discrete_max

    if looks_discrete:
        uniques, counts = np.unique(values, return_counts=True)
        # The probe can undercount, and a passed-in figure can be stale. Either
        # way the real count decides; too many values falls through to binning.
        if uniques.size <= discrete_max:
            rows = []
            # Explicit loop: one row per value, and the place to add anything
            # per-value later.
            for i in range(uniques.size):
                value = float(uniques[i])
                rows.append(
                    {
                        "bin": str(round(value, precision)),
                        "bin_min": value,
                        "bin_max": value,
                        "count": int(counts[i]),
                    }
                )
            return _finish(rows, total, precision, columns)

    # A constant column that got past the discrete check has no distribution.
    if values.min() == values.max():
        return pd.DataFrame(columns=columns)

    # Clip bounds come from the sample too — they are percentiles like any other.
    low, high = np.percentile(sample, clip)

    # Degenerate clip (a column that is mostly one value) leaves nothing to bin.
    if not high > low:
        return pd.DataFrame(columns=columns)

    smallest = float(values.min())
    largest = float(values.max())
    core_span = high - low

    # Decide each side on its own. Measuring the excluded span against the kept
    # span keeps the test scale-free, and unlike a standard deviation it cannot
    # be inflated by the very outliers it is meant to detect.
    if (low - smallest) < tail_min_span * core_span:
        low = smallest
    if (largest - high) < tail_min_span * core_span:
        high = largest

    # range= does the clipping: with a string rule numpy computes the bin width
    # from the data inside the range only. No mask, no filtered copy, and the
    # edges land on the bounds rather than on whichever sampled value sat
    # nearest them.
    edges = np.histogram_bin_edges(sample, bins=bins, range=(low, high))

    # Redo at a fixed count if the rule asked for more bins than we will show.
    # range= again, or the re-derived edges would span the whole column.
    if len(edges) - 1 > max_bins:
        edges = np.histogram_bin_edges(sample, bins=max_bins, range=(low, high))

    # Counts over the FULL column. np.histogram drops values outside the edge
    # range, which is what we want — the tails are counted separately below.
    counts, edges = np.histogram(values, bins=edges)

    below = int((values < edges[0]).sum())
    above = int((values > edges[-1]).sum())

    rows = []

    # Tail rows carry the real extremes, so nothing is lost by not binning them.
    if below:
        rows.append(
            {
                "bin": _label(smallest, edges[0], last=False, precision=precision),
                "bin_min": smallest,
                "bin_max": float(edges[0]),
                "count": below,
            }
        )

    # Explicit loop: each bin needs its own pair of edges, and this is the
    # place to add anything per-bin later.
    for i in range(len(counts)):
        rows.append(
            {
                "bin": _label(edges[i], edges[i + 1], last=(i == len(counts) - 1 and not above), precision=precision),
                "bin_min": float(edges[i]),
                "bin_max": float(edges[i + 1]),
                "count": int(counts[i]),
            }
        )

    if above:
        rows.append(
            {
                "bin": _label(edges[-1], largest, last=True, precision=precision),
                "bin_min": float(edges[-1]),
                "bin_max": largest,
                "count": above,
            }
        )

    return _finish(rows, total, precision, columns)


def _label(bin_min: float, bin_max: float, last: bool, precision: int) -> str:
    """Interval label in pd.cut's style, but with the closure numpy actually uses.

    np.histogram bins are half-open [left, right) so a value on a boundary
    falls in the upper bin — the opposite of pd.cut's (left, right]. Only the
    final bin includes its right edge.
    """
    close = "]" if last else ")"
    return f"[{round(bin_min, precision)}, {round(bin_max, precision)}{close}"


def _finish(rows: list[dict], total: int, precision: int, columns: list[str]) -> pd.DataFrame:
    """Add the derived columns and fix the column order."""
    out = pd.DataFrame(rows)
    # Core bins share a width; tails and discrete rows do not. Stating it beats
    # implying it — a reader can see at a glance which rows are not bins.
    out["width"] = (out["bin_max"] - out["bin_min"]).round(precision)
    out["pct"] = (out["count"] / total * 100).round(2)
    return out[columns]
