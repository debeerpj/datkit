"""Draft: clip-then-histogram binning for numeric distributions."""

import numpy as np
import pandas as pd

NULL_BIN = "(null)"


def bin_numeric(
    s: pd.Series,
    clip: tuple[float, float] = (0.5, 99.5),
    bins: str | int = "doane",
    max_bins: int = 20,
    tail_min_span: float = 0.05,
    special_values: tuple[float, ...] = (),
    n_unique: int | None = None,
    sample_n: int | None = 100_000,
    seed: int = 1,
    precision: int = 3,
) -> pd.DataFrame:
    """Describe a numeric column's distribution as bins, or as values if there are few.

    Two modes, one output shape. A column with at most `max_bins` distinct
    values is reported value by value — a histogram over fewer values than bins
    invents structure, showing empty ranges between values that do not exist.
    Anything wider is binned: edges are chosen from a clipped sample, then
    applied to the whole column.

    Nulls and `special_values` never influence the edges but always appear as
    their own rows, so a column that is half null cannot look complete. Counts
    and percentages are over everything the function was handed.

    The outermost breaks are -inf and +inf, so the bins are contiguous and
    open-ended: data outside the observed range always has somewhere to land.
    The labels are `pd.cut(..., right=False)` intervals, so the same binning can
    be applied elsewhere:

        breaks = list(result["bin_min"]) + [result["bin_max"].iloc[-1]]
        pd.cut(other, bins=breaks, right=False)

    taking only the binned rows, and after removing `special_values` — those
    are extracted before binning, so the breaks alone do not reproduce them.

    Parameters
    ----------
    s : pandas.Series
        Numeric column. Non-numeric returns an empty frame.
    clip : tuple of float, default (0.5, 99.5)
        Percentiles bounding the binned region. Values outside fall in the
        open outer bins instead of stretching the edges. Subject to
        `tail_min_span`.
    bins : str or int, default "doane"
        Passed to numpy. A fixed count keeps the record one size whatever the
        row count, and makes two profiles comparable. The string rules are
        n-dependent: "fd", "scott" and "rice" scale with n**(1/3), so they ask
        for hundreds of bins on a large table; "sturges" and "doane" scale
        with log2(n) and stay readable.
    max_bins : int, default 20
        Ceiling on bin count, and the discrete threshold: a column with no more
        distinct values than this is listed value by value instead of binned.
    tail_min_span : float, default 0.05
        A side is only clipped if the span it excludes is at least this
        fraction of the span kept. Skewed columns usually have a tail on one
        side only; clipping the quiet side carves off a sliver of range into a
        bin narrower than the rest, holding almost nothing, and buys no
        protection for the edges.
    special_values : tuple of float, default ()
        Sentinels — a negative code in an otherwise positive column, say.
        Pulled out before the edges are worked out and reported one row each,
        so they cannot stretch the range. Infinities need no entry here: they
        are excluded from the edge calculation and land in the open outer bins.
    n_unique : int or None, default None
        Distinct count, if the caller already has one. Saves working it out
        again. Must exclude nulls. When None it is estimated from the sample
        and verified before use.
    sample_n : int or None, default 100_000
        Rows sampled to choose edges. None samples nothing — for a caller who
        sampled upstream. Counts are always over everything.
    seed : int, default 1
        Sample seed, so a rerun gives the same edges.
    precision : int, default 3
        Decimal places in the bin label and width.

    Returns
    -------
    pandas.DataFrame
        Columns bin, bin_min, bin_max, width, count, pct — one row per bin,
        per special value, per distinct value, plus a null row when there are
        nulls. Value rows carry bin_min == bin_max and width == 0; the open
        outer bins carry an infinite bound and width. Empty frame only when
        the column is not numeric, or has no rows at all.
    """
    columns = ["bin", "bin_min", "bin_max", "width", "count", "pct"]

    if not pd.api.types.is_numeric_dtype(s.dtype):
        return pd.DataFrame(columns=columns)

    # Everything the caller handed us. Nulls are part of the picture, so this is
    # what count and pct are measured against.
    total = len(s)
    if total == 0:
        return pd.DataFrame(columns=columns)

    # One conversion to a plain float array, because np.histogram cannot read a
    # nullable or Arrow-backed Series — pandas' NA does not survive asarray and
    # the call fails with "autodetected range is not finite". Converting here
    # makes every backend behave the same, and NaN becomes the only null.
    values = s.to_numpy(dtype="float64", na_value=np.nan, copy=False)

    nulls = int(np.isnan(values).sum())

    # Sentinels are extracted before anything measures the range.
    if len(special_values):
        specials = np.asarray(special_values, dtype="float64")
        is_special = np.isin(values, specials)
    else:
        specials = np.empty(0)
        is_special = np.zeros(values.size, dtype=bool)

    # Nulls out, sentinels out. Infinities stay: they are excluded from the edge
    # calculation below but belong in the open outer bins.
    work = values[~np.isnan(values) & ~is_special]

    rows = []

    if work.size > 0:
        # Only finite values can inform an edge. One infinity makes np.percentile
        # return NaN for the whole column.
        finite = work[np.isfinite(work)]
        rows.extend(_value_or_bin_rows(finite, work, bins, max_bins, clip, tail_min_span, n_unique, sample_n, seed, precision))

    # Explicit loop: one row per sentinel, ascending, so they read in order.
    for value in np.sort(specials):
        count = int((values == value).sum())
        if count > 0:
            rows.append({"bin": str(round(float(value), precision)), "bin_min": float(value), "bin_max": float(value), "count": count})

    if nulls > 0:
        rows.append({"bin": NULL_BIN, "bin_min": np.nan, "bin_max": np.nan, "count": nulls})

    if not rows:
        return pd.DataFrame(columns=columns)

    return _finish(rows, total, precision, columns)


def _value_or_bin_rows(finite, work, bins, max_bins, clip, tail_min_span, n_unique, sample_n, seed, precision):
    """Rows for the real values: one per distinct value if few, otherwise binned."""
    rows = []

    if finite.size == 0:
        # Nothing but infinities. Bin them against a degenerate range rather
        # than inventing edges from no finite data.
        edges = np.array([-np.inf, 0.0, np.inf])
        return _bin_rows(work, edges, precision)

    # rng.choice over the raw array rather than Series.sample: 30x faster and
    # far lighter, because .sample carries the index and rebuilds a Series. The
    # seeds are not interchangeable — different RNG streams, different rows —
    # which does not matter for choosing edges.
    if sample_n is not None and finite.size > sample_n:
        rng = np.random.default_rng(seed)
        sample = rng.choice(finite, size=sample_n, replace=False)
    else:
        sample = finite

    # A caller that already counted passes it in. Otherwise probe the sample —
    # nunique() over the full column is a hash pass we usually do not need, and
    # a sample that shows few distinct values is verified below before use.
    if n_unique is None:
        looks_discrete = np.unique(sample).size <= max_bins
    else:
        looks_discrete = n_unique <= max_bins

    if looks_discrete:
        uniques, counts = np.unique(work, return_counts=True)
        # The probe can undercount, and a passed-in figure can be stale. Either
        # way the real count decides; too many values falls through to binning.
        if uniques.size <= max_bins:
            # Explicit loop: one row per value, and the place to add anything
            # per-value later.
            for i in range(uniques.size):
                value = float(uniques[i])
                rows.append({"bin": str(round(value, precision)), "bin_min": value, "bin_max": value, "count": int(counts[i])})
            return rows

    low, high = np.percentile(sample, clip)

    smallest = float(finite.min())
    largest = float(finite.max())

    # A degenerate clip means almost every value is the same one. Bin across the
    # observed range instead of a zero-width one.
    if not high > low:
        low, high = smallest, largest
    if not high > low:
        low, high = smallest, smallest + 1.0

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

    # Open the ends. Bins are then contiguous from -inf to +inf, so nothing can
    # fall between them and future data outside the observed range still lands
    # somewhere. It also puts the infinities in a bin that honestly contains them.
    edges = np.concatenate(([-np.inf], edges, [np.inf]))

    rows.extend(_bin_rows(work, edges, precision))
    return rows


def _bin_rows(work, edges, precision):
    """One row per bin, counted over every value handed in."""
    counts, edges = np.histogram(work, bins=edges)

    rows = []
    # Explicit loop: each bin needs its own pair of edges, and this is the
    # place to add anything per-bin later.
    for i in range(len(counts)):
        rows.append(
            {
                "bin": _label(edges[i], edges[i + 1], precision),
                "bin_min": float(edges[i]),
                "bin_max": float(edges[i + 1]),
                "count": int(counts[i]),
            }
        )
    return rows


def _label(bin_min: float, bin_max: float, precision: int) -> str:
    """Interval label matching pd.cut(..., right=False).

    np.histogram bins are half-open [min, max), so a value on a boundary falls
    in the upper bin — the opposite of pd.cut's default. right=False switches
    pd.cut to the same convention, which is what makes these labels reusable.
    """
    return f"[{round(bin_min, precision)}, {round(bin_max, precision)})"


def _finish(rows: list[dict], total: int, precision: int, columns: list[str]) -> pd.DataFrame:
    """Add the derived columns and fix the column order."""
    # Built as a list of dicts and converted once: appending rows to a DataFrame
    # reallocates the whole frame each time, which is quadratic.
    out = pd.DataFrame(rows)
    # Core bins share a width; value rows are 0 and the open outer bins are inf.
    # Stating it beats implying it — a reader sees at a glance which rows differ.
    out["width"] = (out["bin_max"] - out["bin_min"]).round(precision)
    out["pct"] = (out["count"] / total * 100).round(2)
    return out[columns]
