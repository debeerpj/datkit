"""Draft: clip-then-histogram binning for numeric distributions."""

import numpy as np
import pandas as pd


def bin_numeric(
    s: pd.Series,
    clip: tuple[float, float] = (0.5, 99.5),
    bins: str | int = "fd",
    max_bins: int = 20,
    sample_n: int = 100_000,
    seed: int = 1,
) -> pd.DataFrame:
    """Bin a numeric column, keeping outliers out of the bins but not out of the count.

    Edges are chosen from the clipped *sample*, then applied to the whole
    column.
    Percentiles are the expensive part and barely move between a 100k sample
    and 90M rows.

    Parameters
    ----------
    s : pandas.Series
        Numeric column. Non-numeric returns an empty frame.
    clip : tuple of float, default (0.5, 99.5)
        Percentiles bounding the binned region. Values outside land in the
        tail rows instead of distorting the edges.
    bins : str or int, default "fd"
        Passed to numpy. "fd" (Freedman-Diaconis) is IQR-based and robust;
        "sturges" suits small n; "auto" is the larger of the two.
    max_bins : int, default 60
        Ceiling on bin count. "fd" can ask for thousands on spiky data, which
        is unreadable as a chart and pointless as a record.
    sample_n : int, default 200_000
        Rows sampled to choose edges. The counts are always over everything.
    seed : int, default 1
        Sample seed, so a rerun gives the same edges.

    Returns
    -------
    pandas.DataFrame
        One row per bin plus a row for each non-empty tail, with columns
        region, left, right, count, pct. Empty frame if the column has no
        numeric values to bin.
    """
    columns = ["region", "left", "right", "count", "pct"]

    if not pd.api.types.is_numeric_dtype(s.dtype):
        return pd.DataFrame(columns=columns)

    # One conversion to a plain float array. Arrow, nullable Int64 and numpy
    # columns all land in the same place, and NaN is then the only null.
    values = s.to_numpy(dtype="float64", na_value=np.nan, copy=False)
    values = values[~np.isnan(values)]

    total = values.size
    # Nothing to bin, and a constant column has no distribution to show.
    if total == 0 or values.min() == values.max():
        return pd.DataFrame(columns=columns)

    # Sample for the edge decision only.
    if total > sample_n:
        rng = np.random.default_rng(seed)
        sample = rng.choice(values, size=sample_n, replace=False)
    else:
        sample = values

    # Clip bounds come from the sample too — they are percentiles like any other.
    low, high = np.percentile(sample, clip)

    # Degenerate clip (a column that is mostly one value) leaves nothing to bin.
    if not high > low:
        return pd.DataFrame(columns=columns)

    core_sample = sample[(sample >= low) & (sample <= high)]
    edges = np.histogram_bin_edges(core_sample, bins=bins)

    # Redo at a fixed count if the rule asked for more bins than we will show.
    if len(edges) - 1 > max_bins:
        edges = np.histogram_bin_edges(core_sample, bins=max_bins)

    # Counts over the FULL column. np.histogram drops values outside the edge
    # range, which is what we want — the tails are counted separately below.
    counts, edges = np.histogram(values, bins=edges)

    below = int((values < edges[0]).sum())
    above = int((values > edges[-1]).sum())

    rows = []

    # Tail rows carry the real extremes, so nothing is lost by not binning them.
    if below:
        rows.append({"region": "low_tail", "left": float(values.min()), "right": float(edges[0]), "count": below})

    # Explicit loop: each bin needs its own pair of edges, and this is the
    # place to add anything per-bin later.
    for i in range(len(counts)):
        rows.append(
            {"region": "core", "left": float(edges[i]), "right": float(edges[i + 1]), "count": int(counts[i])}
        )

    if above:
        rows.append({"region": "high_tail", "left": float(edges[-1]), "right": float(values.max()), "count": above})

    out = pd.DataFrame(rows)
    out["pct"] = (out["count"] / total * 100).round(2)
    return out[columns]
