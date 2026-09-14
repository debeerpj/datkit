"""Distribution summaries for a single column: numeric, categorical or date.

Four public functions. `bin_column` picks one of the other three by dtype; all
return the same six columns, so a profile can hold every column of a table in
one frame without branching.

    bin          label for the row
    bin_min      lower bound, or the value itself, or NaN for the null row
    bin_max      upper bound, or the value itself, or NaN for the null row
    width        bin_max - bin_min; 0 for a single value, NaN for the null row
    count        rows in this bin
    pct          count as a percentage of every row handed in, nulls included
"""

import numpy as np
import pandas as pd

NULL_BIN = "(null)"
HIGH_CARDINALITY = "(high cardinality)"
OTHER = "(other)"

COLUMNS = ["bin", "bin_min", "bin_max", "width", "count", "pct"]

# Calendar grains, coarsest-last. Each is (period frequency, years per band);
# a band of 1 is an ordinary period, more than 1 groups whole years together.
# Quarter sits between month and year so the step is not a bare 12x jump.
GRAINS = [("D", 1), ("W", 1), ("M", 1), ("Q", 1), ("Y", 1), ("Y", 5), ("Y", 10)]


def bin_column(
    s: pd.Series,
    special_values: tuple = (),
    n_unique: int | None = None,
    sample_n: int | None = 100_000,
    seed: int = 1,
    **kwargs,
) -> pd.DataFrame:
    """Summarise any column, choosing the numeric or categorical treatment by dtype.

    A dispatcher only. It does not decide what a column *should* be — that is
    the cleaning module's job, and two modules deciding a column's type is how
    a profile ends up disagreeing with the cleaning record. A numeric column
    still stored as text will be summarised as text; fix that in cleaning.

    Parameters
    ----------
    s : pandas.Series
        Any column. A dtype that is none of numeric, text, categorical or
        datetime returns an empty frame.
    special_values : tuple, default ()
        Sentinels to pull out before anything else. See bin_numeric.
    n_unique, sample_n, seed
        Passed through to whichever function is chosen.
    **kwargs
        Extra arguments for the chosen function — `clip`, `bins`, `max_bins`,
        `tail_min_span`, `precision` for numeric; `top_n`, `high_card_ratio`
        for categorical; `max_bins` for dates. An argument the chosen function
        does not accept is a
        TypeError, so a typo is loud rather than ignored.

    Returns
    -------
    pandas.DataFrame
        As described in the module docstring.
    """
    shared = {"special_values": special_values, "n_unique": n_unique, "sample_n": sample_n, "seed": seed}

    if pd.api.types.is_numeric_dtype(s.dtype):
        return bin_numeric(s, **shared, **kwargs)

    # is_string_dtype is False for a categorical even when the categories are
    # strings, so a categorical needs its own test.
    if pd.api.types.is_string_dtype(s.dtype) or isinstance(s.dtype, pd.CategoricalDtype):
        return bin_categorical(s, **shared, **kwargs)

    if pd.api.types.is_datetime64_any_dtype(s.dtype):
        return bin_dates(s, **shared, **kwargs)

    return pd.DataFrame(columns=COLUMNS)


def bin_numeric(
    s: pd.Series,
    clip: tuple[float, float] = (0.5, 99.5),
    bins: str | int = "doane",
    max_bins: int = 20,
    tail_min_span: float = 0.05,
    special_values: tuple = (),
    n_unique: int | None = None,
    sample_n: int | None = 100_000,
    seed: int = 1,
    precision: int = 3,
) -> pd.DataFrame:
    """Describe a numeric column as bins, or as values if there are few enough.

    A column with at most `max_bins` distinct values is listed value by value —
    a histogram over fewer values than bins invents structure, showing empty
    ranges between values that do not exist. Anything wider is binned: edges
    come from a clipped sample, then are applied to the whole column.

    The outermost bins are open, labelled `[-inf, x)` and `[x, inf)`, so the
    bins are contiguous and data outside the observed range always lands
    somewhere. `bin_min` and `bin_max` on those rows carry the observed extreme
    rather than the infinity, because a sentinel nobody declared shows up there
    and is worth seeing. To apply the same bins elsewhere, put the infinities
    back:

        binned = result[result["width"] > 0]
        breaks = [-np.inf] + list(binned["bin_min"])[1:] + [np.inf]
        pd.cut(other, bins=breaks, right=False)

    after removing `special_values`, which are extracted before binning.

    Parameters
    ----------
    s : pandas.Series
        Numeric column. Non-numeric returns an empty frame.
    clip : tuple of float, default (0.5, 99.5)
        Percentiles bounding the binned region. Values outside fall in the open
        outer bins instead of stretching the edges. Subject to `tail_min_span`.
    bins : str or int, default "doane"
        Passed to numpy. A fixed count keeps the record one size whatever the
        row count, and makes two profiles comparable. The string rules are
        n-dependent: "fd", "scott" and "rice" scale with n**(1/3), so they ask
        for hundreds of bins on a large table; "sturges" and "doane" scale with
        log2(n) and stay readable.
    max_bins : int, default 20
        Ceiling on bin count, and the discrete threshold: no more distinct
        values than this and the column is listed rather than binned.
    tail_min_span : float, default 0.05
        A side is only clipped if the span it excludes is at least this
        fraction of the span kept. Skewed columns usually have a tail on one
        side only; clipping the quiet side carves off a sliver of range into a
        bin narrower than the rest, holding almost nothing.
    special_values : tuple, default ()
        Sentinels — a negative code in an otherwise positive column, say.
        Pulled out before the edges are worked out and reported one row each,
        in the order given. Infinities need no entry: they are excluded from
        the edge calculation and land in the open outer bins.
    n_unique : int or None, default None
        Distinct count, if the caller already has one. Must exclude nulls.
        When None it is estimated from the sample and verified before use.
    sample_n : int or None, default 100_000
        Rows sampled to choose edges. None samples nothing, for a caller who
        sampled upstream. Counts are always over everything.
    seed : int, default 1
        Sample seed, so a rerun gives the same edges.
    precision : int, default 3
        Decimal places in the bin label and width.

    Returns
    -------
    pandas.DataFrame
        As described in the module docstring.
    """
    if not pd.api.types.is_numeric_dtype(s.dtype):
        return pd.DataFrame(columns=COLUMNS)

    keep, extra_rows, total = _extract(s, special_values, precision)
    if total == 0:
        return pd.DataFrame(columns=COLUMNS)

    # One conversion to a plain float array, because np.histogram cannot read a
    # nullable or Arrow-backed Series: pandas' NA does not survive asarray and
    # the call fails with "autodetected range is not finite". Converting here
    # makes every backend behave the same.
    work = s.to_numpy(dtype="float64", na_value=np.nan, copy=False)[keep]

    rows = _numeric_rows(work, clip, bins, max_bins, tail_min_span, n_unique, sample_n, seed, precision)
    rows.extend(extra_rows)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return _finish(rows, total, precision)


def bin_categorical(
    s: pd.Series,
    top_n: int = 20,
    high_card_ratio: float = 0.5,
    special_values: tuple = (),
    n_unique: int | None = None,
    sample_n: int | None = 100_000,
    seed: int = 1,
    precision: int = 3,
) -> pd.DataFrame:
    """Describe a text or categorical column as its top values, with the rest pooled.

    `width` carries the number of distinct values behind a row: 1 for a named
    value, the pooled count for "(other)". `bin_min` and `bin_max` are always
    NA — there is no ordering to record — and exist so the frame matches
    bin_numeric.

    A column whose distinct count is a large fraction of its rows — a
    description, a name, a key — has no distribution worth showing, so it
    collapses to a single "(high cardinality)" row rather than hashing every
    value to prove it.

    Parameters
    ----------
    s : pandas.Series
        Text or categorical column. Anything else returns an empty frame.
    top_n : int, default 20
        Values to name individually. The remainder pool into "(other)".
    high_card_ratio : float, default 0.5
        Distinct values as a fraction of non-null rows, above which the column
        is reported as "(high cardinality)" and nothing else.
    special_values : tuple, default ()
        Values to report on their own row before anything else is counted, in
        the order given. They take no part in the cardinality decision.
    n_unique : int or None, default None
        Distinct count, if the caller already has one. Must exclude nulls.
    sample_n : int or None, default 100_000
        Rows sampled for the cardinality probe. None probes the whole column.
    seed : int, default 1
        Sample seed, so a rerun makes the same decision.
    precision : int, default 3
        Unused here; accepted so the two functions take the same arguments.

    Returns
    -------
    pandas.DataFrame
        As described in the module docstring.
    """
    is_text = pd.api.types.is_string_dtype(s.dtype)
    is_categorical = isinstance(s.dtype, pd.CategoricalDtype)
    if not (is_text or is_categorical):
        return pd.DataFrame(columns=COLUMNS)

    keep, extra_rows, total = _extract(s, special_values, precision)
    if total == 0:
        return pd.DataFrame(columns=COLUMNS)

    work = s[keep]
    rows = _categorical_rows(work, top_n, high_card_ratio, n_unique, sample_n, seed)
    rows.extend(extra_rows)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return _finish(rows, total, precision)


# ----------------------------------------------------------------------------
# Shared preamble
# ----------------------------------------------------------------------------


def _extract(s: pd.Series, special_values: tuple, precision: int) -> tuple[np.ndarray, list[dict], int]:
    """Pull nulls and sentinels out, and build their rows.

    Both of those are excluded from anything that measures a range or counts
    cardinality, but both always appear in the result — a column that is half
    null must not be able to look complete.

    Returns
    -------
    tuple of (numpy.ndarray, list of dict, int)
        A boolean mask of the rows left to describe, the rows for the values
        removed, and the total row count every percentage is measured against.
    """
    total = len(s)
    if total == 0:
        return np.zeros(0, dtype=bool), [], 0

    is_null = s.isna().to_numpy(dtype=bool)

    if len(special_values):
        is_special = s.isin(special_values).to_numpy(dtype=bool)
    else:
        is_special = np.zeros(total, dtype=bool)

    rows = []

    # Explicit loop: one row per sentinel, in the order the caller listed them,
    # so the output order is theirs to control.
    for value in special_values:
        count = int((s == value).sum())
        if count > 0:
            rows.append(_value_row(value, count, precision))

    nulls = int(is_null.sum())
    if nulls > 0:
        rows.append({"bin": NULL_BIN, "bin_min": np.nan, "bin_max": np.nan, "count": nulls})

    return ~is_null & ~is_special, rows, total


def _value_row(value, count: int, precision: int) -> dict:
    """A row standing for one exact value rather than a range."""
    if isinstance(value, pd.Timestamp):
        # A sentinel date — 1900-01-01, 9999-12-31 — is one instant, so zero span.
        return {"bin": str(value.date()), "bin_min": value, "bin_max": value, "width": 0.0, "count": count}

    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return {"bin": str(round(number, precision)), "bin_min": number, "bin_max": number, "count": count}
    # Text: width counts distinct values, as it does for every categorical row,
    # so a sentinel reads as the one value it is.
    return {"bin": str(value), "bin_min": pd.NA, "bin_max": pd.NA, "width": 1.0, "count": count}


def _finish(rows: list[dict], total: int, precision: int) -> pd.DataFrame:
    """Add the derived columns and fix the column order.

    Built as a list of dicts and converted once: appending rows to a DataFrame
    reallocates the whole frame each time, which is quadratic.
    """
    out = pd.DataFrame(rows)

    if "width" not in out.columns:
        out["width"] = np.nan

    # Derive the missing widths only where the bounds are numbers. Categorical
    # rows set their own (a count of values) and date rows set theirs (days),
    # and subtracting timestamps would give a Timedelta, not a number.
    if pd.api.types.is_numeric_dtype(out["bin_min"]):
        out["width"] = out["width"].fillna(out["bin_max"] - out["bin_min"])

    out["width"] = pd.to_numeric(out["width"], errors="coerce").round(precision)
    out["pct"] = (out["count"] / total * 100).round(2)
    return out[COLUMNS]


# ----------------------------------------------------------------------------
# Numeric
# ----------------------------------------------------------------------------


def _numeric_rows(work, clip, bins, max_bins, tail_min_span, n_unique, sample_n, seed, precision) -> list[dict]:
    """Rows for the values left after nulls and sentinels: listed if few, else binned."""
    if work.size == 0:
        return []

    # Only finite values can inform an edge. One infinity makes np.percentile
    # return NaN for the whole column.
    finite = work[np.isfinite(work)]
    if finite.size == 0:
        return _bin_rows(work, np.array([-np.inf, 0.0, np.inf]), precision)

    sample = _sample(finite, sample_n, seed)

    if _looks_discrete(sample, n_unique, max_bins):
        uniques, counts = np.unique(work, return_counts=True)
        # The probe can undercount and a passed-in figure can be stale, so the
        # real count decides; too many values falls through to binning.
        if uniques.size <= max_bins:
            rows = []
            # Explicit loop: one row per value, and the place to add anything
            # per-value later.
            for i in range(uniques.size):
                rows.append(_value_row(float(uniques[i]), int(counts[i]), precision))
            return rows

    edges = _choose_edges(finite, sample, clip, bins, max_bins, tail_min_span)
    return _bin_rows(work, edges, precision)


def _sample(values: np.ndarray, sample_n: int | None, seed: int) -> np.ndarray:
    """A bounded sample for the decisions, or everything when sampling is off.

    rng.choice over the raw array rather than Series.sample: about 30x faster
    and far lighter, because .sample carries the index and rebuilds a Series.
    The seeds are not interchangeable — different RNG streams pick different
    rows — which does not matter for choosing edges.
    """
    if sample_n is None or values.size <= sample_n:
        return values
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=sample_n, replace=False)


def _looks_discrete(sample: np.ndarray, n_unique: int | None, max_bins: int) -> bool:
    """Whether to list values instead of binning them.

    A caller that already counted passes it in; otherwise probe the sample,
    because nunique() over the full column is a hash pass usually not needed.
    Either way the answer is a hint — the caller verifies against the real
    count before acting on it.
    """
    if n_unique is not None:
        return n_unique <= max_bins
    return np.unique(sample).size <= max_bins


def _choose_edges(finite, sample, clip, bins, max_bins, tail_min_span) -> np.ndarray:
    """Bin edges, from the clipped sample, with open ends.

    Every decision about where bins fall lives here and nowhere else, so it can
    be tested by looking at an array instead of reading a frame.
    """
    low, high = np.percentile(sample, clip)

    smallest = float(finite.min())
    largest = float(finite.max())

    # A degenerate clip means nearly every value is the same one. Fall back to
    # the observed range, then to an arbitrary unit span for a constant column
    # that reached here.
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

    # Open the ends: bins are then contiguous from -inf to +inf, so nothing can
    # fall between them, future data outside the observed range still lands
    # somewhere, and the infinities sit in a bin that honestly contains them.
    return np.concatenate(([-np.inf], edges, [np.inf]))


def _bin_rows(work: np.ndarray, edges: np.ndarray, precision: int) -> list[dict]:
    """One row per bin, counted over every value handed in.

    The open outer bins report the observed extreme as bin_min/bin_max rather
    than the infinity in their label: an undeclared sentinel turns up there,
    and -inf would hide it.
    """
    counts, edges = np.histogram(work, bins=edges)

    smallest = float(work.min())
    largest = float(work.max())
    last = len(counts) - 1

    rows = []
    # Explicit loop: each bin needs its own pair of edges, and this is the
    # place to add anything per-bin later.
    for i in range(len(counts)):
        rows.append(
            {
                "bin": _label(edges[i], edges[i + 1], precision),
                "bin_min": smallest if i == 0 else float(edges[i]),
                "bin_max": largest if i == last else float(edges[i + 1]),
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


def bin_dates(
    s: pd.Series,
    max_bins: int = 20,
    special_values: tuple = (),
    n_unique: int | None = None,
    sample_n: int | None = 100_000,
    seed: int = 1,
    precision: int = 3,
) -> pd.DataFrame:
    """Describe a date column as calendar periods, at the finest grain that fits.

    Day, week, month, quarter, year, then 5- and 10-year bands. The first grain
    whose span is no more than `max_bins` periods wins, so seven years of data
    comes back as years and three months comes back as weeks.

    The axis is contiguous: a period with no rows still gets a row, at zero. A
    month missing from the middle of a series is a finding, and dropping it
    would hide the gap.

    There are no open outer bins, unlike bin_numeric. Calendar bins are reusable
    by grain rather than by edges — anyone applying them elsewhere writes
    `dt.to_period("M")` and needs no break list — so infinite ends would be dead
    weight.

    Sentinel dates matter more here than anywhere else. One 1900-01-01 or
    9999-12-31 turns a two-year series into a century and forces the coarsest
    grain, so pass them in `special_values`.

    Parameters
    ----------
    s : pandas.Series
        Datetime column, timezone-aware or not. Anything else returns an empty
        frame.
    max_bins : int, default 20
        Most periods to show. The grain escalates until the span fits.
    special_values : tuple, default ()
        Sentinel dates, given as anything pd.to_datetime accepts. Pulled out
        before the span is measured and reported one row each, in the order
        given.
    n_unique : int or None, default None
        Accepted for a common signature; the grain comes from the span, not
        from a distinct count, so it is unused.
    sample_n, seed : int or None
        Accepted for a common signature. Unused — min and max are one pass and
        a sample cannot improve on them.
    precision : int, default 3
        Decimal places in the width.

    Returns
    -------
    pandas.DataFrame
        As described in the module docstring. `bin_min` and `bin_max` are the
        first and last instant of the period; `width` is its length in days.
    """
    if not pd.api.types.is_datetime64_any_dtype(s.dtype):
        return pd.DataFrame(columns=COLUMNS)

    # Sentinels arrive as strings or dates; isin only matches like for like, so
    # a naive sentinel would never match a tz-aware column.
    if len(special_values):
        specials = pd.to_datetime(list(special_values))
        if s.dt.tz is not None:
            specials = specials.tz_localize(s.dt.tz) if specials.tz is None else specials.tz_convert(s.dt.tz)
        specials = tuple(specials)
    else:
        specials = ()

    keep, extra_rows, total = _extract(s, specials, precision)
    if total == 0:
        return pd.DataFrame(columns=COLUMNS)

    work = s[keep]
    rows = _date_rows(work, max_bins)
    rows.extend(extra_rows)

    if not rows:
        return pd.DataFrame(columns=COLUMNS)
    return _finish(rows, total, precision)


def _date_rows(work: pd.Series, max_bins: int) -> list[dict]:
    """Rows for the dates left after nulls and sentinels, at the chosen grain."""
    if work.empty:
        return []

    # Periods have no timezone. Drop it rather than let pandas warn: keeping the
    # local wall time is what someone profiling "orders per month" expects, and
    # converting to UTC first would shift rows across period boundaries.
    if work.dt.tz is not None:
        work = work.dt.tz_localize(None)

    first = work.min()
    last = work.max()

    freq, band = _choose_grain(first, last, max_bins)

    if band > 1:
        return _band_rows(work, band)
    return _period_rows(work, freq)


def _choose_grain(first: pd.Timestamp, last: pd.Timestamp, max_bins: int) -> tuple[str, int]:
    """The finest calendar grain whose span fits in max_bins periods.

    Counted from the two extremes with period arithmetic, so nothing is grouped
    until the grain is settled. Falls back to the coarsest grain rather than
    failing — ten-year bands over a very long series is still a distribution.
    """
    # Explicit loop: each grain is tried in turn and the first that fits wins.
    for freq, band in GRAINS:
        if _period_span(first, last, freq, band) <= max_bins:
            return freq, band
    return GRAINS[-1]


def _period_span(first: pd.Timestamp, last: pd.Timestamp, freq: str, band: int) -> int:
    """How many periods of this grain the range covers, endpoints included."""
    if band > 1:
        # Bands are whole years floored to a multiple of the band width, so
        # 1997 with a 5-year band belongs to 1995.
        return (_floor_year(last.year, band) - _floor_year(first.year, band)) // band + 1
    return (last.to_period(freq) - first.to_period(freq)).n + 1


def _floor_year(year: int, band: int) -> int:
    """The first year of the band this year falls in."""
    return year - (year % band)


def _period_rows(work: pd.Series, freq: str) -> list[dict]:
    """One row per calendar period, including the periods nothing landed in."""
    periods = work.dt.to_period(freq)
    counts = periods.value_counts()

    # Reindexing onto a full period_range is what puts the empty periods back;
    # value_counts alone would silently drop them.
    axis = pd.period_range(periods.min(), periods.max(), freq=freq)
    counts = counts.reindex(axis, fill_value=0)

    rows = []
    # Explicit loop: each period needs its own label and bounds, and this is
    # the place to add anything per-period later.
    for period, count in counts.items():
        rows.append(_period_row(period.start_time, period.end_time, _period_label(period, freq), int(count)))
    return rows


def _band_rows(work: pd.Series, band: int) -> list[dict]:
    """One row per multi-year band, including bands nothing landed in."""
    years = work.dt.year.to_numpy()
    floored = years - (years % band)

    start = int(floored.min())
    stop = int(floored.max())

    rows = []
    # Explicit loop: step through every band in the range so gaps appear as
    # zero rows rather than vanishing.
    for year in range(start, stop + 1, band):
        count = int(((floored >= year) & (floored < year + band)).sum())
        first = pd.Timestamp(year=year, month=1, day=1)
        last = pd.Period(year + band - 1, freq="Y").end_time
        rows.append(_period_row(first, last, f"{year}-{year + band - 1}", count))
    return rows


def _period_row(first: pd.Timestamp, last: pd.Timestamp, label: str, count: int) -> dict:
    """A row for one calendar period, with its width in whole days."""
    return {
        "bin": label,
        "bin_min": first,
        "bin_max": last,
        # end_time is the last instant before the next period, so normalising
        # both ends and adding one counts the days inclusively.
        "width": float((last.normalize() - first.normalize()).days + 1),
        "count": count,
    }


def _period_label(period: pd.Period, freq: str) -> str:
    """Period label. Pandas renders a week as a date range; ISO week is shorter."""
    if freq == "W":
        return period.start_time.strftime("%G-W%V")
    return str(period)


# ----------------------------------------------------------------------------
# Categorical
# ----------------------------------------------------------------------------


def _categorical_rows(work: pd.Series, top_n, high_card_ratio, n_unique, sample_n, seed) -> list[dict]:
    """Rows for the top values, with the remainder pooled — or one row if too varied."""
    present = int(work.notna().sum())
    if present == 0:
        return []

    distinct = _distinct_estimate(work, present, n_unique, sample_n, seed)

    # The ratio alone is not enough: on a short column every value looks
    # distinct, and if there are no more values than we would name anyway,
    # there is nothing to pool.
    if distinct > top_n and distinct / present > high_card_ratio:
        return [
            {
                "bin": HIGH_CARDINALITY,
                "bin_min": pd.NA,
                "bin_max": pd.NA,
                # Exact when the caller supplied it, an estimate otherwise.
                "width": float(distinct) if n_unique is not None else np.nan,
                "count": present,
            }
        ]

    # Cardinality is known to be modest, so the counts table is bounded and a
    # full pass is affordable. Counting on the sample would make every figure
    # an estimate for no memory saving at this point.
    counts = work.value_counts(dropna=True, sort=True)
    # A categorical lists every declared category, including unused ones.
    counts = counts[counts > 0]

    rows = []
    # Explicit loop: each named value gets its own row, and this is where any
    # per-value detail would go.
    for value, count in counts.head(top_n).items():
        rows.append({"bin": str(value), "bin_min": pd.NA, "bin_max": pd.NA, "width": 1.0, "count": int(count)})

    pooled = counts.iloc[top_n:]
    if len(pooled) > 0:
        rows.append(
            {"bin": OTHER, "bin_min": pd.NA, "bin_max": pd.NA, "width": float(len(pooled)), "count": int(pooled.sum())}
        )
    return rows


def _distinct_estimate(s: pd.Series, present: int, n_unique: int | None, sample_n: int | None, seed: int) -> int:
    """Distinct count: the caller's if given, the codes if categorical, else a sample probe.

    A sample undercounts, so this only ever decides whether the column is worth
    counting properly. It is never reported as a figure.
    """
    if n_unique is not None:
        return n_unique

    # A categorical stores integer codes, so nunique() counts those rather than
    # the values — cheap at any row count. len(s.cat.categories) is the wrong
    # figure: it counts categories declared but never used.
    if isinstance(s.dtype, pd.CategoricalDtype):
        return int(s.nunique())

    if sample_n is not None and present > sample_n:
        sample = s.sample(n=sample_n, random_state=seed)
        # Scale back up. Crude, but it only has to separate "a handful of
        # codes" from "one value per row".
        return int(sample.nunique() * (present / sample_n))

    return int(s.nunique())
