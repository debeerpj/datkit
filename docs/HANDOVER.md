# datkit — handover

Where things stand. Updated 2026-09-14.

`CLAUDE.md` at the repo root carries how I work and the target environment.
`docs/SETUP.md` carries the full build log. This file is state and next steps.

---

## Current state

**In the repo, working:** `cleaning.py` and `schema.py`, tests passing, ruff
clean.

**Drafted, reviewed, not merged:** the profiling module and the cleaning
additions. Both live in `Claude outputs/` and need to move into `src/datkit/`
once the outstanding items below are done.

| File | What it is |
|---|---|
| `Claude outputs/binning_v2.py` | **Current** profiling module. Everything else in that folder is superseded — delete `binning.py`, `binning-1.py`, `bin_numeric*.py`, `bin_categorical.py`. |
| `Claude outputs/cleaning_changes.py` | Draft additions to `cleaning.py`: sentinel discovery and the effective-dtype record fields. |

---

## What changed since the last handover

### `clean_dates` rewritten

Was: parse the whole column twice, once per `dayfirst` value, through dateutil.
Now: guess the format from a sample, check the sample agrees, parse once with
`format=`. Written up as **Step 13a** in `docs/SETUP.md`, including the traps —
`guess_datetime_format` is case-sensitive on month names while `to_datetime` is
not, and `dayfirst=True` silently returns `%Y-%d-%m` for ISO strings.

### The profiling module (`binning_v2.py`)

One module, four public functions, same six columns from all of them —
`bin`, `bin_min`, `bin_max`, `width`, `count`, `pct` — so a table profile can
hold every column in one frame without branching.

```
bin_column(s, ...)        dispatch on dtype, or on a recorded effective dtype
bin_numeric(s, ...)       bins, or values listed when there are few
bin_categorical(s, ...)   top-N with the rest pooled, or one high-cardinality row
bin_dates(s, ...)         calendar periods at the finest grain that fits
```

Helpers split so each is testable alone: `_extract`, `_choose_edges`,
`_looks_discrete`, `_sample`, `_bin_rows`, `_categorical_rows`, `_date_rows`,
`_finish`.

Behaviour worth knowing:

- Nulls and `special_values` are pulled out before anything measures a range,
  and always appear as their own rows. `count` and `pct` are over every row
  handed in, so a column that is half null cannot look complete.
- Numeric bins are open at the ends, labelled `[-inf, x)` and `[x, inf)`, so
  future data always lands somewhere. `bin_min`/`bin_max` on those rows carry
  the **observed** extreme, because an undeclared sentinel shows up there.
- Labels are `pd.cut(..., right=False)` intervals, so the same bins can be
  applied elsewhere: `[-inf] + list(binned["bin_min"])[1:] + [inf]`.
- Date grain escalates D → W → M → Q → Y → 5-year → 10-year, first fit wins.
  The period axis is contiguous, so an empty month appears as a zero row.
- `effective_dtype` lets a text column that is numeric-once-sentinels-are-gone
  be summarised as numeric. Whatever still fails to convert becomes an
  `(unconvertible)` row rather than vanishing from the counts.

Tested across 17 column shapes on pandas 3.0.2 **and** pandas 2.2.3; counts and
percentages reconcile on all of them.

### Cleaning additions (`cleaning_changes.py`)

Nothing changes about what `clean_column` returns. The record gains a second
verdict: `dtype_effective`, `specials`, `specials_count`, `n_unique`.

`discover_specials` proposes the few values blocking a conversion — it never
acts on them, and discovered sentinels must **not** join `to_null`'s token
list, because nulling them destroys the information.

Two guards, both needed: at most `MAX_SPECIALS` candidates, and they must be a
minority of the column's distinct values. Without the second, a six-row column
of names proposes all six names as sentinels.

### Vision recorded

`docs/SETUP.md` now has a **High-level vision** section: one repo with
independent modules enforced by an import test, optional extras for viz and
LLM, the four-step LLM flow with a human checkpoint, and the requirement for a
redaction level before any profile leaves the room. PII detection is noted as
much later, and belongs *before* redaction.

### The cleanroom package list arrived

Python **3.10**, pandas **2.2.3**, numpy **2.2.0**. 174 packages. Notable:
`pytest 8.3.4` **is** there, so the test suite can run in the target
environment. `duckdb 1.1.1`, `polars 1.8.2` (old — its streaming engine is the
incomplete one, do not plan around current Polars docs), `pyarrow 18.1.0`,
`tdigest`, `visions`, `splink`, `optbinning`, `ydata-profiling`, `sweetviz`.
Not there: `pandera`, `great-expectations`, `narwhals`, `whylogs`, `dask`.

---

## Next, in order

### 1. Fix the `is_string_dtype` guard — this is a live bug

`is_string_dtype` returns `True` for **any** object column. The cleanroom is
pandas 2.2.3, where `object` is still the default for text. Running the current
`clean_column` there:

```
object of ints   -> AttributeError: Can only use .str accessor with string values!
object mixed     -> [nan, 'a', None]      the integer 1 silently became NaN
object of None   -> float64
object strings   -> ok
```

A crash on one shape, silent data loss on another. Neither shows up locally on
pandas 3, where text arrives as the `str` dtype.

Fix: `pd.api.types.infer_dtype(s, skipna=True) == "string"`, which inspects the
values. Verified identical on 2.2.3 and 3.0.2. Call it once per column in
`clean_column`, not inside each helper — it costs a scan.

### 2. Pin development to the cleanroom versions

A CI job at exactly Python 3.10 / pandas 2.2.3 / numpy 2.2.0. Everything passes
there today, but that is currently luck, not coverage.

### 3. Ship the tests

`pytest` is available in the cleanroom. Put the tests inside the package, or
upload the sdist alongside the wheel, so `pytest --pyargs datkit` can run
against the real pandas, the real memory ceiling and the real data.

### 4. Merge the drafts

`binning_v2.py` → `src/datkit/binning.py`, with tests. Then the cleaning
additions, checking first whether `test_cleaning.py` asserts the record's exact
column set — widening it will break that.

### 5. Recommend Arrow strings in the target

pandas 2.2.3 defaults to object strings; pyarrow is installed. Measured on 2M
rows over 200 distinct values: object 131 MB, Arrow 33 MB, category 4 MB. At
90M rows that is roughly 5.9 GB / 1.5 GB / 180 MB. One setting,
`pd.options.future.infer_string = True`, for a 4x saving — and the profiler
already identifies which columns would benefit from `category` on top.

### 6. Then: the relationship checks

Stage 2 of the EDA work — key uniqueness, referential coverage, nulls in keys.
Prototype in **DuckDB** before writing it in pandas: it queries a pandas frame
in place with no copy, spills to disk under memory pressure, and each check is
one SQL statement. Decide there rather than writing it twice.

---

## Open questions

- `high_card_ratio=0.5` and `tail_min_span=0.05` are untested against real
  columns. Both separate the cases I have by a wide margin, but neither default
  is earned yet.
- `width` now means three things depending on the row — a span for numeric
  bins, days for calendar periods, a count of values for categorical ones.
  It reads well; decide whether that overloading is acceptable before merging.
- Read `visions 0.7.6` before finalising `dtype_effective` — it is the type
  system under ydata-profiling and may already have this vocabulary.
- Run `ydata-profiling` once on something real and large, so "why not just use
  that" has a measured answer rather than an asserted one.
- `plotly` is installed but normally pulls plotly.js from a CDN. Test it in the
  cleanroom before the viz module depends on it; matplotlib and seaborn are safe.
