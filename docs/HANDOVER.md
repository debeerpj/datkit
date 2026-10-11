# datkit — handover

Where things stand. Updated 2026-10-11.

`CLAUDE.md` at the repo root carries how I work and the target environment.
`docs/SETUP.md` carries the full build log. This file is state and next steps.

---

## Current state

**In the repo, working:** `cleaning.py`, `schema.py`, `binning.py`. 229 tests,
ruff clean, all passing on Python 3.10 / pandas 2.2.3 — the cleanroom versions
— as well as locally.

**Shipped shape:** tests live at `src/datkit/tests/`, so they travel inside the
wheel and run in the cleanroom with `pytest --pyargs datkit`. The wheel is
~50 KB and installs there offline with nothing fetched and nothing upgraded.

**`Claude outputs/` is drafts only.** Nothing there is on the import path.
`binning_v2.py` and `cleaning_changes.py` have both been merged and can go.

---

## What changed since the last handover

### The `is_string_dtype` guard became `_is_text`

`is_string_dtype` returns True for **any** object column, and object is still
the default for text on pandas 2.2.3. That let an object column of ints reach
`.str` (AttributeError) and an object column of mixed types lose its
non-string values to coercion. Neither shows up on pandas 3 locally.

`_is_text(s)` wraps `pd.api.types.infer_dtype(s, skipna=True) == "string"`,
which inspects the values rather than the container. It costs one scan, so
callers invoke it **once per column**, not once per step. It lives in one
place because there are now three callers — `clean_column`,
`clean_column_dtype`, `discover_specials` — and an earlier version of this
guard drifted apart across them, which is how the bug survived.

### Sentinel discovery merged

`discover_specials` proposes the few values blocking a conversion; it never
acts on them, and discovered sentinels must **not** join `to_null`'s token
list, because nulling them destroys the information. Two guards, both needed:
at most `MAX_SPECIALS` candidates, and they must be a minority of the
column's distinct values. Without the second, a six-row column of names
proposes all six names.

Record gained `dtype_effective`, `specials`, `specials_count`, `n_unique`.

### Convertibility — contamination, which is a different question

`discover_specials` thresholds on **distinct values** and rejects a column
with too many. `convertibility` thresholds on **row share**. They catch
different things:

|                   | few distinct values | many distinct values |
|-------------------|---------------------|----------------------|
| **few rows**      | rare sentinel       | **contamination**    |
| **many rows**     | common sentinel     | genuinely text       |

IMDb's `runtimeMinutes` is the case in point: 32 bad rows in 12.7M, across 13
distinct values. Over the sentinel cap, far under any row threshold.

**It cannot be sampled.** 32 rows in 12.7M against a 100k sample is a quarter
of an expected hit — missed ~78% of the time. The failure count is one full
pass over the column, deliberately.

Record gained `unconvertible_verdict` (clean / contaminated / text),
`n_unconvertible`, `pct_unconvertible`, `unconvertible_values`,
`n_placeholder`, `n_infinite`.

- The percentage is over **convertible rows** — total less nulls less
  placeholders. A column that is 64% `\N` should not read as 64% clean.
- `n_placeholder` is counted apart from failures: by coercion time those are
  already null and would be invisible.
- `n_infinite` is the opposite failure — `inf` coerces *cleanly* and then
  poisons every mean downstream.
- Verdict `text` zeroes the failure fields. Reporting every title in
  `primaryTitle` as unconvertible is true and useless; it buries the one
  column with 32 bad rows. `CONTAMINATION_MAX_PCT = 50.0`, shared with
  `find_unconvertible` so the two views cannot disagree. Pass `max_pct=100`
  to see raw counts for such a column.

`find_unconvertible(df, max_pct=5.0)` returns the offending rows themselves —
index, column, value — capped at `max_rows` regardless of input size. Run it
on the **raw** frame, before cleaning; afterwards a converted column has
nothing left to find.

### Category profiling

`category_profile(s)` proposes, `to_category(s)` applies, `clean_dataframe`
reports but does **not** convert — `categorize=True` opts in. Record gained
`as_category`, `n_categories`, `category_ratio`, `category_saving`.

Thresholds are measured, not guessed. On 2M rows of 20-character strings:

```
n_unique   ratio    object → category   saving
      10  0.00001    146.9 →   1.9 MB    77x
   1,000  0.0005     146.9 →   3.9 MB    37x
  50,000  0.025      146.9 →  12.3 MB    12x
 400,000  0.20       146.9 →  44.9 MB     3.3x
1,000,000 0.50       146.9 → 103.4 MB     1.4x
```

So memory almost never says no, and speed never does — category was faster at
everything measured, 24x on `sort_values`. The real limits are behavioural:
concat of two categoricals with different categories silently reverts to
object; assigning an unseen value raises TypeError; `groupby(observed=False)`
emits a row per category whether or not it occurs. Hence
`CATEGORY_MAX_RATIO = 0.05` and `CATEGORY_MAX_UNIQUE = 50_000`, where the
saving is still ~12x and the categories array stays small.

### The sampling asymmetry — the idea both two-stage checks rest on

A sample can only ever **under**-count distinct values. So "the sample looks
varied" proves the column is varied (a sound rejection), while "the sample
looks repetitive" proves nothing. That asymmetry is what makes a sample a
valid *filter* and an invalid *decision*.

Applied twice:

- `best_target` picks numeric-or-datetime from a 10k sample (~0.1s flat,
  whatever the column), because that is a majority question. The failure
  count that follows is not sampled, because it is a rare-event question.
- `category_profile` rejects varied columns from a sample without a full
  count, then counts exactly. A category occurring twice in 90M rows is
  invisible to any sample and must still become a category.

Two guards the sampling needs: oversample 5x (nulls and placeholders do not
help the decision) and fall back to the full column below 100 real values.

**Trap worth remembering:** sampling saves nothing if the expensive step
happens before the sample. The first version of `best_target` ran
`remove_whitespace` and `to_null` over all 12.7M rows and *then* sampled —
two column shapes got slower, not faster.

### The Arrow NaN bug — read this before writing any new mask

Under Arrow, a failed coercion produces **NaN, not null**. NaN is a valid
double distinct from null, so `isna()` reports it as *present*:

```
                 converted dtype    isna() sees   failures found
numpy object     float64                      1                1
string[pyarrow]  double[pyarrow]              0                0   <- silent
```

A column read with `dtype_backend="pyarrow"` reported **zero** unconvertible
values on data that had 32. `missing_count`'s docstring has warned about this
since the beginning and three new functions walked into it anyway.

`missing_mask(s)` is now the one implementation; `missing_count` delegates to
it. **Anything asking "which rows did this coercion lose" goes through
`missing_mask`, never `isna()`.**

Why the suite missed it: every convertibility test used `dtype=object`.
`conftest.py` has a `text_dtype` fixture across object / `string` /
`string[pyarrow]` for exactly this, and none of the new tests used it. There
are now 7 backend-parametrised regression tests, verified to fail against the
old code.

**Rule going forward: any new function that inspects missing-ness gets the
`text_dtype` fixture from the start, not after a bug.**

---

## Next, in order

### 1. `binning.py` has the Arrow bug and no tests — this is a live bug

It was merged into `src/datkit/` and ships in the wheel, but the test suite
does not touch it. Measured:

```
                 rows  total count  unconvertible row
object              4          100  present, count=25
string[pyarrow]     3           75  MISSING
```

On an Arrow column 25 rows silently vanish and the counts no longer reconcile
to the row total — the one property that module's docstring promises. Since
the default read is now `dtype_backend="pyarrow"`, this is the common path.

- line 393: `converted.isna() & work.notna()` → `missing_mask`
- line 354: `s.isna().to_numpy(dtype=bool)` → same check
- lines 133 / 149 / 314: `is_string_dtype` where it is judging values → `_is_text`

Then tests, across the three backends. Expect 25–30.

### 2. Relationship checks — stage 2 of the EDA work

Key uniqueness, referential coverage, nulls in keys. Prototype in **DuckDB**
before writing it in pandas: it queries a pandas frame in place with no copy,
spills to disk under memory pressure, and each check is one SQL statement.
`duckdb==1.1.1` is pinned in the dev group to match the cleanroom. Decide
there rather than writing it twice.

`devdata/imdb_schema.yaml` already describes two real relationships, one of
them a composite key (`title_akas` on `titleId, ordering`).

### 3. A second dev dataset — IMDb has no dates and nothing transactional

Checked: the only temporal columns in the whole IMDb set are bare years
(`startYear`, `endYear`, `birthYear`, `deathYear`). All 11.3M non-null
`startYear` values are exactly four digits. There is no day, month or
timestamp anywhere, and every table is a snapshot of entities — `title.ratings`
is already aggregated, with no votes table underneath.

Consequences:

- `clean_dates` — the most intricate function in the repo — has **never met a
  real date column**. Every test of it runs on hand-built fixtures.
- the `datetime` branch of `convertibility` / `best_target` is only exercised
  by fixtures.
- no data for any aggregation-over-time work.

NYC TLC trip records are the obvious candidate: a real fact table, event
timestamps, foreign keys into a zone lookup, `fhvhv` large enough to press the
memory ceiling. One catch — the official files are **Parquet, so the datetimes
arrive already typed** and `clean_dates` still never runs. For text dates use
the NYC Open Data CSV export instead; Socrata renders timestamps as
`MM/DD/YYYY HH:MM:SS AM/PM`, which is month-first *and* 12-hour, exactly the
ambiguity the `dayfirst` logic and the title-casing retry exist for.

(Unverified: I could not download a file to confirm either claim — the Parquet
typing is from the format spec, the CSV format from how Socrata generally
behaves. One download settles both.)

### 4. `datkit.__version__` does not exist

`src/datkit/__init__.py` exports nothing but a docstring, so inside a running
kernel there is no way to tell which version is loaded. Two lines, and it
turns "did my upload take?" into a check rather than a guess.

### 5. Then: redaction, and PII detection before it

Per the vision section in `docs/SETUP.md`: a redaction level is required
before any profile leaves the room, and PII detection belongs *before*
redaction.

---

## Reading the data correctly

IMDb TSVs must be read with `quoting=csv.QUOTE_NONE`. Pandas treats `"` as a
quote character by default, IMDb does not quote its fields, and 35,409 titles
contain a quote character. Those rows swallow the following tabs and every
field shifts left:

```
default quoting    runtimeMinutes: 32 bad rows, 13 distinct (Talk-Show, Reality-TV, ...)
quoting=QUOTE_NONE runtimeMinutes: 0
```

Row counts are identical either way, so nothing warns you. The genre strings
appearing in `runtimeMinutes` were the `genres` column sliding left.

---

## Open questions

- `high_card_ratio=0.5` and `tail_min_span=0.05` in `binning.py` are still
  untested against real columns. Both separate the cases I have by a wide
  margin, but neither default is earned yet.
- `width` in `binning.py` means three things depending on the row — a span for
  numeric bins, days for calendar periods, a count of values for categorical
  ones. It reads well; decide whether the overloading is acceptable.
- `as_category` and `dtype_effective` can disagree usefully: `runtimeMinutes`
  profiles as a good category candidate *and* as numeric-once-cleaned. A column
  that should be an int is not improved by becoming a category of int-strings.
  Decide which verdict wins when they conflict.
- `CATEGORY_MAX_RATIO = 0.05` is a share, so above roughly 1M rows the
  absolute cap is always the binding constraint and tuning the ratio appears
  to do nothing. Intended, but surprising.
- The majority short-circuit in `best_target` is a heuristic, not a proof. It
  is wrong in principle for a column that is 49% numeric and 51% dates.
- Read `visions 0.7.6` before finalising `dtype_effective`? **Answered: no.**
  Installed and tested it — `["1","2","-999","N/A"]` comes back as `String`,
  so it does not solve the sentinel problem at all. `CompleteSet` will not
  even import in the cleanroom (needs `imagehash`, not in the 174 packages),
  leaving `StandardSet`'s ten types. It also parses the whole column via
  dateutil to test its DateTime relation — 5x slower than sample-then-format,
  and the wrong shape for a memory ceiling. Worth an hour for the *model* —
  types as a graph with relations between them, so "String, but Integer under
  this transformation" is an edge rather than a special case — not as a
  dependency.
- Run `ydata-profiling` once on something real and large, so "why not just use
  that" has a measured answer rather than an asserted one.
- `plotly` is installed but normally pulls plotly.js from a CDN. Test it in the
  cleanroom before the viz module depends on it; matplotlib and seaborn are
  safe.

---

## Shipping to the cleanroom

```powershell
Remove-Item dist\*.whl, dist\*.tar.gz   # a stale wheel has the same filename
uv build --wheel                        # expect ~50 KB, not 3 KB
```

Then in the cleanroom:

```python
!pip install --no-deps datkit-0.1.0-py3-none-any.whl
# restart the kernel — the old module stays in memory
!pytest --pyargs datkit -q              # expect 229 passed
```

`--no-deps` on purpose: pip there points at an internal mirror and may
otherwise try to resolve pandas. Verified that a full resolve is a no-op now
the dependency bounds sit at or below the cleanroom versions, so nothing is
skipped by using it.

Dependency lower bounds must stay **at or below** the cleanroom versions
(`pandas>=2.2.3`, `numpy>=2.2.0`, `pydantic>=2.9.2`, `pyyaml>=6.0.2`). When
they did not, the wheel was literally uninstallable there — "no solution
found". The `cleanroom` CI job guards this by resolving with
`--resolution lowest-direct`, which reproduces the target exactly *because*
the bounds are the floor. Raise a bound past the cleanroom and that job goes
red immediately.
