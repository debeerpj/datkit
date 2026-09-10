# datkit — handover

Context for continuing this project in a new conversation. Written 2026-09-10.

---

## Read these first

| File | What it holds |
|---|---|
| `docs/SETUP.md` | The full build log — every decision, the alternatives rejected, and the specific things that went wrong. Long, deliberately. |
| `README.md` | The public framing: purpose, approach, current state. |
| This file | Where things stand and what to do next. |

---

## How I want to work

- **One step at a time.** Do not batch several changes into one message.
- **Explain new concepts** the first time they appear.
- **Guide, do not implement.** When I am building something to learn it, do not
  write the method for me — tell me what it should do, which functions to use,
  and which gotchas to watch for. I will write it. I will ask explicitly when I
  want code, and I do ask (tests especially).
- **Keep responses short.** No long write-ups at every step.
- **Comment everything, concisely.** Prefer explicit loops over comprehensions
  where the loop does real work.
- **Do not change the repo without asking first.** `docs/SETUP.md` is the
  exception — keep it up to date as we go, unprompted.

---

## The project

A Python toolkit that speeds up the first phase of an analysis: cleaning
columns, declaring a schema, and checking the data agrees with it.

**The environment it ships into** drives every decision:

- No internet install. It arrives as a file and imports with what is already
  present.
- Fixed memory ceiling; data can exceed it.
- The SQL catalog gives table names, column names and types — nothing else. Join
  keys and expectations come from asking the client.
- Storage persists between sessions; only compute is rebuilt at each login.
- Data arrives via a magic command that pulls from SQL Server.

Development and testing happen on my Windows machine (16 GB, so memory problems
show up locally); the analysis happens over there.

Repo: `github.com/debeerpj/datkit` — public, MIT.

---

## Current state

**Working and tested — 83 tests passing, ruff clean.**

```
src/datkit/
  cleaning.py     column cleaning, the pipeline, the change record
  schema.py       pydantic models, YAML loader, schema validation
tests/
  conftest.py     text_dtype fixture — object / string / ArrowDtype
  test_cleaning.py
  test_schema.py
docs/SETUP.md
notebooks/        exploration, git-ignored
devdata/          IMDb subset, git-ignored, downloaded by hand
```

### `cleaning.py`

| Function | Does |
|---|---|
| `sample_records(df, n, seed)` | Seeded sample, returns whole frame if smaller |
| `missing_count(s)` | Portable null+NaN count — see the Arrow note below |
| `remove_whitespace(s)` | `.str.strip()` on text |
| `to_null(s, tokens=...)` | Placeholder tokens → NaN, via `mask(isin())` |
| `clean_numbers(s)` | Strips `$ R % ,`, converts to numeric, refuses on loss |
| `clean_dates(s)` | Samples, tries both `dayfirst` values, converts, refuses on loss |
| `uppercase(s)` | Opt-in — **not** in the default pipeline |
| `clean_column(s)` | whitespace → null → numbers → dates |
| `clean_dataframe(df, inplace=False)` | Per column, returns `(frame, record)` |

The record has one row per column: `dtype_before/after`, `converted`,
`bytes_before/after/saved`, `mb_*`, `nulls_before/after/created`.

### `schema.py`

pydantic models `Column`, `Table`, `Relationship`, `Schema`;
`load_schema_from_yaml(path)` which does **not** validate; and
`check_schema(schema)` which collects every problem and raises once.

---

## Decisions already made — do not relitigate

| Decision | Why |
|---|---|
| pydantic over dataclasses for the schema | Removes the hand-written `from_dict` layer; validates nested YAML automatically. Confined to `schema.py` so a fallback means rewriting one module. |
| YAML over JSON/TOML | Comments, for recording *why* an assumption was made. `tomllib` is stdlib only from 3.11. |
| Loader does not validate | A broken schema still loads so it can be inspected — the normal case when hand-editing. |
| Named `check_schema`, not `validate` | pydantic v2's `BaseModel` still carries a deprecated `validate` classmethod. |
| Relationship endpoints are `list[str]` | They *reference* existing columns. Using `Column` objects means `"x" in table.columns` compares a string to objects and is always false. |
| `uppercase` is opt-in | Lossy on titles and names, and the step most likely to exhaust memory. |
| pyarrow is a **dev** dependency | Large compiled wheel; requiring it is another way for the install to fail. Detect at runtime, fall back. |
| Python pinned to 3.10 | Still unconfirmed against the target. Raising later is free; lowering is not. |
| `line-length = 120`, `ignore = ["D401", "SIM108"]` | A rule suppressed every time is noise. |

---

## Gotchas that cost real time

**Arrow and numpy disagree about NaN.** Under numpy NaN is missing; under Arrow
it is a valid double distinct from null, so `isna()` returns False and `count()`
counts it as present. A guard written `len(s) - s.count()` therefore passes under
Arrow, lets a failed coercion through, and turns every text column into NaN —
while the memory report *improves*, because 8 bytes of NaN is smaller than the
text. `missing_count()` exists for this. Use it anywhere a null count feeds a
decision.

**Dtype checks must not compare to literal names.** `s.dtype in ["object",
"string"]` is False for `ArrowDtype`. Use `pd.api.types.is_string_dtype` etc.
But note `is_string_dtype` is **False for a categorical**, even with string
categories — that needs its own branch.

**`pd.api.types.is_category_dtype` was removed in pandas 3.** Use
`isinstance(s.dtype, pd.CategoricalDtype)`.

**`bool(pd.NaT)` is `True`.** Never test a parsed date for truthiness.

**`and`/`or` on a Series raises.** Use `&`/`|` then `.any()`/`.all()`.

**Notebook staleness.** `%autoreload 2` does not reliably pick up a newly added
top-level function. When results look wrong after an edit, restart the kernel
before debugging. Cell outputs also persist until re-executed — I have twice
diagnosed a "bug" that was a stale output.

---

## Known gaps

- **Categorical columns are not handled.** Deliberately deferred — data arriving
  with categories already defined is unlikely. `tests/test_cleaning.py` carries a
  comment with what a future implementation needs to know.
- **`20240101` is read as a number, not a date.** `clean_numbers` runs first and
  claims anything numeric-looking. Note this ordering is load-bearing in the
  other direction: it is what stops a year column (`"2019"`) becoming January
  dates.
- **One unparseable value vetoes a whole column.** Strict by design, but on 90M
  real rows a single malformed value will reject a genuine date column. Wants a
  tolerance parameter.
- **`nulls_created` conflates two things** — placeholders deliberately nulled and
  values lost to coercion. It was meant to be a "should always be zero" alarm.
- **`clean_dates` parses the full column twice**, once per `dayfirst` value.
  Decide from the sample, then parse once.
- **Ambiguous dates resolve by accident.** `03/04/2025` returns 3 April only
  because `dayfirst=True` is tried first. If both parse and disagree, that is the
  ambiguous case and should be reported.

---

## Next: the EDA modules

Three stages, in order:

1. **Table profiling** — rows, nulls, uniques, distributions per column
2. **Relationship checks** — key uniqueness, referential coverage, nulls in keys
3. **Joined-table profiling** — crosstabs and comparisons across a join

### The output-shape decision, unresolved

All three should return the **same shaped record**, so anything written to
filter or display works across all of them. Sketched but not built:

```python
Finding(
    stage,  # "table" | "relationship" | "join"
    subject,  # table name or relationship name
    check,  # "row_count", "coverage", "distribution", ...
    status,  # "ok" | "warn" | "fail" | "info"
    summary,  # one human line
    metrics,  # dict of numbers
    sample,  # bounded rows, never all
    data,  # small DataFrame for charting
    chart,  # hint: "line" | "bar" | "hist" | "heatmap"
)
```

with a `Report` wrapping `list[Finding]` (`.failed()`, `.for_table()`,
`.to_frame()`, `_repr_html_`).

**I found this too abstract when we discussed it.** Build one concrete function
first — `profile_table` returning a plain DataFrame — and let the shape emerge
from what is actually useful. The `clean_dataframe` record is already close to
half of it.

**A dataclass, not pydantic, for findings** — they are produced internally, so
there is nothing to validate, and a DataFrame field needs
`arbitrary_types_allowed` ceremony.

**Charts stay out of the results.** `data` holds the numbers; a separate
`plot(finding)` renders. Keeps matplotlib out of the import path and findings
comparable.

### Memory rules for the checks

- **Counts plus a bounded sample, never the offending rows.** Output stays fixed
  no matter the input.
- **`isin`, not set arithmetic**, for coverage — hashes only the parent side.
- **Composite keys need `MultiIndex.isin`.** Concatenating key columns into a
  string breaks when a value contains the separator.
- **Never a frame-wide method.** `df.isna()` allocates a boolean frame the full
  shape of the data.

### Deferred idea

Mapping columns to a controlled vocabulary of semantic roles — transaction date,
status, amount — so the EDA can suggest relevant checks per role.

---

## Working commands

```powershell
uv sync
uv run pytest -q
uv run ruff format . ; uv run ruff check --fix .
uv build
```

Branch protection is on `main`: branch **before** editing, then
`gh pr create --fill`, then `gh pr merge --squash --delete-branch`.

Git identity is the GitHub noreply address — do not let a real one back in.
