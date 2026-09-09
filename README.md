# datkit

A Python toolkit for the first phase of a data analysis: cleaning columns
safely, declaring what a schema is supposed to look like, and checking that the
data agrees.

Built for **restricted analysis environments** — no internet access, a fixed
memory ceiling, and no development tooling installed. That constraint drives
every design decision here, and is what makes the project interesting to read.

Status: **early**. The cleaning and schema modules work and are tested;
profiling and relationship checks are next.

---

## The problem

Analysis in a locked-down environment looks different from analysis on a laptop:

- **Nothing can be installed from the internet.** Any tooling has to arrive as a
  file and import with only what is already present.
- **Memory is capped**, and the data can be larger than the cap. An operation
  that quietly allocates a copy kills the kernel.
- **The database catalog gives table and column names and types, and nothing
  else.** Join keys, cardinality and expectations come from asking people.
- **The same work repeats on every engagement** — profile the tables, work out
  how they join, find out where the data disagrees with what you were told.

`datkit` is the part of that work that can be written once.

---

## Approach

### Ship as a pure-Python wheel

`uv build` produces a `py3-none-any` wheel that installs with `pip` from a local
file, or — if `pip` is unavailable — can be placed directly on `sys.path`, since
a wheel is a zip and `zipimport` is in the standard library:

```python
import sys

sys.path.insert(0, "datkit-0.1.0-py3-none-any.whl")
import datkit
```

Runtime dependencies are deliberately few and loosely pinned. Every version
bound is a way for an install to fail in an environment whose package index you
do not control.

### Declare expectations, then test the data against them

A schema is a YAML file: tables, columns, primary keys, and the relationships
between them with their expected cardinality and coverage. The generated half
comes from the database catalog; the asserted half is filled in by hand.

```yaml
tables:
  - name: title_akas
    source: dbo.title_akas
    columns:
      - {name: titleId, type: varchar}
      - {name: ordering, type: int}
    primary_key: [titleId, ordering]

relationships:
  - name: akas_to_basics
    from_table: title_akas
    from_columns: [titleId]
    to_table: title_basics
    to_columns: [tconst]
    cardinality: many_to_one
    coverage: full
```

Validation is split in two, because the failures have different audiences:

| | Checks | On failure |
|---|---|---|
| **Schema validation** | Is the YAML internally coherent? | Raises, listing every problem at once |
| **Data validation** | Does the data match what was declared? | Reports — it is a finding, not an error |

The first needs no database connection, so a typo is caught while editing rather
than after waiting on a query.

### Never lose data silently

Every lossy conversion compares the count of missing values before and after,
and **returns the original column if any were lost**. A conversion that half
works is worse than one that does not run, because the loss is invisible
downstream.

`clean_dataframe` returns the cleaned frame *and* a record: dtype before and
after, bytes before and after, nulls before and after, per column. The audit
trail is part of the output, not a log line.

### Treat memory as a design constraint

The rule the module is built on: **operations that materialise a Python object
per row are expensive; type conversions and masks are cheap.** So the functions
work one column at a time and keep only scalars, never calling a frame-wide
method that would allocate a copy of the whole table.

---

## What is here

```
src/datkit/
  cleaning.py     column cleaning, the pipeline, and the change record
  schema.py       pydantic models, YAML loading, schema validation
tests/            pytest, parametrised across string backends
docs/SETUP.md     how this was built, and what was learned doing it
```

| Module | Status |
|---|---|
| `cleaning` | Working — whitespace, placeholder nulls, numeric and date coercion, per-column change record |
| `schema` | Working — models, YAML loader, cross-object validation |
| Table profiling | Next |
| Relationship checks | Planned — key uniqueness, referential coverage |
| Joined-table profiling and charts | Planned |

---

## Testing

```powershell
uv sync
uv run pytest
uv run ruff check .
```

Two things worth noting about the approach:

**Tests run against every string backend.** A fixture parametrises over `object`,
pandas `StringDtype`, and Arrow-backed strings, so each test runs three times.
The target environment decides which backend the data arrives in, and that
decision is not ours — the differences have to be found here rather than there.

That is not theoretical. Arrow treats `NaN` as a valid value distinct from null,
while numpy treats it as missing. A guard written as `len(s) - s.count()` passes
under Arrow, lets a failed numeric coercion through, and turns every text column
in the frame into `NaN` — while the memory report *improves*, because 8 bytes of
`NaN` is smaller than the text it replaced.

**Development data is separate from test data.** Synthetic fixtures in `tests/`
pin down behaviour already decided on; a real dataset, downloaded and
git-ignored, finds behaviour not yet thought of. When real data breaks
something, the fix is a new synthetic case, never a dependency on the download.

---

## Notes

`docs/SETUP.md` records the build in more detail than a README should — the
decisions, the alternatives rejected, and the specific things that went wrong.
It is written as a reference to return to, so it is longer and blunter than this
page.

Requires Python 3.10+. Built with [uv](https://docs.astral.sh/uv/),
[ruff](https://docs.astral.sh/ruff/), [pytest](https://docs.pytest.org/) and
[pydantic](https://docs.pydantic.dev/).

## Licence

MIT — see [LICENSE](LICENSE).
