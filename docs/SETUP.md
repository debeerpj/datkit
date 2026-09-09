# Building a portable data-analysis toolkit

A reference for setting up a Python library that is developed and tested locally,
then shipped as a wheel into a locked-down JupyterHub environment where no
packages can be installed from the internet.

**Target:** a pure-Python package, built to a `py3-none-any` wheel, imported in
notebooks either via `pip install --user` or directly off `sys.path`.

**Toolchain:** `uv` (environments + build), `ruff` (linting/formatting),
`pytest` (tests), `git` (version control).

**Platform:** Windows / PowerShell.

---

## Design constraints

These shape every decision below.

| Constraint | Consequence |
|---|---|
| No internet install in the target environment | The toolkit may only import libraries already present there |
| Compute torn down each login, storage persisted | The wheel lives on persisted storage; a re-install per session is cheap if needed |
| Files uploaded via web UI | Prefer a single-file artifact over a folder of many files |
| Target Python version unknown at start | Pin low (3.10); raising the pin later is free, lowering it is not |

`uv`, `ruff` and `pytest` are **development-time tools only**. They never travel
to the target environment. Only pure-Python source crosses over.

---

## Step 0 — Install git

Version control. Required before anything else so that every later step is
recoverable.

```powershell
winget install --id Git.Git -e
```

Verify:

```powershell
git --version
```

Set your identity once per machine (this is stamped into every commit):

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

---

## Step 1 — Install uv

`uv` is a single binary that replaces `pip`, `venv`, `pyenv` and `pip-tools`.
It resolves dependencies, creates virtual environments, downloads Python
interpreters, and builds wheels.

```powershell
winget install --id astral-sh.uv -e
```

Alternative (official installer script):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```powershell
uv --version
```

### Why uv rather than pip + venv

| | pip + venv | uv |
|---|---|---|
| Environment creation | manual, per-project | automatic, implicit |
| Lockfile | needs pip-tools | built in (`uv.lock`) |
| Python version management | needs pyenv | built in (`.python-version`) |
| Wheel building | needs `build` package | built in (`uv build`) |
| Speed | baseline | 10–100x faster resolution |

---

## Step 2 — Initialise the project

Run in the (already created) repo folder:

```powershell
uv init --lib --python 3.10
```

### What the options mean

| Option | Meaning |
|---|---|
| `--lib` | Create a **library** — code meant to be imported. Produces a `src/` layout and a `[build-system]` block in `pyproject.toml`, which is what makes `uv build` able to produce a wheel. |
| `--app` (default) | Create an **application** — code meant to be run. No build system, so it cannot be packaged. Wrong choice here. |
| `--package` | An application that is *also* packaged, with a console entry point. Overkill for an importable toolkit. |
| `--python 3.10` | Pin the interpreter. Written to `.python-version` and `requires-python` in `pyproject.toml`. |
| `--name mytoolkit` | Override the package name, which otherwise derives from the folder name (lowercased, dashes → underscores). This is the string you will type as `import <name>`. |

### Why pin the Python version low

`requires-python` is baked into the wheel metadata, and **pip refuses to install
a wheel whose `requires-python` excludes the local interpreter**. Ship `>=3.11`
to a 3.9 host and it is rejected before your code is ever imported.

Separately, source syntax is version-sensitive even though a `py3-none-any`
wheel is not: `match` statements and `int | None` annotations require 3.10+.
Code written against a newer interpreter fails to parse on an older one, with no
local warning.

Raising the pin later costs nothing. Lowering it means auditing syntax you have
already written. So pin low until the target version is confirmed.

### Changing the pin later

Three places record the version and must agree:

```
.python-version      # which interpreter uv installs into .venv
pyproject.toml       # requires-python = ">=3.10"
.venv/               # the actual environment
```

Edit the first two, then:

```powershell
uv sync
```

uv fetches the correct interpreter, rebuilds `.venv`, and reinstalls from the
lockfile. Source code is untouched.

### Resulting layout

```
pyproject.toml          # metadata, dependencies, build config, tool config
README.md
.python-version         # interpreter pin
.gitignore
uv.lock                 # exact resolved versions (created on first sync)
src/
  mytoolkit/
    __init__.py
```

**Check before continuing:** that the package name under `src/` is the name you
want to type as `import <name>`, and that the folder is a git repository
(`uv init` runs `git init` for you if it was not already).

---

## Step 3 — Documentation location

This document lives at `docs/SETUP.md`, inside the repo.

| Option | Verdict |
|---|---|
| `docs/` inside the repo | **Chosen.** Version-controlled alongside the code, so it cannot drift out of sync. A "how this was built" guide is an asset in a public repo. |
| A separate folder outside the repo | Becomes the thing you cannot find in six months. Not tracked, so it silently goes stale. |

Keep environment-specific detail **out** of this file — the target hub's name,
internal package index URLs, its installed-library inventory. Those belong in a
`notes/` folder listed in `.gitignore`: private, but nearby.

---

## Step 4 — Add the dev tooling

```powershell
uv add --dev ruff pytest
```

### What `--dev` means

For a library there is no "production server". The useful model is:
**`[project] dependencies` are what your *users* need; `dev` is what *you* need.**
Here, the user is the cleanroom notebook.

| Section | Travels in the wheel? | Read by | Put here |
|---|---|---|---|
| `[project] dependencies` | **Yes** — written into wheel metadata | `pip install` on the target machine, which will try to resolve and install each one | Anything your modules `import` at runtime (e.g. pandas) |
| `[dependency-groups] dev` | **No** — lives only in `pyproject.toml` and `uv.lock` | `uv sync` on your machine only | Anything used to build, lint or test (ruff, pytest) |

Consequences for a restricted target environment:

- Nothing in `dev` can ever break an install. The target never learns it exists.
- Anything in `[project] dependencies` **must** be available on the target's
  internal package index, or `pip install` fails outright.

**The failure mode to watch for:** a runtime import declared in `dev` by mistake.
Everything works locally, the wheel builds cleanly, and it breaks on `import` in
the target environment — because the wheel never declared the dependency and you
were unknowingly relying on it being present. Step 7 exists to catch this.

### Side effects of this command

Running `uv add` also:

- creates `.venv/` (the project's virtual environment)
- creates `uv.lock` (the exact resolved version of every package, so a rebuild
  months later produces an identical environment)
- installs your own project into `.venv` in **editable mode** — the source is
  linked rather than copied, so edits take effect without reinstalling

That last point makes this a useful check as well as an install: if the build
backend cannot locate your package under `src/`, it fails here rather than
silently at build time several steps later.

---

## Naming: distribution name vs import name

These are two different names and are routinely confused.

| | Example | Set in | Used by |
|---|---|---|---|
| Distribution name | `datkit` | `[project] name` in `pyproject.toml` | `pip install`, the wheel filename |
| Import name | `datkit` | the folder name under `src/` | `import datkit` |

Python identifiers cannot contain hyphens, so the packaging convention is
hyphens on the outside, underscores on the inside — e.g. distribution
`data-analysis-toolkit` imports as `data_analysis_toolkit`.

The **build backend** locates your source by convention: it takes the project
name, swaps hyphens for underscores, and looks for that folder under `src/`. If
the two names do not correspond, the build fails.

Current uv versions configure their own backend, `uv_build`:

```toml
[build-system]
requires = ["uv_build>=0.12.5,<0.13.0"]
build-backend = "uv_build"
```

Older versions used **hatchling**. Check `[build-system]` in `pyproject.toml`
rather than assuming — the override syntax for a deliberate name mismatch
differs between them (`[tool.uv.build-backend]` vs
`[tool.hatch.build.targets.wheel]`).

Keeping the two names identical avoids needing an override at all. The repo
folder name on disk is irrelevant to both.

### Choosing the import name

Pick a short one. It is typed at the top of every notebook that uses the toolkit,
and it is cheap to change now and a nuisance once notebooks reference it.
Renaming takes two edits — the folder under `src/`, and `[project] name`.

### `py.typed`

`uv init --lib` creates an empty `py.typed` file in the package. Defined by
PEP 561, its presence tells type checkers that the package's annotations are
real and should be trusted. Without it, mypy and Pylance ignore your type hints
when the package is imported elsewhere, and notebook users get no autocomplete
on your functions. Empty file, real effect — leave it.

---

## Step 5 — Configure ruff

`ruff` is a linter **and** a formatter in one binary. It replaces what used to be
four separate tools: flake8 (linting), isort (import ordering), black
(formatting) and pyupgrade (syntax modernisation).

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "UP", "SIM", "N", "D"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"
```

### What the rule sets mean

`select` switches on families of checks, each identified by a prefix.

| Code | Family | Catches |
|---|---|---|
| `E`, `W` | pycodestyle | Layout and whitespace deviations from PEP 8 |
| `F` | pyflakes | **Real errors** — undefined names, unused imports, unused variables |
| `I` | isort | Import ordering and grouping (stdlib / third-party / local) |
| `B` | flake8-bugbear | Likely bugs — mutable default arguments, loop variable capture |
| `UP` | pyupgrade | Syntax that could be modernised for the target Python version |
| `SIM` | flake8-simplify | Needlessly convoluted constructs |
| `N` | pep8-naming | Naming conventions (`CamelCase` classes, `snake_case` functions) |
| `D` | pydocstyle | Missing or malformed docstrings |

`F` is the one that catches genuine mistakes; the rest are consistency. `D` is
the most opinionated — worth keeping for a library other people import, since
the docstring is the only documentation a notebook user gets from `help()` and
tooltips.

### Why `target-version` matters here

`UP` rewrites code to the newest syntax the target version supports. Set
`target-version = "py310"` and ruff will never suggest a 3.11+ construct — so the
linter actively defends the compatibility pin rather than quietly undermining it.
**Keep this value in step with `requires-python`.**

### `convention = "numpy"`

Selects the docstring style pydocstyle enforces. NumPy style is the norm across
the scientific Python stack (numpy, scipy, pandas, scikit-learn), so it reads as
native to anyone working in a data notebook:

```python
def clean_columns(df, lowercase=True):
    """Normalise DataFrame column names.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame whose columns are renamed.
    lowercase : bool, default True
        Whether to lowercase names in addition to stripping whitespace.

    Returns
    -------
    pandas.DataFrame
        A new frame; the input is not modified.
    """
```

Alternatives are `"google"` (more prose-like) and `"pep257"` (minimal).

### Running it

```powershell
uv run ruff check .           # report problems
uv run ruff check . --fix     # fix the automatically fixable ones
uv run ruff format .          # reformat
```

`uv run <command>` executes that command inside the project's `.venv` without
activating it. This matters more than it looks: there is no
`activate` / `deactivate` cycle to forget, and it is impossible to accidentally
run the wrong project's ruff, or a global one that is a different version. Every
tool invocation in this project is prefixed with `uv run`.

### check vs format

Two distinct operations, often confused:

| Command | Concerned with | Example |
|---|---|---|
| `ruff format` | **Layout** — line breaks, quotes, indentation. Never changes behaviour. | Wrapping a long function call across lines |
| `ruff check` | **Correctness and style rules.** Can change behaviour with `--fix`. | Removing an unused import; flagging an undefined name |

Run `format` first, then `check` — formatting can resolve some line-length
complaints on its own.

---

## Step 6 — Development data strategy

Two kinds of data are needed, for two different jobs. Conflating them produces a
test suite that is slow, fragile and cannot run on a fresh clone.

| | Synthetic | Real |
|---|---|---|
| Purpose | **Regression** — pin down behaviour already decided on | **Discovery** — find behaviour not yet thought of |
| Lives in | `tests/`, committed | `devdata/`, git-ignored, downloaded |
| Size | Tiny — few enough rows to reason about by hand | Large enough to stress memory |
| Tests depend on it? | Yes | **Never** |

### Why both

A generator can only contain the bugs its author already thought of — it is a
specification of current assumptions, so it will never violate one. Real data's
value is precisely that it breaks unanticipated things.

### The loop between them

When real data breaks the library, **do not add the real file to the test
suite.** Reduce the failure to the minimal shape that triggers it, add *that* as
a synthetic case, then fix. Real data finds a bug once; the synthetic case stops
it returning. The alternative slowly turns the test suite into something that
cannot run without a download.

### Placement

- Committed fetch script in `scripts/`
- Data downloaded into `devdata/`, ignored as a whole directory

Keeping the two apart means `.gitignore` can ignore an entire directory rather
than selected files within one — a rule that can be verified at a glance instead
of one that is subtly wrong and reveals itself once.

Only files under `src/datkit/` enter the wheel, so `scripts/`, `tests/` and
`devdata/` cannot leak into the built artifact.

### Get `.gitignore` right before downloading

This is the one irreversible step. Downloading first and then running
`git add .` puts gigabytes into git history, and removing them means rewriting
history.

Verify, do not assume:

```powershell
git status                        # devdata/ should not appear at all
git check-ignore -v devdata\      # names the rule that matched
```

### Choosing a real dataset

The dataset itself is disposable and will change over time — what matters is the
criteria it is chosen against:

1. **Multiple tables with real join keys.** Enough to exercise key-uniqueness and
   referential-coverage checks. A schema where every join is complete and every
   key unique teaches nothing.
2. **Genuinely messy.** Famous teaching datasets have been cleaned to death.
   Wanted: human-written column headings, several null conventions in one file,
   numbers arriving as text, dates in more than one format, at least one
   high-cardinality string column.
3. **Large enough that careless operations hurt.** The memory ceiling in the
   target environment is a design force; development data should be able to
   reproduce that pressure.
4. **No authentication.** A dataset needing an account or API token adds a setup
   step for no analytical benefit.
5. **Licence permits local development use, and contains no personal data.**

### Obtain it manually, do not script it

A download script is worth writing when someone else must recreate the setup, or
when it will be re-run often. Neither applies: this is a one-off download onto
one machine, the test suite never touches it, and it rehearses nothing the target
environment does.

Download by hand into `devdata/` and leave a plain text note beside the files
recording what they are, where they came from and when. That satisfies the whole
reproducibility requirement at a fraction of the effort.

---

## Step 7 — Data model / schema definition

### Why it exists

In the target environment the SQL catalog yields table names, column names and
types — and nothing else. Join keys, cardinality and expectations come from the
client or from stated assumptions. The schema definition is where those two
sources meet.

### Workflow

1. **Introspect** — query the catalog, emit a stub: tables, columns, types
2. **Annotate** — fill in keys, relationships, expectations by hand
3. **Load** — the toolkit reads it and runs checks against the data

### Decisions

| Decision | Choice | Reason |
|---|---|---|
| Describes structure or expectations? | **Expectations** | Turns EDA output into pass/fail rather than reports to interpret |
| Source | **YAML file** | Two sources of truth (generated + hand-edited) rule out a Python literal; must be editable when assumptions prove wrong |
| Format | YAML over JSON | Comments — needed to record *why* an assumption was made |
| | YAML over TOML | `tomllib` is stdlib only from 3.11; the pin is 3.10 |
| Construct | **pydantic**, pending confirmation | Removes the hand-written `from_dict` layer entirely and validates types at runtime. Fall back to stdlib `dataclass` if unavailable |
| Validation trigger | **Explicit `.validate()`** | Allows loading a half-finished schema to see what is wrong with it — the normal case when hand-editing |

**Design rule:** the class takes a **dict**; file loading is a separate thin
function. The format is then not load-bearing, and tests need no files.

### pydantic vs dataclass

| | `dataclass` (stdlib) | pydantic |
|---|---|---|
| Runtime type checking | **None** — annotations are hints; `Column(name=5)` succeeds | Enforced |
| Build from dict | Hand-written `from_dict` | `Model.model_validate(d)` |
| Nested objects from nested dicts | Hand-written recursion | Automatic |
| Error messages | Whatever is raised manually | Structured, naming the field path |
| Mutable defaults | Must use `field(default_factory=list)` | `[]` is safe — copied per instance |

The nesting row saves the most code here, since the YAML nests tables inside
columns and relationships inside endpoints.

**Gotchas either way:**

- `dataclass`: fields without defaults must precede fields with defaults;
  `frozen=True` blocks assignment in `__post_init__`, so validation is possible
  there but normalisation is not.
- pydantic: coerces by default — `"5"` into an `int` field becomes `5` silently.
  `strict=True` disables this.
- pydantic v1 and v2 are effectively different libraries. `model_validate`,
  `model_config` and `model_dump` do not exist in v1. **Confirm
  `pydantic.VERSION` in the target environment, not merely that it is
  installed.**
- Confine pydantic imports to `schema.py`, so an unfavourable answer means
  rewriting one module rather than hunting imports across the toolkit.

**Dependency placement:** pydantic is a runtime dependency — `uv add pydantic`,
without `--dev`. It belongs in `[project] dependencies` and will be resolved by
pip in the target environment.

**Always use `yaml.safe_load`, never `yaml.load`** — the latter can instantiate
arbitrary Python objects from the file.

### Shape

```yaml
tables:
  title_akas:
    source: dbo.title_akas          # generated
    columns:                        # generated
      titleId: {type: varchar}
      ordering: {type: int}
    # one row per localised title variant
    primary_key: [titleId, ordering]   # asserted

relationships:
  - from: {table: title_akas, columns: [titleId]}      # asserted
    to:   {table: title_basics, columns: [tconst]}
    cardinality: many_to_one
    coverage: full                  # full | partial
```

Notes:

- **No `grain` field.** "One row per title" is not assertable — the
  machine-checkable form of it is `primary_key` plus a uniqueness check. Record
  the prose as a YAML comment.
- **Composite keys** make both ends lists, paired by position.
- `coverage: full` — every `from` key must exist on the `to` side.
  `coverage: partial` — the check still runs and reports the rate, but does not
  fail.

### Two validation layers, kept separate

| | Checks | On failure |
|---|---|---|
| **Schema validation** | Is the YAML coherent? Referenced tables and columns exist; `from`/`to` lists are equal length; key columns appear in `columns` | Raise at load, with a clear message |
| **Data validation** | Does the data match the YAML? Uniqueness, coverage, nulls in key columns | Report — do not raise |

Conflating them surfaces a YAML typo as a data quality finding.

### Null handling in keys

A row with a null key is not a missing reference — it is an absent
relationship. By SQL convention, a composite key with *any* null part is not
comparable at all. The EDA always reports nulls found in key columns.

### Deferred

Mapping columns to a controlled vocabulary of semantic roles — transaction date,
status, amount — so the EDA can suggest relevant checks per role. Useful, but
not needed before the core checks work.

---

## Step 8 — Building the schema module

`src/datkit/schema.py` holds four pydantic models — `Column`, `Table`,
`Relationship`, `Schema` — plus two functions:

| Function | Does |
|---|---|
| `load_schema_from_yaml(path)` | Reads the file, returns a `Schema`. Does **not** validate — a broken schema still loads so it can be inspected |
| `check_schema(schema)` | Cross-object consistency. Collects all errors, raises once |

### Checks implemented

- Duplicate table names
- Duplicate column names within a table
- Primary key columns exist on their table
- Relationship `from_table` / `to_table` exist
- Relationship columns exist on the referenced tables
- `from_columns` and `to_columns` are equal length (composite keys pair
  positionally)

Errors are collected into a list and raised together, so a hand-edited YAML
reports every problem in one pass. Where a missing thing is a *lookup* the rest
depends on — an unknown table — record the error and `continue`; there is
nothing downstream to check.

### Gotchas encountered

- **Do not name the method `validate()`.** Pydantic v2's `BaseModel` still
  carries a deprecated `validate` classmethod; defining your own shadows it.
- **Relationship endpoints are `list[str]`, not `list[Column]`.** They
  *reference* existing columns. Using `Column` objects means `"tconst" in
  table.columns` compares a string to objects and is always false — a valid
  schema is rejected.
- **A stream can only be read once.** Calling `yaml.safe_load(file)` twice
  returns the parsed document, then `None`. Read into a variable, then check it.
- **Always pass `encoding="utf-8"` to `open`.** Windows defaults to cp1252, so
  non-ASCII content fails locally and works everywhere else.
- **`yaml.safe_load`, never `yaml.load`** — the latter can instantiate arbitrary
  Python objects.
- `E501` is never auto-fixed. Run `ruff format` *before* `ruff check --fix` — the
  formatter wraps long call arguments, but never splits a string literal. Long
  f-strings are wrapped by hand with implicit concatenation.

---

## Step 9 — Tests with pytest

`tests/` sits at the repo root, outside `src/`, so it never enters the wheel.

Discovery is by naming convention — files `test_*.py`, functions `test_*`. No
registration required.

```powershell
uv run pytest -v      # each test name
uv run pytest -x      # stop at the first failure
```

### `assert` vs `pytest.raises`

| | Question answered | Form |
|---|---|---|
| `assert` | The code ran — is the result right? | `assert normalise(x) == y` |
| `pytest.raises` | The code should refuse — did it refuse correctly? | `with pytest.raises(ValueError, match="empty"):` |

`pytest.raises` fails if **no** exception is raised, or if a different type is.
An `assert` cannot do this job: the exception would abort the test before
reaching it.

To inspect the exception itself:

```python
with pytest.raises(ValueError) as excinfo:
    check_schema(schema)
assert "does_not_exist" in str(excinfo.value)
```

### Conventions used

- **Always pass `match=`.** Without it a test passes on *any* exception of that
  type, including one from an unrelated bug. This is how a test suite quietly
  stops testing anything.
- **Never `pytest.raises(Exception)`** — ruff flags it as `B017`. It swallows
  typos and import errors too. Pydantic's is `ValidationError`.
- **Mutate-one-thing.** A builder function returns a valid object; each test
  breaks exactly one thing. A failure then names the check that misfired,
  instead of leaving you debugging a fixture broken three ways.
- **Build fixtures in Python, not from files.** No YAML, no `devdata` — the
  suite must run on a fresh clone.

---

## Step 10 — Build the wheel and verify it

```powershell
uv build
```

Produces `dist/<name>-<version>-py3-none-any.whl` and an sdist. uv writes a
`dist/.gitignore` automatically.

**Confirm the filename says `py3-none-any`** — that is the proof it is pure
Python and portable.

### Inspect what actually shipped

```powershell
python -m zipfile -l dist\datkit-0.1.0-py3-none-any.whl
```

Expect the package directory and `<name>-<version>.dist-info/`, and nothing
else — no `tests/`, `devdata/` or `docs/`.

### Verify in a clean environment

```powershell
uv venv D:\tmp\wheeltest
uv pip install --python D:\tmp\wheeltest dist\datkit-0.1.0-py3-none-any.whl
uv run --python D:\tmp\wheeltest python -c "import datkit.schema; print('ok')"
```

Importing from the repo proves nothing — the editable install makes it work
regardless. A separate environment is the first place a packaging mistake
appears. Watch whether pip pulls the dependencies automatically; that confirms
the metadata.

Also rehearse the fallback route, which needs the dependencies already present:

```powershell
uv run python -c "import sys; sys.path.insert(0, 'dist/datkit-0.1.0-py3-none-any.whl'); import datkit.schema"
```

### Loosen the dependency bounds before shipping

`uv add` writes lower bounds pinned to whatever is installed locally:

```toml
dependencies = ["pydantic>=2.13.5", "pyyaml>=6.0.3"]
```

In a restricted environment that is a liability. If the target has pydantic 2.9,
pip reads `>=2.13.5`, decides it is unsatisfied, and tries to fetch a newer
version from the internal index — failing, even though 2.9 would have worked.

Set the bound to the oldest version actually required:

```toml
dependencies = ["pydantic>=2", "pyyaml>=6"]
```

**Every version bound is a way for the install to fail.** The same applies to
`requires-python`, which is a hard gate: pip refuses a wheel whose
`Requires-Python` excludes the local interpreter.

Check `Summary:` in the metadata too — the `uv init` placeholder shows up in
`pip show`.

---

## Step 11 — The cleaning module

`src/datkit/cleaning.py`. Small pure functions on a Series, plus two drivers.

| Function | Does |
|---|---|
| `remove_whitespace` | Trim leading/trailing whitespace |
| `to_null` | Replace placeholder tokens with NaN |
| `clean_numbers` | Strip currency symbols and separators, convert to numeric |
| `clean_dates` | Detect and parse date columns |
| `uppercase` | Uppercase text (see caution below) |
| `clean_column` | Runs the above in order on one Series |
| `clean_dataframe` | Applies `clean_column` per column, returns the frame plus a record |

### Two categories, different risk

- **Clean** — changes values. Lossy, needs review.
- **Optimise** — changes dtype only, values identical. Safe.

### The guard that matters

Every lossy conversion compares null counts before and after and **returns the
original if nulls grew**. A conversion that silently drops values is worse than
no conversion, because the loss is invisible downstream. `clean_dataframe`'s
record reports `nulls_created` per column — it should be zero everywhere except
where placeholders were deliberately nulled.

### Ordering is load-bearing

`remove_whitespace` → `to_null` → `clean_numbers` → `clean_dates`

- Padded placeholders (`" NA "`) are only matched after stripping.
- Placeholder text left in a numeric column trips the coercion guard and blocks
  a legitimate conversion.
- `clean_numbers` runs before `clean_dates`, so numeric-looking text never
  reaches the date parser. **This is what stops a year column (`"2019"`) becoming
  1 January dates.** It also means compact dates (`"20240101"`) are read as
  numbers — a known gap.

The ordering test belongs on `clean_dataframe`, not on the individual functions:
a test that calls two functions in sequence documents the constraint but cannot
enforce it.

### Cautions found in use

- **`uppercase` is lossy and expensive.** It destroys titles, names and free
  text irreversibly, and on a large object column it is the step most likely to
  exhaust memory. Make it opt-in rather than a default pipeline step.
- **`clean_dataframe` mutates by default unless it copies.** `df[col] = ...`
  assigns into the caller's frame. An `inplace` parameter makes it a choice
  rather than a surprise; copying is safer but doubles peak memory.

### Python gotchas encountered

- `bool(pd.NaT)` is **True**. Never test a parsed date for truthiness — use
  `pd.isna(...)`.
- `pd.isna(series)` returns a Series. `and` / `or` on a Series raises
  *"truth value is ambiguous"*; use `&` / `|` element-wise then `.any()`
  or `.all()`.
- `to_numeric` is `pd.to_numeric(s)`, a module function — not a Series method.
- A `for` loop that reassigns from the *original* each pass only applies the last
  iteration. Initialise the accumulator before the loop.
- `assert seriesA == seriesB` raises. Use `pandas.testing.assert_series_equal`,
  or compare `.tolist()`.
- Uppercasing a categorical with `.str.upper()` expands it to full strings and
  loses the dtype. Use `s.cat.rename_categories(str.upper)`, which edits the
  category dictionary — and catch `ValueError` for collisions.

---

## Step 12 — Memory discipline in pandas

The target environment has a fixed memory ceiling, so this is a design force,
not an optimisation.

### What is expensive

The rule: **operations that materialise a new Python object per row are
expensive; type conversions and masks are cheap.**

| Operation | Extra memory |
|---|---|
| `s.count()`, `s.min()`, `s.max()` | ~none — C reductions. `len(s) - s.count()` is the cheapest null count |
| `s.isna().sum()` | n bytes — allocates a boolean array first |
| `s.nunique()`, `s.value_counts()` | Proportional to **cardinality**, not length |
| `s.memory_usage(deep=True)` | ~none, but slow on object dtype |
| Anything via `.str` on object dtype | Very expensive — see below |
| `df.describe()`, `.copy()`, `.sort_values()` | Full copy — avoid frame-wide |

**Never call a frame-wide method.** `df.isna()` allocates a boolean frame the
full shape of the data. `df[col].isna()` allocates one column's worth. Loop the
columns and keep only scalars.

### A real failure

`.str.upper()` on a 12.7M-row object column raised `MemoryError` trying to
allocate a **complex128** array. The cause: `map_infer_mask(..., convert=True)`
calls `maybe_convert_objects`, which speculatively allocates a buffer for every
candidate dtype — int64, float64, complex128, bool — to see whether the result
can be narrowed. Those buffers are the cost, not the strings.

### Strings: object vs Arrow

```
object dtype   [ptr][ptr][ptr] → separate Python str objects,
                                 ~50-60 bytes overhead each

Arrow          offsets: [0, 3, 8, 12]
               chars:   "catmousedog"      one contiguous buffer
```

Roughly 5-10x smaller, and `.str` operations run inside Arrow rather than
through the object path.

```python
dtype = "string[pyarrow]"  # Arrow-backed
dtype = "string"  # pandas StringDtype, python-backed
dtype = str  # object — `str` is the Python type, not a pandas dtype
```

**pyarrow as a dev dependency, not a runtime one.** It is a large compiled
wheel; requiring it is another way for the install to fail in a restricted
environment. Detect it at runtime and fall back:

```python
try:
    import pyarrow  # noqa: F401

    dtype = "string[pyarrow]"
except ImportError:
    dtype = "string"
```

### Arrow and numpy disagree about NaN — the sharpest gotcha found so far

Under numpy, NaN **is** missing: both `isna()` and `count()` treat it that way.
Under Arrow, NaN is a **valid double distinct from null**, so `isna()` returns
False for it and `count()` counts it as present.

```
Arrow double[pyarrow] holding [nan, nan, <NA>]

  isna()   → [False, False, True]     count() → 2
  s != s   → [True,  True,  <NA>]
```

**Consequence:** a guard written as `len(s) - s.count()` silently fails under
Arrow. A failed numeric coercion produces NaN, Arrow reports no new nulls, the
guard concludes nothing was lost — and every text column in the frame is
returned as an all-NaN double. The byte counts even look *better*, because
8 bytes of NaN per row is smaller than the text it replaced.

Swapping `count()` for `isna()` does not help — they are two sides of the same
question, and Arrow's answer for NaN is the same either way. The portable form
adds a NaN test:

```python
def missing_count(s: pd.Series) -> int:
    """Count values that are null or NaN."""
    missing = s.isna()
    if pd.api.types.is_float_dtype(s.dtype):
        # NaN is the one value not equal to itself (IEEE 754), so this is True
        # exactly where NaN sits — including the NaNs Arrow calls present.
        missing = missing | (s != s)
    # At a genuine null, `s != s` gives <NA>; a null is missing, so fill True.
    return int(missing.fillna(True).sum())
```

Use it everywhere a null count feeds a decision.

### Other backend differences

- **Dtype checks must not compare to literal names.** `s.dtype in ["object",
  "string"]` is False for `ArrowDtype`, whose name is `"string[pyarrow]"`, so
  every function silently falls through and does nothing. Use
  `pd.api.types.is_string_dtype`, `is_numeric_dtype`, `is_datetime64_any_dtype`.
- `pd.api.types.is_category_dtype` **was removed in pandas 3**. Use
  `isinstance(s.dtype, pd.CategoricalDtype)`.
- `.str` coverage is incomplete on `ArrowDtype` — some methods raise
  `NotImplementedError`.
- scikit-learn, scipy and matplotlib expect numpy arrays; Arrow columns often
  need `.to_numpy()` first, and a nullable integer with nulls cannot become a
  plain numpy int at all.
- **Test against every backend.** A `text_dtype` fixture in `tests/conftest.py`
  parametrised over `object`, `string` and `ArrowDtype(string)` runs each test
  three times. Since the target environment's backend is not under our control,
  this is the only way to find these differences here rather than there.

### Conversion is not always a memory win

A sparse column can grow when converted. A mostly-null short string costs almost
nothing in Arrow; a null `int64` still occupies its 8 bytes. Converting a 96%
null year column from text to `int64` cost 204 KB on 100k rows.

Also expect a small consistent *loss* on columns that were masked but not
converted: Arrow allocates a **validity bitmap** of one bit per row the first
time a column can hold nulls — 12,500 bytes per 100k rows.

### Reading

```python
pd.read_csv(
    path,
    sep="\t",
    usecols=[...],  # biggest win — do not load what you do not need
    nrows=100_000,  # while developing
    dtype=str,  # no inference; cleaning decides the types
    keep_default_na=False,  # stop pandas nulling "NA" before you see it
)
```

`low_memory` is a misnomer: `low_memory=False` uses **more** memory. The default
infers dtypes per internal chunk, so chunks can disagree and the column falls
back to object with a `DtypeWarning`. Setting `dtype=` explicitly skips
inference entirely and makes the option irrelevant.

### Freeing memory

CPython frees by reference counting, so `del df` is usually enough;
`gc.collect()` only helps with reference cycles, which DataFrames do not
normally create.

**In a notebook, `del` is often not enough** — IPython holds references in `_`,
`__` and `Out[n]`. Use `%xdel df`, or `%reset -f out` to clear the output cache.

Freed memory may not return to the OS — Python's allocator keeps arenas — so
Task Manager is not a reliable measure. Use
`df.memory_usage(deep=True).sum()`.

Best habit: avoid creating the second copy at all.

---

## Step 13 — Development notebooks

`notebooks/` at the repo root, outside `src/`, so nothing ships in the wheel.

```powershell
uv add --dev ipykernel      # VS Code's Jupyter extension
uv add --dev jupyterlab     # only if using the browser version
```

**Select the project's `.venv` as the kernel.** Any other interpreter and
`import datkit` fails — the usual cause of "it works in the terminal".

**Autoreload, or you will debug stale code:**

```python
%load_ext autoreload
%autoreload 2
```

The kernel caches imported modules, so editing a module and re-running a cell
silently uses the old version. Autoreload handles most cases; a changed function
signature still needs a kernel restart.

**Anchor paths to the repo root, not the working directory** — a notebook has no
`__file__`, and the cwd differs between VS Code and `jupyter lab`:

```python
REPO = Path.cwd()
while not (REPO / "pyproject.toml").exists():
    REPO = REPO.parent
DEVDATA = REPO / "devdata"
```

**Committing notebooks:** git stores them as JSON with outputs and execution
counts inline, so diffs are unreadable and the repo grows. Either add
`notebooks/` to `.gitignore`, or clear outputs before committing
(`jupyter nbconvert --clear-output --inplace`).

---

## Progress output: logging, not print

`print` in a library cannot be turned off, always goes to stdout, and pollutes
the caller's output.

```python
import logging

logger = logging.getLogger(__name__)
logger.info("Cleaning %s (%s)", col, df[col].dtype)
```

- **Never call `logging.basicConfig()` in library code** — that is the
  application's job.
- **Use `%s` placeholders, not f-strings** — the message is only formatted if it
  is actually emitted.
- In a notebook, enable it with
  `logging.basicConfig(level=logging.INFO, force=True)`. **`force=True` is
  required** — Jupyter has already configured the root logger, so without it
  `basicConfig` silently does nothing.

For long loops, `tqdm.auto` picks the notebook widget or terminal bar
automatically. Import it inside `try/except ImportError` so a missing tqdm
degrades rather than breaking the import.

---

## Git and CI notes

- `git push -u origin <branch>` — a **space**, not `origin/<branch>`. The slash
  form is how git *displays* a remote-tracking branch, not how it is passed.
- `gh pr checks` reports one line per **job**, not per test. A single green tick
  means every step in that job passed. Split lint and test into separate jobs to
  see them separately.
- Branch protection rulesets are free on public repos only; on a private repo
  without a paid plan, CI still runs and reports but cannot block a merge.
- If `gh pr merge --delete-branch` fails to switch branches afterwards, the
  merge itself succeeded — only local cleanup failed. `git stash`, switch, pull,
  `git branch -d <branch>`, `git fetch --prune`, `git stash pop`.
- After a **squash** merge, `git branch -d` may refuse because the squash created
  a new commit with no visible lineage. `git branch -D` is safe once the work is
  on main.

### Before making a repo public: the commit email

Every commit stores an author email, and it is visible on a public repo. Check
what is actually in the history, not just the working tree:

```powershell
git log --format="%an <%ae>" | Sort-Object -Unique
```

Set the identity so it cannot happen again — GitHub → Settings → Emails →
*Keep my email addresses private* gives an address of the form
`<id>+<username>@users.noreply.github.com`, which still links commits to the
account:

```powershell
git config --global user.email "<id>+<username>@users.noreply.github.com"
```

Tick **Block command line pushes that expose my email** on the same page.

**A force-push does not remove the old commits.** GitHub keeps pull-request refs
(`refs/pull/N/head`) pointing at the original commits, so anything that was ever
in a PR stays reachable by SHA even after the branch is rewritten. Rewriting
history with `git filter-repo` has the same limitation.

The only clean fix is to **delete the repository** and push a fresh history. For
a young repo with few commits that is also the simpler path — no history-rewrite
tooling, nothing irreversible to get wrong beyond the deletion itself:

```powershell
Remove-Item -Recurse -Force .git
git init ; git add -A ; git commit -m "Initial commit"
git log --format="%an <%ae>"        # verify BEFORE pushing
gh repo delete <owner>/<repo> --yes
gh repo create <repo> --private --source=. --remote=origin --push
```

`gh repo delete` needs a scope the default login does not have —
`gh auth refresh -h github.com -s delete_repo`, or delete via the web UI under
Settings → Danger Zone.

Note that branch protection rules live on the repository, so they are lost with
it and need re-adding.

---

## Remaining steps

- [ ] Step 14 — Table profiling module
- [ ] Step 15 — Relationship checks: key uniqueness and referential coverage
- [ ] Step 16 — Joined-table profiling and charts
- [ ] Step 17 — Constrain imports to the target environment's baseline
- [ ] Step 18 — Upload and import in the target environment
- [ ] Step 19 — Versioning and update workflow
