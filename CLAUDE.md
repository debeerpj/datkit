# datkit — working agreement

Read this before doing anything in this repo.

## How to work with me

- **One step at a time.** Do not batch several changes into one message.
- **Explain new concepts** the first time they appear.
- **Guide, do not implement.** When I am building something to learn it, tell me
  what the method should do, which functions to use, and which gotchas to watch
  for. I will write it. I ask explicitly when I want code, and I do ask —
  tests especially.
- **Keep responses short.** No long write-ups at every step.
- **Push back.** If a design or a default is wrong, say so with evidence before
  building it. Test your own drafts and report what broke.

## Changing the repo

- **Do not change the repo without asking first.** `docs/SETUP.md` is the
  exception — keep it up to date as we go, unprompted.
- **Edit files in place; do not send whole copies.** Work on a branch, make the
  edit, and tell me what changed so I can read `git diff`. Re-emitting a whole
  module costs tokens and creates competing copies of the truth.
- Branch protection is on `main`: branch **before** editing, then
  `gh pr create --fill`, then `gh pr merge --squash --delete-branch`.
- Git identity is the GitHub noreply address — do not let a real one back in.

## Code style

- **Comment everything, concisely.** Say why, not what.
- **Prefer explicit loops over comprehensions** where the loop does real work.
- Numpydoc docstrings: summary line, Parameters, Returns.
- `line-length = 120`; ruff config lives in `pyproject.toml`.
- Type hints on public functions.

## The environment this ships into

Every design decision comes back to this. A pure-Python wheel, uploaded by
hand, importing with what is already present:

- **Python 3.10**, pandas **2.2.3**, numpy **2.2.0**, pydantic **2.9.2**,
  PyYAML **6.0.2**, pyarrow **18.1.0**, polars **1.8.2**, duckdb **1.1.1**.
- **No internet.** A package that fetches at runtime will fail silently or hang.
- `pytest 8.3.4` **is** available there. `ruff`, `black`, `mypy` and `uv` are not.
- Fixed memory ceiling; data can exceed it. 90M-row tables are normal.
- The SQL catalog gives table, column and type names — nothing else. Join keys
  and expectations come from the client or from assumptions.

I develop on pandas 3.x locally, which is **ahead** of the target. Check
anything dtype-related against 2.2.3 before believing it works.

## Memory rules for anything that touches data

- Counts plus a bounded sample, never the offending rows. Output size fixed
  regardless of input size.
- `isin`, not set arithmetic, for coverage — hashes only the parent side.
- Composite keys need `MultiIndex.isin`; concatenating key columns into a
  string breaks when a value contains the separator.
- Never a frame-wide method. `df.isna()` allocates a boolean frame the full
  shape of the data.
- Decide from a sample, then apply once to the whole column.

## Commands

```powershell
uv sync
uv run pytest -q
uv run ruff format . ; uv run ruff check --fix .
uv build
```

## Layout

```
src/datkit/
  cleaning.py     column cleaning, the pipeline, the change record
  schema.py       pydantic models, YAML loader, schema validation
tests/            conftest.py runs text tests across three string backends
docs/SETUP.md     the full build log — every decision and what went wrong
docs/HANDOVER.md  where things stand and what to do next
notebooks/        exploration, git-ignored
devdata/          IMDb subset, git-ignored
```

`docs/SETUP.md` is long on purpose. Read the gotchas section before debugging
anything that looks like a pandas quirk — it has probably already cost me a day.
