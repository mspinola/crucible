"""The import boundaries that make crucible what it is.

crucible's value is that it is small, MIT, and trivially adoptable: `numpy` + `pandas` and
nothing else. That is not a happy accident to be rediscovered later — it is the property
these tests defend, because it erodes one convenient import at a time.

Three invariants, from the toolchain's ADRs:

* **No engine.** crucible never imports vectorbt/vectorbtpro. The paid, closed engine lives
  *outside* crucible behind a seam, consuming its `TradeLog`
  (ADR-0001, action item 4).
* **No orchestration or durable state.** crucible is a library of pure lenses. Scheduling,
  live state, and the deployment loop live in the orchestrator sibling and never travel back
  (ADR-0003, commitment 5 / action item 6).
* **One-directional dependency.** npf imports crucible; crucible never imports npf. A library
  that imports an application is not a library.

Asserted on the import surface via `ast` rather than on file text, so prose is free to
*discuss* vectorbt and schedulers without tripping the guard.
"""
import ast
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parent.parent / "src" / "crucible"

# Subpackages allowed a plotting dependency; everything else is core.
EXTRA_DEPS = {"ml": {"plotly"}, "report": {"plotly"}}
CORE_THIRD_PARTY = {"numpy", "pandas"}

# Anything that would make crucible something other than a pure, capital-free library.
BANNED = {
    # the wrapped engine — must stay outside crucible (ADR-0001)
    "vectorbt", "vectorbtpro",
    # the application / the siblings — dependency runs one way only
    "npf",
    # schedulers and workflow engines — orchestration is not crucible's job (ADR-0003)
    "croniter", "prefect", "dagster", "airflow", "temporal", "celery", "apscheduler",
    "schedule", "sched",
    # durable state — a lens does not persist anything
    "sqlite3", "shelve", "dbm", "pickle", "sqlalchemy", "redis",
}

STDLIB = {
    "__future__", "dataclasses", "typing", "json", "os", "math", "itertools", "functools",
    "collections", "warnings", "datetime", "pathlib", "abc", "enum", "re", "sys", "copy",
    "textwrap", "random", "time", "string", "io", "contextlib", "inspect", "statistics",
    "urllib", "hashlib", "operator", "types", "numbers", "csv", "importlib", "logging",
}


def _modules():
    return sorted(SRC.rglob("*.py"))


def _imports(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _top_level(names):
    return {n.split(".")[0] for n in names}


def _package_of(path):
    rel = path.relative_to(SRC)
    return rel.parts[0] if len(rel.parts) > 1 else "(top)"


def test_the_guard_can_see_the_package():
    assert len(_modules()) > 10, "boundary test stopped seeing crucible's sources"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.name))
def test_module_imports_nothing_banned(path):
    hits = _top_level(_imports(path)) & BANNED
    assert not hits, (
        f"{path.name} imports {sorted(hits)}. crucible stays a pure, capital-free library: "
        "engines, schedulers, durable state and the npf application all live outside it.")


@pytest.mark.parametrize("path", _modules(), ids=lambda p: str(p.name))
def test_core_modules_import_only_numpy_and_pandas(path):
    pkg = _package_of(path)
    allowed = CORE_THIRD_PARTY | EXTRA_DEPS.get(pkg, set())
    third_party = {m for m in _top_level(_imports(path))
                   if m not in STDLIB and m != "crucible"}
    extra = third_party - allowed
    assert not extra, (
        f"{path.name} (package {pkg!r}) imports {sorted(extra)}. crucible's core dependency "
        f"set is {sorted(CORE_THIRD_PARTY)}; only {sorted(EXTRA_DEPS)} may add a plotting "
        "extra. A new dependency here is a change to what crucible is.")


def test_declared_dependencies_match_the_enforced_ones():
    """The packaging twin: catches a dependency added to pyproject before it is imported."""
    text = (SRC.parent.parent / "pyproject.toml").read_text()
    block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    declared = {line.split(">=")[0].split("==")[0].strip().strip('"\'')
                for line in block.strip().splitlines() if line.strip()}
    assert declared == CORE_THIRD_PARTY, (
        f"pyproject declares {sorted(declared)} but the enforced core set is "
        f"{sorted(CORE_THIRD_PARTY)}; update both together, deliberately.")
