"""examples/README.md is an index of examples/*.py; this keeps it from falling behind.

A new example that is not added to the index is invisible to the reader who lands in
the folder on GitHub, which is exactly where people go looking for examples. The
check is by filename, so wording can change freely; only the roster is pinned.
"""
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_every_example_is_indexed_in_the_readme():
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    missing = sorted(p.name for p in EXAMPLES.glob("*.py") if p.name not in readme)
    assert not missing, f"examples/README.md does not mention: {missing}"


def test_readme_names_only_examples_that_exist():
    readme = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    tokens = (tok.strip("`(),.") for tok in readme.split())
    # bare filenames only: prose like `examples/<name>.py` or `examples/*.py` is not a listing
    listed = {t for t in tokens if t.endswith(".py") and not any(c in t for c in "/*<>")}
    stale = sorted(name for name in listed if not (EXAMPLES / name).exists())
    assert not stale, f"examples/README.md names files that do not exist: {stale}"
