# Publishing the `crucible-quant` redirect

This directory builds a **separate distribution** from the one at the repo root. It is
not part of `crucible` and is not built or published by `.github/workflows/release.yml`.

Its only job: `crucible-quant` 0.1.1 carries no code and depends on `crucible>=0.2`, so
anyone still on the old name gets the real package on their next upgrade, and the
[PyPI page](https://pypi.org/project/crucible-quant/) says where the project went.
`README.md` here is what renders as that page.

This is expected to be a **one-off**. There is no reason to cut another release.

## If you ever do need to rebuild and upload

```bash
cd packaging/crucible-quant-shim
python -m build            # or: uv build --out-dir dist
twine check dist/*
twine upload dist/*        # username __token__, password a crucible-quant scoped token
```

Tokens are created at <https://pypi.org/manage/account/token/>. Scope the token to the
`crucible-quant` project rather than the whole account. There is no Trusted Publisher
configured for this distribution, and setting one up for a dead package is not worth it.

## Checks worth repeating

The upgrade path is the thing that can silently break, because `crucible-quant` 0.1.0
vendored its own `crucible/` package directory. Confirm the real package actually
replaces those files rather than being shadowed by them:

```bash
pip install "crucible-quant==0.1.0"        # the old world
pip install dist/crucible_quant-0.1.1-py3-none-any.whl
python -c "from crucible.validation import run_gauntlet; print('ok')"
```

`run_gauntlet` did not exist in 0.1.0, so that import succeeding proves the 0.2.0+ code
is what loaded.
