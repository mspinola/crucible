# crucible-quant has been renamed to `crucible`

This project now publishes under its own name. Install that instead:

```bash
pip install crucible
```

**Nothing about your code changes.** The import name was always `crucible`, so
`import crucible` works exactly as it did. Only the installer name moved.

This package is a redirect. Version 0.1.1 contains no code of its own, it simply
depends on `crucible`, so an existing `pip install -U crucible-quant` pulls the real
package in. Version 0.1.0 is the last release that shipped actual source, and it is
frozen at the state of the project on 2026-07-14.

No further releases will be made here. Point your dependencies at `crucible`.

- Package: <https://pypi.org/project/crucible/>
- Source: <https://github.com/mspinola/crucible>
- Docs and tutorial: <https://mspinola.github.io/crucible/>

## Why the rename

`crucible` measures the raw mathematical edge of a trading signal, capital-free, with
confidence intervals and a reality check. It always imported as `crucible`. The
distribution carried the `-quant` suffix only because the bare name was unavailable on
PyPI at first release. That is resolved, so the two names now match.

Note that releases under the new name start at **0.2.0**. An unrelated package holds
0.1.0 on the `crucible` project, and PyPI does not allow a version number to be reused.
