# crucible examples

Every file here runs on its own from the repo root (`python examples/<name>.py`).
The ones marked "core" need only `pip install crucible`; the extras are
`pip install "crucible[report]"` (plotly, for HTML pages) and
`pip install "crucible[examples]"` (yfinance, for real prices, which also means
network access). Two of them double as CI smoke tests; the rest are run by hand.

| Example | What it shows | Needs | In CI |
|---|---|---|---|
| `quickstart.py` | The README pipeline on synthetic data: a signal to a trade log, the edge report, the reality check, and a random-entry null. | core | yes |
| `validation.py` | The validation ladder on one signal: holdout, walk-forward, permutation test. | core | no |
| `donchian_gauntlet.py` | A Donchian breakout through the whole gauntlet. Prints the exact numbers the tutorial's section 12 walks through: REAL and STRONG pass, DURABLE fails, so the gauntlet rejects it. | core | no |
| `breadth.py` | How many independent bets a book of correlated markets really holds: effective N and the factors behind it. | core | no |
| `ml_meta_label.py` | A meta-labeling take/skip filter through the ML track and the gauntlet (tutorial section 13). | core | no |
| `edge_monitor.py` | After promotion: freeze a baseline, then watch the edge for decay, HOLDING / SLIPPING / DEGRADED (tutorial section 14). | core | no |
| `tearsheet.py` | A self-contained HTML tearsheet for a signal; `--ticker` swaps in real prices. | `[report]`; `[examples]` + network with `--ticker` | no |
| `real_data_yfinance.py` | The quickstart pipeline on real Yahoo Finance prices. | `[examples]` + network | no |
| `stridsman_postpub.py` | A published system (Stridsman, 2000) judged only on data from after publication: the publication date as the holdout, one look in the ledger, the detrended null scaled into R. Synthetic prices with trend regimes planted only before publication, so the full history flatters and the post-publication gate fails. | core; `--report` needs `[report]` | yes, plus a test that guards the story it prints |
| `stridsman_postpub_yfinance.py` | The same procedure on `ES=F` or `SPY`, where the verdict is not scripted. Read its docstring for the roll-gap caveat. | `[examples]` + network; `--report` needs `[report]` | no |

Two of the rows are pairs: `quickstart.py` and `real_data_yfinance.py` are the same
pipeline offline and online, and the two `stridsman_postpub` files are the same
procedure on synthetic and real data, sharing one implementation.

`tests/test_examples_readme.py` checks that every `examples/*.py` appears in this
table and that nothing listed here has been deleted, so the index cannot fall behind
the folder.
