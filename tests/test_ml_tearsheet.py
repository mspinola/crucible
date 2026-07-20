import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")   # the tearsheet needs the report extra

from crucible.ml import decay_tearsheet, score_by_outcome


def _preds(n=500, seed=1):
    rng = np.random.default_rng(seed)
    score = rng.uniform(0, 1, n)
    label = np.where(rng.uniform(0, 1, n) < score, 1, -1)   # higher score -> more wins
    return pd.DataFrame({"score": score, "label": label})


def test_tearsheet_returns_self_contained_html():
    html = decay_tearsheet(_preds())
    assert html.lstrip().lower().startswith("<!doctype html")
    assert "</html>" in html
    assert "plotly" in html.lower()       # embedded chart
    assert "Q1" in html and "Q5" in html  # quantile table rendered


def test_tearsheet_writes_file(tmp_path):
    out = tmp_path / "sub" / "decay.html"
    html = decay_tearsheet(_preds(), out_path=str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == html


def test_tearsheet_custom_columns():
    df = _preds().rename(columns={"score": "prob", "label": "y"})
    html = decay_tearsheet(df, score="prob", label="y")
    assert "</html>" in html


def test_tearsheet_raises_on_too_few_rows():
    with pytest.raises(ValueError):
        decay_tearsheet(pd.DataFrame({"score": [0.1, 0.2, 0.3], "label": [1, -1, 1]}), q=5)


# ── the extracted winners-vs-losers violin panel ──────────────────────────────

def test_score_by_outcome_is_an_embeddable_panel():
    html = score_by_outcome(_preds())
    assert isinstance(html, str) and html
    assert "<!doctype" not in html.lower()          # a fragment, not a full page
    assert "violin" in html.lower()                  # the winners/losers violin traces
    assert "winners" in html and "losers" in html


def test_score_by_outcome_include_plotlyjs_modes():
    p = _preds()
    assert "cdn.plot" in score_by_outcome(p, include_plotlyjs="cdn")   # standalone
    assert "cdn.plot" not in score_by_outcome(p)                        # default script-less


def test_decay_tearsheet_embeds_the_panel():
    # the tearsheet is now a consumer of the extracted panel — its violins still render
    html = decay_tearsheet(_preds())
    assert "winners" in html and "losers" in html


def test_score_by_outcome_custom_columns_and_empty():
    df = _preds().rename(columns={"score": "prob", "label": "y"})
    assert score_by_outcome(df, score="prob", label="y")
    empty = pd.DataFrame({"score": [np.nan, np.inf], "label": [1, -1]})
    assert score_by_outcome(empty) == ""
    with pytest.raises(ValueError, match="columns"):
        score_by_outcome(pd.DataFrame({"x": [1]}))
