import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")   # the tearsheet needs the report extra

from crucible.ml import decay_tearsheet


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
