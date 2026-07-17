"""Guard the gate-token wrapper's exclusions — the parts that silently regress.

``docs/gate_tokens.py`` styles the gauntlet gate names in rendered HTML. The
value is entirely in *what it leaves alone*: code samples, headings, links, and
tag attributes must never gain a ``<span>``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "docs"))

from gate_tokens import wrap_gate_tokens  # noqa: E402


def test_plain_prose_is_wrapped():
    out = wrap_gate_tokens("<p>The book clears REAL and STRONG.</p>")
    assert out.count('<span class="gate">') == 2
    assert "<span class=\"gate\">REAL</span>" in out


def test_code_and_pre_are_left_literal():
    # audit-report samples like "REAL ✓" must stay literal
    assert wrap_gate_tokens("<pre><code>REAL  DURABLE</code></pre>") == (
        "<pre><code>REAL  DURABLE</code></pre>"
    )
    assert '<span' not in wrap_gate_tokens("<p><code>REAL</code></p>")


def test_headings_and_links_are_skipped():
    assert "<span" not in wrap_gate_tokens('<h3 id="x">the GENERAL gate</h3>')
    assert "<span" not in wrap_gate_tokens('<a href="x">the DURABLE writeup</a>')


def test_image_alt_text_is_never_touched():
    # a gate name inside an attribute must not get a span injected into the tag
    out = wrap_gate_tokens('<p><img alt="the REAL/STRONG gates" src="x.png"> GENERAL</p>')
    assert 'alt="the REAL/STRONG gates"' in out
    assert out.count('<span class="gate">') == 1  # only the visible GENERAL


def test_bold_occurrence_normalizes():
    # the span carries its own weight, so **REAL** matches a plain occurrence
    out = wrap_gate_tokens("<p>on <strong>DURABLE</strong> here</p>")
    assert '<strong><span class="gate">DURABLE</span></strong>' in out


def test_word_boundaries():
    out = wrap_gate_tokens("<p>REAL's and GENERAL-only</p>")
    assert '<span class="gate">REAL</span>\'s' in out
    assert '<span class="gate">GENERAL</span>-only' in out


def test_verdict_words_are_not_gates():
    # FAIL / HELD / FRAGILE keep their own treatment — only gates are wrapped
    assert "<span" not in wrap_gate_tokens("<p>the verdict is FAIL, not HELD</p>")
