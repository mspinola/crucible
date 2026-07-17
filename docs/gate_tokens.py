"""Style the gauntlet gate names as one consistent token wherever they appear.

DECLARE / CLEAN / REAL / STRONG / DURABLE / GENERAL / SURVIVE are a controlled
vocabulary — a named set of gates, not ordinary emphasis. Wrapping each in
``<span class="gate">`` lets one CSS rule (spaced small caps, deep teal) mark
them consistently, so the reader learns to read them as terms of art. The caps
in the Markdown source stay the source of truth; this only styles them.

Shared by both renderers:

  * the MkDocs site, via the ``on_page_content`` hook below (registered under
    ``hooks:`` in ``mkdocs.yml``);
  * ``gen_pdf.py``, which imports :func:`wrap_gate_tokens` and applies it to the
    HTML it hands to xhtml2pdf.

The pass runs on *rendered HTML* and only rewrites text between tags, so it is
safe by construction:

  * ``<pre>`` / ``<code>`` are skipped — audit-report samples like ``REAL ✓``
    stay literal, not styled;
  * headings are skipped — they already carry the heading color;
  * ``<a>`` is skipped — links already carry the link color;
  * tag attributes (e.g. an ``<img alt="…REAL…">`` description) are never
    touched, because only the text nodes between ``>`` and ``<`` are rewritten.

A bold occurrence (``**REAL**`` → ``<strong>REAL</strong>``) normalizes for
free: the span sets ``font-weight`` itself, so it renders identically to a plain
occurrence.
"""
from __future__ import annotations

import re

GATES = ("DECLARE", "CLEAN", "REAL", "STRONG", "DURABLE", "GENERAL", "SURVIVE")

_WORD = re.compile(r"\b(" + "|".join(GATES) + r")\b")
_TAG = re.compile(r"(<[^>]+>)")
# Blocks whose *contents* must be left alone, kept intact by the capturing split.
_PROTECT = re.compile(
    r"(<pre\b.*?</pre>|<code\b.*?</code>|<h[1-6]\b.*?</h[1-6]>|<a\b.*?</a>)",
    re.IGNORECASE | re.DOTALL,
)


def wrap_gate_tokens(html: str) -> str:
    """Wrap each gate name in ``<span class="gate">`` outside protected blocks."""
    out = []
    for i, block in enumerate(_PROTECT.split(html)):
        if i % 2 == 1:  # a protected block (pre/code/heading/link) — leave as is
            out.append(block)
            continue
        # Rewrite only the text nodes; tags (odd indices) — including any
        # attribute values — pass through untouched.
        segs = _TAG.split(block)
        for j in range(0, len(segs), 2):
            segs[j] = _WORD.sub(r'<span class="gate">\1</span>', segs[j])
        out.append("".join(segs))
    return "".join(out)


def on_page_content(html, page=None, config=None, files=None):
    """MkDocs hook — style gate tokens in every rendered page."""
    return wrap_gate_tokens(html)
