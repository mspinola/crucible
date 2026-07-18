#!/usr/bin/env python3
"""Render the tutorial (docs/tutorial.md) to a distribution PDF.

Pure-Python pipeline, no system libraries: python-markdown builds styled HTML,
xhtml2pdf turns it into a paginated PDF. The docs CI runs this after
``mkdocs build`` and drops the result into ``site/`` so the published tutorial
has a downloadable companion at ``.../crucible/tutorial.pdf``.

Run locally the same way CI does::

    pip install -r requirements-docs.txt
    python docs/gen_pdf.py --out tutorial.pdf

The Markdown source stays the single source of truth; this script only adapts a
few things that a print target needs but a web page doesn't (box-drawing glyphs
that xhtml2pdf can't render, a title block, and page numbers).
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import markdown

from gate_tokens import wrap_gate_tokens

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = HERE / "tutorial.md"
SITE_URL = "https://mspinola.github.io/crucible/"

# xhtml2pdf renders on the built-in Helvetica/Courier fonts, which don't cover
# box-drawing characters (they come out as solid black bars). The tutorial uses
# them only as rules inside code blocks, so fold them to ASCII for the PDF while
# leaving the Markdown source — and the web render — untouched.
_BOX_FOLD = {
    ord(c): "-" for c in "─═━"
} | {ord(c): "|" for c in "│║┃"} | {
    ord(c): "+" for c in "┼┬┴├┤┌┐└┘╬╦╩╠╣╔╗╚╝"
}


def _strip_leading_h1(src: str) -> str:
    """Drop the document's opening ``# Title`` — the cover block replaces it."""
    lines = src.splitlines()
    for i, line in enumerate(lines):
        if line.strip():
            if line.lstrip().startswith("# "):
                lines = lines[i + 1 :]
            break
    return "\n".join(lines).lstrip("\n")


def _prepare_source(src: str) -> str:
    """Adapt the web Markdown for print: drop the opening H1 (the cover replaces
    it) and the self-referential "Download this tutorial as a PDF" link."""
    src = _strip_leading_h1(src)
    kept = [ln for ln in src.splitlines() if "tutorial.pdf)" not in ln]
    return "\n".join(kept).lstrip("\n")


def build_html(src: str) -> str:
    body = markdown.markdown(
        _prepare_source(src).translate(_BOX_FOLD),
        extensions=[
            "tables",
            "fenced_code",
            "codehilite",
            "toc",
            "sane_lists",
            "attr_list",
        ],
        extension_configs={"codehilite": {"noclasses": True, "guess_lang": False}},
    )
    body = wrap_gate_tokens(body)
    generated = date.today().isoformat()
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body>
<div id="cover">
  <table style="width:250px;border-collapse:collapse;margin:0 0 1.5cm 0;"><tr>
    <td style="width:52px;border:none;padding:0;vertical-align:middle;"><img src="img/crucible_logo.png" style="width:40px;height:39px;display:block;margin:0;" alt="crucible" /></td>
    <td style="border:none;padding:0;vertical-align:middle;font-size:16pt;letter-spacing:3pt;text-transform:uppercase;color:#00897b;">crucible</td>
  </tr></table>
  <h1 class="cover-title">From Trade Log to Verdict</h1>
  <div class="cover-sub">The Statistics of a Significant Edge</div>
  <div class="cover-meta">{SITE_URL} &middot; generated {generated}</div>
  <hr class="cover-sep" />
</div>
{body}
</body></html>"""


# xhtml2pdf supports a limited CSS subset (no flexbox/grid, simple selectors).
# @page frames drive the running footer + page numbers.
CSS = """
@page {
  size: letter;
  margin: 2.0cm 1.8cm 2.2cm 1.8cm;
  @frame footer { -pdf-frame-content: footerContent; bottom: 1.1cm; margin-left: 1.8cm; margin-right: 1.8cm; height: 1cm; }
}
body { font-family: Helvetica, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a2226; }
#cover { margin-top: 2.6cm; margin-bottom: 0.5cm; }
.cover-title { font-size: 30pt; color: #00695c; margin: 0 0 0.15cm 0; border: none; }
.cover-sub { font-size: 15pt; color: #37474f; }
.cover-meta { font-size: 9pt; color: #78909c; margin-top: 0.7cm; }
.cover-sep { border: none; border-top: 2pt solid #009688; margin: 0.8cm 0 0 0; }
h1, h2, h3, h4 { color: #00695c; font-weight: bold; }
h1 { font-size: 19pt; border-bottom: 1pt solid #b2dfdb; padding-bottom: 3pt; margin-top: 0.6cm; }
h2 { font-size: 15pt; border-bottom: 1pt solid #e0f2f1; padding-bottom: 2pt; margin-top: 0.5cm; }
h3 { font-size: 12.5pt; margin-top: 0.4cm; }
h4 { font-size: 11pt; color: #00796b; }
a { color: #00796b; text-decoration: none; }
p, li { text-align: left; }
code { font-family: Courier, monospace; font-size: 9pt; background: #eceff1; color: #263238; padding: 1pt 3pt; }
pre { background: #f4f6f7; border: 0.5pt solid #cfd8dc; border-left: 2.5pt solid #009688; padding: 8pt 10pt; margin: 6pt 0; }
pre code { background: none; padding: 0; font-size: 8.6pt; line-height: 1.35; }
blockquote { border-left: 3pt solid #009688; background: #f4faf9; margin: 8pt 0; padding: 4pt 12pt; color: #37474f; }
table { border-collapse: collapse; margin: 8pt 0; width: 100%; }
th, td { border: 0.5pt solid #b0bec5; padding: 4pt 7pt; font-size: 9pt; text-align: left; }
th { background: #e0f2f1; color: #004d40; }
hr { border: none; border-top: 0.5pt solid #cfd8dc; margin: 12pt 0; }
img { display: block; margin: 8pt 0; }
.caption { font-size: 8.5pt; color: #78909c; font-style: italic; margin: 0 0 8pt 0; }
.gate { color: #00695c; font-size: 9.6pt; letter-spacing: 1.2pt; font-weight: normal; }
"""

# The frame anchor is a plain block <div> (tables/floats aren't reliably honoured
# here); one centered line keeps it robust across pages.
FOOTER = (
    '<div id="footerContent" style="font-size:8pt;color:#90a4ae;text-align:center;">'
    "crucible &middot; From Trade Log to Verdict &middot; "
    "page <pdf:pagenumber> of <pdf:pagecount>"
    "</div>"
)


def _link_callback(base_dir: Path):
    """Resolve relative image URIs (e.g. ``img/foo.png``) to absolute paths so
    xhtml2pdf can embed them; pass through data:/http(s) URIs untouched."""
    def cb(uri: str, _rel: str) -> str:
        if uri.startswith(("http://", "https://", "data:")):
            return uri
        return str((base_dir / uri).resolve())
    return cb


def render_pdf(src_path: Path, out_path: Path) -> None:
    from xhtml2pdf import pisa  # imported lazily so --help works without deps

    html = build_html(src_path.read_text(encoding="utf-8"))
    # Inject the footer template just inside <body>.
    html = html.replace("<body>", "<body>\n" + FOOTER, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        result = pisa.CreatePDF(html, dest=fh, encoding="utf-8",
                                link_callback=_link_callback(src_path.parent))
    if result.err:
        raise SystemExit(f"xhtml2pdf reported {result.err} error(s) generating {out_path}")
    print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Markdown source (default: docs/tutorial.md)")
    ap.add_argument("--out", type=Path, default=Path("tutorial.pdf"), help="output PDF path")
    args = ap.parse_args()
    render_pdf(args.src, args.out)


if __name__ == "__main__":
    main()
