"""Build PDF documents from the Markdown sources (markdown + xhtml2pdf, no LaTeX).

Public output:
    docs/PROJECT_WALKTHROUGH.pdf

Optional local documents: if ``docs/local_documents.toml`` exists (it is git-ignored), the
documents it lists are built too. Two kinds are supported:

    [[document]]                    # one Markdown file -> one PDF
    kind = "markdown"
    source = "docs/SOME_NOTES.md"
    output = "docs/SOME_NOTES.pdf"
    footer = "EpiDetect - Notes"

    [[document]]                    # a guide: title page, table of contents, walkthrough,
    kind = "guide"                  # key results and figures, limitations and future work,
    output = "docs/GUIDE.pdf"       # the extra parts listed below, data notes in appendix
    footer = "EpiDetect - Guide"
    subtitle = "..."
    author = "..."
    [[document.extra_parts]]
    heading = "Part 4 - ..."
    source = "docs/SOME_NOTES.md"

Fonts: DejaVu Sans / Sans Mono shipped with Matplotlib (Greek letters, arrows and maths symbols
render on every machine without downloading anything).

Usage:
    python scripts/build_pdfs.py
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from typing import Any

import markdown
import matplotlib
from xhtml2pdf import pisa

REPO = Path(__file__).resolve().parents[1]
DOCS, FIGURES, RESULTS = REPO / "docs", REPO / "figures", REPO / "results"
FONTS = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
EXTENSIONS = ["tables", "fenced_code", "sane_lists"]
LOCAL_DOCUMENTS = DOCS / "local_documents.toml"  # optional, git-ignored
KEY_FIGURES = [
    (
        "false_hfo_demo.png",
        "Why a filtered spike looks like an HFO (scripts/05_spike_filter_demo.py).",
    ),
    ("detector_examples.png", "The three detectors on four typical simulated situations."),
    (
        "sim_benchmark_snr.png",
        "Simulation benchmark: recall, precision and spike false alarms vs SNR.",
    ),
    (
        "real_lopo.png",
        "Real data, leave-one-patient-out: F1 per test patient and macro F1 (95 % CI).",
    ),
    (
        "strategy_comparison.png",
        "Tuning on real patients (A), synthetic (B) or hybrid (C) simulation.",
    ),
    ("clinical_roc.png", "Post-resection fast-ripple rate predicting a poor surgical outcome."),
    ("sim_background_calibration.png", "Simulator background calibrated on the real spectra."),
    ("sim_spike_calibration.png", "Spike sharpness calibrated on 8,630 real transients."),
]

CSS = f"""
@font-face {{ font-family: DejaVu; src: url("{(FONTS / "DejaVuSans.ttf").as_posix()}"); }}
@font-face {{ font-family: DejaVu; font-weight: bold;
             src: url("{(FONTS / "DejaVuSans-Bold.ttf").as_posix()}"); }}
@font-face {{ font-family: DejaVu; font-style: italic;
             src: url("{(FONTS / "DejaVuSans-Oblique.ttf").as_posix()}"); }}
@font-face {{ font-family: DejaVuMono; src: url("{(FONTS / "DejaVuSansMono.ttf").as_posix()}"); }}
@page {{ size: a4 portrait; margin: 2cm 1.8cm 2cm 1.8cm;
        @frame footer {{ -pdf-frame-content: footer; bottom: 0.8cm; margin-left: 1.8cm;
                         margin-right: 1.8cm; height: 0.6cm; }} }}
body {{ font-family: DejaVu; font-size: 9.5pt; line-height: 1.35; color: #1a1a1a; }}
h1 {{ font-size: 18pt; color: #0b3d75; margin-top: 10pt; -pdf-outline: true; }}
h2 {{ font-size: 14pt; color: #0b3d75; margin-top: 12pt; border-bottom: 0.5pt solid #9ab;
      -pdf-outline: true; }}
h3 {{ font-size: 11.5pt; color: #1f5fa8; margin-top: 9pt; -pdf-outline: true; }}
h4 {{ font-size: 10pt; color: #333; margin-top: 7pt; }}
code {{ font-family: DejaVuMono; font-size: 8pt; background-color: #f2f2f0; }}
pre {{ font-family: DejaVuMono; font-size: 7pt; background-color: #f4f4f2; padding: 5pt;
       border: 0.5pt solid #ccc; }}
table {{ border-collapse: collapse; margin: 4pt 0 8pt 0; }}
th, td {{ border: 0.5pt solid #999; padding: 2pt 3pt; font-size: 8pt; vertical-align: top; }}
th {{ background-color: #e8eef6; }}
blockquote {{ color: #444; margin-left: 10pt; }}
.title {{ text-align: center; margin-top: 150pt; }}
.title h1 {{ font-size: 30pt; border: none; }}
.subtitle {{ font-size: 13pt; color: #333; }}
.caption {{ font-size: 8pt; color: #444; text-align: center; margin-bottom: 10pt; }}
#footer {{ text-align: right; font-size: 8pt; color: #777; }}
pdftoc {{ color: #333; }}
pdftoc.pdftoclevel0 {{ font-weight: bold; margin-top: 4pt; }}
pdftoc.pdftoclevel1 {{ margin-left: 12pt; }}
pdftoc.pdftoclevel2 {{ margin-left: 24pt; font-size: 8.5pt; }}
"""


def md_to_html(text: str, shift: int = 0) -> str:
    """Markdown to HTML; ``shift`` demotes headings (h1 -> h2 ...) when embedding a document."""
    text = re.sub(r"```mermaid.*?```", "", text, flags=re.S)  # diagrams cannot render in PDF
    html = markdown.markdown(text, extensions=EXTENSIONS)
    if shift:

        def demote(match: re.Match) -> str:
            level = min(int(match.group(2)) + shift, 6)
            return f"<{match.group(1)}h{level}"

        html = re.sub(r"<(/?)h([1-6])", demote, html)
    # Local images: absolute file paths for xhtml2pdf, width limited to the text width.
    html = re.sub(
        r'<img alt="([^"]*)" src="(figures/[^"]+)"',
        lambda m: f'<img alt="{m.group(1)}" src="{(REPO / m.group(2)).as_posix()}" width="480"',
        html,
    )
    return html


def readme_section(title: str) -> str:
    """One ``## title`` section of the README (up to the next ``## ``)."""
    text = (REPO / "README.md").read_text(encoding="utf-8")
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, flags=re.S | re.M)
    if not match:
        raise ValueError(f"README section {title!r} not found")
    return match.group(1)


def render(body: str, path: Path, footer: str) -> None:
    html = (
        f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"
        f"<div id='footer'>{footer} — page <pdf:pagenumber/> / <pdf:pagecount/></div>"
        f"{body}</body></html>"
    )
    with path.open("wb") as f:
        status = pisa.CreatePDF(html, dest=f, encoding="utf-8")
    if status.err:
        raise RuntimeError(f"xhtml2pdf reported {status.err} error(s) for {path.name}")
    print(f"wrote {path.relative_to(REPO)}")


def figures_html() -> str:
    parts = []
    for name, caption in KEY_FIGURES:
        parts.append(
            f'<div><img src="{(FIGURES / name).as_posix()}" width="480"/></div>'
            f'<p class="caption">{caption} (figures/{name})</p>'
        )
    return "".join(parts)


def read(path: str | Path) -> str:
    return (REPO / path).read_text(encoding="utf-8")


def build_markdown(doc: dict[str, Any]) -> None:
    """One Markdown file rendered as one PDF."""
    render(md_to_html(read(doc["source"])), REPO / doc["output"], doc["footer"])


def build_guide(doc: dict[str, Any]) -> None:
    """Title page, table of contents, walkthrough, results, limitations, extra parts, appendix."""
    body = (
        "<div class='title'><h1>EpiDetect</h1>"
        "<p class='subtitle'>A reproducible benchmark of high-frequency oscillation detectors"
        f"<br/>{doc.get('subtitle', '')}</p><p>{doc.get('author', '')}</p></div>"
        "<pdf:nextpage/><h1>Table of contents</h1><pdf:toc/><pdf:nextpage/>"
        "<h1>Part 1 – Project walkthrough</h1>"
        + md_to_html(read("docs/PROJECT_WALKTHROUGH.md"), shift=1)
        + "<pdf:nextpage/><h1>Part 2 – Key results</h1>"
        + "<p>Tables written by scripts/14_results_summary.py from the result files.</p>"
        + md_to_html(read("results/summary_tables.md"))
        + "<h2>Main findings</h2>"
        + md_to_html(readme_section("Main findings"))
        + "<h2>Key figures</h2>"
        + figures_html()
        + "<pdf:nextpage/><h1>Part 3 – Limitations and future work</h1>"
        + md_to_html(readme_section("Limitations"))
        + "<h2>Future work</h2>"
        + md_to_html(readme_section("Future work"))
    )
    for part in doc.get("extra_parts", []):
        body += f"<pdf:nextpage/><h1>{part['heading']}</h1>" + md_to_html(
            read(part["source"]), shift=1
        )
    body += "<pdf:nextpage/><h1>Appendix – Data notes</h1>" + md_to_html(
        read("docs/DATA_NOTES.md"), shift=1
    )
    render(body, REPO / doc["output"], doc["footer"])


def main() -> None:
    render(
        md_to_html(read("docs/PROJECT_WALKTHROUGH.md")),
        DOCS / "PROJECT_WALKTHROUGH.pdf",
        "EpiDetect – Walkthrough",
    )
    if not LOCAL_DOCUMENTS.exists():
        return
    builders = {"markdown": build_markdown, "guide": build_guide}
    with LOCAL_DOCUMENTS.open("rb") as f:
        for doc in tomllib.load(f).get("document", []):
            builders[doc["kind"]](doc)


if __name__ == "__main__":
    main()
