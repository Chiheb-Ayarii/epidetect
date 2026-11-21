"""Audit: is every number of the README found in an output of the scripts? (a helper, not a proof)

Collects all numeric values of results/*.csv, results/summary_tables.md, docs/data_inventory.md
and configs/default.toml (with their roundings to 0-3 decimals and their x100 percentages),
then lists README numbers that match none of them, and the source of every match. The pool of
values is large, so small integers and years match by chance: the output supports a manual
review of every number, it does not replace it. The results tables between the RESULTS
markers are skipped (written by scripts/14_results_summary.py).

Usage:
    python scripts/check_readme_numbers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
NUMBER = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w])")


def variants(value: float) -> set[str]:
    out = set()
    for v in (value, 100 * value):
        for digits in range(4):
            text = f"{v:.{digits}f}"
            out.add(text)
            out.add(text.rstrip("0").rstrip(".") if "." in text else text)
    return out


def known_numbers() -> dict[str, str]:
    """Map every textual variant of every number in the script outputs to its first source."""
    known: dict[str, str] = {}

    def add(value: float, source: str) -> None:
        for v in variants(value):
            known.setdefault(v, source)

    # Small, curated tables first, so that the reported source is the most meaningful one.
    for path in [
        REPO / "results" / "summary_tables.md",
        REPO / "docs" / "data_inventory.md",
        REPO / "configs" / "default.toml",
    ]:
        for token in NUMBER.findall(path.read_text(encoding="utf-8")):
            add(float(token.replace(",", "")), path.name)
    for csv in sorted((REPO / "results").glob("*.csv"), key=lambda p: p.stat().st_size):
        table = pd.read_csv(csv)
        for column in table.columns:
            for v in pd.to_numeric(table[column], errors="coerce").dropna().unique():
                add(float(v), csv.name)
        add(float(len(table)), f"{csv.name} (row count)")
    return known


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # the README contains non-ASCII characters
    known = known_numbers()
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    generated = readme.split("<!-- RESULTS:START -->")[1].split("<!-- RESULTS:END -->")[0]
    unmatched, matched = [], []
    for line_no, line in enumerate(readme.splitlines(), start=1):
        if line and line in generated:
            continue  # results tables are written by scripts/14_results_summary.py
        for token in NUMBER.findall(line):
            plain = token.replace(",", "")
            # Identifiers such as patient "06" or script "01" are compared as integers.
            if plain.isdigit() and plain.startswith("0") and len(plain) > 1:
                plain = str(int(plain))
            if plain not in known:
                unmatched.append((line_no, token, line.strip()[:110]))
            else:
                matched.append((line_no, token, known[plain]))
    print(f"{len(unmatched)} README numbers not found in script outputs (review each):")
    for line_no, token, context in unmatched:
        print(f"  line {line_no:4d}: {token:>8s} | {context}")
    print(f"\n{len(matched)} numbers matched (line, number, source of the match):")
    for line_no, token, source in matched:
        print(f"  line {line_no:4d}: {token:>8s} <- {source}")


if __name__ == "__main__":
    main()
