# -*- coding: utf-8 -*-
"""audit_table_inventory.py -- table inventory with size and reference counts.

For every table environment in the manuscript this prints the order, label,
number of rows, where it sits (section), and how many times the body refers to
it.  That is the evidence needed to decide which tables belong in the main text
and which are appendix material.

Usage:
    python scripts/audit_table_inventory.py
"""

from __future__ import annotations

import io
import re


PATH = r"E:\SDHTG\文章\初稿\【1】Manuscript.tex"


def main() -> None:
    text = io.open(PATH, encoding="utf-8").read()
    lines = text.split("\n")
    section = ""
    records = []
    pending = None
    for number, line in enumerate(lines, 1):
        section_match = re.match(r"\t*\\(section|subsection)\{([^}]*)\}", line)
        if section_match:
            section = section_match.group(2).split("\\label")[0]
        if "\\begin{table}" in line:
            pending = {"start": number, "caption": "", "label": "", "rows": 0}
        elif pending is not None:
            caption = re.search(r"\\caption\{([^}]{0,80})", line)
            if caption:
                pending["caption"] = caption.group(1)
            label = re.search(r"\\label\{(tab:[^}]+)\}", line)
            if label:
                pending["label"] = label.group(1)
            if line.strip().endswith("\\\\") and "toprule" not in line:
                pending["rows"] += 1
            if "\\end{table}" in line:
                pending["section"] = section
                records.append(pending)
                pending = None

    print(f"{'#':>3} {'label':<28} {'rows':>4} {'refs':>5}  section / caption")
    for index, record in enumerate(records, 1):
        refs = 0
        if record["label"]:
            for line in lines:
                if f"\\ref{{{record['label']}}}" in line:
                    refs += line.count(f"\\ref{{{record['label']}}}")
        print(f"{index:>3} {record['label']:<28} {record['rows']:>4} {refs:>5}  "
              f"{record['section'][:22]} | {record['caption'][:52]}")
    print(f"\ntotal tables: {len(records)}")


if __name__ == "__main__":
    main()
