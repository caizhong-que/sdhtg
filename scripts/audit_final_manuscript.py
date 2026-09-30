# -*- coding: utf-8 -*-
"""audit_final_manuscript.py -- inventory of the final (SN template) manuscript."""

from __future__ import annotations

import io
import re


PATH = (r"E:\SDHTG\文章\Download+the+journal+article+template+package+"
        r"(December+2024+version)\sn-article-template\sn-article.tex")


def captions(pattern, text):
    out = []
    for block in re.findall(pattern, text, re.S):
        match = re.search(r"\\caption\{(.{0,160})", block, re.S)
        include = re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]*)\}", block)
        out.append((include.group(1) if include else "", 
                    match.group(1).replace("\n", " ") if match else "(no caption)"))
    return out


def main():
    text = io.open(PATH, encoding="cp1252", errors="replace").read()
    tables = captions(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", text)
    figures = captions(r"\\begin\{figure\*?\}(.*?)\\end\{figure\*?\}", text)
    print("final manuscript: {} tables, {} figures".format(len(tables), len(figures)))
    print("")
    for index, (_, caption) in enumerate(tables, 1):
        print("  T{:<2} {}".format(index, caption[:120]))
    print("")
    for index, (include, caption) in enumerate(figures, 1):
        print("  F{:<2} [{}] {}".format(index, include, caption[:100]))


if __name__ == "__main__":
    main()
