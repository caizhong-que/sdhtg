# -*- coding: utf-8 -*-
"""check_math_shifts.py -- lint for unbalanced inline math in the manuscript.

Writing `$0.9565；` instead of `$0.9565$；` silently puts the rest of the
paragraph into math mode; xeCJK then typesets the Chinese with the Latin font
and LaTeX reports "Missing $ inserted" far away from the real cause.  This lint
reports every line with an odd number of unescaped `$`, ignoring comments and
verbatim-like environments, so the mistake is caught before compiling.

Usage:
    python scripts/check_math_shifts.py [--path <manuscript.tex>]
"""

from __future__ import annotations

import argparse
import io
import re


VERBATIM_ENVIRONMENTS = ("verbatim", "lstlisting", "minted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=r"E:\SDHTG\文章\初稿\【1】Manuscript.tex")
    args = parser.parse_args()

    lines = io.open(args.path, encoding="utf-8").read().split("\n")
    inside_verbatim = False
    offenders = []
    for number, line in enumerate(lines, 1):
        stripped = line.strip()
        for environment in VERBATIM_ENVIRONMENTS:
            if stripped.startswith(f"\\begin{{{environment}}}"):
                inside_verbatim = True
            elif stripped.startswith(f"\\end{{{environment}}}"):
                inside_verbatim = False
        if inside_verbatim or stripped.startswith("%"):
            continue
        # `\$` is an escaped dollar sign, `\\` does not escape a following `$`.
        cleaned = re.sub(r"(?<!\\)\\\$", "", line)
        cleaned = re.sub(r"\\\\", "", cleaned)
        if cleaned.count("$") % 2:
            offenders.append((number, cleaned.count("$"), line.strip()[:110]))

    if offenders:
        print(f"[fail] {len(offenders)} line(s) with an odd number of '$':")
        for number, count, preview in offenders:
            print(f"  line {number}: {count} '$' -- {preview}")
        raise SystemExit(1)
    print(f"[ok] {len(lines)} lines, every line has a balanced number of '$'")


if __name__ == "__main__":
    main()
