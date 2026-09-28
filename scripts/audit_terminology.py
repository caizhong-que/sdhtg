# -*- coding: utf-8 -*-
"""audit_terminology.py -- scan the manuscript for naming inconsistencies.

The manuscript mixes several near-synonyms that must be unified for a
high-level journal submission (baselines, session units, metrics, model
variants, symbol names).  This script counts the candidate variants and prints
the lines where each occurs, so the fixes are evidence-based instead of
guesswork.

Usage:
    python scripts/audit_terminology.py
"""

from __future__ import annotations

import io
import re
from collections import Counter


PATH = r"E:\SDHTG\文章\初稿\【1】Manuscript.tex"

GROUPS = {
    "基线命名": ["统一协议基线", "统一输入协议", "平铺强基线", "平铺基线", "强基线"],
    "样本单位": ["外层样本", "会话", "块", "样本内部", "会话级"],
    "指标口径": ["阈值校准", "阈值口径", "阈值无关", "排序能力", "排序性能"],
    "模型变体": ["完整模型", "主模型", "L7", "L7b", "L0", "SDHTG 模型"],
    "GRU 基线": ["GRU-flat", "平铺 GRU", "GRU\\-flat", "基础 GRU"],
    "边界指标": ["Boundary F1", "边界 F1", "边界定位质量"],
    "温度符号": ["\\tau_b", "\\tau_{\\mathrm{final}}", "边界温度", "最终温度"],
    "原型符号": ["\\lambda_p", "\\tau_p", "原型尺度", "原型温度"],
    "英文残留": [" run", "run ", "dataset", "epoch", "baseline"],
}


def main() -> None:
    text = io.open(PATH, encoding="utf-8").read()
    lines = text.split("\n")
    for group, terms in GROUPS.items():
        print(f"===== {group} =====")
        counts = Counter()
        locations: dict[str, list[int]] = {}
        for term in terms:
            pattern = re.compile(re.escape(term))
            for number, line in enumerate(lines, 1):
                if pattern.search(line):
                    counts[term] += 1
                    locations.setdefault(term, []).append(number)
        for term, count in counts.most_common():
            numbers = locations[term]
            preview = f"L{numbers[0]}" if len(numbers) == 1 else f"L{numbers[0]}..L{numbers[-1]} ({len(numbers)} 处)"
            print(f"  {term:<28} {count:>3}  {preview}")
        missing = [term for term in terms if term not in counts]
        if missing:
            print(f"  (未出现: {', '.join(missing)})")


if __name__ == "__main__":
    main()
