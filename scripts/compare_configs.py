# -*- coding: utf-8 -*-
"""Compare two resolved model configs field by field."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sdhtg.models.factory import load_model_config  # noqa: E402


def main() -> None:
    left, right = sys.argv[1], sys.argv[2]
    overrides = {
        "template_vocab_size": 4096,
        "entity_vocab_size": 1024,
        "action_vocab_size": 1024,
        "status_vocab_size": 4096,
    }
    a = load_model_config(left, overrides)
    b = load_model_config(right, overrides)
    differences = []
    for field in a.__dataclass_fields__:
        va, vb = getattr(a, field), getattr(b, field)
        if va != vb:
            differences.append((field, va, vb))
    print(f"comparing resolved configs:\n  A = {left}\n  B = {right}")
    if not differences:
        print("=> IDENTICAL (same model)")
    else:
        for field, va, vb in differences:
            print(f"  {field}:\n    A={va}\n    B={vb}")


if __name__ == "__main__":
    main()
