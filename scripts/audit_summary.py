# -*- coding: utf-8 -*-
"""Print a compact summary of all regenerated caches and experiment outputs."""
import json
from pathlib import Path


def main() -> None:
    for dataset in ("bgl", "hdfs", "openstack", "ssh", "thunderbird"):
        report = json.loads(
            Path(f"data/processed/{dataset}/quality_report.json").read_text(
                encoding="utf-8"
            )
        )
        test = report["splits"]["test"]
        validation = report["splits"]["validation"]
        print(
            f"{dataset:<12} sessions={report['sessions']:>9d} "
            f"templates={report['templates']:>5d} entities={report['entities']:>6d} "
            f"test_unseen={test['unseen_template_rate']:.4f} "
            f"test_anom={test['anomalous_sessions']} "
            f"val_anom={validation['anomalous_sessions']}"
        )


if __name__ == "__main__":
    main()
