# -*- coding: utf-8 -*-
"""Print the dataset-role table (size, session length, difficulty)."""
import json
from pathlib import Path


def main() -> None:
    for dataset in ["ssh", "hdfs", "bgl", "openstack", "thunderbird"]:
        report = json.loads(
            Path(f"data/processed/{dataset}/quality_report.json").read_text(
                encoding="utf-8"
            )
        )
        test = report["splits"]["test"]
        train = report["splits"]["train"]
        sessions = report["sessions"]
        events = report["events"]
        print(
            f"{dataset:<12} sessions={sessions:>7d} events={events:>9d} "
            f"avg_len={events / sessions:6.1f} entities={report['entities']:>6d} "
            f"train_anom_rate={train['anomalous_sessions'] / train['sessions']:.4f} "
            f"test_anom={test['anomalous_sessions']:>5d}/{test['sessions']:<6d}"
        )


if __name__ == "__main__":
    main()
