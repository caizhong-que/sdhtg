from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class FileSpec:
    path: str
    role: str
    url: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class DataConfig:
    name: str
    adapter: str
    raw_dir: Path
    processed_dir: Path
    files: tuple[FileSpec, ...]
    log_format: str
    encoding: str = "utf-8"
    timestamp_fields: tuple[str, ...] = ("Timestamp",)
    timestamp_formats: tuple[str, ...] = ()
    entity_fields: tuple[str, ...] = ()
    session_fields: tuple[str, ...] = ()
    content_field: str = "Content"
    label_field: str | None = "Label"
    normal_labels: tuple[str, ...] = ("-", "0", "normal", "Normal")
    anomaly_labels: tuple[str, ...] = ()
    label_join_key: str | None = None
    label_file_key: str | None = None
    split: tuple[float, float, float] = (0.6, 0.2, 0.2)
    sessionization: str = "native"
    fixed_window_size: int = 100
    idle_gap_seconds: float = 60.0
    adaptive_idle_multiplier: float = 2.0
    max_session_length: int = 512
    drain: dict[str, Any] = field(default_factory=dict)
    regex: dict[str, str] = field(default_factory=dict)

    def file_by_role(self, role: str) -> FileSpec:
        matches = [x for x in self.files if x.role == role]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one file with role={role!r}, got {len(matches)}")
        return matches[0]


def load_config(path: str | Path) -> DataConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = {"name", "adapter", "raw_dir", "processed_dir", "files", "log_format"}
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"configuration missing keys: {sorted(missing)}")
    files = tuple(FileSpec(**item) for item in raw.pop("files"))
    raw["files"] = files
    raw["raw_dir"] = Path(raw["raw_dir"])
    raw["processed_dir"] = Path(raw["processed_dir"])
    for key in ("timestamp_fields", "timestamp_formats", "entity_fields", "session_fields",
                "normal_labels", "anomaly_labels", "split"):
        if key in raw:
            raw[key] = tuple(raw[key])
    cfg = DataConfig(**raw)
    if abs(sum(cfg.split) - 1.0) > 1e-8 or any(x <= 0 for x in cfg.split):
        raise ValueError("split must contain three positive ratios summing to one")
    if cfg.sessionization not in {"native", "fixed_window", "idle_gap", "adaptive_idle_gap"}:
        raise ValueError(f"unsupported sessionization: {cfg.sessionization}")
    return cfg
