from __future__ import annotations

import csv
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DataConfig
from .format_parser import LogFormatParser


BLOCK_RE = re.compile(r"\bblk_-?\d+\b")
IP_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,36}\b", re.I)
REQUEST_RE = re.compile(r"\breq-[0-9a-f-]+\b", re.I)


class DatasetAdapter(ABC):
    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.parser = LogFormatParser(cfg.log_format)

    def read_log_rows(self) -> Iterable[dict]:
        spec = self.cfg.file_by_role("log")
        return self.parser.parse_file(self.cfg.raw_dir / spec.path, self.cfg.encoding)

    def parse_timestamp(self, row: dict) -> pd.Timestamp:
        values = [row.get(field, "") for field in self.cfg.timestamp_fields]
        value = " ".join(x for x in values if x).strip()
        if not value:
            raise ValueError(f"empty timestamp at source line {row.get('source_line')}")
        for fmt in self.cfg.timestamp_formats:
            try:
                return pd.to_datetime(value, format=fmt).tz_localize("UTC")
            except ValueError:
                pass
        if re.fullmatch(r"\d+(?:\.\d+)?", value):
            unit = "ms" if float(value) > 10_000_000_000 else "s"
            return pd.to_datetime(float(value), unit=unit, utc=True)
        try:
            return pd.to_datetime(value, utc=True, errors="raise")
        except Exception as exc:
            raise ValueError(f"invalid timestamp {value!r} at line {row.get('source_line')}") from exc

    def entity(self, row: dict) -> str:
        values = [str(row.get(x, "")).strip() for x in self.cfg.entity_fields]
        values = [x for x in values if x and x != "-"]
        return "|".join(values) if values else "UNK"

    def explicit_label(self, value: object) -> int:
        text = str(value).strip()
        if text in self.cfg.normal_labels:
            return 0
        if self.cfg.anomaly_labels and text in self.cfg.anomaly_labels:
            return 1
        if not self.cfg.anomaly_labels and text and text not in self.cfg.normal_labels:
            return 1
        raise ValueError(f"unknown label {text!r}; configure normal_labels/anomaly_labels")

    @abstractmethod
    def normalize(self) -> pd.DataFrame:
        raise NotImplementedError

    def base_event(self, row: dict, event_id: int) -> dict:
        content = str(row.get(self.cfg.content_field, "")).strip()
        if not content:
            raise ValueError(f"empty Content at source line {row.get('source_line')}")
        return {"source_event_id": event_id, "source_line": int(row["source_line"]),
                "timestamp": self.parse_timestamp(row), "content": content,
                "entity": self.entity(row)}


class LineLabelAdapter(DatasetAdapter):
    """BGL and Thunderbird: official line label, native node/entity stream."""
    JOB_UNIT_RE = re.compile(r":J\d+-U\d+$")
    def entity(self, row):
        e = super().entity(row)
        return self.JOB_UNIT_RE.sub("", e)

    def normalize(self) -> pd.DataFrame:
        if not self.cfg.label_field:
            raise ValueError("line-labelled adapter requires label_field")
        events = []
        for event_id, row in enumerate(self.read_log_rows()):
            event = self.base_event(row, event_id)
            event["event_label"] = self.explicit_label(row.get(self.cfg.label_field, ""))
            events.append(event)
        return finalize_events(events)


class HDFSAdapter(DatasetAdapter):
    def load_labels(self) -> dict[str, int]:
        spec = self.cfg.file_by_role("labels")
        path = self.cfg.raw_dir / spec.path
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        key = self.cfg.label_file_key or "BlockId"
        label = self.cfg.label_field or "Label"
        if key not in frame or label not in frame:
            raise ValueError(f"HDFS label file must contain {key!r} and {label!r}")
        result = {}
        for _, row in frame.iterrows():
            block = str(row[key]).strip()
            if not BLOCK_RE.fullmatch(block):
                raise ValueError(f"invalid BlockId in label file: {block!r}")
            value = self.explicit_label(row[label])
            if block in result and result[block] != value:
                raise ValueError(f"conflicting labels for {block}")
            result[block] = value
        return result

    def normalize(self) -> pd.DataFrame:
        labels = self.load_labels()
        events = []
        missing = set()
        for event_id, row in enumerate(self.read_log_rows()):
            base = self.base_event(row, event_id)
            blocks = sorted(set(BLOCK_RE.findall(base["content"])))
            if not blocks:
                continue
            for block in blocks:
                if block not in labels:
                    missing.add(block)
                    continue
                event = dict(base)
                event.update(entity=block, native_session_id=block,
                             event_label=labels[block], session_label=labels[block])
                events.append(event)
        if missing:
            sample = sorted(missing)[:20]
            raise ValueError(f"{len(missing)} HDFS blocks have no official label; examples: {sample}")
        return finalize_events(events)


class ExternalLabelAdapter(DatasetAdapter):
    """OpenStack/SSH: explicit in-log labels or an external exact-key label table."""
    # PID->IP cache for SSH: sshd[PID] uniquely maps to a client IP.
    _pid_ip_cache: dict[str, str] = {}
    def load_external_labels(self) -> dict[str, int] | None:
        specs = [x for x in self.cfg.files if x.role == "labels"]
        if not specs:
            return None
        if not self.cfg.label_join_key or not self.cfg.label_file_key or not self.cfg.label_field:
            raise ValueError("external labels require label_join_key, label_file_key and label_field")
        frame = pd.read_csv(self.cfg.raw_dir / specs[0].path, dtype=str, keep_default_na=False)
        if self.cfg.label_file_key not in frame or self.cfg.label_field not in frame:
            raise ValueError("external label file does not contain configured columns")
        result = {}
        for _, row in frame.iterrows():
            key = str(row[self.cfg.label_file_key]).strip()
            value = self.explicit_label(row[self.cfg.label_field])
            if key in result and result[key] != value:
                raise ValueError(f"conflicting labels for key {key!r}")
            result[key] = value
        return result

    def derive_session(self, row: dict, entity: str, event_id: int) -> str:
        values = [str(row.get(x, "")).strip() for x in self.cfg.session_fields]
        values = [x for x in values if x and x != "-"]
        if values:
            return "|".join(values)
        # Search ADDR (OpenStack request UUID) then Content for session IDs.
        addr = str(row.get("ADDR", "")).strip()
        if addr:
            match = REQUEST_RE.search(addr)
            if match:
                return match.group(0).lower()
        content = str(row.get(self.cfg.content_field, ""))
        if not content:
            return f"event:{entity}:{event_id}"
        pid = str(row.get("Pid", "")).strip()
        ip_match = IP_RE.search(content)
        if ip_match:
            ip = ip_match.group(0).lower()
            if pid:
                self._pid_ip_cache[pid] = ip
            return ip
        if pid and pid in self._pid_ip_cache:
            return self._pid_ip_cache[pid]
        for regex in (REQUEST_RE, UUID_RE):
            match = regex.search(content)
            if match:
                return match.group(0).lower()
        return f"event:{entity}:{event_id}"

    def normalize(self) -> pd.DataFrame:
        labels = self.load_external_labels()
        events = []
        for event_id, row in enumerate(self.read_log_rows()):
            event = self.base_event(row, event_id)
            session = self.derive_session(row, event["entity"], event_id)
            event["native_session_id"] = session
            if labels is not None:
                join = str(row.get(self.cfg.label_join_key, session)).strip()
                event["event_label"] = labels.get(join, 0)
            elif self.cfg.label_field and self.cfg.label_field in row:
                event["event_label"] = self.explicit_label(row[self.cfg.label_field])
            else:
                raise ValueError("OpenStack/SSH requires explicit label field or external label file")
            events.append(event)
        return finalize_events(events)


def finalize_events(events: list[dict]) -> pd.DataFrame:
    if not events:
        raise ValueError("adapter produced zero labelled events")
    frame = pd.DataFrame(events)
    required = {"source_event_id", "source_line", "timestamp", "content", "entity", "event_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"normalized events missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("normalized events contain null required values")
    if not set(frame.event_label.unique()).issubset({0, 1}):
        raise ValueError("labels must be binary")
    return frame.sort_values(["timestamp", "source_event_id"], kind="mergesort").reset_index(drop=True)


def create_adapter(cfg: DataConfig) -> DatasetAdapter:
    adapters = {
        "bgl": LineLabelAdapter,
        "thunderbird": LineLabelAdapter,
        "hdfs": HDFSAdapter,
        "openstack": ExternalLabelAdapter,
        "ssh": ExternalLabelAdapter,
    }
    if cfg.adapter not in adapters:
        raise ValueError(f"unknown adapter {cfg.adapter!r}; expected one of {sorted(adapters)}")
    return adapters[cfg.adapter](cfg)
