from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataConfig


def _boundary_cumsum(frame: pd.DataFrame, boundary: pd.Series, prefix: str) -> pd.Series:
    number = boundary.astype(np.int64).groupby(frame["entity"], sort=False).cumsum()
    return frame["entity"].astype(str) + ":" + prefix + ":" + number.astype(str)


def assign_sessions(events: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    frame = events.sort_values(["entity", "timestamp", "source_event_id"], kind="mergesort").copy()
    if cfg.sessionization == "native":
        if "native_session_id" not in frame:
            raise ValueError("native sessionization requested but adapter produced no native_session_id")
        frame["session_id"] = frame["native_session_id"].astype(str)
    elif cfg.sessionization == "fixed_window":
        order = frame.groupby("entity", sort=False).cumcount()
        frame["session_id"] = frame.entity.astype(str) + ":fw:" + (
            order // cfg.fixed_window_size
        ).astype(str)
    else:
        gap = frame.groupby("entity", sort=False).timestamp.diff().dt.total_seconds()
        if (gap.dropna() < 0).any():
            raise ValueError("negative time gap after entity sorting")
        if cfg.sessionization == "idle_gap":
            threshold = pd.Series(cfg.idle_gap_seconds, index=frame.index)
        else:
            positive = gap.where(gap > 0)
            medians = positive.groupby(frame.entity, sort=False).transform("median")
            global_median = positive.median()
            if pd.isna(global_median):
                global_median = cfg.idle_gap_seconds
            threshold = medians.fillna(global_median) * cfg.adaptive_idle_multiplier
        boundary = gap.isna() | (gap > threshold)
        frame["session_id"] = _boundary_cumsum(frame, boundary, "ig")
    # Aggregate labels after grouping. Any anomalous event makes the session anomalous.
    if "session_label" not in frame:
        frame["session_label"] = frame.groupby("session_id").event_label.transform("max")
    else:
        consistency = frame.groupby("session_id").session_label.nunique()
        if (consistency > 1).any():
            raise ValueError("a native session has conflicting labels")
    return frame.sort_values(["timestamp", "source_event_id"], kind="mergesort").reset_index(drop=True)
