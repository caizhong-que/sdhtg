from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataConfig


def estimate_adaptive_idle_thresholds(
    events: pd.DataFrame,
    cfg: DataConfig,
) -> dict[str, float | dict[str, float]]:
    """Estimate per-entity idle-gap thresholds using only training-period events.

    For temporal splits the training window is approximated by the first
    ``split[0]`` fraction of events in time order (the final split is by
    session start time; the two agree up to session-length effects). For
    random session splits (``split_seed`` set) a random ``split[0]`` fraction
    of events is used instead, matching the random training split in
    expectation. Either way, validation/test statistics never enter the
    threshold estimate.

    Returns ``{"per_entity": {entity: seconds}, "global": seconds}``.
    """
    if cfg.split_seed is not None:
        window = events.sample(frac=cfg.split[0], random_state=cfg.split_seed)
    else:
        ordered = events.sort_values(
            ["timestamp", "source_event_id"], kind="mergesort"
        )
        cutoff = max(1, int(len(ordered) * cfg.split[0]))
        window = ordered.iloc[:cutoff].copy()
    window = window.sort_values(["entity", "timestamp", "source_event_id"], kind="mergesort")
    gap = window.groupby("entity", sort=False).timestamp.diff().dt.total_seconds()
    positive = gap.where(gap > 0)
    global_median = positive.median()
    if pd.isna(global_median):
        global_median = float(cfg.idle_gap_seconds)
    # Group-wise medians keyed by ENTITY (not a transform-aligned series,
    # whose to_dict() would be keyed by event index and match nothing).
    group_medians = positive.groupby(window.entity, sort=False).median()
    maximum = float(cfg.adaptive_idle_max_seconds)
    per_entity = (
        (group_medians.fillna(global_median) * cfg.adaptive_idle_multiplier)
        .clip(upper=maximum)
        .to_dict()
    )
    return {
        "per_entity": {str(k): float(v) for k, v in per_entity.items()},
        "global": float(
            min(global_median * cfg.adaptive_idle_multiplier, maximum)
        ),
    }


def _boundary_cumsum(frame: pd.DataFrame, boundary: pd.Series, prefix: str) -> pd.Series:
    number = boundary.astype(np.int64).groupby(frame["entity"], sort=False).cumsum()
    return frame["entity"].astype(str) + ":" + prefix + ":" + number.astype(str)


def assign_sessions(
    events: pd.DataFrame,
    cfg: DataConfig,
    idle_thresholds: dict[str, float | dict[str, float]] | None = None,
) -> pd.DataFrame:
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
            if idle_thresholds is None:
                raise ValueError(
                    "adaptive_idle_gap requires precomputed training-window "
                    "thresholds (see estimate_adaptive_idle_thresholds)"
                )
            per_entity = idle_thresholds["per_entity"]
            threshold = frame.entity.map(per_entity).fillna(
                idle_thresholds["global"]
            )
            threshold = threshold.clip(upper=cfg.adaptive_idle_max_seconds)
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
