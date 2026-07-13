from __future__ import annotations

from typing import Any

import pandas as pd
import torch


class SessionDataset(torch.utils.data.Dataset):
    """Reads preprocessed session chunks from a Parquet file for a given split.

    The Parquet file is produced by ``materialize_sessions`` in
    :mod:`sdhtg.data.cache` and contains one row per chunk, with fields
    that match the expectations of :func:`sdhtg.data.collate.collate_sessions`.

    Parameters
    ----------
    path: str
        Path to ``sessions.parquet``.
    split: str
        One of ``"train"``, ``"validation"``, or ``"test"``.
    """

    def __init__(self, path: str, split: str) -> None:
        super().__init__()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"split must be train/validation/test, got {split!r}")
        self._frame: pd.DataFrame = pd.read_parquet(path)
        self._frame = self._frame[self._frame.split == split].reset_index(drop=True)
        if self._frame.empty:
            raise ValueError(f"split {split!r} produced zero samples in {path}")

    @property
    def rows(self):
        """Lazy row-like iteration; returns a fresh iterator each time."""
        class _Rows:
            def __init__(self, frame):
                self._frame = frame
            def __len__(self):
                return len(self._frame)
            def __getitem__(self, i):
                return self._frame.iloc[i].to_dict()
            def __iter__(self):
                for i in range(len(self._frame)):
                    yield self._frame.iloc[i].to_dict()
        return _Rows(self._frame)

    @property
    def lengths(self):
        """Per-sample sequence lengths for batch bucketing."""
        return [len(x) for x in self._frame["template_ids"]]

    def __len__(self) -> int:
        return len(self._frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._frame.iloc[index].to_dict()
