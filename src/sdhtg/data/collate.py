from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


SEQUENCE_LONG_FIELDS = (
    "template_ids",
    "entity_ids",
    "action_ids",
    "status_ids",
)
SEQUENCE_FLOAT_FIELDS = (
    "delta_t",
    "action_change",
    "entity_change",
)


def _validate_row(row: dict[str, Any], index: int) -> int:
    required = {
        *SEQUENCE_LONG_FIELDS,
        *SEQUENCE_FLOAT_FIELDS,
        "label",
        "sample_id",
        "session_id",
    }
    missing = required - row.keys()
    if missing:
        raise KeyError(f"row {index} is missing fields: {sorted(missing)}")

    length = len(row["template_ids"])
    if length <= 0:
        raise ValueError(f"row {index} has no events")
    for field in (*SEQUENCE_LONG_FIELDS, *SEQUENCE_FLOAT_FIELDS):
        if len(row[field]) != length:
            raise ValueError(
                f"row {index} field {field!r} has length {len(row[field])}, "
                f"expected {length}"
            )
    if int(row["label"]) not in (-1, 0, 1):
        raise ValueError(f"row {index} label must be -1/0/1 (unlabeled/binary)")
    if any(float(value) < 0 for value in row["delta_t"]):
        raise ValueError(f"row {index} has a negative delta_t")
    return length


def collate_sessions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot collate an empty batch")

    lengths = [_validate_row(row, index) for index, row in enumerate(rows)]
    batch_size = len(rows)
    maximum_length = max(lengths)

    mask = torch.zeros(batch_size, maximum_length, dtype=torch.bool)
    long_tensors = {
        field: torch.zeros(batch_size, maximum_length, dtype=torch.long)
        for field in SEQUENCE_LONG_FIELDS
    }
    float_tensors = {
        field: torch.zeros(batch_size, maximum_length, dtype=torch.float32)
        for field in SEQUENCE_FLOAT_FIELDS
    }

    for batch_index, (row, length) in enumerate(zip(rows, lengths)):
        mask[batch_index, :length] = True
        for field in SEQUENCE_LONG_FIELDS:
            long_tensors[field][batch_index, :length] = torch.tensor(
                row[field], dtype=torch.long
            )
        for field in SEQUENCE_FLOAT_FIELDS:
            float_tensors[field][batch_index, :length] = torch.tensor(
                row[field], dtype=torch.float32
            )

    # Public names used by SDHTG.forward.
    batch = {
        "template_id": long_tensors["template_ids"],
        "entity_id": long_tensors["entity_ids"],
        "action_id": long_tensors["action_ids"],
        "status_id": long_tensors["status_ids"],
        "delta_t": float_tensors["delta_t"],
        "action_change": float_tensors["action_change"],
        "entity_change": float_tensors["entity_change"],
        "mask": mask,
        "length": torch.as_tensor(lengths, dtype=torch.long),
        "label": torch.as_tensor(
            [int(row["label"]) for row in rows], dtype=torch.float32
        ),
        "sample_id": [str(row["sample_id"]) for row in rows],
        "session_id": [str(row["session_id"]) for row in rows],
    }
    return batch


def move_batch_to_device(
    batch: dict[str, Any], device: torch.device | str
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, Tensor)
        else value
        for key, value in batch.items()
    }
