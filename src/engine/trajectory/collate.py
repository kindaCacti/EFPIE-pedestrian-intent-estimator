"""Canonical batching with left-padded observations and preserved masks."""

import torch


def trajectory_collate(samples):
    if not samples:
        raise ValueError("Cannot collate an empty trajectory batch")
    batch = {key: [s[key] for s in samples] for key in ("sequence_id", "track_id", "metadata")}
    for key in ("observed_points", "observed_boxes", "observed_time_deltas", "observed_mask",
                "future_points", "future_boxes", "future_mask", "future_time_deltas"):
        if not any(key in s for s in samples):
            continue
        if not all(key in s for s in samples):
            raise ValueError(f"Inconsistent canonical batch: {key} missing from some samples")
        values = [torch.as_tensor(s[key], dtype=torch.bool if key.endswith("mask") else torch.float32) for s in samples]
        length = max(len(v) for v in values)
        if key.startswith("future") and any(len(v) != length for v in values):
            raise ValueError("Future horizons must agree within a batch")
        padded = []
        for value in values:
            pad = value.new_zeros((length - len(value), *value.shape[1:]))
            padded.append(torch.cat([pad, value]))
        batch[key] = torch.stack(padded)
    return batch
