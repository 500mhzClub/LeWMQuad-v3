#!/usr/bin/env python3
"""Complete, durable, verified checkpointing for development training runs.

Two failures this exists to prevent, both of which have already happened here:

1. **Incomplete state.** A checkpoint holding only weights cannot reproduce the
   optimisation trajectory: without optimiser moments, the epoch index, the
   data-order generator and the RNG streams, a "resume" silently becomes a
   different experiment. ``save`` refuses to write unless every required item is
   supplied, and ``load_for_resume`` refuses to resume from an incomplete file.

2. **Not actually on disk.** ``torch.save`` returns before the bytes are durable.
   A crash between the call and the page-cache flush leaves a truncated or
   missing file exactly when it is needed most. ``save`` writes to a temporary
   file, ``flush``es, ``fsync``s the file, atomically ``os.replace``s it into
   position, then ``fsync``s the containing directory so the rename itself is
   durable, and finally reloads the result to confirm it is readable.

Everything needed to re-instantiate the model is stored alongside the weights, so
a checkpoint is self-describing rather than depending on the caller remembering
which architecture produced it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
import torch

SCHEMA = "lewm_dev_checkpoint_v1"
REQUIRED = (
    "schema", "model_state_dict", "optimizer_state_dict", "scheduler_state_dict",
    "epoch", "global_step", "seed", "model_config",
    "python_rng_state", "numpy_rng_state", "torch_cpu_rng_state",
    "torch_cuda_rng_state", "data_order_generator_state",
)


def collect_rng_states(data_order_generator: torch.Generator | None = None) -> dict:
    """Every stream that can influence the next step."""
    return {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "data_order_generator_state": (
            data_order_generator.get_state() if data_order_generator is not None else None
        ),
    }


def restore_rng_states(state: dict, data_order_generator: torch.Generator | None = None) -> None:
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if state.get("torch_cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda_rng_state"])
    if data_order_generator is not None:
        if state.get("data_order_generator_state") is None:
            raise RuntimeError("checkpoint carries no data-order generator state")
        data_order_generator.set_state(state["data_order_generator_state"])


def save(
    path: Path,
    *,
    model,
    optimizer,
    epoch: int,
    global_step: int,
    seed: int,
    model_config: dict,
    scheduler=None,
    data_order_generator: torch.Generator | None = None,
    extra: dict | None = None,
    scheduler_absent_reason: str | None = None,
) -> dict:
    """Write a complete checkpoint durably, then verify it reloads.

    ``scheduler`` may be None only when ``scheduler_absent_reason`` explains why
    (e.g. a fixed learning rate), so a missing scheduler is a recorded decision
    rather than an omission.
    """
    if scheduler is None and not scheduler_absent_reason:
        raise ValueError("scheduler is None: pass scheduler_absent_reason to record why")
    if data_order_generator is None:
        raise ValueError("data_order_generator is required: sample order must be resumable")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scheduler_absent_reason": scheduler_absent_reason,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "seed": int(seed),
        "model_config": dict(model_config),
        **collect_rng_states(data_order_generator),
    }
    if extra:
        payload.update(extra)

    missing = [k for k in REQUIRED if k not in payload]
    if missing:
        raise RuntimeError(f"refusing to write an incomplete checkpoint; missing {missing}")
    optimiser_entries = len(payload["optimizer_state_dict"].get("state", {}))
    if optimiser_entries == 0:
        raise RuntimeError(
            "optimizer state is empty; call save AFTER the first optimizer.step() "
            "or the moments cannot be restored"
        )

    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())          # bytes durable before the rename
    os.replace(temporary, path)            # atomic: readers see old or new, never partial
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)                # the rename itself is now durable
    finally:
        os.close(directory)

    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    absent = [k for k in REQUIRED if k not in reloaded]
    if absent:
        raise RuntimeError(f"checkpoint verification failed; absent after reload: {absent}")

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 22)
            if not block:
                break
            digest.update(block)
    receipt = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "optimizer_state_entries": optimiser_entries,
        "verified_reloadable": True,
        "durable": "fsync(file) + atomic replace + fsync(dir)",
    }
    receipts = path.parent / "checkpoint_receipts.jsonl"
    with open(receipts, "a") as handle:
        handle.write(json.dumps(receipt) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return receipt


def load_for_resume(path: Path, *, model, optimizer, scheduler=None,
                    data_order_generator: torch.Generator | None = None) -> dict:
    """Restore a complete checkpoint, or refuse."""
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if state.get("schema") != SCHEMA:
        raise RuntimeError(
            f"{path} is not a {SCHEMA} checkpoint (schema={state.get('schema')!r}); "
            "resuming from it would change the optimisation trajectory"
        )
    absent = [k for k in REQUIRED if k not in state]
    if absent:
        raise RuntimeError(f"incomplete checkpoint, refusing to resume; absent: {absent}")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None:
        if state["scheduler_state_dict"] is None:
            raise RuntimeError("a scheduler was supplied but the checkpoint holds no scheduler state")
        scheduler.load_state_dict(state["scheduler_state_dict"])
    restore_rng_states(state, data_order_generator)
    return state


def newest_resumable(directory: Path) -> Path | None:
    candidates = []
    for path in Path(directory).glob("checkpoint_epoch*.pt"):
        try:
            head = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        if head.get("schema") == SCHEMA and all(k in head for k in REQUIRED):
            candidates.append((int(head["epoch"]), path))
    return max(candidates)[1] if candidates else None
