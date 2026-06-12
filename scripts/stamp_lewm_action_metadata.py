#!/usr/bin/env python3
"""Stamp canonical active-block metadata into LeWM checkpoints.

Use this for checkpoints produced before ``scripts/train_lewm.py`` started
persisting ``action_metadata``. It does not alter model or optimizer tensors.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from lewm.actions import (
    ACTIVE_BLOCK_ORDER,
    active_block_metadata,
    assert_active_block_metadata_compatible,
)


def iter_checkpoints(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.glob("lewm_seq*_e*.pt")))
        else:
            out.append(path)
    return out


def stamp_checkpoint(path: Path, *, dry_run: bool) -> str:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        return "skip: not a trainer checkpoint dict"

    existing = checkpoint.get("action_metadata")
    if existing is not None:
        assert_active_block_metadata_compatible(existing)
        return f"ok: already {existing.get('active_block_order')}"

    if dry_run:
        return f"would stamp: {ACTIVE_BLOCK_ORDER}"

    checkpoint["action_metadata"] = active_block_metadata()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)
    return f"stamped: {ACTIVE_BLOCK_ORDER}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path, help="Checkpoint files or directories")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for checkpoint in iter_checkpoints(args.paths):
        print(f"{checkpoint}: {stamp_checkpoint(checkpoint, dry_run=args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
