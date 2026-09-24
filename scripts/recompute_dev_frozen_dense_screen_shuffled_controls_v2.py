#!/usr/bin/env python3
"""Recompute only the shuffled-token controls, deranging within each role.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

The v1 control used a single fixed-point-free derangement over the concatenated
train+selection rows, so a selection row could receive a train row's features and
vice versa.  That still measures fixed positional/frustum/decoder priors, but it
mixes the roles.  This recomputes the control with **separate** fixed-point-free
derangements *within* train and *within* checkpoint_selection.

Encoder extraction is not rerun -- the cached frozen features are reused against
their recorded hashes.  The main probes are not rerun and are not touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_frozen_dense_representation_screen_v1 as S  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402

DERANGEMENT_SEED_V2 = 2_026_080_631


def derangement(count: int, rng) -> np.ndarray:
    """Fixed-point-free permutation of ``range(count)``."""
    if count < 2:
        raise ValueError("a derangement needs at least two elements")
    order = np.arange(count)
    while True:
        rng.shuffle(order)
        if not (order == np.arange(count)).any():
            return order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="project_vit_update400,dinov2_vitl14,vjepa2_1_vitl_384")
    args = ap.parse_args()
    device = torch.device(args.device)
    started = time.time()

    result_path = S.OUT / "result.json"
    record = json.loads(result_path.read_text())

    rows = S.load_rows()
    train_rows = [r for r in rows if r["role"] == "train"]
    sel_rows = [r for r in rows if r["role"] == "checkpoint_selection"]
    ordered = train_rows + sel_rows
    if S.ordered_hash(ordered) != record["corpus"]["ordered_pair_sha256"]:
        raise RuntimeError("ordered corpus changed; cached features are not comparable")
    labels = S.load_targets(ordered)
    train_idx = np.arange(len(train_rows))
    sel_idx = np.arange(len(train_rows), len(ordered))

    # within-role derangements, in the same ordered index space as the cache
    rng = np.random.default_rng(DERANGEMENT_SEED_V2)
    derange = np.arange(len(ordered))
    derange[train_idx] = train_idx[derangement(len(train_idx), rng)]
    derange[sel_idx] = sel_idx[derangement(len(sel_idx), rng)]
    if (derange == np.arange(len(ordered))).any():
        raise RuntimeError("derangement has a fixed point")
    if not (np.sort(derange[train_idx]) == train_idx).all():
        raise RuntimeError("train derangement left its role")
    if not (np.sort(derange[sel_idx]) == sel_idx).all():
        raise RuntimeError("selection derangement left its role")

    for name in (a for a in args.arms.split(",") if a):
        entry = record["arms"][name]
        receipt = entry["feature_cache"]
        blob = Path(receipt["cache_path"])
        shape = tuple(receipt["token_shape"])
        if E.file_sha256(blob) != receipt["cache_sha256"]:
            raise RuntimeError(f"{name}: cached features no longer match their receipt")
        grid = tuple(receipt["preprocessing"]["token_grid_hw"])
        dim = int(receipt["preprocessing"]["token_dim"])

        memory = np.memmap(blob, dtype=np.float16, mode="r", shape=shape)
        features = torch.from_numpy(np.ascontiguousarray(memory)).to(device)
        model, history, epoch = S.train_probe(
            features, labels, train_idx, sel_idx, grid, dim, device,
            f"{name}/shuffled_within_role", feature_map=derange,
        )
        sel_pred = S.predict(model, features, sel_idx, grid, device, derange)
        train_pred = S.predict(model, features, train_idx, grid, device, derange)

        previous = entry["shuffled_token_control"]
        entry["shuffled_token_control"] = {
            "description": (
                "complete feature tensors deranged between observations; token positions "
                "preserved; SEPARATE fixed-point-free derangements within train and within "
                "checkpoint_selection, so no role receives another role's features"
            ),
            "derangement_seed": DERANGEMENT_SEED_V2,
            "within_role": True,
            "selected_epoch": epoch,
            "history": history,
            "train": S.summarise(train_pred, labels[train_idx]),
            "checkpoint_selection": S.summarise(sel_pred, labels[sel_idx]),
            "superseded_cross_role_control": {
                "description": previous["description"],
                "checkpoint_selection_observable_occupied_iou": previous[
                    "checkpoint_selection"
                ]["observable_occupied_iou"],
            },
        }
        occ = entry["checkpoint_selection"]["observable_occupied_iou"]
        shuffled_occ = entry["shuffled_token_control"]["checkpoint_selection"][
            "observable_occupied_iou"
        ]
        entry["shuffled_token_margin_observable_occupied_iou"] = float(occ - shuffled_occ)

        del features, model
        torch.cuda.empty_cache()
        record["shuffled_control_recomputed_v2"] = {
            "status": S.STATUS,
            "encoder_extraction_rerun": False,
            "main_probes_rerun": False,
            "wall_seconds": round(time.time() - started, 1),
        }
        result_path.write_text(json.dumps(record, indent=2))

    print(json.dumps(
        {
            name: {
                "occ_iou": a["checkpoint_selection"]["observable_occupied_iou"],
                "shuffled_within_role_occ_iou": a["shuffled_token_control"][
                    "checkpoint_selection"]["observable_occupied_iou"],
                "margin": a["shuffled_token_margin_observable_occupied_iou"],
                "superseded_cross_role_shuffled_occ_iou": a["shuffled_token_control"][
                    "superseded_cross_role_control"][
                    "checkpoint_selection_observable_occupied_iou"],
            }
            for name, a in record["arms"].items()
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
