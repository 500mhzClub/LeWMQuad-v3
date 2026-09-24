#!/usr/bin/env python3
"""Frozen V-JEPA 2.1 spatial reference on centre-cropped v03 current frames.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  The encoder is frozen, so this is not a
JEPA result.

The frozen dense-representation screen measured `0.5103` observable occupied IoU
on ``textured_v04`` frames.  The temporal experiment runs on ``textured_v03``
frames centre-cropped to the v04 field of view, which is a different visual
contract.  This remeasures the same frozen encoder, under the same probe, on the
retained temporal rows, so the encoder-moving arm has a **matched** in-contract
non-regression comparator.  `0.5103` remains a cross-contract reference only.

Caches are written to the root filesystem: the workspace pool is full.
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

from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402
from scripts import run_dev_frozen_dense_representation_screen_v1 as S  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
OUT = CACHE / "frozen_spatial_reference"
ROWS = CACHE / "temporal_rows.jsonl"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
DERANGEMENT_SEED = 2_026_080_641


def load_rows() -> list[dict]:
    return [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]


def load_targets(rows: list[dict]) -> np.ndarray:
    cache: dict[str, np.ndarray] = {}
    out = np.empty((len(rows), 64, 64), dtype=np.uint8)
    for i, row in enumerate(rows):
        shard = row["shard_dir"]
        if shard not in cache:
            cache[shard] = np.fromfile(
                Path(shard) / "raster_labels.u1", dtype=np.uint8
            ).reshape(-1, 64, 64)
        out[i] = cache[shard][row["shard_row"]]
    return out


@torch.no_grad()
def extract(arm, paths, device, dtype, blob: Path, batch_size: int):
    tokens = arm.token_grid[0] * arm.token_grid[1]
    shape = (len(paths), tokens, arm.token_dim)
    memory = np.memmap(blob, dtype=np.float16, mode="w+", shape=shape)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.time()
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        batch = torch.stack([arm.preprocess(p) for p in chunk]).to(device=device, dtype=dtype)
        emitted = arm.tokens(batch)
        memory[start : start + len(chunk)] = emitted.float().cpu().numpy().astype(np.float16)
    memory.flush()
    return shape, int(torch.cuda.max_memory_allocated(device)), time.time() - started


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extract-batch", type=int, default=16)
    args = ap.parse_args()
    device = torch.device(args.device)
    dtype = torch.float32
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rows = load_rows()
    train_rows = [r for r in rows if r["role"] == "train"]
    sel_rows = [r for r in rows if r["role"] == "checkpoint_selection"]
    ordered = train_rows + sel_rows
    overlap = sorted({r["scene"] for r in train_rows} & {r["scene"] for r in sel_rows})
    if overlap:
        raise RuntimeError(f"train/selection scene overlap: {overlap[:5]}")
    labels = load_targets(ordered)
    train_idx = np.arange(len(train_rows))
    sel_idx = np.arange(len(train_rows), len(ordered))

    arm = E.VJepa21CroppedV03Arm()
    blob = OUT / "vjepa2_1_vitl_384_v03crop.f16"
    receipt_path = OUT / "vjepa2_1_vitl_384_v03crop.json"
    paths = [r["frames"][2]["path"] for r in ordered]        # the current frame, t
    if not all(Path(p).name.startswith("frame_") for p in paths):
        raise RuntimeError("current-frame path resolution failed")

    if receipt_path.is_file() and blob.is_file():
        receipt = json.loads(receipt_path.read_text())
        shape = tuple(receipt["token_shape"])
    else:
        module = arm.build(device, dtype)
        parameters = int(sum(p.numel() for p in module.parameters()))
        shape, peak, seconds = extract(arm, paths, device, dtype, blob, args.extract_batch)
        parity = S.encoder_parity_check(arm, [{"path": p} for p in paths], device, dtype)
        del module
        arm._module = None  # noqa: SLF001
        torch.cuda.empty_cache()
        receipt = {
            "status": STATUS,
            "encoder": arm.name,
            **arm.identity(),
            "parameter_count": parameters,
            "inference_dtype": "float32",
            "peak_vram_bytes": peak,
            "peak_vram_gib": round(peak / 2**30, 3),
            "preprocessing": E.preprocessing_identity(arm),
            "preprocessing_sha256": E.preprocessing_hash(arm),
            "token_shape": list(shape),
            "dtype": "float16",
            "cache_path": str(blob),
            "cache_sha256": E.file_sha256(blob),
            "extraction_seconds": round(seconds, 1),
            "parity_check": parity,
        }
        receipt_path.write_text(json.dumps(receipt, indent=2))

    features = torch.from_numpy(
        np.ascontiguousarray(np.memmap(blob, dtype=np.float16, mode="r", shape=shape))
    ).to(device)
    grid = arm.token_grid

    model, history, epoch = S.train_probe(
        features, labels, train_idx, sel_idx, grid, arm.token_dim, device, arm.name
    )
    train_pred = S.predict(model, features, train_idx, grid, device)
    sel_pred = S.predict(model, features, sel_idx, grid, device)

    rng = np.random.default_rng(DERANGEMENT_SEED)
    derange = np.arange(len(ordered))
    for block in (train_idx, sel_idx):
        order = np.arange(len(block))
        while True:
            rng.shuffle(order)
            if not (order == np.arange(len(block))).any():
                break
        derange[block] = block[order]
    s_model, s_history, s_epoch = S.train_probe(
        features, labels, train_idx, sel_idx, grid, arm.token_dim, device,
        f"{arm.name}/shuffled", feature_map=derange,
    )
    s_sel = S.predict(s_model, features, sel_idx, grid, device, derange)
    s_train = S.predict(s_model, features, train_idx, grid, device, derange)

    per_family = S.grouped(sel_pred, labels[sel_idx], [r["family"] for r in sel_rows])
    per_scene = S.grouped(sel_pred, labels[sel_idx], [r["scene"] for r in sel_rows])
    result = {
        "status": STATUS,
        "claim_bearing": False,
        "role": (
            "matched in-contract non-regression comparator for the encoder-moving arm; "
            "the v04 figure 0.5103 is a cross-contract development reference only"
        ),
        "visual_contract": E.preprocessing_identity(arm),
        "feature_cache": receipt,
        "rows": {
            "train": len(train_rows), "checkpoint_selection": len(sel_rows),
            "train_scenes": len({r["scene"] for r in train_rows}),
            "selection_scenes": len({r["scene"] for r in sel_rows}),
            "scene_overlap": 0,
        },
        "probe": {"selected_epoch": epoch, "history": history,
                  "parameters_total": int(sum(p.numel() for p in model.parameters()))},
        "train": S.summarise(train_pred, labels[train_idx]),
        "checkpoint_selection": S.summarise(sel_pred, labels[sel_idx]),
        "shuffled_token_control": {
            "description": "within-role fixed-point-free derangements",
            "selected_epoch": s_epoch,
            "train": S.summarise(s_train, labels[train_idx]),
            "checkpoint_selection": S.summarise(s_sel, labels[sel_idx]),
        },
        "per_family_checkpoint_selection": per_family,
        "per_scene_checkpoint_selection": per_scene,
        "macro_over_selection_scenes": {
            "observable_occupied_iou": S.macro(per_scene, "observable_occupied_iou"),
            "observable_occupied_precision": S.macro(per_scene, "observable_occupied_precision"),
            "observable_occupied_recall": S.macro(per_scene, "observable_occupied_recall"),
        },
        "open_obstacle_field": per_family.get("open_obstacle_field"),
        "cross_contract_reference_v04": {
            "observable_occupied_iou": 0.5102939019022935,
            "observable_occupied_precision": 0.6470894264652014,
            "note": "different render, different textures, different rows; not a comparator",
        },
        "wall_seconds": round(time.time() - started, 1),
    }
    occ = result["checkpoint_selection"]["observable_occupied_iou"]
    sh = result["shuffled_token_control"]["checkpoint_selection"]["observable_occupied_iou"]
    result["shuffled_token_margin_observable_occupied_iou"] = float(occ - sh)
    np.save(OUT / "selection_predictions.npy", sel_pred)
    (OUT / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "occ_iou": occ, "occ_precision": result["checkpoint_selection"]["observable_occupied_precision"],
        "occ_recall": result["checkpoint_selection"]["observable_occupied_recall"],
        "shuffled_margin": result["shuffled_token_margin_observable_occupied_iou"],
        "open_obstacle_field": (result["open_obstacle_field"] or {}).get("observable_occupied_iou"),
        "macro_scene_iou": result["macro_over_selection_scenes"]["observable_occupied_iou"]["macro_mean"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
