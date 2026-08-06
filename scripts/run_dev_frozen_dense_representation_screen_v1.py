#!/usr/bin/env python3
"""Frozen dense-representation screen: can pretrained tokens carry our geometry?

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Every encoder here is frozen, so nothing in
this script is a JEPA result.  Its only jobs are to establish the strongest
achievable frozen spatial baseline, to identify a credible initialisation for
the next encoder-moving JEPA, and to say whether our spatial gate is reachable
from RGB at all with a published representation.

Identical corpus rows, identical targets, identical probe, identical schedule
and seed for every arm.  The only per-arm difference is the frozen encoder and
its own official preprocessing and dense extraction point.

Roles are partitioned before any cap.  ``probability_calibration``,
``evaluation``, ``untouched`` and any sealed data are never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for _r in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))

from scripts import run_go2_representation_qualification_probe_v1 as P  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402

SUP = ROOT / ".generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1"
OUT = ROOT / ".generated/dev/DEVELOPMENT_ONLY_frozen_dense_representation_screen_v1"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
CLASS_NAMES = ("unknown", "free", "occupied")

EXPECTED_TRAIN_PAIRS = 4262
EXPECTED_SELECTION_PAIRS = 495
EXPECTED_TRAIN_SCENES = 72
EXPECTED_SELECTION_SCENES = 8
ALLOWED_ROLES = ("train", "checkpoint_selection")
FORBIDDEN_ROLES = ("probability_calibration", "evaluation", "untouched", "sealed")

# One common probe contract for every arm.  The common grid is the LARGEST arm
# grid, so no arm has spatial detail thrown away by the adapter; smaller grids
# are deterministically upsampled, which adds no information to anybody.
COMMON_GRID = (24, 32)
PROJECTION_CHANNELS = 16
PROBE_HIDDEN = 1024
PROBE_EPOCHS = 30
PROBE_BATCH = 32
PROBE_LR = 1.0e-3
PROBE_WEIGHT_DECAY = 1.0e-4
PROBE_GRAD_CLIP = 1.0
PROBE_SEED = 2_026_080_611
DERANGEMENT_SEED = 2_026_080_617


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------
def load_rows() -> list[dict]:
    """Pairs in corpus order, restricted to the two permitted roles."""
    endpoints = {
        e["endpoint_identity_sha256"]: e
        for e in (
            json.loads(line)
            for line in (SUP / "endpoints.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    rows: list[dict] = []
    seen_roles: set[str] = set()
    for line in (SUP / "pairs.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        pair = json.loads(line)
        seen_roles.add(pair["dataset_role"])
        if pair["dataset_role"] not in ALLOWED_ROLES:
            continue
        current = endpoints[pair["current_endpoint_sha256"]]
        rows.append(
            {
                "path": current["image_path_metadata_only"],
                "shard_dir": str(SUP / Path(current["scene_shard"]).parent),
                "shard_row": int(current["shard_row"]),
                "scene": pair["scene_id"],
                "family": pair["family"],
                "role": pair["dataset_role"],
                "endpoint_sha256": current["endpoint_identity_sha256"],
                "raster_content_sha256": current["raster_content_sha256"],
            }
        )
    leaked = sorted(seen_roles & set(FORBIDDEN_ROLES))
    if leaked:
        # Present in the index is fine; loading them is not.  Nothing below
        # dereferences a row that is not in ALLOWED_ROLES.
        pass
    return rows


def load_targets(rows: list[dict]) -> np.ndarray:
    """``raster_labels.u1`` for each pair's labelled current observation.

    The native target is used exactly as stored.  No occupancy target is
    reconstructed here, and no matched-branch corpus is substituted.
    """
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


def support_provenance(rows: list[dict]) -> dict:
    """Record that the native ground-support evidence backs these labels.

    ``ground_support_in_frustum`` and ``ground_support_clear_to_target`` are the
    128x128x5 arrays the registered rasteriser reduces into ``raster_labels``.
    They are checked for presence and shape only; the target is not re-derived.
    """
    shards = sorted({r["shard_dir"] for r in rows})
    frustum_total = clear_total = 0
    for shard in shards:
        for name, accum in (
            ("ground_support_in_frustum.u1", "frustum"),
            ("ground_support_clear_to_target.u1", "clear"),
        ):
            path = Path(shard) / name
            if not path.is_file():
                raise FileNotFoundError(f"missing native support evidence: {path}")
            count = path.stat().st_size // (128 * 128 * 5)
            if accum == "frustum":
                frustum_total += count
            else:
                clear_total += count
    return {
        "raster_labels_used_verbatim": True,
        "target_reconstructed": False,
        "shards_checked": len(shards),
        "ground_support_in_frustum_rows": frustum_total,
        "ground_support_clear_to_target_rows": clear_total,
        "trailing_shape": [128, 128, 5],
        "class_definitions": "corpus native UNKNOWN=0, FREE=1, OCCUPIED=2",
    }


def ordered_hash(rows: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps([[r["endpoint_sha256"], r["raster_content_sha256"]] for r in rows]).encode()
    ).hexdigest()


# --------------------------------------------------------------------------
# frozen feature cache -- one thin path, not a framework
# --------------------------------------------------------------------------
def cache_features(arm, rows, device, dtype, batch_size: int) -> tuple[np.memmap, dict]:
    features_dir = OUT / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    blob = features_dir / f"{arm.name}.f16"
    receipt_path = features_dir / f"{arm.name}.json"
    tokens = arm.token_grid[0] * arm.token_grid[1]
    shape = (len(rows), tokens, arm.token_dim)
    pair_hash = ordered_hash(rows)
    prep_hash = E.preprocessing_hash(arm)

    if receipt_path.is_file() and blob.is_file():
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("ordered_pair_sha256") == pair_hash
            and receipt.get("preprocessing_sha256") == prep_hash
            and receipt.get("token_shape") == list(shape)
            and receipt.get("dtype") == "float16"
            and receipt.get("checkpoint_sha256") == arm.identity()["checkpoint_sha256"]
        ):
            memory = np.memmap(blob, dtype=np.float16, mode="r", shape=shape)
            receipt["reused_existing_cache"] = True
            return memory, receipt

    torch.cuda.reset_peak_memory_stats(device)
    module = arm.build(device, dtype)
    parameters = int(sum(p.numel() for p in module.parameters()))
    memory = np.memmap(blob, dtype=np.float16, mode="w+", shape=shape)
    started = time.time()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        batch = torch.stack([arm.preprocess(r["path"]) for r in chunk]).to(
            device=device, dtype=dtype
        )
        emitted = arm.tokens(batch)
        if emitted.shape[1:] != (tokens, arm.token_dim):
            raise RuntimeError(
                f"{arm.name}: expected tokens {(tokens, arm.token_dim)}, got {tuple(emitted.shape[1:])}"
            )
        memory[start : start + len(chunk)] = emitted.float().cpu().numpy().astype(np.float16)
    memory.flush()
    peak = int(torch.cuda.max_memory_allocated(device))

    parity = encoder_parity_check(arm, rows, device, dtype)
    del module
    arm._module = None  # noqa: SLF001
    torch.cuda.empty_cache()

    receipt = {
        "status": STATUS,
        "encoder": arm.name,
        "encoder_family": arm.family,
        **arm.identity(),
        "parameter_count": parameters,
        "inference_dtype": str(dtype).replace("torch.", ""),
        "peak_vram_bytes": peak,
        "peak_vram_gib": round(peak / 2**30, 3),
        "preprocessing": E.preprocessing_identity(arm),
        "preprocessing_sha256": prep_hash,
        "ordered_pair_sha256": pair_hash,
        "token_shape": list(shape),
        "dtype": "float16",
        "cache_path": str(blob),
        "cache_sha256": E.file_sha256(blob),
        "extraction_seconds": round(time.time() - started, 1),
        "parity_check": parity,
        "reused_existing_cache": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2))
    return np.memmap(blob, dtype=np.float16, mode="r", shape=shape), receipt


@torch.no_grad()
def encoder_parity_check(arm, rows, device, dtype) -> dict:
    """One visual/numeric parity check per encoder, on two fixed corpus rows."""
    from PIL import Image

    probe_dir = OUT / "parity" / arm.name
    probe_dir.mkdir(parents=True, exist_ok=True)
    first, second = rows[0], rows[len(rows) // 2]
    a = arm.preprocess(first["path"])
    b = arm.preprocess(second["path"])
    batch = torch.stack([a, b]).to(device=device, dtype=dtype)

    once = arm.tokens(batch).float().cpu()
    twice = arm.tokens(batch).float().cpu()
    determinism = float((once - twice).abs().max())

    mean = torch.tensor(E.IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(E.IMAGENET_STD)[:, None, None]
    for tag, tensor in (("row0", a), ("row_mid", b)):
        pixels = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
        Image.fromarray((pixels * 255).astype(np.uint8)).save(probe_dir / f"input_{tag}.png")

    differs = float((once[0] - once[1]).abs().max())
    cosine = float(
        F.cosine_similarity(once[0].flatten(), once[1].flatten(), dim=0)
    )
    tokens = arm.token_grid[0] * arm.token_grid[1]
    return {
        "input_tensor_shape": list(batch.shape),
        "input_tensor_dtype": str(batch.dtype).replace("torch.", ""),
        "input_tensor_min_max_mean_std": [
            float(batch.float().min()),
            float(batch.float().max()),
            float(batch.float().mean()),
            float(batch.float().std()),
        ],
        "input_tensor_sha256": hashlib.sha256(
            batch.float().cpu().numpy().tobytes()
        ).hexdigest(),
        "token_grid_hw": list(arm.token_grid),
        "tokens_total": tokens,
        "tokens_real_image_content": tokens,
        "tokens_pure_padding": 0,
        "deterministic_repeat_max_abs_diff": determinism,
        "distinct_observation_max_abs_diff": differs,
        "distinct_observation_cosine": cosine,
        "denormalised_input_images": [
            str(probe_dir / "input_row0.png"),
            str(probe_dir / "input_row_mid.png"),
        ],
    }


# --------------------------------------------------------------------------
# the one shared probe
# --------------------------------------------------------------------------
class SharedTokenToBev(nn.Module):
    """Dense tokens -> 64x64 UNKNOWN/FREE/OCCUPIED, identical for every arm.

    The only per-arm part is ``project``: one 1x1 convolution mapping the
    encoder's channel count to the common probe width.  Everything after the
    deterministic spatial adapter has the same architecture, parameter count,
    initialisation seed, optimiser and schedule for all arms.
    """

    def __init__(self, token_dim: int, channels: int = PROJECTION_CHANNELS,
                 grid=COMMON_GRID, hidden: int = PROBE_HIDDEN):
        super().__init__()
        self.grid = grid
        self.project = nn.Conv2d(token_dim, channels, 1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * grid[0] * grid[1], hidden),
            nn.GELU(),
            nn.Linear(hidden, 64 * 64 * 3),
        )

    def forward(self, tokens: torch.Tensor, token_grid, content_mask=None):
        b, n, d = tokens.shape
        grid = tokens.transpose(1, 2).reshape(b, d, token_grid[0], token_grid[1])
        x = F.gelu(self.project(grid))
        if content_mask is not None:
            x = x * content_mask
        if (token_grid[0], token_grid[1]) != tuple(self.grid):
            # deterministic, non-learned spatial adapter
            x = F.interpolate(x, size=self.grid, mode="bilinear", align_corners=False)
        return self.head(x).reshape(b, 3, 64, 64)


def class_weights(labels: np.ndarray, device) -> torch.Tensor:
    counts = np.bincount(labels.reshape(-1), minlength=3).astype(np.float64)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32,
                          device=device)
    return weight / weight.mean()


@torch.no_grad()
def predict(model, features, indices, token_grid, device, feature_map=None,
            batch_size=64) -> np.ndarray:
    """``feature_map`` remaps sample -> feature row; identity unless deranged."""
    model.eval()
    out = []
    for start in range(0, len(indices), batch_size):
        sel = indices[start : start + batch_size]
        source = sel if feature_map is None else feature_map[sel]
        rows = torch.as_tensor(np.asarray(source), dtype=torch.long, device=features.device)
        batch = features[rows].to(device=device, dtype=torch.float32)
        out.append(model(batch, token_grid).argmax(1).cpu().numpy().astype(np.uint8))
    return np.concatenate(out, 0)


def train_probe(features, labels, train_idx, sel_idx, token_grid, token_dim, device,
                tag: str, feature_map=None):
    torch.manual_seed(PROBE_SEED)
    model = SharedTokenToBev(token_dim).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY, foreach=False
    )
    weight = class_weights(labels[train_idx], device)
    target = torch.from_numpy(labels).long()
    generator = torch.Generator().manual_seed(PROBE_SEED)
    source = torch.as_tensor(train_idx, dtype=torch.long)
    remap = None if feature_map is None else torch.as_tensor(
        np.asarray(feature_map), dtype=torch.long
    )

    history: list[dict] = []
    best = (-1.0, None, -1)
    for epoch in range(PROBE_EPOCHS):
        model.train()
        order = source[torch.randperm(len(source), generator=generator)]
        for start in range(0, len(order), PROBE_BATCH):
            sel = order[start : start + PROBE_BATCH]
            rows = sel if remap is None else remap[sel]
            optimiser.zero_grad(set_to_none=True)
            batch = features[rows.to(features.device)].to(device=device, dtype=torch.float32)
            loss = F.cross_entropy(
                model(batch, token_grid), target[sel].to(device), weight=weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), PROBE_GRAD_CLIP)
            optimiser.step()
        selection = predict(model, features, sel_idx, token_grid, device, feature_map)
        iou = P.metrics(selection, labels[sel_idx]).get("observable_occupied_iou")
        iou = float(iou) if iou is not None else 0.0
        history.append({"epoch": epoch, "selection_observable_occupied_iou": iou})
        if iou > best[0]:
            best = (iou, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch)
        print(f"  [{tag}] epoch {epoch:02d} selection occ IoU {iou:.4f}", flush=True)
    model.load_state_dict(best[1])
    model.eval()
    return model, history, best[2]


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def all_free_baseline(labels: np.ndarray) -> dict:
    """Predict FREE at every cell: the floor observable free IoU must beat."""
    pred = np.full_like(labels, FREE)
    m = P.metrics(pred, labels)
    return {
        "observable_free_iou": m.get("observable_free_iou"),
        "observable_free_recall": m.get("observable_free_recall"),
        "observable_free_precision": m.get("observable_free_precision"),
        "observable_occupied_iou": m.get("observable_occupied_iou"),
        "observable_balanced_accuracy": m.get("observable_balanced_accuracy"),
    }


def summarise(pred: np.ndarray, labels: np.ndarray) -> dict:
    m = P.metrics(pred, labels)
    cells = int(labels.size)
    observable = int((labels != UNKNOWN).sum())
    return {
        "observable_occupied_iou": m.get("observable_occupied_iou"),
        "observable_occupied_precision": m.get("observable_occupied_precision"),
        "observable_occupied_recall": m.get("observable_occupied_recall"),
        "observable_free_iou": m.get("observable_free_iou"),
        "observable_free_precision": m.get("observable_free_precision"),
        "observable_free_recall": m.get("observable_free_recall"),
        "observable_balanced_accuracy": m.get("observable_balanced_accuracy"),
        "unknown_iou": m.get("unknown_iou"),
        "unknown_precision": m.get("unknown_precision"),
        "unknown_recall": m.get("unknown_recall"),
        "tolerant_occupied": [P.tolerant_occupied(pred, labels, r) for r in (1, 2)],
        "predicted_class_fraction_over_all_cells": {
            n: float((pred == k).mean()) for k, n in enumerate(CLASS_NAMES)
        },
        "target_class_fraction_over_all_cells": {
            n: float((labels == k).mean()) for k, n in enumerate(CLASS_NAMES)
        },
        "denominators": {
            "frames": int(labels.shape[0]),
            "cells_total": cells,
            "cells_observable": observable,
            "cells_occupied": int((labels == OCCUPIED).sum()),
            "occupied_share_of_observable": (
                float((labels == OCCUPIED).sum() / observable) if observable else None
            ),
        },
        "all_free_baseline": all_free_baseline(labels),
    }


def grouped(pred, labels, keys) -> dict:
    out = {}
    unique = sorted(set(keys))
    for key in unique:
        sel = np.array([i for i, k in enumerate(keys) if k == key])
        block = labels[sel]
        m = P.metrics(pred[sel], block)
        out[key] = {
            "frames": int(len(sel)),
            "occupied_support_cells": int((block == OCCUPIED).sum()),
            "observable_occupied_iou": m.get("observable_occupied_iou"),
            "observable_occupied_precision": m.get("observable_occupied_precision"),
            "observable_occupied_recall": m.get("observable_occupied_recall"),
            "observable_free_iou": m.get("observable_free_iou"),
            "all_free_observable_free_iou": all_free_baseline(block)["observable_free_iou"],
            "observable_balanced_accuracy": m.get("observable_balanced_accuracy"),
        }
    return out


def macro(per_group: dict, field: str) -> dict:
    values = [v[field] for v in per_group.values() if v[field] is not None]
    return {
        "macro_mean": float(np.mean(values)) if values else None,
        "groups_contributing": len(values),
        "groups_total": len(per_group),
    }


def per_frame_occupied_iou(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p, t = pred == OCCUPIED, labels == OCCUPIED
    inter = (p & t).reshape(len(pred), -1).sum(1)
    union = (p | t).reshape(len(pred), -1).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def overlays(arm_name: str, rows, pred, labels, path: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    scores = per_frame_occupied_iou(pred, labels)
    valid = np.where(~np.isnan(scores))[0]
    if len(valid) == 0:
        return {}
    order = valid[np.argsort(scores[valid])]
    picks = {
        "worst": int(order[0]),
        "median": int(order[len(order) // 2]),
        "best": int(order[-1]),
    }
    colours = np.array([[40, 40, 48], [70, 130, 180], [220, 80, 60]], dtype=np.uint8)
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for r, (tag, index) in enumerate(picks.items()):
        with Image.open(rows[index]["path"]) as decoded:
            axes[r, 0].imshow(np.asarray(decoded.convert("RGB")))
        axes[r, 0].set_title(f"{tag}: {rows[index]['family']}", fontsize=8)
        axes[r, 1].imshow(colours[labels[index]])
        axes[r, 1].set_title("target", fontsize=8)
        axes[r, 2].imshow(colours[pred[index]])
        axes[r, 2].set_title(f"predicted (occ IoU {scores[index]:.3f})", fontsize=8)
        for c in range(3):
            axes[r, c].axis("off")
    fig.suptitle(f"{arm_name}: checkpoint_selection occupancy overlays", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return {
        tag: {
            "row": index,
            "scene": rows[index]["scene"],
            "family": rows[index]["family"],
            "observable_occupied_iou": float(scores[index]),
        }
        for tag, index in picks.items()
    }


# --------------------------------------------------------------------------
def build_arms(requested: list[str]) -> list:
    available = {
        "project_vit": E.ProjectViTArm(),
        "dinov2": E.DinoV2Arm(),
        "vjepa21": E.VJepa21Arm(),
        "vjepa21_base": E.VJepa21Arm(
            checkpoint=E.VJEPA_FALLBACK_CHECKPOINT, constructor="vjepa2_1_vit_base_384"
        ),
    }
    return [available[name] for name in requested]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="project_vit,dinov2,vjepa21")
    ap.add_argument("--extract-batch", type=int, default=16)
    ap.add_argument("--dtype", default="float32", choices=("float32", "float16", "bfloat16"))
    ap.add_argument("--extract-only", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    rows = load_rows()
    train_rows = [r for r in rows if r["role"] == "train"]
    sel_rows = [r for r in rows if r["role"] == "checkpoint_selection"]
    if len(train_rows) != EXPECTED_TRAIN_PAIRS or len(sel_rows) != EXPECTED_SELECTION_PAIRS:
        raise RuntimeError(
            f"role partition changed: train={len(train_rows)} selection={len(sel_rows)}"
        )
    train_scenes = {r["scene"] for r in train_rows}
    sel_scenes = {r["scene"] for r in sel_rows}
    overlap = sorted(train_scenes & sel_scenes)
    if overlap:
        raise RuntimeError(f"train/selection scene overlap: {overlap[:5]}")
    if len(train_scenes) != EXPECTED_TRAIN_SCENES or len(sel_scenes) != EXPECTED_SELECTION_SCENES:
        raise RuntimeError(
            f"scene split changed: train={len(train_scenes)} selection={len(sel_scenes)}"
        )

    ordered = train_rows + sel_rows          # cache order, fixed for every arm
    labels = load_targets(ordered)
    train_idx = np.arange(len(train_rows))
    sel_idx = np.arange(len(train_rows), len(ordered))

    record: dict = {
        "status": STATUS,
        "claim_bearing": False,
        "purpose": (
            "frozen-representation screen: strongest achievable spatial baseline, "
            "candidate initialisation for the next encoder-moving JEPA, and whether "
            "the spatial gate is reachable from RGB with a published representation. "
            "No frozen encoder plus probe is a JEPA endpoint."
        ),
        "corpus": {
            "name": "development_raw_supervision_v1",
            "path": str(SUP),
            "manifest_content_sha256": json.loads((SUP / "manifest.json").read_text())[
                "content_sha256"
            ],
            "roles_loaded": list(ALLOWED_ROLES),
            "roles_never_loaded": list(FORBIDDEN_ROLES),
            "train_pairs": len(train_rows),
            "checkpoint_selection_pairs": len(sel_rows),
            "train_scenes": len(train_scenes),
            "checkpoint_selection_scenes": len(sel_scenes),
            "scene_overlap": 0,
            "partitioned_before_any_cap": True,
            "ordered_pair_sha256": ordered_hash(ordered),
            "supervision": support_provenance(ordered),
        },
        "probe_contract": {
            "family": "Stage-1 dense-token-to-BEV (run_go2_representation_qualification_probe_v1.TokenToBev)",
            "common_token_grid_hw": list(COMMON_GRID),
            "common_grid_rule": "largest arm grid; smaller grids deterministically bilinear-upsampled",
            "per_arm_input_projection": f"1x1 conv token_dim -> {PROJECTION_CHANNELS}",
            "shared_head": f"flatten -> linear {PROJECTION_CHANNELS*COMMON_GRID[0]*COMMON_GRID[1]} -> {PROBE_HIDDEN} -> GELU -> linear -> 64x64x3",
            "epochs": PROBE_EPOCHS,
            "batch": PROBE_BATCH,
            "lr": PROBE_LR,
            "weight_decay": PROBE_WEIGHT_DECAY,
            "grad_clip": PROBE_GRAD_CLIP,
            "loss": "class-weighted cross entropy, weights from train-role counts",
            "seed": PROBE_SEED,
            "checkpoint_selection_rule": "best checkpoint_selection observable occupied IoU over epochs",
            "encoder_specific_tuning": "none",
        },
        "arms": {},
        "controls": {},
    }

    # Arms may be run in separate invocations (an external checkpoint can
    # arrive late).  Merge into any existing record rather than overwriting it,
    # but only when the corpus and its ordering are byte-identical.
    existing_path = OUT / "result.json"
    if existing_path.is_file():
        existing = json.loads(existing_path.read_text())
        if existing.get("corpus", {}).get("ordered_pair_sha256") == record["corpus"][
            "ordered_pair_sha256"
        ]:
            record["arms"] = existing.get("arms", {})
            record["controls"] = existing.get("controls", {})
        else:
            raise RuntimeError(
                "existing result.json was built on a different ordered corpus; "
                "refusing to merge incomparable arms"
            )

    prior = np.stack([(labels[train_idx] == k).mean(axis=0) for k in range(3)]).argmax(0)
    prior_pred = np.broadcast_to(prior, labels[sel_idx].shape)
    record["controls"]["class_prior_mean_map_no_image_input"] = {
        "description": "per-cell argmax over train-role targets; no image is read",
        "checkpoint_selection": summarise(prior_pred, labels[sel_idx]),
    }

    arm_objects = build_arms([a for a in args.arms.split(",") if a])
    for arm in arm_objects:
        print(f"== arm {arm.name}", flush=True)
        memory, receipt = cache_features(arm, ordered, device, dtype, args.extract_batch)
        entry = {"feature_cache": receipt}
        record["arms"][arm.name] = entry
        (OUT / "result.json").write_text(json.dumps(record, indent=2))
        if args.extract_only:
            continue

        features = torch.from_numpy(np.ascontiguousarray(memory))
        features = features.to(device)          # fp16 resident, cast per batch
        grid = arm.token_grid

        model, history, best_epoch = train_probe(
            features, labels, train_idx, sel_idx, grid, arm.token_dim, device, arm.name
        )
        train_pred = predict(model, features, train_idx, grid, device)
        sel_pred = predict(model, features, sel_idx, grid, device)

        # shuffled-token control: complete feature tensors deranged against
        # observations, token positions untouched, both roles deranged.
        rng = np.random.default_rng(DERANGEMENT_SEED)
        derange = np.arange(len(ordered))
        while True:
            rng.shuffle(derange)
            if not (derange == np.arange(len(ordered))).any():
                break
        s_model, s_history, s_epoch = train_probe(
            features, labels, train_idx, sel_idx, grid, arm.token_dim, device,
            f"{arm.name}/shuffled", feature_map=derange,
        )
        s_sel = predict(s_model, features, sel_idx, grid, device, derange)
        s_train = predict(s_model, features, train_idx, grid, device, derange)

        sel_scene_keys = [r["scene"] for r in sel_rows]
        sel_family_keys = [r["family"] for r in sel_rows]
        per_scene = grouped(sel_pred, labels[sel_idx], sel_scene_keys)
        per_family = grouped(sel_pred, labels[sel_idx], sel_family_keys)

        entry.update(
            {
                "probe": {
                    "selected_epoch": best_epoch,
                    "history": history,
                    "parameters_total": int(sum(p.numel() for p in model.parameters())),
                    "parameters_shared_head": int(
                        sum(p.numel() for p in model.head.parameters())
                    ),
                    "parameters_input_projection": int(
                        sum(p.numel() for p in model.project.parameters())
                    ),
                },
                "train": summarise(train_pred, labels[train_idx]),
                "checkpoint_selection": summarise(sel_pred, labels[sel_idx]),
                "shuffled_token_control": {
                    "description": "complete feature tensors deranged between observations; token positions preserved",
                    "selected_epoch": s_epoch,
                    "train": summarise(s_train, labels[train_idx]),
                    "checkpoint_selection": summarise(s_sel, labels[sel_idx]),
                },
                "per_scene_checkpoint_selection": per_scene,
                "per_family_checkpoint_selection": per_family,
                "macro_over_selection_scenes": {
                    "observable_occupied_iou": macro(per_scene, "observable_occupied_iou"),
                    "observable_occupied_precision": macro(per_scene, "observable_occupied_precision"),
                    "observable_occupied_recall": macro(per_scene, "observable_occupied_recall"),
                },
                "scenes_with_nonzero_occupied_support": int(
                    sum(1 for v in per_scene.values() if v["occupied_support_cells"] > 0)
                ),
                "open_obstacle_field": per_family.get("open_obstacle_field"),
            }
        )
        occ = entry["checkpoint_selection"]["observable_occupied_iou"]
        shuffled_occ = entry["shuffled_token_control"]["checkpoint_selection"][
            "observable_occupied_iou"
        ]
        entry["shuffled_token_margin_observable_occupied_iou"] = (
            None if occ is None or shuffled_occ is None else float(occ - shuffled_occ)
        )
        entry["fit_to_selection_transfer"] = {
            "train_observable_occupied_iou": entry["train"]["observable_occupied_iou"],
            "selection_observable_occupied_iou": occ,
            "gap": (
                None
                if occ is None or entry["train"]["observable_occupied_iou"] is None
                else float(entry["train"]["observable_occupied_iou"] - occ)
            ),
        }
        entry["overlays"] = overlays(
            arm.name, sel_rows, sel_pred, labels[sel_idx],
            OUT / f"overlays_{arm.name}.png",
        )
        np.save(OUT / f"selection_predictions_{arm.name}.npy", sel_pred)

        del features, model, s_model
        torch.cuda.empty_cache()
        record["wall_seconds"] = round(time.time() - started, 1)
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    record["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(
        {
            name: {
                "occ_iou": a.get("checkpoint_selection", {}).get("observable_occupied_iou"),
                "occ_precision": a.get("checkpoint_selection", {}).get("observable_occupied_precision"),
                "occ_recall": a.get("checkpoint_selection", {}).get("observable_occupied_recall"),
                "shuffled_margin": a.get("shuffled_token_margin_observable_occupied_iou"),
                "open_obstacle_field_occ_iou": (a.get("open_obstacle_field") or {}).get(
                    "observable_occupied_iou"),
            }
            for name, a in record["arms"].items()
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
