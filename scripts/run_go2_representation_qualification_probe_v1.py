#!/usr/bin/env python3
"""Native-contract fidelity probe on the direct-BEV ViT (stages 3 and 2).

Runs entirely inside the model's own contract: its corpus
(``development_raw_supervision_v1``), its ``raster_labels.u1`` supervision, its
ImageNet-normalised 112x112 input, and its registered hierarchical decode.

Stage 3 evaluates the two signed-boundary state-head fields through
``hierarchical_class_log_probabilities_v1``.  Stage 2 fits per-cell probes to
the 64x64x64 BEV state, against a per-cell mean-map baseline and a shuffled-BEV
baseline so that fixed frustum and positional priors are not mistaken for
image-conditioned geometry.

Train scenes only, with a scene-disjoint validation split.  Evaluation-role
scenes are sealed and never opened.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
for _r in (REPO_ROOT, REPO_ROOT / "lewm_genesis", REPO_ROOT / "lewm_worlds"):
    if str(_r) not in sys.path:
        sys.path.insert(0, str(_r))

import lewm.models.direct_egocentric_bev_state_jepa_v1 as m1  # noqa: E402
import lewm.models.direct_egocentric_bev_state_jepa_v8_learned_bev_query_prototype_decoder as m8  # noqa: E402
import lewm.models.direct_egocentric_bev_signed_boundary_distance_state_v1 as msb  # noqa: E402

CHECKPOINT = REPO_ROOT / (
    ".generated/go2_shared_observable_camera_ray_jepa_signed_boundary_semantic_anchor_state_v3/"
    "rgb_direct_egocentric_bev_signed_boundary_semantic_anchor_state_probe_v3_update100_trend_gate_timing_v1/"
    "checkpoints/update_400.pt"
)
SUPERVISION = REPO_ROOT / (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/development_raw_supervision_v1"
)
OUT = REPO_ROOT / ".generated/dev/go2_representation_qualification_probe_v1/attempt_v3"

UNKNOWN, FREE, OCCUPIED = 0, 1, 2
CLASS_NAMES = ("unknown", "free", "occupied")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_SEED = 2_026_080_572
PROBE_SEED = 2_026_080_581


def native_preprocess(path: str) -> torch.Tensor:
    """Exactly the training runner's path: 112x112 bilinear, /255, ImageNet norm."""
    with Image.open(path) as decoded:
        image = decoded.convert("RGB").resize((112, 112), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).contiguous()
    mean = tensor.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = tensor.new_tensor(IMAGENET_STD)[:, None, None]
    return (tensor - mean) / std


def load_stack(device):
    enc = m1._construct_n320_encoder_without_rng_draw()  # noqa: SLF001
    dec = m8.LearnedBevQueryDecoderV8()
    head = msb.SignedBoundaryDistanceStateHeadV1()
    sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model_state_dict"]
    sub = lambda p: {k[len(p) + 1 :]: v for k, v in sd.items() if k.startswith(p + ".")}  # noqa: E731
    for name, mod in (("encoder", enc), ("bev_decoder", dec), ("state_head", head)):
        mod.load_state_dict(sub(name), strict=True)
        mod.to(device).eval().requires_grad_(False)
    return enc, dec, head


def native_decode(fields: torch.Tensor) -> np.ndarray:
    """The registered hierarchical interpretation, not an ad hoc sign rule."""
    return msb.hierarchical_class_log_probabilities_v1(fields).argmax(1).cpu().numpy()


def load_index() -> list[dict]:
    rows = []
    for shard in sorted((SUPERVISION / "shards").iterdir()):
        index = shard / "index.jsonl"
        raster = shard / "raster_labels.u1"
        if not index.is_file() or not raster.is_file():
            continue
        entries = [json.loads(line) for line in index.read_text().splitlines() if line.strip()]
        labels = np.fromfile(raster, dtype=np.uint8).reshape(-1, 64, 64)
        if len(entries) != labels.shape[0]:
            raise RuntimeError(f"{shard.name}: index/raster length mismatch")
        for position, entry in enumerate(entries):
            path = str(entry["image_path_metadata_only"])
            if entry.get("dataset_role") != "train" or not os.path.isfile(path):
                continue
            rows.append(
                {
                    "path": path,
                    "scene": str(entry["scene_id"]),
                    "family": str(entry["family"]),
                    "label": labels[position],
                }
            )
    return rows


def metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    out: dict[str, object] = {}
    for k, name in enumerate(CLASS_NAMES):
        actual, chosen = truth == k, pred == k
        support, inter = int(actual.sum()), int((actual & chosen).sum())
        union, picked = int((actual | chosen).sum()), int(chosen.sum())
        out[f"{name}_support"] = support
        out[f"{name}_recall"] = inter / support if support else None
        out[f"{name}_precision"] = inter / picked if picked else None
        out[f"{name}_iou"] = inter / union if union else None
        out[f"{name}_predicted_fraction"] = float(chosen.mean())
    observable = truth != UNKNOWN
    if observable.any():
        po, to = pred[observable], truth[observable]
        recalls = []
        for k, name in ((FREE, "free"), (OCCUPIED, "occupied")):
            actual, chosen = to == k, po == k
            support, inter = int(actual.sum()), int((actual & chosen).sum())
            union = int((actual | chosen).sum())
            rec = inter / support if support else None
            out[f"observable_{name}_recall"] = rec
            out[f"observable_{name}_precision"] = inter / int(chosen.sum()) if int(chosen.sum()) else None
            out[f"observable_{name}_iou"] = inter / union if union else None
            if rec is not None:
                recalls.append(rec)
        out["observable_balanced_accuracy"] = float(np.mean(recalls)) if recalls else None
        out["observable_cells"] = int(observable.sum())
    return out


def tolerant_occupied(pred: np.ndarray, truth: np.ndarray, radius: int) -> dict:
    """Occupied recall/precision allowing a displacement of `radius` cells."""
    from scipy.ndimage import binary_dilation

    structure = np.ones((1, 2 * radius + 1, 2 * radius + 1), dtype=bool)
    p, t = pred == OCCUPIED, truth == OCCUPIED
    p_d = binary_dilation(p, structure=structure)
    t_d = binary_dilation(t, structure=structure)
    rec = float((t & p_d).sum() / t.sum()) if t.sum() else None
    prec = float((p & t_d).sum() / p.sum()) if p.sum() else None
    return {"radius_cells": radius, "occupied_recall": rec, "occupied_precision": prec}


class TokenToBev(nn.Module):
    """Fixed spatial decoder from 16x16 ViT patch tokens to the 64x64 raster.

    Image-plane tokens and the BEV target are not spatially aligned, so this
    must permit a learned reorganisation rather than an upsample.  It is
    deliberately small: a failure here means "this fixed decoder could not
    recover the geometry", not that the tokens lack it.
    """

    def __init__(self, dim=192, c=32, hidden=1024):
        super().__init__()
        self.reduce = nn.Conv2d(dim, c, 1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(c * 16 * 16, hidden), nn.GELU(),
            nn.Linear(hidden, 64 * 64 * 3),
        )

    def forward(self, tokens):
        b, n, d = tokens.shape
        grid = tokens.transpose(1, 2).reshape(b, d, 16, 16)
        x = nn.functional.gelu(self.reduce(grid))
        return self.head(x).reshape(b, 3, 64, 64)


class LinearCell(nn.Module):
    def __init__(self, c=64, k=3):
        super().__init__()
        self.head = nn.Conv2d(c, k, 1)

    def forward(self, x):
        return self.head(x)


class ShallowCell(nn.Module):
    def __init__(self, c=64, k=3, h=128):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(c, h, 1), nn.GELU(), nn.Conv2d(h, k, 1))

    def forward(self, x):
        return self.head(x)


def fit_probe(model, feats, labels, idx, device, epochs=30, bs=32, lr=3e-3):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, foreach=False)
    counts = np.bincount(labels[idx].reshape(-1), minlength=3).astype(np.float64)
    weight = torch.tensor(counts.sum() / np.maximum(counts, 1.0), dtype=torch.float32, device=device)
    weight = weight / weight.mean()
    generator = torch.Generator().manual_seed(PROBE_SEED)
    source = torch.as_tensor(idx, dtype=torch.long)
    model.train()
    for _ in range(epochs):
        order = source[torch.randperm(len(source), generator=generator)]
        for start in range(0, len(order), bs):
            sel = order[start : start + bs]
            opt.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(feats[sel].to(device)),
                torch.from_numpy(labels[sel]).long().to(device),
                weight=weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict(model, feats, idx, device, bs=64):
    out = []
    for start in range(0, len(idx), bs):
        sel = torch.as_tensor(idx[start : start + bs], dtype=torch.long)
        out.append(model(feats[sel].to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="override the checkpoint under diagnostic")
    ap.add_argument("--tag", default=None, help="output subdirectory tag")
    ap.add_argument("--max-frames", type=int, default=2048)
    args = ap.parse_args()
    global CHECKPOINT, OUT
    if args.checkpoint is not None:
        CHECKPOINT = args.checkpoint
    if args.tag:
        OUT = OUT.parent / f"attempt_v3_{args.tag}"
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    enc, dec, head = load_stack(device)
    rows = load_index()[: args.max_frames]
    labels = np.stack([r["label"] for r in rows])
    scenes = sorted({r["scene"] for r in rows})
    rng = np.random.default_rng(SPLIT_SEED)
    val = set(rng.choice(scenes, size=max(1, len(scenes) // 4), replace=False).tolist())
    fit_idx = np.array([i for i, r in enumerate(rows) if r["scene"] not in val])
    val_idx = np.array([i for i, r in enumerate(rows) if r["scene"] in val])

    bevs, decoded, token_cache = [], [], []
    for start in range(0, len(rows), 32):
        batch = torch.stack([native_preprocess(r["path"]) for r in rows[start : start + 32]])
        with torch.no_grad():
            tokens = enc.forward_tokens(batch.to(device))[:, 1:, :]
            bev = dec(tokens)
            decoded.append(native_decode(head(bev)))
        bevs.append(bev.cpu())
        token_cache.append(tokens.cpu())
    bev = torch.cat(bevs, 0)
    stage3 = np.concatenate(decoded, 0)

    result: dict[str, object] = {
        "contract": "native: development_raw_supervision_v1, raster_labels.u1, "
        "112x112 ImageNet-normalised, hierarchical_class_log_probabilities_v1",
        "checkpoint": str(CHECKPOINT),
        "frames": len(rows),
        "scenes": len(scenes),
        "validation_scenes": sorted(val),
        "fit_frames": int(len(fit_idx)),
        "validation_frames": int(len(val_idx)),
        "target_class_fraction": {
            "fit": {n: float((labels[fit_idx] == k).mean()) for k, n in enumerate(CLASS_NAMES)},
            "validation": {n: float((labels[val_idx] == k).mean()) for k, n in enumerate(CLASS_NAMES)},
        },
        "stage3_state_head": {
            "fit": metrics(stage3[fit_idx], labels[fit_idx]),
            "validation": metrics(stage3[val_idx], labels[val_idx]),
            "tolerant_occupied_validation": [
                tolerant_occupied(stage3[val_idx], labels[val_idx], r) for r in (1, 2)
            ],
        },
    }

    # ---- Stage 2 with baselines -------------------------------------------
    stage2: dict[str, object] = {}
    for name, ctor in (("linear", LinearCell), ("shallow_nonlinear", ShallowCell)):
        model = fit_probe(ctor(), bev, labels, fit_idx, device)
        pv = predict(model, bev, val_idx, device)
        stage2[name] = {
            "fit": metrics(predict(model, bev, fit_idx, device), labels[fit_idx]),
            "validation": metrics(pv, labels[val_idx]),
            "tolerant_occupied_validation": [
                tolerant_occupied(pv, labels[val_idx], r) for r in (1, 2)
            ],
        }

    # per-cell mean map: no image input at all, only the fit-split cell prior
    prior = np.stack(
        [(labels[fit_idx] == k).mean(axis=0) for k in range(3)]
    ).argmax(0)
    stage2["mean_map_baseline"] = {
        "validation": metrics(np.broadcast_to(prior, labels[val_idx].shape), labels[val_idx])
    }

    # shuffled BEV: derangement of latents against labels
    perm = np.arange(len(fit_idx))
    while True:
        rng.shuffle(perm)
        if not (perm == np.arange(len(perm))).any():
            break
    shuffled = fit_probe(LinearCell(), bev[fit_idx][perm.copy()], labels[fit_idx], np.arange(len(fit_idx)), device)
    stage2["shuffled_bev_baseline"] = {
        "validation": metrics(predict(shuffled, bev, val_idx, device), labels[val_idx])
    }
    result["stage2_bev_state"] = stage2

    # ---- Stage 1: only because stage 2 genuinely fails on occupied ---------
    tokens = torch.cat(token_cache, 0)
    tok_model = fit_probe(TokenToBev(), tokens, labels, fit_idx, device, epochs=30, lr=1e-3)
    tok_pred = predict(tok_model, tokens, val_idx, device)
    result["stage1_vit_tokens"] = {
        "decoder": "fixed: 1x1 conv 192->32, flatten, 1024 hidden, linear to 64x64x3",
        "caveat": "failure is probe-specific recoverability, not proof of absence",
        "fit": metrics(predict(tok_model, tokens, fit_idx, device), labels[fit_idx]),
        "validation": metrics(tok_pred, labels[val_idx]),
        "tolerant_occupied_validation": [
            tolerant_occupied(tok_pred, labels[val_idx], r) for r in (1, 2)
        ],
    }

    # per-scene validation, NA where a scene has no occupied support
    per_scene = {}
    for scene in sorted(val):
        sel = np.array([i for i, r in enumerate(rows) if r["scene"] == scene])
        m = metrics(stage3[sel], labels[sel])
        per_scene[scene] = {
            "frames": int(len(sel)),
            "occupied_support": m["occupied_support"],
            "stage3_observable_occupied_iou": m.get("observable_occupied_iou"),
            "stage3_observable_free_iou": m.get("observable_free_iou"),
        }
    contributing = sum(1 for v in per_scene.values() if v["stage3_observable_occupied_iou"] is not None)
    result["per_scene_validation"] = per_scene
    result["per_scene_contributing_to_occupied_aggregate"] = contributing
    result["per_scene_total"] = len(per_scene)
    result["wall_seconds"] = time.time() - started

    (OUT / "result.json").write_text(json.dumps(result, indent=2, allow_nan=True))
    print(json.dumps({k: v for k, v in result.items() if k != "per_scene_validation"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
