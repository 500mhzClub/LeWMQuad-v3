#!/usr/bin/env python3
"""Target-space alignment audit: can predicted tokens be mapped onto the
canonical true-future V-JEPA token interface without occupancy labels?

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  Read-only: no encoder moves, no predictor is
retrained, no probe is refitted.

Two label-free channel-space maps per arm, fit on TRAIN-role tokens only:

  1. mean-centred orthogonal Procrustes -- one shared 1024x1024 orthogonal
     transform across every token position, no per-position parameters;
  2. shared affine ridge -- one shared 1024x1024 map plus bias across every
     position, fixed regularisation declared below before any selection result
     is read, no hidden layers, no position-specific parameters.

Both are fit against **true future encoder tokens** only.  Neither sees a raster
label.  Both are applied unchanged to checkpoint_selection predicted tokens and
consumed by the EXISTING fixed probe that was trained on true-future encoder
tokens.  No occupancy probe is refitted after alignment.

The maps are diagnostic.  They do not replace the predictor, the fixed probe or
the operational acceptance criterion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_dev_frozen_dense_representation_screen_v1 as S  # noqa: E402
from scripts import dev_frozen_dense_representation_encoders_v1 as E  # noqa: E402
from scripts import run_dev_v03_temporal_action_jepa_v1 as T  # noqa: E402
from scripts import eval_dev_v03_temporal_action_jepa_v1 as V  # noqa: E402
from scripts import complete_dev_v03_temporal_action_jepa_evaluation_v1 as C  # noqa: E402

CACHE = Path("/home/andrewknowles/.cache/lewm_go2_temporal_v03")
EVAL = CACHE / "temporal_action_jepa_v1" / "evaluation"
COMPLETION = CACHE / "temporal_action_jepa_v1" / "completion"
DIAG = CACHE / "temporal_action_jepa_v1" / "predicted_token_diagnostic"
OUT = CACHE / "temporal_action_jepa_v1" / "alignment_audit"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
EPOCH = 5

# --- declared BEFORE any selection result is read ---------------------------
RIDGE_RELATIVE_LAMBDA = 1e-2   # scaled by mean(diag(Xc^T Xc)); no tuning
CHUNK = 64                     # sequences per Gram-accumulation chunk
DERANGEMENT_SEEDS = (11, 23, 37)


def load(path: Path, count: int, tokens: int, dim: int) -> torch.Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(
            np.memmap(path, dtype=np.float16, mode="r", shape=(count, tokens, dim))
        )
    )


def normalise_chunks(tensor, step=256):
    return torch.cat(
        [T.normalise(tensor[i : min(i + step, len(tensor))].float()).half()
         for i in range(0, len(tensor), step)], 0
    )


@torch.no_grad()
def gram(x_source, y_target, device, dim):
    """Means and cross-products over all token positions, accumulated in float64."""
    n = 0
    sx = torch.zeros(dim, dtype=torch.float64, device=device)
    sy = torch.zeros(dim, dtype=torch.float64, device=device)
    for start in range(0, len(x_source), CHUNK):
        stop = min(start + CHUNK, len(x_source))
        a = x_source[start:stop].to(device).float().reshape(-1, dim).double()
        b = y_target[start:stop].to(device).float().reshape(-1, dim).double()
        sx += a.sum(0); sy += b.sum(0); n += a.shape[0]
    mx, my = sx / n, sy / n
    xx = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    xy = torch.zeros(dim, dim, dtype=torch.float64, device=device)
    for start in range(0, len(x_source), CHUNK):
        stop = min(start + CHUNK, len(x_source))
        a = x_source[start:stop].to(device).float().reshape(-1, dim).double() - mx
        b = y_target[start:stop].to(device).float().reshape(-1, dim).double() - my
        xx += a.T @ a; xy += a.T @ b
    return mx, my, xx, xy, n


@torch.no_grad()
def residual(x_source, y_target, mean_x, mean_y, matrix, device, dim):
    """Mean squared error of the aligned source against the target."""
    total, count = 0.0, 0
    for start in range(0, len(x_source), CHUNK):
        stop = min(start + CHUNK, len(x_source))
        a = x_source[start:stop].to(device).float().reshape(-1, dim).double() - mean_x
        b = y_target[start:stop].to(device).float().reshape(-1, dim).double() - mean_y
        total += float(((a @ matrix) - b).pow(2).sum()); count += a.numel()
    return total / count


@torch.no_grad()
def apply_map(tokens, mean_x, mean_y, matrix, device, dim):
    out = []
    for start in range(0, len(tokens), CHUNK):
        stop = min(start + CHUNK, len(tokens))
        a = tokens[start:stop].to(device).float()
        shape = a.shape
        flat = a.reshape(-1, dim).double() - mean_x
        out.append(((flat @ matrix) + mean_y).reshape(shape).half().cpu())
    return torch.cat(out, 0)


def token_scores(pred, current, future, mask):
    pred = pred.float()
    cos_all = F.cosine_similarity(pred, future, dim=-1)
    err_all = (pred - future).pow(2).mean(-1)
    base_all = (current - future).pow(2).mean(-1)
    return {
        "cosine_all_tokens": float(cos_all.mean()),
        "normalised_error_all_tokens": float(err_all.mean() / base_all.mean().clamp_min(1e-12)),
        "cosine_changed_tokens": float(cos_all[mask].mean()),
        "normalised_error_changed_tokens": float(
            err_all[mask].mean() / base_all[mask].mean().clamp_min(1e-12)
        ),
    }


def spatial_block(probe, tokens, labels, families, grid, device):
    gpu = tokens.half().to(device)
    prediction = S.predict(probe, gpu, np.arange(len(tokens)), grid, device)
    del gpu
    torch.cuda.empty_cache()
    s = S.summarise(prediction, labels)
    per_family = S.grouped(prediction, labels, families)
    return {
        "observable_occupied_precision": s["observable_occupied_precision"],
        "observable_occupied_recall": s["observable_occupied_recall"],
        "observable_occupied_iou": s["observable_occupied_iou"],
        "unknown_iou": s["unknown_iou"],
        "observable_free_iou": s["observable_free_iou"],
        "all_free_baseline_free_iou": s["all_free_baseline"]["observable_free_iou"],
        "predicted_occupied_fraction_all_cells": s["predicted_class_fraction_over_all_cells"]["occupied"],
        "target_occupied_fraction_all_cells": s["target_class_fraction_over_all_cells"]["occupied"],
        "denominators": s["denominators"],
        "per_family": per_family,
        "open_obstacle_field": per_family.get("open_obstacle_field"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = torch.device(args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_rows, sel_rows = T.load_rows()
    ordered = train_rows + sel_rows
    n_train, n_sel = len(train_rows), len(sel_rows)
    arm = E.VJepa21CroppedV03Arm()
    grid, dim = arm.token_grid, arm.token_dim
    tokens = grid[0] * grid[1]
    families = [r["family"] for r in sel_rows]
    labels_future_sel = C.future_labels(sel_rows)

    fixed = S.SharedTokenToBev(dim).to(device)
    fixed.load_state_dict(torch.load(COMPLETION / "future_token_probe.pt", map_location=device))
    fixed.eval()

    completion = json.loads((COMPLETION / "result.json").read_text())
    diagnostic = json.loads((DIAG / "result.json").read_text())

    # shared changed-token mask, frozen-derived, exactly as in the completion pass
    frozen_current = load(EVAL / "frozen_current.f16", len(ordered), tokens, dim)
    frozen_sel_future = load(EVAL / "frozen_sel_future.f16", n_sel, tokens, dim)
    threshold = completion["changed_token_mask"]["threshold"]
    shared_mask = (
        T.normalise(frozen_sel_future.float())
        - T.normalise(frozen_current[n_train:].float())
    ).pow(2).mean(-1) >= threshold

    record = {
        "status": STATUS, "claim_bearing": False,
        "read_only": "no encoder moved, no predictor retrained, no probe refitted after alignment",
        "fixed_probe": {
            "path": str(COMPLETION / "future_token_probe.pt"),
            "trained_on": "normalised FROZEN true-future encoder tokens, train role",
            "labels": "raster_labels of the t+240 endpoint",
            "refitted_after_alignment": False,
        },
        "alignment_contract": {
            "procrustes": "mean-centred orthogonal, one shared 1024x1024, no per-position map, label-free",
            "affine_ridge": "one shared 1024x1024 + bias, no hidden layers, no per-position parameters, label-free",
            "ridge_relative_lambda": RIDGE_RELATIVE_LAMBDA,
            "lambda_rule": "lambda = RIDGE_RELATIVE_LAMBDA * mean(diag(Xc^T Xc)); declared before reading any selection result; not tuned",
            "fit_on": "train-role tokens only",
            "target": "normalised true-future encoder tokens of the SAME arm",
        },
        "changed_token_mask": {
            "threshold": threshold,
            "changed": int(shared_mask.sum()), "total": int(shared_mask.numel()),
            "derived_from": "frozen arm train representation (unchanged from the completion pass)",
        },
        "arms": {},
    }
    del frozen_current, frozen_sel_future
    caveat = (
        "the fresh 25M diagnostic probe may exploit nonlinear information and a "
        "different occupied-class operating point; a shared linear/orthogonal map "
        "closing the gap is evidence for an interface mismatch, but the fresh-probe "
        "result alone does not prove a basis rotation"
    )
    record["caveat"] = caveat

    for name in ("frozen", "moving"):
        checkpoint = torch.load(
            CACHE / "temporal_action_jepa_v1" / f"arm_{name}" / f"checkpoint_epoch{EPOCH}.pt",
            map_location="cpu",
        )
        predictor = T.Predictor().to(device)
        predictor.load_state_dict(checkpoint["predictor"])
        predictor.eval()

        pred_train = load(DIAG / f"{name}_predicted_train.f16", n_train, tokens, dim)
        pred_sel = load(DIAG / f"{name}_predicted_selection.f16", n_sel, tokens, dim)
        true_train = normalise_chunks(load(EVAL / f"{name}_train_future.f16", n_train, tokens, dim))
        true_sel = normalise_chunks(load(EVAL / f"{name}_sel_future.f16", n_sel, tokens, dim))
        current = load(EVAL / f"{name}_current.f16", len(ordered), tokens, dim)
        persistence_sel = T.normalise(current[n_train:].float()).half()

        mx, my, xx, xy, count = gram(pred_train, true_train, device, dim)

        # 1. orthogonal Procrustes
        u, _, vh = torch.linalg.svd(xy)
        rotation = (u @ vh)
        procrustes_train = residual(pred_train, true_train, mx, my, rotation, device, dim)
        procrustes_sel = residual(pred_sel, true_sel, mx, my, rotation, device, dim)

        # 2. shared affine ridge
        lam = RIDGE_RELATIVE_LAMBDA * float(torch.diagonal(xx).mean())
        affine = torch.linalg.solve(
            xx + lam * torch.eye(dim, dtype=torch.float64, device=device), xy
        )
        affine_train = residual(pred_train, true_train, mx, my, affine, device, dim)
        affine_sel = residual(pred_sel, true_sel, mx, my, affine, device, dim)
        singular = torch.linalg.svdvals(affine)
        identity_train = residual(
            pred_train, true_train, torch.zeros_like(mx), torch.zeros_like(my),
            torch.eye(dim, dtype=torch.float64, device=device), device, dim,
        )

        variants = {
            "raw_predicted": pred_sel,
            "procrustes_aligned": apply_map(pred_sel, mx, my, rotation, device, dim),
            "affine_aligned": apply_map(pred_sel, mx, my, affine, device, dim),
            "persistence": persistence_sel,
            "true_future_reference": true_sel,
        }

        entry = {
            "alignment_fit": {
                "train_tokens_used": count,
                "identity_baseline_train_residual": identity_train,
                "procrustes_train_residual": procrustes_train,
                "procrustes_selection_residual": procrustes_sel,
                "procrustes_train_to_selection_gap": procrustes_sel - procrustes_train,
                "affine_lambda_absolute": lam,
                "affine_train_residual": affine_train,
                "affine_selection_residual": affine_sel,
                "affine_train_to_selection_gap": affine_sel - affine_train,
                "affine_singular_value_max": float(singular[0]),
                "affine_singular_value_min": float(singular[-1]),
                "affine_condition_number": float(singular[0] / singular[-1].clamp_min(1e-30)),
                "affine_singular_value_deciles": [
                    float(singular[int(q * (dim - 1))]) for q in np.linspace(0, 1, 11)
                ],
                "procrustes_is_orthogonal_max_dev": float(
                    (rotation.T @ rotation
                     - torch.eye(dim, dtype=torch.float64, device=device)).abs().max()
                ),
            },
            "variants": {},
        }
        for tag, block in variants.items():
            entry["variants"][tag] = {
                "tokens": token_scores(block, persistence_sel.float(), true_sel.float(), shared_mask),
                "fixed_probe": spatial_block(fixed, block, labels_future_sel, families, grid, device),
            }

        # action margin under the SAME map applied to correct and shuffled
        actions = T.action_tensor([r["primitive"] for r in sel_rows], torch.device("cpu"))
        ctx = torch.stack(
            [load(EVAL / f"{name}_ctx{k}.f16", n_sel, tokens, dim) for k in range(3)], dim=1
        )
        context = T.normalise(ctx.float()).half()
        del ctx

        def predict_with(action_tensor):
            out = []
            for start in range(0, n_sel, args.batch):
                stop = min(start + args.batch, n_sel)
                with torch.no_grad():
                    out.append(
                        predictor(
                            context[start:stop].to(device=device, dtype=torch.float32),
                            action_tensor[start:stop].to(device),
                            torch.ones(stop - start, tokens, dtype=torch.bool, device=device),
                        ).half().cpu()
                    )
            return torch.cat(out, 0)

        correct = predict_with(actions)
        orders = [C.derangement(n_sel, s) for s in DERANGEMENT_SEEDS]
        margins = {}
        for tag, matrix in (("raw", None), ("procrustes", rotation), ("affine", affine)):
            def mapped(t):
                return t if matrix is None else apply_map(t, mx, my, matrix, device, dim)
            c = float(F.cosine_similarity(
                mapped(correct).float(), true_sel.float(), dim=-1)[shared_mask].mean())
            sh = float(np.mean([
                float(F.cosine_similarity(
                    mapped(predict_with(actions[o])).float(), true_sel.float(),
                    dim=-1)[shared_mask].mean())
                for o in orders
            ]))
            margins[tag] = {
                "correct_changed_cosine": c,
                "shuffled_changed_cosine": sh,
                "correct_minus_shuffled": c - sh,
            }
        entry["action_margin_under_map"] = margins
        entry["diagnostic_fresh_probe_reference"] = {
            "observable_occupied_iou": diagnostic["probes"][f"{name}_predicted"]["observable_occupied_iou"],
            "observable_occupied_precision": diagnostic["probes"][f"{name}_predicted"]["observable_occupied_precision"],
            "note": "fresh 25M probe refitted on predicted tokens; diagnostic only, not comparable operating point",
        }
        record["arms"][name] = entry
        del predictor, pred_train, pred_sel, true_train, current, context, correct
        torch.cuda.empty_cache()
        (OUT / "result.json").write_text(json.dumps(record, indent=2))

    summary = {}
    for name in ("frozen", "moving"):
        v = record["arms"][name]["variants"]
        summary[name] = {
            k: v[k]["fixed_probe"]["observable_occupied_iou"]
            for k in ("raw_predicted", "procrustes_aligned", "affine_aligned",
                      "persistence", "true_future_reference")
        }
        summary[name]["fresh_probe_diagnostic"] = (
            record["arms"][name]["diagnostic_fresh_probe_reference"]["observable_occupied_iou"]
        )
    record["summary_occupied_iou"] = summary
    record["wall_seconds"] = round(time.time() - started, 1)
    (OUT / "result.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
