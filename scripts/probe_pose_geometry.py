#!/usr/bin/env python3
"""Measure whether a pose-aux checkpoint contains usable local metric geometry.

Two contracts are evaluated on a held-out latent cache:
  - encoded_to_encoded: relative pose between observed frames
  - predicted_to_encoded: predictor-generated endpoint to an encoded final goal,
    matching the deployed ``lewm_pose`` planner input contract

Aligned physical poses are the default labels. Command-integrated poses remain an
explicit ablation, not a substitute for physical geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.pose_head import (  # noqa: E402
    RelPoseHead,
    body_relative,
    integrate_world_poses,
    ordered_pair_indices,
)
from probe_lewm_checkpoint import load_model  # noqa: E402


def _corr(a: np.ndarray, b: np.ndarray, rank: bool = False) -> float:
    if rank:
        a = np.argsort(np.argsort(a)).astype(np.float64)
        b = np.argsort(np.argsort(b)).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def _metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    true_dist = np.linalg.norm(target[:, :2], axis=1)
    pred_dist = np.linalg.norm(pred[:, :2], axis=1)
    true_bearing = np.arctan2(target[:, 1], target[:, 0])
    pred_bearing = np.arctan2(pred[:, 1], pred[:, 0])
    bearing_error = np.abs(
        np.arctan2(
            np.sin(pred_bearing - true_bearing),
            np.cos(pred_bearing - true_bearing),
        )
    )
    yaw_error = np.abs(
        np.arctan2(
            np.sin(pred[:, 2] - target[:, 2]),
            np.cos(pred[:, 2] - target[:, 2]),
        )
    )
    result = {
        "n_pairs": int(target.shape[0]),
        "pose_xy_err_m": float(np.linalg.norm(pred[:, :2] - target[:, :2], axis=1).mean()),
        "pose_yaw_err_rad": float(yaw_error.mean()),
        "mean_predictor_xy_err_m": float(
            np.linalg.norm(target[:, :2] - target[:, :2].mean(axis=0), axis=1).mean()
        ),
        "dist_pearson": _corr(pred_dist, true_dist),
        "dist_spearman": _corr(pred_dist, true_dist, rank=True),
        "bearing_err_rad": float(bearing_error.mean()),
        "true_dist_span_m": [
            float(true_dist.min()),
            float(np.median(true_dist)),
            float(true_dist.max()),
        ],
    }
    bands = {}
    for name, low, high in (
        ("0p0_0p2m", 0.0, 0.2),
        ("0p2_0p5m", 0.2, 0.5),
        ("0p5_1p0m", 0.5, 1.0),
        ("1p0_2p0m", 1.0, 2.0),
        ("2p0_inf_m", 2.0, float("inf")),
    ):
        keep = (true_dist >= low) & (true_dist < high)
        if int(keep.sum()) >= 3:
            bands[name] = {
                "n_pairs": int(keep.sum()),
                "pose_xy_err_m": float(
                    np.linalg.norm(pred[keep, :2] - target[keep, :2], axis=1).mean()
                ),
                "dist_pearson": _corr(pred_dist[keep], true_dist[keep]),
                "dist_spearman": _corr(pred_dist[keep], true_dist[keep], rank=True),
                "bearing_err_rad": float(bearing_error[keep].mean()),
            }
    result["distance_bands"] = bands
    return result


@torch.no_grad()
def _decode_pairs(
    head: RelPoseHead,
    za: torch.Tensor,
    zb: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> dict:
    pred = []
    for start in range(0, za.shape[0], 8192):
        pred.append(
            head(
                za[start : start + 8192].to(device),
                zb[start : start + 8192].to(device),
            ).cpu()
        )
    return _metrics(torch.cat(pred).numpy(), target.cpu().numpy())


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-head-ckpt", type=Path, required=True)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="World-model checkpoint for predicted-to-encoded evaluation. "
        "Defaults to the pose checkpoint's source_checkpoint or cache checkpoint.",
    )
    ap.add_argument("--eval-cache", type=Path, required=True)
    ap.add_argument("--pose-label-source", choices=("actual", "command"), default="actual")
    ap.add_argument("--command-dt-s", type=float, default=0.10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    pck = torch.load(args.pose_head_ckpt.resolve(), map_location=device, weights_only=False)
    head = RelPoseHead(
        latent_dim=int(pck["latent_dim"]),
        hidden=int(pck.get("hidden", 512)),
    ).to(device)
    head.load_state_dict(pck["head_state_dict"])
    head.eval()
    dt = float(pck.get("command_dt_s", args.command_dt_s))

    cache = torch.load(args.eval_cache, map_location="cpu", weights_only=False)
    z_raw = cache["z_raw"].float()
    z_proj = cache["z_proj"].float()
    cmd = cache["cmd"].float()
    if args.pose_label_source == "actual":
        if "pose" not in cache:
            raise SystemExit(
                "eval cache has no aligned physical poses; rebuild it with --include-pose-labels"
            )
        poses = cache["pose"].float()
    else:
        poses = integrate_world_poses(cmd, dt)

    n, t, latent_dim = z_proj.shape
    a_idx, b_idx = ordered_pair_indices(t, torch.device("cpu"))
    encoded_target = body_relative(poses, a_idx, b_idx).reshape(-1, 3)
    encoded = _decode_pairs(
        head,
        z_proj[:, a_idx].reshape(-1, latent_dim),
        z_proj[:, b_idx].reshape(-1, latent_dim),
        encoded_target,
        device,
    )

    checkpoint = args.checkpoint or pck.get("source_checkpoint") or cache.get("checkpoint")
    if checkpoint is None:
        raise SystemExit("no world-model checkpoint available for predicted-to-encoded evaluation")
    checkpoint = Path(checkpoint)
    model, _ = load_model(
        SimpleNamespace(checkpoint=checkpoint.resolve(), max_seq_len=None, sigreg_lambda=None),
        device,
    )
    endpoint_indices = torch.arange(1, t - 1)
    predicted = []
    for endpoint in endpoint_indices.tolist():
        predicted.append(
            model.plan_rollout(
                z_raw[:, 0].to(device),
                cmd[:, :endpoint].to(device),
            )[:, -1].cpu()
        )
    predicted_endpoint = torch.stack(predicted, dim=1)
    goal = z_proj[:, -1:, :].expand(-1, predicted_endpoint.shape[1], -1)
    goal_idx = torch.full_like(endpoint_indices, t - 1)
    predicted_target = body_relative(poses, endpoint_indices, goal_idx).reshape(-1, 3)
    predicted_contract = _decode_pairs(
        head,
        predicted_endpoint.reshape(-1, latent_dim),
        goal.reshape(-1, latent_dim),
        predicted_target,
        device,
    )

    summary = {
        "schema": "pose_geometry_v1",
        "pose_head_ckpt": str(args.pose_head_ckpt.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "eval_cache": str(args.eval_cache.resolve()),
        "pose_label_source": args.pose_label_source,
        "epoch": pck.get("epoch"),
        "n_windows": int(n),
        "seq_len": int(t),
        "encoded_to_encoded": encoded,
        "predicted_to_encoded": predicted_contract,
        # Backward-compatible aliases for existing report readers.
        **{key: value for key, value in encoded.items() if key != "distance_bands"},
    }
    print(
        f"epoch {summary['epoch']} | labels {args.pose_label_source} | "
        f"encoded xy {encoded['pose_xy_err_m']:.3f}m rho {encoded['dist_pearson']:+.3f} | "
        f"predicted->goal xy {predicted_contract['pose_xy_err_m']:.3f}m "
        f"rho {predicted_contract['dist_pearson']:+.3f}",
        flush=True,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
