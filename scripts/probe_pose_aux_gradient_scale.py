#!/usr/bin/env python3
"""Measure pose-aux versus base-objective encoder gradient scale before a run.

The report gives unweighted loss and encoder-gradient norms for the base LeWM,
encoded-pair pose, and predictor-endpoint-to-goal pose objectives. Suggested
weights target auxiliary contributions of 10% and 30% of the base encoder
gradient; they are screening values, not promotion criteria.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.pose_head import RelPoseHead, pose_aux_loss, predicted_pose_aux_loss  # noqa: E402
from probe_lewm_checkpoint import load_model  # noqa: E402
from train_lewm import GenesisWMDataset, make_loader  # noqa: E402


def _grad_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = sum(
        float(gradient.detach().float().square().sum())
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(squared)


def _mean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--render-root", type=Path, default=None)
    ap.add_argument("--seq-len", type=int, default=11)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--max-sessions", type=int, default=8)
    ap.add_argument("--max-batches", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--pose-hidden", type=int, default=512)
    ap.add_argument("--pose-label-source", choices=("actual", "command"), default="actual")
    ap.add_argument("--command-dt-s", type=float, default=0.10)
    ap.add_argument("--holdout-fraction", type=float, default=0.02)
    ap.add_argument("--holdout-seed", type=int, default=20260524)
    ap.add_argument("--allow-material-color-render", action="store_true")
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--sigreg-lambda", type=float, default=None)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--seed", type=int, default=20260606)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    if args.batch_size < 2:
        ap.error("--batch-size must be at least 2 for the BatchNorm projectors")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    model, config = load_model(args, device)
    model.train()
    head = RelPoseHead(latent_dim=model.latent_dim, hidden=args.pose_hidden).to(device)
    head.train()
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]

    render_root = args.render_root or (
        Path(config["render_root"]) if "render_root" in config else None
    )
    dataset = GenesisWMDataset(
        root_dir=args.data_root,
        render_root=render_root,
        seq_len=args.seq_len,
        stride=args.stride,
        max_sessions=args.max_sessions,
        allow_material_color_render=(
            args.allow_material_color_render
            or bool(config.get("allow_material_color_render", False))
        ),
        holdout_fraction=args.holdout_fraction,
        holdout_role="train",
        holdout_seed=args.holdout_seed,
        include_pose_labels=args.pose_label_source == "actual",
    )
    loader = make_loader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=2,
    )

    rows = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.max_batches:
            break
        vis = batch["vis_seq"].to(device)
        cmd = batch["cmd_seq"].to(device)
        poses = batch.get("pose_seq")
        if poses is not None:
            poses = poses.to(device)
        out = model(
            vis_seq=vis,
            prop_seq=None,
            cmd_seq=cmd,
            rollout_lambda=float(config.get("rollout_lambda", 0.0)),
            return_latents=True,
        )
        encoded_loss, _ = pose_aux_loss(
            head, out["z_proj"], cmd, args.command_dt_s, poses=poses
        )
        predicted_loss, _ = predicted_pose_aux_loss(
            head,
            model,
            out["z_raw"],
            out["z_proj"],
            cmd,
            args.command_dt_s,
            poses=poses,
        )
        rows.append(
            {
                "base_loss": float(out["loss"].detach()),
                "encoded_pose_loss": float(encoded_loss.detach()),
                "predicted_pose_loss": float(predicted_loss.detach()),
                "base_encoder_grad_norm": _grad_norm(out["loss"], encoder_parameters),
                "encoded_pose_encoder_grad_norm": _grad_norm(
                    encoded_loss, encoder_parameters
                ),
                "predicted_pose_encoder_grad_norm": _grad_norm(
                    predicted_loss, encoder_parameters
                ),
            }
        )
    if not rows:
        raise SystemExit("no complete batches available for gradient probe")

    aggregate = {key: _mean(rows, key) for key in rows[0]}
    base_norm = aggregate["base_encoder_grad_norm"]
    suggestions = {}
    for objective in ("encoded_pose", "predicted_pose"):
        aux_norm = aggregate[f"{objective}_encoder_grad_norm"]
        suggestions[objective] = {
            "lambda_for_0p1x_base_grad": 0.1 * base_norm / aux_norm if aux_norm > 0 else None,
            "lambda_for_0p3x_base_grad": 0.3 * base_norm / aux_norm if aux_norm > 0 else None,
        }
    report = {
        "schema": "pose_aux_gradient_scale_v0",
        "checkpoint": str(args.checkpoint.resolve()),
        "pose_label_source": args.pose_label_source,
        "n_batches": len(rows),
        "batch_size": args.batch_size,
        "aggregate": aggregate,
        "suggested_screening_weights": suggestions,
        "batches": rows,
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
