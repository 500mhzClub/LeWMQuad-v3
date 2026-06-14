#!/usr/bin/env python3
"""Train the Phase 2B end-to-end spatial-token JEPA on valid future observations."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from lewm.models.spatial_lewm import SpatialLeWorldModel  # noqa: E402
from lewm.models.spatial_predictor import trainable_parameter_count  # noqa: E402
from train_jepa_spatial_predictor import (  # noqa: E402
    _image_tensor,
    _load_rows,
    _selection_metrics,
)


def _batch(rows: list[dict], indices: list[int], device: torch.device):
    selected = [rows[index] for index in indices]
    vision = torch.stack(
        [
            torch.stack(
                [_image_tensor(Path(row["start_frame"]))]
                + [_image_tensor(Path(path)) for path in row["future_frames"]]
            )
            for row in selected
        ]
    ).to(device)
    actions = torch.tensor(
        [row["active_blocks"] for row in selected],
        dtype=torch.float32,
        device=device,
    )
    return selected, vision, actions


@torch.no_grad()
def _evaluate(
    model: SpatialLeWorldModel,
    rows: list[dict],
    *,
    batch_size: int,
    device: torch.device,
) -> dict:
    model.eval()
    rollout_steps: list[list[float]] = []
    persistence_steps: list[list[float]] = []
    goal_costs = []
    for offset in range(0, len(rows), batch_size):
        indices = list(range(offset, min(offset + batch_size, len(rows))))
        selected, vision, actions = _batch(rows, indices, device)
        _appearance, spatial_raw = model.encode_seq(vision)
        target = model.spatial_projector(spatial_raw[:, 1:])
        rollout = model.rollout_spatial(spatial_raw[:, 0], actions)
        persistence = model.spatial_projector(spatial_raw[:, 0])[:, None].expand_as(
            target
        )
        for storage, values in (
            (rollout_steps, (rollout - target).square().mean(dim=(2, 3))),
            (persistence_steps, (persistence - target).square().mean(dim=(2, 3))),
        ):
            while len(storage) < values.shape[1]:
                storage.append([])
            for step in range(values.shape[1]):
                storage[step].extend(values[:, step].cpu().tolist())
        for row, predicted_final in zip(selected, rollout[:, -1], strict=True):
            goal_path = row.get("goal_frame")
            if row.get("goal_present") and goal_path is not None:
                goal = _image_tensor(Path(goal_path))[None].to(device)
                goal_raw = model.encoder.forward_tokens(goal)[:, 1:]
                goal_proj = model.spatial_projector(goal_raw)[0]
                goal_costs.append(float((predicted_final - goal_proj).square().mean()))
            else:
                goal_costs.append(float("inf"))

    per_horizon = []
    for step in range(len(rollout_steps)):
        rollout_mse = float(np.mean(rollout_steps[step]))
        persistence_mse = float(np.mean(persistence_steps[step]))
        per_horizon.append(
            {
                "step": step + 1,
                "free_running_token_mse": rollout_mse,
                "persistence_token_mse": persistence_mse,
                "free_running_vs_persistence_mse_ratio": (
                    rollout_mse / persistence_mse
                    if persistence_mse > 0.0
                    else float("inf")
                ),
                "free_running_beats_persistence": rollout_mse < persistence_mse,
            }
        )
    return {
        "per_horizon_step": per_horizon,
        "selection": _selection_metrics(rows, goal_costs),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--eval-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=192)
    parser.add_argument("--encoder-depth", type=int, default=12)
    parser.add_argument("--encoder-heads", type=int, default=3)
    parser.add_argument("--encoder-mlp-ratio", type=int, default=4)
    parser.add_argument("--pred-layers", type=int, default=6)
    parser.add_argument("--pred-heads", type=int, default=16)
    parser.add_argument("--pred-dim-head", type=int, default=64)
    parser.add_argument("--pred-mlp-dim", type=int, default=2048)
    parser.add_argument("--appearance-sigreg-lambda", type=float, default=0.09)
    parser.add_argument("--spatial-variance-lambda", type=float, default=1.0)
    parser.add_argument("--spatial-target-std", type=float, default=1.0)
    parser.add_argument("--sigreg-projections", type=int, default=1024)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--allow-scene-overlap", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    train_rows, train_audit = _load_rows(args.train_data, args.max_train_rows)
    eval_rows, eval_audit = _load_rows(args.eval_data, args.max_eval_rows)
    train_scenes = {str(row["scene_id"]) for row in train_rows}
    eval_scenes = {str(row["scene_id"]) for row in eval_rows}
    overlap = sorted(train_scenes & eval_scenes)
    if overlap and not args.allow_scene_overlap:
        raise SystemExit(f"train/eval scene overlap: {overlap[:8]}")
    cmd_dim = len(train_rows[0]["active_blocks"][0])
    model = SpatialLeWorldModel(
        latent_dim=args.latent_dim,
        cmd_dim=cmd_dim,
        pred_layers=args.pred_layers,
        pred_heads=args.pred_heads,
        pred_dim_head=args.pred_dim_head,
        pred_mlp_dim=args.pred_mlp_dim,
        encoder_depth=args.encoder_depth,
        encoder_heads=args.encoder_heads,
        encoder_mlp_ratio=args.encoder_mlp_ratio,
        appearance_sigreg_lambda=args.appearance_sigreg_lambda,
        spatial_variance_lambda=args.spatial_variance_lambda,
        spatial_target_std=args.spatial_target_std,
        sigreg_projections=args.sigreg_projections,
        sigreg_knots=args.sigreg_knots,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        totals = {
            "loss": [],
            "prediction_loss": [],
            "appearance_sigreg_loss": [],
            "spatial_variance_loss": [],
        }
        for offset in range(0, len(order), args.batch_size):
            indices = order[offset : offset + args.batch_size]
            _selected, vision, actions = _batch(train_rows, indices, device)
            output = model(vision, actions)
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            optimizer.step()
            for name in totals:
                totals[name].append(float(output[name].detach()))
        record = {
            "epoch": epoch + 1,
            "train": {name: float(np.mean(values)) for name, values in totals.items()},
            "eval": _evaluate(model, eval_rows, batch_size=args.batch_size, device=device),
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    report = {
        "schema": "jepa_spatial_lewm_training_v0",
        "phase": "2B_end_to_end_spatial_token_jepa",
        "train_data": str(args.train_data.resolve()),
        "eval_data": str(args.eval_data.resolve()),
        "train_input_audit": train_audit,
        "eval_input_audit": eval_audit,
        "scene_overlap": overlap,
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model),
        "training_uses_privileged_consequence_labels": False,
        "anti_collapse": {
            "appearance_sigreg_lambda": args.appearance_sigreg_lambda,
            "spatial_variance_lambda": args.spatial_variance_lambda,
            "spatial_target_std": args.spatial_target_std,
        },
        "final": history[-1],
        "history": history,
        "limitations": [
            "token loss excludes renderer-invalid future observations",
            "kinematic future observations are not physics-validated",
            "goal selection cost is direct position-aligned patch-token MSE",
        ],
    }
    payload = {
        "report": report,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "args": vars(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
