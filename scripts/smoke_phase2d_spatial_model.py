#!/usr/bin/env python3
"""Run a deterministic synthetic smoke check of the Phase 2D model contract."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.rollout_diagnostics import summarize_spatial_stability  # noqa: E402
from lewm.models.phase2d_spatial_lewm import Phase2DSpatialLeWorldModel  # noqa: E402


def _scalar(value: torch.Tensor) -> float | int:
    item = value.detach().cpu().item()
    return int(item) if isinstance(item, int) else float(item)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.manual_seed(20260614)
    model = Phase2DSpatialLeWorldModel(
        latent_dim=12,
        cmd_dim=6,
        pred_layers=1,
        pred_heads=3,
        pred_dim_head=4,
        pred_mlp_dim=24,
        pred_dropout=0.0,
        image_size=28,
        patch_size=14,
        encoder_depth=1,
        encoder_heads=3,
        encoder_mlp_ratio=2,
        action_identifiability_lambda=1.0,
        zero_action_lambda=1.0,
        sigreg_projections=8,
        sigreg_knots=5,
        target_ema_momentum=0.99,
    )
    vision = torch.randn(2, 3, 3, 28, 28)
    actions = torch.randn(2, 2, 6)
    output = model(
        vision,
        actions,
        transition_mask=torch.tensor([[True, True], [True, False]]),
        wrong_actions=torch.randn(2, 2, 2, 6),
        wrong_mask=torch.tensor(
            [
                [[True, True], [True, False]],
                [[True, False], [True, True]],
            ]
        ),
        non_hold_mask=torch.tensor([[True, False], [True, True]]),
        return_latents=True,
    )
    output["loss"].backward()
    stability = summarize_spatial_stability(
        pre_normalized_targets=output["target_pre_normalized"],
        normalized_targets=output["target_normalized_all"],
        previous_normalized_targets=torch.cat(
            [
                output["target_normalized_all"][:, :1],
                output["target_normalized_all"][:, :-1],
            ],
            dim=1,
        ),
    )
    scalar_keys = (
        "loss",
        "prediction_loss",
        "action_identifiability_loss",
        "zero_action_loss",
        "appearance_sigreg_loss",
        "spatial_variance_loss",
        "real_prediction_mse",
        "hard_negative_mse",
        "zero_action_mse",
        "mean_target_change_mse",
        "valid_transition_count",
        "eligible_wrong_transition_count",
        "eligible_wrong_pair_count",
        "eligible_zero_count",
    )
    report = {
        "seed": 20260614,
        "metrics": {key: _scalar(output[key]) for key in scalar_keys},
        "mask_counts": {
            "transition_mask": int(output["transition_mask"].sum()),
            "wrong_pair_mask": int(output["wrong_pair_mask"].sum()),
            "eligible_wrong_mask": int(output["eligible_wrong_mask"].sum()),
            "eligible_zero_mask": int(output["eligible_zero_mask"].sum()),
        },
        "shapes": {
            "target_normalized_all": list(output["target_normalized_all"].shape),
            "real_prediction": list(output["real_prediction"].shape),
            "wrong_predictions": list(output["wrong_predictions"].shape),
            "zero_prediction": list(output["zero_prediction"].shape),
        },
        "gradients": {
            "online_encoder": model.encoder.patch_embed.weight.grad is not None,
            "predictor": any(
                parameter.grad is not None for parameter in model.predictor.parameters()
            ),
            "prediction_projector": (
                model.prediction_projector.linear.weight.grad is not None
            ),
            "ema_target": any(
                parameter.grad is not None
                for module in (model.target_encoder, model.target_projector)
                for parameter in module.parameters()
            ),
        },
        "stability": stability,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
