#!/usr/bin/env python3
"""Train a bounded Phase 2D cell using corrected masks and hard negatives."""
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

from lewm.benchmarks.phase2_data import (  # noqa: E402
    build_hard_negative_index,
    load_spatial_future_rows,
    pairwise_split_overlap,
    phase2_dataset_audit,
    source_grouped_batches,
)
from lewm.benchmarks.phase2d_training import (  # noqa: E402
    ACTION_UTILITY_TARGET_VERSION,
    CONSEQUENCE_TARGET_DIM,
    Phase2DCell,
    action_utility_selection_records,
    action_utility_selection_summary,
    batch_contract_audit,
    checkpoint_rule_record,
    materialize_phase2d_batch,
    prediction_control_records,
    primary_source_state_prediction_table,
    registered_cell,
)
from lewm.benchmarks.phase2d_readiness import (  # noqa: E402
    phase2d_training_start_readiness,
)
from lewm.benchmarks.rollout_diagnostics import summarize_spatial_stability  # noqa: E402
from lewm.models.phase2d_spatial_lewm import Phase2DSpatialLeWorldModel  # noqa: E402
from lewm.models.spatial_predictor import trainable_parameter_count  # noqa: E402


MODEL_CONSTANTS = {
    "latent_dim": 48,
    "pred_layers": 2,
    "pred_heads": 4,
    "pred_dim_head": 12,
    "pred_mlp_dim": 96,
    "image_size": 224,
    "patch_size": 14,
    "encoder_depth": 2,
    "encoder_heads": 3,
    "encoder_mlp_ratio": 2,
    "appearance_sigreg_lambda": 0.09,
    "spatial_variance_lambda": 1.0,
    "sigreg_projections": 64,
    "sigreg_knots": 9,
}

METRIC_KEYS = (
    "loss",
    "prediction_loss",
    "action_identifiability_loss",
    "zero_action_loss",
    "consequence_loss",
    "consequence_mse",
    "action_utility_loss",
    "action_utility_ce_loss",
    "action_utility_regression_loss",
    "appearance_sigreg_loss",
    "spatial_variance_loss",
    "real_prediction_mse",
    "hard_negative_mse",
    "zero_action_mse",
    "mean_target_change_mse",
)


def build_model(
    cell: Phase2DCell,
    *,
    command_dim: int,
    overrides: dict | None = None,
) -> Phase2DSpatialLeWorldModel:
    """Build one registered cell with only explicit smoke-test overrides."""

    config = dict(MODEL_CONSTANTS)
    if overrides:
        config.update(overrides)
    return Phase2DSpatialLeWorldModel(
        cmd_dim=command_dim,
        target_ema_momentum=cell.target_ema_momentum,
        action_identifiability_lambda=cell.action_identifiability_lambda,
        zero_action_lambda=cell.zero_action_lambda,
        prediction_input_mode=cell.prediction_input_mode,
        **config,
    )


def _model_inputs(
    cell: Phase2DCell,
    batch,
    *,
    include_controls: bool = False,
) -> dict:
    inputs = {
        "vision": batch.vision,
        "cmd_seq": batch.actions,
        "transition_mask": batch.transition_mask,
        "consequence_targets": batch.consequence_targets,
        "consequence_mask": batch.consequence_mask,
        "action_utility_targets": batch.action_utility_targets,
        "action_utility_mask": batch.action_utility_mask,
        "action_utility_group_ids": batch.action_utility_group_ids,
    }
    if cell.name == "C2" or include_controls:
        inputs.update(
            {
                "wrong_actions": batch.wrong_actions,
                "wrong_mask": batch.wrong_mask,
                "non_hold_mask": batch.non_hold_mask,
            }
        )
    return inputs


def _mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in METRIC_KEYS
    }


def _metric_record(output: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(output[key].detach().cpu()) for key in METRIC_KEYS}


def _assert_finite_metrics(metrics: dict[str, float], *, step: int, phase: str) -> None:
    nonfinite = {
        key: value
        for key, value in metrics.items()
        if not np.isfinite(value)
    }
    if nonfinite:
        raise RuntimeError(
            json.dumps(
                {
                    "error": "nonfinite_phase2d_metrics",
                    "metrics": nonfinite,
                    "phase": phase,
                    "step": int(step),
                },
                sort_keys=True,
            )
        )


@torch.no_grad()
def evaluate(
    model: Phase2DSpatialLeWorldModel,
    cell: Phase2DCell,
    rows: list[dict],
    hard_negatives,
    *,
    source_states_per_batch: int,
    seed: int,
    epoch: int,
    device: torch.device,
) -> dict:
    """Evaluate loss, stability, and per-source-state prediction controls."""

    model.eval()
    records = []
    contracts = []
    pre_normalized = []
    normalized = []
    previous = []
    candidate_step_records = []
    action_utility_records = []
    batches = source_grouped_batches(
        rows,
        source_states_per_batch=source_states_per_batch,
        seed=0,
        shuffle=False,
    )
    for indices in batches:
        batch = materialize_phase2d_batch(
            rows,
            indices,
            hard_negatives=hard_negatives,
        ).to(device)
        output = model(
            **_model_inputs(cell, batch, include_controls=True),
            return_latents=True,
        )
        metric_record = _metric_record(output)
        _assert_finite_metrics(
            metric_record,
            step=len(records) + 1,
            phase="validation_batch",
        )
        records.append(metric_record)
        contracts.append(batch_contract_audit(batch))
        candidate_step_records.extend(
            prediction_control_records(batch, output, seed=seed)
        )
        action_utility_records.extend(
            action_utility_selection_records(batch, output, seed=seed)
        )
        pre_normalized.append(output["target_pre_normalized"].cpu())
        normalized.append(output["target_normalized_all"].cpu())
        previous.append(
            torch.cat(
                [
                    output["target_normalized_all"][:, :1],
                    output["target_normalized_all"][:, :-1],
                ],
                dim=1,
            ).cpu()
        )
    source_state_records = primary_source_state_prediction_table(candidate_step_records)
    stability = summarize_spatial_stability(
        pre_normalized_targets=torch.cat(pre_normalized),
        normalized_targets=torch.cat(normalized),
        previous_normalized_targets=torch.cat(previous),
    )
    checkpoint_rule = checkpoint_rule_record(
        source_state_records,
        epoch=epoch,
        stability=stability,
    )
    return {
        "metrics_unweighted_batch_mean": _mean_records(records),
        "batch_contracts": contracts,
        "candidate_step_prediction_control_records": candidate_step_records,
        "primary_source_state_prediction_control_records": source_state_records,
        "action_utility_selection_records": action_utility_records,
        "action_utility_selection_summary": action_utility_selection_summary(
            action_utility_records
        ),
        "checkpoint_rule_record": checkpoint_rule,
        "stability": stability,
        "checkpoint_selection_permitted": bool(
            cell.participates_in_checkpoint_selection
            and checkpoint_rule["gate_pass"]
        ),
        "checkpoint_selection_basis": {
            "cell_participates_in_checkpoint_selection": (
                cell.participates_in_checkpoint_selection
            ),
            "registered_gate_pass": checkpoint_rule["gate_pass"],
        },
        "limitations": [
            "unweighted batch means are training diagnostics, not confirmatory estimands",
            "source-state records are emitted for confirmatory analysis but pilot runs remain non-confirmatory",
        ],
    }


def _args_for_json(args: argparse.Namespace) -> dict:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _confirmatory_training_start_report(args: argparse.Namespace) -> dict | None:
    if args.run_class != "confirmatory":
        return None
    if args.split_manifest is None:
        raise SystemExit("--split-manifest is required for confirmatory training")
    report = phase2d_training_start_readiness(
        split_manifest_path=args.split_manifest,
        cell=args.cell,
        requested_run_class=args.run_class,
        train_data_path=args.train_data,
        validation_data_path=args.validation_data,
    )
    if not report["passed"]:
        raise SystemExit(
            "Phase 2D confirmatory training preflight failed:\n"
            + json.dumps(report, indent=2, sort_keys=True)
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cell",
        choices=("C0", "C1", "C2", "state_only", "action_only"),
        required=True,
    )
    parser.add_argument(
        "--run-class",
        choices=("smoke", "pilot", "confirmatory"),
        default="smoke",
        help=(
            "Confirmatory execution requires --split-manifest and a passing "
            "training-start preflight."
        ),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        help="Required for confirmatory training; ignored for smoke/pilot runs.",
    )
    parser.add_argument("--optimization-steps", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.0,
        help="Clip gradients to this norm when positive; 0 disables clipping.",
    )
    parser.add_argument(
        "--detach-action-control-state",
        action="store_true",
        help=(
            "Detach the current spatial state for C2 wrong-action and zero-action "
            "contrast branches."
        ),
    )
    parser.add_argument(
        "--target-geometry",
        choices=("patch", "slot"),
        default="patch",
        help=(
            "Prediction/target geometry. 'patch' is registered Phase 2D; 'slot' "
            "is a Phase 2E pilot amendment."
        ),
    )
    parser.add_argument(
        "--num-target-slots",
        type=int,
        default=16,
        help="Number of learned slots when --target-geometry slot is selected.",
    )
    parser.add_argument(
        "--consequence-loss-lambda",
        type=float,
        default=0.0,
        help=(
            "Enable a Phase 2F sequence-level consequence prediction head with "
            "this auxiliary loss weight. Default 0 preserves Phase 2D behavior."
        ),
    )
    parser.add_argument(
        "--action-utility-loss-lambda",
        type=float,
        default=0.0,
        help=(
            "Enable a Phase 2G source-local action-utility ranking head with "
            "this auxiliary loss weight. Default 0 preserves Phase 2D behavior."
        ),
    )
    parser.add_argument(
        "--action-utility-regression-weight",
        type=float,
        default=0.1,
        help=(
            "Regression scale term inside the Phase 2G action-utility loss. "
            "The ranking cross-entropy remains the primary term."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.optimization_steps < 1:
        parser.error("--optimization-steps must be positive")
    if args.evaluation_interval < 1:
        parser.error("--evaluation-interval must be positive")
    if args.max_grad_norm < 0.0:
        parser.error("--max-grad-norm must be non-negative")
    if args.num_target_slots < 1:
        parser.error("--num-target-slots must be positive")
    if args.consequence_loss_lambda < 0.0:
        parser.error("--consequence-loss-lambda must be non-negative")
    if args.action_utility_loss_lambda < 0.0:
        parser.error("--action-utility-loss-lambda must be non-negative")
    if args.action_utility_regression_weight < 0.0:
        parser.error("--action-utility-regression-weight must be non-negative")
    training_start_readiness = _confirmatory_training_start_report(args)

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
    train_rows, train_load_audit = load_spatial_future_rows(
        args.train_data,
        mode="all",
        max_rows=args.max_train_rows,
    )
    validation_rows, validation_load_audit = load_spatial_future_rows(
        args.validation_data,
        mode="all",
        max_rows=args.max_validation_rows,
    )
    overlap = pairwise_split_overlap(
        {"train": train_rows, "validation": validation_rows}
    )
    if any(
        value["scene_ids"] or value["source_keys"]
        for value in overlap.values()
    ):
        raise SystemExit(f"train/validation overlap is prohibited: {overlap}")
    horizons = {len(row["active_blocks"]) for row in train_rows + validation_rows}
    if len(horizons) != 1:
        raise SystemExit(f"all rows must have one common horizon: {sorted(horizons)}")
    horizon = next(iter(horizons))
    command_dim = len(train_rows[0]["active_blocks"][0])
    train_negatives = [
        build_hard_negative_index(train_rows, step=step) for step in range(horizon)
    ]
    validation_negatives = [
        build_hard_negative_index(validation_rows, step=step)
        for step in range(horizon)
    ]
    cell = registered_cell(args.cell)
    model_overrides = {}
    if args.detach_action_control_state:
        model_overrides["detach_action_control_state"] = True
    if args.target_geometry != "patch":
        model_overrides["target_geometry"] = args.target_geometry
        model_overrides["num_target_slots"] = args.num_target_slots
    if args.consequence_loss_lambda > 0.0:
        model_overrides["consequence_dim"] = CONSEQUENCE_TARGET_DIM
        model_overrides["consequence_loss_lambda"] = args.consequence_loss_lambda
    if args.action_utility_loss_lambda > 0.0:
        model_overrides["action_utility_loss_lambda"] = (
            args.action_utility_loss_lambda
        )
        model_overrides["action_utility_regression_weight"] = (
            args.action_utility_regression_weight
        )
    model = build_model(cell, command_dim=command_dim, overrides=model_overrides).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    completed_steps = 0
    epoch = 0
    while completed_steps < args.optimization_steps:
        model.train()
        batches = source_grouped_batches(
            train_rows,
            source_states_per_batch=args.source_states_per_batch,
            seed=args.seed + epoch,
            shuffle=True,
        )
        for indices in batches:
            batch = materialize_phase2d_batch(
                train_rows,
                indices,
                hard_negatives=train_negatives,
            ).to(device)
            output = model(**_model_inputs(cell, batch))
            train_metrics = _metric_record(output)
            _assert_finite_metrics(
                train_metrics,
                step=completed_steps + 1,
                phase="train",
            )
            optimizer.zero_grad(set_to_none=True)
            output["loss"].backward()
            gradient_norm = None
            if args.max_grad_norm > 0.0:
                gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float(args.max_grad_norm),
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(gradient_norm_tensor):
                    raise RuntimeError(
                        json.dumps(
                            {
                                "error": "nonfinite_phase2d_gradient_norm",
                                "gradient_norm": float(
                                    gradient_norm_tensor.detach().cpu()
                                ),
                                "step": int(completed_steps + 1),
                            },
                            sort_keys=True,
                        )
                    )
                gradient_norm = float(gradient_norm_tensor.detach().cpu())
            optimizer.step()
            model.update_target_encoder()
            completed_steps += 1
            record = {
                "optimization_step": completed_steps,
                "epoch": epoch + 1,
                "train_metrics": train_metrics,
                "batch_contract": batch_contract_audit(batch),
            }
            if gradient_norm is not None:
                record["gradient_norm"] = gradient_norm
            if (
                completed_steps % args.evaluation_interval == 0
                or completed_steps == args.optimization_steps
            ):
                record["validation_interface_diagnostic"] = evaluate(
                    model,
                    cell,
                    validation_rows,
                    validation_negatives,
                    source_states_per_batch=args.source_states_per_batch,
                    seed=args.seed,
                    epoch=epoch + 1,
                    device=device,
                )
            history.append(record)
            print(json.dumps(record), flush=True)
            if completed_steps >= args.optimization_steps:
                break
        epoch += 1

    final_validation = next(
        (
            record["validation_interface_diagnostic"]
            for record in reversed(history)
            if "validation_interface_diagnostic" in record
        ),
        None,
    )
    report = {
        "schema": "jepa_phase2d_training_pilot_v0",
        "run_class": args.run_class,
        "confirmatory_result": False,
        "confirmatory_training": args.run_class == "confirmatory",
        "training_start_readiness": training_start_readiness,
        "cell": vars(cell),
        "fixed_model_constants": MODEL_CONSTANTS,
        "model_amendments": {
            "detach_action_control_state": args.detach_action_control_state,
            "target_geometry": args.target_geometry,
            "num_target_slots": args.num_target_slots,
            "consequence_dim": (
                CONSEQUENCE_TARGET_DIM
                if args.consequence_loss_lambda > 0.0
                else 0
            ),
            "consequence_loss_lambda": args.consequence_loss_lambda,
            "action_utility_loss_lambda": args.action_utility_loss_lambda,
            "action_utility_regression_weight": (
                args.action_utility_regression_weight
            ),
            "action_utility_target_version": (
                ACTION_UTILITY_TARGET_VERSION
                if args.action_utility_loss_lambda > 0.0
                else None
            ),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "optimization_steps": args.optimization_steps,
            "evaluation_interval": args.evaluation_interval,
        },
        "seed": args.seed,
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model),
        "train_data": {
            "load_audit": train_load_audit,
            "dataset_audit": phase2_dataset_audit(train_rows),
        },
        "validation_data": {
            "load_audit": validation_load_audit,
            "dataset_audit": phase2_dataset_audit(validation_rows),
        },
        "split_overlap": overlap,
        "history": history,
        "final_validation_gate": (
            final_validation["checkpoint_rule_record"]
            if final_validation is not None
            else None
        ),
        "checkpoint_selection_permitted": (
            bool(final_validation["checkpoint_selection_permitted"])
            if final_validation is not None
            else False
        ),
        "limitations": [
            "confirmatory training is not itself a confirmatory result until the registered analysis is run",
            "state-only and action-only controls do not participate in C0-C2 checkpoint selection",
            "validation values are interface diagnostics and not registered hierarchical estimands",
            "kinematic nominal actions are not physics-validated executed actions",
        ],
    }
    payload = {
        "report": report,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "args": _args_for_json(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
