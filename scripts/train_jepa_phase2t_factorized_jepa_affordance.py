#!/usr/bin/env python3
"""Train a bounded Phase 2T JEPA factorized-affordance integration smoke."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
    Phase2DCell,
    batch_contract_audit,
    checkpoint_rule_record,
    materialize_phase2d_batch,
    prediction_control_records,
    primary_source_state_prediction_table,
    registered_cell,
)
from lewm.benchmarks.phase2m_primitive_affordance import (  # noqa: E402
    build_primitive_affordance_examples,
    evaluate_primitive_action_only_baseline,
    fit_primitive_action_priors,
    primitive_vocabulary,
)
from lewm.benchmarks.phase2o_factorized_affordance import (  # noqa: E402
    FACTORIZED_AFFORDANCE_FACTOR_NAMES,
)
from lewm.benchmarks.phase2t_factorized_jepa_affordance import (  # noqa: E402
    PHASE2T_SEQUENCE_FACTOR_TARGET_VERSION,
    factorized_sequence_primitive_selection_records,
    factorized_sequence_primitive_selection_summary,
    materialize_phase2t_sequence_factor_targets,
    phase2t_sequence_factor_target_audit,
)
from lewm.benchmarks.rollout_diagnostics import summarize_spatial_stability  # noqa: E402
from lewm.models.phase2d_spatial_lewm import Phase2DSpatialLeWorldModel  # noqa: E402
from lewm.models.primitive_affordance import (  # noqa: E402
    factorized_affordance_losses,
    factorized_affordance_values,
)
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

BASE_METRIC_KEYS = (
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
)

METRIC_KEYS = (
    *BASE_METRIC_KEYS,
    "phase2t_total_loss",
    "factorized_affordance_loss",
    "factorized_safety_bce_loss",
    "factorized_value_mse_loss",
)


class FutureFactorizedAffordanceHead(nn.Module):
    """Predict sequence-level factorized affordance values from JEPA futures."""

    def __init__(self, *, latent_dim: int, hidden_dim: int, factor_count: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, factor_count),
        )

    def forward(self, predicted_tokens: torch.Tensor) -> torch.Tensor:
        if predicted_tokens.ndim != 4:
            raise ValueError("predicted_tokens must have shape (B, H, N, D)")
        pooled = predicted_tokens.mean(dim=(1, 2))
        return self.net(pooled)


def build_model(
    cell: Phase2DCell,
    *,
    command_dim: int,
    detach_action_control_state: bool,
) -> Phase2DSpatialLeWorldModel:
    config = dict(MODEL_CONSTANTS)
    if detach_action_control_state:
        config["detach_action_control_state"] = True
    return Phase2DSpatialLeWorldModel(
        cmd_dim=command_dim,
        target_ema_momentum=cell.target_ema_momentum,
        action_identifiability_lambda=cell.action_identifiability_lambda,
        zero_action_lambda=cell.zero_action_lambda,
        prediction_input_mode=cell.prediction_input_mode,
        **config,
    )


def _model_inputs(cell: Phase2DCell, batch, *, include_controls: bool = False) -> dict:
    inputs = {
        "vision": batch.vision,
        "cmd_seq": batch.actions,
        "transition_mask": batch.transition_mask,
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


def _args_for_json(args: argparse.Namespace) -> dict:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in METRIC_KEYS
    }


def _assert_finite_metrics(metrics: dict[str, float], *, step: int, phase: str) -> None:
    nonfinite = {
        key: value for key, value in metrics.items() if not np.isfinite(value)
    }
    if nonfinite:
        raise RuntimeError(
            json.dumps(
                {
                    "error": "nonfinite_phase2t_metrics",
                    "metrics": nonfinite,
                    "phase": phase,
                    "step": int(step),
                },
                sort_keys=True,
            )
        )


def forward_phase2t(
    *,
    model: Phase2DSpatialLeWorldModel,
    head: FutureFactorizedAffordanceHead,
    cell: Phase2DCell,
    batch,
    factor_loss_lambda: float,
    safety_weight: float,
    value_weight: float,
    include_controls: bool,
) -> dict[str, torch.Tensor]:
    output = model(
        **_model_inputs(cell, batch, include_controls=include_controls),
        return_latents=True,
    )
    factor_targets, factor_mask = materialize_phase2t_sequence_factor_targets(
        batch.rows
    )
    factor_targets = factor_targets.to(batch.vision.device)
    factor_mask = factor_mask.to(batch.vision.device)
    factor_logits = head(output["real_prediction"])
    losses = factorized_affordance_losses(
        factor_logits=factor_logits[:, None, :],
        factor_targets=factor_targets[:, None, :],
        factor_mask=factor_mask[:, None, :],
        safety_weight=safety_weight,
        value_weight=value_weight,
    )
    total = output["loss"] + float(factor_loss_lambda) * losses[
        "factorized_affordance_loss"
    ]
    output.update(
        {
            "phase2t_total_loss": total,
            "sequence_factor_logits": factor_logits,
            "sequence_factor_values": factorized_affordance_values(
                factor_logits[:, None, :]
            )[:, 0],
            "sequence_factor_targets": factor_targets,
            "sequence_factor_mask": factor_mask,
            **losses,
        }
    )
    return output


def _metric_record(output: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(output[key].detach().cpu()) for key in METRIC_KEYS}


@torch.no_grad()
def evaluate(
    *,
    model: Phase2DSpatialLeWorldModel,
    head: FutureFactorizedAffordanceHead,
    cell: Phase2DCell,
    rows: list[dict],
    primitive_examples,
    hard_negatives,
    source_states_per_batch: int,
    seed: int,
    epoch: int,
    device: torch.device,
    factor_loss_lambda: float,
    safety_weight: float,
    value_weight: float,
    selection_kwargs: dict,
) -> dict:
    model.eval()
    head.eval()
    metric_records = []
    contracts = []
    pre_normalized = []
    normalized = []
    previous = []
    candidate_step_records = []
    selected_rows = []
    factor_values = []
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
        output = forward_phase2t(
            model=model,
            head=head,
            cell=cell,
            batch=batch,
            factor_loss_lambda=factor_loss_lambda,
            safety_weight=safety_weight,
            value_weight=value_weight,
            include_controls=True,
        )
        metric_record = _metric_record(output)
        _assert_finite_metrics(
            metric_record,
            step=len(metric_records) + 1,
            phase="validation_batch",
        )
        metric_records.append(metric_record)
        contracts.append(batch_contract_audit(batch))
        candidate_step_records.extend(
            prediction_control_records(batch, output, seed=seed)
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
        selected_rows.extend(batch.rows)
        factor_values.append(output["sequence_factor_values"].detach().cpu())
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
    selection_records = factorized_sequence_primitive_selection_records(
        selected_rows,
        torch.cat(factor_values),
        primitive_examples,
        seed=seed,
        split_name="validation",
        scorer_name="phase2t_jepa_factorized_future_head",
        **selection_kwargs,
    )
    return {
        "metrics_unweighted_batch_mean": _mean_records(metric_records),
        "batch_contracts": contracts,
        "candidate_step_prediction_control_records": candidate_step_records,
        "primary_source_state_prediction_control_records": source_state_records,
        "checkpoint_rule_record": checkpoint_rule,
        "stability": stability,
        "checkpoint_selection_permitted": bool(
            cell.participates_in_checkpoint_selection
            and checkpoint_rule["gate_pass"]
        ),
        "primitive_affordance_selection_records": selection_records,
        "primitive_affordance_selection_summary": (
            factorized_sequence_primitive_selection_summary(selection_records)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell", choices=("C2",), default="C2")
    parser.add_argument("--run-class", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--optimization-steps", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--factor-loss-lambda", type=float, default=1.0)
    parser.add_argument("--safety-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--head-hidden-dim", type=int, default=96)
    parser.add_argument("--detach-action-control-state", action="store_true")
    parser.add_argument("--safe-threshold", type=float, default=0.5)
    parser.add_argument("--unsafe-threshold", type=float, default=0.5)
    parser.add_argument("--task-gain-weight", type=float, default=0.75)
    parser.add_argument("--p05-clearance-weight", type=float, default=1.25)
    parser.add_argument("--minimum-clearance-weight", type=float, default=0.75)
    parser.add_argument("--unsafe-penalty-weight", type=float, default=2.0)
    parser.add_argument("--heading-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.optimization_steps < 1:
        parser.error("--optimization-steps must be positive")
    if args.evaluation_interval < 1:
        parser.error("--evaluation-interval must be positive")
    if args.source_states_per_batch < 1:
        parser.error("--source-states-per-batch must be positive")
    if args.max_train_rows < 0:
        parser.error("--max-train-rows must be non-negative")
    if args.max_validation_rows < 0:
        parser.error("--max-validation-rows must be non-negative")
    if args.max_grad_norm < 0.0:
        parser.error("--max-grad-norm must be non-negative")
    if args.factor_loss_lambda < 0.0:
        parser.error("--factor-loss-lambda must be non-negative")
    if args.safety_loss_weight < 0.0:
        parser.error("--safety-loss-weight must be non-negative")
    if args.value_loss_weight < 0.0:
        parser.error("--value-loss-weight must be non-negative")
    if args.head_hidden_dim < 1:
        parser.error("--head-hidden-dim must be positive")
    if not 0.0 <= args.safe_threshold <= 1.0:
        parser.error("--safe-threshold must lie in [0, 1]")
    if not 0.0 <= args.unsafe_threshold <= 1.0:
        parser.error("--unsafe-threshold must lie in [0, 1]")
    if args.log_every < 0:
        parser.error("--log-every must be non-negative")

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
    if any(value["scene_ids"] or value["source_keys"] for value in overlap.values()):
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
    primitive_names = primitive_vocabulary(train_rows)
    baseline_train_examples = build_primitive_affordance_examples(
        train_rows,
        primitive_names=primitive_names,
    )
    baseline_validation_examples = build_primitive_affordance_examples(
        validation_rows,
        primitive_names=primitive_names,
    )
    primitive_priors = fit_primitive_action_priors(baseline_train_examples)
    primitive_action_only_baseline = evaluate_primitive_action_only_baseline(
        baseline_validation_examples,
        primitive_priors,
        split_name="validation",
        seed=args.seed,
    )
    cell = registered_cell(args.cell)
    model = build_model(
        cell,
        command_dim=command_dim,
        detach_action_control_state=args.detach_action_control_state,
    ).to(device)
    head = FutureFactorizedAffordanceHead(
        latent_dim=MODEL_CONSTANTS["latent_dim"],
        hidden_dim=args.head_hidden_dim,
        factor_count=len(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr},
            {"params": head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    selection_kwargs = {
        "safe_threshold": args.safe_threshold,
        "unsafe_threshold": args.unsafe_threshold,
        "task_gain_weight": args.task_gain_weight,
        "p05_clearance_weight": args.p05_clearance_weight,
        "minimum_clearance_weight": args.minimum_clearance_weight,
        "unsafe_penalty_weight": args.unsafe_penalty_weight,
        "heading_weight": args.heading_weight,
    }

    history = []
    completed_steps = 0
    epoch = 0
    while completed_steps < args.optimization_steps:
        model.train()
        head.train()
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
            output = forward_phase2t(
                model=model,
                head=head,
                cell=cell,
                batch=batch,
                factor_loss_lambda=args.factor_loss_lambda,
                safety_weight=args.safety_loss_weight,
                value_weight=args.value_loss_weight,
                include_controls=False,
            )
            train_metrics = _metric_record(output)
            _assert_finite_metrics(
                train_metrics,
                step=completed_steps + 1,
                phase="train",
            )
            optimizer.zero_grad(set_to_none=True)
            output["phase2t_total_loss"].backward()
            gradient_norm = None
            if args.max_grad_norm > 0.0:
                gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    [*model.parameters(), *head.parameters()],
                    max_norm=float(args.max_grad_norm),
                    error_if_nonfinite=False,
                )
                if not torch.isfinite(gradient_norm_tensor):
                    raise RuntimeError(
                        json.dumps(
                            {
                                "error": "nonfinite_phase2t_gradient_norm",
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
                    model=model,
                    head=head,
                    cell=cell,
                    rows=validation_rows,
                    primitive_examples=baseline_validation_examples,
                    hard_negatives=validation_negatives,
                    source_states_per_batch=args.source_states_per_batch,
                    seed=args.seed,
                    epoch=epoch + 1,
                    device=device,
                    factor_loss_lambda=args.factor_loss_lambda,
                    safety_weight=args.safety_loss_weight,
                    value_weight=args.value_loss_weight,
                    selection_kwargs=selection_kwargs,
                )
            history.append(record)
            if args.log_every > 0 and (
                completed_steps % args.log_every == 0
                or "validation_interface_diagnostic" in record
            ):
                compact = {
                    "optimization_step": record["optimization_step"],
                    "epoch": record["epoch"],
                    "train_metrics": train_metrics,
                }
                if gradient_norm is not None:
                    compact["gradient_norm"] = gradient_norm
                if "validation_interface_diagnostic" in record:
                    validation = record["validation_interface_diagnostic"]
                    compact["validation_gate"] = validation["checkpoint_rule_record"]
                    compact["validation_primitive_summary"] = validation[
                        "primitive_affordance_selection_summary"
                    ]
                    compact["validation_metrics"] = validation[
                        "metrics_unweighted_batch_mean"
                    ]
                print(json.dumps(compact), flush=True)
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
        "schema": "jepa_phase2t_factorized_affordance_training_v0",
        "run_class": args.run_class,
        "confirmatory_result": False,
        "cell": vars(cell),
        "fixed_model_constants": MODEL_CONSTANTS,
        "factor_head": {
            "name": "FutureFactorizedAffordanceHead",
            "target_version": PHASE2T_SEQUENCE_FACTOR_TARGET_VERSION,
            "factor_names": list(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
            "factor_loss_lambda": args.factor_loss_lambda,
            "safety_loss_weight": args.safety_loss_weight,
            "value_loss_weight": args.value_loss_weight,
            "head_hidden_dim": args.head_hidden_dim,
        },
        "selection_rule": {
            "schema": "jepa_phase2p_safety_first_selection_rule_v0",
            **selection_kwargs,
        },
        "model_amendments": {
            "detach_action_control_state": args.detach_action_control_state,
            "action_utility_target_version": ACTION_UTILITY_TARGET_VERSION,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.lr,
            "head_learning_rate": args.head_lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "optimization_steps": args.optimization_steps,
            "evaluation_interval": args.evaluation_interval,
        },
        "seed": args.seed,
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model)
        + trainable_parameter_count(head),
        "primitive_action_priors": primitive_priors,
        "primitive_action_only_baseline": primitive_action_only_baseline,
        "train_data": {
            "load_audit": train_load_audit,
            "dataset_audit": phase2_dataset_audit(train_rows),
            "sequence_factor_target_audit": phase2t_sequence_factor_target_audit(
                train_rows
            ),
        },
        "validation_data": {
            "load_audit": validation_load_audit,
            "dataset_audit": phase2_dataset_audit(validation_rows),
            "sequence_factor_target_audit": phase2t_sequence_factor_target_audit(
                validation_rows
            ),
        },
        "split_overlap": overlap,
        "history": history,
        "final_validation": final_validation,
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
            "bounded JEPA integration smoke, not confirmatory training",
            "train and validation evidence only",
            "test_id and test_hard are not used for reported metrics or model selection",
            "factorized labels are generator-derived supervision",
            "future factor head is auxiliary and does not prove deployable navigation",
        ],
    }
    payload = {
        "report": report,
        "model_state_dict": {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        },
        "factor_head_state_dict": {
            name: value.detach().cpu() for name, value in head.state_dict().items()
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
