#!/usr/bin/env python3
"""Train the bounded Phase 2Z local-occupancy affordance diagnostic."""
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
    load_spatial_future_rows,
    pairwise_split_overlap,
    phase2_dataset_audit,
)
from lewm.benchmarks.phase2m_primitive_affordance import (  # noqa: E402
    build_primitive_affordance_examples,
    evaluate_primitive_action_only_baseline,
    fit_primitive_action_priors,
    primitive_affordance_batches,
    primitive_affordance_selection_summary,
    primitive_vocabulary,
)
from lewm.benchmarks.phase2o_factorized_affordance import (  # noqa: E402
    FACTORIZED_AFFORDANCE_FACTOR_NAMES,
    FACTORIZED_AFFORDANCE_TARGET_VERSION,
    factorized_affordance_selection_records,
)
from lewm.benchmarks.phase2z_occupancy_affordance import (  # noqa: E402
    PHASE2Z_OCCUPANCY_FEATURE_SCHEMA,
    build_phase2z_occupancy_affordance_examples,
    materialize_phase2z_occupancy_batch,
    phase2z_grid_channel_names,
    phase2z_occupancy_dataset_audit,
    phase2z_vector_feature_names,
)
from lewm.models.primitive_affordance import (  # noqa: E402
    OccupancyPrimitiveAffordanceModel,
    factorized_affordance_losses,
    factorized_affordance_values,
    primitive_affordance_losses,
)
from lewm.models.spatial_predictor import trainable_parameter_count  # noqa: E402


METRIC_KEYS = (
    "loss",
    "factorized_affordance_loss",
    "factorized_safety_bce_loss",
    "factorized_value_mse_loss",
    "primitive_affordance_loss",
    "primitive_affordance_ce_loss",
    "primitive_affordance_regression_loss",
)


def _metric_record(output: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(output[key].detach().cpu()) for key in METRIC_KEYS}


def _assert_finite_metrics(metrics: dict[str, float], *, step: int, phase: str) -> None:
    nonfinite = {
        key: value for key, value in metrics.items() if not np.isfinite(value)
    }
    if nonfinite:
        raise RuntimeError(
            json.dumps(
                {
                    "error": "nonfinite_phase2z_metrics",
                    "metrics": nonfinite,
                    "phase": phase,
                    "step": int(step),
                },
                sort_keys=True,
            )
        )


def _default_selection_kwargs() -> dict[str, float]:
    return {
        "safe_threshold": 0.5,
        "unsafe_threshold": 0.5,
        "task_gain_weight": 0.75,
        "p05_clearance_weight": 1.25,
        "minimum_clearance_weight": 0.75,
        "unsafe_penalty_weight": 2.0,
        "heading_weight": 0.05,
    }


def _safe_factor_value(values: torch.Tensor, index: int, default: float) -> torch.Tensor:
    if values.shape[-1] <= index:
        return values.new_full(values.shape[:-1], float(default))
    return values[..., index]


def _primitive_scores_from_factor_values(
    factor_values: torch.Tensor,
    *,
    safe_threshold: float,
    unsafe_threshold: float,
    task_gain_weight: float,
    p05_clearance_weight: float,
    minimum_clearance_weight: float,
    unsafe_penalty_weight: float,
    heading_weight: float,
) -> torch.Tensor:
    """Return differentiable primitive scores aligned with the Phase 2P selector."""

    if factor_values.ndim != 3:
        raise ValueError("factor_values must have shape (B, primitive_count, factors)")
    del safe_threshold, unsafe_threshold
    safe = _safe_factor_value(factor_values, 0, 0.0)
    task_gain = _safe_factor_value(factor_values, 1, 0.0)
    p05_clearance = _safe_factor_value(factor_values, 2, 0.0)
    minimum_clearance = _safe_factor_value(factor_values, 3, 0.0)
    unsafe_fraction = _safe_factor_value(factor_values, 4, 1.0)
    heading = _safe_factor_value(factor_values, 5, 0.0)
    base_score = (
        task_gain_weight * task_gain
        + p05_clearance_weight * p05_clearance
        + minimum_clearance_weight * minimum_clearance
        - unsafe_penalty_weight * unsafe_fraction
        + heading_weight * heading
    )
    return base_score + safe - unsafe_fraction


def _forward_batch(
    model: OccupancyPrimitiveAffordanceModel,
    batch,
    *,
    safety_weight: float,
    value_weight: float,
    primitive_ranking_loss_weight: float = 0.0,
    primitive_ranking_regression_weight: float = 1.0,
    selection_kwargs: dict | None = None,
) -> dict[str, torch.Tensor]:
    factor_logits = model(batch.occupancy_action_grids, batch.vector_features)
    losses = factorized_affordance_losses(
        factor_logits=factor_logits,
        factor_targets=batch.factor_targets,
        factor_mask=batch.factor_mask,
        safety_weight=safety_weight,
        value_weight=value_weight,
    )
    factor_values = factorized_affordance_values(factor_logits)
    primitive_scores = _primitive_scores_from_factor_values(
        factor_values,
        **(selection_kwargs or _default_selection_kwargs()),
    )
    primitive_losses = primitive_affordance_losses(
        primitive_scores=primitive_scores,
        primitive_utility_targets=batch.primitive_utility_targets,
        primitive_utility_mask=batch.primitive_utility_mask,
        regression_weight=primitive_ranking_regression_weight,
    )
    total_loss = (
        losses["factorized_affordance_loss"]
        + float(primitive_ranking_loss_weight)
        * primitive_losses["primitive_affordance_loss"]
    )
    return {
        "loss": total_loss,
        "factor_logits": factor_logits,
        "factor_values": factor_values,
        "primitive_scores": primitive_scores,
        "primitive_affordance_loss": primitive_losses["primitive_affordance_loss"],
        "primitive_affordance_ce_loss": primitive_losses[
            "primitive_affordance_ce_loss"
        ],
        "primitive_affordance_regression_loss": primitive_losses[
            "primitive_affordance_regression_loss"
        ],
        **losses,
    }


def _mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in METRIC_KEYS
    }


@torch.no_grad()
def evaluate(
    model: OccupancyPrimitiveAffordanceModel,
    examples,
    *,
    source_states_per_batch: int,
    seed: int,
    device: torch.device,
    safety_weight: float,
    value_weight: float,
    selection_kwargs: dict,
    primitive_ranking_loss_weight: float = 0.0,
    primitive_ranking_regression_weight: float = 1.0,
) -> dict:
    model.eval()
    records = []
    selection_records = []
    batches = primitive_affordance_batches(
        len(examples),
        source_states_per_batch=source_states_per_batch,
        seed=0,
        shuffle=False,
    )
    for indices in batches:
        batch = materialize_phase2z_occupancy_batch(examples, indices).to(device)
        output = _forward_batch(
            model,
            batch,
            safety_weight=safety_weight,
            value_weight=value_weight,
            primitive_ranking_loss_weight=primitive_ranking_loss_weight,
            primitive_ranking_regression_weight=primitive_ranking_regression_weight,
            selection_kwargs=selection_kwargs,
        )
        metric_record = _metric_record(output)
        _assert_finite_metrics(
            metric_record,
            step=len(records) + 1,
            phase="validation_batch",
        )
        records.append(metric_record)
        selection_records.extend(
            factorized_affordance_selection_records(
                batch.base_examples,
                output["factor_values"],
                seed=seed,
                split_name="validation",
                scorer_name="occupancy_affordance_model",
                **selection_kwargs,
            )
        )
    return {
        "metrics_unweighted_batch_mean": _mean_records(records),
        "primitive_affordance_selection_records": selection_records,
        "primitive_affordance_selection_summary": (
            primitive_affordance_selection_summary(selection_records)
        ),
    }


def _args_for_json(args: argparse.Namespace) -> dict:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def _compact_log_record(record: dict) -> dict:
    result = {
        "optimization_step": record["optimization_step"],
        "epoch": record["epoch"],
        "train_metrics": record["train_metrics"],
    }
    if "gradient_norm" in record:
        result["gradient_norm"] = record["gradient_norm"]
    validation = record.get("validation_diagnostic")
    if validation is not None:
        result["validation_summary"] = validation.get(
            "primitive_affordance_selection_summary"
        )
        result["validation_metrics"] = validation.get("metrics_unweighted_batch_mean")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-class", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--optimization-steps", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=32)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--command-dt-s", type=float, default=0.1)
    parser.add_argument("--grid-size", type=int, default=32)
    parser.add_argument("--half-extent-m", type=float, default=4.0)
    parser.add_argument("--occupied-clearance-m", type=float, default=0.15)
    parser.add_argument("--path-radius-m", type=float, default=0.12)
    parser.add_argument("--goal-heat-sigma-m", type=float, default=0.35)
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--safety-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--primitive-ranking-loss-weight", type=float, default=0.10)
    parser.add_argument("--primitive-ranking-regression-weight", type=float, default=1.0)
    parser.add_argument("--safe-threshold", type=float, default=0.5)
    parser.add_argument("--unsafe-threshold", type=float, default=0.5)
    parser.add_argument("--task-gain-weight", type=float, default=0.75)
    parser.add_argument("--p05-clearance-weight", type=float, default=1.25)
    parser.add_argument("--minimum-clearance-weight", type=float, default=0.75)
    parser.add_argument("--unsafe-penalty-weight", type=float, default=2.0)
    parser.add_argument("--heading-weight", type=float, default=0.05)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print every N optimization steps; 0 disables step logging.",
    )
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
    if args.command_dt_s <= 0.0:
        parser.error("--command-dt-s must be positive")
    if args.grid_size < 4:
        parser.error("--grid-size must be at least 4")
    if args.half_extent_m <= 0.0:
        parser.error("--half-extent-m must be positive")
    if args.occupied_clearance_m < 0.0:
        parser.error("--occupied-clearance-m must be non-negative")
    if args.path_radius_m <= 0.0:
        parser.error("--path-radius-m must be positive")
    if args.goal_heat_sigma_m <= 0.0:
        parser.error("--goal-heat-sigma-m must be positive")
    if args.conv_channels < 1:
        parser.error("--conv-channels must be positive")
    if args.hidden_dim < 1:
        parser.error("--hidden-dim must be positive")
    if args.mlp_depth < 1:
        parser.error("--mlp-depth must be positive")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must lie in [0, 1)")
    if args.max_grad_norm < 0.0:
        parser.error("--max-grad-norm must be non-negative")
    if args.safety_loss_weight < 0.0:
        parser.error("--safety-loss-weight must be non-negative")
    if args.value_loss_weight < 0.0:
        parser.error("--value-loss-weight must be non-negative")
    if args.primitive_ranking_loss_weight < 0.0:
        parser.error("--primitive-ranking-loss-weight must be non-negative")
    if args.primitive_ranking_regression_weight < 0.0:
        parser.error("--primitive-ranking-regression-weight must be non-negative")
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

    primitive_names = primitive_vocabulary(train_rows)
    grid_channel_names = phase2z_grid_channel_names()
    vector_feature_names = phase2z_vector_feature_names(primitive_names)
    train_examples = build_phase2z_occupancy_affordance_examples(
        train_rows,
        primitive_names=primitive_names,
        command_dt_s=args.command_dt_s,
        grid_size=args.grid_size,
        half_extent_m=args.half_extent_m,
        occupied_clearance_m=args.occupied_clearance_m,
        path_radius_m=args.path_radius_m,
        goal_heat_sigma_m=args.goal_heat_sigma_m,
    )
    validation_examples = build_phase2z_occupancy_affordance_examples(
        validation_rows,
        primitive_names=primitive_names,
        command_dt_s=args.command_dt_s,
        grid_size=args.grid_size,
        half_extent_m=args.half_extent_m,
        occupied_clearance_m=args.occupied_clearance_m,
        path_radius_m=args.path_radius_m,
        goal_heat_sigma_m=args.goal_heat_sigma_m,
    )
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
    model = OccupancyPrimitiveAffordanceModel(
        grid_channels=len(grid_channel_names),
        vector_dim=len(vector_feature_names),
        factor_count=len(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
        conv_channels=args.conv_channels,
        hidden_dim=args.hidden_dim,
        depth=args.mlp_depth,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
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
        batches = primitive_affordance_batches(
            len(train_examples),
            source_states_per_batch=args.source_states_per_batch,
            seed=args.seed + epoch,
            shuffle=True,
        )
        for indices in batches:
            batch = materialize_phase2z_occupancy_batch(train_examples, indices).to(
                device
            )
            output = _forward_batch(
                model,
                batch,
                safety_weight=args.safety_loss_weight,
                value_weight=args.value_loss_weight,
                primitive_ranking_loss_weight=args.primitive_ranking_loss_weight,
                primitive_ranking_regression_weight=(
                    args.primitive_ranking_regression_weight
                ),
                selection_kwargs=selection_kwargs,
            )
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
                                "error": "nonfinite_phase2z_gradient_norm",
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
            completed_steps += 1
            record = {
                "optimization_step": completed_steps,
                "epoch": epoch + 1,
                "train_metrics": train_metrics,
            }
            if gradient_norm is not None:
                record["gradient_norm"] = gradient_norm
            if (
                completed_steps % args.evaluation_interval == 0
                or completed_steps == args.optimization_steps
            ):
                record["validation_diagnostic"] = evaluate(
                    model,
                    validation_examples,
                    source_states_per_batch=args.source_states_per_batch,
                    seed=args.seed,
                    device=device,
                    safety_weight=args.safety_loss_weight,
                    value_weight=args.value_loss_weight,
                    selection_kwargs=selection_kwargs,
                    primitive_ranking_loss_weight=args.primitive_ranking_loss_weight,
                    primitive_ranking_regression_weight=(
                        args.primitive_ranking_regression_weight
                    ),
                )
            history.append(record)
            if args.log_every > 0 and (
                completed_steps % args.log_every == 0
                or "validation_diagnostic" in record
            ):
                print(json.dumps(_compact_log_record(record)), flush=True)
            if completed_steps >= args.optimization_steps:
                break
        epoch += 1

    final_validation = next(
        (
            record["validation_diagnostic"]
            for record in reversed(history)
            if "validation_diagnostic" in record
        ),
        None,
    )
    report = {
        "schema": "jepa_phase2z_occupancy_affordance_training_v0",
        "run_class": args.run_class,
        "confirmatory_result": False,
        "feature_schema": PHASE2Z_OCCUPANCY_FEATURE_SCHEMA,
        "target_version": FACTORIZED_AFFORDANCE_TARGET_VERSION,
        "model": {
            "name": "OccupancyPrimitiveAffordanceModel",
            "grid_channel_names": list(grid_channel_names),
            "vector_feature_names": list(vector_feature_names),
            "primitive_names": list(primitive_names),
            "factor_names": list(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
            "conv_channels": args.conv_channels,
            "hidden_dim": args.hidden_dim,
            "mlp_depth": args.mlp_depth,
            "dropout": args.dropout,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "optimization_steps": args.optimization_steps,
            "evaluation_interval": args.evaluation_interval,
            "safety_loss_weight": args.safety_loss_weight,
            "value_loss_weight": args.value_loss_weight,
            "primitive_ranking_loss_weight": args.primitive_ranking_loss_weight,
            "primitive_ranking_regression_weight": (
                args.primitive_ranking_regression_weight
            ),
        },
        "feature_config": {
            "command_dt_s": args.command_dt_s,
            "grid_size": args.grid_size,
            "half_extent_m": args.half_extent_m,
            "occupied_clearance_m": args.occupied_clearance_m,
            "path_radius_m": args.path_radius_m,
            "goal_heat_sigma_m": args.goal_heat_sigma_m,
        },
        "selection_rule": {
            "schema": "jepa_phase2p_safety_first_selection_rule_v0",
            **selection_kwargs,
        },
        "primitive_action_priors": primitive_priors,
        "primitive_action_only_baseline": primitive_action_only_baseline,
        "seed": args.seed,
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model),
        "train_data": {
            "load_audit": train_load_audit,
            "dataset_audit": phase2_dataset_audit(train_rows),
            "occupancy_affordance_audit": phase2z_occupancy_dataset_audit(
                train_examples,
                split_name="train",
                grid_channel_names=grid_channel_names,
                vector_feature_names=vector_feature_names,
            ),
        },
        "validation_data": {
            "load_audit": validation_load_audit,
            "dataset_audit": phase2_dataset_audit(validation_rows),
            "occupancy_affordance_audit": phase2z_occupancy_dataset_audit(
                validation_examples,
                split_name="validation",
                grid_channel_names=grid_channel_names,
                vector_feature_names=vector_feature_names,
            ),
        },
        "split_overlap": overlap,
        "history": history,
        "final_validation": final_validation,
        "limitations": [
            "privileged manifest-derived occupancy bridge, not a deployable RGB/depth policy",
            "train and validation evidence only",
            "test_id and test_hard are not used for reported metrics or model selection",
            "factor targets are generator-derived supervision",
            "this is an affordance/utility diagnostic, not a JEPA world model",
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
