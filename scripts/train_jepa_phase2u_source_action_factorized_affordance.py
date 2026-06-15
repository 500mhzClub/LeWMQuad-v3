#!/usr/bin/env python3
"""Train Phase 2U source/action factorized affordance bridge diagnostic."""
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
    source_grouped_batches,
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
from lewm.benchmarks.phase2i_utility_training import (  # noqa: E402
    materialize_phase2i_utility_batch,
    phase2i_batch_contract_audit,
)
from lewm.benchmarks.phase2t_factorized_jepa_affordance import (  # noqa: E402
    PHASE2T_SEQUENCE_FACTOR_TARGET_VERSION,
    factorized_sequence_primitive_selection_records,
    factorized_sequence_primitive_selection_summary,
    materialize_phase2t_sequence_factor_targets,
    phase2t_sequence_factor_target_audit,
)
from lewm.models.primitive_affordance import (  # noqa: E402
    factorized_affordance_losses,
    factorized_affordance_values,
)
from lewm.models.source_action_utility import (  # noqa: E402
    SourceActionFactorizedAffordanceModel,
)
from lewm.models.spatial_predictor import trainable_parameter_count  # noqa: E402


MODEL_CONSTANTS = {
    "latent_dim": 48,
    "action_hidden_dim": 96,
    "image_size": 224,
    "patch_size": 14,
    "encoder_depth": 2,
    "encoder_heads": 3,
    "encoder_mlp_ratio": 2,
}

METRIC_KEYS = (
    "loss",
    "factorized_affordance_loss",
    "factorized_safety_bce_loss",
    "factorized_value_mse_loss",
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
                    "error": "nonfinite_phase2u_metrics",
                    "metrics": nonfinite,
                    "phase": phase,
                    "step": int(step),
                },
                sort_keys=True,
            )
        )


def _forward_batch(
    model: SourceActionFactorizedAffordanceModel,
    batch,
    *,
    safety_weight: float,
    value_weight: float,
) -> dict[str, torch.Tensor]:
    factor_targets, factor_mask = materialize_phase2t_sequence_factor_targets(
        batch.rows
    )
    factor_targets = factor_targets.to(batch.start_vision.device)
    factor_mask = factor_mask.to(batch.start_vision.device)
    factor_logits = model(batch.start_vision, batch.actions)
    losses = factorized_affordance_losses(
        factor_logits=factor_logits[:, None, :],
        factor_targets=factor_targets[:, None, :],
        factor_mask=factor_mask[:, None, :],
        safety_weight=safety_weight,
        value_weight=value_weight,
    )
    return {
        "loss": losses["factorized_affordance_loss"],
        "sequence_factor_logits": factor_logits,
        "sequence_factor_values": factorized_affordance_values(
            factor_logits[:, None, :]
        )[:, 0],
        "sequence_factor_targets": factor_targets,
        "sequence_factor_mask": factor_mask,
        **losses,
    }


def _mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in METRIC_KEYS
    }


@torch.no_grad()
def evaluate(
    model: SourceActionFactorizedAffordanceModel,
    rows: list[dict],
    primitive_examples,
    *,
    source_states_per_batch: int,
    seed: int,
    device: torch.device,
    safety_weight: float,
    value_weight: float,
    selection_kwargs: dict,
) -> dict:
    model.eval()
    records = []
    contracts = []
    selected_rows = []
    factor_values = []
    batches = source_grouped_batches(
        rows,
        source_states_per_batch=source_states_per_batch,
        seed=0,
        shuffle=False,
    )
    for indices in batches:
        batch = materialize_phase2i_utility_batch(rows, indices).to(device)
        output = _forward_batch(
            model,
            batch,
            safety_weight=safety_weight,
            value_weight=value_weight,
        )
        metric_record = _metric_record(output)
        _assert_finite_metrics(
            metric_record,
            step=len(records) + 1,
            phase="validation_batch",
        )
        records.append(metric_record)
        contracts.append(phase2i_batch_contract_audit(batch))
        selected_rows.extend(batch.rows)
        factor_values.append(output["sequence_factor_values"].detach().cpu())
    selection_records = factorized_sequence_primitive_selection_records(
        selected_rows,
        torch.cat(factor_values),
        primitive_examples,
        seed=seed,
        split_name="validation",
        scorer_name="phase2u_source_action_factorized_affordance",
        **selection_kwargs,
    )
    return {
        "metrics_unweighted_batch_mean": _mean_records(records),
        "batch_contracts": contracts,
        "primitive_affordance_selection_records": selection_records,
        "primitive_affordance_selection_summary": (
            factorized_sequence_primitive_selection_summary(selection_records)
        ),
    }


def _args_for_json(args: argparse.Namespace) -> dict:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-class", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--optimization-steps", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--safety-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--input-mode",
        choices=("source_action", "action_only"),
        default="source_action",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("concat", "film_interaction", "interaction_only"),
        default="film_interaction",
    )
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
    if args.max_grad_norm < 0.0:
        parser.error("--max-grad-norm must be non-negative")
    if args.safety_loss_weight < 0.0:
        parser.error("--safety-loss-weight must be non-negative")
    if args.value_loss_weight < 0.0:
        parser.error("--value-loss-weight must be non-negative")
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
    model = SourceActionFactorizedAffordanceModel(
        cmd_dim=command_dim,
        horizon=horizon,
        factor_count=len(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
        input_mode=args.input_mode,
        fusion_mode=args.fusion_mode,
        **MODEL_CONSTANTS,
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
        batches = source_grouped_batches(
            train_rows,
            source_states_per_batch=args.source_states_per_batch,
            seed=args.seed + epoch,
            shuffle=True,
        )
        for indices in batches:
            batch = materialize_phase2i_utility_batch(train_rows, indices).to(device)
            output = _forward_batch(
                model,
                batch,
                safety_weight=args.safety_loss_weight,
                value_weight=args.value_loss_weight,
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
                                "error": "nonfinite_phase2u_gradient_norm",
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
                "batch_contract": phase2i_batch_contract_audit(batch),
            }
            if gradient_norm is not None:
                record["gradient_norm"] = gradient_norm
            if (
                completed_steps % args.evaluation_interval == 0
                or completed_steps == args.optimization_steps
            ):
                record["validation_diagnostic"] = evaluate(
                    model,
                    validation_rows,
                    baseline_validation_examples,
                    source_states_per_batch=args.source_states_per_batch,
                    seed=args.seed,
                    device=device,
                    safety_weight=args.safety_loss_weight,
                    value_weight=args.value_loss_weight,
                    selection_kwargs=selection_kwargs,
                )
            history.append(record)
            if args.log_every > 0 and (
                completed_steps % args.log_every == 0
                or "validation_diagnostic" in record
            ):
                compact = {
                    "optimization_step": record["optimization_step"],
                    "epoch": record["epoch"],
                    "train_metrics": train_metrics,
                }
                if gradient_norm is not None:
                    compact["gradient_norm"] = gradient_norm
                if "validation_diagnostic" in record:
                    validation = record["validation_diagnostic"]
                    compact["validation_summary"] = validation[
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
            record["validation_diagnostic"]
            for record in reversed(history)
            if "validation_diagnostic" in record
        ),
        None,
    )
    report = {
        "schema": "jepa_phase2u_source_action_factorized_affordance_training_v0",
        "run_class": args.run_class,
        "confirmatory_result": False,
        "target_version": PHASE2T_SEQUENCE_FACTOR_TARGET_VERSION,
        "model": {
            "name": "SourceActionFactorizedAffordanceModel",
            "constants": MODEL_CONSTANTS,
            "input_mode": args.input_mode,
            "fusion_mode": args.fusion_mode,
            "factor_names": list(FACTORIZED_AFFORDANCE_FACTOR_NAMES),
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
        "limitations": [
            "source/action factorized affordance diagnostic, not a JEPA world model",
            "train and validation evidence only",
            "test_id and test_hard are not used for reported metrics or model selection",
            "factorized labels are generator-derived supervision",
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
