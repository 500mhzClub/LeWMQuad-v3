#!/usr/bin/env python3
"""Train the bounded Phase 2M source-local primitive-affordance diagnostic."""
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
    PRIMITIVE_AFFORDANCE_TARGET_VERSION,
    build_primitive_affordance_examples,
    evaluate_primitive_action_only_baseline,
    fit_primitive_action_priors,
    materialize_phase2m_primitive_batch,
    oracle_primitive_class_weights,
    phase2m_batch_contract_audit,
    primitive_affordance_batches,
    primitive_affordance_dataset_audit,
    primitive_affordance_selection_records,
    primitive_affordance_selection_summary,
    primitive_vocabulary,
)
from lewm.models.primitive_affordance import (  # noqa: E402
    PrimitiveAffordanceModel,
    primitive_affordance_losses,
)
from lewm.models.spatial_predictor import trainable_parameter_count  # noqa: E402


MODEL_CONSTANTS = {
    "latent_dim": 48,
    "hidden_dim": 96,
    "image_size": 224,
    "patch_size": 14,
    "encoder_depth": 2,
    "encoder_heads": 3,
    "encoder_mlp_ratio": 2,
}

METRIC_KEYS = (
    "loss",
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
                    "error": "nonfinite_phase2m_metrics",
                    "metrics": nonfinite,
                    "phase": phase,
                    "step": int(step),
                },
                sort_keys=True,
            )
        )


def _forward_batch(
    model: PrimitiveAffordanceModel,
    batch,
    *,
    regression_weight: float,
    ranking_loss: str,
    softmax_temperature: float,
    primitive_class_weights: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    scores = model(batch.start_vision)
    losses = primitive_affordance_losses(
        primitive_scores=scores,
        primitive_utility_targets=batch.primitive_utility_targets,
        primitive_utility_mask=batch.primitive_utility_mask,
        primitive_class_weights=primitive_class_weights,
        regression_weight=regression_weight,
        ranking_loss=ranking_loss,
        softmax_temperature=softmax_temperature,
    )
    return {
        "loss": losses["primitive_affordance_loss"],
        "primitive_scores": scores,
        **losses,
    }


def _mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in METRIC_KEYS
    }


@torch.no_grad()
def evaluate(
    model: PrimitiveAffordanceModel,
    examples,
    *,
    source_states_per_batch: int,
    seed: int,
    device: torch.device,
    regression_weight: float,
    ranking_loss: str,
    softmax_temperature: float,
    primitive_class_weights: torch.Tensor | None,
) -> dict:
    model.eval()
    records = []
    contracts = []
    selection_records = []
    batches = primitive_affordance_batches(
        len(examples),
        source_states_per_batch=source_states_per_batch,
        seed=0,
        shuffle=False,
    )
    for indices in batches:
        batch = materialize_phase2m_primitive_batch(examples, indices).to(device)
        output = _forward_batch(
            model,
            batch,
            regression_weight=regression_weight,
            ranking_loss=ranking_loss,
            softmax_temperature=softmax_temperature,
            primitive_class_weights=primitive_class_weights,
        )
        metric_record = _metric_record(output)
        _assert_finite_metrics(
            metric_record,
            step=len(records) + 1,
            phase="validation_batch",
        )
        records.append(metric_record)
        contracts.append(phase2m_batch_contract_audit(batch))
        selection_records.extend(
            primitive_affordance_selection_records(
                batch.examples,
                output["primitive_scores"],
                seed=seed,
                split_name="validation",
                scorer_name="primitive_affordance_model",
            )
        )
    return {
        "metrics_unweighted_batch_mean": _mean_records(records),
        "batch_contracts": contracts,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-class", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--optimization-steps", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=16)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--primitive-regression-weight", type=float, default=1.0)
    parser.add_argument(
        "--primitive-class-balance",
        choices=("none", "oracle_inverse_frequency"),
        default="none",
    )
    parser.add_argument("--primitive-class-weight-max", type=float, default=5.0)
    parser.add_argument(
        "--primitive-ranking-loss",
        choices=("hard_ce", "soft_ce"),
        default="soft_ce",
    )
    parser.add_argument("--primitive-softmax-temperature", type=float, default=0.25)
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
    if args.primitive_regression_weight < 0.0:
        parser.error("--primitive-regression-weight must be non-negative")
    if args.primitive_class_weight_max <= 0.0:
        parser.error("--primitive-class-weight-max must be positive")
    if (
        args.primitive_class_balance != "none"
        and args.primitive_ranking_loss != "hard_ce"
    ):
        parser.error("--primitive-class-balance currently requires hard_ce")
    if args.primitive_softmax_temperature <= 0.0:
        parser.error("--primitive-softmax-temperature must be positive")
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
    train_examples = build_primitive_affordance_examples(
        train_rows,
        primitive_names=primitive_names,
    )
    validation_examples = build_primitive_affordance_examples(
        validation_rows,
        primitive_names=primitive_names,
    )
    primitive_priors = fit_primitive_action_priors(train_examples)
    primitive_class_weights_record = None
    primitive_class_weights = None
    if args.primitive_class_balance == "oracle_inverse_frequency":
        primitive_class_weights_record = oracle_primitive_class_weights(
            train_examples,
            max_weight=args.primitive_class_weight_max,
        )
        primitive_class_weights = torch.tensor(
            primitive_class_weights_record,
            dtype=torch.float32,
            device=device,
        )
    primitive_action_only_baseline = evaluate_primitive_action_only_baseline(
        validation_examples,
        primitive_priors,
        split_name="validation",
        seed=args.seed,
    )

    model = PrimitiveAffordanceModel(
        primitive_count=len(primitive_names),
        **MODEL_CONSTANTS,
    ).to(device)
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
        batches = primitive_affordance_batches(
            len(train_examples),
            source_states_per_batch=args.source_states_per_batch,
            seed=args.seed + epoch,
            shuffle=True,
        )
        for indices in batches:
            batch = materialize_phase2m_primitive_batch(
                train_examples,
                indices,
            ).to(device)
            output = _forward_batch(
                model,
                batch,
                regression_weight=args.primitive_regression_weight,
                ranking_loss=args.primitive_ranking_loss,
                softmax_temperature=args.primitive_softmax_temperature,
                primitive_class_weights=primitive_class_weights,
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
                                "error": "nonfinite_phase2m_gradient_norm",
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
                "batch_contract": phase2m_batch_contract_audit(batch),
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
                    regression_weight=args.primitive_regression_weight,
                    ranking_loss=args.primitive_ranking_loss,
                    softmax_temperature=args.primitive_softmax_temperature,
                    primitive_class_weights=primitive_class_weights,
                )
            history.append(record)
            if args.log_every > 0 and (
                completed_steps % args.log_every == 0
                or "validation_diagnostic" in record
            ):
                print(json.dumps(record), flush=True)
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
        "schema": "jepa_phase2m_primitive_affordance_training_v0",
        "run_class": args.run_class,
        "confirmatory_result": False,
        "target_version": PRIMITIVE_AFFORDANCE_TARGET_VERSION,
        "model": {
            "name": "PrimitiveAffordanceModel",
            "constants": MODEL_CONSTANTS,
            "primitive_names": list(primitive_names),
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "optimization_steps": args.optimization_steps,
            "evaluation_interval": args.evaluation_interval,
            "primitive_regression_weight": args.primitive_regression_weight,
            "primitive_class_balance": args.primitive_class_balance,
            "primitive_class_weight_max": args.primitive_class_weight_max,
            "primitive_class_weights": (
                None
                if primitive_class_weights_record is None
                else list(primitive_class_weights_record)
            ),
            "primitive_ranking_loss": args.primitive_ranking_loss,
            "primitive_softmax_temperature": args.primitive_softmax_temperature,
        },
        "primitive_action_priors": primitive_priors,
        "primitive_action_only_baseline": primitive_action_only_baseline,
        "seed": args.seed,
        "device": str(device),
        "trainable_parameters": trainable_parameter_count(model),
        "train_data": {
            "load_audit": train_load_audit,
            "dataset_audit": phase2_dataset_audit(train_rows),
            "primitive_affordance_audit": primitive_affordance_dataset_audit(
                train_examples,
                split_name="train",
            ),
        },
        "validation_data": {
            "load_audit": validation_load_audit,
            "dataset_audit": phase2_dataset_audit(validation_rows),
            "primitive_affordance_audit": primitive_affordance_dataset_audit(
                validation_examples,
                split_name="validation",
            ),
        },
        "split_overlap": overlap,
        "history": history,
        "final_validation": final_validation,
        "limitations": [
            "this is a source-local affordance diagnostic, not a JEPA world model",
            "train and validation evidence only",
            "test_id and test_hard remain unopened",
            "utility labels are generator-derived supervision",
            "first-primitive aggregation uses the best observed continuation",
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
