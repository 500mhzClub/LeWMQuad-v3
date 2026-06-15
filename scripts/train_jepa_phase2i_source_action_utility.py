#!/usr/bin/env python3
"""Train the bounded Phase 2I source-conditioned action-utility ranker."""
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
from lewm.benchmarks.phase2d_training import (  # noqa: E402
    ACTION_UTILITY_TARGET_VERSION,
    action_utility_selection_records,
    action_utility_selection_summary,
)
from lewm.benchmarks.phase2i_utility_training import (  # noqa: E402
    materialize_phase2i_utility_batch,
    phase2i_batch_contract_audit,
)
from lewm.models.phase2d_spatial_lewm import action_utility_losses  # noqa: E402
from lewm.models.source_action_utility import SourceActionUtilityRanker  # noqa: E402
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
    "action_utility_loss",
    "action_utility_ce_loss",
    "action_utility_regression_loss",
)


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
                    "error": "nonfinite_phase2i_metrics",
                    "metrics": nonfinite,
                    "phase": phase,
                    "step": int(step),
                },
                sort_keys=True,
            )
        )


def _forward_batch(
    model: SourceActionUtilityRanker,
    batch,
    *,
    regression_weight: float,
    ranking_loss: str,
    softmax_temperature: float,
) -> dict[str, torch.Tensor]:
    prediction = model(batch.start_vision, batch.actions)
    losses = action_utility_losses(
        utility_prediction=prediction,
        utility_targets=batch.action_utility_targets,
        utility_mask=batch.action_utility_mask,
        utility_group_ids=batch.action_utility_group_ids,
        regression_weight=regression_weight,
        ranking_loss=ranking_loss,
        softmax_temperature=softmax_temperature,
    )
    return {
        "loss": losses["action_utility_loss"],
        "action_utility_prediction": prediction,
        **losses,
    }


def _mean_records(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([record[key] for record in records]))
        for key in METRIC_KEYS
    }


@torch.no_grad()
def evaluate(
    model: SourceActionUtilityRanker,
    rows: list[dict],
    *,
    source_states_per_batch: int,
    seed: int,
    device: torch.device,
    regression_weight: float,
    ranking_loss: str,
    softmax_temperature: float,
) -> dict:
    model.eval()
    records = []
    contracts = []
    selection_records = []
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
            regression_weight=regression_weight,
            ranking_loss=ranking_loss,
            softmax_temperature=softmax_temperature,
        )
        metric_record = _metric_record(output)
        _assert_finite_metrics(
            metric_record,
            step=len(records) + 1,
            phase="validation_batch",
        )
        records.append(metric_record)
        contracts.append(phase2i_batch_contract_audit(batch))
        selection_records.extend(
            action_utility_selection_records(batch, output, seed=seed)
        )
    return {
        "metrics_unweighted_batch_mean": _mean_records(records),
        "batch_contracts": contracts,
        "action_utility_selection_records": selection_records,
        "action_utility_selection_summary": action_utility_selection_summary(
            selection_records
        ),
    }


def _baseline_reference(path: Path | None) -> list[dict]:
    if path is None:
        return []
    report = json.loads(path.read_text())
    return list(report.get("validation_action_only_baselines", ()))


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
    parser.add_argument("--baseline-audit", type=Path)
    parser.add_argument("--run-class", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--optimization-steps", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=2)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-validation-rows", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--action-utility-regression-weight", type=float, default=0.1)
    parser.add_argument(
        "--action-utility-ranking-loss",
        choices=("hard_ce", "soft_ce"),
        default="hard_ce",
    )
    parser.add_argument("--action-utility-softmax-temperature", type=float, default=0.25)
    parser.add_argument(
        "--input-mode",
        choices=("source_action", "action_only"),
        default="source_action",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("concat", "film_interaction", "interaction_only"),
        default="concat",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print every N optimization steps; 0 disables step logging.",
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
    if args.action_utility_regression_weight < 0.0:
        parser.error("--action-utility-regression-weight must be non-negative")
    if args.action_utility_softmax_temperature <= 0.0:
        parser.error("--action-utility-softmax-temperature must be positive")
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
    model = SourceActionUtilityRanker(
        cmd_dim=command_dim,
        horizon=horizon,
        input_mode=args.input_mode,
        fusion_mode=args.fusion_mode,
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
                regression_weight=args.action_utility_regression_weight,
                ranking_loss=args.action_utility_ranking_loss,
                softmax_temperature=args.action_utility_softmax_temperature,
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
                                "error": "nonfinite_phase2i_gradient_norm",
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
                    source_states_per_batch=args.source_states_per_batch,
                    seed=args.seed,
                    device=device,
                    regression_weight=args.action_utility_regression_weight,
                    ranking_loss=args.action_utility_ranking_loss,
                    softmax_temperature=args.action_utility_softmax_temperature,
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
        "schema": "jepa_phase2i_source_action_utility_training_v0",
        "run_class": args.run_class,
        "confirmatory_result": False,
        "target_version": ACTION_UTILITY_TARGET_VERSION,
        "model": {
            "name": "SourceActionUtilityRanker",
            "constants": MODEL_CONSTANTS,
            "input_mode": args.input_mode,
            "fusion_mode": args.fusion_mode,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "optimization_steps": args.optimization_steps,
            "evaluation_interval": args.evaluation_interval,
            "action_utility_regression_weight": (
                args.action_utility_regression_weight
            ),
            "action_utility_ranking_loss": args.action_utility_ranking_loss,
            "action_utility_softmax_temperature": (
                args.action_utility_softmax_temperature
            ),
        },
        "baseline_reference": _baseline_reference(args.baseline_audit),
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
        "final_validation": final_validation,
        "limitations": [
            "this is a utility/affordance diagnostic, not a JEPA world model",
            "train and validation evidence only",
            "test_id and test_hard remain unopened",
            "utility labels are generator-derived supervision",
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
