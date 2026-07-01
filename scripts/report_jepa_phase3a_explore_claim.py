#!/usr/bin/env python3
"""Report no-beacon explore-then-claim Phase 3A selection metrics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_explore_claim import (  # noqa: E402
    action_sequence_prior_predictions,
    compare_explore_claim_summaries,
    egocentric_explore_claim_predictions,
    egocentric_marker_memory_predictions,
    summarize_explore_claim_predictions,
)
from lewm.benchmarks.phase3a_positive_control import read_jsonl  # noqa: E402
from lewm.benchmarks.phase3a_training import (  # noqa: E402
    CONSEQUENCE_TARGET_NAMES,
    Phase3AMaterializedDataset,
    source_grouped_batches,
)
from lewm.models.phase3a_jepa import Phase3AJepaModel  # noqa: E402


MODEL_CONFIG_KEYS = {
    "view_size",
    "spatial_memory_size",
    "action_dim",
    "latent_dim",
    "pred_layers",
    "target_ema_momentum",
    "prediction_loss_lambda",
    "action_identifiability_lambda",
    "zero_action_lambda",
    "free_running_action_contrast_lambda",
    "free_running_zero_contrast_lambda",
    "utility_loss_lambda",
    "utility_ranking_loss_lambda",
    "utility_ranking_regression_weight",
    "utility_ranking_loss_type",
    "utility_softmax_temperature",
    "utility_source",
    "candidate_score_loss_lambda",
    "candidate_score_regression_weight",
    "candidate_score_ranking_loss_type",
    "candidate_score_softmax_temperature",
    "detach_candidate_score_state",
    "candidate_score_gradient_mode",
    "candidate_score_source_tokens",
    "candidate_score_action_summary",
    "candidate_claim_loss_lambda",
    "candidate_score_claim_logit_weight",
    "online_marker_memory_score_weight",
    "candidate_marker_memory_loss_lambda",
    "candidate_marker_memory_score_weight",
    "candidate_marker_memory_delta_loss_weight",
    "candidate_marker_memory_claim_loss_weight",
    "candidate_marker_memory_ranking_loss_lambda",
    "candidate_marker_memory_ranking_loss_type",
    "candidate_marker_memory_softmax_temperature",
    "candidate_marker_memory_score_mode",
    "structured_marker_memory_loss_lambda",
    "structured_marker_memory_score_weight",
    "structured_marker_memory_ranking_loss_lambda",
    "structured_marker_memory_softmax_temperature",
    "categorical_marker_memory_loss_lambda",
    "categorical_marker_memory_score_weight",
    "categorical_marker_memory_ranking_loss_lambda",
    "categorical_marker_memory_softmax_temperature",
    "categorical_marker_memory_radius",
    "spatial_marker_memory_loss_lambda",
    "spatial_marker_memory_score_weight",
    "spatial_marker_memory_ranking_loss_lambda",
    "spatial_marker_memory_softmax_temperature",
    "spatial_marker_memory_score_temperature",
    "spatial_frontier_memory_loss_lambda",
    "spatial_frontier_observation_loss_lambda",
    "spatial_frontier_memory_score_loss_lambda",
    "spatial_frontier_memory_score_weight",
    "spatial_frontier_memory_ranking_loss_lambda",
    "spatial_frontier_memory_softmax_temperature",
    "spatial_frontier_memory_occupancy_loss_weight",
    "spatial_frontier_memory_marker_loss_weight",
    "spatial_frontier_memory_marker_cell_loss_weight",
    "spatial_frontier_memory_marker_mass_loss_weight",
    "spatial_frontier_memory_detector_init",
    "spatial_frontier_memory_detector_arch",
    "spatial_frontier_memory_gate_mode",
    "spatial_frontier_marker_source",
    "spatial_frontier_collision_penalty",
    "spatial_frontier_novelty_reward",
    "spatial_frontier_marker_gate_threshold",
    "spatial_frontier_marker_gate_width",
    "spatial_frontier_marker_update_threshold",
    "spatial_frontier_marker_update_width",
    "detach_consequence_head_state",
    "consequence_loss_lambda",
    "rollout_delta_loss_lambda",
    "teacher_forced_delta_loss_lambda",
    "decision_token_count",
    "decision_rollout_mode",
    "decision_recurrent_update",
    "decision_target_geometry",
    "decision_target_scale",
    "decision_prediction_loss_lambda",
    "decision_delta_loss_lambda",
    "decision_teacher_forced_prediction_loss_lambda",
    "decision_teacher_forced_delta_loss_lambda",
    "decision_teacher_forced_action_contrast_lambda",
    "decision_teacher_forced_zero_contrast_lambda",
    "decision_action_contrast_lambda",
    "decision_zero_contrast_lambda",
    "use_memory_context",
    "memory_frame_summary",
    "memory_marker_features",
    "spatial_variance_lambda",
}


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_model(path: Path, *, device: torch.device) -> tuple[Phase3AJepaModel, dict]:
    checkpoint = load_checkpoint(path)
    report = checkpoint["report"]
    config = {
        key: value
        for key, value in report["model_config"].items()
        if key in MODEL_CONFIG_KEYS
    }
    model = Phase3AJepaModel(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, report


@torch.no_grad()
def predict_model_scores(
    model: Phase3AJepaModel,
    rows: list[dict],
    *,
    source_states_per_batch: int,
    device: torch.device,
) -> dict[str, list[float]]:
    score_names = (
        "utility",
        "candidate_score",
        "utility_head",
        "consequence_utility",
        "candidate_score_value",
        "candidate_claim_logit",
        "online_marker_memory_score",
        "online_frontier_marker_score",
        "candidate_marker_memory_score",
        "candidate_marker_memory_claim_logit",
        "candidate_marker_memory_distance_score",
        "structured_marker_memory_score",
        "categorical_marker_memory_score",
        "spatial_marker_memory_score",
        "spatial_frontier_memory_score",
        "max_reached_goal_logit",
        "final_reached_goal_logit",
        "mean_reached_goal_logit",
    )
    predictions = {name: [0.0 for _ in rows] for name in score_names}
    reached_goal_index = CONSEQUENCE_TARGET_NAMES.index("reached_goal")
    row_cache = Phase3AMaterializedDataset(rows)
    for indices in source_grouped_batches(
        rows,
        source_states_per_batch=source_states_per_batch,
        shuffle=False,
    ):
        batch = row_cache.materialize_batch(indices).to(device)
        output = model(
            vision=batch.vision,
            history_vision=batch.history_vision,
            history_actions=batch.history_actions,
            actions=batch.actions,
            utility_targets=batch.utility_targets,
            consequence_targets=batch.consequence_targets,
            candidate_marker_memory_valid_mask=batch.marker_memory_valid_mask,
            candidate_marker_memory_delta_targets=batch.marker_memory_delta_targets,
            candidate_marker_memory_claim_targets=batch.marker_memory_claim_targets,
            candidate_marker_memory_score_targets=batch.marker_memory_score_targets,
            structured_marker_memory_valid_mask=batch.marker_memory_start_valid_mask,
            structured_marker_memory_start_delta_targets=(
                batch.marker_memory_start_delta_targets
            ),
            categorical_marker_memory_valid_mask=(
                batch.marker_memory_start_cell_valid_mask
            ),
            categorical_marker_memory_cell_targets=(
                batch.marker_memory_start_cell_targets
            ),
            utility_group_ids=batch.utility_group_ids,
            utility_mask=batch.utility_mask,
            wrong_actions=batch.wrong_actions,
            wrong_mask=batch.wrong_mask,
            non_hold_mask=batch.non_hold_mask,
            return_latents=True,
        )
        reached_goal_logits = output["consequence_prediction"][..., reached_goal_index]
        for local_index, row_index in enumerate(indices):
            predictions["utility"][row_index] = float(
                output["utility_prediction"][local_index].detach().cpu()
            )
            predictions["candidate_score"][row_index] = float(
                output["candidate_score_prediction"][local_index].detach().cpu()
            )
            predictions["candidate_score_value"][row_index] = float(
                output["candidate_score_value_prediction"][local_index].detach().cpu()
            )
            predictions["candidate_claim_logit"][row_index] = float(
                output["candidate_claim_logit"][local_index].detach().cpu()
            )
            predictions["online_marker_memory_score"][row_index] = float(
                output["online_marker_memory_score_prediction"][local_index]
                .detach()
                .cpu()
            )
            predictions["online_frontier_marker_score"][row_index] = float(
                output["online_frontier_marker_score_prediction"][local_index]
                .detach()
                .cpu()
            )
            predictions["candidate_marker_memory_score"][row_index] = float(
                output["candidate_marker_memory_score_prediction"][local_index]
                .detach()
                .cpu()
            )
            predictions["candidate_marker_memory_claim_logit"][row_index] = float(
                output["candidate_marker_memory_claim_logit"][local_index]
                .detach()
                .cpu()
            )
            predictions["candidate_marker_memory_distance_score"][row_index] = float(
                -output["candidate_marker_memory_delta_prediction"][local_index]
                .abs()
                .sum()
                .detach()
                .cpu()
            )
            predictions["structured_marker_memory_score"][row_index] = float(
                output["structured_marker_memory_score_prediction"][local_index]
                .detach()
                .cpu()
            )
            predictions["categorical_marker_memory_score"][row_index] = float(
                output["categorical_marker_memory_score_prediction"][local_index]
                .detach()
                .cpu()
            )
            predictions["spatial_marker_memory_score"][row_index] = float(
                output["spatial_marker_memory_score_prediction"][local_index]
                .detach()
                .cpu()
            )
            predictions["spatial_frontier_memory_score"][row_index] = float(
                output["spatial_frontier_memory_score_prediction"][local_index]
                .detach()
                .cpu()
            )
            predictions["utility_head"][row_index] = float(
                output["utility_head_prediction"][local_index].detach().cpu()
            )
            predictions["consequence_utility"][row_index] = float(
                output["consequence_utility_prediction"][local_index].detach().cpu()
            )
            predictions["max_reached_goal_logit"][row_index] = float(
                reached_goal_logits[local_index].max().detach().cpu()
            )
            predictions["final_reached_goal_logit"][row_index] = float(
                reached_goal_logits[local_index, -1].detach().cpu()
            )
            predictions["mean_reached_goal_logit"][row_index] = float(
                reached_goal_logits[local_index].mean().detach().cpu()
            )
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--memory-checkpoint", type=Path, required=True)
    parser.add_argument("--no-memory-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-states-per-batch", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    train_rows = read_jsonl(args.train_data)
    validation_rows = read_jsonl(args.validation_data)
    memory_model, memory_report = load_model(args.memory_checkpoint, device=device)
    no_memory_model, no_memory_report = load_model(
        args.no_memory_checkpoint,
        device=device,
    )
    memory_scores = predict_model_scores(
        memory_model,
        validation_rows,
        source_states_per_batch=args.source_states_per_batch,
        device=device,
    )
    no_memory_scores = predict_model_scores(
        no_memory_model,
        validation_rows,
        source_states_per_batch=args.source_states_per_batch,
        device=device,
    )
    prior_predictions = action_sequence_prior_predictions(train_rows, validation_rows)
    score_summaries = {
        "memory": {
            name: summarize_explore_claim_predictions(validation_rows, predictions)
            for name, predictions in memory_scores.items()
        },
        "no_memory": {
            name: summarize_explore_claim_predictions(validation_rows, predictions)
            for name, predictions in no_memory_scores.items()
        },
    }
    summaries = {
        "action_sequence_prior": summarize_explore_claim_predictions(
            validation_rows,
            prior_predictions,
        ),
        "egocentric_marker_memory": summarize_explore_claim_predictions(
            validation_rows,
            egocentric_marker_memory_predictions(validation_rows),
        ),
        "egocentric_explore_claim": summarize_explore_claim_predictions(
            validation_rows,
            egocentric_explore_claim_predictions(validation_rows),
        ),
        "no_memory": summarize_explore_claim_predictions(
            validation_rows,
            no_memory_scores["utility"],
        ),
        "memory": summarize_explore_claim_predictions(
            validation_rows,
            memory_scores["utility"],
        ),
    }
    report = {
        "schema": "jepa_phase3a_explore_claim_report_v0",
        "device": str(device),
        "train_data": str(args.train_data.resolve()),
        "validation_data": str(args.validation_data.resolve()),
        "memory_checkpoint": str(args.memory_checkpoint.resolve()),
        "no_memory_checkpoint": str(args.no_memory_checkpoint.resolve()),
        "memory_training_aggregate": memory_report["final_validation"][
            "primitive_selection"
        ],
        "no_memory_training_aggregate": no_memory_report["final_validation"][
            "primitive_selection"
        ],
        "action_only_prior_aggregate": memory_report["action_only_prior"],
        "train_audit": memory_report["train_audit"],
        "validation_audit": memory_report["validation_audit"],
        "summaries": summaries,
        "score_summaries": score_summaries,
        "comparisons": compare_explore_claim_summaries(summaries),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
