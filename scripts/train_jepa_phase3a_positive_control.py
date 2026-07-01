#!/usr/bin/env python3
"""Train the foundational Phase 3A JEPA positive-control model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.phase3a_positive_control import (  # noqa: E402
    phase3a_action_only_prior,
    phase3a_dataset_audit,
    read_jsonl,
)
from lewm.benchmarks.phase3a_training import (  # noqa: E402
    CONSEQUENCE_TARGET_NAMES,
    Phase3AMaterializedDataset,
    primitive_selection_summary,
    source_grouped_batches,
)
from lewm.benchmarks.rollout_diagnostics import (  # noqa: E402
    summarize_rollout_controls,
    summarize_spatial_stability,
)
from lewm.models.phase3a_jepa import Phase3AJepaModel  # noqa: E402
from lewm.models.spatial_predictor import trainable_parameter_count  # noqa: E402


METRIC_KEYS = (
    "loss",
    "prediction_loss",
    "teacher_forced_prediction_loss",
    "rollout_delta_loss",
    "teacher_forced_delta_loss",
    "decision_prediction_loss",
    "decision_delta_loss",
    "decision_teacher_forced_prediction_loss",
    "decision_teacher_forced_delta_loss",
    "decision_action_contrast_loss",
    "decision_zero_contrast_loss",
    "decision_teacher_forced_action_contrast_loss",
    "decision_teacher_forced_zero_contrast_loss",
    "action_identifiability_loss",
    "zero_action_loss",
    "free_running_action_contrast_loss",
    "free_running_zero_contrast_loss",
    "utility_loss",
    "utility_head_loss",
    "candidate_score_loss",
    "candidate_claim_loss",
    "candidate_marker_memory_loss",
    "candidate_marker_memory_delta_loss",
    "candidate_marker_memory_claim_loss",
    "candidate_marker_memory_ranking_loss",
    "candidate_marker_memory_ranking_ce_loss",
    "candidate_marker_memory_ranking_group_count",
    "structured_marker_memory_loss",
    "structured_marker_memory_start_delta_loss",
    "structured_marker_memory_final_delta_loss",
    "structured_marker_memory_ranking_loss",
    "structured_marker_memory_ranking_ce_loss",
    "structured_marker_memory_ranking_group_count",
    "categorical_marker_memory_loss",
    "categorical_marker_memory_ranking_loss",
    "categorical_marker_memory_ranking_ce_loss",
    "categorical_marker_memory_ranking_group_count",
    "spatial_marker_memory_loss",
    "spatial_marker_memory_cell_loss",
    "spatial_marker_memory_mass_loss",
    "spatial_marker_memory_ranking_loss",
    "spatial_marker_memory_ranking_ce_loss",
    "spatial_marker_memory_ranking_group_count",
    "spatial_frontier_memory_loss",
    "spatial_frontier_observation_loss",
    "spatial_frontier_observation_marker_loss",
    "spatial_frontier_observation_occupancy_loss",
    "spatial_frontier_memory_score_loss",
    "spatial_frontier_memory_occupancy_loss",
    "spatial_frontier_memory_marker_loss",
    "spatial_frontier_memory_marker_cell_loss",
    "spatial_frontier_memory_marker_mass_loss",
    "spatial_frontier_memory_ranking_loss",
    "spatial_frontier_memory_ranking_ce_loss",
    "spatial_frontier_memory_ranking_group_count",
    "candidate_score_ranking_loss",
    "candidate_score_ranking_ce_loss",
    "candidate_score_ranking_regression_loss",
    "candidate_score_ranking_group_count",
    "online_marker_memory_score_mean",
    "online_frontier_marker_score_mean",
    "spatial_frontier_memory_score_mean",
    "spatial_frontier_memory_marker_mass_mean",
    "spatial_frontier_memory_observed_mean",
    "spatial_frontier_memory_free_mean",
    "spatial_frontier_memory_blocked_mean",
    "utility_ranking_loss",
    "utility_ranking_ce_loss",
    "utility_ranking_regression_loss",
    "utility_ranking_group_count",
    "consequence_loss",
    "consequence_binary_loss",
    "consequence_scalar_loss",
    "spatial_variance_loss",
    "real_prediction_mse",
    "hard_negative_mse",
    "zero_action_mse",
    "free_running_hard_negative_mse",
    "free_running_zero_action_mse",
    "decision_hard_negative_mse",
    "decision_zero_action_mse",
    "decision_teacher_forced_hard_negative_mse",
    "decision_teacher_forced_zero_action_mse",
    "decision_mean_target_change_mse",
    "mean_target_change_mse",
    "candidate_marker_memory_valid_count",
    "structured_marker_memory_valid_count",
    "categorical_marker_memory_valid_count",
    "spatial_marker_memory_valid_count",
    "spatial_frontier_memory_valid_count",
    "spatial_frontier_observation_frame_count",
)


def _mean(records: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: sum(record[key] for record in records) / max(len(records), 1)
        for key in METRIC_KEYS
    }


def _metric_record(output: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(output[key].detach().cpu()) for key in METRIC_KEYS}


def _save_checkpoint(path: Path, *, model: Phase3AJepaModel, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "report": report}, path)
    path.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


@torch.no_grad()
def evaluate(
    model: Phase3AJepaModel,
    rows: list[dict],
    *,
    source_states_per_batch: int,
    device: torch.device,
    materialized_rows: Phase3AMaterializedDataset | None = None,
) -> dict:
    model.eval()
    row_cache = materialized_rows or Phase3AMaterializedDataset(rows)
    records = []
    rollout_batches = []
    target_batches = []
    previous_batches = []
    zero_batches = []
    shuffled_batches = []
    decision_rollout_batches = []
    decision_target_batches = []
    decision_previous_batches = []
    decision_zero_batches = []
    decision_shuffled_batches = []
    target_pre = []
    target_norm = []
    decision_targets_for_stability = []
    decision_previous_for_stability = []
    utility_predictions = [0.0 for _ in rows]
    batches = source_grouped_batches(
        rows,
        source_states_per_batch=source_states_per_batch,
        shuffle=False,
    )
    for indices in batches:
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
            spatial_frontier_history_observation_targets=(
                batch.spatial_frontier_history_observation_targets
            ),
            spatial_frontier_vision_observation_targets=(
                batch.spatial_frontier_vision_observation_targets
            ),
            utility_group_ids=batch.utility_group_ids,
            utility_mask=batch.utility_mask,
            wrong_actions=batch.wrong_actions,
            wrong_mask=batch.wrong_mask,
            non_hold_mask=batch.non_hold_mask,
            return_latents=True,
        )
        records.append(_metric_record(output))
        rollout_batches.append(output["rollout"].cpu())
        target_batches.append(output["targets"].cpu())
        previous_batches.append(output["previous_targets"].cpu())
        decision_rollout_batches.append(output["decision_rollout"].cpu())
        decision_target_batches.append(output["decision_targets"].cpu())
        decision_previous_batches.append(output["decision_previous_targets"].cpu())
        zero_actions = torch.zeros_like(batch.actions)
        zero_rollout = model.rollout(
            model.encode_seq(batch.vision, target=False)[:, 0],
            zero_actions,
        )
        zero_batches.append(zero_rollout.cpu())
        decision_zero_batches.append(output["decision_zero_rollout"].cpu())
        shuffled_actions = batch.wrong_actions[:, :, 0].clone()
        shuffled_actions[~batch.wrong_mask[:, :, 0]] = batch.actions[
            ~batch.wrong_mask[:, :, 0]
        ]
        shuffled_rollout = model.rollout(
            model.encode_seq(batch.vision, target=False)[:, 0],
            shuffled_actions,
        )
        shuffled_batches.append(shuffled_rollout.cpu())
        shuffled_decision = model.rollout_decision(
            output["decision_start"],
            shuffled_actions,
        )
        decision_shuffled_batches.append(shuffled_decision.cpu())
        target_pre.append(output["target_pre_normalized"][:, 1:].cpu())
        target_norm.append(output["target_normalized_all"].cpu())
        decision_targets_for_stability.append(output["decision_targets"].cpu())
        decision_previous_for_stability.append(
            output["decision_previous_targets"].cpu()
        )
        for local_index, row_index in enumerate(indices):
            utility_predictions[row_index] = float(
                output["utility_prediction"][local_index].detach().cpu()
            )
    rollout = torch.cat(rollout_batches)
    targets = torch.cat(target_batches)
    previous = torch.cat(previous_batches)
    controls = summarize_rollout_controls(
        rollout=rollout,
        targets=targets,
        persistence=previous,
        zero_action=torch.cat(zero_batches),
        shuffled_action=torch.cat(shuffled_batches),
        previous_targets=previous,
    )
    decision_rollout = torch.cat(decision_rollout_batches)
    decision_targets = torch.cat(decision_target_batches)
    decision_previous = torch.cat(decision_previous_batches)
    decision_controls = summarize_rollout_controls(
        rollout=decision_rollout,
        targets=decision_targets,
        persistence=decision_previous,
        zero_action=torch.cat(decision_zero_batches),
        shuffled_action=torch.cat(decision_shuffled_batches),
        previous_targets=decision_previous,
    )
    stability = summarize_spatial_stability(
        pre_normalized_targets=torch.cat(target_pre),
        normalized_targets=torch.cat([tokens[:, 1:] for tokens in target_norm]),
        previous_normalized_targets=torch.cat(
            [tokens[:, :-1] for tokens in target_norm]
        ),
    )
    decision_stability = summarize_spatial_stability(
        pre_normalized_targets=torch.cat(decision_targets_for_stability),
        normalized_targets=torch.cat(decision_targets_for_stability),
        previous_normalized_targets=torch.cat(decision_previous_for_stability),
    )
    return {
        "metrics_unweighted_batch_mean": _mean(records),
        "rollout_controls": controls,
        "decision_rollout_controls": decision_controls,
        "stability": stability,
        "decision_stability": decision_stability,
        "primitive_selection": primitive_selection_summary(rows, utility_predictions),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="optional checkpoint whose model weights initialize training",
    )
    parser.add_argument(
        "--allow-partial-init-checkpoint",
        action="store_true",
        help="load --init-checkpoint with strict=False for newly enabled heads",
    )
    parser.add_argument("--optimization-steps", type=int, default=256)
    parser.add_argument("--evaluation-interval", type=int, default=128)
    parser.add_argument("--source-states-per-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument(
        "--spatial-memory-size",
        type=int,
        default=None,
        help="odd learned online memory size; defaults to the camera view size",
    )
    parser.add_argument("--pred-layers", type=int, default=2)
    parser.add_argument("--target-ema-momentum", type=float, default=0.99)
    parser.add_argument("--prediction-loss-lambda", type=float, default=1.0)
    parser.add_argument("--action-identifiability-lambda", type=float, default=1.0)
    parser.add_argument("--zero-action-lambda", type=float, default=1.0)
    parser.add_argument("--free-running-action-contrast-lambda", type=float, default=1.0)
    parser.add_argument("--free-running-zero-contrast-lambda", type=float, default=1.0)
    parser.add_argument("--utility-loss-lambda", type=float, default=0.1)
    parser.add_argument("--utility-ranking-loss-lambda", type=float, default=0.1)
    parser.add_argument("--utility-ranking-regression-weight", type=float, default=0.1)
    parser.add_argument(
        "--utility-ranking-loss-type",
        choices=("hard_ce", "soft_ce"),
        default="hard_ce",
    )
    parser.add_argument("--utility-softmax-temperature", type=float, default=0.25)
    parser.add_argument(
        "--utility-source",
        choices=("consequence", "head", "candidate_score"),
        default="candidate_score",
    )
    parser.add_argument("--candidate-score-loss-lambda", type=float, default=1.0)
    parser.add_argument(
        "--candidate-score-regression-weight",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--candidate-score-ranking-loss-type",
        choices=("hard_ce", "soft_ce"),
        default="hard_ce",
    )
    parser.add_argument(
        "--candidate-score-softmax-temperature",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--train-candidate-score-state",
        action="store_true",
        help="allow candidate-score losses to update the imagined state",
    )
    parser.add_argument(
        "--candidate-score-gradient-mode",
        choices=("detached", "start_only", "full"),
        default="detached",
        help=(
            "which candidate-score inputs receive gradients; the legacy "
            "--train-candidate-score-state flag maps to full"
        ),
    )
    parser.add_argument(
        "--candidate-score-source-tokens",
        action="store_true",
        help="let the candidate scorer read encoded source image tokens directly",
    )
    parser.add_argument(
        "--candidate-score-action-summary",
        choices=("statistics", "sequence"),
        default="statistics",
        help=(
            "action features for candidate scoring: first/last/mean statistics "
            "or an order-aware GRU sequence summary"
        ),
    )
    parser.add_argument("--candidate-claim-loss-lambda", type=float, default=0.0)
    parser.add_argument(
        "--candidate-score-claim-logit-weight",
        type=float,
        default=0.0,
        help="add this multiple of the candidate claim logit to the candidate score",
    )
    parser.add_argument(
        "--online-marker-memory-score-weight",
        type=float,
        default=0.0,
        help=(
            "add this multiple of the explicit RGB+odometry online memory "
            "score to the candidate score"
        ),
    )
    parser.add_argument("--candidate-marker-memory-loss-lambda", type=float, default=0.0)
    parser.add_argument(
        "--candidate-marker-memory-score-weight",
        type=float,
        default=0.0,
        help=(
            "add this multiple of the learned marker-memory score to the "
            "candidate score"
        ),
    )
    parser.add_argument(
        "--candidate-marker-memory-delta-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--candidate-marker-memory-claim-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--candidate-marker-memory-ranking-loss-lambda",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--candidate-marker-memory-ranking-loss-type",
        choices=("hard_ce", "soft_ce"),
        default="hard_ce",
    )
    parser.add_argument(
        "--candidate-marker-memory-softmax-temperature",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--candidate-marker-memory-score-mode",
        choices=("claim_plus_distance", "distance"),
        default="claim_plus_distance",
    )
    parser.add_argument("--structured-marker-memory-loss-lambda", type=float, default=0.0)
    parser.add_argument(
        "--structured-marker-memory-score-weight",
        type=float,
        default=0.0,
        help=(
            "add this multiple of the structured egocentric marker-memory "
            "score to the candidate score"
        ),
    )
    parser.add_argument(
        "--structured-marker-memory-ranking-loss-lambda",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--structured-marker-memory-softmax-temperature",
        type=float,
        default=0.25,
    )
    parser.add_argument("--categorical-marker-memory-loss-lambda", type=float, default=0.0)
    parser.add_argument(
        "--categorical-marker-memory-score-weight",
        type=float,
        default=0.0,
        help=(
            "add this multiple of the categorical egocentric memory score "
            "to the candidate score"
        ),
    )
    parser.add_argument(
        "--categorical-marker-memory-ranking-loss-lambda",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--categorical-marker-memory-softmax-temperature",
        type=float,
        default=0.25,
    )
    parser.add_argument("--categorical-marker-memory-radius", type=int, default=2)
    parser.add_argument("--spatial-marker-memory-loss-lambda", type=float, default=0.0)
    parser.add_argument(
        "--spatial-marker-memory-score-weight",
        type=float,
        default=0.0,
        help=(
            "add this multiple of the learned spatial belief-map memory score "
            "to the candidate score"
        ),
    )
    parser.add_argument(
        "--spatial-marker-memory-ranking-loss-lambda",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--spatial-marker-memory-softmax-temperature",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--spatial-marker-memory-score-temperature",
        type=float,
        default=1.0,
        help="temperature used to sharpen learned marker belief for claim scoring",
    )
    parser.add_argument("--spatial-frontier-memory-loss-lambda", type=float, default=0.0)
    parser.add_argument(
        "--spatial-frontier-observation-loss-lambda",
        type=float,
        default=0.0,
        help=(
            "train the learned frontier detector against per-frame "
            "marker/observed/free/blocked RGB-map targets"
        ),
    )
    parser.add_argument(
        "--spatial-frontier-memory-score-loss-lambda",
        type=float,
        default=0.0,
        help=(
            "distill the explicit online frontier+marker score into the learned "
            "spatial frontier memory score"
        ),
    )
    parser.add_argument(
        "--spatial-frontier-memory-score-weight",
        type=float,
        default=0.0,
        help=(
            "add this multiple of the learned spatial frontier+marker memory "
            "score to the candidate score"
        ),
    )
    parser.add_argument(
        "--spatial-frontier-memory-ranking-loss-lambda",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--spatial-frontier-memory-softmax-temperature",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--spatial-frontier-memory-occupancy-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--spatial-frontier-memory-marker-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--spatial-frontier-memory-marker-cell-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--spatial-frontier-memory-marker-mass-loss-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--spatial-frontier-memory-detector-init",
        choices=("direct_rgb", "neutral", "random"),
        default="direct_rgb",
    )
    parser.add_argument(
        "--spatial-frontier-memory-detector-arch",
        choices=("linear", "mlp"),
        default="linear",
        help="per-cell RGB detector used by the spatial frontier memory head",
    )
    parser.add_argument(
        "--spatial-frontier-memory-gate-mode",
        choices=("linear", "threshold"),
        default="linear",
        help="how marker confidence gates novelty-vs-claim scoring",
    )
    parser.add_argument(
        "--spatial-frontier-marker-source",
        choices=("frontier", "spatial_marker"),
        default="frontier",
        help=(
            "which learned marker belief feeds the combined frontier+claim "
            "score"
        ),
    )
    parser.add_argument(
        "--spatial-frontier-collision-penalty",
        type=float,
        default=2.0,
        help="learned frontier score penalty for forward motion into predicted blocked cells",
    )
    parser.add_argument(
        "--spatial-frontier-novelty-reward",
        type=float,
        default=0.35,
        help="learned frontier score reward per newly visible predicted cell",
    )
    parser.add_argument(
        "--spatial-frontier-marker-gate-threshold",
        type=float,
        default=0.5,
        help="marker mass where threshold-gated frontier scoring starts claiming",
    )
    parser.add_argument(
        "--spatial-frontier-marker-gate-width",
        type=float,
        default=0.25,
        help="marker mass interval over which threshold-gated frontier scoring switches to claiming",
    )
    parser.add_argument(
        "--spatial-frontier-marker-update-threshold",
        type=float,
        default=0.0,
        help="minimum learned marker presence before writing marker evidence to memory",
    )
    parser.add_argument(
        "--spatial-frontier-marker-update-width",
        type=float,
        default=1.0,
        help="presence interval over which marker memory writes ramp from zero to full",
    )
    parser.add_argument(
        "--train-consequence-head-state",
        action="store_true",
        help="allow consequence readout losses to update the imagined state",
    )
    parser.add_argument("--consequence-loss-lambda", type=float, default=0.2)
    parser.add_argument("--rollout-delta-loss-lambda", type=float, default=1.0)
    parser.add_argument("--teacher-forced-delta-loss-lambda", type=float, default=1.0)
    parser.add_argument("--decision-token-count", type=int, default=4)
    parser.add_argument(
        "--decision-rollout-mode",
        choices=("recurrent", "autoregressive"),
        default="recurrent",
    )
    parser.add_argument(
        "--decision-recurrent-update",
        choices=("absolute", "residual"),
        default="absolute",
    )
    parser.add_argument(
        "--decision-target-geometry",
        choices=("normalized", "linear"),
        default="normalized",
    )
    parser.add_argument("--decision-target-scale", type=float, default=None)
    parser.add_argument("--decision-prediction-loss-lambda", type=float, default=1.0)
    parser.add_argument("--decision-delta-loss-lambda", type=float, default=1.0)
    parser.add_argument(
        "--decision-teacher-forced-prediction-loss-lambda",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--decision-teacher-forced-delta-loss-lambda",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--decision-teacher-forced-action-contrast-lambda",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--decision-teacher-forced-zero-contrast-lambda",
        type=float,
        default=1.0,
    )
    parser.add_argument("--decision-action-contrast-lambda", type=float, default=1.0)
    parser.add_argument("--decision-zero-contrast-lambda", type=float, default=1.0)
    parser.add_argument(
        "--use-memory-context",
        action="store_true",
        help="seed decision tokens from a learned history observation/action memory",
    )
    parser.add_argument(
        "--memory-frame-summary",
        choices=("summary", "spatial"),
        default="summary",
        help=(
            "history frame representation for the memory GRU: compact pooled/"
            "beacon/center summary, or the full spatial token grid"
        ),
    )
    parser.add_argument(
        "--memory-marker-features",
        action="store_true",
        help=(
            "append RGB-only green-marker saliency/centroid features to each "
            "history frame before memory encoding"
        ),
    )
    parser.add_argument("--spatial-variance-lambda", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=64)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help=(
            "write step checkpoints every N optimization steps; 0 disables "
            "intermediate checkpoint writes"
        ),
    )
    args = parser.parse_args()
    if args.checkpoint_interval < 0:
        raise SystemExit("--checkpoint-interval must be non-negative")

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    candidate_score_gradient_mode = (
        "full"
        if args.train_candidate_score_state
        else args.candidate_score_gradient_mode
    )
    train_rows = read_jsonl(args.train_data)
    validation_rows = read_jsonl(args.validation_data)
    if not train_rows or not validation_rows:
        raise SystemExit("train and validation rows must be non-empty")
    view_size = len(train_rows[0]["start_observation_rgb"][0])
    action_dim = len(train_rows[0]["active_blocks"][0])
    model = Phase3AJepaModel(
        view_size=view_size,
        action_dim=action_dim,
        spatial_memory_size=args.spatial_memory_size,
        latent_dim=args.latent_dim,
        pred_layers=args.pred_layers,
        target_ema_momentum=args.target_ema_momentum,
        prediction_loss_lambda=args.prediction_loss_lambda,
        action_identifiability_lambda=args.action_identifiability_lambda,
        zero_action_lambda=args.zero_action_lambda,
        free_running_action_contrast_lambda=(
            args.free_running_action_contrast_lambda
        ),
        free_running_zero_contrast_lambda=args.free_running_zero_contrast_lambda,
        utility_loss_lambda=args.utility_loss_lambda,
        utility_ranking_loss_lambda=args.utility_ranking_loss_lambda,
        utility_ranking_regression_weight=args.utility_ranking_regression_weight,
        utility_ranking_loss_type=args.utility_ranking_loss_type,
        utility_softmax_temperature=args.utility_softmax_temperature,
        utility_source=args.utility_source,
        candidate_score_loss_lambda=args.candidate_score_loss_lambda,
        candidate_score_regression_weight=args.candidate_score_regression_weight,
        candidate_score_ranking_loss_type=args.candidate_score_ranking_loss_type,
        candidate_score_softmax_temperature=(
            args.candidate_score_softmax_temperature
        ),
        detach_candidate_score_state=candidate_score_gradient_mode == "detached",
        candidate_score_gradient_mode=candidate_score_gradient_mode,
        candidate_score_source_tokens=args.candidate_score_source_tokens,
        candidate_score_action_summary=args.candidate_score_action_summary,
        candidate_claim_loss_lambda=args.candidate_claim_loss_lambda,
        candidate_score_claim_logit_weight=args.candidate_score_claim_logit_weight,
        online_marker_memory_score_weight=args.online_marker_memory_score_weight,
        candidate_marker_memory_loss_lambda=(
            args.candidate_marker_memory_loss_lambda
        ),
        candidate_marker_memory_score_weight=(
            args.candidate_marker_memory_score_weight
        ),
        candidate_marker_memory_delta_loss_weight=(
            args.candidate_marker_memory_delta_loss_weight
        ),
        candidate_marker_memory_claim_loss_weight=(
            args.candidate_marker_memory_claim_loss_weight
        ),
        candidate_marker_memory_ranking_loss_lambda=(
            args.candidate_marker_memory_ranking_loss_lambda
        ),
        candidate_marker_memory_ranking_loss_type=(
            args.candidate_marker_memory_ranking_loss_type
        ),
        candidate_marker_memory_softmax_temperature=(
            args.candidate_marker_memory_softmax_temperature
        ),
        candidate_marker_memory_score_mode=args.candidate_marker_memory_score_mode,
        structured_marker_memory_loss_lambda=(
            args.structured_marker_memory_loss_lambda
        ),
        structured_marker_memory_score_weight=(
            args.structured_marker_memory_score_weight
        ),
        structured_marker_memory_ranking_loss_lambda=(
            args.structured_marker_memory_ranking_loss_lambda
        ),
        structured_marker_memory_softmax_temperature=(
            args.structured_marker_memory_softmax_temperature
        ),
        categorical_marker_memory_loss_lambda=(
            args.categorical_marker_memory_loss_lambda
        ),
        categorical_marker_memory_score_weight=(
            args.categorical_marker_memory_score_weight
        ),
        categorical_marker_memory_ranking_loss_lambda=(
            args.categorical_marker_memory_ranking_loss_lambda
        ),
        categorical_marker_memory_softmax_temperature=(
            args.categorical_marker_memory_softmax_temperature
        ),
        categorical_marker_memory_radius=args.categorical_marker_memory_radius,
        spatial_marker_memory_loss_lambda=(
            args.spatial_marker_memory_loss_lambda
        ),
        spatial_marker_memory_score_weight=(
            args.spatial_marker_memory_score_weight
        ),
        spatial_marker_memory_ranking_loss_lambda=(
            args.spatial_marker_memory_ranking_loss_lambda
        ),
        spatial_marker_memory_softmax_temperature=(
            args.spatial_marker_memory_softmax_temperature
        ),
        spatial_marker_memory_score_temperature=(
            args.spatial_marker_memory_score_temperature
        ),
        spatial_frontier_memory_loss_lambda=(
            args.spatial_frontier_memory_loss_lambda
        ),
        spatial_frontier_observation_loss_lambda=(
            args.spatial_frontier_observation_loss_lambda
        ),
        spatial_frontier_memory_score_loss_lambda=(
            args.spatial_frontier_memory_score_loss_lambda
        ),
        spatial_frontier_memory_score_weight=(
            args.spatial_frontier_memory_score_weight
        ),
        spatial_frontier_memory_ranking_loss_lambda=(
            args.spatial_frontier_memory_ranking_loss_lambda
        ),
        spatial_frontier_memory_softmax_temperature=(
            args.spatial_frontier_memory_softmax_temperature
        ),
        spatial_frontier_memory_occupancy_loss_weight=(
            args.spatial_frontier_memory_occupancy_loss_weight
        ),
        spatial_frontier_memory_marker_loss_weight=(
            args.spatial_frontier_memory_marker_loss_weight
        ),
        spatial_frontier_memory_marker_cell_loss_weight=(
            args.spatial_frontier_memory_marker_cell_loss_weight
        ),
        spatial_frontier_memory_marker_mass_loss_weight=(
            args.spatial_frontier_memory_marker_mass_loss_weight
        ),
        spatial_frontier_memory_detector_init=(
            args.spatial_frontier_memory_detector_init
        ),
        spatial_frontier_memory_detector_arch=(
            args.spatial_frontier_memory_detector_arch
        ),
        spatial_frontier_memory_gate_mode=args.spatial_frontier_memory_gate_mode,
        spatial_frontier_marker_source=args.spatial_frontier_marker_source,
        spatial_frontier_collision_penalty=args.spatial_frontier_collision_penalty,
        spatial_frontier_novelty_reward=args.spatial_frontier_novelty_reward,
        spatial_frontier_marker_gate_threshold=(
            args.spatial_frontier_marker_gate_threshold
        ),
        spatial_frontier_marker_gate_width=args.spatial_frontier_marker_gate_width,
        spatial_frontier_marker_update_threshold=(
            args.spatial_frontier_marker_update_threshold
        ),
        spatial_frontier_marker_update_width=(
            args.spatial_frontier_marker_update_width
        ),
        detach_consequence_head_state=not args.train_consequence_head_state,
        consequence_loss_lambda=args.consequence_loss_lambda,
        rollout_delta_loss_lambda=args.rollout_delta_loss_lambda,
        teacher_forced_delta_loss_lambda=args.teacher_forced_delta_loss_lambda,
        decision_token_count=args.decision_token_count,
        decision_rollout_mode=args.decision_rollout_mode,
        decision_recurrent_update=args.decision_recurrent_update,
        decision_target_geometry=args.decision_target_geometry,
        decision_target_scale=args.decision_target_scale,
        decision_prediction_loss_lambda=args.decision_prediction_loss_lambda,
        decision_delta_loss_lambda=args.decision_delta_loss_lambda,
        decision_teacher_forced_prediction_loss_lambda=(
            args.decision_teacher_forced_prediction_loss_lambda
        ),
        decision_teacher_forced_delta_loss_lambda=(
            args.decision_teacher_forced_delta_loss_lambda
        ),
        decision_teacher_forced_action_contrast_lambda=(
            args.decision_teacher_forced_action_contrast_lambda
        ),
        decision_teacher_forced_zero_contrast_lambda=(
            args.decision_teacher_forced_zero_contrast_lambda
        ),
        decision_action_contrast_lambda=args.decision_action_contrast_lambda,
        decision_zero_contrast_lambda=args.decision_zero_contrast_lambda,
        use_memory_context=args.use_memory_context,
        memory_frame_summary=args.memory_frame_summary,
        memory_marker_features=args.memory_marker_features,
        spatial_variance_lambda=args.spatial_variance_lambda,
    ).to(device)
    if args.init_checkpoint is not None:
        try:
            init_checkpoint = torch.load(
                args.init_checkpoint,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            init_checkpoint = torch.load(args.init_checkpoint, map_location=device)
        init_state = init_checkpoint["model_state_dict"]
        skipped_shape_keys: list[dict] = []
        if args.allow_partial_init_checkpoint:
            model_state = model.state_dict()
            compatible_state = {}
            for key, value in init_state.items():
                target = model_state.get(key)
                if target is not None and tuple(target.shape) != tuple(value.shape):
                    skipped_shape_keys.append(
                        {
                            "key": key,
                            "checkpoint_shape": list(value.shape),
                            "model_shape": list(target.shape),
                        }
                    )
                    continue
                compatible_state[key] = value
            init_state = compatible_state
        init_result = model.load_state_dict(
            init_state,
            strict=not args.allow_partial_init_checkpoint,
        )
        if args.allow_partial_init_checkpoint:
            print(
                json.dumps(
                    {
                        "partial_init_checkpoint": str(
                            args.init_checkpoint.resolve()
                        ),
                        "missing_keys": list(init_result.missing_keys),
                        "skipped_shape_keys": skipped_shape_keys,
                        "unexpected_keys": list(init_result.unexpected_keys),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model_config = {
        "view_size": view_size,
        "spatial_memory_size": (
            args.spatial_memory_size
            if args.spatial_memory_size is not None
            else view_size
        ),
        "action_dim": action_dim,
        "latent_dim": args.latent_dim,
        "pred_layers": args.pred_layers,
        "target_ema_momentum": args.target_ema_momentum,
        "prediction_loss_lambda": args.prediction_loss_lambda,
        "action_identifiability_lambda": args.action_identifiability_lambda,
        "zero_action_lambda": args.zero_action_lambda,
        "free_running_action_contrast_lambda": (
            args.free_running_action_contrast_lambda
        ),
        "free_running_zero_contrast_lambda": (
            args.free_running_zero_contrast_lambda
        ),
        "utility_loss_lambda": args.utility_loss_lambda,
        "utility_ranking_loss_lambda": args.utility_ranking_loss_lambda,
        "utility_ranking_regression_weight": (
            args.utility_ranking_regression_weight
        ),
        "utility_ranking_loss_type": args.utility_ranking_loss_type,
        "utility_softmax_temperature": args.utility_softmax_temperature,
        "utility_source": args.utility_source,
        "candidate_score_loss_lambda": args.candidate_score_loss_lambda,
        "candidate_score_regression_weight": (
            args.candidate_score_regression_weight
        ),
        "candidate_score_ranking_loss_type": (
            args.candidate_score_ranking_loss_type
        ),
        "candidate_score_softmax_temperature": (
            args.candidate_score_softmax_temperature
        ),
        "detach_candidate_score_state": (
            candidate_score_gradient_mode == "detached"
        ),
        "candidate_score_gradient_mode": candidate_score_gradient_mode,
        "candidate_score_source_tokens": args.candidate_score_source_tokens,
        "candidate_score_action_summary": args.candidate_score_action_summary,
        "candidate_claim_loss_lambda": args.candidate_claim_loss_lambda,
        "candidate_score_claim_logit_weight": (
            args.candidate_score_claim_logit_weight
        ),
        "online_marker_memory_score_weight": (
            args.online_marker_memory_score_weight
        ),
        "candidate_marker_memory_loss_lambda": (
            args.candidate_marker_memory_loss_lambda
        ),
        "candidate_marker_memory_score_weight": (
            args.candidate_marker_memory_score_weight
        ),
        "candidate_marker_memory_delta_loss_weight": (
            args.candidate_marker_memory_delta_loss_weight
        ),
        "candidate_marker_memory_claim_loss_weight": (
            args.candidate_marker_memory_claim_loss_weight
        ),
        "candidate_marker_memory_ranking_loss_lambda": (
            args.candidate_marker_memory_ranking_loss_lambda
        ),
        "candidate_marker_memory_ranking_loss_type": (
            args.candidate_marker_memory_ranking_loss_type
        ),
        "candidate_marker_memory_softmax_temperature": (
            args.candidate_marker_memory_softmax_temperature
        ),
        "candidate_marker_memory_score_mode": (
            args.candidate_marker_memory_score_mode
        ),
        "structured_marker_memory_loss_lambda": (
            args.structured_marker_memory_loss_lambda
        ),
        "structured_marker_memory_score_weight": (
            args.structured_marker_memory_score_weight
        ),
        "structured_marker_memory_ranking_loss_lambda": (
            args.structured_marker_memory_ranking_loss_lambda
        ),
        "structured_marker_memory_softmax_temperature": (
            args.structured_marker_memory_softmax_temperature
        ),
        "categorical_marker_memory_loss_lambda": (
            args.categorical_marker_memory_loss_lambda
        ),
        "categorical_marker_memory_score_weight": (
            args.categorical_marker_memory_score_weight
        ),
        "categorical_marker_memory_ranking_loss_lambda": (
            args.categorical_marker_memory_ranking_loss_lambda
        ),
        "categorical_marker_memory_softmax_temperature": (
            args.categorical_marker_memory_softmax_temperature
        ),
        "categorical_marker_memory_radius": (
            args.categorical_marker_memory_radius
        ),
        "spatial_marker_memory_loss_lambda": (
            args.spatial_marker_memory_loss_lambda
        ),
        "spatial_marker_memory_score_weight": (
            args.spatial_marker_memory_score_weight
        ),
        "spatial_marker_memory_ranking_loss_lambda": (
            args.spatial_marker_memory_ranking_loss_lambda
        ),
        "spatial_marker_memory_softmax_temperature": (
            args.spatial_marker_memory_softmax_temperature
        ),
        "spatial_marker_memory_score_temperature": (
            args.spatial_marker_memory_score_temperature
        ),
        "spatial_frontier_memory_loss_lambda": (
            args.spatial_frontier_memory_loss_lambda
        ),
        "spatial_frontier_observation_loss_lambda": (
            args.spatial_frontier_observation_loss_lambda
        ),
        "spatial_frontier_memory_score_loss_lambda": (
            args.spatial_frontier_memory_score_loss_lambda
        ),
        "spatial_frontier_memory_score_weight": (
            args.spatial_frontier_memory_score_weight
        ),
        "spatial_frontier_memory_ranking_loss_lambda": (
            args.spatial_frontier_memory_ranking_loss_lambda
        ),
        "spatial_frontier_memory_softmax_temperature": (
            args.spatial_frontier_memory_softmax_temperature
        ),
        "spatial_frontier_memory_occupancy_loss_weight": (
            args.spatial_frontier_memory_occupancy_loss_weight
        ),
        "spatial_frontier_memory_marker_loss_weight": (
            args.spatial_frontier_memory_marker_loss_weight
        ),
        "spatial_frontier_memory_marker_cell_loss_weight": (
            args.spatial_frontier_memory_marker_cell_loss_weight
        ),
        "spatial_frontier_memory_marker_mass_loss_weight": (
            args.spatial_frontier_memory_marker_mass_loss_weight
        ),
        "spatial_frontier_memory_detector_init": (
            args.spatial_frontier_memory_detector_init
        ),
        "spatial_frontier_memory_detector_arch": (
            args.spatial_frontier_memory_detector_arch
        ),
        "spatial_frontier_memory_gate_mode": (
            args.spatial_frontier_memory_gate_mode
        ),
        "spatial_frontier_marker_source": args.spatial_frontier_marker_source,
        "spatial_frontier_collision_penalty": (
            args.spatial_frontier_collision_penalty
        ),
        "spatial_frontier_novelty_reward": args.spatial_frontier_novelty_reward,
        "spatial_frontier_marker_gate_threshold": (
            args.spatial_frontier_marker_gate_threshold
        ),
        "spatial_frontier_marker_gate_width": (
            args.spatial_frontier_marker_gate_width
        ),
        "spatial_frontier_marker_update_threshold": (
            args.spatial_frontier_marker_update_threshold
        ),
        "spatial_frontier_marker_update_width": (
            args.spatial_frontier_marker_update_width
        ),
        "detach_consequence_head_state": not args.train_consequence_head_state,
        "consequence_loss_lambda": args.consequence_loss_lambda,
        "consequence_target_names": CONSEQUENCE_TARGET_NAMES,
        "rollout_delta_loss_lambda": args.rollout_delta_loss_lambda,
        "teacher_forced_delta_loss_lambda": (
            args.teacher_forced_delta_loss_lambda
        ),
        "decision_token_count": args.decision_token_count,
        "decision_rollout_mode": args.decision_rollout_mode,
        "decision_recurrent_update": args.decision_recurrent_update,
        "decision_target_geometry": args.decision_target_geometry,
        "decision_target_scale": (
            args.decision_target_scale
            if args.decision_target_scale is not None
            else args.latent_dim**0.5
        ),
        "decision_prediction_loss_lambda": (
            args.decision_prediction_loss_lambda
        ),
        "decision_delta_loss_lambda": args.decision_delta_loss_lambda,
        "decision_teacher_forced_prediction_loss_lambda": (
            args.decision_teacher_forced_prediction_loss_lambda
        ),
        "decision_teacher_forced_delta_loss_lambda": (
            args.decision_teacher_forced_delta_loss_lambda
        ),
        "decision_teacher_forced_action_contrast_lambda": (
            args.decision_teacher_forced_action_contrast_lambda
        ),
        "decision_teacher_forced_zero_contrast_lambda": (
            args.decision_teacher_forced_zero_contrast_lambda
        ),
        "decision_action_contrast_lambda": args.decision_action_contrast_lambda,
        "decision_zero_contrast_lambda": args.decision_zero_contrast_lambda,
        "use_memory_context": args.use_memory_context,
        "memory_frame_summary": args.memory_frame_summary,
        "memory_marker_features": args.memory_marker_features,
        "spatial_variance_lambda": args.spatial_variance_lambda,
    }
    report_args = {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    train_audit = phase3a_dataset_audit(train_rows)
    validation_audit = phase3a_dataset_audit(validation_rows)
    action_only_prior = phase3a_action_only_prior(train_rows, validation_rows)
    trainable_parameters = trainable_parameter_count(model)
    train_cache = Phase3AMaterializedDataset(train_rows)
    validation_cache = Phase3AMaterializedDataset(validation_rows)
    batches = source_grouped_batches(
        train_rows,
        source_states_per_batch=args.source_states_per_batch,
        shuffle=True,
        seed=args.seed,
    )
    history = []

    def record_step(step: int, grad_norm: torch.Tensor, output: dict[str, torch.Tensor]) -> None:
        record = {
            "step": step,
            "gradient_norm_pre_clip": float(grad_norm.detach().cpu()),
            **_metric_record(output),
        }
        if history and history[-1]["step"] == step:
            history[-1].update(record)
        else:
            history.append(record)

    def build_report(
        *,
        final_validation: dict,
        training_complete: bool,
        completed_steps: int,
    ) -> dict:
        return {
            "schema": "jepa_phase3a_positive_control_training_report_v0",
            "args": report_args,
            "device": str(device),
            "train_data": str(args.train_data.resolve()),
            "validation_data": str(args.validation_data.resolve()),
            "train_audit": train_audit,
            "validation_audit": validation_audit,
            "action_only_prior": action_only_prior,
            "trainable_parameters": trainable_parameters,
            "model_config": model_config,
            "history": history,
            "final_validation": final_validation,
            "training_complete": training_complete,
            "completed_steps": completed_steps,
        }

    for step in range(1, args.optimization_steps + 1):
        indices = batches[(step - 1) % len(batches)]
        batch = train_cache.materialize_batch(indices).to(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
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
            spatial_frontier_history_observation_targets=(
                batch.spatial_frontier_history_observation_targets
            ),
            spatial_frontier_vision_observation_targets=(
                batch.spatial_frontier_vision_observation_targets
            ),
            utility_group_ids=batch.utility_group_ids,
            utility_mask=batch.utility_mask,
            wrong_actions=batch.wrong_actions,
            wrong_mask=batch.wrong_mask,
            non_hold_mask=batch.non_hold_mask,
        )
        output["loss"].backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.max_grad_norm,
        )
        optimizer.step()
        model.update_target_encoder()
        should_checkpoint = (
            args.checkpoint_interval > 0 and step % args.checkpoint_interval == 0
        )
        should_evaluate = (
            step % args.evaluation_interval == 0
            or step == args.optimization_steps
            or should_checkpoint
        )
        if step % args.log_every == 0 or step == 1 or should_evaluate:
            record_step(step, grad_norm, output)
        if should_evaluate:
            validation = evaluate(
                model,
                validation_rows,
                source_states_per_batch=args.source_states_per_batch,
                device=device,
                materialized_rows=validation_cache,
            )
            history[-1]["validation_snapshot"] = validation
            if should_checkpoint and step != args.optimization_steps:
                checkpoint_path = args.output.with_name(
                    f"{args.output.stem}_step{step:06d}{args.output.suffix}"
                )
                _save_checkpoint(
                    checkpoint_path,
                    model=model,
                    report=build_report(
                        final_validation=validation,
                        training_complete=False,
                        completed_steps=step,
                    ),
                )
    final_validation = evaluate(
        model,
        validation_rows,
        source_states_per_batch=args.source_states_per_batch,
        device=device,
        materialized_rows=validation_cache,
    )
    report = build_report(
        final_validation=final_validation,
        training_complete=True,
        completed_steps=args.optimization_steps,
    )
    _save_checkpoint(args.output, model=model, report=report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
