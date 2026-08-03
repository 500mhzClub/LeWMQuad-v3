#!/usr/bin/env python3
"""Fresh-process cache-only replay for the dense shared DINO calibration V1."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_dinov2_dense_shared_spatial_readout_calibration_v1 as mechanism,
)
from lewm.benchmarks.go2_world_model_counterfactual_pilot_v1 import (  # noqa: E402
    FAMILIES,
)
from lewm.models.go2_dinov2_dense_shared_spatial_readout_calibration_v1 import (  # noqa: E402
    DenseSharedSpatialReadoutV1,
    dense_shared_state_identity_v1,
)
from scripts import (  # noqa: E402
    evaluate_go2_world_model_visual_domain_parity_task_relevance_v1
    as task_relevance,
)
from scripts import (  # noqa: E402
    run_go2_dinov2_dense_shared_spatial_readout_calibration_v1 as runner,
)


class DenseSharedReplayError(RuntimeError):
    """Raised when fresh replay differs from the bound primary execution."""


def _canonical_equal(left: object, right: object) -> bool:
    return runner.canonical_bytes_v1(left) == runner.canonical_bytes_v1(right)


def _exact_tree_equal(left: object, right: object) -> bool:
    """Compare checkpoint trees, including exact tensor dtype, shape, and value."""

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and type(left) is type(right)
            and len(left) == len(right)
            and all(
                _exact_tree_equal(left_value, right_value)
                for left_value, right_value in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def _read_checkpoint_v1(
    path: Path, *, expected_sha256: str, expected_byte_count: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = runner.file_binding_v1(path)
    if (
        binding["sha256"] != expected_sha256
        or binding["byte_count"] != expected_byte_count
    ):
        raise DenseSharedReplayError("checkpoint caller binding changed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise DenseSharedReplayError("checkpoint is not a safe Torch payload") from error
    if not isinstance(payload, Mapping):
        raise DenseSharedReplayError("checkpoint payload is not a mapping")
    return dict(payload), binding


@contextmanager
def scoped_replay_compatibility_admission_v1(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
) -> Iterator[dict[str, Any]]:
    """Publish the replay receipt inside the one frozen strict-loader call."""

    receipt_path = Path(str(authority["output_root"])) / (
        "replay_compatibility_receipt.json"
    )
    stored = runner._load_stored_task_relevance_v1(authority)  # noqa: SLF001
    admitted_document, replayed_admission = (
        runner._replay_prior_compatibility_admission_v1(  # noqa: SLF001
            authority, stored
        )
    )
    expected_call = runner._task_relevance_call_bindings_v1(stored)  # noqa: SLF001
    original_evaluator = task_relevance.evaluate_task_relevance_v1
    original_loader = runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1
    state: dict[str, Any] = {
        "evaluator_calls": 0,
        "loader_calls": 0,
        "receipt_binding": None,
        "admission": None,
    }

    def admitted_evaluator(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        state["evaluator_calls"] += 1
        if state["evaluator_calls"] != 1 or args or kwargs != expected_call:
            raise DenseSharedReplayError(
                "task-relevance compatibility-replay call changed"
            )
        receipt = runner._compatibility_receipt_v1(  # noqa: SLF001
            phase="replay",
            authority=authority,
            authority_binding=authority_binding,
            admission=replayed_admission,
        )
        runner._write_json_exclusive(receipt_path, receipt)  # noqa: SLF001
        state["receipt_binding"] = runner.file_binding_v1(receipt_path)
        state["admission"] = dict(replayed_admission)
        return admitted_document

    def admitted_loader(*args: Any, **kwargs: Any) -> object:
        state["loader_calls"] += 1
        if state["loader_calls"] != 1:
            raise DenseSharedReplayError("strict posthoc loader call count changed")
        bundle = original_loader(*args, **kwargs)
        if state["evaluator_calls"] != 1 or state["receipt_binding"] is None:
            raise DenseSharedReplayError(
                "replay compatibility receipt was not published before loader return"
            )
        return bundle

    task_relevance.evaluate_task_relevance_v1 = admitted_evaluator
    runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1 = admitted_loader
    try:
        yield state
    finally:
        task_relevance.evaluate_task_relevance_v1 = original_evaluator
        runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1 = original_loader


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def _selected_summary_v1(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "states": len(rows),
        "normalized_rank_regret": _mean(rows, "normalized_rank_regret"),
        "oracle_equivalent_selection_rate": _mean(rows, "oracle_match"),
        "physical_target_progress_m": _mean(rows, "physical_target_progress_m"),
        "physical_path_length_m": _mean(rows, "physical_path_length_m"),
        "chosen_action_histogram": {
            str(action): sum(
                int(row["selected_action_id"]) == action for row in rows
            )
            for action in range(mechanism.ACTION_COUNT)
        },
    }


def _selection_report_v1(
    groups: Sequence[Any], scores: np.ndarray
) -> dict[str, Any]:
    """Independently select actions and reproduce the frozen arm report."""

    array = np.asarray(scores, dtype=np.float64)
    if array.shape != (len(groups), mechanism.ACTION_COUNT) or not np.isfinite(
        array
    ).all():
        raise DenseSharedReplayError("arm scores are invalid")
    rows: list[dict[str, Any]] = []
    for group_index, group in enumerate(groups):
        branches = tuple(getattr(group, "branches", ()))
        if len(branches) != mechanism.ACTION_COUNT:
            raise DenseSharedReplayError("evaluation branch inventory changed")
        state_scores = array[group_index]
        selected = min(
            range(mechanism.ACTION_COUNT),
            key=lambda action: (state_scores[action], action),
        )
        ranks = np.asarray(
            [branch.oracle_dense_rank for branch in branches], dtype=np.float64
        )
        if ranks.shape != (mechanism.ACTION_COUNT,) or not np.isfinite(ranks).all():
            raise DenseSharedReplayError("physical dense ranks changed")
        denominator = max(1.0, float(ranks.max()))
        oracle_action = min(
            range(mechanism.ACTION_COUNT),
            key=lambda action: (ranks[action], action),
        )
        labels = branches[selected].labels
        oracle_labels = branches[oracle_action].labels
        rows.append(
            {
                "state_id": group.state_id,
                "scene_id": group.scene_id,
                "family": group.family,
                "selected_action_id": selected,
                "oracle_action_id": oracle_action,
                "selected_dense_rank": int(ranks[selected]),
                "normalized_rank_regret": float(ranks[selected] / denominator),
                "random_expected_normalized_rank_regret": float(
                    ranks.mean() / denominator
                ),
                "oracle_match": bool(ranks[selected] == ranks.min()),
                "physical_fell": labels.fell,
                "physical_tipped": labels.tipped,
                "physical_target_progress_m": labels.target_progress_m,
                "physical_path_length_m": labels.path_length_m,
                "physical_progress_delta_to_canonical_oracle_m": (
                    oracle_labels.target_progress_m - labels.target_progress_m
                ),
                "planar_clearance_proxy_min_m": (
                    labels.planar_clearance_proxy_min_m
                ),
                "grid_recoverability_proxy": labels.grid_recoverability_proxy,
            }
        )
    if not rows:
        raise DenseSharedReplayError("at least one evaluation state is required")
    summary: dict[str, Any] = {
        "groups": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "normalized_rank_regret": _mean(rows, "normalized_rank_regret"),
        "random_expected_normalized_rank_regret": _mean(
            rows, "random_expected_normalized_rank_regret"
        ),
        "oracle_match_rate": _mean(rows, "oracle_match"),
        "physical_fall_rate": _mean(rows, "physical_fell"),
        "physical_tip_rate": _mean(rows, "physical_tipped"),
        "physical_target_progress_m": _mean(rows, "physical_target_progress_m"),
        "physical_progress_delta_to_canonical_oracle_m": _mean(
            rows, "physical_progress_delta_to_canonical_oracle_m"
        ),
        "physical_path_length_m": _mean(rows, "physical_path_length_m"),
    }
    proxy_summary = {}
    for key in ("planar_clearance_proxy_min_m", "grid_recoverability_proxy"):
        if all(row[key] is not None for row in rows):
            proxy_summary[key] = _mean(rows, key)
    if proxy_summary:
        summary["nonphysical_proxy_metrics"] = proxy_summary
    summary["oracle_equivalent_selection_rate"] = summary["oracle_match_rate"]
    summary["chosen_action_histogram"] = _selected_summary_v1(rows)[
        "chosen_action_histogram"
    ]
    return {
        "summary": summary,
        "group_results": rows,
        "per_family": {
            family: _selected_summary_v1(
                [row for row in rows if row["family"] == family]
            )
            for family in FAMILIES
        },
        "per_scene": [
            {
                "scene_id": scene,
                "family": next(
                    str(row["family"]) for row in rows if row["scene_id"] == scene
                ),
                **_selected_summary_v1(
                    [row for row in rows if row["scene_id"] == scene]
                ),
            }
            for scene in sorted({str(row["scene_id"]) for row in rows})
        ],
    }


def _random_expected_report_v1(plan: Any) -> dict[str, Any]:
    rows = []
    for state in plan.states:
        ranks = np.asarray(state.dense_ranks, dtype=np.float64)
        rows.append(
            {
                "state_id": state.state_id,
                "scene_id": state.scene_id,
                "family": state.family,
                "selected_action_id": "NOT_APPLICABLE",
                "normalized_rank_regret": float(ranks.mean() / ranks.max()),
                "oracle_equivalent_selection_rate": float(
                    (ranks == ranks.min()).mean()
                ),
                "physical_target_progress_m": "NOT_APPLICABLE",
                "physical_path_length_m": "NOT_APPLICABLE",
            }
        )

    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "states": len(selected),
            "normalized_rank_regret": _mean(selected, "normalized_rank_regret"),
            "oracle_equivalent_selection_rate": _mean(
                selected, "oracle_equivalent_selection_rate"
            ),
            "physical_target_progress_m": "NOT_APPLICABLE",
            "physical_path_length_m": "NOT_APPLICABLE",
            "chosen_action_histogram": "NOT_APPLICABLE",
        }

    return {
        "selection_policy": "uniform_random_expectation_no_realized_action",
        "summary": summarize(rows),
        "group_results": rows,
        "per_family": {
            family: summarize([row for row in rows if row["family"] == family])
            for family in FAMILIES
        },
        "per_scene": [
            {
                "scene_id": scene,
                "family": next(
                    str(row["family"]) for row in rows if row["scene_id"] == scene
                ),
                **summarize([row for row in rows if row["scene_id"] == scene]),
            }
            for scene in sorted({str(row["scene_id"]) for row in rows})
        ],
    }


def paired_family_scene_cluster_comparison_replay_v1(
    candidate_results: Sequence[Mapping[str, Any]],
    baseline_results: Sequence[Mapping[str, Any]],
    *,
    field: str = "normalized_rank_regret",
    resamples: int = 10_000,
    seed: int = 2_026_080_302,
) -> dict[str, Any]:
    """Independent fixed family-stratified, scene-clustered bootstrap."""

    if resamples <= 0:
        raise DenseSharedReplayError("bootstrap resamples must be positive")
    candidate = {str(row["state_id"]): row for row in candidate_results}
    baseline = {str(row["state_id"]): row for row in baseline_results}
    if not candidate or set(candidate) != set(baseline):
        raise DenseSharedReplayError("paired state identities changed")
    by_scene: dict[tuple[str, str], list[float]] = {}
    for state_id in sorted(candidate):
        left = candidate[state_id]
        right = baseline[state_id]
        if left["scene_id"] != right["scene_id"] or left["family"] != right[
            "family"
        ]:
            raise DenseSharedReplayError("paired scene identity changed")
        delta = float(left[field]) - float(right[field])
        if not math.isfinite(delta):
            raise DenseSharedReplayError("paired metric is nonfinite")
        key = (str(left["family"]), str(left["scene_id"]))
        by_scene.setdefault(key, []).append(delta)
    by_family: dict[str, list[float]] = {family: [] for family in FAMILIES}
    for (family, _scene), values in sorted(by_scene.items()):
        if family not in by_family:
            raise DenseSharedReplayError("unexpected bootstrap family")
        by_family[family].append(float(np.mean(values)))
    if any(len(by_family[family]) != 2 for family in FAMILIES):
        raise DenseSharedReplayError("each family must have exactly two scenes")
    rng = np.random.default_rng(seed)
    draws = []
    family_points: dict[str, float] = {}
    for family in FAMILIES:
        values = np.asarray(by_family[family], dtype=np.float64)
        family_points[family] = float(values.mean())
        indices = rng.integers(0, len(values), size=(resamples, len(values)))
        draws.append(values[indices].mean(axis=1))
    samples = np.stack(draws, axis=1).mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "field": field,
        "direction": "candidate_minus_baseline_lower_is_better",
        "paired_states": len(candidate),
        "scene_clusters": len(by_scene),
        "family_strata": len(FAMILIES),
        "scenes_per_family": {
            family: len(by_family[family]) for family in FAMILIES
        },
        "resamples": resamples,
        "seed": seed,
        "mean_delta": float(np.mean(list(family_points.values()))),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "mean_delta_by_family": family_points,
    }


def _model_from_state_v1(
    state: object, *, device: torch.device
) -> DenseSharedSpatialReadoutV1:
    if not isinstance(state, Mapping) or not state:
        raise DenseSharedReplayError("model state changed")
    model = DenseSharedSpatialReadoutV1().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    if dense_shared_state_identity_v1(model.state_dict()) != (
        dense_shared_state_identity_v1(state)
    ):
        raise DenseSharedReplayError("loaded model state changed")
    return model


def _predict_members_v1(
    checkpoint: Mapping[str, Any],
    relations: torch.Tensor,
    conditions: torch.Tensor,
    *,
    state_key: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    members = checkpoint.get("members")
    if not isinstance(members, list) or len(members) != len(mechanism.MODEL_SEEDS):
        raise DenseSharedReplayError("checkpoint members changed")
    score_rows = []
    diagnostics = []
    identity_key = {
        "true_state": "true_identity_sha256",
        "current_state": "current_identity_sha256",
    }.get(state_key)
    training_key = {
        "true_state": "true_training",
        "current_state": "current_training",
    }.get(state_key)
    if identity_key is None or training_key is None:
        raise DenseSharedReplayError("checkpoint state selector changed")
    for expected_seed, member in zip(mechanism.MODEL_SEEDS, members, strict=True):
        if not isinstance(member, Mapping) or member.get("seed") != expected_seed:
            raise DenseSharedReplayError("checkpoint seed changed")
        selected_state = member.get(state_key)
        training = member.get(training_key)
        if not isinstance(selected_state, Mapping) or not isinstance(
            training, Mapping
        ):
            raise DenseSharedReplayError("checkpoint member state changed")
        actual_identity = dense_shared_state_identity_v1(selected_state)
        if (
            member.get(identity_key) != actual_identity
            or training.get("state_identity_sha256") != actual_identity
        ):
            raise DenseSharedReplayError("checkpoint member identity changed")
        model = _model_from_state_v1(selected_state, device=device)
        batches = []
        entropies = []
        with torch.no_grad():
            for start in range(0, mechanism.STATE_COUNT, mechanism.BATCH_STATES):
                stop = start + mechanism.BATCH_STATES
                output = model.forward_with_attention(
                    relations[start:stop]
                    .reshape(
                        -1,
                        mechanism.TOKEN_COUNT,
                        mechanism.RELATIONAL_DIMENSION,
                    )
                    .to(device),
                    conditions[start:stop].reshape(-1, 4).to(device),
                )
                batches.append(
                    output.score.reshape(
                        mechanism.BATCH_STATES, mechanism.ACTION_COUNT
                    ).cpu()
                )
                entropies.append(
                    (
                        -(
                            output.attention
                            * torch.log(output.attention.clamp_min(1.0e-12))
                        ).sum(dim=-1)
                    )
                    .div(math.log(mechanism.TOKEN_COUNT))
                    .cpu()
                )
        member_scores = torch.cat(batches).numpy().astype(np.float64)
        score_rows.append(member_scores)
        diagnostics.append(
            {
                "seed": expected_seed,
                "state_identity_sha256": actual_identity,
                "score_shape": [mechanism.STATE_COUNT, mechanism.ACTION_COUNT],
                "mean_normalized_attention_entropy": float(
                    torch.cat(entropies).mean()
                ),
                "score_sha256": hashlib.sha256(
                    np.ascontiguousarray(member_scores.astype("<f8")).tobytes()
                ).hexdigest(),
            }
        )
    stacked = np.stack(score_rows)
    ensemble = np.mean(stacked, axis=0)
    dispersion = np.std(stacked, axis=0, ddof=0)
    seed_argmins = np.argmin(stacked, axis=2)
    disagreements = np.any(seed_argmins != seed_argmins[:1], axis=0)
    if (
        stacked.shape
        != (
            len(mechanism.MODEL_SEEDS),
            mechanism.STATE_COUNT,
            mechanism.ACTION_COUNT,
        )
        or not np.isfinite(ensemble).all()
        or not np.isfinite(dispersion).all()
    ):
        raise DenseSharedReplayError("ensemble scores are invalid")
    return stacked, ensemble, {
        "members": diagnostics,
        "score_stack_shape": [
            len(mechanism.MODEL_SEEDS),
            mechanism.STATE_COUNT,
            mechanism.ACTION_COUNT,
        ],
        "ensemble_score_sha256": _score_identity_v1(ensemble),
        "seed_dispersion": {
            "definition": (
                "population_std_across_three_seed_scores_per_state_action"
            ),
            "mean_cell_population_std": float(dispersion.mean()),
            "maximum_cell_population_std": float(dispersion.max()),
            "states_with_seed_argmin_disagreement": int(disagreements.sum()),
            "state_seed_argmin_disagreement_rate": float(disagreements.mean()),
        },
    }


def _task_readout_report_v1(readouts: Any) -> dict[str, Any]:
    return {
        "identity_sha256": readouts.identity_sha256,
        "heads": [
            {
                "action_id": action,
                "identity_sha256": head.identity_sha256,
                "training_rows": head.training_rows,
                "feature_dimension": int(head.feature_mean.size),
                "ridge_lambda": head.ridge_lambda,
                "solver": head.solver,
            }
            for action, head in enumerate(readouts.heads)
        ],
    }


def _safety_support_v1(train_plan: Any, eval_plan: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "NOT_TESTABLE_ZERO_EVENT_SUPPORT",
        "applicable": False,
        "passed": False,
        "claim": "NOT_APPLICABLE",
    }
    for role, plan in (("train", train_plan), ("eval", eval_plan)):
        falls = sum(sum(state.physical_fell) for state in plan.states)
        tips = sum(sum(state.physical_tipped) for state in plan.states)
        if falls or tips:
            raise DenseSharedReplayError("zero-event safety support changed")
        result[role] = {
            "branches": len(plan.states) * mechanism.ACTION_COUNT,
            "falls": falls,
            "tips": tips,
        }
    return result


def _gates_v1(
    arms: Mapping[str, Mapping[str, Any]],
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    oracle = arms["privileged_physical_oracle"]["summary"]
    true_future = arms["dense_shared_true_future"]["summary"]
    random_expected = arms["random_expected"]["summary"]
    return {
        "2_privileged_physical_oracle": {
            "passed": oracle["normalized_rank_regret"] == 0.0
            and oracle["oracle_equivalent_selection_rate"] == 1.0,
            "normalized_rank_regret": oracle["normalized_rank_regret"],
            "oracle_equivalent_selection_rate": oracle[
                "oracle_equivalent_selection_rate"
            ],
        },
        "3_true_future_beats_task_action_only": {
            "passed": comparisons["true_future_vs_task_action_only"]["upper_95"]
            < 0.0,
            "measurement": comparisons["true_future_vs_task_action_only"],
        },
        "4_true_future_beats_current_state": {
            "passed": comparisons["true_future_vs_current_state"]["upper_95"]
            < 0.0,
            "measurement": comparisons["true_future_vs_current_state"],
        },
        "5_true_future_beats_relational_persistence": {
            "passed": comparisons["true_future_vs_relational_persistence"][
                "upper_95"
            ]
            < 0.0,
            "measurement": comparisons[
                "true_future_vs_relational_persistence"
            ],
        },
        "6_true_future_beats_random_expected": {
            "passed": true_future["normalized_rank_regret"]
            < random_expected["normalized_rank_regret"],
            "true_future": true_future["normalized_rank_regret"],
            "random_expected": random_expected["normalized_rank_regret"],
            "per_family_true_minus_random": {
                family: arms["dense_shared_true_future"]["per_family"][family][
                    "normalized_rank_regret"
                ]
                - arms["random_expected"]["per_family"][family][
                    "normalized_rank_regret"
                ]
                for family in FAMILIES
            },
        },
    }


def _verdict_v1(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    gates = evaluation.get("gates")
    if (
        evaluation.get("schema") != mechanism.SCHEMA
        or not isinstance(gates, Mapping)
        or set(gates) != mechanism.SCIENTIFIC_GATE_NAMES
        or any(
            not isinstance(gate, Mapping) or type(gate.get("passed")) is not bool
            for gate in gates.values()
        )
    ):
        raise DenseSharedReplayError("independent verdict inputs changed")
    all_scientific = all(bool(gate["passed"]) for gate in gates.values())
    if evaluation.get("scientific_gates_2_to_6_passed") is not all_scientific:
        raise DenseSharedReplayError("scientific gate aggregate changed")
    all_gates = {
        "1_infrastructure_and_custody": {"passed": True},
        **dict(gates),
        "7_deterministic_replay": {"passed": True},
    }
    return {
        "gates": all_gates,
        "passed": all_scientific,
        "terminal_status": (
            runner.PASS_STATUS if all_scientific else runner.STOP_STATUS
        ),
    }


def _evaluation_identity_v1(evaluation: Mapping[str, Any]) -> str:
    document = dict(evaluation)
    document.pop("replay_identity_sha256", None)
    return hashlib.sha256(runner.canonical_bytes_v1(document)).hexdigest()


def _score_identity_v1(scores: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(scores, dtype="<f8")).tobytes()
    ).hexdigest()


def _build_evaluation_v1(
    checkpoint: Mapping[str, Any],
    train_plan: Any,
    eval_plan: Any,
    eval_features: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    combined_plan_identity = hashlib.sha256(
        bytes.fromhex(train_plan.identity_sha256)
        + bytes.fromhex(eval_plan.identity_sha256)
    ).hexdigest()
    mechanism._require_plan_identities_v1(  # noqa: SLF001
        train=train_plan,
        eval_plan=eval_plan,
        combined_identity=combined_plan_identity,
    )
    projected_eval = mechanism.project_cache_v1(
        eval_features, checkpoint["pca"], label="eval"
    )
    task = mechanism._require_refitted_task_payload_v1(  # noqa: SLF001
        checkpoint["task_action_only"], train_plan
    )
    task_scores = mechanism.score_task_action_only_v1(eval_plan, task)
    true_relations, conditions = mechanism._dense_panels(  # noqa: SLF001
        eval_plan, projected_eval, successor_mode="true_future"
    )
    current_relations, current_conditions = mechanism._dense_panels(  # noqa: SLF001
        eval_plan, projected_eval, successor_mode="current_state"
    )
    if not torch.equal(conditions, current_conditions):
        raise DenseSharedReplayError("TRUE/CURRENT conditions changed")
    _true_members, true_residual, true_diagnostics = _predict_members_v1(
        checkpoint,
        true_relations,
        conditions,
        state_key="true_state",
        device=device,
    )
    _current_members, current_residual, current_diagnostics = _predict_members_v1(
        checkpoint,
        current_relations,
        conditions,
        state_key="current_state",
        device=device,
    )
    _persistence_members, persistence_residual, persistence_diagnostics = (
        _predict_members_v1(
            checkpoint,
            current_relations,
            conditions,
            state_key="true_state",
            device=device,
        )
    )
    true_scores = task_scores + true_residual
    current_scores = task_scores + current_residual
    persistence_scores = task_scores + persistence_residual
    oracle_scores = np.stack(
        [np.asarray(state.dense_ranks, dtype=np.float64) for state in eval_plan.states]
    )
    hold_scores = np.ones(
        (mechanism.STATE_COUNT, mechanism.ACTION_COUNT), dtype=np.float64
    )
    hold_scores[:, 6] = 0.0
    arms = {
        "privileged_physical_oracle": _selection_report_v1(
            eval_plan.groups, oracle_scores
        ),
        "dense_shared_true_future": _selection_report_v1(
            eval_plan.groups, true_scores
        ),
        "dense_shared_current_state": _selection_report_v1(
            eval_plan.groups, current_scores
        ),
        "task_action_only": _selection_report_v1(eval_plan.groups, task_scores),
        "dense_relational_persistence": _selection_report_v1(
            eval_plan.groups, persistence_scores
        ),
        "hold_constant": _selection_report_v1(eval_plan.groups, hold_scores),
        "random_expected": _random_expected_report_v1(eval_plan),
    }
    if (
        arms["task_action_only"]["summary"]["normalized_rank_regret"]
        != mechanism.EXPECTED_TASK_EVAL_REGRET
    ):
        raise DenseSharedReplayError("task/action-only evaluation changed")
    true_rows = arms["dense_shared_true_future"]["group_results"]
    comparisons = {
        name: paired_family_scene_cluster_comparison_replay_v1(
            true_rows, arms[baseline]["group_results"]
        )
        for name, baseline in (
            ("true_future_vs_task_action_only", "task_action_only"),
            ("true_future_vs_current_state", "dense_shared_current_state"),
            (
                "true_future_vs_relational_persistence",
                "dense_relational_persistence",
            ),
        )
    }
    gates = _gates_v1(arms, comparisons)
    member_training = [
        {
            "seed": member["seed"],
            "initial_identity_sha256": member["initial_identity_sha256"],
            "true_identity_sha256": member["true_identity_sha256"],
            "current_identity_sha256": member["current_identity_sha256"],
            "true_training": member["true_training"],
            "current_training": member["current_training"],
        }
        for member in checkpoint["members"]
    ]
    result: dict[str, Any] = {
        "schema": mechanism.SCHEMA,
        "status": "COMPLETE_MODEL_INDEPENDENT_EVALUATION",
        "claim_scope": "REUSED_DEVELOPMENT_ROLE_DENSE_ORACLE_FUTURE_CALIBRATION",
        "config": mechanism.config_v1(),
        "feature_plan": {
            "identity_sha256": combined_plan_identity,
            "train_identity_sha256": train_plan.identity_sha256,
            "eval_identity_sha256": eval_plan.identity_sha256,
            "states_per_role": mechanism.STATE_COUNT,
            "artifacts_per_role": runner.prior_evaluator.ROLE_ARTIFACT_COUNT,
        },
        "checkpoint_identity_sha256": checkpoint["identity_sha256"],
        "pca": {
            "identity_sha256": checkpoint["pca"]["identity_sha256"],
            "dimension": mechanism.PCA_DIMENSION,
            "row_count": mechanism.PCA_ROW_COUNT,
            "eigenvalues": [
                float(value) for value in checkpoint["pca"]["eigenvalues"].tolist()
            ],
        },
        "task_action_only_readout": _task_readout_report_v1(task),
        "member_training": member_training,
        "member_diagnostics": {
            "true_future": true_diagnostics,
            "current_state": current_diagnostics,
            "relational_persistence": persistence_diagnostics,
        },
        "score_evidence": {
            name: {
                "shape": [mechanism.STATE_COUNT, mechanism.ACTION_COUNT],
                "sha256_float64_c_order": _score_identity_v1(scores),
            }
            for name, scores in (
                ("dense_shared_true_future", true_scores),
                ("dense_shared_current_state", current_scores),
                ("task_action_only", task_scores),
                ("dense_relational_persistence", persistence_scores),
                ("privileged_physical_oracle", oracle_scores),
                ("hold_constant", hold_scores),
            )
        },
        "arms": arms,
        "paired_scene_cluster_comparisons": comparisons,
        "safety": _safety_support_v1(train_plan, eval_plan),
        "finiteness": {
            "pca": True,
            "training_diagnostics": True,
            "per_seed_scores": True,
            "ensemble_scores": True,
            "reported_metrics": True,
        },
        "gates": gates,
        "scientific_gates_2_to_6_passed": all(
            bool(gate["passed"]) for gate in gates.values()
        ),
    }
    result["replay_identity_sha256"] = _evaluation_identity_v1(result)
    runner.canonical_bytes_v1(result)
    return result


def _selected_actions(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        arm: [row["selected_action_id"] for row in report["group_results"]]
        for arm, report in evaluation["arms"].items()
    }


def _summaries(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        arm: {
            "summary": report["summary"],
            "per_family": report["per_family"],
            "per_scene": report["per_scene"],
        }
        for arm, report in evaluation["arms"].items()
    }


def _per_seed_score_evidence(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = evaluation["member_diagnostics"]
    return {
        arm: report["members"] for arm, report in diagnostics.items()
    }


def _ensemble_score_evidence(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = evaluation["member_diagnostics"]
    return {
        "residual_ensembles": {
            arm: {
                "score_stack_shape": report["score_stack_shape"],
                "ensemble_score_sha256": report["ensemble_score_sha256"],
                "seed_dispersion": report["seed_dispersion"],
            }
            for arm, report in diagnostics.items()
        },
        "total_arm_scores": evaluation["score_evidence"],
    }


def _checkpoint_reproduction_v1(
    fresh: Mapping[str, Any], stored: Mapping[str, Any]
) -> dict[str, bool]:
    fresh_members = fresh.get("members")
    stored_members = stored.get("members")
    if not isinstance(fresh_members, list) or not isinstance(stored_members, list):
        raise DenseSharedReplayError("checkpoint member inventory changed")
    if len(fresh_members) != len(mechanism.MODEL_SEEDS) or len(stored_members) != len(
        mechanism.MODEL_SEEDS
    ):
        raise DenseSharedReplayError("checkpoint member count changed")
    state_equal = True
    step_equal = True
    for expected_seed, left, right in zip(
        mechanism.MODEL_SEEDS, fresh_members, stored_members, strict=True
    ):
        if left.get("seed") != expected_seed or right.get("seed") != expected_seed:
            raise DenseSharedReplayError("checkpoint seed order changed")
        for state_key, identity_key in (
            ("true_state", "true_identity_sha256"),
            ("current_state", "current_identity_sha256"),
        ):
            state_equal = state_equal and _exact_tree_equal(
                left.get(state_key), right.get(state_key)
            )
            state_equal = state_equal and left.get(identity_key) == right.get(
                identity_key
            )
        for training_key in ("true_training", "current_training"):
            left_training = left.get(training_key)
            right_training = right.get(training_key)
            step_equal = step_equal and isinstance(left_training, Mapping)
            step_equal = step_equal and isinstance(right_training, Mapping)
            if isinstance(left_training, Mapping) and isinstance(
                right_training, Mapping
            ):
                step_equal = step_equal and (
                    left_training.get("optimizer_steps")
                    == right_training.get("optimizer_steps")
                    == mechanism.OPTIMIZER_STEPS
                )
    return {
        "pca_identity": _exact_tree_equal(fresh.get("pca"), stored.get("pca")),
        "state_dict_identities": state_equal,
        "step_counts": step_equal,
        "checkpoint_exact": _exact_tree_equal(fresh, stored),
    }


def execute_replay_v1(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    primary_evaluation: Mapping[str, Any],
    primary_evaluation_binding: Mapping[str, Any],
) -> dict[str, Any]:
    output_root = Path(str(authority["output_root"])).resolve()
    if output_root != runner.DEFAULT_OUTPUT_ROOT.resolve():
        raise DenseSharedReplayError("replay output root changed")
    if Path(str(checkpoint_binding["path"])).resolve() != (
        output_root / "pca_readout_checkpoint.pt"
    ):
        raise DenseSharedReplayError("checkpoint escaped the authorized attempt")
    if Path(str(primary_evaluation_binding["path"])).resolve() != (
        output_root / "evaluation.json"
    ):
        raise DenseSharedReplayError("primary evaluation escaped the authorized attempt")

    with scoped_replay_compatibility_admission_v1(
        authority, authority_binding=authority_binding
    ) as compatibility_state:
        bundle = runner.prior_runner.screen_data.load_bound_posthoc_bundle_v1()
    train_groups, _eval_groups, train_plan, eval_plan = runner._feature_plans_v1(  # noqa: SLF001
        bundle
    )
    train_features, _train_receipt = runner._load_train_cache_v1(  # noqa: SLF001
        bundle, train_plan
    )
    device = runner._authorized_device_v1()  # noqa: SLF001
    implementation_source_binding = authority["source_bindings"][
        "dense_shared_evaluator"
    ]
    mechanism.validate_checkpoint_v1(
        checkpoint,
        train_plan=train_plan,
        implementation_source_binding=implementation_source_binding,
    )
    fresh_checkpoint = mechanism.fit_primary_checkpoint_v1(
        train_groups,
        train_features,
        device,
        implementation_source_binding=implementation_source_binding,
    )
    checkpoint_checks = _checkpoint_reproduction_v1(fresh_checkpoint, checkpoint)
    if not all(checkpoint_checks.values()):
        raise DenseSharedReplayError("fresh checkpoint did not reproduce exactly")
    eval_features, _eval_receipt = runner._load_eval_cache_v1(  # noqa: SLF001
        authority, eval_plan
    )
    recomputed = _build_evaluation_v1(
        fresh_checkpoint,
        train_plan,
        eval_plan,
        eval_features,
        device,
    )

    per_seed_scores = _canonical_equal(
        _per_seed_score_evidence(recomputed),
        _per_seed_score_evidence(primary_evaluation),
    )
    ensemble_scores = _canonical_equal(
        _ensemble_score_evidence(recomputed),
        _ensemble_score_evidence(primary_evaluation),
    )
    exact_evaluation = _canonical_equal(recomputed, primary_evaluation)
    reproduction = {
        "pca_identity": checkpoint_checks["pca_identity"],
        "state_dict_identities": checkpoint_checks["state_dict_identities"],
        "step_counts": checkpoint_checks["step_counts"],
        "per_seed_scores": per_seed_scores,
        "ensemble_scores": ensemble_scores,
        "selected_actions": _canonical_equal(
            _selected_actions(recomputed), _selected_actions(primary_evaluation)
        ),
        "summaries": _canonical_equal(
            _summaries(recomputed), _summaries(primary_evaluation)
        ),
        "bootstrap_intervals": _canonical_equal(
            recomputed["paired_scene_cluster_comparisons"],
            primary_evaluation["paired_scene_cluster_comparisons"],
        ),
        "gates": _canonical_equal(
            recomputed["gates"], primary_evaluation["gates"]
        ),
        "verdict": _canonical_equal(
            _verdict_v1(recomputed), _verdict_v1(primary_evaluation)
        ),
        "exactly_reproduced": exact_evaluation,
    }
    if set(reproduction) != runner.REPLAY_REPRODUCTION_FIELDS or any(
        value is not True for value in reproduction.values()
    ):
        failed = sorted(name for name, value in reproduction.items() if not value)
        raise DenseSharedReplayError(
            f"fresh replay reproduction checks failed: {failed}"
        )
    receipt_binding = compatibility_state.get("receipt_binding")
    if not isinstance(receipt_binding, Mapping):
        raise DenseSharedReplayError("replay compatibility receipt binding is absent")
    if runner.file_binding_v1(Path(str(checkpoint_binding["path"]))) != dict(
        checkpoint_binding
    ):
        raise DenseSharedReplayError("checkpoint changed during replay")
    if runner.file_binding_v1(Path(str(primary_evaluation_binding["path"]))) != dict(
        primary_evaluation_binding
    ):
        raise DenseSharedReplayError("primary evaluation changed during replay")
    runner._execution_bindings_unchanged(  # noqa: SLF001
        authority, authority_binding=authority_binding
    )
    report = {
        "schema": runner.REPLAY_SCHEMA,
        "status": runner.REPLAY_STATUS,
        "citable_as_scientific_evidence": False,
        "authority_binding": dict(authority_binding),
        "checkpoint_binding": dict(checkpoint_binding),
        "primary_evaluation_binding": dict(primary_evaluation_binding),
        "compatibility_receipt_binding": dict(receipt_binding),
        "recomputed_evaluation": recomputed,
        "reproduction": reproduction,
        "protected_material_opened": False,
        "rgb_access": {"train": 0, "eval": 0},
    }
    runner.canonical_bytes_v1(report)
    runner._write_json_exclusive(output_root / "replay.json", report)  # noqa: SLF001
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-byte-count", type=int, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--expected-evaluation-sha256", required=True)
    parser.add_argument("--expected-evaluation-byte-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    authority, authority_binding = runner._read_authority(  # noqa: SLF001
        args.authority,
        expected_sha256=args.expected_authority_sha256,
        expected_byte_count=args.expected_authority_byte_count,
    )
    output_root = Path(str(authority["output_root"])).resolve()
    if args.checkpoint.resolve() != output_root / "pca_readout_checkpoint.pt":
        raise DenseSharedReplayError("checkpoint escaped the authorized attempt")
    if args.evaluation.resolve() != output_root / "evaluation.json":
        raise DenseSharedReplayError("primary evaluation escaped the authorized attempt")
    checkpoint, checkpoint_binding = _read_checkpoint_v1(
        args.checkpoint,
        expected_sha256=args.expected_checkpoint_sha256,
        expected_byte_count=args.expected_checkpoint_byte_count,
    )
    evaluation, evaluation_binding = runner._read_bound_json(  # noqa: SLF001
        args.evaluation,
        expected_sha256=args.expected_evaluation_sha256,
        expected_byte_count=args.expected_evaluation_byte_count,
        label="primary evaluation",
    )
    report = execute_replay_v1(
        authority,
        authority_binding=authority_binding,
        checkpoint=checkpoint,
        checkpoint_binding=checkpoint_binding,
        primary_evaluation=evaluation,
        primary_evaluation_binding=evaluation_binding,
    )
    print(runner.canonical_bytes_v1({"status": report["status"]}).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DenseSharedReplayError",
    "build_parser",
    "execute_replay_v1",
    "paired_family_scene_cluster_comparison_replay_v1",
    "scoped_replay_compatibility_admission_v1",
]
