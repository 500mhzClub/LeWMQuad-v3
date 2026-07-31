#!/usr/bin/env python3
"""Development-only scene-disjoint counterfactual action-regret evaluator.

The pure rank, ridge, and clustered-comparison functions in this module do not
import Torch or open RGB/checkpoint files.  Runtime imports and payload access
occur only after the exact pilot manifest has passed the strict receipt-only
consumer.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.datasets.go2_world_model_counterfactual_pilot_v1 import (  # noqa: E402
    ACTION_COUNT,
    CounterfactualGroupV1,
    load_bound_pilot_v1,
    read_bound_rgb_bytes_v1,
)

MASK_ROLE = "val"
MASK_ROW_INDICES = (0, 1, 2, 3)
ARM_NAMES = (
    "forecast",
    "current_state_action",
    "task_action_only",
    "hold_blind",
    "shuffled",
    "true_future_ceiling",
)
HOLD_ACTION_ID = 6


@dataclass(frozen=True)
class RidgeReadoutV1:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    ridge_lambda: float
    training_rows: int
    solver: str
    identity_sha256: str


@dataclass(frozen=True)
class ActionSpecificRidgeReadoutsV1:
    heads: tuple[RidgeReadoutV1, ...]
    identity_sha256: str


def _finite_matrix(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError(f"{name} must be a nonempty two-dimensional matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(array)


def fit_ridge_readout_v1(
    train_features: object,
    train_targets: object,
    *,
    ridge_lambda: float,
) -> RidgeReadoutV1:
    """Fit a deterministic, standardized, train-only ridge readout."""
    features = _finite_matrix(train_features, name="train features")
    targets = np.asarray(train_targets, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[:, None]
    if (
        targets.ndim != 2
        or targets.shape[0] != features.shape[0]
        or not targets.shape[1]
        or not np.isfinite(targets).all()
    ):
        raise ValueError("train targets must be finite and row-aligned")
    if not math.isfinite(ridge_lambda) or ridge_lambda <= 0.0:
        raise ValueError("ridge_lambda must be positive and finite")
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 0.0, scale, 1.0)
    standardized = (features - mean) / scale
    if standardized.shape[1] > standardized.shape[0]:
        solver = "dual"
        target_mean = targets.mean(axis=0, keepdims=True)
        centered_targets = targets - target_mean
        gram = (
            standardized @ standardized.T
            + np.eye(standardized.shape[0], dtype=np.float64) * ridge_lambda
        )
        try:
            dual = np.linalg.solve(gram, centered_targets)
        except np.linalg.LinAlgError:
            dual = np.linalg.pinv(gram, hermitian=True) @ centered_targets
        weights = standardized.T @ dual
        coefficients = np.concatenate([target_mean, weights], axis=0)
    else:
        solver = "primal"
        design = np.concatenate(
            [np.ones((standardized.shape[0], 1)), standardized], axis=1
        )
        penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_lambda
        penalty[0, 0] = 0.0
        gram = design.T @ design + penalty
        rhs = design.T @ targets
        try:
            coefficients = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(gram, hermitian=True) @ rhs
    digest = hashlib.sha256()
    for array in (mean, scale, coefficients):
        canonical = np.ascontiguousarray(array.astype("<f8", copy=False))
        digest.update(str(canonical.shape).encode("ascii") + b"\0")
        digest.update(canonical.tobytes())
    digest.update(np.asarray([ridge_lambda], dtype="<f8").tobytes())
    digest.update(solver.encode("ascii"))
    return RidgeReadoutV1(
        feature_mean=mean,
        feature_scale=scale,
        coefficients=coefficients,
        ridge_lambda=float(ridge_lambda),
        training_rows=int(features.shape[0]),
        solver=solver,
        identity_sha256=digest.hexdigest(),
    )


def predict_ridge_readout_v1(readout: RidgeReadoutV1, features: object) -> np.ndarray:
    matrix = _finite_matrix(features, name="readout features")
    if matrix.shape[1] != readout.feature_mean.shape[0]:
        raise ValueError("readout feature dimension changed")
    standardized = (matrix - readout.feature_mean) / readout.feature_scale
    design = np.concatenate([np.ones((matrix.shape[0], 1)), standardized], axis=1)
    result = design @ readout.coefficients
    return result[:, 0] if result.shape[1] == 1 else result


def fit_action_specific_ridge_readouts_v1(
    features_by_action: Sequence[object],
    targets_by_action: Sequence[object],
    *,
    ridge_lambda: float,
) -> ActionSpecificRidgeReadoutsV1:
    """Fit nine independent train-only heads; actions never share an intercept."""
    if (
        len(features_by_action) != ACTION_COUNT
        or len(targets_by_action) != ACTION_COUNT
    ):
        raise ValueError("action-specific readout requires exactly nine heads")
    heads = tuple(
        fit_ridge_readout_v1(
            features_by_action[action_id],
            targets_by_action[action_id],
            ridge_lambda=ridge_lambda,
        )
        for action_id in range(ACTION_COUNT)
    )
    digest = hashlib.sha256()
    for action_id, head in enumerate(heads):
        digest.update(action_id.to_bytes(2, "little"))
        digest.update(bytes.fromhex(head.identity_sha256))
    return ActionSpecificRidgeReadoutsV1(
        heads=heads,
        identity_sha256=digest.hexdigest(),
    )


def predict_action_specific_scores_v1(
    readouts: ActionSpecificRidgeReadoutsV1,
    action_features: Sequence[object],
) -> np.ndarray:
    if len(readouts.heads) != ACTION_COUNT or len(action_features) != ACTION_COUNT:
        raise ValueError("action-specific scoring requires exactly nine actions")
    scores = []
    for action_id, (head, feature) in enumerate(
        zip(readouts.heads, action_features, strict=True)
    ):
        prediction = np.asarray(
            predict_ridge_readout_v1(head, np.asarray(feature).reshape(1, -1)),
            dtype=np.float64,
        ).reshape(-1)
        if prediction.shape != (1,):
            raise ValueError(f"action head {action_id} did not return one score")
        scores.append(float(prediction[0]))
    return np.asarray(scores, dtype=np.float64)


def masked_token_descriptor_v1(masked_tokens: object) -> np.ndarray:
    """Mean/std descriptor with the exact four-mask feature contract."""
    tokens = np.asarray(masked_tokens, dtype=np.float64)
    if tokens.ndim != 3 or tokens.shape[0] != len(MASK_ROW_INDICES):
        raise ValueError("masked tokens must have shape (4,tokens,dimensions)")
    if not tokens.shape[1] or not tokens.shape[2] or not np.isfinite(tokens).all():
        raise ValueError("masked tokens must be nonempty and finite")
    descriptors = [
        np.concatenate([tokens[index].mean(axis=0), tokens[index].std(axis=0)])
        for index in range(len(MASK_ROW_INDICES))
    ]
    return np.concatenate(descriptors).astype(np.float64, copy=False)


def task_conditioned_feature_v1(
    latent_descriptor: object | None,
    *,
    relative_target_xy_body_m: Sequence[float],
) -> np.ndarray:
    if len(relative_target_xy_body_m) != 2:
        raise ValueError("task target must have two coordinates")
    task = np.asarray(
        [relative_target_xy_body_m[0], relative_target_xy_body_m[1], 1.0],
        dtype=np.float64,
    )
    if not np.isfinite(task).all():
        raise ValueError("task feature is invalid")
    if latent_descriptor is None:
        return task
    latent = np.asarray(latent_descriptor, dtype=np.float64).reshape(-1)
    if not latent.size or not np.isfinite(latent).all():
        raise ValueError("latent descriptor must be nonempty and finite")
    return np.concatenate(
        [latent, task, latent * task[0], latent * task[1]]
    )


def _group_id(group: CounterfactualGroupV1 | Mapping[str, object]) -> str:
    return (
        group.state_id
        if isinstance(group, CounterfactualGroupV1)
        else str(group["state_id"])
    )


def selection_metrics_v1(
    groups: Sequence[CounterfactualGroupV1],
    scores_by_state: Mapping[str, Sequence[float]],
) -> dict[str, object]:
    """Select minimum predicted regret and score against physical dense ranks."""
    results: list[dict[str, object]] = []
    for group in groups:
        scores = np.asarray(scores_by_state.get(group.state_id), dtype=np.float64)
        if scores.shape != (ACTION_COUNT,) or not np.isfinite(scores).all():
            raise ValueError(f"missing/invalid nine-action scores for {group.state_id}")
        selected = min(range(ACTION_COUNT), key=lambda action: (scores[action], action))
        ranks = np.asarray(
            [branch.oracle_dense_rank for branch in group.branches], dtype=np.float64
        )
        denominator = max(1.0, float(ranks.max()))
        oracle_action = min(
            range(ACTION_COUNT), key=lambda action: (ranks[action], action)
        )
        labels = group.branches[selected].labels
        oracle_labels = group.branches[oracle_action].labels
        results.append({
            "state_id": group.state_id,
            "scene_id": group.scene_id,
            "family": group.family,
            "selected_action_id": selected,
            "oracle_action_id": oracle_action,
            "selected_dense_rank": int(ranks[selected]),
            "normalized_rank_regret": float(ranks[selected] / denominator),
            "random_expected_normalized_rank_regret": float(ranks.mean() / denominator),
            "oracle_match": bool(ranks[selected] == ranks.min()),
            "physical_fell": labels.fell,
            "physical_tipped": labels.tipped,
            "physical_target_progress_m": labels.target_progress_m,
            "physical_path_length_m": labels.path_length_m,
            "physical_progress_delta_to_canonical_oracle_m": (
                oracle_labels.target_progress_m - labels.target_progress_m
            ),
            "planar_clearance_proxy_min_m": labels.planar_clearance_proxy_min_m,
            "grid_recoverability_proxy": labels.grid_recoverability_proxy,
        })
    if not results:
        raise ValueError("at least one evaluation group is required")
    numeric = lambda key: float(np.mean([float(row[key]) for row in results]))
    summary = {
        "groups": len(results),
        "scenes": len({row["scene_id"] for row in results}),
        "normalized_rank_regret": numeric("normalized_rank_regret"),
        "random_expected_normalized_rank_regret": numeric(
            "random_expected_normalized_rank_regret"
        ),
        "oracle_match_rate": numeric("oracle_match"),
        "physical_fall_rate": numeric("physical_fell"),
        "physical_tip_rate": numeric("physical_tipped"),
        "physical_target_progress_m": numeric("physical_target_progress_m"),
        "physical_progress_delta_to_canonical_oracle_m": numeric(
            "physical_progress_delta_to_canonical_oracle_m"
        ),
        "physical_path_length_m": numeric("physical_path_length_m"),
    }
    proxy_summary = {}
    for key in ("planar_clearance_proxy_min_m", "grid_recoverability_proxy"):
        values = [row[key] for row in results]
        if all(value is not None for value in values):
            proxy_summary[key] = float(np.mean([float(value) for value in values]))
    if proxy_summary:
        summary["nonphysical_proxy_metrics"] = proxy_summary
    return {"summary": summary, "group_results": results}


def paired_scene_cluster_comparison_v1(
    candidate_results: Sequence[Mapping[str, object]],
    baseline_results: Sequence[Mapping[str, object]],
    *,
    field: str = "normalized_rank_regret",
    resamples: int = 2000,
    seed: int = 20260731,
) -> dict[str, object]:
    """Paired candidate-minus-baseline comparison, resampling whole scenes."""
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    candidate = {str(row["state_id"]): row for row in candidate_results}
    baseline = {str(row["state_id"]): row for row in baseline_results}
    if not candidate or set(candidate) != set(baseline):
        raise ValueError("paired comparisons require identical state identities")
    by_scene: dict[str, list[float]] = defaultdict(list)
    for state_id in sorted(candidate):
        c_row, b_row = candidate[state_id], baseline[state_id]
        if c_row["scene_id"] != b_row["scene_id"]:
            raise ValueError("paired state changed scene identity")
        delta = float(c_row[field]) - float(b_row[field])
        if not math.isfinite(delta):
            raise ValueError("paired metric must be finite")
        by_scene[str(c_row["scene_id"])].append(delta)
    scenes = sorted(by_scene)
    scene_means = np.asarray(
        [np.mean(by_scene[scene]) for scene in scenes], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(scenes), size=(resamples, len(scenes)))
    samples = scene_means[draws].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "field": field,
        "direction": "candidate_minus_baseline_lower_is_better",
        "paired_states": len(candidate),
        "scene_clusters": len(scenes),
        "resamples": resamples,
        "seed": seed,
        "mean_delta": float(scene_means.mean()),
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def evaluator_verdict_v1(comparisons: Mapping[str, Mapping[str, object]]) -> str:
    required = {
        "ceiling_vs_current",
        "forecast_vs_current",
        "forecast_vs_task_action",
        "forecast_vs_hold_blind",
        "forecast_vs_shuffled",
        "forecast_vs_random",
    }
    if set(comparisons) != required:
        raise ValueError("evaluator comparison set changed")
    if float(comparisons["ceiling_vs_current"]["upper_95"]) >= 0.0:
        return "EVALUATOR_SENSITIVITY_NOT_ESTABLISHED"
    if all(
        float(comparisons[name]["upper_95"]) < 0.0
        for name in required - {"ceiling_vs_current"}
    ):
        return "SCENE_DISJOINT_TASK_UTILITY_DEMONSTRATED_DEVELOPMENT_ONLY"
    return "SCENE_DISJOINT_TASK_UTILITY_NOT_ESTABLISHED"


def _masked_descriptor_from_calls(callable_for_mask) -> np.ndarray:
    rows = []
    for mask_row in MASK_ROW_INDICES:
        value = np.asarray(callable_for_mask(mask_row), dtype=np.float64)
        if value.ndim != 2:
            raise ValueError("one-mask token field must be two-dimensional")
        rows.append(value)
    return masked_token_descriptor_v1(np.stack(rows))


def _extract_features(bundle, model, device):
    """Lazy Torch path; returns role/arm/state/nine feature vectors."""
    import torch
    from lewm.benchmarks import (
        go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as mask_contract,
    )
    from scripts import dev_probe_counterfactual_action_fidelity as probe
    from scripts import evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as eval_api

    masks = {
        row: mask_contract.batched_mask_indices(
            MASK_ROLE, [row], device=device
        )[0]
        for row in MASK_ROW_INDICES
    }
    result: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {}
    with torch.no_grad():
        for role in ("train", "eval"):
            role_result: dict[str, dict[str, list[np.ndarray]]] = {
                arm: {} for arm in ARM_NAMES
            }
            for group in bundle.groups_by_role[role]:
                context_images = torch.stack([
                    probe.decode(
                        artifact_id, device, pilot_bundle=bundle
                    )
                    for artifact_id in group.context_rgb_artifact_ids
                ]).unsqueeze(0)
                target_images = [
                    probe.decode(
                        branch.target_rgb_artifact_id,
                        device,
                        pilot_bundle=bundle,
                    )
                    for branch in group.branches
                ]
                current = context_images[0, -1].unsqueeze(0)
                current_descriptor = _masked_descriptor_from_calls(
                    lambda row: eval_api._target_tokens(
                        model, current, masks[row]
                    )[0].detach().cpu().numpy()
                )
                true_descriptors = [
                    _masked_descriptor_from_calls(
                        lambda row, image=image: eval_api._target_tokens(
                            model, image.unsqueeze(0), masks[row]
                        )[0].detach().cpu().numpy()
                    )
                    for image in target_images
                ]

                def forecast_descriptor(used_action: int) -> np.ndarray:
                    sequence = torch.tensor(
                        [[*group.history_action_ids, used_action]],
                        dtype=torch.long,
                        device=device,
                    )
                    return _masked_descriptor_from_calls(
                        lambda row: eval_api._predict_future(
                            model, context_images, sequence, masks[row]
                        ).prediction[0].detach().cpu().numpy()
                    )

                factual = [forecast_descriptor(action) for action in range(ACTION_COUNT)]
                hold = forecast_descriptor(HOLD_ACTION_ID)
                shuffled = [
                    forecast_descriptor((action + 1) % ACTION_COUNT)
                    for action in range(ACTION_COUNT)
                ]
                arm_latents = {
                    "forecast": factual,
                    "current_state_action": [current_descriptor] * ACTION_COUNT,
                    "task_action_only": [None] * ACTION_COUNT,
                    "hold_blind": [hold] * ACTION_COUNT,
                    "shuffled": shuffled,
                    "true_future_ceiling": true_descriptors,
                }
                for arm, latents in arm_latents.items():
                    role_result[arm][group.state_id] = [
                        task_conditioned_feature_v1(
                            latents[action],
                            relative_target_xy_body_m=(
                                group.relative_target_xy_body_m
                            ),
                        )
                        for action in range(ACTION_COUNT)
                    ]
            result[role] = role_result
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--expected-pilot-manifest-byte-count", type=int, required=True)
    parser.add_argument("--expected-pilot-manifest-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-update", type=int, required=True)
    parser.add_argument("--ridge-lambda", type=float, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(".generated/dev/counterfactual/action_regret_v1.json"),
    )
    args = parser.parse_args()
    for value, label in (
        (args.expected_pilot_manifest_sha256, "pilot manifest"),
        (args.expected_checkpoint_sha256, "checkpoint"),
    ):
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"expected {label} SHA-256 must be lowercase hex")
    if args.expected_update < 0:
        raise ValueError("expected-update must be non-negative")
    if not math.isfinite(args.ridge_lambda) or args.ridge_lambda <= 0.0:
        raise ValueError("ridge-lambda must be positive and finite")
    if args.bootstrap_resamples <= 0:
        raise ValueError("bootstrap-resamples must be positive")
    if args.checkpoint.name == "latest.pt":
        raise ValueError("mutable latest.pt checkpoints are forbidden")
    bundle = load_bound_pilot_v1(
        args.pilot_root,
        expected_manifest_byte_count=args.expected_pilot_manifest_byte_count,
        expected_manifest_sha256=args.expected_pilot_manifest_sha256,
    )
    if not bundle.groups_by_role["train"] or not bundle.groups_by_role["eval"]:
        raise ValueError("nonempty train and scene-disjoint eval roles are required")

    import torch
    from lewm.datasets import go2_world_model_counterfactual_pilot_v1 as pilot_consumer
    from scripts import dev_probe_counterfactual_action_fidelity as probe

    output = probe.require_development_output(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite evaluator output: {output}")
    checkpoint = probe.require_development_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model, label, model_identity = probe.build_model(
        checkpoint,
        device,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_update=args.expected_update,
    )
    source_bindings = [
        probe.file_binding(Path(path))
        for path in (
            __file__,
            pilot_consumer.__file__,
            probe.__file__,
            probe.model_module.__file__,
            probe.evaluation.__file__,
            probe.metrics.__file__,
            probe.h6.__file__,
            probe.trainer.__file__,
        )
    ]
    features = _extract_features(bundle, model, device)
    readouts: dict[str, ActionSpecificRidgeReadoutsV1] = {}
    arm_reports: dict[str, dict[str, object]] = {}
    for arm in ARM_NAMES:
        train_x = [[] for _ in range(ACTION_COUNT)]
        train_y = [[] for _ in range(ACTION_COUNT)]
        for group in bundle.groups_by_role["train"]:
            ranks = [branch.oracle_dense_rank for branch in group.branches]
            denominator = max(1, max(ranks))
            for action_id in range(ACTION_COUNT):
                train_x[action_id].append(
                    features["train"][arm][group.state_id][action_id]
                )
                train_y[action_id].append(ranks[action_id] / denominator)
        readout = fit_action_specific_ridge_readouts_v1(
            [np.stack(rows) for rows in train_x],
            train_y,
            ridge_lambda=args.ridge_lambda,
        )
        readouts[arm] = readout
        scores = {
            group.state_id: predict_action_specific_scores_v1(
                readout, features["eval"][arm][group.state_id]
            ).tolist()
            for group in bundle.groups_by_role["eval"]
        }
        arm_reports[arm] = selection_metrics_v1(
            bundle.groups_by_role["eval"], scores
        )
        arm_reports[arm]["readout"] = {
            "identity_sha256": readout.identity_sha256,
            "action_specific_heads": ACTION_COUNT,
            "heads": [
                {
                    "action_id": action_id,
                    "identity_sha256": head.identity_sha256,
                    "training_rows": head.training_rows,
                    "feature_dimension": int(head.feature_mean.size),
                    "ridge_lambda": head.ridge_lambda,
                    "solver": head.solver,
                }
                for action_id, head in enumerate(readout.heads)
            ],
        }

    forecast_rows = arm_reports["forecast"]["group_results"]
    random_rows = [
        {
            **row,
            "normalized_rank_regret": row[
                "random_expected_normalized_rank_regret"
            ],
        }
        for row in forecast_rows
    ]
    baselines = {
        "ceiling_vs_current": ("true_future_ceiling", "current_state_action"),
        "forecast_vs_current": ("forecast", "current_state_action"),
        "forecast_vs_task_action": ("forecast", "task_action_only"),
        "forecast_vs_hold_blind": ("forecast", "hold_blind"),
        "forecast_vs_shuffled": ("forecast", "shuffled"),
    }
    comparisons = {
        name: paired_scene_cluster_comparison_v1(
            arm_reports[candidate]["group_results"],
            arm_reports[baseline]["group_results"],
            resamples=args.bootstrap_resamples,
            seed=args.bootstrap_seed,
        )
        for name, (candidate, baseline) in baselines.items()
    }
    comparisons["forecast_vs_random"] = paired_scene_cluster_comparison_v1(
        forecast_rows,
        random_rows,
        resamples=args.bootstrap_resamples,
        seed=args.bootstrap_seed,
    )
    report = {
        "schema": "lewm_go2_world_model_counterfactual_action_regret_v1",
        "status": "COMPLETE",
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "pilot_manifest_binding": dict(bundle.manifest_binding),
        "checkpoint": str(checkpoint),
        "model_label": label,
        "model_identity": model_identity,
        "source_bindings": source_bindings,
        "feature_contract": {
            "mask_role": MASK_ROLE,
            "mask_row_indices": list(MASK_ROW_INDICES),
            "token_statistics": ["mean", "std"],
            "task_fields": ["target_x_body_m", "target_y_body_m", "target_present"],
            "latent_task_interactions": ["latent_times_target_x", "latent_times_target_y"],
            "action_specific_ridge_heads": ACTION_COUNT,
            "shared_action_one_hot": False,
            "train_only_readout": True,
        },
        "arms": arm_reports,
        "paired_scene_cluster_comparisons": comparisons,
        "verdict": evaluator_verdict_v1(comparisons),
    }
    reloaded = load_bound_pilot_v1(
        args.pilot_root,
        expected_manifest_byte_count=args.expected_pilot_manifest_byte_count,
        expected_manifest_sha256=args.expected_pilot_manifest_sha256,
    )
    if (
        dict(reloaded.manifest_binding) != dict(bundle.manifest_binding)
        or dict(reloaded.rgb_manifest_binding) != dict(bundle.rgb_manifest_binding)
        or {
            role: dict(reloaded.role_bindings[role]) for role in ("train", "eval")
        }
        != {role: dict(bundle.role_bindings[role]) for role in ("train", "eval")}
    ):
        raise RuntimeError("pilot receipts changed during evaluation")
    for artifact_id in sorted(reloaded.artifacts):
        read_bound_rgb_bytes_v1(reloaded, artifact_id)
    probe.assert_file_bindings_unchanged(
        source_bindings, kind="counterfactual action-regret evaluator source"
    )
    probe.write_json_atomic(output, report)
    print(json.dumps({"verdict": report["verdict"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "ActionSpecificRidgeReadoutsV1",
    "MASK_ROW_INDICES",
    "RidgeReadoutV1",
    "evaluator_verdict_v1",
    "fit_action_specific_ridge_readouts_v1",
    "fit_ridge_readout_v1",
    "masked_token_descriptor_v1",
    "paired_scene_cluster_comparison_v1",
    "predict_ridge_readout_v1",
    "predict_action_specific_scores_v1",
    "selection_metrics_v1",
    "task_conditioned_feature_v1",
]
