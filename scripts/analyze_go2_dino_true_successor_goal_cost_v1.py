#!/usr/bin/env python3
"""Validate and analyze the preregistered DINO true-successor ceiling.

The analyzer reads only the supplied benchmark JSON.  It does not discover or
open scene inputs, load a model checkpoint, or run DINO.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "lewm_go2_dino_true_successor_goal_cost_analysis_v1"
INPUT_SCHEMA = "lewm_closed_loop_mpc_benchmark_v0"
EXPECTED_POLICIES = (
    "oracle_mpc",
    "dino_true_successor",
    "dino_true_successor_shuffled",
    "dino_persistence",
    "bearing",
    "hold",
    "random",
)
EXPECTED_PRIMITIVES = (
    "hold",
    "forward_slow",
    "forward_medium",
    "forward_fast",
    "arc_left",
    "arc_right",
    "yaw_left",
    "yaw_right",
    "backward",
)
EXPECTED_CANDIDATE_SEQUENCES = tuple((name,) for name in EXPECTED_PRIMITIVES)
EXPECTED_SCENE_COUNT = 24
EXPECTED_SCENE_CORPUS_SUFFIX = PurePosixPath(
    ".generated/scene_corpus/go2_generalization_v4"
)
EXPECTED_FAMILY = "go2_deployment_medium_maze"
EXPECTED_SPLIT = "development"
EXPECTED_HORIZON = 1
EXPECTED_TRIALS_PER_SCENE = 1
EXPECTED_MAX_BLOCKS = 12
EXPECTED_SEED = 7
EXPECTED_GOAL_RADIUS_M = 0.35
EXPECTED_GOAL_STANDOFF_M = 0.85
EXPECTED_BEACON_APPROACH_DISTANCE_M = 1.2
EXPECTED_BEACON_START_YAW_JITTER_RAD = 0.7
EXPECTED_GRID_CELL_SIZE_M = 0.05
EXPECTED_GRID_INFLATION_M = 0.20
ORACLE_TIE_TOLERANCE_M = 1.0e-9
SHUFFLE_MIX_CONSTANT = 0x9E3779B97F4A7C15

DINO_ASSAY_NAME = "frozen_dinov2_true_successor_goal_cost"
DINO_ASSAY_VERSION = 1
DINO_ENCODER_NAME = "dinov2_vits14"
DINO_REPOSITORY_PATH = (
    "/home/andrewknowles/.cache/"
    "dinov2-7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)
DINO_REPOSITORY_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINO_CHECKPOINT_PATH = (
    "/home/andrewknowles/.cache/torch/hub/checkpoints/"
    "dinov2_vits14_pretrain.pth"
)
DINO_CHECKPOINT_BYTES = 88_283_115
DINO_CHECKPOINT_SHA256 = (
    "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
)
DINO_INPUT_RGB_SHAPE = (3, 224, 224)
DINO_IMAGENET_MEAN = (0.485, 0.456, 0.406)
DINO_IMAGENET_STD = (0.229, 0.224, 0.225)
DINO_PATCH_OUTPUT_SHAPE = (256, 384)
DINO_TOKEN_NORMALIZATION = "per_patch_l2"
DINO_COST_DEFINITION = (
    "mean_j(1-dot(l2_normalize(successor_patch_j),"
    "l2_normalize(single_goal_patch_j)))"
)
DINO_SUCCESSOR_EVALUATION = (
    "reset_observed_pose_execute_one_nominal_kinematic_candidate_render_actual_"
    "successor_restore_observed_pose"
)

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 2_026_080_402
BOOTSTRAP_PERCENTILES = (2.5, 97.5)

METRICS = (
    "progress_m",
    "final_distance_m",
    "success",
    "path_efficiency",
    "scene_mean_oracle_first_action_regret_m",
)
DINO_POLICIES = frozenset(
    {
        "dino_true_successor",
        "dino_true_successor_shuffled",
        "dino_persistence",
    }
)


class DinoCeilingAnalysisError(ValueError):
    """Raised when an input cannot support the preregistered analysis."""


def _path_parts(path: Path | str) -> tuple[str, ...]:
    raw = str(path).replace("\\", "/")
    return tuple(part.lower() for part in PurePosixPath(raw).parts)


def _is_sealed_path(path: Path | str) -> bool:
    return any(
        part == "sealed_test.json" or part == "sealed" or part.startswith("sealed_")
        for part in _path_parts(path)
    )


def _guard_unsealed_path(path: Path | str, *, role: str) -> None:
    if _is_sealed_path(path):
        raise DinoCeilingAnalysisError(f"refusing sealed {role} path")


def _safe_input_path(path: Path) -> Path:
    _guard_unsealed_path(path, role="input")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DinoCeilingAnalysisError(f"input does not exist: {path}") from exc
    _guard_unsealed_path(resolved, role="input")
    if not resolved.is_file():
        raise DinoCeilingAnalysisError("input must resolve to a regular file")
    return resolved


def _safe_output_path(path: Path) -> Path:
    _guard_unsealed_path(path, role="output")
    resolved = path.resolve(strict=False)
    _guard_unsealed_path(resolved, role="output")
    if resolved.exists():
        raise DinoCeilingAnalysisError("refusing to overwrite existing output")
    return resolved


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DinoCeilingAnalysisError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(token: str) -> None:
    raise DinoCeilingAnalysisError(f"non-finite JSON number: {token}")


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DinoCeilingAnalysisError(f"{name} must be an object")
    return value


def _sequence(value: object, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DinoCeilingAnalysisError(f"{name} must be an array")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DinoCeilingAnalysisError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DinoCeilingAnalysisError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DinoCeilingAnalysisError(f"{name} must be finite")
    return number


def _expect_float(value: object, expected: float, *, name: str) -> float:
    number = _finite(value, name=name)
    if number != expected:
        raise DinoCeilingAnalysisError(f"{name} must equal {expected!r}")
    return number


def _expect_exact(mapping: Mapping[str, Any], key: str, expected: object, *, name: str) -> None:
    if key not in mapping or mapping.get(key) != expected:
        raise DinoCeilingAnalysisError(
            f"{name}.{key} must equal {expected!r}, got {mapping.get(key)!r}"
        )


def _validate_source_scope(payload: Mapping[str, Any]) -> None:
    _expect_exact(payload, "schema", INPUT_SCHEMA, name="input")
    _expect_exact(payload, "mode", "kinematic", name="input")
    _expect_exact(payload, "backend", "cpu", name="input")
    _expect_exact(payload, "task", "visible-beacon", name="input")
    _expect_exact(payload, "family", EXPECTED_FAMILY, name="input")
    _expect_exact(payload, "split", EXPECTED_SPLIT, name="input")
    _expect_exact(payload, "horizon", EXPECTED_HORIZON, name="input")
    _expect_exact(
        payload,
        "trials_per_scene",
        EXPECTED_TRIALS_PER_SCENE,
        name="input",
    )
    _expect_exact(payload, "scene_count", EXPECTED_SCENE_COUNT, name="input")
    _expect_exact(payload, "max_blocks", EXPECTED_MAX_BLOCKS, name="input")
    _expect_exact(payload, "seed", EXPECTED_SEED, name="input")
    _expect_exact(payload, "candidate_count", len(EXPECTED_PRIMITIVES), name="input")
    _expect_exact(payload, "max_candidates", None, name="input")
    _expect_exact(payload, "goal_radius_m", EXPECTED_GOAL_RADIUS_M, name="input")
    _expect_exact(payload, "goal_standoff_m", EXPECTED_GOAL_STANDOFF_M, name="input")
    _expect_exact(
        payload,
        "beacon_approach_distance_m",
        EXPECTED_BEACON_APPROACH_DISTANCE_M,
        name="input",
    )
    _expect_exact(
        payload,
        "beacon_start_yaw_jitter_rad",
        EXPECTED_BEACON_START_YAW_JITTER_RAD,
        name="input",
    )

    policies = tuple(_sequence(payload.get("policies"), name="policies"))
    if policies != EXPECTED_POLICIES:
        raise DinoCeilingAnalysisError(
            f"policies must be exactly {list(EXPECTED_POLICIES)!r} in registered order"
        )
    primitives = tuple(_sequence(payload.get("primitive_names"), name="primitive_names"))
    if primitives != EXPECTED_PRIMITIVES:
        raise DinoCeilingAnalysisError("primitive_names do not match the registered order")
    candidate_sequences = tuple(
        tuple(_sequence(item, name=f"candidate_sequences[{index}]"))
        for index, item in enumerate(
            _sequence(payload.get("candidate_sequences"), name="candidate_sequences")
        )
    )
    if candidate_sequences != EXPECTED_CANDIDATE_SEQUENCES:
        raise DinoCeilingAnalysisError("candidate_sequences do not match registered H1 rows")

    corpus = payload.get("scene_corpus")
    if not isinstance(corpus, str) or not corpus:
        raise DinoCeilingAnalysisError("scene_corpus must be a path string")
    _guard_unsealed_path(corpus, role="scene_corpus")
    corpus_parts = _path_parts(corpus)
    suffix_parts = tuple(part.lower() for part in EXPECTED_SCENE_CORPUS_SUFFIX.parts)
    if corpus_parts[-len(suffix_parts) :] != suffix_parts:
        raise DinoCeilingAnalysisError("scene_corpus is not the registered V4 development corpus")

    planning_grid = _mapping(payload.get("planning_grid"), name="planning_grid")
    _expect_float(
        planning_grid.get("cell_size_m"),
        EXPECTED_GRID_CELL_SIZE_M,
        name="planning_grid.cell_size_m",
    )
    _expect_float(
        planning_grid.get("inflation_m"),
        EXPECTED_GRID_INFLATION_M,
        name="planning_grid.inflation_m",
    )
    skipped = _sequence(payload.get("skipped"), name="skipped")
    if skipped:
        raise DinoCeilingAnalysisError("the preregistered panel permits no skipped scene or trial")


def _validate_oracle_provenance(payload: Mapping[str, Any]) -> None:
    oracle = _mapping(payload.get("oracle_assay"), name="oracle_assay")
    _expect_exact(oracle, "name", "privileged_kinematic_endpoint_distance", name="oracle_assay")
    _expect_exact(oracle, "version", 1, name="oracle_assay")
    _expect_exact(oracle, "mode", "kinematic", name="oracle_assay")
    _expect_float(
        oracle.get("tie_tolerance_m"),
        ORACLE_TIE_TOLERANCE_M,
        name="oracle_assay.tie_tolerance_m",
    )
    _expect_exact(
        oracle,
        "tie_break",
        "lowest_candidate_index_within_tolerance",
        name="oracle_assay",
    )
    candidate_bank = _mapping(oracle.get("candidate_bank"), name="oracle_assay.candidate_bank")
    _expect_exact(candidate_bank, "count", len(EXPECTED_PRIMITIVES), name="oracle_assay.candidate_bank")
    _expect_exact(candidate_bank, "max_candidates", None, name="oracle_assay.candidate_bank")
    if candidate_bank.get("ordered_sequences") != [list(item) for item in EXPECTED_CANDIDATE_SEQUENCES]:
        raise DinoCeilingAnalysisError("oracle_assay candidate bank is not the registered H1 bank")


def _validate_dino_provenance(payload: Mapping[str, Any]) -> None:
    dino = _mapping(payload.get("dino_assay"), name="dino_assay")
    expected: dict[str, object] = {
        "name": DINO_ASSAY_NAME,
        "version": DINO_ASSAY_VERSION,
        "encoder_name": DINO_ENCODER_NAME,
        "repository_path": DINO_REPOSITORY_PATH,
        "repository_commit": DINO_REPOSITORY_COMMIT,
        "checkpoint_path": DINO_CHECKPOINT_PATH,
        "checkpoint_bytes": DINO_CHECKPOINT_BYTES,
        "checkpoint_sha256": DINO_CHECKPOINT_SHA256,
        "frozen": True,
        "eval_mode": True,
        "no_grad": True,
        "feature_cache_written": False,
        "input_rgb_shape": list(DINO_INPUT_RGB_SHAPE),
        "imagenet_mean": list(DINO_IMAGENET_MEAN),
        "imagenet_std": list(DINO_IMAGENET_STD),
        "patch_output_shape": list(DINO_PATCH_OUTPUT_SHAPE),
        "token_normalization": DINO_TOKEN_NORMALIZATION,
        "cost_definition": DINO_COST_DEFINITION,
        "goal_view_count": 1,
        "successor_evaluation": DINO_SUCCESSOR_EVALUATION,
    }
    for key, value in expected.items():
        _expect_exact(dino, key, value, name="dino_assay")
    device = dino.get("device")
    if not isinstance(device, str) or not device.startswith("cuda"):
        raise DinoCeilingAnalysisError("dino_assay.device must identify the registered discrete GPU")
    for key in ("torch_version", "hip_version", "device_name"):
        value = dino.get(key)
        if not isinstance(value, str) or not value:
            raise DinoCeilingAnalysisError(f"dino_assay.{key} must be a non-empty string")
    compact_device_name = str(dino["device_name"]).replace(" ", "").upper()
    if "R9700" not in compact_device_name:
        raise DinoCeilingAnalysisError(
            "dino_assay.device_name does not identify the registered R9700 GPU"
        )
    _guard_unsealed_path(str(dino["repository_path"]), role="DINO repository")
    _guard_unsealed_path(str(dino["checkpoint_path"]), role="DINO checkpoint")


def _validate_decision(
    raw_decision: object,
    *,
    name: str,
    policy: str,
) -> float:
    decision = _mapping(raw_decision, name=name)
    regret = _finite(
        decision.get("oracle_first_action_regret_m"),
        name=f"{name}.oracle_first_action_regret_m",
    )
    tolerance = _expect_float(
        decision.get("oracle_cost_tie_tolerance_m"),
        ORACLE_TIE_TOLERANCE_M,
        name=f"{name}.oracle_cost_tie_tolerance_m",
    )
    if regret < -tolerance:
        raise DinoCeilingAnalysisError(f"{name} has negative geometric-oracle regret")
    regret = max(0.0, regret)
    disagreement = decision.get("oracle_first_action_disagreement")
    if not isinstance(disagreement, bool) or disagreement != (regret > tolerance):
        raise DinoCeilingAnalysisError(f"{name} disagreement is not tie-aware regret > tolerance")

    if policy in DINO_POLICIES:
        scores = np.asarray(
            [
                _finite(value, name=f"{name}.policy_candidate_scores[{index}]")
                for index, value in enumerate(
                    _sequence(
                        decision.get("policy_candidate_scores"),
                        name=f"{name}.policy_candidate_scores",
                    )
                )
            ],
            dtype=np.float64,
        )
        unshuffled = np.asarray(
            [
                _finite(value, name=f"{name}.unshuffled_dino_candidate_costs[{index}]")
                for index, value in enumerate(
                    _sequence(
                        decision.get("unshuffled_dino_candidate_costs"),
                        name=f"{name}.unshuffled_dino_candidate_costs",
                    )
                )
            ],
            dtype=np.float64,
        )
        sources = tuple(
            _integer(value, name=f"{name}.score_source_candidate_indices[{index}]")
            for index, value in enumerate(
                _sequence(
                    decision.get("score_source_candidate_indices"),
                    name=f"{name}.score_source_candidate_indices",
                )
            )
        )
        if scores.shape != (len(EXPECTED_PRIMITIVES),):
            raise DinoCeilingAnalysisError(f"{name} must contain nine policy candidate scores")
        if unshuffled.shape != scores.shape:
            raise DinoCeilingAnalysisError(f"{name} must contain nine unshuffled DINO costs")
        if len(sources) != len(EXPECTED_PRIMITIVES) or sorted(sources) != list(
            range(len(EXPECTED_PRIMITIVES))
        ):
            raise DinoCeilingAnalysisError(f"{name} score sources must be a permutation of candidate rows")
        if not np.array_equal(scores, unshuffled[np.asarray(sources, dtype=np.int64)]):
            raise DinoCeilingAnalysisError(f"{name} scores do not match their DINO score sources")
        selected = _integer(
            decision.get("selected_candidate_index"),
            name=f"{name}.selected_candidate_index",
        )
        if selected >= len(EXPECTED_PRIMITIVES):
            raise DinoCeilingAnalysisError(f"{name}.selected_candidate_index is out of range")
        selected_score = _finite(
            decision.get("selected_policy_score"),
            name=f"{name}.selected_policy_score",
        )
        if selected_score != float(scores[selected]):
            raise DinoCeilingAnalysisError(f"{name}.selected_policy_score does not match the score vector")
        selected_source = _integer(
            decision.get("selected_score_source_candidate_index"),
            name=f"{name}.selected_score_source_candidate_index",
        )
        if selected_source != sources[selected]:
            raise DinoCeilingAnalysisError(
                f"{name}.selected_score_source_candidate_index does not match the score vector"
            )
        if selected != int(np.argmin(scores)):
            raise DinoCeilingAnalysisError(
                f"{name}.selected_candidate_index is not the registered exact argmin"
            )
        _expect_exact(decision, "dino_cost_definition", DINO_COST_DEFINITION, name=name)
        _expect_exact(decision, "dino_checkpoint_sha256", DINO_CHECKPOINT_SHA256, name=name)
        if policy == "dino_persistence" and not np.all(scores == scores[0]):
            raise DinoCeilingAnalysisError(f"{name} persistence scores are not all equal")
        if policy == "dino_true_successor" and sources != tuple(range(len(EXPECTED_PRIMITIVES))):
            raise DinoCeilingAnalysisError(f"{name} true-successor score sources are not identity")
        if policy == "dino_true_successor_shuffled":
            block_index = _integer(decision.get("block_index"), name=f"{name}.block_index")
            expected_sources = list(range(len(EXPECTED_PRIMITIVES)))
            mixed_seed = (
                EXPECTED_SEED & ((1 << 64) - 1)
            ) ^ (
                ((block_index + 1) * SHUFFLE_MIX_CONSTANT) & ((1 << 64) - 1)
            )
            random.Random(mixed_seed).shuffle(expected_sources)
            if expected_sources == list(range(len(EXPECTED_PRIMITIVES))):
                expected_sources = expected_sources[1:] + expected_sources[:1]
            if sources != tuple(expected_sources):
                raise DinoCeilingAnalysisError(
                    f"{name} does not use the registered deterministic score permutation"
                )
    return regret


def _metric(row: Mapping[str, Any], key: str, *, name: str) -> float:
    if key == "success":
        value = row.get("success")
        if not isinstance(value, bool):
            raise DinoCeilingAnalysisError(f"{name}.success must be boolean")
        return float(value)
    return _finite(row.get(key), name=f"{name}.{key}")


def _pair_results(
    payload: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Mapping[str, Any]]],
    dict[str, dict[str, float]],
    dict[str, float],
]:
    rows = _sequence(payload.get("results"), name="results")
    if len(rows) != EXPECTED_SCENE_COUNT * len(EXPECTED_POLICIES):
        raise DinoCeilingAnalysisError("results do not contain exactly 24 complete seven-arm scenes")

    paired: dict[str, dict[str, Mapping[str, Any]]] = {}
    scene_regrets: dict[str, dict[str, float]] = {}
    decision_max_regrets = {policy: 0.0 for policy in EXPECTED_POLICIES}
    for row_index, raw_row in enumerate(rows):
        name = f"results[{row_index}]"
        row = _mapping(raw_row, name=name)
        scene_id = row.get("scene_id")
        policy = row.get("policy")
        if not isinstance(scene_id, str) or not scene_id:
            raise DinoCeilingAnalysisError(f"{name}.scene_id must be a non-empty string")
        if policy not in EXPECTED_POLICIES:
            raise DinoCeilingAnalysisError(f"{name}.policy is unexpected")
        scene_rows = paired.setdefault(scene_id, {})
        if policy in scene_rows:
            raise DinoCeilingAnalysisError(f"duplicate {policy} result for scene {scene_id}")

        initial = _finite(row.get("initial_distance_m"), name=f"{name}.initial_distance_m")
        final = _finite(row.get("final_distance_m"), name=f"{name}.final_distance_m")
        progress = _finite(row.get("progress_m"), name=f"{name}.progress_m")
        _finite(row.get("path_efficiency"), name=f"{name}.path_efficiency")
        if initial < 0.0 or final < 0.0:
            raise DinoCeilingAnalysisError(f"{name} has a negative distance")
        if not math.isclose(progress, initial - final, rel_tol=0.0, abs_tol=1.0e-8):
            raise DinoCeilingAnalysisError(f"{name}.progress_m is inconsistent with distances")
        _metric(row, "success", name=name)
        blocks = _integer(row.get("blocks_executed"), name=f"{name}.blocks_executed")
        decisions = _sequence(row.get("decision_log"), name=f"{name}.decision_log")
        if len(decisions) != blocks or not decisions:
            raise DinoCeilingAnalysisError(f"{name} must have one decision record per executed block")
        regrets = [
            _validate_decision(
                decision,
                name=f"{name}.decision_log[{decision_index}]",
                policy=str(policy),
            )
            for decision_index, decision in enumerate(decisions)
        ]
        scene_regrets.setdefault(scene_id, {})[str(policy)] = float(np.mean(regrets))
        decision_max_regrets[str(policy)] = max(
            decision_max_regrets[str(policy)], max(regrets)
        )
        scene_rows[str(policy)] = row

    if len(paired) != EXPECTED_SCENE_COUNT:
        raise DinoCeilingAnalysisError("input does not contain exactly 24 distinct scenes")
    expected = set(EXPECTED_POLICIES)
    for scene_id, scene_rows in paired.items():
        if set(scene_rows) != expected:
            raise DinoCeilingAnalysisError(f"scene {scene_id} is not seven-arm complete")
        goals = {scene_rows[policy].get("goal_object_id") for policy in EXPECTED_POLICIES}
        if len(goals) != 1 or not all(isinstance(value, str) and value for value in goals):
            raise DinoCeilingAnalysisError(f"scene {scene_id} policies do not share one goal")
        starts = [
            _finite(
                scene_rows[policy].get("initial_distance_m"),
                name=f"{scene_id}.{policy}.initial_distance_m",
            )
            for policy in EXPECTED_POLICIES
        ]
        if max(starts) - min(starts) > 1.0e-8:
            raise DinoCeilingAnalysisError(f"scene {scene_id} policies do not share one start")
    return paired, scene_regrets, decision_max_regrets


def _per_scene_metrics(
    paired: Mapping[str, Mapping[str, Mapping[str, Any]]],
    scene_regrets: Mapping[str, Mapping[str, float]],
) -> tuple[list[str], dict[str, dict[str, np.ndarray]]]:
    scenes = sorted(paired)
    metrics: dict[str, dict[str, np.ndarray]] = {}
    for policy in EXPECTED_POLICIES:
        metrics[policy] = {
            "progress_m": np.asarray(
                [_metric(paired[scene][policy], "progress_m", name=scene) for scene in scenes],
                dtype=np.float64,
            ),
            "final_distance_m": np.asarray(
                [_metric(paired[scene][policy], "final_distance_m", name=scene) for scene in scenes],
                dtype=np.float64,
            ),
            "success": np.asarray(
                [_metric(paired[scene][policy], "success", name=scene) for scene in scenes],
                dtype=np.float64,
            ),
            "path_efficiency": np.asarray(
                [_metric(paired[scene][policy], "path_efficiency", name=scene) for scene in scenes],
                dtype=np.float64,
            ),
            "scene_mean_oracle_first_action_regret_m": np.asarray(
                [scene_regrets[scene][policy] for scene in scenes], dtype=np.float64
            ),
        }
    return scenes, metrics


def _aggregate(
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    decision_max_regrets: Mapping[str, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for policy in EXPECTED_POLICIES:
        values = metrics[policy]
        result[policy] = {
            "scene_count": EXPECTED_SCENE_COUNT,
            "progress_m_mean": float(np.mean(values["progress_m"])),
            "final_distance_m_mean": float(np.mean(values["final_distance_m"])),
            "success_count": int(np.sum(values["success"])),
            "success_rate": float(np.mean(values["success"])),
            "path_efficiency_mean": float(np.mean(values["path_efficiency"])),
            "scene_mean_oracle_first_action_regret_m_mean": float(
                np.mean(values["scene_mean_oracle_first_action_regret_m"])
            ),
            "scene_mean_oracle_first_action_regret_m_max": float(
                np.max(values["scene_mean_oracle_first_action_regret_m"])
            ),
            "oracle_first_action_regret_m_max_decision": float(
                decision_max_regrets[policy]
            ),
        }
    return result


def _comparison(
    metrics: Mapping[str, Mapping[str, np.ndarray]],
    *,
    comparator: str,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    true = metrics["dino_true_successor"]
    other = metrics[comparator]
    summaries: dict[str, Any] = {}
    for metric in METRICS:
        difference = true[metric] - other[metric]
        bootstrap = difference[bootstrap_indices].mean(axis=1)
        lower, upper = np.percentile(bootstrap, BOOTSTRAP_PERCENTILES)
        summaries[metric] = {
            "true_mean": float(np.mean(true[metric])),
            "comparator_mean": float(np.mean(other[metric])),
            "true_minus_comparator_mean": float(np.mean(difference)),
            "true_minus_comparator_bootstrap_lower_95": float(lower),
            "true_minus_comparator_bootstrap_upper_95": float(upper),
        }
    return {"comparator": comparator, "metrics": summaries}


def _criterion(value: object, operator: str, threshold: object, passed: bool) -> dict[str, Any]:
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _gate(
    aggregate: Mapping[str, Any], comparisons: Mapping[str, Any]
) -> dict[str, Any]:
    shuffled = comparisons["dino_true_successor_vs_dino_true_successor_shuffled"]["metrics"]
    persistence = comparisons["dino_true_successor_vs_dino_persistence"]["metrics"]
    shuffled_progress = shuffled["progress_m"]
    shuffled_regret = shuffled["scene_mean_oracle_first_action_regret_m"]
    persistence_progress = persistence["progress_m"]
    persistence_regret = persistence["scene_mean_oracle_first_action_regret_m"]
    oracle = aggregate["oracle_mpc"]
    true_progress = float(aggregate["dino_true_successor"]["progress_m_mean"])
    random_progress = float(aggregate["random"]["progress_m_mean"])
    hold_progress = float(aggregate["hold"]["progress_m_mean"])

    progress_advantage = float(shuffled_progress["true_minus_comparator_mean"])
    shuffled_regret_difference = float(shuffled_regret["true_minus_comparator_mean"])
    shuffled_regret_reduction = -shuffled_regret_difference
    criteria = {
        "complete_exact_provenance_finite_panel": _criterion(
            True, "==", True, True
        ),
        "oracle_decision_regret_max_within_tolerance": _criterion(
            float(oracle["oracle_first_action_regret_m_max_decision"]),
            "<=",
            ORACLE_TIE_TOLERANCE_M,
            float(oracle["oracle_first_action_regret_m_max_decision"])
            <= ORACLE_TIE_TOLERANCE_M,
        ),
        "true_vs_shuffled_progress_mean_advantage_m": _criterion(
            progress_advantage, ">=", 0.10, progress_advantage >= 0.10
        ),
        "true_vs_shuffled_progress_bootstrap_lower_95": _criterion(
            float(shuffled_progress["true_minus_comparator_bootstrap_lower_95"]),
            ">",
            0.0,
            float(shuffled_progress["true_minus_comparator_bootstrap_lower_95"]) > 0.0,
        ),
        "true_vs_shuffled_scene_mean_regret_reduction_m": _criterion(
            shuffled_regret_reduction,
            ">=",
            0.02,
            shuffled_regret_reduction >= 0.02,
        ),
        "true_vs_shuffled_regret_bootstrap_upper_95": _criterion(
            float(shuffled_regret["true_minus_comparator_bootstrap_upper_95"]),
            "<",
            0.0,
            float(shuffled_regret["true_minus_comparator_bootstrap_upper_95"]) < 0.0,
        ),
        "true_vs_persistence_progress_bootstrap_interval_favorable": _criterion(
            [
                float(persistence_progress["true_minus_comparator_bootstrap_lower_95"]),
                float(persistence_progress["true_minus_comparator_bootstrap_upper_95"]),
            ],
            "lower_95 >",
            0.0,
            float(persistence_progress["true_minus_comparator_mean"]) > 0.0
            and float(persistence_progress["true_minus_comparator_bootstrap_lower_95"]) > 0.0,
        ),
        "true_vs_persistence_regret_bootstrap_interval_favorable": _criterion(
            [
                float(persistence_regret["true_minus_comparator_bootstrap_lower_95"]),
                float(persistence_regret["true_minus_comparator_bootstrap_upper_95"]),
            ],
            "upper_95 <",
            0.0,
            float(persistence_regret["true_minus_comparator_mean"]) < 0.0
            and float(persistence_regret["true_minus_comparator_bootstrap_upper_95"]) < 0.0,
        ),
        "true_mean_progress_beats_random": _criterion(
            true_progress - random_progress,
            ">",
            0.0,
            true_progress > random_progress,
        ),
        "true_mean_progress_beats_hold": _criterion(
            true_progress - hold_progress,
            ">",
            0.0,
            true_progress > hold_progress,
        ),
    }
    passes = all(item["passed"] for item in criteria.values())
    return {
        "applicable": True,
        "criteria": criteria,
        "passes_all": passes,
        "verdict": (
            "PASS_DINO_TARGET_COST_EARNS_PREDICTOR_TRAINING"
            if passes
            else "FAIL_STOP_FROZEN_DINO_SAME_PATCH_COST_ROUTE"
        ),
        "excluded_from_gate": [
            "bearing_or_geometric_oracle_superiority",
            "kinematic_fall_rate_as_safety_evidence",
        ],
    }


def analyze_payload(payload: Mapping[str, Any], *, input_path: str | None = None) -> dict[str, Any]:
    _validate_source_scope(payload)
    _validate_oracle_provenance(payload)
    _validate_dino_provenance(payload)
    paired, scene_regrets, decision_max_regrets = _pair_results(payload)
    scenes, metrics = _per_scene_metrics(paired, scene_regrets)
    aggregate = _aggregate(metrics, decision_max_regrets)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0,
        EXPECTED_SCENE_COUNT,
        size=(BOOTSTRAP_DRAWS, EXPECTED_SCENE_COUNT),
    )
    comparators = (
        "dino_true_successor_shuffled",
        "dino_persistence",
        "random",
        "hold",
        "bearing",
        "oracle_mpc",
    )
    comparisons = {
        f"dino_true_successor_vs_{comparator}": _comparison(
            metrics, comparator=comparator, bootstrap_indices=indices
        )
        for comparator in comparators
    }
    gate = _gate(aggregate, comparisons)
    return {
        "schema": SCHEMA,
        "input": input_path,
        "input_schema": INPUT_SCHEMA,
        "scope": {
            "development_only": True,
            "mode": "kinematic",
            "scene_count": EXPECTED_SCENE_COUNT,
            "safety_evidence": "not_evaluated_in_kinematic_assay",
            "bearing_role": "ceiling_not_a_superiority_target",
        },
        "validation": {
            "complete_24_scene_seven_arm_pairing": True,
            "no_skips": True,
            "exact_source_checkpoint_cost_provenance": True,
            "all_required_scores_and_metrics_finite": True,
            "per_decision_tie_aware_oracle_regret": True,
        },
        "bootstrap": {
            "unit": "whole_scene_paired_resampling",
            "scene_order": scenes,
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "percentiles": list(BOOTSTRAP_PERCENTILES),
            "numpy_percentile_method": "linear",
            "shared_resample_indices_across_all_metrics_and_contrasts": True,
        },
        "aggregate": aggregate,
        "paired_scene_comparisons": comparisons,
        "preregistered_gate": gate,
    }


def analyze_file(input_path: Path) -> dict[str, Any]:
    selected = _safe_input_path(input_path)
    try:
        raw = selected.read_bytes()
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DinoCeilingAnalysisError(f"cannot read benchmark JSON: {exc}") from exc
    report = analyze_payload(_mapping(payload, name="input"), input_path=str(selected))
    report["input_binding"] = {
        "path": str(selected),
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = analyze_file(args.input)
    output = _safe_output_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as handle:
        handle.write(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
