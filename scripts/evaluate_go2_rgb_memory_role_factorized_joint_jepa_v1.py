#!/usr/bin/env python3
"""Bounded development evaluation for memory-role factorized joint-JEPA V1.

The evaluator consumes only registered checkpoint-selection RGB callbacks.  It
never opens an index, image, checkpoint, or protected role itself, and it never
passes cell, yaw, scene, or family metadata to the model.  Place evaluation
retains the bounded 320-row, at-most-64-candidate key panel needed for
within-scene retrieval;
local evaluation retains scalar row energies only.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from lewm.datasets.go2_memory_role_place_triplets_v1 import (
    PlaceTripletRow,
    RGBTriplet,
)
from lewm.models.memory_role_factorized_joint_jepa_v1 import (
    ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1,
    LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1,
    PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1,
)


SCHEMA_PREFIX_V1 = "lewm_go2_rgb_memory_role_factorized_joint_jepa_v1"
CHECKPOINT_SELECTION_ROLE_V1 = "checkpoint_selection"
FAMILIES_V1 = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
HOLD_ACTION_INDEX_V1 = 6
EVALUATION_BATCH_SIZE_V1 = 16
PLACE_SELECTION_ROW_COUNT_V1 = 320
PLACE_FAMILY_ROW_COUNTS_V1 = dict(
    zip(FAMILIES_V1, (32, 48, 32, 32, 64, 64, 20, 28), strict=True)
)
PLACE_RETRIEVAL_MINIMUM_CANDIDATES_V1 = 40
PLACE_RETRIEVAL_MAXIMUM_CANDIDATES_V1 = 64
LOCAL_SELECTION_MAXIMUM_ROWS_V1 = 2_048
BOOTSTRAP_REPLICATES_V1 = 1_000
BOOTSTRAP_LOWER_INDEX_V1 = 25
BOOTSTRAP_SEED_V1 = 20_260_730

PLACE_ADVANTAGE_MINIMUM_V1 = 0.10
LOCAL_ACTION_ADVANTAGE_MINIMUM_V1 = 0.05
POSITIVE_FAMILY_COUNT_MINIMUM_V1 = 6
PLACE_RETRIEVAL_R5_MINIMUM_V1 = 0.40
PLACE_RETRIEVAL_CHANCE_MULTIPLIER_MINIMUM_V1 = 3.0
PLACE_RETRIEVAL_SCENE_COUNT_MINIMUM_V1 = 6
PLACE_TARGET_EFFECTIVE_RANK_MINIMUM_V1 = 4.0
PLACE_TARGET_RANK_RETENTION_MINIMUM_V1 = 0.75
PHYSICAL_CONTROL_NAMES_V1 = (
    "coordinate_matched_persistence",
    "shuffled_action",
    "wrong_rgb",
    "train_action_mean_prior",
)
PHYSICAL_CONTROL_CHECK_NAMES_V1 = (
    "positive_equal_scene_delta",
    "positive_bootstrap_lower_95",
    "positive_family_count",
)

LOCAL_CURRENT_RGB_KEY_V1 = "current_rgb"
LOCAL_NEXT_RGB_KEY_V1 = "next_rgb"
LOCAL_ACTION_KEY_V1 = "action"
LOCAL_PAIR_KEYS_V1 = (
    LOCAL_CURRENT_RGB_KEY_V1,
    LOCAL_NEXT_RGB_KEY_V1,
    LOCAL_ACTION_KEY_V1,
)


class MemoryRoleEvaluationContractError(RuntimeError):
    """A bounded evaluation invariant failed closed."""


@dataclass(frozen=True, slots=True)
class LocalSelectionRowV1:
    """Non-privileged identity for one immediate-next selection pair."""

    index: int
    role: str
    family: str
    scene_id: str
    action: int


def _finite(value: Any, *, name: str) -> float:
    if type(value) not in (int, float):
        raise MemoryRoleEvaluationContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MemoryRoleEvaluationContractError(f"{name} must be finite")
    return result


def _json_safe(value: Any, *, name: str = "evaluation result") -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise MemoryRoleEvaluationContractError(f"{name} contains nonfinite JSON")
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _json_safe(item, name=f"{name}[{index}]")
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for key, item in value.items():
            _json_safe(item, name=f"{name}.{key}")
        return
    raise MemoryRoleEvaluationContractError(f"{name} is not scalar JSON-safe")


def _validate_rgb(value: Any, *, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (3, 112, 112)
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or not bool(torch.isfinite(value).all())
    ):
        raise MemoryRoleEvaluationContractError(
            f"{name} must be finite CPU float32 3x112x112"
        )
    return value


def _validate_model_device(model: Any, device: Any) -> torch.device:
    requested = torch.device(device)
    parameters = tuple(model.parameters())
    if not parameters or any(parameter.device != requested for parameter in parameters):
        raise MemoryRoleEvaluationContractError(
            "model parameters do not share the requested evaluation device"
        )
    target_method = getattr(model, "target_modules", None)
    if not callable(target_method) or not tuple(target_method()):
        raise MemoryRoleEvaluationContractError("model has no EMA target inventory")
    return requested


def _validate_target_integrity(model: Any) -> dict[str, Any]:
    modules = tuple(model.target_modules())
    checks = {
        "target_inventory_nonempty": bool(modules),
        "target_parameters_frozen": all(
            not parameter.requires_grad
            for module in modules
            for parameter in module.parameters()
        ),
        "target_gradient_tensor_count_zero": all(
            parameter.grad is None
            for module in modules
            for parameter in module.parameters()
        ),
        "target_modules_eval": all(not module.training for module in modules),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_role_and_scenes(
    rows: Sequence[Any],
    *,
    training_scene_ids: Sequence[str] | set[str] | frozenset[str],
) -> tuple[set[str], Counter[str]]:
    training = set(training_scene_ids)
    if not training or any(type(scene) is not str or not scene for scene in training):
        raise MemoryRoleEvaluationContractError(
            "a nonempty validated training-scene inventory is required"
        )
    scenes: set[str] = set()
    families: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if (
            getattr(row, "index", None) != index
            or getattr(row, "role", None) != CHECKPOINT_SELECTION_ROLE_V1
            or getattr(row, "family", None) not in FAMILIES_V1
            or type(getattr(row, "scene_id", None)) is not str
            or not row.scene_id
        ):
            raise MemoryRoleEvaluationContractError(
                "selection row identity, role, family, or order changed"
            )
        scenes.add(row.scene_id)
        families[row.family] += 1
    if scenes & training:
        raise MemoryRoleEvaluationContractError(
            "training and checkpoint-selection scenes overlap"
        )
    if set(families) != set(FAMILIES_V1):
        raise MemoryRoleEvaluationContractError("selection panel lost a family")
    return scenes, families


def _place_energy(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if (
        predicted.shape != target.shape
        or predicted.ndim != 2
        or predicted.shape[1] != PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1
        or predicted.dtype != torch.float32
        or target.dtype != torch.float32
        or predicted.device != target.device
        or target.requires_grad
        or not bool(torch.isfinite(predicted).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise MemoryRoleEvaluationContractError("place energy operands are invalid")
    norms = torch.cat((predicted.norm(dim=1), target.norm(dim=1)))
    if not bool(torch.allclose(norms, torch.ones_like(norms), rtol=1e-4, atol=1e-4)):
        raise MemoryRoleEvaluationContractError("place keys are not unit normalized")
    result = 1.0 - F.cosine_similarity(predicted, target, dim=1, eps=1.0e-6)
    if not bool(torch.isfinite(result).all()):
        raise MemoryRoleEvaluationContractError("place energy is nonfinite")
    return result


def _local_energy(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if (
        predicted.shape != target.shape
        or predicted.ndim != 4
        or tuple(predicted.shape[1:]) != LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1
        or predicted.dtype != torch.float32
        or target.dtype != torch.float32
        or predicted.device != target.device
        or target.requires_grad
        or not bool(torch.isfinite(predicted).all())
        or not bool(torch.isfinite(target).all())
    ):
        raise MemoryRoleEvaluationContractError("local energy operands are invalid")
    return (predicted - target).square().mean(dim=(1, 2, 3))


def _tensor_rows(value: torch.Tensor, *, count: int, name: str) -> list[float]:
    if tuple(value.shape) != (count,) or not bool(torch.isfinite(value).all()):
        raise MemoryRoleEvaluationContractError(f"{name} row vector is invalid")
    return [float(item) for item in value.detach().to(device="cpu").tolist()]


def _family_summary(
    rows: Sequence[Any], values: Sequence[float]
) -> tuple[dict[str, float], int]:
    if len(rows) != len(values):
        raise MemoryRoleEvaluationContractError("family summary length changed")
    grouped: dict[str, list[float]] = {family: [] for family in FAMILIES_V1}
    for row, value in zip(rows, values, strict=True):
        grouped[row.family].append(_finite(value, name="family metric"))
    if any(not grouped[family] for family in FAMILIES_V1):
        raise MemoryRoleEvaluationContractError("family summary has an empty family")
    means = {
        family: sum(grouped[family]) / len(grouped[family])
        for family in FAMILIES_V1
    }
    return means, sum(value > 0.0 for value in means.values())


def _bootstrap_lower_95(
    rows: Sequence[Any], values: Sequence[float], *, seed: int
) -> float:
    grouped: dict[str, list[float]] = {family: [] for family in FAMILIES_V1}
    for row, value in zip(rows, values, strict=True):
        grouped[row.family].append(_finite(value, name="bootstrap value"))
    if any(not grouped[family] for family in FAMILIES_V1):
        raise MemoryRoleEvaluationContractError("bootstrap has an empty family")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_REPLICATES_V1):
        family_means = []
        for family in FAMILIES_V1:
            population = grouped[family]
            sampled = [population[rng.randrange(len(population))] for _ in population]
            family_means.append(sum(sampled) / len(sampled))
        draws.append(sum(family_means) / len(family_means))
    draws.sort()
    return float(draws[BOOTSTRAP_LOWER_INDEX_V1])


def _effective_rank(keys: torch.Tensor) -> float:
    if (
        keys.ndim != 2
        or keys.shape[0] < 2
        or keys.shape[1] != PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1
        or keys.dtype != torch.float32
        or keys.device.type != "cpu"
        or not bool(torch.isfinite(keys).all())
    ):
        raise MemoryRoleEvaluationContractError("effective-rank key panel is invalid")
    centered = keys - keys.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / float(keys.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0.0:
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
    result = float(torch.exp(entropy))
    if not math.isfinite(result):
        raise MemoryRoleEvaluationContractError("effective rank is nonfinite")
    return result


def evaluate_place_checkpoint_selection_v1(
    model: Any,
    rows: Sequence[PlaceTripletRow],
    *,
    load_triplet: Callable[[PlaceTripletRow], RGBTriplet],
    device: Any,
    training_scene_ids: Sequence[str] | set[str] | frozenset[str],
    update: int,
) -> dict[str, Any]:
    """Evaluate exact-pair place discrimination and within-scene R@5."""

    ordered = tuple(rows)
    if type(update) is not int or update not in (0, 100, 400):
        raise MemoryRoleEvaluationContractError("place update must be 0, 100, or 400")
    if len(ordered) != PLACE_SELECTION_ROW_COUNT_V1 or any(
        not isinstance(row, PlaceTripletRow) for row in ordered
    ):
        raise MemoryRoleEvaluationContractError("exact 320-row place panel is required")
    scenes, family_counts = _validate_role_and_scenes(
        ordered, training_scene_ids=training_scene_ids
    )
    if (
        any(
            family_counts[family] != PLACE_FAMILY_ROW_COUNTS_V1[family]
            for family in FAMILIES_V1
        )
        or len(scenes) != len(FAMILIES_V1)
        or any(len({row.scene_id for row in ordered if row.family == family}) != 1 for family in FAMILIES_V1)
    ):
        raise MemoryRoleEvaluationContractError("place panel family quotas changed")
    anchor_ids = [row.anchor.endpoint_identity_sha256 for row in ordered]
    positive_keys = [
        (row.scene_id, row.positive.endpoint_identity_sha256) for row in ordered
    ]
    if len(set(anchor_ids)) != len(anchor_ids) or len(set(positive_keys)) != len(positive_keys):
        raise MemoryRoleEvaluationContractError(
            "place panel anchor or within-scene positive candidates repeat"
        )

    requested = _validate_model_device(model, device)
    positive_energy: list[float] = []
    negative_energy: list[float] = []
    predicted_keys: list[torch.Tensor] = []
    target_anchor_keys: list[torch.Tensor] = []
    target_positive_keys: list[torch.Tensor] = []
    target_negative_keys: list[torch.Tensor] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(ordered), EVALUATION_BATCH_SIZE_V1):
                batch_rows = ordered[start : start + EVALUATION_BATCH_SIZE_V1]
                triplets = tuple(load_triplet(row) for row in batch_rows)
                if any(not isinstance(item, RGBTriplet) for item in triplets):
                    raise MemoryRoleEvaluationContractError("place loader emitted another type")
                anchor = torch.stack(
                    [_validate_rgb(item.anchor_rgb, name="anchor RGB") for item in triplets]
                ).to(requested)
                positive = torch.stack(
                    [_validate_rgb(item.positive_rgb, name="positive RGB") for item in triplets]
                ).to(requested)
                negative = torch.stack(
                    [_validate_rgb(item.negative_rgb, name="negative RGB") for item in triplets]
                ).to(requested)
                online = model.encode_online_roles(anchor)
                prediction = model.place_predictor(online.place_key)
                target = model.encode_target_roles(
                    torch.cat((positive, negative, anchor), dim=0)
                )
                positive_target, negative_target, anchor_target = target.place_key.split(
                    len(batch_rows)
                )
                positive_energy.extend(
                    _tensor_rows(
                        _place_energy(prediction, positive_target),
                        count=len(batch_rows),
                        name="positive place energy",
                    )
                )
                negative_energy.extend(
                    _tensor_rows(
                        _place_energy(prediction, negative_target),
                        count=len(batch_rows),
                        name="negative place energy",
                    )
                )
                predicted_keys.append(prediction.detach().to(device="cpu", dtype=torch.float32))
                target_anchor_keys.append(
                    anchor_target.detach().to(device="cpu", dtype=torch.float32)
                )
                target_positive_keys.append(
                    positive_target.detach().to(device="cpu", dtype=torch.float32)
                )
                target_negative_keys.append(
                    negative_target.detach().to(device="cpu", dtype=torch.float32)
                )
        target_integrity = _validate_target_integrity(model)
    finally:
        model.train(was_training)

    predicted_panel = torch.cat(predicted_keys, dim=0)
    anchor_target_panel = torch.cat(target_anchor_keys, dim=0)
    positive_target_panel = torch.cat(target_positive_keys, dim=0)
    negative_target_panel = torch.cat(target_negative_keys, dim=0)
    retained_panels = (
        predicted_panel,
        anchor_target_panel,
        positive_target_panel,
        negative_target_panel,
    )
    if any(
        tuple(panel.shape) != (PLACE_SELECTION_ROW_COUNT_V1, 64)
        for panel in retained_panels
    ):
        raise MemoryRoleEvaluationContractError("retained place-key panel shape changed")
    advantages = [
        negative - positive
        for positive, negative in zip(positive_energy, negative_energy, strict=True)
    ]
    family_means, positive_family_count = _family_summary(ordered, advantages)
    lower = _bootstrap_lower_95(
        ordered, advantages, seed=BOOTSTRAP_SEED_V1 + update
    )

    scene_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(ordered):
        scene_rows[row.scene_id].append(index)
    scene_retrieval: dict[str, dict[str, Any]] = {}
    row_ranks = [0] * len(ordered)
    for scene in sorted(scene_rows):
        indices = scene_rows[scene]
        predictions = predicted_panel[indices]
        candidate_ids: list[str] = []
        candidate_keys: list[torch.Tensor] = []
        seen_candidates: set[str] = set()
        for reference_name, panel in (
            ("positive", positive_target_panel),
            ("negative", negative_target_panel),
            ("anchor", anchor_target_panel),
        ):
            for index in indices:
                identity = getattr(ordered[index], reference_name).endpoint_identity_sha256
                if identity in seen_candidates:
                    continue
                seen_candidates.add(identity)
                candidate_ids.append(identity)
                candidate_keys.append(panel[index])
                if len(candidate_ids) == PLACE_RETRIEVAL_MAXIMUM_CANDIDATES_V1:
                    break
            if len(candidate_ids) == PLACE_RETRIEVAL_MAXIMUM_CANDIDATES_V1:
                break
        if not (
            PLACE_RETRIEVAL_MINIMUM_CANDIDATES_V1
            <= len(candidate_ids)
            <= PLACE_RETRIEVAL_MAXIMUM_CANDIDATES_V1
        ):
            raise MemoryRoleEvaluationContractError(
                "retrieval scene candidate count left [40,64]"
            )
        candidate_positions = {
            identity: index for index, identity in enumerate(candidate_ids)
        }
        if any(
            ordered[index].positive.endpoint_identity_sha256 not in candidate_positions
            for index in indices
        ):
            raise MemoryRoleEvaluationContractError(
                "a paired positive is absent from its retrieval candidates"
            )
        targets = torch.stack(candidate_keys, dim=0)
        energy = 1.0 - predictions @ targets.T
        if tuple(energy.shape) != (len(indices), len(candidate_ids)) or not bool(
            torch.isfinite(energy).all()
        ):
            raise MemoryRoleEvaluationContractError("retrieval energy matrix is invalid")
        ranks = []
        for local, global_index in enumerate(indices):
            # Pessimistic tie handling makes a collapsed panel rank last, not first.
            relevant = candidate_positions[
                ordered[global_index].positive.endpoint_identity_sha256
            ]
            rank = int((energy[local] <= energy[local, relevant]).sum().item())
            ranks.append(rank)
            row_ranks[global_index] = rank
        recall = sum(rank <= 5 for rank in ranks) / len(ranks)
        exact_chance = 5.0 / len(candidate_ids)
        scene_retrieval[scene] = {
            "family": ordered[indices[0]].family,
            "query_count": len(indices),
            "candidate_count": len(candidate_ids),
            "recall_at_5": recall,
            "exact_chance_recall_at_5": exact_chance,
            "mean_pessimistic_rank": sum(ranks) / len(ranks),
        }
    recall_at_5 = sum(
        item["recall_at_5"] for item in scene_retrieval.values()
    ) / len(scene_retrieval)
    exact_chance_recall_at_5 = sum(
        item["exact_chance_recall_at_5"] for item in scene_retrieval.values()
    ) / len(scene_retrieval)
    scenes_above_chance = sum(
        item["recall_at_5"] > item["exact_chance_recall_at_5"]
        for item in scene_retrieval.values()
    )
    target_rank = _effective_rank(positive_target_panel)
    predicted_rank = _effective_rank(predicted_panel)
    advantage_mean = sum(advantages) / len(advantages)
    checks = {
        "target_integrity_pass": bool(target_integrity["passed"]),
        "mean_negative_minus_positive_at_least_point10": (
            advantage_mean >= PLACE_ADVANTAGE_MINIMUM_V1
        ),
        "negative_minus_positive_bootstrap_l95_positive": lower > 0.0,
        "positive_family_count_at_least_6_of_8": (
            positive_family_count >= POSITIVE_FAMILY_COUNT_MINIMUM_V1
        ),
        "scene_retrieval_r5_at_least_point40": (
            recall_at_5 >= PLACE_RETRIEVAL_R5_MINIMUM_V1
        ),
        "scene_retrieval_r5_at_least_3x_exact_chance": (
            recall_at_5
            >= PLACE_RETRIEVAL_CHANCE_MULTIPLIER_MINIMUM_V1
            * exact_chance_recall_at_5
        ),
        "scene_retrieval_above_chance_at_least_6_of_8": (
            scenes_above_chance >= PLACE_RETRIEVAL_SCENE_COUNT_MINIMUM_V1
        ),
        "target_place_key_effective_rank_at_least_4": (
            target_rank >= PLACE_TARGET_EFFECTIVE_RANK_MINIMUM_V1
        ),
    }
    per_row = [
        {
            "index": row.index,
            "family": row.family,
            "scene_id": row.scene_id,
            "positive_energy": positive_energy[row.index],
            "negative_energy": negative_energy[row.index],
            "negative_minus_positive": advantages[row.index],
            "pessimistic_retrieval_rank": row_ranks[row.index],
        }
        for row in ordered
    ]
    result = {
        "schema": f"{SCHEMA_PREFIX_V1}_place_selection_evaluation_v1",
        "update": update,
        "role": CHECKPOINT_SELECTION_ROLE_V1,
        "row_count": len(ordered),
        "scene_count": len(scenes),
        "family_row_count": dict(family_counts),
        "energy": {
            "positive_mean": sum(positive_energy) / len(positive_energy),
            "negative_mean": sum(negative_energy) / len(negative_energy),
            "negative_minus_positive_mean": advantage_mean,
            "negative_minus_positive_bootstrap_lower_95": lower,
            "positive_family_count": positive_family_count,
            "negative_minus_positive_by_family": family_means,
        },
        "retrieval": {
            "minimum_candidate_count": min(
                item["candidate_count"] for item in scene_retrieval.values()
            ),
            "maximum_candidate_count": max(
                item["candidate_count"] for item in scene_retrieval.values()
            ),
            "recall_at_5": recall_at_5,
            "exact_chance_recall_at_5": exact_chance_recall_at_5,
            "chance_multiple": recall_at_5 / exact_chance_recall_at_5,
            "scene_count_above_chance": scenes_above_chance,
            "by_scene": scene_retrieval,
        },
        "noncollapse": {
            "target_place_key_effective_rank": target_rank,
            "predicted_place_key_effective_rank": predicted_rank,
        },
        "per_row": per_row,
        "access": {
            "triplet_loader_call_count": len(ordered),
            "rgb_tensor_count": 3 * len(ordered),
            "privileged_label_fields_passed_to_model": 0,
            "retained_place_key_rows": 4 * len(ordered),
            "retained_non_scalar_local_rows": 0,
        },
        "target_integrity": target_integrity,
        "checks": checks,
        "passed": all(checks.values()),
    }
    _json_safe(result)
    return result


def evaluate_local_checkpoint_selection_v1(
    model: Any,
    rows: Sequence[LocalSelectionRowV1],
    *,
    load_pair: Callable[[LocalSelectionRowV1], Mapping[str, Any]],
    device: Any,
    training_scene_ids: Sequence[str] | set[str] | frozenset[str],
    update: int,
) -> dict[str, Any]:
    """Evaluate immediate-next action ranking and non-hold persistence."""

    ordered = tuple(rows)
    if type(update) is not int or update not in (0, 100, 400):
        raise MemoryRoleEvaluationContractError("local update must be 0, 100, or 400")
    if not 8 <= len(ordered) <= LOCAL_SELECTION_MAXIMUM_ROWS_V1 or any(
        not isinstance(row, LocalSelectionRowV1) for row in ordered
    ):
        raise MemoryRoleEvaluationContractError("bounded local selection panel is invalid")
    scenes, family_counts = _validate_role_and_scenes(
        ordered, training_scene_ids=training_scene_ids
    )
    if any(
        type(row.action) is not int
        or not 0 <= row.action < ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1
        for row in ordered
    ):
        raise MemoryRoleEvaluationContractError("local action is outside [0,9)")
    if any(
        not any(row.family == family and row.action != HOLD_ACTION_INDEX_V1 for row in ordered)
        for family in FAMILIES_V1
    ):
        raise MemoryRoleEvaluationContractError("every family needs a non-hold local row")

    requested = _validate_model_device(model, device)
    correct_energy: list[float] = []
    wrong_energy: list[float] = []
    persistence_energy: list[float] = []
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(ordered), EVALUATION_BATCH_SIZE_V1):
                batch_rows = ordered[start : start + EVALUATION_BATCH_SIZE_V1]
                pairs = tuple(load_pair(row) for row in batch_rows)
                if any(type(pair) is not dict or tuple(pair) != LOCAL_PAIR_KEYS_V1 for pair in pairs):
                    raise MemoryRoleEvaluationContractError("local loader key order changed")
                if any(
                    type(pair[LOCAL_ACTION_KEY_V1]) is not int
                    or pair[LOCAL_ACTION_KEY_V1] != row.action
                    for row, pair in zip(batch_rows, pairs, strict=True)
                ):
                    raise MemoryRoleEvaluationContractError("local loader action changed")
                current_rgb = torch.stack(
                    [
                        _validate_rgb(pair[LOCAL_CURRENT_RGB_KEY_V1], name="local current RGB")
                        for pair in pairs
                    ]
                ).to(requested)
                next_rgb = torch.stack(
                    [
                        _validate_rgb(pair[LOCAL_NEXT_RGB_KEY_V1], name="local next RGB")
                        for pair in pairs
                    ]
                ).to(requested)
                action = torch.tensor(
                    [row.action for row in batch_rows], dtype=torch.long, device=requested
                )
                wrong_action = (action + 1) % ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1
                current = model.encode_online_roles(current_rgb).local_control
                correct = model.local_predictor(
                    current,
                    F.one_hot(
                        action, num_classes=ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1
                    ).to(dtype=torch.float32),
                )
                wrong = model.local_predictor(
                    current,
                    F.one_hot(
                        wrong_action,
                        num_classes=ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1,
                    ).to(dtype=torch.float32),
                )
                next_target = model.encode_target_roles(next_rgb).local_control
                current_target = model.encode_target_roles(current_rgb).local_control
                batch_count = len(batch_rows)
                correct_energy.extend(
                    _tensor_rows(
                        _local_energy(correct, next_target),
                        count=batch_count,
                        name="correct local energy",
                    )
                )
                wrong_energy.extend(
                    _tensor_rows(
                        _local_energy(wrong, next_target),
                        count=batch_count,
                        name="wrong local energy",
                    )
                )
                persistence_energy.extend(
                    _tensor_rows(
                        _local_energy(current_target, next_target),
                        count=batch_count,
                        name="persistence local energy",
                    )
                )
        target_integrity = _validate_target_integrity(model)
    finally:
        model.train(was_training)

    action_advantage = [
        wrong - correct
        for correct, wrong in zip(correct_energy, wrong_energy, strict=True)
    ]
    family_means, positive_family_count = _family_summary(ordered, action_advantage)
    action_lower = _bootstrap_lower_95(
        ordered, action_advantage, seed=BOOTSTRAP_SEED_V1 + 100 + update
    )
    dynamic_indices = [
        index for index, row in enumerate(ordered) if row.action != HOLD_ACTION_INDEX_V1
    ]
    dynamic_rows = tuple(ordered[index] for index in dynamic_indices)
    dynamic_correct = [correct_energy[index] for index in dynamic_indices]
    dynamic_persistence = [persistence_energy[index] for index in dynamic_indices]
    if any(value <= 0.0 for value in dynamic_persistence):
        raise MemoryRoleEvaluationContractError(
            "a non-hold persistence denominator is not strictly positive"
        )
    persistence_advantage = [
        persistence - correct
        for correct, persistence in zip(
            dynamic_correct, dynamic_persistence, strict=True
        )
    ]
    persistence_ratio = sum(dynamic_correct) / sum(dynamic_persistence)
    persistence_lower = _bootstrap_lower_95(
        dynamic_rows,
        persistence_advantage,
        seed=BOOTSTRAP_SEED_V1 + 200 + update,
    )
    persistence_family_means, persistence_positive_family_count = _family_summary(
        dynamic_rows, persistence_advantage
    )
    action_advantage_mean = sum(action_advantage) / len(action_advantage)
    checks = {
        "target_integrity_pass": bool(target_integrity["passed"]),
        "mean_wrong_minus_correct_at_least_point05": (
            action_advantage_mean >= LOCAL_ACTION_ADVANTAGE_MINIMUM_V1
        ),
        "wrong_minus_correct_bootstrap_l95_positive": action_lower > 0.0,
        "positive_family_count_at_least_6_of_8": (
            positive_family_count >= POSITIVE_FAMILY_COUNT_MINIMUM_V1
        ),
        "non_hold_correct_to_persistence_energy_ratio_strictly_below_one": (
            persistence_ratio < 1.0
        ),
        "non_hold_persistence_minus_correct_bootstrap_l95_positive": (
            persistence_lower > 0.0
        ),
    }
    per_row = [
        {
            "index": row.index,
            "family": row.family,
            "scene_id": row.scene_id,
            "action": row.action,
            "wrong_action": (row.action + 1) % ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1,
            "correct_energy": correct_energy[row.index],
            "wrong_energy": wrong_energy[row.index],
            "wrong_minus_correct": action_advantage[row.index],
            "persistence_energy": persistence_energy[row.index],
            "non_hold_persistence_gate_row": row.action != HOLD_ACTION_INDEX_V1,
        }
        for row in ordered
    ]
    result = {
        "schema": f"{SCHEMA_PREFIX_V1}_local_selection_evaluation_v1",
        "update": update,
        "role": CHECKPOINT_SELECTION_ROLE_V1,
        "row_count": len(ordered),
        "scene_count": len(scenes),
        "family_row_count": dict(family_counts),
        "action": {
            "correct_energy_mean": sum(correct_energy) / len(correct_energy),
            "wrong_energy_mean": sum(wrong_energy) / len(wrong_energy),
            "wrong_minus_correct_mean": action_advantage_mean,
            "wrong_minus_correct_bootstrap_lower_95": action_lower,
            "positive_family_count": positive_family_count,
            "wrong_minus_correct_by_family": family_means,
        },
        "persistence": {
            "non_hold_row_count": len(dynamic_indices),
            "correct_energy_mean": sum(dynamic_correct) / len(dynamic_correct),
            "no_update_energy_mean": sum(dynamic_persistence) / len(dynamic_persistence),
            "correct_to_no_update_energy_ratio": persistence_ratio,
            "no_update_minus_correct_bootstrap_lower_95": persistence_lower,
            "positive_family_count": persistence_positive_family_count,
            "no_update_minus_correct_by_family": persistence_family_means,
        },
        "per_row": per_row,
        "access": {
            "pair_loader_call_count": len(ordered),
            "rgb_tensor_count": 2 * len(ordered),
            "privileged_label_fields_passed_to_model": 0,
            "retained_non_scalar_local_rows": 0,
            "retained_scalar_energy_count": 3 * len(ordered),
        },
        "target_integrity": target_integrity,
        "checks": checks,
        "passed": all(checks.values()),
    }
    _json_safe(result)
    return result


def evaluate_checkpoint_selection_v1(
    model: Any,
    *,
    place_rows: Sequence[PlaceTripletRow],
    load_triplet: Callable[[PlaceTripletRow], RGBTriplet],
    local_rows: Sequence[LocalSelectionRowV1],
    load_local_pair: Callable[[LocalSelectionRowV1], Mapping[str, Any]],
    device: Any,
    training_scene_ids: Sequence[str] | set[str] | frozenset[str],
    update: int,
) -> dict[str, Any]:
    """Run the two bounded role evaluations without retaining RGB tensors."""

    place = evaluate_place_checkpoint_selection_v1(
        model,
        place_rows,
        load_triplet=load_triplet,
        device=device,
        training_scene_ids=training_scene_ids,
        update=update,
    )
    local = evaluate_local_checkpoint_selection_v1(
        model,
        local_rows,
        load_pair=load_local_pair,
        device=device,
        training_scene_ids=training_scene_ids,
        update=update,
    )
    result = {
        "schema": f"{SCHEMA_PREFIX_V1}_checkpoint_selection_evaluation_v1",
        "update": update,
        "place": place,
        "local": local,
        "passed": bool(place["passed"] and local["passed"]),
        "memory_intervention_metrics_present": False,
        "navigation_metrics_present": False,
    }
    _json_safe(result)
    return result


def _validate_role_result(value: Any, *, kind: str, update: int) -> dict[str, Any]:
    expected = f"{SCHEMA_PREFIX_V1}_{kind}_selection_evaluation_v1"
    if (
        type(value) is not dict
        or value.get("schema") != expected
        or value.get("update") != update
        or type(value.get("checks")) is not dict
        or type(value.get("passed")) is not bool
        or value["passed"] != all(value["checks"].values())
    ):
        raise MemoryRoleEvaluationContractError(f"{kind} result contract changed")
    _json_safe(value, name=f"{kind} result")
    return value


def _flatten_controls(value: Any) -> dict[str, bool]:
    if type(value) is not dict or set(value) != set(PHYSICAL_CONTROL_NAMES_V1):
        raise MemoryRoleEvaluationContractError("exact causal control set changed")
    result: dict[str, bool] = {}
    for control in PHYSICAL_CONTROL_NAMES_V1:
        checks = value[control]
        if type(checks) is not dict or set(checks) != set(
            PHYSICAL_CONTROL_CHECK_NAMES_V1
        ):
            raise MemoryRoleEvaluationContractError(
                f"causal control schema changed: {control}"
            )
        for check in PHYSICAL_CONTROL_CHECK_NAMES_V1:
            decision = checks[check]
            if type(decision) is not bool:
                raise MemoryRoleEvaluationContractError("causal control decision changed")
            result[f"{control}:{check}"] = decision
    return result


def _validate_physical_summary(value: Any) -> dict[str, float]:
    if type(value) is not dict:
        raise MemoryRoleEvaluationContractError("physical summary must be a plain object")
    margin_count = value.get("margin_count")
    passed_count = value.get("passed_margin_count")
    rough = value.get("rough_motion")
    if (
        type(margin_count) is not int
        or margin_count != 189
        or type(passed_count) is not int
        or not 0 <= passed_count <= margin_count
        or type(rough) is not dict
    ):
        raise MemoryRoleEvaluationContractError("physical 189-margin schema changed")
    return {
        "margin_count": margin_count,
        "passed_margin_count": passed_count,
        "rough_depth_p95_m": _finite(rough.get("depth_p95_m"), name="rough depth p95"),
    }


def evaluate_terminal_gate_v1(
    *,
    update0_place: Mapping[str, Any],
    update400_place: Mapping[str, Any],
    update400_local: Mapping[str, Any],
    physical_summary: Mapping[str, Any],
    controls: Mapping[str, Mapping[str, bool]],
    integrity_pass: bool,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply only the role-appropriate update-400 falsification gates."""

    if type(integrity_pass) is not bool:
        raise MemoryRoleEvaluationContractError("structural integrity must be Boolean")
    baseline = _validate_role_result(update0_place, kind="place", update=0)
    place = _validate_role_result(update400_place, kind="place", update=400)
    local = _validate_role_result(update400_local, kind="local", update=400)
    physical = _validate_physical_summary(physical_summary)
    causal = _flatten_controls(controls)
    baseline_rank = _finite(
        baseline["noncollapse"].get("target_place_key_effective_rank"),
        name="update-0 target place rank",
    )
    final_rank = _finite(
        place["noncollapse"].get("target_place_key_effective_rank"),
        name="update-400 target place rank",
    )
    checks = {
        "structural_integrity_pass": integrity_pass,
        "place_update400_checks_pass": bool(place["passed"]),
        "place_target_rank_retains_at_least_75_percent_of_update0": (
            final_rank >= PLACE_TARGET_RANK_RETENTION_MINIMUM_V1 * baseline_rank
        ),
        "local_update400_checks_pass": bool(local["passed"]),
        "all_twelve_causal_control_checks_true": all(causal.values()),
        "passed_physical_margin_count_strictly_greater_than_72_of_189": (
            physical["passed_margin_count"] > 72
        ),
    }
    diagnostic_only: dict[str, Any] = {
        "rough_depth_p95_m": physical["rough_depth_p95_m"],
        "rough_depth_is_a_gate": False,
        "tail_metric_is_a_gate": False,
        "prior_metric_is_a_gate": False,
    }
    if diagnostics is not None:
        if type(diagnostics) is not dict:
            raise MemoryRoleEvaluationContractError("diagnostics must be a plain object")
        _json_safe(diagnostics, name="diagnostics")
        diagnostic_only["reported"] = dict(diagnostics)
    passed = all(checks.values())
    result = {
        "schema": f"{SCHEMA_PREFIX_V1}_terminal_gate_v1",
        "update": 400,
        "checks": checks,
        "passed": passed,
        "observed": {
            "update0_target_place_key_effective_rank": baseline_rank,
            "update400_target_place_key_effective_rank": final_rank,
            "target_place_key_rank_retention_ratio": (
                final_rank / baseline_rank if baseline_rank > 0.0 else 0.0
            ),
            "passed_physical_margin_count": physical["passed_margin_count"],
            "physical_margin_count": physical["margin_count"],
        },
        "causal_control_checks": causal,
        "diagnostic_only": diagnostic_only,
        "memory_reset_reverse_shuffle_required": False,
        "navigation_evaluation_required": False,
        "action": (
            "PASS_ROLE_FACTORIZED_JEPA_PROBE_ELIGIBLE_FOR_MEMORY_INTEGRATION"
            if passed
            else "FAIL_TERMINAL_NO_MEMORY_INTEGRATION"
        ),
    }
    _json_safe(result)
    return result


__all__ = [
    "BOOTSTRAP_REPLICATES_V1",
    "CHECKPOINT_SELECTION_ROLE_V1",
    "EVALUATION_BATCH_SIZE_V1",
    "FAMILIES_V1",
    "HOLD_ACTION_INDEX_V1",
    "LOCAL_ACTION_ADVANTAGE_MINIMUM_V1",
    "LOCAL_PAIR_KEYS_V1",
    "LocalSelectionRowV1",
    "MemoryRoleEvaluationContractError",
    "PLACE_ADVANTAGE_MINIMUM_V1",
    "PLACE_FAMILY_ROW_COUNTS_V1",
    "PLACE_RETRIEVAL_MAXIMUM_CANDIDATES_V1",
    "PLACE_RETRIEVAL_MINIMUM_CANDIDATES_V1",
    "PLACE_RETRIEVAL_R5_MINIMUM_V1",
    "PLACE_SELECTION_ROW_COUNT_V1",
    "PLACE_TARGET_EFFECTIVE_RANK_MINIMUM_V1",
    "evaluate_checkpoint_selection_v1",
    "evaluate_local_checkpoint_selection_v1",
    "evaluate_place_checkpoint_selection_v1",
    "evaluate_terminal_gate_v1",
]
