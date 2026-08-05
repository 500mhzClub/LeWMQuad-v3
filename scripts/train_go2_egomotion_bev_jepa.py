#!/usr/bin/env python3
"""Train and gate the egomotion-aligned JEPA traversability model.

New datasets carry label-independent train, checkpoint-selection,
probability-calibration, and untouched-G2 roles assigned by family at build
time. Legacy train/validation datasets retain the deterministic validation
split for compatibility. G2 images are evaluated only after model and
calibration selection are complete.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, deque
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lewm.benchmarks.experiment_manifest import (  # noqa: E402
    build_experiment_manifest,
    sha256_file,
)
from lewm.benchmarks.traversability_metrics import (  # noqa: E402
    FREE_CLASS,
    OCCUPIED_CLASS,
    UNKNOWN_CLASS,
    TraversabilityThresholds,
    evaluate_traversability,
    planned_path_collision_rate,
    select_conservative_thresholds,
)
from lewm.datasets.go2_paired_navigation import (  # noqa: E402
    DATASET_ROLES,
    canonical_json_sha256,
    deterministic_family_role_split,
    scene_id_sha256,
    verify_dataset_provenance,
)
from lewm.hierarchical_probability_calibration import (  # noqa: E402
    CALIBRATION_METHOD as HIERARCHICAL_CALIBRATION_METHOD,
    HierarchicalCalibrationParameters,
    fit_hierarchical_probability_calibration,
    hierarchical_calibrated_probabilities,
    validate_hierarchical_probability_calibration,
)
from lewm.models.egomotion_bev_jepa import (  # noqa: E402
    EgomotionBevJepa,
    GLOBAL_CROSS_ATTENTION_LIFT,
    PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    PROJECTIVE_COLUMN_ATTENTION_LIFT,
    PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
    build_projective_query_support_contract,
    validate_projective_query_support_binding,
    warp_bev_current_to_next,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260709
OCCUPANCY_LOSS_MODE = "hierarchical_equal_capacity_v1"
DEVELOPMENT_DATASET_ROLES = DATASET_ROLES[:-1]
SELECTION_SCORE_MODES = (
    "full_g2_v1",
    "occupancy_ceiling_v1",
    "physical_occupancy_ceiling_v1",
)
CALIBRATION_MODES = ("vector_scaling_v1", "hierarchical_log_odds_v1")
SOURCE_FOV_RECTIFICATION_MODES = (
    "none",
    "legacy_v03_square_vertical_fov_v1",
)
REGISTERED_PROJECTIVE_FOOTPRINT_PERIMETER_SAMPLES = 8
REGISTERED_PHYSICAL_OCCUPIED_DETECTION_PROBABILITY_CANDIDATES = (
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
    0.35,
    0.50,
)
SUPPORTED_DATASET_SCHEMAS = (
    "lewm_go2_paired_navigation_dataset_v2",
    "lewm_go2_paired_navigation_dataset_v3",
)
PHYSICAL_DATASET_LABEL_CONTRACT = "observable_physical_occupancy_v3"
CONFIGURATION_OCCUPANCY_TARGET_SPACE = "body_inflated_configuration_space"
PHYSICAL_OCCUPANCY_TARGET_SPACE = "observable_physical_occupancy"
PHYSICAL_CHECKPOINT_SCHEMA = "lewm_go2_egomotion_bev_jepa_checkpoint_v4"
PHYSICAL_REPORT_SCHEMA = "lewm_go2_egomotion_bev_jepa_training_report_v4"
PHYSICAL_G2_GATE_SCHEMA = "lewm_go2_physical_evidence_g2_v1"


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_normalize(value: Any) -> Any:
    """Recursively convert run configuration values to canonical JSON types."""

    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {
            str(key): _json_normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_normalize(item) for item in value), key=repr)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON-normalizable: {type(value).__name__}")


def _row_subset_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
) -> dict[str, Any]:
    identities = []
    seen_global_rows: set[int] = set()
    for row in sorted(rows, key=lambda item: int(item["global_row"])):
        global_row = int(row["global_row"])
        if global_row in seen_global_rows:
            raise ValueError(f"duplicate global_row in {role} subset: {global_row}")
        seen_global_rows.add(global_row)
        identities.append(
            {
                "global_row": global_row,
                "scene_id": str(row["scene_id"]),
                "scene_id_sha256": scene_id_sha256(str(row["scene_id"])),
                "label_shard_row": int(row["label_shard_row"]),
                "label_shard_sha256": str(row["label_shard_sha256"]),
                "current_image_sha256": str(row["current_image_sha256"]),
                "next_image_sha256": str(row["next_image_sha256"]),
            }
        )
    return {
        "schema": "lewm_go2_deterministic_row_subset_v1",
        "role": str(role),
        "order": "global_row_ascending",
        "count": len(identities),
        "identity_fields": [
            "global_row",
            "scene_id",
            "scene_id_sha256",
            "label_shard_row",
            "label_shard_sha256",
            "current_image_sha256",
            "next_image_sha256",
        ],
        "identities": identities,
        "identity_sha256": _canonical_json_sha256(identities),
    }


def _hash_rank(seed: str, *parts: object) -> int:
    value = "\0".join((str(seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(value.encode()).digest(), "big")


def split_validation_scenes(
    scene_ids: Iterable[str],
    *,
    seed: str,
) -> dict[str, str]:
    """Split validation scenes into selection, calibration, and untouched G2."""

    unique = sorted(set(map(str, scene_ids)))
    if len(unique) < 3:
        raise ValueError("at least three validation scenes are required")
    ranked = sorted(unique, key=lambda scene: (_hash_rank(seed, scene), scene))
    first_cut = max(1, len(ranked) // 3)
    second_cut = max(first_cut + 1, 2 * len(ranked) // 3)
    second_cut = min(second_cut, len(ranked) - 1)
    selection = set(ranked[:first_cut])
    calibration = set(ranked[first_cut:second_cut])
    return {
        scene: (
            "checkpoint_selection"
            if scene in selection
            else "probability_calibration"
            if scene in calibration
            else "g2_evaluation"
        )
        for scene in unique
    }


def resolve_dataset_scene_roles(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    legacy_selection_seed: str,
) -> dict[str, str]:
    """Validate direct dataset roles, or derive roles for a legacy split."""

    if not rows:
        raise ValueError("dataset row index is empty")
    rows_by_scene: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        scene_id = str(row.get("scene_id", "")).strip()
        if not scene_id:
            raise ValueError("every row must declare a nonempty scene_id")
        rows_by_scene.setdefault(scene_id, []).append(row)

    role_contract = manifest.get("scene_roles")
    rows_with_direct_role = sum("dataset_role" in row for row in rows)
    if role_contract is None:
        if rows_with_direct_role:
            raise ValueError(
                "row-level dataset_role values require a manifest scene_roles contract"
            )
        splits: dict[str, str] = {}
        for scene_id, scene_rows in rows_by_scene.items():
            values = {str(row.get("dataset_split", "")) for row in scene_rows}
            if len(values) != 1 or next(iter(values)) not in {"train", "validation"}:
                raise ValueError(
                    f"legacy scene {scene_id!r} must have exactly one train/validation split"
                )
            splits[scene_id] = next(iter(values))
        train_scenes = {scene for scene, split in splits.items() if split == "train"}
        validation_scenes = {
            scene for scene, split in splits.items() if split == "validation"
        }
        if not train_scenes or not validation_scenes:
            raise ValueError(
                "dataset must contain scene-disjoint train and validation rows"
            )
        validation_roles = split_validation_scenes(
            validation_scenes, seed=legacy_selection_seed
        )
        return {
            scene_id: (
                "train" if split == "train" else validation_roles[scene_id]
            )
            for scene_id, split in sorted(splits.items())
        }

    if not isinstance(role_contract, Mapping):
        raise ValueError("manifest scene_roles contract must be an object")
    if role_contract.get("schema") != "lewm_go2_family_scene_roles_v1":
        raise ValueError("unsupported manifest scene_roles schema")
    if rows_with_direct_role != len(rows):
        raise ValueError("every row must declare dataset_role in direct-role mode")
    raw_assignments = role_contract.get("assignments")
    if not isinstance(raw_assignments, Mapping):
        raise ValueError("direct-role contract lacks scene assignments")
    assignments = {
        str(scene_id): str(role) for scene_id, role in raw_assignments.items()
    }
    if set(assignments) != set(rows_by_scene):
        missing = sorted(set(rows_by_scene) - set(assignments))
        empty = sorted(set(assignments) - set(rows_by_scene))
        raise ValueError(
            "direct-role assignment/row scene mismatch: "
            f"unassigned_rows={missing}, assigned_without_rows={empty}"
        )
    invalid_roles = sorted(set(assignments.values()) - set(DATASET_ROLES))
    if invalid_roles:
        raise ValueError(f"unsupported direct dataset roles: {invalid_roles}")
    for role in DATASET_ROLES:
        if role not in assignments.values():
            raise ValueError(f"direct dataset role is empty: {role}")
    if str(role_contract.get("assignments_sha256", "")) != canonical_json_sha256(
        assignments
    ):
        raise ValueError("direct-role assignment commitment mismatch")

    scene_families: dict[str, str] = {}
    role_row_counts = Counter()
    family_role_scene_counts: dict[str, Counter[str]] = {}
    family_role_row_counts: dict[str, Counter[str]] = {}
    for scene_id, scene_rows in rows_by_scene.items():
        row_roles = {str(row["dataset_role"]) for row in scene_rows}
        if row_roles != {assignments[scene_id]}:
            raise ValueError(
                f"scene {scene_id!r} rows disagree with its direct role assignment"
            )
        expected_split = "train" if assignments[scene_id] == "train" else "validation"
        row_splits = {str(row.get("dataset_split", "")) for row in scene_rows}
        if row_splits != {expected_split}:
            raise ValueError(
                f"scene {scene_id!r} dataset_split disagrees with direct role"
            )
        families = {str(row.get("family", "")).strip() for row in scene_rows}
        if len(families) != 1 or not next(iter(families)):
            raise ValueError(
                f"scene {scene_id!r} must declare exactly one nonempty family"
            )
        family = next(iter(families))
        scene_families[scene_id] = family
        role = assignments[scene_id]
        role_row_counts[role] += len(scene_rows)
        family_role_scene_counts.setdefault(family, Counter())[role] += 1
        family_role_row_counts.setdefault(family, Counter())[role] += len(scene_rows)

    expected_assignments = deterministic_family_role_split(
        scene_families,
        role_scenes_per_family=int(role_contract["role_scenes_per_family"]),
        seed=str(role_contract["seed"]),
    )
    if assignments != expected_assignments:
        raise ValueError("direct roles do not reproduce from family, scene ID, and seed")

    expected_role_scenes = {
        role: sorted(
            scene_id for scene_id, assigned in assignments.items() if assigned == role
        )
        for role in DATASET_ROLES
    }
    expected_scene_counts = {
        role: len(scene_ids) for role, scene_ids in expected_role_scenes.items()
    }
    expected_commitments = {
        role: canonical_json_sha256(
            sorted(scene_id_sha256(scene_id) for scene_id in scene_ids)
        )
        for role, scene_ids in expected_role_scenes.items()
    }
    normalized_family_scene_counts = {
        family: {role: int(counts.get(role, 0)) for role in DATASET_ROLES}
        for family, counts in sorted(family_role_scene_counts.items())
    }
    normalized_family_row_counts = {
        family: {role: int(counts.get(role, 0)) for role in DATASET_ROLES}
        for family, counts in sorted(family_role_row_counts.items())
    }
    declared_checks = (
        ("scene_counts", expected_scene_counts),
        ("row_counts", {role: int(role_row_counts[role]) for role in DATASET_ROLES}),
        ("family_scene_counts", normalized_family_scene_counts),
        ("family_row_counts", normalized_family_row_counts),
        ("scene_id_sha256_commitments", expected_commitments),
    )
    for field, expected in declared_checks:
        if role_contract.get(field) != expected:
            raise ValueError(f"direct-role {field} commitment mismatch")
    if role_contract.get("label_independent") is not True:
        raise ValueError("direct-role contract must declare label independence")
    return dict(sorted(assignments.items()))


def _verify_dataset_role_provenance(
    dataset_manifest_path: Path,
    manifest: Mapping[str, Any],
    scene_roles: Mapping[str, str],
    *,
    roles: Iterable[str],
) -> dict[str, Any]:
    """Verify exactly one declared role scope without opening other artifacts."""

    normalized_roles = tuple(dict.fromkeys(str(role) for role in roles))
    if not normalized_roles or set(normalized_roles) - set(DATASET_ROLES):
        raise ValueError(f"invalid dataset provenance role scope: {normalized_roles}")
    selected_scenes = sorted(
        scene_id
        for scene_id, role in scene_roles.items()
        if role in normalized_roles
    )
    if not selected_scenes:
        raise ValueError(f"dataset provenance scope is empty: {normalized_roles}")
    if manifest.get("scene_roles") is not None:
        checked = verify_dataset_provenance(
            dataset_manifest_path,
            verify_images=True,
            roles=normalized_roles,
        )
        selector = "direct_dataset_roles"
    else:
        checked = verify_dataset_provenance(
            dataset_manifest_path,
            verify_images=True,
            scene_ids=selected_scenes,
        )
        selector = "resolved_legacy_scene_ids"
    return {
        "roles": list(normalized_roles),
        "selector": selector,
        "scene_count": len(selected_scenes),
        "scene_id_sha256_commitment": canonical_json_sha256(
            sorted(scene_id_sha256(scene_id) for scene_id in selected_scenes)
        ),
        "images_verified": True,
        "checked": checked,
    }


def deterministic_row_subset(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_rows: int,
    seed: str,
) -> list[dict[str, Any]]:
    """Hash-rank rows while retaining at least one row from every scene."""

    normalized = [dict(row) for row in rows]
    if maximum_rows <= 0 or len(normalized) <= maximum_rows:
        return normalized
    by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in normalized:
        by_scene.setdefault(str(row["scene_id"]), []).append(row)
    if maximum_rows < len(by_scene):
        raise ValueError("maximum_rows is smaller than the number of scenes")
    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for scene, scene_rows in sorted(by_scene.items()):
        ranked = sorted(
            scene_rows,
            key=lambda row: (
                _hash_rank(seed, scene, row.get("global_row", 0)),
                int(row.get("global_row", 0)),
            ),
        )
        selected.append(ranked[0])
        remaining.extend(ranked[1:])
    remaining.sort(
        key=lambda row: (
            _hash_rank(seed, row["scene_id"], row.get("global_row", 0)),
            int(row.get("global_row", 0)),
        )
    )
    selected.extend(remaining[: maximum_rows - len(selected)])
    selected.sort(key=lambda row: int(row.get("global_row", 0)))
    return selected


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(row)
    return rows


def _mask_array(shard: Mapping[str, np.ndarray], prefix: str, labels: np.ndarray) -> np.ndarray:
    """Load the explicit CE mask, with a fail-safe for pre-correction shards."""

    for key in (f"{prefix}_supervision_mask", f"{prefix}_validity"):
        if key in shard:
            mask = np.asarray(shard[key], dtype=bool)
            if mask.shape != labels.shape:
                raise ValueError(f"{key} does not match {prefix}_labels")
            # Old shards defined validity as label != UNKNOWN. Refuse them:
            # they silently remove all unknown-class supervision.
            if key.endswith("validity") and np.array_equal(mask, labels != UNKNOWN_CLASS):
                raise ValueError(
                    "dataset uses legacy validity=label!=UNKNOWN; rebuild with an "
                    "explicit full supervision_mask and observed_mask"
                )
            return mask
    raise KeyError(f"shard is missing {prefix}_supervision_mask")


def _observed_mask_array(
    shard: Mapping[str, np.ndarray],
    prefix: str,
    labels: np.ndarray,
) -> np.ndarray:
    key = f"{prefix}_observed_mask"
    if key not in shard:
        raise KeyError(f"shard is missing {key}")
    mask = np.asarray(shard[key], dtype=bool)
    if mask.shape != labels.shape:
        raise ValueError(f"{key} does not match {prefix}_labels")
    if not np.array_equal(mask, labels != UNKNOWN_CLASS):
        raise ValueError(f"{key} must equal labels != UNKNOWN")
    return mask


def nominal_primitive_delta_table(
    rows: Sequence[Mapping[str, Any]],
    primitive_to_index: Mapping[str, int],
) -> torch.Tensor:
    """Build a frozen runtime table from training rows only."""

    grouped: dict[str, list[np.ndarray]] = {
        primitive: [] for primitive in primitive_to_index
    }
    for row in rows:
        primitive = str(row["primitive"])
        if primitive not in grouped:
            raise ValueError(f"row has unknown primitive {primitive!r}")
        delta = np.asarray(row["relative_se2_current_frame"], dtype=np.float64)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("relative_se2_current_frame must contain three finite values")
        grouped[primitive].append(delta)
    table = np.zeros((len(primitive_to_index), 3), dtype=np.float32)
    for primitive, index in primitive_to_index.items():
        if not grouped[primitive]:
            raise ValueError(f"primitive {primitive!r} has no training deltas")
        values = np.stack(grouped[primitive])
        table[index, :2] = np.median(values[:, :2], axis=0)
        center = math.atan2(
            float(np.sin(values[:, 2]).mean()),
            float(np.cos(values[:, 2]).mean()),
        )
        offsets = np.arctan2(
            np.sin(values[:, 2] - center),
            np.cos(values[:, 2] - center),
        )
        table[index, 2] = math.atan2(
            math.sin(center + float(np.median(offsets))),
            math.cos(center + float(np.median(offsets))),
        )
    return torch.from_numpy(table)


def nominal_primitive_delta_statistics(
    rows: Sequence[Mapping[str, Any]],
    primitive_to_index: Mapping[str, int],
    table: torch.Tensor,
) -> dict[str, dict[str, float | int]]:
    values = table.detach().cpu().numpy()
    grouped: dict[str, list[np.ndarray]] = {
        primitive: [] for primitive in primitive_to_index
    }
    for row in rows:
        grouped[str(row["primitive"])].append(
            np.asarray(row["relative_se2_current_frame"], dtype=np.float64)
        )
    result: dict[str, dict[str, float | int]] = {}
    for primitive, index in primitive_to_index.items():
        samples = np.stack(grouped[primitive])
        translation_error = np.linalg.norm(samples[:, :2] - values[index, :2], axis=1)
        yaw_error = np.abs(
            np.arctan2(
                np.sin(samples[:, 2] - values[index, 2]),
                np.cos(samples[:, 2] - values[index, 2]),
            )
        )
        result[primitive] = {
            "count": int(samples.shape[0]),
            "translation_error_median_m": float(np.median(translation_error)),
            "translation_error_p90_m": float(np.quantile(translation_error, 0.9)),
            "yaw_error_median_rad": float(np.median(yaw_error)),
            "yaw_error_p90_rad": float(np.quantile(yaw_error, 0.9)),
        }
    return result


class PairedNavigationTorchDataset(Dataset[dict[str, torch.Tensor]]):
    """Lazy image/shard reader with one small per-worker NPZ cache."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        primitive_to_index: Mapping[str, int],
        image_size: int,
        source_crop_fraction_xy: tuple[float, float] = (1.0, 1.0),
        shard_cache_size: int = 2,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.primitive_to_index = dict(primitive_to_index)
        self.image_size = int(image_size)
        self.source_crop_fraction_xy = tuple(
            float(value) for value in source_crop_fraction_xy
        )
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._shards: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if (
            len(self.source_crop_fraction_xy) != 2
            or any(
                not math.isfinite(value) or not 0.0 < value <= 1.0
                for value in self.source_crop_fraction_xy
            )
        ):
            raise ValueError("source crop fractions must lie in (0, 1]")

    def __len__(self) -> int:
        return len(self.rows)

    def _shard(self, path: str) -> dict[str, np.ndarray]:
        if path in self._shards:
            self._shards.move_to_end(path)
            return self._shards[path]
        with np.load(path, allow_pickle=False) as archive:
            shard = {key: np.asarray(archive[key]) for key in archive.files}
        self._shards[path] = shard
        self._shards.move_to_end(path)
        while len(self._shards) > self.shard_cache_size:
            self._shards.popitem(last=False)
        return shard

    def _image(self, path: str) -> torch.Tensor:
        with Image.open(path) as image:
            image = image.convert("RGB")
            crop_x, crop_y = self.source_crop_fraction_xy
            if crop_x < 1.0 or crop_y < 1.0:
                crop_width = max(1, int(round(image.width * crop_x)))
                crop_height = max(1, int(round(image.height * crop_y)))
                left = (image.width - crop_width) // 2
                top = (image.height - crop_height) // 2
                image = image.crop(
                    (left, top, left + crop_width, top + crop_height)
                )
            image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.float32) / 255.0
        # Fixed normalization is part of the checkpoint/runtime contract.
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = tensor.new_tensor((0.485, 0.456, 0.406))[:, None, None]
        std = tensor.new_tensor((0.229, 0.224, 0.225))[:, None, None]
        return (tensor - mean) / std

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        shard = self._shard(str(row["label_shard_path"]))
        shard_row = int(row["label_shard_row"])
        current_labels = np.asarray(shard["current_labels"][shard_row], dtype=np.int64)
        next_labels = np.asarray(shard["next_labels"][shard_row], dtype=np.int64)
        current_mask = _mask_array(shard, "current", shard["current_labels"])[shard_row]
        next_mask = _mask_array(shard, "next", shard["next_labels"])[shard_row]
        current_observed = _observed_mask_array(
            shard, "current", shard["current_labels"]
        )[shard_row]
        next_observed = _observed_mask_array(
            shard, "next", shard["next_labels"]
        )[shard_row]
        primitive = str(row["primitive"])
        if primitive not in self.primitive_to_index:
            raise KeyError(f"primitive {primitive!r} is absent from the training vocabulary")
        action = torch.zeros(len(self.primitive_to_index), dtype=torch.float32)
        action[self.primitive_to_index[primitive]] = 1.0
        delta = np.asarray(row["relative_se2_current_frame"], dtype=np.float32)
        if delta.shape != (3,) or not np.isfinite(delta).all():
            raise ValueError("relative_se2_current_frame must contain three finite values")
        return {
            "current_image": self._image(str(row["current_image_path"])),
            "next_image": self._image(str(row["next_image_path"])),
            "action": action,
            "delta": torch.from_numpy(delta),
            "current_labels": torch.from_numpy(current_labels.copy()),
            "next_labels": torch.from_numpy(next_labels.copy()),
            "current_mask": torch.from_numpy(np.asarray(current_mask, dtype=bool).copy()),
            "next_mask": torch.from_numpy(np.asarray(next_mask, dtype=bool).copy()),
            "current_observed_mask": torch.from_numpy(
                np.asarray(current_observed, dtype=bool).copy()
            ),
            "next_observed_mask": torch.from_numpy(
                np.asarray(next_observed, dtype=bool).copy()
            ),
            "global_row": torch.tensor(int(row.get("global_row", index))),
        }


def _balanced_binary_weights(counts: Sequence[int]) -> torch.Tensor:
    """Return inverse-count weights with an irrelevant mean-one normalization."""

    values = np.asarray(counts, dtype=np.float64)
    if values.shape != (2,) or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"both binary classes must occur, got {values.tolist()}")
    inverse = 1.0 / values
    inverse /= inverse.mean()
    return torch.tensor(inverse, dtype=torch.float32)


def _hierarchical_occupancy_objective(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build the preregistered equal-capacity hierarchical occupancy objective."""

    counts = np.zeros(3, dtype=np.int64)
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row["label_shard_path"]), []).append(
            int(row["label_shard_row"])
        )
    for path, indices in grouped.items():
        with np.load(path, allow_pickle=False) as shard:
            for prefix in ("current", "next"):
                labels_all = np.asarray(shard[f"{prefix}_labels"])
                mask_all = _mask_array(shard, prefix, labels_all)
                labels = labels_all[indices]
                mask = mask_all[indices]
                counts += np.bincount(labels[mask].reshape(-1), minlength=3)[:3]
    if (counts == 0).any():
        raise ValueError(
            "every occupancy class must occur in training data, "
            f"got {counts.tolist()}"
        )
    class_counts = {
        "unknown": int(counts[UNKNOWN_CLASS]),
        "free": int(counts[FREE_CLASS]),
        "occupied": int(counts[OCCUPIED_CLASS]),
    }
    unknown_known_counts = (
        class_counts["unknown"],
        class_counts["free"] + class_counts["occupied"],
    )
    free_occupied_counts = (class_counts["free"], class_counts["occupied"])
    unknown_known_weights = _balanced_binary_weights(unknown_known_counts)
    free_occupied_weights = _balanced_binary_weights(free_occupied_counts)
    provenance = {
        "schema": "lewm_hierarchical_occupancy_objective_v1",
        "mode": OCCUPANCY_LOSS_MODE,
        "source": "train_role_current_and_next_supervision_masks",
        "weight_rule": "inverse_global_count_normalized_to_mean_one",
        "reduction": "weighted_mean_over_valid_cells",
        "terms": {
            "unknown_vs_known": {
                "coefficient": 0.5,
                "class_order": ["unknown", "known"],
                "counts": list(unknown_known_counts),
                "weights": unknown_known_weights.tolist(),
            },
            "free_vs_occupied_given_known": {
                "coefficient": 0.5,
                "class_order": ["free", "occupied"],
                "counts": list(free_occupied_counts),
                "weights": free_occupied_weights.tolist(),
            },
        },
        "three_class_weights": None,
        "three_class_class_counts": class_counts,
    }
    return unknown_known_weights, free_occupied_weights, provenance


def _to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "global_row"
    }


def _masked_normalized_error(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[float, int]:
    prediction = torch.nn.functional.normalize(prediction.float(), dim=1)
    target = torch.nn.functional.normalize(target.float(), dim=1)
    error = (prediction - target).square().mean(dim=1)
    weight = mask[:, 0].to(error.dtype)
    return float((error * weight).sum().item()), int(weight.sum().item())


def _forward_with_hierarchical_occupancy(
    model: EgomotionBevJepa,
    batch: Mapping[str, torch.Tensor],
    *,
    nominal_delta_table: torch.Tensor,
    unknown_known_weights: torch.Tensor,
    free_occupied_weights: torch.Tensor,
) -> dict[str, Any]:
    """Run the model through the only occupancy objective allowed by this trainer."""

    commanded_delta = batch["action"] @ nominal_delta_table
    wrong_action = torch.roll(batch["action"], shifts=1, dims=1)
    wrong_action_delta = wrong_action @ nominal_delta_table
    return model(
        batch["current_image"],
        batch["next_image"],
        batch["action"],
        batch["delta"],
        commanded_delta_pose_current=commanded_delta,
        current_occupancy=batch["current_labels"],
        next_occupancy=batch["next_labels"],
        current_occupancy_mask=batch["current_mask"],
        next_occupancy_mask=batch["next_mask"],
        next_prediction_mask=batch["next_observed_mask"],
        occupancy_class_weights=None,
        occupancy_unknown_known_weights=unknown_known_weights,
        occupancy_free_occupied_weights=free_occupied_weights,
        diagnostic_wrong_action=wrong_action,
        diagnostic_wrong_action_delta_pose_current=wrong_action_delta,
    )


def _train_one_epoch(
    model: EgomotionBevJepa,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    unknown_known_weights: torch.Tensor,
    free_occupied_weights: torch.Tensor,
    nominal_delta_table: torch.Tensor,
    gradient_clip: float,
    epoch: int,
) -> dict[str, float]:
    """Train one epoch with the preregistered hierarchical occupancy loss."""

    model.train()
    nominal_delta_table = nominal_delta_table.to(device=device, dtype=torch.float32)
    unknown_known_weights = unknown_known_weights.to(device)
    free_occupied_weights = free_occupied_weights.to(device)
    running: Counter[str] = Counter()
    steps = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = _forward_with_hierarchical_occupancy(
            model,
            batch,
            nominal_delta_table=nominal_delta_table,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
        )
        if not torch.isfinite(output["loss"]):
            raise FloatingPointError(
                f"non-finite loss at epoch={epoch} step={steps}"
            )
        output["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        model.update_target_encoder()
        steps += 1
        for name in (
            "loss",
            "jepa_loss",
            "occupancy_loss",
            "equivariance_loss",
            "action_contrast_loss",
            "variance_loss",
        ):
            running[name] += float(output[name].detach().item())
    if steps == 0:
        raise ValueError("training loader produced no batches")
    return {name: value / steps for name, value in sorted(running.items())}


def apply_vector_calibration(
    logits: torch.Tensor,
    log_scales: torch.Tensor,
    biases: torch.Tensor,
) -> torch.Tensor:
    """Apply positive per-class scales and centered biases along class dim 1."""

    if logits.ndim < 2 or logits.shape[1] != 3:
        raise ValueError("logits must have shape (N, 3, ...)")
    if log_scales.shape != (3,) or biases.shape != (3,):
        raise ValueError("calibration vectors must have shape (3,)")
    scale_shape = (1, 3) + (1,) * (logits.ndim - 2)
    scales = torch.exp(log_scales.clamp(-3.0, 3.0)).reshape(scale_shape)
    centered_biases = (biases - biases.mean()).reshape(scale_shape)
    return logits.float() * scales + centered_biases


def fit_vector_calibration(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    maximum_iterations: int = 80,
) -> dict[str, Any]:
    """Fit six held-out calibration parameters with unweighted multiclass NLL."""

    logits = logits.detach().float().cpu()
    labels = labels.detach().long().cpu()
    if logits.ndim != 2 or logits.shape[1] != 3 or labels.shape != logits.shape[:1]:
        raise ValueError("flat calibration logits/labels have incompatible shapes")
    if logits.shape[0] < 3 or set(torch.unique(labels).tolist()) != {0, 1, 2}:
        raise ValueError("calibration sample must contain all three classes")
    log_scales = torch.zeros(3, requires_grad=True)
    biases = torch.zeros(3, requires_grad=True)
    before_nll = float(torch.nn.functional.cross_entropy(logits, labels).item())
    optimizer = torch.optim.LBFGS(
        (log_scales, biases),
        lr=0.5,
        max_iter=int(maximum_iterations),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        calibrated = apply_vector_calibration(logits, log_scales, biases)
        loss = torch.nn.functional.cross_entropy(calibrated, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        log_scales.clamp_(-3.0, 3.0)
        biases.sub_(biases.mean())
        after_nll = float(
            torch.nn.functional.cross_entropy(
                apply_vector_calibration(logits, log_scales, biases), labels
            ).item()
        )
    if not math.isfinite(after_nll) or after_nll > before_nll + 1e-6:
        raise RuntimeError("held-out vector calibration failed to improve NLL")
    return {
        "method": "positive_diagonal_vector_scaling_with_centered_bias",
        "log_scales": log_scales.detach().tolist(),
        "biases": biases.detach().tolist(),
        "sample_count": int(labels.numel()),
        "nll_before": before_nll,
        "nll_after": after_nll,
    }


@torch.no_grad()
def collect_all_calibration_cells(
    model: EgomotionBevJepa,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Collect every valid calibration-role cell without balancing or backfill."""

    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    source_cell_count = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        logits = model.occupancy_logits(batch["next_image"]).float()
        flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, 3)
        flat_labels = batch["next_labels"].reshape(-1)
        flat_mask = batch["next_mask"].reshape(-1)
        if flat_logits.shape[0] != flat_labels.numel() or flat_mask.shape != flat_labels.shape:
            raise ValueError("calibration logits, labels, and masks have incompatible shapes")
        source_cell_count += int(flat_labels.numel())
        logits_parts.append(flat_logits[flat_mask].detach().cpu())
        label_parts.append(flat_labels[flat_mask].detach().cpu())
    if not logits_parts:
        raise ValueError("calibration loader produced no batches")
    logits = torch.cat(logits_parts, dim=0).contiguous()
    labels = torch.cat(label_parts, dim=0).long().contiguous()
    counts = torch.bincount(labels, minlength=3)[:3]
    if logits.shape != (labels.numel(), 3):
        raise ValueError("collected calibration cells have incompatible shapes")
    if bool((counts == 0).any().item()):
        raise ValueError("calibration role must contain UNKNOWN, FREE, and OCCUPIED")
    return logits, labels, {
        "schema": "lewm_go2_probability_calibration_all_cells_v1",
        "method": "all_valid_cells_in_loader_order",
        "source_cell_count": source_cell_count,
        "valid_cell_count": int(labels.numel()),
        "masked_out_cell_count": source_cell_count - int(labels.numel()),
        "class_counts": {
            "unknown": int(counts[0]),
            "free": int(counts[1]),
            "occupied": int(counts[2]),
        },
        "subsampling": "none",
        "balancing": "none",
        "rare_class_backfill_allowed": False,
        "backfilled_classes": [],
        "appended_cell_count": 0,
        "replacement_count": 0,
    }


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


@torch.no_grad()
def collect_calibration_sample(
    model: EgomotionBevJepa,
    loader: DataLoader,
    *,
    device: torch.device,
    maximum_cells: int = 250_000,
    allow_rare_class_backfill: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Uniformly subsample held-out cells and retain every available class.

    The global stride preserves the calibration role's empirical class
    distribution. If that stride misses an available rare class, one sampled
    cell is deterministically appended or replaced with the first source cell
    of that class. This changes at most one cell per missed class and avoids
    balancing the sample, which would change the class prior fitted by NLL.

    Promotion calibration forbids that backfill: with
    ``allow_rare_class_backfill=False`` a class missed by the stride raises
    instead, because the preregistered G2 contract requires the calibration
    role itself to carry enough support for every class.
    """

    class_names = ("unknown", "free", "occupied")
    if maximum_cells < len(class_names):
        raise ValueError(
            "maximum_cells must be at least three so every occupancy class "
            "can be represented"
        )
    model.eval()
    total_cells = len(loader.dataset) * model.bev_size[0] * model.bev_size[1]
    stride = max(1, math.ceil(total_cells / maximum_cells))
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    source_counts = torch.zeros(3, dtype=torch.long)
    first_source_logits: dict[int, torch.Tensor] = {}
    offset = 0
    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        logits = model.occupancy_logits(batch["next_image"]).float()
        flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, 3)
        flat_labels = batch["next_labels"].reshape(-1)
        flat_mask = batch["next_mask"].reshape(-1)
        if flat_logits.shape[0] != flat_labels.numel() or flat_mask.shape != flat_labels.shape:
            raise ValueError("calibration logits, labels, and masks have incompatible shapes")
        source_labels = flat_labels[flat_mask]
        if source_labels.numel() > 0:
            if bool(((source_labels < 0) | (source_labels > 2)).any().item()):
                raise ValueError("calibration labels must be UNKNOWN/FREE/OCCUPIED")
            source_counts += torch.bincount(
                source_labels.detach().cpu(), minlength=3
            )[:3]
            for class_index in range(3):
                if class_index in first_source_logits:
                    continue
                candidates = torch.nonzero(
                    flat_mask & (flat_labels == class_index), as_tuple=False
                ).flatten()
                if candidates.numel() > 0:
                    first_source_logits[class_index] = (
                        flat_logits[int(candidates[0].item())].detach().cpu().clone()
                    )
        indices = torch.arange(flat_labels.numel(), device=device) + offset
        selected = flat_mask & (indices.remainder(stride) == 0)
        logits_parts.append(flat_logits[selected].cpu())
        label_parts.append(flat_labels[selected].cpu())
        offset += flat_labels.numel()
    if not logits_parts:
        raise ValueError("calibration loader produced no batches")
    if offset != total_cells:
        raise ValueError(
            "calibration loader cell count differs from dataset/model grid contract: "
            f"expected={total_cells}, observed={offset}"
        )

    source_class_counts = {
        name: int(source_counts[index]) for index, name in enumerate(class_names)
    }
    missing_source = [
        class_names[index] for index in range(3) if int(source_counts[index]) == 0
    ]
    if missing_source:
        raise ValueError(
            "probability-calibration role lacks required occupancy classes: "
            f"missing={missing_source}, source_class_counts={source_class_counts}; "
            "rebuild the dataset or deterministically assign a calibration role "
            "containing all three classes"
        )

    logits = torch.cat(logits_parts)[:maximum_cells]
    labels = torch.cat(label_parts)[:maximum_cells].long()
    uniform_counts = torch.bincount(labels, minlength=3)[:3]
    uniform_sample_count = int(labels.numel())
    backfilled_classes: list[str] = []
    replacement_count = 0
    append_count = 0
    for class_index in range(3):
        if bool((labels == class_index).any().item()):
            continue
        if not allow_rare_class_backfill:
            raise RuntimeError(
                "promotion calibration forbids rare-class backfill, but the "
                f"uniform stride sample lacks {class_names[class_index]!r} "
                f"(source_class_counts={source_class_counts}, "
                f"stride={stride}, sample={uniform_sample_count}); increase "
                "--max-calibration-cells or rebuild a calibration role with "
                "adequate class support"
            )
        backfilled_classes.append(class_names[class_index])
        fallback_logit = first_source_logits[class_index].reshape(1, 3)
        fallback_label = torch.tensor([class_index], dtype=torch.long)
        if labels.numel() < maximum_cells:
            logits = torch.cat((logits, fallback_logit), dim=0)
            labels = torch.cat((labels, fallback_label), dim=0)
            append_count += 1
            continue

        sample_counts = torch.bincount(labels, minlength=3)[:3]
        expected = source_counts.to(torch.float64)
        expected *= float(labels.numel()) / float(source_counts.sum())
        donors = [index for index in range(3) if int(sample_counts[index]) > 1]
        if not donors:
            raise RuntimeError("cannot backfill a calibration class without dropping one")
        donor = max(
            donors,
            key=lambda index: (
                float(sample_counts[index]) - float(expected[index]),
                int(sample_counts[index]),
                -index,
            ),
        )
        replace_at = int(torch.nonzero(labels == donor, as_tuple=False)[-1].item())
        logits[replace_at] = fallback_logit[0]
        labels[replace_at] = class_index
        replacement_count += 1

    final_counts = torch.bincount(labels, minlength=3)[:3]
    if bool((final_counts == 0).any().item()):
        raise RuntimeError("calibration class backfill failed")
    sampling = {
        "schema": "lewm_go2_probability_calibration_sampling_v1",
        "method": "global_uniform_stride_with_minimal_missing_class_backfill",
        "rare_class_backfill_allowed": bool(allow_rare_class_backfill),
        "maximum_cells": int(maximum_cells),
        "source_cell_count": int(source_counts.sum()),
        "source_class_counts": source_class_counts,
        "uniform_stride": int(stride),
        "uniform_sample_count": uniform_sample_count,
        "uniform_sample_class_counts": {
            name: int(uniform_counts[index]) for index, name in enumerate(class_names)
        },
        "backfilled_classes": backfilled_classes,
        "appended_cell_count": append_count,
        "replaced_cell_count": replacement_count,
        "final_sample_count": int(labels.numel()),
        "final_sample_class_counts": {
            name: int(final_counts[index]) for index, name in enumerate(class_names)
        },
    }
    return logits, labels, sampling


def _cross_sample_effective_rank(features: torch.Tensor) -> float:
    """Effective channel rank after removing every cell's fixed template."""

    if features.shape[0] < 2:
        return 0.0
    centered = features.float() - features.float().mean(dim=0, keepdim=True)
    samples = centered.permute(0, 2, 3, 1).reshape(-1, centered.shape[1])
    if samples.shape[0] > 65_536:
        stride = math.ceil(samples.shape[0] / 65_536)
        samples = samples[::stride]
    covariance = samples.T @ samples / max(1, samples.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    total = eigenvalues.sum()
    if not bool((total > 0).item()):
        return 0.0
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(torch.exp(entropy).item())


def _nearest_free_start(
    free: np.ndarray,
    forward_axis: np.ndarray,
    left_axis: np.ndarray,
) -> tuple[int, int] | None:
    cells = np.argwhere(free)
    if cells.size == 0:
        return None
    distances = (
        np.square(forward_axis[cells[:, 0]])
        + np.square(left_axis[cells[:, 1]])
    )
    row, col = cells[int(np.argmin(distances))]
    return int(row), int(col)


def _farthest_path(
    admitted: np.ndarray,
    sample_index: int,
    *,
    start: tuple[int, int],
) -> list[tuple[int, int, int]] | None:
    """Return one deterministic longest BFS path from a shared free start."""

    height, width = admitted.shape
    if not admitted[start]:
        return None
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    farthest = start
    while queue:
        cell = queue.popleft()
        farthest = cell
        row, col = cell
        for nxt in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
            nr, nc = nxt
            if 0 <= nr < height and 0 <= nc < width and admitted[nxt] and nxt not in parent:
                parent[nxt] = cell
                queue.append(nxt)
    path = []
    cursor: tuple[int, int] | None = farthest
    while cursor is not None:
        path.append((sample_index, cursor[0], cursor[1]))
        cursor = parent[cursor]
    path.reverse()
    return path


@torch.no_grad()
def evaluate_model(
    model: EgomotionBevJepa,
    loader: DataLoader,
    *,
    device: torch.device,
    unknown_known_weights: torch.Tensor,
    free_occupied_weights: torch.Tensor,
    nominal_delta_table: torch.Tensor,
    calibration: Mapping[str, Any] | None,
    thresholds: TraversabilityThresholds | None,
    select_thresholds: bool,
    occupancy_target_space: str,
) -> dict[str, Any]:
    if occupancy_target_space not in (
        CONFIGURATION_OCCUPANCY_TARGET_SPACE,
        PHYSICAL_OCCUPANCY_TARGET_SPACE,
    ):
        raise ValueError(
            f"unsupported occupancy target space: {occupancy_target_space!r}"
        )
    model.eval()
    nominal_delta_table = nominal_delta_table.to(device=device, dtype=torch.float32)
    unknown_known_weights = unknown_known_weights.to(device)
    free_occupied_weights = free_occupied_weights.to(device)
    if nominal_delta_table.shape != (model.action_dim, 3):
        raise ValueError("nominal_delta_table must have shape (action_dim, 3)")
    calibration_log_scales = torch.zeros(3, device=device)
    calibration_biases = torch.zeros(3, device=device)
    hierarchical_parameters: HierarchicalCalibrationParameters | None = None
    if calibration is not None:
        if calibration.get("method") == HIERARCHICAL_CALIBRATION_METHOD:
            hierarchical_parameters = validate_hierarchical_probability_calibration(
                calibration
            )
        else:
            calibration_log_scales = torch.tensor(
                calibration["log_scales"], device=device, dtype=torch.float32
            )
            calibration_biases = torch.tensor(
                calibration["biases"], device=device, dtype=torch.float32
            )
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    loss_sums: Counter[str] = Counter()
    batches = 0
    panel_sums = {"observed": Counter(), "changed": Counter()}
    target_std_values: list[float] = []
    target_effective_ranks: list[float] = []
    realized_equivariance_sum = 0.0
    realized_equivariance_count = 0

    def add_pair(
        values: Counter[str],
        prefix: str,
        actual: torch.Tensor,
        persistence: torch.Tensor,
        alternative: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        actual_sum, count = _masked_normalized_error(actual, target, mask)
        persistence_sum, _ = _masked_normalized_error(persistence, target, mask)
        alternative_sum, _ = _masked_normalized_error(alternative, target, mask)
        values[f"{prefix}_actual_sum"] += actual_sum
        values[f"{prefix}_persistence_sum"] += persistence_sum
        values[f"{prefix}_alternative_sum"] += alternative_sum
        values[f"{prefix}_count"] += count

    for raw_batch in loader:
        batch = _to_device(raw_batch, device)
        commanded_delta = batch["action"] @ nominal_delta_table
        output = _forward_with_hierarchical_occupancy(
            model,
            batch,
            nominal_delta_table=nominal_delta_table,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
        )
        batches += 1
        for name in (
            "loss",
            "jepa_loss",
            "occupancy_loss",
            "equivariance_loss",
            "action_contrast_loss",
            "variance_loss",
        ):
            loss_sums[name] += float(output[name].item())
        if hierarchical_parameters is None:
            calibrated_logits = apply_vector_calibration(
                output["next_occupancy_logits"],
                calibration_log_scales,
                calibration_biases,
            )
            next_probs = torch.softmax(calibrated_logits, dim=1)
        else:
            next_probs = hierarchical_calibrated_probabilities(
                output["next_occupancy_logits"],
                hierarchical_parameters,
                class_dim=1,
            )
        probabilities.append(next_probs.cpu().numpy())
        labels.append(batch["next_labels"].cpu().numpy())
        masks.append(batch["next_mask"].cpu().numpy())

        wrong_delta_prediction, wrong_delta_warp, wrong_delta_overlap = (
            model.predict_from_command(
                output["current_bev"],
                batch["action"],
                -commanded_delta,
            )
        )
        current_one_hot = torch.nn.functional.one_hot(
            batch["current_labels"].long(), num_classes=3
        ).permute(0, 3, 1, 2).float()
        warped_current_labels, realized_label_overlap = warp_bev_current_to_next(
            current_one_hot,
            batch["delta"],
            forward_range_m=model.forward_range_m,
            left_range_m=model.left_range_m,
        )
        warped_current_observed, _ = warp_bev_current_to_next(
            batch["current_observed_mask"][:, None].float(),
            batch["delta"],
            forward_range_m=model.forward_range_m,
            left_range_m=model.left_range_m,
        )
        previous_known = warped_current_observed[:, 0] >= 0.5
        previous_label = warped_current_labels.argmax(dim=1)
        changed = (
            batch["next_observed_mask"]
            & realized_label_overlap[:, 0]
            & (
                ~previous_known
                | (previous_label != batch["next_labels"])
            )
        )

        for panel_name, panel_mask in (
            ("observed", batch["next_observed_mask"][:, None]),
            ("changed", changed[:, None]),
        ):
            base_mask = output["prediction_valid_mask"] & panel_mask
            actual_sum, count = _masked_normalized_error(
                output["predicted_next_bev"], output["target_next_bev"], base_mask
            )
            persistence_sum, _ = _masked_normalized_error(
                output["commanded_warped_current_bev"],
                output["target_next_bev"],
                base_mask,
            )
            panel_sums[panel_name]["actual_sum"] += actual_sum
            panel_sums[panel_name]["persistence_sum"] += persistence_sum
            panel_sums[panel_name]["count"] += count
            add_pair(
                panel_sums[panel_name],
                "zero_action",
                output["predicted_next_bev"],
                output["commanded_warped_current_bev"],
                output["zero_action_predicted_next_bev"],
                output["target_next_bev"],
                base_mask & output["zero_action_matched_mask"],
            )
            add_pair(
                panel_sums[panel_name],
                "shuffled_action",
                output["predicted_next_bev"],
                output["commanded_warped_current_bev"],
                output["wrong_action_predicted_next_bev"],
                output["target_next_bev"],
                base_mask & output["wrong_action_matched_mask"],
            )
            add_pair(
                panel_sums[panel_name],
                "wrong_commanded_delta",
                output["predicted_next_bev"],
                output["commanded_warped_current_bev"],
                wrong_delta_prediction,
                output["target_next_bev"],
                base_mask & wrong_delta_overlap,
            )

        value, count = _masked_normalized_error(
            output["realized_warped_current_bev"],
            output["target_next_bev"],
            output["realized_equivariance_valid_mask"],
        )
        realized_equivariance_sum += value
        realized_equivariance_count += count

        target = output["target_next_bev"].float()
        if target.shape[0] >= 2:
            target_std_values.append(float(target.std(dim=0, unbiased=False).mean().item()))
            target_effective_ranks.append(_cross_sample_effective_rank(target))

    if batches == 0:
        raise ValueError("evaluation loader produced no batches")
    probs = np.concatenate(probabilities)
    truth = np.concatenate(labels)
    valid = np.concatenate(masks)
    rows, _channels, height, width = probs.shape
    forward = np.linspace(*model.forward_range_m, height, dtype=np.float64)
    left = np.linspace(*model.left_range_m, width, dtype=np.float64)
    distances = np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2)
    distances = np.broadcast_to(distances, truth.shape)
    threshold_selection = None
    if select_thresholds:
        occupied_detection_probability_candidates = (
            REGISTERED_PHYSICAL_OCCUPIED_DETECTION_PROBABILITY_CANDIDATES
            if occupancy_target_space == PHYSICAL_OCCUPANCY_TARGET_SPACE
            else (0.5,)
        )
        threshold_selection = select_conservative_thresholds(
            probs,
            truth,
            distances,
            free_probability_candidates=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99),
            occupied_probability_candidates=(0.01, 0.02, 0.05, 0.10, 0.20, 0.35),
            unknown_probability_candidates=(0.01, 0.02, 0.05, 0.10, 0.20, 0.35),
            occupied_detection_probability_candidates=(
                occupied_detection_probability_candidates
            ),
            evaluation_mask=valid,
            minimum_free_precision=0.99,
            minimum_obstacle_exclusion_recall=0.95,
            minimum_obstacle_detection_recall=0.95,
        )
        thresholds = threshold_selection.thresholds
    if thresholds is None:
        raise ValueError("fixed thresholds are required when select_thresholds=False")
    traversability = evaluate_traversability(
        probs,
        truth,
        distances,
        thresholds=thresholds,
        evaluation_mask=valid,
    )
    admitted = (
        (probs[:, FREE_CLASS] >= thresholds.free_probability_min)
        & (probs[:, OCCUPIED_CLASS] <= thresholds.occupied_probability_max)
        & (probs[:, UNKNOWN_CLASS] <= thresholds.unknown_probability_max)
        & valid
    )
    routing: dict[str, Any]
    if occupancy_target_space == PHYSICAL_OCCUPANCY_TARGET_SPACE:
        routing = {
            "schema": "lewm_go2_routing_not_applicable_v1",
            "valid_for_target_space": False,
            "applicability": "not_applicable",
            "excluded_from_gate": True,
            "deferred_to": "G3_post_memory_multi_view_fusion",
            "reason": (
                "single-frame observable physical evidence is not a fused "
                "configuration-space planning map"
            ),
        }
    else:
        paths: list[list[tuple[int, int, int]]] = []
        oracle_paths: list[list[tuple[int, int, int]]] = []
        route_failures = 0
        route_length_recall_sum = 0.0
        for sample in range(rows):
            oracle_free = valid[sample] & (truth[sample] == FREE_CLASS)
            start = _nearest_free_start(oracle_free, forward, left)
            if start is None:
                continue
            oracle_path = _farthest_path(oracle_free, sample, start=start)
            if oracle_path is None or len(oracle_path) < 2:
                continue
            oracle_paths.append(oracle_path)
            path = _farthest_path(admitted[sample], sample, start=start)
            if path is None or len(path) < 2:
                route_failures += 1
            else:
                paths.append(path)
                route_length_recall_sum += min(
                    1.0,
                    (len(path) - 1) / max(1, len(oracle_path) - 1),
                )
        routing = {
            "valid_for_target_space": True,
            "oracle_routable_paths": len(oracle_paths),
            "predicted_connected_paths": len(paths),
            "route_failures": route_failures,
            "route_success_rate": (
                len(paths) / len(oracle_paths) if oracle_paths else 0.0
            ),
            "mean_route_length_recall": (
                route_length_recall_sum / len(oracle_paths)
                if oracle_paths
                else 0.0
            ),
            "planned_path_collision_rate": planned_path_collision_rate(paths, truth),
            "oracle_map_collision_rate": planned_path_collision_rate(
                oracle_paths, truth
            ),
        }

    def finalize_panel(values: Counter[str]) -> dict[str, float | int]:
        count = int(values["count"])
        persistence_sum = float(values["persistence_sum"])
        result: dict[str, float | int] = {
            "valid_cells": count,
            "normalized_error": float(values["actual_sum"]) / max(1, count),
            "warped_persistence_error": persistence_sum / max(1, count),
            "prediction_to_warped_persistence_ratio": (
                float(values["actual_sum"]) / max(1e-12, persistence_sum)
            ),
        }
        for prefix in (
            "zero_action",
            "shuffled_action",
            "wrong_commanded_delta",
        ):
            prefix_count = int(values[f"{prefix}_count"])
            prefix_persistence = float(values[f"{prefix}_persistence_sum"])
            advantage = float(values[f"{prefix}_alternative_sum"]) - float(
                values[f"{prefix}_actual_sum"]
            )
            result[f"{prefix}_valid_cells"] = prefix_count
            result[f"{prefix}_advantage_over_target_change"] = (
                advantage / max(1e-12, prefix_persistence)
            )
            result[f"{prefix}_error"] = float(
                values[f"{prefix}_alternative_sum"]
            ) / max(1, prefix_count)
        return result

    predictive_panels = {
        name: finalize_panel(values) for name, values in panel_sums.items()
    }
    metrics = {
        "rows": int(rows),
        "occupancy_target_space": str(occupancy_target_space),
        "losses": {name: value / batches for name, value in sorted(loss_sums.items())},
        "traversability": traversability.to_dict(),
        "thresholds": asdict(thresholds),
        "threshold_selection": (
            None
            if threshold_selection is None
            else {
                "candidate_count": threshold_selection.candidate_count,
                "passing_candidate_count": threshold_selection.passing_candidate_count,
                "selection_metrics": threshold_selection.metrics.to_dict(),
            }
        ),
        "calibration": (
            {
                "applied": calibration is not None,
                "id": None if calibration is None else calibration.get("id"),
                "method": None if calibration is None else calibration["method"],
                "log_scales": calibration_log_scales.detach().cpu().tolist(),
                "biases": calibration_biases.detach().cpu().tolist(),
            }
            if hierarchical_parameters is None
            else {
                "applied": True,
                "id": calibration["id"],
                "method": calibration["method"],
                "parameters": calibration["parameters"],
            }
        ),
        "predictive_controls": {
            "panels": predictive_panels,
            "realized_odometry_equivariance_error": (
                realized_equivariance_sum / max(1, realized_equivariance_count)
            ),
            "target_cross_sample_std_mean": (
                float(np.mean(target_std_values)) if target_std_values else 0.0
            ),
            "target_cross_sample_effective_rank_mean": (
                float(np.mean(target_effective_ranks))
                if target_effective_ranks
                else 0.0
            ),
        },
        "routing": routing,
    }
    if occupancy_target_space == PHYSICAL_OCCUPANCY_TARGET_SPACE:
        metrics["physical_evidence"] = {
            "schema": "lewm_go2_observable_physical_evidence_metrics_v1",
            "admitted_observable_physical_free_precision": float(
                traversability.planner_admitted_free_precision
            ),
            "directly_observable_physical_obstacle_recall_within_2m": float(
                traversability.obstacle_detection_recall_within_range
            ),
            "useful_observable_physical_free_recall": float(
                traversability.useful_traversable_recall
            ),
            "observable_physical_obstacle_exclusion_recall_within_2m": float(
                traversability.obstacle_exclusion_recall_within_range
            ),
            "unknown_evidence_admission_rate": float(
                traversability.unknown_admission_rate
            ),
            "free_probability_brier": float(traversability.free_probability_brier),
            "free_probability_ece": float(traversability.free_probability_ece),
        }
        metrics["g2"] = evaluate_physical_evidence_g2_gate(metrics)
    else:
        metrics["g2"] = evaluate_g2_gate(metrics)
    return metrics


def evaluate_physical_evidence_g2_gate(
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    if metrics.get("occupancy_target_space") != PHYSICAL_OCCUPANCY_TARGET_SPACE:
        raise ValueError("physical G2 requires observable physical occupancy")
    physical = metrics.get("physical_evidence")
    if not isinstance(physical, Mapping):
        raise ValueError("physical G2 metrics lack physical-evidence measurements")
    checks = {
        "heldout_probability_calibration_applied": bool(
            metrics["calibration"]["applied"]
        ),
        "admitted_observable_physical_free_precision_ge_0_99": float(
            physical["admitted_observable_physical_free_precision"]
        )
        >= 0.99,
        "directly_observable_physical_obstacle_recall_within_2m_ge_0_95": float(
            physical[
                "directly_observable_physical_obstacle_recall_within_2m"
            ]
        )
        >= 0.95,
        "useful_observable_physical_free_recall_ge_0_90": float(
            physical["useful_observable_physical_free_recall"]
        )
        >= 0.90,
    }
    return {
        "schema": PHYSICAL_G2_GATE_SCHEMA,
        "target_space": PHYSICAL_OCCUPANCY_TARGET_SPACE,
        "routing_included": False,
        "passes": all(checks.values()),
        "checks": checks,
    }


def evaluate_g2_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    traversability = metrics["traversability"]
    predictive = metrics["predictive_controls"]
    routing = metrics["routing"]
    observed = predictive["panels"]["observed"]
    changed = predictive["panels"]["changed"]
    checks = {
        "heldout_probability_calibration_applied": bool(
            metrics["calibration"]["applied"]
        ),
        "planner_admitted_free_precision_ge_0_99": float(
            traversability["planner_admitted_free_precision"]
        ) >= 0.99,
        "obstacle_recall_within_2m_ge_0_95": float(
            traversability["obstacle_detection_recall_within_range"]
        ) >= 0.95,
        "obstacle_exclusion_within_2m_ge_0_95": float(
            traversability["obstacle_exclusion_recall_within_range"]
        ) >= 0.95,
        "useful_traversable_recall_ge_0_90": float(
            traversability["useful_traversable_recall"]
        ) >= 0.90,
        "predicted_routes_no_worse_than_oracle": float(
            routing["planned_path_collision_rate"]
        ) <= float(routing["oracle_map_collision_rate"]),
        "nonzero_oracle_route_panel": int(routing["oracle_routable_paths"]) > 0,
        "predicted_route_success_rate_ge_0_90": float(
            routing["route_success_rate"]
        ) >= 0.90,
        "predicted_route_length_recall_ge_0_90": float(
            routing["mean_route_length_recall"]
        ) >= 0.90,
        "observed_predictor_beats_warped_persistence": float(
            observed["prediction_to_warped_persistence_ratio"]
        ) < 1.0,
        "observed_target_change_is_nontrivial": float(
            observed["warped_persistence_error"]
        ) > 1e-4,
        "changed_predictor_beats_warped_persistence": (
            int(changed["valid_cells"]) > 0
            and float(changed["prediction_to_warped_persistence_ratio"]) < 1.0
        ),
        "changed_target_change_is_nontrivial": float(
            changed["warped_persistence_error"]
        ) > 1e-4,
        "observed_real_action_beats_zero_by_0_10": float(
            observed["zero_action_advantage_over_target_change"]
        ) >= 0.10,
        "observed_real_action_beats_shuffled_by_0_10": float(
            observed["shuffled_action_advantage_over_target_change"]
        ) >= 0.10,
        "changed_real_action_beats_zero_by_0_10": float(
            changed["zero_action_advantage_over_target_change"]
        ) >= 0.10,
        "changed_real_action_beats_shuffled_by_0_10": float(
            changed["shuffled_action_advantage_over_target_change"]
        ) >= 0.10,
        "wrong_commanded_delta_is_worse_on_changed_cells": float(
            changed["wrong_commanded_delta_advantage_over_target_change"]
        ) > 0.0,
        "target_representation_not_collapsed": float(
            predictive["target_cross_sample_std_mean"]
        ) >= 0.05,
        "target_effective_rank_ge_4": float(
            predictive["target_cross_sample_effective_rank_mean"]
        ) >= 4.0,
    }
    return {"passes": all(checks.values()), "checks": checks}


def _selection_score(
    metrics: Mapping[str, Any],
    *,
    mode: str = "full_g2_v1",
) -> tuple[float, ...]:
    if mode not in SELECTION_SCORE_MODES:
        raise ValueError(f"unsupported selection score mode {mode!r}")
    traversability = metrics["traversability"]
    checks = metrics["g2"]["checks"]
    if mode == "physical_occupancy_ceiling_v1":
        if metrics.get("occupancy_target_space") != PHYSICAL_OCCUPANCY_TARGET_SPACE:
            raise ValueError(
                "physical occupancy selection requires physical target semantics"
            )
        physical = metrics.get("physical_evidence")
        if not isinstance(physical, Mapping):
            raise ValueError("physical occupancy selection lacks physical evidence")
        physical_check_names = (
            "admitted_observable_physical_free_precision_ge_0_99",
            "directly_observable_physical_obstacle_recall_within_2m_ge_0_95",
            "useful_observable_physical_free_recall_ge_0_90",
        )
        threshold_selection = metrics.get("threshold_selection")
        if not isinstance(threshold_selection, Mapping):
            raise ValueError(
                "occupancy ceiling selection requires a role-local threshold sweep"
            )
        return (
            float(int(threshold_selection["passing_candidate_count"]) > 0),
            float(sum(bool(checks[name]) for name in physical_check_names)),
            float(physical["useful_observable_physical_free_recall"]),
            float(physical["admitted_observable_physical_free_precision"]),
            float(
                physical[
                    "directly_observable_physical_obstacle_recall_within_2m"
                ]
            ),
            float(
                physical[
                    "observable_physical_obstacle_exclusion_recall_within_2m"
                ]
            ),
            -float(metrics["losses"]["occupancy_loss"]),
        )
    if mode == "occupancy_ceiling_v1":
        occupancy_check_names = (
            "planner_admitted_free_precision_ge_0_99",
            "obstacle_exclusion_within_2m_ge_0_95",
            "obstacle_recall_within_2m_ge_0_95",
            "useful_traversable_recall_ge_0_90",
        )
        threshold_selection = metrics.get("threshold_selection")
        if not isinstance(threshold_selection, Mapping):
            raise ValueError(
                "occupancy ceiling selection requires a role-local threshold sweep"
            )
        return (
            float(int(threshold_selection["passing_candidate_count"]) > 0),
            float(sum(bool(checks[name]) for name in occupancy_check_names)),
            float(traversability["useful_traversable_recall"]),
            float(traversability["planner_admitted_free_precision"]),
            float(traversability["obstacle_detection_recall_within_range"]),
            float(traversability["obstacle_exclusion_recall_within_range"]),
            -float(metrics["losses"]["occupancy_loss"]),
        )
    predictive = metrics["predictive_controls"]["panels"]["changed"]
    return (
        float(sum(bool(value) for value in checks.values())),
        float(traversability["useful_traversable_recall"]),
        float(traversability["planner_admitted_free_precision"]),
        -float(predictive["prediction_to_warped_persistence_ratio"]),
        -float(metrics["losses"]["loss"]),
    )


def _loader(
    rows: Sequence[Mapping[str, Any]],
    *,
    primitive_to_index: Mapping[str, int],
    image_size: int,
    source_crop_fraction_xy: tuple[float, float],
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = PairedNavigationTorchDataset(
        rows,
        primitive_to_index=primitive_to_index,
        image_size=image_size,
        source_crop_fraction_xy=source_crop_fraction_xy,
    )
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(workers) > 0,
        generator=generator,
        drop_last=bool(shuffle and len(dataset) >= batch_size),
    )


def _model_config(
    args: argparse.Namespace,
    action_dim: int,
    grid: Mapping[str, Any],
    *,
    projective_query_support: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = {
        "image_size": int(args.image_size),
        "patch_size": int(args.patch_size),
        "encoder_dim": int(args.encoder_dim),
        "encoder_depth": int(args.encoder_depth),
        "encoder_heads": int(args.encoder_heads),
        "bev_dim": int(args.bev_dim),
        "bev_size": tuple(map(int, grid["shape"])),
        "forward_range_m": tuple(map(float, grid["forward_center_range_m"])),
        "left_range_m": tuple(map(float, grid["left_center_range_m"])),
        "action_dim": int(action_dim),
        "predictor_hidden_dim": int(args.predictor_hidden_dim),
        "target_ema_momentum": float(args.ema_momentum),
        "jepa_weight": float(args.jepa_weight),
        "occupancy_weight": float(args.occupancy_weight),
        "equivariance_weight": float(args.equivariance_weight),
        "action_contrast_weight": float(args.action_contrast_weight),
        "action_margin_fraction": float(args.action_margin_fraction),
        "variance_weight": float(args.variance_weight),
        "variance_target_std": float(args.variance_target_std),
    }
    if args.bev_lift_type in (
        PROJECTIVE_COLUMN_ATTENTION_LIFT,
        PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
    ):
        config.update(
            {
                "bev_lift_type": args.bev_lift_type,
                "projective_horizontal_fov_deg": float(
                    args.projective_horizontal_fov_deg
                ),
                "projective_vertical_fov_deg": float(
                    args.projective_vertical_fov_deg
                ),
                "projective_camera_xyz_body_m": tuple(
                    map(float, args.projective_camera_xyz_body_m)
                ),
                "projective_camera_rpy_body_rad": tuple(
                    map(float, args.projective_camera_rpy_body_rad)
                ),
                "projective_near_m": float(args.projective_near_m),
                "projective_vertical_anchor_z_body_m": tuple(
                    map(float, args.projective_vertical_anchor_z_body_m)
                ),
                "projective_attention_sigma_tokens": float(
                    args.projective_attention_sigma_tokens
                ),
                "projective_attention_bias_floor": float(
                    args.projective_attention_bias_floor
                ),
            }
        )
        if args.bev_lift_type == PROJECTIVE_FOOTPRINT_ATTENTION_LIFT:
            config.update(
                {
                    "projective_footprint_radius_m": float(
                        args.projective_footprint_radius_m
                    ),
                    "projective_footprint_perimeter_samples": int(
                        args.projective_footprint_perimeter_samples
                    ),
                }
            )
        elif args.bev_lift_type == PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT:
            if not isinstance(projective_query_support, Mapping):
                raise ValueError(
                    "cell-square model config requires projective query support"
                )
            config["projective_output_cell_size_m"] = float(
                projective_query_support["output_cell_size_m"]
            )
    elif projective_query_support is not None:
        raise ValueError("projective query support requires cell-square lift")
    return config


def _validate_lift_arguments(args: argparse.Namespace) -> None:
    projective_names = (
        "projective_horizontal_fov_deg",
        "projective_vertical_fov_deg",
        "projective_camera_xyz_body_m",
        "projective_camera_rpy_body_rad",
        "projective_near_m",
        "projective_vertical_anchor_z_body_m",
    )
    footprint_names = (
        "projective_footprint_radius_m",
        "projective_footprint_perimeter_samples",
    )
    if args.bev_lift_type == GLOBAL_CROSS_ATTENTION_LIFT:
        supplied = [
            name
            for name in (*projective_names, *footprint_names)
            if getattr(args, name) is not None
        ]
        if supplied:
            raise SystemExit(
                "projective camera arguments require "
                "a projective --bev-lift-type: " + repr(supplied)
            )
        if (
            float(args.projective_attention_sigma_tokens) != 1.0
            or float(args.projective_attention_bias_floor) != -6.0
        ):
            raise SystemExit(
                "projective attention tuning requires projective-column lift"
            )
        return
    missing = [name for name in projective_names if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"projective lift arguments are missing: {missing}")
    supplied_footprint = [
        name for name in footprint_names if getattr(args, name) is not None
    ]
    if args.bev_lift_type in (
        PROJECTIVE_COLUMN_ATTENTION_LIFT,
        PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    ):
        if supplied_footprint:
            raise SystemExit(
                "footprint arguments require --bev-lift-type "
                f"{PROJECTIVE_FOOTPRINT_ATTENTION_LIFT}: {supplied_footprint}"
                "; body-footprint arguments are forbidden for this lift type"
            )
    elif args.bev_lift_type == PROJECTIVE_FOOTPRINT_ATTENTION_LIFT:
        missing_footprint = [
            name for name in footprint_names if getattr(args, name) is None
        ]
        if missing_footprint:
            raise SystemExit(
                "projective-footprint lift arguments are missing: "
                f"{missing_footprint}"
            )
        if (
            args.projective_footprint_perimeter_samples is not None
            and int(args.projective_footprint_perimeter_samples)
            != REGISTERED_PROJECTIVE_FOOTPRINT_PERIMETER_SAMPLES
        ):
            raise SystemExit(
                "projective footprint perimeter samples differ from the "
                "preregistered value: expected="
                f"{REGISTERED_PROJECTIVE_FOOTPRINT_PERIMETER_SAMPLES}"
            )
    else:
        raise SystemExit(f"unsupported BEV lift type: {args.bev_lift_type!r}")
    if not args.development_only:
        raise SystemExit(
            "projective lift is development-only until checkpoint/runtime "
            "schema v3 is integrated"
        )


def _validate_execution_protocol(
    args: argparse.Namespace,
    *,
    physical_dataset: bool,
) -> None:
    evaluate_physical_once = bool(args.evaluate_physical_g2_once)
    if evaluate_physical_once:
        raise SystemExit(
            "the one-shot G2 path is evaluation-only and must not enter training"
        )
    if physical_dataset:
        if args.selection_score_mode != "physical_occupancy_ceiling_v1":
            raise SystemExit(
                "observable physical dataset v3 requires "
                "--selection-score-mode physical_occupancy_ceiling_v1"
            )
        if not args.development_only:
            raise SystemExit(
                "observable-physical training is development-only; evaluate a "
                "frozen checkpoint with --evaluate-physical-g2-once"
            )
    else:
        if evaluate_physical_once:
            raise SystemExit(
                "--evaluate-physical-g2-once requires observable physical dataset v3"
            )
        if args.selection_score_mode == "physical_occupancy_ceiling_v1":
            raise SystemExit(
                "physical_occupancy_ceiling_v1 requires observable physical dataset v3"
            )
        if args.selection_score_mode != "full_g2_v1" and not args.development_only:
            raise SystemExit(
                "non-G2 checkpoint selection modes require --development-only"
            )
    if (
        getattr(args, "bev_lift_type", GLOBAL_CROSS_ATTENTION_LIFT)
        == PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT
        and not physical_dataset
    ):
        raise SystemExit(
            "projective-cell-square lift requires observable physical dataset v3"
        )
    if (
        args.probability_calibration_mode == "hierarchical_log_odds_v1"
        and not args.development_only
    ):
        raise SystemExit(
            "hierarchical calibration is development-only until checkpoint/runtime "
            "schema v3 is integrated"
        )


def _artifact_schemas(
    *,
    physical_dataset: bool,
    bev_lift_type: str,
    probability_calibration_mode: str,
) -> tuple[str, str]:
    if physical_dataset:
        return PHYSICAL_CHECKPOINT_SCHEMA, PHYSICAL_REPORT_SCHEMA
    legacy_v2 = (
        bev_lift_type == GLOBAL_CROSS_ATTENTION_LIFT
        and probability_calibration_mode == "vector_scaling_v1"
    )
    return (
        "lewm_go2_egomotion_bev_jepa_checkpoint_v2"
        if legacy_v2
        else "lewm_go2_egomotion_bev_jepa_checkpoint_v3",
        "lewm_go2_egomotion_bev_jepa_training_report_v2"
        if legacy_v2
        else "lewm_go2_egomotion_bev_jepa_training_report_v3",
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _vertical_fov_from_horizontal(
    horizontal_fov_deg: float,
    native_resolution: Sequence[int],
) -> float:
    if len(native_resolution) != 2:
        raise ValueError("native camera resolution must contain width and height")
    width, height = (int(value) for value in native_resolution)
    if width <= 0 or height <= 0:
        raise ValueError("native camera resolution must be positive")
    horizontal = float(horizontal_fov_deg)
    if not math.isfinite(horizontal) or not 0.0 < horizontal < 180.0:
        raise ValueError("horizontal camera FOV must lie in (0, 180)")
    return math.degrees(
        2.0
        * math.atan(
            math.tan(math.radians(horizontal) * 0.5)
            * float(height)
            / float(width)
        )
    )


def _source_fov_rectification_contract(
    *,
    mode: str,
    intended_horizontal_fov_deg: float,
    native_resolution: Sequence[int],
) -> dict[str, Any]:
    """Resolve a source-only crop that restores the platform pinhole FOV."""

    if mode not in SOURCE_FOV_RECTIFICATION_MODES:
        raise ValueError(f"unsupported source FOV rectification mode: {mode!r}")
    width, height = (int(value) for value in native_resolution)
    intended_vertical = _vertical_fov_from_horizontal(
        intended_horizontal_fov_deg,
        (width, height),
    )
    if mode == "none":
        crop_fraction_xy = (1.0, 1.0)
        source_horizontal = float(intended_horizontal_fov_deg)
        source_vertical = float(intended_vertical)
        source_contract = "platform_horizontal_fov_rendered_with_correct_aspect"
    else:
        if width <= height:
            raise ValueError(
                "legacy v03 rectification requires a landscape native camera"
            )
        # v03 rendered a square camera and passed the manifest horizontal FOV
        # to Genesis' vertical-FOV API. The square source therefore has H=V;
        # retaining full width and cropping height/native-width restores the
        # platform view after the normal square encoder resize.
        crop_fraction_xy = (1.0, float(height) / float(width))
        source_horizontal = float(intended_horizontal_fov_deg)
        source_vertical = float(intended_horizontal_fov_deg)
        source_contract = "legacy_v03_square_genesis_yfov_equals_manifest_hfov"
    return {
        "schema": "lewm_go2_source_fov_rectification_v1",
        "mode": mode,
        "source_contract": source_contract,
        "native_resolution_wh": [width, height],
        "source_horizontal_fov_deg": source_horizontal,
        "source_vertical_fov_deg": source_vertical,
        "intended_horizontal_fov_deg": float(intended_horizontal_fov_deg),
        "intended_vertical_fov_deg": float(intended_vertical),
        "center_crop_fraction_xy": list(crop_fraction_xy),
        "crop_before_model_resize": True,
        "runtime_crop_required": False,
    }


def _validate_source_camera_contract(
    path: Path,
    *,
    dataset_manifest_path: Path,
    rectification: Mapping[str, Any],
) -> dict[str, Any]:
    artifact_path = path.resolve()
    artifact = _read_json(artifact_path)
    schema = str(artifact.get("schema"))
    if schema not in (
        "lewm_go2_source_camera_contract_v1",
        "lewm_go2_selected_render_audit_v1",
    ):
        raise ValueError("unsupported source camera contract schema")
    core = dict(artifact)
    declared_content_sha = str(core.pop("content_sha256", ""))
    if declared_content_sha != _canonical_json_sha256(core):
        raise ValueError("source camera contract content hash mismatch")
    if schema == "lewm_go2_source_camera_contract_v1":
        dataset_record = artifact.get("dataset_manifest")
        if not isinstance(dataset_record, Mapping) or dataset_record.get(
            "sha256"
        ) != sha256_file(dataset_manifest_path):
            raise ValueError("source camera contract targets a different dataset")
        actual = artifact.get("actual_source_projection")
        platform = artifact.get("platform_projection_after_rectification")
        crop_fraction = (
            platform.get("center_crop_fraction_xy")
            if isinstance(platform, Mapping)
            else None
        )
        if artifact.get("g2_images_opened") is not False:
            raise ValueError("source camera audit must not open G2 images")
    else:
        dataset_manifest = _read_json(dataset_manifest_path)
        bound = dataset_manifest.get("render_audit_contract")
        if (
            not isinstance(bound, Mapping)
            or str(bound.get("file_sha256")) != sha256_file(artifact_path)
            or str(bound.get("content_sha256")) != declared_content_sha
        ):
            raise ValueError("dataset does not bind this v04 render audit")
        source_index = bound.get("output_source_index")
        audited_index = artifact.get("output_source_index")
        if (
            not isinstance(source_index, Mapping)
            or not isinstance(audited_index, Mapping)
            or str(source_index.get("sha256")) != str(audited_index.get("sha256"))
        ):
            raise ValueError("dataset source index differs from the render audit")
        actual = artifact.get("camera_projection")
        platform = actual
        crop_fraction = (1.0, 1.0)
        if (
            artifact.get("g2_image_bytes_hashed_for_integrity") is not True
            or artifact.get("g2_images_decoded_or_inspected") is not False
            or artifact.get("g2_image_content_metrics_computed") is not False
            or artifact.get("g2_label_shards_opened") is not False
            or artifact.get("g2_model_outputs_opened") is not False
        ):
            raise ValueError("v04 render audit touched forbidden G2 evidence")
        object_contract = artifact.get("object_contract")
        if (
            not isinstance(object_contract, Mapping)
            or object_contract.get("collision_distractors_rendered") is not True
            or object_contract.get("full_box_roll_pitch_yaw_rendered") is not True
        ):
            raise ValueError("v04 render audit lacks collision-object parity")
    if not isinstance(actual, Mapping) or not isinstance(platform, Mapping):
        raise ValueError("source camera projection records are missing")
    comparisons = (
        (
            actual.get("horizontal_fov_deg"),
            rectification["source_horizontal_fov_deg"],
            "source horizontal FOV",
        ),
        (
            actual.get("vertical_fov_deg"),
            rectification["source_vertical_fov_deg"],
            "source vertical FOV",
        ),
        (
            platform.get("horizontal_fov_deg"),
            rectification["intended_horizontal_fov_deg"],
            "platform horizontal FOV",
        ),
        (
            platform.get("vertical_fov_deg"),
            rectification["intended_vertical_fov_deg"],
            "platform vertical FOV",
        ),
    )
    for actual_value, expected_value, name in comparisons:
        if not math.isclose(
            float(actual_value), float(expected_value), rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(f"source camera contract changed {name}")
    if not np.allclose(
        np.asarray(crop_fraction, dtype=np.float64),
        np.asarray(rectification["center_crop_fraction_xy"], dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("source camera crop differs from rectification")
    return {
        "schema": schema,
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "content_sha256": declared_content_sha,
        "scene_count": int(artifact["scene_count"]),
    }


def _configure_determinism(enabled: bool) -> dict[str, Any]:
    """Enable strict deterministic execution and return auditable settings."""

    if enabled:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    cudnn_available = bool(torch.backends.cudnn.is_available())
    return {
        "requested": bool(enabled),
        "torch_deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_available": cudnn_available,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "nondeterministic_operation_policy": "error",
    }


def _validate_physical_morphology_contract(
    dataset_label_semantics: Mapping[str, Any],
    checkpoint_output_contract: Mapping[str, Any],
) -> dict[str, Any]:
    dataset_morphology = dataset_label_semantics.get(
        "post_memory_configuration_derivation"
    )
    checkpoint_morphology = checkpoint_output_contract.get(
        "post_memory_configuration_derivation"
    )
    if not isinstance(dataset_morphology, Mapping) or not isinstance(
        checkpoint_morphology, Mapping
    ):
        raise ValueError("physical target lacks post-memory morphology")
    if _canonical_json_sha256(dataset_morphology) != _canonical_json_sha256(
        checkpoint_morphology
    ):
        raise ValueError("checkpoint post-memory morphology differs from dataset")
    if dataset_morphology.get("schema") != (
        "lewm_post_memory_configuration_morphology_v1"
    ):
        raise ValueError("unsupported post-memory morphology schema")
    radius_m = float(dataset_morphology.get("radius_m", float("nan")))
    cell_m = float(dataset_morphology.get("memory_cell_size_m", float("nan")))
    support_sha = str(dataset_morphology.get("support_contract_sha256", ""))
    if (
        not math.isfinite(radius_m)
        or radius_m <= 0.0
        or not math.isfinite(cell_m)
        or cell_m <= 0.0
        or len(support_sha) != 64
        or any(character not in "0123456789abcdef" for character in support_sha)
    ):
        raise ValueError("post-memory morphology parameters are malformed")
    declared_radius = float(
        dataset_label_semantics.get("configuration_inflation_radius_m", float("nan"))
    )
    if not math.isclose(radius_m, declared_radius, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("post-memory morphology radius differs from label semantics")
    return {
        "schema": "lewm_go2_physical_morphology_prerequisite_v1",
        "exact_dataset_checkpoint_match": True,
        "morphology_sha256": _canonical_json_sha256(dataset_morphology),
        "morphology_schema": str(dataset_morphology["schema"]),
        "radius_m": radius_m,
        "memory_cell_size_m": cell_m,
        "support_contract_sha256": support_sha,
    }


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            args,
            cwd=REPOSITORY_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return completed.stdout.strip()

    return {
        "head": run("git", "rev-parse", "HEAD"),
        "status_short": run("git", "status", "--short"),
    }


def _critical_training_inputs(
    dataset_manifest_path: Path,
    manifest: Mapping[str, Any],
    *,
    source_camera_contract_path: Path | None,
) -> dict[str, dict[str, str]]:
    paths = {
        "trainer_source": Path(__file__).resolve(),
        "model_source": REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py",
        "traversability_metrics_source": (
            REPOSITORY_ROOT / "lewm/benchmarks/traversability_metrics.py"
        ),
        "calibration_source": (
            REPOSITORY_ROOT / "lewm/hierarchical_probability_calibration.py"
        ),
        "dataset_contract_source": (
            REPOSITORY_ROOT / "lewm/datasets/go2_paired_navigation.py"
        ),
        "dataset_manifest": dataset_manifest_path.resolve(),
        "dataset_index": Path(str(manifest["index"]["path"])).resolve(),
        "geometry_contract": Path(
            str(manifest["geometry_contract"]["path"])
        ).resolve(),
    }
    if source_camera_contract_path is not None:
        paths["source_camera_contract"] = source_camera_contract_path.resolve()
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _validate_training_run_provenance(
    checkpoint: Mapping[str, Any],
    *,
    dataset_manifest_path: Path,
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    provenance = checkpoint.get("training_run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint lacks embedded training-run provenance")
    core = dict(provenance)
    declared = str(core.pop("content_sha256", ""))
    if declared != _canonical_json_sha256(core):
        raise ValueError("training-run provenance content hash mismatch")
    if provenance.get("schema") != "lewm_go2_training_run_provenance_v1":
        raise ValueError("unsupported training-run provenance schema")
    if provenance.get("checkpoint_artifact_included") is not False:
        raise ValueError("training-run provenance is circular")
    critical = provenance.get("critical_inputs")
    if not isinstance(critical, Mapping):
        raise ValueError("training-run provenance lacks critical inputs")
    expected = _critical_training_inputs(
        dataset_manifest_path,
        manifest,
        source_camera_contract_path=None,
    )
    for name in ("trainer_source", "model_source", "dataset_manifest", "dataset_index"):
        if critical.get(name) != expected[name]:
            raise ValueError(f"training-run provenance changed: {name}")
    return provenance


def _validate_projective_query_support_artifacts(
    checkpoint: Mapping[str, Any],
    dataset_manifest: Mapping[str, Any],
    *,
    report: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    """Fail closed on support metadata according to the recorded lift type."""

    model_config = checkpoint.get("model_config")
    output_contract = checkpoint.get("occupancy_output_contract")
    if not isinstance(model_config, Mapping) or not isinstance(
        output_contract, Mapping
    ):
        raise ValueError("checkpoint lacks model or occupancy-output contract")
    support_value = checkpoint.get("projective_query_support")
    if support_value is not None and not isinstance(support_value, Mapping):
        raise ValueError("checkpoint projective query support is malformed")
    support = validate_projective_query_support_binding(
        model_config=model_config,
        projective_query_support=support_value,
        dataset_manifest=dataset_manifest,
        occupancy_output_contract=output_contract,
    )
    provenance = checkpoint.get("training_run_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("checkpoint lacks training-run provenance")
    if provenance.get("projective_query_support") != support:
        raise ValueError("training provenance projective query support differs")
    if report is not None and report.get("projective_query_support") != support:
        raise ValueError("training report projective query support differs")
    return support


def _evaluate_frozen_physical_g2_once(args: argparse.Namespace) -> int:
    checkpoint_path = args.frozen_physical_checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != args.expected_frozen_checkpoint_sha256:
        raise SystemExit("frozen physical checkpoint SHA-256 mismatch")
    output_path = args.output.resolve()
    report_path = (
        args.report_output.resolve()
        if args.report_output is not None
        else output_path.with_suffix(".report.json")
    )
    if output_path == checkpoint_path:
        raise SystemExit("one-shot G2 output must not replace its frozen checkpoint")
    for path in (output_path, report_path):
        if path.exists():
            raise SystemExit(f"refusing to replace an existing one-shot artifact: {path}")

    dataset_manifest_path = args.dataset_manifest.resolve()
    manifest = _read_json(dataset_manifest_path)
    label_semantics = manifest.get("label_semantics")
    if (
        manifest.get("schema") != "lewm_go2_paired_navigation_dataset_v3"
        or not isinstance(label_semantics, Mapping)
        or label_semantics.get("label_contract")
        != PHYSICAL_DATASET_LABEL_CONTRACT
        or label_semantics.get("target_occupancy_space")
        != PHYSICAL_OCCUPANCY_TARGET_SPACE
        or label_semantics.get("per_frame_configuration_classes_supervised")
        is not False
    ):
        raise SystemExit("one-shot physical G2 requires observable dataset v3")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as exc:  # pragma: no cover - environment contract
        raise SystemExit("one-shot evaluation requires weights_only checkpoint load") from exc
    if not isinstance(checkpoint, Mapping):
        raise SystemExit("frozen physical checkpoint root must be a mapping")
    if checkpoint.get("schema") != PHYSICAL_CHECKPOINT_SCHEMA:
        raise SystemExit("one-shot physical G2 requires checkpoint schema v4")
    if checkpoint.get("dataset_manifest_sha256") != sha256_file(
        dataset_manifest_path
    ):
        raise SystemExit("frozen checkpoint targets a different dataset manifest")
    try:
        _validate_training_run_provenance(
            checkpoint,
            dataset_manifest_path=dataset_manifest_path,
            manifest=manifest,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid frozen training-run provenance: {exc}") from exc
    try:
        _validate_projective_query_support_artifacts(checkpoint, manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid frozen projective query support: {exc}") from exc
    output_contract = checkpoint.get("occupancy_output_contract")
    if (
        not isinstance(output_contract, Mapping)
        or output_contract.get("target_occupancy_space")
        != PHYSICAL_OCCUPANCY_TARGET_SPACE
        or not isinstance(
            output_contract.get("post_memory_configuration_derivation"), Mapping
        )
    ):
        raise SystemExit("frozen checkpoint lacks physical target semantics")
    try:
        morphology_prerequisite = _validate_physical_morphology_contract(
            label_semantics,
            output_contract,
        )
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid physical morphology prerequisite: {exc}") from exc
    if checkpoint.get("selection_score_mode") != "physical_occupancy_ceiling_v1":
        raise SystemExit("frozen checkpoint was not selected by the physical policy")
    if (
        checkpoint.get("g2_evaluation") is not None
        or bool(checkpoint.get("g2_passes", False))
        or checkpoint.get("head_g2_evaluation") is not None
        or bool(checkpoint.get("head_g2_passes", False))
    ):
        raise SystemExit("frozen checkpoint already contains a G2 evaluation")
    if checkpoint.get("runtime_ready") is not False:
        raise SystemExit("physical checkpoint must remain unavailable to runtime")
    provenance = checkpoint.get("dataset_provenance_verification")
    if (
        not isinstance(provenance, Mapping)
        or not isinstance(provenance.get("development"), Mapping)
        or provenance.get("g2_evaluation") is not None
    ):
        raise SystemExit("frozen checkpoint lacks untouched-G2 provenance")

    deterministic_execution = _configure_determinism(args.deterministic)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = _read_rows(Path(manifest["index"]["path"]))
    all_scene_roles = resolve_dataset_scene_roles(
        rows,
        manifest,
        legacy_selection_seed=args.selection_seed,
    )
    g2_rows = [
        row
        for row in rows
        if all_scene_roles[str(row["scene_id"])] == "g2_evaluation"
    ]
    if not g2_rows:
        raise SystemExit("untouched physical G2 role has no rows")
    source_provenance = {
        "git": _git_snapshot(),
        "invocation": {
            "argv": list(sys.argv),
            "mode": "frozen_physical_g2_evaluation_only",
        },
        "sources": {
            "trainer": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "model": {
                "path": str(
                    REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py"
                ),
                "sha256": sha256_file(
                    REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py"
                ),
            },
        },
        "runtime_config": {
            "requested_device": str(args.device),
            "batch_size": int(args.batch_size),
            "workers": int(args.workers),
            "seed": int(args.seed),
            "deterministic_requested": bool(args.deterministic),
            "occupancy_target_space": PHYSICAL_OCCUPANCY_TARGET_SPACE,
            "g2_row_policy": "all_rows_no_subsampling",
        },
    }
    attempt_marker_path = checkpoint_path.with_name(
        checkpoint_path.name + ".physical_g2_attempt.json"
    )
    attempt_core = {
        "schema": "lewm_go2_physical_g2_attempt_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "intent": "open_untouched_physical_g2_exactly_once",
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
        },
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "g2_scene_set_sha256": canonical_json_sha256(
            sorted(
                scene_id_sha256(scene_id)
                for scene_id, role in all_scene_roles.items()
                if role == "g2_evaluation"
            )
        ),
        "g2_row_count": len(g2_rows),
        "morphology_prerequisite": morphology_prerequisite,
        "execution_provenance": source_provenance,
        "status": "attempt_committed_before_g2_byte_open",
    }
    attempt = {
        **attempt_core,
        "content_sha256": _canonical_json_sha256(attempt_core),
    }
    attempt_marker_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with attempt_marker_path.open("x") as stream:
            stream.write(json.dumps(attempt, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise SystemExit(
            "frozen checkpoint already has a physical-G2 attempt marker: "
            f"{attempt_marker_path}"
        ) from exc
    attempt_marker_record = {
        "path": str(attempt_marker_path),
        "sha256": sha256_file(attempt_marker_path),
        "content_sha256": attempt["content_sha256"],
    }
    # First G2 shard/image byte access. The durable attempt marker above must
    # already exist so a crash cannot silently license another one-shot run.
    g2_provenance = _verify_dataset_role_provenance(
        dataset_manifest_path,
        manifest,
        all_scene_roles,
        roles=("g2_evaluation",),
    )

    primitive_to_index = {
        str(name): int(index)
        for name, index in checkpoint["primitive_to_index"].items()
    }
    unseen = {str(row["primitive"]) for row in g2_rows} - set(primitive_to_index)
    if unseen:
        raise SystemExit(f"G2 contains unseen primitives: {sorted(unseen)}")
    model_config = dict(checkpoint["model_config"])
    device = _resolve_device(args.device)
    model = EgomotionBevJepa(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    objective = checkpoint.get("occupancy_training_objective")
    if (
        not isinstance(objective, Mapping)
        or objective.get("mode") != OCCUPANCY_LOSS_MODE
    ):
        raise SystemExit("frozen checkpoint lacks registered occupancy objective")
    terms = objective.get("terms")
    if not isinstance(terms, Mapping):
        raise SystemExit("frozen checkpoint occupancy objective has no terms")
    unknown_known_weights = torch.tensor(
        terms["unknown_vs_known"]["weights"], dtype=torch.float32, device=device
    )
    free_occupied_weights = torch.tensor(
        terms["free_vs_occupied_given_known"]["weights"],
        dtype=torch.float32,
        device=device,
    )
    if unknown_known_weights.shape != (2,) or free_occupied_weights.shape != (2,):
        raise SystemExit("frozen checkpoint occupancy weights are malformed")
    nominal_delta_table = torch.tensor(
        checkpoint["nominal_primitive_delta_current"], dtype=torch.float32
    )
    thresholds = TraversabilityThresholds(
        **dict(checkpoint["traversability_thresholds"])
    )
    thresholds.validate()
    calibration = checkpoint.get("probability_calibration")
    if not isinstance(calibration, Mapping):
        raise SystemExit("frozen checkpoint has no probability calibration")
    rectification = checkpoint.get("source_fov_rectification")
    if not isinstance(rectification, Mapping):
        raise SystemExit("frozen checkpoint has no source camera contract")
    raw_crop = rectification.get("center_crop_fraction_xy")
    if not isinstance(raw_crop, (list, tuple)) or len(raw_crop) != 2:
        raise SystemExit("frozen checkpoint source crop contract is malformed")
    source_crop_fraction_xy = tuple(float(value) for value in raw_crop)
    if source_crop_fraction_xy != (1.0, 1.0):
        raise SystemExit("corrected physical RGB must not be cropped at evaluation")

    g2_loader = _loader(
        g2_rows,
        primitive_to_index=primitive_to_index,
        image_size=int(model_config["image_size"]),
        source_crop_fraction_xy=source_crop_fraction_xy,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    final_g2 = evaluate_model(
        model,
        g2_loader,
        device=device,
        unknown_known_weights=unknown_known_weights,
        free_occupied_weights=free_occupied_weights,
        nominal_delta_table=nominal_delta_table,
        calibration=calibration,
        thresholds=thresholds,
        select_thresholds=False,
        occupancy_target_space=PHYSICAL_OCCUPANCY_TARGET_SPACE,
    )
    head_g2_passes = bool(
        morphology_prerequisite["exact_dataset_checkpoint_match"]
        and final_g2["g2"]["passes"]
    )
    derived = dict(checkpoint)
    derived["dataset_provenance_verification"] = {
        **dict(provenance),
        "g2_evaluation": g2_provenance,
    }
    derived.update(
        {
            "g2_evaluation": None,
            "g2_passes": False,
            "head_g2_evaluation": final_g2,
            "head_g2_passes": head_g2_passes,
            "head_g2_contract_prerequisites": {
                "post_memory_morphology": morphology_prerequisite,
                "attempt_marker": attempt_marker_record,
            },
            "runtime_ready": False,
            "runtime_readiness_reason": "deferred_to_G3_integration",
            "head_g2_source_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            "head_g2_deterministic_execution": deterministic_execution,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(derived, output_path)
    report_core = {
        "schema": "lewm_go2_physical_g2_evaluation_report_v1",
        "source_checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "schema": PHYSICAL_CHECKPOINT_SCHEMA,
        },
        "output_checkpoint": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "schema": PHYSICAL_CHECKPOINT_SCHEMA,
        },
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "g2_role_provenance": g2_provenance,
        "g2_attempt_marker": attempt_marker_record,
        "g2_row_count": len(g2_rows),
        "head_g2_contract_prerequisites": {
            "post_memory_morphology": morphology_prerequisite,
        },
        "execution_provenance": source_provenance,
        "frozen_runtime_contract": {
            "model_config": model_config,
            "projective_query_support": checkpoint.get(
                "projective_query_support"
            ),
            "probability_calibration_id": checkpoint.get(
                "probability_calibration_id"
            ),
            "traversability_thresholds": dict(
                checkpoint["traversability_thresholds"]
            ),
            "occupancy_output_contract": dict(output_contract),
            "runtime_ready": False,
        },
        "final_head_g2_evaluation": final_g2,
        "promotion": {
            "gate": "G2_physical_evidence_head",
            "head_g2_passes": head_g2_passes,
            "head_g2_evaluated": True,
            "runtime_ready": False,
            "runtime_readiness_deferred_to": "G3",
            "frozen_checkpoint_retrained": False,
            "calibration_refit": False,
            "thresholds_reselected": False,
        },
    }
    report = {
        **report_core,
        "content_sha256": _canonical_json_sha256(report_core),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["promotion"], sort_keys=True), flush=True)
    return 0 if head_g2_passes else 2


def main(argv: Sequence[str] | None = None) -> int:
    invocation_argv = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *(str(value) for value in argv)]
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=112)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--encoder-dim", type=int, default=192)
    parser.add_argument("--encoder-depth", type=int, default=6)
    parser.add_argument("--encoder-heads", type=int, default=6)
    parser.add_argument("--bev-dim", type=int, default=64)
    parser.add_argument(
        "--bev-lift-type",
        choices=(
            GLOBAL_CROSS_ATTENTION_LIFT,
            PROJECTIVE_COLUMN_ATTENTION_LIFT,
            PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
            PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
        ),
        default=GLOBAL_CROSS_ATTENTION_LIFT,
    )
    parser.add_argument("--projective-horizontal-fov-deg", type=float)
    parser.add_argument("--projective-vertical-fov-deg", type=float)
    parser.add_argument(
        "--projective-camera-xyz-body-m", type=float, nargs=3, metavar=("X", "Y", "Z")
    )
    parser.add_argument(
        "--projective-camera-rpy-body-rad",
        type=float,
        nargs=3,
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument("--projective-near-m", type=float)
    parser.add_argument(
        "--projective-vertical-anchor-z-body-m", type=float, nargs="+"
    )
    parser.add_argument("--projective-attention-sigma-tokens", type=float, default=1.0)
    parser.add_argument("--projective-attention-bias-floor", type=float, default=-6.0)
    parser.add_argument("--projective-footprint-radius-m", type=float)
    parser.add_argument("--projective-footprint-perimeter-samples", type=int)
    parser.add_argument(
        "--source-fov-rectification-mode",
        choices=SOURCE_FOV_RECTIFICATION_MODES,
        default="none",
        help=(
            "Training-source-only camera rectification. The legacy v03 mode "
            "vertically crops square Genesis frames to the platform pinhole FOV."
        ),
    )
    parser.add_argument(
        "--source-camera-native-resolution",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(640, 480),
    )
    parser.add_argument(
        "--source-camera-contract",
        type=Path,
        help=(
            "Content-addressed renderer audit from "
            "audit_go2_source_camera_contract.py. Required when legacy source "
            "FOV rectification is enabled."
        ),
    )
    parser.add_argument("--predictor-hidden-dim", type=int, default=128)
    parser.add_argument("--ema-momentum", type=float, default=0.996)
    parser.add_argument("--jepa-weight", type=float, default=1.0)
    parser.add_argument("--occupancy-weight", type=float, default=2.0)
    parser.add_argument("--equivariance-weight", type=float, default=0.25)
    parser.add_argument("--action-contrast-weight", type=float, default=1.0)
    parser.add_argument("--action-margin-fraction", type=float, default=0.1)
    parser.add_argument("--variance-weight", type=float, default=0.1)
    parser.add_argument("--variance-target-std", type=float, default=0.5)
    parser.add_argument("--max-train-rows", type=int, default=100000)
    parser.add_argument("--max-selection-rows", type=int, default=12000)
    parser.add_argument("--max-calibration-rows", type=int, default=12000)
    parser.add_argument("--max-g2-rows", type=int, default=12000)
    parser.add_argument("--max-calibration-cells", type=int, default=250000)
    parser.add_argument(
        "--probability-calibration-mode",
        choices=CALIBRATION_MODES,
        default="vector_scaling_v1",
    )
    parser.add_argument("--selection-seed", default="go2_g2_selection_v1")
    parser.add_argument(
        "--selection-score-mode",
        choices=SELECTION_SCORE_MODES,
        default="full_g2_v1",
        help=(
            "Checkpoint ranking contract. occupancy_ceiling_v1 is a "
            "development-only configuration-target interference diagnostic; "
            "physical_occupancy_ceiling_v1 ranks only observable physical "
            "occupancy evidence."
        ),
    )
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Train/calibrate without reading G2 images or claiming promotion.",
    )
    parser.add_argument(
        "--evaluate-physical-g2-once",
        action="store_true",
        help=(
            "Explicitly open the untouched physical-evidence G2 role once after "
            "checkpoint selection, calibration, and threshold freezing. This "
            "promotes only the perception head; runtime remains unavailable."
        ),
    )
    parser.add_argument(
        "--frozen-physical-checkpoint",
        type=Path,
        help=(
            "Development-selected checkpoint-v4 to evaluate without retraining. "
            "Required by --evaluate-physical-g2-once and forbidden otherwise."
        ),
    )
    parser.add_argument(
        "--expected-frozen-checkpoint-sha256",
        help=(
            "Precommitted SHA-256 of --frozen-physical-checkpoint. Required by "
            "the one-shot evaluator so checkpoint selection cannot change."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Require deterministic PyTorch algorithms and fail on unsupported "
            "nondeterministic operations."
        ),
    )
    args = parser.parse_args(argv)

    if args.epochs <= 0 or args.batch_size <= 1:
        raise SystemExit("epochs must be positive and batch-size must exceed one")
    if args.evaluate_physical_g2_once:
        if args.development_only:
            raise SystemExit(
                "--evaluate-physical-g2-once is incompatible with --development-only"
            )
        if args.frozen_physical_checkpoint is None:
            raise SystemExit(
                "--evaluate-physical-g2-once requires --frozen-physical-checkpoint"
            )
        if (
            args.expected_frozen_checkpoint_sha256 is None
            or len(args.expected_frozen_checkpoint_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in args.expected_frozen_checkpoint_sha256
            )
        ):
            raise SystemExit(
                "--evaluate-physical-g2-once requires a lowercase 64-hex "
                "--expected-frozen-checkpoint-sha256"
            )
        return _evaluate_frozen_physical_g2_once(args)
    if (
        args.frozen_physical_checkpoint is not None
        or args.expected_frozen_checkpoint_sha256 is not None
    ):
        raise SystemExit(
            "--frozen-physical-checkpoint requires --evaluate-physical-g2-once"
        )
    deterministic_execution = _configure_determinism(args.deterministic)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_manifest_path = args.dataset_manifest.resolve()
    manifest = _read_json(dataset_manifest_path)
    dataset_schema = str(manifest.get("schema"))
    if dataset_schema not in SUPPORTED_DATASET_SCHEMAS:
        raise SystemExit("unsupported dataset schema")
    label_semantics = manifest.get("label_semantics")
    if not isinstance(label_semantics, Mapping):
        raise SystemExit("dataset manifest has no label-semantics contract")
    occupancy_target_space = str(label_semantics.get("target_occupancy_space", ""))
    physical_dataset = dataset_schema == "lewm_go2_paired_navigation_dataset_v3"
    if physical_dataset:
        if (
            str(label_semantics.get("label_contract"))
            != PHYSICAL_DATASET_LABEL_CONTRACT
            or occupancy_target_space != "observable_physical_occupancy"
            or label_semantics.get("per_frame_configuration_classes_supervised")
            is not False
        ):
            raise SystemExit("dataset v3 is not observable physical occupancy")
    _validate_execution_protocol(args, physical_dataset=physical_dataset)
    _validate_lift_arguments(args)
    projective_query_support = None
    if args.bev_lift_type == PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT:
        try:
            projective_query_support = build_projective_query_support_contract(
                manifest
            )
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid projective query support contract: {exc}") from exc
    geometry_payload = _read_json(Path(manifest["geometry_contract"]["path"]))
    geometry_camera = geometry_payload.get("camera")
    if not isinstance(geometry_camera, Mapping):
        raise SystemExit("geometry contract has no camera mapping")
    try:
        source_fov_rectification = _source_fov_rectification_contract(
            mode=args.source_fov_rectification_mode,
            intended_horizontal_fov_deg=float(
                geometry_camera["horizontal_fov_deg"]
            ),
            native_resolution=args.source_camera_native_resolution,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid source FOV rectification contract: {exc}") from exc
    source_crop_fraction_xy = tuple(
        float(value)
        for value in source_fov_rectification["center_crop_fraction_xy"]
    )
    if physical_dataset and args.source_fov_rectification_mode != "none":
        raise SystemExit("corrected v04 physical RGB forbids source FOV cropping")
    if (
        args.source_fov_rectification_mode
        == "legacy_v03_square_vertical_fov_v1"
        and args.source_camera_contract is None
    ):
        raise SystemExit(
            "legacy v03 source FOV rectification requires --source-camera-contract"
        )
    if physical_dataset and args.source_camera_contract is None:
        raise SystemExit(
            "observable physical dataset v3 requires --source-camera-contract "
            "bound to its v04 render audit"
        )
    source_camera_contract = None
    if args.source_camera_contract is not None:
        try:
            source_camera_contract = _validate_source_camera_contract(
                args.source_camera_contract,
                dataset_manifest_path=dataset_manifest_path,
                rectification=source_fov_rectification,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"invalid source camera contract: {exc}") from exc
    if args.bev_lift_type in (
        PROJECTIVE_COLUMN_ATTENTION_LIFT,
        PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
    ):
        expected_horizontal = float(
            source_fov_rectification["intended_horizontal_fov_deg"]
        )
        expected_vertical = float(
            source_fov_rectification["intended_vertical_fov_deg"]
        )
        if not math.isclose(
            float(args.projective_horizontal_fov_deg),
            expected_horizontal,
            rel_tol=0.0,
            abs_tol=1e-6,
        ) or not math.isclose(
            float(args.projective_vertical_fov_deg),
            expected_vertical,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise SystemExit(
                "projective FOV differs from the rectified platform camera: "
                f"expected=({expected_horizontal:.9f}, {expected_vertical:.9f})"
            )
        expected_xyz = np.asarray(
            geometry_camera["nominal_xyz_body_m"], dtype=np.float64
        )
        expected_rpy = np.asarray(
            geometry_camera["nominal_rpy_body_rad"], dtype=np.float64
        )
        supplied_xyz = np.asarray(
            args.projective_camera_xyz_body_m, dtype=np.float64
        )
        supplied_rpy = np.asarray(
            args.projective_camera_rpy_body_rad, dtype=np.float64
        )
        if (
            expected_xyz.shape != (3,)
            or expected_rpy.shape != (3,)
            or not np.allclose(supplied_xyz, expected_xyz, rtol=0.0, atol=1e-9)
            or not np.allclose(supplied_rpy, expected_rpy, rtol=0.0, atol=1e-9)
            or not math.isclose(
                float(args.projective_near_m),
                float(geometry_camera["near_m"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise SystemExit(
                "projective camera mount/near plane differs from geometry contract"
            )
        if args.bev_lift_type == PROJECTIVE_FOOTPRINT_ATTENTION_LIFT:
            expected_radius = float(
                geometry_payload["swept_footprint"]["planning_disc_radius_m"]
            )
            if not math.isclose(
                float(args.projective_footprint_radius_m),
                expected_radius,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise SystemExit(
                    "projective footprint radius differs from the geometry "
                    f"planning disc: expected={expected_radius}"
                )
    rows = _read_rows(Path(manifest["index"]["path"]))
    try:
        all_scene_roles = resolve_dataset_scene_roles(
            rows, manifest, legacy_selection_seed=args.selection_seed
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid dataset role contract: {exc}") from exc
    development_role_verification = {
        role: _verify_dataset_role_provenance(
            dataset_manifest_path,
            manifest,
            all_scene_roles,
            roles=(role,),
        )
        for role in DEVELOPMENT_DATASET_ROLES
    }
    dataset_provenance_verification: dict[str, Any] = {
        "development": _verify_dataset_role_provenance(
            dataset_manifest_path,
            manifest,
            all_scene_roles,
            roles=DEVELOPMENT_DATASET_ROLES,
        ),
        "g2_evaluation": None,
    }
    g2_role_verification: dict[str, Any] | None = None
    validation_roles = {
        scene_id: role
        for scene_id, role in all_scene_roles.items()
        if role != "train"
    }
    train_all = [
        row for row in rows if all_scene_roles[str(row["scene_id"])] == "train"
    ]
    validation_all = [
        row for row in rows if all_scene_roles[str(row["scene_id"])] != "train"
    ]
    train_scenes = {
        scene_id for scene_id, role in all_scene_roles.items() if role == "train"
    }
    validation_scenes = set(validation_roles)
    selection_all = [
        row for row in validation_all
        if validation_roles[str(row["scene_id"])] == "checkpoint_selection"
    ]
    calibration_all = [
        row for row in validation_all
        if validation_roles[str(row["scene_id"])] == "probability_calibration"
    ]
    g2_all = (
        []
        if args.development_only
        else [
            row
            for row in validation_all
            if validation_roles[str(row["scene_id"])] == "g2_evaluation"
        ]
    )
    train_rows = deterministic_row_subset(
        train_all, maximum_rows=args.max_train_rows, seed=f"{args.seed}:train"
    )
    selection_rows = deterministic_row_subset(
        selection_all,
        maximum_rows=args.max_selection_rows,
        seed=f"{args.seed}:selection",
    )
    calibration_rows = deterministic_row_subset(
        calibration_all,
        maximum_rows=args.max_calibration_rows,
        seed=f"{args.seed}:calibration",
    )
    g2_rows = (
        []
        if args.development_only
        else deterministic_row_subset(
            g2_all,
            maximum_rows=args.max_g2_rows,
            seed=f"{args.seed}:g2",
        )
    )
    selected_rows_by_role = {
        "train": train_rows,
        "checkpoint_selection": selection_rows,
        "probability_calibration": calibration_rows,
        "g2_evaluation": g2_rows,
    }
    row_subset_records = {
        role: _row_subset_record(role_rows, role=role)
        for role, role_rows in selected_rows_by_role.items()
    }
    available_row_counts = {
        role: sum(
            assigned_role == role
            for row in rows
            for assigned_role in (all_scene_roles[str(row["scene_id"])],)
        )
        for role in DATASET_ROLES
    }
    primitives = sorted({str(row["primitive"]) for row in train_rows})
    if not primitives:
        raise SystemExit("training primitive vocabulary is empty")
    primitive_to_index = {primitive: index for index, primitive in enumerate(primitives)}
    nominal_delta_table = nominal_primitive_delta_table(
        train_rows, primitive_to_index
    )
    nominal_delta_table_id = (
        "go2-train-median-delta-"
        + _canonical_json_sha256(
            {
                "primitive_to_index": primitive_to_index,
                "values": nominal_delta_table.tolist(),
            }
        )[:16]
    )
    nominal_delta_statistics = nominal_primitive_delta_statistics(
        train_rows,
        primitive_to_index,
        nominal_delta_table,
    )
    unseen = {
        str(row["primitive"])
        for row in selection_rows + calibration_rows + g2_rows
    } - set(primitives)
    if unseen:
        raise SystemExit(f"validation contains unseen primitives: {sorted(unseen)}")
    (
        unknown_known_weights,
        free_occupied_weights,
        occupancy_loss_provenance,
    ) = _hierarchical_occupancy_objective(train_rows)
    device = _resolve_device(args.device)
    unknown_known_weights = unknown_known_weights.to(device)
    free_occupied_weights = free_occupied_weights.to(device)
    model_config = _model_config(
        args,
        len(primitives),
        manifest["local_grid"],
        projective_query_support=projective_query_support,
    )
    validate_projective_query_support_binding(
        model_config=model_config,
        projective_query_support=projective_query_support,
        dataset_manifest=manifest,
    )
    model = EgomotionBevJepa(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    selection_loader = _loader(
        selection_rows,
        primitive_to_index=primitive_to_index,
        image_size=args.image_size,
        source_crop_fraction_xy=source_crop_fraction_xy,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    calibration_loader = _loader(
        calibration_rows,
        primitive_to_index=primitive_to_index,
        image_size=args.image_size,
        source_crop_fraction_xy=source_crop_fraction_xy,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    history: list[dict[str, Any]] = []
    best_score: tuple[float, ...] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loader = _loader(
            train_rows,
            primitive_to_index=primitive_to_index,
            image_size=args.image_size,
            source_crop_fraction_xy=source_crop_fraction_xy,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=True,
            seed=args.seed + epoch,
        )
        train_metrics = _train_one_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
            nominal_delta_table=nominal_delta_table,
            gradient_clip=args.gradient_clip,
            epoch=epoch,
        )
        selection = evaluate_model(
            model,
            selection_loader,
            device=device,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
            nominal_delta_table=nominal_delta_table,
            calibration=None,
            thresholds=None,
            select_thresholds=True,
            occupancy_target_space=occupancy_target_space,
        )
        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "checkpoint_selection": selection,
        }
        history.append(epoch_record)
        score = _selection_score(selection, mode=args.selection_score_mode)
        if best_score is None or score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        if args.log_every > 0 and (epoch == 1 or epoch % args.log_every == 0):
            tm = selection["traversability"]
            pc = selection["predictive_controls"]["panels"]["changed"]
            print(
                f"epoch={epoch} train_loss={train_metrics['loss']:.5f} "
                f"sel_precision={tm['planner_admitted_free_precision']:.4f} "
                f"sel_recall={tm['useful_traversable_recall']:.4f} "
                f"sel_obstacle={tm['obstacle_detection_recall_within_range']:.4f} "
                f"pred_ratio={pc['prediction_to_warped_persistence_ratio']:.4f} "
                f"g2_checks={sum(selection['g2']['checks'].values())}/{len(selection['g2']['checks'])}",
                flush=True,
            )

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    calibration_provenance = {
        "schema": "lewm_go2_probability_calibration_provenance_v1",
        "role": "probability_calibration",
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "selected_model_state_sha256": _state_dict_sha256(best_state),
        "calibration_row_subset_sha256": row_subset_records[
            "probability_calibration"
        ]["identity_sha256"],
        "calibration_row_count": row_subset_records["probability_calibration"][
            "count"
        ],
        "best_epoch": int(best_epoch),
    }
    if args.probability_calibration_mode == "vector_scaling_v1":
        calibration_logits, calibration_labels, calibration_sampling = (
            collect_calibration_sample(
                model,
                calibration_loader,
                device=device,
                maximum_cells=args.max_calibration_cells,
                allow_rare_class_backfill=bool(args.development_only),
            )
        )
        calibration = fit_vector_calibration(
            calibration_logits,
            calibration_labels,
        )
        calibration["provenance"] = calibration_provenance
        calibration["id"] = (
            "go2-vector-scale-" + _canonical_json_sha256(calibration)[:16]
        )
    else:
        calibration_logits, calibration_labels, calibration_sampling = (
            collect_all_calibration_cells(
                model,
                calibration_loader,
                device=device,
            )
        )
        calibration = fit_hierarchical_probability_calibration(
            calibration_logits,
            calibration_labels,
            provenance=calibration_provenance,
        )
    calibration_metrics = evaluate_model(
        model,
        calibration_loader,
        device=device,
        unknown_known_weights=unknown_known_weights,
        free_occupied_weights=free_occupied_weights,
        nominal_delta_table=nominal_delta_table,
        calibration=calibration,
        thresholds=None,
        select_thresholds=True,
        occupancy_target_space=occupancy_target_space,
    )
    best_thresholds = TraversabilityThresholds(
        **calibration_metrics["thresholds"]
    )
    final_g2 = None
    if not args.development_only:
        g2_role_verification = _verify_dataset_role_provenance(
            dataset_manifest_path,
            manifest,
            all_scene_roles,
            roles=("g2_evaluation",),
        )
        dataset_provenance_verification["g2_evaluation"] = g2_role_verification
        # This is the first permitted byte-level read of G2 artifacts.  Model
        # outputs are evaluated exactly once below.
        g2_loader = _loader(
            g2_rows,
            primitive_to_index=primitive_to_index,
            image_size=args.image_size,
            source_crop_fraction_xy=source_crop_fraction_xy,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=False,
            seed=args.seed,
        )
        final_g2 = evaluate_model(
            model,
            g2_loader,
            device=device,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
            nominal_delta_table=nominal_delta_table,
            calibration=calibration,
            thresholds=best_thresholds,
            select_thresholds=False,
            occupancy_target_space=occupancy_target_space,
        )
    output_path = args.output.resolve()
    report_path = (
        args.report_output.resolve()
        if args.report_output is not None
        else output_path.with_suffix(".report.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_schema, report_schema = _artifact_schemas(
        physical_dataset=physical_dataset,
        bev_lift_type=args.bev_lift_type,
        probability_calibration_mode=args.probability_calibration_mode,
    )
    head_g2_passes = bool(
        physical_dataset
        and final_g2 is not None
        and final_g2["g2"]["passes"]
    )
    legacy_g2_passes = bool(
        not physical_dataset
        and final_g2 is not None
        and final_g2["g2"]["passes"]
    )
    role_specific_provenance = {
        **development_role_verification,
        "g2_evaluation": g2_role_verification,
    }
    dataset_access_ledger = {
        "schema": "lewm_go2_dataset_access_ledger_v1",
        "scope": "trainer_process",
        "row_index_metadata": {
            "read": True,
            "path": str(Path(manifest["index"]["path"]).resolve()),
            "sha256": str(manifest["index"]["sha256"]),
            "row_count": len(rows),
            "all_role_metadata_read": True,
            "g2_row_metadata_count": available_row_counts["g2_evaluation"],
        },
        "roles": {
            role: {
                "available_row_count": available_row_counts[role],
                "selected_row_count": row_subset_records[role]["count"],
                "row_subset_sha256": row_subset_records[role]["identity_sha256"],
                "provenance_verification": role_specific_provenance[role],
                "label_shard_files_hashed": int(
                    (role_specific_provenance[role] or {}).get("checked", {}).get(
                        "shard", 0
                    )
                ),
                "image_files_hashed": int(
                    (role_specific_provenance[role] or {}).get("checked", {}).get(
                        "image", 0
                    )
                ),
                "model_output_rows": (
                    row_subset_records[role]["count"]
                    if role != "g2_evaluation" or final_g2 is not None
                    else 0
                ),
            }
            for role in DATASET_ROLES
        },
    }
    g2_access = dataset_access_ledger["roles"]["g2_evaluation"]
    dataset_access_ledger["g2_contact"] = {
        "row_metadata_read": True,
        "row_metadata_count": available_row_counts["g2_evaluation"],
        "label_shard_byte_opens": int(g2_access["label_shard_files_hashed"]),
        "image_byte_opens": int(g2_access["image_files_hashed"]),
        "model_output_rows": int(g2_access["model_output_rows"]),
    }
    scene_splits = {
        "train": sorted(train_scenes),
        "checkpoint_selection": sorted(
            scene
            for scene, role in validation_roles.items()
            if role == "checkpoint_selection"
        ),
        "probability_calibration": sorted(
            scene
            for scene, role in validation_roles.items()
            if role == "probability_calibration"
        ),
        "g2_evaluation": sorted(
            scene
            for scene, role in validation_roles.items()
            if role == "g2_evaluation"
        ),
    }
    command = " ".join(shlex.quote(value) for value in invocation_argv)
    critical_training_inputs = _critical_training_inputs(
        dataset_manifest_path,
        manifest,
        source_camera_contract_path=args.source_camera_contract,
    )
    training_run_core = {
        "schema": "lewm_go2_training_run_provenance_v1",
        "git": _git_snapshot(),
        "invocation": {"argv": invocation_argv, "command": command},
        "resolved_config": _json_normalize(vars(args)),
        "critical_inputs": critical_training_inputs,
        "scene_splits": scene_splits,
        "row_subsets": row_subset_records,
        "dataset_access_ledger": dataset_access_ledger,
        "checkpoint_artifact_included": False,
        **(
            {"projective_query_support": projective_query_support}
            if projective_query_support is not None
            else {}
        ),
    }
    training_run_provenance = {
        **training_run_core,
        "content_sha256": _canonical_json_sha256(training_run_core),
    }
    checkpoint = {
        "schema": checkpoint_schema,
        "model_state_dict": best_state,
        "model_config": model_config,
        "primitive_to_index": primitive_to_index,
        "nominal_primitive_delta_current": nominal_delta_table.tolist(),
        "nominal_primitive_delta_id": nominal_delta_table_id,
        "nominal_primitive_delta_source": "coordinatewise_train_median_with_circular_yaw",
        "nominal_primitive_delta_statistics": nominal_delta_statistics,
        "probability_calibration": calibration,
        "probability_calibration_id": calibration["id"],
        "probability_calibration_sampling": calibration_sampling,
        "image_normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "source_fov_rectification": source_fov_rectification,
        "source_camera_contract": source_camera_contract,
        **(
            {"projective_query_support": projective_query_support}
            if projective_query_support is not None
            else {}
        ),
        "occupancy_output_contract": {
            "class_order": ["unknown", "free", "occupied"],
            "raw_output": "three_class_logits",
            "runtime_transform": (
                "apply_probability_calibration_then_softmax"
                if args.probability_calibration_mode == "vector_scaling_v1"
                else calibration["output_transform"]
            ),
            "local_grid": manifest["local_grid"],
            "target_occupancy_space": occupancy_target_space,
            "post_memory_configuration_derivation": label_semantics.get(
                "post_memory_configuration_derivation"
            ),
            **(
                {
                    "projective_query_support_contract_sha256": (
                        projective_query_support["contract_sha256"]
                    )
                }
                if projective_query_support is not None
                else {}
            ),
        },
        "occupancy_training_objective": occupancy_loss_provenance,
        "deterministic_execution": deterministic_execution,
        "selection_score_mode": args.selection_score_mode,
        "traversability_thresholds": asdict(best_thresholds),
        "best_epoch": best_epoch,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "dataset_provenance_verification": dataset_provenance_verification,
        "dataset_role_provenance_verification": role_specific_provenance,
        "dataset_access_ledger": dataset_access_ledger,
        "row_subsets": row_subset_records,
        "probability_calibration_provenance": calibration_provenance,
        "training_run_provenance": training_run_provenance,
        "training_scene_ids": sorted(train_scenes),
        "geometry_contract_sha256": manifest["geometry_contract"]["sha256"],
        "scene_roles_sha256": _canonical_json_sha256(validation_roles),
        "selection_metrics": history[best_epoch - 1]["checkpoint_selection"],
        "calibration_metrics": calibration_metrics,
        "g2_evaluation": None if physical_dataset else final_g2,
        "g2_passes": legacy_g2_passes,
        **(
            {
                "head_g2_evaluation": final_g2,
                "head_g2_passes": head_g2_passes,
                "runtime_ready": False,
                "runtime_readiness_reason": "deferred_to_G3_integration",
            }
            if physical_dataset
            else {}
        ),
    }
    _validate_projective_query_support_artifacts(checkpoint, manifest)
    torch.save(checkpoint, output_path)
    experiment_inputs = {
        "dataset_manifest": dataset_manifest_path,
        "dataset_index": Path(manifest["index"]["path"]),
        "model_source": REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py",
        "trainer_source": Path(__file__).resolve(),
    }
    if args.source_camera_contract is not None:
        experiment_inputs["source_camera_contract"] = (
            args.source_camera_contract.resolve()
        )
    experiment = build_experiment_manifest(
        experiment_id=f"go2_egomotion_bev_jepa_seed{args.seed}",
        repository_root=REPOSITORY_ROOT,
        inputs=experiment_inputs,
        artifacts={"checkpoint": output_path},
        config=_json_normalize(
            {
                **vars(args),
                "dataset_manifest": dataset_manifest_path,
                "dataset_provenance_verification": dataset_provenance_verification,
                "dataset_role_provenance_verification": role_specific_provenance,
                "dataset_access_ledger": dataset_access_ledger,
                "row_subsets": row_subset_records,
                "output": output_path,
                "report_output": report_path,
                "model_config": model_config,
                "primitive_to_index": primitive_to_index,
                "nominal_primitive_delta_current": nominal_delta_table.tolist(),
                "nominal_primitive_delta_id": nominal_delta_table_id,
                "nominal_primitive_delta_statistics": nominal_delta_statistics,
                "probability_calibration": calibration,
                "probability_calibration_sampling": calibration_sampling,
                "probability_calibration_provenance": calibration_provenance,
                "occupancy_training_objective": occupancy_loss_provenance,
                "deterministic_execution": deterministic_execution,
                "source_fov_rectification": source_fov_rectification,
                "source_camera_contract": source_camera_contract,
                "projective_query_support": projective_query_support,
                "training_run_provenance": training_run_provenance,
            }
        ),
        seeds=[args.seed],
        run_command=command,
        scene_splits=scene_splits,
        geometry_contract=Path(manifest["geometry_contract"]["path"]),
        runtime_contract={
            "occupancy_model_inputs": ["ego_rgb"],
            "promoted_predictive_inputs": [
                "current_ego_rgb",
                "commanded_primitive",
                "frozen_train_nominal_primitive_delta",
            ],
            "realized_future_odometry_role": "train_and_evaluation_equivariance_auxiliary_only",
            "nominal_primitive_delta_id": nominal_delta_table_id,
            "probability_calibration_id": calibration["id"],
            "privileged_runtime_inputs": [],
            "occupancy_output": (
                "observable physical unknown/free/occupied local evidence"
                if physical_dataset
                else "unknown/free/occupied body-inflated local grid"
            ),
            "occupancy_target_space": occupancy_target_space,
            "post_memory_configuration_derivation": label_semantics.get(
                "post_memory_configuration_derivation"
            ),
            "training_source_fov_rectification": source_fov_rectification,
            "training_source_camera_contract": source_camera_contract,
            "projective_query_support": projective_query_support,
            "runtime_input_camera": {
                "horizontal_fov_deg": source_fov_rectification[
                    "intended_horizontal_fov_deg"
                ],
                "vertical_fov_deg": source_fov_rectification[
                    "intended_vertical_fov_deg"
                ],
                "source_rectification_crop_required": False,
            },
            **(
                {
                    "runtime_ready": False,
                    "physical_head_g2_is_not_runtime_promotion": True,
                }
                if physical_dataset
                else {}
            ),
        },
    )
    report = {
        "schema": report_schema,
        "checkpoint": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "dataset_provenance_verification": dataset_provenance_verification,
        "dataset_role_provenance_verification": role_specific_provenance,
        "dataset_access_ledger": dataset_access_ledger,
        "row_subsets": row_subset_records,
        "row_counts": {
            "train": len(train_rows),
            "checkpoint_selection": len(selection_rows),
            "probability_calibration": len(calibration_rows),
            "g2_evaluation": len(g2_rows),
        },
        "scene_counts": {
            "train": len(train_scenes),
            "checkpoint_selection": sum(
                role == "checkpoint_selection"
                for role in validation_roles.values()
            ),
            "probability_calibration": sum(
                role == "probability_calibration"
                for role in validation_roles.values()
            ),
            "g2_evaluation": sum(role == "g2_evaluation" for role in validation_roles.values()),
        },
        "best_epoch": best_epoch,
        "selection_score_mode": args.selection_score_mode,
        "occupancy_training_objective": occupancy_loss_provenance,
        "deterministic_execution": deterministic_execution,
        "source_fov_rectification": source_fov_rectification,
        "source_camera_contract": source_camera_contract,
        **(
            {"projective_query_support": projective_query_support}
            if projective_query_support is not None
            else {}
        ),
        "label_semantics": label_semantics,
        "history": history,
        "probability_calibration": calibration,
        "probability_calibration_provenance": calibration_provenance,
        "probability_calibration_sampling": calibration_sampling,
        "calibration_metrics": calibration_metrics,
        "training_run_provenance": training_run_provenance,
        "final_g2_evaluation": None if physical_dataset else final_g2,
        **(
            {"final_head_g2_evaluation": final_g2}
            if physical_dataset
            else {}
        ),
        "promotion": (
            {
                "gate": "G2_physical_evidence_head",
                "head_g2_passes": head_g2_passes,
                "head_g2_evaluated": final_g2 is not None,
                "runtime_ready": False,
                "runtime_readiness_deferred_to": "G3",
                "thresholds_selected_without_g2_evaluation": True,
                "calibration_fit_without_g2_evaluation": True,
            }
            if physical_dataset
            else {
                "gate": "G2",
                "passes": legacy_g2_passes,
                "g2_evaluated": final_g2 is not None,
                "thresholds_selected_without_g2_evaluation": True,
                "calibration_fit_without_g2_evaluation": True,
            }
        ),
        "experiment_manifest": experiment,
    }
    _validate_projective_query_support_artifacts(
        checkpoint,
        manifest,
        report=report,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["promotion"], sort_keys=True), flush=True)
    if args.development_only:
        return 0
    assert final_g2 is not None
    if physical_dataset:
        return 0 if head_g2_passes else 2
    return 0 if legacy_g2_passes else 2


if __name__ == "__main__":
    raise SystemExit(main())
