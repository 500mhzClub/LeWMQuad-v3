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
import sys
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
from lewm.models.egomotion_bev_jepa import (  # noqa: E402
    EgomotionBevJepa,
    warp_bev_current_to_next,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20260709


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        shard_cache_size: int = 2,
    ) -> None:
        self.rows = [dict(row) for row in rows]
        self.primitive_to_index = dict(primitive_to_index)
        self.image_size = int(image_size)
        self.shard_cache_size = max(1, int(shard_cache_size))
        self._shards: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

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


def _class_weights(rows: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, dict[str, int]]:
    counts = np.zeros(3, dtype=np.int64)
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(str(row["label_shard_path"]), []).append(int(row["label_shard_row"]))
    for path, indices in grouped.items():
        with np.load(path, allow_pickle=False) as shard:
            for prefix in ("current", "next"):
                labels_all = np.asarray(shard[f"{prefix}_labels"])
                mask_all = _mask_array(shard, prefix, labels_all)
                labels = labels_all[indices]
                mask = mask_all[indices]
                counts += np.bincount(labels[mask].reshape(-1), minlength=3)[:3]
    if (counts == 0).any():
        raise ValueError(f"every occupancy class must occur in training data, got {counts.tolist()}")
    inverse_sqrt = np.sqrt(counts.sum() / counts.astype(np.float64))
    inverse_sqrt /= inverse_sqrt.mean()
    inverse_sqrt = np.clip(inverse_sqrt, 0.25, 4.0)
    return torch.tensor(inverse_sqrt, dtype=torch.float32), {
        "unknown": int(counts[UNKNOWN_CLASS]),
        "free": int(counts[FREE_CLASS]),
        "occupied": int(counts[OCCUPIED_CLASS]),
    }


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
    class_weights: torch.Tensor,
    nominal_delta_table: torch.Tensor,
    calibration: Mapping[str, Any] | None,
    thresholds: TraversabilityThresholds | None,
    select_thresholds: bool,
) -> dict[str, Any]:
    model.eval()
    nominal_delta_table = nominal_delta_table.to(device=device, dtype=torch.float32)
    if nominal_delta_table.shape != (model.action_dim, 3):
        raise ValueError("nominal_delta_table must have shape (action_dim, 3)")
    calibration_log_scales = torch.zeros(3, device=device)
    calibration_biases = torch.zeros(3, device=device)
    if calibration is not None:
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
        wrong_action = torch.roll(batch["action"], shifts=1, dims=1)
        wrong_action_delta = wrong_action @ nominal_delta_table
        output = model(
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
            occupancy_class_weights=class_weights,
            diagnostic_wrong_action=wrong_action,
            diagnostic_wrong_action_delta_pose_current=wrong_action_delta,
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
        calibrated_logits = apply_vector_calibration(
            output["next_occupancy_logits"],
            calibration_log_scales,
            calibration_biases,
        )
        next_probs = torch.softmax(calibrated_logits, dim=1)
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
        threshold_selection = select_conservative_thresholds(
            probs,
            truth,
            distances,
            free_probability_candidates=(0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99),
            occupied_probability_candidates=(0.01, 0.02, 0.05, 0.10, 0.20, 0.35),
            unknown_probability_candidates=(0.01, 0.02, 0.05, 0.10, 0.20, 0.35),
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
    paths: list[list[tuple[int, int, int]]] = []
    oracle_paths: list[list[tuple[int, int, int]]] = []
    route_failures = 0
    route_length_recall_sum = 0.0
    for sample in range(rows):
        oracle_free = valid[sample] & (truth[sample] == FREE_CLASS)
        start = _nearest_free_start(oracle_free, forward, left)
        if start is None:
            continue
        oracle_path = _farthest_path(
            oracle_free, sample, start=start
        )
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
    route_collision_rate = planned_path_collision_rate(paths, truth)

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
        "calibration": {
            "applied": calibration is not None,
            "id": None if calibration is None else calibration.get("id"),
            "method": None if calibration is None else calibration["method"],
            "log_scales": calibration_log_scales.detach().cpu().tolist(),
            "biases": calibration_biases.detach().cpu().tolist(),
        },
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
        "routing": {
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
            "planned_path_collision_rate": route_collision_rate,
            "oracle_map_collision_rate": planned_path_collision_rate(
                oracle_paths, truth
            ),
        },
    }
    metrics["g2"] = evaluate_g2_gate(metrics)
    return metrics


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


def _selection_score(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    traversability = metrics["traversability"]
    predictive = metrics["predictive_controls"]["panels"]["changed"]
    checks = metrics["g2"]["checks"]
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
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = PairedNavigationTorchDataset(
        rows,
        primitive_to_index=primitive_to_index,
        image_size=image_size,
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


def _model_config(args: argparse.Namespace, action_dim: int, grid: Mapping[str, Any]) -> dict[str, Any]:
    return {
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


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main(argv: Sequence[str] | None = None) -> int:
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
    parser.add_argument("--selection-seed", default="go2_g2_selection_v1")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="Train/calibrate without reading G2 images or claiming promotion.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args(argv)

    if args.epochs <= 0 or args.batch_size <= 1:
        raise SystemExit("epochs must be positive and batch-size must exceed one")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_manifest_path = args.dataset_manifest.resolve()
    manifest = _read_json(dataset_manifest_path)
    if manifest.get("schema") != "lewm_go2_paired_navigation_dataset_v2":
        raise SystemExit("unsupported dataset schema")
    verify_dataset_provenance(dataset_manifest_path, verify_images=False)
    rows = _read_rows(Path(manifest["index"]["path"]))
    try:
        all_scene_roles = resolve_dataset_scene_roles(
            rows, manifest, legacy_selection_seed=args.selection_seed
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid dataset role contract: {exc}") from exc
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
    g2_all = [
        row for row in validation_all if validation_roles[str(row["scene_id"])] == "g2_evaluation"
    ]
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
    g2_rows = deterministic_row_subset(
        g2_all, maximum_rows=args.max_g2_rows, seed=f"{args.seed}:g2"
    )
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
    class_weights, class_counts = _class_weights(train_rows)
    device = _resolve_device(args.device)
    class_weights = class_weights.to(device)
    model_config = _model_config(args, len(primitives), manifest["local_grid"])
    model = EgomotionBevJepa(**model_config).to(device)
    nominal_delta_table_device = nominal_delta_table.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    selection_loader = _loader(
        selection_rows,
        primitive_to_index=primitive_to_index,
        image_size=args.image_size,
        batch_size=args.batch_size,
        workers=args.workers,
        shuffle=False,
        seed=args.seed,
    )
    calibration_loader = _loader(
        calibration_rows,
        primitive_to_index=primitive_to_index,
        image_size=args.image_size,
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
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=True,
            seed=args.seed + epoch,
        )
        model.train()
        running = Counter()
        steps = 0
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            commanded_delta = batch["action"] @ nominal_delta_table_device
            wrong_action = torch.roll(batch["action"], shifts=1, dims=1)
            wrong_action_delta = wrong_action @ nominal_delta_table_device
            optimizer.zero_grad(set_to_none=True)
            output = model(
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
                occupancy_class_weights=class_weights,
                diagnostic_wrong_action=wrong_action,
                diagnostic_wrong_action_delta_pose_current=wrong_action_delta,
            )
            if not torch.isfinite(output["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch={epoch} step={steps}")
            output["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
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
        selection = evaluate_model(
            model,
            selection_loader,
            device=device,
            class_weights=class_weights,
            nominal_delta_table=nominal_delta_table,
            calibration=None,
            thresholds=None,
            select_thresholds=True,
        )
        epoch_record = {
            "epoch": epoch,
            "train": {name: value / max(1, steps) for name, value in sorted(running.items())},
            "checkpoint_selection": selection,
        }
        history.append(epoch_record)
        score = _selection_score(selection)
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
                f"epoch={epoch} train_loss={running['loss'] / max(1, steps):.5f} "
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
    calibration_logits, calibration_labels, calibration_sampling = collect_calibration_sample(
        model,
        calibration_loader,
        device=device,
        maximum_cells=args.max_calibration_cells,
        allow_rare_class_backfill=bool(args.development_only),
    )
    calibration = fit_vector_calibration(
        calibration_logits,
        calibration_labels,
    )
    calibration["id"] = (
        "go2-vector-scale-" + _canonical_json_sha256(calibration)[:16]
    )
    calibration_metrics = evaluate_model(
        model,
        calibration_loader,
        device=device,
        class_weights=class_weights,
        nominal_delta_table=nominal_delta_table,
        calibration=calibration,
        thresholds=None,
        select_thresholds=True,
    )
    best_thresholds = TraversabilityThresholds(
        **calibration_metrics["thresholds"]
    )
    final_g2 = None
    if not args.development_only:
        g2_loader = _loader(
            g2_rows,
            primitive_to_index=primitive_to_index,
            image_size=args.image_size,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=False,
            seed=args.seed,
        )
        # This is the first and only read of G2 evaluation model outputs.
        final_g2 = evaluate_model(
            model,
            g2_loader,
            device=device,
            class_weights=class_weights,
            nominal_delta_table=nominal_delta_table,
            calibration=calibration,
            thresholds=best_thresholds,
            select_thresholds=False,
        )
    output_path = args.output.resolve()
    report_path = (
        args.report_output.resolve()
        if args.report_output is not None
        else output_path.with_suffix(".report.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v2",
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
        "occupancy_output_contract": {
            "class_order": ["unknown", "free", "occupied"],
            "raw_output": "three_class_logits",
            "runtime_transform": "apply_probability_calibration_then_softmax",
            "local_grid": manifest["local_grid"],
        },
        "traversability_thresholds": asdict(best_thresholds),
        "best_epoch": best_epoch,
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "training_scene_ids": sorted(train_scenes),
        "geometry_contract_sha256": manifest["geometry_contract"]["sha256"],
        "scene_roles_sha256": _canonical_json_sha256(validation_roles),
        "selection_metrics": history[best_epoch - 1]["checkpoint_selection"],
        "calibration_metrics": calibration_metrics,
        "g2_evaluation": final_g2,
        "g2_passes": bool(
            final_g2 is not None and final_g2["g2"]["passes"]
        ),
    }
    torch.save(checkpoint, output_path)
    command = " ".join(shlex.quote(value) for value in sys.argv)
    experiment = build_experiment_manifest(
        experiment_id=f"go2_egomotion_bev_jepa_seed{args.seed}",
        repository_root=REPOSITORY_ROOT,
        inputs={
            "dataset_manifest": dataset_manifest_path,
            "dataset_index": Path(manifest["index"]["path"]),
            "model_source": REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py",
            "trainer_source": Path(__file__).resolve(),
        },
        artifacts={"checkpoint": output_path},
        config={
            **vars(args),
            "dataset_manifest": str(dataset_manifest_path),
            "output": str(output_path),
            "report_output": str(report_path),
            "model_config": model_config,
            "primitive_to_index": primitive_to_index,
            "nominal_primitive_delta_current": nominal_delta_table.tolist(),
            "nominal_primitive_delta_id": nominal_delta_table_id,
            "nominal_primitive_delta_statistics": nominal_delta_statistics,
            "probability_calibration": calibration,
            "probability_calibration_sampling": calibration_sampling,
            "class_counts": class_counts,
            "class_weights": class_weights.detach().cpu().tolist(),
        },
        seeds=[args.seed],
        run_command=command,
        scene_splits={
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
                scene for scene, role in validation_roles.items() if role == "g2_evaluation"
            ),
        },
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
            "occupancy_output": "unknown/free/occupied body-inflated local grid",
        },
    )
    report = {
        "schema": "lewm_go2_egomotion_bev_jepa_training_report_v2",
        "checkpoint": {"path": str(output_path), "sha256": sha256_file(output_path)},
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": sha256_file(dataset_manifest_path),
        },
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
        "history": history,
        "probability_calibration": calibration,
        "probability_calibration_sampling": calibration_sampling,
        "calibration_metrics": calibration_metrics,
        "final_g2_evaluation": final_g2,
        "promotion": {
            "gate": "G2",
            "passes": bool(
                final_g2 is not None and final_g2["g2"]["passes"]
            ),
            "g2_evaluated": final_g2 is not None,
            "thresholds_selected_without_g2_evaluation": True,
            "calibration_fit_without_g2_evaluation": True,
        },
        "experiment_manifest": experiment,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["promotion"], sort_keys=True), flush=True)
    if args.development_only:
        return 0
    assert final_g2 is not None
    return 0 if final_g2["g2"]["passes"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
