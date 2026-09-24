#!/usr/bin/env python3
"""Run matched patch14/patch7 train-only Go2 micro-overfit probes."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks.go2_physical_micro_overfit import (  # noqa: E402
    AUTHORITATIVE_EXECUTION,
    DISTANCE_BINS_M,
    FAMILIES,
    GATED_DISTANCE_BIN_NAMES,
    PANELS,
    RESULT_SCHEMA,
    SMOKE_EXECUTION,
    SMOKE_RESULT_SCHEMA,
    TRAINING_WEIGHTS,
    attach_role_global_shuffle,
    attach_same_scene_wrong_view,
    canonical_json_sha256,
    classify_cross_arm_decision,
    empty_raw_accumulator,
    finalize_raw_accumulator,
    fit_gate,
    frame_records,
    update_raw_accumulator,
    validate_panel_manifest,
)
from lewm.models.egomotion_bev_jepa import (  # noqa: E402
    EgomotionBevJepa,
    PROJECTIVE_COLUMN_ATTENTION_LIFT,
)
from scripts.diagnose_go2_physical_spatial_grounding import (  # noqa: E402
    PhysicalFrameDataset,
)
from scripts.train_go2_egomotion_bev_jepa import (  # noqa: E402
    PairedNavigationTorchDataset,
)


SOURCE_PATHS = {
    "contract": REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_micro_overfit.py",
    "dataset": REPOSITORY_ROOT / "lewm/datasets/go2_paired_navigation.py",
    "diagnostic_dataset": (
        REPOSITORY_ROOT / "scripts/diagnose_go2_physical_spatial_grounding.py"
    ),
    "encoder": REPOSITORY_ROOT / "lewm/models/encoders.py",
    "generalization_execution_contract": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_generalization_execution_contract_2026-07-09.md"
    ),
    "micro_overfit_protocol": (
        REPOSITORY_ROOT
        / "docs/lewm_go2_physical_micro_overfit_protocol_2026-07-10.md"
    ),
    "model": REPOSITORY_ROOT / "lewm/models/egomotion_bev_jepa.py",
    "panel_preparer": REPOSITORY_ROOT / "scripts/prepare_go2_physical_micro_overfit.py",
    "runner": Path(__file__).resolve(),
    "spatial_metrics": (
        REPOSITORY_ROOT / "lewm/benchmarks/go2_physical_spatial_grounding.py"
    ),
    "trainer": REPOSITORY_ROOT / "scripts/train_go2_egomotion_bev_jepa.py",
}
ARM_CONFIGS = {
    "patch14_8x8": {"patch_size": 14, "token_side": 8, "sigma_tokens": 1.0},
    "patch7_16x16": {"patch_size": 7, "token_side": 16, "sigma_tokens": 2.0},
}
NORMALIZATION_MEAN = (0.485, 0.456, 0.406)
NORMALIZATION_STD = (0.229, 0.224, 0.225)
MIN_AGGREGATE_FREE_CELLS_PER_GATED_BIN = 1000
MIN_FAMILY_FREE_CELLS_PER_GATED_BIN = 100
SMOKE_EXECUTION_BOUNDS = {
    "maximum_batch_size": 4,
    "maximum_faithful_steps": 100,
    "maximum_ceiling_steps": 100,
    "maximum_evaluation_interval": 25,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in sorted(SOURCE_PATHS.items())
    }


def _git_snapshot() -> dict[str, Any]:
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip()
    diff = subprocess.run(
        ("git", "diff", "--binary", "--no-ext-diff"),
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return {
        "head": head,
        "status_short": status,
        "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "tracked_dirty_diff_bytes": len(diff),
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


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def _copy_shared_initial_state(
    baseline: EgomotionBevJepa, variant: EgomotionBevJepa
) -> dict[str, Any]:
    """Copy every identically shaped initialized tensor into the variant."""

    baseline_state = baseline.state_dict()
    variant_state = variant.state_dict()
    copied = []
    variant_only = []
    for name, value in variant_state.items():
        source = baseline_state.get(name)
        if source is not None and source.shape == value.shape and source.dtype == value.dtype:
            variant_state[name] = source.detach().clone()
            copied.append(name)
        else:
            variant_only.append(name)
    variant.load_state_dict(variant_state, strict=True)
    return {
        "schema": "lewm_go2_micro_overfit_shared_initialization_v1",
        "copied_tensor_names": copied,
        "copied_tensor_name_sha256": canonical_json_sha256(copied),
        "variant_specific_tensor_names": variant_only,
        "variant_specific_tensor_name_sha256": canonical_json_sha256(variant_only),
    }


def _independent_query_visibility_report(
    baseline: EgomotionBevJepa, variant: EgomotionBevJepa
) -> dict[str, Any]:
    """Compare geometry masks before any cross-arm state is copied."""

    baseline_visibility = baseline.bev_decoder.projective_query_visibility
    variant_visibility = variant.bev_decoder.projective_query_visibility
    if baseline_visibility is None or variant_visibility is None:
        raise ValueError("matched projective geometry buffers are absent")
    equal = torch.equal(baseline_visibility, variant_visibility)
    report = {
        "checked_before_shared_initialization_copy": True,
        "equal": equal,
        "patch14_sha256": _tensor_sha256(baseline_visibility),
        "patch7_sha256": _tensor_sha256(variant_visibility),
        "patch14_shape": list(baseline_visibility.shape),
        "patch7_shape": list(variant_visibility.shape),
    }
    if not equal:
        raise ValueError("independently constructed patch14/patch7 visibility differs")
    return report


def _validated_grid(
    panel: Mapping[str, Any],
) -> tuple[tuple[int, int], tuple[float, float], tuple[float, float]]:
    grid = panel.get("local_grid")
    if not isinstance(grid, Mapping):
        raise ValueError("micro-overfit panel lacks its local-grid contract")
    shape = tuple(int(value) for value in grid.get("shape", ()))
    forward = tuple(float(value) for value in grid.get("forward_center_range_m", ()))
    left = tuple(float(value) for value in grid.get("left_center_range_m", ()))
    if shape != (64, 64) or len(forward) != 2 or len(left) != 2:
        raise ValueError("micro-overfit runner requires the registered 64x64 grid")
    return shape, forward, left


def _model_config(
    panel: Mapping[str, Any], *, arm: str, action_dim: int
) -> dict[str, Any]:
    if arm not in ARM_CONFIGS:
        raise ValueError(f"unknown micro-overfit arm: {arm}")
    shape, forward, left = _validated_grid(panel)
    camera = panel.get("source_camera_projection")
    if not isinstance(camera, Mapping):
        raise ValueError("micro-overfit panel lacks source-camera projection")
    horizontal_fov = float(camera.get("horizontal_fov_deg", float("nan")))
    vertical_fov = float(camera.get("vertical_fov_deg", float("nan")))
    near_m = float(camera.get("near_m", float("nan")))
    if not math.isclose(horizontal_fov, 78.323, abs_tol=1e-9):
        raise ValueError("source-camera horizontal FOV changed")
    if not math.isclose(vertical_fov, 62.837038636424516, abs_tol=1e-9):
        raise ValueError("source-camera vertical FOV changed")
    if not math.isclose(near_m, 0.05, abs_tol=1e-12):
        raise ValueError("source-camera near plane changed")
    arm_config = ARM_CONFIGS[arm]
    return {
        "image_size": 112,
        "patch_size": int(arm_config["patch_size"]),
        "encoder_dim": 192,
        "encoder_depth": 6,
        "encoder_heads": 6,
        "bev_dim": 64,
        "bev_size": shape,
        "forward_range_m": forward,
        "left_range_m": left,
        "action_dim": int(action_dim),
        "bev_lift_type": PROJECTIVE_COLUMN_ATTENTION_LIFT,
        "projective_horizontal_fov_deg": horizontal_fov,
        "projective_vertical_fov_deg": vertical_fov,
        "projective_camera_xyz_body_m": (0.326, 0.0, 0.043),
        "projective_camera_rpy_body_rad": (0.0, 0.0, 0.0),
        "projective_near_m": near_m,
        "projective_vertical_anchor_z_body_m": (
            -0.333,
            -0.133,
            0.067,
            0.267,
            0.467,
        ),
        "projective_attention_sigma_tokens": float(arm_config["sigma_tokens"]),
        "projective_attention_bias_floor": -6.0,
        "predictor_hidden_dim": 128,
        "target_ema_momentum": 0.996,
        "jepa_weight": 0.0,
        "occupancy_weight": 2.0,
        "equivariance_weight": 0.0,
        "action_contrast_weight": 0.0,
        "action_margin_fraction": 0.1,
        "variance_weight": 0.0,
        "variance_target_std": 0.5,
    }


def _initial_states(
    panel: Mapping[str, Any], *, action_dim: int, seed: int
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    torch.manual_seed(seed)
    baseline = EgomotionBevJepa(
        **_model_config(panel, arm="patch14_8x8", action_dim=action_dim)
    )
    torch.manual_seed(seed)
    variant = EgomotionBevJepa(
        **_model_config(panel, arm="patch7_16x16", action_dim=action_dim)
    )
    independent_visibility = _independent_query_visibility_report(baseline, variant)
    shared = _copy_shared_initial_state(baseline, variant)
    states = {
        "patch14_8x8": _clone_state(baseline.state_dict()),
        "patch7_16x16": _clone_state(variant.state_dict()),
    }
    geometry = {}
    for arm, model in (("patch14_8x8", baseline), ("patch7_16x16", variant)):
        bias = model.bev_decoder.projective_attention_bias
        visibility = model.bev_decoder.projective_query_visibility
        if bias is None or visibility is None:
            raise ValueError("projective geometry buffer is missing")
        geometry[arm] = {
            "attention_bias_sha256": _tensor_sha256(bias),
            "attention_bias_shape": list(bias.shape),
            "query_visibility_sha256": _tensor_sha256(visibility),
            "query_visibility_shape": list(visibility.shape),
            "sigma_over_token_side": (
                float(ARM_CONFIGS[arm]["sigma_tokens"])
                / int(ARM_CONFIGS[arm]["token_side"])
            ),
        }
    baseline_state = baseline.state_dict()
    variant_state = variant.state_dict()
    differing_tensors = {}
    for name in sorted(set(baseline_state) | set(variant_state)):
        baseline_tensor = baseline_state.get(name)
        variant_tensor = variant_state.get(name)
        baseline_shape = None if baseline_tensor is None else list(baseline_tensor.shape)
        variant_shape = None if variant_tensor is None else list(variant_tensor.shape)
        if baseline_shape != variant_shape:
            differing_tensors[name] = {
                "patch14_shape": baseline_shape,
                "patch7_shape": variant_shape,
                "is_parameter": name in dict(baseline.named_parameters())
                or name in dict(variant.named_parameters()),
            }
    parameter_count = {
        "patch14_8x8": sum(parameter.numel() for parameter in baseline.parameters()),
        "patch7_16x16": sum(parameter.numel() for parameter in variant.parameters()),
    }
    report = {
        **shared,
        "seed": int(seed),
        "initial_state_sha256": {
            arm: _state_dict_sha256(state) for arm, state in states.items()
        },
        "parameter_count": parameter_count,
        "patch7_minus_patch14_parameter_count": (
            parameter_count["patch7_16x16"] - parameter_count["patch14_8x8"]
        ),
        "variant_specific_tensor_shapes": differing_tensors,
        "variant_specific_parameter_tensor_names": [
            name for name, record in differing_tensors.items() if record["is_parameter"]
        ],
        "intervention_bundle": (
            "patch_size_token_grid_patch_embedding_and_attention_compute_at_"
            "fixed_samples_updates_and_normalized_projective_sigma"
        ),
        "projective_geometry": geometry,
        "independent_query_visibility": independent_visibility,
        "query_visibility_equal": True,
        "query_visibility_equal_before_shared_initialization_copy": True,
        "input_image_size_equal": True,
        "normalized_attention_sigma_equal": True,
    }
    return states, report


class _AuditedTransitionDataset(PairedNavigationTorchDataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        primitive_to_index: Mapping[str, int],
        allowed_images: set[str],
        allowed_shards: set[str],
    ) -> None:
        super().__init__(
            rows,
            primitive_to_index=primitive_to_index,
            image_size=112,
            source_crop_fraction_xy=(1.0, 1.0),
            shard_cache_size=max(2, len(allowed_shards)),
        )
        self.allowed_images = set(allowed_images)
        self.allowed_shards = set(allowed_shards)
        self.image_decode_events = 0
        self.label_shard_npz_open_events = 0
        self.opened_images: set[str] = set()
        self.opened_shards: set[str] = set()

    def _image(self, path: str) -> torch.Tensor:
        resolved = str(Path(path).resolve())
        if resolved not in self.allowed_images:
            raise PermissionError(f"transition dataset attempted an unapproved image: {path}")
        self.image_decode_events += 1
        self.opened_images.add(resolved)
        return super()._image(path)

    def _shard(self, path: str) -> dict[str, np.ndarray]:
        resolved = str(Path(path).resolve())
        if resolved not in self.allowed_shards:
            raise PermissionError(f"transition dataset attempted an unapproved shard: {path}")
        if path not in self._shards:
            self.label_shard_npz_open_events += 1
            self.opened_shards.add(resolved)
        return super()._shard(path)


class _MicroFrameDataset(PhysicalFrameDataset):
    """Physical frame dataset with an additional same-scene wrong-view image."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        record = self.records[index]
        result["same_scene_control_image"] = self._image(
            str(record["same_scene_control_image_path"])
        )
        return result


def _artifact_contract(
    panels: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[dict[str, str], dict[str, str]]:
    images: dict[str, str] = {}
    shards: dict[str, str] = {}
    for rows in panels.values():
        for row in rows:
            if str(row["dataset_role"]) != "train":
                raise ValueError("forbidden role reached the runner artifact contract")
            shard_path = str(Path(str(row["label_shard_path"])).resolve())
            shard_sha = str(row["label_shard_sha256"])
            if shard_path in shards and shards[shard_path] != shard_sha:
                raise ValueError("conflicting selected shard hashes")
            shards[shard_path] = shard_sha
            for side in ("current", "next"):
                path = str(Path(str(row[f"{side}_image_path"])).resolve())
                sha256 = str(row[f"{side}_image_sha256"])
                if path in images and images[path] != sha256:
                    raise ValueError("conflicting selected image hashes")
                images[path] = sha256
    return images, shards


def _verify_artifacts(
    images: Mapping[str, str], shards: Mapping[str, str]
) -> dict[str, Any]:
    for path, expected in sorted(images.items()):
        if _sha256_file(Path(path)) != expected:
            raise ValueError(f"selected train image SHA-256 mismatch: {path}")
    for path, expected in sorted(shards.items()):
        if _sha256_file(Path(path)) != expected:
            raise ValueError(f"selected train shard SHA-256 mismatch: {path}")
    return {
        "distinct_train_images_hashed": len(images),
        "distinct_train_label_shards_hashed": len(shards),
        "non_train_images_hashed": 0,
        "non_train_label_shards_hashed": 0,
    }


def _distance_grid(panel: Mapping[str, Any]) -> np.ndarray:
    shape, forward_range, left_range = _validated_grid(panel)
    forward = np.linspace(*forward_range, shape[0], dtype=np.float64)
    left = np.linspace(*left_range, shape[1], dtype=np.float64)
    return np.sqrt(forward[:, None] ** 2 + left[None, :] ** 2)


def _validate_support_contract(
    panel_name: str,
    class_counts: np.ndarray,
    distance_free: Mapping[str, int],
    family_class_counts: Mapping[str, np.ndarray],
    family_distance_free: Mapping[str, Mapping[str, int]],
) -> None:
    if (np.asarray(class_counts, dtype=np.int64) == 0).any():
        raise ValueError(f"post-selection class support failed for {panel_name}")
    if set(family_class_counts) != set(FAMILIES) or set(family_distance_free) != set(
        FAMILIES
    ):
        raise ValueError(f"post-selection family support is incomplete for {panel_name}")
    empty_family_classes = {
        family: [
            name
            for index, name in enumerate(("unknown", "free", "occupied"))
            if int(counts[index]) == 0
        ]
        for family, counts in family_class_counts.items()
    }
    empty_family_classes = {
        family: names for family, names in empty_family_classes.items() if names
    }
    if empty_family_classes:
        raise ValueError(
            f"post-selection family class support failed for {panel_name}: "
            f"{empty_family_classes}"
        )
    weak_aggregate = {
        name: int(distance_free[name])
        for name in GATED_DISTANCE_BIN_NAMES
        if int(distance_free[name]) < MIN_AGGREGATE_FREE_CELLS_PER_GATED_BIN
    }
    weak_families = {
        family: {
            name: int(counts[name])
            for name in GATED_DISTANCE_BIN_NAMES
            if int(counts[name]) < MIN_FAMILY_FREE_CELLS_PER_GATED_BIN
        }
        for family, counts in family_distance_free.items()
    }
    weak_families = {
        family: counts for family, counts in weak_families.items() if counts
    }
    if weak_aggregate or weak_families:
        raise ValueError(
            f"post-selection FREE distance support failed for {panel_name}; "
            f"aggregate={weak_aggregate}, families={weak_families}; "
            "the frozen panel must abort without reselection"
        )


def _support_audit(
    panel: Mapping[str, Any], panels: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    distances = _distance_grid(panel)
    report = {}
    for panel_name, rows in panels.items():
        class_counts = np.zeros(3, dtype=np.int64)
        distance_free = {name: 0 for name, _lower, _upper in DISTANCE_BINS_M}
        family_class_counts = {
            family: np.zeros(3, dtype=np.int64) for family in FAMILIES
        }
        family_distance_free = {
            family: {name: 0 for name, _lower, _upper in DISTANCE_BINS_M}
            for family in FAMILIES
        }
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row["label_shard_path"]), []).append(row)
        for path, shard_rows in grouped.items():
            with np.load(path, allow_pickle=False) as shard:
                for row in shard_rows:
                    family = str(row["family"])
                    if family not in family_class_counts:
                        raise ValueError(f"unknown family in support audit: {family}")
                    index = int(row["label_shard_row"])
                    for side in ("current", "next"):
                        labels = np.asarray(shard[f"{side}_labels"][index], dtype=np.int64)
                        mask = np.asarray(
                            shard[f"{side}_supervision_mask"][index], dtype=bool
                        )
                        frame_class_counts = np.bincount(
                            labels[mask], minlength=3
                        )[:3]
                        class_counts += frame_class_counts
                        family_class_counts[family] += frame_class_counts
                        for name, lower, upper in DISTANCE_BINS_M:
                            in_bin = distances >= lower
                            if upper is not None:
                                in_bin &= distances < upper
                            count = int((mask & in_bin & (labels == 1)).sum())
                            distance_free[name] += count
                            family_distance_free[family][name] += count
        _validate_support_contract(
            panel_name,
            class_counts,
            distance_free,
            family_class_counts,
            family_distance_free,
        )
        report[panel_name] = {
            "class_counts": {
                "unknown": int(class_counts[0]),
                "free": int(class_counts[1]),
                "occupied": int(class_counts[2]),
            },
            "distance_free_support": distance_free,
            "family_support": {
                family: {
                    "class_counts": {
                        "unknown": int(family_class_counts[family][0]),
                        "free": int(family_class_counts[family][1]),
                        "occupied": int(family_class_counts[family][2]),
                    },
                    "distance_free_support": family_distance_free[family],
                }
                for family in FAMILIES
            },
            "distance_bins_gated": list(GATED_DISTANCE_BIN_NAMES),
            "minimum_aggregate_free_cells_per_gated_bin": (
                MIN_AGGREGATE_FREE_CELLS_PER_GATED_BIN
            ),
            "minimum_per_family_free_cells_per_gated_bin": (
                MIN_FAMILY_FREE_CELLS_PER_GATED_BIN
            ),
            "distance_bins_reported_only": [
                name
                for name, _lower, _upper in DISTANCE_BINS_M
                if name not in GATED_DISTANCE_BIN_NAMES
            ],
            "asserted_after_label_independent_selection": True,
            "failure_policy": "abort_without_reselection",
            "selected_scene_npz_archives_contain_unselected_train_rows": True,
            "archive_level_arrays_materialized_during_indexed_access": True,
            "optimizer_indexes_only_selected_fit_rows": True,
            "label_shard_npz_open_events": len(grouped),
            "label_frame_access_events": 2 * len(rows),
        }
    return report


@torch.no_grad()
def _evaluate(
    model: EgomotionBevJepa,
    rows: Sequence[Mapping[str, Any]],
    *,
    panel_name: str,
    seed: int,
    device: torch.device,
    batch_size: int,
    distances: np.ndarray,
) -> dict[str, Any]:
    records, cross_scene_shuffle = attach_role_global_shuffle(
        frame_records(rows), seed=seed, namespace=panel_name
    )
    records, same_scene_shuffle = attach_same_scene_wrong_view(
        records, seed=seed, namespace=panel_name
    )
    dataset = _MicroFrameDataset(
        records,
        image_size=112,
        crop_fraction_xy=(1.0, 1.0),
        normalization_mean=NORMALIZATION_MEAN,
        normalization_std=NORMALIZATION_STD,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    conditions = {
        "correct_rgb": empty_raw_accumulator(),
        "role_global_shuffled_rgb": empty_raw_accumulator(),
        "same_scene_wrong_view_rgb": empty_raw_accumulator(),
    }
    family_conditions = {
        family: {
            "correct_rgb": empty_raw_accumulator(),
            "role_global_shuffled_rgb": empty_raw_accumulator(),
            "same_scene_wrong_view_rgb": empty_raw_accumulator(),
        }
        for family in FAMILIES
    }
    model.eval()
    try:
        for batch in loader:
            image = batch["image"].to(device)
            control = batch["control_image"].to(device)
            same_scene_control = batch["same_scene_control_image"].to(device)
            combined = torch.cat((image, control, same_scene_control), dim=0)
            logits = model.occupancy_logits(combined).detach().cpu().numpy()
            correct, shuffled, same_scene = np.split(logits, 3, axis=0)
            labels = batch["labels"].numpy()
            mask = batch["mask"].numpy()
            update_raw_accumulator(
                conditions["correct_rgb"], correct, labels, mask, distances
            )
            update_raw_accumulator(
                conditions["role_global_shuffled_rgb"],
                shuffled,
                labels,
                mask,
                distances,
            )
            update_raw_accumulator(
                conditions["same_scene_wrong_view_rgb"],
                same_scene,
                labels,
                mask,
                distances,
            )
            for index, family in enumerate(batch["family"]):
                family_name = str(family)
                update_raw_accumulator(
                    family_conditions[family_name]["correct_rgb"],
                    correct[index : index + 1],
                    labels[index : index + 1],
                    mask[index : index + 1],
                    distances,
                )
                update_raw_accumulator(
                    family_conditions[family_name]["role_global_shuffled_rgb"],
                    shuffled[index : index + 1],
                    labels[index : index + 1],
                    mask[index : index + 1],
                    distances,
                )
                update_raw_accumulator(
                    family_conditions[family_name]["same_scene_wrong_view_rgb"],
                    same_scene[index : index + 1],
                    labels[index : index + 1],
                    mask[index : index + 1],
                    distances,
                )
    finally:
        dataset.close()
    finalized = {
        name: finalize_raw_accumulator(accumulator)
        for name, accumulator in conditions.items()
    }
    family_metrics = {
        family: {
            name: finalize_raw_accumulator(accumulator)
            for name, accumulator in values.items()
        }
        for family, values in family_conditions.items()
    }
    shuffled_nll = finalized["role_global_shuffled_rgb"][
        "raw_hierarchical_balanced_nll"
    ]
    if shuffled_nll is None:
        raise ValueError("shuffled evaluation lacks hierarchical support")
    same_scene_nll = finalized["same_scene_wrong_view_rgb"][
        "raw_hierarchical_balanced_nll"
    ]
    if same_scene_nll is None:
        raise ValueError("same-scene evaluation lacks hierarchical support")
    families = {}
    for family, metrics in family_metrics.items():
        family_shuffled_nll = metrics["role_global_shuffled_rgb"][
            "raw_hierarchical_balanced_nll"
        ]
        family_same_scene_nll = metrics["same_scene_wrong_view_rgb"][
            "raw_hierarchical_balanced_nll"
        ]
        if family_shuffled_nll is None or family_same_scene_nll is None:
            raise ValueError(f"family {family} lacks hierarchical control support")
        families[family] = {
            "conditions": metrics,
            "fit_gate": fit_gate(
                metrics["correct_rgb"],
                cross_scene_shuffled_nll=float(family_shuffled_nll),
                same_scene_shuffled_nll=float(family_same_scene_nll),
            ),
        }
    return {
        "panel": panel_name,
        "frame_count": len(records),
        "conditions": finalized,
        "families": families,
        "fit_gate": fit_gate(
            finalized["correct_rgb"],
            cross_scene_shuffled_nll=float(shuffled_nll),
            same_scene_shuffled_nll=float(same_scene_nll),
        ),
        "role_global_shuffle": cross_scene_shuffle,
        "same_scene_wrong_view": same_scene_shuffle,
        "access": {
            "image_decode_events": dataset.image_decode_events,
            "label_access_events": dataset.label_access_events,
            "label_shard_npz_open_events": dataset.label_shard_npz_open_events,
            "distinct_image_paths_opened": len(dataset.opened_image_paths),
            "distinct_label_shards_opened": len(dataset.opened_shard_paths),
            "non_train_image_opens": 0,
            "non_train_label_shard_opens": 0,
        },
    }


def _all_family_and_aggregate_fit_gate_pass(report: Mapping[str, Any]) -> bool:
    families = report.get("families")
    if not isinstance(families, Mapping) or set(families) != set(FAMILIES):
        raise ValueError("fit evaluation must contain exactly five family reports")
    aggregate = report.get("fit_gate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("fit evaluation lacks its aggregate gate")
    family_passes = []
    for family in FAMILIES:
        family_report = families[family]
        if not isinstance(family_report, Mapping):
            raise ValueError(f"malformed fit family report: {family}")
        gate = family_report.get("fit_gate")
        if not isinstance(gate, Mapping):
            raise ValueError(f"fit family report lacks its gate: {family}")
        family_passes.append(bool(gate.get("passes", False)))
    return bool(aggregate.get("passes", False)) and all(family_passes)


def _terminal_fit_gate_summary(
    curve: Sequence[Mapping[str, Any]],
    *,
    maximum_steps: int,
    evaluation_interval: int,
) -> dict[str, Any]:
    if maximum_steps < 3 * evaluation_interval:
        raise ValueError("stage budget must contain at least three evaluation intervals")
    if maximum_steps % evaluation_interval:
        raise ValueError("stage budget must be divisible by the evaluation interval")
    expected_steps = [
        maximum_steps - 2 * evaluation_interval,
        maximum_steps - evaluation_interval,
        maximum_steps,
    ]
    by_step = {int(record["step"]): record for record in curve}
    if any(step not in by_step for step in expected_steps):
        raise ValueError("learning curve lacks an exact terminal evaluation")
    terminal_passes = [
        bool(by_step[step].get("all_family_and_aggregate_fit_gate_pass", False))
        for step in expected_steps
    ]
    return {
        "terminal_evaluation_steps": expected_steps,
        "terminal_evaluation_passes": terminal_passes,
        "passes": all(terminal_passes),
        "requires_aggregate_and_all_five_family_gates": True,
    }


def _train_stage(
    panel: Mapping[str, Any],
    panels: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    arm: str,
    initial_state: Mapping[str, torch.Tensor],
    primitive_to_index: Mapping[str, int],
    allowed_images: set[str],
    allowed_shards: set[str],
    device: torch.device,
    seed: int,
    batch_size: int,
    maximum_steps: int,
    evaluation_interval: int,
    learning_rate: float,
    weight_decay: float,
    stage_name: str,
) -> dict[str, Any]:
    if maximum_steps <= 0 or evaluation_interval <= 0:
        raise ValueError("training steps and evaluation interval must be positive")
    if maximum_steps < 3 * evaluation_interval:
        raise ValueError("training requires at least three evaluation intervals")
    if maximum_steps % evaluation_interval:
        raise ValueError("training steps must be divisible by evaluation interval")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    config = _model_config(panel, arm=arm, action_dim=len(primitive_to_index))
    model = EgomotionBevJepa(**config).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    dataset = _AuditedTransitionDataset(
        panels["fit"],
        primitive_to_index=primitive_to_index,
        allowed_images=allowed_images,
        allowed_shards=allowed_shards,
    )
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("fit loader has no full batch")
    iterator = iter(loader)
    uk_weights = torch.tensor(
        TRAINING_WEIGHTS["unknown_known"], dtype=torch.float32, device=device
    )
    fo_weights = torch.tensor(
        TRAINING_WEIGHTS["free_occupied"], dtype=torch.float32, device=device
    )
    distances = _distance_grid(panel)
    curve = []
    consecutive_passes = 0
    first_gate_step = None
    first_three_consecutive_gate_step = None
    step = 0
    while step < maximum_steps:
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        model.train()
        image = torch.cat(
            (raw_batch["current_image"], raw_batch["next_image"]), dim=0
        ).to(device)
        labels = torch.cat(
            (raw_batch["current_labels"], raw_batch["next_labels"]), dim=0
        ).to(device)
        mask = torch.cat(
            (raw_batch["current_mask"], raw_batch["next_mask"]), dim=0
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model.occupancy_logits(image)
        occupancy_loss = model._occupancy_loss(
            logits,
            labels,
            mask,
            None,
            unknown_known_weights=uk_weights,
            free_occupied_weights=fo_weights,
        )
        loss = 2.0 * occupancy_loss
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError(f"non-finite micro-overfit loss at step {step + 1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        if step % evaluation_interval == 0 or step == maximum_steps:
            fit_report = _evaluate(
                model,
                panels["fit"],
                panel_name="fit",
                seed=seed,
                device=device,
                batch_size=batch_size,
                distances=distances,
            )
            passed = _all_family_and_aggregate_fit_gate_pass(fit_report)
            consecutive_passes = consecutive_passes + 1 if passed else 0
            if passed and first_gate_step is None:
                first_gate_step = step
            if consecutive_passes >= 3 and first_three_consecutive_gate_step is None:
                first_three_consecutive_gate_step = step
            curve.append(
                {
                    "step": step,
                    "batch_total_loss": float(loss.detach().item()),
                    "batch_occupancy_loss": float(occupancy_loss.detach().item()),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "fit": fit_report,
                    "all_family_and_aggregate_fit_gate_pass": passed,
                    "consecutive_fit_gate_passes": consecutive_passes,
                }
            )

    if step != maximum_steps:
        raise RuntimeError("micro-overfit stage did not consume its fixed update budget")
    terminal_gate = _terminal_fit_gate_summary(
        curve,
        maximum_steps=maximum_steps,
        evaluation_interval=evaluation_interval,
    )

    final_panels = {
        panel_name: _evaluate(
            model,
            panel_rows,
            panel_name=panel_name,
            seed=seed,
            device=device,
            batch_size=batch_size,
            distances=distances,
        )
        for panel_name, panel_rows in panels.items()
    }
    state = _clone_state(model.state_dict())
    result = {
        "schema": "lewm_go2_physical_micro_overfit_stage_v1",
        "stage": stage_name,
        "arm": arm,
        "model_config": config,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "gradient_clip": 1.0,
        },
        "maximum_steps": int(maximum_steps),
        "completed_steps": int(step),
        "batch_size": int(batch_size),
        "evaluation_interval": int(evaluation_interval),
        "fixed_update_budget_consumed": True,
        "first_single_fit_gate_step": first_gate_step,
        "first_three_consecutive_fit_gate_step": first_three_consecutive_gate_step,
        "terminal_fit_gate": terminal_gate,
        "fit_gate_passed_terminal_three_evaluations": bool(terminal_gate["passes"]),
        "learning_curve": curve,
        "final_panels": final_panels,
        "final_state_sha256": _state_dict_sha256(state),
        "transition_dataset_access": {
            "image_decode_events": dataset.image_decode_events,
            "label_shard_npz_open_events": dataset.label_shard_npz_open_events,
            "distinct_images_opened": len(dataset.opened_images),
            "distinct_label_shards_opened": len(dataset.opened_shards),
            "non_train_image_opens": 0,
            "non_train_label_shard_opens": 0,
        },
    }
    del model, optimizer, state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _determinism(seed: int) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": int(seed),
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
    }


def _reconciled_train_access(
    support: Mapping[str, Mapping[str, Any]],
    faithful: Mapping[str, Mapping[str, Any]],
    ceiling: Mapping[str, Mapping[str, Any]] | None,
    *,
    distinct_images_hashed: int,
    distinct_label_shards_hashed: int,
) -> dict[str, Any]:
    support_shard_opens = sum(
        int(record["label_shard_npz_open_events"]) for record in support.values()
    )
    support_label_frames = sum(
        int(record["label_frame_access_events"]) for record in support.values()
    )
    training = Counter()
    evaluation = Counter()
    stages: list[tuple[str, Mapping[str, Mapping[str, Any]]]] = [
        ("production_faithful", faithful)
    ]
    if ceiling is not None:
        stages.append(("ceiling_optimizer", ceiling))
    per_stage: dict[str, Any] = {}
    for stage_name, stage in stages:
        stage_training = Counter()
        stage_evaluation = Counter()
        for arm, result in stage.items():
            transition = result["transition_dataset_access"]
            completed_steps = int(result["completed_steps"])
            batch_size = int(result["batch_size"])
            stage_training["image_decode_events"] += int(
                transition["image_decode_events"]
            )
            stage_training["label_shard_npz_open_events"] += int(
                transition["label_shard_npz_open_events"]
            )
            stage_training["transition_row_access_events"] += (
                completed_steps * batch_size
            )
            stage_training["label_frame_access_events"] += (
                2 * completed_steps * batch_size
            )
            stage_training["model_output_frames"] += (
                2 * completed_steps * batch_size
            )
            reports = [entry["fit"] for entry in result["learning_curve"]]
            reports.extend(result["final_panels"].values())
            for report in reports:
                access = report["access"]
                stage_evaluation["image_decode_events"] += int(
                    access["image_decode_events"]
                )
                stage_evaluation["label_frame_access_events"] += int(
                    access["label_access_events"]
                )
                stage_evaluation["label_shard_npz_open_events"] += int(
                    access["label_shard_npz_open_events"]
                )
                stage_evaluation["model_output_frames"] += 3 * int(
                    access["label_access_events"]
                )
        training.update(stage_training)
        evaluation.update(stage_evaluation)
        per_stage[stage_name] = {
            "training": dict(sorted(stage_training.items())),
            "evaluation": dict(sorted(stage_evaluation.items())),
        }
    integrity_image_opens = 2 * int(distinct_images_hashed)
    integrity_shard_opens = 2 * int(distinct_label_shards_hashed)
    total_image_decodes = training["image_decode_events"] + evaluation[
        "image_decode_events"
    ]
    total_shard_npz_opens = (
        support_shard_opens
        + training["label_shard_npz_open_events"]
        + evaluation["label_shard_npz_open_events"]
    )
    total_label_frames = (
        support_label_frames
        + training["label_frame_access_events"]
        + evaluation["label_frame_access_events"]
    )
    return {
        "schema": "lewm_go2_physical_micro_overfit_train_access_reconciliation_v1",
        "support_audit": {
            "image_decode_events": 0,
            "label_frame_access_events": support_label_frames,
            "label_shard_npz_open_events": support_shard_opens,
            "model_output_frames": 0,
        },
        "integrity_hashing": {
            "passes": 2,
            "distinct_image_files_per_pass": int(distinct_images_hashed),
            "distinct_label_shard_files_per_pass": int(distinct_label_shards_hashed),
            "image_byte_open_events": integrity_image_opens,
            "label_shard_byte_open_events": integrity_shard_opens,
        },
        "per_stage": per_stage,
        "training_totals": dict(sorted(training.items())),
        "evaluation_totals": dict(sorted(evaluation.items())),
        "all_train_role_totals": {
            "image_integrity_hash_open_events": integrity_image_opens,
            "image_decode_events": total_image_decodes,
            "image_byte_open_events": integrity_image_opens + total_image_decodes,
            "label_frame_access_events": total_label_frames,
            "label_shard_integrity_hash_open_events": integrity_shard_opens,
            "label_shard_npz_open_events": total_shard_npz_opens,
            "label_shard_byte_open_events": (
                integrity_shard_opens + total_shard_npz_opens
            ),
            "model_output_frames": (
                training["model_output_frames"] + evaluation["model_output_frames"]
            ),
        },
        "events_reconciled": True,
        "non_train_image_byte_open_events": 0,
        "non_train_label_shard_byte_open_events": 0,
        "non_train_model_output_frames": 0,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--expected-panel-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--faithful-steps",
        type=int,
    )
    parser.add_argument(
        "--ceiling-steps",
        type=int,
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
    )
    parser.add_argument("--non-authoritative-smoke", action="store_true")
    parser.add_argument(
        "--run-ceiling-on-failure",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; result artifacts are immutable")
    mode_defaults = (
        SMOKE_EXECUTION if args.non_authoritative_smoke else AUTHORITATIVE_EXECUTION
    )
    for argument_name in (
        "batch_size",
        "faithful_steps",
        "ceiling_steps",
        "evaluation_interval",
    ):
        if getattr(args, argument_name) is None:
            setattr(args, argument_name, int(mode_defaults[argument_name]))
    actual_execution = {
        "batch_size": int(args.batch_size),
        "faithful_steps": int(args.faithful_steps),
        "ceiling_steps": int(args.ceiling_steps),
        "evaluation_interval": int(args.evaluation_interval),
    }
    if args.non_authoritative_smoke:
        bounded = all(
            (
                1 <= actual_execution["batch_size"]
                <= SMOKE_EXECUTION_BOUNDS["maximum_batch_size"],
                1 <= actual_execution["faithful_steps"]
                <= SMOKE_EXECUTION_BOUNDS["maximum_faithful_steps"],
                1 <= actual_execution["ceiling_steps"]
                <= SMOKE_EXECUTION_BOUNDS["maximum_ceiling_steps"],
                1 <= actual_execution["evaluation_interval"]
                <= SMOKE_EXECUTION_BOUNDS["maximum_evaluation_interval"],
            )
        )
        interval = actual_execution["evaluation_interval"]
        stage_budgets_are_valid = all(
            steps >= 3 * interval and steps % interval == 0
            for steps in (
                actual_execution["faithful_steps"],
                actual_execution["ceiling_steps"],
            )
        )
        if not bounded or not stage_budgets_are_valid:
            parser.error(
                "non-authoritative smoke execution must stay within "
                f"{SMOKE_EXECUTION_BOUNDS}, contain at least three evaluation "
                "intervals per stage, and use exactly divisible stage budgets"
            )
    elif actual_execution != AUTHORITATIVE_EXECUTION:
        parser.error(
            "authoritative execution must be exactly "
            f"{AUTHORITATIVE_EXECUTION}; use --non-authoritative-smoke for "
            "bounded alternate settings"
        )
    if args.seed not in (20260710, 20260711):
        parser.error("authoritative seed must be 20260710 or 20260711")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    invocation_argv = (
        list(sys.argv)
        if argv is None
        else [str(Path(__file__).resolve()), *(str(value) for value in argv)]
    )
    started_at = datetime.now(timezone.utc).isoformat()
    panel_path = args.panel_manifest.resolve()
    output_path = args.output.resolve()
    panel_file_sha256 = _sha256_file(panel_path)
    if panel_file_sha256 != str(args.expected_panel_sha256):
        raise ValueError("panel manifest differs from the precommitted SHA-256")
    source_start = _source_hashes()
    git_start = _git_snapshot()
    deterministic = _determinism(int(args.seed))
    panel = _read_json(panel_path)
    panels = validate_panel_manifest(panel)
    panel_source_hashes = panel.get("source_hashes")
    if not isinstance(panel_source_hashes, Mapping):
        raise ValueError("panel lacks preparer source provenance")
    shared_source_keys = {
        "contract": "contract",
        "execution_contract": "generalization_execution_contract",
        "micro_overfit_protocol": "micro_overfit_protocol",
        "preparer": "panel_preparer",
    }
    for panel_key, runner_key in shared_source_keys.items():
        panel_record = panel_source_hashes.get(panel_key)
        if not isinstance(panel_record, Mapping) or str(
            panel_record.get("sha256", "")
        ) != str(source_start[runner_key]["sha256"]):
            raise ValueError(
                f"panel was prepared under a different frozen source: {panel_key}"
            )

    preparer_ledger = panel.get("artifact_access_ledger")
    metadata_ledger = panel.get("metadata_access")
    if not isinstance(preparer_ledger, Mapping) or not isinstance(
        metadata_ledger, Mapping
    ):
        raise ValueError("panel lacks its isolation ledgers")
    if (
        metadata_ledger.get(
            "full_row_objects_parsed_including_non_train_path_metadata"
        )
        is not True
        or metadata_ledger.get("non_train_artifact_paths_emitted_to_panel")
        is not False
        or metadata_ledger.get("non_train_artifact_paths_dereferenced") is not False
    ):
        raise ValueError("panel metadata-access ledger is incomplete or dishonest")
    for role in ("checkpoint_selection", "probability_calibration", "g2_evaluation"):
        record = preparer_ledger.get(role)
        if not isinstance(record, Mapping) or any(
            int(record.get(name, -1)) != 0
            for name in ("label_shard_byte_opens", "image_byte_opens", "model_outputs")
        ):
            raise ValueError(f"panel records forbidden artifact contact: {role}")

    images, shards = _artifact_contract(panels)
    verification_start = _verify_artifacts(images, shards)
    support = _support_audit(panel, panels)
    primitive_vocabulary = tuple(map(str, panel.get("primitive_vocabulary", ())))
    if len(primitive_vocabulary) != 9 or len(set(primitive_vocabulary)) != 9:
        raise ValueError("micro-overfit panel must bind the nine train primitives")
    primitive_to_index = {
        primitive: index for index, primitive in enumerate(primitive_vocabulary)
    }
    used_primitives = {
        str(row["primitive"]) for rows in panels.values() for row in rows
    }
    if not used_primitives <= set(primitive_to_index):
        raise ValueError("micro-overfit panel uses an unbound primitive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    initial_states, initialization = _initial_states(
        panel, action_dim=len(primitive_to_index), seed=int(args.seed)
    )
    allowed_images = set(images)
    allowed_shards = set(shards)

    faithful = {}
    for arm in ARM_CONFIGS:
        faithful[arm] = _train_stage(
            panel,
            panels,
            arm=arm,
            initial_state=initial_states[arm],
            primitive_to_index=primitive_to_index,
            allowed_images=allowed_images,
            allowed_shards=allowed_shards,
            device=device,
            seed=int(args.seed),
            batch_size=int(args.batch_size),
            maximum_steps=int(args.faithful_steps),
            evaluation_interval=int(args.evaluation_interval),
            learning_rate=2e-4,
            weight_decay=1e-4,
            stage_name="production_faithful",
        )
    any_failure = any(
        not bool(result["fit_gate_passed_terminal_three_evaluations"])
        for result in faithful.values()
    )
    ceiling = None
    if any_failure:
        ceiling = {}
        for arm in ARM_CONFIGS:
            ceiling[arm] = _train_stage(
                panel,
                panels,
                arm=arm,
                initial_state=initial_states[arm],
                primitive_to_index=primitive_to_index,
                allowed_images=allowed_images,
                allowed_shards=allowed_shards,
                device=device,
                seed=int(args.seed),
                batch_size=int(args.batch_size),
                maximum_steps=int(args.ceiling_steps),
                evaluation_interval=int(args.evaluation_interval),
                learning_rate=1e-3,
                weight_decay=0.0,
                stage_name="ceiling_optimizer",
            )

    decision = classify_cross_arm_decision(faithful, ceiling, seed=int(args.seed))
    access_reconciliation = _reconciled_train_access(
        support,
        faithful,
        ceiling,
        distinct_images_hashed=len(images),
        distinct_label_shards_hashed=len(shards),
    )

    verification_end = _verify_artifacts(images, shards)
    if verification_end != verification_start:
        raise RuntimeError("selected train artifact set changed during execution")
    source_end = _source_hashes()
    if source_end != source_start:
        raise RuntimeError("micro-overfit source changed during execution")
    git_end = _git_snapshot()
    if _sha256_file(panel_path) != panel_file_sha256:
        raise RuntimeError("micro-overfit panel changed during execution")

    authoritative = not bool(args.non_authoritative_smoke)
    core = {
        "schema": RESULT_SCHEMA if authoritative else SMOKE_RESULT_SCHEMA,
        "authoritative": authoritative,
        "promotion_eligible": authoritative,
        "created_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "invocation": {
            "argv": invocation_argv,
            "resolved": {
                "panel_manifest": str(panel_path),
                "expected_panel_sha256": str(args.expected_panel_sha256),
                "output": str(output_path),
                "device": str(device),
                "seed": int(args.seed),
                "batch_size": int(args.batch_size),
                "faithful_steps": int(args.faithful_steps),
                "ceiling_steps": int(args.ceiling_steps),
                "evaluation_interval": int(args.evaluation_interval),
                "legacy_run_ceiling_on_failure_flag_supplied": bool(
                    args.run_ceiling_on_failure
                ),
                "non_authoritative_smoke": bool(args.non_authoritative_smoke),
            },
        },
        "inputs": {
            "panel_manifest": {
                "path": str(panel_path),
                "sha256": panel_file_sha256,
                "expected_sha256": str(args.expected_panel_sha256),
                "content_sha256": str(panel["content_sha256"]),
                "pre_deserialization_hash_match": True,
            }
        },
        "execution": {
            "authoritative": authoritative,
            "promotion_eligible": authoritative,
            "device": str(device),
            "batch_size": int(args.batch_size),
            "faithful_steps": int(args.faithful_steps),
            "ceiling_steps": int(args.ceiling_steps),
            "evaluation_interval": int(args.evaluation_interval),
            "ceiling_policy": "mandatory_for_both_arms_if_either_faithful_arm_fails",
            "legacy_run_ceiling_on_failure_flag_has_no_behavioral_effect": True,
            "non_authoritative_smoke": bool(args.non_authoritative_smoke),
            "smoke_execution_bounds": (
                SMOKE_EXECUTION_BOUNDS if args.non_authoritative_smoke else None
            ),
            "determinism": deterministic,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "hip_version": torch.version.hip,
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "contract": {
            "authoritative": authoritative,
            "promotion_eligible": authoritative,
            "arms": ARM_CONFIGS,
            "families": list(FAMILIES),
            "panels": list(PANELS),
            "training_weights": TRAINING_WEIGHTS,
            "image_preprocessing_identical_between_arms": True,
            "lift_type_identical_between_arms": PROJECTIVE_COLUMN_ATTENTION_LIFT,
            "multi_view_fusion_used": False,
            "calibration_fitted_or_applied": False,
            "threshold_search_performed": False,
            "intervention_claim": (
                "patch7_16x16_tokenization_and_patch_embedding_bundle_not_"
                "token_resolution_alone"
            ),
            "equal_samples_and_fixed_updates_between_arms": True,
            "independent_early_stopping_used": False,
            "terminal_fit_gate_requires_aggregate_and_all_five_families": True,
            "source_scene_npz_archives_include_unselected_train_rows_and_arrays": True,
            "archive_level_arrays_materialized_by_npz_access": True,
            "optimizer_indexes_only_selected_fit_rows": True,
        },
        "initialization": initialization,
        "post_selection_support_audit": support,
        "stages": {"production_faithful": faithful, "ceiling_optimizer": ceiling},
        "cross_arm_decision": decision,
        "artifact_verification": verification_end,
        "access_ledger": {
            "global_index_contact": "preparer_metadata_only",
            "runner_input_contains_only_train_rows": True,
            "train_image_paths_available": len(images),
            "train_label_shard_paths_available": len(shards),
            "train_role_event_reconciliation": access_reconciliation,
            "checkpoint_selection": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "probability_calibration": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
            "g2_evaluation": {
                "image_byte_opens": 0,
                "label_shard_byte_opens": 0,
                "model_outputs": 0,
            },
        },
        "source_hashes": source_end,
        "git": {"start": git_start, "end": git_end},
    }
    payload = {**core, "content_sha256": canonical_json_sha256(core)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "file_sha256": _sha256_file(output_path),
                "content_sha256": payload["content_sha256"],
                "authoritative": authoritative,
                "promotion_eligible": authoritative,
                "faithful_fit_gate": {
                    arm: bool(
                        result["fit_gate_passed_terminal_three_evaluations"]
                    )
                    for arm, result in faithful.items()
                },
                "ceiling_ran": ceiling is not None,
                "classification": decision["classification"],
                "patch7_full_train_candidate_licensed": decision[
                    "patch7_full_train_candidate_licensed"
                ],
                "second_seed_needed": decision["second_seed_needed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
