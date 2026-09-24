#!/usr/bin/env python3
"""Run the capped RGB swept-progress survival joint-JEPA V1 development probe.

This is intentionally a lean experiment runner, not a new custody framework.
It accepts only the already-reviewed development inputs used by the predecessor,
checks the model-free label bundle, trains one fresh full arm, evaluates cheap
inference controls, and writes a development-only result.  It never names or
opens any final-test role.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
LABEL_ROOT_RELATIVE_PATH = ".generated/go2_swept_progress_survival_labels_v1"
OUTPUT_RELATIVE_PATH = (
    ".generated/go2_rgb_swept_progress_survival_joint_jepa_v1/attempt_v1"
)
LABEL_MANIFEST_NAME = "manifest.json"
LABEL_MANIFEST_CONTENT_SHA256 = (
    "6e0ea572612cdf94cb6dd91dffb90e50c828053617f69b42307161c958700c03"
)
LABEL_MANIFEST_FILE_SHA256 = (
    "edc0df8c796f97d3f91c8c3796e9795a4355dceac79770b91de382132fe8e1d3"
)
LABEL_MANIFEST_BYTE_COUNT = 5_914
REQUIRED_GPU_NAME = "AMD Radeon AI PRO R9700"
REQUIRED_GPU_MEMORY_BYTES = 34_208_743_424
ACTION_ORDER = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
ROLE_FILES = {
    "train": "train.jsonl",
    "probability_calibration": "calibration.jsonl",
    "checkpoint_selection": "selection.jsonl",
}
NON_HOLD_INDICES = (0, 1, 2, 3, 4, 5, 7, 8)
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
PRESENTATIONS_PER_UPDATE = 16
MAXIMUM_UPDATES = 1_000
MAXIMUM_PRESENTATIONS = 16_000
CONSTRUCTOR_INITIALIZATION_SEED = 20_260_712
EXPERIMENT_SEED = 20_260_728
BOOTSTRAP_SEED = 20_260_728
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_LOWER_INDEX = 249
PROGRESS_SEGMENT_M = 0.1
PROGRESS_HORIZON_M = 1.5
CONTROL_NAMES = (
    "coordinate_matched_persistence",
    "shuffled_action",
    "wrong_rgb",
    "train_action_mean_prior",
)
ALL_ARM_NAMES = ("full", *CONTROL_NAMES)
REGISTERED_FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)

# Frozen before GPU execution.  Equality passes the absolute floors but not the
# strictly-positive control deltas.
GATE_THRESHOLDS = {
    "semantic_balanced_accuracy_min": 0.80,
    "semantic_free_recall_min": 0.85,
    "semantic_occupied_recall_min": 0.70,
    "semantic_unknown_recall_min": 0.90,
    "semantic_rough_occupied_recall_min": 0.65,
    "informative_utility_min": 0.85,
    "family_informative_utility_min": 0.70,
    "selected_zero_prefix_rate_max": 0.05,
    "family_selected_zero_prefix_rate_max": 0.20,
    "pair_concordance_min": 0.75,
    "family_pair_concordance_min": 0.60,
    "positive_control_family_count_min": 6,
}


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(core)
    value["content_sha256"] = _canonical_json_sha256(value)
    return value


def _read_bound_file(path: Path, record: Mapping[str, Any]) -> bytes:
    """Read one exact regular file after validating its small binding."""

    if (
        type(record) is not dict
        or record.get("path") != path.name
        or type(record.get("file_sha256")) is not str
        or type(record.get("byte_count")) is not int
        or record["byte_count"] < 0
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError(f"invalid bound input {path.name!r}")
    raw = path.read_bytes()
    if (
        len(raw) != record["byte_count"]
        or hashlib.sha256(raw).hexdigest() != record["file_sha256"]
    ):
        raise PermissionError(f"bound input changed: {path.name}")
    return raw


def _parse_canonical_object(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PermissionError(f"{name} is not canonical JSON") from error
    if type(value) is not dict or raw != _canonical_json_bytes(value) + b"\n":
        raise PermissionError(f"{name} is not one canonical JSON object")
    content_sha256 = value.get("content_sha256")
    core = {key: item for key, item in value.items() if key != "content_sha256"}
    if content_sha256 != _canonical_json_sha256(core):
        raise PermissionError(f"{name} content hash changed")
    return value


def _parse_canonical_jsonl(raw: bytes, *, name: str) -> tuple[Mapping[str, Any], ...]:
    if not raw or not raw.endswith(b"\n"):
        raise PermissionError(f"{name} is empty or lacks a terminal newline")
    rows: list[Mapping[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PermissionError(f"{name} row {index} is invalid JSON") from error
        if type(row) is not dict or line != _canonical_json_bytes(row):
            raise PermissionError(f"{name} row {index} is not canonical")
        rows.append(row)
    return tuple(rows)


def load_label_bundle_v1(
    repository_root: Path,
    *,
    labels_api: Any,
) -> tuple[Mapping[str, Any], Mapping[str, tuple[Mapping[str, Any], ...]]]:
    """Verify the manifest and all three role JSONLs before returning rows."""

    label_root = Path(repository_root) / LABEL_ROOT_RELATIVE_PATH
    manifest_path = label_root / LABEL_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise PermissionError("swept-progress label manifest is absent")
    manifest_raw = manifest_path.read_bytes()
    if (
        len(manifest_raw) != LABEL_MANIFEST_BYTE_COUNT
        or hashlib.sha256(manifest_raw).hexdigest() != LABEL_MANIFEST_FILE_SHA256
    ):
        raise PermissionError("frozen swept-progress label manifest changed")
    manifest = _parse_canonical_object(
        manifest_raw, name="swept-progress label manifest"
    )
    if (
        manifest.get("schema") != labels_api.MANIFEST_SCHEMA
        or manifest.get("content_sha256") != LABEL_MANIFEST_CONTENT_SHA256
        or manifest.get("status") != "complete_model_free_development_labels"
        or manifest.get("roles") != list(labels_api.ROLE_ORDER)
        or manifest.get("role_files") != ROLE_FILES
        or manifest.get("action_order") != list(ACTION_ORDER)
    ):
        raise PermissionError("swept-progress label manifest contract changed")
    records = manifest.get("files")
    if type(records) is not list or len(records) != 3:
        raise PermissionError("label manifest must bind exactly three JSONLs")
    by_name = {record.get("path"): record for record in records if type(record) is dict}
    if set(by_name) != set(ROLE_FILES.values()):
        raise PermissionError("label manifest role-file set changed")
    rows_by_role: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for role in labels_api.ROLE_ORDER:
        filename = ROLE_FILES[role]
        record = by_name[filename]
        raw = _read_bound_file(label_root / filename, record)
        rows = _parse_canonical_jsonl(raw, name=f"{role} labels")
        if (
            record.get("dataset_role") != role
            or record.get("state_count") != labels_api.v4.ROLE_STATE_COUNTS[role]
            or record.get("action_row_count") != len(rows)
            or len(rows) != labels_api.v4.ROLE_STATE_COUNTS[role] * len(ACTION_ORDER)
            or record.get("ordered_row_content_sha256")
            != _canonical_json_sha256([row.get("content_sha256") for row in rows])
        ):
            raise PermissionError(f"{role} label population or order changed")
        labels_api._state_groups(rows, role=role, frozen=True)
        rows_by_role[role] = rows
    return manifest, rows_by_role


def _exact_runtime_bindings_v1(
    manifest: Mapping[str, Any], *, direct_contract: Any, labels_api: Any
) -> Mapping[str, Mapping[str, Any]]:
    inputs = manifest.get("input_bindings", {}).get("inputs")
    if type(inputs) is not dict:
        raise PermissionError("label manifest raw input bindings are absent")
    expected = {
        "raw_manifest": direct_contract.RUNTIME_BINDINGS[
            direct_contract.RAW_MANIFEST_RELATIVE_PATH
        ],
        "raw_audit": direct_contract.RUNTIME_BINDINGS[
            direct_contract.RAW_AUDIT_RELATIVE_PATH
        ],
        "schedule": direct_contract.RUNTIME_BINDINGS[
            direct_contract.SCHEDULE_RELATIVE_PATH
        ],
    }
    if any(inputs.get(name) != record for name, record in expected.items()):
        raise PermissionError("raw manifest, audit, or schedule binding changed")
    raw_root = direct_contract.RAW_ROOT_RELATIVE_PATH
    for name, filename, digest in (
        ("raw_pairs", "pairs.jsonl", labels_api.v4.RAW_PAIRS_FILE_SHA256),
        ("raw_endpoints", "endpoints.jsonl", labels_api.v4.RAW_ENDPOINTS_FILE_SHA256),
    ):
        record = inputs.get(name)
        if (
            type(record) is not dict
            or record.get("path") != f"{raw_root}/{filename}"
            or record.get("file_sha256") != digest
        ):
            raise PermissionError(f"exact {name} binding changed")
    return inputs


def _wrong_rgb_mapping_v1(labels: Any, *, metrics_api: Any) -> Any:
    endpoints = tuple(
        (labels.role, scene, endpoint)
        for scene, endpoint in zip(labels.scene_ids, labels.endpoint_ids, strict=True)
    )
    return metrics_api.wrong_rgb_endpoint_mapping(endpoints)


def _shared_head_logits_with_masks_v1(
    model: Any,
    action_latents: Any,
    masks: Any,
    *,
    torch: Any,
) -> Any:
    """Pool with alternate fixed masks through the exact learned shared output."""

    if action_latents.ndim != 5 or tuple(action_latents.shape[1:]) != (
        9, 64, 64, 64
    ):
        raise ValueError("control latents must have shape (B,9,64,64,64)")
    if tuple(masks.shape) != (9, 16, 64, 64) or masks.dtype != torch.bool:
        raise ValueError("control masks must be bool with shape (9,16,64,64)")
    weights = masks.to(
        device=action_latents.device, dtype=action_latents.dtype
    )
    counts = weights.sum(dim=(-2, -1))
    if not bool((counts > 0).all()):
        raise ValueError("every control mask must be nonempty")
    pooled = torch.einsum("bachw,akhw->bakc", action_latents, weights)
    pooled = pooled / counts[None, :, :, None]
    logits = model.predictor.swept_progress_head.output(pooled).squeeze(-1)
    if tuple(logits.shape) != (action_latents.shape[0], 9, 16):
        raise RuntimeError("control survival logits changed shape")
    return logits


def score_role_v1(
    model: Any,
    loader: Any,
    pairs: Sequence[Mapping[str, Any]],
    labels: Any,
    action_prior_m: Any,
    device: Any,
    *,
    torch: Any,
    np: Any,
    training_core: Any,
    current_frame_persistence_masks: Any,
    metrics_api: Any,
) -> Mapping[str, Any]:
    training_core.validate_pairs_against_labels_v1(pairs, labels)
    wrong = _wrong_rgb_mapping_v1(labels, metrics_api=metrics_api)
    scores: dict[str, list[Any]] = {name: [] for name in ALL_ARM_NAMES[:-1]}
    confusion = torch.zeros((3, 3), dtype=torch.long)
    rough_confusion = torch.zeros((3, 3), dtype=torch.long)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(pairs), MICROBATCH_SIZE):
                selected = pairs[start : start + MICROBATCH_SIZE]
                current_rgb = torch.stack([
                    loader.image(
                        str(pair["current_endpoint_sha256"]), role=labels.role,
                        stage=f"score_{labels.role}", kind="current",
                    ) for pair in selected
                ]).to(device)
                next_rgb = torch.stack([
                    loader.image(
                        str(pair["next_endpoint_sha256"]), role=labels.role,
                        stage=f"score_{labels.role}", kind="next",
                    ) for pair in selected
                ]).to(device)
                current_labels = torch.stack([
                    loader.raster_label(
                        str(pair["current_endpoint_sha256"]), role=labels.role,
                        stage=f"score_{labels.role}", scope="observation",
                    ) for pair in selected
                ]).long().to(device)
                next_labels = torch.stack([
                    loader.raster_label(
                        str(pair["next_endpoint_sha256"]), role=labels.role,
                        stage=f"score_{labels.role}", scope="observation",
                    ) for pair in selected
                ]).long().to(device)
                wrong_rgb = torch.stack([
                    loader.image(
                        wrong.by_endpoint[(
                            labels.role,
                            str(pair["scene_id"]),
                            str(pair["current_endpoint_sha256"]),
                        )],
                        role=labels.role,
                        stage=f"score_{labels.role}_wrong_rgb",
                        kind="endpoint",
                    ) for pair in selected
                ]).to(device)

                current = model.encode_online(current_rgb)
                next_latent = model.encode_online(next_rgb)
                full = training_core.score_full_control_v1(model, current)
                persistence_latents = current[:, None].expand(-1, 9, -1, -1, -1)
                persistence_logits = _shared_head_logits_with_masks_v1(
                    model,
                    persistence_latents,
                    current_frame_persistence_masks,
                    torch=torch,
                )
                persistence = training_core._control_scores(None, persistence_logits)
                shuffled = training_core.score_shuffled_action_control_v1(
                    model, full.predicted_latents
                )
                wrong_control = training_core.score_wrong_rgb_control_v1(
                    model, wrong_rgb
                )
                scores["full"].append(full.expected_progress_m.cpu())
                scores["coordinate_matched_persistence"].append(
                    persistence.expected_progress_m.cpu()
                )
                scores["shuffled_action"].append(
                    shuffled.expected_progress_m.cpu()
                )
                scores["wrong_rgb"].append(
                    wrong_control.expected_progress_m.cpu()
                )

                for logits, target in (
                    (model.semantic_logits_from_latent(current), current_labels),
                    (model.semantic_logits_from_latent(next_latent), next_labels),
                ):
                    predicted = logits.argmax(dim=1)
                    counts = torch.bincount(
                        (target.reshape(-1) * 3 + predicted.reshape(-1)).cpu(),
                        minlength=9,
                    ).reshape(3, 3)
                    confusion += counts
                    rough_rows = torch.tensor(
                        [pair["family"] == "rough_local_dynamics" for pair in selected],
                        dtype=torch.bool,
                        device=device,
                    )
                    if bool(rough_rows.any()):
                        rough_target = target[rough_rows]
                        rough_predicted = predicted[rough_rows]
                        rough_confusion += torch.bincount(
                            (rough_target.reshape(-1) * 3 + rough_predicted.reshape(-1)).cpu(),
                            minlength=9,
                        ).reshape(3, 3)
    finally:
        model.train(was_training)
    arrays = {
        name: torch.cat(parts, dim=0).numpy() for name, parts in scores.items()
    }
    arrays["train_action_mean_prior"] = np.broadcast_to(
        np.asarray(action_prior_m, dtype=np.float64)[None, :],
        (len(pairs), len(ACTION_ORDER)),
    ).copy()
    if any(
        value.shape != (len(pairs), len(ACTION_ORDER))
        or not np.isfinite(value).all()
        or (value < 0.0).any()
        or (value > PROGRESS_HORIZON_M + 1e-6).any()
        for value in arrays.values()
    ):
        raise FloatingPointError("role progress scores changed shape or range")
    return {
        "scores_m": arrays,
        "semantic_confusion": confusion.numpy(),
        "rough_semantic_confusion": rough_confusion.numpy(),
        "wrong_rgb_mapping_sha256": wrong.mapping_sha256,
    }


def _utility_values_v1(predicted_m: Any, target_prefix: Any, informative: Any, np: Any) -> Any:
    predicted = np.asarray(predicted_m, dtype=np.float64)[:, NON_HOLD_INDICES]
    target = np.asarray(target_prefix, dtype=np.int64)[:, NON_HOLD_INDICES]
    mask = np.asarray(informative, dtype=np.bool_)
    chosen = predicted.argmax(axis=1)
    oracle = target.max(axis=1)
    if bool((oracle[mask] <= 0).any()):
        raise ValueError("informative state has no positive oracle prefix")
    values = np.full(len(target), np.nan, dtype=np.float64)
    values[mask] = target[np.arange(len(target)), chosen][mask] / oracle[mask]
    return values


def _pair_concordance_v1(predicted_m: Any, target_prefix: Any, selected: Any, np: Any) -> tuple[float, int]:
    predicted = np.asarray(predicted_m, dtype=np.float64)[selected][:, NON_HOLD_INDICES]
    target = np.asarray(target_prefix, dtype=np.int64)[selected][:, NON_HOLD_INDICES]
    better = target[:, :, None] > target[:, None, :]
    pair_count = int(better.sum())
    if pair_count == 0:
        raise ValueError("pair concordance has no unequal target pair")
    correct = (predicted[:, :, None] > predicted[:, None, :]) & better
    return float(correct.sum() / pair_count), pair_count


def _progress_calibration_v1(predicted_m: Any, target_m: Any, np: Any) -> Mapping[str, Any]:
    predicted = np.asarray(predicted_m, dtype=np.float64).reshape(-1)
    target = np.asarray(target_m, dtype=np.float64).reshape(-1)
    edges = np.linspace(0.0, PROGRESS_HORIZON_M, 11)
    rows = []
    weighted_gap = 0.0
    for index in range(10):
        mask = (predicted >= edges[index]) & (
            predicted <= edges[index + 1]
            if index == 9 else predicted < edges[index + 1]
        )
        count = int(mask.sum())
        predicted_mean = float(predicted[mask].mean()) if count else None
        target_mean = float(target[mask].mean()) if count else None
        gap = abs(predicted_mean - target_mean) if count else None
        if gap is not None:
            weighted_gap += gap * count / len(predicted)
        rows.append({
            "lower_m": float(edges[index]),
            "upper_m": float(edges[index + 1]),
            "count": count,
            "predicted_mean_m": predicted_mean,
            "target_mean_m": target_mean,
            "absolute_gap_m": gap,
        })
    return {"bin_count": 10, "weighted_absolute_gap_m": weighted_gap, "bins": rows}


def scientific_metrics_v1(
    predicted_m: Any,
    target_prefix: Any,
    informative: Any,
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    np: Any,
) -> Mapping[str, Any]:
    predicted = np.asarray(predicted_m, dtype=np.float64)
    target_prefix = np.asarray(target_prefix, dtype=np.int64)
    target_m = target_prefix.astype(np.float64) * PROGRESS_SEGMENT_M
    informative = np.asarray(informative, dtype=np.bool_)
    if predicted.shape != target_prefix.shape or predicted.shape[1:] != (9,):
        raise ValueError("progress metrics require matching (N,9) arrays")
    if len(scene_ids) != len(predicted) or len(family_ids) != len(predicted):
        raise ValueError("metric identities do not match score rows")
    utility = _utility_values_v1(predicted, target_prefix, informative, np)

    def group(mask: Any) -> Mapping[str, Any]:
        selected = informative & mask
        count = int(selected.sum())
        if count == 0:
            return {
                "informative_state_count": 0,
                "normalized_chosen_prefix_utility": None,
                "selected_zero_prefix_rate": None,
                "unequal_pair_concordance": None,
                "unequal_pair_count": 0,
            }
        concordance, pair_count = _pair_concordance_v1(
            predicted, target_prefix, selected, np
        )
        chosen = predicted[:, NON_HOLD_INDICES].argmax(axis=1)
        chosen_target = target_prefix[:, NON_HOLD_INDICES][
            np.arange(len(target_prefix)), chosen
        ]
        return {
            "informative_state_count": count,
            "normalized_chosen_prefix_utility": float(utility[selected].mean()),
            "selected_zero_prefix_rate": float((chosen_target[selected] == 0).mean()),
            "unequal_pair_concordance": concordance,
            "unequal_pair_count": pair_count,
        }

    family_array = np.asarray(family_ids, dtype=object)
    scene_array = np.asarray(scene_ids, dtype=object)
    return {
        "state_count": len(predicted),
        "informative_state_count": int(informative.sum()),
        "expected_progress_mae_m": float(np.abs(predicted - target_m).mean()),
        "informative_expected_progress_mae_m": float(
            np.abs(predicted[informative] - target_m[informative]).mean()
        ),
        "overall": group(np.ones(len(predicted), dtype=np.bool_)),
        "families": {
            family: group(family_array == family)
            for family in sorted(set(family_ids))
        },
        "scene_informative_state_counts": {
            scene: int((informative & (scene_array == scene)).sum())
            for scene in sorted(set(scene_ids))
        },
        "progress_calibration": _progress_calibration_v1(predicted, target_m, np),
    }


def semantic_metrics_v1(confusion: Any, rough_confusion: Any, *, np: Any) -> Mapping[str, Any]:
    matrix = np.asarray(confusion, dtype=np.float64)
    rough = np.asarray(rough_confusion, dtype=np.float64)
    if matrix.shape != (3, 3) or rough.shape != (3, 3):
        raise ValueError("semantic confusion matrices must be 3x3")
    supports = matrix.sum(axis=1)
    if bool((supports <= 0).any()) or rough[2].sum() <= 0:
        raise ValueError("semantic confusion lacks required class support")
    recalls = np.diag(matrix) / supports
    return {
        "confusion_true_row_predicted_column": matrix.astype(np.int64).tolist(),
        "rough_confusion_true_row_predicted_column": rough.astype(np.int64).tolist(),
        "balanced_accuracy": float(recalls.mean()),
        "unknown_recall": float(recalls[0]),
        "free_recall": float(recalls[1]),
        "occupied_recall": float(recalls[2]),
        "rough_occupied_recall": float(rough[2, 2] / rough[2].sum()),
    }


def paired_control_comparison_v1(
    full_m: Any,
    control_m: Any,
    target_prefix: Any,
    informative: Any,
    scene_ids: Sequence[str],
    family_ids: Sequence[str],
    *,
    np: Any,
) -> Mapping[str, Any]:
    full = _utility_values_v1(full_m, target_prefix, informative, np)
    control = _utility_values_v1(control_m, target_prefix, informative, np)
    informative = np.asarray(informative, dtype=np.bool_)
    scenes = np.asarray(scene_ids, dtype=object)
    families = np.asarray(family_ids, dtype=object)
    scene_names = sorted(set(scene_ids))
    if len(scene_names) != 8:
        raise ValueError("paired control comparison requires exactly eight scenes")
    scene_deltas = np.asarray([
        float((full[informative & (scenes == scene)] - control[
            informative & (scenes == scene)
        ]).mean())
        for scene in scene_names
    ])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, len(scene_deltas), size=(BOOTSTRAP_REPLICATES, len(scene_deltas)))
    replicates = np.sort(scene_deltas[draws].mean(axis=1))
    family_deltas = {
        family: float((full[informative & (families == family)] - control[
            informative & (families == family)
        ]).mean())
        for family in REGISTERED_FAMILIES
    }
    return {
        "scene_count": 8,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "equal_scene_mean_delta": float(scene_deltas.mean()),
        "bootstrap_lower_95": float(replicates[BOOTSTRAP_LOWER_INDEX]),
        "per_scene_delta": dict(zip(scene_names, map(float, scene_deltas), strict=True)),
        "family_deltas": family_deltas,
        "positive_family_count": sum(value > 0.0 for value in family_deltas.values()),
    }


def evaluate_gate_v1(
    selection_metrics: Mapping[str, Mapping[str, Any]],
    semantic: Mapping[str, Any],
    comparisons: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    full = selection_metrics["full"]
    overall = full["overall"]
    checks = {
        "semantic_balanced_accuracy": semantic["balanced_accuracy"]
        >= GATE_THRESHOLDS["semantic_balanced_accuracy_min"],
        "semantic_free_recall": semantic["free_recall"]
        >= GATE_THRESHOLDS["semantic_free_recall_min"],
        "semantic_occupied_recall": semantic["occupied_recall"]
        >= GATE_THRESHOLDS["semantic_occupied_recall_min"],
        "semantic_unknown_recall": semantic["unknown_recall"]
        >= GATE_THRESHOLDS["semantic_unknown_recall_min"],
        "semantic_rough_occupied_recall": semantic["rough_occupied_recall"]
        >= GATE_THRESHOLDS["semantic_rough_occupied_recall_min"],
        "selection_registered_families": set(full["families"]) == set(REGISTERED_FAMILIES),
        "selection_informative_utility": overall["normalized_chosen_prefix_utility"]
        >= GATE_THRESHOLDS["informative_utility_min"],
        "selection_zero_prefix_rate": overall["selected_zero_prefix_rate"]
        <= GATE_THRESHOLDS["selected_zero_prefix_rate_max"],
        "selection_pair_concordance": overall["unequal_pair_concordance"]
        >= GATE_THRESHOLDS["pair_concordance_min"],
        "all_family_utility": all(
            row["normalized_chosen_prefix_utility"]
            >= GATE_THRESHOLDS["family_informative_utility_min"]
            for row in full["families"].values()
        ),
        "all_family_zero_prefix_rate": all(
            row["selected_zero_prefix_rate"]
            <= GATE_THRESHOLDS["family_selected_zero_prefix_rate_max"]
            for row in full["families"].values()
        ),
        "all_family_pair_concordance": all(
            row["unequal_pair_concordance"]
            >= GATE_THRESHOLDS["family_pair_concordance_min"]
            for row in full["families"].values()
        ),
    }
    if set(comparisons) != set(CONTROL_NAMES):
        raise ValueError("control comparison set changed")
    for name, comparison in comparisons.items():
        checks[f"{name}:positive_equal_scene_delta"] = (
            comparison["equal_scene_mean_delta"] > 0.0
        )
        checks[f"{name}:positive_bootstrap_lower_95"] = (
            comparison["bootstrap_lower_95"] > 0.0
        )
        checks[f"{name}:positive_family_count"] = (
            comparison["positive_family_count"]
            >= GATE_THRESHOLDS["positive_control_family_count_min"]
        )
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "PASS_FULL_ARM" if not failed else "FAIL_FULL_ARM",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "thresholds": dict(GATE_THRESHOLDS),
    }


def _atomic_write_v1(path: Path, raw: bytes) -> Mapping[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"write-once artifact exists: {path}")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "path": path.name,
        "byte_count": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _write_json_v1(path: Path, core: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    value = _with_content_sha256(core)
    raw = _canonical_json_bytes(value) + b"\n"
    binding = dict(_atomic_write_v1(path, raw))
    binding["content_sha256"] = value["content_sha256"]
    return value, binding


def _prepare_runtime_v1(repository_root: Path, manifest: Mapping[str, Any], labels_api: Any) -> Mapping[str, Any]:
    if repository_root.resolve() != ROOT.resolve():
        raise PermissionError("this runtime accepts only the canonical repository root")
    direct_contract = importlib.import_module(
        "lewm.benchmarks.go2_direct_egocentric_bev_state_jepa_v1"
    )
    inputs_binding = _exact_runtime_bindings_v1(
        manifest, direct_contract=direct_contract, labels_api=labels_api
    )
    direct = importlib.import_module("scripts.run_go2_direct_egocentric_bev_state_jepa_v1")
    matched = importlib.import_module("scripts.run_go2_shared_jepa_v5_matched_training_v1")
    runtime = matched._load_runtime()
    torch = runtime.torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one visible GPU is required")
    properties = torch.cuda.get_device_properties(0)
    hardware = {
        "visible_device_count": int(torch.cuda.device_count()),
        "name": torch.cuda.get_device_name(0),
        "total_memory_bytes": int(properties.total_memory),
        "hip_visible_devices": os.environ.get("HIP_VISIBLE_DEVICES"),
    }
    if (
        hardware["name"] != REQUIRED_GPU_NAME
        or hardware["total_memory_bytes"] != REQUIRED_GPU_MEMORY_BYTES
    ):
        raise RuntimeError("the exact reviewed R9700 runtime is required")
    progress: dict[str, Any] = {}
    authorization = {
        "raw": {
            "manifest": dict(inputs_binding["raw_manifest"]),
            "audit": dict(inputs_binding["raw_audit"]),
        },
        "camera": {
            "gate": dict(direct_contract.RUNTIME_BINDINGS[
                direct_contract.N320_GATE_RELATIVE_PATH
            ]),
            "checkpoint": dict(direct_contract.RUNTIME_BINDINGS[
                direct_contract.N320_CHECKPOINT_RELATIVE_PATH
            ]),
        },
    }
    raw_indexes = labels_api.v4.load_and_validate_raw_indexes(
        repository_root / inputs_binding["raw_manifest"]["path"],
        repository_root / inputs_binding["raw_pairs"]["path"],
        repository_root / inputs_binding["raw_endpoints"]["path"],
    )
    labels_api.v4.validate_raw_audit_v1(
        repository_root / inputs_binding["raw_audit"]["path"]
    )
    schedule = labels_api.v4.load_schedule_indices_v1(
        repository_root / inputs_binding["schedule"]["path"], raw_indexes=raw_indexes
    )
    inputs = direct._construct_raw_inputs_with_progress(
        matched, runtime, authorization, progress
    )
    direct._normalize_endpoint_paths(inputs)
    loader = direct.DirectBevNarrowLoader(runtime, inputs, progress=progress)
    fit, gate, checkpoint = direct._load_n320_with_progress(
        matched, runtime, authorization, progress
    )
    return {
        "direct": direct,
        "runtime": runtime,
        "torch": torch,
        "np": runtime.np,
        "device": torch.device("cuda:0"),
        "hardware": hardware,
        "inputs": inputs,
        "loader": loader,
        "fit": fit,
        "n320_gate": gate,
        "n320_checkpoint": checkpoint,
        "schedule": schedule,
        "progress": progress,
    }


def _mask_receipt_v1(masks: Any) -> Mapping[str, Any]:
    tensor = masks.detach().cpu().contiguous()
    payload = tensor.numpy().tobytes(order="C")
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "cell_counts": tensor.sum(dim=(-2, -1)).tolist(),
        "all_nonempty": bool(tensor.flatten(start_dim=2).any(dim=2).all()),
    }


def _access_receipt_v1(context: Mapping[str, Any]) -> Mapping[str, Any]:
    narrow = context["loader"].receipt()
    progress = context["progress"]
    consumed = context["inputs"].consumed
    identities = [
        {
            key: record[key]
            for key in ("path", "file_sha256", "byte_count", "kind", "roles", "arms")
        }
        for _, record in sorted(consumed.items())
    ]
    semantic_forbidden = sum(
        int(value) for value in narrow["forbidden_semantic_counters"].values()
    )
    fixed_negative = int(narrow["rgb_request_count"]["fixed_negative"])
    forbidden = semantic_forbidden + fixed_negative
    if forbidden != 0:
        raise PermissionError("narrow loader recorded a forbidden input request")
    receipt = {
        "n320_gate_open_attempted": bool(progress.get("n320_gate_open_attempted")),
        "n320_gate_open_succeeded": bool(progress.get("n320_gate_open_succeeded")),
        "n320_checkpoint_open_attempted": bool(
            progress.get("n320_checkpoint_open_attempted")
        ),
        "n320_checkpoint_open_succeeded": bool(
            progress.get("n320_checkpoint_open_succeeded")
        ),
        "raw_constructor_reads": progress.get("_raw_constructor_reads", {}),
        "raw_consumed_record_count": len(consumed),
        "raw_consumed_identity_sha256": _canonical_json_sha256(identities),
        "raw_consumed_roles": sorted({
            role for record in consumed.values() for role in record.get("roles", [])
        }),
        "narrow_loader": narrow,
        "forbidden_input_count": forbidden,
        "g2_navigation_final_evaluation_open_count": 0,
    }
    if not all(
        receipt[name]
        for name in (
            "n320_gate_open_attempted",
            "n320_gate_open_succeeded",
            "n320_checkpoint_open_attempted",
            "n320_checkpoint_open_succeeded",
        )
    ):
        raise PermissionError("exact N320 gate/checkpoint access did not complete")
    return receipt


def _install_repository_import_roots_v1(repository_root: Path) -> None:
    """Make the exact checkout importable under the reviewed ``python -I`` run."""

    if repository_root.resolve() != ROOT.resolve():
        raise PermissionError("this runtime accepts only the canonical repository root")
    for path in reversed((repository_root, repository_root / "lewm_worlds")):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def execute_v1(*, repository_root: Path = ROOT) -> Mapping[str, Any]:
    repository_root = Path(repository_root).absolute()
    _install_repository_import_roots_v1(repository_root)
    output = repository_root / OUTPUT_RELATIVE_PATH
    if output.exists() or output.is_symlink():
        raise FileExistsError("fresh swept-progress attempt_v1 already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    try:
        labels_api = importlib.import_module(
            "lewm.benchmarks.go2_swept_progress_survival_labels_v1"
        )
        manifest, rows_by_role = load_label_bundle_v1(
            repository_root, labels_api=labels_api
        )
        context = _prepare_runtime_v1(repository_root, manifest, labels_api)
        torch, np = context["torch"], context["np"]
        preflight = labels_api.summarize_preflight_v1(
            rows_by_role, context["schedule"]
        )
        if preflight != manifest.get("preflight"):
            raise PermissionError("label preflight no longer matches its manifest")
        training_core = importlib.import_module(
            "scripts.run_go2_rgb_swept_progress_survival_joint_jepa_v1"
        )
        frozen = {
            role: training_core.freeze_role_labels_v1(rows, role=role, np=np)
            for role, rows in rows_by_role.items()
        }
        informative = {
            role: np.asarray(
                [group[0]["informative_state"] for group in labels.state_groups],
                dtype=np.bool_,
            )
            for role, labels in frozen.items()
        }
        pairs = {role: context["inputs"].role_pairs(role) for role in ROLE_FILES}
        for role in ROLE_FILES:
            training_core.validate_pairs_against_labels_v1(pairs[role], frozen[role])

        model_api = importlib.import_module(
            "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v1"
        )
        survival_scoring = importlib.import_module(
            "lewm.benchmarks.go2_swept_progress_survival_joint_jepa_v1"
        )
        metrics_api = importlib.import_module(
            "lewm.benchmarks.go2_post_action_projective_support_metrics_v1"
        )
        torch.manual_seed(EXPERIMENT_SEED)
        torch.cuda.manual_seed_all(EXPERIMENT_SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        n320_state = {
            name: value.detach().cpu().float().contiguous().clone()
            for name, value in context["fit"].encoder.state_dict().items()
        }
        masks = survival_scoring.build_swept_progress_masks_v1()
        current_frame_persistence_masks = (
            survival_scoring.build_current_frame_swept_progress_masks_v1()
        )
        model = model_api.GeometryAnchoredSweptProgressSurvivalJointJepaV1(
            n320_state, masks
        ).to(context["device"])
        model.train()
        partition = training_core.partition_parameters_v1(model)
        optimizer = training_core.build_frozen_optimizer_v1(partition)
        if not any(
            name.startswith("predictor.swept_progress_head.")
            for name in partition.names["predictor"]
        ):
            raise RuntimeError("survival head escaped the predictor optimizer group")

        accounting_state, trace, training_diagnostics = (
            training_core.run_fixed_training_v1(
                model,
                optimizer,
                context["loader"],
                pairs["train"],
                frozen["train"],
                context["schedule"],
                context["device"],
            )
        )
        accounting = dict(accounting_state.__dict__)
        model.eval()
        model.requires_grad_(False)
        state = {
            name: value.detach().cpu().contiguous()
            for name, value in model.state_dict().items()
        }
        checkpoint_buffer = io.BytesIO()
        torch.save(
            {
                "schema": "lewm_go2_rgb_swept_progress_survival_joint_jepa_v1_checkpoint_v1",
                "development_only": True,
                "resume_authorized": False,
                "qualified": False,
                "constructor_initialization_seed": CONSTRUCTOR_INITIALIZATION_SEED,
                "experiment_seed": EXPERIMENT_SEED,
                "accounting": accounting,
                "model_state_dict": state,
            },
            checkpoint_buffer,
        )
        checkpoint_binding = _atomic_write_v1(
            output / "checkpoint_update_1000.pt", checkpoint_buffer.getvalue()
        )
        _, trace_binding = _write_json_v1(
            output / "training_trace.json",
            {
                "schema": "lewm_go2_rgb_swept_progress_survival_joint_jepa_v1_trace_v1",
                "status": "COMPLETE",
                "accounting": accounting,
                "rows": list(trace),
            },
        )
        action_prior_m = (
            frozen["train"].prefix_lengths.mean(axis=0, dtype=np.float64)
            * PROGRESS_SEGMENT_M
        )
        scored = {
            role: score_role_v1(
                model,
                context["loader"],
                pairs[role],
                frozen[role],
                action_prior_m,
                context["device"],
                torch=torch,
                np=np,
                training_core=training_core,
                current_frame_persistence_masks=current_frame_persistence_masks,
                metrics_api=metrics_api,
            )
            for role in ("probability_calibration", "checkpoint_selection")
        }
        role_metrics = {
            role: {
                arm: scientific_metrics_v1(
                    scored[role]["scores_m"][arm],
                    frozen[role].prefix_lengths,
                    informative[role],
                    frozen[role].scene_ids,
                    frozen[role].family_ids,
                    np=np,
                )
                for arm in ALL_ARM_NAMES
            }
            for role in scored
        }
        selection_semantic = semantic_metrics_v1(
            scored["checkpoint_selection"]["semantic_confusion"],
            scored["checkpoint_selection"]["rough_semantic_confusion"],
            np=np,
        )
        selection_scores = scored["checkpoint_selection"]["scores_m"]
        selection_labels = frozen["checkpoint_selection"]
        comparisons = {
            name: paired_control_comparison_v1(
                selection_scores["full"],
                selection_scores[name],
                selection_labels.prefix_lengths,
                informative["checkpoint_selection"],
                selection_labels.scene_ids,
                selection_labels.family_ids,
                np=np,
            )
            for name in CONTROL_NAMES
        }
        gate = evaluate_gate_v1(
            role_metrics["checkpoint_selection"], selection_semantic, comparisons
        )
        access_receipt = _access_receipt_v1(context)
        mask_receipts = {
            "predicted_next_post_action_frame": _mask_receipt_v1(masks),
            "coordinate_matched_current_frame_persistence": _mask_receipt_v1(
                current_frame_persistence_masks
            ),
        }
        result, _ = _write_json_v1(
            output / "result.json",
            {
                "schema": "lewm_go2_rgb_swept_progress_survival_joint_jepa_v1_result_v1",
                "status": gate["status"],
                "gate": gate,
                "caps": {"updates": MAXIMUM_UPDATES, "presentations": MAXIMUM_PRESENTATIONS},
                "seeds": {
                    "inherited_fresh_component_constructor": CONSTRUCTOR_INITIALIZATION_SEED,
                    "experiment_and_stochastic_execution": EXPERIMENT_SEED,
                    "bootstrap": BOOTSTRAP_SEED,
                },
                "label_manifest": {
                    "path": f"{LABEL_ROOT_RELATIVE_PATH}/{LABEL_MANIFEST_NAME}",
                    "file_sha256": LABEL_MANIFEST_FILE_SHA256,
                    "content_sha256": manifest["content_sha256"],
                    "byte_count": LABEL_MANIFEST_BYTE_COUNT,
                    "role_files": manifest["files"],
                },
                "n320": {
                    "gate_content_sha256": context["n320_gate"]["content_sha256"],
                    "checkpoint": context["n320_checkpoint"],
                    "encoder_only_initialization": True,
                },
                "hardware": context["hardware"],
                "schedule_prefix_sha256": labels_api.v4.SCHEDULE_PREFIX_SHA256,
                "masks": mask_receipts,
                "training": {
                    "accounting": accounting,
                    "diagnostics": training_diagnostics,
                    "joint_from_update_one": True,
                    "separate_head_or_predictor_training": False,
                    "checkpoint": checkpoint_binding,
                    "trace": trace_binding,
                },
                "action_prior_mean_progress_m": action_prior_m.tolist(),
                "roles": role_metrics,
                "selection_semantic": selection_semantic,
                "selection_control_comparisons": comparisons,
                "wrong_rgb_mapping_sha256": {
                    role: scored[role]["wrong_rgb_mapping_sha256"] for role in scored
                },
                "determinism": {
                    "algorithms_enabled": bool(
                        torch.are_deterministic_algorithms_enabled()
                    ),
                    "warn_only": True,
                    "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                    "cudnn_deterministic": bool(
                        torch.backends.cudnn.deterministic
                    ),
                    "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                    "matmul_allow_tf32": bool(
                        torch.backends.cuda.matmul.allow_tf32
                    ),
                },
                "access": access_receipt,
                "matched_no_jepa": {
                    "status": "STAGED_ONLY_IF_FULL_ARM_PASSES",
                    "run_in_this_attempt": False,
                    "jepa_treatment_effect_claimed": False,
                },
                "authority": {
                    "development_only": True,
                    "g2_navigation_final_evaluation_opened": False,
                    "checkpoint_qualified": False,
                    "promotion_performed": False,
                    "retry_or_resume_authorized": False,
                },
            },
        )
        return result
    except Exception as error:
        if not (output / "result.json").exists() and not (output / "failure.json").exists():
            try:
                _write_json_v1(
                    output / "failure.json",
                    {
                        "schema": "lewm_go2_rgb_swept_progress_survival_joint_jepa_v1_failure_v1",
                        "status": "FAILED_NO_RETRY_OR_RESUME",
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                        "traceback": traceback.format_exc(),
                        "authority": {
                            "development_only": True,
                            "g2_navigation_final_evaluation_opened": False,
                            "retry_or_resume_authorized": False,
                        },
                    },
                )
            except Exception:
                pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    result = execute_v1(repository_root=args.repository_root)
    print(_canonical_json_bytes({
        "status": result["status"],
        "result": f"{OUTPUT_RELATIVE_PATH}/result.json",
    }).decode("utf-8"))
    return 0 if result["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
