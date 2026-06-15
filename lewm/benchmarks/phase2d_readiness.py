"""Immutable split and run-readiness gates for Phase 2D."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .experiment_manifest import artifact_record, sha256_file, verify_manifest_files
from .phase2_data import (
    CONFIRMATORY_SPLIT_REQUIREMENTS,
    confirmatory_data_gate,
    load_spatial_future_rows,
    materialized_frame_paths,
    phase2_dataset_audit,
    source_key,
)

REQUIRED_SPLITS = tuple(CONFIRMATORY_SPLIT_REQUIREMENTS)
REQUIRED_CELLS = ("C0", "C1", "C2")
DIAGNOSTIC_CELLS = ("state_only", "action_only")
TRAINING_START_RUN_CLASSES = ("pilot", "confirmatory")
CANONICAL_SPLIT_ALIASES = {
    "train": {"train"},
    "validation": {"validation", "val", "eval"},
    "test_id": {"test_id", "test-id", "testid"},
    "test_hard": {"test_hard", "test-hard", "testhard"},
}
LINEAGE_FIELD_ALIASES = {
    "topology_seed": (
        "topology_seed",
        "topology_id",
        "topology_hash",
        "maze_seed",
        "layout_seed",
    ),
    "visual_seed": (
        "visual_seed",
        "visual_id",
        "visual_hash",
        "texture_seed",
        "material_seed",
    ),
}


def canonical_split_name(name: str) -> str:
    """Return the registered split name for a user/file label."""

    normalized = name.strip().lower().replace("-", "_")
    for canonical, aliases in CANONICAL_SPLIT_ALIASES.items():
        if normalized in {alias.replace("-", "_") for alias in aliases}:
            return canonical
    raise ValueError(f"unknown Phase 2D split name: {name}")


def _lineage_value(row: Mapping, canonical_field: str):
    for alias in LINEAGE_FIELD_ALIASES[canonical_field]:
        if alias in row and row[alias] not in (None, ""):
            return row[alias]
    metadata = row.get("scene_metadata")
    if isinstance(metadata, Mapping):
        for alias in LINEAGE_FIELD_ALIASES[canonical_field]:
            if alias in metadata and metadata[alias] not in (None, ""):
                return metadata[alias]
    return None


def _split_label_matches(row: Mapping, split_name: str) -> bool:
    label = str(row.get("split", "")).strip().lower().replace("-", "_")
    return label in {alias.replace("-", "_") for alias in CANONICAL_SPLIT_ALIASES[split_name]}


def source_state_lineage_audit(rows: Sequence[dict], *, split_name: str) -> dict:
    """Audit source-state lineage fields required before confirmatory access."""

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[source_key(row)].append(row)

    missing_field_counts = Counter()
    inconsistent_field_counts = Counter()
    split_label_mismatch_rows = 0
    records = []
    for key, source_rows in sorted(grouped.items()):
        scene_id, source_index = key
        if any(not _split_label_matches(row, split_name) for row in source_rows):
            split_label_mismatch_rows += sum(
                not _split_label_matches(row, split_name) for row in source_rows
            )
        lineage = {}
        for field in LINEAGE_FIELD_ALIASES:
            values = {
                _lineage_value(row, field)
                for row in source_rows
                if _lineage_value(row, field) is not None
            }
            if not values:
                missing_field_counts[field] += 1
                lineage[field] = None
            elif len(values) > 1:
                inconsistent_field_counts[field] += 1
                lineage[field] = sorted(str(value) for value in values)
            else:
                lineage[field] = next(iter(values))
        family_values = {str(row.get("family", "")) for row in source_rows}
        if len(family_values) > 1:
            inconsistent_field_counts["family"] += 1
        records.append(
            {
                "scene_id": scene_id,
                "source_index": source_index,
                "row_count": len(source_rows),
                "family": sorted(family_values),
                **lineage,
            }
        )
    verified = (
        not missing_field_counts
        and not inconsistent_field_counts
        and split_label_mismatch_rows == 0
        and bool(grouped)
    )
    return {
        "schema": "jepa_phase2d_source_state_lineage_audit_v0",
        "split": split_name,
        "source_states": len(grouped),
        "missing_lineage_field_counts": dict(sorted(missing_field_counts.items())),
        "inconsistent_lineage_field_counts": dict(
            sorted(inconsistent_field_counts.items())
        ),
        "split_label_mismatch_rows": split_label_mismatch_rows,
        "lineage_verified": verified,
        "source_state_records": records,
    }


def _image_artifacts(rows: Sequence[dict]) -> tuple[dict[str, dict], dict]:
    paths = []
    for row in rows:
        frame_paths, _mask = materialized_frame_paths(row)
        paths.extend(frame_paths)
        goal_frame = row.get("goal_frame")
        if goal_frame is not None:
            paths.append(Path(str(goal_frame)))
    unique_paths = sorted({Path(path) for path in paths}, key=lambda path: str(path))
    artifacts = {}
    missing = []
    for index, path in enumerate(unique_paths):
        try:
            artifacts[f"image_{index:08d}"] = artifact_record(path)
        except FileNotFoundError:
            missing.append(str(path))
    return artifacts, {
        "schema": "jepa_phase2d_image_lineage_audit_v0",
        "unique_referenced_images": len(unique_paths),
        "hashed_images": len(artifacts),
        "missing_images": missing,
        "all_images_hashed": not missing,
    }


def build_phase2d_split_manifest(
    split_paths: Mapping[str, Path],
    *,
    hash_images: bool = False,
) -> dict:
    """Build the immutable split, data-gate, and lineage manifest."""

    canonical_paths = {
        canonical_split_name(name): Path(path) for name, path in split_paths.items()
    }
    if len(canonical_paths) != len(split_paths):
        raise ValueError("split names must be unique after canonicalization")

    split_rows = {}
    splits = {}
    lineage_verified = True
    image_lineage_verified = True
    for split_name, path in sorted(canonical_paths.items()):
        rows, load_audit = load_spatial_future_rows(path, mode="all")
        split_rows[split_name] = rows
        lineage = source_state_lineage_audit(rows, split_name=split_name)
        image_artifacts = {}
        image_audit = None
        if hash_images:
            image_artifacts, image_audit = _image_artifacts(rows)
            image_lineage_verified = (
                image_lineage_verified and image_audit["all_images_hashed"]
            )
        lineage_verified = lineage_verified and lineage["lineage_verified"]
        splits[split_name] = {
            "dataset_file": artifact_record(path),
            "load_audit": load_audit,
            "dataset_audit": phase2_dataset_audit(rows),
            "lineage": lineage,
            "image_artifacts": image_artifacts,
            "image_lineage": image_audit,
        }
    gate = confirmatory_data_gate(
        split_rows,
        lineage_verified=lineage_verified and image_lineage_verified,
    )
    return {
        "schema": "jepa_phase2d_split_manifest_v0",
        "required_splits": list(REQUIRED_SPLITS),
        "hash_images": hash_images,
        "splits": splits,
        "lineage": {
            "source_state_lineage_verified": lineage_verified,
            "image_lineage_verified": image_lineage_verified,
            "lineage_verified": lineage_verified and image_lineage_verified,
        },
        "confirmatory_gate": gate,
    }


def verify_phase2d_split_manifest(manifest: Mapping) -> dict:
    """Verify split and optional image hashes recorded in a Phase 2D split manifest."""

    file_results = {}
    for split_name, split in manifest.get("splits", {}).items():
        dataset = split["dataset_file"]
        path = Path(str(dataset["path"]))
        actual = sha256_file(path) if path.is_file() else None
        file_results[f"splits.{split_name}.dataset_file"] = {
            "exists": path.is_file(),
            "expected_sha256": dataset["sha256"],
            "actual_sha256": actual,
            "matches": path.is_file() and actual == dataset["sha256"],
        }
        for image_name, image in split.get("image_artifacts", {}).items():
            image_path = Path(str(image["path"]))
            image_actual = sha256_file(image_path) if image_path.is_file() else None
            file_results[f"splits.{split_name}.image_artifacts.{image_name}"] = {
                "exists": image_path.is_file(),
                "expected_sha256": image["sha256"],
                "actual_sha256": image_actual,
                "matches": image_path.is_file() and image_actual == image["sha256"],
            }
    return {
        "schema": "jepa_phase2d_split_manifest_verification_v0",
        "files": file_results,
        "confirmatory_gate_passed": bool(
            manifest.get("confirmatory_gate", {}).get("passed", False)
        ),
        "lineage_verified": bool(
            manifest.get("lineage", {}).get("lineage_verified", False)
        ),
        "passes": bool(file_results)
        and all(result["matches"] for result in file_results.values())
        and bool(manifest.get("confirmatory_gate", {}).get("passed", False))
        and bool(manifest.get("lineage", {}).get("lineage_verified", False)),
    }


def _all_manifest_files_match(verification: Mapping) -> bool:
    files = verification.get("files", {})
    return bool(files) and all(result.get("matches", False) for result in files.values())


def _manifest_split_dataset_path(manifest: Mapping, split_name: str) -> Path | None:
    split = manifest.get("splits", {}).get(split_name)
    if not isinstance(split, Mapping):
        return None
    dataset = split.get("dataset_file")
    if not isinstance(dataset, Mapping) or "path" not in dataset:
        return None
    return Path(str(dataset["path"])).resolve()


def _path_matches(left: Path | None, right: Path | None) -> bool:
    return left is not None and right is not None and left.resolve() == right.resolve()


def phase2d_training_start_readiness(
    *,
    split_manifest_path: Path,
    cell: str,
    requested_run_class: str,
    train_data_path: Path | None,
    validation_data_path: Path | None,
) -> dict:
    """Return a guard decision for starting a Phase 2D training run.

    This preflight is intentionally stricter for confirmatory training than
    ordinary smoke/pilot work. It does not grant validation/test-result access;
    that remains guarded by :func:`phase2d_run_readiness` after selected
    checkpoints are frozen.
    """

    run_class = requested_run_class.strip().lower().replace("-", "_")
    normalized_cell = cell.strip()
    manifest = _load_manifest(split_manifest_path)
    split_verification = verify_phase2d_split_manifest(manifest)
    train_manifest_path = _manifest_split_dataset_path(manifest, "train")
    validation_manifest_path = _manifest_split_dataset_path(manifest, "validation")
    registered_training_cells = set(REQUIRED_CELLS) | set(DIAGNOSTIC_CELLS)
    confirmatory_requested = run_class == "confirmatory"
    checks = {
        "run_class_supported": run_class in TRAINING_START_RUN_CLASSES,
        "cell_registered": normalized_cell in registered_training_cells,
        "confirmatory_cell_participates_in_primary_comparison": (
            not confirmatory_requested or normalized_cell in REQUIRED_CELLS
        ),
        "split_manifest_files_verified": _all_manifest_files_match(split_verification),
        "split_manifest_lineage_verified": bool(
            split_verification.get("lineage_verified", False)
        ),
        "confirmatory_data_gate_required_and_passed": (
            not confirmatory_requested
            or bool(split_verification.get("confirmatory_gate_passed", False))
        ),
        "train_split_present": train_manifest_path is not None,
        "validation_split_present": validation_manifest_path is not None,
        "train_data_matches_manifest": _path_matches(
            train_data_path,
            train_manifest_path,
        ),
        "validation_data_matches_manifest": _path_matches(
            validation_data_path,
            validation_manifest_path,
        ),
    }
    return {
        "schema": "jepa_phase2d_training_start_readiness_v0",
        "requested_run_class": run_class,
        "cell": normalized_cell,
        "split_manifest": {
            "path": str(split_manifest_path.resolve()),
            "verification": split_verification,
        },
        "manifest_dataset_paths": {
            "train": None if train_manifest_path is None else str(train_manifest_path),
            "validation": (
                None if validation_manifest_path is None else str(validation_manifest_path)
            ),
        },
        "requested_dataset_paths": {
            "train": None if train_data_path is None else str(train_data_path.resolve()),
            "validation": (
                None
                if validation_data_path is None
                else str(validation_data_path.resolve())
            ),
        },
        "access_scope": (
            "training_start_only_no_validation_or_test_result_access_granted"
        ),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text())


def _verify_cell_manifest(path: Path, *, expected_cell: str) -> dict:
    manifest = _load_manifest(path)
    verification = verify_manifest_files(manifest)
    config = manifest.get("config", {})
    artifacts = manifest.get("artifacts", {})
    return {
        "path": str(path.resolve()),
        "experiment_id": manifest.get("experiment_id"),
        "files_verified": verification["passes"],
        "cell_matches": config.get("cell") == expected_cell,
        "run_class_is_confirmatory": config.get("run_class") == "confirmatory",
        "checkpoint_rule_registered": bool(config.get("checkpoint_rule")),
        "selected_checkpoint_present": "selected_checkpoint" in artifacts,
        "verification": verification,
        "passes": all(
            (
                verification["passes"],
                config.get("cell") == expected_cell,
                config.get("run_class") == "confirmatory",
                bool(config.get("checkpoint_rule")),
                "selected_checkpoint" in artifacts,
            )
        ),
    }


def phase2d_run_readiness(
    *,
    split_manifest_path: Path,
    cell_manifest_paths: Mapping[str, Path],
    requested_stage: str,
    test_id_report_manifest_path: Path | None = None,
) -> dict:
    """Return a machine-readable guard decision for validation/test access."""

    stage = requested_stage.strip().lower().replace("-", "_")
    if stage not in {"validation", "test_id", "test_hard"}:
        raise ValueError(f"unsupported requested stage: {requested_stage}")
    split_manifest = _load_manifest(split_manifest_path)
    split_verification = verify_phase2d_split_manifest(split_manifest)
    canonical_cells = {
        cell.strip(): Path(path) for cell, path in cell_manifest_paths.items()
    }
    missing_cells = sorted(set(REQUIRED_CELLS) - set(canonical_cells))
    extra_cells = sorted(set(canonical_cells) - set(REQUIRED_CELLS))
    cell_results = {
        cell: _verify_cell_manifest(canonical_cells[cell], expected_cell=cell)
        for cell in sorted(set(REQUIRED_CELLS) & set(canonical_cells))
    }
    test_id_report = None
    if test_id_report_manifest_path is not None:
        report_manifest = _load_manifest(test_id_report_manifest_path)
        test_id_report = {
            "path": str(test_id_report_manifest_path.resolve()),
            "verification": verify_manifest_files(report_manifest),
        }
        test_id_report["passes"] = test_id_report["verification"]["passes"]
    test_stage_rule_passed = True
    test_stage_reason = "validation_or_test_id_access"
    if stage == "test_hard":
        test_stage_rule_passed = bool(test_id_report and test_id_report["passes"])
        test_stage_reason = "test_hard_requires_verified_test_id_report"

    checks = {
        "split_manifest_verified_and_gate_passed": split_verification["passes"],
        "all_required_cell_manifests_present": not missing_cells,
        "no_unregistered_cell_manifests": not extra_cells,
        "all_cell_manifests_ready": bool(cell_results)
        and all(result["passes"] for result in cell_results.values())
        and not missing_cells
        and not extra_cells,
        "test_stage_access_rule_passed": test_stage_rule_passed,
    }
    return {
        "schema": "jepa_phase2d_run_readiness_v0",
        "requested_stage": stage,
        "split_manifest": {
            "path": str(split_manifest_path.resolve()),
            "verification": split_verification,
        },
        "cell_manifests": cell_results,
        "missing_cells": missing_cells,
        "extra_cells": extra_cells,
        "test_id_report_manifest": test_id_report,
        "test_stage_access_rule": test_stage_reason,
        "checks": checks,
        "passed": all(checks.values()),
    }
