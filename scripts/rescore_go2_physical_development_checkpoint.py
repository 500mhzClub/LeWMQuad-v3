#!/usr/bin/env python3
"""Rescore a frozen physical-head development checkpoint without touching G2.

This migration diagnostic exists only to recompute the calibration-role
operating point after a threshold-selector correction. It never trains, never
opens G2 image/label bytes, never emits a checkpoint, and is ineligible for
one-shot or runtime promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lewm.benchmarks.experiment_manifest import sha256_file  # noqa: E402
from lewm.benchmarks.traversability_metrics import (  # noqa: E402
    TraversabilityThresholds,
)
from lewm.datasets.go2_paired_navigation import DATASET_ROLES  # noqa: E402
from lewm.hierarchical_probability_calibration import (  # noqa: E402
    CALIBRATION_METHOD as HIERARCHICAL_CALIBRATION_METHOD,
    validate_hierarchical_probability_calibration,
)
from lewm.models.egomotion_bev_jepa import EgomotionBevJepa  # noqa: E402
from scripts.train_go2_egomotion_bev_jepa import (  # noqa: E402
    PHYSICAL_CHECKPOINT_SCHEMA,
    PHYSICAL_OCCUPANCY_TARGET_SPACE,
    PHYSICAL_REPORT_SCHEMA,
    REPOSITORY_ROOT,
    _canonical_json_sha256,
    _git_snapshot,
    _json_normalize,
    _loader,
    _read_json,
    _read_rows,
    _resolve_device,
    _row_subset_record,
    _state_dict_sha256,
    _validate_projective_query_support_artifacts,
    evaluate_model,
    resolve_dataset_scene_roles,
)


OUTPUT_SCHEMA = "lewm_go2_physical_development_threshold_rescore_v1"
DEVELOPMENT_ROLES = ("checkpoint_selection", "probability_calibration")
TRAINER_SOURCE = REPOSITORY_ROOT / "scripts/train_go2_egomotion_bev_jepa.py"
METRICS_SOURCE = REPOSITORY_ROOT / "lewm/benchmarks/traversability_metrics.py"


def _require_sha256(value: str, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-checkpoint-sha256",
        required=True,
    )
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument(
        "--expected-parent-report-sha256",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260709)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch-size must be positive and workers must be nonnegative")
    try:
        args.expected_parent_checkpoint_sha256 = _require_sha256(
            args.expected_parent_checkpoint_sha256,
            name="expected parent checkpoint SHA-256",
        )
        args.expected_parent_report_sha256 = _require_sha256(
            args.expected_parent_report_sha256,
            name="expected parent report SHA-256",
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def _load_parent_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - environment contract
        raise RuntimeError("torch.load(..., weights_only=True) is required") from error
    if not isinstance(payload, Mapping):
        raise ValueError("parent checkpoint root must be an object")
    return payload


def _validate_content_hash(record: Mapping[str, Any], *, name: str) -> None:
    declared = _require_sha256(
        str(record.get("content_sha256", "")),
        name=f"{name} content SHA-256",
    )
    core = dict(record)
    core.pop("content_sha256", None)
    if _canonical_json_sha256(core) != declared:
        raise ValueError(f"{name} content SHA-256 mismatch")


def _require_zero_g2_contact(payload: Mapping[str, Any], *, name: str) -> None:
    if payload.get("g2_evaluation") is not None:
        raise ValueError(f"{name} contains a prior G2 evaluation")
    if payload.get("head_g2_evaluation") is not None:
        raise ValueError(f"{name} contains a prior physical-head G2 evaluation")
    if payload.get("final_g2_evaluation") is not None:
        raise ValueError(f"{name} contains a prior final G2 evaluation")
    if payload.get("final_head_g2_evaluation") is not None:
        raise ValueError(f"{name} contains a prior final physical-head G2 evaluation")
    if bool(payload.get("g2_passes", False)) or bool(
        payload.get("head_g2_passes", False)
    ):
        raise ValueError(f"{name} claims a prior G2 pass")

    role_provenance = payload.get("dataset_role_provenance_verification")
    if not isinstance(role_provenance, Mapping):
        raise ValueError(f"{name} lacks role-specific dataset provenance")
    if role_provenance.get("g2_evaluation") is not None:
        raise ValueError(f"{name} previously verified G2 artifact bytes")
    combined_provenance = payload.get("dataset_provenance_verification")
    if not isinstance(combined_provenance, Mapping):
        raise ValueError(f"{name} lacks dataset provenance")
    if combined_provenance.get("g2_evaluation") is not None:
        raise ValueError(f"{name} previously verified G2 artifact bytes")

    row_subsets = payload.get("row_subsets")
    if not isinstance(row_subsets, Mapping):
        raise ValueError(f"{name} lacks row-subset provenance")
    g2_subset = row_subsets.get("g2_evaluation")
    if not isinstance(g2_subset, Mapping) or dict(g2_subset) != _row_subset_record(
        (), role="g2_evaluation"
    ):
        raise ValueError(f"{name} contains G2 rows in its selected subset")

    ledger = payload.get("dataset_access_ledger")
    if not isinstance(ledger, Mapping):
        raise ValueError(f"{name} lacks a dataset access ledger")
    roles = ledger.get("roles")
    if not isinstance(roles, Mapping) or not isinstance(
        roles.get("g2_evaluation"), Mapping
    ):
        raise ValueError(f"{name} lacks a G2 access-ledger role")
    g2_role = roles["g2_evaluation"]
    for field in (
        "label_shard_files_hashed",
        "image_files_hashed",
        "model_output_rows",
        "selected_row_count",
    ):
        if int(g2_role.get(field, -1)) != 0:
            raise ValueError(f"{name} G2 access ledger is nonzero: {field}")
    if g2_role.get("provenance_verification") is not None:
        raise ValueError(f"{name} G2 provenance verification is nonzero")
    contact = ledger.get("g2_contact")
    if not isinstance(contact, Mapping):
        raise ValueError(f"{name} lacks an explicit G2 contact ledger")
    for field in ("label_shard_byte_opens", "image_byte_opens", "model_output_rows"):
        if int(contact.get(field, -1)) != 0:
            raise ValueError(f"{name} G2 contact ledger is nonzero: {field}")

    promotion = payload.get("promotion")
    if promotion is not None:
        if not isinstance(promotion, Mapping):
            raise ValueError(f"{name} promotion record is malformed")
        if bool(promotion.get("head_g2_evaluated", False)) or bool(
            promotion.get("g2_evaluated", False)
        ):
            raise ValueError(f"{name} promotion record says G2 was evaluated")
        if bool(promotion.get("head_g2_passes", False)) or bool(
            promotion.get("passes", False)
        ):
            raise ValueError(f"{name} promotion record claims a G2 pass")


def _validate_parent_contracts(
    checkpoint: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    dataset_manifest_path: Path,
    dataset_manifest_sha256: str,
) -> dict[str, Any]:
    if checkpoint.get("schema") != PHYSICAL_CHECKPOINT_SCHEMA:
        raise ValueError("parent checkpoint is not physical checkpoint schema v4")
    if report.get("schema") != PHYSICAL_REPORT_SCHEMA:
        raise ValueError("parent report is not physical report schema v4")
    output_contract = checkpoint.get("occupancy_output_contract")
    if not isinstance(output_contract, Mapping) or output_contract.get(
        "target_occupancy_space"
    ) != PHYSICAL_OCCUPANCY_TARGET_SPACE:
        raise ValueError("parent checkpoint does not target observable physical occupancy")
    if checkpoint.get("runtime_ready") is not False:
        raise ValueError("physical parent checkpoint must remain runtime-ineligible")

    report_checkpoint = report.get("checkpoint")
    if not isinstance(report_checkpoint, Mapping) or str(
        report_checkpoint.get("sha256", "")
    ) != checkpoint_sha256:
        raise ValueError("parent report/checkpoint SHA-256 agreement failed")
    report_dataset = report.get("dataset_manifest")
    if not isinstance(report_dataset, Mapping) or str(
        report_dataset.get("sha256", "")
    ) != dataset_manifest_sha256:
        raise ValueError("parent report/dataset SHA-256 agreement failed")
    if str(checkpoint.get("dataset_manifest_sha256", "")) != dataset_manifest_sha256:
        raise ValueError("parent checkpoint/dataset SHA-256 agreement failed")

    report_label_semantics = report.get("label_semantics")
    if not isinstance(report_label_semantics, Mapping) or report_label_semantics.get(
        "target_occupancy_space"
    ) != PHYSICAL_OCCUPANCY_TARGET_SPACE:
        raise ValueError("parent report does not bind physical target semantics")

    agreement_fields = (
        "dataset_provenance_verification",
        "dataset_role_provenance_verification",
        "dataset_access_ledger",
        "row_subsets",
        "probability_calibration",
        "probability_calibration_provenance",
        "training_run_provenance",
        "best_epoch",
    )
    for field in agreement_fields:
        if checkpoint.get(field) != report.get(field):
            raise ValueError(f"parent report/checkpoint disagree on {field}")
    _require_zero_g2_contact(checkpoint, name="parent checkpoint")
    _require_zero_g2_contact(report, name="parent report")

    training_run = checkpoint.get("training_run_provenance")
    if not isinstance(training_run, Mapping) or training_run.get("schema") != (
        "lewm_go2_training_run_provenance_v1"
    ):
        raise ValueError("parent checkpoint lacks training-run provenance")
    _validate_content_hash(training_run, name="parent training-run provenance")
    critical_inputs = training_run.get("critical_inputs")
    if not isinstance(critical_inputs, Mapping):
        raise ValueError("parent training-run provenance lacks critical inputs")
    old_sources = {}
    for field in ("trainer_source", "traversability_metrics_source"):
        record = critical_inputs.get(field)
        if not isinstance(record, Mapping):
            raise ValueError(f"parent training provenance lacks {field}")
        old_sources[field] = {
            "path": str(record.get("path", "")),
            "sha256": _require_sha256(
                str(record.get("sha256", "")),
                name=f"parent {field} SHA-256",
            ),
        }

    return {
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
        "dataset_manifest": {
            "path": str(dataset_manifest_path),
            "sha256": dataset_manifest_sha256,
        },
        "old_sources": old_sources,
    }


def _validate_physical_dataset_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema") != "lewm_go2_paired_navigation_dataset_v3":
        raise ValueError("physical rescore requires dataset schema v3")
    semantics = manifest.get("label_semantics")
    if not isinstance(semantics, Mapping) or semantics.get(
        "target_occupancy_space"
    ) != PHYSICAL_OCCUPANCY_TARGET_SPACE:
        raise ValueError("dataset does not target observable physical occupancy")
    if (
        semantics.get("per_frame_configuration_classes_supervised") is not False
        or semantics.get("post_memory_configuration_derivation_is_evaluation_only")
        is not True
    ):
        raise ValueError("dataset physical-label contract is malformed")


def _exact_saved_subset_rows(
    rows: Sequence[Mapping[str, Any]],
    scene_roles: Mapping[str, str],
    saved_subsets: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    rows_by_global: dict[int, dict[str, Any]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        global_row = int(row["global_row"])
        if global_row in rows_by_global:
            raise ValueError(f"dataset index repeats global_row {global_row}")
        rows_by_global[global_row] = row

    selected: dict[str, list[dict[str, Any]]] = {}
    for role in DEVELOPMENT_ROLES:
        saved = saved_subsets.get(role)
        if not isinstance(saved, Mapping):
            raise ValueError(f"parent checkpoint lacks saved {role} subset")
        identities = saved.get("identities")
        if not isinstance(identities, list) or not identities:
            raise ValueError(f"parent checkpoint saved {role} subset is empty")
        role_rows = []
        for identity in identities:
            if not isinstance(identity, Mapping):
                raise ValueError(f"parent checkpoint {role} identity is malformed")
            global_row = int(identity.get("global_row", -1))
            if global_row not in rows_by_global:
                raise ValueError(f"saved {role} global_row is absent: {global_row}")
            row = rows_by_global[global_row]
            if scene_roles.get(str(row["scene_id"])) != role:
                raise ValueError(f"saved {role} row belongs to another dataset role")
            role_rows.append(row)
        reconstructed = _row_subset_record(role_rows, role=role)
        if reconstructed != dict(saved):
            raise ValueError(f"saved {role} subset identity does not match dataset index")
        selected[role] = role_rows
    overlap = {
        int(row["global_row"])
        for row in selected["checkpoint_selection"]
    } & {
        int(row["global_row"])
        for row in selected["probability_calibration"]
    }
    if overlap:
        raise ValueError(f"saved development subsets overlap: {sorted(overlap)}")
    return selected


def _verify_selected_artifacts(
    rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    records: dict[str, Any] = {}

    def add_expected(
        files: dict[Path, str],
        path: Path,
        expected_sha256: str,
        *,
        role: str,
    ) -> None:
        prior = files.setdefault(path, expected_sha256)
        if prior != expected_sha256:
            raise ValueError(f"selected {role} artifact has conflicting SHA-256")

    for role in DEVELOPMENT_ROLES:
        label_files: dict[Path, str] = {}
        image_files: dict[Path, str] = {}
        for row in rows_by_role[role]:
            add_expected(
                label_files,
                Path(str(row["label_shard_path"])).resolve(),
                str(row["label_shard_sha256"]),
                role=role,
            )
            for path_field, hash_field in (
                ("current_image_path", "current_image_sha256"),
                ("next_image_path", "next_image_sha256"),
            ):
                add_expected(
                    image_files,
                    Path(str(row[path_field])).resolve(),
                    str(row[hash_field]),
                    role=role,
                )
        for path, expected in (*label_files.items(), *image_files.items()):
            if sha256_file(path) != expected:
                raise ValueError(f"selected {role} artifact SHA-256 mismatch: {path}")
        records[role] = {
            "selected_row_count": len(rows_by_role[role]),
            "label_shard_files_verified": len(label_files),
            "image_files_verified": len(image_files),
            "artifact_identity_sha256": _canonical_json_sha256(
                {
                    "label_shards": [
                        {"path": str(path), "sha256": digest}
                        for path, digest in sorted(
                            label_files.items(), key=lambda item: str(item[0])
                        )
                    ],
                    "images": [
                        {"path": str(path), "sha256": digest}
                        for path, digest in sorted(
                            image_files.items(), key=lambda item: str(item[0])
                        )
                    ],
                }
            ),
        }
    return records


def _calibration_identity(
    checkpoint: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("parent checkpoint model state is malformed")
    model_state_sha256 = _state_dict_sha256(state)
    calibration = checkpoint.get("probability_calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError("parent checkpoint calibration is malformed")
    if calibration.get("method") != HIERARCHICAL_CALIBRATION_METHOD:
        raise ValueError("physical rescore requires hierarchical calibration")
    validate_hierarchical_probability_calibration(calibration)
    calibration_id = str(calibration.get("id", ""))
    if checkpoint.get("probability_calibration_id") != calibration_id:
        raise ValueError("parent checkpoint calibration ID mismatch")
    provenance = checkpoint.get("probability_calibration_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("parent checkpoint calibration provenance is malformed")
    if provenance.get("selected_model_state_sha256") != model_state_sha256:
        raise ValueError("parent calibration is not bound to its frozen model state")
    source = calibration.get("provenance", {}).get("source")
    if source != provenance:
        raise ValueError("calibration artifact/source provenance disagreement")
    if report.get("probability_calibration") != calibration:
        raise ValueError("parent report calibration differs from checkpoint")
    return calibration, {
        "model_state_sha256": model_state_sha256,
        "calibration_id": calibration_id,
        "calibration_content_sha256": _require_sha256(
            str(calibration.get("content_sha256", "")),
            name="calibration content SHA-256",
        ),
        "calibration_canonical_sha256": _canonical_json_sha256(calibration),
        "model_state_unchanged": True,
        "calibration_unchanged": True,
    }


def _source_crop_fraction_xy(checkpoint: Mapping[str, Any]) -> tuple[float, float]:
    rectification = checkpoint.get("source_fov_rectification")
    if not isinstance(rectification, Mapping):
        raise ValueError("parent checkpoint lacks source FOV rectification")
    raw = rectification.get("center_crop_fraction_xy")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("parent checkpoint source crop contract is malformed")
    result = tuple(float(value) for value in raw)
    if result != (1.0, 1.0):
        raise ValueError("corrected physical RGB must not be cropped")
    return result


def _occupancy_weights(checkpoint: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    objective = checkpoint.get("occupancy_training_objective")
    if not isinstance(objective, Mapping):
        raise ValueError("parent checkpoint lacks occupancy objective")
    terms = objective.get("terms")
    if not isinstance(terms, Mapping):
        raise ValueError("parent checkpoint occupancy objective has no terms")
    unknown_known = torch.tensor(terms["unknown_vs_known"]["weights"], dtype=torch.float32)
    free_occupied = torch.tensor(
        terms["free_vs_occupied_given_known"]["weights"], dtype=torch.float32
    )
    if unknown_known.shape != (2,) or free_occupied.shape != (2,):
        raise ValueError("parent checkpoint occupancy weights are malformed")
    return unknown_known, free_occupied


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    invocation_argv = [
        str(Path(__file__).resolve()),
        *(sys.argv[1:] if argv is None else argv),
    ]
    dataset_manifest_path = args.dataset_manifest.resolve()
    checkpoint_path = args.parent_checkpoint.resolve()
    report_path = args.parent_report.resolve()
    output_path = args.output.resolve()
    if output_path in {checkpoint_path, report_path, dataset_manifest_path}:
        raise ValueError("rescore output must not overwrite any parent/input artifact")

    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != args.expected_parent_checkpoint_sha256:
        raise ValueError("parent checkpoint SHA-256 does not match expected digest")
    report_sha256 = sha256_file(report_path)
    if report_sha256 != args.expected_parent_report_sha256:
        raise ValueError("parent report SHA-256 does not match expected digest")
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)

    checkpoint = _load_parent_checkpoint(checkpoint_path)
    report = _read_json(report_path)
    parent_contract = _validate_parent_contracts(
        checkpoint,
        report,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        dataset_manifest_path=dataset_manifest_path,
        dataset_manifest_sha256=dataset_manifest_sha256,
    )
    parent_contract["report"] = {"path": str(report_path), "sha256": report_sha256}
    manifest = _read_json(dataset_manifest_path)
    _validate_physical_dataset_manifest(manifest)
    projective_query_support = _validate_projective_query_support_artifacts(
        checkpoint,
        manifest,
        report=report,
    )
    if projective_query_support is not None:
        parent_contract["projective_query_support"] = projective_query_support
    index_path = Path(str(manifest["index"]["path"])).resolve()
    if sha256_file(index_path) != str(manifest["index"]["sha256"]):
        raise ValueError("dataset row-index SHA-256 mismatch")
    rows = _read_rows(index_path)
    scene_roles = resolve_dataset_scene_roles(
        rows,
        manifest,
        legacy_selection_seed="physical_development_rescore_forbidden_legacy",
    )
    saved_subsets = checkpoint.get("row_subsets")
    assert isinstance(saved_subsets, Mapping)
    rows_by_role = _exact_saved_subset_rows(rows, scene_roles, saved_subsets)

    calibration_provenance = checkpoint["probability_calibration_provenance"]
    calibration_subset = saved_subsets["probability_calibration"]
    if (
        calibration_provenance.get("dataset_manifest_sha256")
        != dataset_manifest_sha256
        or calibration_provenance.get("calibration_row_subset_sha256")
        != calibration_subset["identity_sha256"]
        or int(calibration_provenance.get("calibration_row_count", -1))
        != len(rows_by_role["probability_calibration"])
        or int(calibration_provenance.get("best_epoch", -1))
        != int(checkpoint["best_epoch"])
    ):
        raise ValueError("parent calibration provenance is not bound to saved subset")
    artifact_verification = _verify_selected_artifacts(rows_by_role)
    calibration, frozen_identity = _calibration_identity(checkpoint, report)

    primitive_to_index = {
        str(name): int(index)
        for name, index in checkpoint["primitive_to_index"].items()
    }
    unseen = {
        str(row["primitive"])
        for role in DEVELOPMENT_ROLES
        for row in rows_by_role[role]
        if str(row["primitive"]) not in primitive_to_index
    }
    if unseen:
        raise ValueError(f"development rescore contains unseen primitives: {sorted(unseen)}")
    device = _resolve_device(args.device)
    model_config = dict(checkpoint["model_config"])
    model = EgomotionBevJepa(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    unknown_known_weights, free_occupied_weights = _occupancy_weights(checkpoint)
    nominal_delta_table = torch.tensor(
        checkpoint["nominal_primitive_delta_current"], dtype=torch.float32
    )
    source_crop_fraction_xy = _source_crop_fraction_xy(checkpoint)

    loaders = {
        role: _loader(
            rows_by_role[role],
            primitive_to_index=primitive_to_index,
            image_size=int(model_config["image_size"]),
            source_crop_fraction_xy=source_crop_fraction_xy,
            batch_size=args.batch_size,
            workers=args.workers,
            shuffle=False,
            seed=args.seed,
        )
        for role in DEVELOPMENT_ROLES
    }
    calibration_metrics = evaluate_model(
        model,
        loaders["probability_calibration"],
        device=device,
        unknown_known_weights=unknown_known_weights,
        free_occupied_weights=free_occupied_weights,
        nominal_delta_table=nominal_delta_table,
        calibration=calibration,
        thresholds=None,
        select_thresholds=True,
        occupancy_target_space=PHYSICAL_OCCUPANCY_TARGET_SPACE,
    )
    corrected_thresholds = TraversabilityThresholds(
        **dict(calibration_metrics["thresholds"])
    )
    corrected_thresholds.validate()
    selection_metrics = evaluate_model(
        model,
        loaders["checkpoint_selection"],
        device=device,
        unknown_known_weights=unknown_known_weights,
        free_occupied_weights=free_occupied_weights,
        nominal_delta_table=nominal_delta_table,
        calibration=calibration,
        thresholds=corrected_thresholds,
        select_thresholds=False,
        occupancy_target_space=PHYSICAL_OCCUPANCY_TARGET_SPACE,
    )
    if int(calibration_metrics.get("rows", -1)) != len(
        rows_by_role["probability_calibration"]
    ) or int(selection_metrics.get("rows", -1)) != len(
        rows_by_role["checkpoint_selection"]
    ):
        raise ValueError("development rescore evaluation row count mismatch")

    current_sources = {
        "rescore_helper_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "trainer_source": {
            "path": str(TRAINER_SOURCE.resolve()),
            "sha256": sha256_file(TRAINER_SOURCE),
        },
        "traversability_metrics_source": {
            "path": str(METRICS_SOURCE.resolve()),
            "sha256": sha256_file(METRICS_SOURCE),
        },
    }
    access_ledger = {
        "schema": "lewm_go2_development_rescore_access_ledger_v1",
        "scope": "rescore_process",
        "row_index_metadata": {
            "read": True,
            "path": str(index_path),
            "sha256": str(manifest["index"]["sha256"]),
            "all_role_metadata_read": True,
            "g2_row_metadata_count": sum(
                scene_roles[str(row["scene_id"])] == "g2_evaluation" for row in rows
            ),
        },
        "roles": {
            role: {
                **artifact_verification[role],
                "row_subset_sha256": saved_subsets[role]["identity_sha256"],
                "model_output_rows": len(rows_by_role[role]),
            }
            for role in DEVELOPMENT_ROLES
        },
        "train": {
            "label_shard_byte_opens": 0,
            "image_byte_opens": 0,
            "model_output_rows": 0,
        },
        "g2_contact": {
            "row_metadata_read": True,
            "row_metadata_count": sum(
                scene_roles[str(row["scene_id"])] == "g2_evaluation" for row in rows
            ),
            "label_shard_byte_opens": 0,
            "image_byte_opens": 0,
            "model_output_rows": 0,
        },
    }
    old_sources = parent_contract.pop("old_sources")
    core = {
        "schema": OUTPUT_SCHEMA,
        "classification": "development_diagnostic_only",
        "purpose": "post_smoke_occupied_detection_threshold_selector_correction",
        "parent_artifacts": parent_contract,
        "source_transition": {
            "parent": old_sources,
            "current": current_sources,
        },
        "frozen_identity": frozen_identity,
        "dataset_access_ledger": access_ledger,
        "row_subsets": {
            role: dict(saved_subsets[role]) for role in DEVELOPMENT_ROLES
        },
        "parent_thresholds": dict(checkpoint["traversability_thresholds"]),
        "corrected_calibration_role_evaluation": calibration_metrics,
        "corrected_thresholds": dict(calibration_metrics["thresholds"]),
        "checkpoint_selection_at_corrected_thresholds": selection_metrics,
        "eligibility": {
            "training_performed": False,
            "calibration_refit": False,
            "checkpoint_emitted_or_mutated": False,
            "g2_evaluated": False,
            "one_shot_promotion_eligible": False,
            "runtime_promotion_eligible": False,
            "wiring_evidence_only": True,
            "reason": "parent_is_two_epoch_development_smoke",
        },
        "execution": {
            "invocation": {
                "argv": invocation_argv,
                "command": " ".join(shlex.quote(value) for value in invocation_argv),
            },
            "resolved_config": _json_normalize(vars(args)),
            "git": _git_snapshot(),
        },
    }
    output = {**core, "content_sha256": _canonical_json_sha256(core)}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
