#!/usr/bin/env python3
"""Diagnose G2 train-vs-selection behavior without opening untouched G2.

The checkpoint's frozen probability calibration and traversability thresholds
are used for both permitted roles.  Evaluation is reported both pooled and with
equal scene weight.  Only ``train`` and ``checkpoint_selection`` image/label
artifacts are ever selected or verified by this script.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lewm.benchmarks.experiment_manifest import sha256_file  # noqa: E402
from lewm.benchmarks.g2_train_selection_diagnostic import (  # noqa: E402
    ALLOWED_DIAGNOSTIC_ROLES,
    compact_scene_metrics,
    compare_roles,
    decompose_role_evaluations,
    family_summaries,
    learning_curve_inputs,
    scene_balanced_summary,
    select_diagnostic_rows,
)
from lewm.benchmarks.traversability_metrics import (  # noqa: E402
    TraversabilityThresholds,
)
from lewm.datasets.go2_paired_navigation import (  # noqa: E402
    LABEL_CONTRACT_CENTER_VISIBLE_V2,
    LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3,
)
from lewm.models.egomotion_bev_jepa import EgomotionBevJepa  # noqa: E402
from scripts.train_go2_egomotion_bev_jepa import (  # noqa: E402
    _hierarchical_occupancy_objective,
    _loader,
    _read_json,
    _read_rows,
    evaluate_model,
    resolve_dataset_scene_roles,
)


SUPPORTED_CHECKPOINT_SCHEMAS = (
    "lewm_go2_egomotion_bev_jepa_checkpoint_v2",
    "lewm_go2_egomotion_bev_jepa_checkpoint_v3",
    "lewm_go2_egomotion_bev_jepa_checkpoint_v4",
)
DATASET_LABEL_CONTRACTS = {
    "lewm_go2_paired_navigation_dataset_v2": {
        "label_contract": LABEL_CONTRACT_CENTER_VISIBLE_V2,
        "target_occupancy_space": "body_inflated_configuration_space",
    },
    "lewm_go2_paired_navigation_dataset_v3": {
        "label_contract": LABEL_CONTRACT_OBSERVABLE_PHYSICAL_V3,
        "target_occupancy_space": "observable_physical_occupancy",
    },
}
REPORT_SCHEMA_BY_CHECKPOINT_SCHEMA = {
    "lewm_go2_egomotion_bev_jepa_checkpoint_v2": (
        "lewm_go2_egomotion_bev_jepa_training_report_v2"
    ),
    "lewm_go2_egomotion_bev_jepa_checkpoint_v3": (
        "lewm_go2_egomotion_bev_jepa_training_report_v3"
    ),
    "lewm_go2_egomotion_bev_jepa_checkpoint_v4": (
        "lewm_go2_egomotion_bev_jepa_training_report_v4"
    ),
}
OCCUPANCY_OBJECTIVE_MODE = "hierarchical_equal_capacity_v1"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--training-report",
        type=Path,
        help="Optional trainer report; defaults to CHECKPOINT.report.json when present.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument(
        "--verify-selected-images",
        action="store_true",
        help="Hash train/selection RGB files too; G2 files remain excluded.",
    )
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("batch-size must be positive and workers must be nonnegative")
    return args


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - environment contract
        raise RuntimeError("torch.load(..., weights_only=True) is required") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be an object")
    if payload.get("schema") not in SUPPORTED_CHECKPOINT_SCHEMAS:
        raise ValueError("unsupported checkpoint schema")
    required = {
        "model_state_dict",
        "model_config",
        "primitive_to_index",
        "nominal_primitive_delta_current",
        "probability_calibration",
        "traversability_thresholds",
        "occupancy_training_objective",
        "dataset_manifest_sha256",
        "training_scene_ids",
        "g2_evaluation",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"checkpoint lacks required fields: {missing}")
    return payload


def _resolve_dataset_label_contract(
    manifest: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve schema-bound occupancy semantics without reading row artifacts."""

    dataset_schema = str(manifest.get("schema", ""))
    expected = DATASET_LABEL_CONTRACTS.get(dataset_schema)
    if expected is None:
        raise ValueError("unsupported dataset schema")
    semantics = manifest.get("label_semantics")
    if dataset_schema == "lewm_go2_paired_navigation_dataset_v3":
        if not isinstance(semantics, Mapping):
            raise ValueError("dataset v3 lacks observable-physical label semantics")
        if (
            semantics.get("per_frame_configuration_classes_supervised") is not False
            or semantics.get(
                "post_memory_configuration_derivation_is_evaluation_only"
            )
            is not True
        ):
            raise ValueError("dataset v3 is not observable physical occupancy")
    elif semantics is not None and not isinstance(semantics, Mapping):
        raise ValueError("dataset label_semantics must be an object")
    semantics = semantics if isinstance(semantics, Mapping) else {}
    actual = {
        "label_contract": str(
            semantics.get("label_contract", expected["label_contract"])
        ),
        "target_occupancy_space": str(
            semantics.get(
                "target_occupancy_space", expected["target_occupancy_space"]
            )
        ),
    }
    if actual != expected:
        raise ValueError(
            "dataset label semantics disagree with its schema: "
            f"expected {expected}, got {actual}"
        )
    return {"dataset_schema": dataset_schema, **actual}


def _validate_checkpoint_dataset_contract(
    checkpoint: Mapping[str, Any],
    dataset_contract: Mapping[str, str],
) -> dict[str, Any]:
    """Require the checkpoint's learned target to match the bound dataset."""

    checkpoint_schema = str(checkpoint.get("schema", ""))
    if checkpoint_schema not in SUPPORTED_CHECKPOINT_SCHEMAS:
        raise ValueError("unsupported checkpoint schema")
    output_contract = checkpoint.get("occupancy_output_contract")
    # Legacy projective/calibration checkpoints used schema v3 while still
    # targeting dataset-v2 configuration occupancy. Their exact manifest hash
    # is checked by main, so v2 can retain its schema-implied target. Dataset
    # v3 must carry the physical target explicitly in the checkpoint.
    requires_explicit_target = (
        dataset_contract["dataset_schema"]
        == "lewm_go2_paired_navigation_dataset_v3"
    )
    if requires_explicit_target and checkpoint_schema != (
        "lewm_go2_egomotion_bev_jepa_checkpoint_v4"
    ):
        raise ValueError("observable-physical dataset requires checkpoint schema v4")
    if not requires_explicit_target and checkpoint_schema == (
        "lewm_go2_egomotion_bev_jepa_checkpoint_v4"
    ):
        raise ValueError("checkpoint schema v4 requires observable-physical dataset")
    if not isinstance(output_contract, Mapping):
        if requires_explicit_target:
            raise ValueError("checkpoint lacks occupancy output semantics")
        output_contract = {}
    expected_target = str(dataset_contract["target_occupancy_space"])
    actual_target = str(
        output_contract.get("target_occupancy_space", expected_target)
    )
    if actual_target != expected_target:
        raise ValueError(
            "checkpoint occupancy target disagrees with dataset label semantics"
        )
    if (
        expected_target == "observable_physical_occupancy"
        and not isinstance(
            output_contract.get("post_memory_configuration_derivation"), Mapping
        )
    ):
        raise ValueError(
            "observable-physical checkpoint lacks post-memory configuration derivation"
        )
    return {
        "checkpoint_schema": checkpoint_schema,
        "target_occupancy_space": actual_target,
    }


def _source_crop_fraction_xy(checkpoint: Mapping[str, Any]) -> tuple[float, float]:
    rectification = checkpoint.get("source_fov_rectification")
    if rectification is None:
        return (1.0, 1.0)
    if not isinstance(rectification, Mapping):
        raise ValueError("checkpoint source FOV rectification must be an object")
    raw = rectification.get("center_crop_fraction_xy")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("checkpoint source FOV rectification lacks crop fractions")
    crop = tuple(float(value) for value in raw)
    if any(not 0.0 < value <= 1.0 for value in crop):
        raise ValueError("checkpoint source crop fractions must lie in (0, 1]")
    return crop


def _validate_occupancy_training_objective(payload: Mapping[str, Any]) -> None:
    objective = payload.get("occupancy_training_objective")
    if not isinstance(objective, Mapping):
        raise ValueError("checkpoint lacks occupancy training-objective provenance")
    if objective.get("mode") != OCCUPANCY_OBJECTIVE_MODE:
        raise ValueError(
            "checkpoint occupancy objective is not the preregistered "
            f"{OCCUPANCY_OBJECTIVE_MODE!r} mode"
        )


def _require_untouched_checkpoint(payload: Mapping[str, Any]) -> None:
    if payload.get("g2_evaluation") is not None or bool(
        payload.get("g2_passes", False)
    ) or payload.get("head_g2_evaluation") is not None or bool(
        payload.get("head_g2_passes", False)
    ):
        raise ValueError(
            "diagnostic accepts only development checkpoints with no stored G2 evaluation"
        )


def _selected_artifact_provenance(
    rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    verify_images: bool,
) -> dict[str, Any]:
    """Verify only artifacts owned by the two permitted roles."""

    shard_hashes: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    for role, rows in rows_by_role.items():
        if role not in ALLOWED_DIAGNOSTIC_ROLES:
            raise ValueError(f"forbidden role reached artifact verifier: {role}")
        for row in rows:
            shard_path = str(row["label_shard_path"])
            expected_shard = str(row["label_shard_sha256"])
            prior = shard_hashes.setdefault(shard_path, expected_shard)
            if prior != expected_shard:
                raise ValueError(f"inconsistent shard hash in row index: {shard_path}")
            if verify_images:
                for prefix in ("current", "next"):
                    image_path = str(row[f"{prefix}_image_path"])
                    expected_image = str(row[f"{prefix}_image_sha256"])
                    prior_image = image_hashes.setdefault(image_path, expected_image)
                    if prior_image != expected_image:
                        raise ValueError(
                            f"inconsistent image hash in row index: {image_path}"
                        )
    for path, expected in sorted(shard_hashes.items()):
        actual = sha256_file(Path(path))
        if actual != expected:
            raise ValueError(f"selected shard hash mismatch: {path}")
    for path, expected in sorted(image_hashes.items()):
        actual = sha256_file(Path(path))
        if actual != expected:
            raise ValueError(f"selected image hash mismatch: {path}")
    return {
        "verified_roles": sorted(rows_by_role),
        "selected_shards_verified": len(shard_hashes),
        "selected_images_verified": len(image_hashes),
        "verify_selected_images": bool(verify_images),
        "g2_shards_opened": False,
        "g2_images_opened": False,
    }


def _load_training_curve(
    checkpoint_path: Path,
    checkpoint_sha256: str,
    requested_report: Path | None,
    *,
    checkpoint_schema: str,
    dataset_manifest_sha256: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    report_path = requested_report
    if report_path is None:
        candidate = checkpoint_path.with_suffix(".report.json")
        report_path = candidate if candidate.is_file() else None
    if report_path is None:
        return None, None
    report_path = report_path.resolve()
    report = _read_json(report_path)
    expected_report_schema = REPORT_SCHEMA_BY_CHECKPOINT_SCHEMA[checkpoint_schema]
    if report.get("schema") != expected_report_schema:
        raise ValueError("unsupported training report schema")
    promotion = report.get("promotion")
    if (
        report.get("final_g2_evaluation") is not None
        or report.get("final_head_g2_evaluation") is not None
        or (
            isinstance(promotion, Mapping)
            and (
                bool(promotion.get("g2_evaluated", False))
                or bool(promotion.get("head_g2_evaluated", False))
            )
        )
    ):
        raise ValueError("training report already contains G2 evaluation output")
    report_checkpoint = report.get("checkpoint")
    if not isinstance(report_checkpoint, Mapping):
        raise ValueError("training report lacks checkpoint provenance")
    if str(report_checkpoint.get("sha256")) != checkpoint_sha256:
        raise ValueError("training report checkpoint SHA-256 mismatch")
    report_dataset = report.get("dataset_manifest")
    if not isinstance(report_dataset, Mapping):
        raise ValueError("training report lacks dataset provenance")
    if str(report_dataset.get("sha256")) != dataset_manifest_sha256:
        raise ValueError("training report dataset manifest SHA-256 mismatch")
    row_counts = report.get("row_counts")
    if not isinstance(row_counts, Mapping):
        raise ValueError("training report lacks role row counts")
    return (
        learning_curve_inputs(report),
        {
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "schema": expected_report_schema,
            "row_counts": {str(key): int(value) for key, value in row_counts.items()},
        },
    )


def _evaluate_rows(
    model: EgomotionBevJepa,
    rows: Sequence[Mapping[str, Any]],
    *,
    primitive_to_index: Mapping[str, int],
    device: torch.device,
    unknown_known_weights: torch.Tensor,
    free_occupied_weights: torch.Tensor,
    nominal_delta_table: torch.Tensor,
    calibration: Mapping[str, Any] | None,
    thresholds: TraversabilityThresholds | None,
    select_thresholds: bool,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
    source_crop_fraction_xy: tuple[float, float],
    occupancy_target_space: str,
) -> dict[str, Any]:
    loader = _loader(
        rows,
        primitive_to_index=primitive_to_index,
        image_size=image_size,
        source_crop_fraction_xy=source_crop_fraction_xy,
        batch_size=batch_size,
        workers=workers,
        shuffle=False,
        seed=seed,
    )
    return evaluate_model(
        model,
        loader,
        device=device,
        unknown_known_weights=unknown_known_weights,
        free_occupied_weights=free_occupied_weights,
        nominal_delta_table=nominal_delta_table,
        calibration=calibration,
        thresholds=thresholds,
        select_thresholds=select_thresholds,
        occupancy_target_space=occupancy_target_space,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    manifest_path = args.dataset_manifest.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    manifest = _read_json(manifest_path)
    dataset_contract = _resolve_dataset_label_contract(manifest)
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_dataset_contract = _validate_checkpoint_dataset_contract(
        checkpoint, dataset_contract
    )
    _validate_occupancy_training_objective(checkpoint)
    _require_untouched_checkpoint(checkpoint)
    manifest_sha256 = sha256_file(manifest_path)
    if str(checkpoint["dataset_manifest_sha256"]) != manifest_sha256:
        raise ValueError("checkpoint dataset manifest SHA-256 mismatch")
    source_crop_fraction_xy = _source_crop_fraction_xy(checkpoint)

    index_path = Path(str(manifest["index"]["path"])).resolve()
    if sha256_file(index_path) != str(manifest["index"]["sha256"]):
        raise ValueError("dataset row-index SHA-256 mismatch")
    rows = _read_rows(index_path)
    scene_roles = resolve_dataset_scene_roles(
        rows,
        manifest,
        legacy_selection_seed="g2_train_selection_diagnostic_legacy_v1",
    )
    rows_by_role = select_diagnostic_rows(rows, scene_roles)
    selected_provenance = _selected_artifact_provenance(
        rows_by_role,
        verify_images=bool(args.verify_selected_images),
    )

    train_scene_ids = sorted(
        scene for scene, role in scene_roles.items() if role == "train"
    )
    if sorted(map(str, checkpoint["training_scene_ids"])) != train_scene_ids:
        raise ValueError("checkpoint training scenes disagree with dataset role contract")
    primitive_to_index = {
        str(key): int(value)
        for key, value in checkpoint["primitive_to_index"].items()
    }
    unseen = {
        str(row["primitive"])
        for role_rows in rows_by_role.values()
        for row in role_rows
        if str(row["primitive"]) not in primitive_to_index
    }
    if unseen:
        raise ValueError(f"diagnostic rows contain unseen primitives: {sorted(unseen)}")

    device = _resolve_device(args.device)
    model_config = dict(checkpoint["model_config"])
    model = EgomotionBevJepa(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    unknown_known_weights, free_occupied_weights, occupancy_objective = (
        _hierarchical_occupancy_objective(rows_by_role["train"])
    )
    unknown_known_weights = unknown_known_weights.to(device)
    free_occupied_weights = free_occupied_weights.to(device)
    if occupancy_objective != checkpoint["occupancy_training_objective"]:
        raise ValueError(
            "checkpoint occupancy training objective disagrees with train-role labels"
        )
    nominal_delta_table = torch.tensor(
        checkpoint["nominal_primitive_delta_current"], dtype=torch.float32
    )
    calibration = checkpoint["probability_calibration"]
    if not isinstance(calibration, Mapping):
        raise ValueError("checkpoint probability calibration must be an object")
    thresholds = TraversabilityThresholds(
        **dict(checkpoint["traversability_thresholds"])
    )
    thresholds.validate()

    checkpoint_sha256 = sha256_file(checkpoint_path)
    curve_inputs, report_provenance = _load_training_curve(
        checkpoint_path,
        checkpoint_sha256,
        args.training_report,
        checkpoint_schema=checkpoint_dataset_contract["checkpoint_schema"],
        dataset_manifest_sha256=manifest_sha256,
    )
    if curve_inputs is not None:
        if int(curve_inputs["best_epoch"]) != int(checkpoint.get("best_epoch", 0)):
            raise ValueError("training report best epoch disagrees with checkpoint")
        assert report_provenance is not None
        evaluated_row_counts = {
            role: len(rows_by_role[role]) for role in ALLOWED_DIAGNOSTIC_ROLES
        }
        trained_row_counts = report_provenance["row_counts"]
        cap_mismatches = {
            role: {
                "training_report": int(trained_row_counts.get(role, -1)),
                "diagnostic": count,
            }
            for role, count in evaluated_row_counts.items()
            if int(trained_row_counts.get(role, -1)) != count
        }
        if cap_mismatches:
            raise ValueError(
                "diagnostic cannot call full-role evaluation training fit when "
                f"training row caps bound: {cap_mismatches}"
            )
    role_reports: dict[str, Any] = {}
    role_decompositions: dict[str, Any] = {}
    pooled_raw: dict[str, dict[str, Any]] = {}
    for role in sorted(ALLOWED_DIAGNOSTIC_ROLES):
        role_rows = rows_by_role[role]
        pooled = _evaluate_rows(
            model,
            role_rows,
            primitive_to_index=primitive_to_index,
            device=device,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
            nominal_delta_table=nominal_delta_table,
            calibration=calibration,
            thresholds=thresholds,
            select_thresholds=False,
            image_size=int(model_config["image_size"]),
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            seed=int(args.seed),
            source_crop_fraction_xy=source_crop_fraction_xy,
            occupancy_target_space=checkpoint_dataset_contract[
                "target_occupancy_space"
            ],
        )
        pooled_raw[role] = pooled
        uncalibrated_frozen = _evaluate_rows(
            model,
            role_rows,
            primitive_to_index=primitive_to_index,
            device=device,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
            nominal_delta_table=nominal_delta_table,
            calibration=None,
            thresholds=thresholds,
            select_thresholds=False,
            image_size=int(model_config["image_size"]),
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            seed=int(args.seed),
            source_crop_fraction_xy=source_crop_fraction_xy,
            occupancy_target_space=checkpoint_dataset_contract[
                "target_occupancy_space"
            ],
        )
        calibrated_role_local = _evaluate_rows(
            model,
            role_rows,
            primitive_to_index=primitive_to_index,
            device=device,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
            nominal_delta_table=nominal_delta_table,
            calibration=calibration,
            thresholds=None,
            select_thresholds=True,
            image_size=int(model_config["image_size"]),
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            seed=int(args.seed),
            source_crop_fraction_xy=source_crop_fraction_xy,
            occupancy_target_space=checkpoint_dataset_contract[
                "target_occupancy_space"
            ],
        )
        uncalibrated_role_local = _evaluate_rows(
            model,
            role_rows,
            primitive_to_index=primitive_to_index,
            device=device,
            unknown_known_weights=unknown_known_weights,
            free_occupied_weights=free_occupied_weights,
            nominal_delta_table=nominal_delta_table,
            calibration=None,
            thresholds=None,
            select_thresholds=True,
            image_size=int(model_config["image_size"]),
            batch_size=int(args.batch_size),
            workers=int(args.workers),
            seed=int(args.seed),
            source_crop_fraction_xy=source_crop_fraction_xy,
            occupancy_target_space=checkpoint_dataset_contract[
                "target_occupancy_space"
            ],
        )
        decomposition = decompose_role_evaluations(
            pooled,
            uncalibrated_frozen,
            calibrated_role_local,
            uncalibrated_role_local,
            occupancy_target_space=checkpoint_dataset_contract[
                "target_occupancy_space"
            ],
        )
        role_decompositions[role] = decomposition
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        families: dict[str, str] = {}
        for row in role_rows:
            scene_id = str(row["scene_id"])
            grouped[scene_id].append(row)
            family = str(row["family"])
            prior_family = families.setdefault(scene_id, family)
            if prior_family != family:
                raise ValueError(f"scene spans multiple families: {scene_id}")
        scene_records = []
        for scene_index, scene_id in enumerate(sorted(grouped)):
            scene_raw = _evaluate_rows(
                model,
                grouped[scene_id],
                primitive_to_index=primitive_to_index,
                device=device,
                unknown_known_weights=unknown_known_weights,
                free_occupied_weights=free_occupied_weights,
                nominal_delta_table=nominal_delta_table,
                calibration=calibration,
                thresholds=thresholds,
                select_thresholds=False,
                image_size=int(model_config["image_size"]),
                batch_size=int(args.batch_size),
                workers=int(args.workers),
                seed=int(args.seed) + scene_index + 1,
                source_crop_fraction_xy=source_crop_fraction_xy,
                occupancy_target_space=checkpoint_dataset_contract[
                    "target_occupancy_space"
                ],
            )
            scene_records.append(
                {
                    "scene_id": scene_id,
                    "family": families[scene_id],
                    "evaluation": compact_scene_metrics(scene_raw),
                }
            )
        role_reports[role] = {
            "pooled_evaluation": pooled,
            "diagnostic_pooled_views": {
                "uncalibrated_frozen_promotion_thresholds": uncalibrated_frozen,
                "checkpoint_calibrated_role_local_thresholds": calibrated_role_local,
                "uncalibrated_role_local_thresholds": uncalibrated_role_local,
            },
            "component_decomposition": decomposition,
            "scene_balanced": scene_balanced_summary(scene_records),
            "family_summaries": family_summaries(scene_records),
            "per_scene": scene_records,
        }

    diagnostic_inputs = compare_roles(
        pooled_raw["train"],
        pooled_raw["checkpoint_selection"],
        role_reports["train"]["scene_balanced"],
        role_reports["checkpoint_selection"]["scene_balanced"],
        curve_inputs=curve_inputs,
        train_decomposition=role_decompositions["train"],
        selection_decomposition=role_decompositions["checkpoint_selection"],
    )
    output = {
        "schema": "lewm_go2_g2_train_selection_diagnostic_v1",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha256,
            "best_epoch": int(checkpoint.get("best_epoch", 0)),
            "schema": checkpoint_dataset_contract["checkpoint_schema"],
            "target_occupancy_space": checkpoint_dataset_contract[
                "target_occupancy_space"
            ],
            "g2_was_already_present_in_checkpoint": False,
        },
        "dataset_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
            "index_path": str(index_path),
            "index_sha256": str(manifest["index"]["sha256"]),
            "scene_roles_sha256": str(
                manifest.get("scene_roles", {}).get("assignments_sha256", "")
            ),
            **dataset_contract,
        },
        "training_report": report_provenance,
        "evaluation_contract": {
            "roles_evaluated": sorted(ALLOWED_DIAGNOSTIC_ROLES),
            "roles_forbidden": ["probability_calibration", "g2_evaluation"],
            "fixed_checkpoint_probability_calibration": True,
            "fixed_checkpoint_thresholds": True,
            "promotion_threshold_selection_performed": False,
            "diagnostic_role_local_threshold_sweeps_performed": True,
            "uncalibrated_role_local_threshold_sweep_performed": True,
            "diagnostic_thresholds_are_not_promotion_eligible": True,
            "uncalibrated_counterfactual_evaluated": True,
            "g2_evaluated": False,
            "g2_images_or_labels_opened": False,
            "label_contract": dataset_contract["label_contract"],
            "target_occupancy_space": dataset_contract[
                "target_occupancy_space"
            ],
            "source_crop_fraction_xy": list(source_crop_fraction_xy),
            "selected_artifact_provenance": selected_provenance,
        },
        "checkpoint_probability_calibration": dict(calibration),
        "occupancy_training_objective_from_train_only": occupancy_objective,
        "role_reports": role_reports,
        "diagnostic_inputs": diagnostic_inputs,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "bounded_read": diagnostic_inputs["bounded_read"],
                "g2_evaluated": False,
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
