from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from lewm.benchmarks.traversability_metrics import (
    TraversabilityThresholds,
    evaluate_traversability,
)

from lewm.models.egomotion_bev_jepa import EgomotionBevJepa
from lewm.models.egomotion_bev_jepa import (
    GLOBAL_CROSS_ATTENTION_LIFT,
    PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
    PROJECTIVE_COLUMN_ATTENTION_LIFT,
    PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
    build_projective_query_support_contract,
)
from lewm.datasets.go2_paired_navigation import (
    DATASET_ROLES,
    canonical_json_sha256,
    deterministic_family_role_split,
    scene_id_sha256,
)
from scripts import train_go2_egomotion_bev_jepa as trainer_script

from scripts.train_go2_egomotion_bev_jepa import (
    DEVELOPMENT_DATASET_ROLES,
    OCCUPANCY_LOSS_MODE,
    PHYSICAL_CHECKPOINT_SCHEMA,
    PHYSICAL_G2_GATE_SCHEMA,
    PHYSICAL_REPORT_SCHEMA,
    PairedNavigationTorchDataset,
    REGISTERED_PHYSICAL_OCCUPIED_DETECTION_PROBABILITY_CANDIDATES,
    REGISTERED_PROJECTIVE_FOOTPRINT_PERIMETER_SAMPLES,
    _artifact_schemas,
    _configure_determinism,
    _hierarchical_occupancy_objective,
    _json_normalize,
    _mask_array,
    _model_config,
    _observed_mask_array,
    _row_subset_record,
    _selection_score,
    _source_fov_rectification_contract,
    _train_one_epoch,
    _validate_execution_protocol,
    _validate_source_camera_contract,
    _validate_lift_arguments,
    _validate_projective_query_support_artifacts,
    _validate_physical_morphology_contract,
    _verify_dataset_role_provenance,
    collect_calibration_sample,
    deterministic_row_subset,
    evaluate_model,
    evaluate_g2_gate,
    evaluate_physical_evidence_g2_gate,
    fit_vector_calibration,
    main as train_main,
    nominal_primitive_delta_table,
    resolve_dataset_scene_roles,
    split_validation_scenes,
)


def test_legacy_v03_fov_rectification_matches_platform_pinhole() -> None:
    contract = _source_fov_rectification_contract(
        mode="legacy_v03_square_vertical_fov_v1",
        intended_horizontal_fov_deg=78.323,
        native_resolution=(640, 480),
    )

    assert contract["source_horizontal_fov_deg"] == pytest.approx(78.323)
    assert contract["source_vertical_fov_deg"] == pytest.approx(78.323)
    assert contract["intended_vertical_fov_deg"] == pytest.approx(62.8370386364)
    assert contract["center_crop_fraction_xy"] == pytest.approx([1.0, 0.75])
    assert contract["runtime_crop_required"] is False


def test_dataset_applies_source_vertical_crop_before_resize(tmp_path) -> None:
    pixels = np.zeros((4, 4, 3), dtype=np.uint8)
    pixels[[0, 3], :, 0] = 255
    pixels[1:3, :, 1] = 255
    image_path = tmp_path / "fov_fixture.png"
    Image.fromarray(pixels).save(image_path)
    dataset = PairedNavigationTorchDataset(
        [],
        primitive_to_index={"hold": 0},
        image_size=2,
        source_crop_fraction_xy=(1.0, 0.5),
    )

    normalized = dataset._image(str(image_path))
    mean = normalized.new_tensor((0.485, 0.456, 0.406))[:, None, None]
    std = normalized.new_tensor((0.229, 0.224, 0.225))[:, None, None]
    restored = normalized * std + mean

    assert torch.allclose(restored[0], torch.zeros_like(restored[0]), atol=1e-6)
    assert torch.allclose(restored[1], torch.ones_like(restored[1]), atol=1e-6)
    assert torch.allclose(restored[2], torch.zeros_like(restored[2]), atol=1e-6)


def test_source_camera_contract_is_content_and_dataset_bound(tmp_path) -> None:
    dataset_path = tmp_path / "dataset_manifest.json"
    dataset_path.write_text("{}\n")
    dataset_sha = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    rectification = _source_fov_rectification_contract(
        mode="legacy_v03_square_vertical_fov_v1",
        intended_horizontal_fov_deg=78.323,
        native_resolution=(640, 480),
    )
    core = {
        "schema": "lewm_go2_source_camera_contract_v1",
        "dataset_manifest": {"path": str(dataset_path), "sha256": dataset_sha},
        "actual_source_projection": {
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 78.323,
        },
        "platform_projection_after_rectification": {
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 62.8370386364,
            "center_crop_fraction_xy": [1.0, 0.75],
        },
        "scene_count": 96,
        "g2_images_opened": False,
    }
    artifact = {**core, "content_sha256": canonical_json_sha256(core)}
    artifact_path = tmp_path / "source_camera_contract.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True) + "\n")

    record = _validate_source_camera_contract(
        artifact_path,
        dataset_manifest_path=dataset_path,
        rectification=rectification,
    )
    assert record["content_sha256"] == artifact["content_sha256"]
    assert record["scene_count"] == 96

    artifact["scene_count"] = 95
    artifact_path.write_text(json.dumps(artifact, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="content hash mismatch"):
        _validate_source_camera_contract(
            artifact_path,
            dataset_manifest_path=dataset_path,
            rectification=rectification,
        )


def test_v04_render_audit_binds_camera_objects_and_source_index(tmp_path) -> None:
    index_sha = "1" * 64
    core = {
        "schema": "lewm_go2_selected_render_audit_v1",
        "scene_count": 96,
        "output_source_index": {"sha256": index_sha},
        "camera_projection": {
            "horizontal_fov_deg": 78.323,
            "vertical_fov_deg": 62.8370386364,
        },
        "object_contract": {
            "collision_distractors_rendered": True,
            "full_box_roll_pitch_yaw_rendered": True,
        },
        "g2_image_bytes_hashed_for_integrity": True,
        "g2_images_decoded_or_inspected": False,
        "g2_image_content_metrics_computed": False,
        "g2_label_shards_opened": False,
        "g2_model_outputs_opened": False,
    }
    artifact = {**core, "content_sha256": canonical_json_sha256(core)}
    artifact_path = tmp_path / "render_audit.json"
    artifact_path.write_text(json.dumps(artifact, sort_keys=True) + "\n")
    artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    dataset_path = tmp_path / "dataset_manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "render_audit_contract": {
                    "file_sha256": artifact_sha,
                    "content_sha256": artifact["content_sha256"],
                    "output_source_index": {"sha256": index_sha},
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    rectification = _source_fov_rectification_contract(
        mode="none",
        intended_horizontal_fov_deg=78.323,
        native_resolution=(640, 480),
    )

    record = _validate_source_camera_contract(
        artifact_path,
        dataset_manifest_path=dataset_path,
        rectification=rectification,
    )
    assert record["schema"] == "lewm_go2_selected_render_audit_v1"
    assert record["scene_count"] == 96

    artifact["object_contract"]["collision_distractors_rendered"] = False
    core = dict(artifact)
    core.pop("content_sha256")
    artifact["content_sha256"] = canonical_json_sha256(core)
    artifact_path.write_text(json.dumps(artifact, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="does not bind"):
        _validate_source_camera_contract(
            artifact_path,
            dataset_manifest_path=dataset_path,
            rectification=rectification,
        )


def _model_args(**overrides) -> SimpleNamespace:
    values = {
        "image_size": 28,
        "patch_size": 14,
        "encoder_dim": 12,
        "encoder_depth": 1,
        "encoder_heads": 3,
        "bev_dim": 8,
        "predictor_hidden_dim": 12,
        "ema_momentum": 0.5,
        "jepa_weight": 1.0,
        "occupancy_weight": 2.0,
        "equivariance_weight": 0.25,
        "action_contrast_weight": 1.0,
        "action_margin_fraction": 0.1,
        "variance_weight": 0.1,
        "variance_target_std": 0.5,
        "bev_lift_type": GLOBAL_CROSS_ATTENTION_LIFT,
        "projective_horizontal_fov_deg": None,
        "projective_vertical_fov_deg": None,
        "projective_camera_xyz_body_m": None,
        "projective_camera_rpy_body_rad": None,
        "projective_near_m": None,
        "projective_vertical_anchor_z_body_m": None,
        "projective_attention_sigma_tokens": 1.0,
        "projective_attention_bias_floor": -6.0,
        "projective_footprint_radius_m": None,
        "projective_footprint_perimeter_samples": None,
        "development_only": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _small_grid() -> dict:
    return {
        "shape": [4, 4],
        "forward_center_range_m": [0.1, 0.7],
        "left_center_range_m": [-0.3, 0.3],
    }


def _physical_query_support_manifest() -> dict:
    aggregation = {
        "schema": "lewm_observable_physical_aggregation_v1",
        "source_cell_size_m": 0.05,
        "output_cell_size_m": 0.1,
        "free_rule": "unit free rule",
        "occupied_rule": "unit occupied rule",
        "known_class_precedence": "OCCUPIED_then_FREE_else_UNKNOWN",
        "collision_geometry_veto": "unit collision veto",
    }
    aggregation["contract_sha256"] = canonical_json_sha256(aggregation)
    return {
        "schema": "lewm_go2_paired_navigation_dataset_v3",
        "local_grid": {**_small_grid(), "cell_size_m": 0.1},
        "label_semantics": {
            "label_contract": "observable_physical_occupancy_v3",
            "target_occupancy_space": "observable_physical_occupancy",
            "per_frame_configuration_classes_supervised": False,
            "physical_aggregation": aggregation,
        },
    }


def test_legacy_model_config_retains_v2_key_contract() -> None:
    args = _model_args()
    _validate_lift_arguments(args)
    config = _model_config(args, 4, _small_grid())
    assert "bev_lift_type" not in config
    EgomotionBevJepa(**config)


def test_projective_model_config_is_explicit_and_constructible() -> None:
    args = _model_args(
        bev_lift_type=PROJECTIVE_COLUMN_ATTENTION_LIFT,
        projective_horizontal_fov_deg=78.323,
        projective_vertical_fov_deg=78.323,
        projective_camera_xyz_body_m=(0.326, 0.0, 0.043),
        projective_camera_rpy_body_rad=(0.0, 0.0, 0.0),
        projective_near_m=0.05,
        projective_vertical_anchor_z_body_m=(-0.33, -0.13, 0.07, 0.27, 0.47),
    )
    _validate_lift_arguments(args)
    config = _model_config(args, 4, _small_grid())
    assert config["bev_lift_type"] == PROJECTIVE_COLUMN_ATTENTION_LIFT
    model = EgomotionBevJepa(**config)
    assert model.bev_decoder.projective_attention_bias is not None


def test_cell_square_model_config_uses_only_manifest_derived_cell_support() -> None:
    args = _model_args(
        bev_lift_type=PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        projective_horizontal_fov_deg=78.323,
        projective_vertical_fov_deg=62.8370386364,
        projective_camera_xyz_body_m=(0.326, 0.0, 0.043),
        projective_camera_rpy_body_rad=(0.0, 0.0, 0.0),
        projective_near_m=0.05,
        projective_vertical_anchor_z_body_m=(-0.33, 0.07, 0.47),
    )
    manifest = _physical_query_support_manifest()
    support = build_projective_query_support_contract(manifest)
    _validate_lift_arguments(args)
    config = _model_config(
        args,
        4,
        manifest["local_grid"],
        projective_query_support=support,
    )

    assert config["bev_lift_type"] == PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT
    assert config["projective_output_cell_size_m"] == pytest.approx(0.1)
    assert "projective_footprint_radius_m" not in config
    model = EgomotionBevJepa(**config)
    assert model.bev_decoder.projective_horizontal_offsets_body_m == (
        (0.0, 0.0),
        (-0.05, -0.05),
        (-0.05, 0.05),
        (0.05, -0.05),
        (0.05, 0.05),
    )
    with pytest.raises(ValueError, match="requires projective query support"):
        _model_config(args, 4, manifest["local_grid"])


def test_cell_square_lift_rejects_body_footprint_arguments() -> None:
    args = _model_args(
        bev_lift_type=PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
        projective_horizontal_fov_deg=78.323,
        projective_vertical_fov_deg=62.8370386364,
        projective_camera_xyz_body_m=(0.326, 0.0, 0.043),
        projective_camera_rpy_body_rad=(0.0, 0.0, 0.0),
        projective_near_m=0.05,
        projective_vertical_anchor_z_body_m=(-0.33, 0.07, 0.47),
        projective_footprint_radius_m=0.47,
        projective_footprint_perimeter_samples=8,
    )
    with pytest.raises(SystemExit, match="forbidden"):
        _validate_lift_arguments(args)


def test_footprint_projective_model_config_is_explicit_and_constructible() -> None:
    args = _model_args(
        bev_lift_type=PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
        projective_horizontal_fov_deg=78.323,
        projective_vertical_fov_deg=62.8370386364,
        projective_camera_xyz_body_m=(0.326, 0.0, 0.043),
        projective_camera_rpy_body_rad=(0.0, 0.0, 0.0),
        projective_near_m=0.05,
        projective_vertical_anchor_z_body_m=(-0.33, -0.13, 0.07, 0.27, 0.47),
        projective_footprint_radius_m=0.47,
        projective_footprint_perimeter_samples=8,
    )
    _validate_lift_arguments(args)
    config = _model_config(args, 4, _small_grid())
    assert config["bev_lift_type"] == PROJECTIVE_FOOTPRINT_ATTENTION_LIFT
    assert config["projective_footprint_radius_m"] == pytest.approx(0.47)
    assert config["projective_footprint_perimeter_samples"] == 8
    model = EgomotionBevJepa(**config)
    assert len(model.bev_decoder.projective_horizontal_offsets_body_m) == 9


def test_center_projective_lift_rejects_footprint_arguments() -> None:
    args = _model_args(
        bev_lift_type=PROJECTIVE_COLUMN_ATTENTION_LIFT,
        projective_horizontal_fov_deg=78.323,
        projective_vertical_fov_deg=62.8370386364,
        projective_camera_xyz_body_m=(0.326, 0.0, 0.043),
        projective_camera_rpy_body_rad=(0.0, 0.0, 0.0),
        projective_near_m=0.05,
        projective_vertical_anchor_z_body_m=(-0.33, 0.07, 0.47),
        projective_footprint_radius_m=0.47,
        projective_footprint_perimeter_samples=8,
    )
    with pytest.raises(SystemExit, match="footprint arguments require"):
        _validate_lift_arguments(args)


def test_footprint_projective_lift_binds_registered_support_count() -> None:
    args = _model_args(
        bev_lift_type=PROJECTIVE_FOOTPRINT_ATTENTION_LIFT,
        projective_horizontal_fov_deg=78.323,
        projective_vertical_fov_deg=62.8370386364,
        projective_camera_xyz_body_m=(0.326, 0.0, 0.043),
        projective_camera_rpy_body_rad=(0.0, 0.0, 0.0),
        projective_near_m=0.05,
        projective_vertical_anchor_z_body_m=(-0.33, 0.07, 0.47),
        projective_footprint_radius_m=0.47,
        projective_footprint_perimeter_samples=4,
    )
    assert REGISTERED_PROJECTIVE_FOOTPRINT_PERIMETER_SAMPLES == 8
    with pytest.raises(SystemExit, match="preregistered value"):
        _validate_lift_arguments(args)


def test_projective_lift_is_development_only_and_requires_geometry() -> None:
    missing = _model_args(bev_lift_type=PROJECTIVE_COLUMN_ATTENTION_LIFT)
    with pytest.raises(SystemExit, match="missing"):
        _validate_lift_arguments(missing)

    promoted = _model_args(
        bev_lift_type=PROJECTIVE_COLUMN_ATTENTION_LIFT,
        projective_horizontal_fov_deg=78.323,
        projective_vertical_fov_deg=78.323,
        projective_camera_xyz_body_m=(0.326, 0.0, 0.043),
        projective_camera_rpy_body_rad=(0.0, 0.0, 0.0),
        projective_near_m=0.05,
        projective_vertical_anchor_z_body_m=(-0.33, 0.07, 0.47),
        development_only=False,
    )
    with pytest.raises(SystemExit, match="development-only"):
        _validate_lift_arguments(promoted)


def test_cell_square_support_is_explicit_in_all_training_artifacts() -> None:
    manifest = _physical_query_support_manifest()
    support = build_projective_query_support_contract(manifest)
    checkpoint = {
        "model_config": {
            "bev_lift_type": PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT,
            "projective_output_cell_size_m": 0.1,
        },
        "projective_query_support": support,
        "occupancy_output_contract": {
            "projective_query_support_contract_sha256": support["contract_sha256"]
        },
        "training_run_provenance": {"projective_query_support": support},
    }
    report = {"projective_query_support": support}

    assert _validate_projective_query_support_artifacts(
        checkpoint,
        manifest,
        report=report,
    ) == support
    changed_report = copy.deepcopy(report)
    changed_report["projective_query_support"]["uses_body_footprint"] = True
    with pytest.raises(ValueError, match="training report"):
        _validate_projective_query_support_artifacts(
            checkpoint,
            manifest,
            report=changed_report,
        )


def test_old_projective_artifacts_remain_valid_without_query_support() -> None:
    checkpoint = {
        "model_config": {"bev_lift_type": PROJECTIVE_COLUMN_ATTENTION_LIFT},
        "occupancy_output_contract": {},
        "training_run_provenance": {},
    }
    assert _validate_projective_query_support_artifacts(
        checkpoint,
        _physical_query_support_manifest(),
        report={},
    ) is None


def _selection_metrics_for_score() -> dict:
    return {
        "traversability": {
            "useful_traversable_recall": 0.7,
            "planner_admitted_free_precision": 0.995,
            "obstacle_detection_recall_within_range": 0.96,
            "obstacle_exclusion_recall_within_range": 0.97,
        },
        "predictive_controls": {
            "panels": {"changed": {"prediction_to_warped_persistence_ratio": 0.2}}
        },
        "g2": {
            "checks": {
                "planner_admitted_free_precision_ge_0_99": True,
                "obstacle_exclusion_within_2m_ge_0_95": True,
                "obstacle_recall_within_2m_ge_0_95": True,
                "useful_traversable_recall_ge_0_90": False,
                "predictive_check": False,
            }
        },
        "threshold_selection": {
            "candidate_count": 288,
            "passing_candidate_count": 2,
        },
        "losses": {"loss": 0.5, "occupancy_loss": 0.25},
    }


def test_occupancy_ceiling_selection_score_ignores_predictive_checks() -> None:
    metrics = _selection_metrics_for_score()
    first = _selection_score(metrics, mode="occupancy_ceiling_v1")
    metrics["g2"]["checks"]["predictive_check"] = True
    metrics["predictive_controls"]["panels"]["changed"][
        "prediction_to_warped_persistence_ratio"
    ] = 0.99
    assert _selection_score(metrics, mode="occupancy_ceiling_v1") == first
    assert first[:2] == (1.0, 3.0)


def test_physical_occupancy_selection_uses_only_occupancy_evidence() -> None:
    metrics = _selection_metrics_for_score()
    metrics["occupancy_target_space"] = "observable_physical_occupancy"
    metrics["physical_evidence"] = {
        "admitted_observable_physical_free_precision": 0.995,
        "directly_observable_physical_obstacle_recall_within_2m": 0.96,
        "useful_observable_physical_free_recall": 0.7,
        "observable_physical_obstacle_exclusion_recall_within_2m": 0.97,
    }
    metrics["g2"]["checks"] = {
        "heldout_probability_calibration_applied": False,
        "admitted_observable_physical_free_precision_ge_0_99": True,
        "directly_observable_physical_obstacle_recall_within_2m_ge_0_95": True,
        "useful_observable_physical_free_recall_ge_0_90": False,
        "predictive_check": False,
    }
    physical = _selection_score(
        metrics,
        mode="physical_occupancy_ceiling_v1",
    )
    metrics["routing"] = {"route_success_rate": 0.0}
    metrics["g2"]["checks"]["predictive_check"] = True
    metrics["predictive_controls"]["panels"]["changed"][
        "prediction_to_warped_persistence_ratio"
    ] = 100.0
    assert _selection_score(
        metrics,
        mode="physical_occupancy_ceiling_v1",
    ) == physical
    assert physical[:2] == (1.0, 2.0)


def test_physical_execution_protocol_never_retrains_for_one_shot_g2() -> None:
    development = SimpleNamespace(
        evaluate_physical_g2_once=False,
        selection_score_mode="physical_occupancy_ceiling_v1",
        development_only=True,
        probability_calibration_mode="hierarchical_log_odds_v1",
    )
    _validate_execution_protocol(development, physical_dataset=True)

    retraining = SimpleNamespace(**{**vars(development), "development_only": False})
    with pytest.raises(SystemExit, match="training is development-only"):
        _validate_execution_protocol(retraining, physical_dataset=True)

    mislabeled_one_shot = SimpleNamespace(
        **{**vars(retraining), "evaluate_physical_g2_once": True}
    )
    with pytest.raises(SystemExit, match="evaluation-only"):
        _validate_execution_protocol(mislabeled_one_shot, physical_dataset=True)

    legacy = SimpleNamespace(
        evaluate_physical_g2_once=False,
        selection_score_mode="full_g2_v1",
        development_only=False,
        probability_calibration_mode="vector_scaling_v1",
    )
    _validate_execution_protocol(legacy, physical_dataset=False)


def test_one_shot_cli_requires_a_frozen_physical_checkpoint(tmp_path) -> None:
    with pytest.raises(SystemExit, match="requires --frozen-physical-checkpoint"):
        train_main(
            [
                "--dataset-manifest",
                str(tmp_path / "must-not-be-opened.json"),
                "--output",
                str(tmp_path / "out.pt"),
                "--evaluate-physical-g2-once",
            ]
        )
    with pytest.raises(SystemExit, match="expected-frozen-checkpoint-sha256"):
        train_main(
            [
                "--dataset-manifest",
                str(tmp_path / "must-not-be-opened.json"),
                "--output",
                str(tmp_path / "out.pt"),
                "--evaluate-physical-g2-once",
                "--frozen-physical-checkpoint",
                str(tmp_path / "must-not-be-opened.pt"),
            ]
        )


def test_physical_target_has_dedicated_nonruntime_artifact_schemas() -> None:
    assert _artifact_schemas(
        physical_dataset=True,
        bev_lift_type=GLOBAL_CROSS_ATTENTION_LIFT,
        probability_calibration_mode="vector_scaling_v1",
    ) == (PHYSICAL_CHECKPOINT_SCHEMA, PHYSICAL_REPORT_SCHEMA)
    assert PHYSICAL_CHECKPOINT_SCHEMA.endswith("_v4")

    assert _artifact_schemas(
        physical_dataset=False,
        bev_lift_type=GLOBAL_CROSS_ATTENTION_LIFT,
        probability_calibration_mode="vector_scaling_v1",
    )[0].endswith("_v2")


def test_physical_morphology_prerequisite_is_exact_and_fail_closed() -> None:
    morphology = {
        "schema": "lewm_post_memory_configuration_morphology_v1",
        "radius_m": 0.47,
        "memory_cell_size_m": 0.10,
        "support_contract_sha256": "a" * 64,
        "operation": "conservative_inflation",
    }
    semantics = {
        "configuration_inflation_radius_m": 0.47,
        "post_memory_configuration_derivation": morphology,
    }
    output_contract = {
        "post_memory_configuration_derivation": copy.deepcopy(morphology),
    }
    record = _validate_physical_morphology_contract(semantics, output_contract)
    assert record["exact_dataset_checkpoint_match"] is True
    assert record["radius_m"] == pytest.approx(0.47)
    assert record["support_contract_sha256"] == "a" * 64

    changed = copy.deepcopy(output_contract)
    changed["post_memory_configuration_derivation"]["radius_m"] = 0.46
    with pytest.raises(ValueError, match="differs from dataset"):
        _validate_physical_morphology_contract(semantics, changed)


def _passing_physical_gate_metrics() -> dict:
    return {
        "occupancy_target_space": "observable_physical_occupancy",
        "calibration": {"applied": True},
        "routing": {
            "applicability": "not_applicable",
            "valid_for_target_space": False,
            "excluded_from_gate": True,
        },
        "physical_evidence": {
            "admitted_observable_physical_free_precision": 0.99,
            "directly_observable_physical_obstacle_recall_within_2m": 0.95,
            "useful_observable_physical_free_recall": 0.90,
        },
    }


def test_physical_g2_gate_is_route_independent_and_exactly_99_95_90() -> None:
    metrics = _passing_physical_gate_metrics()
    result = evaluate_physical_evidence_g2_gate(metrics)
    assert result["schema"] == PHYSICAL_G2_GATE_SCHEMA
    assert result["routing_included"] is False
    assert result["passes"]

    metrics["routing"] = {"route_success_rate": 0.0, "collision_rate": 1.0}
    assert evaluate_physical_evidence_g2_gate(metrics) == result
    for name in (
        "admitted_observable_physical_free_precision",
        "directly_observable_physical_obstacle_recall_within_2m",
        "useful_observable_physical_free_recall",
    ):
        failed = copy.deepcopy(metrics)
        failed["physical_evidence"][name] -= 1e-6
        assert not evaluate_physical_evidence_g2_gate(failed)["passes"]


def test_physical_obstacle_gate_ignores_occupied_cells_beyond_two_metres() -> None:
    probabilities = np.asarray(
        [[[[0.01, 0.01]], [[0.01, 0.98]], [[0.98, 0.01]]]],
        dtype=np.float64,
    )
    labels = np.asarray([[[2, 2]]], dtype=np.int64)
    distances = np.asarray([[[1.0, 3.0]]], dtype=np.float64)
    thresholds = TraversabilityThresholds(
        free_probability_min=0.9,
        occupied_probability_max=0.1,
        unknown_probability_max=0.1,
    )
    first = evaluate_traversability(
        probabilities,
        labels,
        distances,
        thresholds=thresholds,
    )
    probabilities[:, :, :, 1] = np.asarray([0.01, 0.01, 0.98])[:, None]
    second = evaluate_traversability(
        probabilities,
        labels,
        distances,
        thresholds=thresholds,
    )
    assert first.true_occupied_count == second.true_occupied_count == 2
    assert first.true_occupied_within_range_count == 1
    assert second.true_occupied_within_range_count == 1
    assert first.obstacle_detection_recall_within_range == pytest.approx(1.0)
    assert second.obstacle_detection_recall_within_range == pytest.approx(1.0)


def test_physical_evaluator_selects_low_calibrated_occupied_detection_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert REGISTERED_PHYSICAL_OCCUPIED_DETECTION_PROBABILITY_CANDIDATES == (
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.35,
        0.50,
    )
    model = EgomotionBevJepa(
        image_size=28,
        patch_size=14,
        encoder_dim=12,
        encoder_depth=1,
        encoder_heads=3,
        bev_dim=8,
        bev_size=(8, 8),
        forward_range_m=(-0.35, 0.35),
        left_range_m=(-0.35, 0.35),
        action_dim=4,
        predictor_hidden_dim=12,
    )
    label_grid = torch.zeros(8, 8, dtype=torch.long)
    label_grid[2:6, 2:6] = 1
    label_grid[2:6, 6] = 2
    samples = [
        {
            "current_image": torch.randn(3, 28, 28),
            "next_image": torch.randn(3, 28, 28),
            "action": torch.eye(4)[index],
            "delta": torch.tensor([0.05, 0.0, 0.0]),
            "current_labels": label_grid,
            "next_labels": label_grid,
            "current_mask": torch.ones(8, 8, dtype=torch.bool),
            "next_mask": torch.ones(8, 8, dtype=torch.bool),
            "current_observed_mask": label_grid != 0,
            "next_observed_mask": label_grid != 0,
        }
        for index in range(2)
    ]
    calibrated_class_probabilities = torch.tensor(
        (
            (0.98, 0.01, 0.01),
            (0.01, 0.98, 0.01),
            (0.965, 0.01, 0.025),
        ),
        dtype=torch.float32,
    )

    def low_occupied_posterior(logits, _log_scales, _biases):
        probabilities = calibrated_class_probabilities.to(logits.device)[
            label_grid.to(logits.device)
        ]
        return probabilities.permute(2, 0, 1).log().expand_as(logits)

    monkeypatch.setattr(
        trainer_script,
        "apply_vector_calibration",
        low_occupied_posterior,
    )
    common = {
        "device": torch.device("cpu"),
        "unknown_known_weights": torch.ones(2),
        "free_occupied_weights": torch.ones(2),
        "nominal_delta_table": torch.tensor(
            (
                (0.05, 0.0, 0.0),
                (0.0, 0.05, 0.0),
                (0.0, 0.0, 0.1),
                (0.0, 0.0, -0.1),
            )
        ),
        "calibration": {
            "id": "unit-low-occupied-posterior",
            "method": "positive_diagonal_vector_scaling_with_centered_bias",
            "log_scales": [0.0, 0.0, 0.0],
            "biases": [0.0, 0.0, 0.0],
        },
        "thresholds": None,
        "select_thresholds": True,
    }
    physical = evaluate_model(
        model,
        DataLoader(samples, batch_size=2),
        occupancy_target_space="observable_physical_occupancy",
        **common,
    )
    legacy = evaluate_model(
        model,
        DataLoader(samples, batch_size=2),
        occupancy_target_space="body_inflated_configuration_space",
        **common,
    )

    assert physical["thresholds"]["occupied_detection_min"] == pytest.approx(0.02)
    assert (
        physical["thresholds"]["occupied_probability_max"]
        < physical["thresholds"]["occupied_detection_min"]
    )
    assert physical["threshold_selection"]["candidate_count"] == 2016
    assert physical["threshold_selection"]["passing_candidate_count"] > 0
    assert physical["traversability"][
        "obstacle_detection_recall_within_range"
    ] == pytest.approx(1.0)
    assert physical["g2"]["passes"]
    assert legacy["thresholds"]["occupied_detection_min"] == pytest.approx(0.5)
    assert legacy["threshold_selection"]["candidate_count"] == 288
    assert legacy["traversability"][
        "obstacle_detection_recall_within_range"
    ] == pytest.approx(0.0)


def test_occupancy_ceiling_selection_requires_threshold_sweep() -> None:
    metrics = _selection_metrics_for_score()
    metrics["threshold_selection"] = None
    with pytest.raises(ValueError, match="role-local threshold sweep"):
        _selection_score(metrics, mode="occupancy_ceiling_v1")


def test_selection_score_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="unsupported selection score mode"):
        _selection_score(_selection_metrics_for_score(), mode="unknown")


class _CalibrationModel(torch.nn.Module):
    bev_size = (4, 4)

    def occupancy_logits(self, images: torch.Tensor) -> torch.Tensor:
        batch = images.shape[0]
        values = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
        return values.repeat(batch, 1, 1, 1)


def test_validation_roles_are_deterministic_disjoint_and_nonempty() -> None:
    scenes = [f"scene_{index:02d}" for index in range(11)]
    first = split_validation_scenes(scenes, seed="fixed")
    second = split_validation_scenes(reversed(scenes), seed="fixed")
    assert first == second
    assert set(first) == set(scenes)
    assert set(first.values()) == {
        "checkpoint_selection",
        "probability_calibration",
        "g2_evaluation",
    }


def _direct_role_rows_and_manifest() -> tuple[list[dict], dict]:
    families = {
        f"{family}_{index}": family
        for family in ("alpha", "beta")
        for index in range(4)
    }
    assignments = deterministic_family_role_split(
        families, role_scenes_per_family=1, seed="direct-fixed"
    )
    rows = [
        {
            "scene_id": scene_id,
            "family": family,
            "dataset_role": assignments[scene_id],
            "dataset_split": (
                "train" if assignments[scene_id] == "train" else "validation"
            ),
        }
        for scene_id, family in families.items()
    ]
    role_scene_ids = {
        role: sorted(scene for scene, assigned in assignments.items() if assigned == role)
        for role in DATASET_ROLES
    }
    scene_counts = {role: len(role_scene_ids[role]) for role in DATASET_ROLES}
    row_counts = dict(scene_counts)
    family_scene_counts = {
        family: {
            role: sum(
                assignments[scene] == role
                for scene, item in families.items()
                if item == family
            )
            for role in DATASET_ROLES
        }
        for family in ("alpha", "beta")
    }
    manifest = {
        "scene_roles": {
            "schema": "lewm_go2_family_scene_roles_v1",
            "seed": "direct-fixed",
            "role_scenes_per_family": 1,
            "assignments": assignments,
            "assignments_sha256": canonical_json_sha256(assignments),
            "scene_counts": scene_counts,
            "row_counts": row_counts,
            "family_scene_counts": family_scene_counts,
            "family_row_counts": family_scene_counts,
            "scene_id_sha256_commitments": {
                role: canonical_json_sha256(
                    sorted(scene_id_sha256(scene) for scene in role_scene_ids[role])
                )
                for role in DATASET_ROLES
            },
            "label_independent": True,
        }
    }
    return rows, manifest


def test_trainer_honors_direct_roles_without_resplitting() -> None:
    rows, manifest = _direct_role_rows_and_manifest()
    first = resolve_dataset_scene_roles(
        rows, manifest, legacy_selection_seed="ignored-one"
    )
    second = resolve_dataset_scene_roles(
        list(reversed(rows)), manifest, legacy_selection_seed="ignored-two"
    )
    assert first == second == manifest["scene_roles"]["assignments"]


def test_trainer_verifies_development_and_g2_as_separate_role_scopes(
    tmp_path,
    monkeypatch,
) -> None:
    _rows, manifest = _direct_role_rows_and_manifest()
    assignments = manifest["scene_roles"]["assignments"]
    calls = []

    def recording_verifier(path, **kwargs):
        calls.append((path, kwargs))
        return {"selected_scene": 6}

    monkeypatch.setitem(
        _verify_dataset_role_provenance.__globals__,
        "verify_dataset_provenance",
        recording_verifier,
    )
    development = _verify_dataset_role_provenance(
        tmp_path / "dataset_manifest.json",
        manifest,
        assignments,
        roles=DEVELOPMENT_DATASET_ROLES,
    )
    assert development["roles"] == list(DEVELOPMENT_DATASET_ROLES)
    assert calls[0][1] == {
        "verify_images": True,
        "roles": DEVELOPMENT_DATASET_ROLES,
    }
    assert "g2_evaluation" not in calls[0][1]["roles"]

    g2 = _verify_dataset_role_provenance(
        tmp_path / "dataset_manifest.json",
        manifest,
        assignments,
        roles=("g2_evaluation",),
    )
    assert g2["roles"] == ["g2_evaluation"]
    assert calls[1][1] == {
        "verify_images": True,
        "roles": ("g2_evaluation",),
    }


def test_trainer_rejects_tampered_or_empty_direct_role_contract() -> None:
    rows, manifest = _direct_role_rows_and_manifest()
    tampered_rows = copy.deepcopy(rows)
    tampered_rows[0]["dataset_role"] = "train"
    with pytest.raises(ValueError, match="disagree with its direct role"):
        resolve_dataset_scene_roles(
            tampered_rows, manifest, legacy_selection_seed="ignored"
        )

    missing_rows = [row for row in rows if row["dataset_role"] != "g2_evaluation"]
    with pytest.raises(ValueError, match="assigned_without_rows"):
        resolve_dataset_scene_roles(
            missing_rows, manifest, legacy_selection_seed="ignored"
        )


def test_trainer_preserves_legacy_train_validation_api() -> None:
    rows = [
        {"scene_id": "train", "dataset_split": "train"},
        *[
            {"scene_id": f"validation_{index}", "dataset_split": "validation"}
            for index in range(3)
        ],
    ]
    roles = resolve_dataset_scene_roles(
        rows, {}, legacy_selection_seed="legacy-fixed"
    )
    assert roles["train"] == "train"
    assert set(roles.values()) == set(DATASET_ROLES)


def test_row_subset_retains_every_scene_and_is_order_independent() -> None:
    rows = [
        {"scene_id": f"scene_{scene}", "global_row": scene * 100 + row}
        for scene in range(5)
        for row in range(20)
    ]
    first = deterministic_row_subset(rows, maximum_rows=17, seed="fixed")
    second = deterministic_row_subset(list(reversed(rows)), maximum_rows=17, seed="fixed")
    assert first == second
    assert len(first) == 17
    assert {row["scene_id"] for row in first} == {f"scene_{index}" for index in range(5)}


def test_json_normalization_resolves_paths_and_tuples(tmp_path) -> None:
    normalized = _json_normalize(
        {
            "path": tmp_path / "artifact.pt",
            "nested": (Path("relative.json"), (1, 2)),
        }
    )
    assert normalized == {
        "nested": [str(Path("relative.json").resolve()), [1, 2]],
        "path": str((tmp_path / "artifact.pt").resolve()),
    }
    json.dumps(normalized, sort_keys=True)


def test_row_subset_record_binds_exact_global_order_and_artifact_hashes() -> None:
    rows = [
        {
            "global_row": global_row,
            "scene_id": f"scene_{global_row}",
            "label_shard_row": global_row + 10,
            "label_shard_sha256": str(global_row) * 64,
            "current_image_sha256": "a" * 64,
            "next_image_sha256": "b" * 64,
        }
        for global_row in (2, 1)
    ]
    record = _row_subset_record(rows, role="train")
    assert record["count"] == 2
    assert [item["global_row"] for item in record["identities"]] == [1, 2]
    assert record["identities"][0]["label_shard_row"] == 11
    assert record["identity_sha256"] == canonical_json_sha256(
        record["identities"]
    )


def test_legacy_unknown_excluding_validity_mask_is_rejected() -> None:
    labels = np.array([[[0, 1], [2, 0]]], dtype=np.uint8)
    shard = {
        "current_labels": labels,
        "current_validity": labels != 0,
    }
    with pytest.raises(ValueError, match="legacy validity"):
        _mask_array(shard, "current", labels)


def test_supervision_mask_keeps_unknown_targets() -> None:
    labels = np.array([[[0, 1], [2, 0]]], dtype=np.uint8)
    mask = np.ones_like(labels, dtype=bool)
    shard = {
        "current_labels": labels,
        "current_supervision_mask": mask,
        "current_observed_mask": labels != 0,
    }
    np.testing.assert_array_equal(_mask_array(shard, "current", labels), mask)


def test_hierarchical_objective_uses_equal_capacity_binary_terms(tmp_path) -> None:
    shard_path = tmp_path / "labels.npz"
    current = np.array([[[0, 0, 0], [1, 1, 2]]], dtype=np.uint8)
    nxt = np.array([[[0, 0, 1], [1, 1, 2]]], dtype=np.uint8)
    np.savez(
        shard_path,
        current_labels=current,
        next_labels=nxt,
        current_supervision_mask=np.ones_like(current, dtype=bool),
        next_supervision_mask=np.ones_like(nxt, dtype=bool),
    )

    unknown_known, free_occupied, provenance = _hierarchical_occupancy_objective(
        [{"label_shard_path": str(shard_path), "label_shard_row": 0}]
    )

    assert provenance["mode"] == OCCUPANCY_LOSS_MODE
    assert provenance["three_class_weights"] is None
    assert provenance["three_class_class_counts"] == {
        "unknown": 5,
        "free": 5,
        "occupied": 2,
    }
    unknown_known_term = provenance["terms"]["unknown_vs_known"]
    free_occupied_term = provenance["terms"]["free_vs_occupied_given_known"]
    assert unknown_known_term["counts"] == [5, 7]
    assert free_occupied_term["counts"] == [5, 2]
    assert 5 * float(unknown_known[0]) == pytest.approx(
        7 * float(unknown_known[1])
    )
    assert 5 * float(free_occupied[0]) == pytest.approx(
        2 * float(free_occupied[1])
    )


def test_deterministic_execution_is_strict_and_auditable() -> None:
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        provenance = _configure_determinism(True)
        assert provenance["requested"] is True
        assert provenance["torch_deterministic_algorithms"] is True
        assert provenance["torch_deterministic_warn_only"] is False
        assert provenance["cudnn_deterministic"] is True
        assert provenance["cudnn_benchmark"] is False
        assert provenance["nondeterministic_operation_policy"] == "error"
    finally:
        torch.use_deterministic_algorithms(
            previous_algorithms,
            warn_only=previous_warn_only,
        )
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark


def test_observed_mask_is_explicit_and_matches_known_labels() -> None:
    labels = np.array([[[0, 1], [2, 0]]], dtype=np.uint8)
    observed = labels != 0
    shard = {"current_observed_mask": observed}
    np.testing.assert_array_equal(
        _observed_mask_array(shard, "current", labels), observed
    )
    with pytest.raises(ValueError, match="labels != UNKNOWN"):
        _observed_mask_array(
            {"current_observed_mask": np.ones_like(observed)},
            "current",
            labels,
        )


def test_nominal_delta_table_uses_training_medians_and_circular_yaw() -> None:
    rows = [
        {
            "primitive": "forward",
            "relative_se2_current_frame": [value, 0.0, 0.01],
        }
        for value in (0.1, 0.2, 4.0)
    ] + [
        {
            "primitive": "turn",
            "relative_se2_current_frame": [0.0, 0.0, yaw],
        }
        for yaw in (3.12, -3.13, 3.10)
    ]
    table = nominal_primitive_delta_table(
        rows, {"forward": 0, "turn": 1}
    )
    assert table.shape == (2, 3)
    assert table[0, 0] == pytest.approx(0.2)
    assert abs(abs(float(table[1, 2])) - np.pi) < 0.05


def test_vector_calibration_reduces_heldout_multiclass_nll() -> None:
    labels = torch.arange(3).repeat_interleave(100)
    logits = torch.nn.functional.one_hot(labels, num_classes=3).float() * 2.0
    logits = logits + torch.tensor([3.0, 0.0, -2.0])
    fitted = fit_vector_calibration(logits, labels, maximum_iterations=40)
    assert fitted["sample_count"] == 300
    assert fitted["nll_after"] < fitted["nll_before"]


def test_calibration_sampler_minimally_backfills_rare_available_class() -> None:
    labels = torch.tensor(
        [[0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.long,
    )
    loader = DataLoader(
        [
            {
                "next_image": torch.zeros(3, 4, 4),
                "next_labels": labels,
                "next_mask": torch.ones(4, 4, dtype=torch.bool),
            }
        ],
        batch_size=1,
    )

    first_logits, first_labels, first_sampling = collect_calibration_sample(
        _CalibrationModel(), loader, device=torch.device("cpu"), maximum_cells=3
    )
    second_logits, second_labels, second_sampling = collect_calibration_sample(
        _CalibrationModel(), loader, device=torch.device("cpu"), maximum_cells=3
    )

    torch.testing.assert_close(first_logits, second_logits)
    torch.testing.assert_close(first_labels, second_labels)
    assert first_sampling == second_sampling
    assert first_sampling["source_class_counts"] == {
        "unknown": 14,
        "free": 1,
        "occupied": 1,
    }
    assert first_sampling["uniform_sample_class_counts"] == {
        "unknown": 2,
        "free": 1,
        "occupied": 0,
    }
    assert first_sampling["backfilled_classes"] == ["occupied"]
    assert first_sampling["replaced_cell_count"] == 1
    assert first_sampling["final_sample_class_counts"] == {
        "unknown": 1,
        "free": 1,
        "occupied": 1,
    }


def test_promotion_calibration_forbids_rare_class_backfill() -> None:
    labels = torch.tensor(
        [[0, 2, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.long,
    )
    loader = DataLoader(
        [
            {
                "next_image": torch.zeros(3, 4, 4),
                "next_labels": labels,
                "next_mask": torch.ones(4, 4, dtype=torch.bool),
            }
        ],
        batch_size=1,
    )

    with pytest.raises(RuntimeError, match=r"forbids rare-class backfill.*'occupied'"):
        collect_calibration_sample(
            _CalibrationModel(),
            loader,
            device=torch.device("cpu"),
            maximum_cells=3,
            allow_rare_class_backfill=False,
        )

    _, _, sampling = collect_calibration_sample(
        _CalibrationModel(),
        loader,
        device=torch.device("cpu"),
        maximum_cells=16,
        allow_rare_class_backfill=False,
    )
    assert sampling["rare_class_backfill_allowed"] is False
    assert sampling["backfilled_classes"] == []


def test_calibration_sampler_names_class_missing_from_source_role() -> None:
    labels = torch.zeros(4, 4, dtype=torch.long)
    labels[0, 1] = 2
    loader = DataLoader(
        [
            {
                "next_image": torch.zeros(3, 4, 4),
                "next_labels": labels,
                "next_mask": torch.ones(4, 4, dtype=torch.bool),
            }
        ],
        batch_size=1,
    )

    with pytest.raises(
        ValueError,
        match=r"missing=\['free'\].*'unknown': 15.*'free': 0.*'occupied': 1",
    ):
        collect_calibration_sample(
            _CalibrationModel(), loader, device=torch.device("cpu"), maximum_cells=8
        )


def test_g2_requires_historical_action_margin_and_connected_routes() -> None:
    panel = {
        "valid_cells": 100,
        "prediction_to_warped_persistence_ratio": 0.9,
        "warped_persistence_error": 0.01,
        "zero_action_advantage_over_target_change": 0.11,
        "shuffled_action_advantage_over_target_change": 0.11,
        "wrong_commanded_delta_advantage_over_target_change": 0.01,
    }
    metrics = {
        "calibration": {"applied": True},
        "traversability": {
            "planner_admitted_free_precision": 0.995,
            "obstacle_detection_recall_within_range": 0.96,
            "obstacle_exclusion_recall_within_range": 0.96,
            "useful_traversable_recall": 0.91,
        },
        "predictive_controls": {
            "panels": {
                "observed": dict(panel),
                "changed": dict(panel),
            },
            "target_cross_sample_std_mean": 0.1,
            "target_cross_sample_effective_rank_mean": 5.0,
        },
        "routing": {
            "planned_path_collision_rate": 0.0,
            "oracle_map_collision_rate": 0.0,
            "oracle_routable_paths": 10,
            "route_success_rate": 0.9,
            "mean_route_length_recall": 0.9,
        },
    }
    assert evaluate_g2_gate(metrics)["passes"]

    weak_action = copy.deepcopy(metrics)
    weak_action["predictive_controls"]["panels"]["changed"][
        "shuffled_action_advantage_over_target_change"
    ] = 0.099
    assert not evaluate_g2_gate(weak_action)["passes"]

    no_routes = copy.deepcopy(metrics)
    no_routes["routing"]["route_success_rate"] = 0.0
    assert not evaluate_g2_gate(no_routes)["passes"]


def test_epoch_trainer_selects_hierarchical_occupancy_branch(monkeypatch) -> None:
    torch.manual_seed(19)
    model = EgomotionBevJepa(
        image_size=28,
        patch_size=14,
        encoder_dim=12,
        encoder_depth=1,
        encoder_heads=3,
        bev_dim=8,
        bev_size=(8, 8),
        forward_range_m=(-0.35, 0.35),
        left_range_m=(-0.35, 0.35),
        action_dim=4,
        predictor_hidden_dim=12,
    )
    labels = torch.zeros(8, 8, dtype=torch.long)
    labels[2:6, 2:6] = 1
    labels[2:6, 6] = 2
    samples = [
        {
            "current_image": torch.randn(3, 28, 28),
            "next_image": torch.randn(3, 28, 28),
            "action": torch.eye(4)[index],
            "delta": torch.tensor([0.05, 0.0, 0.0]),
            "current_labels": labels,
            "next_labels": labels,
            "current_mask": torch.ones(8, 8, dtype=torch.bool),
            "next_mask": torch.ones(8, 8, dtype=torch.bool),
            "current_observed_mask": labels != 0,
            "next_observed_mask": labels != 0,
        }
        for index in range(2)
    ]
    recorded_loss_arguments = []
    original_forward = model.forward

    def recording_forward(*args, **kwargs):
        recorded_loss_arguments.append(
            {
                "class_weights": kwargs.get("occupancy_class_weights"),
                "unknown_known": kwargs.get("occupancy_unknown_known_weights"),
                "free_occupied": kwargs.get("occupancy_free_occupied_weights"),
            }
        )
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", recording_forward)
    unknown_known_weights = torch.tensor([1.25, 0.75])
    free_occupied_weights = torch.tensor([0.4, 1.6])
    metrics = _train_one_epoch(
        model,
        DataLoader(samples, batch_size=2),
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
        device=torch.device("cpu"),
        unknown_known_weights=unknown_known_weights,
        free_occupied_weights=free_occupied_weights,
        nominal_delta_table=torch.tensor(
            [
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
                [0.0, 0.0, 0.1],
                [0.0, 0.0, -0.1],
            ]
        ),
        gradient_clip=1.0,
        epoch=1,
    )

    assert np.isfinite(metrics["loss"])
    assert recorded_loss_arguments
    for arguments in recorded_loss_arguments:
        assert arguments["class_weights"] is None
        torch.testing.assert_close(arguments["unknown_known"], unknown_known_weights)
        torch.testing.assert_close(arguments["free_occupied"], free_occupied_weights)


def test_evaluator_runs_hierarchical_loss_and_matched_panels(monkeypatch) -> None:
    torch.manual_seed(23)
    model = EgomotionBevJepa(
        image_size=28,
        patch_size=14,
        encoder_dim=12,
        encoder_depth=1,
        encoder_heads=3,
        bev_dim=8,
        bev_size=(8, 8),
        forward_range_m=(-0.35, 0.35),
        left_range_m=(-0.35, 0.35),
        action_dim=4,
        predictor_hidden_dim=12,
    )
    labels = torch.zeros(8, 8, dtype=torch.long)
    labels[2:6, 2:6] = 1
    labels[2:6, 6] = 2
    samples = []
    for index in range(4):
        current_labels = torch.roll(labels, shifts=index % 2, dims=0)
        samples.append(
            {
                "current_image": torch.randn(3, 28, 28),
                "next_image": torch.randn(3, 28, 28),
                "action": torch.eye(4)[index],
                "delta": torch.tensor([0.05, 0.0, 0.0]),
                "current_labels": current_labels,
                "next_labels": labels,
                "current_mask": torch.ones(8, 8, dtype=torch.bool),
                "next_mask": torch.ones(8, 8, dtype=torch.bool),
                "current_observed_mask": current_labels != 0,
                "next_observed_mask": labels != 0,
            }
        )
    recorded_loss_arguments = []
    original_forward = model.forward

    def recording_forward(*args, **kwargs):
        recorded_loss_arguments.append(
            {
                "class_weights": kwargs.get("occupancy_class_weights"),
                "unknown_known": kwargs.get("occupancy_unknown_known_weights"),
                "free_occupied": kwargs.get("occupancy_free_occupied_weights"),
            }
        )
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", recording_forward)
    unknown_known_weights = torch.tensor([1.25, 0.75])
    free_occupied_weights = torch.tensor([0.4, 1.6])
    metrics = evaluate_model(
        model,
        DataLoader(samples, batch_size=2),
        device=torch.device("cpu"),
        unknown_known_weights=unknown_known_weights,
        free_occupied_weights=free_occupied_weights,
        nominal_delta_table=torch.tensor(
            [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]
        ),
        calibration=None,
        thresholds=None,
        select_thresholds=True,
        occupancy_target_space="body_inflated_configuration_space",
    )
    assert metrics["predictive_controls"]["panels"]["observed"]["valid_cells"] > 0
    assert metrics["predictive_controls"]["panels"]["changed"]["valid_cells"] > 0
    assert metrics["routing"]["oracle_routable_paths"] > 0
    assert recorded_loss_arguments
    for arguments in recorded_loss_arguments:
        assert arguments["class_weights"] is None
        torch.testing.assert_close(arguments["unknown_known"], unknown_known_weights)
        torch.testing.assert_close(arguments["free_occupied"], free_occupied_weights)


def test_full_main_serializes_normalized_role_scoped_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    geometry_path = tmp_path / "geometry.json"
    geometry_path.write_text(
        json.dumps(
            {
                "camera": {
                    "horizontal_fov_deg": 78.323,
                    "nominal_xyz_body_m": [0.326, 0.0, 0.043],
                    "nominal_rpy_body_rad": [0.0, 0.0, 0.0],
                    "near_m": 0.05,
                },
                "swept_footprint": {"planning_disc_radius_m": 0.47},
            }
        )
        + "\n"
    )
    roles = {
        "train_scene": "train",
        "selection_scene": "checkpoint_selection",
        "calibration_scene": "probability_calibration",
        "g2_scene": "g2_evaluation",
    }
    rows = []
    for global_row, (scene_id, role) in enumerate(roles.items()):
        rows.append(
            {
                "global_row": global_row,
                "scene_id": scene_id,
                "family": "unit",
                "dataset_role": role,
                "dataset_split": "train" if role == "train" else "validation",
                "primitive": "hold",
                "relative_se2_current_frame": [0.0, 0.0, 0.0],
                "label_shard_path": str(tmp_path / f"{scene_id}.npz"),
                "label_shard_row": 0,
                "label_shard_sha256": f"{global_row:x}" * 64,
                "current_image_path": str(tmp_path / f"{scene_id}-current.png"),
                "next_image_path": str(tmp_path / f"{scene_id}-next.png"),
                "current_image_sha256": "a" * 64,
                "next_image_sha256": "b" * 64,
            }
        )
    index_path = tmp_path / "rows.jsonl"
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    dataset_path = tmp_path / "dataset_manifest.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema": "lewm_go2_paired_navigation_dataset_v2",
                "label_semantics": {
                    "target_occupancy_space": "body_inflated_configuration_space"
                },
                "geometry_contract": {
                    "path": str(geometry_path),
                    "sha256": "c" * 64,
                },
                "index": {
                    "path": str(index_path),
                    "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
                },
                "local_grid": {
                    "shape": [2, 2],
                    "cell_size_m": 0.1,
                    "forward_edge_range_m": [0.0, 0.2],
                    "left_edge_range_m": [-0.1, 0.1],
                    "forward_center_range_m": [0.05, 0.15],
                    "left_center_range_m": [-0.05, 0.05],
                    "array_axes": {
                        "row": "base_forward_increasing",
                        "column": "base_left_increasing",
                    },
                    "base_frame_axes": {
                        "forward": "+x_base_link",
                        "left": "+y_base_link",
                    },
                    "bounds_are": "cell_edges",
                },
            }
        )
        + "\n"
    )

    monkeypatch.setattr(
        trainer_script,
        "resolve_dataset_scene_roles",
        lambda *_args, **_kwargs: dict(roles),
    )

    def fake_verify(_path, _manifest, scene_roles, *, roles):
        selected = tuple(roles)
        return {
            "roles": list(selected),
            "selector": "unit-test",
            "scene_count": sum(role in selected for role in scene_roles.values()),
            "scene_id_sha256_commitment": "d" * 64,
            "images_verified": True,
            "checked": {
                "shard": len(selected),
                "image": 2 * len(selected),
            },
        }

    monkeypatch.setattr(trainer_script, "_verify_dataset_role_provenance", fake_verify)
    objective = {
        "schema": "lewm_hierarchical_occupancy_objective_v1",
        "mode": OCCUPANCY_LOSS_MODE,
        "terms": {
            "unknown_vs_known": {"weights": [1.0, 1.0]},
            "free_vs_occupied_given_known": {"weights": [1.0, 1.0]},
        },
        "three_class_weights": None,
    }
    monkeypatch.setattr(
        trainer_script,
        "_hierarchical_occupancy_objective",
        lambda _rows: (torch.ones(2), torch.ones(2), objective),
    )

    class FakeModel(torch.nn.Module):
        def __init__(self, **_kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    monkeypatch.setattr(trainer_script, "EgomotionBevJepa", FakeModel)
    monkeypatch.setattr(
        trainer_script,
        "_train_one_epoch",
        lambda *_args, **_kwargs: {
            "loss": 0.5,
            "jepa_loss": 0.2,
            "occupancy_loss": 0.3,
        },
    )
    monkeypatch.setattr(trainer_script, "_loader", lambda rows, **_kwargs: list(rows))

    thresholds = {
        "free_probability_min": 0.9,
        "occupied_probability_max": 0.1,
        "unknown_probability_max": 0.1,
        "occupied_detection_min": 0.5,
    }

    def fake_evaluate(_model, _loader, *, calibration, select_thresholds, **_kwargs):
        return {
            "rows": 1,
            "occupancy_target_space": "body_inflated_configuration_space",
            "losses": {"loss": 0.5, "occupancy_loss": 0.3},
            "traversability": {
                "planner_admitted_free_precision": 0.995,
                "useful_traversable_recall": 0.91,
                "obstacle_detection_recall_within_range": 0.96,
                "obstacle_exclusion_recall_within_range": 0.97,
            },
            "thresholds": thresholds,
            "threshold_selection": (
                {"candidate_count": 288, "passing_candidate_count": 2}
                if select_thresholds
                else None
            ),
            "calibration": {
                "applied": calibration is not None,
                "id": None if calibration is None else calibration.get("id"),
            },
            "predictive_controls": {
                "panels": {
                    "changed": {"prediction_to_warped_persistence_ratio": 0.5}
                }
            },
            "routing": {"valid_for_target_space": True},
            "g2": {
                "passes": False,
                "checks": {
                    "planner_admitted_free_precision_ge_0_99": True,
                    "obstacle_exclusion_within_2m_ge_0_95": True,
                    "obstacle_recall_within_2m_ge_0_95": True,
                    "useful_traversable_recall_ge_0_90": True,
                },
            },
        }

    monkeypatch.setattr(trainer_script, "evaluate_model", fake_evaluate)
    monkeypatch.setattr(
        trainer_script,
        "collect_calibration_sample",
        lambda *_args, **_kwargs: (
            torch.eye(3),
            torch.arange(3),
            {"schema": "unit-calibration-sample", "final_sample_count": 3},
        ),
    )
    monkeypatch.setattr(
        trainer_script,
        "fit_vector_calibration",
        lambda *_args, **_kwargs: {
            "method": "positive_diagonal_vector_scaling_with_centered_bias",
            "log_scales": [0.0, 0.0, 0.0],
            "biases": [0.0, 0.0, 0.0],
            "sample_count": 3,
            "nll_before": 1.0,
            "nll_after": 0.9,
        },
    )
    monkeypatch.setattr(
        trainer_script,
        "_git_snapshot",
        lambda: {"head": "e" * 40, "status_short": "unit-test"},
    )
    experiment_calls = []

    def fake_experiment(**kwargs):
        json.dumps(kwargs["config"], sort_keys=True)
        experiment_calls.append(kwargs)
        return {"schema": "unit_experiment_manifest"}

    monkeypatch.setattr(trainer_script, "build_experiment_manifest", fake_experiment)
    output_path = tmp_path / "model.pt"
    report_path = tmp_path / "model.report.json"
    assert (
        trainer_script.main(
            [
                "--dataset-manifest",
                str(dataset_path),
                "--output",
                str(output_path),
                "--report-output",
                str(report_path),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--workers",
                "0",
                "--device",
                "cpu",
                "--development-only",
                "--selection-score-mode",
                "occupancy_ceiling_v1",
                "--log-every",
                "0",
            ]
        )
        == 0
    )

    checkpoint = torch.load(output_path, map_location="cpu", weights_only=True)
    assert checkpoint["row_subsets"]["train"]["count"] == 1
    assert checkpoint["row_subsets"]["g2_evaluation"]["count"] == 0
    calibration_provenance = checkpoint["probability_calibration_provenance"]
    assert calibration_provenance["dataset_manifest_sha256"] == hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()
    assert calibration_provenance["calibration_row_subset_sha256"] == checkpoint[
        "row_subsets"
    ]["probability_calibration"]["identity_sha256"]
    assert checkpoint["probability_calibration"]["provenance"] == (
        calibration_provenance
    )
    g2_access = checkpoint["dataset_access_ledger"]["roles"]["g2_evaluation"]
    assert g2_access["available_row_count"] == 1
    assert g2_access["selected_row_count"] == 0
    assert g2_access["label_shard_files_hashed"] == 0
    assert g2_access["image_files_hashed"] == 0
    assert g2_access["model_output_rows"] == 0
    assert checkpoint["dataset_access_ledger"]["g2_contact"] == {
        "row_metadata_read": True,
        "row_metadata_count": 1,
        "label_shard_byte_opens": 0,
        "image_byte_opens": 0,
        "model_output_rows": 0,
    }
    assert checkpoint["dataset_role_provenance_verification"][
        "g2_evaluation"
    ] is None
    run = checkpoint["training_run_provenance"]
    run_core = dict(run)
    assert run_core.pop("content_sha256") == canonical_json_sha256(run_core)
    assert run["checkpoint_artifact_included"] is False
    assert run["resolved_config"]["dataset_manifest"] == str(
        dataset_path.resolve()
    )
    assert "checkpoint" not in run["critical_inputs"]
    report = json.loads(report_path.read_text())
    assert report["training_run_provenance"] == run
    assert report["dataset_access_ledger"] == checkpoint["dataset_access_ledger"]
    assert experiment_calls
    assert experiment_calls[0]["config"]["output"] == str(output_path.resolve())
