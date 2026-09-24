from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from lewm.benchmarks.g2_train_selection_diagnostic import (
    OCCUPANCY_CHECKS,
    PHYSICAL_OCCUPANCY_CHECKS,
    REPRESENTATION_PREDICTION_CHECKS,
    ROUTING_CHECKS,
    compact_scene_metrics,
    compare_roles,
    decompose_role_evaluations,
    family_summaries,
    learning_curve_inputs,
    scene_balanced_summary,
    select_diagnostic_rows,
)
from scripts import diagnose_go2_g2_train_selection as diagnostic_script


def _metrics(*, quality: float, passes: bool, rows: int = 10) -> dict:
    return {
        "rows": rows,
        "occupancy_target_space": "body_inflated_configuration_space",
        "losses": {
            "loss": 1.0 - quality,
            "jepa_loss": 0.5 * (1.0 - quality),
            "occupancy_loss": 0.75 * (1.0 - quality),
        },
        "traversability": {
            "planner_admitted_free_precision": quality,
            "obstacle_detection_recall_within_range": quality,
            "obstacle_exclusion_recall_within_range": quality,
            "useful_traversable_recall": quality,
            "free_probability_ece": 1.0 - quality,
            "free_probability_brier": 0.5 * (1.0 - quality),
            "unknown_admission_rate": 1.0 - quality,
        },
        "predictive_controls": {
            "panels": {
                "observed": {
                    "prediction_to_warped_persistence_ratio": 1.0 - quality,
                    "shuffled_action_advantage_over_target_change": quality,
                },
                "changed": {
                    "prediction_to_warped_persistence_ratio": 1.0 - quality,
                    "shuffled_action_advantage_over_target_change": quality,
                },
            },
            "target_cross_sample_effective_rank_mean": 8.0 * quality,
        },
        "routing": {
            "route_success_rate": quality,
            "mean_route_length_recall": quality,
            "planned_path_collision_rate": 1.0 - quality,
        },
        "g2": {
            "passes": passes,
            "checks": {"first": passes, "second": quality >= 0.5},
        },
    }


def test_row_selection_excludes_calibration_and_g2_artifacts() -> None:
    roles = {
        "train_scene": "train",
        "selection_scene": "checkpoint_selection",
        "calibration_scene": "probability_calibration",
        "g2_scene": "g2_evaluation",
    }
    rows = [
        {
            "scene_id": scene,
            "dataset_role": role,
            "label_shard_path": f"/{scene}.npz",
        }
        for scene, role in roles.items()
    ]
    selected = select_diagnostic_rows(rows, roles)
    selected_paths = {
        row["label_shard_path"]
        for role_rows in selected.values()
        for row in role_rows
    }
    assert selected_paths == {"/train_scene.npz", "/selection_scene.npz"}
    assert "/calibration_scene.npz" not in selected_paths
    assert "/g2_scene.npz" not in selected_paths


def test_row_selection_rejects_role_disagreement() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        select_diagnostic_rows(
            [{"scene_id": "scene", "dataset_role": "g2_evaluation"}],
            {"scene": "train"},
        )


def test_scene_summary_weights_scenes_not_rows() -> None:
    low = {
        "scene_id": "low",
        "family": "alpha",
        "evaluation": compact_scene_metrics(
            _metrics(quality=0.2, passes=False, rows=1000)
        ),
    }
    high = {
        "scene_id": "high",
        "family": "alpha",
        "evaluation": compact_scene_metrics(
            _metrics(quality=0.8, passes=True, rows=1)
        ),
    }
    summary = scene_balanced_summary([low, high])
    assert summary["scene_count"] == 2
    assert summary["row_count"] == 1001
    assert summary["scene_g2_pass_fraction"] == 0.5
    assert summary["metrics"]["planner_admitted_free_precision"]["mean"] == 0.5
    assert summary["metrics"]["total_loss"]["mean"] == 0.5


def test_family_summary_is_explicitly_family_balanced() -> None:
    records = [
        {
            "scene_id": "a1",
            "family": "alpha",
            "evaluation": compact_scene_metrics(
                _metrics(quality=0.2, passes=False)
            ),
        },
        {
            "scene_id": "a2",
            "family": "alpha",
            "evaluation": compact_scene_metrics(
                _metrics(quality=0.4, passes=False)
            ),
        },
        {
            "scene_id": "b1",
            "family": "beta",
            "evaluation": compact_scene_metrics(
                _metrics(quality=0.9, passes=True)
            ),
        },
    ]
    summary = family_summaries(records)
    assert summary["family_count"] == 2
    assert summary["per_family"]["alpha"]["scene_count"] == 2
    assert summary["family_balanced_metric_means"][
        "planner_admitted_free_precision"
    ] == pytest.approx((0.3 + 0.9) / 2.0)


def test_role_decomposition_separates_calibration_thresholds_and_components() -> None:
    def view(*, calibrated: bool, local: bool) -> dict:
        metrics = _metrics(quality=0.8, passes=False)
        metrics["calibration"] = {"applied": calibrated}
        metrics["thresholds"] = {
            "free_probability_min": 0.9 if not local else 0.7,
            "occupied_probability_max": 0.1,
            "unknown_probability_max": 0.1,
            "occupied_detection_min": 0.5,
        }
        metrics["threshold_selection"] = (
            {
                "candidate_count": 288,
                "passing_candidate_count": 2,
                "selection_metrics": {},
            }
            if local
            else None
        )
        metrics["g2"]["checks"] = {
            name: True
            for name in (
                *OCCUPANCY_CHECKS,
                *REPRESENTATION_PREDICTION_CHECKS,
                *ROUTING_CHECKS,
            )
        }
        return metrics

    frozen = view(calibrated=True, local=False)
    frozen["g2"]["checks"]["useful_traversable_recall_ge_0_90"] = False
    frozen["traversability"]["free_probability_ece"] = 0.2
    frozen["traversability"]["free_probability_brier"] = 0.15
    uncalibrated = view(calibrated=False, local=False)
    uncalibrated["traversability"]["free_probability_ece"] = 0.4
    uncalibrated["traversability"]["free_probability_brier"] = 0.3
    role_local = view(calibrated=True, local=True)
    uncalibrated_role_local = view(calibrated=False, local=True)

    result = decompose_role_evaluations(
        frozen,
        uncalibrated,
        role_local,
        uncalibrated_role_local,
        occupancy_target_space="body_inflated_configuration_space",
    )

    assert result["representation_prediction"]["gate"]["passes"] is True
    assert result["threshold_attainability"]["frozen_promotion"][
        "occupancy_gate"
    ]["passes"] is False
    assert result["threshold_attainability"]["calibrated_role_local_diagnostic"][
        "occupancy_gate"
    ]["passes"] is True
    assert result["threshold_attainability"][
        "uncalibrated_role_local_diagnostic"
    ]["occupancy_gate"]["passes"] is True
    assert (
        result["threshold_attainability"]["bounded_read"]
        == "role_local_thresholds_attain_gate_but_frozen_do_not"
    )
    quality = result["occupancy_probability_quality"]
    assert quality["checkpoint_calibration_improves_or_preserves_both"] is True
    assert quality["calibrated_minus_uncalibrated_negative_is_improvement"][
        "free_probability_ece"
    ] == pytest.approx(-0.2)

    damaging_calibration = view(calibrated=True, local=True)
    damaging_calibration["threshold_selection"]["passing_candidate_count"] = 0
    damaging_calibration["g2"]["checks"][
        "useful_traversable_recall_ge_0_90"
    ] = False
    damaging_calibration["g2"]["checks"][
        "planner_admitted_free_precision_ge_0_99"
    ] = False
    destroyed = decompose_role_evaluations(
        frozen,
        uncalibrated,
        damaging_calibration,
        uncalibrated_role_local,
        occupancy_target_space="body_inflated_configuration_space",
    )
    assert (
        destroyed["threshold_attainability"][
            "calibration_effect_on_attainability"
        ]
        == "checkpoint_calibration_destroys_full_gate_attainability"
    )
    assert (
        destroyed["threshold_attainability"]["bounded_read"]
        == "raw_head_attains_gate_but_checkpoint_calibration_destroys_it"
    )

    raw_head_failure = copy.deepcopy(uncalibrated_role_local)
    raw_head_failure["threshold_selection"]["passing_candidate_count"] = 0
    raw_head_failure["g2"]["checks"][
        "useful_traversable_recall_ge_0_90"
    ] = False
    raw_head_failure["g2"]["checks"][
        "planner_admitted_free_precision_ge_0_99"
    ] = False
    unavailable = decompose_role_evaluations(
        frozen,
        uncalibrated,
        damaging_calibration,
        raw_head_failure,
        occupancy_target_space="body_inflated_configuration_space",
    )
    assert (
        unavailable["threshold_attainability"]["raw_head_bounded_read"]
        == "raw_head_has_no_registered_grid_safety_candidate"
    )
    assert (
        unavailable["threshold_attainability"]["bounded_read"]
        == "raw_head_has_no_role_local_safety_candidate_in_registered_grid"
    )


def _physical_metrics(*, calibrated: bool, local: bool) -> dict:
    metrics = _metrics(quality=0.96, passes=calibrated)
    metrics["occupancy_target_space"] = "observable_physical_occupancy"
    metrics["calibration"] = {"applied": calibrated}
    metrics["thresholds"] = {
        "free_probability_min": 0.9,
        "occupied_probability_max": 0.1,
        "unknown_probability_max": 0.1,
        "occupied_detection_min": 0.5,
    }
    metrics["threshold_selection"] = (
        {
            "candidate_count": 288,
            "passing_candidate_count": 2,
            "selection_metrics": {},
        }
        if local
        else None
    )
    metrics["physical_evidence"] = {
        "schema": "lewm_go2_observable_physical_evidence_metrics_v1",
        "admitted_observable_physical_free_precision": 0.995,
        "directly_observable_physical_obstacle_recall_within_2m": 0.96,
        "observable_physical_obstacle_exclusion_recall_within_2m": 0.97,
        "useful_observable_physical_free_recall": 0.91,
        "unknown_evidence_admission_rate": 0.01,
        "free_probability_brier": 0.02,
        "free_probability_ece": 0.03,
    }
    metrics["routing"] = {
        "schema": "lewm_go2_routing_not_applicable_v1",
        "valid_for_target_space": False,
        "applicability": "not_applicable",
        "excluded_from_gate": True,
        "deferred_to": "G3_post_memory_multi_view_fusion",
    }
    checks = {name: True for name in PHYSICAL_OCCUPANCY_CHECKS}
    checks["heldout_probability_calibration_applied"] = calibrated
    metrics["g2"] = {
        "schema": "lewm_go2_physical_evidence_g2_v1",
        "passes": all(checks.values()),
        "routing_included": False,
        "checks": checks,
    }
    return metrics


def test_physical_diagnostic_uses_physical_metrics_and_excludes_routing() -> None:
    frozen = _physical_metrics(calibrated=True, local=False)
    raw = _physical_metrics(calibrated=False, local=False)
    calibrated_local = _physical_metrics(calibrated=True, local=True)
    raw_local = _physical_metrics(calibrated=False, local=True)

    compact = compact_scene_metrics(frozen)
    assert compact["occupancy_target_space"] == "observable_physical_occupancy"
    assert "admitted_observable_physical_free_precision" in compact["metrics"]
    assert "route_success_rate" not in compact["metrics"]
    result = decompose_role_evaluations(
        frozen,
        raw,
        calibrated_local,
        raw_local,
        occupancy_target_space="observable_physical_occupancy",
    )
    assert result["routing"] == {
        "applicability": "not_applicable",
        "included_in_head_g2": False,
        "deferred_to": "G3_post_memory_multi_view_fusion",
    }
    assert result["representation_prediction"]["gate"][
        "included_in_head_g2"
    ] is False


def test_role_comparison_does_not_call_train_failure_data_limited() -> None:
    train_raw = _metrics(quality=0.7, passes=False)
    selection_raw = _metrics(quality=0.4, passes=False)
    train_record = {
        "scene_id": "train",
        "family": "alpha",
        "evaluation": compact_scene_metrics(train_raw),
    }
    selection_record = {
        "scene_id": "selection",
        "family": "alpha",
        "evaluation": compact_scene_metrics(selection_raw),
    }
    comparison = compare_roles(
        train_raw,
        selection_raw,
        scene_balanced_summary([train_record]),
        scene_balanced_summary([selection_record]),
        curve_inputs=None,
    )
    assert (
        comparison["bounded_read"]
        == "train_role_runtime_contract_failure_blocks_generalization_attribution"
    )
    assert comparison["scene_balanced_metric_gaps"][
        "planner_admitted_free_precision"
    ]["directional_gap_positive_means_selection_worse"] == pytest.approx(0.3)
    assert comparison["scene_balanced_metric_gaps"]["total_loss"][
        "directional_gap_positive_means_selection_worse"
    ] == pytest.approx(0.3)


def test_role_comparison_reports_bounded_generalization_gap() -> None:
    train_raw = _metrics(quality=0.99, passes=True)
    selection_raw = _metrics(quality=0.6, passes=False)
    train_record = {
        "scene_id": "train",
        "family": "alpha",
        "evaluation": compact_scene_metrics(train_raw),
    }
    selection_record = {
        "scene_id": "selection",
        "family": "alpha",
        "evaluation": compact_scene_metrics(selection_raw),
    }
    train_decomposition = {
        "threshold_attainability": {
            "frozen_promotion": {"occupancy_gate": {"passes": True}}
        },
        "representation_prediction": {"gate": {"passes": True}},
        "routing": {"frozen_promotion": {"gate": {"passes": True}}},
    }
    selection_decomposition = copy.deepcopy(train_decomposition)
    selection_decomposition["threshold_attainability"]["frozen_promotion"][
        "occupancy_gate"
    ]["passes"] = False
    comparison = compare_roles(
        train_raw,
        selection_raw,
        scene_balanced_summary([train_record]),
        scene_balanced_summary([selection_record]),
        curve_inputs=None,
        train_decomposition=train_decomposition,
        selection_decomposition=selection_decomposition,
    )
    assert comparison["bounded_read"] == "selection_role_generalization_gap_observed"
    assert "does not distinguish" in comparison["explanation"]
    assert comparison["component_gate_inputs"][
        "occupancy_at_frozen_thresholds"
    ]["train"]["passes"] is True
    assert comparison["component_gate_inputs"][
        "occupancy_at_frozen_thresholds"
    ]["checkpoint_selection"]["passes"] is False


def test_learning_curve_inputs_expose_late_slopes_and_caveat() -> None:
    history = []
    for epoch in range(1, 9):
        selection = _metrics(quality=0.5, passes=False)
        selection["g2"]["checks"] = {
            "a": True,
            "b": epoch >= 3,
            "c": False,
        }
        history.append(
            {
                "epoch": epoch,
                "train": {
                    "loss": 1.0 - 0.05 * epoch,
                    "occupancy_loss": 0.8 - 0.04 * epoch,
                    "jepa_loss": 0.6 - 0.02 * epoch,
                },
                "checkpoint_selection": selection,
            }
        )
    inputs = learning_curve_inputs({"best_epoch": 3, "history": history})
    assert inputs["epoch_count"] == 8
    assert inputs["best_epoch"] == 3
    assert inputs["selected_checkpoint_record"]["epoch"] == 3
    assert inputs["epochs_after_selected_checkpoint"] == 5
    assert inputs["train_loss_fractional_change"] < 0.0
    assert inputs["late_slopes_per_epoch"]["train_loss"] == pytest.approx(-0.05)
    assert inputs["late_slopes_per_epoch"]["train_occupancy_loss"] == pytest.approx(
        -0.04
    )
    assert inputs["late_slopes_per_epoch"]["selection_checks_passed"] == 0.0
    assert "cannot" in inputs["caveat"]


def test_compact_metrics_rejects_missing_required_values() -> None:
    broken = copy.deepcopy(_metrics(quality=0.8, passes=False))
    del broken["routing"]["route_success_rate"]
    with pytest.raises(ValueError, match="route_success_rate"):
        compact_scene_metrics(broken)


def test_checkpoint_must_declare_corrected_hierarchical_objective() -> None:
    diagnostic_script._validate_occupancy_training_objective(
        {"occupancy_training_objective": {"mode": "hierarchical_equal_capacity_v1"}}
    )
    with pytest.raises(ValueError, match="training-objective provenance"):
        diagnostic_script._validate_occupancy_training_objective({})
    with pytest.raises(ValueError, match="preregistered"):
        diagnostic_script._validate_occupancy_training_objective(
            {"occupancy_training_objective": {"mode": "legacy_three_class"}}
        )


def test_v2_and_v3_dataset_checkpoint_semantics_are_explicitly_bound() -> None:
    v2_contract = diagnostic_script._resolve_dataset_label_contract(
        {"schema": "lewm_go2_paired_navigation_dataset_v2"}
    )
    assert v2_contract == {
        "dataset_schema": "lewm_go2_paired_navigation_dataset_v2",
        "label_contract": "center_visible_configuration_v2",
        "target_occupancy_space": "body_inflated_configuration_space",
    }
    assert diagnostic_script._validate_checkpoint_dataset_contract(
        {"schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v2"},
        v2_contract,
    )["target_occupancy_space"] == "body_inflated_configuration_space"
    assert diagnostic_script._validate_checkpoint_dataset_contract(
        {"schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v3"},
        v2_contract,
    )["target_occupancy_space"] == "body_inflated_configuration_space"

    v3_contract = diagnostic_script._resolve_dataset_label_contract(
        {
            "schema": "lewm_go2_paired_navigation_dataset_v3",
            "label_semantics": {
                "label_contract": "observable_physical_occupancy_v3",
                "target_occupancy_space": "observable_physical_occupancy",
                "per_frame_configuration_classes_supervised": False,
                "post_memory_configuration_derivation_is_evaluation_only": True,
            },
        }
    )
    checkpoint_contract = diagnostic_script._validate_checkpoint_dataset_contract(
        {
            "schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v4",
            "occupancy_output_contract": {
                "target_occupancy_space": "observable_physical_occupancy",
                "post_memory_configuration_derivation": {"operation": "inflate"},
            },
        },
        v3_contract,
    )
    assert checkpoint_contract == {
        "checkpoint_schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v4",
        "target_occupancy_space": "observable_physical_occupancy",
    }

    with pytest.raises(ValueError, match="requires checkpoint schema v4"):
        diagnostic_script._validate_checkpoint_dataset_contract(
            {
                "schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v3",
                "occupancy_output_contract": {
                    "target_occupancy_space": "observable_physical_occupancy",
                    "post_memory_configuration_derivation": {"operation": "inflate"},
                },
            },
            v3_contract,
        )

    with pytest.raises(ValueError, match="occupancy target disagrees"):
        diagnostic_script._validate_checkpoint_dataset_contract(
            {
                "schema": "lewm_go2_egomotion_bev_jepa_checkpoint_v4",
                "occupancy_output_contract": {
                    "target_occupancy_space": "body_inflated_configuration_space"
                },
            },
            v3_contract,
        )


def test_v3_report_schema_and_dataset_provenance_must_match(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "model.report.json"
    report = {
        "schema": "lewm_go2_egomotion_bev_jepa_training_report_v3",
        "checkpoint": {"sha256": "c" * 64},
        "dataset_manifest": {"sha256": "d" * 64},
        "final_g2_evaluation": None,
        "promotion": {"g2_evaluated": False},
        "row_counts": {"train": 10, "checkpoint_selection": 10},
        "best_epoch": 1,
        "history": [
            {
                "epoch": 1,
                "train": {
                    "loss": 1.0,
                    "occupancy_loss": 0.75,
                    "jepa_loss": 0.5,
                },
                "checkpoint_selection": _metrics(quality=0.8, passes=False),
            }
        ],
    }
    report_path.write_text(json.dumps(report))
    curve, provenance = diagnostic_script._load_training_curve(
        tmp_path / "model.pt",
        "c" * 64,
        report_path,
        checkpoint_schema="lewm_go2_egomotion_bev_jepa_checkpoint_v3",
        dataset_manifest_sha256="d" * 64,
    )
    assert curve is not None
    assert provenance is not None
    assert provenance["schema"] == (
        "lewm_go2_egomotion_bev_jepa_training_report_v3"
    )

    report["dataset_manifest"]["sha256"] = "0" * 64
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="dataset manifest SHA-256 mismatch"):
        diagnostic_script._load_training_curve(
            tmp_path / "model.pt",
            "c" * 64,
            report_path,
            checkpoint_schema="lewm_go2_egomotion_bev_jepa_checkpoint_v3",
            dataset_manifest_sha256="d" * 64,
        )


def test_source_crop_contract_preserves_legacy_and_rectified_inputs() -> None:
    assert diagnostic_script._source_crop_fraction_xy({}) == (1.0, 1.0)
    assert diagnostic_script._source_crop_fraction_xy(
        {"source_fov_rectification": {"center_crop_fraction_xy": [1.0, 0.75]}}
    ) == (1.0, 0.75)
    with pytest.raises(ValueError, match="crop fractions"):
        diagnostic_script._source_crop_fraction_xy(
            {"source_fov_rectification": {"center_crop_fraction_xy": [1.1, 1.0]}}
        )


def test_diagnostic_rejects_checkpoint_with_stored_g2_output() -> None:
    diagnostic_script._require_untouched_checkpoint(
        {"g2_evaluation": None, "g2_passes": False}
    )
    with pytest.raises(ValueError, match="no stored G2"):
        diagnostic_script._require_untouched_checkpoint(
            {"g2_evaluation": {"g2": {"passes": False}}, "g2_passes": False}
        )


def test_evaluator_uses_hierarchical_weight_interface(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    loader_kwargs: dict = {}

    def fake_loader(*args, **kwargs):
        loader_kwargs.update(kwargs)
        return "loader"

    monkeypatch.setattr(diagnostic_script, "_loader", fake_loader)

    def fake_evaluate(model, loader, **kwargs):
        captured.update(kwargs)
        return {"model": model, "loader": loader}

    monkeypatch.setattr(diagnostic_script, "evaluate_model", fake_evaluate)
    result = diagnostic_script._evaluate_rows(
        object(),
        [{"scene_id": "train"}],
        primitive_to_index={"forward": 0},
        device=torch.device("cpu"),
        unknown_known_weights=torch.tensor([1.0, 1.0]),
        free_occupied_weights=torch.tensor([1.0, 1.0]),
        nominal_delta_table=torch.zeros((1, 3)),
        calibration={
            "method": "test",
            "log_scales": [0.0] * 3,
            "biases": [0.0] * 3,
        },
        thresholds=object(),
        select_thresholds=False,
        image_size=16,
        batch_size=2,
        workers=0,
        seed=1,
        source_crop_fraction_xy=(1.0, 0.75),
        occupancy_target_space="observable_physical_occupancy",
    )
    assert result["loader"] == "loader"
    assert "class_weights" not in captured
    assert torch.equal(captured["unknown_known_weights"], torch.tensor([1.0, 1.0]))
    assert torch.equal(captured["free_occupied_weights"], torch.tensor([1.0, 1.0]))
    assert captured["occupancy_target_space"] == "observable_physical_occupancy"
    assert loader_kwargs["source_crop_fraction_xy"] == (1.0, 0.75)
