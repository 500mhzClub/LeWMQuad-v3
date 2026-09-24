from __future__ import annotations

import copy
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

from lewm.benchmarks.go2_physical_spatial_grounding import (
    alignment_accumulators_for_batch,
    canonical_json_sha256,
    deterministic_maximum_mismatch_permutation,
    distance_bin_masks,
    finalize_loss_accumulator,
    finalize_physical_accumulator,
    grounding_contrast,
    loss_accumulator_for_batch,
    physical_accumulator_for_batch,
    visibility_regions,
)
from scripts import diagnose_go2_physical_spatial_grounding as diagnostic


def _perfect_logits(labels: np.ndarray, strength: float = 12.0) -> np.ndarray:
    logits = np.full((labels.shape[0], 3, *labels.shape[1:]), -strength)
    for class_index in range(3):
        logits[:, class_index][labels == class_index] = strength
    return logits


def _zero_g2_ledger() -> dict:
    return {
        "dataset_access_ledger": {
            "g2_contact": {
                "row_metadata_read": True,
                "row_metadata_count": 5,
                "label_shard_byte_opens": 0,
                "image_byte_opens": 0,
                "model_output_rows": 0,
            }
        }
    }


def test_visibility_regions_match_two_3x3_convolution_rings() -> None:
    visible = np.zeros((7, 7), dtype=bool)
    visible[3, 3] = True
    regions = visibility_regions(visible)

    assert regions["center_visible_interior"].sum() == 1
    assert regions["exterior_ring_1"].sum() == 8
    assert regions["exterior_ring_2"].sum() == 16
    assert regions["outside_ring_2"].sum() == 24
    stacked = np.stack(tuple(regions.values()))
    assert np.all(stacked.sum(axis=0) == 1)


def test_distance_bins_are_disjoint_and_cover_grid() -> None:
    distances = np.asarray([[0.0, 0.5, 1.0, 2.0, 3.0, 8.0]])
    masks = distance_bin_masks(distances)
    stacked = np.stack(tuple(masks.values()))

    assert np.all(stacked.sum(axis=0) == 1)
    assert masks["0.0_to_0.5"][0, 0]
    assert masks["0.5_to_1.0"][0, 1]
    assert masks["3.0_plus"][0, -1]


def test_deterministic_global_derangement_changes_unique_images() -> None:
    keys = ("a", "b", "c", "d")
    first = deterministic_maximum_mismatch_permutation(
        keys, seed=17, namespace="train"
    )
    second = deterministic_maximum_mismatch_permutation(
        keys, seed=17, namespace="train"
    )

    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(len(keys)))
    assert all(keys[index] != keys[int(first[index])] for index in range(len(keys)))


def test_derangement_maximizes_changes_when_duplicate_images_exist() -> None:
    keys = ("same", "same", "other")
    indices = deterministic_maximum_mismatch_permutation(keys, seed=1)
    changed = sum(keys[index] != keys[int(indices[index])] for index in range(3))
    assert changed == 2


def test_derangement_adversarial_multiset_achieves_five_of_five() -> None:
    keys = ("a", "a", "b", "c", "b")
    indices = deterministic_maximum_mismatch_permutation(keys, seed=11)
    changed = sum(
        keys[index] != keys[int(indices[index])] for index in range(len(keys))
    )
    assert changed == 5


def test_loss_accumulator_reports_near_zero_for_perfect_logits() -> None:
    labels = np.asarray([[[0, 1], [2, 1]]])
    logits = _perfect_logits(labels)
    accumulator = loss_accumulator_for_batch(
        logits,
        labels,
        np.ones_like(labels, dtype=bool),
        unknown_known_weights=(0.2, 1.8),
        free_occupied_weights=(0.5, 1.5),
    )
    result = finalize_loss_accumulator(accumulator)

    assert result["raw_joint_nll"] < 1e-8
    assert result["raw_hierarchical_balanced_nll"] < 1e-8
    assert result["raw_joint_accuracy"] == 1.0
    assert result["raw_known_free_occupied_accuracy"] == 1.0


def test_physical_accumulator_uses_independent_admission_and_detection() -> None:
    labels = np.asarray([[[1, 2, 0]]])
    probabilities = np.asarray(
        [[[[0.01, 0.02, 0.98]], [[0.98, 0.01, 0.01]], [[0.01, 0.97, 0.01]]]]
    )
    accumulator = physical_accumulator_for_batch(
        probabilities,
        labels,
        np.ones_like(labels, dtype=bool),
        free_probability_min=0.9,
        occupied_probability_max=0.1,
        unknown_probability_max=0.1,
        occupied_detection_min=0.9,
    )
    result = finalize_physical_accumulator(accumulator)

    assert result["admitted_observable_physical_free_precision"] == 1.0
    assert result["useful_observable_physical_free_recall"] == 1.0
    assert result["directly_observable_physical_obstacle_recall"] == 1.0
    assert result["unknown_evidence_admission_rate"] == 0.0


@pytest.mark.parametrize("failure", ["simplex", "nonfinite", "label"])
def test_physical_accumulator_rejects_invalid_calibrated_contract(
    failure: str,
) -> None:
    labels = np.asarray([[[0, 1]]])
    probabilities = np.asarray(
        [[[[0.2, 0.1]], [[0.3, 0.8]], [[0.5, 0.1]]]], dtype=np.float64
    )
    if failure == "simplex":
        probabilities[0, 0, 0, 0] = 0.1
    elif failure == "nonfinite":
        probabilities[0, 0, 0, 0] = np.nan
    else:
        labels[0, 0, 0] = 3
    with pytest.raises(ValueError):
        physical_accumulator_for_batch(
            probabilities,
            labels,
            np.ones_like(labels, dtype=bool),
            free_probability_min=0.9,
            occupied_probability_max=0.1,
            unknown_probability_max=0.1,
            occupied_detection_min=0.9,
        )


def test_alignment_grid_recovers_known_positive_row_shift() -> None:
    labels = np.random.default_rng(41).integers(0, 3, size=(1, 10, 10))
    aligned = _perfect_logits(labels)
    shifted = np.roll(aligned, shift=1, axis=2)
    accumulators = alignment_accumulators_for_batch(
        shifted,
        labels,
        np.ones_like(labels, dtype=bool),
        unknown_known_weights=(1.0, 1.0),
        free_occupied_weights=(1.0, 1.0),
        max_shift=3,
    )
    metrics = {
        name: finalize_loss_accumulator(value) for name, value in accumulators.items()
    }
    best = min(metrics, key=lambda name: metrics[name]["raw_hierarchical_balanced_nll"])

    assert best == "shift_row_+1_col_+0"
    assert metrics[best]["raw_hierarchical_balanced_nll"] < 1e-8
    assert metrics["identity"]["raw_hierarchical_balanced_nll"] > 1.0


def test_alignment_grid_recovers_horizontal_flip() -> None:
    labels = np.random.default_rng(73).integers(0, 3, size=(1, 10, 10))
    flipped_logits = _perfect_logits(labels)[:, :, :, ::-1]
    accumulators = alignment_accumulators_for_batch(
        flipped_logits,
        labels,
        np.ones_like(labels, dtype=bool),
        unknown_known_weights=(1.0, 1.0),
        free_occupied_weights=(1.0, 1.0),
        max_shift=3,
    )
    metrics = {
        name: finalize_loss_accumulator(value) for name, value in accumulators.items()
    }

    assert metrics["horizontal_flip"]["raw_hierarchical_balanced_nll"] < 1e-8
    assert metrics["identity"]["raw_hierarchical_balanced_nll"] > 1.0


def test_grounding_contrast_uses_loss_increase_sign() -> None:
    conditions = {
        "correct_rgb": {
            "raw_joint_nll": 0.1,
            "raw_hierarchical_balanced_nll": 0.2,
            "raw_known_free_occupied_nll": 0.3,
        },
        "role_global_shuffled_rgb": {
            "raw_joint_nll": 0.4,
            "raw_hierarchical_balanced_nll": 0.6,
            "raw_known_free_occupied_nll": 0.8,
        },
        "per_channel_mean_rgb": {
            "raw_joint_nll": 0.5,
            "raw_hierarchical_balanced_nll": 0.7,
            "raw_known_free_occupied_nll": 0.9,
        },
    }
    result = grounding_contrast(conditions)
    assert result["role_global_shuffled_rgb"][
        "role_global_shuffled_rgb_minus_correct_raw_hierarchical_balanced_nll"
    ] == pytest.approx(0.4)


def test_frame_record_selection_never_materializes_forbidden_role_paths() -> None:
    assignments = {
        "train_scene": "train",
        "selection_scene": "checkpoint_selection",
        "calibration_scene": "probability_calibration",
        "g2_scene": "g2_evaluation",
    }

    def row(scene: str, role: str, global_row: int) -> dict:
        base = {
            "scene_id": scene,
            "dataset_role": role,
            "global_row": global_row,
            "family": "family",
        }
        if role in {"train", "checkpoint_selection"}:
            base.update(
                {
                    "current_image_path": f"/{scene}_current.png",
                    "current_image_sha256": "a" * 64,
                    "next_image_path": f"/{scene}_next.png",
                    "next_image_sha256": "b" * 64,
                    "label_shard_path": f"/{scene}.npz",
                    "label_shard_sha256": "c" * 64,
                    "label_shard_row": 0,
                }
            )
        else:
            base["forbidden_artifact_path"] = f"/{scene}.npz"
        return base

    rows = [row(scene, role, index) for index, (scene, role) in enumerate(assignments.items())]
    selected, counts = diagnostic._frame_records(rows, assignments)

    assert counts == {
        "checkpoint_selection": 1,
        "g2_evaluation": 1,
        "probability_calibration": 1,
        "train": 1,
    }
    serialized = repr(selected)
    assert "g2_scene" not in serialized
    assert "calibration_scene" not in serialized
    assert len(selected["train"]) == 2
    assert len(selected["checkpoint_selection"]) == 2


def _control_record(index: int, scene: str, image_hash: str) -> dict:
    return {
        "scene_id": scene,
        "global_row": index,
        "side": "current",
        "image_path": f"/{image_hash}.png",
        "image_sha256": image_hash,
    }


def test_role_global_control_is_cross_scene_image_and_transition() -> None:
    records = [
        _control_record(0, "scene_a", "image_a0"),
        _control_record(1, "scene_a", "image_a1"),
        _control_record(2, "scene_b", "image_b0"),
        _control_record(3, "scene_c", "image_c0"),
    ]
    report = diagnostic._attach_role_global_controls(
        records, role="train", seed=101
    )

    assert report["record_count"] == 4
    assert report["image"]["same_hash_pair_count"] == 0
    assert report["scene"]["same_scene_pair_count"] == 0
    assert report["transition"]["same_transition_pair_count"] == 0
    for record in records:
        assert record["image_sha256"] != record["control_image_sha256"]
        assert record["scene_id"] != record["control_scene_id"]
        assert (record["scene_id"], record["global_row"]) != (
            record["control_scene_id"],
            record["control_global_row"],
        )


def test_role_global_control_exact_fallback_finds_joint_assignment() -> None:
    records = [
        _control_record(0, "scene_a", "image_x"),
        _control_record(1, "scene_a", "image_y"),
        _control_record(2, "scene_b", "image_y"),
        _control_record(3, "scene_b", "image_x"),
    ]
    report = diagnostic._attach_role_global_controls(records, role="train", seed=1)
    assert report["assignment_method"] == "exact_joint_bipartite_matching_fallback"
    assert report["image"]["same_hash_pair_count"] == 0
    assert report["scene"]["same_scene_pair_count"] == 0


@pytest.mark.parametrize(
    "records",
    [
        [
            _control_record(0, "scene_a", "image_0"),
            _control_record(1, "scene_a", "image_1"),
            _control_record(2, "scene_a", "image_2"),
            _control_record(3, "scene_b", "image_3"),
        ],
        [
            _control_record(0, "scene_a", "same"),
            _control_record(1, "scene_a", "same"),
            _control_record(2, "scene_b", "same"),
            _control_record(3, "scene_b", "different"),
        ],
    ],
)
def test_role_global_control_rejects_impossible_zero_match_dataset(
    records: list[dict],
) -> None:
    with pytest.raises(ValueError, match="cannot support zero-pair"):
        diagnostic._attach_role_global_controls(records, role="train", seed=101)


def test_checkpoint_and_report_reject_any_stored_g2_output() -> None:
    checkpoint = {
        "schema": diagnostic.CHECKPOINT_SCHEMA,
        "g2_evaluation": None,
        "g2_passes": False,
        "head_g2_evaluation": None,
        "head_g2_passes": False,
        "runtime_ready": False,
        **_zero_g2_ledger(),
    }
    diagnostic._require_no_g2_checkpoint(checkpoint)
    touched_checkpoint = copy.deepcopy(checkpoint)
    touched_checkpoint["head_g2_evaluation"] = {"g2": {"passes": False}}
    with pytest.raises(ValueError, match="already contains"):
        diagnostic._require_no_g2_checkpoint(touched_checkpoint)

    report = {
        "schema": diagnostic.REPORT_SCHEMA,
        "final_g2_evaluation": None,
        "final_head_g2_evaluation": None,
        "promotion": {"head_g2_evaluated": False, "head_g2_passes": False},
        "row_counts": {"g2_evaluation": 0},
        **_zero_g2_ledger(),
    }
    diagnostic._require_no_g2_report(report)
    touched_report = copy.deepcopy(report)
    touched_report["dataset_access_ledger"]["g2_contact"]["image_byte_opens"] = 1
    with pytest.raises(ValueError, match="forbidden G2 access"):
        diagnostic._require_no_g2_report(touched_report)


def test_dataset_manifest_role_hash_and_physical_contract_are_required() -> None:
    assignments = {"scene": "train"}
    manifest = {
        "schema": diagnostic.DATASET_SCHEMA,
        "label_semantics": {
            "label_contract": diagnostic.LABEL_CONTRACT,
            "target_occupancy_space": diagnostic.TARGET_SPACE,
            "per_frame_configuration_classes_supervised": False,
            "post_memory_configuration_derivation_is_evaluation_only": True,
        },
        "scene_roles": {
            "assignments": assignments,
            "assignments_sha256": canonical_json_sha256(assignments),
        },
    }
    assert diagnostic._validate_dataset_manifest(manifest) == assignments
    broken = copy.deepcopy(manifest)
    broken["scene_roles"]["assignments_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        diagnostic._validate_dataset_manifest(broken)


def test_development_only_flag_is_mandatory(tmp_path: Path) -> None:
    common = [
        "--dataset-manifest",
        str(tmp_path / "dataset.json"),
        "--expected-dataset-manifest-sha256",
        "a" * 64,
        "--checkpoint",
        str(tmp_path / "model.pt"),
        "--expected-checkpoint-sha256",
        "b" * 64,
        "--training-report",
        str(tmp_path / "report.json"),
        "--expected-training-report-sha256",
        "c" * 64,
        "--output",
        str(tmp_path / "output.json"),
    ]
    with pytest.raises(SystemExit):
        diagnostic._parse_args(common)
    parsed = diagnostic._parse_args([*common, "--development-only"])
    assert parsed.development_only is True


def test_precommitted_hash_mismatch_precedes_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "dataset.json"
    checkpoint = tmp_path / "model.pt"
    report = tmp_path / "report.json"
    manifest.write_text("{}\n")
    checkpoint.write_bytes(b"not a checkpoint")
    report.write_text("{}\n")
    calls: list[str] = []
    monkeypatch.setattr(
        diagnostic,
        "_read_json",
        lambda _path: calls.append("json") or (_ for _ in ()).throw(AssertionError()),
    )
    monkeypatch.setattr(
        diagnostic,
        "_load_checkpoint",
        lambda _path: calls.append("torch") or (_ for _ in ()).throw(AssertionError()),
    )

    with pytest.raises(ValueError, match="precommitted"):
        diagnostic.main(
            [
                "--dataset-manifest",
                str(manifest),
                "--expected-dataset-manifest-sha256",
                "0" * 64,
                "--checkpoint",
                str(checkpoint),
                "--expected-checkpoint-sha256",
                diagnostic._sha256_file(checkpoint),
                "--training-report",
                str(report),
                "--expected-training-report-sha256",
                diagnostic._sha256_file(report),
                "--output",
                str(tmp_path / "output.json"),
                "--development-only",
            ]
        )
    assert calls == []


def _training_source_checkpoint(*, status_short: str = "") -> dict:
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=diagnostic.REPOSITORY_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    critical = {
        name: {"path": str(path.resolve()), "sha256": diagnostic._sha256_file(path)}
        for name, path in diagnostic.TRAINING_CRITICAL_SOURCE_PATHS.items()
    }
    core = {
        "schema": "lewm_go2_training_run_provenance_v1",
        "git": {"head": head, "status_short": status_short},
        "critical_inputs": critical,
        "checkpoint_artifact_included": False,
    }
    return {
        "training_run_provenance": {
            **core,
            "content_sha256": canonical_json_sha256(core),
        }
    }


def test_training_source_provenance_uses_clean_head_for_legacy_encoder() -> None:
    checkpoint = _training_source_checkpoint()
    result = diagnostic._validate_training_source_provenance(checkpoint)
    assert result["encoder_source"]["mode"].startswith("clean_git_blob")
    assert set(result["critical_sources"]) == set(
        diagnostic.TRAINING_CRITICAL_SOURCE_PATHS
    )


def test_training_source_provenance_rejects_changed_or_dirty_source() -> None:
    checkpoint = _training_source_checkpoint()
    checkpoint["training_run_provenance"]["critical_inputs"]["model_source"][
        "sha256"
    ] = "0" * 64
    core = dict(checkpoint["training_run_provenance"])
    core.pop("content_sha256")
    checkpoint["training_run_provenance"]["content_sha256"] = canonical_json_sha256(core)
    with pytest.raises(ValueError, match="model_source"):
        diagnostic._validate_training_source_provenance(checkpoint)
    allowed = diagnostic._validate_training_source_provenance(
        checkpoint,
        allowed_counterfactual_source_changes=frozenset({"model_source"}),
    )
    assert allowed["critical_sources"]["model_source"][
        "counterfactual_source_transition_allowed"
    ] is True

    dirty = _training_source_checkpoint(
        status_short=" M lewm/models/encoders.py"
    )
    with pytest.raises(ValueError, match="dirty"):
        diagnostic._validate_training_source_provenance(dirty)


def _complete_condition_metrics() -> dict:
    return {
        "raw_joint_nll": 0.3,
        "raw_hierarchical_balanced_nll": 0.4,
        "raw_unknown_known_weighted_nll": 0.2,
        "raw_known_free_occupied_weighted_nll": 0.6,
        "raw_known_free_occupied_nll": 0.5,
        "cell_count": 7,
        "known_cell_count": 5,
        "frozen_calibrated_physical": {
            "true_free": 3,
            "true_occupied": 2,
            "true_unknown": 2,
        },
    }


def test_paired_condition_support_is_nonnull_and_equal() -> None:
    conditions = {
        condition: _complete_condition_metrics() for condition in diagnostic.CONDITIONS
    }
    result = diagnostic._validate_paired_condition_support(
        conditions, context="train/scene/example"
    )
    assert result["supports_equal"] is True
    assert result["cell_count"] == 7

    conditions["role_global_shuffled_rgb"]["cell_count"] = 8
    with pytest.raises(ValueError, match="support"):
        diagnostic._validate_paired_condition_support(
            conditions, context="train/scene/example"
        )


def test_alignment_support_is_nonnull_and_equal() -> None:
    base = _complete_condition_metrics()
    base.pop("frozen_calibrated_physical")
    transforms = {"identity": dict(base), "shift_row_+1_col_+0": dict(base)}
    assert diagnostic._validate_alignment_support(
        transforms, context="train/alignment"
    )["supports_equal"]
    transforms["shift_row_+1_col_+0"]["raw_joint_nll"] = None
    with pytest.raises(ValueError, match="finite"):
        diagnostic._validate_alignment_support(transforms, context="train/alignment")


def test_projective_geometry_record_is_canonical_and_sensitive() -> None:
    bias = torch.tensor([[[0.0, -1.0], [-2.0, -3.0]]], dtype=torch.float32)
    visibility = torch.tensor([[[True, True], [False, True]]])
    first = diagnostic._projective_geometry_buffer_record(bias, visibility)
    second = diagnostic._projective_geometry_buffer_record(bias.clone(), visibility.clone())
    changed = diagnostic._projective_geometry_buffer_record(
        bias + torch.tensor(0.25), visibility
    )
    assert first == second
    assert first["content_sha256"] != changed["content_sha256"]


def _physical_support_manifest() -> dict:
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
        "schema": diagnostic.DATASET_SCHEMA,
        "local_grid": {"cell_size_m": 0.1},
        "label_semantics": {
            "label_contract": diagnostic.LABEL_CONTRACT,
            "target_occupancy_space": diagnostic.TARGET_SPACE,
            "per_frame_configuration_classes_supervised": False,
            "physical_aggregation": aggregation,
        },
    }


def test_frozen_cell_square_counterfactual_changes_only_geometry_contract() -> None:
    checkpoint = {
        "model_config": {
            "bev_lift_type": diagnostic.PROJECTIVE_COLUMN_ATTENTION_LIFT,
        },
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "occupancy_output_contract": {},
    }
    config, support, record = diagnostic._frozen_cell_square_counterfactual(
        checkpoint,
        _physical_support_manifest(),
    )

    assert config["bev_lift_type"] == (
        diagnostic.PROJECTIVE_CELL_SQUARE_ATTENTION_LIFT
    )
    assert config["projective_output_cell_size_m"] == pytest.approx(0.1)
    assert support["uses_body_footprint"] is False
    assert record["learned_state_unchanged"] is True
    assert record["checkpoint_emitted_or_mutated"] is False
    core = dict(record)
    assert core.pop("content_sha256") == canonical_json_sha256(core)


def test_geometry_contract_file_hash_is_verified(tmp_path: Path) -> None:
    contract = tmp_path / "geometry.json"
    contract.write_text("{}\n")
    actual = diagnostic._sha256_file(contract)
    path, digest = diagnostic._validated_geometry_contract(
        {"geometry_contract": {"path": str(contract), "file_sha256": actual}}
    )
    assert path == contract.resolve()
    assert digest == actual

    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        diagnostic._validated_geometry_contract(
            {"geometry_contract": {"path": str(contract), "file_sha256": "0" * 64}}
        )


def test_spatial_diagnostic_requires_disjoint_thresholds() -> None:
    valid = {
        "traversability_thresholds": {
            "free_probability_min": 0.5,
            "occupied_probability_max": 0.01,
            "unknown_probability_max": 0.05,
            "occupied_detection_min": 0.02,
        }
    }
    assert (
        diagnostic._validated_traversability_thresholds(valid).occupied_detection_min
        == 0.02
    )

    overlapping = copy.deepcopy(valid)
    overlapping["traversability_thresholds"]["occupied_detection_min"] = 0.01
    with pytest.raises(ValueError, match="intervals overlap"):
        diagnostic._validated_traversability_thresholds(overlapping)
