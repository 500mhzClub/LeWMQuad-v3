from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from lewm.models.egomotion_bev_jepa import EgomotionBevJepa
from lewm.datasets.go2_paired_navigation import (
    DATASET_ROLES,
    canonical_json_sha256,
    deterministic_family_role_split,
    scene_id_sha256,
)

from scripts.train_go2_egomotion_bev_jepa import (
    _mask_array,
    _observed_mask_array,
    collect_calibration_sample,
    deterministic_row_subset,
    evaluate_model,
    evaluate_g2_gate,
    fit_vector_calibration,
    nominal_primitive_delta_table,
    resolve_dataset_scene_roles,
    split_validation_scenes,
)


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


def test_evaluator_runs_observed_and_changed_matched_panels() -> None:
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
    metrics = evaluate_model(
        model,
        DataLoader(samples, batch_size=2),
        device=torch.device("cpu"),
        class_weights=torch.ones(3),
        nominal_delta_table=torch.tensor(
            [[0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, -0.1]]
        ),
        calibration=None,
        thresholds=None,
        select_thresholds=True,
    )
    assert metrics["predictive_controls"]["panels"]["observed"]["valid_cells"] > 0
    assert metrics["predictive_controls"]["panels"]["changed"]["valid_cells"] > 0
    assert metrics["routing"]["oracle_routable_paths"] > 0
