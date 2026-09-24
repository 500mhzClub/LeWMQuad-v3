from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as calibration
from lewm.benchmarks.go2_world_model_counterfactual_pilot_v1 import FAMILIES
from lewm.benchmarks.go2_dinov2_physical_readout_calibration_v1 import (
    DESCRIPTOR_DIMENSION,
    INFRASTRUCTURE_FAILURE_STATUS,
    PASS_STATUS,
    SCHEMA,
    STOP_STATUS,
    DINOv2PhysicalReadoutCalibrationError,
    RoleFeaturePlanV1,
    RoleStateIndexV1,
    build_calibration_feature_plans_v1,
    build_role_feature_plan_v1,
    calibration_replay_identity_v1,
    calibration_verdict_v1,
    dinov2_quadrant_descriptor_v1,
    fit_calibration_readouts_v1,
    paired_family_scene_cluster_comparison_v1,
    relational_descriptor_v1,
)


def _groups(role: str, *, reverse: bool = False) -> list[SimpleNamespace]:
    role_offset = 0 if role == "train" else 128
    result = []
    for local_index in range(128):
        family_index = local_index // 16
        within_family = local_index % 16
        scene_index = within_family // 8
        state_index = within_family % 8
        family = FAMILIES[family_index]
        prefix = f"{role}_{local_index:03d}"
        labels = [
            SimpleNamespace(
                target_progress_m=float(8 - action),
                path_length_m=1.0,
                fell=False,
                tipped=False,
                planar_clearance_proxy_min_m=None,
                grid_recoverability_proxy=None,
            )
            for action in range(9)
        ]
        result.append(
            SimpleNamespace(
                role=role,
                state_id=f"state_{prefix}",
                family=family,
                scene_id=f"{role}_{family}_scene_{scene_index}",
                group_index=role_offset + local_index,
                state_index_in_scene=state_index,
                relative_target_xy_body_m=(float(state_index - 4), float(scene_index)),
                context_rgb_artifact_ids=tuple(
                    f"{prefix}_context_{index}" for index in range(3)
                ),
                branches=tuple(
                    SimpleNamespace(
                        action_id=action,
                        target_rgb_artifact_id=f"{prefix}_target_{action}",
                        oracle_dense_rank=action,
                        labels=labels[action],
                    )
                    for action in range(9)
                ),
            )
        )
    return list(reversed(result)) if reverse else result


def test_exact_quadrant_descriptor_is_float64_row_major_and_ddof_zero() -> None:
    grid = np.empty((16, 16, 384), dtype=np.float16)
    grid[:8, :8] = 1.0
    grid[:8, 8:] = 2.0
    grid[8:, :8] = 3.0
    grid[8:, 8:] = 4.0
    descriptor = dinov2_quadrant_descriptor_v1(grid.reshape(256, 384))
    assert descriptor.dtype == np.float64
    assert descriptor.shape == (DESCRIPTOR_DIMENSION,)
    for quadrant, value in enumerate((1.0, 2.0, 3.0, 4.0)):
        start = quadrant * 768
        np.testing.assert_array_equal(descriptor[start : start + 384], value)
        np.testing.assert_array_equal(descriptor[start + 384 : start + 768], 0.0)

    with pytest.raises(DINOv2PhysicalReadoutCalibrationError, match="float16"):
        dinov2_quadrant_descriptor_v1(grid.astype(np.float32).reshape(256, 384))
    changed = grid.copy().reshape(256, 384)
    changed[0, 0] = np.nan
    with pytest.raises(DINOv2PhysicalReadoutCalibrationError, match="finite"):
        dinov2_quadrant_descriptor_v1(changed)


def test_relational_descriptor_is_current_successor_and_delta() -> None:
    current = np.arange(DESCRIPTOR_DIMENSION, dtype=np.float64)
    successor = current + 2.0
    relation = relational_descriptor_v1(current, successor)
    assert relation.shape == (3 * DESCRIPTOR_DIMENSION,)
    np.testing.assert_array_equal(relation[:DESCRIPTOR_DIMENSION], current)
    np.testing.assert_array_equal(
        relation[DESCRIPTOR_DIMENSION : 2 * DESCRIPTOR_DIMENSION], successor
    )
    np.testing.assert_array_equal(relation[2 * DESCRIPTOR_DIMENSION :], 2.0)


def test_role_plans_are_metadata_only_ordered_and_role_disjoint() -> None:
    reversed_train = _groups("train", reverse=True)
    train = build_role_feature_plan_v1(reversed_train, role="train")
    replay = build_role_feature_plan_v1(list(reversed(reversed_train)), role="train")
    assert train.identity_sha256 == replay.identity_sha256
    assert len(train.states) == 128
    assert len(train.artifact_ids) == 1_536
    assert train.states[0].group_index == 0
    assert train.artifact_ids[:3] == (
        "train_000_context_0",
        "train_000_context_1",
        "train_000_context_2",
    )
    assert train.artifact_ids[3:12] == tuple(
        f"train_000_target_{action}" for action in range(9)
    )

    plans = build_calibration_feature_plans_v1(reversed_train, _groups("eval"))
    assert plans.train.identity_sha256 == train.identity_sha256
    assert not set(plans.train.artifact_ids) & set(plans.eval.artifact_ids)
    assert not {state.scene_id for state in plans.train.states} & {
        state.scene_id for state in plans.eval.states
    }


def _small_train_plan() -> tuple[RoleFeaturePlanV1, list[np.ndarray]]:
    groups = _groups("train")[:2]
    states = []
    artifacts: list[str] = []
    features: list[np.ndarray] = []
    for index, group in enumerate(groups):
        base = len(artifacts)
        artifacts.extend([*group.context_rgb_artifact_ids, *(branch.target_rgb_artifact_id for branch in group.branches)])
        for slot in range(12):
            token = np.full((256, 384), index * 16 + slot + 1, dtype=np.float16)
            features.append(token)
        states.append(
            RoleStateIndexV1(
                role_state_index=index,
                state_id=group.state_id,
                role="train",
                family=group.family,
                scene_id=group.scene_id,
                group_index=group.group_index,
                state_index_in_scene=group.state_index_in_scene,
                relative_target_xy_body_m=group.relative_target_xy_body_m,
                context_artifact_indices=(base, base + 1, base + 2),
                target_artifact_indices=tuple(range(base + 3, base + 12)),
                dense_ranks=tuple(branch.oracle_dense_rank for branch in group.branches),
                target_progress_m=tuple(branch.labels.target_progress_m for branch in group.branches),
                physical_fell=(False,) * 9,
                physical_tipped=(False,) * 9,
            )
        )
    plan = RoleFeaturePlanV1(
        role="train",
        artifact_ids=tuple(artifacts),
        artifact_index_by_id=MappingProxyType({value: index for index, value in enumerate(artifacts)}),
        states=tuple(states),
        groups=tuple(groups),
        identity_sha256="0" * 64,
    )
    return plan, features


def test_three_readout_sets_are_separate_deterministic_and_exactly_dimensioned() -> None:
    plan, features = _small_train_plan()
    first = fit_calibration_readouts_v1(plan, features)
    second = fit_calibration_readouts_v1(plan, features)
    assert first.identity_sha256 == second.identity_sha256
    assert first.relational.identity_sha256 != first.current_state.identity_sha256
    assert first.relational.heads[0].feature_mean.size == 27_651
    assert first.current_state.heads[0].feature_mean.size == 9_219
    assert first.task_action_only.heads[0].feature_mean.size == 3
    assert all(head.solver == "dual" for head in first.relational.heads)
    # This deliberately tiny two-row unit fixture selects the helper's dual
    # branch even for the three task coordinates; the fixed 128-row run is
    # primal for this control.
    assert all(head.solver == "dual" for head in first.task_action_only.heads)


def _paired_rows(delta: float) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidate = []
    baseline = []
    for family in FAMILIES:
        for scene_index in range(2):
            for state_index in range(8):
                common = {
                    "state_id": f"{family}_{scene_index}_{state_index}",
                    "family": family,
                    "scene_id": f"{family}_scene_{scene_index}",
                }
                baseline.append({**common, "normalized_rank_regret": 0.5})
                candidate.append({**common, "normalized_rank_regret": 0.5 + delta})
    return candidate, baseline


def test_family_equal_scene_bootstrap_is_paired_seeded_and_strictly_oriented() -> None:
    candidate, baseline = _paired_rows(-0.25)
    first = paired_family_scene_cluster_comparison_v1(candidate, baseline)
    second = paired_family_scene_cluster_comparison_v1(candidate, baseline)
    assert first == second
    assert first["paired_states"] == 128
    assert first["scene_clusters"] == 16
    assert first["family_strata"] == 8
    assert first["mean_delta"] == pytest.approx(-0.25)
    assert first["lower_95"] == pytest.approx(-0.25)
    assert first["upper_95"] == pytest.approx(-0.25)


def test_family_equal_scene_bootstrap_handles_heterogeneous_scene_effects() -> None:
    candidate = []
    baseline = []
    for family_index, family in enumerate(FAMILIES):
        for scene_index in range(2):
            delta = family_index * 0.01 + scene_index * 0.10
            for state_index in range(8):
                common = {
                    "state_id": f"{family}_{scene_index}_{state_index}",
                    "family": family,
                    "scene_id": f"{family}_scene_{scene_index}",
                }
                baseline.append({**common, "normalized_rank_regret": 0.25})
                candidate.append(
                    {**common, "normalized_rank_regret": 0.25 + delta}
                )
    first = paired_family_scene_cluster_comparison_v1(candidate, baseline)
    second = paired_family_scene_cluster_comparison_v1(candidate, baseline)
    assert first == second
    assert first["mean_delta"] == pytest.approx(0.085)
    assert first["lower_95"] < first["mean_delta"] < first["upper_95"]


def test_score_arm_wiring_uses_separate_heads_and_same_head_persistence(
    monkeypatch,
) -> None:
    plan = build_role_feature_plan_v1(_groups("eval"), role="eval")
    relational_heads = object()
    current_heads = object()
    task_heads = object()
    readouts = SimpleNamespace(
        relational=relational_heads,
        current_state=current_heads,
        task_action_only=task_heads,
    )
    monkeypatch.setattr(
        calibration,
        "_descriptor_cache",
        lambda selected_plan, _features: tuple(
            range(len(selected_plan.artifact_ids))
        ),
    )
    monkeypatch.setattr(
        calibration,
        "relational_descriptor_v1",
        lambda current, successor: ("relational", current, successor),
    )
    monkeypatch.setattr(
        calibration,
        "task_conditioned_feature_v1",
        lambda latent, *, relative_target_xy_body_m: (
            latent,
            tuple(relative_target_xy_body_m),
        ),
    )

    def predict(heads, features):
        if heads is current_heads:
            selected = 2
        elif heads is task_heads:
            selected = 3
        elif heads is relational_heads:
            relation = features[0][0]
            selected = 8 if relation[1] == relation[2] else 0
        else:  # pragma: no cover - a new unregistered readout set is a failure
            raise AssertionError("unexpected readout set")
        scores = np.ones(9, dtype=np.float64)
        scores[selected] = 0.0
        return scores

    monkeypatch.setattr(calibration, "predict_action_specific_scores_v1", predict)
    arms = calibration.score_calibration_arms_v1(plan, (), readouts)
    assert arms["privileged_physical_oracle"]["summary"][
        "normalized_rank_regret"
    ] == 0.0
    assert arms["dinov2_true_future"]["summary"][
        "chosen_action_histogram"
    ]["0"] == 128
    assert arms["dinov2_current_state"]["summary"][
        "chosen_action_histogram"
    ]["2"] == 128
    assert arms["task_action_only"]["summary"]["chosen_action_histogram"][
        "3"
    ] == 128
    assert arms["relational_persistence"]["summary"][
        "chosen_action_histogram"
    ]["8"] == 128
    assert arms["hold_constant"]["summary"]["chosen_action_histogram"]["6"] == 128
    assert arms["random_expected"]["summary"]["chosen_action_histogram"] == (
        "NOT_APPLICABLE"
    )
    assert "physical_path_length_m" in arms["dinov2_true_future"]["per_family"][
        FAMILIES[0]
    ]


def test_evaluate_gate_wiring_requires_every_registered_control(monkeypatch) -> None:
    plans = SimpleNamespace(
        identity_sha256="1" * 64,
        train=SimpleNamespace(identity_sha256="2" * 64),
        eval=SimpleNamespace(identity_sha256="3" * 64),
    )
    monkeypatch.setattr(
        calibration, "build_calibration_feature_plans_v1", lambda *_args: plans
    )
    monkeypatch.setattr(
        calibration, "fit_calibration_readouts_v1", lambda *_args: object()
    )
    monkeypatch.setattr(
        calibration,
        "_readout_report",
        lambda _readouts: {"identity_sha256": "4" * 64},
    )
    monkeypatch.setattr(
        calibration,
        "_safety_support",
        lambda _plans: {"status": calibration.SAFETY_STATUS, "passed": False},
    )

    def arm(regret: float, *, oracle_rate: float = 0.5):
        rows = []
        for family in FAMILIES:
            for scene_index in range(2):
                for state_index in range(8):
                    rows.append(
                        {
                            "state_id": f"{family}_{scene_index}_{state_index}",
                            "family": family,
                            "scene_id": f"{family}_scene_{scene_index}",
                            "normalized_rank_regret": regret,
                        }
                    )
        return {
            "summary": {
                "normalized_rank_regret": regret,
                "oracle_equivalent_selection_rate": oracle_rate,
            },
            "group_results": rows,
            "per_family": {
                family: {"normalized_rank_regret": regret} for family in FAMILIES
            },
        }

    arms = {
        "privileged_physical_oracle": arm(0.0, oracle_rate=1.0),
        "dinov2_true_future": arm(0.10),
        "dinov2_current_state": arm(0.30),
        "task_action_only": arm(0.40),
        "relational_persistence": arm(0.20),
        "random_expected": arm(0.50),
        "hold_constant": arm(0.60),
    }
    monkeypatch.setattr(
        calibration, "score_calibration_arms_v1", lambda *_args: arms
    )
    result = calibration.evaluate_calibration_v1((), (), (), ())
    assert result["scientific_gates_2_to_6_passed"] is True
    assert all(gate["passed"] is True for gate in result["gates"].values())
    assert result["paired_scene_cluster_comparisons"][
        "true_future_vs_relational_persistence"
    ]["upper_95"] < 0.0
    assert result["safety"]["passed"] is False


def test_replay_identity_and_seven_gate_verdict_statuses_are_deterministic() -> None:
    scientific_gates = {
        name: {"passed": True} for name in calibration.SCIENTIFIC_GATE_NAMES
    }
    evaluation = {
        "schema": SCHEMA,
        "gates": scientific_gates,
        "scientific_gates_2_to_6_passed": True,
        "value": 1.0,
    }
    assert calibration_replay_identity_v1(evaluation) == calibration_replay_identity_v1(
        dict(evaluation)
    )
    passed = calibration_verdict_v1(
        evaluation,
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )
    assert passed["passed"]
    assert passed["terminal_status"] == PASS_STATUS

    stopped_evaluation = {
        **evaluation,
        "gates": {
            **scientific_gates,
            "3_true_future_beats_task_action_only": {"passed": False},
        },
        "scientific_gates_2_to_6_passed": False,
    }
    stopped = calibration_verdict_v1(
        stopped_evaluation,
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )
    assert not stopped["passed"]
    assert stopped["terminal_status"] == STOP_STATUS

    failed = calibration_verdict_v1(
        evaluation,
        infrastructure_checks_passed=False,
        deterministic_replay_passed=True,
    )
    assert failed["terminal_status"] == INFRASTRUCTURE_FAILURE_STATUS

    missing_gate = {
        **evaluation,
        "gates": {
            name: gate
            for name, gate in scientific_gates.items()
            if name != "6_true_future_beats_random_expected"
        },
    }
    with pytest.raises(
        DINOv2PhysicalReadoutCalibrationError,
        match="registered scientific gates changed",
    ):
        calibration_verdict_v1(
            missing_gate,
            infrastructure_checks_passed=True,
            deterministic_replay_passed=True,
        )
