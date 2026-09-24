from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

from lewm.benchmarks import go2_matched_branch_physical_outcome_screen_v1 as subject
from lewm.models.go2_matched_branch_physical_outcome_screen_v1 import (
    PARAMETER_COUNT,
    initialize_physical_outcome_mlp_v1,
    physical_outcome_state_identity_v1,
)


def _pose(x: float, y: float, yaw: float) -> dict[str, list[float]]:
    return {
        "position_xyz_m": [x, y, 0.4],
        "quaternion_wxyz": [
            float(np.cos(yaw / 2.0)),
            0.0,
            0.0,
            float(np.sin(yaw / 2.0)),
        ],
    }


def _plan(states: int, *, role: str = "train") -> SimpleNamespace:
    rows = tuple(
        SimpleNamespace(
            state_id=f"{role}-{index}",
            context_artifact_indices=(index * 12, index * 12 + 1, index * 12 + 2),
            relative_target_xy_body_m=(1.0, 0.0),
            dense_ranks=tuple(range(9)),
        )
        for index in range(states)
    )
    return SimpleNamespace(role=role, states=rows, identity_sha256="a" * 64)


def _role_data(
    states: int,
    *,
    role: str = "train",
    physical: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    pooled: torch.Tensor | None = None,
) -> subject.RolePhysicalDataV1:
    return subject.RolePhysicalDataV1(
        role=role,
        plan=_plan(states, role=role),
        physical_inputs=(
            physical
            if physical is not None
            else torch.zeros(states, 9, 12, dtype=torch.float32)
        ),
        targets=(
            targets
            if targets is not None
            else torch.zeros(states, 9, 4, dtype=torch.float32)
        ),
        pooled_context=(
            pooled
            if pooled is not None
            else torch.zeros(states, subject.PCA_INPUT_DIMENSION, dtype=torch.float64)
        ),
        identity_sha256="b" * 64,
    )


def _receipt() -> dict[str, object]:
    poses = [_pose(0.0, 0.0, 0.0), _pose(0.1, 0.0, 0.1), _pose(0.2, 0.0, 0.2)]
    branches = []
    for action_id, command in enumerate(subject.EXPECTED_ACTION_COMMANDS):
        branches.append(
            {
                "action_id": action_id,
                "requested_block": [list(command) for _ in range(5)],
                "executed_block": object(),
                "endpoint_state": {
                    "base_pos_world": [0.3 + action_id * 0.01, 0.02, 0.4],
                    "base_quat_wxyz": _pose(0.0, 0.0, 0.25)["quaternion_wxyz"],
                },
                "physical_path_length_m": 0.2 + action_id * 0.01,
                "physical_target_progress_m": 0.1 - action_id * 0.01,
                "physical_fell": False,
                "physical_tipped": False,
                "frame_receipt": {"artifact_id": f"train-0-target-{action_id}"},
            }
        )
    return {
        "state": {
            "role": "train",
            "state_id": "train-0",
            "family": "large_enclosed_maze",
            "scene_id": "train-scene-0",
            "group_index": 0,
            "state_index_in_scene": 0,
        },
        "context": {
            "rgb_artifact_ids": [f"train-0-context-{index}" for index in range(3)],
            "target_relative_body_xy_m": [1.0, 0.0],
            "context_base_pose_world_sequence": poses,
            "history_executed_blocks": [
                [[0.1, 99.0, 0.2] for _ in range(5)],
                [[0.3, -99.0, -0.4] for _ in range(5)],
            ],
        },
        "branches": branches,
    }


def test_fixed_configuration_matches_preregistered_mechanism_and_schedule() -> None:
    config = subject.config_v1()
    assert PARAMETER_COUNT == 532
    assert config["model"]["architecture"] == [28, 16, 4]
    assert config["model"]["seeds"] == [2_026_080_311, 2_026_080_312, 2_026_080_313]
    assert config["training"]["updates_per_member"] == 1_024
    assert config["training"]["batch_states"] == 16
    assert config["training"]["batches_per_seed_local_permutation"] == 8
    assert config["training"]["seed_local_permutations"] == 128
    assert config["training"]["same_schedule_and_initial_state_for_matched_b_c"] is True
    assert config["visual_projection"]["pooled_grid"] == [4, 4, 384]
    assert config["visual_projection"]["pca_dimension"] == 16
    assert config["bootstrap_resamples"] == 10_000
    assert config["bootstrap_seed"] == 2_026_080_314


def test_access_accounting_separates_validation_from_model_input() -> None:
    accounting = subject.access_accounting_v1()
    assert accounting == {
        "rgb_leaf_opens": 0,
        "encoder_executions": 0,
        "target_or_successor_token_grids_used_as_model_input": 0,
        "target_or_successor_token_grids_validation_only": 2_304,
        "protected_material_accessed": False,
    }


def test_body_local_pose_increment_and_wrapped_yaw_are_exact() -> None:
    result = subject.body_local_increment_v1(
        _pose(2.0, 3.0, np.pi / 2.0),
        _pose(2.0, 4.0, -np.pi + 0.1),
    )
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0, abs=1e-15)
    assert result[2] == pytest.approx(np.pi / 2.0 + 0.1)


def test_physical_projection_uses_only_preaction_history_requested_action_and_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 1)
    receipt = _receipt()
    plan = _plan(1)
    physical, targets = subject._role_arrays(  # noqa: SLF001
        plan, MappingProxyType({"train-0": receipt})
    )
    assert physical.shape == (1, 9, 12)
    assert physical[0, 0, 6:10].tolist() == pytest.approx([0.1, 0.2, 0.3, -0.4])
    assert physical[0, 0, 10:].tolist() == pytest.approx([0.2, 0.45])
    assert physical[0, 6, 10:].tolist() == [0.0, 0.0]
    assert targets.shape == (1, 9, 4)
    assert targets[0, 0, 3].item() == pytest.approx(0.2)
    # The deliberately unusable future executed object did not enter the input path.
    assert receipt["branches"][0]["executed_block"].__class__ is object


def test_context_pooling_never_indexes_target_or_successor_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 2)
    plan = _plan(2)
    cache = torch.full((24, 256, 384), float("nan"), dtype=torch.float16)
    for state in plan.states:
        for index in state.context_artifact_indices:
            values = torch.arange(384, dtype=torch.float32) + float(index + 1)
            cache[index] = torch.nn.functional.normalize(
                values.repeat(256, 1), dim=-1
            ).to(torch.float16)
    pooled = subject.pool_context_grids_v1(plan, cache)
    assert pooled.shape == (2, 18_432)
    assert pooled.dtype == torch.float64
    assert torch.isfinite(pooled).all()


def test_train_only_thin_svd_pca_is_signed_deterministic_and_identity_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 6)
    monkeypatch.setattr(subject, "PCA_INPUT_DIMENSION", 20)
    monkeypatch.setattr(subject, "PCA_DIMENSION", 3)
    monkeypatch.setattr(subject, "EXPECTED_TRAIN_PLAN_IDENTITY", "a" * 64)
    generator = torch.Generator().manual_seed(4)
    pooled = torch.randn(6, 20, generator=generator, dtype=torch.float64)
    train = _role_data(6, pooled=pooled)
    binding = {"path": "/synthetic/evaluator.py", "sha256": "c" * 64, "byte_count": 7}
    first = subject.fit_train_pca_v1(train, implementation_source_binding=binding)
    second = subject.fit_train_pca_v1(train, implementation_source_binding=binding)
    assert first["identity_sha256"] == second["identity_sha256"]
    assert torch.equal(first["mean"], second["mean"])
    assert torch.equal(first["components"], second["components"])
    for column in range(3):
        component = first["components"][:, column]
        pivot = int(torch.argmax(torch.abs(component)))
        assert component[pivot] > 0.0
    changed = dict(first)
    changed["components"] = first["components"].clone()
    changed["components"][0, 0] += 1.0
    with pytest.raises(subject.PhysicalOutcomeScreenError, match="PCA basis"):
        subject.pca_identity_v1(changed)


def test_arm_inputs_have_zero_slots_or_replicated_current_visual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 2)
    physical = torch.arange(2 * 9 * 12, dtype=torch.float32).reshape(2, 9, 12)
    role = _role_data(2, physical=physical)
    visual = torch.arange(32, dtype=torch.float32).reshape(2, 16)
    odometry = subject.assemble_model_inputs_v1(
        role, visual, arm=subject.ODOMETRY_ARM
    )
    combined = subject.assemble_model_inputs_v1(
        role, visual, arm=subject.VISUAL_ARM
    )
    assert torch.equal(odometry[..., :12], physical)
    assert torch.count_nonzero(odometry[..., 12:]) == 0
    assert torch.equal(combined[..., :12], physical)
    assert torch.equal(combined[:, 0, 12:], visual)
    assert torch.equal(combined[:, -1, 12:], visual)


def test_action_means_and_train_only_population_scales_replace_degenerate_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 2)
    targets = torch.zeros(2, 9, 4, dtype=torch.float32)
    targets[1, :, 0] = 2.0
    role = _role_data(2, targets=targets)
    outcome = subject.fit_outcome_statistics_v1(role)
    assert torch.equal(outcome["action_means"][:, 0], torch.ones(9))
    assert outcome["residual_scales"].tolist() == [1.0, 1.0, 1.0, 1.0]
    inputs = torch.zeros(2, 9, 28, dtype=torch.float32)
    inputs[1, :, 0] = 2.0
    stats = subject.fit_input_statistics_v1(
        inputs, arm=subject.ODOMETRY_ARM, train_data_identity_sha256=role.identity_sha256
    )
    assert stats["mean"][0].item() == 1.0
    assert stats["scale"][0].item() == 1.0
    assert torch.equal(stats["scale"][1:], torch.ones(27))


def test_seed_local_schedule_is_exact_and_reused_by_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 4)
    monkeypatch.setattr(subject, "BATCH_STATES", 2)
    monkeypatch.setattr(subject, "UPDATES_PER_MEMBER", 6)
    seed = subject.MODEL_SEEDS[0]
    first = subject.training_orders_v1(seed)
    second = subject.training_orders_v1(seed)
    assert len(first) == 3
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    assert all(torch.equal(torch.sort(order).values, torch.arange(4)) for order in first)


def test_member_training_is_exact_from_same_initial_state_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 4)
    monkeypatch.setattr(subject, "BATCH_STATES", 2)
    monkeypatch.setattr(subject, "UPDATES_PER_MEMBER", 4)
    seed = subject.MODEL_SEEDS[0]
    generator = torch.Generator().manual_seed(9)
    inputs = torch.randn(4, 9, 28, generator=generator)
    targets = torch.randn(4, 9, 4, generator=generator)
    initial = initialize_physical_outcome_mlp_v1(seed).state_dict()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_threads = torch.get_num_threads()
    try:
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)
        orders = subject.training_orders_v1(seed)
        first_state, first_report = subject.fit_member_v1(
            initial, inputs, targets, orders
        )
        second_state, second_report = subject.fit_member_v1(
            initial, inputs, targets, orders
        )
    finally:
        torch.use_deterministic_algorithms(previous_determinism)
        torch.set_num_threads(previous_threads)
    assert first_report == second_report
    assert first_report["optimizer_steps"] == 4
    assert physical_outcome_state_identity_v1(first_state) == physical_outcome_state_identity_v1(second_state)
    assert all(torch.equal(first_state[name], second_state[name]) for name in first_state)


def test_physical_scoring_uses_progress_then_clamped_path_then_action_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 1)
    plan = _plan(1, role="eval")
    outcomes = torch.zeros(1, 9, 4, dtype=torch.float32)
    outcomes[0, :, 0] = torch.tensor([0.10, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03])
    outcomes[0, 0, 3] = 0.2
    outcomes[0, 1, 3] = -100.0
    scores = subject.physical_score_matrix_v1(plan, outcomes)
    assert scores.shape == (1, 9)
    assert scores[0, 1] == 0.0  # same quantized progress, shorter clamped path
    assert scores[0, 0] == 1.0
    outcomes[0, 0, 3] = 0.0
    scores = subject.physical_score_matrix_v1(plan, outcomes)
    assert scores[0, 0] == 0.0  # exact tie broken by action ID


def test_receipt_label_adapter_exercises_complete_selection_report_interface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 1)
    monkeypatch.setattr(subject.prior, "FAMILIES", ("large_enclosed_maze",))
    groups, _receipts_by_id = subject._groups_from_receipts(  # noqa: SLF001
        [_receipt()], role="train"
    )
    plan = _plan(1)
    plan.groups = groups
    scores = np.ones((1, 9), dtype=np.float64)
    scores[0, 2] = 0.0

    report = subject.report_arm_v1(plan, scores)

    row = report["group_results"][0]
    assert {
        "physical_fell": row["physical_fell"],
        "physical_tipped": row["physical_tipped"],
        "physical_target_progress_m": row["physical_target_progress_m"],
        "physical_path_length_m": row["physical_path_length_m"],
        "physical_progress_delta_to_canonical_oracle_m": row[
            "physical_progress_delta_to_canonical_oracle_m"
        ],
        "planar_clearance_proxy_min_m": row["planar_clearance_proxy_min_m"],
        "grid_recoverability_proxy": row["grid_recoverability_proxy"],
    } == {
        "physical_fell": False,
        "physical_tipped": False,
        "physical_target_progress_m": pytest.approx(0.08),
        "physical_path_length_m": pytest.approx(0.22),
        "physical_progress_delta_to_canonical_oracle_m": pytest.approx(0.02),
        "planar_clearance_proxy_min_m": None,
        "grid_recoverability_proxy": None,
    }
    assert "nonphysical_proxy_metrics" not in report["summary"]


def _summary(regret: float, oracle_rate: float = 0.0) -> dict[str, object]:
    return {
        "summary": {
            "normalized_rank_regret": regret,
            "oracle_equivalent_selection_rate": oracle_rate,
        }
    }


def test_scientific_gates_require_intervals_and_every_matched_seed() -> None:
    arms = {
        "privileged_physical_oracle": _summary(0.0, 1.0),
        "task_action_only": _summary(0.20),
        subject.ODOMETRY_ARM: _summary(0.15),
        subject.VISUAL_ARM: _summary(0.10),
        "random_expected": _summary(0.50),
    }
    artifacts = {
        subject.ODOMETRY_ARM: {
            "members": [
                {"report": _summary(value)} for value in (0.16, 0.17, 0.18)
            ]
        },
        subject.VISUAL_ARM: {
            "members": [
                {"report": _summary(value)} for value in (0.11, 0.12, 0.13)
            ]
        },
    }
    comparison = lambda upper: {"upper_95": upper}  # noqa: E731
    comparisons = {
        "odometry_vs_task_action_only": comparison(-0.01),
        "visual_vs_task_action_only": comparison(-0.02),
        "visual_vs_odometry": comparison(-0.005),
    }
    gates = subject.scientific_gates_v1(arms, artifacts, comparisons)
    assert all(gate["passed"] for gate in gates.values())
    artifacts[subject.VISUAL_ARM]["members"][1]["report"] = _summary(0.18)
    gates = subject.scientific_gates_v1(arms, artifacts, comparisons)
    assert gates["5_visual_beats_odometry"]["passed"] is False


def test_verdict_prioritizes_visual_then_odometry_then_stop_and_infrastructure() -> None:
    base_gates = {
        name: {"passed": True} for name in subject.SCIENTIFIC_GATE_NAMES
    }

    def evaluation(gates: dict[str, object]) -> dict[str, object]:
        result = {"schema": subject.SCHEMA, "gates": gates}
        result["evaluation_identity_sha256"] = subject.evaluation_identity_v1(result)
        return result

    assert subject.verdict_v1(
        evaluation(base_gates),
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )["terminal_status"] == subject.PASS_VISUAL_STATUS
    odometry = {name: dict(value) for name, value in base_gates.items()}
    odometry["4_visual_beats_task_action_only"]["passed"] = False
    assert subject.verdict_v1(
        evaluation(odometry),
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )["terminal_status"] == subject.PASS_ODOMETRY_STATUS
    stopped = {name: dict(value) for name, value in odometry.items()}
    stopped["3_odometry_beats_task_action_only"]["passed"] = False
    assert subject.verdict_v1(
        evaluation(stopped),
        infrastructure_checks_passed=True,
        deterministic_replay_passed=True,
    )["terminal_status"] == subject.STOP_STATUS
    assert subject.verdict_v1(
        evaluation(base_gates),
        infrastructure_checks_passed=False,
        deterministic_replay_passed=True,
    )["terminal_status"] == subject.INFRASTRUCTURE_FAILURE_STATUS
