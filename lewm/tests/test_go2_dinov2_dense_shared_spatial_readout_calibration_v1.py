from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest
import torch

from lewm.benchmarks import (
    go2_dinov2_dense_shared_spatial_readout_calibration_v1 as subject,
)
from lewm.benchmarks import go2_dinov2_physical_readout_calibration_v1 as prior
from lewm.models.go2_dinov2_dense_shared_spatial_readout_calibration_v1 import (
    PARAMETER_COUNT,
    dense_shared_state_identity_v1,
    initialize_dense_shared_spatial_readout_v1,
)


def _small_plan(*, role: str = "train", states: int = 2) -> prior.RoleFeaturePlanV1:
    groups = []
    state_rows = []
    artifact_ids: list[str] = []
    for state_index in range(states):
        base = len(artifact_ids)
        prefix = f"{role}_{state_index}"
        contexts = tuple(f"{prefix}_context_{slot}" for slot in range(3))
        targets = tuple(f"{prefix}_target_{action}" for action in range(9))
        artifact_ids.extend((*contexts, *targets))
        labels = tuple(
            SimpleNamespace(
                target_progress_m=float(9 - action),
                path_length_m=1.0 + action / 10.0,
                fell=False,
                tipped=False,
            )
            for action in range(9)
        )
        branches = tuple(
            SimpleNamespace(
                action_id=action,
                target_rgb_artifact_id=targets[action],
                oracle_dense_rank=action,
                labels=labels[action],
            )
            for action in range(9)
        )
        group = SimpleNamespace(
            role=role,
            state_id=f"state_{prefix}",
            family="straight_open",
            scene_id=f"scene_{role}_{state_index // 8}",
            group_index=state_index,
            state_index_in_scene=state_index % 8,
            relative_target_xy_body_m=(float(state_index + 1), -2.0),
            context_rgb_artifact_ids=contexts,
            branches=branches,
        )
        groups.append(group)
        state_rows.append(
            prior.RoleStateIndexV1(
                role_state_index=state_index,
                state_id=group.state_id,
                role=role,
                family=group.family,
                scene_id=group.scene_id,
                group_index=group.group_index,
                state_index_in_scene=group.state_index_in_scene,
                relative_target_xy_body_m=group.relative_target_xy_body_m,
                context_artifact_indices=(base, base + 1, base + 2),
                target_artifact_indices=tuple(range(base + 3, base + 12)),
                dense_ranks=tuple(range(9)),
                target_progress_m=tuple(label.target_progress_m for label in labels),
                physical_fell=(False,) * 9,
                physical_tipped=(False,) * 9,
            )
        )
    return prior.RoleFeaturePlanV1(
        role=role,
        artifact_ids=tuple(artifact_ids),
        artifact_index_by_id=MappingProxyType(
            {artifact_id: index for index, artifact_id in enumerate(artifact_ids)}
        ),
        states=tuple(state_rows),
        groups=tuple(groups),
        identity_sha256=("1" if role == "train" else "2") * 64,
    )


def test_fixed_configuration_matches_frozen_capacity_and_schedule() -> None:
    config = subject.config_v1()
    assert PARAMETER_COUNT == 245
    assert config["model_seeds"] == [2_026_080_303, 2_026_080_304, 2_026_080_305]
    assert config["parameter_count_per_member"] == 245
    assert config["true_ensemble_parameter_count"] == 735
    assert config["current_ensemble_parameter_count"] == 735
    assert config["checkpoint_dense_parameter_count"] == 1_470
    assert config["pca"]["dimension"] == 8
    assert config["pca"]["row_count"] == 327_680
    assert config["epochs"] == 256
    assert config["batch_states"] == 16
    assert config["steps_per_epoch"] == 8
    assert config["optimizer_steps"] == 2_048
    assert config["optimizer"] == {
        "name": "AdamW",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-2,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "fused": False,
    }


def test_patch_coordinates_are_exact_row_major_centres() -> None:
    coordinates = subject.patch_coordinates_v1()
    assert coordinates.dtype == torch.float32
    assert coordinates.shape == (256, 2)
    assert torch.equal(coordinates[0], torch.tensor((-0.9375, -0.9375)))
    assert torch.equal(coordinates[15], torch.tensor((0.9375, -0.9375)))
    assert torch.equal(coordinates[16], torch.tensor((-0.9375, -0.8125)))
    assert torch.equal(coordinates[-1], torch.tensor((0.9375, 0.9375)))


def test_train_pca_uses_only_registered_current_and_successor_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _small_plan()
    monkeypatch.setattr(subject, "STATE_COUNT", 2)
    monkeypatch.setattr(subject, "TOKEN_COUNT", 4)
    monkeypatch.setattr(subject, "TOKEN_DIMENSION", 10)
    monkeypatch.setattr(subject, "PCA_DIMENSION", 3)
    monkeypatch.setattr(subject, "PCA_ROW_COUNT", 80)
    monkeypatch.setattr(subject, "TOKEN_NORM_TOLERANCE", 1.0e-3)
    monkeypatch.setattr(prior, "ROLE_ARTIFACT_COUNT", 24)

    values = torch.arange(24 * 4 * 10, dtype=torch.float32).reshape(24, 4, 10)
    values = torch.nn.functional.normalize(values + 1.0, dim=-1).to(torch.float16)
    implementation = {"path": "/synthetic/evaluator.py", "sha256": "a" * 64, "byte_count": 1}
    first = subject.fit_train_pca_v1(
        plan, values, implementation_source_binding=implementation
    )
    second = subject.fit_train_pca_v1(
        plan, values.clone(), implementation_source_binding=implementation
    )
    assert first["identity_sha256"] == second["identity_sha256"]
    assert first["source"]["artifact_indices"] == [
        2,
        14,
        *range(3, 12),
        *range(15, 24),
    ]
    assert first["source"]["row_count"] == 80
    assert first["mean"].dtype == torch.float64
    assert first["components"].shape == (10, 3)
    assert first["scales"].shape == (3,)
    projected = subject.project_cache_v1(values, first, label="synthetic")
    assert projected.shape == (24, 4, 3)
    assert projected.dtype == torch.float32
    assert torch.isfinite(projected).all()

    changed_unused_context = values.clone()
    changed_unused_context[0] = torch.nn.functional.normalize(
        torch.flip(changed_unused_context[0].float(), dims=(-1,)), dim=-1
    ).to(torch.float16)
    unchanged = subject.fit_train_pca_v1(
        plan,
        changed_unused_context,
        implementation_source_binding=implementation,
    )
    assert unchanged["identity_sha256"] == first["identity_sha256"]


def test_dense_panels_use_colocated_true_current_delta_and_registered_condition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _small_plan(states=1)
    monkeypatch.setattr(subject, "STATE_COUNT", 1)
    monkeypatch.setattr(prior, "ROLE_ARTIFACT_COUNT", 12)
    projected = torch.arange(12 * 256 * 8, dtype=torch.float32).reshape(12, 256, 8)
    true_relations, conditions = subject._dense_panels(  # noqa: SLF001
        plan, projected, successor_mode="true_future"
    )
    current_relations, current_conditions = subject._dense_panels(  # noqa: SLF001
        plan, projected, successor_mode="current_state"
    )
    current = projected[2]
    successor = projected[3]
    assert torch.equal(true_relations[0, 0, :, :8], current)
    assert torch.equal(true_relations[0, 0, :, 8:16], successor)
    assert torch.equal(true_relations[0, 0, :, 16:], successor - current)
    assert torch.equal(current_relations[0, 0, :, :8], current)
    assert torch.equal(current_relations[0, 0, :, 8:16], current)
    assert torch.count_nonzero(current_relations[0, 0, :, 16:]) == 0
    assert torch.equal(conditions, current_conditions)
    assert torch.equal(conditions[0, 0], torch.tensor((0.1, -0.2, 2.0 / 3.0, 1.0)))
    assert torch.equal(conditions[0, 6], torch.tensor((0.1, -0.2, 0.0, 0.0)))


def test_training_from_same_initial_state_and_orders_is_exact_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 2)
    monkeypatch.setattr(subject, "BATCH_STATES", 1)
    monkeypatch.setattr(subject, "EPOCHS", 2)
    monkeypatch.setattr(subject, "OPTIMIZER_STEPS", 4)
    generator = torch.Generator(device="cpu").manual_seed(7)
    relations = torch.randn(2, 9, 256, 24, generator=generator)
    conditions = torch.randn(2, 9, 4, generator=generator)
    targets = torch.randn(2, 9, generator=generator)
    orders = (torch.tensor((1, 0)), torch.tensor((0, 1)))
    initial = initialize_dense_shared_spatial_readout_v1(2_026_080_303).state_dict()

    first_state, first_report = subject._train_one_model(  # noqa: SLF001
        initial,
        relations,
        conditions,
        targets,
        orders,
        device=torch.device("cpu"),
    )
    second_state, second_report = subject._train_one_model(  # noqa: SLF001
        initial,
        relations,
        conditions,
        targets,
        orders,
        device=torch.device("cpu"),
    )
    assert first_report == second_report
    assert first_report["optimizer_steps"] == 4
    assert dense_shared_state_identity_v1(first_state) == dense_shared_state_identity_v1(
        second_state
    )
    for name in first_state:
        assert torch.equal(first_state[name], second_state[name])


def test_member_prediction_reports_exact_score_hashes_and_seed_dispersion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 2)
    monkeypatch.setattr(subject, "BATCH_STATES", 1)
    members = []
    for seed in subject.MODEL_SEEDS:
        state = initialize_dense_shared_spatial_readout_v1(seed).state_dict()
        identity = dense_shared_state_identity_v1(state)
        members.append(
            {
                "seed": seed,
                "true_state": state,
                "true_identity_sha256": identity,
                "true_training": {"state_identity_sha256": identity},
            }
        )
    generator = torch.Generator(device="cpu").manual_seed(11)
    relations = torch.randn(2, 9, 256, 24, generator=generator)
    conditions = torch.randn(2, 9, 4, generator=generator)
    first_scores, first = subject._predict_members(  # noqa: SLF001
        {"members": members},
        relations,
        conditions,
        state_key="true_state",
        device=torch.device("cpu"),
    )
    second_scores, second = subject._predict_members(  # noqa: SLF001
        {"members": members},
        relations,
        conditions,
        state_key="true_state",
        device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(first_scores, second_scores)
    assert first == second
    assert first["score_stack_shape"] == [3, 2, 9]
    assert len(first["members"]) == 3
    assert all(len(item["score_sha256"]) == 64 for item in first["members"])
    assert len(first["ensemble_score_sha256"]) == 64
    assert first["seed_dispersion"]["mean_cell_population_std"] >= 0.0
    assert first["seed_dispersion"]["maximum_cell_population_std"] >= 0.0
    assert 0.0 <= first["seed_dispersion"][
        "state_seed_argmin_disagreement_rate"
    ] <= 1.0


def _synthetic_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], prior.RoleFeaturePlanV1, dict[str, object]]:
    plan = _small_plan(states=4)
    monkeypatch.setattr(subject, "STATE_COUNT", 4)
    monkeypatch.setattr(subject, "TOKEN_COUNT", 4)
    monkeypatch.setattr(subject, "TOKEN_DIMENSION", 10)
    monkeypatch.setattr(subject, "PCA_DIMENSION", 3)
    monkeypatch.setattr(subject, "PCA_ROW_COUNT", 160)
    monkeypatch.setattr(subject, "OPTIMIZER_STEPS", 1)
    monkeypatch.setattr(subject, "EXPECTED_TRAIN_PLAN_IDENTITY", plan.identity_sha256)
    monkeypatch.setattr(prior, "ROLE_ARTIFACT_COUNT", 48)
    implementation = {
        "path": "/synthetic/evaluator.py",
        "sha256": "b" * 64,
        "byte_count": 2,
    }
    values = torch.arange(48 * 4 * 10, dtype=torch.float32).reshape(48, 4, 10)
    values = torch.nn.functional.normalize(values + 1.0, dim=-1).to(torch.float16)
    pca = subject.fit_train_pca_v1(
        plan, values, implementation_source_binding=implementation
    )

    targets = subject._normalized_rank_targets(plan)  # noqa: SLF001
    task_features = np.stack(
        [
            subject.task_conditioned_feature_v1(
                None, relative_target_xy_body_m=state.relative_target_xy_body_m
            )
            for state in plan.states
        ]
    )
    heads = tuple(
        subject.fit_ridge_readout_v1(
            task_features,
            targets[:, action],
            ridge_lambda=subject.TASK_RIDGE_LAMBDA,
        )
        for action in range(9)
    )
    readout = subject._assemble_task_heads(heads)  # noqa: SLF001
    monkeypatch.setattr(subject, "EXPECTED_TASK_IDENTITY", readout.identity_sha256)
    task_payload = subject._task_payload(readout)  # noqa: SLF001

    members = []
    for seed in subject.MODEL_SEEDS:
        state = {
            name: value.detach().clone()
            for name, value in initialize_dense_shared_spatial_readout_v1(
                seed
            ).state_dict().items()
        }
        identity = dense_shared_state_identity_v1(state)
        report = {
            "optimizer_steps": 1,
            "last_minibatch_residual_mse": 0.1,
            "last_gradient_norm_before_clip": 0.2,
            "full_train_residual_mse": 0.1,
            "mean_normalized_attention_entropy": 1.0,
            "state_identity_sha256": identity,
        }
        members.append(
            {
                "seed": seed,
                "initial_identity_sha256": identity,
                "true_state": deepcopy(state),
                "true_identity_sha256": identity,
                "true_training": deepcopy(report),
                "current_state": deepcopy(state),
                "current_identity_sha256": identity,
                "current_training": deepcopy(report),
            }
        )
    checkpoint: dict[str, object] = {
        "schema": subject.CHECKPOINT_SCHEMA,
        "config": subject.config_v1(),
        "train_plan_identity": plan.identity_sha256,
        "pca": pca,
        "task_action_only": task_payload,
        "members": members,
    }
    checkpoint["identity_sha256"] = subject._checkpoint_content_identity_v1(  # noqa: SLF001
        checkpoint
    )
    return checkpoint, plan, implementation


@pytest.mark.parametrize("mutation", ["pca", "task", "true_state"])
def test_checkpoint_validation_rederives_tensor_identities(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    checkpoint, plan, implementation = _synthetic_checkpoint(monkeypatch)
    subject.validate_checkpoint_v1(
        checkpoint,
        train_plan=plan,
        implementation_source_binding=implementation,
    )
    changed = deepcopy(checkpoint)
    if mutation == "pca":
        changed["pca"]["mean"][0] += 1.0
    elif mutation == "task":
        changed["task_action_only"]["heads"][0]["coefficients"][0, 0] += 1.0
    else:
        changed["members"][0]["true_state"]["W_r"][0, 0] += 1.0
    with pytest.raises(subject.DenseSharedCalibrationError):
        subject.validate_checkpoint_v1(
            changed,
            train_plan=plan,
            implementation_source_binding=implementation,
        )


def _evaluation_with_gates(value: bool) -> dict[str, object]:
    family_panel = {
        family: {"normalized_rank_regret": 0.1} for family in prior.FAMILIES
    }
    random_panel = {
        family: {"normalized_rank_regret": 0.5} for family in prior.FAMILIES
    }
    arms = {
        "privileged_physical_oracle": {
            "summary": {
                "normalized_rank_regret": 0.0,
                "oracle_equivalent_selection_rate": 1.0,
            }
        },
        "dense_shared_true_future": {
            "summary": {"normalized_rank_regret": 0.1},
            "per_family": family_panel,
        },
        "random_expected": {
            "summary": {"normalized_rank_regret": 0.5},
            "per_family": random_panel,
        },
    }
    comparisons = {
        name: {"upper_95": -0.1 if value else 0.1}
        for name in (
            "true_future_vs_task_action_only",
            "true_future_vs_current_state",
            "true_future_vs_relational_persistence",
        )
    }
    gates = subject._scientific_gates_v1(arms, comparisons)  # noqa: SLF001
    evaluation = {
        "schema": subject.SCHEMA,
        "status": "COMPLETE_MODEL_INDEPENDENT_EVALUATION",
        "claim_scope": "REUSED_DEVELOPMENT_ROLE_DENSE_ORACLE_FUTURE_CALIBRATION",
        "config": subject.config_v1(),
        "feature_plan": {},
        "checkpoint_identity_sha256": "a" * 64,
        "pca": {},
        "task_action_only_readout": {},
        "member_training": [],
        "member_diagnostics": {},
        "score_evidence": {},
        "arms": arms,
        "paired_scene_cluster_comparisons": comparisons,
        "safety": {},
        "finiteness": {},
        "gates": gates,
        "scientific_gates_2_to_6_passed": all(
            gate["passed"] for gate in gates.values()
        ),
    }
    evaluation["replay_identity_sha256"] = subject.evaluation_identity_v1(evaluation)
    return evaluation


@pytest.mark.parametrize(
    ("science", "infrastructure", "replay", "expected"),
    [
        (True, True, True, subject.PASS_STATUS),
        (False, True, True, subject.STOP_STATUS),
        (True, False, True, subject.INFRASTRUCTURE_FAILURE_STATUS),
        (True, True, False, subject.INFRASTRUCTURE_FAILURE_STATUS),
    ],
)
def test_verdict_has_exact_terminal_routes(
    science: bool, infrastructure: bool, replay: bool, expected: str
) -> None:
    verdict = subject.verdict_v1(
        _evaluation_with_gates(science),
        infrastructure_checks_passed=infrastructure,
        deterministic_replay_passed=replay,
    )
    assert verdict["terminal_status"] == expected
    assert verdict["passed"] is (expected == subject.PASS_STATUS)
    assert list(verdict["gates"]) == [
        "1_infrastructure_and_custody",
        *list(_evaluation_with_gates(science)["gates"]),
        "7_deterministic_replay",
    ]


def test_verdict_rejects_nonboolean_or_inconsistent_gate_contract() -> None:
    nonboolean = _evaluation_with_gates(True)
    nonboolean["gates"]["2_privileged_physical_oracle"]["passed"] = 1
    nonboolean["replay_identity_sha256"] = subject.evaluation_identity_v1(nonboolean)
    with pytest.raises(subject.DenseSharedCalibrationError):
        subject.verdict_v1(
            nonboolean,
            infrastructure_checks_passed=True,
            deterministic_replay_passed=True,
        )

    inconsistent = _evaluation_with_gates(True)
    inconsistent["scientific_gates_2_to_6_passed"] = False
    inconsistent["replay_identity_sha256"] = subject.evaluation_identity_v1(inconsistent)
    with pytest.raises(subject.DenseSharedCalibrationError):
        subject.verdict_v1(
            inconsistent,
            infrastructure_checks_passed=True,
            deterministic_replay_passed=True,
        )
