from __future__ import annotations

import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lewm.benchmarks import (
    go2_dense_vjepa2_1_physical_interface_ceiling_v1 as subject,
)


def _binding() -> dict[str, object]:
    return {
        "path": "/synthetic/evaluator.py",
        "sha256": "a" * 64,
        "byte_count": 1,
    }


def _state(
    state_id: str,
    *,
    current: int,
    targets: tuple[int, ...],
    family: str = "family",
    scene: str = "scene",
    goal: tuple[float, float] = (1.0, 2.0),
) -> SimpleNamespace:
    return SimpleNamespace(
        state_id=state_id,
        family=family,
        scene_id=scene,
        relative_target_xy_body_m=goal,
        context_artifact_indices=(current, current, current),
        target_artifact_indices=targets,
    )


def _two_state_plan(identity: str = "p" * 64) -> SimpleNamespace:
    return SimpleNamespace(
        role="train",
        identity_sha256=identity,
        artifact_ids=tuple(f"artifact-{index}" for index in range(6)),
        states=(
            _state("state-0", current=0, targets=(2, 3)),
            _state("state-1", current=1, targets=(4, 5)),
        ),
    )


def _patch_tiny_numeric_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 2)
    monkeypatch.setattr(subject, "ACTION_COUNT", 2)
    monkeypatch.setattr(subject, "ROLE_ARTIFACT_COUNT", 6)
    monkeypatch.setattr(subject, "TOKEN_COUNT", 2)
    monkeypatch.setattr(subject, "TOKEN_DIMENSION", 4)
    monkeypatch.setattr(subject, "PCA_DIMENSION", 2)
    monkeypatch.setattr(subject, "RELATIONAL_DIMENSION", 6)
    monkeypatch.setattr(subject, "PCA_ROW_COUNT", 12)


def _gate_inputs(
    *,
    upper_95: float = -0.01,
    true_regret: float = 0.10,
    random_regret: float = 0.20,
    oracle_regret: float = 0.0,
    oracle_rate: float = 1.0,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    arms = {
        "privileged_physical_oracle": {
            "summary": {
                "normalized_rank_regret": oracle_regret,
                "oracle_equivalent_selection_rate": oracle_rate,
            }
        },
        subject.TRUE_ARM: {
            "summary": {"normalized_rank_regret": true_regret}
        },
        "random_expected": {
            "summary": {"normalized_rank_regret": random_regret}
        },
    }
    comparisons = {
        name: {
            "mean_delta": upper_95 - 0.01,
            "lower_95": upper_95 - 0.02,
            "upper_95": upper_95,
        }
        for name in subject.COMPARISON_BASELINES
    }
    return arms, comparisons


def _evaluation(
    arms: dict[str, dict[str, object]],
    comparisons: dict[str, dict[str, object]],
) -> dict[str, object]:
    gates = subject.scientific_gates_v1(arms, comparisons)
    result: dict[str, object] = {
        "schema": subject.SCHEMA,
        "status": "COMPLETE_DEVELOPMENT_ONLY_PHYSICAL_INTERFACE_EVALUATION",
        "config": subject.config_v1(),
        "arms": arms,
        "paired_family_scene_cluster_comparisons": comparisons,
        "gates": gates,
        "scientific_gates_2_to_9_passed": all(
            gate["passed"] for gate in gates.values()
        ),
    }
    result["evaluation_identity_sha256"] = subject.evaluation_identity_v1(result)
    return result


def test_exact_frozen_configuration() -> None:
    assert subject.config_v1() == {
        "schema": "lewm_go2_dense_vjepa2_1_physical_interface_ceiling_config_v1",
        "role_contract": {
            "states_per_role": 128,
            "scenes_per_role": 16,
            "actions": 9,
            "artifacts_per_role": 1_536,
            "tokens_per_grid": 256,
            "token_dimension": 768,
            "storage_dtype": "float16",
            "token_norm_tolerance": 2.0e-3,
            "train_plan_identity": (
                "f6f94cf589ec44324fdefe0939aa7076e25543d984464d5b264a0b2f0ff9535b"
            ),
            "eval_plan_identity": (
                "5dbf9733fd245caff27ce5c5c2b3dc90a3fe9ca9e1bc894dc10a97d64dad9231"
            ),
            "combined_plan_identity": (
                "99e60638634eff6ac244cff023cd2ae8f7aa0c53326263ba7a36fa6847386375"
            ),
        },
        "pca": {
            "dimension": 8,
            "row_count": 327_680,
            "source_grids": 1_280,
            "source_order": "all_current_then_state_major_action_major_successors",
            "patch_order": "row_major_16x16",
            "statistics_dtype": "float64",
            "covariance": "population",
            "eigensolver": "numpy.linalg.eigh",
            "ordering": "descending_eigenvalue_then_original_ascending_index",
            "sign": "largest_absolute_loading_smallest_channel_positive",
            "whitening_epsilon": 1.0e-12,
            "clipping": False,
        },
        "readout": {
            "model": "unchanged_DenseSharedSpatialReadoutV1",
            "parameter_count_per_member": 245,
            "true_ensemble_parameters": 735,
            "current_ensemble_parameters": 735,
            "condition": [
                "goal_x_div_10",
                "goal_y_div_10",
                "vx_div_0.30",
                "wz_div_0.45",
            ],
            "relational": ["current", "successor", "successor_minus_current"],
        },
        "training": {
            "seeds": [2_026_080_303, 2_026_080_304, 2_026_080_305],
            "epochs": 256,
            "batch_states": 16,
            "steps_per_epoch": 8,
            "optimizer_steps": 2_048,
            "optimizer": "AdamW",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-2,
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "amsgrad": False,
            "foreach": False,
            "fused": False,
            "gradient_clip_norm": 1.0,
            "complete_state_minibatches": True,
            "deterministic_algorithms": True,
            "float32_matmul_precision": "highest",
            "device": "ROCm",
        },
        "task_action_only": {
            "identity_sha256": (
                "69895316b19bc179e35fdd76905aadbd50b6ad3e22e965b662ba59672c52886a"
            ),
            "required_eval_regret": 0.17441406250000002,
            "ridge_lambda": 1.0e-3,
        },
        "retained_physical_predecessor": {
            "evaluation_identity_sha256": (
                "5e19a2547187f1101a4a19ee6ffd9d8892f38efc1f5c52842430d860430091cc"
            ),
            "required_eval_regret": 0.14896763392857143,
            "checkpoint_loaded": False,
        },
        "wrong_scene": {
            "pairing": (
                "other_lexicographic_scene_same_family_same_role_plan_ordinal"
            ),
            "states_per_scene": 8,
            "same_action": True,
        },
        "train_action_mean_innovation": {
            "source": "train_only_projected_successor_minus_current",
            "mean_axes": "128_states_per_action_patch",
            "accumulator_dtype": "float64",
            "scorer_dtype": "float32",
        },
        "bootstrap": {
            "resamples": 10_000,
            "seed": 2_026_080_314,
            "unit": "whole_scene",
            "family_weighting": "equal_eight_families",
            "interval": "percentile_95",
        },
    }


def test_config_rejects_shared_readout_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "PARAMETER_COUNT", 246)
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="shared readout"):
        subject.config_v1()


def test_tiny_pca_freezes_source_order_tie_break_sign_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_numeric_contract(monkeypatch)
    plan = _two_state_plan()
    basis = torch.eye(4, dtype=torch.float16)
    features = torch.stack(
        [
            torch.stack((basis[index % 4], basis[(index + 1) % 4]))
            for index in range(6)
        ]
    )
    root_half = np.sqrt(0.5)

    def fixed_eigh(_covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vectors = np.eye(4, dtype=np.float64)
        vectors[:, 1] = (-root_half, root_half, 0.0, 0.0)
        vectors[:, 2] = (0.0, 0.0, -1.0, 0.0)
        return np.asarray((1.0, 4.0, 4.0, 2.0)), vectors

    monkeypatch.setattr(subject.np.linalg, "eigh", fixed_eigh)
    pca = subject.fit_train_pca_v1(
        plan,
        features,
        implementation_source_binding=_binding(),
    )

    assert pca["source"]["artifact_indices"] == [0, 1, 2, 3, 4, 5]
    assert pca["source"]["artifact_ids"] == list(plan.artifact_ids)
    assert torch.equal(pca["eigenvalues"], torch.tensor([4.0, 4.0]))
    expected_components = torch.tensor(
        [
            [root_half, 0.0],
            [-root_half, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    assert torch.equal(pca["components"], expected_components)
    expected_mean = features.numpy().astype(np.float64).reshape(12, 4).mean(axis=0)
    assert np.array_equal(pca["mean"].numpy(), expected_mean)
    assert pca["identity_sha256"] == subject.pca_identity_v1(pca)

    tampered = copy.deepcopy(pca)
    tampered["mean"][0] += 0.125
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="PCA identity"):
        subject.project_cache_v1(features, tampered, label="synthetic")

    wrong_sign = copy.deepcopy(pca)
    wrong_sign["components"][:, 0].mul_(-1.0)
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="component sign"):
        subject.pca_identity_v1(wrong_sign)


def test_synthetic_cache_validation_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ROLE_ARTIFACT_COUNT", 1)
    monkeypatch.setattr(subject, "TOKEN_COUNT", 1)
    monkeypatch.setattr(subject, "TOKEN_DIMENSION", 2)
    valid = torch.tensor([[[1.0, 0.0]]], dtype=torch.float16)
    assert subject._validate_cache_v1(valid, label="synthetic") is valid  # noqa: SLF001

    invalid_values = (
        valid.float(),
        torch.ones((1, 2, 2), dtype=torch.float16),
        torch.tensor([[[0.0, 0.0]]], dtype=torch.float16),
        torch.tensor([[[float("nan"), 0.0]]], dtype=torch.float16),
    )
    for value in invalid_values:
        with pytest.raises(subject.DenseVJEPAInterfaceError):
            subject._validate_cache_v1(value, label="synthetic")  # noqa: SLF001


def test_action_mean_uses_float64_reduction_then_float32_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_numeric_contract(monkeypatch)
    plan = _two_state_plan()
    monkeypatch.setattr(subject, "EXPECTED_TRAIN_PLAN_IDENTITY", plan.identity_sha256)
    projected = torch.tensor(
        [
            [[0.10, -0.20], [0.30, 0.40]],
            [[1.00, 2.00], [3.00, 4.00]],
            [[0.25, -0.05], [0.35, 0.90]],
            [[-0.20, 0.80], [1.30, -0.60]],
            [[1.40, 2.20], [3.80, 3.50]],
            [[0.90, 1.10], [4.20, 5.30]],
        ],
        dtype=torch.float32,
    )

    payload = subject.train_action_mean_innovation_v1(plan, projected)
    values = projected.numpy()
    rows = np.empty((2, 2, 2, 2), dtype=np.float64)
    for state_index, state in enumerate(plan.states):
        current = values[state.context_artifact_indices[-1]].astype(np.float64)
        for action in range(2):
            successor = values[state.target_artifact_indices[action]].astype(np.float64)
            rows[action, state_index] = successor - current
    expected = rows.mean(axis=1, dtype=np.float64).astype(np.float32)

    assert payload["accumulator_dtype"] == "float64"
    assert payload["storage_dtype"] == "float32"
    assert payload["values"].dtype == torch.float32
    assert np.array_equal(payload["values"].numpy(), expected)
    assert subject.validate_action_mean_innovation_v1(payload) is payload["values"]

    replay = subject.train_action_mean_innovation_v1(plan, projected.clone())
    assert replay["identity_sha256"] == payload["identity_sha256"]
    assert torch.equal(replay["values"], payload["values"])

    tampered = copy.deepcopy(payload)
    tampered["values"][0, 0, 0] += 1.0
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="identity"):
        subject.validate_action_mean_innovation_v1(tampered)


def test_action_mean_rejects_wrong_projected_dtype_shape_and_finiteness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_numeric_contract(monkeypatch)
    plan = _two_state_plan()
    valid = torch.zeros((6, 2, 2), dtype=torch.float32)
    invalid_values = (
        valid.to(torch.float64),
        torch.zeros((5, 2, 2), dtype=torch.float32),
        valid.clone(),
    )
    invalid_values[2][0, 0, 0] = float("nan")
    for value in invalid_values:
        with pytest.raises(subject.DenseVJEPAInterfaceError, match="cache contract"):
            subject.train_action_mean_innovation_v1(plan, value)


def _wrong_scene_plan() -> SimpleNamespace:
    states = []
    for ordinal in range(8):
        states.extend(
            (
                _state(
                    f"z-{ordinal}",
                    current=2 * (2 * ordinal),
                    targets=(2 * (2 * ordinal) + 1,),
                    scene="scene-z",
                    goal=(float(ordinal + 1), 2.0),
                ),
                _state(
                    f"a-{ordinal}",
                    current=2 * (2 * ordinal + 1),
                    targets=(2 * (2 * ordinal + 1) + 1,),
                    scene="scene-a",
                    goal=(float(ordinal + 11), 3.0),
                ),
            )
        )
    return SimpleNamespace(
        role="eval",
        identity_sha256="e" * 64,
        states=tuple(states),
    )


def test_wrong_scene_mapping_is_unique_involutive_role_plan_ordinal_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 16)
    plan = _wrong_scene_plan()
    donors, document = subject.same_action_wrong_scene_donors_v1(plan)

    assert donors == tuple(index ^ 1 for index in range(16))
    assert all(donors[donors[index]] == index for index in range(16))
    assert [
        (pair["left_state_id"], pair["right_state_id"])
        for pair in document["scene_pairs"]
    ] == [(f"a-{ordinal}", f"z-{ordinal}") for ordinal in range(8)]
    identity_document = dict(document)
    identity = identity_document.pop("identity_sha256")
    assert identity == hashlib.sha256(
        subject.canonical_bytes_v1(identity_document)
    ).hexdigest()


def test_wrong_scene_relations_use_donor_successor_but_source_goal_and_current(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 16)
    monkeypatch.setattr(subject, "ACTION_COUNT", 1)
    monkeypatch.setattr(subject, "ROLE_ARTIFACT_COUNT", 32)
    monkeypatch.setattr(subject, "TOKEN_COUNT", 1)
    monkeypatch.setattr(subject, "PCA_DIMENSION", 1)
    monkeypatch.setattr(subject, "RELATIONAL_DIMENSION", 3)
    plan = _wrong_scene_plan()
    donors, _document = subject.same_action_wrong_scene_donors_v1(plan)
    projected = torch.empty((32, 1, 1), dtype=torch.float32)
    for index in range(16):
        projected[2 * index, 0, 0] = 100.0 + index
        projected[2 * index + 1, 0, 0] = 1_000.0 + index

    relations, conditions = subject.relational_panels_v1(
        plan,
        projected,
        mode="wrong_scene",
        wrong_scene_donors=donors,
    )
    assert torch.equal(
        relations[0, 0, 0], torch.tensor([100.0, 1_001.0, 901.0])
    )
    assert tuple(conditions[0, 0, :2].tolist()) == pytest.approx((0.1, 0.2))

    with pytest.raises(subject.DenseVJEPAInterfaceError, match="donor role"):
        subject.relational_panels_v1(
            plan,
            projected,
            mode="wrong_scene",
            wrong_scene_donors=tuple(range(16)),
        )


def test_wrong_scene_mapping_rejects_role_scene_count_and_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "STATE_COUNT", 16)
    plan = _wrong_scene_plan()

    wrong_role = copy.copy(plan)
    wrong_role.role = "train"
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="exact eval role"):
        subject.same_action_wrong_scene_donors_v1(wrong_role)

    unbalanced_states = list(plan.states)
    unbalanced_states[1] = copy.copy(unbalanced_states[1])
    unbalanced_states[1].scene_id = "scene-z"
    unbalanced = copy.copy(plan)
    unbalanced.states = tuple(unbalanced_states)
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="eight states"):
        subject.same_action_wrong_scene_donors_v1(unbalanced)

    three_scene_states = list(plan.states)
    three_scene_states[0] = copy.copy(three_scene_states[0])
    three_scene_states[0].scene_id = "scene-third"
    three_scenes = copy.copy(plan)
    three_scenes.states = tuple(three_scene_states)
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="two scenes"):
        subject.same_action_wrong_scene_donors_v1(three_scenes)


def test_scientific_gates_use_strict_zero_effect_boundaries() -> None:
    arms, comparisons = _gate_inputs()
    passing = subject.scientific_gates_v1(arms, comparisons)
    assert set(passing) == subject.SCIENTIFIC_GATE_NAMES
    assert all(gate["passed"] is True for gate in passing.values())

    comparison_gate_names = {
        "true_future_vs_task_action_only": "3_true_future_beats_task_action_only",
        "true_future_vs_retained_physical_predecessor": (
            "4_true_future_beats_retained_physical_predecessor"
        ),
        "true_future_vs_current_state": "5_true_future_beats_current_state",
        "true_future_vs_relational_persistence": (
            "6_true_future_beats_relational_persistence"
        ),
        "true_future_vs_same_action_wrong_scene": (
            "7_true_future_beats_same_action_wrong_scene"
        ),
        "true_future_vs_train_action_mean_innovation": (
            "8_true_future_beats_train_action_mean_innovation"
        ),
    }
    for comparison_name, gate_name in comparison_gate_names.items():
        boundary = copy.deepcopy(comparisons)
        boundary[comparison_name]["upper_95"] = 0.0
        assert subject.scientific_gates_v1(arms, boundary)[gate_name]["passed"] is False

    equal_random = copy.deepcopy(arms)
    equal_random[subject.TRUE_ARM]["summary"]["normalized_rank_regret"] = 0.20
    assert (
        subject.scientific_gates_v1(equal_random, comparisons)[
            "9_true_future_beats_random_expected"
        ]["passed"]
        is False
    )


def test_scientific_gates_reject_nonexact_comparison_inventory() -> None:
    arms, comparisons = _gate_inputs()
    missing = dict(comparisons)
    missing.pop("true_future_vs_task_action_only")
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="inventory"):
        subject.scientific_gates_v1(arms, missing)

    extra = dict(comparisons)
    extra["unregistered_posthoc_comparison"] = {
        "mean_delta": -1.0,
        "lower_95": -1.0,
        "upper_95": -1.0,
    }
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="inventory"):
        subject.scientific_gates_v1(arms, extra)

    non_oracle = copy.deepcopy(arms)
    non_oracle["privileged_physical_oracle"]["summary"][
        "oracle_equivalent_selection_rate"
    ] = 0.999
    assert (
        subject.scientific_gates_v1(non_oracle, comparisons)[
            "2_privileged_physical_oracle"
        ]["passed"]
        is False
    )


@pytest.mark.parametrize(
    ("science_passes", "infrastructure", "replay", "expected_status"),
    (
        (True, True, True, subject.PASS_STATUS),
        (False, True, True, subject.STOP_STATUS),
        (True, False, True, subject.INFRASTRUCTURE_FAILURE_STATUS),
        (True, True, False, subject.INFRASTRUCTURE_FAILURE_STATUS),
    ),
)
def test_verdict_has_exact_pass_stop_and_infrastructure_routes(
    science_passes: bool,
    infrastructure: bool,
    replay: bool,
    expected_status: str,
) -> None:
    arms, comparisons = _gate_inputs(upper_95=-0.01 if science_passes else 0.0)
    evaluation = _evaluation(arms, comparisons)
    verdict = subject.verdict_v1(
        evaluation,
        infrastructure_checks_passed=infrastructure,
        deterministic_replay_passed=replay,
    )
    assert verdict["terminal_status"] == expected_status
    assert verdict["passed"] is (expected_status == subject.PASS_STATUS)
    assert set(verdict["gates"]) == {
        "1_infrastructure_and_custody",
        *subject.SCIENTIFIC_GATE_NAMES,
        "10_exact_fresh_process_cache_only_replay",
    }


def test_verdict_rejects_nonbooleans_stale_gates_and_wrong_aggregate() -> None:
    arms, comparisons = _gate_inputs()
    evaluation = _evaluation(arms, comparisons)
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="verdict inputs"):
        subject.verdict_v1(
            evaluation,
            infrastructure_checks_passed=1,  # type: ignore[arg-type]
            deterministic_replay_passed=True,
        )

    stale_gate = copy.deepcopy(evaluation)
    stale_gate["gates"]["3_true_future_beats_task_action_only"]["passed"] = False
    stale_gate["scientific_gates_2_to_9_passed"] = False
    stale_gate["evaluation_identity_sha256"] = subject.evaluation_identity_v1(stale_gate)
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="scientific gates"):
        subject.verdict_v1(
            stale_gate,
            infrastructure_checks_passed=True,
            deterministic_replay_passed=True,
        )

    wrong_aggregate = copy.deepcopy(evaluation)
    wrong_aggregate["scientific_gates_2_to_9_passed"] = False
    wrong_aggregate["evaluation_identity_sha256"] = subject.evaluation_identity_v1(
        wrong_aggregate
    )
    with pytest.raises(subject.DenseVJEPAInterfaceError, match="aggregate"):
        subject.verdict_v1(
            wrong_aggregate,
            infrastructure_checks_passed=True,
            deterministic_replay_passed=True,
        )
