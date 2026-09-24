from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lewm.benchmarks import go2_dinov2_dense_shared_spatial_readout_calibration_v1 as dense
from scripts import run_go2_grounded_dense_dino_joint_jepa_v1 as base
from scripts import (
    run_go2_grounded_dense_dino_joint_jepa_v1_evaluation_integrity_replacement_v1
    as recovery,
)
from scripts.evaluate_go2_world_model_counterfactual_action_regret_v1 import (
    fit_action_specific_ridge_readouts_v1,
    task_conditioned_feature_v1,
)


def _synthetic_plan() -> SimpleNamespace:
    states = []
    for index in range(base.STATE_COUNT):
        ranks = tuple(
            float(((index * 5 + action * 7) % base.ACTION_COUNT) + 1)
            for action in range(base.ACTION_COUNT)
        )
        states.append(
            SimpleNamespace(
                relative_target_xy_body_m=(
                    float((index % 17) - 8) / 5.0,
                    float((index % 13) - 6) / 7.0,
                ),
                dense_ranks=ranks,
            )
        )
    return SimpleNamespace(identity_sha256="synthetic-plan", states=tuple(states))


def _fit_unchecked(plan: SimpleNamespace):
    features = np.stack(
        [
            task_conditioned_feature_v1(
                None, relative_target_xy_body_m=state.relative_target_xy_body_m
            )
            for state in plan.states
        ]
    )
    targets = np.stack(
        [
            np.asarray(state.dense_ranks, dtype=np.float64)
            / np.asarray(state.dense_ranks, dtype=np.float64).max()
            for state in plan.states
        ]
    )
    return fit_action_specific_ridge_readouts_v1(
        [features] * base.ACTION_COUNT,
        [targets[:, action] for action in range(base.ACTION_COUNT)],
        ridge_lambda=recovery.TASK_RIDGE_LAMBDA,
    )


def _checkpoint_payload(arm: str) -> dict[str, object]:
    trace = [
        {
            "update": 0,
            "all_finite": True,
            "normalized_physical_rank_regret": 0.24,
            "branch_retrieval_accuracy": 0.11 if arm == "joint_jepa_grounded" else 0.0,
            "successor_cosine_error": 0.26 if arm == "joint_jepa_grounded" else 0.0,
            "persistence_cosine_error": 0.26 if arm == "joint_jepa_grounded" else 0.0,
        },
        {
            "update": 400,
            "all_finite": True,
            "normalized_physical_rank_regret": 0.08 if arm == "joint_jepa_grounded" else 0.20,
            "branch_retrieval_accuracy": 0.37 if arm == "joint_jepa_grounded" else 0.0,
            "successor_cosine_error": 0.18 if arm == "joint_jepa_grounded" else 0.0,
            "persistence_cosine_error": 0.25 if arm == "joint_jepa_grounded" else 0.0,
        },
        {
            "update": 800,
            "all_finite": True,
            "normalized_physical_rank_regret": 0.03,
            "branch_retrieval_accuracy": 0.54 if arm == "joint_jepa_grounded" else 0.0,
            "successor_cosine_error": 0.12 if arm == "joint_jepa_grounded" else 0.0,
            "persistence_cosine_error": 0.23 if arm == "joint_jepa_grounded" else 0.0,
        },
    ]
    return {
        "schema": base.CHECKPOINT_SCHEMA,
        "arm": arm,
        "update": base.MAX_UPDATES,
        "model_seed": base.MODEL_SEED,
        "sampler_seed": base.SAMPLER_SEED,
        "config": base.runner_config_v1(),
        "train_identity_sha256": recovery.EXPECTED_TRAIN_RUNTIME_IDENTITY,
        "initial_model_identity_sha256": "a" * 64,
        "input_statistics": {
            "identity_sha256": "b" * 64,
            "value": torch.tensor([1.0]),
        },
        "outcome_statistics": {
            "identity_sha256": "c" * 64,
            "value": torch.tensor([2.0]),
        },
        "model_state_dict": {"weight": torch.tensor([3.0])},
        "optimizer_state_dict": {"state": {}, "param_groups": []},
        "trace": trace,
    }


def test_recovery_preregistration_binding_is_frozen() -> None:
    raw = recovery.PREREGISTRATION.read_bytes()
    assert len(raw) == recovery.PREREGISTRATION_BYTE_COUNT
    assert hashlib.sha256(raw).hexdigest() == recovery.PREREGISTRATION_SHA256


def test_predecessor_bindings_are_exact_and_distinct() -> None:
    bindings = recovery.predecessor_bindings_v1()
    assert set(bindings) == {
        "original_authority",
        "original_reservation",
        "original_terminal",
        "physical_only_update_800_checkpoint",
        "joint_jepa_update_800_checkpoint",
    }
    assert bindings["physical_only_update_800_checkpoint"]["sha256"] != bindings[
        "joint_jepa_update_800_checkpoint"
    ]["sha256"]
    assert all("sealed" not in value["path"].lower() for value in bindings.values())


def test_recovery_task_fit_is_exactly_the_legacy_algorithm_without_old_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _synthetic_plan()
    expected = _fit_unchecked(plan)
    monkeypatch.setattr(recovery, "EXPECTED_TRAIN_PLAN_IDENTITY", "synthetic-plan")
    monkeypatch.setattr(recovery, "EXPECTED_TASK_IDENTITY", expected.identity_sha256)
    actual = recovery.fit_current_task_action_only_v1(plan)
    assert recovery._readouts_exactly_equal(actual, expected)  # noqa: SLF001

    monkeypatch.setattr(dense, "EXPECTED_TASK_IDENTITY", expected.identity_sha256)
    legacy = dense.fit_task_action_only_v1(plan)
    assert recovery._readouts_exactly_equal(actual, legacy)  # noqa: SLF001
    assert all(head.solver == "primal" for head in actual.heads)
    assert all(head.training_rows == base.STATE_COUNT for head in actual.heads)


@pytest.mark.parametrize("mutation", ["plan", "goal", "rank"])
def test_recovery_task_fit_rejects_every_bound_input_change(
    mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _synthetic_plan()
    expected = _fit_unchecked(plan)
    monkeypatch.setattr(recovery, "EXPECTED_TRAIN_PLAN_IDENTITY", "synthetic-plan")
    monkeypatch.setattr(recovery, "EXPECTED_TASK_IDENTITY", expected.identity_sha256)
    if mutation == "plan":
        plan.identity_sha256 = "changed"
    elif mutation == "goal":
        plan.states[0].relative_target_xy_body_m = (9.0, 9.0)
    else:
        plan.states[0].dense_ranks = tuple(reversed(plan.states[0].dense_ranks))
    with pytest.raises(recovery.RecoveryError):
        recovery.fit_current_task_action_only_v1(plan)


def test_checkpoint_is_bound_deserialized_once_and_optimizer_is_discarded(
    tmp_path: Path,
) -> None:
    path = tmp_path / "joint.pt"
    torch.save(_checkpoint_payload("joint_jepa_grounded"), path)
    binding = base.file_binding_v1(path)
    loaded = recovery.read_checkpoint_once_v1(
        binding, expected_arm="joint_jepa_grounded"
    )
    assert loaded["update"] == 800
    assert "optimizer_state_dict" not in loaded
    changed = dict(binding)
    changed["sha256"] = "0" * 64
    with pytest.raises(recovery.RecoveryError, match="binding changed"):
        recovery.read_checkpoint_once_v1(
            changed, expected_arm="joint_jepa_grounded"
        )


def test_checkpoint_pair_and_train_futility_are_qualified() -> None:
    physical = _checkpoint_payload("physical_only_matched")
    joint = _checkpoint_payload("joint_jepa_grounded")
    physical.pop("optimizer_state_dict")
    joint.pop("optimizer_state_dict")
    evidence = recovery.qualify_checkpoints_v1(physical, joint)
    assert evidence["joint_update_400_futility"]["continue_to_update_800"] is True
    joint["trace"][1]["branch_retrieval_accuracy"] = 0.20
    with pytest.raises(recovery.RecoveryError, match="ineligible"):
        recovery.qualify_checkpoints_v1(physical, joint)


def test_original_checkpoint_directory_symlink_is_rejected_before_enumeration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    elsewhere = tmp_path / "elsewhere"
    attempt.mkdir()
    elsewhere.mkdir()
    (attempt / "reservation.json").write_text("{}")
    (attempt / "terminal.json").write_text("{}")
    (attempt / "checkpoints").symlink_to(elsewhere, target_is_directory=True)
    monkeypatch.setattr(recovery, "ORIGINAL_OUTPUT_ROOT", attempt)
    with pytest.raises(base.GroundedRunnerError, match="traverses a symlink"):
        recovery.validate_original_inventory_v1()


def test_recovery_access_audit_has_no_train_or_successor_rgb() -> None:
    ledger = base.AccessLedgerV1()
    ledger.load_receipts("train")
    ledger.open_role_index("train", "/train/index")
    for index in range(base.STATE_COUNT):
        ledger.open_state_receipt("train", f"/train/{index}")
    ledger.checkpoint("physical_only_matched")
    ledger.checkpoint("joint_jepa_grounded")
    ledger.load_receipts("eval")
    ledger.open_role_index("eval", "/eval/index")
    for index in range(base.STATE_COUNT):
        ledger.open_state_receipt("eval", f"/eval/{index}")
        for frame in range(base.CONTEXT_COUNT):
            ledger.open_rgb("eval", "context", f"eval-{index}-{frame}")
    audit = recovery.recovery_access_audit_v1(
        ledger,
        checkpoint_reads={
            "physical_only_matched": 1,
            "joint_jepa_grounded": 1,
        },
    )
    assert audit["rgb_opens"]["train_context"] == 0
    assert audit["rgb_opens"]["train_successor"] == 0
    assert audit["rgb_opens"]["eval_successor"] == 0
    assert audit["rgb_opens"]["eval_context"] == 384


def test_recovery_models_are_frozen_and_in_eval_mode() -> None:
    class FakeDino:
        @staticmethod
        def fresh_tail():
            return torch.nn.ModuleList(), torch.nn.Identity()

    class FakeModel(torch.nn.Module):
        def __init__(self, blocks, norm, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0]))

    payload = {"model_state_dict": {"weight": torch.tensor([4.0])}}
    model = recovery._model_from_payload_v1(  # noqa: SLF001
        payload,
        dino=FakeDino(),
        model_class=FakeModel,
        device=torch.device("cpu"),
    )
    assert model.training is False
    assert model.weight.item() == 4.0
    assert all(parameter.requires_grad is False for parameter in model.parameters())


def test_runner_contains_no_training_or_backward_path() -> None:
    source = Path(recovery.__file__).read_text()
    assert ".backward(" not in source
    assert "torch.optim." not in source
    assert "train_arm_v1(" not in source
    assert "execute_v1(original_authority" not in source
    assert "EXPECTED_TASK_EVAL_REGRET = 0.17441406250000002" in source


def test_main_writes_one_consumed_failure_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "attempt"
    authority_path = tmp_path / "authority.json"
    authority_path.write_text("{}")
    monkeypatch.setattr(
        recovery,
        "_load_authority_v1",
        lambda *args, **kwargs: {"output_root": str(output)},
    )

    def fail(_authority: object) -> object:
        output.mkdir()
        raise RuntimeError("sentinel")

    monkeypatch.setattr(recovery, "execute_v1", fail)
    with pytest.raises(RuntimeError, match="sentinel"):
        recovery.main(
            [
                "--authority",
                str(authority_path),
                "--expected-authority-sha256",
                "0" * 64,
                "--expected-authority-byte-count",
                "2",
            ]
        )
    terminal = base._strict_json_loads(  # noqa: SLF001
        (output / "terminal.json").read_bytes(), label="terminal"
    )
    assert terminal["status"] == "CONSUMED_TERMINAL_INFRASTRUCTURE_FAILURE"
    assert terminal["retry_authorized"] is False
    assert terminal["result_binding"] is None


def test_recovery_config_has_no_training_authority() -> None:
    assert recovery.config_v1()["training_allowed"] is False
    assert recovery.permissions_v1()["training_or_optimizer_access"] is False
    assert recovery.permissions_v1()["eval_successor_rgb_access"] is False
    assert recovery.permissions_v1()["retry_resume_overwrite"] is False
