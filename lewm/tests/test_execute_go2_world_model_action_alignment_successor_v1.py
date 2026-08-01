from __future__ import annotations

import copy
import gc
import io
import json
from pathlib import Path
from unittest import mock
import weakref

import pytest
import torch

from scripts import execute_go2_world_model_action_alignment_successor_v1 as worker
from scripts import check_go2_world_model_action_alignment_successor_v1 as checker


class FakeArm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.energy = torch.nn.Parameter(
            torch.tensor([0.20, 0.10, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
        )


class TinyRestoreArm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor_position = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        self.temporal_gru = torch.nn.Linear(2, 2, bias=False)


def fake_predict(arm, encoded, actions, target_indices, *, candidate_blind):
    del encoded, target_indices, candidate_blind
    values = arm.energy[actions[:, 2]]
    raw = values[:, None, None].expand(-1, 64, 192)
    return worker.base.ArmPrediction(raw=raw, normalized=raw, recurrent_memory=raw)


def fake_energy(raw, target):
    del target
    return raw[:, 0, 0]


def b32_actions(*, factual_id: int | None = None) -> torch.Tensor:
    rows = torch.arange(worker.CANDIDATE_SCAN_BATCH_ROWS, dtype=torch.long)
    factual = (
        torch.full_like(rows, factual_id)
        if factual_id is not None
        else (rows * 5 + 2) % worker.ACTION_COUNT
    )
    return torch.stack(((rows + 1) % 9, (rows + 4) % 9, factual), dim=1)


def objective_inputs(actions: torch.Tensor):
    batch = worker.CANDIDATE_SCAN_BATCH_ROWS
    assert actions.shape == (batch, 3)
    return {
        "encoded_history": torch.zeros((batch, 3, 256, 192)),
        "factual_actions": actions,
        "target_indices": torch.zeros((batch, 64), dtype=torch.long),
        "target": torch.zeros((batch, 64, 192)),
    }


def test_two_pass_objective_matches_full_unique_minimum_gradient():
    factual_actions = b32_actions()
    arm = FakeArm()
    with mock.patch.object(worker.base, "predict_from_shared_encoding", fake_predict), mock.patch.object(
        worker, "normalized_half_squared_token_energy_v1", fake_energy
    ):
        terms = worker._action_objective_two_pass(
            arm=arm, coefficient=1.0, **objective_inputs(factual_actions)
        )
        assert arm.energy.grad is None
        terms.total.backward()
    observed_gradient = arm.energy.grad.detach().clone()
    expected_selected = [0 if value == 1 else 1 for value in factual_actions[:, 2]]
    assert terms.selected_wrong_action_ids.tolist() == expected_selected
    assert terms.scan_energy.requires_grad is False
    assert terms.scan_energy.grad_fn is None

    arm.energy.grad = None
    factual = arm.energy[factual_actions[:, 2]]
    grid = arm.energy[None, :].expand(worker.CANDIDATE_SCAN_BATCH_ROWS, -1)
    masked = grid.clone()
    masked[torch.arange(worker.CANDIDATE_SCAN_BATCH_ROWS), factual_actions[:, 2]] = torch.inf
    expected = factual.mean() + torch.relu(
        worker.ALIGNMENT_MARGIN + factual - masked.min(dim=1).values
    ).mean()
    expected.backward()
    assert terms.total.detach() == expected.detach()
    assert torch.equal(observed_gradient, arm.energy.grad)


def test_zero_coefficient_retains_only_factual_gradient_but_runs_same_route():
    actions = b32_actions()
    arm = FakeArm()
    with mock.patch.object(worker.base, "predict_from_shared_encoding", fake_predict), mock.patch.object(
        worker, "normalized_half_squared_token_energy_v1", fake_energy
    ):
        terms = worker._action_objective_two_pass(
            arm=arm, coefficient=0.0, **objective_inputs(actions)
        )
        terms.total.backward()
    expected = torch.bincount(actions[:, 2], minlength=9).float() / len(actions)
    assert torch.equal(arm.energy.grad, expected)


def test_row_stable_scan_has_exact_coverage_order_shape_and_dispatch():
    actions = b32_actions()
    arm = FakeArm()
    calls: list[dict[str, object]] = []

    def traced_predict(arm, encoded, selected_actions, target_indices, *, candidate_blind):
        calls.append(
            {
                "grad_enabled": torch.is_grad_enabled(),
                "batch": encoded.shape[0],
                "row_witness": encoded[:, 0, 0, 0].detach().clone(),
                "candidate_ids": selected_actions[:, 2].detach().clone(),
            }
        )
        return fake_predict(
            arm, encoded, selected_actions, target_indices,
            candidate_blind=candidate_blind,
        )

    inputs = objective_inputs(actions)
    inputs["encoded_history"][:, 0, 0, 0] = torch.arange(
        worker.CANDIDATE_SCAN_BATCH_ROWS
    )
    with mock.patch.object(
        worker.base, "predict_from_shared_encoding", traced_predict
    ), mock.patch.object(worker, "normalized_half_squared_token_energy_v1", fake_energy):
        terms = worker._action_objective_two_pass(
            arm=arm, coefficient=1.0, **inputs
        )

    assert len(calls) == 10
    expected_rows = torch.arange(worker.CANDIDATE_SCAN_BATCH_ROWS).float()
    assert all(call["grad_enabled"] is True for call in calls)
    assert all(call["batch"] == worker.CANDIDATE_SCAN_BATCH_ROWS for call in calls)
    assert all(torch.equal(call["row_witness"], expected_rows) for call in calls)
    observed_wrong = torch.stack(
        [call["candidate_ids"] for call in calls[:8]], dim=1
    )
    expected_wrong = torch.tensor(
        [
            [candidate for candidate in range(9) if candidate != int(factual)]
            for factual in actions[:, 2]
        ],
        dtype=torch.long,
    )
    assert torch.equal(observed_wrong, expected_wrong)
    assert torch.equal(calls[8]["candidate_ids"], actions[:, 2])
    assert torch.equal(calls[9]["candidate_ids"], terms.selected_wrong_action_ids)
    assert terms.scan_energy.requires_grad is False
    assert arm.energy.grad is None


def test_absolute_action_columns_resolve_ties_to_lowest_id():
    actions = b32_actions(factual_id=8)
    arm = FakeArm()
    with torch.no_grad():
        arm.energy.copy_(torch.tensor([0.4, 0.1, 0.5, 0.1, 0.6, 0.7, 0.8, 0.9, 0.3]))
    with mock.patch.object(worker.base, "predict_from_shared_encoding", fake_predict), mock.patch.object(
        worker, "normalized_half_squared_token_energy_v1", fake_energy
    ):
        terms = worker._action_objective_two_pass(
            arm=arm, coefficient=1.0, **objective_inputs(actions)
        )
    assert terms.selected_wrong_action_ids.tolist() == [1] * len(actions)


def test_scan_rejects_no_grad_dispatch_and_wrong_batch_shape():
    actions = b32_actions()
    arm = FakeArm()
    with torch.no_grad(), pytest.raises(
        worker.AlignmentWorkerError, match="requires autograd dispatch"
    ):
        worker._action_objective_two_pass(
            arm=arm, coefficient=1.0, **objective_inputs(actions)
        )
    shortened = actions[:-1]
    inputs = objective_inputs(actions)
    inputs = {name: value[:-1] for name, value in inputs.items()}
    with pytest.raises(worker.AlignmentWorkerError, match="input shape"):
        worker._action_objective_two_pass(
            arm=arm, coefficient=1.0, **inputs
        )


def test_each_temporary_scan_graph_is_released_before_the_next_slot():
    actions = b32_actions()
    arm = FakeArm()
    raw_references: list[weakref.ReferenceType[torch.Tensor]] = []
    call_count = 0

    def tracked_predict(arm, encoded, selected_actions, target_indices, *, candidate_blind):
        nonlocal call_count
        if call_count <= 8:
            gc.collect()
            assert not any(reference() is not None for reference in raw_references)
        call_count += 1
        prediction = fake_predict(
            arm, encoded, selected_actions, target_indices,
            candidate_blind=candidate_blind,
        )
        if call_count <= 8:
            raw_references.append(weakref.ref(prediction.raw))
        return prediction

    with mock.patch.object(
        worker.base, "predict_from_shared_encoding", tracked_predict
    ), mock.patch.object(worker, "normalized_half_squared_token_energy_v1", fake_energy):
        terms = worker._action_objective_two_pass(
            arm=arm, coefficient=1.0, **objective_inputs(actions)
        )
    gc.collect()
    assert call_count == 10
    assert not any(reference() is not None for reference in raw_references)
    assert terms.scan_energy.grad_fn is None


def test_exact_model_route_has_no_dropout_or_cross_row_state():
    config = worker.base.temporal_model.RGBRecurrentPatchMemoryTemporalJepaV1Config()
    assert config.predictor_dropout == 0.0
    assert config.encoder_dropout == 0.0
    assert config.context_length == 3
    assert config.temporal_hidden_dim == config.feature_dim


def test_continuation_plan_validator_accepts_exact_plan_and_rejects_escapes():
    plan = json.loads(worker.PLAN_PATH.read_text())
    worker._validate_continuation_plan(plan)
    for section, field, value in (
        ("continuation", "optimizer_reset", True),
        ("training", "global_schedule_updates", 901),
        ("attempt", "further_continuation", True),
        ("finality", "meaningful_progress_requires_separate_preregistration", False),
    ):
        changed = copy.deepcopy(plan)
        changed[section][field] = value
        with pytest.raises(worker.AlignmentWorkerError, match="continuation plan"):
            worker._validate_continuation_plan(changed)
    changed = copy.deepcopy(plan)
    changed["arms"][0]["u700_snapshot"] = changed["arms"][1]["u700_snapshot"]
    with pytest.raises(worker.AlignmentWorkerError, match="continuation plan"):
        worker._validate_continuation_plan(changed)


def test_schedule_is_exact_900_with_completed_700_prefix():
    observed, audit = worker.base.build_bound_training_schedule(updates=900)
    prefix, prefix_audit = worker.base.build_bound_training_schedule(updates=700)
    assert torch.equal(observed[:700], prefix)
    assert audit["presentations"] == 230_400
    assert prefix_audit["presentations"] == 179_200


def test_snapshot_restore_validates_and_reloads_model_and_own_adamw_moments():
    arm = TinyRestoreArm()
    optimizer, _partition = worker.base.build_arm_optimizer(arm)
    worker.base._set_optimizer_learning_rates(
        optimizer, fraction=worker.base.learning_rate_fraction(worker.START_UPDATE)
    )
    for parameter in arm.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(worker.START_UPDATE), dtype=torch.float32),
            "exp_avg": torch.full_like(parameter, 0.25),
            "exp_avg_sq": torch.full_like(parameter, 0.5),
        }
    _schedule, schedule_audit = worker.base.build_bound_training_schedule(
        updates=worker.START_UPDATE
    )
    substrate_receipt = {"synthetic": True}
    snapshot = {
        "schema": worker.PREDECESSOR_SNAPSHOT_SCHEMA,
        "status": "COMPLETE",
        "arm": "baseline",
        "alignment_coefficient": 0.0,
        "update": worker.START_UPDATE,
        "authority_binding": worker.EXPECTED_EVIDENCE_BINDINGS[
            "completed_successor_authority"
        ],
        "reservation_binding": worker.EXPECTED_EVIDENCE_BINDINGS[
            "completed_successor_reservation"
        ],
        "substrate": substrate_receipt,
        "schedule": schedule_audit,
        "arm_state_dict": worker.base._clone_cpu(arm.state_dict()),
        "optimizer_state_dict": worker.base._clone_cpu(optimizer.state_dict()),
    }
    expected_model_hash = worker.base.module_state_sha256(arm)
    stream = io.BytesIO()
    torch.save(snapshot, stream)
    optimizer.state.clear()
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    with torch.no_grad():
        for parameter in arm.parameters():
            parameter.zero_()
    with mock.patch.object(
        worker.custody, "_read_absolute_regular_once", return_value=stream.getvalue()
    ):
        receipt = worker._load_and_restore_u700_snapshot(
            arm_name="baseline",
            arm=arm,
            optimizer=optimizer,
            substrate_receipt=substrate_receipt,
            schedule_u700_audit=schedule_audit,
        )
    assert worker.base.module_state_sha256(arm) == expected_model_hash
    assert receipt["optimizer_parameter_count"] == len(tuple(arm.parameters()))
    assert receipt["optimizer_step"] == worker.START_UPDATE
    assert all(
        float(state["step"]) == worker.START_UPDATE
        for state in optimizer.state.values()
    )


def test_reused_pack_binding_rejects_digest_mutation():
    observed = {
        "manifest_sha256": worker.EXPECTED_INPUT_BINDINGS["pack_manifest"]["file_sha256"],
        "frames": {
            "byte_count": worker.EXPECTED_INPUT_BINDINGS["pack_train_frames"]["byte_count"],
            "sha256": "0" * 64,
        },
        "actions": {
            "byte_count": worker.EXPECTED_INPUT_BINDINGS["pack_train_actions"]["byte_count"],
            "sha256": worker.EXPECTED_INPUT_BINDINGS["pack_train_actions"]["file_sha256"],
        },
        "metadata": {
            "byte_count": worker.EXPECTED_INPUT_BINDINGS["pack_train_metadata"]["byte_count"],
            "sha256": worker.EXPECTED_INPUT_BINDINGS["pack_train_metadata"]["file_sha256"],
        },
    }
    with pytest.raises(worker.AlignmentWorkerError):
        worker._validate_reused_pack_binding("train", observed)


def test_claim_and_access_contract_forbid_expansion():
    assert worker.CLAIM_BOUNDARY[-1].startswith("no planning")
    assert all("sealed" not in path and "heldout" not in path for path in worker.REQUIRED_SOURCE_PATHS.values())
    assert worker.EXPECTED_SUCCESS_FILES_BEFORE_CHECKER == {
        "reservation.json", "baseline_update_000900.pt",
        "alignment_update_000900.pt", "metrics.pt", "result.json",
    }


def test_rank_covariance_round_trip_supports_independent_effective_rank():
    generator = torch.Generator().manual_seed(17)
    with mock.patch.object(worker, "EXPECTED_VALIDATION_ROWS", 4), mock.patch.object(
        worker, "RANK_TOKEN_COUNT", 3
    ), mock.patch.object(worker, "RANK_FEATURE_DIMENSION", 5):
        target = torch.randn((4, 3, 5), generator=generator)
        prediction = torch.randn((4, 3, 5), generator=generator)
        target_covariance = worker._rank_covariance(target)
        prediction_covariance = worker._rank_covariance(prediction)
        target_rank = worker._effective_rank_from_covariance(target_covariance)
        prediction_rank = worker._effective_rank_from_covariance(
            prediction_covariance
        )
        expected_target, _variance = worker.scaled.effective_rank(target)
        expected_prediction, _variance = worker.scaled.effective_rank(prediction)
        assert torch.equal(target_covariance, target_covariance.T)
        assert torch.equal(prediction_covariance, prediction_covariance.T)
        assert target_rank == pytest.approx(expected_target, rel=0.0, abs=1.0e-12)
        assert prediction_rank == pytest.approx(
            expected_prediction, rel=0.0, abs=1.0e-12
        )
        assert prediction_rank / target_rank == pytest.approx(
            expected_prediction / expected_target, rel=0.0, abs=1.0e-12
        )
        assert checker._effective_rank_from_covariance(
            prediction_covariance, label="synthetic"
        ) == pytest.approx(expected_prediction, rel=0.0, abs=1.0e-12)


def test_checker_rank_covariance_rejects_shape_and_symmetry_changes():
    with mock.patch.object(worker, "RANK_FEATURE_DIMENSION", 3):
        with pytest.raises(checker.AlignmentCheckError, match="covariance changed"):
            checker._effective_rank_from_covariance(
                torch.eye(2, dtype=torch.float64), label="bad shape"
            )
        nonsymmetric = torch.eye(3, dtype=torch.float64)
        nonsymmetric[0, 1] = 0.5
        with pytest.raises(checker.AlignmentCheckError, match="covariance changed"):
            checker._effective_rank_from_covariance(
                nonsymmetric, label="nonsymmetric"
            )


def test_independent_reviewer_identity_and_disclosure_review_are_enforced():
    review = {
        "reviewer": {"identity": worker.INDEPENDENT_SOURCE_REVIEWER_IDENTITY},
        "verification": {
            "all_focused_tests_passed": True,
            "focused_tests": {"passed": 24, "failed": 0},
            "restoration_contract_reviewed": True,
            "absolute_progress_decision_reviewed": True,
            "schedule_prefix_and_absolute_update_reviewed": True,
            "preauthority_identity_read_disclosure_and_exclusions_reviewed": True,
            "governance_correction_reviewed": True,
            "no_real_runtime_payload_opened": True,
        },
        "custody": {
            "runtime_payloads_opened": False,
            "sealed_or_heldout_opened": False,
        },
    }
    worker._validate_independent_source_reviewer_evidence(review)
    for identity in worker.PREAUTHORITY_REVIEW_EXCLUDED_IDENTITIES:
        changed = copy.deepcopy(review)
        changed["reviewer"]["identity"] = identity
        with pytest.raises(worker.AlignmentWorkerError, match="review evidence"):
            worker._validate_independent_source_reviewer_evidence(changed)
    changed = copy.deepcopy(review)
    changed["verification"][
        "preauthority_identity_read_disclosure_and_exclusions_reviewed"
    ] = False
    with pytest.raises(worker.AlignmentWorkerError, match="review evidence"):
        worker._validate_independent_source_reviewer_evidence(changed)


def test_u700_replay_anchors_bind_points_and_lower_quantiles():
    for name in worker.ARM_NAMES:
        anchor = worker.PUBLIC_U700_REPLAY_ANCHORS[name]
        assert len(anchor["per_action_points"]) == worker.ACTION_COUNT
        assert len(anchor["per_action_q05"]) == worker.ACTION_COUNT
