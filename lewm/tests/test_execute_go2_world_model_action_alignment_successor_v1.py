from __future__ import annotations

import copy
import gc
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


def test_replacement_plan_has_zero_normalized_scientific_differences():
    original = json.loads(
        Path(
            "docs/lewm_go2_world_model_action_alignment_successor_v1_"
            "plan_2026-08-01.json"
        ).read_text()
    )
    replacement = json.loads(worker.PLAN_PATH.read_text())
    for key in (
        "development_only", "citable_as_original_factual_learnability_claim",
        "authorizes_execution", "route", "arms", "objective", "action_margin",
        "training", "paired_decision", "reuse", "caps",
    ):
        assert replacement[key] == original[key]
    original_attempt = {
        key: value for key, value in original["attempt"].items() if key != "id"
    }
    replacement_attempt = {
        key: value
        for key, value in replacement["attempt"].items()
        if key not in {"id", "original_attempt_runtime_reuse"}
    }
    assert replacement_attempt == original_attempt
    assert set(original["forbidden"]) <= set(replacement["forbidden"])
    assert replacement["integrity_replacement"]["scientific_fields_changed"] is False
    assert replacement["integrity_replacement"]["tolerance_relaxed"] is False
    assert all(replacement["science_identity"].values())


def test_replacement_plan_validator_rejects_every_finality_escape():
    plan = json.loads(worker.PLAN_PATH.read_text())
    worker._validate_replacement_plan(plan)
    mutations = (
        ("maximum_integrity_replacements_after_this", 1),
        ("failed_attempt_state_reused", True),
        ("scientific_fields_changed", True),
        ("tolerance_relaxed", True),
    )
    for field, value in mutations:
        changed = copy.deepcopy(plan)
        changed["integrity_replacement"][field] = value
        with pytest.raises(worker.AlignmentWorkerError, match="replacement plan"):
            worker._validate_replacement_plan(changed)
    changed = copy.deepcopy(plan)
    changed["forbidden"].remove("further_integrity_replacement")
    with pytest.raises(worker.AlignmentWorkerError, match="replacement plan"):
        worker._validate_replacement_plan(changed)
    changed = copy.deepcopy(plan)
    changed["attempt"]["original_attempt_runtime_reuse"] = True
    with pytest.raises(worker.AlignmentWorkerError, match="replacement plan"):
        worker._validate_replacement_plan(changed)


def test_schedule_is_exact_v3_schedule():
    observed, audit = worker.base.build_bound_training_schedule()
    expected, expected_audit = worker.base.build_bound_training_schedule()
    assert torch.equal(observed, expected)
    assert audit == expected_audit
    assert audit["presentations"] == 179_200


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
        "reservation.json", "baseline_update_000700.pt",
        "alignment_update_000700.pt", "metrics.pt", "result.json",
    }


def test_checker_rejects_nonreproducing_concurrent_baseline():
    with pytest.raises(checker.AlignmentCheckError, match="did not reproduce"):
        checker._require_passing_baseline_anchor_audit(
            {"exact_within_1e_15": False, "checks": {"anchor": False}}
        )
    checker._require_passing_baseline_anchor_audit(
        {"exact_within_1e_15": True, "checks": {"anchor": True}}
    )
