from __future__ import annotations

from unittest import mock

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


def test_two_pass_objective_matches_full_unique_minimum_gradient():
    factual_actions = torch.tensor(((0, 0, 0), (0, 0, 2)), dtype=torch.long)
    encoded = torch.zeros((2, 3, 256, 192), dtype=torch.float32)
    targets = torch.zeros((2, 64), dtype=torch.long)
    target = torch.zeros((2, 64, 192), dtype=torch.float32)
    arm = FakeArm()
    with mock.patch.object(worker.base, "predict_from_shared_encoding", fake_predict), mock.patch.object(
        worker, "normalized_half_squared_token_energy_v1", fake_energy
    ):
        terms = worker._action_objective_two_pass(
            arm=arm, encoded_history=encoded, factual_actions=factual_actions,
            target_indices=targets, target=target, coefficient=1.0,
        )
        terms.total.backward()
    observed_gradient = arm.energy.grad.detach().clone()
    assert terms.selected_wrong_action_ids.tolist() == [1, 1]

    arm.energy.grad = None
    factual = arm.energy[factual_actions[:, 2]]
    grid = arm.energy[None, :].expand(2, -1)
    masked = grid.clone()
    masked[torch.arange(2), factual_actions[:, 2]] = torch.inf
    expected = factual.mean() + torch.relu(
        worker.ALIGNMENT_MARGIN + factual - masked.min(dim=1).values
    ).mean()
    expected.backward()
    assert terms.total.detach() == expected.detach()
    assert torch.equal(observed_gradient, arm.energy.grad)


def test_zero_coefficient_retains_only_factual_gradient_but_runs_same_route():
    actions = torch.tensor(((0, 0, 0), (0, 0, 2)), dtype=torch.long)
    arm = FakeArm()
    with mock.patch.object(worker.base, "predict_from_shared_encoding", fake_predict), mock.patch.object(
        worker, "normalized_half_squared_token_energy_v1", fake_energy
    ):
        terms = worker._action_objective_two_pass(
            arm=arm,
            encoded_history=torch.zeros((2, 3, 256, 192)),
            factual_actions=actions,
            target_indices=torch.zeros((2, 64), dtype=torch.long),
            target=torch.zeros((2, 64, 192)),
            coefficient=0.0,
        )
        terms.total.backward()
    assert terms.selected_wrong_action_ids.tolist() == [1, 1]
    assert torch.equal(
        arm.energy.grad,
        torch.tensor([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )


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
