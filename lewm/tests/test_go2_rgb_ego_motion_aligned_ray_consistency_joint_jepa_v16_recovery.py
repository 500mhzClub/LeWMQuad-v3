"""CPU-only exactness proofs for the narrow V16 recovery seam."""
from __future__ import annotations

import copy
import hashlib
import io
from typing import Any

import pytest
import torch
from torch import nn

from scripts import (
    go2_rgb_ego_motion_aligned_ray_consistency_joint_jepa_v16_recovery
    as recovery,
)


class _TinyLift(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.evidence_head = nn.Linear(1, 1, bias=False)
        self.free_projection = nn.Linear(1, 1, bias=False)


class _TinyV16Model(nn.Module):
    """Small model with the exact public V13 parameter-role prefixes."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.bev_lift = _TinyLift()
        self.semantic_head = nn.Linear(1, 1, bias=False)
        self.predictor = nn.Linear(1, 1, bias=False)
        self.target_encoder = nn.Linear(1, 1, bias=False)
        self.target_bev_lift = _TinyLift()
        for parameter in (
            *self.target_encoder.parameters(),
            *self.target_bev_lift.parameters(),
        ):
            parameter.requires_grad_(False)
        with torch.no_grad():
            self.target_encoder.weight.copy_(self.encoder.weight)
            self.target_bev_lift.evidence_head.weight.copy_(
                self.bev_lift.evidence_head.weight
            )
            self.target_bev_lift.free_projection.weight.copy_(
                self.bev_lift.free_projection.weight
            )
        self.register_buffer(
            "target_hard_sync_count", torch.tensor(1, dtype=torch.int64)
        )
        self.register_buffer(
            "ema_update_count", torch.tensor(0, dtype=torch.int64)
        )
        self.config = {"tiny_width": 1, "target_ema_momentum": 0.9}

    @torch.no_grad()
    def update_target(self) -> None:
        pairs = (
            (self.encoder.weight, self.target_encoder.weight),
            (
                self.bev_lift.evidence_head.weight,
                self.target_bev_lift.evidence_head.weight,
            ),
            (
                self.bev_lift.free_projection.weight,
                self.target_bev_lift.free_projection.weight,
            ),
        )
        for online, target in pairs:
            target.mul_(0.9).add_(online, alpha=0.1)
        self.ema_update_count.add_(1)


def _new_training_state() -> tuple[_TinyV16Model, Any]:
    model = _TinyV16Model()
    partition = recovery.v13_training.partition_parameters_v13(model)
    optimizer = recovery.v13_training.build_frozen_optimizer_v13(partition)
    return model, optimizer


def _one_step(model: _TinyV16Model, optimizer: Any) -> None:
    optimizer.zero_grad(set_to_none=True)
    sample = torch.rand((), dtype=torch.float32)
    online = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    loss = sum(
        ((index + 1.0) * parameter * sample).square().sum()
        for index, parameter in enumerate(online)
    )
    loss.backward()
    optimizer.step()
    model.update_target()


def _accounting(update: int) -> Any:
    accounting_type = recovery.v13_training.JointTrainingAccountingV13
    return accounting_type(
        updates=update,
        presentations=update * 16,
        microbatch_graphs=update * 4,
        backward_calls=update * 8,
        camera_route_grad_calls=update * 4,
        joint_route_grad_calls=update * 4,
        camera_frame_objectives=update * 32,
        optimizer_steps=update,
        ema_steps=update,
        predictor_forwards=update * 4,
        predictor_objectives=update * 4,
    )


def _schedule() -> list[int]:
    return [index % recovery.TRAIN_PAIR_COUNT_V16 for index in range(16_000)]


def _schedule_receipt() -> dict[str, Any]:
    return {
        "schema": "synthetic_v16_schedule_receipt_v1",
        "presentation_count": 16_000,
        "schedule_regeneration_count": 0,
    }


def _authority() -> dict[str, Any]:
    return {
        "schema": "synthetic_v16_authority_v1",
        "preregistration_commit": recovery.PREREGISTRATION_COMMIT_V16,
        "frozen_source_and_review_commit": "1" * 40,
        "recursive_source_closure_manifest_sha256": "2" * 64,
        "execution_binding_commit": "3" * 40,
        "output_root": ".generated/synthetic_v16/attempt_v1",
        "scientific_payload_authorized": True,
    }


def _trace(update: int) -> list[dict[str, Any]]:
    return [
        {
            "schema": "synthetic_v16_trace_row_v1",
            "event": "recovery_milestone",
            "update": update,
        }
    ]


def _metric_bindings(update: int) -> list[dict[str, Any]]:
    return [
        {
            "path": f"metrics/update_{update}.json",
            "file_sha256": "4" * 64,
            "byte_count": 1,
        }
    ]


def _serialize(model: _TinyV16Model, optimizer: Any) -> tuple[bytes, dict]:
    return recovery.serialize_recovery_checkpoint_v16(
        torch,
        model,
        optimizer,
        _accounting(400),
        update=400,
        schedule=_schedule(),
        schedule_receipt=_schedule_receipt(),
        authority=_authority(),
        trace=_trace(400),
        metric_bindings=_metric_bindings(400),
        access_receipt={"opened_roles": ["train", "checkpoint_selection"]},
        consumed_inputs={
            "train/example.rgb": {
                "file_sha256": "5" * 64,
                "byte_count": 1,
                "roles": ["train"],
            }
        },
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right)
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_cpu_save_restore_matches_uninterrupted_next_step_exactly() -> None:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.manual_seed(20_260_730)
        model, optimizer = _new_training_state()
        for _ in range(400):
            _one_step(model, optimizer)

        model_before = copy.deepcopy(model.state_dict())
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        rng_before = torch.random.get_rng_state().clone()
        raw, binding = _serialize(model, optimizer)
        _assert_nested_equal(model.state_dict(), model_before)
        _assert_nested_equal(optimizer.state_dict(), optimizer_before)
        assert torch.equal(torch.random.get_rng_state(), rng_before)

        _one_step(model, optimizer)
        uninterrupted_model = copy.deepcopy(model.state_dict())
        uninterrupted_optimizer = copy.deepcopy(optimizer.state_dict())
        uninterrupted_rng = torch.random.get_rng_state().clone()

        restored_model, restored_optimizer = _new_training_state()
        restored = recovery.restore_recovery_checkpoint_v16(
            raw,
            restored_model,
            restored_optimizer,
            torch_module=torch,
            binding=binding,
            expected_update=400,
            schedule=_schedule(),
            schedule_receipt=_schedule_receipt(),
            authority=_authority(),
        )
        assert restored["completed_update"] == 400
        assert restored["next_update"] == 401
        assert restored["presentation_cursor"] == 6_400
        assert restored["accounting"] == _accounting(400)

        _one_step(restored_model, restored_optimizer)
        _assert_nested_equal(restored_model.state_dict(), uninterrupted_model)
        _assert_nested_equal(
            restored_optimizer.state_dict(), uninterrupted_optimizer
        )
        assert torch.equal(torch.random.get_rng_state(), uninterrupted_rng)
        assert int(restored_model.ema_update_count) == 401
        assert _accounting(401).updates == restored["next_update"]
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.fixture(scope="module")
def checkpoint() -> tuple[bytes, dict[str, Any]]:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.manual_seed(20_260_731)
        model, optimizer = _new_training_state()
        for _ in range(400):
            _one_step(model, optimizer)
        return _serialize(model, optimizer)
    finally:
        torch.random.set_rng_state(caller_rng)


def test_validation_rejects_orphan_schedule_source_and_update_mismatch(
    checkpoint: tuple[bytes, dict[str, Any]],
) -> None:
    raw, binding = checkpoint
    common = {
        "torch_module": torch,
        "expected_update": 400,
        "schedule": _schedule(),
        "schedule_receipt": _schedule_receipt(),
        "authority": _authority(),
    }
    with pytest.raises(PermissionError, match="orphan"):
        recovery.validate_recovery_checkpoint_v16(
            raw, binding=None, **common
        )

    changed_schedule = _schedule()
    changed_schedule[0] = (changed_schedule[0] + 1) % 4_262
    with pytest.raises(PermissionError, match="schedule"):
        recovery.validate_recovery_checkpoint_v16(
            raw,
            binding=binding,
            **{**common, "schedule": changed_schedule},
        )

    changed_authority = _authority()
    changed_authority["frozen_source_and_review_commit"] = "6" * 40
    with pytest.raises(PermissionError, match="source"):
        recovery.validate_recovery_checkpoint_v16(
            raw,
            binding=binding,
            **{**common, "authority": changed_authority},
        )

    with pytest.raises(PermissionError):
        recovery.validate_recovery_checkpoint_v16(
            raw,
            binding=binding,
            **{**common, "expected_update": 1_000},
        )


def test_update1000_checkpoint_has_next_update1001_and_full_cursor(
    checkpoint: tuple[bytes, dict[str, Any]],
) -> None:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        raw400, binding400 = checkpoint
        model, optimizer = _new_training_state()
        recovery.restore_recovery_checkpoint_v16(
            raw400,
            model,
            optimizer,
            torch_module=torch,
            binding=binding400,
            expected_update=400,
            schedule=_schedule(),
            schedule_receipt=_schedule_receipt(),
            authority=_authority(),
        )
        for _ in range(600):
            _one_step(model, optimizer)
        raw1000, binding1000 = recovery.serialize_recovery_checkpoint_v16(
            torch,
            model,
            optimizer,
            _accounting(1_000),
            update=1_000,
            schedule=_schedule(),
            schedule_receipt=_schedule_receipt(),
            authority=_authority(),
            trace=_trace(1_000),
            metric_bindings=_metric_bindings(1_000),
            access_receipt={"opened_roles": ["train", "checkpoint_selection"]},
            consumed_inputs={
                "train/example.rgb": {
                    "file_sha256": "5" * 64,
                    "byte_count": 1,
                    "roles": ["train"],
                }
            },
        )
        payload = recovery.validate_recovery_checkpoint_v16(
            raw1000,
            torch_module=torch,
            binding=binding1000,
            expected_update=1_000,
            schedule=_schedule(),
            schedule_receipt=_schedule_receipt(),
            authority=_authority(),
        )
        assert payload["next_update"] == 1_001
        assert payload["presentation_cursor"] == 16_000
        assert (
            payload["metadata"]["schedule"]["consumed_prefix_sha256"]
            == payload["metadata"]["schedule"]["full_schedule_sha256"]
        )
    finally:
        torch.random.set_rng_state(caller_rng)


@pytest.mark.parametrize("mutation", ("cursor", "accounting", "ema", "optimizer"))
def test_validation_rejects_internal_state_mismatch(
    checkpoint: tuple[bytes, dict[str, Any]], mutation: str
) -> None:
    raw, _ = checkpoint
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if mutation == "cursor":
        payload["presentation_cursor"] -= 16
    elif mutation == "accounting":
        payload["accounting"]["updates"] -= 1
    elif mutation == "ema":
        payload["model_state_dict"]["ema_update_count"].sub_(1)
    else:
        first = next(iter(payload["optimizer_state_dict"]["state"].values()))
        first["step"].sub_(1)
    stream = io.BytesIO()
    torch.save(payload, stream)
    changed_raw = stream.getvalue()
    changed_binding = recovery.build_recovery_binding_v16(
        changed_raw, payload["metadata"]
    )
    with pytest.raises(PermissionError):
        recovery.validate_recovery_checkpoint_v16(
            changed_raw,
            torch_module=torch,
            binding=changed_binding,
            expected_update=400,
            schedule=_schedule(),
            schedule_receipt=_schedule_receipt(),
            authority=_authority(),
        )


def test_payload_then_binding_publication_is_write_once(
    checkpoint: tuple[bytes, dict[str, Any]],
) -> None:
    raw, binding = checkpoint
    files: dict[str, bytes] = {}
    order: list[str] = []

    def publish(path: str, value: bytes) -> dict[str, Any]:
        if path in files:
            raise FileExistsError(path)
        files[path] = value
        order.append(path)
        return {
            "path": path,
            "file_sha256": hashlib.sha256(value).hexdigest(),
            "byte_count": len(value),
        }

    receipt = recovery.publish_recovery_checkpoint_v16(raw, binding, publish)
    assert order == [
        "recovery/checkpoint_update_400.pt",
        "recovery/checkpoint_update_400.binding.json",
    ]
    assert receipt["value"] == binding
    with pytest.raises(FileExistsError):
        recovery.publish_recovery_checkpoint_v16(raw, binding, publish)
