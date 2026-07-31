from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import (
    execute_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1
    as executor,
)
from scripts import (
    launch_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1
    as launcher,
)
from scripts import (
    run_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1
    as training,
)


def _binding(path: str) -> dict[str, object]:
    return {"path": path, "file_sha256": "a" * 64, "byte_count": 1}


def _authority() -> dict[str, object]:
    core: dict[str, object] = {
        "schema": f"{executor.SCHEMA_PREFIX}_execution_authority_v1",
        "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_STAGED_ONE_SHOT",
        "scientific_payload_authorized": True,
        "one_shot": True,
        "maximum_updates": 1_000,
        "stage_a_updates": 500,
        "maximum_memory_presentations": 16_000,
        "maximum_physical_presentations": 8_000,
        "maximum_presentations": 24_000,
        "retry_authorized": False,
        "scientific_resume_authorized": False,
        "infrastructure_recovery_authorized": True,
        "certified_source_root": executor.CERTIFIED_SOURCE_ROOT,
        "output_root": executor.OUTPUT_ROOT_RELATIVE_PATH,
        "runtime_data_root": "/home/andrewknowles/Workspace/LeWMQuad-v3",
        "preregistration_commit": executor.PREREGISTRATION_COMMIT,
        "pinned_source_and_review_commit": "b" * 40,
        "selectors": {
            "executor_module": executor.__name__,
            "model_module": executor.MODEL_MODULE_NAME,
            "model_class": executor.MODEL_CLASS_NAME,
            "training_module": executor.TRAINING_MODULE_NAME,
            "evaluation_module": executor.EVALUATION_MODULE_NAME,
        },
        "runtime_inputs": {
            name: _binding(f"inputs/{name}.json")
            for name in executor.RUNTIME_INPUT_BINDING_NAMES
        },
        "clean_export_certification": _binding("docs/certification.json"),
    }
    return executor._content_bound(core)


def test_authority_binds_staged_caps_and_runtime_selectors() -> None:
    authority = _authority()
    assert executor.validate_future_execution_prerequisites_v1(authority) == authority
    changed = dict(authority)
    changed["maximum_memory_presentations"] = 16_001
    changed = executor._content_bound(changed)
    with pytest.raises(PermissionError):
        executor.validate_future_execution_prerequisites_v1(changed)


def test_reservation_is_one_shot_and_recovery_requires_snapshot(tmp_path: Path) -> None:
    parent = tmp_path / Path(executor.OUTPUT_ROOT_RELATIVE_PATH).parent
    parent.mkdir(parents=True)
    authority = _authority()
    reservation, recovery = executor.reserve_or_recover_attempt_v1(
        tmp_path, authority, created_utc="2026-07-31T00:00:00Z"
    )
    assert recovery is None
    assert reservation["attempt_consumed"] is True
    with pytest.raises(PermissionError, match="no complete exact snapshot"):
        executor.reserve_or_recover_attempt_v1(
            tmp_path, authority, created_utc="2026-07-31T00:01:00Z"
        )


@dataclass(frozen=True)
class _Route:
    gradient_norm: float
    scale: float


class _Target(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1), requires_grad=False)


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.online = torch.nn.Parameter(torch.ones(1))
        self.target = _Target()
        self.register_buffer("ema_update_count", torch.tensor(1, dtype=torch.long))

    def target_modules(self):
        return (self.target,)


def test_update_integrity_serializes_nested_route_dataclasses() -> None:
    accounting = training.JointTrainingAccountingV1(
        updates=1,
        presentations=24,
        physical_presentations=8,
        memory_presentations=16,
        physical_microbatch_graphs=2,
        memory_microbatch_graphs=8,
        autograd_grad_calls=14,
        optimizer_steps=1,
        ema_steps=1,
    )
    result = SimpleNamespace(
        accounting=accounting,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
        target_gradient_tensor_count=0,
        mean_losses={"total": 1.0},
        memory_diagnostics={"per_microbatch": (1.0, 0.5)},
        gradient_routes={"memory": _Route(1.0, 0.25)},
    )
    receipt = executor.validate_update_integrity_v1(
        SimpleNamespace(torch=torch), _Model(), result, update=1
    )
    assert receipt["passed"] is True
    assert receipt["gradient_routes"]["memory"] == {
        "gradient_norm": 1.0,
        "scale": 0.25,
    }
    executor._canonical_json_bytes(receipt)


def test_update_zero_gate_requires_prediction_identity_and_substrate() -> None:
    zero = SimpleNamespace(macro=(0.0, 0.0, 0.0, 0.0))
    observation = SimpleNamespace(
        update=0,
        temporal=SimpleNamespace(
            score=SimpleNamespace(macro=(1.0, 1.0, 1.0, 1.0)),
            persistence_lift=zero,
            action_lift=zero,
            history_lift=zero,
        ),
        safeguards=SimpleNamespace(
            integrity_pass=True,
            target_noncollapsed=True,
            online_noncollapsed=True,
        ),
        memory_state=SimpleNamespace(noncollapsed=True),
        substrate=SimpleNamespace(
            place_chance_multiple=2.1,
            place_scene_count_above_chance=6,
            target_place_rank=2.0,
        ),
    )
    receipt = {
        "temporal": {
            "integrity": {
                "checks": {"update_zero_controls_equal_persistence": True},
                "update_zero_max_control_prediction_delta": 0.0,
            }
        }
    }
    decision = executor.evaluate_update0_gate_v1(observation, receipt)
    assert decision.passed is True
    assert decision.observed["maximum_prediction_persistence_delta"] == 0.0

    receipt["temporal"]["integrity"]["checks"][
        "update_zero_controls_equal_persistence"
    ] = False
    failed = executor.evaluate_update0_gate_v1(observation, receipt)
    assert failed.passed is False
    assert "prediction_level_persistence_identity" in failed.failed_checks


def test_launcher_rejects_ambiguous_gpu_visibility_before_reservation() -> None:
    receipt = launcher.validate_pre_reservation_gpu_visibility_v1(
        {"HIP_VISIBLE_DEVICES": "0"}
    )
    assert receipt["passed"] is True
    with pytest.raises(PermissionError, match="HIP_VISIBLE_DEVICES=0"):
        launcher.validate_pre_reservation_gpu_visibility_v1({})
    with pytest.raises(PermissionError, match="conflicting selector"):
        launcher.validate_pre_reservation_gpu_visibility_v1(
            {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"}
        )


def test_exact_recovery_reuses_only_identical_write_once_artifacts(
    tmp_path: Path,
) -> None:
    delegate = launcher._BASE.V13WriteOncePublisher(tmp_path, executor)
    delegate.publish_json("metrics/update_500.json", {"value": 1})
    delegate.publish_bytes("snapshots/update_500.pt", b"exact-state")
    recovery = executor.ExactRecoveryReplayPublisherV1(delegate)

    replayed = recovery.publish_json("metrics/update_500.json", {"value": 1})
    assert replayed["value"]["value"] == 1
    assert recovery.publish_bytes(
        "snapshots/update_500.pt", b"exact-state"
    )["byte_count"] == len(b"exact-state")
    with pytest.raises(PermissionError, match="JSON replay changed"):
        recovery.publish_json("metrics/update_500.json", {"value": 2})
    with pytest.raises(PermissionError, match="byte replay changed"):
        recovery.publish_bytes("snapshots/update_500.pt", b"changed-state")


def test_launcher_without_authority_denies_before_reservation(capsys) -> None:
    assert launcher.main([]) == 4
    output = capsys.readouterr().out
    assert "DENIED_NO_FUTURE_AUTHORITY" in output
    assert "reservation_created" in output
