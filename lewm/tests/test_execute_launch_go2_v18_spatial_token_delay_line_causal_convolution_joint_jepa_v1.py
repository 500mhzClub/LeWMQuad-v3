from __future__ import annotations

import copy
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


def test_every_observation_uses_the_controls_producing_physical_panel() -> None:
    assert set(executor.PHYSICAL_OBSERVATION_ALIAS) == set(
        executor.OBSERVATION_UPDATES
    )
    assert set(executor.PHYSICAL_OBSERVATION_ALIAS.values()) == {400}


def _comparison_payload(call: int) -> dict[str, dict[str, int]]:
    return {name: {"call": call} for name in executor.CONTROL_NAMES}


def test_physical_alias_adapter_rescores_current_model_at_all_six_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace()
    calls: list[tuple[object, int, bool]] = []

    def observe(
        observed_runtime: object,
        model: object,
        *,
        update: int,
        integrity_pass: bool,
    ) -> dict[str, object]:
        assert observed_runtime is runtime
        cache = getattr(runtime, "causal_comparisons_v19", None)
        if cache is None:
            cache = {}
            runtime.causal_comparisons_v19 = cache
        assert cache == {}
        calls.append((model, update, integrity_pass))
        cache[update] = _comparison_payload(len(calls))
        return {"model": model, "call": len(calls)}

    monkeypatch.setattr(executor.physical_executor, "_observation_v13", observe)
    models = {update: object() for update in executor.OBSERVATION_UPDATES}
    returned = [
        executor._physical_observation_v1(
            runtime,
            models[update],
            outer_update=update,
            integrity_pass=True,
        )
        for update in executor.OBSERVATION_UPDATES
    ]
    assert [call[0] for call in calls] == list(models.values())
    assert [call[1] for call in calls] == [400] * 6
    assert all(call[2] is True for call in calls)
    assert [value["model"] for value in returned] == list(models.values())
    assert [value["call"] for value in returned] == list(range(1, 7))
    assert runtime.causal_comparisons_v19 == {}


def test_physical_alias_adapter_preserves_partial_state_when_observer_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = SimpleNamespace()
    payload = _comparison_payload(1)

    def fail_after_population(*args: object, **kwargs: object) -> object:
        runtime.causal_comparisons_v19 = {400: payload}
        raise LookupError("inherited observation failed")

    monkeypatch.setattr(
        executor.physical_executor,
        "_observation_v13",
        fail_after_population,
    )
    with pytest.raises(LookupError, match="inherited observation failed"):
        executor._physical_observation_v1(
            runtime,
            object(),
            outer_update=0,
            integrity_pass=True,
        )
    assert runtime.causal_comparisons_v19 == {400: payload}


@pytest.mark.parametrize(
    "initial_cache",
    (None, [], {400: _comparison_payload(0)}),
    ids=("none", "non_dict", "preexisting_alias"),
)
def test_physical_alias_adapter_rejects_invalid_pre_call_state(
    monkeypatch: pytest.MonkeyPatch,
    initial_cache: object,
) -> None:
    runtime = SimpleNamespace(causal_comparisons_v19=initial_cache)
    calls = 0

    def observe(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr(executor.physical_executor, "_observation_v13", observe)
    with pytest.raises(RuntimeError, match="malformed or nonempty before"):
        executor._physical_observation_v1(
            runtime,
            object(),
            outer_update=0,
            integrity_pass=True,
        )
    assert calls == 0


@pytest.mark.parametrize(
    "mode",
    ("missing", "wrong_key", "extra_key", "malformed_controls"),
)
def test_physical_alias_adapter_rejects_invalid_post_call_state(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    runtime = SimpleNamespace()

    def observe(*args: object, **kwargs: object) -> object:
        if mode == "missing":
            runtime.causal_comparisons_v19 = {}
        elif mode == "wrong_key":
            runtime.causal_comparisons_v19 = {100: _comparison_payload(1)}
        elif mode == "extra_key":
            runtime.causal_comparisons_v19 = {
                400: _comparison_payload(1),
                100: _comparison_payload(2),
            }
        else:
            runtime.causal_comparisons_v19 = {
                400: {executor.CONTROL_NAMES[0]: {"call": 1}}
            }
        return {"unchanged": True}

    monkeypatch.setattr(executor.physical_executor, "_observation_v13", observe)
    with pytest.raises(RuntimeError, match="alias cache"):
        executor._physical_observation_v1(
            runtime,
            object(),
            outer_update=0,
            integrity_pass=True,
        )
    if mode == "missing":
        assert runtime.causal_comparisons_v19 == {}
    else:
        assert runtime.causal_comparisons_v19


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
            target_noncollapsed=False,
            online_noncollapsed=False,
        ),
        memory_state=SimpleNamespace(noncollapsed=False),
        substrate=SimpleNamespace(
            place_chance_multiple=2.1,
            place_scene_count_above_chance=6,
            target_place_rank=2.0,
        ),
    )
    receipt = {
        "temporal": {
            "integrity": {
                "absolute_noncollapse_enforced": False,
                "checks": {
                    "target_state_finite": True,
                    "target_state_nonzero_scale": True,
                    "online_state_finite": True,
                    "online_state_nonzero_scale": True,
                    "memory_state_finite": True,
                    "memory_state_nonzero_scale": True,
                    "update_zero_controls_equal_persistence": True,
                },
                "update_zero_max_control_prediction_delta": 0.0,
            }
        }
    }
    decision = executor.evaluate_update0_gate_v1(observation, receipt)
    assert decision.passed is True
    assert decision.observed["maximum_prediction_persistence_delta"] == 0.0
    assert decision.observed["absolute_noncollapse_enforced"] is False
    assert decision.observed["target_noncollapsed_diagnostic"] is False
    assert decision.observed["online_noncollapsed_diagnostic"] is False
    assert decision.observed["memory_noncollapsed_diagnostic"] is False

    receipt["temporal"]["integrity"]["checks"][
        "memory_state_nonzero_scale"
    ] = False
    structurally_failed = executor.evaluate_update0_gate_v1(observation, receipt)
    assert structurally_failed.passed is False
    assert "memory_finite_nonzero" in structurally_failed.failed_checks
    receipt["temporal"]["integrity"]["checks"][
        "memory_state_nonzero_scale"
    ] = True

    receipt["temporal"]["integrity"]["checks"][
        "update_zero_controls_equal_persistence"
    ] = False
    failed = executor.evaluate_update0_gate_v1(observation, receipt)
    assert failed.passed is False
    assert "prediction_level_persistence_identity" in failed.failed_checks


def test_absolute_noncollapse_is_first_enforced_at_update_250() -> None:
    safeguards = SimpleNamespace(
        integrity_pass=True,
        gradient_accounting_pass=True,
        target_noncollapsed=False,
        online_noncollapsed=False,
    )
    memory_state = SimpleNamespace(
        noncollapsed=False,
        participation_rank_ratio=0.02,
        near_zero_fraction=0.10,
    )
    update_100 = SimpleNamespace(
        update=100,
        safeguards=safeguards,
        memory_state=memory_state,
    )
    assert executor.evaluate_observation_integrity_v1(update_100) is None

    update_250 = SimpleNamespace(
        update=250,
        safeguards=safeguards,
        memory_state=memory_state,
    )
    failed = executor.evaluate_observation_integrity_v1(update_250)
    assert failed is not None
    assert failed.passed is False
    assert set(failed.failed_checks) == {
        "target_noncollapsed",
        "online_noncollapsed",
        "memory_noncollapsed",
    }


def test_frozen_nested_v25_physical_builder_accepts_v18_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v25_builder = executor.v5_launcher._V25_LAUNCHER._build_one_microbatch_v25
    v25_globals = v25_builder.__globals__
    v24_builder = v25_globals["_INHERITED_BUILD_ONE_MICROBATCH_V24"]
    v24_globals = v24_builder.__globals__
    v23_builder = v24_globals["_INHERITED_BUILD_ONE_MICROBATCH_V23"]
    v23_globals = v23_builder.__globals__
    v21_builder = v23_globals["_INHERITED_BUILD_ONE_MICROBATCH_V21"]
    v21_globals = v21_builder.__globals__

    monkeypatch.setitem(
        v25_globals,
        "preflight_schedule_per_row_persistence_contrastive_v25",
        lambda _runtime: {"passed": True},
    )
    monkeypatch.setitem(
        v24_globals,
        "preflight_schedule_predictor_core_protected_survival_v24",
        lambda _runtime: {"passed": True},
    )
    monkeypatch.setitem(
        v23_globals,
        "preflight_schedule_state_residual_survival_v23",
        lambda _runtime: {"passed": True},
    )
    monkeypatch.setitem(
        v21_globals,
        "preflight_schedule_negative_rows_v21",
        lambda _runtime: {"passed": True},
    )
    monkeypatch.setitem(
        v21_globals,
        "negative_rows_from_train_metadata_v21",
        lambda _runtime, _indices: (1, 2, 3, 0),
    )
    base = {
        key: torch.zeros((4, 1), dtype=torch.float32)
        for key in training.REQUIRED_BATCH_KEYS
    }
    monkeypatch.setitem(
        v21_globals,
        "_ORIGINAL_BUILD_ONE_MICROBATCH_V13",
        lambda **_kwargs: dict(base),
    )
    runtime = SimpleNamespace(
        training_module=training,
        torch=torch,
        action_prior_m=(0.0,) * 9,
    )

    result = v25_builder(
        runtime=runtime,
        indices=(0, 1, 2, 3),
        stage="synthetic_schema_only",
    )

    assert tuple(result) == training.REQUIRED_BATCH_KEYS_V25
    assert tuple(result)[: len(base)] == training.REQUIRED_BATCH_KEYS
    assert tuple(result)[-2:] == (
        training.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        training.ACTION_PRIOR_M_KEY_V23,
    )
    assert result[
        training.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
    ].tolist() == [1, 2, 3, 0]
    prior = result[training.ACTION_PRIOR_M_KEY_V23]
    assert tuple(prior.shape) == (9,)
    assert prior.dtype == torch.float32
    assert prior.requires_grad is False


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


def _access_runtime_triplet(*, populated: bool):
    physical_record = {
        "path": "raw/file.bin",
        "file_sha256": "1" * 64,
        "byte_count": 10,
        "kind": "raw_supervision",
        "roles": ["authority", "train"] if populated else ["authority"],
        "arms": ["shared", "correct"] if populated else ["shared"],
        "stages": ["input_validation", "training"] if populated else ["input_validation"],
    }
    runtime = SimpleNamespace(
        raw_inputs=SimpleNamespace(
            consumed={"raw/file.bin": copy.deepcopy(physical_record)}
        ),
        _access_consumed_count=1,
        _access_opened_roles=("authority",),
    )
    h6_loader = SimpleNamespace(
        _access={"rgb_open_success_count": 7 if populated else 0}
    )
    h6_runtime = SimpleNamespace(_loader=h6_loader)
    h6_runtime._require_loader = lambda: h6_runtime._loader
    local_record = {
        "path": "scene/frame.png",
        "file_sha256": "2" * 64,
        "byte_count": 20,
        "role": "train",
        "row_index": 4,
        "leaf": "scene/frame.png",
    }
    local_loader = SimpleNamespace(
        _consumed=(
            {"scene/frame.png": copy.deepcopy(local_record)} if populated else {}
        ),
        _tensor_requests=5 if populated else 0,
        _open_attempts=5 if populated else 0,
        _open_successes=5 if populated else 0,
        _decode_successes=5 if populated else 0,
        _byte_count=100 if populated else 0,
    )
    role_runtime = SimpleNamespace(
        _local_loader=local_loader,
        _place_reference_counts={
            "attempt": 3 if populated else 0,
            "sha256_verified": 3 if populated else 0,
            "success": 3 if populated else 0,
            "failure": 0,
        },
        _place_loader_calls=1 if populated else 0,
        _place_loaded_row_keys=(
            {("checkpoint_selection", 3)} if populated else set()
        ),
        _place_rows={("checkpoint_selection", 3): object()},
    )
    return runtime, h6_runtime, role_runtime


def _add_post_snapshot_access(runtime, h6_runtime, role_runtime) -> None:
    runtime.raw_inputs.consumed["raw/file.bin"]["stages"].append("post_recovery")
    h6_runtime._loader._access["rgb_open_success_count"] += 2
    role_runtime._local_loader._tensor_requests += 2
    role_runtime._local_loader._open_attempts += 2
    role_runtime._local_loader._open_successes += 2
    role_runtime._local_loader._decode_successes += 2
    role_runtime._local_loader._byte_count += 40
    role_runtime._place_reference_counts["attempt"] += 1
    role_runtime._place_reference_counts["sha256_verified"] += 1
    role_runtime._place_reference_counts["success"] += 1
    role_runtime._place_loader_calls += 1


def test_recovery_restores_cumulative_access_ledgers_exactly() -> None:
    uninterrupted = _access_runtime_triplet(populated=True)
    snapshot_access = executor._capture_exact_access_state_v1(*uninterrupted)

    recovered = _access_runtime_triplet(populated=False)
    executor._restore_exact_access_state_v1(*recovered, snapshot_access)
    assert executor._capture_exact_access_state_v1(*recovered) == snapshot_access

    _add_post_snapshot_access(*uninterrupted)
    _add_post_snapshot_access(*recovered)
    assert executor._capture_exact_access_state_v1(
        *recovered
    ) == executor._capture_exact_access_state_v1(*uninterrupted)


def test_recovered_snapshot_metadata_binding_is_complete() -> None:
    value = executor._content_bound({"schema": "synthetic_snapshot_binding"})
    binding = executor._content_bound_json_artifact_binding_v1(
        "snapshots/update_1000.binding.json", value
    )
    assert set(binding) == {
        "path",
        "file_sha256",
        "byte_count",
        "content_sha256",
    }
    assert binding["content_sha256"] == value["content_sha256"]
    assert len(binding["file_sha256"]) == 64
    assert binding["byte_count"] > 0


def test_launcher_without_authority_denies_before_reservation(capsys) -> None:
    assert launcher.main([]) == 4
    output = capsys.readouterr().out
    assert "DENIED_NO_FUTURE_AUTHORITY" in output
    assert "reservation_created" in output
