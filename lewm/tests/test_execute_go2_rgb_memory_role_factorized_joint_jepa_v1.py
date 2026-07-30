from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import torch

from scripts import execute_go2_rgb_memory_role_factorized_joint_jepa_v1 as executor


def _authority() -> dict:
    binding = {
        "path": ".generated/synthetic.jsonl",
        "file_sha256": "a" * 64,
        "byte_count": 1,
    }
    return executor._content_bound(
        {
            "schema": f"{executor.SCHEMA_PREFIX}_future_execution_authority_v1",
            "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT",
            "scientific_payload_authorized": True,
            "one_shot": True,
            "maximum_updates": 400,
            "maximum_presentations": 12_800,
            "retry_authorized": False,
            "resume_authorized": False,
            "certified_source_root": executor.CERTIFIED_SOURCE_ROOT,
            "output_root": executor.OUTPUT_ROOT_RELATIVE_PATH,
            "preregistration_commit": executor.PREREGISTRATION_COMMIT,
            "split_integrity_amendment_commit": (
                executor.SPLIT_INTEGRITY_AMENDMENT_COMMIT
            ),
            "runtime_data_root": str(ROOT),
            "selectors": {
                "executor_module": executor.__name__,
                "model_module": executor.MODEL_MODULE_NAME,
                "model_class": executor.MODEL_CLASS_NAME,
                "training_module": executor.TRAINING_MODULE_NAME,
                "evaluation_module": executor.EVALUATION_MODULE_NAME,
            },
            "clean_export_certification": {
                "path": executor.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
                "file_sha256": "b" * 64,
                "byte_count": 1,
                "content_sha256": "c" * 64,
            },
            "runtime_inputs": {
                name: dict(binding) for name in executor.RUNTIME_INPUT_BINDING_NAMES
            },
        }
    )


def test_authority_and_reservation_are_exactly_one_shot(tmp_path: Path) -> None:
    authority = _authority()
    runtime_inputs = dict(authority["runtime_inputs"])
    runtime_inputs["raw_manifest"] = {
        **runtime_inputs["raw_manifest"],
        "content_sha256": "d" * 64,
    }
    authority = executor._content_bound(
        {**authority, "runtime_inputs": runtime_inputs}
    )
    assert executor.validate_future_execution_prerequisites_v1(authority) == authority
    reservation = executor.reserve_attempt_v1(
        tmp_path, authority, created_utc="2026-07-30T00:00:00Z"
    )
    output = tmp_path / executor.OUTPUT_ROOT_RELATIVE_PATH
    assert stat.S_IMODE(os.lstat(output).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(output / "reservation.json").st_mode) == 0o444
    assert executor.validate_attempt_reservation_v1(reservation) == reservation
    assert reservation["maximum_updates"] == 400
    assert reservation["maximum_presentations"] == 12_800
    with pytest.raises(FileExistsError):
        executor.reserve_attempt_v1(
            tmp_path, authority, created_utc="2026-07-30T00:00:01Z"
        )

    changed = dict(authority)
    changed["maximum_presentations"] = 12_832
    changed = executor._content_bound(changed)
    with pytest.raises(PermissionError):
        executor.validate_future_execution_prerequisites_v1(changed)

    malformed = dict(authority)
    malformed_inputs = dict(authority["runtime_inputs"])
    malformed_inputs["raw_manifest"] = {
        **malformed_inputs["raw_manifest"],
        "content_sha256": "not-a-sha256",
    }
    malformed["runtime_inputs"] = malformed_inputs
    with pytest.raises(TypeError):
        executor.validate_future_execution_prerequisites_v1(
            executor._content_bound(malformed)
        )


def test_exception_terminalizer_quarantines_partial_checkpoint(tmp_path: Path) -> None:
    authority = _authority()
    reservation = executor.reserve_attempt_v1(
        tmp_path, authority, created_utc="2026-07-30T00:00:00Z"
    )
    output = tmp_path / executor.OUTPUT_ROOT_RELATIVE_PATH
    raw = b"synthetic partial checkpoint"
    checkpoint = output / "checkpoint_update_400.pt"
    checkpoint.write_bytes(raw)
    checkpoint.chmod(0o444)
    binding = {
        "path": checkpoint.name,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }
    failure = executor.terminalize_failure_v1(
        output,
        reservation,
        stage="publish_pass_checkpoint",
        error=RuntimeError("synthetic publication failure"),
        created_utc="2026-07-30T00:00:01Z",
        partial_checkpoint_binding=binding,
        failure_context={"last_completed_update": 17},
    )
    assert failure["checkpoint"] == binding
    assert failure["checkpoint_published"] is True
    assert failure["checkpoint_quarantined"] is True
    assert failure["checkpoint_access_authorized"] is False
    assert failure["failure_context"] == {"last_completed_update": 17}
    assert failure["retry_authorized"] is False
    assert failure["resume_authorized"] is False
    assert stat.S_IMODE(os.lstat(checkpoint).st_mode) == 0o000


def test_private_restart_state_is_exact_and_never_self_authorizes_resume() -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    accounting = {"updates": 100, "presentations": 3_200}
    raw, metadata = executor._serialize_private_restart_state_v3(
        SimpleNamespace(torch=torch),
        model,
        optimizer,
        accounting,
        _authority(),
        update=100,
    )
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    assert state["schema"].endswith("_private_restart_state_v1")
    assert state["update"] == 100
    assert state["accounting"] == accounting
    assert state["model_module"] == executor.MODEL_MODULE_NAME
    assert state["model_class"] == executor.MODEL_CLASS_NAME
    assert state["private_restart_state"] is True
    assert state["scientific_promotion"] is False
    assert state["resume_authorized"] is False
    assert metadata["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["resume_authorized"] is False
    with pytest.raises(ValueError, match="not registered"):
        executor._serialize_private_restart_state_v3(
            SimpleNamespace(torch=torch),
            model,
            optimizer,
            accounting,
            _authority(),
            update=101,
        )


@pytest.mark.parametrize("update", (100, 400))
def test_exception_terminalizer_receipts_orphan_restart_state(
    tmp_path: Path, update: int
) -> None:
    authority = _authority()
    reservation = executor.reserve_attempt_v1(
        tmp_path, authority, created_utc="2026-07-30T00:00:00Z"
    )
    output = tmp_path / executor.OUTPUT_ROOT_RELATIVE_PATH
    restart = output / f"private_restart/update_{update}.pt"
    restart.parent.mkdir(mode=0o700)
    raw = b"published state before metadata publication failed"
    restart.write_bytes(raw)
    restart.chmod(0o444)

    failure = executor.terminalize_failure_v1(
        output,
        reservation,
        stage=f"publish_private_restart_update_{update}",
        error=RuntimeError("synthetic metadata publication failure"),
        created_utc="2026-07-30T00:00:01Z",
    )

    assert failure["status"] == "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME"
    assert failure["private_restart_state_published"] is True
    assert failure["private_restart_state_hash_read_count"] == 1
    assert failure["private_restart_state_deserialized"] is False
    assert failure["private_restart_resume_authorized"] is False
    assert failure["private_restart_states"] == [
        {
            "update": update,
            "state": {
                "path": f"private_restart/update_{update}.pt",
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            },
            "immutable_mode": True,
            "state_deserialized": False,
            "resume_authorized": False,
        }
    ]


def test_engine_exception_publishes_complete_in_memory_failure_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority()
    reservation = executor.reserve_attempt_v1(
        tmp_path, authority, created_utc="2026-07-30T00:00:00Z"
    )

    class RoleRuntime:
        closed = False

        @staticmethod
        def failure_access_snapshot():
            return {"rgb_open_count": 0}

        def close(self):
            self.closed = True

    role_runtime = RoleRuntime()
    monkeypatch.setattr(
        executor,
        "load_memory_role_runtime_v1",
        lambda *args, **kwargs: role_runtime,
    )
    runtime = SimpleNamespace(
        runtime_data_root=ROOT,
        pairs={
            "train": tuple({"scene_id": f"train_{index}"} for index in range(72)),
            "checkpoint_selection": tuple(
                {"scene_id": f"selection_{index}"} for index in range(8)
            ),
        },
        initialize_model_v13=lambda: (_ for _ in ()).throw(
            RuntimeError("synthetic initialization failure")
        ),
        access_receipt_v13=lambda: {"opened_roles": ["authority", "index"]},
    )
    publisher = SimpleNamespace(
        output_root=tmp_path / executor.OUTPUT_ROOT_RELATIVE_PATH
    )
    failure = executor.run_future_authorized_engine_v1(
        authority=authority,
        reservation=reservation,
        runtime=runtime,
        publisher=publisher,
    )

    assert failure["status"] == "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME"
    context = failure["failure_context"]
    assert context["last_completed_update"] == 0
    assert context["accounting"] is None
    assert context["trace_event_count"] == 0
    assert context["physical_access"]["status"] == "CAPTURED"
    assert context["role_access"]["status"] == "CAPTURED"
    assert role_runtime.closed is True


def test_engine_stops_at_update100_and_keeps_nonresumable_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority()
    reservation = executor.reserve_attempt_v1(
        tmp_path, authority, created_utc="2026-07-30T00:00:00Z"
    )

    class Publisher:
        output_root = tmp_path / executor.OUTPUT_ROOT_RELATIVE_PATH

        def __init__(self):
            self.raw: dict[str, bytes] = {}
            self.values: dict[str, dict] = {}

        @staticmethod
        def _binding(path: str, raw: bytes, *, content_sha256=None):
            value = {
                "path": path,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "byte_count": len(raw),
            }
            if content_sha256 is not None:
                value["content_sha256"] = content_sha256
            return value

        def publish_bytes(self, path, raw):
            assert path not in self.raw
            self.raw[path] = raw
            return self._binding(path, raw)

        def publish_json(self, path, core):
            assert path not in self.values
            value = executor._content_bound(core)
            raw = executor._canonical_json_bytes(value) + b"\n"
            self.values[path] = value
            return {
                "value": value,
                "binding": self._binding(
                    path, raw, content_sha256=value["content_sha256"]
                ),
            }

    class RoleRuntime:
        closed = False

        @staticmethod
        def preflight_receipt():
            return {"passed": True}

        @staticmethod
        def build_local_train_microbatches(update, device):
            return (update, device)

        @staticmethod
        def build_place_train_microbatches(update, device):
            return (update, device)

        @staticmethod
        def evaluate_role_metrics(model, *, update, device):
            return {
                "place": {"update": update},
                "local": {"update": update},
                "integrity": {"passed": True},
            }

        @staticmethod
        def terminal_access_receipt():
            return {"passed": True}

        @staticmethod
        def failure_access_snapshot():
            return {"passed": True}

        def close(self):
            self.closed = True

    role_runtime = RoleRuntime()
    monkeypatch.setattr(
        executor, "load_memory_role_runtime_v1", lambda *args, **kwargs: role_runtime
    )
    monkeypatch.setattr(
        executor.physical_executor,
        "_derive_initial_structural_integrity_v13",
        lambda runtime, model: {"passed": True},
    )
    monkeypatch.setattr(
        executor.physical_executor,
        "_observation_v13",
        lambda runtime, model, *, update, integrity_pass: {
            "integrity_pass": True,
            "physical": {
                "margin_count": 189,
                "passed_margin_count": 60,
                "rough_motion": {"depth_p95_m": 1.0},
            },
            "controls": {},
        },
    )
    monkeypatch.setattr(
        executor,
        "validate_update_integrity_v1",
        lambda runtime, model, result, *, update: {
            "schema": "synthetic_update_integrity",
            "update": update,
            "passed": True,
        },
    )
    from scripts import evaluate_go2_rgb_memory_role_factorized_joint_jepa_v1 as evaluation

    monkeypatch.setattr(
        evaluation,
        "evaluate_update100_continuation_gate_v3",
        lambda **kwargs: {"update": 100, "passed": False},
    )

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    class Training:
        calls = 0

        @classmethod
        def joint_training_update_v1(
            cls, model, optimizer, physical, local, place, *, accounting
        ):
            cls.calls += 1
            return SimpleNamespace(
                accounting={
                    "updates": cls.calls,
                    "presentations": 32 * cls.calls,
                }
            )

    runtime = SimpleNamespace(
        runtime_data_root=ROOT,
        pairs={
            "train": ({"scene_id": "train"},),
            "checkpoint_selection": ({"scene_id": "selection"},),
        },
        schedule=tuple(range(1_600)),
        device="cpu",
        torch=torch,
        training_module=Training,
        initialize_model_v13=lambda: (model, optimizer, {"fresh": True}),
        build_microbatches_v13=lambda indices, *, update: (indices, update),
        terminal_access_receipt_v13=lambda: {"passed": True},
        access_receipt_v13=lambda: {"passed": True},
    )
    publisher = Publisher()
    failure = executor.run_future_authorized_engine_v1(
        authority=authority,
        reservation=reservation,
        runtime=runtime,
        publisher=publisher,
    )

    assert failure["status"] == (
        "FAIL_SCIENTIFIC_UPDATE100_CONTINUATION_GATE_TERMINAL"
    )
    assert failure["terminal_update"] == 100
    assert failure["accounting"] == {"updates": 100, "presentations": 3_200}
    assert failure["private_restart_state_published"] is True
    assert failure["private_restart_states"][0]["resume_authorized"] is False
    assert Training.calls == 100
    assert "private_restart/update_100.pt" in publisher.raw
    assert "private_restart/update_400.pt" not in publisher.raw
    assert "metrics/update_400.json" not in publisher.values
    assert role_runtime.closed is True


def _route() -> dict:
    return {
        "preclip_l2": 1.0,
        "applied_scale": 1.0,
        "parameter_tensor_count": 1,
        "absent_tensor_gradient_count": 0,
    }


def test_update_integrity_accepts_only_exact_three_route_accounting() -> None:
    update = 3
    multipliers = {
        "updates": 1,
        "presentations": 32,
        "physical_presentations": 16,
        "local_presentations": 8,
        "place_presentations": 8,
        "rgb_decodes": 72,
        "physical_rgb_decodes": 32,
        "local_rgb_decodes": 16,
        "place_rgb_decodes": 24,
        "online_rgb_encodings": 48,
        "ema_target_rgb_encodings": 24,
        "physical_microbatch_graphs": 4,
        "local_microbatch_graphs": 2,
        "place_microbatch_graphs": 1,
        "autograd_grad_calls": 15,
        "optimizer_steps": 1,
        "ema_steps": 1,
    }
    route_names = (
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
        "predictor_core_protected_survival_output",
        "immediate_action_local_control",
        "same_place_retrieval_key",
    )
    result = SimpleNamespace(
        accounting={name: update * value for name, value in multipliers.items()},
        gradient_routes={name: _route() for name in route_names},
        mean_losses={
            name: 1.0
            for name in (
                "S",
                "U",
                "R",
                "O",
                "N",
                "C",
                "J24",
                "L",
                "local",
                "place",
                "total",
            )
        },
        local_diagnostics={
            "mechanism": "immediate_action_local_control",
            "correct_energy_per_row": (0.1,) * 8,
            "wrong_energy_per_row": (0.2,) * 8,
        },
        place_diagnostics={
            "mechanism": "same_place_retrieval_key",
            "positive_energy_per_row": (0.1,) * 8,
            "negative_energy_per_row": (0.2,) * 8,
        },
        target_gradient_tensor_count=0,
        optimizer_steps_this_update=1,
        ema_steps_this_update=1,
    )
    model = SimpleNamespace(
        ema_update_count=torch.tensor(update),
        target_modules=lambda: (),
        state_dict=lambda: {"online": torch.tensor((1.0,))},
    )
    receipt = executor.validate_update_integrity_v1(
        SimpleNamespace(torch=torch), model, result, update=update
    )
    assert receipt["passed"] is True
    assert receipt["accounting"]["presentations"] == 96

    result.accounting = {**result.accounting, "local_presentations": 23}
    with pytest.raises(RuntimeError, match="accounting"):
        executor.validate_update_integrity_v1(
            SimpleNamespace(torch=torch), model, result, update=update
        )


def test_source_path_filter_rejects_protected_roots_without_open(tmp_path: Path) -> None:
    for relative in (
        ".generated/runtime.json",
        "sealed/secret.py",
        "sealed_test.json",
        "held_out/maze.py",
        "data/copy.py",
    ):
        assert executor._protected_source_path(relative) is True
    assert executor._protected_source_path("lewm/datasets/safe.py") is False


def test_role_training_schedule_uses_exact_noncycling_eight_row_windows() -> None:
    local_seen: list[int] = []
    place_seen: list[int] = []

    class LocalLoader:
        @staticmethod
        def load_pair(row):
            local_seen.append(row.index)
            return {
                "current_rgb": torch.zeros(3, 112, 112),
                "next_rgb": torch.ones(3, 112, 112),
                "action": row.index % 9,
            }

        @staticmethod
        def access_receipt():
            return {"rgb_open_attempt_count": 0}

    class PlaceData:
        @staticmethod
        def load_rgb_triplet(_root, row, *, record_reference_access):
            place_seen.append(row.index)
            for role in ("anchor", "positive", "negative"):
                record_reference_access(role, "attempt")
                record_reference_access(role, "sha256_verified")
                record_reference_access(role, "success")
            return SimpleNamespace(
                anchor_rgb=torch.zeros(3, 112, 112),
                positive_rgb=torch.ones(3, 112, 112),
                negative_rgb=torch.full((3, 112, 112), 2.0),
            )

    tensor_core = SimpleNamespace(_runtime_apis=lambda: (torch,))
    training = SimpleNamespace(
        v25=SimpleNamespace(_tensor_core=tensor_core),
        LOCAL_CURRENT_RGB_KEY_V1="current_rgb",
        LOCAL_NEXT_RGB_KEY_V1="next_rgb",
        LOCAL_ACTION_KEY_V1="action",
        REQUIRED_LOCAL_BATCH_KEYS_V1=("current_rgb", "next_rgb", "action"),
        PLACE_ANCHOR_RGB_KEY_V1="anchor_rgb",
        PLACE_POSITIVE_RGB_KEY_V1="positive_rgb",
        PLACE_NEGATIVE_RGB_KEY_V1="negative_rgb",
        REQUIRED_PLACE_BATCH_KEYS_V1=(
            "anchor_rgb",
            "positive_rgb",
            "negative_rgb",
        ),
    )
    runtime = object.__new__(executor.MemoryRoleRuntimeV1)
    runtime._closed = False
    runtime.runtime_data_root = ROOT
    runtime.training = training
    runtime._local_loader = LocalLoader()
    runtime.local_train_rows = tuple(
        SimpleNamespace(index=index) for index in range(3_200)
    )
    runtime.place_train_rows = tuple(
        SimpleNamespace(role="train", index=index) for index in range(3_200)
    )
    runtime._place_rows = {
        (row.role, row.index): row for row in runtime.place_train_rows
    }
    runtime.place_data = PlaceData()
    runtime._place_loader_calls = 0
    runtime._place_loaded_row_keys = set()
    runtime._place_reference_counts = {
        "attempt": 0,
        "sha256_verified": 0,
        "success": 0,
        "failure": 0,
    }

    local_batches = runtime.build_local_train_microbatches(400, "cpu")
    place_batches = runtime.build_place_train_microbatches(400, "cpu")
    assert local_seen == list(range(3_192, 3_200))
    assert place_seen == list(range(3_192, 3_200))
    assert len(local_batches) == len(place_batches) == 2
    assert all(batch["current_rgb"].shape == (4, 3, 112, 112) for batch in local_batches)
    assert all(batch["anchor_rgb"].shape == (4, 3, 112, 112) for batch in place_batches)
    assert runtime._place_loader_calls == 8
    successful = runtime.failure_access_snapshot()
    assert successful["place_triplet_loader_call_count"] == 8
    assert successful["place_rgb_reference_attempt_count"] == 24
    assert successful["place_rgb_sha256_verified_per_access_count"] == 24
    assert successful["place_rgb_reference_success_count"] == 24
    assert successful["place_rgb_reference_failure_count"] == 0

    class PartialFailurePlaceData:
        @staticmethod
        def load_rgb_triplet(_root, _row, *, record_reference_access):
            record_reference_access("anchor", "attempt")
            record_reference_access("anchor", "sha256_verified")
            record_reference_access("anchor", "success")
            record_reference_access("positive", "attempt")
            record_reference_access("positive", "sha256_verified")
            record_reference_access("positive", "failure")
            raise RuntimeError("synthetic decode failure")

    runtime.place_data = PartialFailurePlaceData()
    with pytest.raises(RuntimeError, match="synthetic"):
        runtime._load_place_triplet(runtime.place_train_rows[0])
    partial = runtime.failure_access_snapshot()
    assert partial["place_triplet_loader_call_count"] == 8
    assert partial["place_rgb_reference_attempt_count"] == 26
    assert partial["place_rgb_sha256_verified_per_access_count"] == 26
    assert partial["place_rgb_reference_success_count"] == 25
    assert partial["place_rgb_reference_failure_count"] == 1
    assert partial["place_unique_row_count_opened"] == 8
    runtime._place_reference_counts["attempt"] += 1
    with pytest.raises(RuntimeError, match="accounting is incomplete"):
        runtime.failure_access_snapshot()

    with pytest.raises(PermissionError):
        runtime.build_local_train_microbatches(401, "cpu")
    with pytest.raises(PermissionError):
        runtime.build_place_train_microbatches(0, "cpu")
