from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "scripts/launch_go2_rgb_object_space_explicit_plan_discounted_successor_"
    "state_joint_jepa_v27.py"
)
V26_AUTHORITY_PATH = ROOT / (
    "docs/lewm_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_execution_authorization_2026-07-30.json"
)
PRIVATE_PREFIXES = (
    "_lewm_v27_explicit_plan_private_v25_launcher",
    "_lewm_v25_per_row_temporal_private_v24_launcher",
    "_lewm_v24_core_protected_private_v23_launcher",
    "_lewm_v23_scene_action_private_v21_launcher",
    "_lewm_v21_scene_innovation_private_v20_launcher",
)


def _load(name: str):
    for loaded in tuple(sys.modules):
        if loaded.startswith(PRIVATE_PREFIXES):
            sys.modules.pop(loaded, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _content_bound(module, core):
    value = dict(core)
    value["content_sha256"] = hashlib.sha256(
        module._canonical_json_bytes_v27(value)
    ).hexdigest()
    return value


def _authority(module):
    inherited = json.loads(V26_AUTHORITY_PATH.read_bytes())
    runtime_inputs = {
        name: inherited["runtime_inputs"][name]
        for name in module.PHYSICAL_RUNTIME_INPUT_NAMES
    }
    runtime_inputs.update(
        {
            "h6_train_index": dict(module.H6_TRAIN_BINDING),
            "h6_validation_index": dict(module.H6_VALIDATION_BINDING),
        }
    )
    return _content_bound(
        module,
        {
            "schema": module.AUTHORITY_SCHEMA,
            "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT",
            "scientific_payload_authorized": True,
            "one_shot": True,
            "maximum_updates": 400,
            "maximum_presentations": 12_800,
            "retry_authorized": False,
            "resume_authorized": False,
            "certified_source_root": module.CERTIFIED_SOURCE_ROOT,
            "output_root": module.OUTPUT_ROOT_RELATIVE_PATH,
            "preregistration_commit": module.PREREGISTRATION_COMMIT,
            "pinned_source_and_review_commit": "a" * 40,
            "implementation_commit": "b" * 40,
            "selectors": {
                "executor_module": module.EXECUTOR_MODULE_NAME,
                "model_module": module.MODEL_MODULE_NAME,
                "model_class": module.MODEL_CLASS_NAME,
                "training_module": module.TRAINING_MODULE_NAME,
                "evaluation_module": module.EVALUATION_MODULE_NAME,
            },
            "runtime_data_root": module.RUNTIME_DATA_ROOT,
            "runtime_inputs": runtime_inputs,
            "rgb_root_relative_path": module.RGB_ROOT_RELATIVE_PATH,
            "hardware": inherited["hardware"],
            "runtime": inherited["runtime"],
            "authorized_roles": inherited["authorized_roles"],
            "clean_export_certification": {
                "path": module.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
                "file_sha256": "c" * 64,
                "byte_count": 1,
                "content_sha256": "d" * 64,
            },
        },
    )


def _recertify(module, value):
    core = dict(value)
    core.pop("content_sha256", None)
    return _content_bound(module, core)


def test_import_and_no_argument_path_open_no_scientific_payload(capsys) -> None:
    watched = {"torch", "numpy", "PIL"}
    before = {name for name in sys.modules if name.split(".", 1)[0] in watched}
    scientific_names = (
        "scripts.execute_go2_rgb_object_space_explicit_plan_discounted_"
        "successor_state_joint_jepa_v27",
        "lewm.models.geometry_anchored_explicit_plan_discounted_successor_"
        "state_joint_jepa_v27",
        "scripts.run_go2_rgb_object_space_explicit_plan_discounted_successor_"
        "state_joint_jepa_v27",
        "scripts.evaluate_go2_rgb_object_space_explicit_plan_discounted_"
        "successor_state_joint_jepa_v27",
    )
    scientific_before = {name: sys.modules.get(name) for name in scientific_names}
    module = _load("_v27_launcher_denial")
    after = {name for name in sys.modules if name.split(".", 1)[0] in watched}
    assert after == before
    assert {name: sys.modules.get(name) for name in scientific_names} == (
        scientific_before
    )

    assert module.main([]) == 4
    receipt = json.loads(capsys.readouterr().out)
    assert receipt == {
        "schema": module.LAUNCHER_SCHEMA,
        "status": "DENIED_NO_FUTURE_AUTHORITY",
        "scientific_payload_opened": False,
        "reservation_created": False,
    }


def test_adapter_keeps_v25_builder_and_full_v13_physical_schedule() -> None:
    module = _load("_v27_launcher_adapter")
    module._assert_runtime_adapter_v27()
    assert module.SCHEMA_PREFIX.endswith("v27_integrity_replacement_v1")
    assert "integrity-replacement-v1-source" in module.CERTIFIED_SOURCE_ROOT
    assert "integrity_replacement_v1" in module.OUTPUT_ROOT_RELATIVE_PATH
    assert module._BASE_LAUNCHER._build_one_microbatch_v13 is (
        module._V25_LAUNCHER._build_one_microbatch_v25
    )
    assert module._BASE_LAUNCHER.MAXIMUM_UPDATES == 1_000
    assert module._BASE_LAUNCHER.MAXIMUM_PRESENTATIONS == 16_000
    receipt = module.private_launcher_adapter_receipt_v27()
    assert receipt["physical_microbatch_builder"] == "v25_exact"
    assert receipt["physical_runtime_composer"] == "v13_exact"
    assert receipt["maximum_updates"] == 400
    assert receipt["maximum_presentations"] == 12_800
    assert receipt["execution_authorized_by_source"] is False


def test_pre_reservation_gpu_visibility_contract_is_exact() -> None:
    module = _load("_v27_launcher_gpu_visibility")
    assert module.validate_pre_reservation_gpu_visibility_v27(
        {"HIP_VISIBLE_DEVICES": "0"}
    ) == {
        "schema": (
            f"{module.SCHEMA_PREFIX}_pre_reservation_gpu_visibility_v1"
        ),
        "hip_visible_devices": "0",
        "conflicting_selectors_present": [],
        "hardware_queried": False,
        "passed": True,
    }

    for environment in ({}, {"HIP_VISIBLE_DEVICES": ""}, {"HIP_VISIBLE_DEVICES": "1"}):
        with pytest.raises(PermissionError, match="HIP_VISIBLE_DEVICES=0"):
            module.validate_pre_reservation_gpu_visibility_v27(environment)
    for name in module.CONFLICTING_GPU_VISIBILITY_ENVIRONMENT_KEYS:
        with pytest.raises(PermissionError, match="conflicting selector"):
            module.validate_pre_reservation_gpu_visibility_v27(
                {"HIP_VISIBLE_DEVICES": "0", name: ""}
            )


def test_authority_is_exact_and_rejects_scientific_drift() -> None:
    module = _load("_v27_launcher_authority")
    authority = _authority(module)
    assert module.validate_authority_v27(authority) == authority

    mutations = []
    changed = dict(authority)
    changed["maximum_updates"] = 401
    mutations.append(changed)
    changed = dict(authority)
    changed["retry_authorized"] = True
    mutations.append(changed)
    changed = dict(authority)
    changed["selectors"] = {**authority["selectors"], "model_class": "Changed"}
    mutations.append(changed)
    changed = dict(authority)
    changed_inputs = dict(authority["runtime_inputs"])
    changed_inputs["h6_train_index"] = {
        **changed_inputs["h6_train_index"],
        "file_sha256": "0" * 64,
    }
    changed["runtime_inputs"] = changed_inputs
    mutations.append(changed)
    changed = dict(authority)
    changed["unexpected"] = True
    mutations.append(changed)

    for mutation in mutations:
        with pytest.raises(PermissionError):
            module.validate_authority_v27(_recertify(module, mutation))


def _synthetic_certification(module, root: Path):
    bindings = []
    for index, relative in enumerate(sorted(module.REQUIRED_CERTIFIED_SOURCE_PATHS)):
        payload = f"source-{index}\n".encode("ascii")
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        bindings.append(
            {
                "path": relative,
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "byte_count": len(payload),
            }
        )
    core = {
        "schema": module.CERTIFICATION_SCHEMA,
        "status": "PASS_CLEAN_EXPORT_CERTIFIED",
        "passed": True,
        "certified_source_root": str(root),
        "pinned_source_and_review_commit": "a" * 40,
        "bindings_sha256": module._sha256_canonical_v27(bindings),
        "bindings": bindings,
    }
    certification = _content_bound(module, core)
    raw = module._canonical_json_bytes_v27(certification) + b"\n"
    path = root / module.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    authority = {
        "certified_source_root": str(root),
        "pinned_source_and_review_commit": "a" * 40,
        "clean_export_certification": {
            "path": module.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "content_sha256": certification["content_sha256"],
        },
    }
    return authority, bindings


def test_thin_certification_validates_exact_files_and_allows_python_dataset(
    tmp_path,
) -> None:
    module = _load("_v27_launcher_certification")
    authority, bindings = _synthetic_certification(module, tmp_path)
    receipt = module.validate_source_certification_v27(tmp_path, authority)
    assert receipt["passed"] is True
    assert receipt["validated_path_count"] == len(bindings)
    assert receipt["certified_source_bindings"] == bindings

    dataset = next(row for row in bindings if row["path"].startswith("lewm/datasets/"))
    assert module._validate_certified_source_binding_for_base_v27(
        tmp_path, dataset
    ) == dataset["path"]

    tampered = tmp_path / bindings[0]["path"]
    tampered.write_bytes(b"tampered\n")
    with pytest.raises(PermissionError):
        module.validate_source_certification_v27(tmp_path, authority)


@pytest.mark.parametrize(
    "relative",
    (
        ".generated/runtime.json",
        "sealed/secret.py",
        "sealed_test.json",
        "held_out/maze.py",
    ),
)
def test_source_validator_rejects_protected_paths_before_open(
    tmp_path, relative
) -> None:
    module = _load(f"_v27_launcher_protected_{relative.replace('/', '_')}")
    binding = {
        "path": relative,
        "file_sha256": "0" * 64,
        "byte_count": 1,
    }
    with pytest.raises(PermissionError, match="protected path"):
        module._validate_certified_path_v27(tmp_path.resolve(), binding)


def _patch_execution_shell(monkeypatch, module, authority, *, compose):
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    for name in module.CONFLICTING_GPU_VISIBILITY_ENVIRONMENT_KEYS:
        monkeypatch.delenv(name, raising=False)
    events = []
    runtime = SimpleNamespace(close_v13=lambda: events.append("close"))
    publisher = object()
    reservation = {"reservation": True}

    def validate_sources(source_root, supplied):
        assert source_root == module.ROOT.resolve()
        assert supplied is authority
        events.append("sources")
        return {
            "passed": True,
            "validated_path_count": 8,
            "bindings_sha256": "e" * 64,
            "certification_content_sha256": "f" * 64,
            "certified_export_binding_count": 1,
            "certified_source_bindings_sha256": "e" * 64,
            "certified_source_bindings": [
                {"path": "source.py", "file_sha256": "0" * 64, "byte_count": 1}
            ],
        }

    def reserve(repository, supplied, *, created_utc):
        assert repository == module.ROOT.resolve()
        assert supplied is authority
        assert created_utc == "time"
        events.append("reserve")
        return reservation

    executor = SimpleNamespace(
        validate_future_execution_prerequisites_v27=lambda supplied: events.append(
            "executor_validate"
        ),
        reserve_attempt_v27=reserve,
        run_future_authorized_engine_v27=None,
        terminalize_failure_v27=None,
    )
    monkeypatch.setattr(module, "_load_authority_file_v27", lambda _: authority)
    monkeypatch.setattr(module, "validate_authority_v27", lambda value: value)
    monkeypatch.setattr(module, "validate_source_certification_v27", validate_sources)
    monkeypatch.setattr(module.importlib, "import_module", lambda name: executor)
    monkeypatch.setattr(
        module._BASE_LAUNCHER,
        "_validate_certified_source_root_v13",
        lambda *_: events.append("root"),
    )
    monkeypatch.setattr(
        module._BASE_LAUNCHER,
        "_activate_certified_source_root_v13",
        lambda *_: events.append("activate"),
    )
    monkeypatch.setattr(
        module._BASE_LAUNCHER,
        "_validate_runtime_data_root_v13",
        lambda *_: events.append("runtime_root"),
    )
    monkeypatch.setattr(
        module._BASE_LAUNCHER,
        "_ensure_output_parent_v13",
        lambda *_: events.append("output_parent"),
    )
    monkeypatch.setattr(module._BASE_LAUNCHER, "_utc_now_v13", lambda: "time")
    monkeypatch.setattr(
        module._BASE_LAUNCHER,
        "compose_runtime_v13",
        lambda **kwargs: compose(events, runtime, kwargs),
    )
    monkeypatch.setattr(
        module._BASE_LAUNCHER,
        "V13WriteOncePublisher",
        lambda *_: publisher,
    )
    return events, executor, runtime, publisher, reservation


def test_invalid_gpu_visibility_is_rejected_before_reservation(monkeypatch) -> None:
    module = _load("_v27_launcher_gpu_visibility_ordering")
    authority = _authority(module)

    def compose(*_args, **_kwargs):
        pytest.fail("runtime composed after GPU visibility rejection")

    events, executor, _runtime, _publisher, _reservation = _patch_execution_shell(
        monkeypatch, module, authority, compose=compose
    )
    executor.run_future_authorized_engine_v27 = lambda **_: pytest.fail(
        "engine ran after GPU visibility rejection"
    )
    executor.terminalize_failure_v27 = lambda *_args, **_kwargs: pytest.fail(
        "an unreserved visibility rejection was terminalized"
    )
    monkeypatch.delenv("HIP_VISIBLE_DEVICES")

    with pytest.raises(PermissionError, match="HIP_VISIBLE_DEVICES=0"):
        module.execute_future_authorized_v27(
            repository_root=module.ROOT, authority=authority
        )
    assert events == []


def test_execution_reserves_then_composes_calls_exact_v27_engine_and_closes(
    monkeypatch,
) -> None:
    module = _load("_v27_launcher_orchestration")
    authority = _authority(module)

    def compose(events, runtime, kwargs):
        events.append("compose")
        assert kwargs["authority"] is authority
        assert kwargs["reservation"] is reservation
        return runtime

    events, executor, runtime, publisher, reservation = _patch_execution_shell(
        monkeypatch, module, authority, compose=compose
    )

    def engine(**kwargs):
        events.append("terminal_rehash")
        executor.validate_bound_sources_v13(module.ROOT.resolve())
        events.append("engine")
        assert kwargs == {
            "authority": authority,
            "reservation": reservation,
            "runtime": runtime,
            "publisher": publisher,
        }
        return {"status": "PASS_DEVELOPMENT_UPDATE400_TERMINAL"}

    executor.run_future_authorized_engine_v27 = engine
    executor.terminalize_failure_v27 = lambda *_args, **_kwargs: pytest.fail(
        "success path terminalized failure"
    )
    result = module.execute_future_authorized_v27(
        repository_root=module.ROOT, authority=authority
    )
    assert result["status"] == "PASS_DEVELOPMENT_UPDATE400_TERMINAL"
    assert events.index("reserve") < events.index("compose") < events.index("engine")
    assert events[-1] == "close"


def test_post_reservation_composition_error_terminalizes_once(monkeypatch) -> None:
    module = _load("_v27_launcher_composition_failure")
    authority = _authority(module)

    def compose(events, _runtime, _kwargs):
        events.append("compose")
        raise RuntimeError("composition failed")

    events, executor, _runtime, _publisher, reservation = _patch_execution_shell(
        monkeypatch, module, authority, compose=compose
    )
    terminal = []

    def terminalize(output_root, supplied, **kwargs):
        terminal.append((output_root, supplied, kwargs))
        return {"status": "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME"}

    executor.run_future_authorized_engine_v27 = lambda **_: pytest.fail(
        "engine ran after composition failure"
    )
    executor.terminalize_failure_v27 = terminalize
    monkeypatch.setattr(module, "_terminal_exists_v27", lambda _root: False)
    with pytest.raises(RuntimeError, match="composition failed"):
        module.execute_future_authorized_v27(
            repository_root=module.ROOT, authority=authority
        )
    assert events.index("reserve") < events.index("compose")
    assert "close" not in events
    assert len(terminal) == 1
    assert terminal[0][1] is reservation
    assert terminal[0][2]["stage"] == "post_reservation_runtime_composition"
    assert terminal[0][2]["created_utc"] == "time"
