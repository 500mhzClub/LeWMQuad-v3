from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODULE_PATH = ROOT / "scripts/launch_go2_rgb_memory_role_factorized_joint_jepa_v1.py"
PRIVATE_PREFIXES = (
    "_lewm_memory_role_v1_private_v25_launcher",
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
        module._canonical_json_bytes_v1(value)
    ).hexdigest()
    return value


def _authority(module, monkeypatch):
    binding = {
        "path": ".generated/synthetic.jsonl",
        "file_sha256": "a" * 64,
        "byte_count": 1,
    }
    physical = {name: dict(binding) for name in module.PHYSICAL_RUNTIME_INPUT_NAMES}
    hardware = {"synthetic": "hardware"}
    runtime = {"synthetic": "runtime"}
    roles = {"synthetic": "roles"}
    monkeypatch.setattr(
        module, "PHYSICAL_RUNTIME_INPUTS_SHA256", module._sha256_canonical_v1(physical)
    )
    monkeypatch.setattr(module, "HARDWARE_SHA256", module._sha256_canonical_v1(hardware))
    monkeypatch.setattr(module, "RUNTIME_SHA256", module._sha256_canonical_v1(runtime))
    monkeypatch.setattr(
        module, "AUTHORIZED_ROLES_SHA256", module._sha256_canonical_v1(roles)
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
            "split_integrity_amendment_commit": (
                module.SPLIT_INTEGRITY_AMENDMENT_COMMIT
            ),
            "pinned_source_and_review_commit": "b" * 40,
            "implementation_commit": "c" * 40,
            "selectors": {
                "executor_module": module.EXECUTOR_MODULE_NAME,
                "model_module": module.MODEL_MODULE_NAME,
                "model_class": module.MODEL_CLASS_NAME,
                "training_module": module.TRAINING_MODULE_NAME,
                "evaluation_module": module.EVALUATION_MODULE_NAME,
            },
            "runtime_data_root": module.RUNTIME_DATA_ROOT,
            "runtime_inputs": {**physical, **module.ROLE_RUNTIME_BINDINGS},
            "rgb_root_relative_path": module.RGB_ROOT_RELATIVE_PATH,
            "hardware": hardware,
            "runtime": runtime,
            "authorized_roles": roles,
            "clean_export_certification": {
                "path": module.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
                "file_sha256": "d" * 64,
                "byte_count": 1,
                "content_sha256": "e" * 64,
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
        "scripts.execute_go2_rgb_memory_role_factorized_joint_jepa_v1",
        "lewm.models.memory_role_spatial_contrastive_joint_jepa_v3",
        "scripts.run_go2_rgb_memory_role_factorized_joint_jepa_v1",
        "scripts.evaluate_go2_rgb_memory_role_factorized_joint_jepa_v1",
    )
    scientific_before = {name: sys.modules.get(name) for name in scientific_names}
    module = _load("_memory_role_launcher_denial")
    after = {name for name in sys.modules if name.split(".", 1)[0] in watched}
    assert after == before
    assert {name: sys.modules.get(name) for name in scientific_names} == scientific_before
    assert module.main([]) == 4
    assert json.loads(capsys.readouterr().out) == {
        "schema": module.LAUNCHER_SCHEMA,
        "status": "DENIED_NO_FUTURE_AUTHORITY",
        "scientific_payload_opened": False,
        "reservation_created": False,
    }


def test_adapter_keeps_exact_v25_builder_and_fixed_three_way_budget() -> None:
    module = _load("_memory_role_launcher_adapter")
    module._assert_runtime_adapter_v1()
    assert module._BASE_LAUNCHER._build_one_microbatch_v13 is (
        module._V25_LAUNCHER._build_one_microbatch_v25
    )
    receipt = module.private_launcher_adapter_receipt_v1()
    assert receipt["physical_presentations_per_update"] == 16
    assert receipt["local_presentations_per_update"] == 8
    assert receipt["place_presentations_per_update"] == 8
    assert receipt["maximum_updates"] == 400
    assert receipt["maximum_presentations"] == 12_800
    assert receipt["retry_authorized"] is False
    assert receipt["resume_authorized"] is False


def test_v4_uses_fresh_lifecycle_identity() -> None:
    module = _load("_memory_role_launcher_v4_identity")
    assert module.SCHEMA_PREFIX.endswith("_v4")
    assert module.EXPERIMENT_ARM_NAME.endswith("_v4")
    assert module.PREREGISTRATION_COMMIT == (
        "b079504940103f2cbd127552d337a90b6028b749"
    )
    assert module.CERTIFIED_SOURCE_ROOT.endswith("joint-jepa-v4-source")
    assert module.OUTPUT_ROOT_RELATIVE_PATH.endswith("v4/attempt_v1")
    assert module.PREREGISTRATION_RELATIVE_PATH in (
        module.REQUIRED_CERTIFIED_SOURCE_PATHS
    )
    assert module.RETRIEVAL_METADATA_PREFLIGHT_RELATIVE_PATH in (
        module.REQUIRED_CERTIFIED_SOURCE_PATHS
    )
    assert module.V2_SCIENTIFIC_RESULT_RELATIVE_PATH in (
        module.REQUIRED_CERTIFIED_SOURCE_PATHS
    )
    assert module.V3_TERMINAL_INFRASTRUCTURE_FAILURE_RESULT_RELATIVE_PATH in (
        module.REQUIRED_CERTIFIED_SOURCE_PATHS
    )
    assert module.TERMINAL_FAILURE_RESULT_RELATIVE_PATH in (
        module.REQUIRED_CERTIFIED_SOURCE_PATHS
    )
    assert module.INTEGRITY_REPLACEMENT_TERMINAL_FAILURE_RESULT_RELATIVE_PATH in (
        module.REQUIRED_CERTIFIED_SOURCE_PATHS
    )


def test_pre_reservation_gpu_visibility_is_exact() -> None:
    module = _load("_memory_role_launcher_gpu")
    receipt = module.validate_pre_reservation_gpu_visibility_v1(
        {"HIP_VISIBLE_DEVICES": "0"}
    )
    assert receipt["passed"] is True
    assert receipt["hardware_queried"] is False
    with pytest.raises(PermissionError, match="HIP_VISIBLE_DEVICES=0"):
        module.validate_pre_reservation_gpu_visibility_v1({})
    for name in module.CONFLICTING_GPU_VISIBILITY_ENVIRONMENT_KEYS:
        with pytest.raises(PermissionError, match="conflicting selector"):
            module.validate_pre_reservation_gpu_visibility_v1(
                {"HIP_VISIBLE_DEVICES": "0", name: ""}
            )


def test_authority_rejects_schedule_role_and_retry_drift(monkeypatch) -> None:
    module = _load("_memory_role_launcher_authority")
    authority = _authority(module, monkeypatch)
    assert module.validate_authority_v1(authority) == authority
    mutations = []
    changed = dict(authority)
    changed["maximum_updates"] = 401
    mutations.append(changed)
    changed = dict(authority)
    changed["retry_authorized"] = True
    mutations.append(changed)
    changed = dict(authority)
    changed_inputs = dict(authority["runtime_inputs"])
    changed_inputs["place_triplet_train_index"] = {
        **changed_inputs["place_triplet_train_index"],
        "file_sha256": "f" * 64,
    }
    changed["runtime_inputs"] = changed_inputs
    mutations.append(changed)
    changed = dict(authority)
    changed["unexpected"] = True
    mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(PermissionError):
            module.validate_authority_v1(_recertify(module, mutation))


@pytest.mark.parametrize(
    "relative",
    (
        ".generated/runtime.json",
        "sealed/secret.py",
        "sealed_test.json",
        "held_out/maze.py",
        "data/copied.py",
    ),
)
def test_source_validator_rejects_protected_paths_before_open(
    tmp_path: Path, relative: str
) -> None:
    module = _load(f"_memory_role_launcher_protected_{relative.replace('/', '_')}")
    binding = {"path": relative, "file_sha256": "0" * 64, "byte_count": 1}
    with pytest.raises(PermissionError, match="protected path"):
        module._validate_certified_path_v1(tmp_path.resolve(), binding)
