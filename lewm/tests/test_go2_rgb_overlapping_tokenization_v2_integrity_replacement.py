from __future__ import annotations

import builtins
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/"
    "go2_rgb_overlapping_tokenization_v2_integrity_replacement.py"
)
V1_CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_overlapping_tokenization_v1.py"
)
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_overlapping_tokenization_v2_"
    "integrity_replacement.py"
)
LAUNCHER_PATH = (
    ROOT
    / "scripts/launch_go2_rgb_overlapping_tokenization_v2_"
    "integrity_replacement.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


contract = _load("_test_overlap_v2_contract", CONTRACT_PATH)
v1_contract = _load("_test_overlap_v2_frozen_v1_contract", V1_CONTRACT_PATH)


def _runtime_leaf(path: str) -> dict[str, object]:
    return {
        "path": path,
        "file_sha256": contract.RUNTIME_FILE_SHA256[path],
        "content_sha256": contract.RUNTIME_CONTENT_SHA256[path],
        "byte_count": contract.RUNTIME_BYTE_COUNTS[path],
    }


def _runtime_inputs() -> dict[str, object]:
    return {
        "raw": {
            "root": contract.RAW_ROOT_RELATIVE_PATH,
            "manifest": _runtime_leaf(contract.RAW_MANIFEST_RELATIVE_PATH),
            "audit": _runtime_leaf(contract.RAW_AUDIT_RELATIVE_PATH),
            "role_counts": {
                "train": dict(contract.TRAIN_ROLE_COUNTS),
                "checkpoint_selection":
                    dict(contract.SELECTION_ROLE_COUNTS),
            },
            "grant": {
                "allowed_roles": ["train", "checkpoint_selection"],
                "allowed_operations": [
                    "development_rgb_decode",
                    "overlapping_tokenization_training",
                    "physical_checkpoint_selection",
                ],
                "calibration_g2_navigation_heldout_or_production_use": False,
            },
        },
        "camera": {
            "root": contract.N320_ROOT_RELATIVE_PATH,
            "gate": _runtime_leaf(contract.N320_GATE_RELATIVE_PATH),
            "checkpoint": _runtime_leaf(
                contract.N320_CHECKPOINT_RELATIVE_PATH
            ),
            "seed": 20_260_710,
            "fit_size": 320,
            "updates": 40_000,
            "gate_must_pass_all_checks": 26,
        },
        "schedule": _runtime_leaf(contract.SCHEDULE_RELATIVE_PATH),
    }


def test_v2_science_contract_is_deep_equal_to_frozen_v1() -> None:
    v1 = v1_contract.science_contract()
    v2 = contract.science_contract()
    assert v2 == v1
    assert contract.canonical_json_bytes(v2) == v1_contract.canonical_json_bytes(
        v1
    )
    assert v2["model_family"] == v1["model_family"]
    assert v2["model_runtime_version"] == v1["model_runtime_version"]
    assert v2["operation_cap"]["maximum_optimizer_updates"] == 1_000
    assert (
        v2["operation_cap"]["maximum_pair_index_presentations"] == 16_000
    )
    assert v2["operation_cap"]["jepa_objective_count"] == 0
    assert v2["operation_cap"]["jepa_backward_count"] == 0


def test_exact_raw_counts_are_mandatory_and_fallback_free() -> None:
    assert contract.RUNTIME_BYTE_COUNTS[
        contract.RAW_MANIFEST_RELATIVE_PATH
    ] == 311_598
    assert contract.RUNTIME_BYTE_COUNTS[
        contract.RAW_AUDIT_RELATIVE_PATH
    ] == 26_975
    inputs = _runtime_inputs()
    assert contract.validate_runtime_inputs(inputs) == inputs

    for key in ("manifest", "audit"):
        changed = deepcopy(inputs)
        changed["raw"][key]["byte_count"] = 1
        with pytest.raises(PermissionError, match="runtime binding changed"):
            contract.validate_runtime_inputs(changed)

        missing = deepcopy(inputs)
        del missing["raw"][key]["byte_count"]
        with pytest.raises(PermissionError, match="runtime binding changed"):
            contract.validate_runtime_inputs(missing)


def test_v2_identity_is_fresh_and_v1_root_is_prohibited() -> None:
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "/rgb_overlapping_tokenization_probe_v2_integrity_replacement"
    )
    assert (
        contract.OUTPUT_ROOT_RELATIVE_PATH
        != contract.V1_OUTPUT_ROOT_RELATIVE_PATH
    )
    assert (
        contract.V1_OUTPUT_ROOT_RELATIVE_PATH
        in contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
    )
    assert (
        contract.OUTPUT_ROOT_RELATIVE_PATH
        not in contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
    )
    assert contract.INTEGRITY_REPLACEMENT_DELTA == {
        "science_changed": False,
        "v1_runtime_output_open_authorized": False,
        "raw_manifest_byte_count": 311_598,
        "raw_audit_byte_count": 26_975,
        "missing_authorized_byte_count_fallback": None,
    }


def test_contract_runner_and_launcher_import_are_source_only() -> None:
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {
            "torch", "numpy", "PIL", "cv2", "jax", "tensorflow",
        }:
            raise AssertionError(f"source-only import loaded {name}")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded):
        runner = _load("_test_overlap_v2_runner", RUNNER_PATH)
        launcher = _load("_test_overlap_v2_launcher", LAUNCHER_PATH)

    assert runner.contract.OUTPUT_ROOT_RELATIVE_PATH == (
        contract.OUTPUT_ROOT_RELATIVE_PATH
    )
    assert launcher.contract.OUTPUT_ROOT_RELATIVE_PATH == (
        contract.OUTPUT_ROOT_RELATIVE_PATH
    )
    assert runner._BASE.contract is runner.contract
    assert launcher._BASE.contract is launcher.contract
    assert runner.PREFLIGHT_ENVIRONMENT_KEY == (
        launcher.PREFLIGHT_ENVIRONMENT_KEY
    )
