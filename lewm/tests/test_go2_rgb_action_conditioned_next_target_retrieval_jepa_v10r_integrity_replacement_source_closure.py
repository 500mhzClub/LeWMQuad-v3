from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = (
    ROOT
    / "scripts/"
    "check_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement_source_closure.py"
)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_v10r_checker_import_is_source_only_and_uses_v10r_contract() -> None:
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {
            "torch", "numpy", "PIL", "cv2", "jax", "tensorflow",
        }:
            raise AssertionError(f"source-only import loaded {name}")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded):
        checker = _load("_test_retrieval_jepa_v10r_source_closure")

    assert checker.contract.SCHEMA_PREFIX.endswith(
        "v10r_integrity_replacement"
    )
    assert checker._V10.contract is checker.contract
    assert checker._V10._BASE.ENTRYPOINTS == (
        checker.contract.SOURCE_MANIFEST_ENTRYPOINTS
    )


def test_v10r_manifest_candidate_is_complete_and_valid() -> None:
    checker = _load("_test_retrieval_jepa_v10r_manifest_candidate")
    value = checker.build_manifest()
    raw = checker.contract.canonical_json_bytes(value) + b"\n"
    assert checker.contract.validate_source_manifest(raw) == value
    paths = set(value["source_paths"])
    assert set(checker.contract.SOURCE_PATHS).issubset(paths)
    assert checker.REQUIRED_DYNAMIC_SOURCE_PATHS.issubset(paths)
    assert {
        checker.contract.CONTRACT_RELATIVE_PATH,
        checker.contract.RUNNER_RELATIVE_PATH,
        checker.contract.LAUNCHER_RELATIVE_PATH,
        checker.contract.CONTRACT_TEST_RELATIVE_PATH,
        checker.contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
        checker.contract.SOURCE_CLOSURE_TEST_RELATIVE_PATH,
    }.issubset(paths)
    assert value["entrypoints"] == list(
        checker.contract.SOURCE_MANIFEST_ENTRYPOINTS
    )
    assert value["generated_input_open_count"] == 0
    assert value["checkpoint_or_tensor_open_count"] == 0
    assert value["sealed_or_heldout_open_count"] == 0
