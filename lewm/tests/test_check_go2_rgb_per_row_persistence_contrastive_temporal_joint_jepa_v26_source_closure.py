from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / (
    "scripts/check_go2_rgb_per_row_persistence_contrastive_temporal_joint_jepa_"
    "v26_source_closure.py"
)


def _load(name: str):
    for module_name in tuple(sys.modules):
        if (
            module_name.startswith("_lewm_v26_schema_compat_source_closure_base")
            or module_name.startswith("_lewm_v25_per_row_temporal_source_closure_base")
            or (
                "source_closure_base" in module_name
                and module_name.startswith("_lewm_v24")
            )
        ):
            sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_checker_is_denied_and_binds_frozen_v25(capsys) -> None:
    checker = _load("_v26_checker_denial")
    receipt = checker.execution_denial_receipt_v26()
    assert receipt["execution_authorized"] is False
    assert receipt["recovery_state_opened"] is False
    assert checker.BASE_FROZEN_SOURCE_AND_REVIEW_COMMIT == (
        "43231c689547b66de83f3cafbfac270455a7a234"
    )
    assert checker.PREREGISTRATION_COMMIT == (
        "0c277fd7350931a7993d5affc2d1d4633ffed916"
    )
    assert checker.main(["--emit"]) == 0
    assert checker.SCHEMA in capsys.readouterr().out


def test_preregistration_and_v25_failure_bindings_are_exact() -> None:
    checker = _load("_v26_checker_bindings")
    for relative, expected_sha, expected_bytes in (
        (
            checker.PREREGISTRATION_RELATIVE_PATH,
            checker.PREREGISTRATION_FILE_SHA256,
            checker.PREREGISTRATION_BYTE_COUNT,
        ),
        (
            checker.V25_TERMINAL_FAILURE_RESULT_RELATIVE_PATH,
            checker.V25_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            checker.V25_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
    ):
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == expected_bytes
        assert hashlib.sha256(raw).hexdigest() == expected_sha


def test_closure_is_exact_v25_plus_three_v26_runtime_wrappers() -> None:
    checker = _load("_v26_checker_entrypoints")
    assert checker.ENTRYPOINTS[1:] == checker.IMPLEMENTATION_PATHS
    assert set(checker.V25_PARENT_ENTRYPOINTS).issubset(
        checker.FORCED_DYNAMIC_SOURCES
    )
    manifest = checker.build_manifest()
    paths = tuple(manifest["source_paths"])
    assert len(paths) == checker.EXPECTED_SOURCE_COUNT == 107
    assert len(paths) == len(set(paths))
    assert all(path in paths for path in checker.IMPLEMENTATION_PATHS)
    assert all(path in paths for path in checker.V25_PARENT_ENTRYPOINTS)
    assert checker.BASE_CHECKER_PATH not in paths
    assert checker.PREREGISTRATION_RELATIVE_PATH not in paths
    assert checker.V25_TERMINAL_FAILURE_RESULT_RELATIVE_PATH not in paths


def test_candidate_contains_no_runtime_or_protected_material() -> None:
    checker = _load("_v26_checker_custody")
    manifest = checker.build_manifest()
    paths = tuple(manifest["source_paths"])
    forbidden_parts = {".generated", "sealed", "heldout", "held_out"}
    assert all((ROOT / path).is_file() and not (ROOT / path).is_symlink() for path in paths)
    assert all(not (forbidden_parts & set(Path(path).parts)) for path in paths)
    assert not any("recovery/update_400_training_state" in path for path in paths)
    assert manifest["excluded_runtime_categories"] == list(
        checker.EXCLUDED_RUNTIME_CATEGORIES
    )
