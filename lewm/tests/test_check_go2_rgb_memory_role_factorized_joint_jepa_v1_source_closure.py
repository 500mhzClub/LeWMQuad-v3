from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CHECKER_PATH = ROOT / (
    "scripts/check_go2_rgb_memory_role_factorized_joint_jepa_v1_"
    "source_closure.py"
)


def _load(name: str):
    for module_name in tuple(sys.modules):
        if (
            module_name.startswith(
                "_lewm_memory_role_factorized_v1_source_closure_base"
            )
            or module_name.startswith(
                "_lewm_v13_camera_evidence_source_closure_base"
            )
        ):
            sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(name, CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_checker_is_denied_and_binds_frozen_preregistration(capsys) -> None:
    checker = _load("_memory_role_v1_checker_denial")
    receipt = checker.execution_denial_receipt_v1()
    assert receipt["execution_authorized"] is False
    assert receipt["dataset_payload_or_rgb_opened"] is False
    assert receipt["checkpoint_opened"] is False
    assert checker.PREREGISTRATION_COMMIT == (
        "01d78284a22a52816a41f31a78411491714b4f9c"
    )
    raw = (ROOT / checker.PREREGISTRATION_RELATIVE_PATH).read_bytes()
    assert len(raw) == checker.PREREGISTRATION_BYTE_COUNT
    assert hashlib.sha256(raw).hexdigest() == checker.PREREGISTRATION_FILE_SHA256
    assert checker.main(["--emit"]) == 0
    assert checker.SCHEMA in capsys.readouterr().out


def test_recursive_closure_contains_candidate_and_dynamic_sources() -> None:
    checker = _load("_memory_role_v1_checker_closure")
    manifest = checker.build_manifest()
    paths = tuple(manifest["source_paths"])
    assert len(paths) == len(set(paths))
    assert all(path in paths for path in checker.IMPLEMENTATION_PATHS)
    assert all(path in paths for path in checker.FORCED_DYNAMIC_SOURCES)
    assert all(path in paths for path in checker.ALLOWED_DATASET_SOURCES)
    assert checker.BASE_CHECKER_PATH not in paths
    assert checker.PREREGISTRATION_RELATIVE_PATH not in paths
    assert manifest["entrypoints"] == list(checker.ENTRYPOINTS)
    assert manifest["execution_authorized"] is False


def test_only_exact_dataset_sources_are_admitted() -> None:
    checker = _load("_memory_role_v1_checker_dataset_allowlist")
    for relative in checker.ALLOWED_DATASET_SOURCES:
        checker._safe_source_path(relative)
    with pytest.raises(PermissionError):
        checker._safe_source_path("lewm/datasets/unreviewed_adapter.py")


@pytest.mark.parametrize(
    "relative",
    (
        ".generated/runtime/source.py",
        "runtime_artifacts/attempt/source.py",
        "checkpoints/rejected/source.py",
        "scripts/run_go2_shared_jepa_v5_protected_camera_adaptation_v99.py",
        "scripts/probability_calibration/source.py",
        "scripts/g2/source.py",
        "scripts/heldout_probe.py",
        "scripts/held_out_probe.py",
        "sealed/source.py",
        "sealed_v4/source.py",
    ),
)
def test_protected_paths_fail_closed(relative: str) -> None:
    checker = _load("_memory_role_v1_checker_custody")
    with pytest.raises(PermissionError):
        checker._safe_source_path(relative)


def test_manifest_is_source_only_and_contains_no_protected_path() -> None:
    checker = _load("_memory_role_v1_checker_manifest_custody")
    manifest = checker.build_manifest()
    paths = tuple(manifest["source_paths"])
    forbidden_parts = {
        ".generated",
        "artifacts",
        "checkpoints",
        "runtime",
        "runtime_artifacts",
        "runtime_inputs",
        "sealed",
        "heldout",
        "held_out",
        "probability_calibration",
        "g2",
    }
    assert all(
        (ROOT / path).is_file() and not (ROOT / path).is_symlink()
        for path in paths
    )
    assert all(
        not (forbidden_parts & {part.casefold() for part in Path(path).parts})
        for path in paths
    )
    assert manifest["dataset_payload_or_rgb_open_count"] == 0
    assert manifest["generated_or_runtime_artifact_open_count"] == 0
    assert manifest["probability_calibration_open_count"] == 0
    assert manifest["g2_or_heldout_open_count"] == 0
    assert manifest["excluded_runtime_categories"] == list(
        checker.EXCLUDED_RUNTIME_CATEGORIES
    )
