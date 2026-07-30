from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
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
        "ba6e37d63f099cd51184642dea39808ae1f2f99e"
    )
    bindings = (
        (
            checker.PREREGISTRATION_RELATIVE_PATH,
            checker.PREREGISTRATION_FILE_SHA256,
            checker.PREREGISTRATION_BYTE_COUNT,
        ),
        (
            checker.ORIGINAL_PREREGISTRATION_RELATIVE_PATH,
            checker.ORIGINAL_PREREGISTRATION_FILE_SHA256,
            checker.ORIGINAL_PREREGISTRATION_BYTE_COUNT,
        ),
        (
            checker.TERMINAL_FAILURE_RESULT_RELATIVE_PATH,
            checker.TERMINAL_FAILURE_RESULT_FILE_SHA256,
            checker.TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
    )
    for relative, expected_sha256, expected_bytes in bindings:
        raw = (ROOT / relative).read_bytes()
        assert len(raw) == expected_bytes
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
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


def test_recursive_closure_contains_every_private_predecessor_chain() -> None:
    checker = _load("_memory_role_v1_checker_private_chains")
    paths = set(checker.build_manifest()["source_paths"])
    for chain in (
        checker.RUNNER_PREDECESSOR_SOURCES,
        checker.EXECUTOR_PREDECESSOR_SOURCES,
        checker.LAUNCHER_PREDECESSOR_SOURCES,
    ):
        assert len(chain) == len(set(chain))
        assert set(chain) <= paths


def test_manifest_isolated_tree_imports_runtime_shells(tmp_path: Path) -> None:
    checker = _load("_memory_role_v1_checker_isolated_import")
    manifest = checker.build_manifest()
    for relative in manifest["source_paths"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)

    program = """
import importlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(root))
importlib.import_module(
    "scripts.run_go2_rgb_memory_role_factorized_joint_jepa_v1"
)
importlib.import_module(
    "scripts.execute_go2_rgb_memory_role_factorized_joint_jepa_v1"
)
launcher = importlib.import_module(
    "scripts.launch_go2_rgb_memory_role_factorized_joint_jepa_v1"
)
status = launcher.main([])
raise SystemExit(status)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4, result.stderr
    assert json.loads(result.stdout) == {
        "schema": (
            "lewm_go2_rgb_memory_role_factorized_joint_jepa_v1_"
            "integrity_replacement_v1_launcher_v1"
        ),
        "status": "DENIED_NO_FUTURE_AUTHORITY",
        "scientific_payload_opened": False,
        "reservation_created": False,
    }


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
