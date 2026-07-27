from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "lewm/benchmarks/"
    "go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)
RUNNER = (
    ROOT / "scripts/"
    "run_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)
LAUNCHER = (
    ROOT / "scripts/"
    "launch_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement.py"
)
CHECKER = (
    ROOT / "scripts/"
    "check_go2_geometry_anchored_deformable_bev_lift_joint_jepa_v2_"
    "runtime_import_integrity_replacement_source_closure.py"
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_contract_is_exact_v1_science_with_distinct_v2_lifecycle() -> None:
    contract = _load("_joint_jepa_v2_import_contract_test", CONTRACT)
    v1 = contract._v1
    assert len(v1.SOURCE_PATHS) == 74
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 5
    assert len(contract.SOURCE_PATHS) == 79
    assert set(contract.SOURCE_PATHS) == {
        *v1.SOURCE_PATHS,
        *contract.ADDITIVE_SOURCE_PATHS,
    }
    assert contract.MODEL_RELATIVE_PATH == v1.MODEL_RELATIVE_PATH
    assert not any("models/" in path for path in contract.ADDITIVE_SOURCE_PATHS)
    assert contract.model_config() == v1.model_config()
    assert contract.objective_contract() == v1.objective_contract()
    assert contract.optimizer_contract() == v1.optimizer_contract()
    assert contract.build_schedule_identity() == v1.build_schedule_identity()
    assert contract.runtime_authorization_template() == (
        v1.runtime_authorization_template()
    )
    assert contract.canonical_json_sha256(v1.science_contract()) == (
        contract.FROZEN_V1_SCIENCE_CONTRACT_SHA256
    )
    observed = {
        "model": contract.canonical_json_sha256(contract.model_config()),
        "objective": contract.canonical_json_sha256(contract.objective_contract()),
        "optimizer": contract.canonical_json_sha256(contract.optimizer_contract()),
        "schedule": contract.canonical_json_sha256(
            contract.build_schedule_identity()
        ),
        "gate_thresholds": contract.canonical_json_sha256(
            contract.GATE_THRESHOLDS
        ),
    }
    assert observed == contract.FROZEN_V1_SCIENCE_COMPONENT_SHA256
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != v1.OUTPUT_ROOT_RELATIVE_PATH
    assert not (ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH).exists()
    assert contract.EXECUTION_AUTHORITY["maximum_updates"] == 1_000
    assert contract.EXECUTION_AUTHORITY["maximum_presentations"] == 16_000
    assert contract.EXECUTION_AUTHORITY["gpu_active_minutes_maximum"] == 30
    assert contract.validate_governing_documents(ROOT)


@pytest.mark.parametrize("path", [RUNNER, LAUNCHER])
def test_wrapper_import_is_source_only_under_isolation(path: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(path)!r})
spec = importlib.util.spec_from_file_location("_joint_jepa_v2_wrapper", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert "PIL" not in sys.modules
assert module._V1.contract is module.contract
assert Path(module._V1.__file__).resolve() == path
print("PASS")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PASS\n"
    assert completed.stderr == ""


def test_corrected_loader_keeps_one_root_then_restores_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load("_joint_jepa_v2_import_runner_unit", RUNNER)
    calls: list[str] = []

    def assert_one_root(stage: str) -> None:
        calls.append(stage)
        assert Path(sys.path[0]).resolve() == ROOT
        assert sum(runner._canonical_root_entry(item) for item in sys.path) == 1

    class Matched:
        @staticmethod
        def _load_runtime() -> str:
            assert_one_root("runtime")
            return "runtime"

    def fake_source(name: str, path: Path) -> Any:
        del name
        if path == runner._V1.MATCHED_RUNNER_PATH:
            assert_one_root("matched")
            return Matched
        if path == runner._V1.SCHEDULE_ADAPTER_PATH:
            assert_one_root("schedule")
            return "schedule"
        if path == ROOT / runner.contract.MODEL_RELATIVE_PATH:
            assert_one_root("model")
            return "model"
        raise AssertionError(path)

    monkeypatch.setattr(runner._V1, "_read_regular", lambda *args, **kwargs: b"")
    monkeypatch.setattr(runner._V1, "_source_module", fake_source)
    original = list(sys.path)
    duplicated = [str(ROOT), str(ROOT / "."), *original]
    sys.path[:] = duplicated
    try:
        loaded = runner._load_post_reservation_stack({"source.py": "0" * 64})
        assert loaded == (Matched, "runtime", "schedule", "model")
        assert sys.path == duplicated
    finally:
        sys.path[:] = original
    assert calls == ["matched", "runtime", "schedule", "model"]


def test_corrected_loader_restores_path_on_lazy_import_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load("_joint_jepa_v2_import_runner_exception", RUNNER)

    class Matched:
        @staticmethod
        def _load_runtime() -> None:
            assert sum(runner._canonical_root_entry(item) for item in sys.path) == 1
            raise ModuleNotFoundError("synthetic delayed import")

    monkeypatch.setattr(runner._V1, "_read_regular", lambda *args, **kwargs: b"")
    monkeypatch.setattr(
        runner._V1,
        "_source_module",
        lambda name, path: Matched,
    )
    baseline = list(sys.path)
    with pytest.raises(ModuleNotFoundError, match="synthetic delayed import"):
        runner._load_post_reservation_stack({"source.py": "0" * 64})
    assert sys.path == baseline


def test_runner_and_launcher_delegate_with_v2_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load("_joint_jepa_v2_import_runner_delegate", RUNNER)
    called: list[list[str] | None] = []

    def fake_runner_main(argv: list[str] | None = None) -> int:
        assert runner._V1.contract is runner.contract
        assert runner._V1._load_post_reservation_stack is (
            runner._load_post_reservation_stack
        )
        called.append(argv)
        return 17

    monkeypatch.setattr(runner._V1, "main", fake_runner_main)
    args = ["--review-sha256", "a" * 64, "--authorization-sha256", "b" * 64]
    assert runner.main(args) == 17
    assert called == [args]

    launcher = _load("_joint_jepa_v2_import_launcher_delegate", LAUNCHER)
    parsed = launcher.parse_args(args)
    argv = launcher._V1._runtime_argv(parsed)
    assert argv == [
        launcher.contract.RUNTIME_INTERPRETER_PATH,
        *launcher.contract.RUNTIME_INTERPRETER_ARGUMENTS,
        str(ROOT / launcher.contract.RUNNER_RELATIVE_PATH),
        "--review-sha256",
        "a" * 64,
        "--authorization-sha256",
        "b" * 64,
    ]
    environment = launcher._V1._launch_environment()
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["HIP_VISIBLE_DEVICES"] == "0"
    assert launcher._V1.OUTPUT_ROOT == (
        ROOT / launcher.contract.OUTPUT_ROOT_RELATIVE_PATH
    )


def test_failure_receipts_and_one_shot_denials_are_preserved() -> None:
    contract = _load("_joint_jepa_v2_import_failure_contract", CONTRACT)
    assert contract.NORMAL_RECEIPT_PATHS == contract._v1.NORMAL_RECEIPT_PATHS
    assert contract.OPERATIONAL_FAILURE_RECEIPT_PATHS == (
        contract._v1.OPERATIONAL_FAILURE_RECEIPT_PATHS
    )
    assert contract.OPERATIONAL_FAILURE_RECEIPT_PATHS == (
        "metrics.json", "artifact.json", "access.json", "result.json",
        "failure.json", "completed.json",
    )
    assert contract.EXECUTION_AUTHORITY["maximum_attempts"] == 1
    assert contract.EXECUTION_AUTHORITY[
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt_authorized"
    ] is False
    assert contract.EXECUTION_AUTHORITY["v1_retry_authorized"] is False
    assert contract.EXECUTION_AUTHORITY["checkpoint_read_authorized"] is False
    assert contract.EXECUTION_AUTHORITY["navigation_authorized"] is False
    assert contract.EXECUTION_AUTHORITY["heldout_authorized"] is False


def test_recursive_closure_is_exactly_79_sources() -> None:
    checker = _load("_joint_jepa_v2_import_closure_test", CHECKER)
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 79
    assert manifest["source_paths"] == list(checker.contract.SOURCE_PATHS)
    assert set(manifest["source_paths"]) == {
        *checker._V1.contract.REUSED_SOURCE_PATHS,
        *checker.contract.ADDITIVE_SOURCE_PATHS,
    }
    assert manifest["entrypoints"] == list(
        checker.contract.SOURCE_MANIFEST_ENTRYPOINTS
    )


def test_exact_reviewed_runtime_post_stack_import_preflight() -> None:
    contract = _load("_joint_jepa_v2_import_preflight_contract", CONTRACT)
    assert not (ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH).exists()
    program = f"""
import importlib.util
import json
from pathlib import Path
import sys
path = Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location("_joint_jepa_v2_exact_preflight", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
before = list(sys.path)
result = module.run_isolated_import_preflight()
assert result == module.contract.IMPORT_PREFLIGHT_REQUIREMENTS
assert sys.path == before
assert not (module.ROOT / module.contract.OUTPUT_ROOT_RELATIVE_PATH).exists()
print(json.dumps(result, sort_keys=True, separators=(",", ":")))
"""
    completed = subprocess.run(
        [
            contract.RUNTIME_INTERPRETER_PATH,
            *contract.RUNTIME_INTERPRETER_ARGUMENTS,
            "-c",
            program,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert "post_reservation_stack_imported" in completed.stdout
    assert not (ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH).exists()
