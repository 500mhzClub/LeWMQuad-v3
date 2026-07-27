from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = (
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v2_"
    "runtime_interpreter_integrity_replacement"
)
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
TEST = ROOT / "lewm/tests" / f"test_{STEM}.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"
V1_CONTRACT = (
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py"
)
V1_TEST = (
    ROOT
    / "lewm/tests/"
    "test_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py"
)
V1_MODEL = (
    "lewm/models/"
    "direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py"
)
EXPECTED_RUNTIME_INTERPRETER = (
    "/home/andrewknowles/.local/share/"
    "lewmquad-v12-runtime-torch291-rocm64/bin/python"
)
EXPECTED_RUNTIME_PREFIX = (
    "/home/andrewknowles/.local/share/"
    "lewmquad-v12-runtime-torch291-rocm64"
)
EXPECTED_SCIENCE_SHA256 = (
    "2d42031e0586c205cfcae783991a497a4b3f4a5b1c5b8013aa3e65ac5ca673f1"
)
EXPECTED_OUTPUT_ROOT = (
    ".generated/go2_shared_observable_camera_ray_jepa_signed_boundary_"
    "semantic_anchor_state_v2/rgb_direct_egocentric_bev_signed_boundary_"
    "semantic_anchor_state_probe_v2_runtime_interpreter_integrity_"
    "replacement_v1"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("source", [CONTRACT, RUNNER, LAUNCHER, CHECKER])
def test_v2_sources_import_source_only(source: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(source)!r})
spec = importlib.util.spec_from_file_location('_semantic_anchor_v2_source_only', path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
print('PASS')
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


def test_v2_science_model_schedule_gates_and_caps_are_exact_v1() -> None:
    v1 = _load(V1_CONTRACT, "_semantic_anchor_v2_frozen_v1_science")
    contract = _load(CONTRACT, "_semantic_anchor_v2_science_identity")

    assert contract.science_contract() == v1.science_contract()
    assert (
        contract.canonical_json_sha256(contract.science_contract())
        == EXPECTED_SCIENCE_SHA256
    )
    assert contract.science_identity_receipt()[
        "v1_science_contract_sha256"
    ] == EXPECTED_SCIENCE_SHA256
    assert contract.build_schedule_identity() == v1.build_schedule_identity()
    assert contract.runtime_authorization_template()["schedule"] == (
        v1.runtime_authorization_template()["schedule"]
    )
    assert contract.model_config() == v1.model_config()
    assert contract.MODEL_PARAMETER_INVENTORY == v1.MODEL_PARAMETER_INVENTORY
    assert contract.MODEL_RELATIVE_PATH == V1_MODEL
    assert contract.MODEL_RELATIVE_PATH not in contract.ADDITIVE_SOURCE_PATHS
    assert set(contract.REUSED_SOURCE_PATHS) == set(v1.SOURCE_PATHS)

    for name in (
        "MAXIMUM_ATTEMPTS",
        "ATTEMPT_INDEX",
        "MAXIMUM_UPDATES",
        "MAXIMUM_PRESENTATIONS",
        "GPU_ACTIVE_TIME_CAP_MINUTES",
        "EFFECTIVE_BATCH_SIZE",
        "MICROBATCH_SIZE",
        "MICROBATCHES_PER_UPDATE",
        "CHECKPOINT_UPDATES",
        "SNAPSHOT_UPDATES",
        "OBSERVATION_UPDATES",
        "SCHEDULE_PREFIX_SHA256",
        "GATE_THRESHOLDS",
        "INTEGRITY_FIELDS",
        "SEMANTIC_ANCHOR_WEIGHT",
        "STATE_FIELD_CHANNEL_ORDER",
    ):
        assert copy.deepcopy(getattr(contract, name)) == copy.deepcopy(
            getattr(v1, name)
        ), name
    assert contract.perception_accounting is contract._V1.perception_accounting
    assert contract.evaluate_gate is not contract._V1.evaluate_gate
    assert contract.validate_failure_status_chain is not (
        contract._V1.validate_failure_status_chain
    )
    assert set(contract.GATE_CONTROLS) == set(v1.GATE_CONTROLS)
    assert all(
        "STATE_V2_RUNTIME_INTERPRETER_INTEGRITY_REPLACEMENT" in control
        for controls in contract.GATE_CONTROLS.values()
        for control in controls
    )
    assert contract.OPERATIONAL_FAILURE_STATUS == (
        "TERMINAL_SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_"
        "INTERPRETER_INTEGRITY_OR_OPERATIONAL_FAILURE_NO_RETRY"
    )
    for governance_validator in (
        "validate_source_manifest",
        "current_source_bindings",
        "validate_governing_documents",
        "validate_review",
        "validate_authorization",
    ):
        assert getattr(contract, governance_validator) is not getattr(
            contract._V1, governance_validator
        )
    assert contract.SCHEDULE_SCHEMA_ADAPTER_CHANGED is False
    assert contract.SCIENCE_DELTA_COUNT == 0
    assert contract.OUTPUT_ROOT_RELATIVE_PATH == EXPECTED_OUTPUT_ROOT
    assert contract.MAXIMUM_ATTEMPTS == 1
    assert contract.MAXIMUM_UPDATES == 1_000
    assert contract.MAXIMUM_PRESENTATIONS == 16_000
    assert contract.EXECUTION_AUTHORITY[
        "signed_boundary_v1_runtime_output_or_state_reuse_authorized"
    ] is False
    assert contract.EXECUTION_AUTHORITY[
        "retry_resume_repair_recovery_replacement_or_second_seed_authorized"
    ] is False


def test_five_file_delta_closure_and_terminal_audit_binding_are_exact() -> None:
    v1 = _load(V1_CONTRACT, "_semantic_anchor_v2_frozen_v1_sources")
    contract = _load(CONTRACT, "_semantic_anchor_v2_source_identity")
    checker = _load(CHECKER, "_semantic_anchor_v2_closure_identity")

    expected_additive = {
        CONTRACT.relative_to(ROOT).as_posix(),
        TEST.relative_to(ROOT).as_posix(),
        RUNNER.relative_to(ROOT).as_posix(),
        LAUNCHER.relative_to(ROOT).as_posix(),
        CHECKER.relative_to(ROOT).as_posix(),
    }
    assert set(contract.ADDITIVE_SOURCE_PATHS) == expected_additive
    assert len(contract.REUSED_SOURCE_PATHS) == 155
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 5
    assert len(contract.SOURCE_PATHS) == 160
    assert set(contract.SOURCE_PATHS) == set(v1.SOURCE_PATHS) | expected_additive
    assert contract.frozen_v1_terminal_audit_binding() == {
        "path": v1.AUTHORIZATION_RELATIVE_PATH.replace(
            "execution_authorization", "terminal_audit"
        ),
        "commit": "d5f56905202318be1651e8db7b8537909a602432",
        "file_sha256": (
            "59456a174c0adf13800fa3998cd19d454d0a57ac007cd3bd43771d655aa9ce5c"
        ),
        "content_sha256": (
            "6a31cef7757816786f62e7b9e770557bbb2f59f9365a41058b481fbcdcf29ab0"
        ),
        "byte_count": 9_861,
        "status": (
            "PASS_VALID_TERMINAL_RECEIPT_CHAIN_ZERO_WORK_POST_RESERVATION_"
            "INTERPRETER_PREFLIGHT_FAILURE_SEMANTIC_ANCHOR_STATE_V1_CLOSED_"
            "NO_RETRY"
        ),
        "classification": (
            "VALID_POST_RESERVATION_INTERPRETER_PREFLIGHT_OPERATIONAL_FAILURE_"
            "ZERO_WORK_SCIENTIFICALLY_UNEVALUATED_SEMANTIC_ANCHOR_STATE_V1_"
            "CLOSED_NO_RETRY"
        ),
    }

    manifest = checker.build_manifest()
    assert manifest["source_count"] == 160
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
    assert manifest["generated_input_open_count"] == 0
    assert manifest["checkpoint_or_tensor_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0
    assert manifest["whole_tree_export_authorized"] is False


def test_v2_gate_receipt_identity_is_the_only_gate_delta() -> None:
    v1 = _load(V1_CONTRACT, "_semantic_anchor_v2_gate_frozen_v1")
    contract = _load(CONTRACT, "_semantic_anchor_v2_gate_identity")
    fixtures = _load(V1_TEST, "_semantic_anchor_v2_gate_fixtures")
    zero = fixtures._update_zero_metrics(v1)
    hundred = fixtures._update_100_metrics(v1)
    four_hundred = fixtures._update_400_metrics(v1)
    thousand = fixtures._update_1000_metrics(v1)
    cases = (
        (0, zero, {}),
        (100, hundred, {"update_zero": zero}),
        (400, four_hundred, {"update_100": hundred}),
        (1_000, thousand, {"update_400": four_hundred}),
    )
    for update, metrics, kwargs in cases:
        frozen = v1.evaluate_gate(update, metrics, **kwargs)
        replacement = contract.evaluate_gate(update, metrics, **kwargs)
        assert replacement["passed"] is frozen["passed"]
        assert replacement["control"] == contract.GATE_CONTROLS[update][1]
        for receipt_only in ("control", "gate_mode"):
            frozen.pop(receipt_only)
            replacement.pop(receipt_only)
        assert replacement == frozen

    frozen_preliminary = v1.evaluate_gate(0, {})
    replacement_preliminary = contract.evaluate_gate(0, {})
    assert replacement_preliminary["control"] == contract.CONTROL_PRELIMINARY
    for receipt_only in ("control", "gate_mode"):
        frozen_preliminary.pop(receipt_only)
        replacement_preliminary.pop(receipt_only)
    assert replacement_preliminary == frozen_preliminary


def test_launcher_wrong_identity_execs_exact_runtime_once_with_exact_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load(LAUNCHER, "_semantic_anchor_v2_launcher_wrong_identity")
    raw_argv = [
        "--review-sha256",
        "a" * 64,
        "--authorization-sha256",
        "b" * 64,
    ]
    inherited_environment = {
        "HIP_VISIBLE_DEVICES": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
    }
    calls: list[tuple[str, list[str], dict[str, str]]] = []

    class ExecCalled(Exception):
        pass

    def fake_execve(
        executable: str,
        argv: list[str],
        environment: dict[str, str],
    ) -> None:
        calls.append((executable, argv, environment))
        raise ExecCalled

    monkeypatch.delenv(
        launcher.INTERPRETER_HANDOFF_ENVIRONMENT_KEY,
        raising=False,
    )
    # Isolation flags alone are insufficient: the lexical executable must be
    # the preregistered venv path too.
    monkeypatch.setattr(launcher.sys, "executable", "/wrong/current/python")
    monkeypatch.setattr(launcher.sys, "prefix", EXPECTED_RUNTIME_PREFIX)
    monkeypatch.setattr(
        launcher.sys,
        "flags",
        SimpleNamespace(isolated=1),
    )
    monkeypatch.setattr(launcher.sys, "dont_write_bytecode", True)
    monkeypatch.setattr(
        launcher._V1._LEAF._V11._BASE,
        "_launch_environment",
        lambda: dict(inherited_environment),
    )
    monkeypatch.setattr(launcher.os, "execve", fake_execve)
    monkeypatch.setattr(
        launcher._V1,
        "main",
        lambda _argv: pytest.fail("V1 launcher reached before runtime handoff"),
    )

    with pytest.raises(ExecCalled):
        launcher.main(raw_argv)
    assert len(calls) == 1
    executable, argv, environment = calls[0]
    assert executable == EXPECTED_RUNTIME_INTERPRETER
    assert argv == [
        EXPECTED_RUNTIME_INTERPRETER,
        "-I",
        "-B",
        str(LAUNCHER),
        *raw_argv,
    ]
    assert environment == {
        **inherited_environment,
        launcher.INTERPRETER_HANDOFF_ENVIRONMENT_KEY: "1",
    }


def test_launcher_handoff_environment_is_the_inherited_sanitized_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load(LAUNCHER, "_semantic_anchor_v2_launcher_environment")
    monkeypatch.delenv(
        launcher.INTERPRETER_HANDOFF_ENVIRONMENT_KEY,
        raising=False,
    )
    hostile = (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "NVIDIA_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "ZE_AFFINITY_MASK",
        launcher.PREFLIGHT_ENVIRONMENT_KEY,
    )
    for name in hostile:
        monkeypatch.setenv(name, "hostile")
    expected = launcher._V1._LEAF._V11._BASE._launch_environment()
    observed = launcher._runtime_handoff_environment()
    assert observed == {
        **expected,
        launcher.INTERPRETER_HANDOFF_ENVIRONMENT_KEY: "1",
    }
    assert observed["HIP_VISIBLE_DEVICES"] == "0"
    assert observed["PYTHONNOUSERSITE"] == "1"
    assert observed["PYTHONDONTWRITEBYTECODE"] == "1"
    assert all(name not in observed for name in hostile)


def test_launcher_correct_lexical_identity_enters_v1_without_exec_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load(LAUNCHER, "_semantic_anchor_v2_launcher_correct_identity")
    marker = launcher.INTERPRETER_HANDOFF_ENVIRONMENT_KEY
    monkeypatch.setattr(launcher.sys, "executable", EXPECTED_RUNTIME_INTERPRETER)
    monkeypatch.setattr(launcher.sys, "prefix", EXPECTED_RUNTIME_PREFIX)
    monkeypatch.setattr(
        launcher.sys,
        "flags",
        SimpleNamespace(isolated=1),
    )
    monkeypatch.setattr(launcher.sys, "dont_write_bytecode", True)
    monkeypatch.setenv(marker, "1")
    monkeypatch.setattr(
        launcher.os,
        "execve",
        lambda *_args: pytest.fail("correct runtime identity re-execed"),
    )
    observed: list[list[str]] = []

    def fake_main(argv: list[str]) -> int:
        observed.append(argv)
        assert marker not in launcher.os.environ
        return 17

    monkeypatch.setattr(launcher._V1, "main", fake_main)
    assert launcher._runtime_interpreter_matches() is True
    assert launcher.main(["--review-sha256", "a" * 64]) == 17
    assert observed == [["--review-sha256", "a" * 64]]


@pytest.mark.parametrize(
    ("executable", "prefix", "isolated", "dont_write_bytecode"),
    [
        ("/wrong/python", EXPECTED_RUNTIME_PREFIX, 1, True),
        (EXPECTED_RUNTIME_INTERPRETER, "/wrong/prefix", 1, True),
        (EXPECTED_RUNTIME_INTERPRETER, EXPECTED_RUNTIME_PREFIX, 0, True),
        (EXPECTED_RUNTIME_INTERPRETER, EXPECTED_RUNTIME_PREFIX, 1, False),
    ],
)
def test_runtime_identity_rejects_each_wrong_lexical_or_flag_dimension(
    monkeypatch: pytest.MonkeyPatch,
    executable: str,
    prefix: str,
    isolated: int,
    dont_write_bytecode: bool,
) -> None:
    launcher = _load(
        LAUNCHER,
        "_semantic_anchor_v2_launcher_identity_dimension_"
        f"{isolated}_{int(dont_write_bytecode)}_{abs(hash(executable + prefix))}",
    )
    monkeypatch.setattr(launcher.sys, "executable", executable)
    monkeypatch.setattr(launcher.sys, "prefix", prefix)
    monkeypatch.setattr(
        launcher.sys,
        "flags",
        SimpleNamespace(isolated=isolated),
    )
    monkeypatch.setattr(
        launcher.sys,
        "dont_write_bytecode",
        dont_write_bytecode,
    )
    assert launcher._runtime_interpreter_matches() is False


def test_failed_or_repeated_runtime_handoff_rejects_before_v1_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load(LAUNCHER, "_semantic_anchor_v2_launcher_failed_handoff")
    monkeypatch.delenv(
        launcher.INTERPRETER_HANDOFF_ENVIRONMENT_KEY,
        raising=False,
    )
    monkeypatch.setattr(
        launcher._V1,
        "main",
        lambda _argv: pytest.fail("V1 launcher reached after failed exec"),
    )
    calls = 0

    def missing_exec(*_args: object) -> None:
        nonlocal calls
        calls += 1
        raise FileNotFoundError("reviewed interpreter absent")

    monkeypatch.setattr(launcher.os, "execve", missing_exec)
    with pytest.raises(FileNotFoundError, match="interpreter absent"):
        launcher._exec_reviewed_runtime([])
    assert calls == 1

    monkeypatch.setenv(launcher.INTERPRETER_HANDOFF_ENVIRONMENT_KEY, "1")
    with pytest.raises(PermissionError, match="did not establish"):
        launcher._exec_reviewed_runtime([])
    assert calls == 1


def test_direct_runner_wrong_interpreter_rejects_before_inherited_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUNNER, "_semantic_anchor_v2_runner_wrong_interpreter")
    reached: list[str] = []
    monkeypatch.setattr(runner, "_runtime_interpreter_matches", lambda: False)
    monkeypatch.setattr(
        runner._V1,
        "run_parent",
        lambda **_kwargs: reached.append("inherited_run"),
    )
    with pytest.raises(PermissionError, match="before reservation"):
        runner.run_parent(
            review_file_sha256="a" * 64,
            authorization_file_sha256="b" * 64,
        )
    assert reached == []


def test_correct_runner_identity_has_one_inherited_call_and_receipts_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load(RUNNER, "_semantic_anchor_v2_runner_correct_interpreter")
    assert runner.RUNTIME_INTERPRETER_PATH == EXPECTED_RUNTIME_INTERPRETER
    assert runner.RUNTIME_SYS_PREFIX == EXPECTED_RUNTIME_PREFIX

    leaf = runner._V1._LEAF
    assert leaf._snapshot_model is runner._V1._V9._v9_snapshot_model
    assert leaf._terminal_failure is runner._V1._V9._v9_terminal_failure
    assert leaf.contract.validate_failure_status_chain is (
        runner.contract.validate_failure_status_chain
    )
    assert runner.contract.validate_failure_status_chain is not (
        runner.contract._V1.validate_failure_status_chain
    )
    failure_control = runner.contract.GATE_CONTROLS[100][0]
    assert runner.contract.validate_failure_status_chain({
        "metrics": failure_control,
        "artifact": failure_control,
        "result": failure_control,
        "completion": failure_control,
    })["completion"] == failure_control

    calls: list[dict[str, str]] = []
    monkeypatch.setattr(runner, "_runtime_interpreter_matches", lambda: True)

    def fake_run_parent(**kwargs: str) -> int:
        calls.append(kwargs)
        return 23

    monkeypatch.setattr(runner._V1, "run_parent", fake_run_parent)
    assert runner.run_parent(
        review_file_sha256="a" * 64,
        authorization_file_sha256="b" * 64,
    ) == 23
    assert calls == [{
        "review_file_sha256": "a" * 64,
        "authorization_file_sha256": "b" * 64,
    }]
