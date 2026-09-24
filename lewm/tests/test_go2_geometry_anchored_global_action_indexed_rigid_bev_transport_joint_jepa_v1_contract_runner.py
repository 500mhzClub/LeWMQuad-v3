from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import warnings

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "lewm/benchmarks/"
    "go2_geometry_anchored_global_action_indexed_rigid_bev_transport_"
    "joint_jepa_v1.py"
)
RUNNER = (
    ROOT / "scripts/"
    "run_go2_geometry_anchored_global_action_indexed_rigid_bev_transport_"
    "joint_jepa_v1.py"
)
LAUNCHER = (
    ROOT / "scripts/"
    "launch_go2_geometry_anchored_global_action_indexed_rigid_bev_transport_"
    "joint_jepa_v1.py"
)
CHECKER = (
    ROOT / "scripts/"
    "check_go2_geometry_anchored_global_action_indexed_rigid_bev_transport_"
    "joint_jepa_v1_source_closure.py"
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("path", [RUNNER, LAUNCHER])
def test_wrapper_import_is_source_only_and_fully_rebound(path: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(path)!r})
spec = importlib.util.spec_from_file_location("_rigid_transport_wrapper", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert "PIL" not in sys.modules
assert module._V3.contract is module.contract
assert module._V3._V2.contract is module.contract
assert module._V3._V2._V1.contract is module.contract
if hasattr(module, "_run_deterministic"):
    assert module._V3._V2._V1._run_deterministic is module._run_deterministic
    assert module._V3._V2._V1._tensor_state_sha256 is module._tensor_state_sha256
else:
    assert module._V3._V2._V1.RUNNER_PATH == Path(module.contract.RUNNER_RELATIVE_PATH).resolve()
    assert module._V3._V2._V1.OUTPUT_ROOT == (module.ROOT / module.contract.OUTPUT_ROOT_RELATIVE_PATH)
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


def test_contract_is_exact_v3_science_plus_registered_predictor() -> None:
    contract = _load("_rigid_transport_contract_science", CONTRACT)
    v3 = contract._v3
    assert contract.FROZEN_V3_SOURCE_COMMIT == (
        "ebcde189628b1a7040ffaf95aafaf9fd8f404fc4"
    )
    assert contract.FROZEN_V3_EXECUTION_AUTHORIZATION_COMMIT == (
        "3681264a7365d48ad43cbb75e73dba290b8b0134"
    )
    assert len(v3.SOURCE_PATHS) == 84
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 7
    assert len(contract.SOURCE_PATHS) == 91
    assert set(contract.SOURCE_PATHS) == {
        *v3.SOURCE_PATHS,
        *contract.ADDITIVE_SOURCE_PATHS,
    }
    assert contract.MODEL_RELATIVE_PATH != v3.MODEL_RELATIVE_PATH
    assert contract.objective_contract() == v3.objective_contract()
    assert contract.optimizer_contract() == v3.optimizer_contract()
    assert contract.build_schedule_identity() == v3.build_schedule_identity()
    assert contract.GATE_THRESHOLDS == v3.GATE_THRESHOLDS
    assert contract.EXECUTION_AUTHORITY["maximum_updates"] == 1_000
    assert contract.EXECUTION_AUTHORITY["maximum_presentations"] == 16_000
    assert contract.EXECUTION_AUTHORITY["maximum_attempts"] == 1
    assert contract.EXECUTION_AUTHORITY[
        "v3_checkpoint_tensor_trace_optimizer_rng_or_runtime_state_reuse_authorized"
    ] is False
    predictor = contract.model_config()["predictor"]
    assert predictor["parameter_count"] == 184_667
    assert predictor["parameter_tensor_count"] == 11
    assert len(contract.PREDICTOR_ORDERED_PARAMETER_NAMES) == 11
    assert contract.PREDICTOR_ORDERED_PARAMETER_NAMES[0] == "predictor.raw_twist"
    science = contract.science_contract()
    assert science["scientific_question"].startswith(
        "can_a_learned_global_action_indexed_rigid_transform"
    )
    assert "integrity_replacement" not in science
    for stale_name in (
        "integrity_replacement_of",
        "v1_retry",
        "v2_retry_or_resume",
    ):
        assert stale_name not in science["lifecycle"]
    assert science["lifecycle"]["scientific_successor_of"] == (
        contract._v3.EXPERIMENT_ID
    )


def test_governing_preregistration_and_v3_audit_bindings_are_exact() -> None:
    contract = _load("_rigid_transport_contract_documents", CONTRACT)
    validated = contract.validate_governing_documents(ROOT)
    assert validated[contract.PREREGISTRATION_RELATIVE_PATH] == (
        "90bf02ecf88a8ae3d691ca56714556d6b7cbf903a4030e0b05c6806c485bf5bb"
    )
    assert validated[contract.V3_TERMINAL_AUDIT_RELATIVE_PATH] == (
        "bbb1d82faefc62c0358df531941ab07f2b3253d274eca2156df378ffb17a52c4"
    )


def test_gate_and_phase_controls_are_rebound_without_metric_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _load("_rigid_transport_contract_controls", CONTRACT)
    for update in (0, 100, 400, 1_000):
        old_fail, old_pass = contract._v3.GATE_CONTROLS[update]
        for old_control, expected in zip(
            (old_fail, old_pass), contract.GATE_CONTROLS[update], strict=True
        ):
            inherited = {"control": old_control, "passed": old_control == old_pass}
            monkeypatch.setattr(
                contract._v3,
                "evaluate_gate",
                lambda *args, value=inherited, **kwargs: value,
            )
            observed = contract.evaluate_gate(update, {})
            assert observed == {**inherited, "control": expected}
    for old_control, expected in zip(
        contract._v3.PHASE_SWITCH_CONTROLS,
        contract.PHASE_SWITCH_CONTROLS,
        strict=True,
    ):
        inherited = {"control": old_control, "passed": True}
        monkeypatch.setattr(
            contract._v3,
            "evaluate_update_401_phase_switch",
            lambda *args, value=inherited, **kwargs: value,
        )
        assert contract.evaluate_update_401_phase_switch({}) == {
            **inherited,
            "control": expected,
        }


def test_scalar_state_hash_preserves_v3_fix() -> None:
    runner = _load("_rigid_transport_runner_scalar_hash", RUNNER)
    values = {
        "bool": torch.tensor(True, dtype=torch.bool),
        "float": torch.tensor(-3.25, dtype=torch.float32),
        "integer": torch.tensor(20260727, dtype=torch.int64),
    }
    assert runner._tensor_state_sha256(torch, values) == (
        runner._V3._tensor_state_sha256(torch, values)
    )


@pytest.mark.parametrize(
    "suffix",
    (
        " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:.)",
        " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:12)",
        " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:12. )",
        " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:12.) ",
        " (Triggered internally at /tmp/Context.cpp:12.)",
        "\n",
    ),
)
def test_warning_canonicalizer_rejects_every_altered_suffix(suffix: str) -> None:
    runner = _load(f"_rigid_transport_warning_bad_{hash(suffix)}", RUNNER)
    assert runner.canonicalize_rocm_determinism_warning(
        runner.ROCM_GRID_SAMPLE_DETERMINISM_WARNING + suffix
    ) is None


def test_deterministic_wrapper_accepts_only_exact_userwarning_forms() -> None:
    runner = _load("_rigid_transport_warning_good", RUNNER)
    base = runner.ROCM_GRID_SAMPLE_DETERMINISM_WARNING
    suffixed = (
        base
        + " (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:203.)"
    )
    assert runner.canonicalize_rocm_determinism_warning(base) == base
    assert runner.canonicalize_rocm_determinism_warning(suffixed) == base
    sentinel = {"updates": 1_000, "presentations": 16_000}

    def operation() -> dict[str, int]:
        warnings.warn(base, UserWarning)
        warnings.warn(suffixed, UserWarning)
        return sentinel

    observed, receipt = runner._run_deterministic(
        type("Runtime", (), {"torch": torch})(), operation
    )
    assert observed is sentinel
    assert receipt["warning_count"] == 2
    assert receipt["unexpected_warning_count"] == 0
    assert receipt["warning_provenance_suffix_count"] == 1
    assert receipt["warning_categories"] == ["UserWarning", "UserWarning"]
    assert receipt["canonical_warning_message_sha256"] == [
        hashlib.sha256(base.encode("utf-8")).hexdigest()
    ]
    assert runner._BASE._run_deterministic is runner._run_deterministic


@pytest.mark.parametrize("category", [RuntimeWarning, type("Derived", (UserWarning,), {})])
def test_warning_category_failure_retains_returned_science(
    category: type[Warning],
) -> None:
    runner = _load(f"_rigid_transport_warning_category_{category.__name__}", RUNNER)
    sentinel = {"updates": 1_000, "presentations": 16_000, "gate": "computed"}

    def operation() -> dict[str, Any]:
        warnings.warn(runner.ROCM_GRID_SAMPLE_DETERMINISM_WARNING, category)
        return sentinel

    with pytest.raises(runner.DeterministicWarningFailure) as caught:
        runner._run_deterministic(
            type("Runtime", (), {"torch": torch})(), operation
        )
    assert caught.value.scientific_result is sentinel
    receipt = caught.value.warning_receipt
    assert receipt["scientific_callable_returned_before_warning_finalization"] is True
    assert receipt["unexpected_warning_count"] == 1
    assert receipt["warning_categories"] == [category.__name__]


def _terminalize_synthetic_warning_failure(
    runner: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: BaseException,
    *,
    scientific_result: Any | None,
) -> dict[str, Any]:
    progress: dict[str, Any] = {"stage": "synthetic_warning_test"}

    def failed_execute(**_kwargs: Any) -> int:
        raise error

    monkeypatch.setattr(runner, "_BASE_EXECUTE", failed_execute)
    output_root = tmp_path / "attempt"
    output_root.mkdir()
    reservation = {"content_sha256": "a" * 64}
    reservation_raw = b"{}\n"
    with pytest.raises(type(error)):
        runner._execute(
            sources={},
            authorization={},
            reservation=reservation,
            reservation_raw=reservation_raw,
            output_root=output_root,
            progress=progress,
        )
    if scientific_result is not None:
        assert progress["_probe"] is scientific_result[1]
    expected_receipt = getattr(error, "warning_receipt", None)
    if expected_receipt is None:
        expected_receipt = getattr(error, "determinism_warning_receipt")
    assert progress["_determinism"] == expected_receipt
    monkeypatch.setattr(runner._BASE, "_seal", lambda _path: {})
    runner._BASE._terminal_failure(
        output_root,
        reservation,
        reservation_raw,
        progress,
        error,
    )
    return json.loads((output_root / "result.json").read_text(encoding="utf-8"))


def test_post_return_warning_failure_is_persisted_with_computed_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_rigid_transport_warning_terminal_returned", RUNNER)
    receipt = {
        "warning_count": 1,
        "unexpected_warning_count": 1,
        "warning_categories": ["RuntimeWarning"],
        "scientific_callable_returned_before_warning_finalization": True,
    }
    probe = {
        "observations": [{"update": 1_000}],
        "checkpoints": [],
        "training_trace": None,
        "terminal_gate": {"passed": False, "control": "SYNTHETIC_GATE"},
    }
    scientific_result = (object(), probe)
    error = runner.DeterministicWarningFailure(
        "synthetic post-return warning rejection",
        scientific_result=scientific_result,
        warning_receipt=receipt,
    )
    result = _terminalize_synthetic_warning_failure(
        runner,
        monkeypatch,
        tmp_path,
        error,
        scientific_result=scientific_result,
    )
    assert result["determinism"] == receipt
    metrics = json.loads(
        (tmp_path / "attempt" / "metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["terminal_gate"] == probe["terminal_gate"]


def test_callable_exception_warning_receipt_is_persisted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_rigid_transport_warning_terminal_exception", RUNNER)

    def operation() -> None:
        warnings.warn(runner.ROCM_GRID_SAMPLE_DETERMINISM_WARNING, UserWarning)
        raise RuntimeError("synthetic callable failure")

    with pytest.raises(RuntimeError) as caught:
        runner._run_deterministic(
            type("Runtime", (), {"torch": torch})(), operation
        )
    error = caught.value
    receipt = error.determinism_warning_receipt
    assert receipt["warning_count"] == 1
    assert receipt["unexpected_warning_count"] == 0
    assert receipt[
        "scientific_callable_returned_before_warning_finalization"
    ] is False
    result = _terminalize_synthetic_warning_failure(
        runner,
        monkeypatch,
        tmp_path,
        error,
        scientific_result=None,
    )
    assert result["determinism"] == receipt


def test_runner_and_launcher_delegate_with_new_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load("_rigid_transport_runner_delegate", RUNNER)
    calls: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        assert runner._BASE.contract is runner.contract
        assert runner._BASE._load_post_reservation_stack is (
            runner._load_post_reservation_stack
        )
        assert runner._BASE._parameter_receipt is runner._parameter_receipt
        assert runner._BASE._run_deterministic is runner._run_deterministic
        calls.append(argv)
        return 29

    monkeypatch.setattr(runner._V3, "main", fake_main)
    args = ["--review-sha256", "a" * 64, "--authorization-sha256", "b" * 64]
    assert runner.main(args) == 29
    assert calls == [args]

    launcher = _load("_rigid_transport_launcher_delegate", LAUNCHER)
    parsed = launcher.parse_args(args)
    argv = launcher._V3._V2._V1._runtime_argv(parsed)
    assert argv == [
        launcher.contract.RUNTIME_INTERPRETER_PATH,
        *launcher.contract.RUNTIME_INTERPRETER_ARGUMENTS,
        str(ROOT / launcher.contract.RUNNER_RELATIVE_PATH),
        "--review-sha256",
        "a" * 64,
        "--authorization-sha256",
        "b" * 64,
    ]
    assert launcher._V3._V2._V1.OUTPUT_ROOT == (
        ROOT / launcher.contract.OUTPUT_ROOT_RELATIVE_PATH
    )


def test_recursive_closure_is_exactly_v3_plus_seven_sources() -> None:
    checker = _load("_rigid_transport_closure", CHECKER)
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 91
    assert manifest["source_paths"] == list(checker.contract.SOURCE_PATHS)
    assert set(manifest["source_paths"]) == {
        *checker.contract.REUSED_SOURCE_PATHS,
        *checker.contract.ADDITIVE_SOURCE_PATHS,
    }
    assert manifest["entrypoints"] == list(
        checker.contract.SOURCE_MANIFEST_ENTRYPOINTS
    )
    assert manifest["forced_dynamic_sources"] == list(
        checker.contract.SOURCE_PATHS
    )
