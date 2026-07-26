from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_masked_current_next_pair_tubelet_jepa_v13_learning_curve_continuation.py"
LAUNCHER_PATH = ROOT / "scripts/launch_go2_rgb_masked_current_next_pair_tubelet_jepa_v13_learning_curve_continuation.py"
V12_CONTRACT_PATH = ROOT / "lewm/benchmarks/go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py"
V11_FIXTURES_PATH = ROOT / "lewm/tests/test_go2_rgb_masked_current_next_pair_tubelet_jepa_v11_contract.py"
V13_ROOT = ".generated/go2_shared_observable_camera_ray_jepa_v5/rgb_masked_current_next_pair_tubelet_jepa_probe_v13"
V11_ROOT = V13_ROOT[:-3] + "v11"
V12_ROOT = V13_ROOT[:-3] + "v12"
PREFLIGHT_KEY = "LEWM_RGB_MASKED_CURRENT_NEXT_PAIR_TUBELET_JEPA_V13_LEARNING_CURVE_CONTINUATION_PREFLIGHT_JSON"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_imports_are_source_only_and_all_layers_are_v13_bound() -> None:
    program = f"""
import importlib.util, sys
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
r = load({str(RUNNER_PATH)!r}, "_v13_source_runner")
l = load({str(LAUNCHER_PATH)!r}, "_v13_source_launcher")
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert r.contract.OUTPUT_ROOT_RELATIVE_PATH == {V13_ROOT!r}
assert r.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert r._V12.contract is r.contract
assert r._V12._V11.contract is r.contract
assert r._V12._V11._V10.contract is r.contract
assert l._V12.contract is l.contract
assert l._V12._V11.contract is l.contract
assert l._V12._V11._BASE.contract is l.contract
assert l._V12._V11._BASE.RUNNER_PATH == l.ROOT / l.contract.RUNNER_RELATIVE_PATH
assert l._V12._V11._BASE.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
print("PASS")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "PASS\n"
    assert result.stderr == ""


def test_inherited_function_globals_and_preflight_globals_are_v13() -> None:
    runner = _load(RUNNER_PATH, "_v13_nested_runner")
    launcher = _load(LAUNCHER_PATH, "_v13_nested_launcher")
    inherited = runner._V12._V11

    assert inherited._phase_a_train.__globals__["contract"] is runner.contract
    assert inherited._execute_after_reservation.__globals__["contract"] is runner.contract
    assert inherited._load_authority_pre_reservation.__globals__["contract"] is runner.contract
    assert inherited._reserve.__globals__["contract"] is runner.contract
    assert inherited._V10.PREFLIGHT_ENVIRONMENT_KEY == PREFLIGHT_KEY
    assert runner._V12.PREFLIGHT_ENVIRONMENT_KEY == PREFLIGHT_KEY

    base = launcher._V12._V11._BASE
    assert base._load_authority_before_hardware.__globals__["contract"] is launcher.contract
    assert base.PREFLIGHT_ENVIRONMENT_KEY == PREFLIGHT_KEY
    assert str(launcher._V12.__file__) == str(LAUNCHER_PATH)
    assert str(launcher._V12._V11.__file__) == str(LAUNCHER_PATH)
    assert str(base.__file__) == str(LAUNCHER_PATH)


def test_cli_lifecycle_one_phase_and_caps_are_exact_v13() -> None:
    runner = _load(RUNNER_PATH, "_v13_cli_runner")
    launcher = _load(LAUNCHER_PATH, "_v13_cli_launcher")
    digest_a, digest_b = "a" * 64, "b" * 64

    args = runner.parse_args([
        "--run", "--review-sha256", digest_a,
        "--authorization-sha256", digest_b,
    ])
    assert (args.run, args.review_sha256, args.authorization_sha256) == (True, digest_a, digest_b)
    launch_args = launcher.parse_args([
        "--review-sha256", digest_a,
        "--authorization-sha256", digest_b,
    ])
    assert (launch_args.review_sha256, launch_args.authorization_sha256) == (digest_a, digest_b)

    contract = runner.contract
    science = contract.science_contract()
    assert contract.OUTPUT_ROOT_RELATIVE_PATH == V13_ROOT
    assert contract.RUNNER_RELATIVE_PATH == RUNNER_PATH.relative_to(ROOT).as_posix()
    assert contract.LAUNCHER_RELATIVE_PATH == LAUNCHER_PATH.relative_to(ROOT).as_posix()
    assert science["lifecycle"]["attempt_index"] == 1
    assert science["lifecycle"]["maximum_attempts"] == 1
    assert science["lifecycle"]["output_root"] == V13_ROOT
    assert science["lifecycle"]["phase_b_authorized"] is False
    assert science["lifecycle"]["retry_resume_replacement_authorized"] is False
    assert contract.PHASE_A_MAXIMUM_UPDATE == 1_000
    assert contract.MAXIMUM_PRESENTATIONS == 16_000
    assert tuple(contract.CHECKPOINT_UPDATES) == (100, 400, 1_000)
    assert contract.PHASE_B_MAXIMUM_UPDATE == 0
    inherited_source = inspect.getsource(runner._V12._V11)
    assert "_phase_b_train(" not in inherited_source
    assert '"phase_b": None' in inherited_source
    for path in (RUNNER_PATH, LAUNCHER_PATH):
        source = path.read_text(encoding="utf-8")
        assert V11_ROOT not in source
        assert V12_ROOT not in source


def test_fresh_v13_reservation_precedes_inherited_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _load(RUNNER_PATH, "_v13_reservation_runner")
    inherited = runner._V12._V11
    expected_root = ROOT / V13_ROOT
    events: list[str] = []

    def authority(review: str, authorization: str):
        events.append("authority")
        assert (review, authorization) == ("a" * 64, "b" * 64)
        return ({}, b"review", {}, b"authorization", {})

    def reserve(root: Path, **kwargs: Any):
        events.append("reserve")
        assert root == expected_root
        assert events == ["authority", "reserve"]
        return ({"reserved": True}, b"reservation")

    def execute(**kwargs: Any) -> int:
        events.append("runtime")
        assert events == ["authority", "reserve", "runtime"]
        assert kwargs["output_root"] == expected_root
        assert kwargs["progress"]["stage"] == "reserved"
        return 13

    monkeypatch.setattr(inherited, "_load_authority_pre_reservation", authority)
    monkeypatch.setattr(inherited, "_reserve", reserve)
    monkeypatch.setattr(inherited, "_execute_after_reservation", execute)
    assert runner.run_parent(
        review_file_sha256="a" * 64,
        authorization_file_sha256="b" * 64,
    ) == 13
    assert events == ["authority", "reserve", "runtime"]


def test_update100_uses_v13_gate_and_update400_matches_frozen_v12() -> None:
    runner = _load(RUNNER_PATH, "_v13_gate_runner")
    contract = runner.contract
    v12 = _load(V12_CONTRACT_PATH, "_v12_frozen_gate_reference")
    fixtures = _load(V11_FIXTURES_PATH, "_v11_gate_fixtures_for_v13")

    update0 = fixtures._update0()
    update100 = fixtures._update100()
    update100["true_pair_mse"] = 0.70
    gate100 = contract.evaluate_phase_a_continuation(
        100, update100, update0, fixtures._integrity(100)
    )
    expected_directional = {
        "non_hold_correct_to_fixed_current_strictly_below_one",
        "normalized_projected_future_effective_rank_strictly_above_update0",
        "action_retrieval_nll_strictly_below_update0",
        "action_retrieval_macro_balanced_accuracy_strictly_above_update0",
        "target_retrieval_nll_strictly_below_update0",
        "same_action_target_retrieval_nll_strictly_below_update0",
        "same_action_two_target_nll_strictly_below_update0",
        "same_action_strict_win_rate_strictly_above_update0",
        "same_action_correct_to_deranged_ratio_strictly_below_update0",
        "true_to_mean_target_ratio_strictly_below_update0",
    }
    assert gate100["passed"] is True
    assert len(gate100["conjuncts"]) == 22
    assert expected_directional <= set(gate100["conjuncts"])

    update100_original = fixtures._update100()
    update400 = fixtures._update400()
    observed = contract.evaluate_phase_a_continuation(
        400, update400, update0, fixtures._integrity(400), update100_original
    )
    expected = v12.evaluate_phase_a_continuation(
        400, update400, update0, fixtures._integrity(400), update100_original
    )
    assert observed == expected
    assert "contract.evaluate_phase_a_continuation(" in inspect.getsource(
        runner._V12._V11._phase_a_train
    )
    assert runner._V12._V11._phase_a_train.__globals__["contract"] is contract
