from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/"
    "run_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py"
)
LAUNCHER_PATH = (
    ROOT
    / "scripts/"
    "launch_go2_rgb_masked_current_next_pair_tubelet_jepa_v12_gate_timing.py"
)
V11_CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/"
    "go2_rgb_masked_current_next_pair_tubelet_jepa_v11.py"
)
V12_OUTPUT_ROOT = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "rgb_masked_current_next_pair_tubelet_jepa_probe_v12"
)
V11_OUTPUT_ROOT = V12_OUTPUT_ROOT[:-3] + "v11"
PREFLIGHT_ENVIRONMENT_KEY = (
    "LEWM_RGB_MASKED_CURRENT_NEXT_PAIR_TUBELET_JEPA_"
    "V12_GATE_TIMING_PREFLIGHT_JSON"
)
REMOVED_AT_UPDATE_ZERO = {
    "true_at_most_point90_shuffled_next",
    "true_at_most_point95_shuffled_current",
}


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner(name: str) -> Any:
    return _load(RUNNER_PATH, name)


def _load_launcher(name: str) -> Any:
    return _load(LAUNCHER_PATH, name)


def test_runner_and_launcher_import_source_only_with_v12_bindings() -> None:
    program = f"""
import importlib.util
import sys

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

runner = load({str(RUNNER_PATH)!r}, "_v12_runner_source_only")
launcher = load({str(LAUNCHER_PATH)!r}, "_v12_launcher_source_only")
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert runner.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_ENVIRONMENT_KEY!r}
assert launcher.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_ENVIRONMENT_KEY!r}
assert runner.contract.OUTPUT_ROOT_RELATIVE_PATH == {V12_OUTPUT_ROOT!r}
assert runner._V11.contract is runner.contract
assert runner._V11._V10.contract is runner.contract
assert launcher._V11.contract is launcher.contract
assert launcher._V11._BASE.contract is launcher.contract
assert launcher._V11._BASE.RUNNER_PATH == launcher.ROOT / launcher.contract.RUNNER_RELATIVE_PATH
assert launcher._V11._BASE.PREFLIGHT_ENVIRONMENT_KEY == launcher.PREFLIGHT_ENVIRONMENT_KEY
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


def test_both_inherited_runner_contract_levels_resolve_v12() -> None:
    runner = _load_runner("_v12_inherited_runner_globals")
    launcher = _load_launcher("_v12_inherited_launcher_globals")

    assert runner._V11._phase_a_train.__globals__["contract"] is runner.contract
    assert (
        runner._V11._execute_after_reservation.__globals__["contract"]
        is runner.contract
    )
    assert (
        runner._V11._load_authority_pre_reservation.__globals__["contract"]
        is runner.contract
    )
    assert runner._V11._reserve.__globals__["contract"] is runner.contract
    assert runner._V11._V10.contract is runner.contract
    assert (
        runner._V11._V10.PREFLIGHT_ENVIRONMENT_KEY
        == PREFLIGHT_ENVIRONMENT_KEY
    )

    assert launcher._V11.contract is launcher.contract
    assert launcher._V11._BASE.contract is launcher.contract
    assert (
        launcher._V11._BASE._load_authority_before_hardware
        .__globals__["contract"]
        is launcher.contract
    )
    assert str(launcher._V11._BASE.__file__) == str(LAUNCHER_PATH)


def test_cli_is_v12_bound_without_execution() -> None:
    runner = _load_runner("_v12_runner_cli")
    launcher = _load_launcher("_v12_launcher_cli")
    review = "a" * 64
    authorization = "b" * 64

    runner_args = runner.parse_args(
        [
            "--run",
            "--review-sha256",
            review,
            "--authorization-sha256",
            authorization,
        ]
    )
    assert runner_args.run is True
    assert runner_args.review_sha256 == review
    assert runner_args.authorization_sha256 == authorization

    launcher_args = launcher.parse_args(
        [
            "--review-sha256",
            review,
            "--authorization-sha256",
            authorization,
        ]
    )
    assert launcher_args.review_sha256 == review
    assert launcher_args.authorization_sha256 == authorization

    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--review-sha256",
                review,
                "--authorization-sha256",
                authorization,
            ]
        )


def test_v12_lifecycle_is_fresh_one_phase_and_exactly_capped() -> None:
    runner = _load_runner("_v12_lifecycle")
    contract = runner.contract
    science = contract.science_contract()

    assert contract.OUTPUT_ROOT_RELATIVE_PATH == V12_OUTPUT_ROOT
    assert V11_OUTPUT_ROOT not in contract.OUTPUT_ROOT_RELATIVE_PATH
    assert contract.RUNNER_RELATIVE_PATH == RUNNER_PATH.relative_to(ROOT).as_posix()
    assert (
        contract.LAUNCHER_RELATIVE_PATH
        == LAUNCHER_PATH.relative_to(ROOT).as_posix()
    )
    assert science["lifecycle"] == {
        **science["lifecycle"],
        "attempt_index": 1,
        "maximum_attempts": 1,
        "output_root": V12_OUTPUT_ROOT,
        "phase_b_authorized": False,
        "retry_resume_replacement_authorized": False,
    }
    assert contract.PHASE_A_MAXIMUM_UPDATE == 1_000
    assert contract.MAXIMUM_PRESENTATIONS == 16_000
    assert tuple(contract.CHECKPOINT_UPDATES) == (100, 400, 1_000)
    assert contract.PHASE_B_MAXIMUM_UPDATE == 0

    inherited_source = inspect.getsource(runner._V11)
    assert "_phase_b_train(" not in inherited_source
    assert 'progress["phase_b_entered"] = True' not in inherited_source
    assert '"phase_b": None' in inherited_source
    assert V11_OUTPUT_ROOT not in RUNNER_PATH.read_text(encoding="utf-8")
    assert V11_OUTPUT_ROOT not in LAUNCHER_PATH.read_text(encoding="utf-8")


def test_inherited_run_reserves_fresh_v12_root_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner("_v12_reservation_order")
    events: list[str] = []
    expected_root = ROOT / V12_OUTPUT_ROOT

    def load_authority(review_sha: str, authorization_sha: str):
        events.append("authority")
        assert review_sha == "a" * 64
        assert authorization_sha == "b" * 64
        return ({"review": True}, b"review", {"authorization": True}, b"auth", {})

    def reserve(output_root: Path, **kwargs: Any):
        events.append("reserve")
        assert output_root == expected_root
        assert events == ["authority", "reserve"]
        assert kwargs["sources"] == {}
        return ({"reserved": True}, b"reservation")

    def execute(**kwargs: Any) -> int:
        events.append("runtime")
        assert events == ["authority", "reserve", "runtime"]
        assert kwargs["output_root"] == expected_root
        assert kwargs["reservation"] == {"reserved": True}
        assert kwargs["progress"]["stage"] == "reserved"
        return 17

    monkeypatch.setattr(
        runner._V11, "_load_authority_pre_reservation", load_authority
    )
    monkeypatch.setattr(runner._V11, "_reserve", reserve)
    monkeypatch.setattr(runner._V11, "_execute_after_reservation", execute)

    assert runner.run_parent(
        review_file_sha256="a" * 64,
        authorization_file_sha256="b" * 64,
    ) == 17
    assert events == ["authority", "reserve", "runtime"]


def test_update_zero_uses_v12_delta_and_later_gate_code_is_frozen_v11(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner("_v12_gate_dispatch")
    contract = runner.contract
    v11 = _load(V11_CONTRACT_PATH, "_v11_gate_reference_for_v12_runner")

    common = {
        "control_populations_and_one_scene_per_family_exact": True,
        "ema_inventory_and_update_count_exact": True,
        "factorized_retrieval_health_exact": True,
        "finite_and_ema_gradient_free": True,
        "future_leakage_target_nonavuity_and_autograd_isolation": True,
        "normalized_population_and_all_nine_candidates_exact": True,
        "normalized_projected_future_spatial_diversity_at_least_quarter_update0": True,
        "normalized_projected_future_variance_at_least_quarter_update0": True,
        "observation_rng_and_model_state_preserved": True,
        "retrieval_references_and_mappings_immutable": True,
        "true_at_most_point90_shuffled_next": False,
        "true_at_most_point95_shuffled_current": False,
        "update_zero_action_symmetry_and_chance_exact": True,
        "update_zero_factorized_retrieval_health_exact": True,
    }
    normalized = {
        "common": common,
        "ratios": {
            "true_to_shuffled_next": 0.94,
            "true_to_shuffled_current": 0.96,
        },
        "per_family": {},
        "retrieval": {},
    }
    monkeypatch.setattr(
        contract,
        "_normalize_v11_phase_a_inputs",
        lambda *_args: normalized,
    )
    gate = contract.evaluate_phase_a_update_zero({}, {}, {})
    assert set(gate["conjuncts"]) == set(common) - REMOVED_AT_UPDATE_ZERO
    assert gate["passed"] is True
    assert gate["control"] == contract.CONTROL_CONTINUE
    assert gate["ratios"] == normalized["ratios"]

    common["future_leakage_target_nonavuity_and_autograd_isolation"] = False
    failed = contract.evaluate_phase_a_update_zero({}, {}, {})
    assert failed["passed"] is False
    assert failed["control"] == contract.CONTROL_PHASE_A_UPDATE_ZERO_FAIL

    for name in ("evaluate_phase_a_continuation", "evaluate_phase_a"):
        v12_code = getattr(contract, name).__code__
        v11_code = getattr(v11, name).__code__
        assert v12_code.co_code == v11_code.co_code
        assert v12_code.co_names == v11_code.co_names
        assert v12_code.co_varnames == v11_code.co_varnames
    assert (
        contract.PHASE_A_UPDATE_100_THRESHOLDS
        == v11.PHASE_A_UPDATE_100_THRESHOLDS
    )
    assert (
        contract.PHASE_A_UPDATE_400_THRESHOLDS
        == v11.PHASE_A_UPDATE_400_THRESHOLDS
    )
    assert contract.PHASE_A_PASS_THRESHOLDS == v11.PHASE_A_PASS_THRESHOLDS
    assert "contract.evaluate_phase_a_update_zero(" in inspect.getsource(
        runner._V11._phase_a_train
    )
    assert runner._V11._phase_a_train.__globals__["contract"] is contract
