from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = (
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v3_"
    "update100_trend_gate_timing"
)
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
TEST = ROOT / "lewm/tests" / f"test_{STEM}.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"
V2_CONTRACT = (
    ROOT
    / "lewm/benchmarks/"
    "go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v2_"
    "runtime_interpreter_integrity_replacement.py"
)
V1_TEST = (
    ROOT
    / "lewm/tests/"
    "test_go2_direct_egocentric_bev_signed_boundary_semantic_anchor_state_v1.py"
)
EXPECTED_SCIENCE_SHA256 = (
    "2d42031e0586c205cfcae783991a497a4b3f4a5b1c5b8013aa3e65ac5ca673f1"
)
EXPECTED_RUNTIME_INTERPRETER = (
    "/home/andrewknowles/.local/share/"
    "lewmquad-v12-runtime-torch291-rocm64/bin/python"
)
EXPECTED_RUNTIME_PREFIX = (
    "/home/andrewknowles/.local/share/"
    "lewmquad-v12-runtime-torch291-rocm64"
)
EXPECTED_OUTPUT_ROOT = (
    ".generated/go2_shared_observable_camera_ray_jepa_signed_boundary_"
    "semantic_anchor_state_v3/rgb_direct_egocentric_bev_signed_boundary_"
    "semantic_anchor_state_probe_v3_update100_trend_gate_timing_v1"
)
BA_CONJUNCT = (
    "balanced_accuracy_at_least_max_point68_or_update_zero_plus_point08"
)


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _without_family_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(value))
    result.pop("control", None)
    result.pop("gate_mode", None)
    return result


@pytest.mark.parametrize("source", [CONTRACT, RUNNER, LAUNCHER, CHECKER])
def test_v3_sources_import_source_only(source: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(source)!r})
spec = importlib.util.spec_from_file_location('_semantic_anchor_v3_source_only', path)
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


def test_training_mechanism_is_exact_v2_and_decision_delta_is_disclosed() -> None:
    v2 = _load(V2_CONTRACT, "_semantic_anchor_v3_frozen_v2_science")
    contract = _load(CONTRACT, "_semantic_anchor_v3_science_identity")

    normalized = contract.normalize_v3_decision_protocol(
        contract.science_contract()
    )
    assert normalized == v2.science_contract()
    assert contract.canonical_json_sha256(normalized) == EXPECTED_SCIENCE_SHA256
    receipt = contract.science_identity_receipt()
    assert receipt["inherited_v2_science_contract_sha256"] == (
        EXPECTED_SCIENCE_SHA256
    )
    assert receipt["normalized_v3_decision_protocol_sha256"] == (
        EXPECTED_SCIENCE_SHA256
    )
    assert receipt["normalized_exactly_equals_v2"] is True
    assert receipt["training_science_delta_count"] == 0
    assert receipt["evaluation_decision_protocol_delta_count"] == 1
    assert receipt["changed_evaluation_paths"] == [
        "schema",
        "gates.controls",
        "gates.evaluation_decision_protocol",
    ]
    assert receipt["v2_runtime_reuse_authorized"] is False
    assert receipt["predictor_training_or_evaluation_authorized"] is False
    assert receipt["v3_full_evaluation_contract_sha256"] != (
        EXPECTED_SCIENCE_SHA256
    )

    assert contract.build_schedule_identity() == v2.build_schedule_identity()
    assert contract.model_config() == v2.model_config()
    assert contract.MODEL_PARAMETER_INVENTORY == v2.MODEL_PARAMETER_INVENTORY
    assert contract.MODEL_RELATIVE_PATH == v2.MODEL_RELATIVE_PATH
    assert contract.MODEL_RELATIVE_PATH not in contract.ADDITIVE_SOURCE_PATHS
    assert contract.runtime_authorization_template()["schedule"] == (
        v2.runtime_authorization_template()["schedule"]
    )
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
            getattr(v2, name)
        ), name


def test_v3_source_surface_and_v2_terminal_audit_binding_are_exact() -> None:
    v2 = _load(V2_CONTRACT, "_semantic_anchor_v3_frozen_v2_sources")
    contract = _load(CONTRACT, "_semantic_anchor_v3_source_identity")
    checker = _load(CHECKER, "_semantic_anchor_v3_closure_identity")

    expected_additive = {
        CONTRACT.relative_to(ROOT).as_posix(),
        TEST.relative_to(ROOT).as_posix(),
        RUNNER.relative_to(ROOT).as_posix(),
        LAUNCHER.relative_to(ROOT).as_posix(),
        CHECKER.relative_to(ROOT).as_posix(),
    }
    assert set(contract.ADDITIVE_SOURCE_PATHS) == expected_additive
    assert set(contract.REUSED_SOURCE_PATHS) == set(v2.SOURCE_PATHS)
    assert len(contract.REUSED_SOURCE_PATHS) == 160
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 5
    assert len(contract.SOURCE_PATHS) == 165
    assert set(contract.SOURCE_PATHS) == set(v2.SOURCE_PATHS) | expected_additive
    assert contract.frozen_v2_terminal_audit_binding() == {
        "path": (
            "docs/lewm_go2_rgb_direct_egocentric_bev_signed_boundary_"
            "semantic_anchor_state_v2_runtime_interpreter_integrity_"
            "replacement_terminal_audit_2026-07-27.json"
        ),
        "commit": "ef8dfcf5ed659b64cf6adf7480c904cbeb61357c",
        "file_sha256": (
            "88a0b03fde5f5cda2088576ae0fd12ef5c8d5dc47925a4df7e7defa85c8132b8"
        ),
        "content_sha256": (
            "3e31ce22a5c19b553299c32155291936d033aedb4a2588842c3251bc0d8c7bc3"
        ),
        "byte_count": 23_895,
        "status": (
            "PASS_VALID_TERMINAL_RECEIPT_CHAIN_UPDATE_100_SCIENTIFIC_FAILURE_"
            "SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_INTERPRETER_"
            "INTEGRITY_REPLACEMENT_CLOSED_NO_RETRY"
        ),
        "classification": (
            "VALID_UPDATE_100_SINGLE_BALANCED_ACCURACY_CONJUNCT_SCIENTIFIC_"
            "FAILURE_AFTER_STRONG_OBJECTIVE_NLL_AND_SEMANTIC_IMPROVEMENT_"
            "SIGNED_BOUNDARY_SEMANTIC_ANCHOR_STATE_V2_RUNTIME_INTERPRETER_"
            "INTEGRITY_REPLACEMENT_CLOSED_NO_RETRY"
        ),
    }
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 165
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
    assert manifest["generated_input_open_count"] == 0
    assert manifest["checkpoint_or_tensor_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0
    assert manifest["whole_tree_export_authorized"] is False


def test_update100_balanced_accuracy_only_miss_continues_with_exact_evidence() -> None:
    v2 = _load(V2_CONTRACT, "_semantic_anchor_v3_ba_only_frozen_v2")
    contract = _load(CONTRACT, "_semantic_anchor_v3_ba_only_gate")
    fixtures = _load(V1_TEST, "_semantic_anchor_v3_ba_only_fixtures")
    zero = fixtures._update_zero_metrics(contract)
    hundred = fixtures._update_100_metrics(contract)
    threshold = max(
        0.68,
        zero["aggregate_raster_balanced_accuracy"] + 0.08,
    )
    hundred["aggregate_raster_balanced_accuracy"] = math.nextafter(
        threshold, -math.inf
    )
    before = copy.deepcopy(hundred)

    frozen = v2.evaluate_gate(100, hundred, update_zero=zero)
    observed = contract.evaluate_gate(100, hundred, update_zero=zero)
    assert hundred == before
    assert frozen["passed"] is False
    assert [key for key, value in frozen["conjuncts"].items() if not value] == [
        BA_CONJUNCT
    ]
    assert observed["passed"] is True
    assert observed["control"] == contract.GATE_CONTROLS[100][1]
    assert observed["conjuncts"] == frozen["conjuncts"]
    assert observed["original_v2_conjuncts"] == frozen["conjuncts"]
    assert observed["active_conjuncts"] == {
        key: value
        for key, value in frozen["conjuncts"].items()
        if key != BA_CONJUNCT
    }
    assert observed["deferred_conjunct"] == BA_CONJUNCT
    assert observed["original_v2_gate_passed"] is False
    assert observed["original_v2_control"] == frozen["control"]
    evidence = observed["balanced_accuracy_evidence"]
    assert evidence == {
        "BA_0": zero["aggregate_raster_balanced_accuracy"],
        "BA_100": hundred["aggregate_raster_balanced_accuracy"],
        "BA_threshold": threshold,
        "BA_threshold_formula": "max(0.68,BA_0+0.08)",
        "BA_pass": False,
        "BA_margin": hundred["aggregate_raster_balanced_accuracy"] - threshold,
        "original_V2_gate_would_pass": False,
        "balanced_accuracy_recorded": True,
        "balanced_accuracy_applied_as_terminal_conjunct": False,
    }


def test_update100_all_pass_continues_and_any_other_miss_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    v2 = _load(V2_CONTRACT, "_semantic_anchor_v3_truth_frozen_v2")
    contract = _load(CONTRACT, "_semantic_anchor_v3_truth_gate")
    fixtures = _load(V1_TEST, "_semantic_anchor_v3_truth_fixtures")
    zero = fixtures._update_zero_metrics(contract)
    hundred = fixtures._update_100_metrics(contract)

    frozen_pass = v2.evaluate_gate(100, hundred, update_zero=zero)
    observed_pass = contract.evaluate_gate(100, hundred, update_zero=zero)
    assert frozen_pass["passed"] is True
    assert observed_pass["passed"] is True
    assert observed_pass["original_v2_gate_passed"] is True
    assert observed_pass["balanced_accuracy_evidence"]["BA_pass"] is True

    receipt = copy.deepcopy(frozen_pass)
    calls = 0

    def fake_v2_gate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return copy.deepcopy(receipt)

    monkeypatch.setattr(contract._V2, "evaluate_gate", fake_v2_gate)
    active_keys = tuple(
        key for key in frozen_pass["conjuncts"] if key != BA_CONJUNCT
    )
    for key in active_keys:
        receipt = copy.deepcopy(frozen_pass)
        receipt["conjuncts"][key] = False
        receipt["passed"] = False
        receipt["control"] = contract._V2.GATE_CONTROLS[100][0]
        observed = contract.evaluate_gate(100, hundred, update_zero=zero)
        assert observed["passed"] is False, key
        assert observed["control"] == contract.GATE_CONTROLS[100][0], key
        assert observed["active_conjuncts"][key] is False, key
    assert calls == len(active_keys)


def test_u0_u400_u1000_math_is_exact_v2_with_only_v3_identity() -> None:
    v2 = _load(V2_CONTRACT, "_semantic_anchor_v3_other_gates_frozen_v2")
    contract = _load(CONTRACT, "_semantic_anchor_v3_other_gates")
    fixtures = _load(V1_TEST, "_semantic_anchor_v3_other_gate_fixtures")
    zero = fixtures._update_zero_metrics(contract)
    hundred = fixtures._update_100_metrics(contract)
    four_hundred = fixtures._update_400_metrics(contract)
    thousand = fixtures._update_1000_metrics(contract)
    cases = (
        (0, zero, {}),
        (400, four_hundred, {"update_100": hundred}),
        (1_000, thousand, {"update_400": four_hundred}),
    )
    for update, metrics, kwargs in cases:
        before = copy.deepcopy(metrics)
        frozen = v2.evaluate_gate(update, metrics, **kwargs)
        observed = contract.evaluate_gate(update, metrics, **kwargs)
        assert metrics == before
        assert observed["passed"] is frozen["passed"]
        assert observed["control"] == contract.GATE_CONTROLS[update][1]
        assert _without_family_identity(observed) == _without_family_identity(
            frozen
        )


def test_no_u200_and_caps_schedule_interpreter_root_are_exact() -> None:
    contract = _load(CONTRACT, "_semantic_anchor_v3_caps")
    assert contract.OBSERVATION_UPDATES == (0, 100, 400, 1_000)
    assert contract.CHECKPOINT_UPDATES == (100, 400, 1_000)
    assert contract.SNAPSHOT_UPDATES == (100, 400, 1_000)
    assert 200 not in contract.SCHEDULE_PREFIX_SHA256
    assert contract.MAXIMUM_ATTEMPTS == 1
    assert contract.MAXIMUM_UPDATES == 1_000
    assert contract.MAXIMUM_PRESENTATIONS == 16_000
    assert contract.RUNTIME_INTERPRETER_PATH == EXPECTED_RUNTIME_INTERPRETER
    assert contract.RUNTIME_SYS_PREFIX == EXPECTED_RUNTIME_PREFIX
    assert contract.OUTPUT_ROOT_RELATIVE_PATH == EXPECTED_OUTPUT_ROOT
    with pytest.raises(ValueError):
        contract.evaluate_gate(200, {})

    runtime = contract.runtime_authorization_template()
    scope = runtime["experiment_scope"]
    assert runtime["schedule"]["observation_updates"] == [0, 100, 400, 1000]
    assert scope["maximum_attempts"] == 1
    assert scope["maximum_updates"] == 1_000
    assert scope["maximum_presentations"] == 16_000
    assert scope["output_root"] == EXPECTED_OUTPUT_ROOT
    assert scope["prior_runtime_or_checkpoint_reuse"] is False
    assert scope[
        "retry_resume_repair_recovery_extension_second_seed_or_second_attempt"
    ] is False


def test_failure_chain_and_downstream_reuse_denials_are_exact() -> None:
    contract = _load(CONTRACT, "_semantic_anchor_v3_failure_and_denials")
    for update in contract.GATE_CONTROLS:
        failure = contract.GATE_CONTROLS[update][0]
        chain = {
            "metrics": failure,
            "artifact": failure,
            "result": failure,
            "completion": failure,
        }
        assert contract.validate_failure_status_chain(chain) == chain
    mismatch = {
        "metrics": contract.GATE_CONTROLS[100][0],
        "artifact": contract.GATE_CONTROLS[100][0],
        "result": contract.GATE_CONTROLS[100][0],
        "completion": contract._V2.GATE_CONTROLS[100][0],
    }
    with pytest.raises(ValueError):
        contract.validate_failure_status_chain(mismatch)

    authority = contract.EXECUTION_AUTHORITY
    for key in (
        "predictor_training_or_evaluation_authorized",
        "g2_authorized",
        "navigation_authorized",
        "heldout_authorized",
        "sealed_authorized",
        "production_authorized",
        "promotion_authorized",
        "deployment_authorized",
        "v2_runtime_output_or_state_reuse_authorized",
        "retry_resume_repair_recovery_replacement_or_second_seed_authorized",
    ):
        assert authority[key] is False, key
