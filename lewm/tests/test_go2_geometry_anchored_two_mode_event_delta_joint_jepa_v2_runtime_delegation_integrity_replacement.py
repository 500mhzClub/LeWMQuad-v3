from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = (
    "go2_geometry_anchored_two_mode_event_delta_joint_jepa_v2_"
    "runtime_delegation_integrity_replacement"
)
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"
TEST_RELATIVE_PATH = f"lewm/tests/test_{STEM}.py"

FROZEN_V1_SCIENCE_CONTRACT_SHA256 = (
    "26c095f0b330e6e43952814e6a3b910f15b72a906d1c2f3d931a70c959ae6974"
)
FROZEN_V1_SCIENCE_COMPONENT_SHA256 = {
    "model": "4c84691d76eaf2c3b5eee345bb3b1c9cf8dd747e9512fc91c9d6f74b37337b03",
    "objective": "85017d1618e75970a2e70e1ace6f6930650aa5b351c60855753bcdceaa3515d4",
    "optimizer": "2bb70f943838b656540b3dac3b6e0f30bb384547180270274abfc5077e264b34",
    "schedule": "bc0ad45c06171cff7533fbfcb054e5afecf6086de0a58060c35cb5ca0256c2e3",
    "gate_thresholds": (
        "97fa8bb4b2740e68cadf90974ab80ff33419a854b07a16a258e2f49c3f177036"
    ),
    "work_accounting": (
        "013837055e693ae754324d7c9b8b098d47efed5f569505cf0f58fca8b432e359"
    ),
    "warning_policy": (
        "01a958d0de33a399453c7262d07f6328aabb3bbeaa83cfa045f52cdd03b6a67b"
    ),
    "runtime_input_template": (
        "393563699929bbfd7ca4d9c97c2c63b8a2583bfcc093f61ca0926cb63d24924b"
    ),
}


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


@pytest.mark.parametrize("path", [CONTRACT, RUNNER, LAUNCHER, CHECKER])
def test_v2_entrypoints_import_source_only_under_isolation(path: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(path)!r})
spec = importlib.util.spec_from_file_location("_event_delta_v2_isolated", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert "PIL" not in sys.modules
if hasattr(module, "_assert_final_runner_bindings"):
    module._assert_final_runner_bindings()
if hasattr(module, "_assert_final_launcher_bindings"):
    module._assert_final_launcher_bindings()
print("PASS")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PASS\n"
    assert completed.stderr == ""


def test_v2_contract_is_exact_frozen_v1_science_and_five_file_delta() -> None:
    contract = _load("_event_delta_v2_contract_identity", CONTRACT)
    v1 = contract._v1
    assert contract.FROZEN_V1_SCIENCE_CONTRACT_SHA256 == (
        FROZEN_V1_SCIENCE_CONTRACT_SHA256
    )
    assert contract.FROZEN_V1_SCIENCE_COMPONENT_SHA256 == (
        FROZEN_V1_SCIENCE_COMPONENT_SHA256
    )
    assert contract.canonical_json_sha256(v1.science_contract()) == (
        FROZEN_V1_SCIENCE_CONTRACT_SHA256
    )
    observed = {
        "model": contract.canonical_json_sha256(v1.model_config()),
        "objective": contract.canonical_json_sha256(v1.objective_contract()),
        "optimizer": contract.canonical_json_sha256(v1.optimizer_contract()),
        "schedule": contract.canonical_json_sha256(
            v1.build_schedule_identity()
        ),
        "gate_thresholds": contract.canonical_json_sha256(v1.GATE_THRESHOLDS),
        "work_accounting": contract.canonical_json_sha256(
            v1.WORK_ACCOUNTING_CONTRACT
        ),
        "warning_policy": contract.canonical_json_sha256(v1.WARNING_POLICY),
        "runtime_input_template": contract.canonical_json_sha256(
            v1.runtime_authorization_template()
        ),
    }
    assert observed == FROZEN_V1_SCIENCE_COMPONENT_SHA256
    assert contract.model_config() == v1.model_config()
    assert contract.objective_contract() == v1.objective_contract()
    assert contract.optimizer_contract() == v1.optimizer_contract()
    assert contract.build_schedule_identity() == v1.build_schedule_identity()
    assert contract.runtime_authorization_template() == (
        v1.runtime_authorization_template()
    )

    expected_additive = {
        contract.CONTRACT_RELATIVE_PATH,
        contract.RUNNER_RELATIVE_PATH,
        contract.LAUNCHER_RELATIVE_PATH,
        contract.SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
    }
    assert tuple(contract.REUSED_SOURCE_PATHS) == tuple(v1.SOURCE_PATHS)
    assert len(contract.REUSED_SOURCE_PATHS) == 98
    assert set(contract.ADDITIVE_SOURCE_PATHS) == expected_additive
    assert len(contract.SOURCE_PATHS) == 103
    assert set(contract.SOURCE_PATHS) == {
        *v1.SOURCE_PATHS,
        *expected_additive,
    }
    assert not any("lewm/models/" in path for path in expected_additive)
    assert contract.MODEL_TEST_RELATIVE_PATH == (
        "lewm/tests/test_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
    )
    assert contract.MODEL_TEST_RELATIVE_PATH in contract.SOURCE_PATHS
    assert contract.MODEL_TEST_RELATIVE_PATH in (
        contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES
    )
    assert contract.DELEGATION_PREFLIGHT_REQUIREMENTS[
        "reviewed_source_witness"
    ] == {
        "source_map_installed_before_inherited_execute_body": True,
        "path": contract.MODEL_TEST_RELATIVE_PATH,
        "file_sha256": (
            "09170a2cceb297df65bfd6c3bf6f4f3aedda077777c8f837095cbde3a53198d6"
        ),
        "runtime_value_non_null_and_exact": True,
        "fallback_hard_coded_sha_used": False,
    }
    assert contract.FROZEN_V1_RUNNER_RELATIVE_PATH == (
        "scripts/run_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
    )
    assert contract.FROZEN_V1_LAUNCHER_RELATIVE_PATH == (
        "scripts/launch_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
    )
    assert contract.FROZEN_V1_MODEL_RELATIVE_PATH == v1.MODEL_RELATIVE_PATH
    assert contract.FROZEN_V1_SOURCE_COMMIT == (
        "c414231d6d0e0d0cbf9282aec16944d4d4b7cfca"
    )
    assert contract.FROZEN_V1_SOURCE_MANIFEST_FILE_SHA256 == (
        "f87aa717fd118f3fb6e0a0e169dd0f4aec812f5a305cf95eb5b809e0c6c13e50"
    )
    assert contract.FROZEN_V1_SOURCE_MANIFEST_CONTENT_SHA256 == (
        "db5c7fdab152f75a3bafd7c94ba555bac5c5441e44fbb1ddb7ddb439ae74aa70"
    )
    assert contract.FROZEN_V1_SOURCE_BINDINGS_SHA256 == (
        "d7f6d4302c6e5ab6ff1ce24089ba8c7b20df80dda92dcaea3897ccb200315f8b"
    )
    assert contract.FROZEN_V1_SOURCE_COUNT == 98
    assert contract.FROZEN_V1_SOURCE_MANIFEST_BYTE_COUNT == 33_275
    assert contract.OUTPUT_ROOT_RELATIVE_PATH == (
        ".generated/go2_rgb_geometry_anchored_two_mode_event_delta_joint_jepa_"
        "v2_runtime_delegation_integrity_replacement/attempt_v1"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != v1.OUTPUT_ROOT_RELATIVE_PATH
    assert not (ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH).exists()


def test_final_event_hooks_and_terminal_lifecycles_are_frozen_v1() -> None:
    contract = _load("_event_delta_v2_terminal_contract", CONTRACT)
    runner = _load("_event_delta_v2_terminal_runner", RUNNER)
    runner._assert_final_runner_bindings()
    assert runner.run_isolated_import_preflight() == (
        contract.DELEGATION_PREFLIGHT_REQUIREMENTS
    )
    for name in runner._EVENT_BINDING_NAMES:
        expected = runner._FROZEN_V1_EVENT_BINDINGS[name]
        assert getattr(runner._V1, name) is expected
        assert getattr(runner._BASE, name) is expected
    terminal_globals = runner._V1._terminal_failure.__globals__
    assert terminal_globals["_publish_partial_scientific_failure"] is (
        runner._V1._publish_partial_scientific_failure
    )
    assert terminal_globals["_publish_compact_operational_failure"] is (
        runner._V1._publish_compact_operational_failure
    )
    assert contract.NORMAL_RECEIPT_PATHS == runner._V1.contract.NORMAL_RECEIPT_PATHS
    assert contract.NORMAL_RECEIPT_PATHS == (
        "reservation.json",
        "metrics.json",
        "artifact.json",
        "access.json",
        "result.json",
        "completed.json",
    )
    assert contract.OPERATIONAL_FAILURE_RECEIPT_PATHS == (
        "failure.json",
        "completed.json",
    )


def test_runner_and_launcher_dispatch_directly_to_deepest_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fatal_predecessor(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a predecessor wrapper was called")

    runner = _load("_event_delta_v2_direct_runner", RUNNER)
    runner_calls: list[tuple[str, Any]] = []
    for predecessor in (
        runner._V1,
        runner._V1._RIGID,
        runner._V1._V3,
        runner._V1._V2,
    ):
        monkeypatch.setattr(predecessor, "parse_args", fatal_predecessor)
        monkeypatch.setattr(predecessor, "main", fatal_predecessor)
    monkeypatch.setattr(
        runner._BASE,
        "parse_args",
        lambda argv=None: runner_calls.append(("parse", argv)) or "parsed",
    )
    monkeypatch.setattr(
        runner._BASE,
        "main",
        lambda argv=None: runner_calls.append(("main", argv)) or 17,
    )
    assert runner.parse_args(["runner-parse"]) == "parsed"
    assert runner.main(["runner-main"]) == 17
    assert runner_calls == [
        ("parse", ["runner-parse"]),
        ("main", ["runner-main"]),
    ]

    launcher = _load("_event_delta_v2_direct_launcher", LAUNCHER)
    launcher_calls: list[tuple[str, Any]] = []
    for predecessor in (
        launcher._V1,
        launcher._V1._RIGID,
        launcher._V1._RIGID._V3,
        launcher._V1._RIGID._V3._V2,
    ):
        monkeypatch.setattr(predecessor, "parse_args", fatal_predecessor)
        monkeypatch.setattr(predecessor, "main", fatal_predecessor)
    monkeypatch.setattr(
        launcher._BASE,
        "parse_args",
        lambda argv=None: launcher_calls.append(("parse", argv)) or "parsed",
    )
    monkeypatch.setattr(
        launcher._BASE,
        "main",
        lambda argv=None: launcher_calls.append(("main", argv)) or 19,
    )
    assert launcher.parse_args(["launcher-parse"]) == "parsed"
    assert launcher.main(["launcher-main"]) == 19
    assert launcher_calls == [
        ("parse", ["launcher-parse"]),
        ("main", ["launcher-main"]),
    ]


def test_event_execute_populates_exact_reviewed_cpu_witness_before_body(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_event_delta_v2_witness_runner", RUNNER)
    model_test = runner.contract.MODEL_TEST_RELATIVE_PATH
    sources = {model_test: "a" * 64}
    observed: dict[str, Any] = {}

    class BodyReached(RuntimeError):
        pass

    def inherited_body(**kwargs: Any) -> int:
        assert kwargs["sources"] == sources
        observed["witness"] = runner._V1._reviewed_cpu_source_witness(
            source_authority_exact=True
        )
        raise BodyReached("synthetic inherited body reached")

    base_globals_before = {
        "_publish_json": runner._BASE._publish_json,
        "_seal": runner._BASE._seal,
        "_load_development_inputs": runner._BASE._load_development_inputs,
    }
    bindings_before = dict(runner._V1._ACTIVE_SOURCE_BINDINGS)
    monkeypatch.setattr(runner._V1, "_BASE_EXECUTE", inherited_body)
    try:
        with pytest.raises(BodyReached, match="inherited body reached"):
            runner._BASE._execute(
                sources=sources,
                authorization={},
                reservation={},
                reservation_raw=b"{}\n",
                output_root=tmp_path,
                progress={},
            )
    finally:
        runner._V1._ACTIVE_SOURCE_BINDINGS.clear()
        runner._V1._ACTIVE_SOURCE_BINDINGS.update(bindings_before)

    witness = observed["witness"]
    assert witness["reviewed_model_source_synthetic_witness_path"] == model_test
    assert witness["reviewed_model_source_synthetic_witness_sha256"] == "a" * 64
    assert witness["reviewed_model_source_synthetic_witness_sha256"] is not None
    boolean_witnesses = {
        name: value for name, value in witness.items()
        if isinstance(value, bool)
    }
    assert boolean_witnesses
    assert all(boolean_witnesses.values())
    assert witness["runtime_update_zero_synthetic_accelerator_call_count"] == 0
    assert runner._BASE._publish_json is base_globals_before["_publish_json"]
    assert runner._BASE._seal is base_globals_before["_seal"]
    assert runner._BASE._load_development_inputs is (
        base_globals_before["_load_development_inputs"]
    )
    assert not any(tmp_path.iterdir())
