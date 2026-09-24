from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import (
    execute_go2_rgb_unified_ray_survival_joint_jepa_v14 as canonical_v14,
)
from scripts import (
    execute_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon as v15,
)
from scripts import (
    go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_lifecycle
    as lifecycle,
)
from scripts import (
    run_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
    as canonical_training,
)
from scripts import (
    run_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon as training,
)


ROOT = Path(__file__).resolve().parents[2]


def _summary(
    *,
    passed: int = 120,
    shortfall: float = 30.0,
    pixel: float = 0.84,
    ground: float = 0.68,
    depth: float = 0.90,
    complete: int = 1,
) -> dict[str, object]:
    return {
        "complete_physical_scope_count": complete,
        "margin_count": 189,
        "passed_margin_count": passed,
        "total_shortfall": shortfall,
        "worst_margin": -0.1,
        "rough_motion": {
            "pixel_balanced_accuracy": pixel,
            "ground_balanced_accuracy": ground,
            "depth_p95_m": depth,
        },
    }


def _positive_controls() -> dict[str, dict[str, bool]]:
    return {
        name: {check: True for check in v15.CONTROL_CHECK_NAMES}
        for name in v15.CONTROL_NAMES
    }


def _v12_gate(*, passed: bool = True) -> dict[str, object]:
    checks = {name: True for name in v15.V12_GATE_CHECK_NAMES}
    if not passed:
        checks[v15.V12_GATE_CHECK_NAMES[0]] = False
    return {"passed": passed, "checks": checks}


def test_v15_replacement_preserves_science_and_changes_only_attempt_identity() -> None:
    assert v15._base is not canonical_v14
    assert v15.PRIVATE_V14_MODULE_NAME not in sys.modules
    assert canonical_v14.MAXIMUM_UPDATES == 1_000
    assert canonical_v14.MAXIMUM_PRESENTATIONS == 16_000
    assert canonical_v14.OBSERVATION_UPDATES == (0, 100, 400, 1_000)
    assert canonical_v14.TERMINAL_UPDATES == (400, 1_000)
    assert canonical_v14.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH.endswith(
        "update_1000.pt"
    )

    assert v15.SCHEMA_PREFIX == (
        "lewm_go2_rgb_unified_ray_survival_joint_jepa_v15_extended_horizon_"
        "integrity_replacement_v1"
    )
    assert v15.PREREGISTRATION_COMMIT == (
        "86d19b29171ee8d08dda6b04361466f420aec42d"
    )
    assert v15.PREREGISTRATION_PATH.endswith(
        "v15_extended_horizon_integrity_replacement_v1_"
        "preregistration_2026-07-29.md"
    )
    assert v15.OUTPUT_ROOT_RELATIVE_PATH == (
        ".generated/go2_rgb_unified_ray_survival_joint_jepa_v15_"
        "extended_horizon_integrity_replacement_v1/attempt_v1"
    )
    assert v15.MAXIMUM_UPDATES == 2_000
    assert v15.MAXIMUM_PRESENTATIONS == 32_000
    assert v15.OBSERVATION_UPDATES == (0, 100, 400, 1_000, 1_400, 2_000)
    assert v15.TERMINAL_UPDATES == (400, 1_400, 2_000)
    assert v15.MODEL_CLASS_NAME == canonical_v14.MODEL_CLASS_NAME
    assert v15.MODEL_REQUIRED_CONSTANTS == canonical_v14.MODEL_REQUIRED_CONSTANTS
    assert v15.FINAL_PHYSICAL_THRESHOLDS == canonical_v14.FINAL_PHYSICAL_THRESHOLDS
    assert v15.MATCHED_UPDATE400_THRESHOLDS == (
        canonical_v14.MATCHED_UPDATE400_THRESHOLDS
    )
    assert v15.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH == (
        "checkpoints/update_2000.pt"
    )
    assert v15.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH == (
        "checkpoints/update_2000.binding.json"
    )
    receipt = v15.private_adapter_receipt_v15()
    assert receipt["public_v14_loaded_by_adapter"] is False
    assert receipt["private_module_registered"] is False
    assert receipt["execution_authorized"] is False
    assert receipt["original_v15_preregistration_commit"] == (
        "af0f786841b1404d1f42542b507ad198ee574250"
    )
    assert receipt["v15_terminal_failure_result_commit"] == (
        "51cfeb7fd5dbc1743bf043d21f350937755c0647"
    )


def test_v15_replacement_binds_preregistrations_and_failure_receipt() -> None:
    expected = {
        v15.PREREGISTRATION_PATH: (
            v15.PREREGISTRATION_FILE_SHA256,
            v15.PREREGISTRATION_BYTE_COUNT,
        ),
        v15.ORIGINAL_V15_PREREGISTRATION_PATH: (
            v15.ORIGINAL_V15_PREREGISTRATION_FILE_SHA256,
            v15.ORIGINAL_V15_PREREGISTRATION_BYTE_COUNT,
        ),
        v15.V15_TERMINAL_FAILURE_RESULT_PATH: (
            v15.V15_TERMINAL_FAILURE_RESULT_FILE_SHA256,
            v15.V15_TERMINAL_FAILURE_RESULT_BYTE_COUNT,
        ),
        v15.V14_RESULT_PATH: (
            v15.V14_RESULT_FILE_SHA256,
            v15.V14_RESULT_BYTE_COUNT,
        ),
    }
    for path, binding in expected.items():
        assert v15.BOUND_PARENT_SOURCES[path] == binding
    assert v15.validate_bound_sources_v15(ROOT, expected)[
        "validated_path_count"
    ] == len(expected)


def test_private_training_adapter_extends_caps_without_mutating_public_base() -> None:
    assert training._training is not canonical_training
    assert training.PRIVATE_BASE_MODULE_NAME not in sys.modules
    assert canonical_training.MAXIMUM_UPDATES == 1_000
    assert canonical_training.MAXIMUM_PRESENTATIONS == 16_000
    assert training.MAXIMUM_UPDATES == 2_000
    assert training.MAXIMUM_PRESENTATIONS == 32_000
    assert training.PRESENTATIONS_PER_UPDATE == 16
    assert training.joint_training_update_v13.__globals__ is training._training.__dict__
    assert training.joint_training_update_v13.__globals__["MAXIMUM_UPDATES"] == 2_000
    receipt = training.private_training_adapter_receipt_v15()
    assert receipt["public_base_loaded_by_adapter"] is False
    assert receipt["private_module_registered"] is False
    assert receipt["scientific_change"] == "terminal_accounting_caps_only"


def test_update400_is_unchanged_except_successor_update() -> None:
    before = _summary(
        passed=60,
        shortfall=80.0,
        pixel=0.70,
        ground=0.60,
        depth=2.1,
        complete=0,
    )
    after = _summary(
        passed=72,
        shortfall=71.0,
        pixel=0.71,
        ground=0.61,
        depth=1.9,
        complete=0,
    )
    decision = v15.evaluate_update400_gate_v15(
        before,
        after,
        _positive_controls(),
        integrity_pass=True,
    )
    assert decision["passed"] is True
    assert decision["action"] == "CONTINUE_TO_UPDATE_1400"
    assert decision["next_update"] == 1_400
    assert len(decision["checks"]) == 8
    assert decision["matched_update400_thresholds"] == (
        canonical_v14.MATCHED_UPDATE400_THRESHOLDS
    )


def test_update1400_gate_exact_pass_and_non_strict_integer_minimum() -> None:
    passing = _summary(
        passed=99,
        shortfall=38.0,
        pixel=0.83,
        ground=0.66,
        depth=1.30,
        complete=0,
    )
    decision = v15.evaluate_update1400_gate_v15(
        _v12_gate(),
        passing,
        _positive_controls(),
        integrity_pass=True,
    )
    assert decision["passed"] is True
    assert decision["update"] == 1_400
    assert decision["action"] == "CONTINUE_TO_UPDATE_2000"
    assert decision["next_update"] == 2_000
    assert len(decision["causal_control_checks"]) == 12
    assert decision["checkpoint_authorized"] is False
    assert decision["g2_authorized"] is False
    assert decision["retry_authorized"] is False
    assert decision["resume_authorized"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("total_shortfall", 38.1),
        ("depth_p95_m", 1.304),
        ("pixel_balanced_accuracy", 0.8198594673963917),
        ("ground_balanced_accuracy", 0.647134926562893),
    ),
)
def test_update1400_strict_continuous_threshold_equalities_fail(
    field: str,
    value: float,
) -> None:
    physical = _summary(
        passed=99,
        shortfall=38.0,
        pixel=0.83,
        ground=0.66,
        depth=1.30,
        complete=0,
    )
    if field in {
        "depth_p95_m",
        "pixel_balanced_accuracy",
        "ground_balanced_accuracy",
    }:
        physical["rough_motion"][field] = value  # type: ignore[index]
    else:
        physical[field] = value
    decision = v15.evaluate_update1400_gate_v15(
        _v12_gate(),
        physical,
        _positive_controls(),
        integrity_pass=True,
    )
    assert decision["passed"] is False
    assert decision["action"] == "FAIL_TERMINAL_NO_RETRY_NO_RESUME"
    assert decision["next_update"] is None


def test_update1400_requires_fresh_complete_v12_and_control_evidence() -> None:
    physical = _summary(
        passed=99,
        shortfall=38.0,
        pixel=0.83,
        ground=0.66,
        depth=1.30,
        complete=0,
    )
    controls = _positive_controls()
    controls[v15.CONTROL_NAMES[0]][v15.CONTROL_CHECK_NAMES[0]] = False
    failed_control = v15.evaluate_update1400_gate_v15(
        _v12_gate(), physical, controls, integrity_pass=True
    )
    assert failed_control["passed"] is False
    assert len(failed_control["causal_control_checks"]) == 12
    assert v15.evaluate_update1400_gate_v15(
        _v12_gate(passed=False),
        physical,
        _positive_controls(),
        integrity_pass=True,
    )["passed"] is False
    assert v15.evaluate_update1400_gate_v15(
        _v12_gate(),
        physical,
        _positive_controls(),
        integrity_pass=False,
    )["passed"] is False


def test_update2000_final_gate_keeps_exact_v14_threshold_semantics() -> None:
    passing = _summary(
        passed=112,
        shortfall=33.0,
        pixel=0.82,
        ground=0.65,
        depth=0.97,
        complete=1,
    )
    decision = v15.evaluate_final_gate_v15(
        _v12_gate(), passing, integrity_pass=True
    )
    assert decision["passed"] is True
    assert decision["update"] == 2_000
    assert decision["physical_adapter_preregistration_eligible"] is True
    assert decision["probability_calibration_authorized"] is False
    assert decision["g2_authorized"] is False
    assert decision["navigation_authorized"] is False
    assert decision["held_out_authorized"] is False

    equality = _summary(
        passed=112,
        shortfall=33.05143763708337,
        pixel=0.82,
        ground=0.65,
        depth=0.97,
        complete=1,
    )
    assert v15.evaluate_final_gate_v15(
        _v12_gate(), equality, integrity_pass=True
    )["passed"] is False


def test_update1000_is_observation_only_and_never_checkpoint_eligible() -> None:
    assert 1_000 in v15.OBSERVATION_UPDATES
    assert 1_000 not in v15.TERMINAL_UPDATES
    assert v15.METRIC_RELATIVE_PATHS[1_000] == "metrics/update_1000.json"
    assert "1000" not in v15.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH
    assert "1000" not in v15.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH


@pytest.mark.parametrize("terminal_update", (400, 1_400, 2_000))
def test_terminal_accounting_is_exact_and_capped(terminal_update: int) -> None:
    accounting = {
        name: terminal_update * multiplier
        for name, multiplier in v15._engine.ACCOUNTING_MULTIPLIERS.items()
    }
    assert v15.validate_terminal_accounting_v15(
        accounting, terminal_update=terminal_update
    ) == accounting
    accounting["ema_steps"] -= 1
    with pytest.raises(RuntimeError, match="inconsistent"):
        v15.validate_terminal_accounting_v15(
            accounting, terminal_update=terminal_update
        )


def test_update1000_cannot_be_terminal_accounting_or_publish_a_checkpoint() -> None:
    accounting = {
        name: 1_000 * multiplier
        for name, multiplier in v15._engine.ACCOUNTING_MULTIPLIERS.items()
    }
    with pytest.raises(ValueError):
        v15.validate_terminal_accounting_v15(
            accounting, terminal_update=1_000
        )


def test_denial_shell_remains_source_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert v15.CURRENT_EXECUTION_AUTHORIZED is False
    assert v15.main([]) == 4
    receipt = v15.validate_content_bound_v15(
        __import__("json").loads(capsys.readouterr().out)
    )
    assert receipt["preregistration_commit"] == v15.PREREGISTRATION_COMMIT
    assert receipt["output_root"] == v15.OUTPUT_ROOT_RELATIVE_PATH
    assert receipt["scientific_payload_opened"] is False
    assert receipt["reservation_created"] is False


class _MemoryPublisher:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    @staticmethod
    def _binding(path: str, raw: bytes) -> dict[str, object]:
        return {
            "path": path,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }

    def publish_json(
        self,
        relative_path: str,
        core: dict[str, object],
    ) -> dict[str, object]:
        if relative_path in self.files:
            raise FileExistsError(relative_path)
        value = v15._engine._content_bound(core)
        raw = v15._engine._canonical_json_bytes(value) + b"\n"
        self.files[relative_path] = raw
        return {
            "value": value,
            "binding": self._binding(relative_path, raw),
        }

    def publish_bytes(self, relative_path: str, raw: bytes) -> dict[str, object]:
        if relative_path in self.files:
            raise FileExistsError(relative_path)
        self.files[relative_path] = raw
        return self._binding(relative_path, raw)


class _LifecycleStubEngine:
    SCHEMA_PREFIX = v15.SCHEMA_PREFIX
    MAXIMUM_UPDATES = v15.MAXIMUM_UPDATES
    MAXIMUM_PRESENTATIONS = v15.MAXIMUM_PRESENTATIONS
    OBSERVATION_UPDATES = v15.OBSERVATION_UPDATES
    TERMINAL_UPDATES = v15.TERMINAL_UPDATES
    PRESENTATIONS_PER_UPDATE = v15.PRESENTATIONS_PER_UPDATE
    METRIC_RELATIVE_PATHS = v15.METRIC_RELATIVE_PATHS
    TRACE_RELATIVE_PATH = v15.TRACE_RELATIVE_PATH
    TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH = (
        v15.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH
    )
    SCIENTIFIC_FAILURE_RELATIVE_PATH = v15.SCIENTIFIC_FAILURE_RELATIVE_PATH
    SUCCESS_RELATIVE_PATH = v15.SUCCESS_RELATIVE_PATH
    DEVELOPMENT_CHECKPOINT_RELATIVE_PATH = (
        v15.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH
    )
    DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH = (
        v15.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH
    )
    MATCHED_UPDATE400_THRESHOLDS = v15.MATCHED_UPDATE400_THRESHOLDS
    CONTROL_NAMES = v15.CONTROL_NAMES
    CONTROL_CHECK_NAMES = v15.CONTROL_CHECK_NAMES
    V12_GATE_CHECK_NAMES = v15.V12_GATE_CHECK_NAMES

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.observed_updates: list[int] = []

    @staticmethod
    def _canonical_json_bytes(value: dict[str, object]) -> bytes:
        return v15._engine._canonical_json_bytes(value)

    @staticmethod
    def _canonical_value_sha256(value: object) -> str:
        return v15._engine._canonical_value_sha256(value)

    @staticmethod
    def _content_bound(value: dict[str, object]) -> dict[str, object]:
        return v15._engine._content_bound(value)

    @staticmethod
    def _publisher_json_v13(
        publisher: object,
        relative_path: str,
        core: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object]]:
        return v15._engine._publisher_json_v13(publisher, relative_path, core)

    @staticmethod
    def _publisher_bytes_v13(
        publisher: object,
        relative_path: str,
        raw: bytes,
    ) -> dict[str, object]:
        return v15._engine._publisher_bytes_v13(publisher, relative_path, raw)

    @staticmethod
    def _validate_physical_summary(value: dict[str, object]) -> dict[str, object]:
        return v15._engine._validate_physical_summary(value)

    @staticmethod
    def validate_future_execution_prerequisites_v13(
        authority: dict[str, object],
    ) -> dict[str, object]:
        return dict(authority)

    @staticmethod
    def validate_attempt_reservation_v13(
        reservation: dict[str, object],
    ) -> dict[str, object]:
        return dict(reservation)

    @staticmethod
    def validate_schedule_v13(
        schedule: list[int],
        *,
        train_pair_count: int,
    ) -> dict[str, object]:
        assert train_pair_count == 4_262
        assert len(schedule) == 32_000
        assert schedule[:16_000] == schedule[16_000:]
        return {
            "presentation_count": 32_000,
            "repeated_halves_elementwise_identical": True,
        }

    @staticmethod
    def _validate_initialization_v13(
        runtime: object,
        model: object,
        initialization: dict[str, object],
    ) -> dict[str, object]:
        return dict(initialization)

    @staticmethod
    def _derive_initial_structural_integrity_v13(
        runtime: object,
        model: object,
    ) -> dict[str, bool]:
        return {"passed": True}

    @staticmethod
    def _validate_access_receipt_v13(
        value: dict[str, object],
        *,
        terminal: bool = False,
    ) -> dict[str, object]:
        return dict(value)

    @staticmethod
    def _validate_microbatches_for_engine_v13(
        runtime: object,
        model: object,
        microbatches: object,
    ) -> None:
        return None

    @staticmethod
    def _validate_update_integrity_v13(
        runtime: object,
        model: object,
        result: object,
        *,
        update: int,
        access_receipt: dict[str, object],
    ) -> dict[str, object]:
        return {"update": update, "passed": True}

    def _observation_v13(
        self,
        runtime: object,
        model: object,
        *,
        update: int,
        integrity_pass: bool,
    ) -> dict[str, object]:
        self.observed_updates.append(update)
        physical_by_update = {
            0: _summary(
                passed=40,
                shortfall=100.0,
                pixel=0.60,
                ground=0.50,
                depth=2.5,
                complete=0,
            ),
            100: _summary(
                passed=60,
                shortfall=80.0,
                pixel=0.70,
                ground=0.60,
                depth=2.1,
                complete=0,
            ),
            400: _summary(
                passed=72,
                shortfall=71.0,
                pixel=0.71,
                ground=0.61,
                depth=1.9,
                complete=0,
            ),
            1_000: _summary(
                passed=89,
                shortfall=41.0,
                pixel=0.80,
                ground=0.64,
                depth=1.5,
                complete=0,
            ),
            1_400: _summary(
                passed=99,
                shortfall=(38.1 if self.scenario == "fail1400" else 38.0),
                pixel=0.83,
                ground=0.66,
                depth=1.30,
                complete=0,
            ),
            2_000: _summary(
                passed=112,
                shortfall=30.0,
                pixel=0.84,
                ground=0.68,
                depth=0.9777327477931971,
                complete=1,
            ),
        }
        return {
            "integrity_pass": integrity_pass,
            "physical": physical_by_update[update],
            "v12_gate": _v12_gate(),
            "controls": (
                _positive_controls() if update in (400, 1_400) else None
            ),
        }

    @staticmethod
    def evaluate_update400_gate_v13(
        update100: dict[str, object],
        update400: dict[str, object],
        controls: dict[str, dict[str, bool]],
        *,
        integrity_pass: bool,
        matched_update400_thresholds: dict[str, int | float],
    ) -> dict[str, object]:
        return v15.evaluate_update400_gate_v15(
            update100,
            update400,
            controls,
            integrity_pass=integrity_pass,
            matched_update400_thresholds=matched_update400_thresholds,
        )

    @staticmethod
    def evaluate_final_gate_v13(
        v12_gate: dict[str, object],
        physical: dict[str, object],
        *,
        integrity_pass: bool,
    ) -> dict[str, object]:
        return v15.evaluate_final_gate_v15(
            v12_gate,
            physical,
            integrity_pass=integrity_pass,
        )

    @staticmethod
    def validate_terminal_accounting_v13(
        accounting: dict[str, int],
        *,
        terminal_update: int,
    ) -> dict[str, int]:
        return v15.validate_terminal_accounting_v15(
            accounting,
            terminal_update=terminal_update,
        )


class _LifecycleStubRuntime:
    train_pair_count = 4_262

    def __init__(self) -> None:
        base = [index % self.train_pair_count for index in range(16_000)]
        self.schedule = base + base
        self.training_module = SimpleNamespace(
            joint_training_update_v13=self._training_update
        )
        self.close_calls = 0
        self.terminal_access_calls = 0

    @staticmethod
    def initialize_model_v13() -> tuple[object, object, dict[str, object]]:
        return object(), object(), {"fresh_initialization": True}

    @staticmethod
    def build_microbatches_v13(
        indices: list[int],
        *,
        update: int,
    ) -> tuple[()]:
        assert len(indices) == 16
        return ()

    @staticmethod
    def _training_update(
        model: object,
        optimizer: object,
        microbatches: object,
        *,
        accounting: dict[str, int] | None,
    ) -> SimpleNamespace:
        update = 1 if accounting is None else accounting["updates"] + 1
        return SimpleNamespace(
            accounting={
                name: update * multiplier
                for name, multiplier in v15._engine.ACCOUNTING_MULTIPLIERS.items()
            }
        )

    @staticmethod
    def access_receipt_v13() -> dict[str, object]:
        return {"receipt_kind": "lightweight_in_memory"}

    def terminal_access_receipt_v13(self) -> dict[str, object]:
        self.terminal_access_calls += 1
        return {
            "runtime_data_root": "/stub/runtime-data",
            "source_root": "/stub/certified-source",
            "runtime_fingerprint": {"runtime": "stub"},
        }

    def close_v13(self) -> None:
        self.close_calls += 1


@pytest.mark.parametrize(
    ("scenario", "terminal_update", "trace_rows", "metric_count"),
    (
        ("fail1400", 1_400, 1_403, 5),
        ("fail2000", 2_000, 2_004, 6),
    ),
)
def test_controller_inventory_and_failed_gates_never_publish_checkpoint(
    scenario: str,
    terminal_update: int,
    trace_rows: int,
    metric_count: int,
) -> None:
    engine = _LifecycleStubEngine(scenario)
    runtime = _LifecycleStubRuntime()
    publisher = _MemoryPublisher()
    authority = {
        "runtime_data_root": "/stub/runtime-data",
        "certified_source_root": "/stub/certified-source",
        "runtime": {"runtime": "stub"},
    }
    reservation = {
        "authority_sha256": hashlib.sha256(
            engine._canonical_json_bytes(authority)
        ).hexdigest()
    }
    result = lifecycle.run_future_authorized_engine_v15(
        authority=authority,
        reservation=reservation,
        runtime=runtime,
        publisher=publisher,
        engine=engine,
    )
    assert result["status"] == (
        f"FAIL_SCIENTIFIC_UPDATE{terminal_update}_GATE_TERMINAL"
    )
    assert result["terminal_update"] == terminal_update
    assert result["checkpoint_published"] is False
    assert engine.observed_updates == [
        update for update in v15.OBSERVATION_UPDATES if update <= terminal_update
    ]
    assert v15.METRIC_RELATIVE_PATHS[1_000] in publisher.files
    assert v15.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH not in publisher.files
    assert v15.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH not in publisher.files
    assert v15.SUCCESS_RELATIVE_PATH not in publisher.files
    assert v15.SCIENTIFIC_FAILURE_RELATIVE_PATH in publisher.files
    assert len(result["metrics"]) == metric_count
    trace = publisher.files[v15.TRACE_RELATIVE_PATH].decode("utf-8").splitlines()
    assert len(trace) == trace_rows
    events = [json.loads(row)["event"] for row in trace]
    assert "update1000_final_gate" not in events
    assert events[-1] == (
        "update1400_feasibility_gate"
        if terminal_update == 1_400
        else "update2000_final_gate"
    )
    # The controller publishes metrics plus trace, terminal access, and failure;
    # the one reservation file is created by the launcher before this call.
    assert len(publisher.files) == metric_count + 3
    assert len(publisher.files) + 1 == (9 if terminal_update == 1_400 else 10)
    assert runtime.terminal_access_calls == 1
    assert runtime.close_calls == 1
