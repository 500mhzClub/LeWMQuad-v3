from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from scripts import (
    run_go2_rgb_fixed_teacher_factual_shared_transition_trajectory_h4_jepa_v2_schedule_integrity
    as runner,
)


_CORE_MUTABLE_NAMES = (
    "MODEL_MODULE",
    "MODEL_SOURCE",
    "MODEL_SOURCE_SHA256",
    "MODEL_SOURCE_BYTES",
    "TRAIN_INDEX",
    "TRAIN_INDEX_SHA256",
    "TRAIN_INDEX_BYTES",
    "VAL_INDEX",
    "VAL_INDEX_SHA256",
    "VAL_INDEX_BYTES",
    "INDEX_ROW_SCHEMA",
    "OUTPUT_ROOT",
    "SCHEMA",
    "PASS_DECISION",
    "STOP_DECISION",
    "PREDICTION_WEIGHT",
    "VARIANCE_WEIGHT",
    "ACTION_RANKING_WEIGHT",
    "TRAIN_WRONG_ACTION_CONTRAST",
    "UPDATE_TARGET_EMA",
    "TARGET_DESCRIPTION",
    "OBJECTIVE_DESCRIPTION",
    "ADDITIONAL_SCIENCE",
    "AUXILIARY_TRAINING_CONTROL_MULTIPLIER",
    "EXECUTION_SOURCE_BINDINGS",
    "_evaluate",
    "_decision",
    "_run",
)
_BASE_MUTABLE_NAMES = (
    "MODEL_MODULE",
    "MODEL_SOURCE",
    "MODEL_SOURCE_SHA256",
    "MODEL_SOURCE_BYTES",
    "OUTPUT_ROOT",
    "SCHEMA",
    "PASS_DECISION",
    "STOP_DECISION",
)
_SCIENCE_NAMES = (
    "MODEL_MODULE",
    "MODEL_SOURCE",
    "MODEL_SOURCE_SHA256",
    "MODEL_SOURCE_BYTES",
    "PREDICTION_WEIGHT",
    "VARIANCE_WEIGHT",
    "ACTION_RANKING_WEIGHT",
    "TRAIN_WRONG_ACTION_CONTRAST",
    "UPDATE_TARGET_EMA",
    "TARGET_DESCRIPTION",
    "OBJECTIVE_DESCRIPTION",
    "AUXILIARY_TRAINING_CONTROL_MULTIPLIER",
    "UPDATES",
    "BATCH_SIZE",
    "PRESENTATIONS",
    "VAL_PRESENTATIONS",
    "OBSERVATION_UPDATES",
    "MAX_GPU_SECONDS",
    "SEED",
    "BOOTSTRAP_REPLICATES",
)


def _snapshots() -> tuple[dict[str, object], dict[str, object]]:
    return (
        {name: getattr(runner.core, name) for name in _CORE_MUTABLE_NAMES},
        {name: getattr(runner.v1.base, name) for name in _BASE_MUTABLE_NAMES},
    )


def _restore(
    core_values: dict[str, object],
    base_values: dict[str, object],
) -> None:
    for name, value in base_values.items():
        setattr(runner.v1.base, name, value)
    for name, value in core_values.items():
        setattr(runner.core, name, value)


def _observation(
    update: int,
    *,
    real: float,
    joint: float,
    action: float,
    hold: float,
    persistence: float,
    history: float,
    distribution: float,
    spread: float,
    p0_p1_score: float,
    p0_p1_gap: float,
) -> dict:
    aggregate = {
        "real_normalized_energy_score": [real] * 4,
        "action_gap": [action] * 4,
        "hold_gap": [hold] * 4,
        "persistence_gap": [persistence] * 4,
        "history_gap": [history] * 4,
        "distribution_value_gap": [distribution] * 4,
        "normalized_pairwise_spread": [spread] * 4,
        "best_atom_normalized_squared_error": [real] * 4,
        "centroid_normalized_squared_error": [real] * 4,
        "joint_trajectory_normalized_energy_score": joint,
        "combined_normalized_energy_score": 0.5 * (joint + real),
        "combined_distribution_value_gap": distribution,
        "p0_p1_local_prior_combined_normalized_energy_score": p0_p1_score,
        "p0_p1_local_prior_persistence_gap": p0_p1_gap,
        "future_p2_p5_local_combined_normalized_energy_score": real,
    }
    return {
        "update": update,
        "presentations": update * 16,
        "aggregate": deepcopy(aggregate),
        "family": {
            family: deepcopy(aggregate) for family in runner.core.FAMILIES
        },
        "bootstrap_lower_95": {
            "action_gap_h4": action,
            "persistence_gap_h4": persistence,
            "history_gap_h4": history,
            "distribution_value_gap_h4": distribution,
            "combined_distribution_value_gap": distribution,
            "p0_p1_local_prior_persistence_gap": p0_p1_gap,
        },
        "noncollapse": {
            "target_effective_rank_ratio": 0.17,
            "online_effective_rank_ratio": 0.20,
            "target_near_zero_variance_fraction": 0.0,
            "online_near_zero_variance_fraction": 0.0,
        },
        "all_registered_values_finite": True,
    }


def _passing_observations() -> list[dict]:
    return [
        _observation(
            0,
            real=1.0,
            joint=1.0,
            action=0.0,
            hold=0.0,
            persistence=0.0,
            history=0.0,
            distribution=0.0,
            spread=0.0,
            p0_p1_score=1.0,
            p0_p1_gap=0.0,
        ),
        _observation(
            750,
            real=0.80,
            joint=0.82,
            action=0.10,
            hold=0.08,
            persistence=0.20,
            history=0.05,
            distribution=0.06,
            spread=0.10,
            p0_p1_score=0.75,
            p0_p1_gap=0.25,
        ),
    ]


def test_configuration_changes_only_v2_schedule_and_identity() -> None:
    core_original, base_original = _snapshots()
    bindings = {
        "factual_shared_transition_v2_schedule_integrity_wrapper": {
            "path": "runner.py",
            "file_sha256": "0" * 64,
            "byte_count": 1,
        }
    }
    try:
        runner.v1._configure_core(bindings)
        expected_science = {
            name: getattr(runner.core, name) for name in _SCIENCE_NAMES
        }
        expected_additional = deepcopy(runner.core.ADDITIONAL_SCIENCE)

        runner._configure_core(bindings)
        assert {
            name: getattr(runner.core, name) for name in _SCIENCE_NAMES
        } == expected_science
        observed_additional = dict(runner.core.ADDITIONAL_SCIENCE)
        schedule = observed_additional.pop("schedule_integrity")
        assert observed_additional == expected_additional
        assert schedule["manifest"] == {
            "path": str(runner.INDEX_MANIFEST.relative_to(runner.ROOT)),
            "file_sha256": runner.INDEX_MANIFEST_SHA256,
            "byte_count": runner.INDEX_MANIFEST_BYTES,
        }
        assert runner.core.TRAIN_INDEX == runner.TRAIN_INDEX
        assert runner.core.TRAIN_INDEX_SHA256 == runner.TRAIN_INDEX_SHA256
        assert runner.core.TRAIN_INDEX_BYTES == runner.TRAIN_INDEX_BYTES
        assert runner.core.VAL_INDEX == runner.VAL_INDEX
        assert runner.core.VAL_INDEX_SHA256 == runner.VAL_INDEX_SHA256
        assert runner.core.VAL_INDEX_BYTES == runner.VAL_INDEX_BYTES
        assert runner.core.INDEX_ROW_SCHEMA == runner.INDEX_ROW_SCHEMA
        assert runner.core.OUTPUT_ROOT == runner.OUTPUT_ROOT
        assert runner.core.SCHEMA == runner.SCHEMA
        assert runner.core.PASS_DECISION == runner.PASS_DECISION
        assert runner.core.STOP_DECISION == runner.STOP_DECISION
        assert runner.core.EXECUTION_SOURCE_BINDINGS == bindings
    finally:
        _restore(core_original, base_original)


def test_exact_v2_index_bindings_are_defaults_and_cannot_be_overridden() -> None:
    core_original, base_original = _snapshots()
    try:
        runner._configure_core({})
        args = runner.core.parse_args(["--preflight-only"])
        assert args.train_index == runner.TRAIN_INDEX
        assert args.train_index_sha256 == runner.TRAIN_INDEX_SHA256
        assert args.train_index_bytes == 10_328_000
        assert args.val_index == runner.VAL_INDEX
        assert args.val_index_sha256 == runner.VAL_INDEX_SHA256
        assert args.val_index_bytes == 1_317_888
        with pytest.raises(SystemExit):
            runner.core.parse_args(
                [
                    "--preflight-only",
                    "--train-index",
                    str(runner.ROOT / ".generated/other/train.jsonl"),
                ]
            )
    finally:
        _restore(core_original, base_original)


def test_decision_reuses_all_v1_gates_and_changes_only_terminal_identity() -> None:
    observations = _passing_observations()
    v1_pass = runner._V1_DECISION(deepcopy(observations), 1_000)
    v2_pass = runner._schedule_integrity_decision(
        deepcopy(observations), 1_000
    )
    assert len(v2_pass["gates"]) == 32
    assert v2_pass["gates"] == v1_pass["gates"]
    assert v2_pass["failed_gates"] == v1_pass["failed_gates"] == []
    assert v2_pass["diagnostics"] == v1_pass["diagnostics"]
    assert v1_pass["decision"] == runner.v1.PASS_DECISION
    assert v2_pass["decision"] == runner.PASS_DECISION

    stopped_observations = _passing_observations()
    stopped_observations[1]["aggregate"]["action_gap"] = [0.0] * 4
    v1_stop = runner._V1_DECISION(deepcopy(stopped_observations), 1_000)
    v2_stop = runner._schedule_integrity_decision(
        deepcopy(stopped_observations), 1_000
    )
    assert v2_stop["gates"] == v1_stop["gates"]
    assert v2_stop["failed_gates"] == v1_stop["failed_gates"]
    assert v2_stop["diagnostics"] == v1_stop["diagnostics"]
    assert v1_stop["decision"] == runner.v1.STOP_DECISION
    assert v2_stop["decision"] == runner.STOP_DECISION


def test_runtime_install_preserves_v1_evaluator_and_run_handler() -> None:
    core_original, base_original = _snapshots()
    try:
        runner.core._evaluate = runner.v1.base._CORE_EVALUATE
        runner.core._decision = runner.v1.base._CORE_DECISION
        runner.core._run = runner.v1.base._CORE_RUN
        runner._install_runtime_adapters()
        assert (
            runner.core._evaluate
            is runner.v1._factual_shared_transition_evaluate
        )
        assert runner.core._run is runner.v1._factual_shared_transition_run
        assert runner.core._decision is runner._schedule_integrity_decision
        runner._install_runtime_adapters()
    finally:
        _restore(core_original, base_original)


def test_source_closure_requires_external_wrapper_and_binds_head_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha_name = (
        "LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_V2_SCHEDULE_INTEGRITY_"
        "WRAPPER_SHA256"
    )
    bytes_name = (
        "LEWM_FACTUAL_SHARED_TRANSITION_TRAJECTORY_H4_V2_SCHEDULE_INTEGRITY_"
        "WRAPPER_BYTES"
    )
    monkeypatch.delenv(sha_name, raising=False)
    monkeypatch.delenv(bytes_name, raising=False)
    with pytest.raises(runner.core.ContractError):
        runner._verify_source_closure()

    calls: list[tuple[Path, str, int]] = []

    def fake_binding(path: Path, sha256: str, byte_count: int) -> dict:
        calls.append((path, sha256, byte_count))
        return {
            "path": str(path),
            "file_sha256": sha256,
            "byte_count": byte_count,
        }

    monkeypatch.setattr(runner.v1.base, "_source_binding", fake_binding)
    monkeypatch.setenv(sha_name, "a" * 64)
    monkeypatch.setenv(bytes_name, "123")
    closure = runner._verify_source_closure()
    assert calls[:4] == [
        (Path(runner.__file__).resolve(), "a" * 64, 123),
        (
            runner.V1_RUNNER_SOURCE,
            runner.V1_RUNNER_SOURCE_SHA256,
            runner.V1_RUNNER_SOURCE_BYTES,
        ),
        (
            runner.V2_ADAPTER_SOURCE,
            runner.V2_ADAPTER_SOURCE_SHA256,
            runner.V2_ADAPTER_SOURCE_BYTES,
        ),
        (
            runner.V2_BUILDER_SOURCE,
            runner.V2_BUILDER_SOURCE_SHA256,
            runner.V2_BUILDER_SOURCE_BYTES,
        ),
    ]
    assert set(closure) == {
        "factual_shared_transition_v2_schedule_integrity_wrapper",
        "factual_shared_transition_v1_runner",
        "v2_schedule_integrity_index_adapter",
        "v2_schedule_integrity_index_builder",
        "trajectory_h4_wrapper_dependency",
        "shared_runner",
        "factual_shared_transition_trajectory_h4_model",
        "trajectory_h4_model_dependency",
        "local_innovation_trajectory_h4_model_dependency",
        "dense_h4_model_dependency",
        "inherited_v1_model",
        "encoder_dependency",
    }


@pytest.mark.parametrize(
    ("path", "sha256", "byte_count"),
    (
        (
            runner.V1_RUNNER_SOURCE,
            runner.V1_RUNNER_SOURCE_SHA256,
            runner.V1_RUNNER_SOURCE_BYTES,
        ),
        (
            runner.v1.MODEL_SOURCE,
            runner.v1.MODEL_SOURCE_SHA256,
            runner.v1.MODEL_SOURCE_BYTES,
        ),
        (
            runner.V2_ADAPTER_SOURCE,
            runner.V2_ADAPTER_SOURCE_SHA256,
            runner.V2_ADAPTER_SOURCE_BYTES,
        ),
        (
            runner.V2_BUILDER_SOURCE,
            runner.V2_BUILDER_SOURCE_SHA256,
            runner.V2_BUILDER_SOURCE_BYTES,
        ),
    ),
)
def test_committed_head_source_bindings_are_exact(
    path: Path,
    sha256: str,
    byte_count: int,
) -> None:
    raw = path.read_bytes()
    assert len(raw) == byte_count
    assert hashlib.sha256(raw).hexdigest() == sha256


def test_main_delegates_without_runtime_io_in_the_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    bindings = {"wrapper": {"path": "x", "file_sha256": "0" * 64}}
    monkeypatch.setattr(
        runner,
        "_verify_source_closure",
        lambda: calls.append("closure") or bindings,
    )
    monkeypatch.setattr(
        runner.v1.base,
        "_install_bound_model_package_stubs",
        lambda: calls.append("stubs"),
    )
    monkeypatch.setattr(
        runner,
        "_configure_core",
        lambda value: calls.append(("configure", value)),
    )
    monkeypatch.setattr(
        runner,
        "_install_runtime_adapters",
        lambda: calls.append("adapters"),
    )
    monkeypatch.setattr(
        runner.core,
        "main",
        lambda argv: calls.append(("core.main", argv)) or 17,
    )
    assert runner.main(["--preflight-only"]) == 17
    assert calls == [
        "closure",
        "stubs",
        ("configure", bindings),
        "adapters",
        ("core.main", ["--preflight-only"]),
    ]
