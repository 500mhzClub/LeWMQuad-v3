from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from scripts import (
    run_go2_rgb_fixed_teacher_factorized_conditional_increment_trajectory_h4_jepa_v1
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
_PRESERVED_V2_NAMES = (
    "TRAIN_INDEX",
    "TRAIN_INDEX_SHA256",
    "TRAIN_INDEX_BYTES",
    "VAL_INDEX",
    "VAL_INDEX_SHA256",
    "VAL_INDEX_BYTES",
    "INDEX_ROW_SCHEMA",
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


def test_configuration_preserves_v2_science_schedule_gates_and_cap() -> None:
    core_original, base_original = _snapshots()
    bindings = {
        "factorized_conditional_increment_trajectory_h4_wrapper": {
            "path": "runner.py",
            "file_sha256": "0" * 64,
            "byte_count": 1,
        }
    }
    try:
        runner.v2._configure_core(bindings)
        expected = {
            name: getattr(runner.core, name) for name in _PRESERVED_V2_NAMES
        }
        expected_schedule = deepcopy(
            runner.core.ADDITIONAL_SCIENCE["schedule_integrity"]
        )

        runner._configure_core(bindings)
        assert {
            name: getattr(runner.core, name) for name in _PRESERVED_V2_NAMES
        } == expected
        assert runner.core.MODEL_MODULE == runner.MODEL_MODULE
        assert runner.core.MODEL_SOURCE == runner.MODEL_SOURCE
        assert runner.core.MODEL_SOURCE_SHA256 == runner.MODEL_SOURCE_SHA256
        assert runner.core.MODEL_SOURCE_BYTES == runner.MODEL_SOURCE_BYTES
        assert runner.core.OUTPUT_ROOT == runner.OUTPUT_ROOT
        assert runner.core.SCHEMA == runner.SCHEMA
        assert runner.core.PASS_DECISION == runner.PASS_DECISION
        assert runner.core.STOP_DECISION == runner.STOP_DECISION
        observed_schedule = dict(
            runner.core.ADDITIONAL_SCIENCE["schedule_integrity"]
        )
        assert "replacement" not in observed_schedule
        assert observed_schedule["reuse"] == (
            "exact_causal_v2_schedule_with_new_factorized_model"
        )
        expected_schedule.pop("replacement")
        observed_schedule.pop("reuse")
        assert observed_schedule == expected_schedule
        factorization = runner.core.ADDITIONAL_SCIENCE["factorization"]
        assert factorization["generic_current_state_successor_bypass"] is False
        assert factorization["uniform_action_mean"] == "W0(d_t)"
        assert runner.core.ADDITIONAL_SCIENCE["evaluation"] == (
            "exact_V2_evaluator_selection_and_all_32_gates"
        )
        assert runner.core.EXECUTION_SOURCE_BINDINGS == bindings
    finally:
        _restore(core_original, base_original)


def test_configuration_keeps_exact_v2_indexes_and_argument_lock() -> None:
    core_original, base_original = _snapshots()
    try:
        runner._configure_core({})
        args = runner.core.parse_args(["--preflight-only"])
        assert args.train_index == runner.v2.TRAIN_INDEX
        assert args.train_index_sha256 == runner.v2.TRAIN_INDEX_SHA256
        assert args.train_index_bytes == runner.v2.TRAIN_INDEX_BYTES
        assert args.val_index == runner.v2.VAL_INDEX
        assert args.val_index_sha256 == runner.v2.VAL_INDEX_SHA256
        assert args.val_index_bytes == runner.v2.VAL_INDEX_BYTES
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


def test_decision_reuses_all_32_v2_gates_and_only_relabels() -> None:
    observations = _passing_observations()
    v2_pass = runner._V2_DECISION(deepcopy(observations), 1_000)
    factorized_pass = runner._factorized_conditional_increment_decision(
        deepcopy(observations), 1_000
    )
    assert len(factorized_pass["gates"]) == 32
    assert factorized_pass["gates"] == v2_pass["gates"]
    assert factorized_pass["failed_gates"] == v2_pass["failed_gates"] == []
    assert factorized_pass["diagnostics"] == v2_pass["diagnostics"]
    assert v2_pass["decision"] == runner.v2.PASS_DECISION
    assert factorized_pass["decision"] == runner.PASS_DECISION

    stopped = _passing_observations()
    stopped[1]["aggregate"]["action_gap"] = [0.0] * 4
    v2_stop = runner._V2_DECISION(deepcopy(stopped), 1_000)
    factorized_stop = runner._factorized_conditional_increment_decision(
        deepcopy(stopped), 1_000
    )
    assert factorized_stop["gates"] == v2_stop["gates"]
    assert factorized_stop["failed_gates"] == v2_stop["failed_gates"]
    assert factorized_stop["diagnostics"] == v2_stop["diagnostics"]
    assert v2_stop["decision"] == runner.v2.STOP_DECISION
    assert factorized_stop["decision"] == runner.STOP_DECISION


def test_run_wraps_only_v1_factual_artifact_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {"selection_rule": "unchanged"}
    decision = {"decision": runner.PASS_DECISION}
    artifact = {
        "schema": "artifact",
        "fresh_shared_transition_mode_and_residual_head_initialization": True,
        "factual_shared_transition_objective_enabled": True,
        "factual_shared_transition_score_weights": {
            "all_six_factual_local_innovation": 0.5,
            "open_loop_future_cumulative_trajectory": 0.5,
        },
    }
    monkeypatch.setattr(
        runner,
        "_V1_FACTUAL_RUN",
        lambda *args, **kwargs: (metrics, artifact, decision),
    )
    observed_metrics, observed_artifact, observed_decision = (
        runner._factorized_conditional_increment_run("x", access={})
    )
    assert observed_metrics is metrics
    assert observed_decision is decision
    assert (
        "fresh_shared_transition_mode_and_residual_head_initialization"
        not in observed_artifact
    )
    assert observed_artifact[
        "fresh_factorized_belief_increment_action_and_shared_projection_"
        "initialization"
    ] is True
    assert observed_artifact["factorized_conditional_increment_mechanism_enabled"]
    assert observed_artifact["factual_shared_transition_objective_enabled"]
    assert observed_artifact["factual_shared_transition_score_weights"] == (
        artifact["factual_shared_transition_score_weights"]
    )


def test_run_fails_closed_if_inherited_initialization_receipt_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_V1_FACTUAL_RUN",
        lambda *args, **kwargs: ({}, {}, {}),
    )
    with pytest.raises(runner.core.ContractError):
        runner._factorized_conditional_increment_run(access={})


def test_runtime_install_preserves_v2_evaluator_and_wraps_run_decision() -> None:
    core_original, base_original = _snapshots()
    try:
        runner.core._evaluate = runner.v1.base._CORE_EVALUATE
        runner.core._decision = runner.v1.base._CORE_DECISION
        runner.core._run = runner.v1.base._CORE_RUN
        runner._install_runtime_adapters()
        assert runner.core._evaluate is runner.v1._factual_shared_transition_evaluate
        assert runner.core._run is runner._factorized_conditional_increment_run
        assert (
            runner.core._decision
            is runner._factorized_conditional_increment_decision
        )
        runner._install_runtime_adapters()
    finally:
        _restore(core_original, base_original)


def test_source_closure_requires_external_self_binding_and_is_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha_name = (
        "LEWM_FACTORIZED_CONDITIONAL_INCREMENT_TRAJECTORY_H4_V1_WRAPPER_"
        "SHA256"
    )
    bytes_name = (
        "LEWM_FACTORIZED_CONDITIONAL_INCREMENT_TRAJECTORY_H4_V1_WRAPPER_BYTES"
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
    assert calls[:6] == [
        (Path(runner.__file__).resolve(), "a" * 64, 123),
        (
            runner.V2_RUNNER_SOURCE,
            runner.V2_RUNNER_SOURCE_SHA256,
            runner.V2_RUNNER_SOURCE_BYTES,
        ),
        (
            runner.v2.V1_RUNNER_SOURCE,
            runner.v2.V1_RUNNER_SOURCE_SHA256,
            runner.v2.V1_RUNNER_SOURCE_BYTES,
        ),
        (
            runner.v2.V2_ADAPTER_SOURCE,
            runner.v2.V2_ADAPTER_SOURCE_SHA256,
            runner.v2.V2_ADAPTER_SOURCE_BYTES,
        ),
        (
            runner.v2.V2_BUILDER_SOURCE,
            runner.v2.V2_BUILDER_SOURCE_SHA256,
            runner.v2.V2_BUILDER_SOURCE_BYTES,
        ),
        (
            runner.MODEL_SOURCE,
            runner.MODEL_SOURCE_SHA256,
            runner.MODEL_SOURCE_BYTES,
        ),
    ]
    assert set(closure) == {
        "factorized_conditional_increment_trajectory_h4_wrapper",
        "v2_schedule_integrity_wrapper_dependency",
        "factual_shared_transition_v1_runner_dependency",
        "v2_schedule_integrity_index_adapter",
        "v2_schedule_integrity_index_builder",
        "factorized_conditional_increment_trajectory_h4_model",
        "factual_shared_transition_trajectory_h4_model_dependency",
        "trajectory_h4_wrapper_dependency",
        "shared_runner",
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
            runner.V2_RUNNER_SOURCE,
            runner.V2_RUNNER_SOURCE_SHA256,
            runner.V2_RUNNER_SOURCE_BYTES,
        ),
        (
            runner.v2.V1_RUNNER_SOURCE,
            runner.v2.V1_RUNNER_SOURCE_SHA256,
            runner.v2.V1_RUNNER_SOURCE_BYTES,
        ),
        (
            runner.MODEL_SOURCE,
            runner.MODEL_SOURCE_SHA256,
            runner.MODEL_SOURCE_BYTES,
        ),
    ),
)
def test_bound_source_hashes_and_bytes_are_exact(
    path: Path,
    sha256: str,
    byte_count: int,
) -> None:
    raw = path.read_bytes()
    assert len(raw) == byte_count
    assert hashlib.sha256(raw).hexdigest() == sha256


def test_main_is_a_thin_delegation_without_wrapper_runtime_io(
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
