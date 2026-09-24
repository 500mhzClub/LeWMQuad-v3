from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from scripts import (
    run_go2_rgb_fixed_teacher_latent_momentum_causal_innovation_filter_trajectory_h4_jepa_v1
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
    "_terminal_failure",
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


def _training_receipt() -> dict:
    return {
        "mean_over_completed_updates": {
            "history_teacher_alignment": 1.0,
            "half_all_six_factual_local_innovation_energy_score": 2.0,
            "half_open_loop_future_cumulative_trajectory_energy_score": 3.0,
        },
        "last_completed_update": {
            "history_teacher_alignment": 4.0,
            "half_all_six_factual_local_innovation_energy_score": 5.0,
            "half_open_loop_future_cumulative_trajectory_energy_score": 6.0,
        },
        "receipt_field_semantics": {
            "history_teacher_alignment": "objective_term_weighted_one",
            "half_all_six_factual_local_innovation_energy_score": (
                "objective_term_already_weighted_one_half"
            ),
            "half_open_loop_future_cumulative_trajectory_energy_score": (
                "objective_term_already_weighted_one_half"
            ),
        },
        "objective": runner.OBJECTIVE_DESCRIPTION,
    }


def _factorized_artifact() -> dict:
    return {
        "schema": "artifact",
        "fresh_factorized_belief_increment_action_and_shared_projection_"
        "initialization": True,
        "factorized_conditional_increment_mechanism_enabled": True,
        "factual_shared_transition_objective_enabled": True,
        "factorized_conditional_increment_contract": {"inherited": True},
        "factual_shared_transition_score_weights": {
            "all_six_factual_local_innovation": 0.5,
            "open_loop_future_cumulative_trajectory": 0.5,
        },
    }


def test_configuration_changes_only_mechanism_identity_and_receipts() -> None:
    core_original, base_original = _snapshots()
    bindings = {"wrapper": {"path": "runner.py"}}
    try:
        runner.factorized._configure_core(bindings)
        expected = {
            name: getattr(runner.core, name) for name in _PRESERVED_V2_NAMES
        }
        runner._configure_core(bindings)

        assert {
            name: getattr(runner.core, name) for name in _PRESERVED_V2_NAMES
        } == expected
        assert runner.core.MODEL_MODULE == runner.MODEL_MODULE
        assert runner.core.MODEL_SOURCE == runner.MODEL_SOURCE
        assert runner.core.OUTPUT_ROOT == runner.OUTPUT_ROOT
        assert runner.core.SCHEMA == runner.SCHEMA
        assert runner.core.PASS_DECISION == runner.PASS_DECISION
        assert runner.core.STOP_DECISION == runner.STOP_DECISION
        assert runner.core.OBJECTIVE_DESCRIPTION == runner.OBJECTIVE_DESCRIPTION
        science = runner.core.ADDITIONAL_SCIENCE
        assert science["schedule_integrity"]["reuse"] == (
            "exact_causal_v2_schedule_with_new_latent_momentum_filter_model"
        )
        assert science["evaluation"] == (
            "exact_V2_evaluator_selection_and_all_32_gates"
        )
        assert science["observer"]["calls"] == (
            "exactly_twice_after_scored_observed_priors"
        )
        assert science["transition"]["shared_core"] == (
            "one_exact_parameter_set_for_p0_through_p5"
        )
        assert science["training_losses"] == [
            "half_all_six_realized_local_innovation_energy_score",
            "half_open_loop_future_cumulative_trajectory_energy_score",
            "history_teacher_alignment",
        ]
        assert runner.core.EXECUTION_SOURCE_BINDINGS == bindings
    finally:
        _restore(core_original, base_original)


def test_exact_schedule_cap_seed_and_argument_lock_are_retained() -> None:
    core_original, base_original = _snapshots()
    try:
        runner._configure_core({})
        args = runner.core.parse_args(["--preflight-only"])
        assert args.train_index == runner.v2.TRAIN_INDEX
        assert args.val_index == runner.v2.VAL_INDEX
        assert runner.core.UPDATES == 1_000
        assert runner.core.BATCH_SIZE == 16
        assert runner.core.PRESENTATIONS == 16_000
        assert runner.core.VAL_PRESENTATIONS == 2_048
        assert runner.core.OBSERVATION_UPDATES == (0, 250, 500, 750, 1_000)
        assert runner.core.MAX_GPU_SECONDS == 5_400
        assert runner.core.SEED == 20_260_727
        assert runner.core.BOOTSTRAP_REPLICATES == 1_000
        assert (
            len(runner.core.OBSERVATION_UPDATES)
            * runner.core.VAL_PRESENTATIONS
            == 10_240
        )
        assert (
            runner.core.PRESENTATIONS
            + len(runner.core.OBSERVATION_UPDATES)
            * runner.core.VAL_PRESENTATIONS
        ) * 7 == 183_680
        for override in (
            ("--resume",),
            ("--seed", "1"),
            ("--updates", "999"),
            ("--presentations", "15984"),
            ("--batch-size", "8"),
            ("--max-gpu-seconds", "1"),
            ("--checkpoint", "checkpoint.pt"),
        ):
            with pytest.raises(SystemExit):
                runner.core.parse_args(["--preflight-only", *override])
    finally:
        _restore(core_original, base_original)


@pytest.mark.parametrize(
    ("flag", "value"),
    (
        ("--train-index", ".generated/other/train.jsonl"),
        ("--train-index-sha256", "0" * 64),
        ("--train-index-bytes", str(runner.v2.TRAIN_INDEX_BYTES + 1)),
        ("--val-index", ".generated/other/val.jsonl"),
        ("--val-index-sha256", "0" * 64),
        ("--val-index-bytes", str(runner.v2.VAL_INDEX_BYTES + 1)),
        ("--model-sha256", "0" * 64),
        ("--model-bytes", str(runner.MODEL_SOURCE_BYTES + 1)),
    ),
)
def test_argument_lock_rejects_every_bound_input_override(
    flag: str,
    value: str,
) -> None:
    core_original, base_original = _snapshots()
    try:
        runner._configure_core({})
        with pytest.raises(SystemExit):
            runner.core.parse_args(["--preflight-only", flag, value])
    finally:
        _restore(core_original, base_original)


@pytest.mark.parametrize("failed", (False, True))
def test_decision_preserves_all_gates_and_only_relabels(
    monkeypatch: pytest.MonkeyPatch,
    failed: bool,
) -> None:
    inherited = {
        "decision": (
            runner.factorized.STOP_DECISION
            if failed
            else runner.factorized.PASS_DECISION
        ),
        "gates": {f"gate_{index}": True for index in range(32)},
        "failed_gates": ["gate_3"] if failed else [],
        "diagnostics": {"selected_update": 750},
    }
    monkeypatch.setattr(
        runner,
        "_FACTORIZED_DECISION",
        lambda observations, updates: deepcopy(inherited),
    )
    result = runner._latent_momentum_decision([], 1_000)
    assert result["gates"] == inherited["gates"]
    assert len(result["gates"]) == 32
    assert result["failed_gates"] == inherited["failed_gates"]
    assert result["diagnostics"] == inherited["diagnostics"]
    assert result["decision"] == (
        runner.STOP_DECISION if failed else runner.PASS_DECISION
    )


def test_run_truthfully_relabels_loss_and_mechanism_receipts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics = {"training_losses": _training_receipt(), "other": 17}
    artifact = _factorized_artifact()
    decision = {"decision": runner.PASS_DECISION}
    monkeypatch.setattr(
        runner,
        "_FACTORIZED_RUN",
        lambda *args, **kwargs: (metrics, artifact, decision),
    )
    observed_metrics, observed_artifact, observed_decision = (
        runner._latent_momentum_run("x")
    )

    assert observed_decision is decision
    assert observed_metrics["other"] == 17
    training = observed_metrics["training_losses"]
    assert training["objective"] == runner.OBJECTIVE_DESCRIPTION
    new_name = "half_all_six_realized_local_innovation_energy_score"
    old_name = "half_all_six_factual_local_innovation_energy_score"
    for bucket_name in (
        "mean_over_completed_updates",
        "last_completed_update",
    ):
        assert old_name not in training[bucket_name]
        assert new_name in training[bucket_name]
    assert old_name not in training["receipt_field_semantics"]
    assert "future_recursive_q_t" in training["receipt_field_semantics"][new_name]
    assert "factorized_conditional_increment_contract" not in observed_artifact
    assert observed_artifact[
        "latent_momentum_causal_innovation_filter_enabled"
    ] is True
    contract = observed_artifact[
        "latent_momentum_causal_innovation_filter_contract"
    ]
    assert contract["shared_prior_calls"] == 6
    assert contract["post_prior_observer_calls"] == 2
    assert contract["future_raw_z_or_explicit_increment_bypass"] is False


@pytest.mark.parametrize("corruption", ("loss", "objective", "artifact"))
def test_run_fails_closed_if_inherited_receipts_change(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    metrics = {"training_losses": _training_receipt()}
    artifact = _factorized_artifact()
    if corruption == "loss":
        del metrics["training_losses"]["last_completed_update"]
    elif corruption == "objective":
        metrics["training_losses"]["objective"] = (
            runner.INHERITED_OBJECTIVE_DESCRIPTION
        )
    else:
        artifact["factorized_conditional_increment_mechanism_enabled"] = False
    monkeypatch.setattr(
        runner,
        "_FACTORIZED_RUN",
        lambda *args, **kwargs: (metrics, artifact, {}),
    )
    with pytest.raises(runner.core.ContractError):
        runner._latent_momentum_run()


def test_runtime_install_keeps_evaluator_and_complete_terminal_handler() -> None:
    core_original, base_original = _snapshots()
    try:
        runner.core._evaluate = runner.v1.base._CORE_EVALUATE
        runner.core._decision = runner.v1.base._CORE_DECISION
        runner.core._run = runner.v1.base._CORE_RUN
        runner._install_runtime_adapters()
        assert runner.core._evaluate is runner.v1._factual_shared_transition_evaluate
        assert runner.core._run is runner._latent_momentum_run
        assert runner.core._decision is runner._latent_momentum_decision
        assert runner.core._terminal_failure is runner._FACTORIZED_TERMINAL_FAILURE
        runner._install_runtime_adapters()
    finally:
        _restore(core_original, base_original)


def test_source_closure_is_complete_and_requires_external_self_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = (
        "LEWM_LATENT_MOMENTUM_CAUSAL_INNOVATION_FILTER_TRAJECTORY_H4_V1_"
    )
    monkeypatch.delenv(prefix + "WRAPPER_SHA256", raising=False)
    monkeypatch.delenv(prefix + "WRAPPER_BYTES", raising=False)
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
    monkeypatch.setenv(prefix + "WRAPPER_SHA256", "a" * 64)
    monkeypatch.setenv(prefix + "WRAPPER_BYTES", "18715")
    closure = runner._verify_source_closure()
    assert len(closure) == len(calls) == 16
    assert calls[:3] == [
        (Path(runner.__file__).resolve(), "a" * 64, 18_715),
        (
            runner.FACTORIZED_RUNNER_SOURCE,
            runner.FACTORIZED_RUNNER_SOURCE_SHA256,
            runner.FACTORIZED_RUNNER_SOURCE_BYTES,
        ),
        (
            runner.MODEL_SOURCE,
            runner.MODEL_SOURCE_SHA256,
            runner.MODEL_SOURCE_BYTES,
        ),
    ]
    assert {
        "latent_momentum_causal_innovation_filter_wrapper",
        "latent_momentum_causal_innovation_filter_model",
        "factorized_conditional_increment_wrapper_dependency",
        "factorized_conditional_increment_model_dependency",
        "encoder_dependency",
    } <= set(closure)


@pytest.mark.parametrize(
    ("path", "sha256", "byte_count"),
    (
        (
            runner.FACTORIZED_RUNNER_SOURCE,
            runner.FACTORIZED_RUNNER_SOURCE_SHA256,
            runner.FACTORIZED_RUNNER_SOURCE_BYTES,
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


def test_main_is_thin_source_only_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    bindings = {"wrapper": {"path": "x"}}
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
        lambda argv: calls.append(("main", argv)) or 17,
    )
    assert runner.main(["--preflight-only"]) == 17
    assert calls == [
        "closure",
        "stubs",
        ("configure", bindings),
        "adapters",
        ("main", ["--preflight-only"]),
    ]
