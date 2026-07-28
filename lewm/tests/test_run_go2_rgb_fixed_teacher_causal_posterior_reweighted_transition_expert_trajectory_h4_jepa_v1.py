from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lewm.models.go2_rgb_fixed_teacher_causal_posterior_reweighted_transition_expert_trajectory_h4_jepa_v1 import (
    weighted_pairwise_spread,
    weighted_spherical_centroid,
    weighted_trajectory_energy_score,
)
from scripts import (
    run_go2_rgb_fixed_teacher_causal_posterior_reweighted_transition_expert_trajectory_h4_jepa_v1
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
_PRESERVED_NAMES = (
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
        {name: getattr(runner.base, name) for name in _BASE_MUTABLE_NAMES},
    )


def _restore(
    core_values: dict[str, object],
    base_values: dict[str, object],
) -> None:
    for name, value in base_values.items():
        setattr(runner.base, name, value)
    for name, value in core_values.items():
        setattr(runner.core, name, value)


def _training_receipt() -> dict:
    buckets = {
        "history_teacher_alignment": 1.0,
        "half_all_six_factual_local_innovation_energy_score": 2.0,
        "half_open_loop_future_cumulative_trajectory_energy_score": 3.0,
        "diagnostic_centroid_absolute_future_error": 4.0,
        "total": 6.0,
    }
    return {
        "mean_over_completed_updates": dict(buckets),
        "last_completed_update": dict(buckets),
        "receipt_field_semantics": {
            "diagnostic_centroid_absolute_future_error": (
                "measured_by_shared_runner_but_weight_zero"
            ),
            "history_teacher_alignment": "objective_weight_one",
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
        "factorized_conditional_increment_contract": {
            "incoming_increment": (
                "factual_for_observed_edges_and_post_renormalization_realized_"
                "for_open_loop_edges"
            ),
            "action_code": "uniformly_centered_after_complete_action_tower",
            "action_free_belief_current_action_access": False,
            "shared_projection_bias": False,
            "shared_projection_zero_initialized": True,
            "generic_current_state_successor_bypass": False,
        },
        "factual_shared_transition_score_weights": {
            "all_six_factual_local_innovation": 0.5,
            "open_loop_future_cumulative_trajectory": 0.5,
        },
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_publish_spy(
    published: list[tuple[str, dict, dict]],
):
    def publish(
        output_fd: int,
        name: str,
        payload: dict,
    ) -> tuple[dict, dict]:
        assert output_fd == 17
        observed = dict(payload)
        observed["content_sha256"] = hashlib.sha256(
            _canonical_bytes(payload)
        ).hexdigest()
        raw = _canonical_bytes(observed) + b"\n"
        binding = {
            "path": name,
            "byte_count": len(raw),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": observed["content_sha256"],
        }
        published.append((name, observed, binding))
        return observed, binding

    return publish


def test_configuration_preserves_schedule_and_installs_exact_science() -> None:
    core_original, base_original = _snapshots()
    bindings = {"wrapper": {"path": "runner.py"}}
    try:
        runner.system_id._configure_core(bindings)
        expected = {
            name: getattr(runner.core, name) for name in _PRESERVED_NAMES
        }
        runner._configure_core(bindings)

        assert {
            name: getattr(runner.core, name) for name in _PRESERVED_NAMES
        } == expected
        assert runner.core.MODEL_MODULE == runner.MODEL_MODULE
        assert runner.core.MODEL_SOURCE == runner.MODEL_SOURCE
        assert runner.core.MODEL_SOURCE_SHA256 == runner.MODEL_SOURCE_SHA256
        assert runner.core.MODEL_SOURCE_BYTES == runner.MODEL_SOURCE_BYTES
        assert runner.core.OUTPUT_ROOT == runner.OUTPUT_ROOT
        assert runner.core.SCHEMA == runner.SCHEMA
        assert runner.core.OBJECTIVE_DESCRIPTION == runner.OBJECTIVE_DESCRIPTION
        science = runner.core.ADDITIONAL_SCIENCE
        assert science["state"]["sole_history"] == (
            "four_strictly_positive_simplex_probabilities"
        )
        assert science["posterior"] == {
            "initial_mass": [0.25, 0.25, 0.25, 0.25],
            "update_calls": 2,
            "future_update_calls": 0,
            "squared_error": (
                "mean_token_sum_feature_squared_prior_minus_online_destination"
            ),
            "likelihood": "exp(-d_k/(mean_four_d+1e-6))",
            "update": "normalize(w_previous_times_likelihood)",
            "learned_temperature_gain_gate_prior_or_detach": False,
        }
        assert science["state"]["final_hidden_particles"] == (
            "compatibility_alias_of_posterior_probabilities_not_extra_state"
        )
        assert science["proper_score"]["all_six_local_mass"] == "equal_quarter"
        assert science["proper_score"]["future_mass"] == "causal_w2"
        assert science["evaluation_weight_rules"] == {
            "real_wrong_action_and_all_hold": "factual_branch_w2",
            "reversed_history": "independently_recomputed_reversed_w2",
            "reset_history": "independently_recomputed_reset_w2",
            "centroid_and_pair_spread": "posterior_weighted",
            "persistence": "weight_invariant_identical_atoms",
        }
        assert science["schedule_integrity"]["reuse"] == (
            "exact_causal_v2_schedule_with_new_causal_posterior_reweighted_"
            "transition_expert_model"
        )
        assert runner.core.EXECUTION_SOURCE_BINDINGS == bindings
    finally:
        _restore(core_original, base_original)


def test_exact_schedule_cap_seed_and_all_argument_locks_are_retained() -> None:
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
        assert len(runner.core.OBSERVATION_UPDATES) * 2_048 == 10_240
        assert (16_000 + 10_240) * 7 == 183_680
        for override in (
            ("--resume",),
            ("--seed", "1"),
            ("--updates", "999"),
            ("--presentations", "15984"),
            ("--batch-size", "8"),
            ("--max-gpu-seconds", "1"),
            ("--checkpoint", "checkpoint.pt"),
            ("--temperature", "1"),
            ("--likelihood-epsilon", "0.1"),
            ("--expert-count", "8"),
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
        ("--model-bytes", "1"),
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


def test_weighted_math_exactly_matches_model_helpers_and_uniform_k4() -> None:
    generator = torch.Generator(device="cpu").manual_seed(19)
    atoms = F.normalize(torch.randn(2, 4, 4, 3, 5, generator=generator), dim=-1)
    target = F.normalize(torch.randn(2, 4, 3, 5, generator=generator), dim=-1)
    weights = torch.tensor(
        [[0.55, 0.25, 0.15, 0.05], [0.1, 0.2, 0.3, 0.4]],
        dtype=atoms.dtype,
    )
    runtime = SimpleNamespace(torch=torch)

    actual = runner._weighted_energy(atoms, target, weights, runtime)
    expected = weighted_trajectory_energy_score(atoms, target, weights)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(
        runner._weighted_centroid(atoms, weights, runtime),
        weighted_spherical_centroid(atoms, weights),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        runner._weighted_spread(atoms, weights, runtime),
        weighted_pairwise_spread(atoms, weights),
        rtol=0.0,
        atol=1e-7,
    )

    uniform = atoms.new_full((2, 4), 0.25)
    horizon, joint, combined = runner._weighted_energy(
        atoms, target, uniform, runtime
    )
    inherited_horizon = runner.base._marginal_energy(atoms, target, runtime)
    inherited_joint = runner.base._joint_energy(atoms, target, runtime)
    torch.testing.assert_close(horizon, inherited_horizon, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(joint, inherited_joint, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(
        combined,
        0.5 * inherited_joint + 0.5 * inherited_horizon.mean(dim=1),
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize("corruption", ("shape", "zero", "sum", "nan"))
def test_posterior_extraction_fails_closed(corruption: str) -> None:
    atoms = torch.zeros(2, 4, 4, 1, 2)
    weights = torch.full((2, 4), 0.25)
    if corruption == "shape":
        weights = weights[:, :3]
    elif corruption == "zero":
        weights[0] = torch.tensor([0.0, 0.5, 0.25, 0.25])
    elif corruption == "sum":
        weights[0, 0] = 0.3
    else:
        weights[0, 0] = torch.nan
    with pytest.raises(runner.core.ContractError):
        runner._probabilities(
            {"posterior_probabilities": weights},
            atoms,
            SimpleNamespace(torch=torch),
        )


def test_future_control_must_return_factual_posterior_unchanged() -> None:
    atoms = torch.zeros(2, 4, 4, 1, 2)
    weights = torch.tensor(
        [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]]
    )
    actions = torch.zeros(2, 4, dtype=torch.long)

    class Model:
        returned_weights = weights

        def predict_trajectory_atoms_and_probabilities_from_belief(
            self,
            belief: torch.Tensor,
            future_actions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del belief, future_actions
            return atoms, self.returned_weights

    model = Model()
    observed = runner._control_distribution(
        model,
        torch.zeros(1),
        actions,
        weights,
        atoms,
        SimpleNamespace(torch=torch),
    )
    assert observed is atoms
    model.returned_weights = weights.roll(1, dims=1)
    with pytest.raises(runner.core.ContractError):
        runner._control_distribution(
            model,
            torch.zeros(1),
            actions,
            weights,
            atoms,
            SimpleNamespace(torch=torch),
        )


def test_evaluator_routes_each_branch_mass_and_keeps_p0p1_equal_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = len(runner.core.FAMILIES)
    generator = torch.Generator(device="cpu").manual_seed(23)
    atoms = F.normalize(
        torch.randn(batch, 4, 4, 2, 3, generator=generator), dim=-1
    )
    target = F.normalize(
        torch.randn(batch, 4, 2, 3, generator=generator), dim=-1
    )
    online = F.normalize(
        torch.randn(batch, 3, 2, 3, generator=generator), dim=-1
    )
    teacher = F.normalize(
        torch.randn(batch, 3, 2, 3, generator=generator), dim=-1
    )
    factual = torch.tensor([0.4, 0.3, 0.2, 0.1]).expand(batch, -1).clone()
    reversed_mass = (
        torch.tensor([0.1, 0.4, 0.3, 0.2]).expand(batch, -1).clone()
    )
    reset_mass = (
        torch.tensor([0.2, 0.1, 0.4, 0.3]).expand(batch, -1).clone()
    )

    def belief_for(weights: torch.Tensor) -> torch.Tensor:
        belief = torch.zeros(batch, 5, 2, 3)
        belief[:, 4].reshape(batch, -1)[:, :4] = weights
        return belief

    def output_for(weights: torch.Tensor, offset: float) -> dict:
        return {
            "trajectory_latents": F.normalize(atoms + offset, dim=-1),
            "posterior_probabilities": weights,
            "observed_prior_latents": F.normalize(
                torch.randn(batch, 4, 2, 2, 3, generator=generator), dim=-1
            ),
            "trajectory_innovations": torch.randn(
                batch, 4, 4, 2, 3, generator=generator
            ),
            "history_latents": online,
            "belief_latents": belief_for(weights),
        }

    outputs = iter(
        (
            output_for(factual, 0.0),
            output_for(reversed_mass, 0.1),
            output_for(reset_mass, -0.1),
        )
    )
    monkeypatch.setattr(
        runner.core,
        "_model_forward",
        lambda model, history, past, future: next(outputs),
    )
    rgb = torch.zeros(batch, 7, 1, 1, 1)
    actions = torch.zeros(batch, 6, dtype=torch.long)
    monkeypatch.setattr(
        runner.core,
        "_load_batch",
        lambda rows, **kwargs: (rgb, actions),
    )
    monkeypatch.setattr(runner.core, "_target_encode", lambda model, value: target)
    monkeypatch.setattr(
        runner.core,
        "_pool_features",
        lambda value, time_index: value[:, time_index].mean(dim=1),
    )
    monkeypatch.setattr(
        runner.core,
        "_effective_rank",
        lambda values, runtime: (0.2, 0.0),
    )
    monkeypatch.setattr(
        runner.core,
        "_bootstrap_lower",
        lambda values, seed: 0.1,
    )

    equal_mass_calls: list[tuple[torch.Size, torch.Size]] = []

    def equal_mass_local(
        innovations: torch.Tensor,
        targets: torch.Tensor,
        runtime: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del runtime
        equal_mass_calls.append((innovations.shape, targets.shape))
        return torch.ones(batch), torch.zeros(batch)

    monkeypatch.setattr(
        runner.v1,
        "_normalized_local_combined_score",
        equal_mass_local,
    )
    weighted_calls: list[torch.Tensor] = []
    original_weighted = runner._weighted_energy

    def weighted_spy(
        value_atoms: torch.Tensor,
        value_target: torch.Tensor,
        value_weights: torch.Tensor,
        runtime: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weighted_calls.append(value_weights.detach().clone())
        return original_weighted(
            value_atoms, value_target, value_weights, runtime
        )

    monkeypatch.setattr(runner, "_weighted_energy", weighted_spy)

    class Model:
        training = True
        control_weights: list[torch.Tensor]

        def __init__(self) -> None:
            self.control_weights = []

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

        def _encode_fixed_teacher_history(self, history: torch.Tensor) -> torch.Tensor:
            del history
            return teacher

        def posterior_probabilities_from_belief(
            self, belief: torch.Tensor
        ) -> torch.Tensor:
            return belief[:, 4].reshape(batch, -1)[:, :4]

        def predict_trajectory_atoms_and_probabilities_from_belief(
            self,
            belief: torch.Tensor,
            future_actions: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del future_actions
            weights = self.posterior_probabilities_from_belief(belief)
            self.control_weights.append(weights.detach().clone())
            return atoms, weights

    model = Model()
    rows = [
        SimpleNamespace(family=family, scene_id=f"scene_{index}")
        for index, family in enumerate(runner.core.FAMILIES)
    ]
    result = runner._posterior_evaluate(
        model,
        rows,
        root_fd=-1,
        runtime=SimpleNamespace(torch=torch),
        access=Counter(),
        device=torch.device("cpu"),
        update=250,
    )

    assert model.training is True
    assert len(model.control_weights) == 2
    for observed in model.control_weights:
        torch.testing.assert_close(observed, factual)
    assert len(equal_mass_calls) == 1
    assert equal_mass_calls[0] == (
        torch.Size((batch, 4, 2, 2, 3)),
        torch.Size((batch, 2, 2, 3)),
    )
    assert len(weighted_calls) == 8
    for index in (0, 1, 2, 5, 6, 7):
        torch.testing.assert_close(weighted_calls[index], factual)
    torch.testing.assert_close(weighted_calls[3], reversed_mass)
    torch.testing.assert_close(weighted_calls[4], reset_mass)
    assert result["update"] == 250
    assert result["validation_rows"] == batch
    assert set(result["family"]) == set(runner.core.FAMILIES)
    assert result["all_registered_values_finite"] is True


@pytest.mark.parametrize("failed", (False, True))
def test_decision_preserves_all_32_gates_and_only_relabels(
    monkeypatch: pytest.MonkeyPatch,
    failed: bool,
) -> None:
    inherited = {
        "decision": (
            runner.system_id.STOP_DECISION
            if failed
            else runner.system_id.PASS_DECISION
        ),
        "gates": {f"gate_{index}": True for index in range(32)},
        "failed_gates": ["gate_3"] if failed else [],
        "diagnostics": {"selected_update": 1_000},
    }
    monkeypatch.setattr(
        runner,
        "_SYSTEM_ID_DECISION",
        lambda observations, updates: deepcopy(inherited),
    )
    result = runner._posterior_decision([], 1_000)
    assert result["gates"] == inherited["gates"]
    assert len(result["gates"]) == 32
    assert result["failed_gates"] == inherited["failed_gates"]
    assert result["diagnostics"] == inherited["diagnostics"]
    assert result["decision"] == (
        runner.STOP_DECISION if failed else runner.PASS_DECISION
    )


def test_run_truthfully_replaces_factorized_receipts(
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
        runner._posterior_run("x")
    )

    assert observed_decision is decision
    assert "half_all_six_factual_local_innovation_energy_score" not in (
        observed_metrics["training_losses"]["last_completed_update"]
    )
    assert (
        "half_all_six_realized_local_innovation_energy_score"
        in observed_metrics["training_losses"]["last_completed_update"]
    )
    inherited_cumulative = (
        "half_open_loop_future_cumulative_trajectory_energy_score"
    )
    weighted_cumulative = (
        "half_open_loop_future_posterior_weighted_cumulative_trajectory_"
        "energy_score"
    )
    for bucket_name in (
        "mean_over_completed_updates",
        "last_completed_update",
    ):
        bucket = observed_metrics["training_losses"][bucket_name]
        assert inherited_cumulative not in bucket
        assert bucket[weighted_cumulative] == 3.0
    semantics = observed_metrics["training_losses"][
        "receipt_field_semantics"
    ]
    assert inherited_cumulative not in semantics
    assert semantics[weighted_cumulative] == (
        "objective_term_already_weighted_one_half_with_causal_w2_"
        "posterior_mass"
    )
    assert "factorized_conditional_increment_mechanism_enabled" not in (
        observed_artifact
    )
    assert observed_artifact[
        "causal_posterior_reweighted_transition_expert_enabled"
    ] is True
    contract = observed_artifact[
        "causal_posterior_reweighted_transition_expert_contract"
    ]
    assert contract["post_prior_evidence_update_calls"] == 2
    assert contract["future_evidence_update_calls"] == 0
    assert contract["final_hidden_particles"] == (
        "compatibility_alias_of_posterior_probabilities_not_extra_state"
    )
    assert contract["future_probabilities_bitwise_fixed"] is True
    assert contract["wrong_and_hold_mass"] == "factual_w2"
    assert contract["reverse_and_reset_mass"] == (
        "independently_recomputed_branch_w2"
    )


@pytest.mark.parametrize("corruption", ("objective", "artifact", "contract"))
def test_run_fails_closed_if_inherited_receipts_change(
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    metrics = {"training_losses": _training_receipt()}
    artifact = _factorized_artifact()
    if corruption == "objective":
        metrics["training_losses"]["objective"] = "changed"
    elif corruption == "artifact":
        artifact["factorized_conditional_increment_mechanism_enabled"] = False
    else:
        artifact["factorized_conditional_increment_contract"] = {}
    monkeypatch.setattr(
        runner,
        "_FACTORIZED_RUN",
        lambda *args, **kwargs: (metrics, artifact, {}),
    )
    with pytest.raises(runner.core.ContractError):
        runner._posterior_run()


def test_runtime_install_replaces_evaluator_and_keeps_terminal_handler() -> None:
    core_original, base_original = _snapshots()
    try:
        runner.core._evaluate = runner.base._CORE_EVALUATE
        runner.core._decision = runner.base._CORE_DECISION
        runner.core._run = runner.base._CORE_RUN
        runner._install_runtime_adapters()
        assert runner.core._evaluate is runner._posterior_evaluate
        assert runner.core._run is runner._posterior_run
        assert runner.core._decision is runner._posterior_decision
        assert runner.core._terminal_failure is runner._SYSTEM_ID_TERMINAL_FAILURE
        runner._install_runtime_adapters()
    finally:
        _restore(core_original, base_original)


def test_normal_main_publishes_canonical_successor_receipt_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_original, base_original = _snapshots()
    published: list[tuple[str, dict, dict]] = []
    closed: list[int] = []
    try:
        runner._configure_core({})
        runner.core._run = runner._posterior_run
        runner.core._terminal_failure = runner._SYSTEM_ID_TERMINAL_FAILURE
        metrics = {
            "schema": f"{runner.SCHEMA}_metrics_v1",
            "training_losses": _training_receipt(),
        }
        artifact = {
            **_factorized_artifact(),
            "schema": f"{runner.SCHEMA}_artifact_v1",
        }
        decision = {
            "decision": runner.PASS_DECISION,
            "gates": {f"gate_{index}": True for index in range(32)},
            "failed_gates": [],
            "diagnostics": {"selected_update": 1_000},
        }
        monkeypatch.setattr(
            runner,
            "_FACTORIZED_RUN",
            lambda *args, **kwargs: (metrics, artifact, decision),
        )
        monkeypatch.setattr(
            runner.core,
            "_validate_census_receipt",
            lambda: {"path": "census.json", "file_sha256": "1" * 64},
        )
        monkeypatch.setattr(runner.core, "_reserve_output", lambda: 17)
        monkeypatch.setattr(
            runner.core,
            "_publish_json",
            _canonical_publish_spy(published),
        )
        monkeypatch.setattr(
            runner.core.os,
            "close",
            lambda descriptor: closed.append(descriptor),
        )

        assert runner.core.main(["--execute"]) == 0

        assert [name for name, _payload, _binding in published] == [
            "reservation.json",
            "metrics.json",
            "artifact.json",
            "access.json",
            "result.json",
            "completed.json",
        ]
        payloads = {name: payload for name, payload, _binding in published}
        bindings = {name: binding for name, _payload, binding in published}
        assert payloads["reservation.json"]["schema"] == (
            f"{runner.SCHEMA}_reservation_v1"
        )
        assert payloads["metrics.json"]["schema"] == (
            f"{runner.SCHEMA}_metrics_v1"
        )
        assert payloads["artifact.json"]["schema"] == (
            f"{runner.SCHEMA}_artifact_v1"
        )
        assert payloads["access.json"]["schema"] == (
            f"{runner.SCHEMA}_access_v1"
        )
        assert payloads["result.json"]["schema"] == (
            f"{runner.SCHEMA}_result_v1"
        )
        completion = payloads["completed.json"]
        assert completion["schema"] == f"{runner.SCHEMA}_completion_v1"
        assert completion["status"] == "COMPLETE"
        assert completion["decision"] == runner.PASS_DECISION
        for field, name in (
            ("reservation", "reservation.json"),
            ("metrics", "metrics.json"),
            ("artifact", "artifact.json"),
            ("access", "access.json"),
            ("result", "result.json"),
        ):
            assert completion[field] == bindings[name]
            assert completion["cross_bindings"][
                f"{field}_content_sha256"
            ] == payloads[name]["content_sha256"]
        assert closed == [17]
    finally:
        _restore(core_original, base_original)


def test_caught_failure_publishes_canonical_successor_receipt_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core_original, base_original = _snapshots()
    published: list[tuple[str, dict, dict]] = []
    closed: list[int] = []
    try:
        runner._configure_core({})
        runner.core._terminal_failure = runner._SYSTEM_ID_TERMINAL_FAILURE

        def fail_run(*args, **kwargs):
            access = kwargs["access"]
            access["optimizer_update_count"] = 7
            access["train_sequence_presentation_count"] = 112
            raise RuntimeError("synthetic caught failure")

        runner.core._run = fail_run
        monkeypatch.setattr(
            runner.core,
            "_validate_census_receipt",
            lambda: {"path": "census.json", "file_sha256": "1" * 64},
        )
        monkeypatch.setattr(runner.core, "_reserve_output", lambda: 17)
        monkeypatch.setattr(
            runner.core,
            "_publish_json",
            _canonical_publish_spy(published),
        )
        monkeypatch.setattr(
            runner.core.os,
            "close",
            lambda descriptor: closed.append(descriptor),
        )

        assert runner.core.main(["--execute"]) == 3

        assert [name for name, _payload, _binding in published] == [
            "reservation.json",
            "failure.json",
            "failure_access.json",
            "completed.json",
        ]
        payloads = {name: payload for name, payload, _binding in published}
        bindings = {name: binding for name, _payload, binding in published}
        failure = payloads["failure.json"]
        failure_access = payloads["failure_access.json"]
        completion = payloads["completed.json"]
        assert failure["schema"] == f"{runner.SCHEMA}_failure_v1"
        assert failure["updates_completed"] == 7
        assert failure["presentations_completed"] == 112
        assert failure_access["schema"] == f"{runner.SCHEMA}_access_v1"
        assert failure_access["counts_complete"] is True
        assert failure_access["forbidden_all_zero"] is True
        assert completion["schema"] == f"{runner.SCHEMA}_completion_v1"
        assert completion["status"] == "TERMINAL_FAILURE_COMPLETE"
        assert completion["reservation"] == bindings["reservation.json"]
        assert completion["failure"] == bindings["failure.json"]
        assert completion["access"] == bindings["failure_access.json"]
        assert completion["failure_content_sha256"] == failure[
            "content_sha256"
        ]
        assert completion["access_content_sha256"] == failure_access[
            "content_sha256"
        ]
        assert closed == [17, 17]
    finally:
        _restore(core_original, base_original)


def test_source_closure_is_complete_and_requires_external_self_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = (
        "LEWM_CAUSAL_POSTERIOR_REWEIGHTED_TRANSITION_EXPERT_TRAJECTORY_H4_"
        "V1_WRAPPER_"
    )
    monkeypatch.delenv(prefix + "SHA256", raising=False)
    monkeypatch.delenv(prefix + "BYTES", raising=False)
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

    monkeypatch.setattr(runner.base, "_source_binding", fake_binding)
    monkeypatch.setenv(prefix + "SHA256", "a" * 64)
    monkeypatch.setenv(prefix + "BYTES", "123")
    closure = runner._verify_source_closure()
    assert len(closure) == len(calls) == 20
    assert calls[:3] == [
        (Path(runner.__file__).resolve(), "a" * 64, 123),
        (
            runner.SYSTEM_ID_RUNNER_SOURCE,
            runner.SYSTEM_ID_RUNNER_SOURCE_SHA256,
            runner.SYSTEM_ID_RUNNER_SOURCE_BYTES,
        ),
        (
            runner.MODEL_SOURCE,
            runner.MODEL_SOURCE_SHA256,
            runner.MODEL_SOURCE_BYTES,
        ),
    ]
    assert {
        "causal_posterior_reweighted_transition_expert_wrapper",
        "causal_posterior_reweighted_transition_expert_model",
        "action_attributed_system_id_wrapper_dependency",
        "action_attributed_system_id_model_dependency",
        "factorized_wrapper_dependency",
        "encoder_dependency",
    } <= set(closure)


def test_bound_system_runner_hash_and_model_placeholder_are_fail_closed() -> None:
    raw = runner.SYSTEM_ID_RUNNER_SOURCE.read_bytes()
    assert len(raw) == runner.SYSTEM_ID_RUNNER_SOURCE_BYTES
    assert hashlib.sha256(raw).hexdigest() == (
        runner.SYSTEM_ID_RUNNER_SOURCE_SHA256
    )
    if runner.MODEL_SOURCE_SHA256 == "ROOT_FREEZE_MODEL_SHA256_PLACEHOLDER":
        assert runner.MODEL_SOURCE_BYTES == -1
    else:
        model_raw = runner.MODEL_SOURCE.read_bytes()
        assert len(model_raw) == runner.MODEL_SOURCE_BYTES
        assert hashlib.sha256(model_raw).hexdigest() == runner.MODEL_SOURCE_SHA256


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
        runner.base,
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
