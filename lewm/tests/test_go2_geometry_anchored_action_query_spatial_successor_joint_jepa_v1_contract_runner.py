from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = "go2_geometry_anchored_action_query_spatial_successor_joint_jepa_v1"
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"


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


@pytest.mark.parametrize("path", [CONTRACT, RUNNER, LAUNCHER])
def test_entrypoints_import_source_only_under_isolation(path: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(path)!r})
spec = importlib.util.spec_from_file_location("_action_query_isolated", path)
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


def test_runner_and_launcher_dispatch_directly_to_deepest_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fatal_predecessor(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("a predecessor wrapper was called")

    runner = _load("_action_query_direct_runner", RUNNER)
    calls: list[tuple[str, Any]] = []
    for predecessor in (runner._V3, runner._V2):
        monkeypatch.setattr(predecessor, "parse_args", fatal_predecessor)
        monkeypatch.setattr(predecessor, "main", fatal_predecessor)
    monkeypatch.setattr(
        runner._BASE,
        "parse_args",
        lambda argv=None: calls.append(("parse", argv)) or "parsed",
    )
    monkeypatch.setattr(
        runner._BASE,
        "main",
        lambda argv=None: calls.append(("main", argv)) or 17,
    )
    assert runner.parse_args(["parse"]) == "parsed"
    assert runner.main(["main"]) == 17
    assert calls == [("parse", ["parse"]), ("main", ["main"])]

    launcher = _load("_action_query_direct_launcher", LAUNCHER)
    calls.clear()
    for predecessor in (launcher._V3, launcher._V2):
        monkeypatch.setattr(predecessor, "parse_args", fatal_predecessor)
        monkeypatch.setattr(predecessor, "main", fatal_predecessor)
    monkeypatch.setattr(
        launcher._BASE,
        "parse_args",
        lambda argv=None: calls.append(("parse", argv)) or "parsed",
    )
    monkeypatch.setattr(
        launcher._BASE,
        "main",
        lambda argv=None: calls.append(("main", argv)) or 19,
    )
    assert launcher.parse_args(["parse"]) == "parsed"
    assert launcher.main(["main"]) == 19
    assert calls == [("parse", ["parse"]), ("main", ["main"])]


def test_final_hook_identities_and_reviewed_source_witness_have_no_fallback() -> None:
    runner = _load("_action_query_binding_runner", RUNNER)
    runner._assert_final_runner_bindings()
    for name in runner._RUNNER_BINDING_NAMES:
        assert getattr(runner._BASE, name) is getattr(runner, name)
    path = runner.contract.MODEL_TEST_RELATIVE_PATH
    runner._ACTIVE_SOURCE_BINDINGS.clear()
    with pytest.raises(PermissionError, match="witness is absent"):
        runner._reviewed_cpu_source_witness(source_authority_exact=True)
    runner._ACTIVE_SOURCE_BINDINGS[path] = "a" * 64
    witness = runner._reviewed_cpu_source_witness(source_authority_exact=True)
    assert witness["reviewed_model_source_synthetic_witness_path"] == path
    assert witness["reviewed_model_source_synthetic_witness_sha256"] == "a" * 64
    assert witness["reviewed_model_source_synthetic_witness_sha256_non_null"]
    assert witness["runtime_update_zero_synthetic_accelerator_call_count"] == 0
    assert witness["fallback_hard_coded_sha_used"] is False
    source = RUNNER.read_text(encoding="utf-8")
    assert "_ACTIVE_SOURCE_BINDINGS.get(path)" in source
    assert "reviewed_model_source_synthetic_witness_sha256\": digest" in source


@pytest.mark.parametrize("update", [0, 1, 100, 400, 1_000])
def test_exact_registered_work_at_scientific_boundaries(update: int) -> None:
    runner = _load(f"_action_query_work_{update}", RUNNER)
    work = runner._expected_work_at_update(update)
    microbatches = update * 4
    assert work["training_microbatch_count"] == microbatches
    assert work["scheduled_pair_presentations_loaded"] == update * 16
    assert work["joint_combined_objective_evaluation_count"] == microbatches
    assert work["combined_backward_call_count"] == microbatches
    assert work["effective_batch_divided_backward_count"] == microbatches
    assert work["registered_scalar_term_evaluation_count"] == microbatches * 4
    assert work["all_action_predictor_training_forward_count"] == microbatches
    assert work["candidate_row_successor_count"] == microbatches * 36
    assert work["online_encoder_lift_training_forward_count"] == microbatches * 2
    assert work["semantic_head_training_forward_count"] == microbatches * 2
    assert work["target_encoder_lift_training_forward_count"] == microbatches * 2
    assert work["online_optimizer_update_count"] == update
    assert work["target_ema_update_count"] == update
    assert work["predictor_optimizer_update_count"] == update
    assert work["perception_only_update_count"] == 0
    assert work["predictor_only_update_count"] == 0
    assert work["separately_trained_predictor_update_count"] == 0
    assert work["route_probe_call_count"] == (8 if update else 0)


def test_terminal_operation_uses_nonzero_live_partial_work_aliases() -> None:
    runner = _load("_action_query_partial_terminal_operation_runner", RUNNER)
    runner._reset_work()
    runner._WORK.update({
        "scheduled_pair_presentations_loaded": 11,
        "joint_combined_objective_evaluation_count": 7,
        "combined_backward_call_count": 5,
        "training_microbatch_count": 3,
    })
    operation = runner._terminal_operation(
        {"updates": 0, "presentations": 0},
        {
            "pair_presentations_loaded": 0,
            "objective_evaluations": 0,
            "backward_calls": 0,
            "training_microbatch_count": 0,
        },
    )
    assert operation["pair_presentations_loaded"] == 11
    assert operation["objective_evaluations"] == 7
    assert operation["backward_calls"] == 5
    assert operation["training_microbatch_count"] == 3


def _passing_gate_metrics(runner: Any, update: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "presentations": update * 16,
        "source_authority_exact": True,
        "runtime_input_bindings_exact": True,
        "schedule_prefix_exact": True,
        "role_and_mapping_bindings_exact": True,
        "model_parameter_inventory_exact": True,
        "optimizer_inventory_exact": True,
        "rgb_only_causal_call_graph_exact": True,
        "forbidden_input_and_bypass_counts_zero": True,
        "target_requires_grad_false": True,
        "all_forbidden_access_counts_zero": True,
        "all_registered_values_finite": True,
        "state_nonconstant": True,
        "paired_rgb_latents_nonidentical": True,
        "out_of_frustum_semantic_unknown_exact": True,
        "work_accounting_exact": True,
        "target_gradient_tensor_count": 0,
        "target_optimizer_membership_count": 0,
        **runner._expected_work_at_update(update),
    }
    if update == 0:
        metrics.update({
            "A": 1.0,
            "aggregate_raster_nll": 1.0,
            "rough_raster_balanced_accuracy": 0.40,
            "rough_raster_occupied_recall": 0.20,
            "paired_rgb_margin": -0.10,
            "online_target_representation_bitwise_equal": True,
            "predictor_parameter_group_present": True,
            "semantic_objective_formula_exact": True,
            "action_query_objective_formula_exact": True,
            "reviewed_model_source_synthetic_witness_sha256_non_null": True,
            "initial_target_hard_sync_count": 1,
        })
    elif update == 100:
        metrics.update({
            "A": 0.80,
            "aggregate_raster_nll": 0.80,
            "aggregate_raster_balanced_accuracy": 0.61,
            "aggregate_free_recall": 0.56,
            "aggregate_occupied_recall": 0.31,
            "free_occupied_recall_gap": 0.25,
            "rough_raster_balanced_accuracy": 0.50,
            "rough_raster_occupied_recall": 0.30,
            "paired_rgb_margin": 0.05,
            "paired_rgb_scene_wins": 6,
            "action_raw_nll": math.log(9.0) - 0.01,
            "action_macro_balanced_accuracy": 1.0 / 9.0 + 0.01,
            "hardest_wrong_positive_margin_family_count": 1,
            "correct_next_deranged_raw_nll": math.log(2.0) - 0.01,
            "correct_next_deranged_strict_win_rate": 0.51,
            "encoder_parameter_displaced": True,
            "all_predictor_components_displaced": True,
            "target_effective_rank": 1.0,
            "target_channel_variance": 1.0,
            "target_spatial_diversity": 1.0,
        })
    elif update == 400:
        metrics.update({
            "A": 0.50,
            "aggregate_raster_nll": 0.40,
            "aggregate_raster_balanced_accuracy": 0.81,
            "aggregate_occupied_recall": 0.61,
            "rough_raster_balanced_accuracy": 0.78,
            "rough_raster_occupied_recall": 0.56,
            "paired_rgb_margin": 0.10,
            "paired_rgb_scene_wins": 6,
            "action_raw_nll": 0.98 * math.log(9.0) - 0.01,
            "action_macro_balanced_accuracy": 0.18,
            "hardest_wrong_positive_margin_family_count": 3,
            "correct_next_deranged_strict_win_rate": 0.70,
            "target_effective_rank": 0.80,
            "target_channel_variance": 0.80,
            "target_spatial_diversity": 0.80,
        })
    else:
        metrics.update({
            "A": 0.49,
            "aggregate_raster_nll": 0.37,
            "aggregate_raster_balanced_accuracy": 0.81,
            "aggregate_unknown_recall": 0.80,
            "aggregate_free_recall": 0.75,
            "aggregate_occupied_recall": 0.70,
            "free_occupied_recall_gap": 0.05,
            "rough_raster_balanced_accuracy": 0.772,
            "rough_raster_occupied_recall": 0.65,
            "paired_rgb_margin": 0.10,
            "paired_rgb_scene_wins": 8,
            "target_effective_rank": 0.70,
            "target_channel_variance": 0.70,
            "target_spatial_diversity": 0.70,
            "action_raw_nll": 0.95 * math.log(9.0) - 0.01,
            "action_macro_balanced_accuracy": 2.0 / 9.0 + 0.01,
            "hardest_wrong_positive_margin_family_count": 6,
            "mean_wrong_action_energy": 1.01,
            "mean_executed_action_energy": 1.00,
            "mean_non_hold_hold_action_energy": 1.01,
            "mean_non_hold_executed_action_energy": 1.00,
            "correct_next_deranged_raw_nll": 0.95 * math.log(2.0) - 0.01,
            "correct_next_deranged_strict_win_rate": 0.70,
            "correct_next_positive_margin_family_count": 6,
            "mean_successor_unscaled_local_energy": 0.90,
            "mean_persistence_unscaled_local_energy": 1.00,
            "successor_over_persistence_strict_win_family_count": 6,
            "autoregressive_rollout_step_count": 8,
            "autoregressive_rollout_action_count": 9,
            "autoregressive_rollout_all_intermediate_and_final_finite": True,
            "autoregressive_rollout_future_rgb_input_count": 0,
            "autoregressive_rollout_objective_backward_step_ema_count": 0,
            "autoregressive_rollout_renormalization_count": 0,
            "encoder_parameter_displaced": True,
            "lift_parameter_displaced": True,
            "all_predictor_components_displaced": True,
        })
    return metrics


def test_all_scientific_gate_boundaries_pass_and_strict_ties_fail() -> None:
    runner = _load("_action_query_gate_runner", RUNNER)
    contract = runner.contract
    metrics = {
        update: _passing_gate_metrics(runner, update)
        for update in (0, 100, 400, 1_000)
    }
    prior: dict[int, dict[str, Any]] = {}
    for update in (0, 100, 400, 1_000):
        gate = contract.evaluate_gate(update, metrics[update], prior_metrics=prior)
        assert gate["passed"], gate["conjuncts"]
        prior[update] = metrics[update]
    assert gate["control"] == contract.CONTROL_PASS

    strict_ties = (
        (100, "action_raw_nll", math.log(9.0)),
        (100, "action_macro_balanced_accuracy", 1.0 / 9.0),
        (100, "correct_next_deranged_strict_win_rate", 0.50),
        (400, "action_raw_nll", 0.98 * math.log(9.0)),
        (1_000, "action_raw_nll", 0.95 * math.log(9.0)),
        (1_000, "action_macro_balanced_accuracy", 2.0 / 9.0),
        (1_000, "mean_wrong_action_energy", 1.00),
    )
    for update, name, value in strict_ties:
        tied = dict(metrics[update])
        tied[name] = value
        gate = contract.evaluate_gate(update, tied, prior_metrics=prior)
        assert not gate["passed"]
        assert gate["control"] == contract.GATE_CONTROLS[update][0]


def test_update_400_zero_anti_collapse_statistics_are_scientific_failure() -> None:
    runner = _load("_action_query_zero_collapse_gate_runner", RUNNER)
    contract = runner.contract
    zero = _passing_gate_metrics(runner, 0)
    hundred = _passing_gate_metrics(runner, 100)
    metrics = _passing_gate_metrics(runner, 400)
    metrics.update({
        "target_effective_rank": 0.0,
        "target_channel_variance": 0.0,
        "target_spatial_diversity": 0.0,
    })

    gate = contract.evaluate_gate(
        400,
        metrics,
        prior_metrics={0: zero, 100: hundred},
    )

    assert gate["passed"] is False
    assert gate["control"] == contract.GATE_CONTROLS[400][0]
    assert gate["scientific_gate_evidence"] is True
    for name in (
        "target_effective_rank_strictly_positive",
        "target_channel_variance_strictly_positive",
        "target_spatial_diversity_strictly_positive",
    ):
        assert gate["conjuncts"][name] is False


def test_metrics_use_all_fixed_mapping_vocabulary_without_same_action_aliases() -> None:
    runner = _load("_action_query_metric_vocabulary_runner", RUNNER)
    source = inspect.getsource(runner._evaluate_observation_body)

    for current_name in (
        "correct_next_deranged_raw_nll",
        "correct_next_deranged_strict_win_rate",
        "correct_next_positive_margin_family_count",
    ):
        assert f'"{current_name}"' in source
    for obsolete_name in (
        "same_action_target_nll",
        "same_action_target_strict_win_rate",
    ):
        assert f'"{obsolete_name}"' not in source


def test_target_collapse_statistics_measure_normalized_objective_target() -> None:
    runner = _load("_action_query_normalized_target_statistics_runner", RUNNER)
    parameters = tuple(
        inspect.signature(runner._target_statistics_for_gate).parameters
    )
    assert parameters == (
        "runtime",
        "model_api",
        "model",
        "loader",
        "identities",
        "device",
        "update",
    )

    statistics_source = inspect.getsource(runner._target_statistics_for_gate)
    normalization = "latent = model_api.normalize_latent_per_cell_v1("
    assert normalization in statistics_source
    assert "model.encode_target(images)" in statistics_source
    assert statistics_source.index(normalization) < statistics_source.index(
        "flat = latent.permute"
    )
    assert statistics_source.index(normalization) < statistics_source.index(
        "horizontal = latent["
    )

    observation_source = inspect.getsource(runner._evaluate_observation_body)
    assert "_target_statistics_for_gate(\n                runtime,\n                model_api," in (
        observation_source
    )


def test_exact_observation_work_at_all_scientific_boundaries() -> None:
    runner = _load("_action_query_observation_work_runner", RUNNER)
    update_zero = {
        "observation_pair_microbatch_count": 124,
        "observation_endpoint_microbatch_count": 231,
        "observation_pair_rows_loaded": 495,
        "observation_endpoint_rows_loaded": 924,
        "observation_online_encoder_lift_forward_count": 603,
        "observation_target_encoder_lift_forward_count": 0,
        "observation_semantic_head_forward_count": 603,
        "observation_all_action_predictor_forward_count": 0,
        "observation_all_action_candidate_successor_count": 0,
        "observation_selected_action_predictor_forward_count": 0,
        "observation_selected_action_successor_count": 0,
        "observation_semantic_term_evaluation_count": 124,
        "observation_action_query_objective_evaluation_count": 0,
        "observation_semantic_nll_helper_call_count": 620,
        "observation_action_score_helper_call_count": 0,
        "observation_target_score_helper_call_count": 0,
        "observation_action_ce_reporting_call_count": 0,
        "observation_target_ce_reporting_call_count": 0,
        "observation_confusion_metric_helper_call_count": 2,
        "observation_target_statistics_pass_count": 0,
        "observation_backward_call_count": 0,
        "observation_optimizer_update_count": 0,
        "observation_ema_update_count": 0,
        "observation_predictor_forward_count": 0,
        "observation_objective_evaluation_count": 124,
        "observation_reporting_helper_call_count": 622,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
    }
    learned = {
        **update_zero,
        "observation_endpoint_microbatch_count": 462,
        "observation_endpoint_rows_loaded": 1_848,
        "observation_target_encoder_lift_forward_count": 479,
        "observation_all_action_predictor_forward_count": 124,
        "observation_all_action_candidate_successor_count": 4_455,
        "observation_action_query_objective_evaluation_count": 124,
        "observation_action_score_helper_call_count": 124,
        "observation_target_score_helper_call_count": 124,
        "observation_action_ce_reporting_call_count": 124,
        "observation_target_ce_reporting_call_count": 124,
        "observation_target_statistics_pass_count": 1,
        "observation_predictor_forward_count": 124,
        "observation_objective_evaluation_count": 248,
        "observation_reporting_helper_call_count": 1_118,
    }
    terminal = {
        **learned,
        "observation_selected_action_predictor_forward_count": 868,
        "observation_selected_action_successor_count": 31_185,
        "observation_predictor_forward_count": 992,
    }

    assert runner._expected_observation_work(0) == update_zero
    assert runner._expected_observation_work(100) == learned
    assert runner._expected_observation_work(400) == learned
    assert runner._expected_observation_work(1_000) == terminal


def test_update_zero_requires_each_observation_preservation_boolean() -> None:
    runner = _load("_action_query_update_zero_preservation_runner", RUNNER)
    source = inspect.getsource(runner._train_probe)
    preservation_names = (
        "model_state_hash_unchanged",
        "optimizer_state_hash_unchanged",
        "rng_state_unchanged",
        "training_work_counters_unchanged",
        "model_training_mode_restored",
        "schedule_not_advanced",
    )
    for name in preservation_names:
        assert f'"{name}"' in source
    preservation_state = {
        "committed_update": 0,
        **{name: True for name in preservation_names},
    }
    assert preservation_state["committed_update"] == 0
    assert all(bool(preservation_state[name]) for name in preservation_names)
    preservation_block = source[
        source.index("preservation_names = ("):
        source.index("if not all(bool(state_receipt[name])", source.index(
            "preservation_names = ("
        ))
    ]
    assert '"committed_update"' not in preservation_block
    explicit_check = (
        "if not all(bool(state_receipt[name]) for name in preservation_names):"
    )
    assert explicit_check in source
    assert source.index(explicit_check) < source.index(
        "gate = contract.evaluate_gate(update, metrics, prior_metrics=prior_metrics)"
    )
    assert "_metrics_zero, gate_zero = observe(0)" in source


def test_cumulative_observation_receipt_includes_completed_and_partial_work() -> None:
    runner = _load("_action_query_cumulative_observation_runner", RUNNER)
    runner._OBSERVATION_HISTORY.clear()
    runner._OBSERVATION_LIVE.clear()

    expected_zero = runner._expected_observation_work(0)
    runner._begin_observation_accounting(0)
    runner._OBSERVATION_LIVE.update(expected_zero)
    completed = runner._complete_observation_accounting(0)
    assert completed["observation_status"] == "complete"
    assert runner._cumulative_observation_receipt() == {
        "completed_observation_count": 1,
        "completed_observation_updates": [0],
        "included_partial_observation": False,
        "totals": expected_zero,
    }

    runner._begin_observation_accounting(100)
    runner._OBSERVATION_LIVE.update({
        "observation_pair_microbatch_count": 3,
        "observation_pair_rows_loaded": 12,
        "observation_semantic_term_evaluation_count": 3,
        "observation_action_query_objective_evaluation_count": 2,
        "observation_semantic_nll_helper_call_count": 15,
        "observation_action_score_helper_call_count": 2,
        "observation_target_score_helper_call_count": 2,
        "observation_action_ce_reporting_call_count": 2,
        "observation_target_ce_reporting_call_count": 2,
        "last_started_call": "predict_all_actions",
        "last_successful_call": "encode_target",
    })
    runner._mark_observation_failure(RuntimeError("synthetic partial observation"))
    partial = runner._observation_live_receipt()
    assert partial is not None
    assert partial["observation_status"] == "failed"
    assert partial["failure_type"] == "RuntimeError"
    assert runner.contract.is_sha256(partial["failure_message_sha256"])
    assert partial["observation_objective_evaluation_count"] == 5
    assert partial["observation_reporting_helper_call_count"] == 23

    expected_totals = {
        name: expected_zero[name] + int(partial[name])
        for name in expected_zero
    }
    assert runner._cumulative_observation_receipt() == {
        "completed_observation_count": 1,
        "completed_observation_updates": [0],
        "included_partial_observation": True,
        "totals": expected_totals,
    }


def test_four_microbatches_use_eight_route_probes_and_only_divisor_four() -> None:
    runner = _load("_action_query_arithmetic_runner", RUNNER)
    runner._reset_work()
    divisions: list[int] = []
    backwards: list[bool] = []
    grad_calls: list[dict[str, Any]] = []

    class Scalar:
        def __truediv__(self, divisor: int) -> "Scalar":
            divisions.append(divisor)
            return self

        def backward(self) -> None:
            backwards.append(True)

    class Autograd:
        @staticmethod
        def grad(
            value: Any,
            parameters: Any,
            *,
            retain_graph: bool,
            allow_unused: bool,
        ) -> tuple[str]:
            grad_calls.append({
                "value": value,
                "parameters": parameters,
                "retain_graph": retain_graph,
                "allow_unused": allow_unused,
            })
            return ("gradient",)

    fake_torch = SimpleNamespace(autograd=Autograd())
    scalar = Scalar()
    for _ in range(4):
        semantic, dynamics = runner._route_probes_for_microbatch(
            fake_torch, scalar, scalar, ("parameter",)
        )
        assert semantic == dynamics == ("gradient",)
        runner._combined_backward(scalar)
    assert len(grad_calls) == 8
    assert all(row["retain_graph"] and not row["allow_unused"] for row in grad_calls)
    assert len(backwards) == 4
    assert divisions == [4, 4, 4] * 4
    assert runner._WORK["semantic_route_probe_call_count"] == 4
    assert runner._WORK["dynamics_route_probe_call_count"] == 4
    assert runner._WORK["route_probe_call_count"] == 8
    assert runner._WORK["combined_backward_call_count"] == 4
    source = inspect.getsource(runner._train_probe)
    assert "optimizer.zero_grad(set_to_none=True)" in source
    assert "if update == 1:" in source
    assert "ratio_abort_applied\": False" in source
    assert "phase_joint" not in source
    assert '"perception_warmup_update_count": 0' in source


def test_warning_receipt_survives_raising_operation_and_execute_wrapper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_action_query_warning_failure_runner", RUNNER)

    class FakeTorch:
        def __init__(self) -> None:
            self.enabled = False
            self.warn_only = False
            self.backends = SimpleNamespace(
                cudnn=SimpleNamespace(benchmark=True, deterministic=False)
            )

        def are_deterministic_algorithms_enabled(self) -> bool:
            return self.enabled

        def is_deterministic_algorithms_warn_only_enabled(self) -> bool:
            return self.warn_only

        def use_deterministic_algorithms(
            self, enabled: bool, *, warn_only: bool
        ) -> None:
            self.enabled = enabled
            self.warn_only = warn_only

    fake_torch = FakeTorch()
    failure = RuntimeError("synthetic scientific operation failure")

    def raising_operation() -> None:
        import warnings

        warnings.warn("synthetic deterministic warning", UserWarning)
        raise failure

    with pytest.raises(RuntimeError) as caught:
        runner._run_deterministic(
            SimpleNamespace(torch=fake_torch), raising_operation
        )
    assert caught.value is failure
    receipt = failure.determinism_warning_receipt
    assert receipt["warning_count"] == 1
    assert receipt["unexpected_warning_count"] == 1
    assert receipt[
        "scientific_callable_returned_before_warning_finalization"
    ] is False
    assert fake_torch.enabled is False
    assert fake_torch.warn_only is False
    assert fake_torch.backends.cudnn.benchmark is True
    assert fake_torch.backends.cudnn.deterministic is False

    monkeypatch.setattr(
        runner,
        "_BASE_EXECUTE",
        lambda **_kwargs: (_ for _ in ()).throw(failure),
    )
    progress: dict[str, Any] = {}
    with pytest.raises(RuntimeError) as wrapped:
        runner._execute(
            sources={},
            authorization={},
            reservation={},
            reservation_raw=b"{}\n",
            output_root=tmp_path,
            progress=progress,
        )
    assert wrapped.value is failure
    assert progress["_determinism"] == receipt


def test_authority_is_checked_before_absent_root_and_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    launcher = _load("_action_query_authority_launcher", LAUNCHER)
    order: list[str] = []
    digest = "b" * 64
    monkeypatch.setattr(
        launcher._BASE,
        "parse_args",
        lambda _argv=None: argparse.Namespace(
            review_sha256=digest, authorization_sha256=digest
        ),
    )
    monkeypatch.setattr(
        launcher._BASE,
        "_validate_authority",
        lambda **_kwargs: order.append("authority") or {},
    )
    monkeypatch.setattr(launcher._BASE, "OUTPUT_ROOT", tmp_path / "absent")
    monkeypatch.setattr(launcher, "_assert_final_launcher_bindings", lambda: None)

    class ExecutionReached(RuntimeError):
        pass

    def execution(_args: Any) -> None:
        order.append("execution")
        raise ExecutionReached("synthetic execution boundary")

    monkeypatch.setattr(launcher._BASE, "_exec_runtime", execution)
    with pytest.raises(ExecutionReached, match="execution boundary"):
        launcher.main([])
    assert order == ["authority", "execution"]
    assert not (tmp_path / "absent").exists()


def test_operational_failure_receipt_contains_complete_partial_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_action_query_failure_runner", RUNNER)
    (tmp_path / "reservation.json").write_bytes(b"{}\n")
    published: dict[str, dict[str, Any]] = {}

    def publish(path: Path, core: Any) -> tuple[dict[str, Any], bytes]:
        value = dict(core)
        raw = b"{}\n"
        path.write_bytes(raw)
        published[path.name] = value
        return value, raw

    monkeypatch.setattr(runner._BASE, "_publish_json", publish)
    monkeypatch.setattr(
        runner._BASE,
        "_binding",
        lambda path, _value, raw: {"path": path, "byte_count": len(raw)},
    )
    monkeypatch.setattr(runner, "_BASE_SEAL", lambda _path: {"sealed": True})
    model_test = runner.contract.MODEL_TEST_RELATIVE_PATH
    runner._ACTIVE_SOURCE_BINDINGS.clear()
    runner._ACTIVE_SOURCE_BINDINGS[model_test] = "c" * 64
    runner._OBSERVATION_HISTORY.clear()
    runner._OBSERVATION_LIVE.clear()
    runner._begin_observation_accounting(100)
    runner._OBSERVATION_LIVE.update({
        "observation_pair_microbatch_count": 1,
        "observation_pair_rows_loaded": 4,
        "observation_semantic_term_evaluation_count": 1,
        "observation_semantic_nll_helper_call_count": 2,
        "last_started_call": "loader.batch:pair_observation_update_100",
        "last_successful_call": "encode_online",
    })
    runner._mark_observation_failure(RuntimeError("synthetic observation failure"))
    partial = {
        "updates": 0,
        "presentations": 0,
        "terminal_gate": None,
        "integrity": {},
        **runner._expected_work_at_update(0),
    }
    last_committed = {
        "update": 0,
        "model_state_sha256": "1" * 64,
        "optimizer_state_sha256": "2" * 64,
        "rng_state_sha256": "3" * 64,
        "audited_immediately_before_non_mutating_observation": True,
    }
    terminal_capture = {
        "available": False,
        "reason": "optimizer_or_ema_commit_was_in_progress",
        "last_audited_committed_state_hashes": last_committed,
    }
    progress = {
        "stage": "synthetic_failure",
        "_probe_failure_state": partial,
        "_observations": [],
        "_checkpoint_bindings": [],
        "_last_committed_state_hashes": last_committed,
        "_terminal_state_hash_capture": terminal_capture,
    }
    runner._terminal_failure(
        tmp_path,
        {"content_sha256": "d" * 64},
        b"{}\n",
        progress,
        RuntimeError("synthetic"),
    )
    assert set(published) == {"failure.json", "completed.json"}
    failure = published["failure.json"]
    assert failure["complete_failure_receipt"] is True
    assert failure["retry_resume_repair_or_replacement_authorized"] is False
    assert failure["checkpoint_read_count_after_write"] == 0
    assert failure["training_trace_read_count_after_write"] == 0
    assert failure["source_bindings"][model_test] == "c" * 64
    assert failure["observation_work"]["observation_update"] == 100
    assert failure["observation_work"]["observation_status"] == "failed"
    assert failure["observation_work"]["observation_pair_microbatch_count"] == 1
    assert failure["observation_work"]["observation_pair_rows_loaded"] == 4
    assert failure["cumulative_observation_work"][
        "included_partial_observation"
    ] is True
    assert failure["cumulative_observation_work"]["totals"][
        "observation_pair_microbatch_count"
    ] == 1
    assert failure["last_committed_state_hashes"] == last_committed
    assert failure["terminal_state_hash_capture"] == terminal_capture
    assert failure["reviewed_cpu_source_witness"][
        "reviewed_model_source_synthetic_witness_sha256"
    ] == "c" * 64
    for name in runner._expected_work_at_update(0):
        assert name in failure["operation"]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "completed.json", "failure.json", "reservation.json"
    ]
