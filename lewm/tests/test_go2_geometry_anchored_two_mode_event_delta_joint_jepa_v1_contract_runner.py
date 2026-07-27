from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "lewm/benchmarks/"
    "go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
MODEL = (
    ROOT / "lewm/models/"
    "geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
RUNNER = (
    ROOT / "scripts/"
    "run_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
LAUNCHER = (
    ROOT / "scripts/"
    "launch_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1.py"
)
CHECKER = (
    ROOT / "scripts/"
    "check_go2_geometry_anchored_two_mode_event_delta_joint_jepa_v1_source_closure.py"
)


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


@pytest.mark.parametrize("path", [RUNNER, LAUNCHER, CHECKER])
def test_entrypoint_import_is_source_only_and_rebound(path: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(path)!r})
spec = importlib.util.spec_from_file_location("_event_delta_source_import", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
if hasattr(module, "_joint_terms"):
    assert module._BASE.contract is module.contract
    assert module._BASE._joint_terms is module._joint_terms
    assert module._BASE._evaluate_observation is module._evaluate_observation
    assert module._BASE._train_probe is module._train_probe
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


def test_contract_and_recursive_closure_are_rigid_plus_seven() -> None:
    contract = _load("_event_delta_contract_identity", CONTRACT)
    checker = _load("_event_delta_checker_identity", CHECKER)
    assert contract.FROZEN_RIGID_SOURCE_COUNT == 91
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 7
    assert len(contract.SOURCE_PATHS) == 98
    assert set(contract.SOURCE_PATHS) == {
        *contract.REUSED_SOURCE_PATHS,
        *contract.ADDITIVE_SOURCE_PATHS,
    }
    assert len(checker.EXPECTED_SOURCE_PATHS) == 98
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 98
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
    assert manifest["entrypoints"] == list(contract.SOURCE_MANIFEST_ENTRYPOINTS)
    assert manifest["forced_dynamic_sources"] == list(contract.SOURCE_PATHS)


def test_actual_model_helper_accepts_plain_delta_for_all_action_prediction() -> None:
    model_api = _load("_event_delta_model_helper", MODEL)
    runner = _load("_event_delta_runner_helper", RUNNER)
    runner._FROZEN_REFERENCES.clear()
    runner._FROZEN_REFERENCES["T400"] = 0.25
    delta = torch.linspace(
        -0.3, 0.3, 64 * 64 * 64, dtype=torch.float32
    ).reshape(1, 64, 64, 64)
    prediction = model_api.EventDeltaPrediction(
        torch.zeros(1, 9, 64, 64, 64, dtype=torch.float32),
        torch.linspace(
            -2.0, 2.0, 9 * 64 * 64, dtype=torch.float32
        ).reshape(1, 9, 1, 64, 64),
    )
    zero, learned, mixed = runner._event_cells(model_api, delta, prediction)
    assert zero.shape == learned.shape == mixed.shape == (1, 9, 64, 64)
    assert torch.allclose(mixed, zero, atol=2e-6, rtol=2e-6)
    weight = torch.full((1, 64, 64), 0.25, dtype=torch.float32)
    reduced = runner._balanced(mixed, weight)
    assert reduced.changed.shape == reduced.static.shape == (1, 9)
    assert torch.allclose(
        reduced.balanced,
        mixed.mean(dim=(-2, -1)),
        atol=2e-6,
        rtol=2e-6,
    )


class _SyntheticEventModel(torch.nn.Module):
    def __init__(self, model_api: Any) -> None:
        super().__init__()
        self.model_api = model_api
        self.encoder_gain = torch.nn.Parameter(torch.tensor(1.0))
        self.mean_gain = torch.nn.Parameter(torch.tensor(0.02))
        self.logit_gain = torch.nn.Parameter(torch.tensor(0.03))
        self.fixed_context_requires_grad: bool | None = None

    def _latent(self, value: torch.Tensor) -> torch.Tensor:
        batch = value.shape[0]
        tag = value[:, 0, 0, 0].reshape(batch, 1, 1, 1)
        channel = torch.linspace(
            -1.0, 1.0, 64, dtype=torch.float32, device=value.device
        ).reshape(1, 64, 1, 1)
        return self.encoder_gain * (
            channel + tag * channel.square()
        ).expand(batch, 64, 64, 64)

    def encode_target(self, value: torch.Tensor) -> torch.Tensor:
        return self._latent(value)

    def encode_online(self, value: torch.Tensor) -> torch.Tensor:
        result = self._latent(value)
        if float(value[0, 0, 0, 0]) == 2.0:
            self.fixed_context_requires_grad = result.requires_grad
        return result

    def predict_all_action_event_deltas(
        self, x: torch.Tensor
    ) -> Any:
        action = torch.arange(9, device=x.device, dtype=x.dtype).reshape(1, 9, 1, 1, 1)
        mu = self.mean_gain * x[:, None] + 0.001 * action
        logit = self.logit_gain * x[:, None, :1]
        return self.model_api.EventDeltaPrediction(
            mu.expand(-1, 9, -1, -1, -1),
            logit.expand(-1, 9, -1, -1, -1),
        )

    def predict_event_delta(
        self, x: torch.Tensor, action_one_hot: torch.Tensor
    ) -> Any:
        action = (
            action_one_hot
            * torch.arange(9, device=x.device, dtype=x.dtype)[None]
        ).sum(dim=1).reshape(-1, 1, 1, 1)
        return self.model_api.EventDeltaPrediction(
            self.mean_gain * x + 0.001 * action,
            (self.logit_gain * x[:, :1]).expand(-1, 1, -1, -1),
        )


def test_joint_terms_use_target_delta_and_autograd_enabled_detached_context() -> None:
    model_api = _load("_event_delta_model_joint", MODEL)
    runner = _load("_event_delta_runner_joint", RUNNER)
    runner._reset_event_runtime_state()
    runner._FROZEN_REFERENCES.update({"T400": 0.25, "B400": 0.4})
    model = _SyntheticEventModel(model_api)
    model.eval()  # Observation mode avoids training counters but preserves autograd.
    current_rgb = torch.zeros(1, 1, 1, 1)
    next_rgb = torch.ones(1, 1, 1, 1)
    fixed_rgb = torch.full((1, 1, 1, 1), 2.0)
    current_latent = model.encode_online(current_rgb)
    batch = {
        "current_rgb": current_rgb,
        "next_rgb": next_rgb,
        "fixed_negative_rgb": fixed_rgb,
        "action_indices": torch.tensor([3], dtype=torch.long),
    }
    runtime = type("Runtime", (), {"torch": torch})()
    joint = runner._joint_terms(
        runtime,
        model_api,
        model,
        batch,
        current_latent,
        persistence_baseline=0.4,
    )
    assert torch.allclose(
        joint["D"],
        joint["P_event"] + joint["R_action"]
        + joint["C_target"] + joint["C_context"],
    )
    assert joint["energies"].shape == (1, 9)
    assert joint["prediction_context"].mu_event.shape == (1, 64, 64, 64)
    joint["D"].backward()
    assert model.fixed_context_requires_grad is False
    assert model.encoder_gain.grad is not None and bool(model.encoder_gain.grad != 0)
    assert model.mean_gain.grad is not None and bool(model.mean_gain.grad != 0)
    assert model.logit_gain.grad is not None and bool(model.logit_gain.grad != 0)
    assert runner._EVENT_ACCOUNTING["training_microbatch_count"] == 0


def test_update_401_enrichment_reaches_contract_with_all_event_routes() -> None:
    runner = _load("_event_delta_runner_phase", RUNNER)
    runner._reset_event_runtime_state()
    for key in (
        "action_embedding_dynamics_gradient_update_count",
        "predictor_trunk_dynamics_gradient_update_count",
        "event_mean_head_dynamics_gradient_update_count",
        "event_logit_head_dynamics_gradient_update_count",
    ):
        runner._EVENT_ACCOUNTING[key] = 1
    receipt = runner._evaluate_update_401_phase_switch({
        "optimizer_identity_unchanged": True,
        "optimizer_parameter_group_membership_unchanged": True,
        "joint_objective_formula_exact": True,
        "online_representation_gradient_finite_nonzero": True,
        "predictor_gradient_finite_nonzero": True,
        "target_gradients_absent": True,
        "shared_gradient_contribution_gate_passed": True,
        "online_optimizer_update_count": 401,
        "target_ema_update_count": 401,
        "predictor_optimizer_update_count": 1,
        "joint_optimizer_update_count": 1,
    })
    assert receipt["passed"] is True
    assert all(receipt["conjuncts"].values())
    assert receipt["conjuncts"]["event_logit_head_gradient_finite_nonzero"] is True


def _u0_contract_metrics(contract: Any) -> dict[str, Any]:
    return {
        **{field: True for field in contract._v3.COMMON_GATE_BOOLEAN_FIELDS},
        **contract.OBSERVATION_ACCOUNTING_EXPECTATIONS[0],
        "presentations": 0,
        "A": 1.0,
        "aggregate_raster_nll": 0.8,
        "aggregate_raster_balanced_accuracy": 0.34,
        "aggregate_unknown_recall": 0.8,
        "aggregate_free_recall": 0.1,
        "aggregate_occupied_recall": 0.1,
        "rough_raster_balanced_accuracy": 0.34,
        "rough_raster_occupied_recall": 0.1,
        "paired_rgb_margin": 0.1,
        "paired_rgb_scene_wins": 5,
        "online_optimizer_update_count": 0,
        "target_ema_update_count": 0,
        "predictor_forward_count": 0,
        "predictor_objective_count": 0,
        "predictor_backward_count": 0,
        "predictor_optimizer_update_count": 0,
        "joint_optimizer_update_count": 0,
        "shared_gradient_ratio_evaluation_count": 0,
        "online_target_representation_bitwise_equal": True,
        "predictor_parameter_group_present": True,
        "semantic_objective_formula_exact": True,
        "latent_prediction_objective_formula_exact": True,
        "action_objective_formula_exact": True,
        "same_action_contrast_formula_exact": True,
        "deformable_lift_synthetic_mechanism_exact": True,
        "paired_correct_wrong_rgb_latents_finite_nonidentical": True,
        "initial_target_hard_sync_count": 1,
        **{
            field: True
            for field in (
                "event_tensor_shapes_exact",
                "fixed_mode_identities_exact",
                "output_parameter_action_symmetry_exact",
                "synthetic_positive_temperature_stable_energy_exact",
                "zero_mean_persistence_identity_exact",
                "synthetic_action_nll_log9_exact",
                "synthetic_action_macro_balanced_accuracy_one_ninth_exact",
                "event_initialization_and_gradient_witness_exact",
                "action_embedding_exact_zero_at_update_zero",
                "event_prior_bitwise_float32_half_at_update_zero",
                "mean_and_logit_head_initialization_and_rng_order_exact",
                "online_encoder_lift_and_each_predictor_submodule_gradient_witness_exact",
                "reviewed_model_source_synthetic_encoder_lift_and_each_residual_conv_witness_bound",
                "no_scale_inverse_pair_posterior_transport_or_future_bypass",
            )
        },
    }


def test_update_zero_runner_metric_names_pass_contract() -> None:
    contract = _load("_event_delta_contract_u0", CONTRACT)
    result = contract.evaluate_gate(0, _u0_contract_metrics(contract))
    assert result["passed"] is True
    assert all(result["conjuncts"].values())


def test_update_zero_witness_is_hash_bound_and_executes_no_runtime_tensor_work() -> None:
    runner = _load("_event_delta_runner_u0_binding", RUNNER)
    runner._ACTIVE_SOURCE_BINDINGS.clear()
    runner._ACTIVE_SOURCE_BINDINGS[
        runner.contract.MODEL_TEST_RELATIVE_PATH
    ] = "a" * 64
    witness = runner._reviewed_cpu_source_witness(
        source_authority_exact=True
    )
    assert witness["event_initialization_and_gradient_witness_exact"] is True
    assert witness["action_embedding_exact_zero_at_update_zero"] is True
    assert witness["event_prior_bitwise_float32_half_at_update_zero"] is True
    assert witness["runtime_update_zero_synthetic_accelerator_call_count"] == 0
    assert witness["reviewed_model_source_synthetic_witness_sha256"] == "a" * 64
    runner._ACTIVE_SOURCE_BINDINGS.clear()
    assert runner._reviewed_cpu_source_witness(
        source_authority_exact=True
    )["event_tensor_shapes_exact"] is False


def test_update_zero_gate_failure_return_gets_complete_normal_probe_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load("_event_delta_runner_u0_probe_shape", RUNNER)
    probe = {
        "status": "SYNTHETIC_U0_FAIL",
        "terminal_gate": {"passed": False},
        "observations": [],
        "checkpoints": [],
        "training_trace": None,
        "updates": 0,
        "presentations": 0,
        "objective_evaluations": 0,
        "backward_calls": 0,
        "integrity": {},
    }
    monkeypatch.setattr(
        runner, "_BASE_TRAIN_PROBE", lambda *_args, **_kwargs: (object(), probe)
    )
    _model, observed = runner._train_probe(progress={})
    for key in (
        "phase_switch_receipt", "predictor_forward_count",
        "predictor_objective_count", "predictor_backward_count",
        "predictor_optimizer_updates", "joint_optimizer_updates",
        "shared_gradient_gate_pass_count",
    ):
        assert key in observed


def test_mid_update_failure_accounting_uses_completed_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load("_event_delta_runner_failure_accounting", RUNNER)
    progress: dict[str, Any] = {}

    def fail_after_seven(*_args: Any, **kwargs: Any) -> Any:
        runner._EVENT_ACCOUNTING["scheduled_pair_presentations_loaded"] = 28
        runner._EVENT_ACCOUNTING["semantic_term_evaluation_count"] = 7
        runner._EVENT_ACCOUNTING["training_microbatch_count"] = 3
        runner._EVENT_ACCOUNTING["online_encoder_lift_training_forward_count"] = 17
        runner._EVENT_ACCOUNTING["semantic_head_training_forward_count"] = 14
        runner._EVENT_ACCOUNTING["target_encoder_lift_training_forward_count"] = 9
        for key in (
            "event_persistence_term_evaluation_count",
            "action_term_evaluation_count",
            "target_term_evaluation_count",
            "context_term_evaluation_count",
            "all_action_predictor_training_forward_count",
            "context_swap_predictor_training_forward_count",
        ):
            runner._EVENT_ACCOUNTING[key] = 3
        kwargs["progress"]["_probe_failure_state"] = {
            "updates": 1,
            "objective_evaluations": 7,
            "backward_calls": 7,
            "integrity": {},
        }
        raise RuntimeError("synthetic mid-update failure")

    monkeypatch.setattr(runner, "_BASE_TRAIN_PROBE", fail_after_seven)
    with pytest.raises(RuntimeError, match="synthetic mid-update failure"):
        runner._train_probe(progress=progress)
    state = progress["_probe_failure_state"]
    assert state["combined_objective_evaluation_count"] == 7
    assert state["backward_call_count"] == 7
    assert state["semantic_term_evaluation_count"] == 7
    assert state["registered_scalar_term_evaluation_count"] == 19
    assert state["presentations"] == 28
    assert state["pair_presentations_loaded"] == 28
    assert state["online_encoder_lift_training_forward_count"] == 17
    assert state["target_encoder_lift_training_forward_count"] == 9
    assert state["integrity"]["combined_objective_evaluation_count"] == 7


def test_semantic_and_joint_failure_counters_advance_at_successful_boundaries() -> None:
    model_api = _load("_event_delta_model_boundary", MODEL)
    runner = _load("_event_delta_runner_boundary", RUNNER)
    runner._reset_event_runtime_state()

    class SemanticFailure(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def encode_online(self, _value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic next encode failure")
            return torch.zeros(4, 64, 64, 64)

    semantic_model = SemanticFailure()
    semantic_model.train()
    with pytest.raises(RuntimeError, match="next encode failure"):
        runner._semantic_terms(
            model_api,
            semantic_model,
            {
                "action_indices": torch.zeros(4, dtype=torch.long),
                "current_rgb": torch.zeros(4, 1, 1, 1),
                "next_rgb": torch.ones(4, 1, 1, 1),
                "current_labels": torch.zeros(4, 64, 64, dtype=torch.long),
                "next_labels": torch.zeros(4, 64, 64, dtype=torch.long),
            },
        )
    assert runner._EVENT_ACCOUNTING["scheduled_pair_presentations_loaded"] == 4
    assert runner._EVENT_ACCOUNTING["online_encoder_lift_training_forward_count"] == 1
    assert runner._EVENT_ACCOUNTING["semantic_head_training_forward_count"] == 0

    class TargetFailure(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def encode_target(self, _value: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic target-next failure")
            channel = torch.linspace(-1, 1, 64).reshape(1, 64, 1, 1)
            return channel.expand(1, 64, 64, 64).contiguous()

    runner._reset_event_runtime_state()
    runner._FROZEN_REFERENCES.update({"T400": 0.25, "B400": 0.4})
    target_model = TargetFailure()
    target_model.train()
    with pytest.raises(RuntimeError, match="target-next failure"):
        runner._joint_terms(
            type("Runtime", (), {"torch": torch})(),
            model_api,
            target_model,
            {
                "current_rgb": torch.zeros(1, 1, 1, 1),
                "next_rgb": torch.ones(1, 1, 1, 1),
                "fixed_negative_rgb": torch.full((1, 1, 1, 1), 2.0),
                "action_indices": torch.tensor([1]),
            },
            torch.zeros(1, 64, 64, 64),
            persistence_baseline=0.4,
        )
    assert runner._EVENT_ACCOUNTING["target_encoder_lift_training_forward_count"] == 1
    assert runner._EVENT_ACCOUNTING["all_action_predictor_training_forward_count"] == 0

    class ContextFailure(_SyntheticEventModel):
        def predict_event_delta(
            self, x: torch.Tensor, action_one_hot: torch.Tensor
        ) -> Any:
            raise RuntimeError("synthetic context predictor failure")

    runner._reset_event_runtime_state()
    runner._FROZEN_REFERENCES.update({"T400": 0.25, "B400": 0.4})
    context_model = ContextFailure(model_api)
    context_model.train()
    current = torch.zeros(1, 1, 1, 1)
    current_latent = context_model.encode_online(current)
    with pytest.raises(RuntimeError, match="context predictor failure"):
        runner._joint_terms(
            type("Runtime", (), {"torch": torch})(),
            model_api,
            context_model,
            {
                "current_rgb": current,
                "next_rgb": torch.ones(1, 1, 1, 1),
                "fixed_negative_rgb": torch.full((1, 1, 1, 1), 2.0),
                "action_indices": torch.tensor([2]),
            },
            current_latent,
            persistence_baseline=0.4,
        )
    assert runner._EVENT_ACCOUNTING["target_encoder_lift_training_forward_count"] == 3
    assert runner._EVENT_ACCOUNTING["online_encoder_lift_training_forward_count"] == 1
    assert runner._EVENT_ACCOUNTING["all_action_predictor_training_forward_count"] == 1
    assert runner._EVENT_ACCOUNTING["context_swap_predictor_training_forward_count"] == 0
    assert runner._EVENT_ACCOUNTING["event_persistence_term_evaluation_count"] == 0


def _reservation(runner: Any, root: Path) -> tuple[dict[str, Any], bytes]:
    root.mkdir()
    return runner._BASE._publish_json(
        root / "reservation.json",
        {"schema": runner.contract.RESERVATION_SCHEMA, "attempt": "synthetic"},
    )


def test_terminal_receipt_inventories_split_science_and_operations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_event_delta_runner_terminal_inventory", RUNNER)
    monkeypatch.setattr(runner._BASE, "_seal", lambda _root: {})
    progress = {
        "stage": "synthetic_joint_gradient",
        "_observations": [],
        "_checkpoint_bindings": [],
        "_probe_failure_state": {
            "updates": 401,
            "presentations": 6_416,
            "pair_presentations_loaded": 6_416,
            "objective_evaluations": 1_604,
            "backward_calls": 1_604,
            "integrity": {},
        },
    }
    science_root = tmp_path / "science"
    science_reservation, science_raw = _reservation(runner, science_root)
    science_error = runner._BASE.ScientificGateFailure(
        "synthetic gradient ratio",
        control=runner.contract.CONTROL_FAIL_JOINT_GRADIENT,
    )
    runner._terminal_failure(
        science_root, science_reservation, science_raw, progress, science_error
    )
    assert runner._receipt_inventory(science_root) == [
        "access.json", "artifact.json", "completed.json", "metrics.json",
        "reservation.json", "result.json",
    ]
    assert not (science_root / "failure.json").exists()

    operation_root = tmp_path / "operation"
    operation_reservation, operation_raw = _reservation(runner, operation_root)
    phase_error = runner._BASE.ScientificGateFailure(
        "synthetic phase integrity",
        control=runner.contract.PHASE_SWITCH_CONTROLS[0],
    )
    runner._terminal_failure(
        operation_root, operation_reservation, operation_raw, progress, phase_error
    )
    assert runner._receipt_inventory(operation_root) == [
        "completed.json", "failure.json", "reservation.json"
    ]
    failure = __import__("json").loads(
        (operation_root / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["complete_failure_receipt"] is True
    assert "operation" in failure and "access" in failure
    assert failure["access"]["access_phase"] == "before_loader_construction"
    assert failure["access"]["sealed_open_count"] == 0


def test_staged_normal_receipts_do_not_leak_into_operational_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_event_delta_runner_staged_receipts", RUNNER)
    root = tmp_path / "staged"
    reservation, reservation_raw = _reservation(runner, root)
    progress: dict[str, Any] = {"stage": "synthetic_access_validation"}

    def fail_after_staging(**_kwargs: Any) -> int:
        runner._BASE._publish_json(
            root / "metrics.json",
            {"schema": runner.contract.METRICS_SCHEMA, "status": "staged"},
        )
        runner._BASE._publish_json(
            root / "artifact.json",
            {"schema": runner.contract.ARTIFACT_SCHEMA, "status": "staged"},
        )
        raise RuntimeError("synthetic access validation failure")

    monkeypatch.setattr(runner, "_BASE_EXECUTE", fail_after_staging)
    with pytest.raises(RuntimeError, match="access validation failure"):
        runner._execute(
            sources={"synthetic.py": "b" * 64},
            authorization={},
            reservation=reservation,
            reservation_raw=reservation_raw,
            output_root=root,
            progress=progress,
        )
    assert not (root / "metrics.json").exists()
    assert not (root / "artifact.json").exists()
    monkeypatch.setattr(runner._BASE, "_seal", lambda _root: {})
    runner._terminal_failure(
        root,
        reservation,
        reservation_raw,
        progress,
        RuntimeError("synthetic access validation failure"),
    )
    assert runner._receipt_inventory(root) == [
        "completed.json", "failure.json", "reservation.json"
    ]


def test_access_validation_failure_preserves_raw_nonzero_counters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load("_event_delta_runner_access_truth", RUNNER)

    class Loader:
        def receipt(self) -> dict[str, int]:
            return {"sealed_open_count": 2, "consumed_record_count": 17}

    monkeypatch.setattr(
        runner._BASE,
        "_access_receipt",
        lambda *_args: (_ for _ in ()).throw(
            PermissionError("synthetic forbidden access")
        ),
    )
    receipt = runner._terminal_access({"_loader": Loader(), "_inputs": object()})
    assert receipt["loader_receipt_available"] is False
    assert receipt["validated_forbidden_access_counts"] is None
    assert receipt["raw_loader_receipt"]["sealed_open_count"] == 2
    assert receipt["sealed_open_count"] == 2


def test_observation_accounting_is_separate_and_zero_presentation() -> None:
    runner = _load("_event_delta_runner_observation_counts", RUNNER)
    assert runner._observation_accounting(0) == {
        "observation_pair_population_count": 495,
        "observation_endpoint_population_count": 924,
        "observation_pair_pass_count": 1,
        "observation_endpoint_pass_count": 1,
        "observation_pair_microbatch_count": 124,
        "observation_endpoint_microbatch_count": 231,
        "observation_microbatch_count": 355,
        "observation_online_encoder_lift_forward_count": 603,
        "observation_target_encoder_lift_forward_count": 0,
        "observation_semantic_head_forward_count": 603,
        "observation_all_action_predictor_forward_count": 0,
        "observation_one_action_predictor_forward_count": 0,
        "observation_predictor_forward_count": 0,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
    }
    final = runner._observation_accounting(1_000)
    assert final["observation_microbatch_count"] == 834
    assert final["observation_online_encoder_lift_forward_count"] == 1_099
    assert final["observation_target_encoder_lift_forward_count"] == 975
    assert final["observation_predictor_forward_count"] == 620


@pytest.mark.parametrize(
    ("update", "expected"),
    (
        (0, (1, 0, 0, 0, 0, 0)),
        (100, (1, 1, 0, 0, 0, 0)),
        (400, (1, 1, 1, 1, 0, 0)),
        (1_000, (1, 1, 1, 1, 1, 1)),
    ),
)
def test_partial_observation_work_records_successful_call_boundaries(
    update: int,
    expected: tuple[int, int, int, int, int, int],
) -> None:
    runner = _load(f"_event_delta_runner_live_{update}", RUNNER)

    class Model:
        def encode_online(self, value: torch.Tensor) -> torch.Tensor:
            return value

        def encode_target(self, value: torch.Tensor) -> torch.Tensor:
            return value

        def semantic_logits_from_latent(
            self, value: torch.Tensor
        ) -> torch.Tensor:
            return value

        def predict_all_action_event_deltas(
            self, value: torch.Tensor
        ) -> torch.Tensor:
            return value

        def predict_event_delta(
            self, value: torch.Tensor, _action: torch.Tensor
        ) -> torch.Tensor:
            return value

    class Loader:
        def batch(
            self,
            _pairs: list[dict[str, Any]],
            indices: list[int],
            _device: str,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            return {"action_indices": torch.zeros(len(indices), dtype=torch.long)}

        def endpoint_batch(
            self,
            identities: list[str],
            _device: str,
            **_kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros(len(identities), 1), torch.zeros(len(identities), 1)

    model = Model()
    loader = Loader()
    runner._begin_observation_accounting(update)
    failure = RuntimeError(f"synthetic partial observation U{update}")
    try:
        with runner._instrument_observation_calls(model, loader):
            loader.batch(
                [{}, {}], [0, 1], "cpu",
                stage=f"partial_pair_update_{update}",
            )
            model.encode_online(torch.zeros(1))
            if update >= 100:
                model.semantic_logits_from_latent(torch.zeros(1))
            if update >= 400:
                model.encode_target(torch.zeros(1))
                loader.endpoint_batch(
                    ["a", "b", "c"], "cpu",
                    stage=f"partial_endpoint_update_{update}",
                )
            if update >= 1_000:
                model.predict_all_action_event_deltas(torch.zeros(1))
                model.predict_event_delta(torch.zeros(1), torch.zeros(1))
            raise failure
    except RuntimeError as caught:
        assert caught is failure
        runner._mark_observation_failure(caught)

    receipt = runner._observation_live_receipt()
    assert receipt is not None
    assert receipt["observation_status"] == "failed"
    assert receipt["observation_pair_successful_microbatch_count"] == 1
    assert receipt["observation_pair_rows_loaded"] == 2
    assert receipt["observation_pair_completed_pass_count"] == 0
    assert receipt["observation_presentations_count"] == 0
    assert receipt["observation_schedule_advance_count"] == 0
    names = (
        "observation_online_encoder_lift_forward_count",
        "observation_semantic_head_forward_count",
        "observation_target_encoder_lift_forward_count",
        "observation_endpoint_successful_microbatch_count",
        "observation_all_action_predictor_forward_count",
        "observation_one_action_predictor_forward_count",
    )
    assert tuple(receipt[name] for name in names) == expected


def test_operational_failure_binds_partial_observation_runtime_inputs_and_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_event_delta_runner_partial_failure", RUNNER)
    root = tmp_path / "partial_failure"
    reservation, reservation_raw = _reservation(runner, root)

    class Inputs:
        consumed = {
            "record-a": {"role": "train"},
            "record-b": {"role": "checkpoint_selection"},
        }

    partial = {
        "observation_update": 400,
        "observation_status": "failed",
        "observation_pair_successful_microbatch_count": 7,
        "observation_endpoint_successful_microbatch_count": 0,
        "observation_presentations_count": 0,
        "observation_schedule_advance_count": 0,
    }
    runtime_inputs = runner.contract.runtime_authorization_template()
    progress = {
        "stage": "observation_update_400",
        "_inputs": Inputs(),
        "_partial_observation_work": partial,
        "_authorized_runtime_inputs": runner._runtime_input_authority_receipt(
            {"runtime_inputs": runtime_inputs}
        ),
    }
    monkeypatch.setattr(runner._BASE, "_seal", lambda _root: {})
    runner._terminal_failure(
        root, reservation, reservation_raw, progress,
        RuntimeError("synthetic partial observation failure"),
    )
    failure = __import__("json").loads(
        (root / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["partial_observation_work"] == partial
    assert failure["authorized_runtime_inputs"]["availability"] == "available"
    assert failure["authorized_runtime_inputs"]["binding"] == runtime_inputs
    assert failure["access"]["roles_opened"] == [
        "checkpoint_selection", "train"
    ]
    assert failure["access"]["consumed_record_count"] == 2
    assert failure["access"]["development_input_consumption_known_exact"] is True
    assert failure["access"]["model_facing_counts"] is None
    assert failure["access"]["heldout_open_count"] == 0


def test_completed_observer_work_survives_following_gate_evaluation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_event_delta_runner_post_observer_gate", RUNNER)
    progress: dict[str, Any] = {
        "stage": "observation_update_100",
        "_probe_failure_state": {
            "updates": 100,
            "presentations": 1_600,
            "objective_evaluations": 0,
            "backward_calls": 0,
            "integrity": {},
        },
    }

    def fail_after_completed_observer(*_args: Any, **_kwargs: Any) -> Any:
        runner._begin_observation_accounting(100)
        runner._OBSERVATION_LIVE.update({
            "observation_pair_successful_microbatch_count": 124,
            "observation_endpoint_successful_microbatch_count": 231,
            "observation_pair_rows_loaded": 495,
            "observation_endpoint_rows_loaded": 924,
            "observation_pair_stage_successful_microbatch_counts": {
                "pair_observation_update_100": 124,
            },
            "observation_pair_stage_rows_loaded": {
                "pair_observation_update_100": 495,
            },
            "observation_endpoint_stage_successful_microbatch_counts": {
                "raster_observation_update_100": 231,
            },
            "observation_endpoint_stage_rows_loaded": {
                "raster_observation_update_100": 924,
            },
            "observation_online_encoder_lift_forward_count": 603,
            "observation_semantic_head_forward_count": 603,
        })
        completed = runner._validate_completed_observation_accounting(100)
        assert completed["observation_status"] == "complete"
        raise ValueError("synthetic gate evaluation failure")

    monkeypatch.setattr(runner, "_BASE_TRAIN_PROBE", fail_after_completed_observer)
    with pytest.raises(ValueError, match="gate evaluation failure"):
        runner._train_probe(progress=progress)
    terminal = progress["_terminal_observation_work"]
    assert terminal["observation_status"] == "complete"
    assert terminal["observation_microbatch_count"] == 355
    assert terminal["observation_presentations_count"] == 0
    assert terminal["observation_schedule_advance_count"] == 0
    assert "_partial_observation_work" not in progress
    assert progress["_probe_failure_state"]["terminal_observation_work"] == terminal

    root = tmp_path / "post_observer_gate"
    reservation, reservation_raw = _reservation(runner, root)
    monkeypatch.setattr(runner._BASE, "_seal", lambda _root: {})
    runner._terminal_failure(
        root, reservation, reservation_raw, progress,
        ValueError("synthetic gate evaluation failure"),
    )
    failure = __import__("json").loads(
        (root / "failure.json").read_text(encoding="utf-8")
    )
    assert failure["terminal_observation_work"] == terminal
    assert failure["partial_observation_work"] is None


def test_gpu_elapsed_timer_starts_before_gpu_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = _load("_event_delta_runner_gpu_timer", RUNNER)
    root = tmp_path / "gpu_timer"
    root.mkdir()
    progress: dict[str, Any] = {}

    def load_inputs(*_args: Any, **_kwargs: Any) -> tuple[str]:
        return ("inputs",)

    def fail_during_gpu_validation(**_kwargs: Any) -> int:
        runner._BASE._load_development_inputs()
        assert "_gpu_active_started_monotonic" in progress
        raise RuntimeError("synthetic GPU validation failure")

    monkeypatch.setattr(runner, "_BASE_LOAD_DEVELOPMENT_INPUTS", load_inputs)
    monkeypatch.setattr(runner, "_BASE_EXECUTE", fail_during_gpu_validation)
    with pytest.raises(RuntimeError, match="GPU validation failure"):
        runner._execute(
            sources={},
            authorization={},
            reservation={},
            reservation_raw=b"{}\n",
            output_root=root,
            progress=progress,
        )
    assert progress["_development_inputs_loaded"] is True
    assert progress["_gpu_active_elapsed_seconds"] >= 0.0
    assert progress["_authorized_runtime_inputs"]["availability"] == (
        "unknown_or_absent"
    )


def test_runner_and_launcher_delegate_with_event_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load("_event_delta_runner_delegate", RUNNER)
    calls: list[list[str] | None] = []

    def fake_main(argv: list[str] | None = None) -> int:
        assert runner._BASE.contract is runner.contract
        assert runner._BASE._joint_terms is runner._joint_terms
        assert runner._BASE._evaluate_observation is runner._evaluate_observation
        calls.append(argv)
        return 41

    monkeypatch.setattr(runner._RIGID, "main", fake_main)
    args = ["--review-sha256", "a" * 64, "--authorization-sha256", "b" * 64]
    assert runner.main(args) == 41
    assert calls == [args]

    launcher = _load("_event_delta_launcher_delegate", LAUNCHER)
    parsed = launcher.parse_args(args)
    base = launcher._RIGID._V3._V2._V1
    assert base._runtime_argv(parsed) == [
        launcher.contract.RUNTIME_INTERPRETER_PATH,
        *launcher.contract.RUNTIME_INTERPRETER_ARGUMENTS,
        str(ROOT / launcher.contract.RUNNER_RELATIVE_PATH),
        "--review-sha256", "a" * 64,
        "--authorization-sha256", "b" * 64,
    ]
    assert base.OUTPUT_ROOT == ROOT / launcher.contract.OUTPUT_ROOT_RELATIVE_PATH
