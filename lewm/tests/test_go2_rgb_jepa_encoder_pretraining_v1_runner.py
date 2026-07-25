from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import warnings

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_jepa_encoder_pretraining_v1.py"


def _load_runner(name: str = "_jepa_encoder_v9_runner_test"):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_import_is_source_only_and_bound_to_v9() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_source_only_runner", {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert module.PREFLIGHT_ENVIRONMENT_KEY == (
    "LEWM_RGB_DENSE_PAIRWISE_SPATIAL_COST_VOLUME_"
    "INVERSE_JEPA_V9_PREFLIGHT_JSON"
)
assert module.contract.PREREGISTRATION_COMMIT == (
    "b775093897669c91d8c1b9e7d148e257881bcedf"
)
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


def test_runner_source_contains_v9_only_and_phase_b_rejects_the_head() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for removed in (
        "ActionConditionedLocalCorrespondenceTransport",
        "predict_action_conditioned_local_transports",
        "local_correspondence",
        "all_candidate_correspondence",
        "transport_weight",
    ):
        assert removed not in source
    assert "ActionConditionedLatentFlow" in source
    assert "DensePairwiseSpatialCostVolumeInverseHead" in source
    assert "predict_action_conditioned_flow_warps" in source
    assert (
        "fresh Phase-B model unexpectedly contains the V9 inverse head"
        in source
    )

    module = _load_runner("_jepa_encoder_v9_phase_b_scope")
    assert "dense_pairwise_inverse_head." not in (
        module.contract.PHASE_B_FROZEN_PARAMETER_PREFIXES
    )


def test_deferred_preflight_accepts_exact_unsigned_child_observation() -> None:
    module = _load_runner("_jepa_encoder_v9_preflight_observation")
    observation = {
        "preflight_child_process_id": 123,
        "visible_device_count": 1,
        "visible_device_index": 0,
        "visible_device_name": "AMD Radeon PRO R9700",
        "total_memory_bytes": 32_000_000_000,
        "torch_version": "test",
        "hip_version": "test",
        "tensor_allocation_count": 0,
        "payload_open_count": 0,
        "torch_device_api_call_count": 3,
    }
    stdout = (
        module.contract.canonical_json_bytes(observation).decode("ascii")
        + "\n"
    )
    assert module._parse_preflight_observation(stdout) == observation
    with pytest.raises(RuntimeError, match="not canonical"):
        module._parse_preflight_observation(stdout + "\n")


def _objective_model(torch):
    nn = torch.nn

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.feature_scale = nn.Parameter(
                torch.linspace(0.6, 1.4, 192)
            )
            self.seen: list[object] = []

        def forward_tokens(self, image):
            self.seen.append(image.detach().clone())
            pixels = image.mean(dim=1).reshape(image.shape[0], 256)
            feature_bias = torch.linspace(
                -0.7,
                0.9,
                192,
                dtype=image.dtype,
                device=image.device,
            )
            patches = (
                pixels[:, :, None] * self.feature_scale[None, None, :]
                + feature_bias[None, None, :]
            )
            return torch.cat(
                (
                    torch.zeros(
                        image.shape[0],
                        1,
                        192,
                        dtype=image.dtype,
                        device=image.device,
                    ),
                    patches,
                ),
                dim=1,
            )

    class Model(nn.Module):
        def __init__(self, head) -> None:
            super().__init__()
            self.encoder = Encoder()
            self.online_geometry = nn.Identity()
            self.online_target_projector = nn.Identity()
            self.prediction_projector = nn.Identity()
            self.predictor = nn.Identity()
            self.appearance_projector = nn.Linear(192, 192)
            self.target_encoder = Encoder()
            self.target_geometry_module = nn.Identity()
            self.target_projector = nn.Identity()
            self.dense_pairwise_inverse_head = head
            for frozen in (
                self.target_encoder,
                self.target_geometry_module,
                self.target_projector,
            ):
                frozen.requires_grad_(False)

    return Model


def _objective_ops(torch, actual_ops):
    def predict(_predictor, _projector, state, action, _target_current):
        all_predictions = state[:, None].expand(-1, 9, -1, -1)
        requested = action.argmax(dim=1)
        candidates = torch.stack([
            torch.tensor(
                [index for index in range(9) if index != int(row)],
                dtype=torch.long,
                device=state.device,
            )
            for row in requested
        ])
        return SimpleNamespace(
            executed=state,
            all_predictions=all_predictions,
            controls=all_predictions.gather(
                1,
                candidates[:, :, None, None].expand(-1, -1, 256, 192),
            ),
            control_indices=candidates,
            all_flows_cell=torch.zeros(
                state.shape[0], 9, 256, 2, device=state.device
            ),
        )

    def indexed_energy(predictions, target):
        jepa = (predictions.executed - target).square().mean()
        identification = predictions.executed.square().mean() * 0.01
        return SimpleNamespace(
            jepa=jepa,
            identification=identification,
            row_scale=torch.ones(
                predictions.executed.shape[0],
                dtype=torch.float32,
                device=predictions.executed.device,
            ),
        )

    def whitening(value):
        return SimpleNamespace(
            variance=value.square().mean() * 0.001,
            covariance=value.mean().square() * 0.001,
        )

    return SimpleNamespace(
        normalize_spatial_tokens=lambda value: value,
        predict_action_conditioned_flow_warps=predict,
        action_indexed_energy_nll=indexed_energy,
        dense_pairwise_spatial_cost_volume_inverse_terms=(
            actual_ops.dense_pairwise_spatial_cost_volume_inverse_terms
        ),
        patch_whitening_terms=whitening,
    )


def test_v9_objective_uses_both_live_encoder_branches_and_dense_head() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(44)
    module = _load_runner("_jepa_encoder_v9_objective")
    actual_ops = module._phase_a_ops()
    model = _objective_model(torch)(
        actual_ops.DensePairwiseSpatialCostVolumeInverseHead()
    )
    ops = _objective_ops(torch, actual_ops)
    current = torch.randn(3, 3, 16, 16, requires_grad=True)
    next_rgb = torch.randn(3, 3, 16, 16, requires_grad=True)
    action = torch.zeros(3, 9)
    action[torch.arange(3), torch.tensor([0, 4, 8])] = 1.0

    output = module._phase_a_current_only_loss(
        model,
        current,
        next_rgb,
        action,
        ops=ops,
    )
    expected = (
        output["jepa_loss"]
        + module.contract.ACTION_INDEXED_ENERGY_NLL_WEIGHT
        * output["action_identification_loss"]
        + module.contract.DENSE_PAIRWISE_INVERSE_LOSS_WEIGHT
        * output["dense_pairwise_inverse_loss"]
        + module.contract.WHITENING_VARIANCE_WEIGHT
        * (
            output["raw_whitening_variance_loss"]
            + output["projected_whitening_variance_loss"]
        )
        + module.contract.WHITENING_COVARIANCE_WEIGHT
        * (
            output["raw_whitening_covariance_loss"]
            + output["projected_whitening_covariance_loss"]
        )
    )
    assert torch.equal(output["loss"], expected)
    assert output["dense_pairwise_inverse_logits"].shape == (3, 9)
    assert output["dense_pairwise_current_next_cost_volume"].shape == (
        3, 256, 256
    )
    assert output["dense_pairwise_current_current_cost_volume"].shape == (
        3, 256, 256
    )
    assert output["dense_pairwise_volume"].shape == (3, 256, 16, 16)
    assert output["dense_pairwise_displacement"].shape == (3, 2, 16, 16)
    assert output["all_flows_cell"].shape == (3, 9, 256, 2)
    assert len(model.encoder.seen) == 2
    assert torch.equal(model.encoder.seen[0], current.detach())
    assert torch.equal(model.encoder.seen[1], next_rgb.detach())
    assert len(model.target_encoder.seen) == 2

    output["loss"].backward()
    assert current.grad is not None
    assert next_rgb.grad is not None
    assert torch.count_nonzero(current.grad).item() > 0
    assert torch.count_nonzero(next_rgb.grad).item() > 0
    assert model.encoder.feature_scale.grad is not None
    assert torch.count_nonzero(model.encoder.feature_scale.grad).item() > 0
    for parameter in (
        model.dense_pairwise_inverse_head.channel_projection.weight,
        model.dense_pairwise_inverse_head.spatial_projection.weight,
        model.dense_pairwise_inverse_head.classifier.weight,
        model.dense_pairwise_inverse_head.classifier.bias,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad).item() > 0
    assert all(
        parameter.grad is None and not parameter.requires_grad
        for parameter in model.target_encoder.parameters()
    )


def _initialization_model(torch):
    nn = torch.nn

    class GateBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(192, 6 * 192),
            )
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)

    class Predictor(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([GateBlock()])
            self.action_embed = nn.Linear(9, 192, bias=False)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Linear(2, 2)
            self.online_geometry = nn.Identity()
            self.online_target_projector = nn.Linear(192, 192)
            self.prediction_projector = nn.Linear(192, 192)
            self.predictor = Predictor()
            self.appearance_projector = nn.Linear(192, 192)
            self.target_encoder = nn.Linear(2, 2)
            self.target_geometry_module = nn.Identity()
            self.target_projector = nn.Linear(192, 192)
            for frozen in (
                self.target_encoder,
                self.target_geometry_module,
                self.target_projector,
            ):
                frozen.requires_grad_(False)

    return Model


def test_phase_a_initialization_restores_v5_flow_and_exact_v9_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v9_initialization")
    monkeypatch.setattr(module, "_state_sha", lambda *_args: "a" * 64)
    phase2d = SimpleNamespace(
        Phase2DSpatialLeWorldModel=lambda **_kwargs: (
            _initialization_model(torch)()
        )
    )
    fit = SimpleNamespace(encoder=torch.nn.Linear(2, 2))
    model, partition, receipt = module._phase_a_model(
        SimpleNamespace(torch=torch),
        phase2d,
        fit,
        torch.device("cpu"),
    )

    flow = receipt["state_dependent_latent_flow_initialization"]
    assert flow["weight_shape"] == [2, 192]
    assert flow["exact_zero_weight_scalar_count"] == 384
    assert torch.count_nonzero(model.prediction_projector.flow_weight) == 0
    head = receipt["dense_pairwise_inverse_head_initialization"]
    assert head["seed"] == 20260725
    assert head["parameter_count"] == 8_713
    assert head["all_three_weights_every_scalar_nonzero"] is True
    assert head["classifier_bias_exact_zero"] is True
    assert sum(
        parameter.numel()
        for parameter in model.dense_pairwise_inverse_head.parameters()
    ) == 8_713
    assert all(
        any(parameter is candidate for candidate in partition["other"])
        for parameter in model.dense_pairwise_inverse_head.parameters()
    )
    assert all(
        parameter is not candidate
        for parameter in model.dense_pairwise_inverse_head.parameters()
        for candidate in partition["encoder"]
    )
    assert partition["receipt"]["appearance_projector_frozen"] is True

    model.unregistered_science_delta = torch.nn.Parameter(torch.ones(()))
    with pytest.raises(PermissionError, match="partition changed"):
        module._phase_a_parameter_partition(model)


def test_update_zero_compares_all_action_pairs_across_495_rows() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v9_update_zero")
    base = torch.arange(495 * 6, dtype=torch.float32).reshape(
        495, 1, 2, 3
    )
    predictions = base.expand(-1, 9, -1, -1).clone()
    row_count = 0
    comparison_count = None
    for batch in predictions.split(16):
        rows, pairs = module._verify_update_zero_action_symmetry_batch(
            torch, batch
        )
        row_count += rows
        comparison_count = pairs
    assert module._update_zero_action_symmetry_receipt(
        row_count=row_count,
        comparison_count=comparison_count,
    ) == {
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": 36,
        "all_action_prediction_row_count": 495,
    }
    predictions[-1, -1, -1, -1] += 1
    with pytest.raises(PermissionError, match="not bitwise equal"):
        module._verify_update_zero_action_symmetry_batch(
            torch, predictions[-15:]
        )


def _pair(scene: str, content: str, current: str, next_: str):
    return {
        "scene_id": scene,
        "content_sha256": content,
        "current_endpoint_sha256": current,
        "next_endpoint_sha256": next_,
    }


def test_scene_derangement_is_local_deterministic_and_identity_safe() -> None:
    module = _load_runner("_jepa_encoder_v9_derangement")
    pairs = [
        _pair("s1", "c", "a", "n1"),
        _pair("s1", "a", "b", "n2"),
        _pair("s1", "b", "b", "n3"),
        _pair("s2", "e", "d", "n5"),
        _pair("s2", "d", "e", "n4"),
    ]
    mapping = module._scene_derangement(
        pairs, endpoint_key="current_endpoint_sha256"
    )
    assert mapping == module._scene_derangement(
        pairs, endpoint_key="current_endpoint_sha256"
    )
    for row, candidate in enumerate(mapping):
        assert pairs[row]["scene_id"] == pairs[candidate]["scene_id"]
        assert (
            pairs[row]["current_endpoint_sha256"]
            != pairs[candidate]["current_endpoint_sha256"]
        )


def _latent_flow(contract, *, update_zero: bool):
    active = {
        action: (not update_zero and action != "hold")
        for action in contract.ACTION_VOCABULARY
    }
    return {
        "all_values_finite": True,
        "all_components_within_closed_one_patch_bound": True,
        "hold_flow_exactly_zero": True,
        "maximum_absolute_flow_cell": 0.0 if update_zero else 0.4,
        "non_hold_action_nonzero_count": 0 if update_zero else 8,
        "per_action_any_nonzero": active,
    }


def _dense(contract, nll: float):
    non_hold_counts = iter((55, 55, 55, 54, 54, 54, 54, 54))
    counts = {
        action: (60 if action == "hold" else next(non_hold_counts))
        for action in contract.ACTION_VOCABULARY
    }
    correct = {action: count // 2 for action, count in counts.items()}
    recalls = {
        action: correct[action] / counts[action]
        for action in contract.ACTION_VOCABULARY
    }
    top1 = sum(correct.values()) / 495
    macro = sum(recalls.values()) / 9
    deranged = nll / 0.8
    family = {name: 0.1 for name in contract.SCENE_FAMILIES}
    return {
        "all_values_finite": True,
        "probabilities_all_values_finite": True,
        "probability_rows_normalized": True,
        "volume_all_values_finite": True,
        "volume_values_within_closed_unit_interval": True,
        "volume_channel_conservation": True,
        "displacement_all_values_finite": True,
        "displacement_values_within_closed_two_bound": True,
        "maximum_absolute_displacement_component": 0.2,
        "cross_pair_displacement_rms": 0.05,
        "cross_pair_displacement_value_count": 495 * 2 * 16 * 16,
        "same_tensor_diff_exact_zero": True,
        "same_tensor_volume_exact_zero": True,
        "same_tensor_displacement_exact_zero": True,
        "head_parameters_all_values_finite": True,
        "head_parameter_count": 8_713,
        "head_weight_tensors_all_nonzero": True,
        "zero_logit_reference_nll": math.log(9.0),
        "unscaled_dense_inverse_nll": nll,
        "dense_inverse_top1_accuracy": top1,
        "per_executed_action_dense_inverse": {
            action: {
                "row_count": counts[action],
                "mean_nll": nll,
                "recall": recalls[action],
            }
            for action in contract.ACTION_VOCABULARY
        },
        "dense_inverse_macro_balanced_accuracy": macro,
        "correct_pair_nll": nll,
        "correct_pair_count": 495,
        "deranged_next_nll": deranged,
        "deranged_next_pair_count": 495,
        "correct_to_deranged_nll_ratio": 0.8,
        "non_hold_correct_pair_nll": nll,
        "non_hold_correct_pair_count": 435,
        "non_hold_current_current_nll": deranged,
        "non_hold_current_current_pair_count": 435,
        "non_hold_correct_to_current_current_nll_ratio": 0.8,
        "deranged_positive_family_margin_count": 8,
        "per_family_deranged_minus_correct_nll": family,
    }


def _metric(contract, nll: float):
    return {
        "all_values_finite": True,
        "ema_target_gradient_free": True,
        "pair_count": 495,
        "scene_family_count": 8,
        "cyclic_wrong_action_pair_count": 495,
        "all_wrong_action_candidate_count": 495 * 8,
        "non_hold_pair_count": 435,
        "hold_action_pair_count": 435,
        "hold_action_rows_match_non_hold_rows": True,
        "centered_raw_patch_effective_rank": 50.0,
        "centered_projected_target_effective_rank": 50.0,
        "raw_cross_sample_variance": 1.0,
        "content_residual_spatial_diversity": 1.0,
        "true_pair_mse": 0.8,
        "shuffled_next_mse": 1.0,
        "mean_target_mse": 1.0,
        "cyclic_wrong_action_mse": 1.0,
        "hardest_wrong_action_mse": 1.0,
        "non_hold_true_pair_mse": 0.8,
        "hold_action_mse": 1.0,
        "shuffled_current_mse": 1.0,
        "per_family": {
            family: {
                "cyclic_wrong_action_minus_true_mse": 0.2,
                "hardest_wrong_action_minus_true_mse": 0.2,
                "hold_action_minus_non_hold_true_mse": 0.2,
                "hold_action_rows_match_non_hold_rows": True,
            }
            for family in contract.SCENE_FAMILIES
        },
        "latent_flow": _latent_flow(contract, update_zero=False),
        "dense_pairwise_inverse": _dense(contract, nll),
    }


def _update0(contract):
    return {
        "raw_cross_sample_variance": 1.0,
        "content_residual_spatial_diversity": 1.0,
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": 36,
        "all_action_prediction_row_count": 495,
        "latent_flow": _latent_flow(contract, update_zero=True),
        "dense_pairwise_inverse": _dense(contract, math.log(9.0)),
    }


def test_staged_gates_compare_100_to_400_to_1000_without_retry() -> None:
    module = _load_runner("_jepa_encoder_v9_staged_gates")
    contract = module.contract
    integrity = {"rng_state_preserved": True, "state_mutation_count": 0}
    update0 = _update0(contract)
    metric100 = _metric(contract, 1.9)
    gate100 = contract.evaluate_phase_a_continuation(
        100, metric100, update0, integrity, None
    )
    assert gate100["passed"] is True

    metric400 = _metric(contract, 1.8)
    gate400 = contract.evaluate_phase_a_continuation(
        400, metric400, update0, integrity, metric100
    )
    assert gate400["passed"] is True

    metric1000 = _metric(contract, 1.7)
    terminal = contract.evaluate_phase_a(
        metric1000, update0, integrity, metric400
    )
    assert terminal["passed"] is True
    assert terminal["control"] == contract.CONTROL_PHASE_A_PASS

    no_improvement = _metric(contract, 1.9)
    failed = contract.evaluate_phase_a_continuation(
        400, no_improvement, update0, integrity, metric100
    )
    assert failed["passed"] is False
    assert failed["control"] == contract.CONTROL_PHASE_A_UPDATE_400_FAIL
    assert module.contract.MAXIMUM_ATTEMPTS == 1


def test_train_wires_previous_metrics_and_exact_phase_a_status() -> None:
    module = _load_runner("_jepa_encoder_v9_train_wiring")
    source = inspect.getsource(module._phase_a_train)
    assert "previous_metric" in source
    assert "diagnostics[-2][\"metric\"]" in source
    assert "phase_status = str(terminal_gate[\"control\"])" in source
    assert "contract.CONTROL_PHASE_A_PASS" in source


def test_phase_a_determinism_restores_exact_v5_warning_scope() -> None:
    module = _load_runner("_jepa_encoder_v9_determinism")
    calls = []

    class Torch:
        @staticmethod
        def use_deterministic_algorithms(enabled, *, warn_only):
            calls.append((enabled, warn_only))

    runtime = SimpleNamespace(torch=Torch())

    def operation():
        warnings.warn(
            module.contract.PHASE_A_GRID_SAMPLE_DETERMINISM_WARNING_PREFIX
            + " synthetic",
            UserWarning,
        )
        return "done"

    result, receipt = module._run_phase_a_with_reviewed_determinism(
        runtime, operation
    )
    assert result == "done"
    assert calls == [(True, True), (True, False)]
    assert receipt["warn_only_scope"] == (
        "phase_a_training_and_checkpoint_selection"
    )
    assert receipt["expected_grid_sampler_warning_count"] == 1


def _failure_authorization():
    binding = {
        "path": "not-opened",
        "file_sha256": "a" * 64,
        "byte_count": 1,
    }
    return {
        "runtime_inputs": {
            "raw": {
                "manifest": dict(binding),
                "audit": dict(binding),
            },
            "schedule": dict(binding),
            "camera": {
                "gate": dict(binding),
                "checkpoint": dict(binding),
            },
        },
    }


def test_failure_custody_has_every_distinct_zero_counter() -> None:
    module = _load_runner("_jepa_encoder_v9_failure_custody")
    progress = {
        "phase_a_updates": 0,
        "phase_a_presentations": 0,
        "phase_b_updates": 0,
        "phase_b_presentations": 0,
    }
    receipt = module._failure_custody_attestation(
        _failure_authorization(),
        {"reviewed_sources": {}},
        progress,
    )
    counters = receipt["forbidden_access_counts"]
    assert tuple(counters) == module.contract.ACCESS_ZERO_COUNTER_FIELDS
    assert all(value == 0 for value in counters.values())
    assert receipt["operation_counts"]["cumulative_optimizer_updates"] == 0
    assert receipt["operation_counts"]["cumulative_pair_presentations"] == 0
    assert receipt["consumed"]["status"] == "TRACKER_NOT_YET_CONSTRUCTED"


def test_failed_exclusive_write_preserves_bound_partial_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_jepa_encoder_v9_partial_evidence")
    module._reset_output_binding_registry(tmp_path)
    path = tmp_path / "phase_a" / "metrics.json"
    raw = b"scientific evidence that must survive"
    original_fsync = module.os.fsync
    calls = 0

    def fail_first_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("bounded synthetic fsync failure")
        return original_fsync(descriptor)

    monkeypatch.setattr(module.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="synthetic fsync"):
        module._write_exclusive(path, raw)
    assert not path.exists()
    inventory = module._terminal_inventory(tmp_path)
    assert len(inventory["partial_evidence_bindings"]) == 1
    binding = inventory["partial_evidence_bindings"][0]
    assert binding["byte_count"] == len(raw)
    assert binding["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert binding["path"] in inventory["files"]

    partial_path = tmp_path / binding["path"]
    partial_path.chmod(0o644)
    partial_path.write_bytes(b"x" * len(raw))
    with pytest.raises(PermissionError, match="write-time binding"):
        module._terminal_inventory(tmp_path)

    malformed = tmp_path / "malformed.json"
    malformed.write_bytes(b"not canonical")
    assert module._canonical_receipt_present(
        tmp_path, "malformed.json"
    ) is False


def test_phase_b_failure_inventory_extends_normal_receipt_set() -> None:
    module = _load_runner("_jepa_encoder_v9_failure_inventory")
    source = inspect.getsource(module._terminal_failure)
    assert "contract.PHASE_B_RECEIPT_PATHS" in source
    assert "_canonical_receipt_present" in source
    assert "missing_normal_receipts_synthesized" in source


def test_late_failure_binds_complete_checkpoint_trace_and_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_jepa_encoder_v9_complete_failure_inventory")
    output_root = tmp_path / "attempt"
    output_root.mkdir(mode=0o700)
    module._reset_output_binding_registry(output_root)
    reservation, reservation_raw = module._publish_json(
        output_root / "reservation.json",
        {
            "schema": module.contract.RESERVATION_SCHEMA,
            "status": "SYNTHETIC_RESERVED",
            "attempt_identity": "synthetic-attempt",
            "reviewed_sources": {},
        },
    )
    metrics, _ = module._publish_json(
        output_root / "phase_a/metrics.json",
        {
            "schema": module.contract.PHASE_A_METRICS_SCHEMA,
            "status": "SYNTHETIC_COMPLETE_BEFORE_LATER_FAILURE",
        },
    )
    checkpoint_raw = b"synthetic complete write-only checkpoint"
    trace_raw = b'{"synthetic":"trace"}\n'
    checkpoint_content_sha256 = "c" * 64
    checkpoint_state_sha256 = "d" * 64
    trace_content_sha256 = module.contract.canonical_json_sha256([
        {"synthetic": "trace"}
    ])
    checkpoint_path = output_root / "phase_a/checkpoints/update_100.pt"
    module._write_exclusive(checkpoint_path, checkpoint_raw)
    module._register_output_semantic_metadata(
        checkpoint_path,
        content_sha256=checkpoint_content_sha256,
        state_sha256=checkpoint_state_sha256,
        phase="phase_a",
        update=100,
        schedule_prefix_sha256=(
            module.contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[100]
        ),
    )
    trace_path = output_root / "phase_a/training_trace.jsonl"
    module._write_exclusive(trace_path, trace_raw)
    module._register_output_semantic_metadata(
        trace_path,
        content_sha256=trace_content_sha256,
        row_count=1,
    )

    original_read_regular = module._read_regular

    def forbid_write_only_reopen(path, **kwargs):
        if Path(path).suffix in {".pt", ".jsonl"}:
            raise AssertionError("write-only artifact payload was reopened")
        return original_read_regular(path, **kwargs)

    monkeypatch.setattr(module, "_read_regular", forbid_write_only_reopen)

    progress = {
        "stage": "synthetic_late_failure",
        "phase_a_updates": 100,
        "phase_a_presentations": 1_600,
        "phase_a_optimizer_updates": 100,
        "phase_a_ema_updates": 100,
        "phase_b_entered": False,
        "phase_b_updates": 0,
        "phase_b_presentations": 0,
    }
    module._terminal_failure(
        output_root,
        reservation,
        reservation_raw,
        authorization=_failure_authorization(),
        error=RuntimeError("synthetic failure after complete files"),
        progress=progress,
    )
    failure = module.contract.parse_canonical_json(
        (output_root / "failure.json").read_bytes(),
        name="synthetic failure receipt",
    )
    inventory = failure[
        "exact_partial_inventory_before_failure_receipt"
    ]
    bindings = {
        binding["path"]: binding
        for binding in inventory["file_bindings"]
    }
    expected_raw = {
        "reservation.json": reservation_raw,
        "phase_a/metrics.json": (
            module.contract.canonical_json_bytes(metrics) + b"\n"
        ),
        "phase_a/checkpoints/update_100.pt": checkpoint_raw,
        "phase_a/training_trace.jsonl": trace_raw,
    }
    assert set(bindings) == set(expected_raw)
    for path, raw in expected_raw.items():
        assert bindings[path]["byte_count"] == len(raw)
        assert bindings[path]["file_sha256"] == hashlib.sha256(raw).hexdigest()
        assert bindings[path]["filesystem_fingerprint"]["size"] == len(raw)
        assert set(bindings[path]["filesystem_fingerprint"]) == {
            "device",
            "inode",
            "mode",
            "size",
            "mtime_ns",
            "ctime_ns",
        }
    assert bindings["reservation.json"]["content_sha256"] == (
        reservation["content_sha256"]
    )
    assert bindings["phase_a/metrics.json"]["content_sha256"] == (
        metrics["content_sha256"]
    )
    checkpoint_binding = bindings["phase_a/checkpoints/update_100.pt"]
    assert checkpoint_binding["content_sha256"] == (
        checkpoint_content_sha256
    )
    assert checkpoint_binding["state_sha256"] == checkpoint_state_sha256
    assert checkpoint_binding["phase"] == "phase_a"
    assert checkpoint_binding["update"] == 100
    assert checkpoint_binding["schedule_prefix_sha256"] == (
        module.contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[100]
    )
    trace_binding = bindings["phase_a/training_trace.jsonl"]
    assert trace_binding["content_sha256"] == trace_content_sha256
    assert trace_binding["row_count"] == 1
    assert failure["normal_receipts_present"] == [
        "reservation.json",
        "phase_a/metrics.json",
    ]
    assert failure["normal_receipt_bindings_present"] == [
        bindings["reservation.json"],
        bindings["phase_a/metrics.json"],
    ]

    for current, directories, filenames in module.os.walk(
        output_root,
        topdown=False,
    ):
        for filename in filenames:
            module.os.chmod(Path(current) / filename, 0o644)
        for directory in directories:
            module.os.chmod(Path(current) / directory, 0o755)
    module.os.chmod(output_root, 0o755)


def test_access_counter_order_and_failure_status_chain_are_exact() -> None:
    module = _load_runner("_jepa_encoder_v9_lifecycle_contract")
    counters = {name: 0 for name in module.contract.ACCESS_ZERO_COUNTER_FIELDS}
    assert module.contract.validate_access_zero_counters(counters) == counters
    for control in module.contract.PHASE_A_FAILURE_CONTROLS:
        chain = {
            "metrics": control,
            "artifact": control,
            "result": control,
            "completion": control,
        }
        assert (
            module.contract.validate_phase_a_failure_status_chain(chain)
            == chain
        )
