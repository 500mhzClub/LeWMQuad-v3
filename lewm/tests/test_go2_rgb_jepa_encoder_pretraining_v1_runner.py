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


def _load_runner(name: str = "_jepa_encoder_v10_runner_test"):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_import_is_source_only_and_bound_to_v10() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_source_only_runner", {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert module.PREFLIGHT_ENVIRONMENT_KEY == (
    "LEWM_RGB_ACTION_CONDITIONED_NEXT_TARGET_RETRIEVAL_"
    "JEPA_V10_PREFLIGHT_JSON"
)
assert module.contract.PREREGISTRATION_COMMIT == (
    "25b93c92fbfb2816d52f0dfc27603c759e7c3c68"
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


def test_runner_source_contains_only_v10_phase_a_mechanism() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for removed in (
        "ActionConditionedLocalCorrespondenceTransport",
        "predict_action_conditioned_local_transports",
        "local_correspondence",
        "all_candidate_correspondence",
        "transport_weight",
        "DensePairwiseSpatialCostVolumeInverseHead",
        "dense_pairwise_spatial_cost_volume_inverse_terms",
        "dense_pairwise_inverse_head",
        "action_indexed_energy_nll",
    ):
        assert removed not in source
    assert "ActionConditionedLatentFlow" in source
    assert "predict_action_conditioned_flow_warps" in source
    assert "action_conditioned_next_target_retrieval_terms" in source
    assert "factorized_retrieval_state_transfer_count" in source
    assert "v9" not in source.casefold()

    module = _load_runner("_jepa_encoder_v10_phase_b_scope")
    assert all(
        "retrieval" not in prefix
        for prefix in module.contract.PHASE_B_FROZEN_PARAMETER_PREFIXES
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
        def __init__(self) -> None:
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
            for frozen in (
                self.target_encoder,
                self.target_geometry_module,
                self.target_projector,
            ):
                frozen.requires_grad_(False)

    return Model


def _objective_ops(torch, actual_ops):
    def predict(_predictor, _projector, state, action, _target_current):
        normalized = actual_ops.normalize_spatial_tokens(state)
        offsets = torch.linspace(
            -0.04,
            0.04,
            9,
            dtype=state.dtype,
            device=state.device,
        )[None, :, None, None]
        direction = torch.linspace(
            -1.0,
            1.0,
            192,
            dtype=state.dtype,
            device=state.device,
        )[None, None, None, :]
        all_predictions = actual_ops.normalize_spatial_tokens(
            normalized[:, None] + offsets * direction
        )
        requested = action.argmax(dim=1)
        executed = all_predictions.gather(
            1,
            requested[:, None, None, None].expand(-1, 1, 256, 192),
        ).squeeze(1)
        candidates = torch.stack([
            torch.tensor(
                [index for index in range(9) if index != int(row)],
                dtype=torch.long,
                device=state.device,
            )
            for row in requested
        ])
        return SimpleNamespace(
            executed_indices=requested,
            executed=executed,
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

    def whitening(value):
        return SimpleNamespace(
            variance=value.square().mean() * 0.001,
            covariance=value.mean().square() * 0.001,
        )

    return SimpleNamespace(
        normalize_spatial_tokens=actual_ops.normalize_spatial_tokens,
        predict_action_conditioned_flow_warps=predict,
        action_conditioned_next_target_retrieval_terms=(
            actual_ops.action_conditioned_next_target_retrieval_terms
        ),
        normalized_token_l2_energy=actual_ops.normalized_token_l2_energy,
        patch_whitening_terms=whitening,
    )


def test_v10_objective_routes_only_current_rgb_through_online_encoder() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(44)
    module = _load_runner("_jepa_encoder_v10_objective")
    actual_ops = module._phase_a_ops()
    model = _objective_model(torch)()
    ops = _objective_ops(torch, actual_ops)
    current = torch.randn(3, 3, 16, 16, requires_grad=True)
    next_rgb = torch.randn(3, 3, 16, 16, requires_grad=True)
    deranged_next_rgb = torch.randn(
        3, 3, 16, 16, requires_grad=True
    )
    action = torch.zeros(3, 9)
    action[torch.arange(3), torch.tensor([0, 6, 8])] = 1.0
    non_hold = torch.tensor([True, False, True])

    output = module._phase_a_current_only_loss(
        model,
        current,
        next_rgb,
        deranged_next_rgb,
        action,
        non_hold,
        ops=ops,
    )
    expected = (
        output["jepa_loss"]
        + output["action_retrieval_loss"]
        + output["target_retrieval_loss"]
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
    assert output["action_retrieval_energies"].shape == (3, 9)
    assert output["action_retrieval_logits"].shape == (3, 9)
    assert output["target_retrieval_energies"].shape == (3, 3)
    assert output["target_retrieval_logits"].shape == (3, 3)
    assert output["target_candidate_mask"].tolist() == [
        [True, True, True],
        [True, True, False],
        [True, True, True],
    ]
    assert torch.equal(
        output["action_retrieval_logits"],
        -output["action_retrieval_energies"],
    )
    assert len(model.encoder.seen) == 1
    assert torch.equal(model.encoder.seen[0], current.detach())
    assert len(model.target_encoder.seen) == 3
    assert torch.equal(model.target_encoder.seen[0], current.detach())
    assert torch.equal(model.target_encoder.seen[1], next_rgb.detach())
    assert torch.equal(
        model.target_encoder.seen[2], deranged_next_rgb.detach()
    )

    output["loss"].backward()
    assert current.grad is not None
    assert torch.count_nonzero(current.grad).item() > 0
    assert next_rgb.grad is None
    assert deranged_next_rgb.grad is None
    assert model.encoder.feature_scale.grad is not None
    assert torch.count_nonzero(model.encoder.feature_scale.grad).item() > 0
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


def test_phase_a_initialization_restores_v5_flow_without_new_v10_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v10_initialization")
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
    retrieval = receipt["factorized_retrieval_initialization"]
    assert retrieval["new_parameter_count"] == 0
    assert retrieval["new_initialization_draw_count"] == 0
    assert retrieval["temperature"] is None
    assert retrieval["row_scale"] is None
    assert retrieval["action_retrieval_coefficient"] == 1.0
    assert retrieval["target_retrieval_coefficient"] == 1.0
    assert not any(
        "retrieval" in name for name, _ in model.named_parameters()
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


def test_gate_references_use_exact_float32_494x2_and_495x9_formulas() -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v10_gate_references")
    pairs = [
        {"primitive": module.contract.ACTION_VOCABULARY[index % 9]}
        for index in range(495)
    ]
    mapping = {
        "same_action_eligible": (True,) * 494 + (False,),
    }
    observed = module._phase_a_gate_references(
        SimpleNamespace(torch=torch),
        pairs,
        mapping,
        torch.device("cpu"),
    )
    expected_action = torch.nn.functional.cross_entropy(
        torch.zeros((495, 9), dtype=torch.float32),
        torch.tensor([index % 9 for index in range(495)]),
    )
    expected_two = torch.nn.functional.cross_entropy(
        torch.zeros((494, 2), dtype=torch.float32),
        torch.zeros(494, dtype=torch.long),
    )
    assert observed == {
        "action_equal_logit_reference": float(expected_action),
        "two_target_equal_logit_reference": float(expected_two),
    }


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


def test_mapped_negative_requests_are_counted_separately_from_cache_io() -> None:
    module = _load_runner("_jepa_encoder_v10_candidate_counters")
    endpoint = "a" * 64
    loader = module.RGBOnlyLoader(
        SimpleNamespace(),
        SimpleNamespace(
            endpoints={
                endpoint: {"dataset_role": "train"},
            }
        ),
    )
    sentinel = object()
    loader.cache[endpoint] = sentinel
    assert loader.image(
        endpoint,
        role="train",
        stage="phase_a_gradient",
        mapped_negative_scope="training",
    ) is sentinel
    assert loader.image(
        endpoint,
        role="train",
        stage="phase_a_diagnostic_update_0",
        mapped_negative_scope="observation",
    ) is sentinel
    receipt = loader.mapped_negative_io_receipt()
    assert receipt["by_scope"]["training"] == {
        "endpoint_request_count": 1,
        "cache_hit_count": 1,
        "cache_miss_count": 0,
        "physical_read_attempt_count": 0,
        "physical_read_success_count": 0,
    }
    assert receipt["by_scope"]["observation"] == {
        "endpoint_request_count": 1,
        "cache_hit_count": 1,
        "cache_miss_count": 0,
        "physical_read_attempt_count": 0,
        "physical_read_success_count": 0,
    }
    assert receipt["total_endpoint_request_count"] == 2
    assert receipt["total_cache_hit_count"] == 2


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


def _per_action_retrieval(contract, nll: float, *, update_zero: bool):
    counts = {
        action: count
        for action, count in zip(
            contract.ACTION_VOCABULARY,
            (55, 55, 55, 54, 54, 54, 60, 54, 54),
            strict=True,
        )
    }
    if update_zero:
        correct = {
            action: (count if index == 0 else 0)
            for index, (action, count) in enumerate(counts.items())
        }
    else:
        correct = {
            action: (count + 1) // 2
            for action, count in counts.items()
        }
    per_action = {
        action: {
            "row_count": count,
            "mean_nll": nll,
            "recall": correct[action] / count,
        }
        for action, count in counts.items()
    }
    return (
        per_action,
        sum(correct.values()) / 495,
        sum(row["recall"] for row in per_action.values()) / 9,
    )


def _factorized(
    contract,
    *,
    action_nll: float,
    target_nll: float,
    two_target_nll: float,
    update_zero: bool,
):
    per_action, action_top1, macro = _per_action_retrieval(
        contract, action_nll, update_zero=update_zero
    )
    margin = 0.0 if update_zero else 0.2
    per_family = {
        family: {
            "scene_id":
                contract.SELECTION_FAMILY_BINDINGS[family]["scene_id"],
            "row_count":
                contract.SELECTION_FAMILY_BINDINGS[family]["row_count"],
            "same_action_row_count":
                contract.SELECTION_FAMILY_BINDINGS[family]
                ["same_action_row_count"],
            "non_hold_row_count":
                contract.SELECTION_FAMILY_BINDINGS[family]
                ["non_hold_row_count"],
            "deranged_minus_correct_energy": margin,
            "current_target_minus_correct_energy": margin,
            "cyclic_wrong_minus_executed_energy": margin,
            "hardest_wrong_minus_executed_energy": margin,
            "hold_minus_non_hold_executed_energy": margin,
            "permuted_minus_executed_energy": margin,
            "hold_action_rows_match_non_hold_rows": True,
        }
        for family in contract.SCENE_FAMILIES
    }
    correct = 1.0 if update_zero else 0.8
    control = 1.0
    strict_wins = 0 if update_zero else 300
    target_top1 = 495 if update_zero else 350
    positives = 0 if update_zero else 8
    return {
        "all_values_finite": True,
        "energy_values_within_closed_zero_four": True,
        "target_candidate_order_and_counts_exact": True,
        "same_action_target_mapping_exact": True,
        "selection_action_permutation_exact": True,
        "reference_values_immutable": True,
        "action_equal_logit_reference": math.log(9.0),
        "two_target_equal_logit_reference": math.log(2.0),
        "action_retrieval_nll": action_nll,
        "action_retrieval_top1_accuracy": action_top1,
        "per_executed_action_action_retrieval": per_action,
        "action_retrieval_macro_balanced_accuracy": macro,
        "target_retrieval_nll": target_nll,
        "same_action_target_retrieval_nll": target_nll,
        "hold_target_retrieval_nll": target_nll,
        "non_hold_target_retrieval_nll": target_nll,
        "same_action_two_target_nll": two_target_nll,
        "target_retrieval_top1_count": target_top1,
        "target_retrieval_top1_accuracy": target_top1 / 495,
        "same_action_strict_win_count": strict_wins,
        "same_action_strict_win_rate": strict_wins / 494,
        "same_action_correct_energy": correct,
        "same_action_deranged_energy": control,
        "same_action_correct_to_deranged_ratio": correct / control,
        "non_hold_correct_energy": correct,
        "non_hold_current_target_energy": control,
        "non_hold_correct_to_current_ratio": correct / control,
        "executed_action_energy": correct,
        "cyclic_wrong_action_energy": control,
        "hardest_wrong_action_energy": control,
        "permuted_action_energy": control,
        "non_hold_executed_action_energy": correct,
        "non_hold_hold_action_energy": control,
        "executed_to_cyclic_ratio": correct / control,
        "executed_to_hardest_wrong_ratio": correct / control,
        "executed_to_permuted_ratio": correct / control,
        "non_hold_executed_to_hold_ratio": correct / control,
        "all_row_count": 495,
        "same_action_row_count": 494,
        "fallback_row_count": 1,
        "hold_row_count": 60,
        "non_hold_row_count": 435,
        "target_candidate_count": 1_425,
        "action_candidate_count": 9,
        "all_wrong_action_candidate_count": 495 * 8,
        "selection_target_mapping_sha256":
            contract.TARGET_MAPPING_BINDINGS[
                "checkpoint_selection"
            ]["mapping_sha256"],
        "selection_action_permutation_sha256":
            contract.SELECTION_ACTION_PERMUTATION_BINDING[
                "mapping_sha256"
            ],
        "per_family": per_family,
        "deranged_positive_family_margin_count": positives,
        "current_target_positive_family_margin_count": positives,
        "cyclic_positive_family_margin_count": positives,
        "hold_positive_family_margin_count": positives,
        "permuted_positive_family_margin_count": positives,
    }


def _metric(
    contract,
    *,
    action_nll: float,
    target_nll: float,
    two_target_nll: float,
):
    return {
        "all_values_finite": True,
        "ema_target_gradient_free": True,
        "pair_count": 495,
        "scene_family_count": 8,
        "centered_raw_patch_effective_rank": 50.0,
        "centered_projected_target_effective_rank": 50.0,
        "raw_cross_sample_variance": 1.0,
        "content_residual_spatial_diversity": 1.0,
        "true_pair_mse": 0.8,
        "shuffled_next_mse": 1.0,
        "mean_target_mse": 1.0,
        "non_hold_pair_count": 435,
        "shuffled_current_mse": 1.0,
        "latent_flow": _latent_flow(contract, update_zero=False),
        "factorized_retrieval": _factorized(
            contract,
            action_nll=action_nll,
            target_nll=target_nll,
            two_target_nll=two_target_nll,
            update_zero=False,
        ),
    }


def _update0(contract):
    return {
        "raw_cross_sample_variance": 1.0,
        "content_residual_spatial_diversity": 1.0,
        "all_action_predictions_bitwise_equal": True,
        "all_action_unordered_pair_count": 36,
        "all_action_prediction_row_count": 495,
        "latent_flow": _latent_flow(contract, update_zero=True),
        "factorized_retrieval": _factorized(
            contract,
            action_nll=math.log(9.0),
            target_nll=contract.SELECTION_EQUAL_LOGIT_REFERENCE_BINARY64,
            two_target_nll=math.log(2.0),
            update_zero=True,
        ),
    }


def test_staged_gates_compare_100_to_400_to_1000_without_retry() -> None:
    module = _load_runner("_jepa_encoder_v10_staged_gates")
    contract = module.contract
    integrity = {"rng_state_preserved": True, "state_mutation_count": 0}
    update0 = _update0(contract)
    metric100 = _metric(
        contract,
        action_nll=1.9,
        target_nll=0.6,
        two_target_nll=0.6,
    )
    gate100 = contract.evaluate_phase_a_continuation(
        100, metric100, update0, integrity, None
    )
    assert gate100["passed"] is True

    metric400 = _metric(
        contract,
        action_nll=1.8,
        target_nll=0.5,
        two_target_nll=0.5,
    )
    gate400 = contract.evaluate_phase_a_continuation(
        400, metric400, update0, integrity, metric100
    )
    assert gate400["passed"] is True

    metric1000 = _metric(
        contract,
        action_nll=1.7,
        target_nll=0.4,
        two_target_nll=0.4,
    )
    terminal = contract.evaluate_phase_a(
        metric1000, update0, integrity, metric400
    )
    assert terminal["passed"] is True
    assert terminal["control"] == contract.CONTROL_PHASE_A_PASS

    no_improvement = _metric(
        contract,
        action_nll=1.9,
        target_nll=0.6,
        two_target_nll=0.6,
    )
    failed = contract.evaluate_phase_a_continuation(
        400, no_improvement, update0, integrity, metric100
    )
    assert failed["passed"] is False
    assert failed["control"] == contract.CONTROL_PHASE_A_UPDATE_400_FAIL
    assert module.contract.MAXIMUM_ATTEMPTS == 1


def test_constant_query_cyclic_same_action_targets_stay_at_exact_log2(
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v10_constant_query_shortcut")
    ops = module._phase_a_ops()
    query = torch.zeros((4, 256, 192), dtype=torch.float32)
    query[..., 0] = 1.0
    positive = torch.zeros_like(query)
    for row, feature in enumerate((1, 2, 3, 4)):
        positive[row, :, feature] = 1.0
    cyclic_negative = positive.roll(shifts=1, dims=0)
    positive_energy = ops.normalized_token_l2_energy(query, positive)
    negative_energy = ops.normalized_token_l2_energy(
        query, cyclic_negative
    )
    assert torch.equal(positive_energy, negative_energy)
    logits = -torch.stack((positive_energy, negative_energy), dim=1)
    labels = torch.zeros(4, dtype=torch.long)
    nll = torch.nn.functional.cross_entropy(
        logits, labels, reduction="none"
    )
    exact_log2 = torch.log(torch.tensor(2.0, dtype=torch.float32))
    assert torch.equal(nll, exact_log2.expand_as(nll))
    assert int((positive_energy < negative_energy).sum()) == 0

    metrics = _metric(
        module.contract,
        action_nll=1.9,
        target_nll=0.6,
        two_target_nll=float(nll.mean()),
    )
    retrieval = metrics["factorized_retrieval"]
    retrieval["same_action_strict_win_count"] = 0
    retrieval["same_action_strict_win_rate"] = 0.0
    gate = module.contract.evaluate_phase_a_continuation(
        100,
        metrics,
        _update0(module.contract),
        {"rng_state_preserved": True, "state_mutation_count": 0},
    )
    assert gate["passed"] is False
    assert gate["conjuncts"][
        "same_action_two_target_nll_strictly_below_reference"
    ] is False
    assert gate["conjuncts"][
        "same_action_strict_win_rate_strictly_above_half"
    ] is False


def test_action_ignoring_energies_make_four_exact_unit_ratios_and_fail_100(
) -> None:
    torch = pytest.importorskip("torch")
    module = _load_runner("_jepa_encoder_v10_action_ignoring_shortcut")
    ops = module._phase_a_ops()
    rows = 4
    query = torch.zeros((rows, 256, 192), dtype=torch.float32)
    query[..., 0] = 1.0
    target = torch.zeros_like(query)
    target[..., 1] = 1.0
    all_predictions = query[:, None].expand(-1, 9, -1, -1)
    all_targets = target[:, None].expand_as(all_predictions)
    energies = ops.normalized_token_l2_energy(
        all_predictions, all_targets
    )
    executed_indices = torch.tensor([0, 1, 6, 8], dtype=torch.long)
    row_indices = torch.arange(rows)
    executed = energies[row_indices, executed_indices]
    cyclic = energies[row_indices, (executed_indices + 1) % 9]
    hardest_wrong = energies[row_indices, (executed_indices + 2) % 9]
    permuted = energies[row_indices, (executed_indices + 3) % 9]
    non_hold = executed_indices.ne(module.contract.HOLD_ACTION_INDEX)
    hold = energies[non_hold, module.contract.HOLD_ACTION_INDEX]
    ratios = (
        float(executed.mean() / cyclic.mean()),
        float(executed.mean() / hardest_wrong.mean()),
        float(executed.mean() / permuted.mean()),
        float(executed[non_hold].mean() / hold.mean()),
    )
    assert ratios == (1.0, 1.0, 1.0, 1.0)

    metrics = _metric(
        module.contract,
        action_nll=math.log(9.0),
        target_nll=module.contract.SELECTION_EQUAL_LOGIT_REFERENCE_BINARY64,
        two_target_nll=math.log(2.0),
    )
    metrics["factorized_retrieval"] = _factorized(
        module.contract,
        action_nll=math.log(9.0),
        target_nll=module.contract.SELECTION_EQUAL_LOGIT_REFERENCE_BINARY64,
        two_target_nll=math.log(2.0),
        update_zero=True,
    )
    retrieval = metrics["factorized_retrieval"]
    assert tuple(
        retrieval[field]
        for field in (
            "executed_to_cyclic_ratio",
            "executed_to_hardest_wrong_ratio",
            "executed_to_permuted_ratio",
            "non_hold_executed_to_hold_ratio",
        )
    ) == ratios
    gate = module.contract.evaluate_phase_a_continuation(
        100,
        metrics,
        _update0(module.contract),
        {"rng_state_preserved": True, "state_mutation_count": 0},
    )
    assert gate["passed"] is False
    assert gate["control"] == module.contract.CONTROL_PHASE_A_UPDATE_100_FAIL


def test_train_wires_previous_metrics_and_exact_phase_a_status() -> None:
    module = _load_runner("_jepa_encoder_v10_train_wiring")
    source = inspect.getsource(module._phase_a_train)
    assert source.count("_phase_a_gate_references(") == 1
    assert "train_target_mapping[\"negative_indices\"]" in source
    assert "mapped_negative_scope=\"training\"" in source
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


def test_terminal_authority_rehash_rereads_both_files_and_binds_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_jepa_encoder_v10_terminal_authority_rehash")
    monkeypatch.setattr(module, "ROOT", tmp_path)

    manifest = module.contract.with_content_sha256({"kind": "manifest"})
    review = module.contract.with_content_sha256({
        "kind": "review",
        "reviewer": "/root/synthetic_reviewer",
    })
    authorization = module.contract.with_content_sha256({
        "kind": "authorization",
    })
    manifest_raw = module.contract.canonical_json_bytes(manifest) + b"\n"
    review_raw = module.contract.canonical_json_bytes(review) + b"\n"
    authorization_raw = (
        module.contract.canonical_json_bytes(authorization) + b"\n"
    )
    paths_and_raw = {
        module.contract.SOURCE_MANIFEST_RELATIVE_PATH: manifest_raw,
        module.contract.REVIEW_RELATIVE_PATH: review_raw,
        module.contract.AUTHORIZATION_RELATIVE_PATH: authorization_raw,
    }
    for relative, raw in paths_and_raw.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def accept_review(value, *, expected_sources, source_manifest_raw):
        assert expected_sources == sources
        assert source_manifest_raw == manifest_raw
        return value

    def accept_authorization(value, *, review_binding, reviewer):
        assert review_binding["file_sha256"] == hashlib.sha256(
            review_raw
        ).hexdigest()
        assert reviewer == review["reviewer"]
        return value

    sources = {
        module.contract.SOURCE_MANIFEST_RELATIVE_PATH:
            hashlib.sha256(manifest_raw).hexdigest(),
    }
    monkeypatch.setattr(module.contract, "validate_review", accept_review)
    monkeypatch.setattr(
        module.contract,
        "validate_authorization",
        accept_authorization,
    )
    observed = module._terminal_authority_rehash(
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
    )
    assert set(observed) == {"source_review", "execution_authorization"}
    assert all(
        binding["exact_pre_reservation_bytes_match"] is True
        and binding["observed_file_sha256"] == binding["file_sha256"]
        and binding["observed_byte_count"] == binding["byte_count"]
        for binding in observed.values()
    )
    execute_source = inspect.getsource(module._execute_after_reservation)
    assert execute_source.index("authority_rehash =") < execute_source.index(
        "access, access_raw ="
    )
    assert '"source_authority_rehash": authority_rehash' in execute_source

    for relative, original_raw in (
        (module.contract.REVIEW_RELATIVE_PATH, review_raw),
        (module.contract.AUTHORIZATION_RELATIVE_PATH, authorization_raw),
    ):
        path = tmp_path / relative
        path.write_bytes(original_raw + b" ")
        with pytest.raises(PermissionError, match="input (hash|byte count) changed"):
            module._terminal_authority_rehash(
                review=review,
                review_raw=review_raw,
                authorization=authorization,
                authorization_raw=authorization_raw,
                sources=sources,
            )
        path.write_bytes(original_raw)


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
    assert receipt["phase_a_loader"]["mapped_negative_io"][
        "total_endpoint_request_count"
    ] == 0
    assert receipt["operation_counts"]["phase_a_mapped_negative_io"] == (
        receipt["phase_a_loader"]["mapped_negative_io"]
    )
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
