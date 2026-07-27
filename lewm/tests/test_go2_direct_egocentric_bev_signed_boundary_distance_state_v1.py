from __future__ import annotations

import importlib.util
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
STEM = "go2_direct_egocentric_bev_signed_boundary_distance_state_v1"
CONTRACT = ROOT / "lewm/benchmarks" / f"{STEM}.py"
MODEL = ROOT / "lewm/models/direct_egocentric_bev_signed_boundary_distance_state_v1.py"
RUNNER = ROOT / "scripts" / f"run_{STEM}.py"
LAUNCHER = ROOT / "scripts" / f"launch_{STEM}.py"
CHECKER = ROOT / "scripts" / f"check_{STEM}_source_closure.py"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tensor_state_sha256(torch: Any, state: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _common_gate_metrics(contract: Any, update: int) -> dict[str, Any]:
    return {
        **contract.PERCEPTION_ACCOUNTING[update],
        "signed_boundary_distance_mechanism_receipt_ready": True,
        "active_training_scope_signed_boundary_distance_v1": "perception_only",
        "all_registered_values_finite": True,
        "state_nonconstant": True,
        "all_forbidden_access_counts_zero": True,
    }


@pytest.mark.parametrize("source", [CONTRACT, RUNNER, LAUNCHER, CHECKER])
def test_control_sources_import_without_tensor_or_image_runtime(source: Path) -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(source)!r})
spec = importlib.util.spec_from_file_location('_signed_distance_source_only', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name.startswith('torch.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert 'PIL' not in sys.modules
print('PASS')
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


def test_contract_inventory_caps_source_closure_and_exact_gates() -> None:
    contract = _load(CONTRACT, "_signed_distance_contract")
    checker = _load(CHECKER, "_signed_distance_checker")
    assert contract.MODEL_BINDINGS_FROZEN is True
    assert contract.SIGNED_BOUNDARY_DISTANCE_INITIAL_HEAD_STATE_SHA256 == (
        "a3582ca41e41963592f4bf76ba7de432b51fed783408dcda3e3de9c070c9f40f"
    )
    assert contract.MODEL_PARAMETER_INVENTORY["decoder_state"] == {
        "parameter_count": 87_746,
        "tensor_count": 32,
        "ordered_parameter_name_sha256": (
            "93facb8b8d4059e7270ebb90dbff26572c6f1700bb302d8a2b7177bb5777c147"
        ),
    }
    assert contract.MODEL_PARAMETER_INVENTORY["total"] == {
        "parameter_count": 5_987_639,
        "tensor_count": 299,
    }
    assert contract.MAXIMUM_ATTEMPTS == 1
    assert contract.MAXIMUM_UPDATES == 1_000
    assert contract.MAXIMUM_PRESENTATIONS == 16_000
    assert contract.EXECUTION_AUTHORITY["maximum_updates"] == 1_000
    assert contract.EXECUTION_AUTHORITY["maximum_presentations"] == 16_000
    assert not contract.EXECUTION_AUTHORITY[
        "learned_bev_query_prototype_perception_only"
    ]
    assert not contract.EXECUTION_AUTHORITY[
        "final_class_macro_grounding_perception_only"
    ]
    assert contract.OBSERVATION_UPDATES == (0, 100, 400, 1_000)
    assert contract.SCHEDULE_PREFIX_SHA256 == {
        100: "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
        400: "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
        1_000: "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
    }
    assert len(contract.REUSED_SOURCE_PATHS) == 143
    assert len(contract.ADDITIVE_SOURCE_PATHS) == 6
    assert len(contract.SOURCE_PATHS) == 149
    manifest = checker.build_manifest()
    assert manifest["source_count"] == 149
    assert manifest["source_paths"] == list(contract.SOURCE_PATHS)
    assert manifest["generated_input_open_count"] == 0
    assert manifest["checkpoint_or_tensor_open_count"] == 0
    assert manifest["sealed_or_heldout_open_count"] == 0

    for update in contract.OBSERVATION_UPDATES:
        preliminary = contract.evaluate_gate(update, {})
        assert preliminary["passed"] is True
        assert preliminary["scientific_gate_evidence"] is False

    zero = {
        **_common_gate_metrics(contract, 0),
        **{field: True for field in contract.INTEGRITY_FIELDS},
        "initial_online_to_target_hard_sync_count": 1,
        "G": 1.0,
        "aggregate_raster_nll": 1.0,
        "aggregate_raster_balanced_accuracy": 0.40,
    }
    assert "correct_rgb_scene_win_count" not in zero
    assert contract.evaluate_gate(0, zero)["passed"] is True

    hundred = {
        **_common_gate_metrics(contract, 100),
        "G": 0.80,
        "aggregate_raster_nll": 1.0 - 0.15,
        "aggregate_raster_balanced_accuracy": 0.40 + 0.08,
        "aggregate_free_recall": 0.10,
        "aggregate_occupied_recall": 0.10,
        "rough_raster_balanced_accuracy": 0.50,
        "rough_raster_occupied_recall": 0.20,
        "paired_rgb_aggregate_margin": 1e-9,
        "correct_rgb_scene_win_count": 6,
    }
    assert contract.evaluate_gate(100, hundred, update_zero=zero)["passed"] is True

    four_hundred = {
        **_common_gate_metrics(contract, 400),
        "G": 0.70,
        "aggregate_raster_nll": 0.55,
        "aggregate_raster_balanced_accuracy": 0.68,
        "aggregate_free_recall": 0.50,
        "aggregate_occupied_recall": 0.55,
        "rough_raster_balanced_accuracy": 0.65,
        "rough_raster_occupied_recall": 0.50,
        "paired_rgb_aggregate_margin": 1e-9,
        "correct_rgb_scene_win_count": 7,
    }
    assert contract.evaluate_gate(
        400, four_hundred, update_100=hundred
    )["passed"] is True

    thousand = {
        **_common_gate_metrics(contract, 1_000),
        "G": 0.60,
        "aggregate_raster_nll": 0.42,
        "aggregate_raster_balanced_accuracy": 0.80,
        "aggregate_unknown_recall": 0.80,
        "aggregate_free_recall": 0.68,
        "aggregate_occupied_recall": 0.88,
        "rough_raster_balanced_accuracy": 0.772,
        "rough_raster_occupied_recall": 0.65,
        "paired_rgb_aggregate_margin": 1e-9,
        "correct_rgb_scene_win_count": 7,
    }
    assert contract.evaluate_gate(
        1_000, thousand, update_400=four_hundred
    )["passed"] is True
    assert contract.evaluate_gate(
        1_000,
        {**thousand, "paired_rgb_aggregate_margin": 0.0},
        update_400=four_hundred,
    )["passed"] is False

    epsilon = 1e-7
    for mutation in (
        {"G": zero["G"]},
        {"aggregate_raster_nll": zero["aggregate_raster_nll"] - 0.15 + epsilon},
        {
            "aggregate_raster_balanced_accuracy": (
                zero["aggregate_raster_balanced_accuracy"] + 0.08 - epsilon
            )
        },
        {"aggregate_free_recall": 0.10 - epsilon},
        {"aggregate_occupied_recall": 0.10 - epsilon},
        {"paired_rgb_aggregate_margin": 0.0},
        {"correct_rgb_scene_win_count": 5},
    ):
        assert not contract.evaluate_gate(
            100, {**hundred, **mutation}, update_zero=zero
        )["passed"]

    for mutation in (
        {"G": hundred["G"]},
        {"aggregate_raster_nll": 0.55 + epsilon},
        {"aggregate_raster_balanced_accuracy": 0.68 - epsilon},
        {"aggregate_free_recall": 0.50 - epsilon},
        {"aggregate_occupied_recall": 0.55 - epsilon},
        {"aggregate_free_recall": 0.50, "aggregate_occupied_recall": 0.86},
        {"rough_raster_balanced_accuracy": 0.65 - epsilon},
        {"rough_raster_occupied_recall": 0.50 - epsilon},
        {"paired_rgb_aggregate_margin": 0.0},
        {"correct_rgb_scene_win_count": 6},
    ):
        assert not contract.evaluate_gate(
            400, {**four_hundred, **mutation}, update_100=hundred
        )["passed"]
    relative_hundred = {
        **hundred,
        "aggregate_raster_nll": 0.50,
        "aggregate_raster_balanced_accuracy": 0.80,
        "rough_raster_balanced_accuracy": 0.75,
        "rough_raster_occupied_recall": 0.60,
    }
    relative_four_hundred = {
        **four_hundred,
        "aggregate_raster_nll": relative_hundred["aggregate_raster_nll"] - 0.03,
        "aggregate_raster_balanced_accuracy": (
            relative_hundred["aggregate_raster_balanced_accuracy"] + 0.03
        ),
        "rough_raster_balanced_accuracy": (
            relative_hundred["rough_raster_balanced_accuracy"] + 0.02
        ),
        "rough_raster_occupied_recall": (
            relative_hundred["rough_raster_occupied_recall"] + 0.05
        ),
    }
    assert contract.evaluate_gate(
        400, relative_four_hundred, update_100=relative_hundred
    )["passed"]
    for field, direction in (
        ("aggregate_raster_nll", 1.0),
        ("aggregate_raster_balanced_accuracy", -1.0),
        ("rough_raster_balanced_accuracy", -1.0),
        ("rough_raster_occupied_recall", -1.0),
    ):
        assert not contract.evaluate_gate(
            400,
            {
                **relative_four_hundred,
                field: relative_four_hundred[field] + direction * epsilon,
            },
            update_100=relative_hundred,
        )["passed"]

    for mutation in (
        {"G": four_hundred["G"]},
        {"aggregate_raster_nll": 0.42 + epsilon},
        {"aggregate_raster_balanced_accuracy": 0.80 - epsilon},
        {"aggregate_unknown_recall": 0.80 - epsilon},
        {"aggregate_free_recall": 0.68 - epsilon},
        {"aggregate_occupied_recall": 0.88 - epsilon},
        {"aggregate_free_recall": 0.68, "aggregate_occupied_recall": 0.94},
        {"rough_raster_balanced_accuracy": 0.772 - epsilon},
        {"rough_raster_occupied_recall": 0.65 - epsilon},
        {"paired_rgb_aggregate_margin": 0.0},
        {"correct_rgb_scene_win_count": 6},
    ):
        assert not contract.evaluate_gate(
            1_000, {**thousand, **mutation}, update_400=four_hundred
        )["passed"]
    late_reference = {
        **four_hundred,
        "aggregate_raster_nll": 0.30,
        "aggregate_raster_balanced_accuracy": 0.85,
    }
    late_pass = {
        **thousand,
        "aggregate_raster_nll": 0.30,
        "aggregate_raster_balanced_accuracy": 0.85,
    }
    assert contract.evaluate_gate(
        1_000, late_pass, update_400=late_reference
    )["passed"]
    for field, threshold in (
        ("aggregate_raster_nll", 0.30 + epsilon),
        ("aggregate_raster_balanced_accuracy", 0.85 - epsilon),
    ):
        assert not contract.evaluate_gate(
            1_000,
            {**late_pass, field: threshold},
            update_400=late_reference,
        )["passed"]


def test_runner_three_seams_preserve_v9_custody_and_synthetic_mechanism() -> None:
    runner = _load(RUNNER, "_signed_distance_runner_topology")
    launcher = _load(LAUNCHER, "_signed_distance_launcher_topology")
    runner._assert_signed_boundary_bindings()
    launcher._assert_signed_boundary_bindings()
    assert tuple(name for name, _ in runner._SIGNED_BOUNDARY_SEAM_TABLE) == (
        "_gradient_integrity_probe",
        "_evaluate_observation_impl",
        "_load_schedule",
    )
    assert runner._LEAF._snapshot_model is runner._V9._v9_snapshot_model
    assert runner._LEAF._terminal_failure is runner._V9._v9_terminal_failure
    assert runner._LEAF._load_schedule is runner._signed_boundary_load_schedule
    assert all(
        name == runner.SIGNED_BOUNDARY_MODEL_RUNTIME_MODULE_NAME
        for name in runner._runtime_module_names()
    )
    model_api = _load(MODEL, "_signed_distance_runner_synthetic_model")
    receipt = runner._synthetic_mechanism_receipt(
        SimpleNamespace(torch=model_api.torch), model_api
    )
    assert receipt
    assert all(receipt.values()), receipt


def _slow_targets(torch: Any, labels: Any) -> tuple[Any, Any]:
    result = torch.zeros(
        labels.shape[0], 2, labels.shape[1], labels.shape[2], dtype=torch.float64
    )
    masks = torch.zeros_like(result, dtype=torch.bool)
    for batch in range(labels.shape[0]):
        for row in range(labels.shape[1]):
            for column in range(labels.shape[2]):
                state = int(labels[batch, row, column])
                masks[batch, 0, row, column] = True
                masks[batch, 1, row, column] = state != 0
                for channel in range(2):
                    if channel == 1 and state == 0:
                        continue
                    positive = state != 0 if channel == 0 else state == 1
                    opposite = []
                    for source_row in range(labels.shape[1]):
                        for source_column in range(labels.shape[2]):
                            source = int(labels[batch, source_row, source_column])
                            is_opposite = (
                                (source == 0) != (state == 0)
                                if channel == 0
                                else source in (1, 2) and source != state
                            )
                            if is_opposite:
                                opposite.append((source_row, source_column))
                    if opposite:
                        distance = min(
                            math.hypot(row - sr, column - sc)
                            for sr, sc in opposite
                        )
                        magnitude = min(1.0, max(0.0, distance - 0.5) / 8.0)
                    else:
                        magnitude = 1.0
                    result[batch, channel, row, column] = (
                        magnitude if positive else -magnitude
                    )
    return result, masks


def test_exact_half_cell_center_distance_targets_and_masks() -> None:
    api = _load(MODEL, "_signed_distance_target_math")
    torch = api.torch
    labels = torch.tensor(
        [
            [
                [0, 1, 1, 1, 1],
                [1, 1, 2, 1, 1],
                [1, 2, 2, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ]
        ],
        dtype=torch.uint8,
    )
    observed, masks = api.signed_boundary_distance_targets_v1(labels)
    expected, expected_masks = _slow_targets(torch, labels)
    assert observed.dtype == torch.float32
    torch.testing.assert_close(
        observed.cpu(), expected.to(dtype=observed.dtype), rtol=0.0, atol=1e-7
    )
    assert torch.equal(masks.cpu(), expected_masks)
    assert float(observed[0, 0, 0, 0]) == -1.0 / 16.0
    assert float(observed[0, 0, 1, 1]) == pytest.approx(
        (math.sqrt(2.0) - 0.5) / 8.0, abs=1e-7
    )
    assert float(observed[0, 1, 1, 1]) == 1.0 / 16.0
    assert float(observed[0, 1, 1, 2]) == -1.0 / 16.0
    assert float(observed[0, 1, 0, 0]) == 0.0

    all_unknown = torch.zeros((1, 4, 4), dtype=torch.long)
    unknown_fields, unknown_masks = api.signed_boundary_distance_targets_v1(
        all_unknown
    )
    assert bool((unknown_fields[:, 0] == -1.0).all())
    assert bool((unknown_fields[:, 1] == 0.0).all())
    assert bool(unknown_masks[:, 0].all())
    assert not bool(unknown_masks[:, 1].any())

    all_free = torch.ones((1, 4, 4), dtype=torch.long)
    free_fields, free_masks = api.signed_boundary_distance_targets_v1(all_free)
    assert bool((free_fields == 1.0).all())
    assert bool(free_masks.all())

    saturation = torch.ones((1, 10, 5), dtype=torch.long)
    saturation[0, 0, 0] = 0
    saturation_fields, _ = api.signed_boundary_distance_targets_v1(saturation)
    assert float(saturation_fields[0, 0, 8, 2]) == pytest.approx(
        (math.sqrt(68.0) - 0.5) / 8.0, abs=1e-7
    )
    assert float(saturation_fields[0, 0, 8, 3]) == 1.0


def _manual_huber(error: Any, delta: float) -> Any:
    absolute = error.abs()
    return error.square().mul(0.5).where(
        absolute < delta,
        delta * (absolute - 0.5 * delta),
    )


def test_per_row_sign_macro_huber_and_unknown_o_gradient_mask() -> None:
    api = _load(MODEL, "_signed_distance_huber_math")
    torch = api.torch
    labels = torch.tensor(
        [
            [[0, 0, 1], [1, 2, 2]],
            [[0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 1, 1]],
        ],
        dtype=torch.long,
    )
    targets, _ = api.signed_boundary_distance_targets_v1(labels)
    predicted = torch.linspace(
        -0.75, 0.85, targets.numel(), dtype=torch.float32
    ).reshape_as(targets).requires_grad_()
    observed = api._boundary_huber_per_row_v1(predicted, targets, labels)
    pointwise = _manual_huber(
        predicted - targets.to(dtype=predicted.dtype, device=predicted.device),
        0.125,
    )
    expected_rows = []
    for row in range(labels.shape[0]):
        k_groups = []
        if bool((labels[row] == 0).any()):
            k_groups.append(pointwise[row, 0][labels[row] == 0].mean())
        if bool((labels[row] != 0).any()):
            k_groups.append(pointwise[row, 0][labels[row] != 0].mean())
        k_macro = torch.stack(k_groups).mean()
        o_groups = [
            pointwise[row, 1][labels[row] == state].mean()
            for state in (1, 2)
            if bool((labels[row] == state).any())
        ]
        expected_rows.append(
            0.5 * (k_macro + torch.stack(o_groups).mean())
            if o_groups
            else k_macro
        )
    torch.testing.assert_close(observed, torch.stack(expected_rows))
    observed.sum().backward()
    assert bool((predicted.grad[:, 1][labels == 0] == 0.0).all())
    assert bool((predicted.grad[:, 1][labels != 0] != 0.0).any())


def test_scale_16_hierarchical_adapter_is_normalized_stable_and_semantic() -> None:
    api = _load(MODEL, "_signed_distance_adapter_math")
    torch = api.torch
    labels = torch.tensor(
        [[[0, 1, 2], [1, 2, 0], [2, 1, 0]]], dtype=torch.long
    )
    fields, _ = api.signed_boundary_distance_targets_v1(labels)
    logits = api.hierarchical_class_log_probabilities_v1(fields)
    probabilities = logits.exp()
    torch.testing.assert_close(
        probabilities.sum(dim=1),
        torch.ones_like(probabilities[:, 0]),
        rtol=2e-6,
        atol=2e-7,
    )
    assert torch.equal(logits.argmax(dim=1), labels)
    torch.testing.assert_close(
        torch.softmax(logits, dim=1), probabilities, rtol=2e-6, atol=2e-7
    )

    extreme = torch.tensor(
        [[[[1.0, -1.0, 0.0]], [[1.0, -1.0, 0.0]]]],
        dtype=torch.float32,
    )
    extreme_logits = api.hierarchical_class_log_probabilities_v1(extreme)
    assert bool(torch.isfinite(extreme_logits).all())
    assert int(extreme_logits[0, :, 0, 2].argmax()) == 0
    for code in (0, 1, 2):
        saturated_labels = torch.full((1, 3, 3), code, dtype=torch.long)
        saturated_fields, _ = api.signed_boundary_distance_targets_v1(
            saturated_labels
        )
        saturated_logits = api.hierarchical_class_log_probabilities_v1(
            saturated_fields
        )
        assert torch.equal(saturated_logits.argmax(dim=1), saturated_labels)
    free_occupied_tie = torch.tensor([[[[1.0]], [[0.0]]]])
    tie_logits = api.hierarchical_class_log_probabilities_v1(free_occupied_tie)
    assert torch.equal(tie_logits[0, 1], tie_logits[0, 2])
    assert int(tie_logits.argmax(dim=1)) == 1
    assert api.HIERARCHICAL_ADAPTER_SCALE_V1 == 16.0


def test_head_formula_inventory_and_both_rows_receive_gradient() -> None:
    api = _load(MODEL, "_signed_distance_head")
    torch = api.torch
    head = api.SignedBoundaryDistanceStateHeadV1()
    assert sum(parameter.numel() for parameter in head.parameters()) == 130
    assert len(tuple(head.parameters())) == 2
    assert tuple(head.projection.weight.shape) == (2, 64, 1, 1)
    assert tuple(head.projection.bias.shape) == (2,)
    features = torch.randn(2, 64, 64, 64, generator=torch.Generator().manual_seed(9))
    output = head(features)
    assert output.shape == (2, 2, 64, 64)
    detached = output.detach()
    assert float(detached.min()) >= -1.0 and float(detached.max()) <= 1.0
    output[:, 0].mean().add(output[:, 1].square().mean()).backward()
    assert head.projection.weight.grad is not None
    assert head.projection.bias.grad is not None
    assert bool((head.projection.weight.grad.reshape(2, -1).norm(dim=1) > 0).all())
    assert bool((head.projection.bias.grad != 0).all())


def test_model_initialization_objective_ema_and_perception_only_accounting() -> None:
    api = _load(MODEL, "_signed_distance_model_integration")
    torch = api.torch
    encoder = api._v10._v8._v6._v3._v1._construct_n320_encoder_without_rng_draw()
    for value in encoder.state_dict().values():
        if value.is_floating_point():
            value.zero_()
    caller_rng = torch.random.get_rng_state().clone()
    model = api.DirectEgocentricBevStateJepaV1(encoder.state_dict())
    assert torch.equal(torch.random.get_rng_state(), caller_rng)
    assert _tensor_state_sha256(torch, model.state_head.state_dict()) == (
        "a3582ca41e41963592f4bf76ba7de432b51fed783408dcda3e3de9c070c9f40f"
    )
    assert (sum(p.numel() for p in model.parameters()), len(tuple(model.parameters()))) == (
        5_987_639,
        299,
    )
    for online, target in zip(
        model._online_modules(), model._target_modules(), strict=True
    ):
        assert tuple(online.state_dict()) == tuple(target.state_dict())
        for name, value in online.state_dict().items():
            assert torch.equal(value, target.state_dict()[name]), name
    assert all(
        not parameter.requires_grad
        for module in model._target_modules()
        for parameter in module.parameters()
    )
    assert int(model.ema_update_count) == 0

    calls = {"online": 0, "target": 0}
    pattern = torch.linspace(-0.8, 0.8, 2 * 64 * 64).reshape(1, 2, 64, 64)

    def online_fields(self: Any, rgb: Any) -> Any:
        calls["online"] += 1
        scalar = rgb.mean(dim=(1, 2, 3), keepdim=True)
        return torch.tanh(pattern.to(rgb).expand(rgb.shape[0], -1, -1, -1) + scalar)

    def target_fields(self: Any, rgb: Any) -> Any:
        calls["target"] += 1
        scalar = rgb.mean(dim=(1, 2, 3), keepdim=True)
        return torch.tanh(
            pattern.to(rgb).expand(rgb.shape[0], -1, -1, -1) + scalar
        ).detach()

    model.online_state_fields = MethodType(online_fields, model)
    model.target_state_fields = MethodType(target_fields, model)
    model.arm_phase_schedule_v6()
    batch = 2
    current_rgb = torch.zeros(batch, 3, 112, 112, dtype=torch.float32)
    next_rgb = torch.full_like(current_rgb, 0.1)
    fixed_rgb = torch.full_like(current_rgb, -0.1)
    labels = torch.arange(batch * 64 * 64).reshape(batch, 64, 64).remainder(3)
    executed = torch.tensor([0, 1], dtype=torch.long)
    action = torch.nn.functional.one_hot(
        executed, num_classes=len(model.action_vocabulary)
    ).to(dtype=torch.float32)
    result = model.training_objective(
        current_rgb=current_rgb,
        next_rgb=next_rgb,
        fixed_negative_rgb=fixed_rgb,
        action_one_hot=action,
        non_hold_mask=executed != api.HOLD_ACTION_INDEX_V1,
        current_labels=labels,
        next_labels=labels.roll(1, dims=1),
    )
    assert calls == {"online": 2, "target": 3}
    assert torch.equal(result.total, result.G)
    assert torch.equal(result.G, 0.5 * result.G_current + 0.5 * result.G_next)
    assert bool(torch.isfinite(result.total))

    target_before = model.target_state_head.projection.weight.detach().clone()
    with torch.no_grad():
        model.state_head.projection.weight.add_(0.125)
    online_before = model.state_head.projection.weight.detach().clone()
    model.update_target_ema_after_optimizer_step()
    expected = target_before * 0.996 + online_before * 0.004
    torch.testing.assert_close(
        model.target_state_head.projection.weight,
        expected,
        rtol=0.0,
        atol=torch.finfo(expected.dtype).eps,
    )
    assert model.active_phase_v6 == api.PHASE_ONE_V6
    counters = model.phase_counters_v6()
    assert counters["perception_optimizer_update_count"] == 1
    assert counters["predictor_optimizer_update_count"] == 0
    assert counters["ema_arithmetic_update_count"] == 1
    assert counters["boundary_hard_sync_count"] == 0
