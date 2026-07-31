from __future__ import annotations

import copy
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
)


def _load_runner(
    name: str = "_single_frame_masked_spatial_runner_test",
) -> Any:
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _FakeMaskedSpatialJepa:
    """Constructed lazily so importing this test does not require Torch."""

    @staticmethod
    def build(torch: Any) -> Any:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                torch.manual_seed(81)
                self.encoder = torch.nn.Linear(3, 8)
                self.predictor_output = torch.nn.Linear(8, 8)
                self.target_encoder = copy.deepcopy(self.encoder)
                self.target_encoder.requires_grad_(False)
                self.target_encoder.eval()
                self.register_buffer(
                    "ema_update_count",
                    torch.zeros((), dtype=torch.long),
                )
                self.online_calls = 0
                self.target_calls = 0
                self.ema_calls = 0

            def forward_online(self, rgb, target_indices):
                self.online_calls += 1
                features = self.encoder(rgb.mean(dim=(2, 3)))
                predicted = self.predictor_output(features)[:, None, :].expand(
                    -1,
                    target_indices.shape[1],
                    -1,
                )
                universe = torch.arange(
                    256,
                    dtype=torch.long,
                    device=target_indices.device,
                ).expand(target_indices.shape[0], -1)
                target_mask = torch.zeros_like(universe, dtype=torch.bool)
                target_mask.scatter_(1, target_indices, True)
                visible_indices = universe[~target_mask].reshape(
                    target_indices.shape[0],
                    192,
                )
                return SimpleNamespace(
                    normalized_predicted_target_tokens=(
                        torch.nn.functional.normalize(predicted, dim=-1)
                    ),
                    target_indices=target_indices,
                    visible_indices=visible_indices,
                )

            @torch.no_grad()
            def encode_target(self, rgb, target_indices):
                self.target_calls += 1
                features = self.target_encoder(rgb.mean(dim=(2, 3)))
                target = features[:, None, :].expand(
                    -1,
                    target_indices.shape[1],
                    -1,
                )
                return SimpleNamespace(
                    normalized_target_tokens=torch.nn.functional.normalize(
                        target,
                        dim=-1,
                    ),
                    target_indices=target_indices,
                )

            def forward(self, rgb, target_indices):
                prediction = self.forward_online(rgb, target_indices)
                target = self.encode_target(rgb, target_indices)
                loss = (
                    0.5
                    * (
                        prediction.normalized_predicted_target_tokens
                        - target.normalized_target_tokens
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                )
                return SimpleNamespace(
                    prediction=prediction,
                    target=target,
                    loss=loss,
                )

            @torch.no_grad()
            def update_target_ema(self):
                self.ema_calls += 1
                for online, target in zip(
                    self.encoder.parameters(),
                    self.target_encoder.parameters(),
                    strict=True,
                ):
                    target.mul_(0.996).add_(online, alpha=0.004)
                self.ema_update_count.add_(1)
                self.target_encoder.eval()

        return Model()


def _inputs(torch: Any, *, start: int = 0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260731 + start)
    rgb = tuple(
        torch.randn(
            4,
            3,
            112,
            112,
            dtype=torch.float32,
            generator=generator,
        )
        for _ in range(4)
    )
    rows = tuple(
        tuple(range(start + microbatch * 4, start + (microbatch + 1) * 4))
        for microbatch in range(4)
    )
    return rgb, rows


def _all_mapping_keys(value: Any) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_all_mapping_keys(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            keys.extend(_all_mapping_keys(item))
    return tuple(keys)


def test_runner_import_is_source_only() -> None:
    program = f"""
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("_masked_spatial_source_only", {str(RUNNER_PATH)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert module.PRESENTATIONS_PER_UPDATE_V1 == 16
assert module.MAXIMUM_UPDATES_V1 == 1000
assert module.MAXIMUM_PRESENTATIONS_V1 == 16000
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


def test_one_update_has_exact_accounting_routes_optimizer_and_ema() -> None:
    torch = pytest.importorskip("torch")
    runner = _load_runner("_masked_spatial_update")
    model = _FakeMaskedSpatialJepa.build(torch)
    optimizer = runner.build_optimizer_v1(model)
    step_count = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    target_before = tuple(
        parameter.detach().clone() for parameter in model.target_encoder.parameters()
    )
    rgb, rows = _inputs(torch)
    result = runner.training_update_v1(model, optimizer, rgb, rows)

    assert result.accounting == runner.accounting_for_completed_updates_v1(1)
    assert result.accounting.presentations == 16
    assert result.accounting.microbatch_graphs == 4
    assert result.accounting.backward_calls == 4
    assert result.accounting.global_gradient_clips == 1
    assert result.accounting.optimizer_steps == 1
    assert result.accounting.ema_steps == 1
    assert result.optimizer_steps_this_update == 1
    assert result.ema_steps_this_update == 1
    assert step_count == 1
    assert model.ema_calls == 1
    assert model.online_calls == 4
    assert model.target_calls == 4
    assert len(result.microbatch_jepa_losses) == 4
    assert result.mean_jepa_loss == pytest.approx(
        sum(result.microbatch_jepa_losses) / 4
    )
    assert result.mean_jepa_loss > 0.0
    assert result.target_gradient_tensor_count == 0
    assert all(
        parameter.grad is None for parameter in model.target_encoder.parameters()
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(
            target_before,
            model.target_encoder.parameters(),
            strict=True,
        )
    )
    receipt = result.gradient_receipt
    assert receipt["sole_jepa_route"] is True
    assert receipt["encoder_gradient_norm_before_global_clip"] > 0.0
    assert receipt["predictor_gradient_norm_before_global_clip"] > 0.0
    assert receipt["global_gradient_norm_before_clip"] > 0.0
    assert receipt["global_gradient_norm_after_clip"] <= 1.00001
    assert receipt["missing_encoder_gradient_tensor_count"] == 0
    assert receipt["missing_predictor_gradient_tensor_count"] == 0
    assert receipt["all_gradient_receipts_finite"] is True


def test_optimizer_has_two_constant_science_groups_and_no_target() -> None:
    torch = pytest.importorskip("torch")
    runner = _load_runner("_masked_spatial_optimizer")
    model = _FakeMaskedSpatialJepa.build(torch)
    partition = runner.partition_parameters_v1(model)
    optimizer = runner.build_optimizer_v1(partition)
    runner.validate_optimizer_v1(optimizer, partition)

    assert [group["group_name"] for group in optimizer.param_groups] == [
        "encoder",
        "predictor",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [
        1.0e-4,
        3.0e-4,
    ]
    assert all(group["weight_decay"] == 1.0e-4 for group in optimizer.param_groups)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimizer_ids == {id(parameter) for parameter in partition.online}
    assert optimizer_ids.isdisjoint(
        {id(parameter) for parameter in partition.target}
    )

    optimizer.param_groups[0]["lr"] = 9.0e-4
    with pytest.raises(RuntimeError, match="optimizer group"):
        runner.validate_optimizer_v1(optimizer, partition)


def test_masks_are_benchmark_derived_and_repeat_exactly_by_row_identity() -> None:
    torch = pytest.importorskip("torch")
    runner = _load_runner("_masked_spatial_masks")
    _, masks = runner._runtime_apis()
    rows = (0, 1, 2, 3)
    first, first_visible = runner._mask_indices_for_microbatch_v1(
        torch,
        masks,
        rows,
        torch.device("cpu"),
    )
    second, second_visible = runner._mask_indices_for_microbatch_v1(
        torch,
        masks,
        rows,
        torch.device("cpu"),
    )
    expected = torch.tensor(
        [masks.mask_indices("train", row)[0] for row in rows],
        dtype=torch.long,
    )
    expected_visible = tuple(
        masks.mask_indices("train", row)[1] for row in rows
    )
    assert torch.equal(first, second)
    assert torch.equal(first, expected)
    assert first_visible == second_visible == expected_visible
    assert tuple(first.shape) == (4, 64)
    assert all(len(values) == 192 for values in first_visible)
    for target, visible in zip(first.tolist(), first_visible, strict=True):
        assert set(target).isdisjoint(visible)
        assert set(target) | set(visible) == set(range(256))


def test_cap_and_frozen_row_order_fail_before_mutation() -> None:
    torch = pytest.importorskip("torch")
    runner = _load_runner("_masked_spatial_caps")
    model = _FakeMaskedSpatialJepa.build(torch)
    optimizer = runner.build_optimizer_v1(model)
    rgb, rows = _inputs(torch)

    model.ema_update_count.fill_(1000)
    at_cap = runner.accounting_for_completed_updates_v1(1000)
    with pytest.raises(PermissionError, match="no complete update"):
        runner.training_update_v1(
            model,
            optimizer,
            rgb,
            rows,
            accounting=at_cap,
        )
    assert model.ema_calls == 0
    assert model.online_calls == 0
    assert model.target_calls == 0

    model.ema_update_count.zero_()
    wrong_rows = tuple(
        tuple(value + (1 if microbatch == 2 else 0) for value in values)
        for microbatch, values in enumerate(rows)
    )
    with pytest.raises(PermissionError, match="corrected-H6 order"):
        runner.training_update_v1(model, optimizer, rgb, wrong_rows)
    assert model.ema_calls == 0
    assert model.online_calls == 0
    assert model.target_calls == 0


def test_checkpoint_is_complete_cpu_state_without_temporal_inputs() -> None:
    torch = pytest.importorskip("torch")
    runner = _load_runner("_masked_spatial_checkpoint")
    model = _FakeMaskedSpatialJepa.build(torch)
    optimizer = runner.build_optimizer_v1(model)
    rgb, rows = _inputs(torch)
    result = runner.training_update_v1(model, optimizer, rgb, rows)
    payload = runner.checkpoint_payload_v1(
        model,
        optimizer,
        result.accounting,
    )

    assert tuple(payload) == (
        "schema",
        "model_state_dict",
        "optimizer_state_dict",
        "accounting",
        "model_state_inventory",
        "training_contract",
    )
    assert payload["accounting"]["updates"] == 1
    assert payload["accounting"]["presentations"] == 16
    assert payload["training_contract"]["sole_objective"] == (
        "normalized_half_squared_masked_spatial_jepa"
    )
    assert payload["model_state_inventory"]["target_optimizer_excluded"] is True
    assert all(
        value.device.type == "cpu"
        for value in payload["model_state_dict"].values()
    )
    forbidden = ("action", "history", "future", "next_rgb", "sequence", "pose", "depth")
    assert not any(
        token in key
        for key in _all_mapping_keys(payload)
        for token in forbidden
    )
    parameters = tuple(inspect.signature(runner.training_update_v1).parameters)
    assert parameters == (
        "model",
        "optimizer",
        "rgb_microbatches",
        "row_index_microbatches",
        "accounting",
    )
