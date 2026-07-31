from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import importlib.util

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"


def _load():
    spec = importlib.util.spec_from_file_location("temporal_runner_v1_tested", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _TinyTemporalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(7)
        self.encoder = nn.Sequential(nn.Linear(1, 192), nn.Tanh())
        self.predictor_position = nn.Parameter(torch.randn(256, 192) * 0.01)
        self.predictor_mask_token = nn.Parameter(torch.randn(1, 1, 192) * 0.01)
        self.predictor_blocks = nn.ModuleList([nn.Linear(192, 192)])
        self.predictor_norm = nn.LayerNorm(192)
        self.predictor_output = nn.Linear(192, 192)
        self.action_embedding = nn.Embedding(9, 192)
        self.time_embedding = nn.Embedding(3, 192)
        self.temporal_gru = nn.GRU(192, 192, batch_first=True)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_encoder.eval()
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target_ema(self) -> None:
        for online, target in zip(
            self.encoder.parameters(), self.target_encoder.parameters(), strict=True
        ):
            target.mul_(0.996).add_(online, alpha=0.004)
        self.ema_update_count.add_(1)
        self.target_encoder.eval()

    def forward(self, context, actions, future, target_indices):
        del target_indices
        batch = context.shape[0]
        scalars = context.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        encoded = self.encoder(scalars.reshape(-1, 1)).reshape(batch, 3, 192)
        time = self.time_embedding(torch.arange(3, device=context.device)).unsqueeze(0)
        recurrent, _ = self.temporal_gru(
            encoded + self.action_embedding(actions) + time
        )
        hidden = recurrent[:, -1:].expand(-1, 64, -1)
        hidden = hidden + self.predictor_position[:64].unsqueeze(0)
        hidden = hidden + self.predictor_mask_token
        for block in self.predictor_blocks:
            hidden = torch.tanh(block(hidden))
        raw_prediction = self.predictor_output(self.predictor_norm(hidden))
        with torch.no_grad():
            future_scalar = future.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
            raw_target = self.target_encoder(future_scalar).unsqueeze(1).expand(-1, 64, -1)
        prediction = F.normalize(raw_prediction, dim=-1, eps=1e-8)
        target = F.normalize(raw_target, dim=-1, eps=1e-8).detach()
        loss = 0.5 * (prediction - target).square().sum(dim=-1).mean()
        return SimpleNamespace(
            prediction=SimpleNamespace(normalized_predicted_target_tokens=prediction),
            target=SimpleNamespace(normalized_target_tokens=target),
            loss=loss,
        )


def test_accounting_is_exact_and_capped() -> None:
    runner = _load()
    state = runner.accounting_for_completed_updates_v1(400)
    assert state.sequence_rows == 4_000
    assert state.logical_rgb_presentations == 16_000
    assert state.online_frame_encodings == 12_000
    assert state.ema_target_frame_encodings == 4_000
    assert state.backward_calls == 2_000
    runner.validate_accounting_v1(state)
    with pytest.raises(PermissionError):
        runner.validate_accounting_v1(
            runner.accounting_for_completed_updates_v1(401)
        )


def test_optimizer_partition_has_exact_three_online_roles() -> None:
    runner = _load()
    model = _TinyTemporalModel()
    partition = runner.partition_parameters_v1(model)
    optimizer = runner.build_optimizer_v1(partition)
    assert [group["group_name"] for group in optimizer.param_groups] == [
        "encoder",
        "predictor",
        "memory",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [3e-5, 1e-4, 3e-4]
    assert not any(parameter.requires_grad for parameter in partition.target)
    with pytest.raises(PermissionError, match="200/400"):
        runner.checkpoint_payload_v1(
            model,
            optimizer,
            runner.TemporalTrainingAccountingV1(),
        )


def test_one_update_routes_future_jepa_through_all_online_roles() -> None:
    runner = _load()
    model = _TinyTemporalModel().train()
    optimizer = runner.build_optimizer_v1(model)
    generator = torch.Generator().manual_seed(11)
    context = [torch.rand(2, 3, 3, 112, 112, generator=generator) for _ in range(5)]
    future = [torch.rand(2, 3, 112, 112, generator=generator) for _ in range(5)]
    actions = [torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long) for _ in range(5)]
    row_batches = [tuple(range(2 * index, 2 * index + 2)) for index in range(5)]
    result = runner.training_update_v1(
        model,
        optimizer,
        context,
        actions,
        future,
        row_batches,
        expected_row_indices=tuple(range(10)),
        schedule_offset=0,
    )
    assert result.accounting == runner.accounting_for_completed_updates_v1(1)
    assert result.target_gradient_tensor_count == 0
    assert result.optimizer_steps_this_update == 1
    assert result.ema_steps_this_update == 1
    assert result.gradient_receipt["encoder_nonzero_gradient_tensor_count"] > 0
    assert result.gradient_receipt["predictor_nonzero_gradient_tensor_count"] > 0
    assert result.gradient_receipt["memory_nonzero_gradient_tensor_count"] > 0
    assert int(model.ema_update_count) == 1


def test_schedule_slice_mismatch_fails_before_training() -> None:
    runner = _load()
    model = _TinyTemporalModel().train()
    optimizer = runner.build_optimizer_v1(model)
    context = [torch.zeros(2, 3, 3, 112, 112) for _ in range(5)]
    future = [torch.zeros(2, 3, 112, 112) for _ in range(5)]
    actions = [torch.zeros(2, 3, dtype=torch.long) for _ in range(5)]
    rows = [tuple(range(2 * index, 2 * index + 2)) for index in range(5)]
    with pytest.raises(PermissionError):
        runner.training_update_v1(
            model,
            optimizer,
            context,
            actions,
            future,
            rows,
            expected_row_indices=tuple(reversed(range(10))),
            schedule_offset=0,
        )
    with pytest.raises(PermissionError):
        runner.training_update_v1(
            model,
            optimizer,
            context,
            actions,
            future,
            rows,
            expected_row_indices=tuple(range(10)),
            schedule_offset=10,
        )
