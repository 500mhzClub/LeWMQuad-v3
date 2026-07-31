from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import NamedTuple

from PIL import Image
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as h6
from scripts import (
    evaluate_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1
    as evaluation,
)


def _row(
    role: str,
    index: int,
    *,
    family: str | None = None,
) -> h6.H6V2Row:
    selected_family = family or h6.FAMILIES[0]
    scene = f"{selected_family}_{index + (0 if role == 'train' else 4096):012x}"
    environment = index % 48
    first_frame = environment + 48 * (100 + index * 50)
    leaves = tuple(
        f"{scene}/rgb/frame_{first_frame + 240 * horizon:06d}_env_"
        f"{environment:02d}.png"
        for horizon in range(7)
    )
    return h6.H6V2Row(
        index=index,
        role=role,
        family=selected_family,
        scene_id=scene,
        rgb=leaves,
        actions=tuple((index + offset) % 9 for offset in range(6)),
    )


def _leaf_path(root: Path, leaf: str) -> Path:
    return root / evaluation.RGB_ROOT_RELATIVE_PATH_V1 / leaf


def _write_images(root: Path, row: h6.H6V2Row) -> None:
    for horizon, leaf in enumerate(row.rgb):
        path = _leaf_path(root, leaf)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = 20 + horizon * 20
        Image.new("RGB", (224, 224), color=(value, value + 2, value + 5)).save(
            path, format="PNG"
        )


class _SyntheticLoader:
    def __init__(self) -> None:
        self.loaded_indices: list[int] = []
        self.requests = 0

    def load_sequence(self, row: h6.H6V2Row) -> torch.Tensor:
        self.loaded_indices.append(row.index)
        self.requests += 7
        values = [
            ((row.index % 31) * 0.011 + horizon * 0.047) % 0.9
            for horizon in range(7)
        ]
        return torch.stack(
            [
                torch.full((3, 112, 112), value, dtype=torch.float32)
                for value in values
            ]
        )

    def access_snapshot(self) -> dict[str, int]:
        return {
            "rgb_tensor_request_count": self.requests,
            "rgb_open_attempt_count": self.requests,
            "rgb_open_success_count": self.requests,
            "rgb_decode_success_count": self.requests,
            "rgb_byte_count": self.requests * 100,
            "validation_cache_hit_count": 0,
            "validation_cache_miss_count": 0,
            "validation_cache_insert_count": 0,
            "validation_cache_entry_count": 0,
            "validation_cache_bytes": 0,
        }


class _State(NamedTuple):
    tokens: torch.Tensor
    valid: torch.Tensor
    actions: torch.Tensor


class _Step(NamedTuple):
    prediction: torch.Tensor
    state: _State


class _TinyDelayLineModel(nn.Module):
    """Newest-tap model with a rich deterministic synthetic token geometry."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.ema_target = nn.Linear(1, 1, bias=False)
        self.ema_target.requires_grad_(False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(17)
        self.register_buffer(
            "basis",
            torch.randn(1, 64, 16, 16, generator=generator),
        )
        self.register_buffer(
            "direction",
            torch.randn(1, 64, 16, 16, generator=generator),
        )
        self.train(True)

    def train(self, mode: bool = True) -> _TinyDelayLineModel:
        super().train(mode)
        self.ema_target.eval()
        return self

    def target_modules(self) -> tuple[nn.Module, ...]:
        return (self.ema_target,)

    def _encode(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, time = sequence.shape[:2]
        signal = sequence.mean(dim=(2, 3, 4)).reshape(batch * time, 1, 1, 1)
        value = self.basis + signal * self.direction
        value = F.normalize(value + self.anchor * 0.0, dim=1)
        return value.reshape(batch, time, 64, 16, 16)

    def encode_online_memory_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        return self._encode(sequence)

    def encode_target_memory_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        return self._encode(sequence).detach()

    def build_history_state(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
    ) -> _State:
        batch, time = tokens.shape[:2]
        state_tokens = torch.zeros(
            batch, 4, 64, 16, 16, dtype=tokens.dtype, device=tokens.device
        )
        state_tokens[:, :time] = tokens.flip(1)
        valid = torch.zeros(batch, 4, dtype=torch.bool, device=tokens.device)
        valid[:, :time] = True
        state_actions = torch.zeros(
            batch, 4, 9, dtype=torch.float32, device=tokens.device
        )
        if time > 1:
            state_actions[:, : time - 1] = actions.flip(1)
        return _State(state_tokens, valid, state_actions)

    def reset_history_state(
        self,
        state: _State,
        reset_mask: torch.Tensor | None = None,
    ) -> _State:
        del reset_mask
        tokens = torch.zeros_like(state.tokens)
        tokens[:, 0] = state.tokens[:, 0]
        valid = torch.zeros_like(state.valid)
        valid[:, 0] = True
        return _State(tokens, valid, torch.zeros_like(state.actions))

    def predict_from_state(
        self,
        state: _State,
        action_one_hot: torch.Tensor,
    ) -> _Step:
        prediction = state.tokens[:, 0]
        tokens = torch.cat((prediction[:, None], state.tokens[:, :-1]), dim=1)
        valid = torch.cat(
            (
                torch.ones(
                    state.valid.shape[0],
                    1,
                    dtype=torch.bool,
                    device=state.valid.device,
                ),
                state.valid[:, :-1],
            ),
            dim=1,
        )
        actions = torch.cat(
            (action_one_hot[:, None], state.actions[:, :-1]), dim=1
        )
        return _Step(prediction, _State(tokens, valid, actions))


def test_safe_loader_reads_all_seven_registered_leaves_without_cache(
    tmp_path: Path,
) -> None:
    row = _row("val", 0)
    _write_images(tmp_path, row)

    with evaluation.SafeDelayLineRGBLoader(tmp_path, (row,)) as loader:
        sequence = loader.load_sequence(row)
        receipt = loader.access_snapshot()

    assert sequence.shape == (7, 3, 112, 112)
    assert sequence.dtype == torch.float32
    assert receipt["rgb_tensor_request_count"] == 7
    assert receipt["rgb_open_success_count"] == 7
    assert receipt["rgb_decode_success_count"] == 7
    assert receipt["validation_cache_entry_count"] == 0
    assert receipt["validation_cache_insert_count"] == 0


def test_safe_loader_rejects_a_symlinked_history_leaf(tmp_path: Path) -> None:
    row = _row("val", 0)
    _write_images(tmp_path, row)
    first = _leaf_path(tmp_path, row.rgb[0])
    first.unlink()
    first.symlink_to(Path(row.rgb[1]).name)

    with evaluation.SafeDelayLineRGBLoader(tmp_path, (row,)) as loader:
        with pytest.raises(evaluation.DelayLineEvaluationContractError, match="no-follow"):
            loader.load_sequence(row)
        assert loader.access_snapshot()["rgb_open_attempt_count"] == 1
        assert loader.access_snapshot()["rgb_open_success_count"] == 0


def test_update_builder_returns_exact_eight_consecutive_b2_batches() -> None:
    prototype = _row("train", 0)
    rows = tuple(replace(prototype, index=index) for index in range(16_000))
    loader = _SyntheticLoader()

    batches = evaluation.build_train_h6_microbatches_for_update(
        rows,
        update=2,
        loader=loader,
        device="cpu",
    )

    assert len(batches) == 8
    assert loader.loaded_indices == list(range(16, 32))
    for batch in batches:
        assert tuple(batch) == evaluation.REQUIRED_MEMORY_BATCH_KEYS_V1
        assert batch[evaluation.HISTORY_RGB_KEY_V1].shape == (
            2,
            3,
            3,
            112,
            112,
        )
        assert batch[evaluation.HISTORY_ACTIONS_KEY_V1].shape == (2, 2)
        assert batch[evaluation.HISTORY_ACTIONS_KEY_V1].dtype == torch.long
        assert batch[evaluation.FUTURE_RGB_KEY_V1].shape == (
            2,
            4,
            3,
            112,
            112,
        )
        assert batch[evaluation.FUTURE_ACTIONS_KEY_V1].shape == (2, 4)
        assert batch[evaluation.FUTURE_ACTIONS_KEY_V1].dtype == torch.long


def test_streamed_temporal_evaluation_is_deterministic_and_update_zero_exact() -> None:
    rows = tuple(
        _row("val", index, family=h6.FAMILIES[index % len(h6.FAMILIES)])
        for index in range(16)
    )
    model = _TinyDelayLineModel()
    first_loader = _SyntheticLoader()

    first = evaluation.stream_validation_delay_line_metrics(
        model,
        rows,
        update=0,
        loader=first_loader,
        device="cpu",
        bootstrap_replicates=40,
        bootstrap_seed=123,
    )

    assert model.training is True
    assert first.temporal.row_count == 16
    assert first.temporal.family_count == 8
    assert first.energies.real.shape == (16, 4)
    for arm in (
        first.energies.real,
        first.energies.wrong_action,
        first.energies.reset,
        first.energies.reverse,
        first.energies.shuffle,
    ):
        torch.testing.assert_close(arm, first.energies.persistence, rtol=0.0, atol=0.0)
    assert first.target_rank.noncollapsed
    assert first.online_rank.noncollapsed
    assert first.memory_rank.noncollapsed
    assert first.integrity["passed"] is True
    assert first.integrity["future_rgb_online_access_count"] == 0
    assert first.access_receipt["rgb_tensor_request_count"] == 16 * 7
    assert first.memory_receipt["retained_full_row_latent_count"] == 0
    assert first.memory_receipt["retained_scalar_energy_count"] == 16 * 4 * 6

    second = evaluation.stream_validation_delay_line_metrics(
        model,
        rows,
        update=0,
        loader=_SyntheticLoader(),
        device="cpu",
        bootstrap_replicates=40,
        bootstrap_seed=123,
    )
    torch.testing.assert_close(first.energies.real, second.energies.real)
    assert first.temporal == second.temporal
    assert first.target_rank == second.target_rank
    assert first.memory_rank == second.memory_rank


def test_validation_rejects_non_b8_or_out_of_order_rows() -> None:
    rows = tuple(
        _row("val", index, family=h6.FAMILIES[index % len(h6.FAMILIES)])
        for index in range(8)
    )
    model = _TinyDelayLineModel()
    with pytest.raises(
        evaluation.DelayLineEvaluationContractError,
        match="ordered B8",
    ):
        evaluation.stream_validation_delay_line_metrics(
            model,
            (*rows, replace(rows[-1], index=8)),
            update=100,
            loader=_SyntheticLoader(),
            device="cpu",
            bootstrap_replicates=40,
        )
