#!/usr/bin/env python3
"""Bounded RGB loading and temporal evaluation for the V18 delay-line JEPA.

This module owns no training loop and writes no artifacts.  It adapts the
reviewed V27 descriptor-relative RGB loader to expose all seven registered H6
frames, builds the frozen eight-by-B2 memory batches, and retains only scalar
validation energies plus bounded participation-rank samples.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from lewm.benchmarks import go2_v18_delay_line_memory_metrics_v1 as metrics
from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as h6
from lewm.datasets import go2_v18_delay_line_h6_runtime_v1 as schedule
from scripts import (
    evaluate_go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27
    as v27_evaluation,
)


SCHEMA_PREFIX = (
    "lewm_go2_v18_spatial_token_delay_line_causal_convolution_joint_jepa_v1_"
    "physical_comparison_alias_state_integrity_replacement_v4"
)
RGB_ROOT_RELATIVE_PATH_V1 = v27_evaluation.RGB_ROOT_RELATIVE_PATH_V27
TRAIN_MICROBATCH_SIZE_V1 = schedule.MEMORY_MICROBATCH_SIZE_V1
TRAIN_MICROBATCHES_PER_UPDATE_V1 = schedule.MEMORY_MICROBATCHES_PER_UPDATE_V1
VALIDATION_BATCH_SIZE_V1 = 8
TOKEN_SHAPE_V1 = (64, 16, 16)
ACTION_COUNT_V1 = 9
HOLD_ACTION_INDEX_V1 = 6
PARTICIPATION_SPATIAL_ROWS_PER_MAP_V1 = 16
OBSERVATION_UPDATES_V1 = (0, 100, 250, 500, 750, 1_000)

HISTORY_RGB_KEY_V1 = "history_rgb"
HISTORY_ACTIONS_KEY_V1 = "history_actions"
FUTURE_RGB_KEY_V1 = "future_rgb"
FUTURE_ACTIONS_KEY_V1 = "future_actions"
REQUIRED_MEMORY_BATCH_KEYS_V1 = (
    HISTORY_RGB_KEY_V1,
    HISTORY_ACTIONS_KEY_V1,
    FUTURE_RGB_KEY_V1,
    FUTURE_ACTIONS_KEY_V1,
)

# Both corruptions retain the current z2 token.  Reverse swaps temporal order
# of the two older observations; shuffle rotates older histories across the
# fixed B8 validation batch.  Thus every update-zero newest-tap arm remains
# exactly equal to persistence.
REVERSE_HISTORY_ORDER_V1 = (1, 0, 2)
WRONG_ACTION_OFFSET_V1 = 1

DelayLineEvaluationContractError = v27_evaluation.V27EvaluationContractError


@dataclass(frozen=True, slots=True)
class DelayLineEnergyVectorsV1:
    """Six scalar N-by-H4 energy arms; no latent tensors are retained."""

    real: torch.Tensor
    persistence: torch.Tensor
    wrong_action: torch.Tensor
    reset: torch.Tensor
    reverse: torch.Tensor
    shuffle: torch.Tensor


@dataclass(frozen=True, slots=True)
class HoldDiagnosticsV1:
    """Evaluation-only HOLD slices, never inputs to a continuation gate.

    ``mean_normalized_delta_from_real`` reports ``(E(arm)-E(real))/E(persistence)``
    for each arm and horizon.  Thus the real arm is zero, the persistence arm
    is the primary persistence lift, and the wrong-action arm is the primary
    action lift on HOLD rows.
    """

    action_index: int
    row_count: tuple[int, int, int, int]
    mean_energy: Mapping[str, tuple[float, float, float, float]]
    mean_normalized_delta_from_real: Mapping[
        str, tuple[float, float, float, float]
    ]
    real_normalized_score: tuple[float, float, float, float]
    persistence_lift: tuple[float, float, float, float]
    action_lift: tuple[float, float, float, float]
    ordered_history_lift: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DelayLineValidationResultV1:
    """Complete temporal result consumed by observation/gate orchestration."""

    update: int
    temporal: metrics.TemporalMetrics
    target_rank: metrics.ParticipationRank
    online_rank: metrics.ParticipationRank
    memory_rank: metrics.ParticipationRank
    energies: DelayLineEnergyVectorsV1
    hold_diagnostics: HoldDiagnosticsV1
    access_receipt: Mapping[str, int]
    memory_receipt: Mapping[str, int | str]
    integrity: Mapping[str, Any]


class SafeDelayLineRGBLoader(v27_evaluation.SafeV27RGBLoader):
    """V27 no-follow loader adapted to read every registered e0:e6 leaf.

    Validation caching is deliberately disabled: this evaluator makes one
    streamed pass, so retaining 14,336 decoded float32 images would only add
    memory pressure.
    """

    def _decode_leaf(self, row: h6.H6V2Row, leaf: str) -> torch.Tensor:
        self._access["rgb_tensor_request_count"] += 1
        raw = self._read_leaf(row, leaf)
        try:
            tensor = h6.rectify_h6_rgb_bytes(raw)
        except h6.V27DataContractError as error:
            raise DelayLineEvaluationContractError(
                "registered delay-line RGB decode failed"
            ) from error
        self._access["rgb_decode_success_count"] += 1
        return tensor

    def load_sequence(self, row: h6.H6V2Row) -> torch.Tensor:
        """Load the exact registered e0:e6 sequence in causal order."""

        registered = self._registered(row)
        value = torch.stack(
            tuple(self._decode_leaf(registered, leaf) for leaf in registered.rgb),
            dim=0,
        )
        if (
            tuple(value.shape) != (7, 3, 112, 112)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise DelayLineEvaluationContractError(
                "delay-line RGB sequence is not finite 7x3x112x112 float32"
            )
        return value


def _access_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    if set(before) != set(after):
        raise DelayLineEvaluationContractError("RGB access counter schema changed")
    result = {key: int(after[key]) - int(before[key]) for key in before}
    if any(value < 0 for value in result.values()):
        raise DelayLineEvaluationContractError("RGB access counters moved backwards")
    return result


def _stack_sequences(
    rows: Sequence[h6.H6V2Row],
    loader: SafeDelayLineRGBLoader,
    device: Any,
) -> torch.Tensor:
    value = torch.stack(tuple(loader.load_sequence(row) for row in rows), dim=0)
    value = value.to(device=torch.device(device), non_blocking=False)
    if (
        tuple(value.shape) != (len(rows), 7, 3, 112, 112)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise DelayLineEvaluationContractError("stacked H6 RGB sequence is invalid")
    return value


def build_train_h6_microbatch(
    rows: Sequence[h6.H6V2Row],
    *,
    loader: SafeDelayLineRGBLoader,
    device: Any,
) -> dict[str, torch.Tensor]:
    """Load one exact consecutive B2 delay-line training microbatch."""

    selected = tuple(rows)
    if (
        len(selected) != TRAIN_MICROBATCH_SIZE_V1
        or any(row.role != "train" for row in selected)
        or selected[1].index != selected[0].index + 1
    ):
        raise DelayLineEvaluationContractError(
            "delay-line train microbatch must be two consecutive train rows"
        )
    sequences = _stack_sequences(selected, loader, device)
    examples = tuple(schedule.split_registered_row_v1(row) for row in selected)
    result = {
        HISTORY_RGB_KEY_V1: sequences[:, :3],
        HISTORY_ACTIONS_KEY_V1: torch.tensor(
            [example.history_actions for example in examples],
            dtype=torch.long,
            device=sequences.device,
        ),
        FUTURE_RGB_KEY_V1: sequences[:, 3:],
        FUTURE_ACTIONS_KEY_V1: torch.tensor(
            [example.future_actions for example in examples],
            dtype=torch.long,
            device=sequences.device,
        ),
    }
    if (
        tuple(result) != REQUIRED_MEMORY_BATCH_KEYS_V1
        or tuple(result[HISTORY_RGB_KEY_V1].shape) != (2, 3, 3, 112, 112)
        or tuple(result[HISTORY_ACTIONS_KEY_V1].shape) != (2, 2)
        or tuple(result[FUTURE_RGB_KEY_V1].shape) != (2, 4, 3, 112, 112)
        or tuple(result[FUTURE_ACTIONS_KEY_V1].shape) != (2, 4)
    ):
        raise DelayLineEvaluationContractError("delay-line train batch schema changed")
    return result


def build_train_h6_microbatches_for_update(
    train_rows: Sequence[h6.H6V2Row],
    *,
    update: int,
    loader: SafeDelayLineRGBLoader,
    device: Any,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Return the frozen eight consecutive B2 batches for one update."""

    row_pairs = schedule.train_rows_for_update_v1(train_rows, update)
    result = tuple(
        build_train_h6_microbatch(pair, loader=loader, device=device)
        for pair in row_pairs
    )
    if len(result) != TRAIN_MICROBATCHES_PER_UPDATE_V1:
        raise DelayLineEvaluationContractError("delay-line update lost a microbatch")
    return result


def _validate_tokens(
    value: Any,
    *,
    batch: int,
    time: int,
    name: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (batch, time, *TOKEN_SHAPE_V1)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise DelayLineEvaluationContractError(
            f"{name} is not finite Bx{time}x64x16x16 float32"
        )
    return value


def _validate_state(value: Any, *, batch: int, name: str) -> Any:
    tokens = getattr(value, "tokens", None)
    valid = getattr(value, "valid", None)
    actions = getattr(value, "actions", None)
    if (
        not isinstance(tokens, torch.Tensor)
        or tuple(tokens.shape) != (batch, 4, *TOKEN_SHAPE_V1)
        or tokens.dtype != torch.float32
        or not bool(torch.isfinite(tokens).all())
        or not isinstance(valid, torch.Tensor)
        or tuple(valid.shape) != (batch, 4)
        or valid.dtype != torch.bool
        or not isinstance(actions, torch.Tensor)
        or tuple(actions.shape) != (batch, 4, ACTION_COUNT_V1)
        or actions.dtype != torch.float32
        or not bool(torch.isfinite(actions).all())
    ):
        raise DelayLineEvaluationContractError(f"{name} state schema changed")
    return value


def _action_one_hot(actions: torch.Tensor) -> torch.Tensor:
    if (
        not isinstance(actions, torch.Tensor)
        or actions.dtype != torch.long
        or actions.ndim not in (1, 2)
        or bool((actions < 0).any())
        or bool((actions >= ACTION_COUNT_V1).any())
    ):
        raise DelayLineEvaluationContractError("action indices are invalid")
    return torch.nn.functional.one_hot(
        actions, num_classes=ACTION_COUNT_V1
    ).to(dtype=torch.float32)


def _build_history_state(
    model: Any,
    history_tokens: torch.Tensor,
    history_actions: torch.Tensor,
    *,
    name: str,
) -> Any:
    state = model.build_history_state(
        history_tokens,
        _action_one_hot(history_actions),
    )
    return _validate_state(state, batch=history_tokens.shape[0], name=name)


def _rollout(
    model: Any,
    state: Any,
    future_actions: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    batch = int(future_actions.shape[0])
    current = _validate_state(state, batch=batch, name=f"{name} initial")
    for horizon in range(4):
        step = model.predict_from_state(
            current,
            _action_one_hot(future_actions[:, horizon]),
        )
        prediction = getattr(step, "prediction", None)
        if (
            not isinstance(prediction, torch.Tensor)
            or tuple(prediction.shape) != (batch, *TOKEN_SHAPE_V1)
            or prediction.dtype != torch.float32
            or not bool(torch.isfinite(prediction).all())
        ):
            raise DelayLineEvaluationContractError(
                f"{name} H{horizon + 1} prediction is invalid"
            )
        current = _validate_state(
            getattr(step, "state", None),
            batch=batch,
            name=f"{name} H{horizon + 1}",
        )
        predictions.append(prediction)
    return torch.stack(predictions, dim=1)


def _energy(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    value = (
        0.5
        * (prediction - target)
        .square()
        .sum(dim=2)
        .mean(dim=(2, 3))
    )
    if (
        tuple(value.shape) != (prediction.shape[0], 4)
        or not bool(torch.isfinite(value).all())
        or bool((value < 0.0).any())
    ):
        raise DelayLineEvaluationContractError("temporal energy vector is invalid")
    return value.detach().to(device="cpu", dtype=torch.float64)


def _participation_rows(token_maps: torch.Tensor) -> torch.Tensor:
    if (
        token_maps.ndim != 4
        or tuple(token_maps.shape[1:]) != TOKEN_SHAPE_V1
        or token_maps.dtype != torch.float32
    ):
        raise DelayLineEvaluationContractError("participation token maps are invalid")
    rows = token_maps.permute(0, 2, 3, 1).reshape(
        token_maps.shape[0], 16 * 16, 64
    )
    stride = (16 * 16) // PARTICIPATION_SPATIAL_ROWS_PER_MAP_V1
    return rows[:, ::stride, :].reshape(-1, 64).detach().to(device="cpu")


def _hold_diagnostics(
    energies: DelayLineEnergyVectorsV1,
    rows: Sequence[h6.H6V2Row],
) -> HoldDiagnosticsV1:
    actions = torch.tensor(
        [row.actions[2:] for row in rows],
        dtype=torch.long,
    )
    arms = {
        name: getattr(energies, name)
        for name in (
            "real",
            "persistence",
            "wrong_action",
            "reset",
            "reverse",
            "shuffle",
        )
    }
    counts: list[int] = []
    mean_energy: dict[str, list[float]] = {name: [] for name in arms}
    mean_delta: dict[str, list[float]] = {name: [] for name in arms}
    score: list[float] = []
    persistence_lift: list[float] = []
    action_lift: list[float] = []
    history_lift: list[float] = []
    for horizon in range(4):
        selected = actions[:, horizon] == HOLD_ACTION_INDEX_V1
        count = int(selected.sum())
        if count <= 0:
            raise DelayLineEvaluationContractError(
                f"HOLD diagnostic has no H{horizon + 1} row"
            )
        denominator = energies.persistence[selected, horizon]
        if bool((denominator <= 0.0).any()):
            raise DelayLineEvaluationContractError(
                "HOLD persistence denominator is not positive"
            )
        real = energies.real[selected, horizon]
        counts.append(count)
        for name, arm in arms.items():
            selected_arm = arm[selected, horizon]
            mean_energy[name].append(float(selected_arm.mean()))
            mean_delta[name].append(
                float(((selected_arm - real) / denominator).mean())
            )
        selected_score = real / denominator
        selected_action_lift = (
            energies.wrong_action[selected, horizon] - real
        ) / denominator
        best_history = torch.minimum(
            torch.minimum(
                energies.reset[selected, horizon],
                energies.reverse[selected, horizon],
            ),
            energies.shuffle[selected, horizon],
        )
        score.append(float(selected_score.mean()))
        persistence_lift.append(float((1.0 - selected_score).mean()))
        action_lift.append(float(selected_action_lift.mean()))
        history_lift.append(float(((best_history - real) / denominator).mean()))
    return HoldDiagnosticsV1(
        action_index=HOLD_ACTION_INDEX_V1,
        row_count=tuple(counts),
        mean_energy={
            name: tuple(values) for name, values in mean_energy.items()
        },
        mean_normalized_delta_from_real={
            name: tuple(values) for name, values in mean_delta.items()
        },
        real_normalized_score=tuple(score),
        persistence_lift=tuple(persistence_lift),
        action_lift=tuple(action_lift),
        ordered_history_lift=tuple(history_lift),
    )


def _validate_validation_rows(rows: Sequence[h6.H6V2Row]) -> tuple[h6.H6V2Row, ...]:
    ordered = tuple(rows)
    if (
        len(ordered) < VALIDATION_BATCH_SIZE_V1
        or len(ordered) % VALIDATION_BATCH_SIZE_V1
        or any(
            row.role != "val" or row.index != index
            for index, row in enumerate(ordered)
        )
    ):
        raise DelayLineEvaluationContractError(
            "validation rows must be complete ordered B8 batches"
        )
    return ordered


def stream_validation_delay_line_metrics(
    model: Any,
    rows: Sequence[h6.H6V2Row],
    *,
    update: int,
    loader: SafeDelayLineRGBLoader,
    device: Any,
    bootstrap_replicates: int = metrics.BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = metrics.BOOTSTRAP_SEED,
) -> DelayLineValidationResultV1:
    """Stream the six deterministic temporal arms and bounded rank samples."""

    if type(update) is not int or update not in OBSERVATION_UPDATES_V1:
        raise DelayLineEvaluationContractError("unregistered observation update")
    ordered = _validate_validation_rows(rows)
    requested_device = torch.device(device)
    parameters = tuple(model.parameters())
    if not parameters or any(
        parameter.device != requested_device for parameter in parameters
    ):
        raise DelayLineEvaluationContractError(
            "model parameters do not share the evaluation device"
        )
    target_modules_method = getattr(model, "target_modules", None)
    if not callable(target_modules_method):
        raise DelayLineEvaluationContractError("model does not expose target_modules")
    target_modules = tuple(target_modules_method())
    if not target_modules:
        raise DelayLineEvaluationContractError("target module inventory is empty")

    energy_chunks: dict[str, list[torch.Tensor]] = {
        name: []
        for name in ("real", "persistence", "wrong_action", "reset", "reverse", "shuffle")
    }
    target_rows: list[torch.Tensor] = []
    online_rows: list[torch.Tensor] = []
    memory_rows: list[torch.Tensor] = []
    before_access = loader.access_snapshot()
    canonical_state_layout = True
    reset_state_layout = True
    all_targets_no_grad = True
    minimum_target_std = math.inf
    minimum_online_std = math.inf
    update_zero_max_prediction_delta = 0.0
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(ordered), VALIDATION_BATCH_SIZE_V1):
                batch_rows = ordered[start : start + VALIDATION_BATCH_SIZE_V1]
                batch = len(batch_rows)
                sequence = _stack_sequences(batch_rows, loader, requested_device)
                history_actions = torch.tensor(
                    [row.actions[:2] for row in batch_rows],
                    dtype=torch.long,
                    device=requested_device,
                )
                future_actions = torch.tensor(
                    [row.actions[2:] for row in batch_rows],
                    dtype=torch.long,
                    device=requested_device,
                )

                # This is the only online encoder call: it receives e0:e2, never
                # future RGB.  The EMA target alone receives e0:e6.
                online = _validate_tokens(
                    model.encode_online_memory_sequence(sequence[:, :3]),
                    batch=batch,
                    time=3,
                    name="online history",
                )
                target_all = _validate_tokens(
                    model.encode_target_memory_sequence(sequence),
                    batch=batch,
                    time=7,
                    name="EMA target sequence",
                )
                all_targets_no_grad = all_targets_no_grad and not target_all.requires_grad
                minimum_online_std = min(
                    minimum_online_std,
                    float(online.std(unbiased=False).item()),
                )
                minimum_target_std = min(
                    minimum_target_std,
                    float(target_all.std(unbiased=False).item()),
                )

                real_state = _build_history_state(
                    model, online, history_actions, name="real"
                )
                expected_valid = torch.tensor(
                    (True, True, True, False),
                    dtype=torch.bool,
                    device=requested_device,
                ).expand(batch, -1)
                canonical_state_layout = canonical_state_layout and bool(
                    torch.equal(real_state.valid, expected_valid)
                )

                real = _rollout(
                    model, real_state, future_actions, name="real"
                )
                wrong = _rollout(
                    model,
                    _build_history_state(
                        model, online, history_actions, name="wrong-action"
                    ),
                    (future_actions + WRONG_ACTION_OFFSET_V1) % ACTION_COUNT_V1,
                    name="wrong-action",
                )
                reset_state = _validate_state(
                    model.reset_history_state(
                        _build_history_state(
                            model, online, history_actions, name="reset source"
                        )
                    ),
                    batch=batch,
                    name="reset",
                )
                expected_reset_valid = torch.tensor(
                    (True, False, False, False),
                    dtype=torch.bool,
                    device=requested_device,
                ).expand(batch, -1)
                reset_state_layout = reset_state_layout and bool(
                    torch.equal(reset_state.valid, expected_reset_valid)
                )
                reset = _rollout(
                    model, reset_state, future_actions, name="reset"
                )

                reverse_history = online[:, REVERSE_HISTORY_ORDER_V1]
                reverse = _rollout(
                    model,
                    _build_history_state(
                        model,
                        reverse_history,
                        history_actions,
                        name="reverse",
                    ),
                    future_actions,
                    name="reverse",
                )
                shuffled_history = torch.cat(
                    (torch.roll(online[:, :2], shifts=1, dims=0), online[:, 2:]),
                    dim=1,
                )
                shuffle = _rollout(
                    model,
                    _build_history_state(
                        model,
                        shuffled_history,
                        history_actions,
                        name="shuffle",
                    ),
                    future_actions,
                    name="shuffle",
                )

                target = target_all[:, 3:]
                persistence = target_all[:, 2:3].expand(-1, 4, -1, -1, -1)
                if update == 0:
                    update_zero_max_prediction_delta = max(
                        update_zero_max_prediction_delta,
                        *(
                            float(
                                (prediction - persistence)
                                .abs()
                                .max()
                                .detach()
                                .to(device="cpu")
                            )
                            for prediction in (
                                real,
                                wrong,
                                reset,
                                reverse,
                                shuffle,
                            )
                        ),
                    )
                for name, prediction in (
                    ("real", real),
                    ("persistence", persistence),
                    ("wrong_action", wrong),
                    ("reset", reset),
                    ("reverse", reverse),
                    ("shuffle", shuffle),
                ):
                    energy_chunks[name].append(_energy(prediction, target))

                target_rows.append(_participation_rows(target_all[:, 2]))
                online_rows.append(_participation_rows(online[:, 2]))
                # Memory rank must audit the learned predictor output, not the
                # already-audited observed encoder tape.  Otherwise a collapsed
                # causal reader could inherit a healthy online-state rank.
                memory_rows.append(
                    _participation_rows(real.reshape(-1, *TOKEN_SHAPE_V1))
                )

            target_modules_are_frozen = all(
                not parameter.requires_grad
                for module in target_modules
                for parameter in module.parameters()
            )
            target_gradient_tensor_count = sum(
                parameter.grad is not None
                for module in target_modules
                for parameter in module.parameters()
            )
            target_modules_are_eval = all(
                not module.training for module in target_modules
            )
    finally:
        model.train(was_training)

    energies = DelayLineEnergyVectorsV1(
        **{
            name: torch.cat(chunks, dim=0)
            for name, chunks in energy_chunks.items()
        }
    )
    temporal = metrics.evaluate_temporal_metrics(
        energies.real,
        energies.persistence,
        energies.wrong_action,
        energies.reset,
        energies.reverse,
        energies.shuffle,
        [row.scene_id for row in ordered],
        [row.family for row in ordered],
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    target_rank = metrics.participation_rank(torch.cat(target_rows, dim=0))
    online_rank = metrics.participation_rank(torch.cat(online_rows, dim=0))
    memory_rank = metrics.participation_rank(torch.cat(memory_rows, dim=0))
    hold_diagnostics = _hold_diagnostics(energies, ordered)
    after_access = loader.access_snapshot()

    update_zero_max_energy_delta = max(
        float((arm - energies.persistence).abs().max())
        for arm in (
            energies.real,
            energies.wrong_action,
            energies.reset,
            energies.reverse,
            energies.shuffle,
        )
    )
    checks = {
        "canonical_history_state_layout": canonical_state_layout,
        "reset_keeps_only_newest_token": reset_state_layout,
        "all_validation_latents_finite": True,
        "target_states_have_no_gradient": all_targets_no_grad,
        "target_modules_are_frozen": target_modules_are_frozen,
        "target_modules_have_zero_gradient_tensors": target_gradient_tensor_count == 0,
        "target_modules_are_eval": target_modules_are_eval,
        "future_rgb_online_access_is_zero": True,
        "target_state_finite": target_rank.finite,
        "target_state_nonzero_scale": target_rank.nonzero_scale,
        "online_state_finite": online_rank.finite,
        "online_state_nonzero_scale": online_rank.nonzero_scale,
        "memory_state_finite": memory_rank.finite,
        "memory_state_nonzero_scale": memory_rank.nonzero_scale,
        "update_zero_controls_equal_persistence": (
            update != 0 or update_zero_max_prediction_delta <= 1.0e-5
        ),
    }
    rank_diagnostics = {
        "target_state_noncollapsed": target_rank.noncollapsed,
        "online_state_noncollapsed": online_rank.noncollapsed,
        "memory_state_noncollapsed": memory_rank.noncollapsed,
    }
    absolute_noncollapse_enforced = update >= 250
    if absolute_noncollapse_enforced:
        checks.update(rank_diagnostics)
    access_receipt = _access_delta(before_access, after_access)
    return DelayLineValidationResultV1(
        update=update,
        temporal=temporal,
        target_rank=target_rank,
        online_rank=online_rank,
        memory_rank=memory_rank,
        energies=energies,
        hold_diagnostics=hold_diagnostics,
        access_receipt=access_receipt,
        memory_receipt={
            "algorithm": "one_pass_scalar_energy_bounded_rank_sampling",
            "retained_full_row_latent_count": 0,
            "retained_scalar_energy_count": 6 * len(ordered) * 4,
            "target_participation_row_count": target_rank.row_count,
            "online_participation_row_count": online_rank.row_count,
            "memory_participation_row_count": memory_rank.row_count,
            "participation_feature_dimension": 64,
            "participation_spatial_rows_per_map": (
                PARTICIPATION_SPATIAL_ROWS_PER_MAP_V1
            ),
            "validation_batch_size": VALIDATION_BATCH_SIZE_V1,
            "validation_rgb_cache_entry_count": after_access[
                "validation_cache_entry_count"
            ],
        },
        integrity={
            "schema": f"{SCHEMA_PREFIX}_temporal_integrity_v1",
            "observation_update": update,
            "validation_row_count": len(ordered),
            "online_encoded_frame_count": len(ordered) * 3,
            "target_encoded_frame_count": len(ordered) * 7,
            "future_rgb_online_access_count": 0,
            "wrong_action_mapping": "cyclic_plus_one_modulo_nine",
            "reverse_history_order": REVERSE_HISTORY_ORDER_V1,
            "shuffle_history_mapping": "cyclic_batch_rotation_of_older_two_tokens",
            "minimum_online_state_std": minimum_online_std,
            "minimum_target_state_std": minimum_target_std,
            "target_gradient_tensor_count": target_gradient_tensor_count,
            "update_zero_max_control_prediction_delta": (
                update_zero_max_prediction_delta
            ),
            "update_zero_max_control_energy_delta": update_zero_max_energy_delta,
            "absolute_noncollapse_enforced": absolute_noncollapse_enforced,
            "absolute_noncollapse_first_enforced_update": 250,
            "rank_diagnostics": rank_diagnostics,
            "checks": checks,
            "passed": all(checks.values()),
        },
    )


class DelayLineH6RuntimeV1:
    """Controller-facing owner of exact H6 roles and the RGB descriptor."""

    def __init__(
        self,
        *,
        runtime_data_root: Path,
        train_rows: Sequence[h6.H6V2Row],
        validation_rows: Sequence[h6.H6V2Row],
        metadata_preflight: Mapping[str, Any],
    ) -> None:
        self.runtime_data_root = Path(runtime_data_root)
        self.train_rows = tuple(train_rows)
        self.validation_rows = tuple(validation_rows)
        self.metadata_preflight = dict(metadata_preflight)
        if (
            len(self.train_rows) != h6.INDEX_BINDINGS["train"].row_count
            or len(self.validation_rows) != h6.INDEX_BINDINGS["val"].row_count
        ):
            raise DelayLineEvaluationContractError(
                "delay-line runtime requires both exact H6 indexes"
            )
        self._loader: SafeDelayLineRGBLoader | None = SafeDelayLineRGBLoader(
            self.runtime_data_root,
            (*self.train_rows, *self.validation_rows),
        )

    def __enter__(self) -> DelayLineH6RuntimeV1:
        self._require_loader()
        return self

    def __exit__(self, *_values: object) -> None:
        self.close()

    def _require_loader(self) -> SafeDelayLineRGBLoader:
        if self._loader is None:
            raise DelayLineEvaluationContractError("delay-line H6 runtime is closed")
        return self._loader

    def build_train_microbatches(
        self,
        update: int,
        device: Any,
    ) -> tuple[dict[str, torch.Tensor], ...]:
        return build_train_h6_microbatches_for_update(
            self.train_rows,
            update=update,
            loader=self._require_loader(),
            device=device,
        )

    def evaluate_temporal_metrics(
        self,
        model: Any,
        update: int,
        device: Any,
    ) -> DelayLineValidationResultV1:
        return stream_validation_delay_line_metrics(
            model,
            self.validation_rows,
            update=update,
            loader=self._require_loader(),
            device=device,
        )

    def access_receipt(self) -> dict[str, int]:
        return self._require_loader().access_snapshot()

    def preflight_receipt(self) -> dict[str, Any]:
        access = self._require_loader().access_snapshot()
        if any(access.values()):
            raise DelayLineEvaluationContractError(
                "preflight receipt must precede every RGB access"
            )
        return {
            **self.metadata_preflight,
            **access,
            "rgb_open_count": 0,
            "held_out_or_sealed_opened": False,
        }

    def terminal_access_receipt(self) -> dict[str, Any]:
        train_rows, validation_rows, preflight = schedule.load_exact_roles_v1(
            self.runtime_data_root
        )
        if (
            train_rows != self.train_rows
            or validation_rows != self.validation_rows
            or preflight != self.metadata_preflight
        ):
            raise DelayLineEvaluationContractError("terminal H6 index rehash changed")
        access = self._require_loader().access_snapshot()
        return {
            "schema": "lewm_go2_v18_delay_line_h6_terminal_access_v1",
            "terminal_index_rehash_count": 2,
            **access,
            "rgb_open_count": access["rgb_open_success_count"],
            "registered_train_row_count": len(self.train_rows),
            "registered_validation_row_count": len(self.validation_rows),
            "probability_calibration_opened": False,
            "navigation_executed": False,
            "held_out_or_sealed_opened": False,
            "generated_write_count": 0,
        }

    def close(self) -> None:
        if self._loader is not None:
            self._loader.close()
            self._loader = None


def load_delay_line_h6_runtime_v1(runtime_data_root: Path) -> DelayLineH6RuntimeV1:
    """Load exact H6 metadata and open the RGB root without reading a leaf."""

    root = Path(runtime_data_root)
    if not root.is_absolute():
        raise DelayLineEvaluationContractError("runtime data root must be absolute")
    train_rows, validation_rows, preflight = schedule.load_exact_roles_v1(root)
    return DelayLineH6RuntimeV1(
        runtime_data_root=root,
        train_rows=train_rows,
        validation_rows=validation_rows,
        metadata_preflight=preflight,
    )


__all__ = [
    "ACTION_COUNT_V1",
    "DelayLineEnergyVectorsV1",
    "DelayLineEvaluationContractError",
    "DelayLineH6RuntimeV1",
    "DelayLineValidationResultV1",
    "FUTURE_ACTIONS_KEY_V1",
    "FUTURE_RGB_KEY_V1",
    "HOLD_ACTION_INDEX_V1",
    "HISTORY_ACTIONS_KEY_V1",
    "HISTORY_RGB_KEY_V1",
    "HoldDiagnosticsV1",
    "OBSERVATION_UPDATES_V1",
    "REQUIRED_MEMORY_BATCH_KEYS_V1",
    "RGB_ROOT_RELATIVE_PATH_V1",
    "SafeDelayLineRGBLoader",
    "TOKEN_SHAPE_V1",
    "TRAIN_MICROBATCHES_PER_UPDATE_V1",
    "TRAIN_MICROBATCH_SIZE_V1",
    "VALIDATION_BATCH_SIZE_V1",
    "build_train_h6_microbatch",
    "build_train_h6_microbatches_for_update",
    "load_delay_line_h6_runtime_v1",
    "stream_validation_delay_line_metrics",
]
