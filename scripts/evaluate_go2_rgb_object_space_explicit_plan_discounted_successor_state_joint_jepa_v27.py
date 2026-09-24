#!/usr/bin/env python3
"""Bounded H6 loading and temporal evaluation for joint-JEPA V27.

This module has no lifecycle or artifact-writing authority.  It exposes one
descriptor-relative RGB runtime for the exact corrected-H6 indexes, four-row
training microbatches, and a two-pass validation evaluator that retains scalar
row energies and group sums, never all 2,048 row latents.  A validation-only
normalized-RGB cache avoids repeating Pillow and disk work across observations;
training RGB is never cached here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, Sequence

import torch

from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as data
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_object_space_height_volume import (
    latent_energy_per_row,
)
from scripts import (
    run_go2_rgb_object_space_explicit_plan_discounted_successor_state_joint_jepa_v27
    as training,
)


RGB_ROOT_RELATIVE_PATH_V27 = Path(
    ".generated/datagen_full/render_textured_v03"
)
TRAIN_MICROBATCH_SIZE_V27 = 4
TRAIN_MICROBATCHES_PER_UPDATE_V27 = 4
VALIDATION_BATCH_SIZE_V27 = 16
MAXIMUM_RGB_FILE_BYTES_V27 = 4 * 1024 * 1024
LATENT_SHAPE_V27 = (64, 64, 64)
VALIDATION_RGB_TENSOR_BYTES_V27 = 3 * 112 * 112 * 4
MAXIMUM_VALIDATION_CACHE_ENTRIES_V27 = data.INDEX_BINDINGS["val"].row_count * 5
MAXIMUM_VALIDATION_CACHE_BYTES_V27 = (
    MAXIMUM_VALIDATION_CACHE_ENTRIES_V27 * VALIDATION_RGB_TENSOR_BYTES_V27
)

_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


class V27EvaluationContractError(RuntimeError):
    """A V27 RGB-loading, batching, or evaluation invariant failed."""


@dataclass(frozen=True, slots=True)
class ValidationPlanEnergyVectorsV27:
    """Scalar-only result of the bounded validation forward passes."""

    correct: tuple[float, ...]
    persistence: tuple[float, ...]
    wrong_plan: tuple[float, ...]
    tail: tuple[float, ...]
    wrong_scene: Mapping[int, float]
    mean_prior: tuple[float, ...]
    access_receipt: Mapping[str, int]
    memory_receipt: Mapping[str, Any]
    integrity: Mapping[str, Any]


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _open_absolute_directory(path: Path) -> int:
    value = Path(path)
    if (
        not value.is_absolute()
        or any(part in {"", ".", ".."} for part in value.parts[1:])
        or not getattr(os, "O_NOFOLLOW", 0)
        or not getattr(os, "O_DIRECTORY", 0)
    ):
        raise V27EvaluationContractError(
            "RGB root must be canonical absolute and support no-follow opens"
        )
    descriptor = os.open(value.anchor, _DIR_FLAGS)
    try:
        for component in value.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class SafeV27RGBLoader:
    """No-follow loader restricted to registered corrected-H6 row leaves."""

    def __init__(
        self,
        runtime_data_root: Path,
        rows: Sequence[data.H6V2Row],
    ) -> None:
        root = Path(runtime_data_root)
        if not root.is_absolute():
            raise V27EvaluationContractError("runtime data root must be absolute")
        registered: dict[tuple[str, int], data.H6V2Row] = {}
        for row in rows:
            if not isinstance(row, data.H6V2Row):
                raise TypeError("registered RGB rows must be H6V2Row values")
            key = (row.role, row.index)
            if key in registered:
                raise V27EvaluationContractError("registered RGB row identity repeats")
            registered[key] = row
        if not registered:
            raise V27EvaluationContractError("RGB loader requires registered rows")
        try:
            descriptor = _open_absolute_directory(
                root / RGB_ROOT_RELATIVE_PATH_V27
            )
        except OSError as error:
            raise V27EvaluationContractError("frozen RGB root open failed") from error
        self._root_fd: int | None = descriptor
        self._rows = registered
        self._validation_cache_allowlist = frozenset(
            leaf
            for row in registered.values()
            if row.role == "val"
            for leaf in (row.current_rgb, *row.future_rgb)
        )
        self._validation_cache: dict[str, torch.Tensor] = {}
        if len(self._validation_cache_allowlist) > MAXIMUM_VALIDATION_CACHE_ENTRIES_V27:
            self.close()
            raise V27EvaluationContractError("validation RGB cache allowlist is too large")
        self._access: Counter[str] = Counter(
            {
                "rgb_tensor_request_count": 0,
                "rgb_open_attempt_count": 0,
                "rgb_open_success_count": 0,
                "rgb_decode_success_count": 0,
                "rgb_byte_count": 0,
                "validation_cache_hit_count": 0,
                "validation_cache_miss_count": 0,
                "validation_cache_insert_count": 0,
            }
        )

    def __enter__(self) -> SafeV27RGBLoader:
        self._require_open()
        return self

    def __exit__(self, *_values: object) -> None:
        self.close()

    def _require_open(self) -> int:
        if self._root_fd is None:
            raise V27EvaluationContractError("V27 RGB loader is closed")
        return self._root_fd

    def _registered(self, row: data.H6V2Row) -> data.H6V2Row:
        if not isinstance(row, data.H6V2Row):
            raise TypeError("RGB row must be H6V2Row")
        registered = self._rows.get((row.role, row.index))
        if registered is None or registered != row:
            raise V27EvaluationContractError("RGB row is not the registered index row")
        return registered

    def _read_leaf(self, row: data.H6V2Row, leaf: str) -> bytes:
        registered = self._registered(row)
        if leaf not in registered.rgb:
            raise V27EvaluationContractError("RGB leaf is absent from its registered row")
        try:
            canonical, _frame, _environment = data._validate_rgb_leaf(
                leaf,
                scene_id=registered.scene_id,
            )
        except data.V27DataContractError as error:
            raise V27EvaluationContractError("registered RGB leaf is invalid") from error
        parts = PurePosixPath(canonical).parts
        if len(parts) != 3:
            raise V27EvaluationContractError("registered RGB leaf shape changed")

        descriptor = os.dup(self._require_open())
        image_fd: int | None = None
        try:
            for component in parts[:-1]:
                child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            self._access["rgb_open_attempt_count"] += 1
            image_fd = os.open(parts[-1], _READ_FLAGS, dir_fd=descriptor)
            before = os.fstat(image_fd)
            if not stat.S_ISREG(before.st_mode):
                raise V27EvaluationContractError("RGB leaf is not a regular file")
            if not 0 < before.st_size <= MAXIMUM_RGB_FILE_BYTES_V27:
                raise V27EvaluationContractError("RGB leaf byte count is unsafe")
            chunks: list[bytes] = []
            consumed = 0
            while True:
                chunk = os.read(image_fd, min(1024 * 1024, MAXIMUM_RGB_FILE_BYTES_V27))
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > MAXIMUM_RGB_FILE_BYTES_V27:
                    raise V27EvaluationContractError("RGB leaf exceeded its byte cap")
                chunks.append(chunk)
            after = os.fstat(image_fd)
            raw = b"".join(chunks)
            if (
                _fingerprint(before) != _fingerprint(after)
                or len(raw) != before.st_size
            ):
                raise V27EvaluationContractError("RGB leaf changed while being read")
            self._access["rgb_open_success_count"] += 1
            self._access["rgb_byte_count"] += len(raw)
            return raw
        except V27EvaluationContractError:
            raise
        except OSError as error:
            raise V27EvaluationContractError("no-follow RGB leaf open failed") from error
        finally:
            if image_fd is not None:
                os.close(image_fd)
            os.close(descriptor)

    def _decode_leaf(self, row: data.H6V2Row, leaf: str) -> torch.Tensor:
        self._access["rgb_tensor_request_count"] += 1
        cacheable = row.role == "val" and leaf in self._validation_cache_allowlist
        if row.role == "val" and not cacheable:
            raise V27EvaluationContractError("validation RGB leaf left cache allowlist")
        if cacheable and leaf in self._validation_cache:
            self._access["validation_cache_hit_count"] += 1
            return self._validation_cache[leaf]
        if cacheable:
            self._access["validation_cache_miss_count"] += 1
        raw = self._read_leaf(row, leaf)
        try:
            tensor = data.rectify_h6_rgb_bytes(raw)
        except data.V27DataContractError as error:
            raise V27EvaluationContractError("registered RGB decode failed") from error
        self._access["rgb_decode_success_count"] += 1
        if cacheable:
            if len(self._validation_cache) >= MAXIMUM_VALIDATION_CACHE_ENTRIES_V27:
                raise V27EvaluationContractError("validation RGB cache capacity exceeded")
            self._validation_cache[leaf] = tensor
            self._access["validation_cache_insert_count"] += 1
        return tensor

    def load_current(self, row: data.H6V2Row) -> torch.Tensor:
        registered = self._registered(row)
        return self._decode_leaf(registered, registered.current_rgb)

    def load_future(self, row: data.H6V2Row) -> torch.Tensor:
        registered = self._registered(row)
        result = torch.stack(
            tuple(self._decode_leaf(registered, leaf) for leaf in registered.future_rgb),
            dim=0,
        )
        if tuple(result.shape) != (4, 3, 112, 112) or result.dtype != torch.float32:
            raise V27EvaluationContractError("future RGB tensor contract changed")
        return result

    def access_snapshot(self) -> dict[str, int]:
        return {
            **self._access,
            "validation_cache_entry_count": len(self._validation_cache),
            "validation_cache_bytes": (
                len(self._validation_cache) * VALIDATION_RGB_TENSOR_BYTES_V27
            ),
        }

    def close(self) -> None:
        self._validation_cache.clear()
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None


def _access_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    if set(before) != set(after):
        raise V27EvaluationContractError("RGB access counter schema changed")
    result = {name: int(after[name]) - int(before[name]) for name in before}
    if any(value < 0 for value in result.values()):
        raise V27EvaluationContractError("RGB access counters moved backwards")
    return result


def _stack_current(
    rows: Sequence[data.H6V2Row],
    loader: SafeV27RGBLoader,
    device: Any,
) -> torch.Tensor:
    value = torch.stack(tuple(loader.load_current(row) for row in rows), dim=0)
    value = value.to(device=torch.device(device), non_blocking=False)
    if (
        tuple(value.shape) != (len(rows), 3, 112, 112)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise V27EvaluationContractError("current RGB batch is invalid")
    return value


def _stack_future(
    rows: Sequence[data.H6V2Row],
    loader: SafeV27RGBLoader,
    device: Any,
) -> torch.Tensor:
    value = torch.stack(tuple(loader.load_future(row) for row in rows), dim=0)
    value = value.to(device=torch.device(device), non_blocking=False)
    if (
        tuple(value.shape) != (len(rows), 4, 3, 112, 112)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise V27EvaluationContractError("future RGB batch is invalid")
    return value


def build_train_h6_microbatch(
    train_rows: Sequence[data.H6V2Row],
    *,
    row_start: int,
    loader: SafeV27RGBLoader,
    device: Any,
) -> dict[str, torch.Tensor]:
    """Load one exact consecutive four-row H6 training microbatch."""

    if (
        type(row_start) is not int
        or row_start < 0
        or row_start % TRAIN_MICROBATCH_SIZE_V27
        or row_start + TRAIN_MICROBATCH_SIZE_V27 > data.TRAIN_PREFIX_ROWS
        or row_start + TRAIN_MICROBATCH_SIZE_V27 > len(train_rows)
    ):
        raise V27EvaluationContractError("train H6 microbatch offset is invalid")
    rows = tuple(train_rows[row_start : row_start + TRAIN_MICROBATCH_SIZE_V27])
    if any(
        row.role != "train" or row.index != row_start + offset
        for offset, row in enumerate(rows)
    ):
        raise V27EvaluationContractError("train H6 rows left frozen consecutive order")
    current = _stack_current(rows, loader, device)
    future = _stack_future(rows, loader, device)
    plans = torch.tensor(
        [row.plan for row in rows],
        dtype=torch.long,
        device=torch.device(device),
    )
    result = {
        training.H6_CURRENT_RGB_KEY_V27: current,
        training.H6_FUTURE_RGB_KEY_V27: future,
        training.H6_FUTURE_ACTIONS_KEY_V27: plans,
    }
    if tuple(result) != training.REQUIRED_H6_BATCH_KEYS_V27:
        raise V27EvaluationContractError("train H6 microbatch key order changed")
    return result


def build_train_h6_microbatches_for_update(
    train_rows: Sequence[data.H6V2Row],
    *,
    update: int,
    loader: SafeV27RGBLoader,
    device: Any,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Return the four exact B4 H6 microbatches for one V27 update."""

    if type(update) is not int or not 1 <= update <= training.MAXIMUM_UPDATES_V27:
        raise V27EvaluationContractError("V27 update must be in the range 1 through 400")
    if len(train_rows) != data.INDEX_BINDINGS["train"].row_count:
        raise V27EvaluationContractError("complete frozen train index is required")
    first = 16 * (update - 1)
    result = tuple(
        build_train_h6_microbatch(
            train_rows,
            row_start=first + microbatch * TRAIN_MICROBATCH_SIZE_V27,
            loader=loader,
            device=device,
        )
        for microbatch in range(TRAIN_MICROBATCHES_PER_UPDATE_V27)
    )
    if len(result) != TRAIN_MICROBATCHES_PER_UPDATE_V27:
        raise V27EvaluationContractError("V27 update lost an H6 microbatch")
    return result


def _validate_latent(value: Any, *, batch: int, name: str) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (batch, *LATENT_SHAPE_V27)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise V27EvaluationContractError(f"{name} is not finite Bx64x64x64 float32")
    return value


def _target_batch(
    model: Any,
    rows: Sequence[data.H6V2Row],
    loader: SafeV27RGBLoader,
    device: Any,
) -> torch.Tensor:
    future = _stack_future(rows, loader, device)
    states = model.encode_target(
        future.reshape(len(rows) * 4, 3, 112, 112)
    )
    states = _validate_latent(
        states,
        batch=len(rows) * 4,
        name="future EMA state",
    ).reshape(len(rows), 4, *LATENT_SHAPE_V27)
    return _validate_latent(
        data.discounted_successor_target(states),
        batch=len(rows),
        name="discounted successor target",
    )


def _current_latents(
    model: Any,
    rows: Sequence[data.H6V2Row],
    loader: SafeV27RGBLoader,
    device: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    current_rgb = _stack_current(rows, loader, device)
    encoding = model.encode_online_with_evidence(current_rgb)
    online = getattr(encoding, "latent", None)
    online = _validate_latent(
        online,
        batch=len(rows),
        name="online current state",
    )
    persistence = _validate_latent(
        model.encode_target(current_rgb),
        batch=len(rows),
        name="EMA current persistence state",
    )
    return online, persistence


def _online_current_batch(
    model: Any,
    rows: Sequence[data.H6V2Row],
    loader: SafeV27RGBLoader,
    device: Any,
) -> torch.Tensor:
    current_rgb = _stack_current(rows, loader, device)
    return _validate_latent(
        getattr(model.encode_online_with_evidence(current_rgb), "latent", None),
        batch=len(rows),
        name="online current state",
    )


def _prediction_batch(
    model: Any,
    current: torch.Tensor,
    plans: Sequence[tuple[int, ...]],
) -> torch.Tensor:
    action_plan = torch.tensor(
        plans,
        dtype=torch.long,
        device=current.device,
    )
    return _validate_latent(
        model.predict_plan_successor(current, action_plan),
        batch=len(plans),
        name="plan successor prediction",
    )


def _energy_values(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    energy = latent_energy_per_row(prediction, target)
    if (
        tuple(energy.shape) != (prediction.shape[0],)
        or not bool(torch.isfinite(energy).all())
        or bool((energy < 0.0).any())
    ):
        raise V27EvaluationContractError("V27 latent energy vector is invalid")
    return [float(value) for value in energy.detach().to(device="cpu").tolist()]


def _validate_validation_panel(
    rows: Sequence[data.H6V2Row],
    donors: data.DonorPanels,
) -> None:
    count = len(rows)
    if count <= 0 or any(
        row.role != "val" or row.index != index
        for index, row in enumerate(rows)
    ):
        raise V27EvaluationContractError("validation rows must be ordered from zero")
    if (
        len(donors.tail_donor_indices) != count
        or len(donors.wrong_plan_donor_indices) != count
        or len(donors.exact_plan_wrong_scene_donor_indices) != count
    ):
        raise V27EvaluationContractError("validation donor vector length changed")
    for index, row in enumerate(rows):
        tail = donors.tail_donor_indices[index]
        wrong = donors.wrong_plan_donor_indices[index]
        exact = donors.exact_plan_wrong_scene_donor_indices[index]
        if not 0 <= tail < count or not 0 <= wrong < count:
            raise V27EvaluationContractError("mandatory donor index is out of range")
        if (
            rows[tail].scene_id == row.scene_id
            or rows[tail].family != row.family
            or rows[tail].first_plan_action != row.first_plan_action
            or sum(
                left != right
                for left, right in zip(row.plan[1:], rows[tail].plan[1:], strict=True)
            )
            < 2
        ):
            raise V27EvaluationContractError("same-a0 tail donor changed")
        if (
            rows[wrong].scene_id == row.scene_id
            or rows[wrong].family != row.family
            or rows[wrong].plan == row.plan
        ):
            raise V27EvaluationContractError("full wrong-plan donor changed")
        if exact is not None and (
            not 0 <= exact < count
            or rows[exact].scene_id == row.scene_id
            or rows[exact].family != row.family
            or rows[exact].plan != row.plan
        ):
            raise V27EvaluationContractError("exact-plan wrong-scene donor changed")


def stream_validation_plan_energy_vectors(
    model: Any,
    rows: Sequence[data.H6V2Row],
    donors: data.DonorPanels,
    *,
    loader: SafeV27RGBLoader,
    device: Any,
) -> ValidationPlanEnergyVectorsV27:
    """Compute every temporal energy with bounded group-sum memory.

    Pass one computes the correct, persistence, wrong-plan, tail, and
    wrong-scene controls while accumulating float32 family/action target sums.
    Pass two recomputes targets scene by scene, subtracts each scene sum, and
    recomputes only the correct prediction needed for the leave-one-scene
    family/action mean prior.  No row latent survives a batch boundary.
    """

    ordered_rows = tuple(rows)
    _validate_validation_panel(ordered_rows, donors)
    requested_device = torch.device(device)
    parameters = tuple(model.parameters())
    if not parameters or any(parameter.device != requested_device for parameter in parameters):
        raise V27EvaluationContractError("model parameters do not share evaluation device")
    target_modules_method = getattr(model, "target_modules", None)
    if not callable(target_modules_method):
        raise V27EvaluationContractError("model does not expose EMA target modules")
    target_modules = tuple(target_modules_method())
    if not target_modules:
        raise V27EvaluationContractError("model EMA target module inventory is empty")

    count = len(ordered_rows)
    correct = [math.nan] * count
    persistence_energy = [math.nan] * count
    wrong_plan = [math.nan] * count
    tail = [math.nan] * count
    wrong_scene: dict[int, float] = {}
    mean_prior = [math.nan] * count
    global_sums: dict[tuple[str, int], torch.Tensor] = {}
    global_counts: Counter[tuple[str, int]] = Counter()
    before_access = loader.access_snapshot()
    peak_scene_sum_count = 0
    minimum_online_state_std = math.inf
    minimum_target_state_std = math.inf
    all_target_states_no_grad = True
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, count, VALIDATION_BATCH_SIZE_V27):
                batch_rows = ordered_rows[start : start + VALIDATION_BATCH_SIZE_V27]
                target = _target_batch(model, batch_rows, loader, requested_device)
                minimum_target_state_std = min(
                    minimum_target_state_std,
                    float(target.std(unbiased=False).item()),
                )
                all_target_states_no_grad = (
                    all_target_states_no_grad and not target.requires_grad
                )
                target_cpu = target.detach().to(device="cpu", dtype=torch.float32)
                for local, row in enumerate(batch_rows):
                    key = (row.family, row.first_plan_action)
                    if key not in global_sums:
                        global_sums[key] = torch.zeros_like(target_cpu[local])
                    global_sums[key].add_(target_cpu[local])
                    global_counts[key] += 1

                online, current_target = _current_latents(
                    model,
                    batch_rows,
                    loader,
                    requested_device,
                )
                minimum_online_state_std = min(
                    minimum_online_state_std,
                    float(online.std(unbiased=False).item()),
                )
                minimum_target_state_std = min(
                    minimum_target_state_std,
                    float(current_target.std(unbiased=False).item()),
                )
                all_target_states_no_grad = (
                    all_target_states_no_grad and not current_target.requires_grad
                )
                correct_plans = [row.plan for row in batch_rows]
                wrong_plans = [
                    ordered_rows[donors.wrong_plan_donor_indices[row.index]].plan
                    for row in batch_rows
                ]
                tail_plans = [
                    ordered_rows[donors.tail_donor_indices[row.index]].plan
                    for row in batch_rows
                ]
                all_predictions = _prediction_batch(
                    model,
                    torch.cat((online, online, online), dim=0),
                    (*correct_plans, *wrong_plans, *tail_plans),
                )
                batch_count = len(batch_rows)
                predicted_correct, predicted_wrong, predicted_tail = (
                    all_predictions[:batch_count],
                    all_predictions[batch_count : 2 * batch_count],
                    all_predictions[2 * batch_count :],
                )
                for destination, values in (
                    (correct, _energy_values(predicted_correct, target)),
                    (persistence_energy, _energy_values(current_target, target)),
                    (wrong_plan, _energy_values(predicted_wrong, target)),
                    (tail, _energy_values(predicted_tail, target)),
                ):
                    destination[start : start + batch_count] = values

                eligible = [
                    (local, donors.exact_plan_wrong_scene_donor_indices[row.index])
                    for local, row in enumerate(batch_rows)
                    if donors.exact_plan_wrong_scene_donor_indices[row.index] is not None
                ]
                if eligible:
                    unique_indices = tuple(
                        dict.fromkeys(int(donor) for _local, donor in eligible)
                    )
                    donor_target = _target_batch(
                        model,
                        tuple(ordered_rows[index] for index in unique_indices),
                        loader,
                        requested_device,
                    )
                    donor_by_index = {
                        index: donor_target[position]
                        for position, index in enumerate(unique_indices)
                    }
                    selected_prediction = torch.stack(
                        tuple(predicted_correct[local] for local, _donor in eligible),
                        dim=0,
                    )
                    selected_target = torch.stack(
                        tuple(donor_by_index[int(donor)] for _local, donor in eligible),
                        dim=0,
                    )
                    values = _energy_values(selected_prediction, selected_target)
                    for (local, _donor), value in zip(eligible, values, strict=True):
                        wrong_scene[batch_rows[local].index] = value

            scene_rows: dict[tuple[str, str], list[data.H6V2Row]] = defaultdict(list)
            for row in ordered_rows:
                scene_rows[(row.family, row.scene_id)].append(row)
            for scene_key in sorted(scene_rows):
                selected_rows = tuple(sorted(scene_rows[scene_key], key=lambda row: row.index))
                scene_sums: dict[tuple[str, int], torch.Tensor] = {}
                scene_counts: Counter[tuple[str, int]] = Counter()
                for start in range(0, len(selected_rows), VALIDATION_BATCH_SIZE_V27):
                    batch_rows = selected_rows[start : start + VALIDATION_BATCH_SIZE_V27]
                    target = _target_batch(model, batch_rows, loader, requested_device)
                    target_cpu = target.detach().to(device="cpu", dtype=torch.float32)
                    for local, row in enumerate(batch_rows):
                        key = (row.family, row.first_plan_action)
                        if key not in scene_sums:
                            scene_sums[key] = torch.zeros_like(target_cpu[local])
                        scene_sums[key].add_(target_cpu[local])
                        scene_counts[key] += 1
                peak_scene_sum_count = max(peak_scene_sum_count, len(scene_sums))

                for start in range(0, len(selected_rows), VALIDATION_BATCH_SIZE_V27):
                    batch_rows = selected_rows[start : start + VALIDATION_BATCH_SIZE_V27]
                    online = _online_current_batch(
                        model,
                        batch_rows,
                        loader,
                        requested_device,
                    )
                    prediction = _prediction_batch(
                        model,
                        online,
                        [row.plan for row in batch_rows],
                    )
                    priors: list[torch.Tensor] = []
                    for row in batch_rows:
                        key = (row.family, row.first_plan_action)
                        prior_count = global_counts[key] - scene_counts[key]
                        if prior_count <= 0:
                            raise V27EvaluationContractError(
                                "leave-one-scene family/action prior has no donor row"
                            )
                        prior = (global_sums[key] - scene_sums[key]) / float(prior_count)
                        priors.append(prior)
                    prior_batch = torch.stack(priors, dim=0).to(
                        device=requested_device,
                        dtype=torch.float32,
                        non_blocking=False,
                    )
                    values = _energy_values(prediction, prior_batch)
                    for row, value in zip(batch_rows, values, strict=True):
                        mean_prior[row.index] = value
    finally:
        model.train(was_training)

    scalar_vectors = (correct, persistence_energy, wrong_plan, tail, mean_prior)
    if (
        any(not math.isfinite(value) or value < 0.0 for vector in scalar_vectors for value in vector)
        or set(wrong_scene) != set(donors.exact_plan_eligible_indices)
        or any(not math.isfinite(value) or value < 0.0 for value in wrong_scene.values())
    ):
        raise V27EvaluationContractError("streamed validation energies are incomplete")
    after_access = loader.access_snapshot()
    global_sum_bytes = sum(
        value.numel() * value.element_size() for value in global_sums.values()
    )
    target_gradient_tensor_count = sum(
        parameter.grad is not None
        for module in target_modules
        for parameter in module.parameters()
    )
    integrity_checks = {
        "all_validation_latents_finite": True,
        "online_states_noncollapsed": minimum_online_state_std > 0.0,
        "target_states_noncollapsed": minimum_target_state_std > 0.0,
        "target_states_have_no_gradient": all_target_states_no_grad,
        "target_modules_are_frozen": all(
            not parameter.requires_grad
            for module in target_modules
            for parameter in module.parameters()
        ),
        "target_modules_have_zero_gradient_tensors": target_gradient_tensor_count == 0,
        "target_modules_are_eval": all(not module.training for module in target_modules),
    }
    return ValidationPlanEnergyVectorsV27(
        correct=tuple(correct),
        persistence=tuple(persistence_energy),
        wrong_plan=tuple(wrong_plan),
        tail=tuple(tail),
        wrong_scene=dict(sorted(wrong_scene.items())),
        mean_prior=tuple(mean_prior),
        access_receipt=_access_delta(before_access, after_access),
        memory_receipt={
            "algorithm": "two_pass_float32_group_sum_recompute",
            "retained_full_row_latent_count": 0,
            "retained_scalar_row_energy_count": 5 * count + len(wrong_scene),
            "global_family_a0_sum_tensor_count": len(global_sums),
            "global_family_a0_sum_bytes": global_sum_bytes,
            "peak_scene_a0_sum_tensor_count": peak_scene_sum_count,
            "validation_batch_size": VALIDATION_BATCH_SIZE_V27,
            "complete_future_target_pass_count": 2,
            "complete_online_current_pass_count": 2,
            "complete_ema_current_pass_count": 1,
            "wrong_scene_donor_target_row_count": len(wrong_scene),
            "validation_rgb_cache_entry_count": after_access[
                "validation_cache_entry_count"
            ],
            "validation_rgb_cache_bytes": after_access["validation_cache_bytes"],
            "validation_rgb_cache_maximum_entries": MAXIMUM_VALIDATION_CACHE_ENTRIES_V27,
            "validation_rgb_cache_maximum_bytes": MAXIMUM_VALIDATION_CACHE_BYTES_V27,
            "validation_rgb_cache_scope": "validation_only_normalized_float32",
        },
        integrity={
            "schema": "lewm_go2_v27_plan_observation_integrity_v1",
            "minimum_online_state_std": minimum_online_state_std,
            "minimum_target_state_std": minimum_target_state_std,
            "target_gradient_tensor_count": target_gradient_tensor_count,
            "checks": integrity_checks,
            "passed": all(integrity_checks.values()),
        },
    )


class V27H6Runtime:
    """Controller-facing owner of exact indexes, donors, and RGB descriptor."""

    def __init__(
        self,
        *,
        runtime_data_root: Path,
        train_rows: Sequence[data.H6V2Row],
        validation_rows: Sequence[data.H6V2Row],
        donor_panels: data.DonorPanels,
        train_audit: Mapping[str, Any],
        validation_audit: Mapping[str, Any],
    ) -> None:
        self.runtime_data_root = Path(runtime_data_root)
        self.train_rows = tuple(train_rows)
        self.validation_rows = tuple(validation_rows)
        self.donor_panels = donor_panels
        self.train_audit = dict(train_audit)
        self.validation_audit = dict(validation_audit)
        if (
            len(self.train_rows) != data.INDEX_BINDINGS["train"].row_count
            or len(self.validation_rows) != data.INDEX_BINDINGS["val"].row_count
        ):
            raise V27EvaluationContractError("V27 runtime requires both exact indexes")
        self._loader: SafeV27RGBLoader | None = SafeV27RGBLoader(
            self.runtime_data_root,
            (*self.train_rows, *self.validation_rows),
        )

    def __enter__(self) -> V27H6Runtime:
        self._require_loader()
        return self

    def __exit__(self, *_values: object) -> None:
        self.close()

    def _require_loader(self) -> SafeV27RGBLoader:
        if self._loader is None:
            raise V27EvaluationContractError("V27 H6 runtime is closed")
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

    def evaluate_plan_metrics(
        self,
        model: Any,
        update: int,
        device: Any,
    ) -> dict[str, Any]:
        if type(update) is not int or update not in (0, 100, 400):
            raise V27EvaluationContractError("V27 observation update must be 0, 100, or 400")
        vectors = stream_validation_plan_energy_vectors(
            model,
            self.validation_rows,
            self.donor_panels,
            loader=self._require_loader(),
            device=device,
        )
        summary = data.summarize_plan_energies(
            self.validation_rows,
            observation_update=update,
            correct_energy=vectors.correct,
            persistence_energy=vectors.persistence,
            wrong_plan_energy=vectors.wrong_plan,
            tail_energy=vectors.tail,
            wrong_scene_energy=vectors.wrong_scene,
            mean_prior_energy=vectors.mean_prior,
        )
        return {
            **summary,
            "integrity": dict(vectors.integrity),
            "donor_panel": self.donor_panels.audit(),
            "temporal_access": dict(vectors.access_receipt),
            "bounded_memory": dict(vectors.memory_receipt),
            "energy_means": {
                "wrong_plan": sum(vectors.wrong_plan) / len(vectors.wrong_plan),
                "tail": sum(vectors.tail) / len(vectors.tail),
                "wrong_scene": sum(vectors.wrong_scene.values())
                / len(vectors.wrong_scene),
                "mean_prior": sum(vectors.mean_prior) / len(vectors.mean_prior),
            },
        }

    def access_receipt(self) -> dict[str, int]:
        return self._require_loader().access_snapshot()

    def preflight_receipt(self) -> dict[str, Any]:
        """Return bound metadata facts before the first RGB leaf access."""

        access = self._require_loader().access_snapshot()
        if any(access.values()):
            raise V27EvaluationContractError(
                "V27 H6 preflight receipt must precede every RGB access"
            )
        return {
            "schema": "lewm_go2_v27_h6_runtime_preflight_v1",
            "status": "PASS_METADATA_ONLY_PREFLIGHT",
            "train": dict(self.train_audit),
            "validation": dict(self.validation_audit),
            "train_prefix_rows": data.TRAIN_PREFIX_ROWS,
            "train_validation_scene_overlap_count": 0,
            "train_validation_rgb_path_overlap_count": 0,
            "donors": self.donor_panels.audit(),
            **access,
            "rgb_open_count": access["rgb_open_success_count"],
            "gpu_use_count": 0,
            "generated_write_count": 0,
        }

    def terminal_access_receipt(self) -> dict[str, Any]:
        """Rehash both frozen indexes and report cumulative RGB leaf access."""

        loader = self._require_loader()
        train_rows, train_audit = data.load_bound_index(
            self.runtime_data_root,
            role="train",
        )
        validation_rows, validation_audit = data.load_bound_index(
            self.runtime_data_root,
            role="val",
        )
        if (
            train_rows != self.train_rows
            or validation_rows != self.validation_rows
            or train_audit != self.train_audit
            or validation_audit != self.validation_audit
        ):
            raise V27EvaluationContractError("terminal H6 index rehash changed")
        access = loader.access_snapshot()
        return {
            "schema": "lewm_go2_v27_h6_terminal_access_v1",
            "terminal_index_rehash_count": 2,
            "train_index": dict(train_audit),
            "validation_index": dict(validation_audit),
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


def load_v27_h6_runtime(runtime_data_root: Path) -> V27H6Runtime:
    """Load both exact indexes and open the fixed RGB root without reading RGB."""

    root = Path(runtime_data_root)
    if not root.is_absolute():
        raise V27EvaluationContractError("runtime data root must be absolute")
    train_rows, train_audit = data.load_bound_index(root, role="train")
    validation_rows, validation_audit = data.load_bound_index(root, role="val")
    if (
        {row.scene_id for row in train_rows}
        & {row.scene_id for row in validation_rows}
        or {leaf for row in train_rows for leaf in row.rgb}
        & {leaf for row in validation_rows for leaf in row.rgb}
    ):
        raise V27EvaluationContractError("V27 train and validation indexes overlap")
    donors = data.build_donor_panels(validation_rows)
    return V27H6Runtime(
        runtime_data_root=root,
        train_rows=train_rows,
        validation_rows=validation_rows,
        donor_panels=donors,
        train_audit=train_audit,
        validation_audit=validation_audit,
    )


__all__ = [
    "LATENT_SHAPE_V27",
    "RGB_ROOT_RELATIVE_PATH_V27",
    "SafeV27RGBLoader",
    "TRAIN_MICROBATCHES_PER_UPDATE_V27",
    "TRAIN_MICROBATCH_SIZE_V27",
    "V27EvaluationContractError",
    "V27H6Runtime",
    "VALIDATION_BATCH_SIZE_V27",
    "ValidationPlanEnergyVectorsV27",
    "build_train_h6_microbatch",
    "build_train_h6_microbatches_for_update",
    "load_v27_h6_runtime",
    "stream_validation_plan_energy_vectors",
]
