#!/usr/bin/env python3
"""Guarded H6 runtime and evaluator for recurrent patch-memory JEPA V1.

This module owns no training loop, checkpoint loading, lifecycle policy, or
artifact publication.  Its RGB loader is deliberately narrower than the
legacy V18 and V27 evaluators: a factual row can expose only ``rgb[0:4]`` and
a wrong-history donor can expose only ``rgb[0:2]``.  Every request, open, and
decode is counted by role, access kind, and registered H6 position.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
import math
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Mapping, NamedTuple, Sequence

import torch

from lewm.benchmarks import (
    go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as metrics,
)
from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as h6
from scripts import (
    evaluate_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1
    as spatial_evaluation,
)


SCHEMA_PREFIX_V1 = "lewm_go2_rgb_recurrent_patch_memory_temporal_jepa_v1"
RGB_ROOT_RELATIVE_PATH_V1 = Path(".generated/datagen_full/render_textured_v03")
MAXIMUM_RGB_FILE_BYTES_V1 = 4 * 1024 * 1024
TRAIN_ROW_COUNT_V1 = 16_000
VALIDATION_ROW_COUNT_V1 = 2_048
TRAIN_SCHEDULE_ROW_COUNT_V1 = 4_000
MAXIMUM_UPDATES_V1 = 400
TRAIN_MICROBATCH_SIZE_V1 = 2
TRAIN_MICROBATCHES_PER_UPDATE_V1 = 5
TRAIN_SEQUENCES_PER_UPDATE_V1 = 10
TRAIN_LOGICAL_RGB_PRESENTATIONS_PER_UPDATE_V1 = 40
VALIDATION_BATCH_SIZE_V1 = 32
ACTION_COUNT_V1 = 9
HOLD_ACTION_INDEX_V1 = 6
TARGET_TOKEN_COUNT_V1 = 64
FEATURE_DIMENSION_V1 = 192

CONTEXT_RGB_KEY_V1 = "context_rgb"
ACTION_SEQUENCE_KEY_V1 = "action_sequence"
TARGET_RGB_KEY_V1 = "target_rgb"
REQUIRED_MODEL_BATCH_KEYS_V1 = (
    CONTEXT_RGB_KEY_V1,
    ACTION_SEQUENCE_KEY_V1,
    TARGET_RGB_KEY_V1,
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
_ACCESS_ROLES = ("train", "val")
_ACCESS_KINDS = ("factual", "donor")
_ACCESS_OPERATIONS = ("request", "open_attempt", "open_success", "decode_success")
_FACTUAL_POSITIONS = frozenset((0, 1, 2, 3))
_DONOR_POSITIONS = frozenset((0, 1))


class TemporalEvaluationContractError(RuntimeError):
    """A frozen temporal data, loader, model, or metric contract failed."""


class TemporalModelContractError(TemporalEvaluationContractError):
    """The model did not expose the exact temporal evaluation interface."""


def _fingerprint(info: os.stat_result) -> tuple[int, ...]:
    return (
        int(info.st_dev),
        int(info.st_ino),
        int(info.st_mode),
        int(info.st_size),
        int(info.st_mtime_ns),
        int(info.st_ctime_ns),
    )


def _open_absolute_directory(value: Path) -> int:
    selected = Path(value)
    if (
        not selected.is_absolute()
        or any(part in {"", ".", ".."} for part in selected.parts[1:])
        or not getattr(os, "O_NOFOLLOW", 0)
        or not getattr(os, "O_DIRECTORY", 0)
    ):
        raise TemporalEvaluationContractError(
            "RGB root must be canonical absolute and support no-follow opens"
        )
    descriptor = os.open(selected.anchor, _DIR_FLAGS)
    try:
        for component in selected.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _position_counter_key(
    role: str,
    access_kind: str,
    position: int,
    operation: str,
) -> str:
    return (
        f"{role}_{access_kind}_rgb_position_{position}_{operation}_count"
    )


def _initial_access_counters() -> Counter[str]:
    result: Counter[str] = Counter(
        {
            "rgb_tensor_request_count": 0,
            "rgb_open_attempt_count": 0,
            "rgb_open_success_count": 0,
            "rgb_decode_success_count": 0,
            "rgb_byte_count": 0,
            "denied_rgb_position_request_count": 0,
        }
    )
    for role in _ACCESS_ROLES:
        for access_kind in _ACCESS_KINDS:
            for position in range(7):
                for operation in _ACCESS_OPERATIONS:
                    result[
                        _position_counter_key(
                            role,
                            access_kind,
                            position,
                            operation,
                        )
                    ] = 0
    return result


class SafeTemporalH6RGBLoaderV1:
    """No-follow loader limited to four factual and two donor H6 leaves."""

    def __init__(
        self,
        runtime_data_root: Path,
        rows: Sequence[h6.H6V2Row],
    ) -> None:
        root = Path(runtime_data_root)
        if not root.is_absolute():
            raise TemporalEvaluationContractError(
                "runtime data root must be absolute"
            )
        registered: dict[tuple[str, int], h6.H6V2Row] = {}
        for row in rows:
            if not isinstance(row, h6.H6V2Row):
                raise TypeError("registered RGB rows must be H6V2Row values")
            key = (row.role, row.index)
            if key in registered:
                raise TemporalEvaluationContractError(
                    "registered RGB row identity repeats"
                )
            registered[key] = row
        if not registered:
            raise TemporalEvaluationContractError(
                "temporal RGB loader requires registered rows"
            )
        try:
            descriptor = _open_absolute_directory(
                root / RGB_ROOT_RELATIVE_PATH_V1
            )
        except OSError as error:
            raise TemporalEvaluationContractError(
                "frozen temporal RGB root open failed"
            ) from error
        self._root_fd: int | None = descriptor
        self._rows = registered
        self._access = _initial_access_counters()

    def __enter__(self) -> SafeTemporalH6RGBLoaderV1:
        self._require_open()
        return self

    def __exit__(self, *_values: object) -> None:
        self.close()

    def _require_open(self) -> int:
        if self._root_fd is None:
            raise TemporalEvaluationContractError(
                "temporal RGB loader is closed"
            )
        return self._root_fd

    def _registered(self, row: h6.H6V2Row) -> h6.H6V2Row:
        if not isinstance(row, h6.H6V2Row):
            raise TypeError("temporal RGB row must be H6V2Row")
        registered = self._rows.get((row.role, row.index))
        if registered is None or registered != row:
            raise TemporalEvaluationContractError(
                "temporal RGB row is not its registered index row"
            )
        return registered

    def _validate_access(
        self,
        row: h6.H6V2Row,
        *,
        position: int,
        access_kind: str,
    ) -> h6.H6V2Row:
        registered = self._registered(row)
        if (
            type(position) is not int
            or type(access_kind) is not str
            or access_kind not in _ACCESS_KINDS
        ):
            raise TypeError("temporal RGB position or access kind is invalid")
        allowed = (
            access_kind == "factual"
            and position in _FACTUAL_POSITIONS
        ) or (
            access_kind == "donor"
            and registered.role == "val"
            and position in _DONOR_POSITIONS
        )
        if not allowed:
            self._access["denied_rgb_position_request_count"] += 1
            raise PermissionError(
                f"{access_kind} RGB position {position} is outside V1 authority"
            )
        return registered

    def _read_leaf(
        self,
        row: h6.H6V2Row,
        *,
        position: int,
        access_kind: str,
    ) -> bytes:
        registered = self._validate_access(
            row,
            position=position,
            access_kind=access_kind,
        )
        leaf = registered.rgb[position]
        try:
            canonical, _frame, _environment = h6._validate_rgb_leaf(
                leaf,
                scene_id=registered.scene_id,
            )
        except h6.V27DataContractError as error:
            raise TemporalEvaluationContractError(
                "registered temporal RGB leaf is invalid"
            ) from error
        parts = PurePosixPath(canonical).parts
        if len(parts) != 3:
            raise TemporalEvaluationContractError(
                "registered temporal RGB leaf shape changed"
            )

        descriptor = os.dup(self._require_open())
        image_fd: int | None = None
        attempt_key = _position_counter_key(
            registered.role,
            access_kind,
            position,
            "open_attempt",
        )
        success_key = _position_counter_key(
            registered.role,
            access_kind,
            position,
            "open_success",
        )
        try:
            for component in parts[:-1]:
                child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            self._access["rgb_open_attempt_count"] += 1
            self._access[attempt_key] += 1
            image_fd = os.open(parts[-1], _READ_FLAGS, dir_fd=descriptor)
            before = os.fstat(image_fd)
            if not stat.S_ISREG(before.st_mode):
                raise TemporalEvaluationContractError(
                    "temporal RGB leaf is not a regular file"
                )
            if not 0 < before.st_size <= MAXIMUM_RGB_FILE_BYTES_V1:
                raise TemporalEvaluationContractError(
                    "temporal RGB leaf byte count is unsafe"
                )
            chunks: list[bytes] = []
            consumed = 0
            while True:
                chunk = os.read(
                    image_fd,
                    min(1024 * 1024, MAXIMUM_RGB_FILE_BYTES_V1),
                )
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > MAXIMUM_RGB_FILE_BYTES_V1:
                    raise TemporalEvaluationContractError(
                        "temporal RGB leaf exceeded its byte cap"
                    )
                chunks.append(chunk)
            after = os.fstat(image_fd)
            raw = b"".join(chunks)
            if (
                _fingerprint(before) != _fingerprint(after)
                or len(raw) != before.st_size
            ):
                raise TemporalEvaluationContractError(
                    "temporal RGB leaf changed while being read"
                )
            self._access["rgb_open_success_count"] += 1
            self._access[success_key] += 1
            self._access["rgb_byte_count"] += len(raw)
            return raw
        except TemporalEvaluationContractError:
            raise
        except OSError as error:
            raise TemporalEvaluationContractError(
                "no-follow temporal RGB leaf open failed"
            ) from error
        finally:
            if image_fd is not None:
                os.close(image_fd)
            os.close(descriptor)

    def _decode_position(
        self,
        row: h6.H6V2Row,
        *,
        position: int,
        access_kind: str,
    ) -> torch.Tensor:
        registered = self._validate_access(
            row,
            position=position,
            access_kind=access_kind,
        )
        self._access["rgb_tensor_request_count"] += 1
        self._access[
            _position_counter_key(
                registered.role,
                access_kind,
                position,
                "request",
            )
        ] += 1
        raw = self._read_leaf(
            registered,
            position=position,
            access_kind=access_kind,
        )
        try:
            value = h6.rectify_h6_rgb_bytes(raw)
        except h6.V27DataContractError as error:
            raise TemporalEvaluationContractError(
                "registered temporal RGB decode failed"
            ) from error
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != (3, 112, 112)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise TemporalEvaluationContractError(
                "temporal RGB decode returned an invalid tensor"
            )
        self._access["rgb_decode_success_count"] += 1
        self._access[
            _position_counter_key(
                registered.role,
                access_kind,
                position,
                "decode_success",
            )
        ] += 1
        return value

    def load_factual(self, row: h6.H6V2Row) -> torch.Tensor:
        """Return exact factual ``rgb[0:4]`` as a finite ``4x3x112x112``."""

        registered = self._registered(row)
        value = torch.stack(
            tuple(
                self._decode_position(
                    registered,
                    position=position,
                    access_kind="factual",
                )
                for position in range(4)
            ),
            dim=0,
        )
        if (
            tuple(value.shape) != (4, 3, 112, 112)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise TemporalEvaluationContractError(
                "factual temporal RGB tensor contract changed"
            )
        return value

    def load_donor_history(self, row: h6.H6V2Row) -> torch.Tensor:
        """Return only a validation donor's registered ``rgb[0:2]``."""

        registered = self._registered(row)
        value = torch.stack(
            tuple(
                self._decode_position(
                    registered,
                    position=position,
                    access_kind="donor",
                )
                for position in range(2)
            ),
            dim=0,
        )
        if (
            tuple(value.shape) != (2, 3, 112, 112)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
        ):
            raise TemporalEvaluationContractError(
                "donor temporal RGB tensor contract changed"
            )
        return value

    def access_snapshot(self) -> dict[str, int]:
        return dict(self._access)

    def close(self) -> None:
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None


def access_delta_v1(
    before: Mapping[str, int],
    after: Mapping[str, int],
) -> dict[str, int]:
    """Return a strict nonnegative delta over the fixed access schema."""

    if set(before) != set(after):
        raise TemporalEvaluationContractError(
            "temporal RGB access counter schema changed"
        )
    result = {key: int(after[key]) - int(before[key]) for key in before}
    if any(value < 0 for value in result.values()):
        raise TemporalEvaluationContractError(
            "temporal RGB access counters moved backwards"
        )
    return result


def forbidden_rgb_open_count_v1(receipt: Mapping[str, int]) -> int:
    """Count every factual/donor open outside the preregistered boundary."""

    total = 0
    for role in _ACCESS_ROLES:
        for position in range(4, 7):
            total += int(
                receipt[
                    _position_counter_key(
                        role,
                        "factual",
                        position,
                        "open_success",
                    )
                ]
            )
        donor_start = 0 if role == "train" else 2
        for position in range(donor_start, 7):
            total += int(
                receipt[
                    _position_counter_key(
                        role,
                        "donor",
                        position,
                        "open_success",
                    )
                ]
            )
    return total


@dataclass(frozen=True, slots=True)
class TemporalSequenceBatchV1:
    """The only tensors supplied to the temporal model for factual rows."""

    row_indices: tuple[int, ...]
    families: tuple[str, ...]
    scene_ids: tuple[str, ...]
    context_rgb: torch.Tensor
    action_sequence: torch.Tensor
    target_rgb: torch.Tensor

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            CONTEXT_RGB_KEY_V1: self.context_rgb,
            ACTION_SEQUENCE_KEY_V1: self.action_sequence,
            TARGET_RGB_KEY_V1: self.target_rgb,
        }


@dataclass(frozen=True, slots=True)
class TemporalControlBatchV1:
    """Factual tensors and exact validation-only corruptions."""

    factual: TemporalSequenceBatchV1
    donor_indices: tuple[int, ...]
    wrong_history_context_rgb: torch.Tensor
    wrong_history_action_sequence: torch.Tensor
    wrong_action_sequence: torch.Tensor
    wrong_action_eligible: torch.Tensor


def _stack_factual_sequences(
    rows: Sequence[h6.H6V2Row],
    *,
    loader: SafeTemporalH6RGBLoaderV1,
    device: Any,
) -> torch.Tensor:
    selected = tuple(rows)
    if not selected:
        raise TemporalEvaluationContractError(
            "a factual temporal batch cannot be empty"
        )
    value = torch.stack(
        tuple(loader.load_factual(row) for row in selected),
        dim=0,
    ).to(device=torch.device(device), non_blocking=False)
    if (
        tuple(value.shape) != (len(selected), 4, 3, 112, 112)
        or value.dtype != torch.float32
        or value.requires_grad
        or not bool(torch.isfinite(value).all())
    ):
        raise TemporalEvaluationContractError(
            "stacked factual temporal RGB is invalid"
        )
    return value


def build_sequence_batch_v1(
    rows: Sequence[h6.H6V2Row],
    *,
    loader: SafeTemporalH6RGBLoaderV1,
    device: Any,
) -> TemporalSequenceBatchV1:
    """Build model tensors from exactly ``rgb[0:4]`` and ``actions[0:3]``."""

    selected = tuple(rows)
    if (
        not selected
        or any(
            not isinstance(row, h6.H6V2Row)
            or row.role not in _ACCESS_ROLES
            or len(row.rgb) != 7
            or len(row.actions) != 6
            for row in selected
        )
    ):
        raise TemporalEvaluationContractError(
            "temporal batch rows do not match registered H6 metadata"
        )
    sequences = _stack_factual_sequences(
        selected,
        loader=loader,
        device=device,
    )
    actions = torch.tensor(
        [row.actions[:3] for row in selected],
        dtype=torch.long,
        device=sequences.device,
    )
    if (
        tuple(actions.shape) != (len(selected), 3)
        or bool((actions < 0).any())
        or bool((actions >= ACTION_COUNT_V1).any())
    ):
        raise TemporalEvaluationContractError(
            "model-visible temporal action tensor is invalid"
        )
    result = TemporalSequenceBatchV1(
        row_indices=tuple(row.index for row in selected),
        families=tuple(row.family for row in selected),
        scene_ids=tuple(row.scene_id for row in selected),
        context_rgb=sequences[:, :3],
        action_sequence=actions,
        target_rgb=sequences[:, 3],
    )
    if (
        tuple(result.context_rgb.shape)
        != (len(selected), 3, 3, 112, 112)
        or tuple(result.target_rgb.shape)
        != (len(selected), 3, 112, 112)
        or tuple(result.model_inputs()) != REQUIRED_MODEL_BATCH_KEYS_V1
    ):
        raise TemporalEvaluationContractError(
            "temporal model batch schema changed"
        )
    return result


def build_control_batch_v1(
    rows: Sequence[h6.H6V2Row],
    *,
    validation_rows: Sequence[h6.H6V2Row],
    donor_indices: Sequence[int],
    loader: SafeTemporalH6RGBLoaderV1,
    device: Any,
) -> TemporalControlBatchV1:
    """Build all temporal corruptions without opening a donor's current RGB."""

    selected = tuple(rows)
    ordered_validation = tuple(validation_rows)
    selected_donors = tuple(int(value) for value in donor_indices)
    if (
        not selected
        or len(selected_donors) != len(selected)
        or len(ordered_validation) != VALIDATION_ROW_COUNT_V1
        or any(
            row.role != "val" or row.index < 0
            for row in selected
        )
        or any(
            row.role != "val" or row.index != index
            for index, row in enumerate(ordered_validation)
        )
        or any(
            not 0 <= donor < len(ordered_validation)
            for donor in selected_donors
        )
    ):
        raise TemporalEvaluationContractError(
            "temporal control rows or donor identities are invalid"
        )
    factual = build_sequence_batch_v1(
        selected,
        loader=loader,
        device=device,
    )
    donor_rows = tuple(ordered_validation[index] for index in selected_donors)
    if any(
        donor.family != row.family or donor.scene_id == row.scene_id
        for row, donor in zip(selected, donor_rows, strict=True)
    ):
        raise TemporalEvaluationContractError(
            "wrong-history donor left its same-family different-scene panel"
        )
    donor_history = torch.stack(
        tuple(loader.load_donor_history(row) for row in donor_rows),
        dim=0,
    ).to(device=factual.context_rgb.device, non_blocking=False)
    donor_actions = torch.tensor(
        [row.actions[:2] for row in donor_rows],
        dtype=torch.long,
        device=factual.action_sequence.device,
    )
    wrong_history_context = torch.cat(
        (donor_history, factual.context_rgb[:, 2:3]),
        dim=1,
    )
    wrong_history_actions = torch.cat(
        (donor_actions, factual.action_sequence[:, 2:3]),
        dim=1,
    )
    wrong_actions = factual.action_sequence.clone()
    wrong_actions[:, 2] = (wrong_actions[:, 2] + 1).remainder(ACTION_COUNT_V1)
    eligible = factual.action_sequence[:, 2].ne(HOLD_ACTION_INDEX_V1)
    if (
        tuple(donor_history.shape)
        != (len(selected), 2, 3, 112, 112)
        or tuple(wrong_history_context.shape)
        != tuple(factual.context_rgb.shape)
        or tuple(wrong_history_actions.shape)
        != tuple(factual.action_sequence.shape)
        or tuple(wrong_actions.shape) != tuple(factual.action_sequence.shape)
        or tuple(eligible.shape) != (len(selected),)
        or eligible.dtype != torch.bool
        or bool(
            wrong_actions[:, 2].eq(factual.action_sequence[:, 2]).any()
        )
    ):
        raise TemporalEvaluationContractError(
            "temporal control tensor schema changed"
        )
    return TemporalControlBatchV1(
        factual=factual,
        donor_indices=selected_donors,
        wrong_history_context_rgb=wrong_history_context,
        wrong_history_action_sequence=wrong_history_actions,
        wrong_action_sequence=wrong_actions,
        wrong_action_eligible=eligible,
    )


def _metadata_rows(
    rows: Sequence[h6.H6V2Row],
) -> tuple[metrics.MetadataRow, ...]:
    """Adapt registered H6 rows to the pure metric module without RGB access."""

    return tuple(
        metrics.MetadataRow(
            index=row.index,
            role=row.role,
            family=row.family,
            scene_id=row.scene_id,
            rgb=tuple(row.rgb),
            actions=tuple(row.actions),
        )
        for row in rows
    )


def _indices_sha256(indices: Sequence[int]) -> str:
    return metrics.canonical_json_sha256(tuple(int(value) for value in indices))


class TemporalH6RuntimeV1:
    """Exact one-pass train schedule and frozen validation-control runtime."""

    def __init__(
        self,
        train_rows: Sequence[h6.H6V2Row],
        validation_rows: Sequence[h6.H6V2Row],
        *,
        train_schedule_indices: Sequence[int],
        sentinel_indices: Sequence[int],
        donor_indices: Sequence[int],
        loader: SafeTemporalH6RGBLoaderV1,
        device: Any,
        panel_identity: Mapping[str, Any],
    ) -> None:
        self._train = tuple(train_rows)
        self._validation = tuple(validation_rows)
        self.train_schedule_indices = tuple(
            int(value) for value in train_schedule_indices
        )
        self.sentinel_indices = tuple(int(value) for value in sentinel_indices)
        self.donor_indices = tuple(int(value) for value in donor_indices)
        if (
            len(self._train) != TRAIN_ROW_COUNT_V1
            or len(self._validation) != VALIDATION_ROW_COUNT_V1
            or any(
                row.role != "train" or row.index != index
                for index, row in enumerate(self._train)
            )
            or any(
                row.role != "val" or row.index != index
                for index, row in enumerate(self._validation)
            )
            or len(self.train_schedule_indices) != TRAIN_SCHEDULE_ROW_COUNT_V1
            or len(set(self.train_schedule_indices))
            != TRAIN_SCHEDULE_ROW_COUNT_V1
            or any(
                not 0 <= index < len(self._train)
                for index in self.train_schedule_indices
            )
            or len(self.sentinel_indices) != 256
            or len(set(self.sentinel_indices)) != 256
            or any(
                not 0 <= index < len(self._validation)
                for index in self.sentinel_indices
            )
            or len(self.donor_indices) != VALIDATION_ROW_COUNT_V1
            or any(
                not 0 <= donor < len(self._validation)
                or self._validation[donor].family
                != self._validation[index].family
                or self._validation[donor].scene_id
                == self._validation[index].scene_id
                for index, donor in enumerate(self.donor_indices)
            )
        ):
            raise TemporalEvaluationContractError(
                "exact ordered temporal H6 roles or panels changed"
            )
        train_scenes = {row.scene_id for row in self._train}
        validation_scenes = {row.scene_id for row in self._validation}
        train_rgb = {leaf for row in self._train for leaf in row.rgb}
        validation_rgb = {
            leaf for row in self._validation for leaf in row.rgb
        }
        if train_scenes & validation_scenes or train_rgb & validation_rgb:
            raise TemporalEvaluationContractError(
                "temporal train and validation roles overlap"
            )
        self.loader = loader
        self.device = torch.device(device)
        self.panel_identity = dict(panel_identity)

    @property
    def train_rows(self) -> tuple[h6.H6V2Row, ...]:
        return self._train

    @property
    def validation_rows(self) -> tuple[h6.H6V2Row, ...]:
        return self._validation

    @property
    def val_rows(self) -> tuple[h6.H6V2Row, ...]:
        return self._validation

    @property
    def training_schedule(self) -> tuple[int, ...]:
        return self.train_schedule_indices

    def train_microbatches_for_update(
        self,
        update: int,
    ) -> tuple[TemporalSequenceBatchV1, ...]:
        if type(update) is not int or not 1 <= update <= MAXIMUM_UPDATES_V1:
            raise TemporalEvaluationContractError(
                "temporal update must be in the closed range [1,400]"
            )
        start = (update - 1) * TRAIN_SEQUENCES_PER_UPDATE_V1
        selected_indices = self.train_schedule_indices[
            start : start + TRAIN_SEQUENCES_PER_UPDATE_V1
        ]
        if len(selected_indices) != TRAIN_SEQUENCES_PER_UPDATE_V1:
            raise TemporalEvaluationContractError(
                "temporal update lost a scheduled row"
            )
        selected = tuple(self._train[index] for index in selected_indices)
        result = tuple(
            build_sequence_batch_v1(
                selected[offset : offset + TRAIN_MICROBATCH_SIZE_V1],
                loader=self.loader,
                device=self.device,
            )
            for offset in range(
                0,
                len(selected),
                TRAIN_MICROBATCH_SIZE_V1,
            )
        )
        if (
            len(result) != TRAIN_MICROBATCHES_PER_UPDATE_V1
            or tuple(
                row_index
                for batch in result
                for row_index in batch.row_indices
            )
            != selected_indices
        ):
            raise TemporalEvaluationContractError(
                "temporal update microbatch order changed"
            )
        return result

    def load_training_microbatches(
        self,
        schedule_slice: Sequence[int],
        device: Any,
    ) -> tuple[TemporalSequenceBatchV1, ...]:
        """Load one exact contiguous ten-row update slice as five B=2 batches."""

        selected_indices = tuple(int(value) for value in schedule_slice)
        requested_device = torch.device(device)
        if requested_device != self.device:
            raise TemporalEvaluationContractError(
                "runtime/training device mismatch"
            )
        if len(selected_indices) != TRAIN_SEQUENCES_PER_UPDATE_V1:
            raise TemporalEvaluationContractError(
                "training schedule slice must contain exactly ten rows"
            )
        starts = tuple(
            offset
            for offset in range(
                0,
                TRAIN_SCHEDULE_ROW_COUNT_V1,
                TRAIN_SEQUENCES_PER_UPDATE_V1,
            )
            if self.train_schedule_indices[
                offset : offset + TRAIN_SEQUENCES_PER_UPDATE_V1
            ]
            == selected_indices
        )
        if len(starts) != 1:
            raise TemporalEvaluationContractError(
                "training schedule slice is not one registered update"
            )
        return self.train_microbatches_for_update(
            starts[0] // TRAIN_SEQUENCES_PER_UPDATE_V1 + 1
        )

    def validation_batch(
        self,
        indices: Sequence[int],
    ) -> TemporalSequenceBatchV1:
        selected_indices = tuple(int(value) for value in indices)
        if (
            not selected_indices
            or len(set(selected_indices)) != len(selected_indices)
            or any(
                not 0 <= index < len(self._validation)
                for index in selected_indices
            )
        ):
            raise TemporalEvaluationContractError(
                "validation batch indices are invalid"
            )
        return build_sequence_batch_v1(
            tuple(self._validation[index] for index in selected_indices),
            loader=self.loader,
            device=self.device,
        )

    def validation_control_batch(
        self,
        indices: Sequence[int],
    ) -> TemporalControlBatchV1:
        selected_indices = tuple(int(value) for value in indices)
        if (
            not selected_indices
            or len(set(selected_indices)) != len(selected_indices)
            or any(
                not 0 <= index < len(self._validation)
                for index in selected_indices
            )
        ):
            raise TemporalEvaluationContractError(
                "validation control indices are invalid"
            )
        return build_control_batch_v1(
            tuple(self._validation[index] for index in selected_indices),
            validation_rows=self._validation,
            donor_indices=tuple(
                self.donor_indices[index] for index in selected_indices
            ),
            loader=self.loader,
            device=self.device,
        )

    def access_snapshot(self) -> dict[str, int]:
        return self.loader.access_snapshot()

    def access_audit(self) -> dict[str, Any]:
        receipt = self.access_snapshot()
        request_count = int(receipt["rgb_tensor_request_count"])
        attempt_count = int(receipt["rgb_open_attempt_count"])
        success_count = int(receipt["rgb_open_success_count"])
        decode_count = int(receipt["rgb_decode_success_count"])
        denied_count = int(receipt["denied_rgb_position_request_count"])
        exact_open_pipeline = (
            request_count == attempt_count == success_count == decode_count
        )
        forbidden_count = forbidden_rgb_open_count_v1(receipt)
        return {
            "schema": f"{SCHEMA_PREFIX_V1}_rgb_access_audit_v1",
            "counters": receipt,
            "forbidden_rgb_open_count": forbidden_count,
            "denied_rgb_position_request_count": denied_count,
            "exact_request_attempt_success_decode_counts": exact_open_pipeline,
            "passed": (
                forbidden_count == 0
                and denied_count == 0
                and exact_open_pipeline
            ),
        }

    def close(self) -> None:
        self.loader.close()


def open_bound_runtime_v1(
    runtime_data_root: Path,
    *,
    device: Any,
) -> tuple[TemporalH6RuntimeV1, dict[str, Any]]:
    """Open exact H6 metadata roles, then construct the narrow RGB runtime."""

    root = Path(runtime_data_root)
    if not root.is_absolute():
        raise TemporalEvaluationContractError(
            "runtime data root must be absolute"
        )
    train_rows, train_audit = h6.load_bound_index(root, role="train")
    validation_rows, validation_audit = h6.load_bound_index(root, role="val")
    train_metadata = _metadata_rows(train_rows)
    validation_metadata = _metadata_rows(validation_rows)
    train_schedule = tuple(metrics.build_training_schedule(train_metadata))
    sentinel_indices = tuple(metrics.build_sentinel_indices(validation_metadata))
    donor_indices = tuple(
        metrics.build_wrong_history_donor_indices(validation_metadata)
    )
    panel_identity = {
        "training_schedule_indices_sha256": _indices_sha256(train_schedule),
        "sentinel_indices_sha256": _indices_sha256(sentinel_indices),
        "wrong_history_donor_indices_sha256": _indices_sha256(donor_indices),
        "sentinel_wrong_history_donor_indices_sha256": (
            _indices_sha256(
                tuple(donor_indices[index] for index in sentinel_indices)
            )
        ),
    }
    expected_panel_identity = {
        "training_schedule_indices_sha256": (
            metrics.TRAIN_SCHEDULE_SHA256
        ),
        "sentinel_indices_sha256": metrics.SENTINEL_INDICES_SHA256,
        "wrong_history_donor_indices_sha256": (
            metrics.FULL_WRONG_HISTORY_DONORS_SHA256
        ),
        "sentinel_wrong_history_donor_indices_sha256": (
            metrics.SENTINEL_WRONG_HISTORY_DONORS_SHA256
        ),
    }
    if panel_identity != expected_panel_identity:
        raise TemporalEvaluationContractError(
            "frozen temporal schedule or validation panel identity changed"
        )
    loader = SafeTemporalH6RGBLoaderV1(
        root,
        (*train_rows, *validation_rows),
    )
    try:
        runtime = TemporalH6RuntimeV1(
            train_rows,
            validation_rows,
            train_schedule_indices=train_schedule,
            sentinel_indices=sentinel_indices,
            donor_indices=donor_indices,
            loader=loader,
            device=device,
            panel_identity=panel_identity,
        )
    except BaseException:
        loader.close()
        raise
    audit = {
        "schema": f"{SCHEMA_PREFIX_V1}_runtime_preflight_v1",
        "status": "PASS_EXACT_TEMPORAL_RUNTIME_METADATA",
        "train": train_audit,
        "validation": validation_audit,
        "panels": panel_identity,
        "model_visible_rgb_positions": [0, 1, 2, 3],
        "model_visible_action_positions": [0, 1, 2],
        "forbidden_rgb_positions": [4, 5, 6],
        "rgb_open_count": 0,
        "checkpoint_open_count": 0,
        "gpu_tensor_allocation_count": 0,
    }
    return runtime, audit


def open_bound_runtime(
    repo_root: Path,
    *,
    device: Any,
) -> tuple[TemporalH6RuntimeV1, dict[str, Any]]:
    """Lifecycle-facing alias for the exact bound-root runtime constructor."""

    return open_bound_runtime_v1(repo_root, device=device)


def _require_prediction_tensor(
    value: Any,
    *,
    batch: int,
    name: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape)
        != (batch, TARGET_TOKEN_COUNT_V1, FEATURE_DIMENSION_V1)
        or value.dtype != torch.float32
        or not bool(torch.isfinite(value).all())
    ):
        raise TemporalModelContractError(
            f"{name} must be finite Bx64x192 float32"
        )
    return value


class TemporalPredictionFieldsV1(NamedTuple):
    prediction: torch.Tensor
    memory: torch.Tensor
    step_states: torch.Tensor


def _prediction_fields(
    value: Any,
    *,
    batch: int,
    expected_steps: int,
) -> TemporalPredictionFieldsV1:
    prediction = getattr(value, "raw_predicted_target_tokens", None)
    memory = getattr(value, "recurrent_memory", None)
    step_states = getattr(value, "recurrent_step_states", None)
    _require_prediction_tensor(
        prediction,
        batch=batch,
        name="temporal raw prediction",
    )
    if (
        not isinstance(memory, torch.Tensor)
        or tuple(memory.shape)
        != (batch, 256, FEATURE_DIMENSION_V1)
        or memory.dtype != torch.float32
        or not bool(torch.isfinite(memory).all())
    ):
        raise TemporalModelContractError(
            "temporal recurrent memory must be finite Bx256x192 float32"
        )
    if (
        not isinstance(step_states, torch.Tensor)
        or tuple(step_states.shape)
        != (batch, expected_steps, 256, FEATURE_DIMENSION_V1)
        or step_states.dtype != torch.float32
        or not bool(torch.isfinite(step_states).all())
        or not torch.equal(step_states[:, -1], memory)
    ):
        raise TemporalModelContractError(
            "temporal recurrent step states changed shape, type, or final state"
        )
    return TemporalPredictionFieldsV1(prediction, memory, step_states)


def _target_tokens(
    model: Any,
    rgb: torch.Tensor,
    target_indices: torch.Tensor,
) -> torch.Tensor:
    method = getattr(model, "encode_target", None)
    if not callable(method):
        raise TemporalModelContractError(
            "temporal model lacks encode_target(rgb,target_indices)"
        )
    value = method(rgb, target_indices)
    tokens = getattr(value, "raw_target_tokens", value)
    return _require_prediction_tensor(
        tokens,
        batch=rgb.shape[0],
        name="EMA target tokens",
    ).detach()


def _predict_future(
    model: Any,
    context_rgb: torch.Tensor,
    action_sequence: torch.Tensor,
    target_indices: torch.Tensor,
) -> TemporalPredictionFieldsV1:
    method = getattr(model, "predict_future", None)
    if not callable(method):
        raise TemporalModelContractError(
            "temporal model lacks predict_future(context,actions,indices)"
        )
    return _prediction_fields(
        method(context_rgb, action_sequence, target_indices),
        batch=context_rgb.shape[0],
        expected_steps=3,
    )


def _predict_current_only(
    model: Any,
    current_rgb: torch.Tensor,
    action: torch.Tensor,
    target_indices: torch.Tensor,
) -> TemporalPredictionFieldsV1:
    method = getattr(model, "predict_current_only", None)
    if not callable(method):
        raise TemporalModelContractError(
            "temporal model lacks predict_current_only(rgb,action,indices)"
        )
    return _prediction_fields(
        method(current_rgb, action, target_indices),
        batch=current_rgb.shape[0],
        expected_steps=1,
    )


class TemporalEnergyVectorsV1(NamedTuple):
    real: torch.Tensor
    persistence: torch.Tensor
    current_only_reset: torch.Tensor
    wrong_history: torch.Tensor
    wrong_action: torch.Tensor
    wrong_action_eligible: torch.Tensor


@dataclass(frozen=True, slots=True)
class TemporalEvaluationResultV1:
    """Scalar controls plus bounded health accumulations for one panel."""

    row_indices: tuple[int, ...]
    energies: TemporalEnergyVectorsV1
    controls: Mapping[str, Any]
    recurrent_health: Any
    prediction_health: Any
    target_health: Any
    recurrent_temporal_change: float
    causal_deltas: Mapping[str, Any]
    token_populations: Mapping[str, torch.Tensor]
    diagnostic_vectors: Mapping[str, torch.Tensor]
    source_row_indices: tuple[int, ...]
    access_receipt: Mapping[str, int]
    panel_identity: Mapping[str, Any]
    integrity: Mapping[str, Any]


def _energy(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    value = metrics.normalized_half_squared_energy(prediction, target)
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (prediction.shape[0],)
        or not bool(torch.isfinite(value).all())
        or bool((value < 0).any())
    ):
        raise TemporalEvaluationContractError(
            "temporal energy vector is invalid"
        )
    return value.detach().to(device="cpu", dtype=torch.float64)


def _row_delta(
    factual: torch.Tensor,
    counterfactual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        not isinstance(factual, torch.Tensor)
        or not isinstance(counterfactual, torch.Tensor)
        or factual.shape != counterfactual.shape
        or factual.ndim != 3
        or not factual.is_floating_point()
        or not counterfactual.is_floating_point()
        or not bool(torch.isfinite(factual).all())
        or not bool(torch.isfinite(counterfactual).all())
    ):
        raise TemporalEvaluationContractError(
            "causal-delta operands are invalid"
        )
    difference = (
        factual.detach().to(device="cpu", dtype=torch.float64)
        - counterfactual.detach().to(device="cpu", dtype=torch.float64)
    )
    return (
        difference.square().mean(dim=(1, 2)),
        difference.abs().mean(dim=(1, 2)),
    )


def _summarize_delta(
    squared: torch.Tensor,
    absolute: torch.Tensor,
    *,
    eligible: torch.Tensor | None = None,
) -> dict[str, Any]:
    selected_squared = squared
    selected_absolute = absolute
    if eligible is not None:
        mask = eligible.detach().to(device="cpu", dtype=torch.bool)
        if tuple(mask.shape) != tuple(squared.shape) or not bool(mask.any()):
            raise TemporalEvaluationContractError(
                "causal-delta eligibility is invalid"
            )
        selected_squared = squared[mask]
        selected_absolute = absolute[mask]
    if (
        selected_squared.numel() < 1
        or not bool(torch.isfinite(selected_squared).all())
        or not bool(torch.isfinite(selected_absolute).all())
    ):
        raise TemporalEvaluationContractError(
            "causal-delta summary is nonfinite or empty"
        )
    return {
        "row_count": int(selected_squared.numel()),
        "mean_squared_delta": float(selected_squared.mean()),
        "mean_absolute_delta": float(selected_absolute.mean()),
        "positive_row_count": int((selected_squared > 0.0).sum()),
    }


def _summarize_controls(
    energy_vectors: Mapping[str, torch.Tensor],
    eligible: torch.Tensor,
    rows: Sequence[h6.H6V2Row],
) -> dict[str, metrics.ControlSummary]:
    scene_ids = tuple(row.scene_id for row in rows)
    family_ids = tuple(row.family for row in rows)
    keep_values = tuple(bool(value) for value in eligible.tolist())
    return {
        name: metrics.summarize_control(
            energy_vectors["real"][
                eligible if name == "wrong_action" else slice(None)
            ],
            energy_vectors[name][
                eligible if name == "wrong_action" else slice(None)
            ],
            tuple(
                scene
                for scene, keep in zip(
                    scene_ids,
                    keep_values,
                    strict=True,
                )
                if name != "wrong_action" or keep
            ),
            tuple(
                family
                for family, keep in zip(
                    family_ids,
                    keep_values,
                    strict=True,
                )
                if name != "wrong_action" or keep
            ),
            control_name=name,
        )
        for name in (
            "persistence",
            "current_only_reset",
            "wrong_history",
            "wrong_action",
        )
    }


def stream_temporal_panel_v1(
    model: Any,
    runtime: TemporalH6RuntimeV1,
    indices: Sequence[int],
) -> TemporalEvaluationResultV1:
    """Evaluate all temporal controls while retaining only CPU tensors."""

    selected_indices = tuple(int(value) for value in indices)
    if (
        not selected_indices
        or len(set(selected_indices)) != len(selected_indices)
        or any(
            not 0 <= index < VALIDATION_ROW_COUNT_V1
            for index in selected_indices
        )
    ):
        raise TemporalEvaluationContractError(
            "temporal evaluation panel indices are invalid"
        )
    rows = tuple(runtime.validation_rows[index] for index in selected_indices)
    access_before = runtime.access_snapshot()
    energies: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "real",
            "persistence",
            "current_only_reset",
            "wrong_history",
            "wrong_action",
        )
    }
    eligibility: list[torch.Tensor] = []
    recurrent_chunks: list[torch.Tensor] = []
    prediction_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    penultimate_state_chunks: list[torch.Tensor] = []
    final_state_chunks: list[torch.Tensor] = []
    causal_delta_chunks: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "wrong_history_prediction_squared",
            "wrong_history_prediction_absolute",
            "wrong_history_state_squared",
            "wrong_history_state_absolute",
            "wrong_action_prediction_squared",
            "wrong_action_prediction_absolute",
            "wrong_action_state_squared",
            "wrong_action_state_absolute",
        )
    }
    module_states = {module: bool(module.training) for module in model.modules()}
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, len(selected_indices), VALIDATION_BATCH_SIZE_V1):
                batch_indices = selected_indices[
                    start : start + VALIDATION_BATCH_SIZE_V1
                ]
                controls = runtime.validation_control_batch(batch_indices)
                factual = controls.factual
                target_indices, _visible = metrics.batched_mask_indices(
                    "val",
                    factual.row_indices,
                    device=runtime.device,
                )
                real_fields = _predict_future(
                    model,
                    factual.context_rgb,
                    factual.action_sequence,
                    target_indices,
                )
                current_only_fields = _predict_current_only(
                    model,
                    factual.context_rgb[:, 2],
                    factual.action_sequence[:, 2],
                    target_indices,
                )
                wrong_history_fields = _predict_future(
                    model,
                    controls.wrong_history_context_rgb,
                    controls.wrong_history_action_sequence,
                    target_indices,
                )
                wrong_action_fields = _predict_future(
                    model,
                    factual.context_rgb,
                    controls.wrong_action_sequence,
                    target_indices,
                )
                real = real_fields.prediction
                memory = real_fields.memory
                current_only = current_only_fields.prediction
                wrong_history = wrong_history_fields.prediction
                wrong_action = wrong_action_fields.prediction
                future_target = _target_tokens(
                    model,
                    factual.target_rgb,
                    target_indices,
                )
                current_target = _target_tokens(
                    model,
                    factual.context_rgb[:, 2],
                    target_indices,
                )
                for name, prediction in (
                    ("real", real),
                    ("persistence", current_target),
                    ("current_only_reset", current_only),
                    ("wrong_history", wrong_history),
                    ("wrong_action", wrong_action),
                ):
                    energies[name].append(_energy(prediction, future_target))
                eligibility.append(
                    controls.wrong_action_eligible.detach().to(device="cpu")
                )
                recurrent_chunks.append(memory.detach().to(device="cpu"))
                prediction_chunks.append(real.detach().to(device="cpu"))
                target_chunks.append(future_target.detach().to(device="cpu"))
                penultimate_state_chunks.append(
                    real_fields.step_states[:, 1].detach().to(device="cpu")
                )
                final_state_chunks.append(
                    real_fields.step_states[:, 2].detach().to(device="cpu")
                )
                for prefix, counterfactual_prediction, counterfactual_state in (
                    (
                        "wrong_history",
                        wrong_history_fields.prediction,
                        wrong_history_fields.memory,
                    ),
                    (
                        "wrong_action",
                        wrong_action_fields.prediction,
                        wrong_action_fields.memory,
                    ),
                ):
                    prediction_squared, prediction_absolute = _row_delta(
                        real,
                        counterfactual_prediction,
                    )
                    state_squared, state_absolute = _row_delta(
                        memory,
                        counterfactual_state,
                    )
                    causal_delta_chunks[
                        f"{prefix}_prediction_squared"
                    ].append(prediction_squared)
                    causal_delta_chunks[
                        f"{prefix}_prediction_absolute"
                    ].append(prediction_absolute)
                    causal_delta_chunks[
                        f"{prefix}_state_squared"
                    ].append(state_squared)
                    causal_delta_chunks[
                        f"{prefix}_state_absolute"
                    ].append(state_absolute)
    finally:
        for module, training in module_states.items():
            module.training = training

    energy_vectors = {
        name: torch.cat(chunks)
        for name, chunks in energies.items()
    }
    eligible = torch.cat(eligibility)
    controls_summary = _summarize_controls(
        energy_vectors,
        eligible,
        rows,
    )
    recurrent = torch.cat(recurrent_chunks)
    prediction = torch.cat(prediction_chunks)
    target = torch.cat(target_chunks)
    penultimate_state = torch.cat(penultimate_state_chunks)
    final_state = torch.cat(final_state_chunks)
    delta_vectors = {
        name: torch.cat(chunks)
        for name, chunks in causal_delta_chunks.items()
    }
    recurrent_change_vector = (
        final_state.to(dtype=torch.float64)
        - penultimate_state.to(dtype=torch.float64)
    ).square().mean(dim=(1, 2))
    causal_deltas = {
        "wrong_history": {
            "prediction": _summarize_delta(
                delta_vectors["wrong_history_prediction_squared"],
                delta_vectors["wrong_history_prediction_absolute"],
            ),
            "recurrent_state": _summarize_delta(
                delta_vectors["wrong_history_state_squared"],
                delta_vectors["wrong_history_state_absolute"],
            ),
        },
        "wrong_action_non_hold": {
            "prediction": _summarize_delta(
                delta_vectors["wrong_action_prediction_squared"],
                delta_vectors["wrong_action_prediction_absolute"],
                eligible=eligible,
            ),
            "recurrent_state": _summarize_delta(
                delta_vectors["wrong_action_state_squared"],
                delta_vectors["wrong_action_state_absolute"],
                eligible=eligible,
            ),
        },
        "wrong_action_all_rows_diagnostic": {
            "prediction": _summarize_delta(
                delta_vectors["wrong_action_prediction_squared"],
                delta_vectors["wrong_action_prediction_absolute"],
            ),
            "recurrent_state": _summarize_delta(
                delta_vectors["wrong_action_state_squared"],
                delta_vectors["wrong_action_state_absolute"],
            ),
        },
    }
    access_after = runtime.access_snapshot()
    access = access_delta_v1(access_before, access_after)
    forbidden = forbidden_rgb_open_count_v1(access)
    target_module = getattr(model, "target_encoder", None)
    target_parameters = (
        tuple(target_module.parameters())
        if isinstance(target_module, torch.nn.Module)
        else ()
    )
    integrity = {
        "panel_row_count": len(selected_indices),
        "future_rgb_online_access_count": 0,
        "forbidden_rgb_open_count": forbidden,
        "model_visible_action_positions": [0, 1, 2],
        "target_inventory_nonempty": bool(target_parameters),
        "target_parameters_frozen": bool(target_parameters)
        and all(not parameter.requires_grad for parameter in target_parameters),
        "target_module_eval": isinstance(target_module, torch.nn.Module)
        and not target_module.training,
        "target_gradient_tensor_count": sum(
            parameter.grad is not None for parameter in target_parameters
        ),
        "finite_energies": all(
            bool(torch.isfinite(value).all())
            for value in energy_vectors.values()
        ),
        "passed": (
            forbidden == 0
            and bool(target_parameters)
            and all(
                not parameter.requires_grad
                for parameter in target_parameters
            )
            and not target_module.training
            and all(
                bool(torch.isfinite(value).all())
                for value in energy_vectors.values()
            )
            and all(parameter.grad is None for parameter in target_parameters)
        ),
    }
    return TemporalEvaluationResultV1(
        row_indices=selected_indices,
        energies=TemporalEnergyVectorsV1(
            real=energy_vectors["real"],
            persistence=energy_vectors["persistence"],
            current_only_reset=energy_vectors["current_only_reset"],
            wrong_history=energy_vectors["wrong_history"],
            wrong_action=energy_vectors["wrong_action"],
            wrong_action_eligible=eligible,
        ),
        controls=controls_summary,
        recurrent_health=metrics.representation_health(recurrent),
        prediction_health=metrics.representation_health(prediction),
        target_health=metrics.representation_health(target),
        recurrent_temporal_change=metrics.recurrent_temporal_change(
            penultimate_state,
            final_state,
        ),
        causal_deltas=causal_deltas,
        token_populations={
            "recurrent": recurrent,
            "prediction": prediction,
            "target": target,
        },
        diagnostic_vectors={
            **delta_vectors,
            "recurrent_temporal_change": recurrent_change_vector,
        },
        source_row_indices=selected_indices,
        access_receipt=access,
        panel_identity=dict(runtime.panel_identity),
        integrity=integrity,
    )


def slice_temporal_result_v1(
    result: TemporalEvaluationResultV1,
    runtime: TemporalH6RuntimeV1,
    indices: Sequence[int],
) -> TemporalEvaluationResultV1:
    """Re-aggregate an exact row subset without another RGB or model call."""

    selected_indices = tuple(int(value) for value in indices)
    if (
        not isinstance(result, TemporalEvaluationResultV1)
        or not selected_indices
        or len(set(selected_indices)) != len(selected_indices)
        or result.row_indices != result.source_row_indices
    ):
        raise TemporalEvaluationContractError(
            "only an unsliced temporal source result can be sliced"
        )
    source_positions = {
        row_index: position
        for position, row_index in enumerate(result.row_indices)
    }
    if any(index not in source_positions for index in selected_indices):
        raise TemporalEvaluationContractError(
            "temporal result slice escaped its source panel"
        )
    positions = torch.tensor(
        [source_positions[index] for index in selected_indices],
        dtype=torch.long,
        device="cpu",
    )
    energy_vectors = {
        "real": result.energies.real.index_select(0, positions),
        "persistence": result.energies.persistence.index_select(0, positions),
        "current_only_reset": result.energies.current_only_reset.index_select(
            0, positions
        ),
        "wrong_history": result.energies.wrong_history.index_select(
            0, positions
        ),
        "wrong_action": result.energies.wrong_action.index_select(0, positions),
    }
    eligible = result.energies.wrong_action_eligible.index_select(0, positions)
    rows = tuple(runtime.validation_rows[index] for index in selected_indices)
    controls = _summarize_controls(energy_vectors, eligible, rows)
    populations = {
        name: value.index_select(0, positions)
        for name, value in result.token_populations.items()
    }
    vectors = {
        name: value.index_select(0, positions)
        for name, value in result.diagnostic_vectors.items()
    }
    causal_deltas = {
        "wrong_history": {
            "prediction": _summarize_delta(
                vectors["wrong_history_prediction_squared"],
                vectors["wrong_history_prediction_absolute"],
            ),
            "recurrent_state": _summarize_delta(
                vectors["wrong_history_state_squared"],
                vectors["wrong_history_state_absolute"],
            ),
        },
        "wrong_action_non_hold": {
            "prediction": _summarize_delta(
                vectors["wrong_action_prediction_squared"],
                vectors["wrong_action_prediction_absolute"],
                eligible=eligible,
            ),
            "recurrent_state": _summarize_delta(
                vectors["wrong_action_state_squared"],
                vectors["wrong_action_state_absolute"],
                eligible=eligible,
            ),
        },
        "wrong_action_all_rows_diagnostic": {
            "prediction": _summarize_delta(
                vectors["wrong_action_prediction_squared"],
                vectors["wrong_action_prediction_absolute"],
            ),
            "recurrent_state": _summarize_delta(
                vectors["wrong_action_state_squared"],
                vectors["wrong_action_state_absolute"],
            ),
        },
    }
    recurrent_temporal_change = float(
        vectors["recurrent_temporal_change"].mean()
    )
    if (
        not math.isfinite(recurrent_temporal_change)
        or recurrent_temporal_change < 0.0
    ):
        raise TemporalEvaluationContractError(
            "sliced recurrent temporal change is invalid"
        )
    integrity = {
        **dict(result.integrity),
        "panel_row_count": len(selected_indices),
        "derived_from_source_panel": True,
        "source_panel_row_count": len(result.source_row_indices),
        "additional_rgb_open_count": 0,
        "additional_model_call_count": 0,
    }
    same_population = (
        len(selected_indices) == len(result.row_indices)
        and set(selected_indices) == set(result.row_indices)
    )
    return TemporalEvaluationResultV1(
        row_indices=selected_indices,
        energies=TemporalEnergyVectorsV1(
            real=energy_vectors["real"],
            persistence=energy_vectors["persistence"],
            current_only_reset=energy_vectors["current_only_reset"],
            wrong_history=energy_vectors["wrong_history"],
            wrong_action=energy_vectors["wrong_action"],
            wrong_action_eligible=eligible,
        ),
        controls=controls,
        recurrent_health=(
            result.recurrent_health
            if same_population
            else metrics.representation_health(populations["recurrent"])
        ),
        prediction_health=(
            result.prediction_health
            if same_population
            else metrics.representation_health(populations["prediction"])
        ),
        target_health=(
            result.target_health
            if same_population
            else metrics.representation_health(populations["target"])
        ),
        recurrent_temporal_change=(
            result.recurrent_temporal_change
            if same_population
            else recurrent_temporal_change
        ),
        causal_deltas=causal_deltas,
        token_populations=populations,
        diagnostic_vectors=vectors,
        source_row_indices=result.source_row_indices,
        access_receipt=result.access_receipt,
        panel_identity=dict(result.panel_identity),
        integrity=integrity,
    )


def _evaluation_access_exact(
    receipt: Mapping[str, int],
    *,
    row_count: int,
) -> bool:
    expected_total = 6 * row_count
    if (
        int(receipt.get("rgb_tensor_request_count", -1)) != expected_total
        or int(receipt.get("rgb_open_attempt_count", -1)) != expected_total
        or int(receipt.get("rgb_open_success_count", -1)) != expected_total
        or int(receipt.get("rgb_decode_success_count", -1)) != expected_total
        or int(receipt.get("denied_rgb_position_request_count", -1)) != 0
        or forbidden_rgb_open_count_v1(receipt) != 0
    ):
        return False
    for role in _ACCESS_ROLES:
        for access_kind in _ACCESS_KINDS:
            for position in range(7):
                expected = (
                    row_count
                    if role == "val"
                    and (
                        (
                            access_kind == "factual"
                            and position in _FACTUAL_POSITIONS
                        )
                        or (
                            access_kind == "donor"
                            and position in _DONOR_POSITIONS
                        )
                    )
                    else 0
                )
                for operation in _ACCESS_OPERATIONS:
                    if (
                        int(
                            receipt.get(
                                _position_counter_key(
                                    role,
                                    access_kind,
                                    position,
                                    operation,
                                ),
                                -1,
                            )
                        )
                        != expected
                    ):
                        return False
    return True


def _control_summary(value: Any) -> metrics.ControlSummary:
    if isinstance(value, metrics.ControlSummary):
        return value
    if not isinstance(value, Mapping):
        raise TemporalEvaluationContractError(
            "predecessor control summary is invalid"
        )
    try:
        return metrics.ControlSummary(**dict(value))
    except (TypeError, ValueError) as error:
        raise TemporalEvaluationContractError(
            "predecessor control summary fields changed"
        ) from error


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.detach().to(device="cpu").tolist()
    return value


def _checkpoint_payload_from_result(
    model: Any,
    runtime: TemporalH6RuntimeV1,
    update: int,
    device: Any,
    *,
    full: bool,
    baseline: Mapping[str, Any] | None = None,
    result: TemporalEvaluationResultV1,
) -> dict[str, Any]:
    if type(full) is not bool or type(update) is not int:
        raise TypeError("checkpoint update/full selector types changed")
    allowed_updates = (
        metrics.FULL_OBSERVATION_UPDATES
        if full
        else metrics.SENTINEL_OBSERVATION_UPDATES
    )
    if update not in allowed_updates:
        raise TemporalEvaluationContractError(
            "checkpoint requested an unregistered observation"
        )
    if torch.device(device) != runtime.device:
        raise TemporalEvaluationContractError(
            "runtime/evaluation device mismatch"
        )
    expected_indices = (
        tuple(range(VALIDATION_ROW_COUNT_V1))
        if full
        else runtime.sentinel_indices
    )
    selected_indices = result.row_indices
    if selected_indices != expected_indices:
        raise TemporalEvaluationContractError(
            "temporal result does not match the registered observation panel"
        )
    valid_source_identity = result.source_row_indices == selected_indices or (
        not full
        and update == 0
        and result.source_row_indices
        == tuple(range(VALIDATION_ROW_COUNT_V1))
    )
    if not valid_source_identity:
        raise TemporalEvaluationContractError(
            "temporal result source panel is invalid"
        )
    metadata_rows = _metadata_rows(runtime.validation_rows)
    selected_donors = tuple(
        runtime.donor_indices[index] for index in selected_indices
    )
    wrong_action_indices = tuple(
        index
        for index in selected_indices
        if runtime.validation_rows[index].actions[2] != HOLD_ACTION_INDEX_V1
    )
    panel_identity_sha256 = metrics.validation_panel_identity(
        metadata_rows,
        selected_indices,
        selected_donors,
        wrong_action_indices,
    )

    supplied = dict(baseline or {})
    model_state = model.state_dict()
    model_state_finite = bool(model_state) and all(
        isinstance(value, torch.Tensor)
        and (
            not value.is_floating_point()
            or bool(torch.isfinite(value).all())
        )
        for value in model_state.values()
    )
    executor_state_finite = (
        supplied.get("model_and_optimizer_state_finite") is True
    )
    predecessor = supplied.get("predecessor_controls")
    predecessor_controls = (
        None
        if predecessor is None
        else {
            str(name): _control_summary(value)
            for name, value in dict(predecessor).items()
        }
    )
    target_module = getattr(model, "target_encoder", None)
    target_parameters = (
        tuple(target_module.parameters())
        if isinstance(target_module, torch.nn.Module)
        else ()
    )
    ema_value = getattr(model, "ema_update_count", -1)
    if isinstance(ema_value, torch.Tensor) and ema_value.numel() == 1:
        ema_count = int(ema_value.detach().to(device="cpu"))
    elif type(ema_value) is int:
        ema_count = ema_value
    else:
        ema_count = -1
    access_exact = _evaluation_access_exact(
        result.access_receipt,
        row_count=len(result.source_row_indices),
    )
    accounting_exact = supplied.get(
        "training_accounting_exact",
        update == 0,
    )
    latest_receipt = supplied.get(
        "latest_training_receipt_pass",
        None,
    )
    baseline_noncollapsed = bool(
        supplied.get(
            "baseline_health_noncollapsed",
            result.target_health.effective_rank > 0.0
            and result.target_health.cross_sample_variance > 0.0,
        )
    )
    integrity = metrics.IntegrityFacts(
        access_and_accounting_exact=(
            access_exact and accounting_exact is True
        ),
        all_evaluated_finite=bool(result.integrity["finite_energies"])
        and model_state_finite
        and executor_state_finite
        and all(
            value.finite
            for value in (
                result.recurrent_health,
                result.prediction_health,
                result.target_health,
            )
        ),
        target_frozen_eval=bool(target_parameters)
        and all(not parameter.requires_grad for parameter in target_parameters)
        and isinstance(target_module, torch.nn.Module)
        and not target_module.training,
        target_gradient_tensor_count=sum(
            parameter.grad is not None for parameter in target_parameters
        ),
        ema_count=ema_count,
        latest_training_receipt_pass=(
            latest_receipt
            if latest_receipt in {None, True, False}
            else False
        ),
        baseline_health_noncollapsed=baseline_noncollapsed,
    )
    observation = metrics.TemporalObservation(
        update=update,
        panel_kind="full" if full else "sentinel",
        panel_identity_sha256=panel_identity_sha256,
        controls=result.controls,
        recurrent_health=result.recurrent_health,
        prediction_health=result.prediction_health,
        target_health=result.target_health,
        integrity=integrity,
        predecessor_controls=predecessor_controls,
        raw_health_retentions=supplied.get("raw_health_retentions"),
        place_chance_multiple_retention=supplied.get(
            "place_chance_multiple_retention"
        ),
        target_place_rank_retention=supplied.get(
            "target_place_rank_retention"
        ),
    )
    gate_checks = metrics.observation_survival_checks(observation)
    if full and update in (200, 400):
        gate_checks = metrics.qualification_checks(observation)
    return {
        "schema": f"{SCHEMA_PREFIX_V1}_checkpoint_evaluation_v1",
        "update": update,
        "panel_kind": observation.panel_kind,
        "panel_identity_sha256": panel_identity_sha256,
        "row_count": len(selected_indices),
        "controls": _jsonable(result.controls),
        "health": {
            "recurrent": _jsonable(result.recurrent_health),
            "prediction": _jsonable(result.prediction_health),
            "target": _jsonable(result.target_health),
        },
        "diagnostics": {
            "recurrent_temporal_change": result.recurrent_temporal_change,
            "causal_deltas": _jsonable(result.causal_deltas),
        },
        "integrity": _jsonable(integrity),
        "state_finiteness": {
            "model_state_finite": model_state_finite,
            "executor_model_and_optimizer_state_finite": executor_state_finite,
        },
        "access": dict(result.access_receipt),
        "access_provenance": {
            "source_panel_row_count": len(result.source_row_indices),
            "derived_from_source_panel": (
                result.source_row_indices != result.row_indices
            ),
            "additional_rgb_open_count": 0
            if result.source_row_indices != result.row_indices
            else None,
            "additional_model_call_count": 0
            if result.source_row_indices != result.row_indices
            else None,
        },
        "gate": {
            "checks": dict(gate_checks),
            "passed": all(gate_checks.values()),
        },
    }


def evaluate_checkpoint(
    model: Any,
    runtime: TemporalH6RuntimeV1,
    update: int,
    device: Any,
    *,
    full: bool,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one JSONable registered temporal observation.

    ``baseline`` is an executor-supplied bridge for the predecessor spatial,
    raw-health, place-retention, and latest-training-receipt facts.  Keeping
    those facts injectable lets this evaluator enforce temporal RGB custody
    without coupling it to checkpoint or place-panel I/O.
    """

    allowed_updates = (
        metrics.FULL_OBSERVATION_UPDATES
        if full
        else metrics.SENTINEL_OBSERVATION_UPDATES
    )
    if update not in allowed_updates:
        raise TemporalEvaluationContractError(
            "checkpoint requested an unregistered observation"
        )
    if torch.device(device) != runtime.device:
        raise TemporalEvaluationContractError(
            "runtime/evaluation device mismatch"
        )
    selected_indices = (
        tuple(range(VALIDATION_ROW_COUNT_V1))
        if full
        else runtime.sentinel_indices
    )
    result = stream_temporal_panel_v1(model, runtime, selected_indices)
    return _checkpoint_payload_from_result(
        model,
        runtime,
        update,
        device,
        full=full,
        baseline=baseline,
        result=result,
    )


def evaluate_update_zero_full_and_sentinel_v1(
    model: Any,
    runtime: TemporalH6RuntimeV1,
    device: Any,
    *,
    baseline: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate update zero once and derive its registered sentinel exactly."""

    if torch.device(device) != runtime.device:
        raise TemporalEvaluationContractError(
            "runtime/evaluation device mismatch"
        )
    full_result = stream_temporal_panel_v1(
        model,
        runtime,
        tuple(range(VALIDATION_ROW_COUNT_V1)),
    )
    sentinel_result = slice_temporal_result_v1(
        full_result,
        runtime,
        runtime.sentinel_indices,
    )
    return {
        "schema": f"{SCHEMA_PREFIX_V1}_update_zero_full_and_sentinel_v1",
        "update": 0,
        "single_temporal_rgb_and_model_pass": True,
        "full": _checkpoint_payload_from_result(
            model,
            runtime,
            0,
            device,
            full=True,
            baseline=baseline,
            result=full_result,
        ),
        "sentinel": _checkpoint_payload_from_result(
            model,
            runtime,
            0,
            device,
            full=False,
            baseline=baseline,
            result=sentinel_result,
        ),
    }


def evaluate_predecessor_retention_panel_v1(
    model: Any,
    repo_root: Path,
    temporal_update: int,
    device: Any,
) -> dict[str, Any]:
    """Run the frozen update-independent predecessor panel at temporal u0/u200/u400.

    The predecessor evaluator's ``update`` parameter is only an observation
    token; its panel is update-independent.  This adapter therefore calls that
    frozen evaluator with its accepted update-zero token, preserves that fact
    in the receipt, and binds the result to the requested temporal update
    without pretending the predecessor trained for 200 or 400 updates.
    """

    if (
        type(temporal_update) is not int
        or temporal_update not in metrics.FULL_OBSERVATION_UPDATES
    ):
        raise TemporalEvaluationContractError(
            "predecessor retention is registered only at temporal 0/200/400"
        )
    runtime, runtime_audit = spatial_evaluation.open_bound_runtime(
        Path(repo_root),
        device=device,
        include_place=True,
    )
    try:
        result = spatial_evaluation.evaluate_checkpoint(
            model,
            runtime,
            0,
            device,
        )
    finally:
        closer = getattr(runtime, "close", None)
        if callable(closer):
            closer()
        else:
            loader = getattr(runtime, "loader", None)
            loader_closer = getattr(loader, "close", None)
            if callable(loader_closer):
                loader_closer()
    if (
        not isinstance(result, Mapping)
        or result.get("update") != 0
        or not isinstance(runtime_audit, Mapping)
    ):
        raise TemporalEvaluationContractError(
            "frozen predecessor evaluator result identity changed"
        )
    return {
        "schema": f"{SCHEMA_PREFIX_V1}_predecessor_retention_panel_v1",
        "temporal_update": temporal_update,
        "underlying_spatial_evaluator_update": 0,
        "underlying_spatial_evaluator_schema": result.get("schema"),
        "runtime_audit": _jsonable(runtime_audit),
        "evaluation": _jsonable(result),
    }


def expected_training_access_v1(update: int) -> dict[str, int]:
    """Return exact logical training access counts through ``update``."""

    if type(update) is not int or not 0 <= update <= MAXIMUM_UPDATES_V1:
        raise TemporalEvaluationContractError(
            "training access update must be in [0,400]"
        )
    sequences = TRAIN_SEQUENCES_PER_UPDATE_V1 * update
    return {
        "updates": update,
        "sequence_presentations": sequences,
        "logical_rgb_frame_presentations": 4 * sequences,
        "online_encoder_frame_presentations": 3 * sequences,
        "ema_target_encoder_frame_presentations": sequences,
        "microbatch_graphs": TRAIN_MICROBATCHES_PER_UPDATE_V1 * update,
        "backward_calls": TRAIN_MICROBATCHES_PER_UPDATE_V1 * update,
        "global_gradient_clips": update,
        "optimizer_steps": update,
        "ema_steps": update,
        "forbidden_rgb_open_count": 0,
    }


__all__ = [
    "ACTION_COUNT_V1",
    "ACTION_SEQUENCE_KEY_V1",
    "CONTEXT_RGB_KEY_V1",
    "FEATURE_DIMENSION_V1",
    "HOLD_ACTION_INDEX_V1",
    "MAXIMUM_UPDATES_V1",
    "REQUIRED_MODEL_BATCH_KEYS_V1",
    "RGB_ROOT_RELATIVE_PATH_V1",
    "SafeTemporalH6RGBLoaderV1",
    "TARGET_RGB_KEY_V1",
    "TARGET_TOKEN_COUNT_V1",
    "TemporalControlBatchV1",
    "TemporalEnergyVectorsV1",
    "TemporalEvaluationContractError",
    "TemporalEvaluationResultV1",
    "TemporalH6RuntimeV1",
    "TemporalModelContractError",
    "TemporalPredictionFieldsV1",
    "TemporalSequenceBatchV1",
    "access_delta_v1",
    "build_control_batch_v1",
    "build_sequence_batch_v1",
    "evaluate_checkpoint",
    "evaluate_predecessor_retention_panel_v1",
    "evaluate_update_zero_full_and_sentinel_v1",
    "expected_training_access_v1",
    "forbidden_rgb_open_count_v1",
    "open_bound_runtime",
    "open_bound_runtime_v1",
    "slice_temporal_result_v1",
    "stream_temporal_panel_v1",
]
