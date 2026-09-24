#!/usr/bin/env python3
"""Current-RGB-only runtime and observation evaluator for masked spatial JEPA."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Callable, Mapping, Sequence

import torch

from lewm.benchmarks import (
    go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1 as metrics,
)
from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as h6
from lewm.datasets import go2_memory_role_place_triplets_v1 as place_data


PLACE_INDEX_ROOT = Path(".generated/go2_memory_role_place_triplet_index_v1")
PLACE_MANIFEST_SHA256 = (
    "a5997d93838419cabaaf8e262db70ed51f6f928195f1a312cadc4768f74ca6ca"
)
PLACE_INDEX_SHA256 = (
    "a628a1047b6f15223a4fd7d30c5c87fa1914efef0955d70d9bd2f5330c77dcb0"
)
MICROBATCH_SIZE = 4
MICROBATCHES_PER_UPDATE = 4
VALIDATION_BATCH_SIZE = 16
PLACE_BATCH_SIZE = 16
RGB_ROOT_RELATIVE_PATH = Path(".generated/datagen_full/render_textured_v03")
MAXIMUM_RGB_FILE_BYTES = 4 * 1024 * 1024
VALIDATION_RGB_TENSOR_BYTES = 3 * 112 * 112 * 4
MAXIMUM_VALIDATION_CACHE_ENTRIES = metrics.VALIDATION_ROW_COUNT

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


class MaskedSpatialEvaluationError(RuntimeError):
    """The current-frame boundary or frozen evaluation contract changed."""


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
        raise MaskedSpatialEvaluationError(
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


class SafeCurrentRGBLoader:
    """No-follow loader restricted to registered H6 current-frame leaves."""

    def __init__(self, runtime_data_root: Path, rows: Sequence[h6.H6V2Row]) -> None:
        root = Path(runtime_data_root)
        if not root.is_absolute():
            raise MaskedSpatialEvaluationError(
                "runtime data root must be absolute"
            )
        registered: dict[tuple[str, int], h6.H6V2Row] = {}
        for row in rows:
            if not isinstance(row, h6.H6V2Row):
                raise TypeError("registered RGB rows must be H6V2Row values")
            key = (row.role, row.index)
            if key in registered:
                raise MaskedSpatialEvaluationError(
                    "registered RGB row identity repeats"
                )
            registered[key] = row
        if not registered:
            raise MaskedSpatialEvaluationError(
                "RGB loader requires registered rows"
            )
        try:
            descriptor = _open_absolute_directory(root / RGB_ROOT_RELATIVE_PATH)
        except OSError as error:
            raise MaskedSpatialEvaluationError("frozen RGB root open failed") from error
        self._root_fd: int | None = descriptor
        self._rows = registered
        self._validation_cache_allowlist = frozenset(
            row.current_rgb
            for row in registered.values()
            if row.role == "val"
        )
        self._validation_cache: dict[str, torch.Tensor] = {}
        if len(self._validation_cache_allowlist) > MAXIMUM_VALIDATION_CACHE_ENTRIES:
            self.close()
            raise MaskedSpatialEvaluationError(
                "validation RGB cache allowlist is too large"
            )
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

    def _require_open(self) -> int:
        if self._root_fd is None:
            raise MaskedSpatialEvaluationError("RGB loader is closed")
        return self._root_fd

    def _registered(self, row: h6.H6V2Row) -> h6.H6V2Row:
        if not isinstance(row, h6.H6V2Row):
            raise TypeError("RGB row must be H6V2Row")
        registered = self._rows.get((row.role, row.index))
        if registered is None or registered != row:
            raise MaskedSpatialEvaluationError(
                "RGB row is not the registered index row"
            )
        return registered

    def _read_current_leaf(self, row: h6.H6V2Row) -> bytes:
        registered = self._registered(row)
        leaf = registered.current_rgb
        try:
            canonical, _frame, _environment = h6._validate_rgb_leaf(
                leaf, scene_id=registered.scene_id
            )
        except h6.V27DataContractError as error:
            raise MaskedSpatialEvaluationError(
                "registered current RGB leaf is invalid"
            ) from error
        parts = PurePosixPath(canonical).parts
        if len(parts) != 3:
            raise MaskedSpatialEvaluationError(
                "registered current RGB leaf shape changed"
            )

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
                raise MaskedSpatialEvaluationError(
                    "current RGB leaf is not a regular file"
                )
            if not 0 < before.st_size <= MAXIMUM_RGB_FILE_BYTES:
                raise MaskedSpatialEvaluationError(
                    "current RGB leaf byte count is unsafe"
                )
            chunks: list[bytes] = []
            consumed = 0
            while True:
                chunk = os.read(image_fd, min(1024 * 1024, MAXIMUM_RGB_FILE_BYTES))
                if not chunk:
                    break
                consumed += len(chunk)
                if consumed > MAXIMUM_RGB_FILE_BYTES:
                    raise MaskedSpatialEvaluationError(
                        "current RGB leaf exceeded its byte cap"
                    )
                chunks.append(chunk)
            after = os.fstat(image_fd)
            raw = b"".join(chunks)
            if _fingerprint(before) != _fingerprint(after) or len(raw) != before.st_size:
                raise MaskedSpatialEvaluationError(
                    "current RGB leaf changed while being read"
                )
            self._access["rgb_open_success_count"] += 1
            self._access["rgb_byte_count"] += len(raw)
            return raw
        except MaskedSpatialEvaluationError:
            raise
        except OSError as error:
            raise MaskedSpatialEvaluationError(
                "no-follow current RGB leaf open failed"
            ) from error
        finally:
            if image_fd is not None:
                os.close(image_fd)
            os.close(descriptor)

    def load_current(self, row: h6.H6V2Row) -> torch.Tensor:
        registered = self._registered(row)
        leaf = registered.current_rgb
        self._access["rgb_tensor_request_count"] += 1
        cacheable = row.role == "val" and leaf in self._validation_cache_allowlist
        if row.role == "val" and not cacheable:
            raise MaskedSpatialEvaluationError(
                "validation current RGB left cache allowlist"
            )
        if cacheable and leaf in self._validation_cache:
            self._access["validation_cache_hit_count"] += 1
            return self._validation_cache[leaf]
        if cacheable:
            self._access["validation_cache_miss_count"] += 1
        raw = self._read_current_leaf(registered)
        try:
            tensor = h6.rectify_h6_rgb_bytes(raw)
        except h6.V27DataContractError as error:
            raise MaskedSpatialEvaluationError(
                "registered current RGB decode failed"
            ) from error
        self._access["rgb_decode_success_count"] += 1
        if cacheable:
            if len(self._validation_cache) >= MAXIMUM_VALIDATION_CACHE_ENTRIES:
                raise MaskedSpatialEvaluationError(
                    "validation current RGB cache capacity exceeded"
                )
            self._validation_cache[leaf] = tensor
            self._access["validation_cache_insert_count"] += 1
        return tensor

    def access_snapshot(self) -> dict[str, int]:
        return {
            **self._access,
            "validation_cache_entry_count": len(self._validation_cache),
            "validation_cache_bytes": (
                len(self._validation_cache) * VALIDATION_RGB_TENSOR_BYTES
            ),
        }

    def close(self) -> None:
        self._validation_cache.clear()
        if self._root_fd is not None:
            os.close(self._root_fd)
            self._root_fd = None


@dataclass(frozen=True, slots=True)
class CurrentFrameRow:
    index: int
    role: str
    family: str
    scene_id: str
    current_rgb: str


@dataclass(frozen=True, slots=True)
class CurrentFrameBatch:
    row_indices: tuple[int, ...]
    rows: tuple[CurrentFrameRow, ...]
    rgb: torch.Tensor


class CurrentFrameH6Runtime:
    """Expose only H6 ``current_rgb`` tensors and non-privileged metadata."""

    def __init__(
        self,
        train_rows: Sequence[h6.H6V2Row],
        validation_rows: Sequence[h6.H6V2Row],
        *,
        loader: SafeCurrentRGBLoader,
        device: Any,
        place_rows: Sequence[place_data.PlaceTripletRow] = (),
        load_place_triplet: Callable[
            [place_data.PlaceTripletRow], place_data.RGBTriplet
        ]
        | None = None,
    ) -> None:
        self._train = tuple(train_rows)
        self._validation = tuple(validation_rows)
        if (
            len(self._train) != metrics.TRAIN_ROW_COUNT
            or len(self._validation) != metrics.VALIDATION_ROW_COUNT
            or any(
                row.index != index or row.role != "train"
                for index, row in enumerate(self._train)
            )
            or any(
                row.index != index or row.role != "val"
                for index, row in enumerate(self._validation)
            )
        ):
            raise MaskedSpatialEvaluationError(
                "exact ordered corrected-H6 train/validation roles are required"
            )
        train_scenes = {row.scene_id for row in self._train}
        validation_scenes = {row.scene_id for row in self._validation}
        train_rgb = {row.current_rgb for row in self._train}
        validation_rgb = {row.current_rgb for row in self._validation}
        if train_scenes & validation_scenes or train_rgb & validation_rgb:
            raise MaskedSpatialEvaluationError(
                "train and validation current-frame roles overlap"
            )
        self.loader = loader
        self.device = torch.device(device)
        self.train_rows = tuple(self._public(row) for row in self._train)
        self.validation_rows = tuple(
            self._public(row) for row in self._validation
        )
        self.donor_indices = metrics.build_validation_donor_indices(
            self.validation_rows
        )
        self.place_rows = tuple(place_rows)
        self.load_place_triplet = load_place_triplet

    @staticmethod
    def _public(row: h6.H6V2Row) -> CurrentFrameRow:
        return CurrentFrameRow(
            index=row.index,
            role=row.role,
            family=row.family,
            scene_id=row.scene_id,
            current_rgb=row.current_rgb,
        )

    def _batch(self, rows: Sequence[h6.H6V2Row]) -> CurrentFrameBatch:
        ordered = tuple(rows)
        rgb = torch.stack(
            tuple(self.loader.load_current(row) for row in ordered)
        ).to(device=self.device, non_blocking=False)
        if (
            tuple(rgb.shape) != (len(ordered), 3, 112, 112)
            or rgb.dtype != torch.float32
            or not bool(torch.isfinite(rgb).all())
        ):
            raise MaskedSpatialEvaluationError("current RGB batch is invalid")
        public = tuple(self._public(row) for row in ordered)
        return CurrentFrameBatch(
            row_indices=tuple(row.index for row in ordered),
            rows=public,
            rgb=rgb,
        )

    def train_rows_for_update(self, update: int) -> tuple[CurrentFrameBatch, ...]:
        if type(update) is not int or not 1 <= update <= 1_000:
            raise MaskedSpatialEvaluationError("update must be in [1,1000]")
        first = 16 * (update - 1)
        batches = tuple(
            self._batch(
                self._train[
                    first + microbatch * MICROBATCH_SIZE :
                    first + (microbatch + 1) * MICROBATCH_SIZE
                ]
            )
            for microbatch in range(MICROBATCHES_PER_UPDATE)
        )
        if tuple(index for batch in batches for index in batch.row_indices) != tuple(
            range(first, first + 16)
        ):
            raise MaskedSpatialEvaluationError("training row order changed")
        return batches

    def validation_batch(self, indices: Sequence[int]) -> CurrentFrameBatch:
        ordered = tuple(indices)
        if (
            not ordered
            or len(set(ordered)) != len(ordered)
            or any(
                type(index) is not int
                or not 0 <= index < metrics.VALIDATION_ROW_COUNT
                for index in ordered
            )
        ):
            raise MaskedSpatialEvaluationError("validation indices are invalid")
        return self._batch(tuple(self._validation[index] for index in ordered))


def open_bound_runtime(
    repo_root: Path, *, device: Any, include_place: bool = True
) -> tuple[CurrentFrameH6Runtime, Mapping[str, Any]]:
    """Open only the exact preregistered H6 and optional place roles."""

    root = Path(repo_root).resolve(strict=True)
    train_rows, train_audit = h6.load_bound_index(root, role="train")
    validation_rows, validation_audit = h6.load_bound_index(root, role="val")
    loader = SafeCurrentRGBLoader(root, (*train_rows, *validation_rows))
    place_rows: tuple[place_data.PlaceTripletRow, ...] = ()
    place_audit: Mapping[str, Any] = {}
    place_loader = None
    if include_place:
        place_rows, place_audit = place_data.load_index(
            root,
            PLACE_INDEX_ROOT,
            role="checkpoint_selection",
            expected_manifest_sha256=PLACE_MANIFEST_SHA256,
        )
        if (
            place_audit.get("index_file_sha256") != PLACE_INDEX_SHA256
            or len(place_rows) != metrics.PLACE_SELECTION_ROW_COUNT
        ):
            loader.close()
            raise MaskedSpatialEvaluationError("place index binding changed")
        place_loader = lambda row: place_data.load_rgb_triplet(root, row)
    runtime = CurrentFrameH6Runtime(
        train_rows,
        validation_rows,
        loader=loader,
        device=device,
        place_rows=place_rows,
        load_place_triplet=place_loader,
    )
    return runtime, {
        "train": train_audit,
        "validation": validation_audit,
        "place": dict(place_audit),
        "future_rgb_tensor_count": 0,
        "action_tensor_count": 0,
    }


def _target_integrity(model: Any) -> dict[str, Any]:
    method = getattr(model, "target_modules", None)
    modules = tuple(method()) if callable(method) else (
        (model.target_encoder,) if hasattr(model, "target_encoder") else ()
    )
    checks = {
        "target_inventory_nonempty": bool(modules),
        "target_parameters_frozen": all(
            not parameter.requires_grad
            for module in modules
            for parameter in module.parameters()
        ),
        "target_gradient_tensor_count_zero": all(
            parameter.grad is None
            for module in modules
            for parameter in module.parameters()
        ),
        "target_modules_eval": all(not module.training for module in modules),
    }
    return {"checks": checks, "passed": all(checks.values())}


def _evaluate_place(model: Any, runtime: CurrentFrameH6Runtime) -> dict[str, Any]:
    if not runtime.place_rows or runtime.load_place_triplet is None:
        raise MaskedSpatialEvaluationError("place panel is not configured")
    key_chunks: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("online", "anchor", "positive", "negative")
    }
    for start in range(0, len(runtime.place_rows), PLACE_BATCH_SIZE):
        rows = runtime.place_rows[start : start + PLACE_BATCH_SIZE]
        triplets = tuple(runtime.load_place_triplet(row) for row in rows)
        anchor = torch.stack(tuple(item.anchor_rgb for item in triplets)).to(
            runtime.device
        )
        positive = torch.stack(tuple(item.positive_rgb for item in triplets)).to(
            runtime.device
        )
        negative = torch.stack(tuple(item.negative_rgb for item in triplets)).to(
            runtime.device
        )
        online = model.encode_online_full_frame(anchor)
        target = model.encode_target_full_frame(
            torch.cat((anchor, positive, negative), dim=0)
        )
        target_anchor, target_positive, target_negative = target.split(len(rows))
        for name, tokens in (
            ("online", online),
            ("anchor", target_anchor),
            ("positive", target_positive),
            ("negative", target_negative),
        ):
            key_chunks[name].append(
                metrics.flatten_spatial_keys(tokens)
            )
    return metrics.evaluate_place_keys(
        runtime.place_rows,
        *(torch.cat(key_chunks[name]) for name in key_chunks),
    )


def evaluate_checkpoint(
    model: Any,
    runtime: CurrentFrameH6Runtime,
    update: int,
    device: Any,
) -> dict[str, Any]:
    """Evaluate controls, raw health, and place without actions or future RGB."""

    if update not in metrics.OBSERVATION_UPDATES:
        raise MaskedSpatialEvaluationError("unregistered observation update")
    if torch.device(device) != runtime.device:
        raise MaskedSpatialEvaluationError("runtime/evaluation device mismatch")
    module_states = {module: bool(module.training) for module in model.modules()}
    access_before = runtime.loader.access_snapshot()
    online_health = metrics.RawHealthAccumulator()
    target_health = metrics.RawHealthAccumulator()
    position_sum = torch.zeros(
        metrics.TOKEN_COUNT,
        metrics.FEATURE_DIMENSION,
        dtype=torch.float64,
    )
    energies = {
        name: []
        for name in ("correct", "wrong_target", "wrong_context", "position_mean")
    }
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, metrics.VALIDATION_ROW_COUNT, VALIDATION_BATCH_SIZE):
                indices = tuple(
                    range(
                        start,
                        min(start + VALIDATION_BATCH_SIZE, metrics.VALIDATION_ROW_COUNT),
                    )
                )
                rgb = runtime.validation_batch(indices).rgb
                online = model.encode_online_full_frame(rgb)
                target = model.encode_target_full_frame(rgb)
                online_health.update(online)
                target_health.update(target)
                position_sum += metrics.normalize_tokens(target).sum(dim=0).to(
                    device="cpu", dtype=torch.float64
                )
            position_mean = (
                position_sum / metrics.VALIDATION_ROW_COUNT
            ).to(device=runtime.device, dtype=torch.float32)

            for start in range(0, metrics.VALIDATION_ROW_COUNT, VALIDATION_BATCH_SIZE):
                indices = tuple(
                    range(
                        start,
                        min(start + VALIDATION_BATCH_SIZE, metrics.VALIDATION_ROW_COUNT),
                    )
                )
                donor_indices = tuple(runtime.donor_indices[index] for index in indices)
                rgb = runtime.validation_batch(indices).rgb
                donor_rgb = runtime.validation_batch(donor_indices).rgb
                target_indices, _visible = metrics.batched_mask_indices(
                    "val", indices, device=runtime.device
                )
                prediction = model.forward_online(
                    rgb, target_indices
                ).normalized_predicted_target_tokens
                donor_prediction = model.forward_online(
                    donor_rgb, target_indices
                ).normalized_predicted_target_tokens
                target = metrics.normalize_tokens(
                    model.encode_target_full_frame(rgb)
                )
                donor_target = metrics.normalize_tokens(
                    model.encode_target_full_frame(donor_rgb)
                )
                gathered_target = metrics.gather_target_tokens(
                    target, target_indices
                )
                gathered_donor = metrics.gather_target_tokens(
                    donor_target, target_indices
                )
                gathered_mean = metrics.gather_target_tokens(
                    position_mean.expand(len(indices), -1, -1), target_indices
                )
                for name, predicted, expected in (
                    ("correct", prediction, gathered_target),
                    ("wrong_target", prediction, gathered_donor),
                    ("wrong_context", donor_prediction, gathered_target),
                    ("position_mean", prediction, gathered_mean),
                ):
                    energies[name].append(
                        metrics.half_squared_token_energy(predicted, expected)
                    )
            controls = {
                name: metrics.summarize_control(
                    torch.cat(energies["correct"]),
                    torch.cat(energies[name]),
                    [row.scene_id for row in runtime.validation_rows],
                    [row.family for row in runtime.validation_rows],
                    control_name=name,
                ).to_dict()
                for name in ("wrong_target", "wrong_context", "position_mean")
            }
            place = _evaluate_place(model, runtime)
            integrity = _target_integrity(model)
    finally:
        for module, training in module_states.items():
            module.training = training
    access_after = runtime.loader.access_snapshot()
    access = {
        key: int(access_after[key]) - int(access_before[key])
        for key in access_before
    }
    result = {
        "schema": f"{metrics.SCHEMA_PREFIX}_checkpoint_evaluation_v1",
        "update": update,
        "controls": controls,
        "raw_health": {
            "online": online_health.finalize().to_dict(),
            "target": target_health.finalize().to_dict(),
        },
        "place": place,
        "access": {
            "h6_current_rgb": access,
            "place_triplet_loader_call_count": len(runtime.place_rows),
            "place_rgb_tensor_count": 3 * len(runtime.place_rows),
            "future_rgb_tensor_count": 0,
            "action_tensor_count": 0,
            "model_input_fields": ["normalized_current_rgb", "target_indices"],
        },
        "integrity": integrity,
    }
    return result


__all__ = [
    "CurrentFrameBatch",
    "CurrentFrameH6Runtime",
    "CurrentFrameRow",
    "MaskedSpatialEvaluationError",
    "evaluate_checkpoint",
    "open_bound_runtime",
]
