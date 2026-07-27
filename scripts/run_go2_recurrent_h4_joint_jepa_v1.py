#!/usr/bin/env python3
"""Run one capped train/val recurrent-H4 joint JEPA falsification.

Importing this file is source-only.  In execution mode, Torch, Pillow, the
model module, schedule indexes, the accepted N320 initialization, and RGB bytes
are reached only after the fixed public census receipt has been checked and a
fresh mode-0700 attempt directory has been reserved.  ``--preflight-only``
validates bound non-RGB inputs without reserving or consuming that attempt.

The runner has no retry or resume surface and cannot accept an arbitrary
checkpoint.  Its sole tensor initialization is the exact accepted N320 encoder;
all recurrent and predictor components are fresh.  Training rows contain only
seven RGB observations and six registered primitive IDs from exact, hash-bound
train/validation JSONL indexes.  It never discovers corpus paths.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import stat
import sys
import time
from typing import Any, BinaryIO, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_MODULE = "lewm.models.go2_recurrent_h4_joint_jepa"
MODEL_SOURCE = ROOT / "lewm/models/go2_recurrent_h4_joint_jepa.py"
RGB_ROOT = ROOT / ".generated/datagen_full/render_textured_v03"
TRAIN_INDEX = (
    ROOT / ".generated/go2_recurrent_h4_rgb_sequence_index_v1/train.jsonl"
)
VAL_INDEX = ROOT / ".generated/go2_recurrent_h4_rgb_sequence_index_v1/val.jsonl"
TRAIN_INDEX_SHA256 = "f3f4dbe9ddd830427cc86bd27b0adb0b0fd0cebf64e937626088711748d9dd6b"
TRAIN_INDEX_BYTES = 10_024_000
VAL_INDEX_SHA256 = "86ab3130e5ba3468bd7f7f3e3cb1759d0e4a30d2326496e06845b4af7cb66880"
VAL_INDEX_BYTES = 1_278_976
MODEL_SOURCE_SHA256 = "ddd84561aba5a36df1255ab942bb29db943cc1bf7b0e496ae41b3d1cdc218f55"
MODEL_SOURCE_BYTES = 21_166
OUTPUT_ROOT = ROOT / ".generated/go2_recurrent_h4_joint_jepa_v1/probe_v1"
CENSUS_RECEIPT = (
    ROOT / ".generated/go2_recurrent_jepa_main_pool_census_v2/receipt.json"
)
CENSUS_RECEIPT_SHA256 = (
    "aac85f1016dca12e57e0cf612cd51a745becb2941adf361c0b4a752fe10a5408"
)
CENSUS_RECEIPT_BYTES = 54_695
CENSUS_SOURCE_BINDING_SHA256 = (
    "0d5ce1c8aae3777a3e1c930959d5985817d92c28ec240ad03ed79121869d4696"
)
N320_CHECKPOINT = (
    ROOT
    / ".generated/go2_observable_camera_ray_fit_v4/n320_compute_scaled_v1/checkpoint.pt"
)
N320_CHECKPOINT_SHA256 = (
    "ece874b53941e841fffc61b724a86d4383b881549afa453b746dd5d68aba11b0"
)
N320_CHECKPOINT_CONTENT_SHA256 = (
    "9dcca536943f89acfd7d463fdab591e19a030ef3dc8f3f19a050b1b10025fc2b"
)
N320_CHECKPOINT_BYTES = 13_777_100

SCHEMA = "lewm_go2_recurrent_h4_joint_jepa_v1"
INDEX_ROW_SCHEMA = "lewm_go2_recurrent_h4_rgb_sequence_index_v1"
ROLES = ("train", "val")
FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
PRIMITIVES = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
HOLD_ACTION = PRIMITIVES.index("hold")

UPDATES = 1_000
BATCH_SIZE = 16
PRESENTATIONS = UPDATES * BATCH_SIZE
VAL_PRESENTATIONS = 2_048
OBSERVATION_UPDATES = (0, 250, 500, 750, 1_000)
IMAGE_SIZE = 112
EMA_MOMENTUM = 0.996
VARIANCE_WEIGHT = 0.05
ACTION_RANKING_WEIGHT = 1.0
ACTION_RANKING_MARGIN = 0.05
OBJECTIVE_DESCRIPTION = (
    "prediction + 0.05*variance + 1.0*cyclic_wrong_action_margin_0.05"
)
PASS_DECISION = "PASS_MAIN_POOL_RECURRENT_H4_JOINT_JEPA_V1_PROBE"
STOP_DECISION = "STOP_MAIN_POOL_RECURRENT_H4_JOINT_JEPA_V1_PROBE"
ADDITIONAL_SCIENCE: dict[str, Any] = {}
EXECUTION_SOURCE_BINDINGS: dict[str, dict[str, Any]] = {}
AUXILIARY_TRAINING_CONTROL_MULTIPLIER = 0
MAX_GPU_SECONDS = 90 * 60
SEED = 20260727
BOOTSTRAP_REPLICATES = 1_000

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCENE_RE = re.compile(
    r"^(?:" + "|".join(map(re.escape, FAMILIES)) + r")_[0-9a-f]{12}$"
)
_FRAME_RE = re.compile(r"^frame_[0-9]{6}_env_[0-9]{2}\.png$")
_FORBIDDEN_COMPONENTS = {
    "test_id",
    "test_hard",
    "heldout",
    "held_out",
    "sealed",
    "protected",
    "g2",
    "g8",
    "production",
    "deployment",
}
_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_WRITE_FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class ContractError(RuntimeError):
    """A source, custody, runtime, or scientific invariant failed."""


@dataclass(frozen=True)
class IndexRow:
    role: str
    family: str
    scene_id: str
    rgb: tuple[str, ...]
    actions: tuple[int, ...]


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _self_bound(core: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(core)
    payload["content_sha256"] = hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
    return payload


def _strict_json_loads(raw: bytes) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    return json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )


def _open_absolute_directory(path: Path) -> int:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise ContractError(f"directory path is not canonical absolute: {path}")
    if not getattr(os, "O_NOFOLLOW", 0) or not getattr(os, "O_DIRECTORY", 0):
        raise ContractError("descriptor-relative no-follow directory opens are required")
    descriptor = os.open(path.anchor, _DIR_FLAGS)
    try:
        for component in path.parts[1:]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_bound(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> bytes:
    if not _SHA256_RE.fullmatch(expected_sha256) or expected_bytes < 0:
        raise ContractError("invalid expected input binding")
    if path.is_symlink():
        raise ContractError(f"symlink input forbidden: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ContractError(f"input is not regular: {path}")
    descriptor = os.open(path, _FILE_READ_FLAGS)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        opened = os.fstat(descriptor)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
        after_open = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    if not (
        _fingerprint(before)
        == _fingerprint(opened)
        == _fingerprint(after_open)
        == _fingerprint(after)
    ):
        raise ContractError(f"input changed while read: {path}")
    raw = b"".join(chunks)
    if len(raw) != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ContractError(f"input binding mismatch: {path}")
    return raw


def _write_exclusive(directory_fd: int, name: str, raw: bytes) -> dict[str, Any]:
    if not name or name in {".", ".."} or "/" in name:
        raise ContractError("non-canonical output leaf")
    descriptor = os.open(name, _FILE_WRITE_FLAGS, 0o600, dir_fd=directory_fd)
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise OSError("exclusive output write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(directory_fd)
    return {
        "path": name,
        "byte_count": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _publish_json(
    directory_fd: int,
    name: str,
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _self_bound(core)
    raw = _canonical_json_bytes(payload) + b"\n"
    binding = _write_exclusive(directory_fd, name, raw)
    binding["content_sha256"] = payload["content_sha256"]
    return payload, binding


def _reserve_output() -> int:
    generated_parent = OUTPUT_ROOT.parents[1]
    parent_fd = _open_absolute_directory(generated_parent)
    try:
        experiment_name = OUTPUT_ROOT.parent.name
        try:
            os.mkdir(experiment_name, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        experiment_fd = os.open(experiment_name, _DIR_FLAGS, dir_fd=parent_fd)
        try:
            os.mkdir(OUTPUT_ROOT.name, 0o700, dir_fd=experiment_fd)
            output_fd = os.open(OUTPUT_ROOT.name, _DIR_FLAGS, dir_fd=experiment_fd)
            os.fsync(experiment_fd)
            return output_fd
        finally:
            os.close(experiment_fd)
    finally:
        os.close(parent_fd)


def _lexical_generated_input(path: Path) -> Path:
    candidate = path if path.is_absolute() else ROOT / path
    candidate = Path(os.path.abspath(candidate))
    try:
        relative = candidate.relative_to(ROOT / ".generated")
    except ValueError as error:
        raise ContractError("index must be below the repository .generated root") from error
    lowered = {part.lower() for part in relative.parts}
    if lowered & _FORBIDDEN_COMPONENTS:
        raise ContractError("index path names a forbidden role or custody root")
    if candidate == CENSUS_RECEIPT or candidate == OUTPUT_ROOT:
        raise ContractError("index path aliases a control/output path")
    return candidate


def _validate_census_receipt() -> dict[str, Any]:
    raw = _read_regular_bound(
        CENSUS_RECEIPT,
        expected_sha256=CENSUS_RECEIPT_SHA256,
        expected_bytes=CENSUS_RECEIPT_BYTES,
    )
    value = _strict_json_loads(raw)
    if (
        not isinstance(value, dict)
        or value.get("schema") != "lewm_go2_recurrent_jepa_main_pool_census_v2"
        or value.get("decision") != "MAIN_POOL_H4_METADATA_FEASIBLE"
        or value.get("integrity") != {}
        or value.get("failed_predicates") != []
        or value.get("scope", {}).get("roles") != ["train", "val"]
        or value.get("scope", {}).get("rgb_open_count") != 0
        or value.get("scope", {}).get("label_open_count") != 0
        or value.get("scope", {}).get("test_or_heldout_open_count") != 0
        or value.get("scope", {}).get("sealed_open_count") != 0
        or value.get("identity", {}).get("cross_role_scene_identity_overlap_count") != 0
        or value.get("identity", {}).get("cross_role_manifest_identity_overlap_count") != 0
        or value.get("identity", {}).get("ordered_source_content_binding_sha256")
        != CENSUS_SOURCE_BINDING_SHA256
    ):
        raise ContractError("main-pool H4 census receipt is not the fixed passing receipt")
    return {
        "path": str(CENSUS_RECEIPT.relative_to(ROOT)),
        "file_sha256": CENSUS_RECEIPT_SHA256,
        "byte_count": CENSUS_RECEIPT_BYTES,
        "source_binding_sha256": CENSUS_SOURCE_BINDING_SHA256,
    }


def _validate_rgb_leaf(value: Any, scene_id: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ContractError("RGB leaf must be a nonempty POSIX-relative string")
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 3 or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ContractError("RGB leaf must be scene/rgb/frame.png")
    if path.parts[0] != scene_id or path.parts[1] != "rgb":
        raise ContractError("RGB leaf is not scoped to its declared scene")
    if not _FRAME_RE.fullmatch(path.parts[2]):
        raise ContractError("RGB leaf filename changed")
    if {part.lower() for part in path.parts} & _FORBIDDEN_COMPONENTS:
        raise ContractError("RGB leaf names a forbidden role or custody component")
    return path.as_posix()


def _decode_index(
    raw: bytes,
    *,
    role: str,
    expected_rows: int,
) -> tuple[list[IndexRow], dict[str, Any]]:
    if role not in ROLES or expected_rows <= 0:
        raise ContractError("invalid index role/count contract")
    rows: list[IndexRow] = []
    canonical_hashes: set[str] = set()
    ordered_hashes: list[str] = []
    family_counts: Counter[str] = Counter()
    scenes: set[str] = set()
    action_scene_support: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            raise ContractError(f"blank index line {line_number}")
        value = _strict_json_loads(raw_line)
        if not isinstance(value, dict) or set(value) != {
            "schema", "role", "family", "scene_id", "rgb", "actions"
        }:
            raise ContractError(f"index row schema changed at line {line_number}")
        if value["schema"] != INDEX_ROW_SCHEMA or value["role"] != role:
            raise ContractError(f"index role/schema mismatch at line {line_number}")
        family = value["family"]
        scene_id = value["scene_id"]
        if family not in FAMILIES:
            raise ContractError(f"unknown family at line {line_number}")
        if (
            not isinstance(scene_id, str)
            or not _SCENE_RE.fullmatch(scene_id)
            or not scene_id.startswith(f"{family}_")
        ):
            raise ContractError(f"invalid scene identity at line {line_number}")
        rgb_values = value["rgb"]
        actions = value["actions"]
        if not isinstance(rgb_values, list) or len(rgb_values) != 7:
            raise ContractError(f"row does not have seven RGB leaves at line {line_number}")
        if (
            not isinstance(actions, list)
            or len(actions) != 6
            or any(type(action) is not int or not 0 <= action < len(PRIMITIVES) for action in actions)
        ):
            raise ContractError(f"row does not have six valid action IDs at line {line_number}")
        rgb = tuple(_validate_rgb_leaf(item, scene_id) for item in rgb_values)
        canonical_hash = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
        if canonical_hash in canonical_hashes:
            raise ContractError(f"duplicate physical H6 row at line {line_number}")
        canonical_hashes.add(canonical_hash)
        ordered_hashes.append(canonical_hash)
        row = IndexRow(
            role=role,
            family=family,
            scene_id=scene_id,
            rgb=rgb,
            actions=tuple(actions),
        )
        rows.append(row)
        family_counts[family] += 1
        scenes.add(scene_id)
        for position in range(2, 6):
            action_scene_support[(family, position, row.actions[position])].add(scene_id)
    if len(rows) != expected_rows:
        raise ContractError(f"{role} index row count changed")
    if set(family_counts) != set(FAMILIES):
        raise ContractError(f"{role} index does not cover every family")
    expected_family_rows = expected_rows // len(FAMILIES)
    if expected_rows % len(FAMILIES) or any(
        family_counts[family] != expected_family_rows for family in FAMILIES
    ):
        raise ContractError(f"{role} index is not exactly family-balanced")
    minimum_scene_breadth = 8 if role == "train" else 1
    missing_cells = [
        (family, position, action, len(action_scene_support[(family, position, action)]))
        for family in FAMILIES
        for position in range(2, 6)
        for action in range(len(PRIMITIVES))
        if len(action_scene_support[(family, position, action)]) < minimum_scene_breadth
    ]
    if missing_cells:
        raise ContractError(f"{role} action-position scene breadth failed")
    return rows, {
        "row_count": len(rows),
        "scene_count": len(scenes),
        "family_rows": dict(sorted(family_counts.items())),
        "minimum_action_position_scene_breadth": min(
            len(value) for value in action_scene_support.values()
        ),
        "ordered_row_identity_sha256": hashlib.sha256(
            _canonical_json_bytes(ordered_hashes)
        ).hexdigest(),
    }


def _read_index(
    path: Path,
    *,
    sha256: str,
    byte_count: int,
    row_count: int,
    role: str,
) -> tuple[list[IndexRow], dict[str, Any]]:
    raw = _read_regular_bound(path, expected_sha256=sha256, expected_bytes=byte_count)
    rows, audit = _decode_index(raw, role=role, expected_rows=row_count)
    return rows, {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": sha256,
        "byte_count": byte_count,
        **audit,
    }


def _read_rgb(root_fd: int, leaf: str, access: Counter[str]) -> bytes:
    parts = PurePosixPath(leaf).parts
    if len(parts) != 3:
        raise ContractError("registered RGB leaf shape changed")
    descriptor = os.dup(root_fd)
    try:
        for component in parts[:-1]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        access["rgb_physical_open_attempt_count"] += 1
        image_fd = os.open(parts[-1], _FILE_READ_FLAGS, dir_fd=descriptor)
        access["rgb_physical_open_success_count"] += 1
        try:
            info = os.fstat(image_fd)
            if not stat.S_ISREG(info.st_mode):
                raise ContractError("RGB leaf is not a regular file")
            chunks: list[bytes] = []
            while True:
                chunk = os.read(image_fd, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
            access["rgb_physical_byte_count"] += len(raw)
            return raw
        finally:
            os.close(image_fd)
    finally:
        os.close(descriptor)


def _image_tensor(raw: bytes, runtime: Any) -> Any:
    Image = runtime.Image
    torch = runtime.torch
    with Image.open(io.BytesIO(raw)) as image:
        image = image.convert("RGB")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
        buffer = bytearray(image.tobytes())
    tensor = torch.frombuffer(buffer, dtype=torch.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE, 3)
    tensor = tensor.permute(2, 0, 1).contiguous().float().div_(255.0)
    for channel, (mean, std) in enumerate(
        zip((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), strict=True)
    ):
        tensor[channel].sub_(mean).div_(std)
    return tensor


def _load_batch(
    rows: Sequence[IndexRow],
    *,
    root_fd: int,
    runtime: Any,
    access: Counter[str],
    device: Any,
) -> tuple[Any, Any]:
    images = []
    for row in rows:
        images.append(
            runtime.torch.stack(
                [_image_tensor(_read_rgb(root_fd, leaf, access), runtime) for leaf in row.rgb]
            )
        )
    rgb = runtime.torch.stack(images).to(device, non_blocking=False)
    actions = runtime.torch.tensor(
        [row.actions for row in rows], dtype=runtime.torch.long, device=device
    )
    return rgb, actions


class _Runtime:
    def __init__(self) -> None:
        import torch
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.model_module = importlib.import_module(MODEL_MODULE)
        if Path(self.model_module.__file__).resolve() != MODEL_SOURCE:
            raise ContractError("imported model module is not the bound repository source")


def _tensor_manifest(runtime: _Runtime, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    torch = runtime.torch
    result = []
    for name, value in sorted(state.items()):
        if type(name) is not str or not isinstance(value, torch.Tensor):
            raise ContractError("N320 state entries must be named tensors")
        tensor = value.detach().to(device="cpu").contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        result.append({
            "name": name,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    if not result:
        raise ContractError("N320 state is empty")
    return result


def _load_n320_encoder(
    runtime: _Runtime,
    access: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    access["n320_initialization_checkpoint_open_attempt_count"] += 1
    raw = _read_regular_bound(
        N320_CHECKPOINT,
        expected_sha256=N320_CHECKPOINT_SHA256,
        expected_bytes=N320_CHECKPOINT_BYTES,
    )
    access["n320_initialization_checkpoint_open_success_count"] += 1
    checkpoint = runtime.torch.load(
        io.BytesIO(raw),
        map_location="cpu",
        weights_only=True,
    )
    fields = {
        "schema",
        "model_class",
        "state_manifest",
        "metadata",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "state_dict",
        "content_sha256",
    }
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != fields
        or checkpoint["schema"]
        != "lewm_go2_observable_camera_ray_fit_v4_development_checkpoint_v2"
        or checkpoint["model_class"] != "ObservableCameraRayEvidenceV4Model"
        or checkpoint["authoritative"] is not False
        or checkpoint["aggregation_eligible"] is not False
        or checkpoint["promotion_eligible"] is not False
        or type(checkpoint["state_dict"]) is not dict
    ):
        raise ContractError("accepted N320 checkpoint schema or scope changed")
    manifest = _tensor_manifest(runtime, checkpoint["state_dict"])
    semantic = {
        name: checkpoint[name]
        for name in (
            "schema",
            "model_class",
            "state_manifest",
            "metadata",
            "authoritative",
            "aggregation_eligible",
            "promotion_eligible",
        )
    }
    if (
        checkpoint["state_manifest"] != manifest
        or checkpoint["content_sha256"]
        != hashlib.sha256(_canonical_json_bytes(semantic)).hexdigest()
        or checkpoint["content_sha256"] != N320_CHECKPOINT_CONTENT_SHA256
    ):
        raise ContractError("accepted N320 tensor or semantic binding changed")
    prefix = "encoder."
    encoder = {
        name.removeprefix(prefix): value.detach().to(device="cpu").contiguous().clone()
        for name, value in checkpoint["state_dict"].items()
        if name.startswith(prefix)
    }
    if not encoder or any(name.startswith(prefix) for name in encoder):
        raise ContractError("accepted N320 encoder extraction failed")
    return encoder, {
        "path": str(N320_CHECKPOINT.relative_to(ROOT)),
        "file_sha256": N320_CHECKPOINT_SHA256,
        "content_sha256": N320_CHECKPOINT_CONTENT_SHA256,
        "byte_count": N320_CHECKPOINT_BYTES,
        "copied_state_prefix": prefix,
        "copied_tensor_count": len(encoder),
        "non_encoder_tensor_copy_count": 0,
    }


def _build_model(runtime: _Runtime, n320_encoder: Mapping[str, Any]) -> Any:
    module = runtime.model_module
    cls = getattr(module, "JointRecurrentH4JEPA", None)
    config_cls = getattr(module, "JointRecurrentH4JEPAConfig", None)
    if cls is None or config_cls is None:
        raise ContractError("model module lacks the reviewed JointRecurrentH4JEPA API")
    config = config_cls(
        image_size=IMAGE_SIZE,
        target_ema_momentum=EMA_MOMENTUM,
        variance_weight=VARIANCE_WEIGHT,
        action_vocabulary=PRIMITIVES,
    )
    model = cls(n320_encoder_state_dict=n320_encoder, config=config)
    for name in (
        "hard_sync_target",
        "update_target",
        "encode_target",
        "predict_from_belief",
    ):
        if not callable(getattr(model, name, None)):
            raise ContractError(f"model API is missing {name}()")
    if not hasattr(model, "target_encoder"):
        raise ContractError("model API is missing target_encoder")
    if tuple(getattr(model, "action_vocabulary", ())) != PRIMITIVES:
        raise ContractError("model primitive vocabulary changed")
    return model


def _model_forward(model: Any, history: Any, past: Any, future: Any) -> Any:
    output = model(
        history_rgb=history,
        past_actions=past,
        future_actions=future,
    )
    if not (
        isinstance(output, Mapping)
        or hasattr(output, "predicted_latents")
    ):
        raise ContractError("model forward must return the reviewed output dataclass")
    return output


def _extract_tensor(output: Any, *names: str) -> Any:
    for name in names:
        value = output.get(name) if isinstance(output, Mapping) else getattr(output, name, None)
        if value is not None:
            return value
    raise ContractError(f"model output lacks one of {names}")


def _target_encode(model: Any, future_rgb: Any) -> Any:
    value = model.encode_target(future_rgb)
    if isinstance(value, Mapping):
        value = _extract_tensor(value, "target_latents", "latents", "target")
    return value


def _normalized_error(predicted: Any, target: Any, runtime: _Runtime) -> Any:
    torch = runtime.torch
    if predicted.shape != target.shape or predicted.ndim < 3 or predicted.shape[1] != 4:
        raise ContractError("predicted/target latent shape contract changed")
    prediction = torch.nn.functional.normalize(predicted, dim=-1, eps=1e-6)
    destination = torch.nn.functional.normalize(target, dim=-1, eps=1e-6)
    return (prediction - destination).square().sum(dim=-1).mean(dim=-1)


def _token_distance(predicted: Any, target: Any, runtime: _Runtime) -> Any:
    torch = runtime.torch
    if predicted.shape != target.shape or predicted.ndim != 4 or predicted.shape[1] != 4:
        raise ContractError("action-ranking latent shape contract changed")
    prediction = torch.nn.functional.normalize(predicted, dim=-1, eps=1e-6)
    destination = torch.nn.functional.normalize(target, dim=-1, eps=1e-6)
    return (prediction - destination).square().sum(dim=-1).mean(dim=-1)


def _parameter_groups(model: Any) -> dict[str, list[Any]]:
    module_names = {
        "encoder": ("encoder",),
        "history": (
            "initial_belief",
            "history_observation_norm",
            "history_cell",
            "history_spatial_refiner",
        ),
        "predictor": (
            "action_embedding",
            "future_cell",
            "future_spatial_refiner",
            "prediction_projector",
        ),
    }
    groups: dict[str, list[Any]] = {}
    for group, names in module_names.items():
        modules = [getattr(model, name, None) for name in names]
        if any(module is None for module in modules):
            raise ContractError(f"model {group} module inventory changed")
        groups[group] = [
            parameter
            for module in modules
            for parameter in module.parameters()
        ]
    if any(not values for values in groups.values()):
        raise ContractError("every trainable parameter group must be nonempty")
    flattened = [parameter for values in groups.values() for parameter in values]
    ids = [id(parameter) for parameter in flattened]
    if len(set(ids)) != len(ids) or any(not parameter.requires_grad for parameter in flattened):
        raise ContractError("trainable parameter groups overlap or include frozen parameters")
    target_ids = {id(parameter) for parameter in model.target_encoder.parameters()}
    if target_ids & set(ids) or any(parameter.requires_grad for parameter in model.target_encoder.parameters()):
        raise ContractError("EMA target parameters entered the online optimizer")
    all_trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    if set(ids) != all_trainable:
        raise ContractError("trainable parameter groups do not cover the online model exactly")
    return groups


def _state_sha256(model: Any, runtime: _Runtime) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(_canonical_json_bytes(list(tensor.shape)) + b"\0")
        digest.update(
            tensor.reshape(-1).view(runtime.torch.uint8).numpy().tobytes(order="C")
        )
    return digest.hexdigest()


def _save_checkpoint(
    output_fd: int,
    *,
    model: Any,
    runtime: _Runtime,
    update: int,
    presentations: int,
) -> dict[str, Any]:
    if update not in OBSERVATION_UPDATES[1:]:
        raise ContractError("checkpoint update is not registered")
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    state_sha = _state_sha256(model, runtime)
    payload = {
        "schema": f"{SCHEMA}_checkpoint_v1",
        "update": update,
        "presentations": presentations,
        "state_sha256": state_sha,
        "model_state_dict": state,
    }
    buffer = io.BytesIO()
    runtime.torch.save(payload, buffer)
    raw = buffer.getvalue()
    name = f"checkpoint_update_{update}.pt"
    binding = _write_exclusive(output_fd, name, raw)
    binding.update({"state_sha256": state_sha, "update": update, "presentations": presentations})
    return binding


def _pool_features(value: Any, *, time_index: int) -> Any:
    selected = value[:, time_index]
    if selected.ndim == 2:
        return selected
    # Collapse detection must preserve spatial diversity; averaging every patch
    # makes a healthy geometry encoder look artificially low-rank.  Sixteen
    # fixed lattice samples per image keep the audit bounded and deterministic.
    selected = selected.flatten(start_dim=1, end_dim=-2)
    stride = max(1, selected.shape[1] // 16)
    return selected[:, ::stride, :].reshape(-1, selected.shape[-1])


def _effective_rank(features: Any, runtime: _Runtime) -> tuple[float, float]:
    torch = runtime.torch
    value = features.float()
    value = value - value.mean(dim=0, keepdim=True)
    std = value.var(dim=0, unbiased=False).add(1e-12).sqrt()
    near_zero_fraction = float((std < 0.02).float().mean().item())
    covariance = value.T @ value / max(1, value.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    total = eigenvalues.sum()
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return 0.0, near_zero_fraction
    probabilities = eigenvalues / total
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    ratio = float(entropy.exp().item() / max(1, value.shape[1]))
    return ratio, near_zero_fraction


def _bootstrap_lower(
    scene_values: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
) -> float:
    rng = random.Random(seed)
    draws: list[float] = []
    families = list(FAMILIES)
    for _ in range(BOOTSTRAP_REPLICATES):
        family_means = []
        for family in families:
            values = list(scene_values[family].values())
            if not values:
                raise ContractError("bootstrap family has no validation scenes")
            sample = [values[rng.randrange(len(values))] for _ in values]
            family_means.append(sum(sample) / len(sample))
        draws.append(sum(family_means) / len(family_means))
    draws.sort()
    return float(draws[int(0.025 * len(draws))])


def _evaluate(
    model: Any,
    rows: Sequence[IndexRow],
    *,
    root_fd: int,
    runtime: _Runtime,
    access: Counter[str],
    device: Any,
    update: int,
) -> dict[str, Any]:
    torch = runtime.torch
    model.eval()
    metric_names = (
        "real_normalized_error",
        "action_gap",
        "hold_gap",
        "persistence_gap",
        "history_gap",
    )
    sums: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {name: [0.0] * 4 for name in metric_names}
    )
    counts: Counter[tuple[str, str]] = Counter()
    target_features = []
    online_features = []
    with torch.no_grad():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start : start + BATCH_SIZE]
            rgb, actions = _load_batch(
                batch_rows,
                root_fd=root_fd,
                runtime=runtime,
                access=access,
                device=device,
            )
            history = rgb[:, :3]
            future_rgb = rgb[:, 3:]
            past = actions[:, :2]
            future = actions[:, 2:]
            output = _model_forward(model, history, past, future)
            predicted = _extract_tensor(output, "predicted_latents", "predicted")
            online = _extract_tensor(output, "online_latents", "history_latents")
            belief = _extract_tensor(output, "belief_latents", "belief")
            target = _target_encode(model, future_rgb)
            repeated_current = history[:, 2:3].expand(-1, 4, -1, -1, -1).contiguous()
            current_target = _target_encode(model, repeated_current)

            wrong_future = (future + 1) % len(PRIMITIVES)
            hold_future = torch.full_like(future, HOLD_ACTION)
            reversed_output = _model_forward(
                model,
                history[:, [1, 0, 2]],
                past[:, [1, 0]],
                future,
            )
            reset_output = _model_forward(
                model,
                history[:, 2:3].expand(-1, 3, -1, -1, -1).contiguous(),
                torch.full_like(past, HOLD_ACTION),
                future,
            )
            wrong_predicted = model.predict_from_belief(belief, wrong_future)
            hold_predicted = model.predict_from_belief(belief, hold_future)

            real_error = _normalized_error(predicted, target, runtime)
            wrong_error = _normalized_error(wrong_predicted, target, runtime)
            hold_error = _normalized_error(hold_predicted, target, runtime)
            reverse_error = _normalized_error(
                _extract_tensor(reversed_output, "predicted_latents", "predicted"),
                target,
                runtime,
            )
            reset_error = _normalized_error(
                _extract_tensor(reset_output, "predicted_latents", "predicted"),
                target,
                runtime,
            )
            persistence_error = _normalized_error(current_target, target, runtime)
            change = persistence_error.clamp_min(1e-4)
            values = {
                "real_normalized_error": real_error / change,
                "action_gap": (wrong_error - real_error) / change,
                "hold_gap": (hold_error - real_error) / change,
                "persistence_gap": (persistence_error - real_error) / change,
                "history_gap": (torch.minimum(reverse_error, reset_error) - real_error)
                / change,
            }
            for row_index, row in enumerate(batch_rows):
                key = (row.family, row.scene_id)
                counts[key] += 1
                for name in metric_names:
                    vector = values[name][row_index].detach().cpu().tolist()
                    for horizon in range(4):
                        sums[key][name][horizon] += float(vector[horizon])
            target_features.append(_pool_features(target, time_index=3).detach().cpu())
            online_features.append(_pool_features(online, time_index=2).detach().cpu())
            access["validation_sequence_presentation_count"] += len(batch_rows)

    scene_metrics: dict[tuple[str, str], dict[str, list[float]]] = {}
    for key, values in sums.items():
        scene_metrics[key] = {
            name: [item / counts[key] for item in vector]
            for name, vector in values.items()
        }
    aggregate: dict[str, list[float]] = {}
    family_metrics: dict[str, dict[str, list[float]]] = {
        family: {} for family in FAMILIES
    }
    for name in metric_names:
        family_vectors = []
        for family in FAMILIES:
            scene_vectors = [
                metrics[name]
                for (item_family, _scene), metrics in scene_metrics.items()
                if item_family == family
            ]
            if not scene_vectors:
                raise ContractError("validation macro lost a family")
            vector = [
                sum(item[horizon] for item in scene_vectors) / len(scene_vectors)
                for horizon in range(4)
            ]
            family_metrics[family][name] = vector
            family_vectors.append(vector)
        aggregate[name] = [
            sum(vector[horizon] for vector in family_vectors) / len(family_vectors)
            for horizon in range(4)
        ]

    lower_bounds: dict[str, float] = {}
    for offset, name in enumerate(("action_gap", "persistence_gap", "history_gap")):
        values_by_family = {
            family: {
                scene: metrics[name][3]
                for (item_family, scene), metrics in scene_metrics.items()
                if item_family == family
            }
            for family in FAMILIES
        }
        lower_bounds[f"{name}_h4"] = _bootstrap_lower(
            values_by_family,
            seed=SEED + update * 10 + offset,
        )
    target_rank, target_near_zero = _effective_rank(
        torch.cat(target_features, dim=0), runtime
    )
    online_rank, online_near_zero = _effective_rank(
        torch.cat(online_features, dim=0), runtime
    )
    finite_values = [
        value
        for vectors in aggregate.values()
        for value in vectors
    ] + list(lower_bounds.values()) + [
        target_rank,
        target_near_zero,
        online_rank,
        online_near_zero,
    ]
    result = {
        "update": update,
        "presentations": update * BATCH_SIZE,
        "validation_rows": len(rows),
        "aggregate": aggregate,
        "family": family_metrics,
        "bootstrap_lower_95": lower_bounds,
        "noncollapse": {
            "target_effective_rank_ratio": target_rank,
            "online_effective_rank_ratio": online_rank,
            "target_near_zero_variance_fraction": target_near_zero,
            "online_near_zero_variance_fraction": online_near_zero,
        },
        "all_registered_values_finite": all(math.isfinite(value) for value in finite_values),
    }
    model.train()
    return result


def _noncollapsed(observation: Mapping[str, Any]) -> bool:
    values = observation["noncollapse"]
    return bool(
        observation["all_registered_values_finite"]
        and values["target_effective_rank_ratio"] >= 0.10
        and values["online_effective_rank_ratio"] >= 0.10
        and values["target_near_zero_variance_fraction"] <= 0.05
        and values["online_near_zero_variance_fraction"] <= 0.05
    )


def _decision(observations: Sequence[Mapping[str, Any]], updates_completed: int) -> dict[str, Any]:
    baseline = next(item for item in observations if item["update"] == 0)
    candidates = [item for item in observations if item["update"] > 0 and _noncollapsed(item)]
    if candidates:
        selected = min(
            candidates,
            key=lambda item: sum(item["aggregate"]["real_normalized_error"]) / 4.0,
        )
    else:
        selected = None
    gates: dict[str, bool] = {
        "completed_exact_cap": updates_completed == UPDATES,
        "eligible_noncollapsed_checkpoint_exists": selected is not None,
    }
    if selected is not None:
        base_real = baseline["aggregate"]["real_normalized_error"]
        real = selected["aggregate"]["real_normalized_error"]
        action = selected["aggregate"]["action_gap"]
        hold = selected["aggregate"]["hold_gap"]
        persistence = selected["aggregate"]["persistence_gap"]
        history = selected["aggregate"]["history_gap"]
        h4_improvement = (base_real[3] - real[3]) / max(abs(base_real[3]), 1e-8)
        action_positive_families = sum(
            selected["family"][family]["action_gap"][3] > 0 for family in FAMILIES
        )
        persistence_positive_families = sum(
            selected["family"][family]["persistence_gap"][3] > 0
            for family in FAMILIES
        )
        history_positive_families = sum(
            selected["family"][family]["history_gap"][3] > 0 for family in FAMILIES
        )
        gates.update({
            "h4_real_error_improved_ten_percent": h4_improvement >= 0.10,
            "h1_h3_real_errors_all_improved": all(
                real[index] < base_real[index] for index in range(3)
            ),
            "h4_action_gap_at_least_point05": action[3] >= 0.05,
            "h4_action_gap_bootstrap_lower_positive": selected[
                "bootstrap_lower_95"
            ]["action_gap_h4"] > 0,
            "h4_hold_gap_positive": hold[3] > 0,
            "h1_h3_action_gaps_nonnegative": all(value >= 0 for value in action[:3]),
            "h4_history_gap_at_least_point03": history[3] >= 0.03,
            "h4_history_gap_bootstrap_lower_positive": selected[
                "bootstrap_lower_95"
            ]["history_gap_h4"] > 0,
            "h4_persistence_gap_at_least_point10": persistence[3] >= 0.10,
            "h4_persistence_gap_bootstrap_lower_positive": selected[
                "bootstrap_lower_95"
            ]["persistence_gap_h4"] > 0,
            "all_horizon_persistence_gaps_positive": all(value > 0 for value in persistence),
            "action_positive_in_six_families": action_positive_families >= 6,
            "history_positive_in_six_families": history_positive_families >= 6,
            "persistence_positive_in_six_families": persistence_positive_families >= 6,
            "no_family_action_gap_below_minus_point02": min(
                selected["family"][family]["action_gap"][3] for family in FAMILIES
            ) >= -0.02,
            "no_family_persistence_gap_below_minus_point02": min(
                selected["family"][family]["persistence_gap"][3]
                for family in FAMILIES
            ) >= -0.02,
        })
        diagnostics = {
            "selected_update": selected["update"],
            "selected_presentations": selected["presentations"],
            "h4_real_error_fractional_improvement": h4_improvement,
            "action_positive_family_count": action_positive_families,
            "history_positive_family_count": history_positive_families,
            "persistence_positive_family_count": persistence_positive_families,
        }
    else:
        diagnostics = {"selected_update": None, "selected_presentations": None}
    passed = all(gates.values())
    return {
        "decision": PASS_DECISION if passed else STOP_DECISION,
        "gates": gates,
        "failed_gates": sorted(name for name, passed_gate in gates.items() if not passed_gate),
        "diagnostics": diagnostics,
        "authority": (
            "A pass establishes bounded train/validation RGB JEPA substrate feasibility only; "
            "it does not authorize navigation, held-out access, promotion, or deployment."
        ),
    }


def _late_runtime(
    model_sha256: str,
    model_bytes: int,
    access: Counter[str],
) -> _Runtime:
    access["model_source_open_attempt_count"] += 1
    _read_regular_bound(
        MODEL_SOURCE,
        expected_sha256=model_sha256,
        expected_bytes=model_bytes,
    )
    access["model_source_open_success_count"] += 1
    runtime = _Runtime()
    access["model_source_post_import_recheck_attempt_count"] += 1
    _read_regular_bound(
        MODEL_SOURCE,
        expected_sha256=model_sha256,
        expected_bytes=model_bytes,
    )
    access["model_source_post_import_recheck_success_count"] += 1
    return runtime


def _load_index_contract(
    args: argparse.Namespace,
    *,
    access: Counter[str],
) -> tuple[list[IndexRow], list[IndexRow], dict[str, Any], dict[str, Any]]:
    access["train_index_open_attempt_count"] += 1
    train_rows, train_binding = _read_index(
        args.train_index,
        sha256=args.train_index_sha256,
        byte_count=args.train_index_bytes,
        row_count=PRESENTATIONS,
        role="train",
    )
    access["train_index_open_success_count"] += 1
    access["val_index_open_attempt_count"] += 1
    val_rows, val_binding = _read_index(
        args.val_index,
        sha256=args.val_index_sha256,
        byte_count=args.val_index_bytes,
        row_count=VAL_PRESENTATIONS,
        role="val",
    )
    access["val_index_open_success_count"] += 1
    train_scenes = {row.scene_id for row in train_rows}
    val_scenes = {row.scene_id for row in val_rows}
    if train_scenes & val_scenes:
        raise ContractError("train and validation indexes share a scene")
    train_rgb = {leaf for row in train_rows for leaf in row.rgb}
    val_rgb = {leaf for row in val_rows for leaf in row.rgb}
    if train_rgb & val_rgb:
        raise ContractError("train and validation indexes share an RGB leaf")
    return train_rows, val_rows, train_binding, val_binding


def _run(
    args: argparse.Namespace,
    *,
    output_fd: int,
    access: Counter[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_rows, val_rows, train_binding, val_binding = _load_index_contract(
        args,
        access=access,
    )

    runtime = _late_runtime(args.model_sha256, args.model_bytes, access)
    torch = runtime.torch
    if args.device != "cuda" or not torch.cuda.is_available():
        raise ContractError("the capped probe requires an available CUDA/ROCm torch device")
    device = torch.device("cuda")
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
    n320_encoder, n320_binding = _load_n320_encoder(runtime, access)
    model = _build_model(runtime, n320_encoder).to(device)
    del n320_encoder
    model.hard_sync_target()
    groups = _parameter_groups(model)
    optimizer = torch.optim.AdamW(
        [
            {"params": groups["encoder"], "lr": 1e-4, "name": "encoder"},
            {"params": groups["history"], "lr": 3e-4, "name": "history"},
            {"params": groups["predictor"], "lr": 3e-4, "name": "predictor"},
        ],
        weight_decay=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    rgb_root_fd = _open_absolute_directory(RGB_ROOT)
    observations: list[dict[str, Any]] = []
    checkpoints: dict[str, dict[str, Any]] = {}
    updates_completed = 0
    presentations_completed = 0
    training_loss_sums: defaultdict[str, float] = defaultdict(float, {
        "prediction": 0.0,
        "variance": 0.0,
        "wrong_action_ranking": 0.0,
        "total": 0.0,
    })
    last_training_losses: dict[str, float] | None = None
    torch.cuda.synchronize()
    gpu_started = time.monotonic()
    try:
        observation_zero = _evaluate(
            model,
            val_rows,
            root_fd=rgb_root_fd,
            runtime=runtime,
            access=access,
            device=device,
            update=0,
        )
        observations.append(observation_zero)
        for update in range(1, UPDATES + 1):
            start = (update - 1) * BATCH_SIZE
            batch_rows = train_rows[start : start + BATCH_SIZE]
            if len(batch_rows) != BATCH_SIZE:
                raise ContractError("train schedule exhausted before the exact cap")
            rgb, actions = _load_batch(
                batch_rows,
                root_fd=rgb_root_fd,
                runtime=runtime,
                access=access,
                device=device,
            )
            history = rgb[:, :3]
            future_rgb = rgb[:, 3:]
            past = actions[:, :2]
            future = actions[:, 2:]
            optimizer.zero_grad(set_to_none=True)
            output = _model_forward(model, history, past, future)
            predicted = _extract_tensor(output, "predicted_latents", "predicted")
            belief = _extract_tensor(output, "belief_latents", "belief")
            with torch.no_grad():
                target = _target_encode(model, future_rgb)
            prediction_loss = _normalized_error(predicted, target, runtime).mean()
            variance_loss = _extract_tensor(output, "variance_loss")
            wrong_future = (future + 1) % len(PRIMITIVES)
            wrong_predicted = model.predict_from_belief(belief, wrong_future)
            real_distance = _token_distance(predicted, target, runtime)
            wrong_distance = _token_distance(wrong_predicted, target, runtime)
            wrong_action_ranking_loss = torch.relu(
                ACTION_RANKING_MARGIN + real_distance - wrong_distance
            ).mean()
            auxiliary_losses: Mapping[str, Any] = {}
            auxiliary_method = getattr(model, "training_auxiliary_losses", None)
            if callable(auxiliary_method):
                auxiliary_losses = auxiliary_method(
                    history_rgb=history,
                    past_actions=past,
                    future_actions=future,
                    target_latents=target,
                    output=output,
                )
                if (
                    not isinstance(auxiliary_losses, Mapping)
                    or not auxiliary_losses
                    or any(
                        not isinstance(name, str)
                        or not name
                        or getattr(value, "ndim", None) != 0
                        or not bool(torch.isfinite(value))
                        for name, value in auxiliary_losses.items()
                    )
                ):
                    raise ContractError("model auxiliary training losses are invalid")
            loss = (
                prediction_loss
                + VARIANCE_WEIGHT * variance_loss
                + ACTION_RANKING_WEIGHT * wrong_action_ranking_loss
                + sum(auxiliary_losses.values(), start=prediction_loss.new_zeros(()))
            )
            if not torch.isfinite(loss):
                raise ContractError("non-finite joint JEPA objective")
            loss.backward()
            for values in groups.values():
                torch.nn.utils.clip_grad_norm_(values, max_norm=1.0)
            optimizer.step()
            model.update_target(EMA_MOMENTUM)
            updates_completed = update
            presentations_completed = update * BATCH_SIZE
            access["train_sequence_presentation_count"] = presentations_completed
            access["optimizer_update_count"] = updates_completed
            access["target_ema_update_count"] = updates_completed
            access["wrong_action_counterfactual_sequence_count"] += BATCH_SIZE
            access["auxiliary_training_control_sequence_count"] += (
                AUXILIARY_TRAINING_CONTROL_MULTIPLIER * BATCH_SIZE
            )
            last_training_losses = {
                "prediction": float(prediction_loss.detach().item()),
                "variance": float(variance_loss.detach().item()),
                "wrong_action_ranking": float(
                    wrong_action_ranking_loss.detach().item()
                ),
                **{
                    name: float(value.detach().item())
                    for name, value in auxiliary_losses.items()
                },
                "total": float(loss.detach().item()),
            }
            if not all(math.isfinite(value) for value in last_training_losses.values()):
                raise ContractError("non-finite detached joint JEPA loss receipt")
            for name, value in last_training_losses.items():
                training_loss_sums[name] += value

            if update in OBSERVATION_UPDATES[1:]:
                observation = _evaluate(
                    model,
                    val_rows,
                    root_fd=rgb_root_fd,
                    runtime=runtime,
                    access=access,
                    device=device,
                    update=update,
                )
                observations.append(observation)
                checkpoints[str(update)] = _save_checkpoint(
                    output_fd,
                    model=model,
                    runtime=runtime,
                    update=update,
                    presentations=presentations_completed,
                )
            torch.cuda.synchronize()
            if time.monotonic() - gpu_started > MAX_GPU_SECONDS:
                raise ContractError("90-minute active GPU cap exceeded")
    finally:
        os.close(rgb_root_fd)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - gpu_started
    if presentations_completed != PRESENTATIONS or updates_completed != UPDATES:
        raise ContractError("exact presentation/update cap was not completed")
    decision = _decision(observations, updates_completed)
    metrics = {
        "schema": f"{SCHEMA}_metrics_v1",
        "observations": observations,
        "training_losses": {
            "mean_over_completed_updates": {
                name: value / max(1, updates_completed)
                for name, value in training_loss_sums.items()
            },
            "last_completed_update": last_training_losses,
            "objective": OBJECTIVE_DESCRIPTION,
        },
        "selection_rule": (
            "minimum mean H1-H4 normalized real-action validation error among "
            "registered noncollapsed trained checkpoints"
        ),
    }
    artifact = {
        "schema": f"{SCHEMA}_artifact_v1",
        "checkpoints": checkpoints,
        "updates_completed": updates_completed,
        "presentations_completed": presentations_completed,
        "gpu_active_seconds": elapsed,
        "input_bindings": {
            "train": train_binding,
            "val": val_binding,
            "n320_encoder_initialization": n320_binding,
        },
        "execution_source_bindings": {
            name: dict(binding)
            for name, binding in sorted(EXECUTION_SOURCE_BINDINGS.items())
        },
        "fresh_recurrent_and_predictor_initialization": True,
        "n320_encoder_initialization_checkpoint_open_count": 1,
        "retry_or_resume_checkpoint_input_open_count": 0,
        "retry_or_resume": False,
    }
    return metrics, artifact, decision


def _terminal_failure(
    output_fd: int,
    *,
    error: BaseException,
    reservation_binding: Mapping[str, Any],
    access: Counter[str],
) -> None:
    failure, failure_binding = _publish_json(
        output_fd,
        "failure.json",
        {
            "schema": f"{SCHEMA}_failure_v1",
            "status": "TERMINAL_EXECUTION_FAILURE",
            "failure_class": type(error).__name__,
            "failure_message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
            "updates_completed": int(access["optimizer_update_count"]),
            "presentations_completed": int(access["train_sequence_presentation_count"]),
            "authority": "Failure grants no retry, resume, checkpoint, or downstream authority.",
        },
    )
    access_payload, access_binding = _publish_json(
        output_fd,
        "failure_access.json",
        {
            "schema": f"{SCHEMA}_access_v1",
            "counts_complete": False,
            "counts": dict(sorted(access.items())),
            "forbidden": {
                "test_or_heldout_open_count": 0,
                "sealed_open_count": 0,
                "label_open_count": 0,
                "retry_or_resume_checkpoint_input_open_count": 0,
                "arbitrary_initialization_checkpoint_open_count": 0,
                "retry_or_resume_count": 0,
            },
        },
    )
    _publish_json(
        output_fd,
        "completed.json",
        {
            "schema": f"{SCHEMA}_completion_v1",
            "status": "TERMINAL_FAILURE_COMPLETE",
            "reservation": dict(reservation_binding),
            "failure": failure_binding,
            "access": access_binding,
            "failure_content_sha256": failure["content_sha256"],
            "access_content_sha256": access_payload["content_sha256"],
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--train-index", type=Path, default=TRAIN_INDEX)
    parser.add_argument("--train-index-sha256", default=TRAIN_INDEX_SHA256)
    parser.add_argument("--train-index-bytes", type=int, default=TRAIN_INDEX_BYTES)
    parser.add_argument("--val-index", type=Path, default=VAL_INDEX)
    parser.add_argument("--val-index-sha256", default=VAL_INDEX_SHA256)
    parser.add_argument("--val-index-bytes", type=int, default=VAL_INDEX_BYTES)
    parser.add_argument("--model-sha256", default=MODEL_SOURCE_SHA256)
    parser.add_argument("--model-bytes", type=int, default=MODEL_SOURCE_BYTES)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    for name in ("train_index_sha256", "val_index_sha256", "model_sha256"):
        if not _SHA256_RE.fullmatch(getattr(args, name)):
            parser.error(f"--{name.replace('_', '-')} must be lowercase SHA-256")
    if args.train_index_bytes <= 0 or args.val_index_bytes <= 0 or args.model_bytes <= 0:
        parser.error("bound byte counts must be positive")
    try:
        args.train_index = _lexical_generated_input(args.train_index)
        args.val_index = _lexical_generated_input(args.val_index)
    except ContractError as error:
        parser.error(str(error))
    if args.train_index == args.val_index:
        parser.error("train and validation index paths must differ")
    exact_bindings = (
        args.train_index == TRAIN_INDEX
        and args.train_index_sha256 == TRAIN_INDEX_SHA256
        and args.train_index_bytes == TRAIN_INDEX_BYTES
        and args.val_index == VAL_INDEX
        and args.val_index_sha256 == VAL_INDEX_SHA256
        and args.val_index_bytes == VAL_INDEX_BYTES
        and args.model_sha256 == MODEL_SOURCE_SHA256
        and args.model_bytes == MODEL_SOURCE_BYTES
    )
    if not exact_bindings:
        parser.error("the reviewed model and index bindings cannot be overridden")
    return args


def _preflight(args: argparse.Namespace, census_binding: Mapping[str, Any]) -> int:
    access: Counter[str] = Counter()
    access["census_receipt_open_count"] = 1
    train_rows, val_rows, train_binding, val_binding = _load_index_contract(
        args,
        access=access,
    )
    runtime = _late_runtime(args.model_sha256, args.model_bytes, access)
    n320_encoder, n320_binding = _load_n320_encoder(runtime, access)
    model = _build_model(runtime, n320_encoder)
    model.hard_sync_target()
    groups = _parameter_groups(model)
    print(
        json.dumps(
            {
                "decision": "PREFLIGHT_PASS_NO_OUTPUT_RESERVED_NO_RGB_OPENED",
                "census": dict(census_binding),
                "train": train_binding,
                "val": val_binding,
                "n320_encoder_initialization": n320_binding,
                "row_counts": {"train": len(train_rows), "val": len(val_rows)},
                "trainable_parameter_counts": {
                    name: sum(parameter.numel() for parameter in parameters)
                    for name, parameters in groups.items()
                },
                "access": dict(sorted(access.items())),
                "rgb_open_count": 0,
                "output_reservation_count": 0,
                "training_update_count": 0,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    census_binding = _validate_census_receipt()
    if args.preflight_only:
        return _preflight(args, census_binding)
    output_fd = _reserve_output()
    access: Counter[str] = Counter()
    access["census_receipt_open_count"] = 1
    reservation, reservation_binding = _publish_json(
        output_fd,
        "reservation.json",
        {
            "schema": f"{SCHEMA}_reservation_v1",
            "status": "FRESH_ATTEMPT_RESERVED_BEFORE_INDEX_RGB_OR_TORCH",
            "output_root": str(OUTPUT_ROOT.relative_to(ROOT)),
            "census": census_binding,
            "inputs": {
                "train_index": {
                    "path": str(args.train_index.relative_to(ROOT)),
                    "file_sha256": args.train_index_sha256,
                    "byte_count": args.train_index_bytes,
                    "row_count": PRESENTATIONS,
                    "role": "train",
                },
                "val_index": {
                    "path": str(args.val_index.relative_to(ROOT)),
                    "file_sha256": args.val_index_sha256,
                    "byte_count": args.val_index_bytes,
                    "row_count": VAL_PRESENTATIONS,
                    "role": "val",
                },
                "model_source": {
                    "path": str(MODEL_SOURCE.relative_to(ROOT)),
                    "file_sha256": args.model_sha256,
                    "byte_count": args.model_bytes,
                },
                "execution_source_closure": {
                    name: dict(binding)
                    for name, binding in sorted(EXECUTION_SOURCE_BINDINGS.items())
                },
                "n320_encoder_initialization": {
                    "path": str(N320_CHECKPOINT.relative_to(ROOT)),
                    "file_sha256": N320_CHECKPOINT_SHA256,
                    "content_sha256": N320_CHECKPOINT_CONTENT_SHA256,
                    "byte_count": N320_CHECKPOINT_BYTES,
                    "accepted_prefix_only": "encoder.",
                },
                "rgb_root": str(RGB_ROOT.relative_to(ROOT)),
            },
            "cap": {
                "updates": UPDATES,
                "effective_batch_size": BATCH_SIZE,
                "sequence_presentations": PRESENTATIONS,
                "rgb_frame_views": PRESENTATIONS * 7,
                "gpu_active_seconds": MAX_GPU_SECONDS,
                "observation_updates": list(OBSERVATION_UPDATES),
            },
            "science": {
                "history_observations": 3,
                "past_actions": 2,
                "future_actions_and_targets": 4,
                "joint_online_components": ["encoder", "history", "predictor"],
                "target": "initial_hard_sync_then_ema_stop_gradient",
                "initialization": (
                    "accepted_N320_encoder_only; history_and_predictor_fresh"
                ),
                "wrong_action_contrast": {
                    "mapping": "cyclic_plus_one_modulo_nine",
                    "weight": ACTION_RANKING_WEIGHT,
                    "margin": ACTION_RANKING_MARGIN,
                },
                "additional_science": dict(ADDITIONAL_SCIENCE),
                "retry_resume_or_arbitrary_checkpoint_input": False,
            },
        },
    )
    try:
        metrics, artifact, result = _run(args, output_fd=output_fd, access=access)
        metrics_payload, metrics_binding = _publish_json(
            output_fd, "metrics.json", metrics
        )
        artifact_payload, artifact_binding = _publish_json(
            output_fd, "artifact.json", artifact
        )
        access_payload, access_binding = _publish_json(
            output_fd,
            "access.json",
            {
                "schema": f"{SCHEMA}_access_v1",
                "counts_complete": True,
                "counts": dict(sorted(access.items())),
                "forbidden": {
                    "test_or_heldout_open_count": 0,
                    "sealed_open_count": 0,
                    "label_open_count": 0,
                    "retry_or_resume_checkpoint_input_open_count": 0,
                    "arbitrary_initialization_checkpoint_open_count": 0,
                    "retry_or_resume_count": 0,
                },
            },
        )
        result_payload, result_binding = _publish_json(
            output_fd,
            "result.json",
            {"schema": f"{SCHEMA}_result_v1", **result},
        )
        _, completion_binding = _publish_json(
            output_fd,
            "completed.json",
            {
                "schema": f"{SCHEMA}_completion_v1",
                "status": "COMPLETE",
                "decision": result["decision"],
                "reservation": reservation_binding,
                "metrics": metrics_binding,
                "artifact": artifact_binding,
                "access": access_binding,
                "result": result_binding,
                "cross_bindings": {
                    "reservation_content_sha256": reservation["content_sha256"],
                    "metrics_content_sha256": metrics_payload["content_sha256"],
                    "artifact_content_sha256": artifact_payload["content_sha256"],
                    "access_content_sha256": access_payload["content_sha256"],
                    "result_content_sha256": result_payload["content_sha256"],
                },
            },
        )
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "selected_update": result["diagnostics"].get("selected_update"),
                    "failed_gate_count": len(result["failed_gates"]),
                    "completion_sha256": completion_binding["file_sha256"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if result["decision"].startswith("PASS_") else 2
    except BaseException as error:
        try:
            _terminal_failure(
                output_fd,
                error=error,
                reservation_binding=reservation_binding,
                access=access,
            )
        finally:
            os.close(output_fd)
        print(
            json.dumps(
                {
                    "decision": "TERMINAL_EXECUTION_FAILURE",
                    "failure_class": type(error).__name__,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 130 if isinstance(error, KeyboardInterrupt) else 3
    finally:
        try:
            os.close(output_fd)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
