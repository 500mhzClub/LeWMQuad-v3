"""Frozen H6 metadata and evaluation utilities for explicit-plan JEPA V27.

The metadata preflight in this module opens only the two exact, SHA-256-bound
corrected-H6 V2 JSONL files.  It never follows an RGB leaf.  RGB decoding is a
separate, explicit function so callers cannot accidentally turn the preflight
into image access.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import (
    BLOCK_SIZE,
    ENVS_PER_SOURCE,
    EXPECTED_SCENES,
    FAMILIES,
    PRIMITIVES,
    ROWS_PER_SOURCE,
)
from lewm.datasets.go2_recurrent_h4_rgb_sequences_v2 import (
    SCHEMA as CORRECTED_H6_V2_ROW_SCHEMA,
)


SCHEMA = "lewm_go2_explicit_plan_discounted_successor_state_v27_metadata_v1"
INDEX_ROOT = Path(
    ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity"
)
TRAIN_INDEX = INDEX_ROOT / "train.jsonl"
VALIDATION_INDEX = INDEX_ROOT / "val.jsonl"
TRAIN_INDEX_ROWS = 16_000
TRAIN_INDEX_BYTES = 10_328_000
TRAIN_INDEX_SHA256 = "aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77"
VALIDATION_INDEX_ROWS = 2_048
VALIDATION_INDEX_BYTES = 1_317_888
VALIDATION_INDEX_SHA256 = (
    "83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6"
)
TRAIN_PREFIX_ROWS = 6_400

SOURCE_IMAGE_SIZE = (224, 224)
CROP_BOX = (0, 28, 224, 196)
MODEL_IMAGE_SIZE = (112, 112)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PLAN_GAMMA = 0.9
PLAN_WEIGHTS = (1.0, 0.9, 0.81, 0.729)
PLAN_WEIGHT_SUM = 3.439
ADVANTAGE_DENOMINATOR_FLOOR = 1e-4
BOOTSTRAP_SEED = 20_260_730
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_LOWER_INDEX = 50

DONOR_MODULUS = VALIDATION_INDEX_ROWS
DONOR_RULE = (
    "minimize (((donor_index-row_index) mod 2048), donor_index); "
    "exclude zero offset"
)
EXPECTED_TAIL_DONOR_COUNT = 2_048
EXPECTED_WRONG_PLAN_DONOR_COUNT = 2_048
EXPECTED_EXACT_PLAN_DONOR_COUNT = 1_212
LEXICOGRAPHIC_FAMILIES = tuple(sorted(FAMILIES))
EXPECTED_EXACT_PLAN_COUNTS = MappingProxyType(
    dict(
        zip(
            LEXICOGRAPHIC_FAMILIES,
            (137, 144, 141, 159, 184, 170, 127, 150),
            strict=True,
        )
    )
)

_SCENE_RE = re.compile(
    r"^(?:" + "|".join(map(re.escape, FAMILIES)) + r")_[0-9a-f]{12}$"
)
_RGB_NAME_RE = re.compile(r"^frame_([0-9]{6})_env_([0-9]{2})\.png$")
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_DIR_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


class V27DataContractError(RuntimeError):
    """A frozen V27 metadata, image, donor, or metric contract failed."""


@dataclass(frozen=True, slots=True)
class IndexBinding:
    role: str
    path: Path
    row_count: int
    byte_count: int
    sha256: str


INDEX_BINDINGS = MappingProxyType(
    {
        "train": IndexBinding(
            role="train",
            path=TRAIN_INDEX,
            row_count=TRAIN_INDEX_ROWS,
            byte_count=TRAIN_INDEX_BYTES,
            sha256=TRAIN_INDEX_SHA256,
        ),
        "val": IndexBinding(
            role="val",
            path=VALIDATION_INDEX,
            row_count=VALIDATION_INDEX_ROWS,
            byte_count=VALIDATION_INDEX_BYTES,
            sha256=VALIDATION_INDEX_SHA256,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class H6V2Row:
    index: int
    role: str
    family: str
    scene_id: str
    rgb: tuple[str, ...]
    actions: tuple[int, ...]

    @property
    def current_rgb(self) -> str:
        return self.rgb[2]

    @property
    def future_rgb(self) -> tuple[str, ...]:
        return self.rgb[3:7]

    @property
    def plan(self) -> tuple[int, ...]:
        return self.actions[2:6]

    @property
    def first_plan_action(self) -> int:
        return self.actions[2]


@dataclass(frozen=True, slots=True)
class DonorPanels:
    tail_donor_indices: tuple[int, ...]
    wrong_plan_donor_indices: tuple[int, ...]
    exact_plan_wrong_scene_donor_indices: tuple[int | None, ...]
    exact_plan_eligible_indices: tuple[int, ...]
    exact_plan_counts_by_family: Mapping[str, int]
    panel_sha256: str

    def audit(self) -> dict[str, Any]:
        return {
            "rule": DONOR_RULE,
            "modulus": DONOR_MODULUS,
            "tail_donor_count": len(self.tail_donor_indices),
            "wrong_plan_donor_count": len(self.wrong_plan_donor_indices),
            "exact_plan_wrong_scene_row_count": len(
                self.exact_plan_eligible_indices
            ),
            "exact_plan_counts_by_family": dict(
                sorted(self.exact_plan_counts_by_family.items())
            ),
            "panel_sha256": self.panel_sha256,
        }


def _reject_constant(value: str) -> Any:
    raise V27DataContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise V27DataContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json_loads(raw: bytes) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except V27DataContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V27DataContractError("invalid UTF-8 JSON") from error


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise V27DataContractError("value is not canonical finite JSON") from error


def _validate_rgb_leaf(value: Any, *, scene_id: str) -> tuple[str, int, int]:
    if type(value) is not str:
        raise V27DataContractError("RGB leaf is not a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 3
        or path.parts[0] != scene_id
        or path.parts[1] != "rgb"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise V27DataContractError("RGB leaf left the exact scene/rgb allowlist")
    match = _RGB_NAME_RE.fullmatch(path.parts[2])
    if match is None:
        raise V27DataContractError("RGB filename is not canonical")
    frame_index, env_index = map(int, match.groups())
    if (
        not 0 <= frame_index < ROWS_PER_SOURCE
        or not 0 <= env_index < ENVS_PER_SOURCE
        or frame_index % ENVS_PER_SOURCE != env_index
    ):
        raise V27DataContractError("RGB numeric identity is invalid")
    return value, frame_index, env_index


def _decode_row(value: Any, *, role: str, index: int) -> H6V2Row:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "role",
        "family",
        "scene_id",
        "rgb",
        "actions",
    }:
        raise V27DataContractError(f"H6 row {index} fields changed")
    if value["schema"] != CORRECTED_H6_V2_ROW_SCHEMA or value["role"] != role:
        raise V27DataContractError(f"H6 row {index} schema or role changed")
    family = value["family"]
    scene_id = value["scene_id"]
    if (
        type(family) is not str
        or family not in FAMILIES
        or type(scene_id) is not str
        or not _SCENE_RE.fullmatch(scene_id)
        or not scene_id.startswith(f"{family}_")
    ):
        raise V27DataContractError(f"H6 row {index} family or scene changed")
    rgb_values = value["rgb"]
    actions = value["actions"]
    if type(rgb_values) is not list or len(rgb_values) != 7:
        raise V27DataContractError(f"H6 row {index} must contain seven RGB leaves")
    if (
        type(actions) is not list
        or len(actions) != 6
        or any(
            type(action) is not int or not 0 <= action < len(PRIMITIVES)
            for action in actions
        )
    ):
        raise V27DataContractError(f"H6 row {index} must contain six action IDs")

    leaves: list[str] = []
    frames: list[int] = []
    environments: list[int] = []
    for leaf in rgb_values:
        canonical, frame_index, env_index = _validate_rgb_leaf(
            leaf, scene_id=scene_id
        )
        leaves.append(canonical)
        frames.append(frame_index)
        environments.append(env_index)
    if len(set(leaves)) != 7 or len(set(environments)) != 1:
        raise V27DataContractError(f"H6 row {index} is not one causal stream")
    expected_delta = BLOCK_SIZE * ENVS_PER_SOURCE
    if any(right - left != expected_delta for left, right in zip(frames, frames[1:])):
        raise V27DataContractError(
            f"H6 row {index} does not use corrected five-tick causal endpoints"
        )
    return H6V2Row(
        index=index,
        role=role,
        family=family,
        scene_id=scene_id,
        rgb=tuple(leaves),
        actions=tuple(actions),
    )


def decode_index_bytes(
    raw: bytes,
    *,
    role: str,
    expected_rows: int,
) -> tuple[tuple[H6V2Row, ...], dict[str, Any]]:
    """Strictly decode canonical corrected-H6 V2 JSONL without following RGB."""

    if role not in INDEX_BINDINGS or type(expected_rows) is not int or expected_rows <= 0:
        raise V27DataContractError("invalid index role or expected row count")
    if type(raw) is not bytes or not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise V27DataContractError("index must be nonempty canonical LF JSONL")

    rows: list[H6V2Row] = []
    row_hashes: set[str] = set()
    transition_identities: set[tuple[str, str, int, str]] = set()
    family_counts: Counter[str] = Counter()
    family_scenes: dict[str, set[str]] = defaultdict(set)
    action_scene_support: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if not line.endswith(b"\n") or line == b"\n":
            raise V27DataContractError(f"index row {index} is not canonical JSONL")
        body = line[:-1]
        value = _strict_json_loads(body)
        if _canonical_json_bytes(value) != body:
            raise V27DataContractError(f"index row {index} is not canonical JSON")
        row_hash = hashlib.sha256(body).hexdigest()
        if row_hash in row_hashes:
            raise V27DataContractError(f"duplicate H6 row at index {index}")
        row_hashes.add(row_hash)
        row = _decode_row(value, role=role, index=index)
        for position, action in enumerate(row.actions):
            transition = (row.scene_id, row.rgb[position], action, row.rgb[position + 1])
            if transition in transition_identities:
                raise V27DataContractError("duplicate corrected H6 transition")
            transition_identities.add(transition)
        rows.append(row)
        family_counts[row.family] += 1
        family_scenes[row.family].add(row.scene_id)
        for position in range(2, 6):
            action_scene_support[(row.family, position, row.actions[position])].add(
                row.scene_id
            )

    if len(rows) != expected_rows:
        raise V27DataContractError("bound H6 row count changed")
    expected_per_family = expected_rows // len(FAMILIES)
    if expected_rows % len(FAMILIES) or any(
        family_counts[family] != expected_per_family for family in FAMILIES
    ):
        raise V27DataContractError("H6 role is not exactly family-balanced")
    if any(
        len(family_scenes[family]) != EXPECTED_SCENES[role][family]
        for family in FAMILIES
    ):
        raise V27DataContractError("H6 family scene inventory changed")
    minimum_scene_breadth = 8 if role == "train" else 1
    if any(
        len(action_scene_support[(family, position, action)])
        < minimum_scene_breadth
        for family in FAMILIES
        for position in range(2, 6)
        for action in range(len(PRIMITIVES))
    ):
        raise V27DataContractError("H6 future action-position coverage changed")
    return tuple(rows), {
        "row_count": len(rows),
        "scene_count": len({row.scene_id for row in rows}),
        "family_rows": dict(sorted(family_counts.items())),
        "family_scenes": {
            family: len(family_scenes[family]) for family in LEXICOGRAPHIC_FAMILIES
        },
        "minimum_future_action_position_scene_breadth": min(
            len(value) for value in action_scene_support.values()
        ),
        "ordered_row_identity_sha256": hashlib.sha256(
            _canonical_json_bytes(
                [
                    hashlib.sha256(line[:-1]).hexdigest()
                    for line in raw.splitlines(keepends=True)
                ]
            )
        ).hexdigest(),
        "rgb_open_count": 0,
    }


def _read_bound_relative_file(repo_root: Path, binding: IndexBinding) -> bytes:
    root = Path(repo_root)
    if not root.is_absolute():
        raise V27DataContractError("repository root must be absolute")
    parts = binding.path.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise V27DataContractError("bound index path is not canonical relative")
    descriptor = os.open(root, _DIR_FLAGS)
    file_descriptor: int | None = None
    try:
        for component in parts[:-1]:
            child = os.open(component, _DIR_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(parts[-1], _READ_FLAGS, dir_fd=descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise V27DataContractError("bound H6 index is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        raw = b"".join(chunks)
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or len(raw) != binding.byte_count
            or before.st_size != binding.byte_count
            or hashlib.sha256(raw).hexdigest() != binding.sha256
        ):
            raise V27DataContractError("bound H6 index bytes or identity changed")
        return raw
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)


def load_bound_index(
    repo_root: Path,
    *,
    role: str,
) -> tuple[tuple[H6V2Row, ...], dict[str, Any]]:
    """Open and validate one exact frozen H6 V2 index, never an RGB leaf."""

    binding = INDEX_BINDINGS.get(role)
    if binding is None:
        raise V27DataContractError("only frozen train and val roles are available")
    raw = _read_bound_relative_file(Path(repo_root), binding)
    rows, audit = decode_index_bytes(
        raw,
        role=binding.role,
        expected_rows=binding.row_count,
    )
    return rows, {
        "role": binding.role,
        "path": binding.path.as_posix(),
        "file_sha256": binding.sha256,
        "byte_count": binding.byte_count,
        **audit,
    }


def _donor_key(row_index: int, donor_index: int, *, modulus: int) -> tuple[int, int]:
    if (
        type(row_index) is not int
        or type(donor_index) is not int
        or type(modulus) is not int
        or modulus <= 1
        or not 0 <= row_index < modulus
        or not 0 <= donor_index < modulus
    ):
        raise V27DataContractError("donor indices left the frozen modulus")
    offset = (donor_index - row_index) % modulus
    if offset == 0:
        raise V27DataContractError("zero-offset/self donor is forbidden")
    return offset, donor_index


def select_donor_index(
    *,
    row_index: int,
    candidate_indices: Sequence[int],
    predicate: Callable[[int], bool],
    modulus: int,
) -> int | None:
    """Apply the frozen cyclic-offset donor rule to an eligible candidate set."""

    candidates = [
        candidate
        for candidate in candidate_indices
        if candidate != row_index and predicate(candidate)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda donor: _donor_key(row_index, donor, modulus=modulus))


def _frame_disjoint(left: frozenset[str], right: frozenset[str]) -> bool:
    return left.isdisjoint(right)


def build_donor_panels(rows: Sequence[H6V2Row]) -> DonorPanels:
    """Build and verify all frozen V27 validation-only donor panels."""

    if len(rows) != VALIDATION_INDEX_ROWS or any(
        row.index != index or row.role != "val" for index, row in enumerate(rows)
    ):
        raise V27DataContractError("donor panels require the exact ordered val role")
    frames = tuple(frozenset(row.rgb) for row in rows)
    family_indices: dict[str, list[int]] = defaultdict(list)
    family_a0_indices: dict[tuple[str, int], list[int]] = defaultdict(list)
    family_plan_indices: dict[tuple[str, tuple[int, ...]], list[int]] = defaultdict(list)
    for row in rows:
        family_indices[row.family].append(row.index)
        family_a0_indices[(row.family, row.first_plan_action)].append(row.index)
        family_plan_indices[(row.family, row.plan)].append(row.index)

    tail: list[int] = []
    wrong_plan: list[int] = []
    exact: list[int | None] = []
    exact_eligible: list[int] = []
    exact_counts: Counter[str] = Counter()
    for row in rows:
        def common(candidate_index: int) -> bool:
            donor = rows[candidate_index]
            return (
                donor.scene_id != row.scene_id
                and _frame_disjoint(frames[row.index], frames[candidate_index])
            )

        tail_donor = select_donor_index(
            row_index=row.index,
            candidate_indices=family_a0_indices[(row.family, row.first_plan_action)],
            predicate=lambda candidate: common(candidate)
            and sum(
                left != right
                for left, right in zip(row.plan[1:], rows[candidate].plan[1:], strict=True)
            )
            >= 2,
            modulus=DONOR_MODULUS,
        )
        full_donor = select_donor_index(
            row_index=row.index,
            candidate_indices=family_indices[row.family],
            predicate=lambda candidate: common(candidate)
            and rows[candidate].plan != row.plan,
            modulus=DONOR_MODULUS,
        )
        exact_donor = select_donor_index(
            row_index=row.index,
            candidate_indices=family_plan_indices[(row.family, row.plan)],
            predicate=common,
            modulus=DONOR_MODULUS,
        )
        if tail_donor is None or full_donor is None:
            raise V27DataContractError("a mandatory validation donor is absent")
        tail.append(tail_donor)
        wrong_plan.append(full_donor)
        exact.append(exact_donor)
        if exact_donor is not None:
            exact_eligible.append(row.index)
            exact_counts[row.family] += 1

    exact_count_mapping = {
        family: exact_counts[family] for family in LEXICOGRAPHIC_FAMILIES
    }
    if (
        len(tail) != EXPECTED_TAIL_DONOR_COUNT
        or len(wrong_plan) != EXPECTED_WRONG_PLAN_DONOR_COUNT
        or len(exact_eligible) != EXPECTED_EXACT_PLAN_DONOR_COUNT
        or exact_count_mapping != EXPECTED_EXACT_PLAN_COUNTS
    ):
        raise V27DataContractError("observed donor panel counts changed")
    panel_core = {
        "tail": tail,
        "wrong_plan": wrong_plan,
        "exact_plan_wrong_scene": exact,
    }
    return DonorPanels(
        tail_donor_indices=tuple(tail),
        wrong_plan_donor_indices=tuple(wrong_plan),
        exact_plan_wrong_scene_donor_indices=tuple(exact),
        exact_plan_eligible_indices=tuple(exact_eligible),
        exact_plan_counts_by_family=exact_count_mapping,
        panel_sha256=hashlib.sha256(_canonical_json_bytes(panel_core)).hexdigest(),
    )


def metadata_only_preflight(repo_root: Path) -> dict[str, Any]:
    """Reproduce the frozen index and donor facts without importing image code."""

    train_rows, train_audit = load_bound_index(Path(repo_root), role="train")
    validation_rows, validation_audit = load_bound_index(Path(repo_root), role="val")
    train_scenes = {row.scene_id for row in train_rows}
    validation_scenes = {row.scene_id for row in validation_rows}
    train_rgb = {leaf for row in train_rows for leaf in row.rgb}
    validation_rgb = {leaf for row in validation_rows for leaf in row.rgb}
    if train_scenes & validation_scenes or train_rgb & validation_rgb:
        raise V27DataContractError("train and validation metadata are not disjoint")
    if len(train_rows[:TRAIN_PREFIX_ROWS]) != TRAIN_PREFIX_ROWS:
        raise V27DataContractError("frozen V27 train prefix is unavailable")
    panels = build_donor_panels(validation_rows)
    return {
        "schema": SCHEMA,
        "status": "PASS_METADATA_ONLY_PREFLIGHT",
        "train": train_audit,
        "validation": validation_audit,
        "train_prefix_rows": TRAIN_PREFIX_ROWS,
        "train_validation_scene_overlap_count": 0,
        "train_validation_rgb_path_overlap_count": 0,
        "donors": panels.audit(),
        "rgb_open_count": 0,
        "gpu_use_count": 0,
        "generated_write_count": 0,
    }


def rectify_h6_rgb_bytes(raw: bytes) -> Any:
    """Decode exact 224-square RGB PNG bytes and apply the frozen V27 crop."""

    if type(raw) is not bytes or not raw:
        raise V27DataContractError("RGB payload must be nonempty bytes")
    try:
        import torch
        from PIL import Image
    except ImportError as error:  # pragma: no cover - runtime dependency failure
        raise V27DataContractError("Pillow and torch are required for RGB decoding") from error

    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "RGB" or image.size != SOURCE_IMAGE_SIZE:
                raise V27DataContractError("H6 image must be exact 224x224 RGB PNG")
            image.load()
            image = image.crop(CROP_BOX)
            if image.size != (224, 168):
                raise V27DataContractError("H6 crop geometry changed")
            image = image.resize(MODEL_IMAGE_SIZE, Image.Resampling.BILINEAR)
            pixels = bytearray(image.tobytes())
    except V27DataContractError:
        raise
    except Exception as error:
        raise V27DataContractError("H6 PNG decode failed") from error

    tensor = torch.frombuffer(pixels, dtype=torch.uint8).reshape(112, 112, 3)
    tensor = tensor.permute(2, 0, 1).contiguous().to(dtype=torch.float32).div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    tensor.sub_(mean).div_(std)
    if tuple(tensor.shape) != (3, 112, 112) or not bool(torch.isfinite(tensor).all()):
        raise V27DataContractError("rectified H6 tensor is invalid")
    return tensor


def discounted_successor_target(target_states: Any) -> Any:
    """Form the fixed-gamma stop-gradient V27 target in float32."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependency failure
        raise V27DataContractError("torch is required for target construction") from error
    if not isinstance(target_states, torch.Tensor):
        raise TypeError("target_states must be a torch.Tensor")
    if target_states.ndim != 5 or target_states.shape[0] <= 0 or tuple(
        target_states.shape[1:]
    ) != (4, 64, 64, 64):
        raise ValueError("target_states must have shape (B,4,64,64,64)")
    if target_states.dtype != torch.float32 or not bool(torch.isfinite(target_states).all()):
        raise V27DataContractError("target states must be finite float32")
    detached = target_states.detach()
    weights = torch.tensor(PLAN_WEIGHTS, dtype=torch.float32, device=detached.device)
    weights = weights / weights.sum()
    result = (detached * weights[None, :, None, None, None]).sum(dim=1)
    if result.dtype != torch.float32 or not bool(torch.isfinite(result).all()):
        raise V27DataContractError("discounted successor target is invalid")
    return result


def _coerce_row_values(
    rows: Sequence[H6V2Row],
    values: Sequence[float] | Mapping[int, float],
) -> dict[int, float]:
    row_by_index = {row.index: row for row in rows}
    if len(row_by_index) != len(rows):
        raise V27DataContractError("metric rows contain duplicate indices")
    if isinstance(values, Mapping):
        if any(type(index) is not int for index in values):
            raise V27DataContractError("metric mapping keys must be integer row indices")
        result = {index: float(value) for index, value in values.items()}
    else:
        if len(values) != len(rows):
            raise V27DataContractError("metric vector length differs from row count")
        result = {row.index: float(value) for row, value in zip(rows, values, strict=True)}
    if not result or not set(result).issubset(row_by_index):
        raise V27DataContractError("metric values reference an unknown or empty row panel")
    if any(not math.isfinite(value) for value in result.values()):
        raise V27DataContractError("metric values must all be finite")
    return result


def bootstrap_scene_family_lower_95(
    scene_means: Mapping[str, Mapping[str, float]],
    *,
    observation_update: int,
    metric_name: str,
) -> float:
    """Run the exact fresh-PCG64, family-macro V27 scene bootstrap."""

    if (
        type(observation_update) is not int
        or observation_update < 0
        or type(metric_name) is not str
        or not metric_name
    ):
        raise V27DataContractError("bootstrap observation identity is invalid")
    if set(scene_means) != set(LEXICOGRAPHIC_FAMILIES):
        raise V27DataContractError("bootstrap scene means do not cover eight families")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    family_replicates: list[np.ndarray] = []
    for family in LEXICOGRAPHIC_FAMILIES:
        by_scene = scene_means[family]
        if not by_scene:
            raise V27DataContractError("bootstrap family has no scene means")
        scene_vector = np.asarray(
            [float(by_scene[scene]) for scene in sorted(by_scene)], dtype=np.float64
        )
        if not np.isfinite(scene_vector).all():
            raise V27DataContractError("bootstrap scene vector is nonfinite")
        indices = rng.integers(
            0,
            scene_vector.size,
            size=(BOOTSTRAP_REPLICATES, scene_vector.size),
        )
        family_replicates.append(scene_vector[indices].mean(axis=1))
    replicates = np.stack(family_replicates, axis=0).mean(axis=0)
    if replicates.shape != (BOOTSTRAP_REPLICATES,) or not np.isfinite(replicates).all():
        raise V27DataContractError("bootstrap replicate vector is invalid")
    return float(np.sort(replicates)[BOOTSTRAP_LOWER_INDEX])


def aggregate_normalized_advantage(
    rows: Sequence[H6V2Row],
    row_values: Sequence[float] | Mapping[int, float],
    *,
    observation_update: int,
    metric_name: str,
) -> dict[str, Any]:
    """Aggregate rows to scenes to families, then bootstrap complete scenes."""

    values = _coerce_row_values(rows, row_values)
    row_by_index = {row.index: row for row in rows}
    scene_rows: dict[tuple[str, str], list[float]] = defaultdict(list)
    for index, value in values.items():
        row = row_by_index[index]
        scene_rows[(row.family, row.scene_id)].append(value)
    scene_means: dict[str, dict[str, float]] = {
        family: {} for family in LEXICOGRAPHIC_FAMILIES
    }
    for (family, scene), items in sorted(scene_rows.items()):
        scene_means[family][scene] = float(np.asarray(items, dtype=np.float64).mean())
    if any(not scene_means[family] for family in LEXICOGRAPHIC_FAMILIES):
        raise V27DataContractError("metric panel lost a family")
    family_means = {
        family: float(
            np.asarray(list(scene_means[family].values()), dtype=np.float64).mean()
        )
        for family in LEXICOGRAPHIC_FAMILIES
    }
    aggregate_mean = float(
        np.asarray(list(family_means.values()), dtype=np.float64).mean()
    )
    lower = bootstrap_scene_family_lower_95(
        scene_means,
        observation_update=observation_update,
        metric_name=metric_name,
    )
    if not all(math.isfinite(value) for value in (*family_means.values(), aggregate_mean, lower)):
        raise V27DataContractError("aggregated advantage is nonfinite")
    return {
        "metric_name": metric_name,
        "observation_update": observation_update,
        "row_count": len(values),
        "scene_count": sum(len(value) for value in scene_means.values()),
        "equal_family_mean": aggregate_mean,
        "bootstrap_lower_95": lower,
        "positive_family_count": sum(value > 0.0 for value in family_means.values()),
        "family_equal_scene_means": family_means,
        "family_scene_counts": {
            family: len(scene_means[family]) for family in LEXICOGRAPHIC_FAMILIES
        },
    }


def summarize_plan_energies(
    rows: Sequence[H6V2Row],
    *,
    observation_update: int,
    correct_energy: Sequence[float],
    persistence_energy: Sequence[float],
    wrong_plan_energy: Sequence[float],
    tail_energy: Sequence[float],
    wrong_scene_energy: Mapping[int, float],
    mean_prior_energy: Sequence[float],
) -> dict[str, Any]:
    """Compute the exact V27 ratio and five normalized control summaries."""

    arrays: dict[str, np.ndarray] = {}
    for name, values in (
        ("correct", correct_energy),
        ("persistence", persistence_energy),
        ("wrong_plan", wrong_plan_energy),
        ("tail", tail_energy),
        ("mean_prior", mean_prior_energy),
    ):
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(rows),) or not np.isfinite(array).all() or (array < 0.0).any():
            raise V27DataContractError(f"{name} energy vector is invalid")
        arrays[name] = array
    wrong_scene = _coerce_row_values(rows, wrong_scene_energy)
    if any(value < 0.0 for value in wrong_scene.values()):
        raise V27DataContractError("wrong-scene energies must be nonnegative")
    persistence_mean = float(arrays["persistence"].mean())
    if not persistence_mean > 1e-6:
        raise V27DataContractError("aggregate unclamped persistence energy is too small")
    denominator = np.maximum(arrays["persistence"], ADVANTAGE_DENOMINATOR_FLOOR)
    if not np.isfinite(denominator).all():
        raise V27DataContractError("per-row advantage denominator is nonfinite")
    correct_ratio = float(arrays["correct"].mean() / persistence_mean)
    advantages: dict[str, Sequence[float] | Mapping[int, float]] = {
        "persistence_advantage": (
            arrays["persistence"] - arrays["correct"]
        )
        / denominator,
        "wrong_plan_advantage": (
            arrays["wrong_plan"] - arrays["correct"]
        )
        / denominator,
        "tail_advantage": (arrays["tail"] - arrays["correct"]) / denominator,
        "wrong_scene_advantage": {
            index: (energy - arrays["correct"][index]) / denominator[index]
            for index, energy in wrong_scene.items()
        },
        "mean_prior_advantage": (
            arrays["mean_prior"] - arrays["correct"]
        )
        / denominator,
    }
    summaries = {
        metric_name: aggregate_normalized_advantage(
            rows,
            values,
            observation_update=observation_update,
            metric_name=metric_name,
        )
        for metric_name, values in advantages.items()
    }
    if not math.isfinite(correct_ratio):
        raise V27DataContractError("correct prediction ratio is nonfinite")
    return {
        "observation_update": observation_update,
        "correct_ratio": correct_ratio,
        "mean_correct_energy": float(arrays["correct"].mean()),
        "mean_unclamped_persistence_energy": persistence_mean,
        "advantages": summaries,
        "all_registered_values_finite": True,
    }


__all__ = [
    "ADVANTAGE_DENOMINATOR_FLOOR",
    "BOOTSTRAP_LOWER_INDEX",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "CORRECTED_H6_V2_ROW_SCHEMA",
    "CROP_BOX",
    "DONOR_MODULUS",
    "DONOR_RULE",
    "DonorPanels",
    "EXPECTED_EXACT_PLAN_COUNTS",
    "H6V2Row",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "INDEX_BINDINGS",
    "LEXICOGRAPHIC_FAMILIES",
    "MODEL_IMAGE_SIZE",
    "PLAN_GAMMA",
    "PLAN_WEIGHTS",
    "PLAN_WEIGHT_SUM",
    "SCHEMA",
    "SOURCE_IMAGE_SIZE",
    "TRAIN_PREFIX_ROWS",
    "V27DataContractError",
    "aggregate_normalized_advantage",
    "bootstrap_scene_family_lower_95",
    "build_donor_panels",
    "decode_index_bytes",
    "discounted_successor_target",
    "load_bound_index",
    "metadata_only_preflight",
    "rectify_h6_rgb_bytes",
    "select_donor_index",
    "summarize_plan_energies",
]
