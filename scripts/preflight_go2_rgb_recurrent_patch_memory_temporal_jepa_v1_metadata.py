#!/usr/bin/env python3
"""Metadata-only preflight for recurrent patch-memory temporal JEPA V1.

The preflight opens only two explicitly bound corrected-H6 JSONL indices.  It
never follows any RGB path and never opens a checkpoint, tensor, generated
model output, navigation role, held-out role, or sealed role.  Its byte-level
entry point is intentionally injectable so synthetic tests need no repository
payload.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence

from lewm.benchmarks import (
    go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as metrics,
)


ROW_SCHEMA = "lewm_go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity"
RECEIPT_SCHEMA = f"{metrics.SCHEMA_PREFIX}_metadata_preflight_receipt_v1"
AUTHORITY_SCHEMA = f"{metrics.SCHEMA_PREFIX}_metadata_preflight_authority_v1"
RESERVATION_SCHEMA = (
    f"{metrics.SCHEMA_PREFIX}_metadata_preflight_attempt_reservation_v1"
)
AUTHORITY_STATUS = "AUTHORIZED_METADATA_ONLY_PREFLIGHT_ONE_SHOT"
TRAIN_INDEX_PATH = (
    ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/"
    "train.jsonl"
)
VALIDATION_INDEX_PATH = (
    ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/"
    "val.jsonl"
)
TRAIN_INDEX_SHA256 = (
    "aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77"
)
TRAIN_INDEX_BYTE_COUNT = 10_328_000
VALIDATION_INDEX_SHA256 = (
    "83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6"
)
VALIDATION_INDEX_BYTE_COUNT = 1_317_888
FROZEN_OUTPUT_ROOT = (
    ".generated/go2_rgb_recurrent_patch_memory_temporal_jepa_v1_"
    "metadata_preflight/attempt_v1"
)

ENVIRONMENT_COUNT = 48
FRAME_COUNT_PER_SOURCE = 48_000
CAUSAL_FRAME_DELTA = 5 * ENVIRONMENT_COUNT
_SCENE_RE = re.compile(
    r"^(?:"
    + "|".join(map(re.escape, metrics.REGISTERED_FAMILIES))
    + r")_[0-9a-f]{12}$"
)
_RGB_RE = re.compile(r"^frame_([0-9]{6})_env_([0-9]{2})\.png$")
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


class MetadataPreflightError(RuntimeError):
    """The metadata preflight failed closed."""


@dataclass(frozen=True, slots=True)
class IndexBinding:
    role: str
    path: str
    file_sha256: str
    byte_count: int
    row_count: int


@dataclass(frozen=True, slots=True)
class PreflightPolicy:
    """Exact expected identities, injectable only for focused synthetic tests."""

    train_binding: IndexBinding
    validation_binding: IndexBinding
    output_root: str
    train_rows_per_family: int
    sentinel_rows_per_family: int
    train_schedule_sha256: str
    sentinel_indices_sha256: str
    full_donor_sha256: str
    sentinel_donor_sha256: str
    train_scene_counts: Mapping[str, int]
    validation_scene_counts: Mapping[str, int]
    sentinel_scene_count: int
    full_wrong_action_row_counts: Mapping[str, int]
    full_wrong_action_scene_counts: Mapping[str, int]
    sentinel_wrong_action_row_counts: Mapping[str, int]
    sentinel_wrong_action_scene_counts: Mapping[str, int]
    require_all_visible_actions: bool = True


@dataclass(frozen=True, slots=True)
class ValidatedAuthority:
    """Canonical authority value and its exact source-file binding."""

    repository_root: str
    output_root: str
    binding: Mapping[str, Any]
    value: Mapping[str, Any]


FROZEN_POLICY = PreflightPolicy(
    train_binding=IndexBinding(
        role="train",
        path=TRAIN_INDEX_PATH,
        file_sha256=TRAIN_INDEX_SHA256,
        byte_count=TRAIN_INDEX_BYTE_COUNT,
        row_count=metrics.TRAIN_INDEX_ROW_COUNT,
    ),
    validation_binding=IndexBinding(
        role="val",
        path=VALIDATION_INDEX_PATH,
        file_sha256=VALIDATION_INDEX_SHA256,
        byte_count=VALIDATION_INDEX_BYTE_COUNT,
        row_count=metrics.VALIDATION_INDEX_ROW_COUNT,
    ),
    output_root=FROZEN_OUTPUT_ROOT,
    train_rows_per_family=metrics.TRAIN_ROWS_PER_FAMILY,
    sentinel_rows_per_family=metrics.SENTINEL_ROWS_PER_FAMILY,
    train_schedule_sha256=metrics.TRAIN_SCHEDULE_SHA256,
    sentinel_indices_sha256=metrics.SENTINEL_INDICES_SHA256,
    full_donor_sha256=metrics.FULL_WRONG_HISTORY_DONORS_SHA256,
    sentinel_donor_sha256=metrics.SENTINEL_WRONG_HISTORY_DONORS_SHA256,
    train_scene_counts=metrics.TRAIN_SCENE_COUNTS,
    validation_scene_counts=metrics.VALIDATION_SCENE_COUNTS,
    sentinel_scene_count=metrics.SENTINEL_SCENE_COUNT,
    full_wrong_action_row_counts=metrics.FULL_WRONG_ACTION_ROW_COUNTS,
    full_wrong_action_scene_counts=metrics.FULL_WRONG_ACTION_SCENE_COUNTS,
    sentinel_wrong_action_row_counts=metrics.SENTINEL_WRONG_ACTION_ROW_COUNTS,
    sentinel_wrong_action_scene_counts=metrics.SENTINEL_WRONG_ACTION_SCENE_COUNTS,
)


def _reject_constant(value: str) -> Any:
    raise MetadataPreflightError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise MetadataPreflightError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_json_loads(raw: bytes) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except MetadataPreflightError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MetadataPreflightError("invalid UTF-8 JSON") from error


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
        raise MetadataPreflightError("value is not canonical finite JSON") from error


def _with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(core)
    if "content_sha256" in value:
        raise MetadataPreflightError("receipt core already has a content hash")
    value["content_sha256"] = hashlib.sha256(
        _canonical_json_bytes(value)
    ).hexdigest()
    return value


def _validate_content_bound_json(raw: bytes, *, name: str) -> dict[str, Any]:
    if (
        type(raw) is not bytes
        or not raw.endswith(b"\n")
        or raw.count(b"\n") != 1
        or b"\r" in raw
    ):
        raise MetadataPreflightError(f"{name} is not one canonical JSON line")
    body = raw[:-1]
    value = _strict_json_loads(body)
    if type(value) is not dict or _canonical_json_bytes(value) != body:
        raise MetadataPreflightError(f"{name} is not canonical JSON")
    core = dict(value)
    content_sha256 = core.pop("content_sha256", None)
    if (
        type(content_sha256) is not str
        or len(content_sha256) != 64
        or hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
        != content_sha256
    ):
        raise MetadataPreflightError(f"{name} content binding changed")
    return value


def validate_authority_bytes(
    raw: bytes,
    *,
    authority_path: str,
    policy: PreflightPolicy = FROZEN_POLICY,
) -> ValidatedAuthority:
    """Validate one canonical metadata-only one-shot authority."""

    if not isinstance(policy, PreflightPolicy):
        raise MetadataPreflightError("authority policy type changed")
    path = PurePosixPath(authority_path)
    if (
        type(authority_path) is not str
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MetadataPreflightError("authority path is not canonical relative")
    value = _validate_content_bound_json(raw, name="metadata preflight authority")
    if set(value) != {
        "schema",
        "status",
        "preregistration_commit",
        "one_shot",
        "repository_root",
        "output_root",
        "output_root_absent_at_authorization",
        "train_index",
        "validation_index",
        "content_sha256",
    }:
        raise MetadataPreflightError("metadata preflight authority fields changed")
    repository_root = value["repository_root"]
    output_root = value["output_root"]
    output_path = PurePosixPath(output_root) if type(output_root) is str else None
    if (
        value["schema"] != AUTHORITY_SCHEMA
        or value["status"] != AUTHORITY_STATUS
        or value["preregistration_commit"] != metrics.PREREGISTRATION_COMMIT
        or value["one_shot"] is not True
        or value["output_root_absent_at_authorization"] is not True
        or type(repository_root) is not str
        or not Path(repository_root).is_absolute()
        or output_path is None
        or output_root != policy.output_root
        or output_path.is_absolute()
        or not output_path.parts
        or any(part in {"", ".", ".."} for part in output_path.parts)
        or value["train_index"] != asdict(policy.train_binding)
        or value["validation_index"] != asdict(policy.validation_binding)
    ):
        raise MetadataPreflightError("metadata preflight authority scope changed")
    binding = {
        "path": path.as_posix(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": value["content_sha256"],
        "byte_count": len(raw),
    }
    return ValidatedAuthority(
        repository_root=repository_root,
        output_root=output_path.as_posix(),
        binding=binding,
        value=value,
    )


def build_reservation(
    authority: ValidatedAuthority,
    *,
    created_utc: str,
) -> dict[str, Any]:
    """Build the in-memory record bound to the atomic attempt reservation."""

    if (
        not isinstance(authority, ValidatedAuthority)
        or type(created_utc) is not str
        or not created_utc
    ):
        raise MetadataPreflightError("reservation inputs changed")
    return _with_content_sha256(
        {
            "schema": RESERVATION_SCHEMA,
            "status": "RESERVED_METADATA_PREFLIGHT_ONE_SHOT",
            "preregistration_commit": metrics.PREREGISTRATION_COMMIT,
            "authority": dict(authority.binding),
            "repository_root": authority.repository_root,
            "output_root": authority.output_root,
            "created_utc": created_utc,
        }
    )


def _validate_reservation(
    reservation: Mapping[str, Any],
    authority: ValidatedAuthority,
) -> dict[str, Any]:
    if type(reservation) is not dict:
        raise MetadataPreflightError("reservation is not an exact object")
    value = dict(reservation)
    core = dict(value)
    content_sha256 = core.pop("content_sha256", None)
    if (
        set(value)
        != {
            "schema",
            "status",
            "preregistration_commit",
            "authority",
            "repository_root",
            "output_root",
            "created_utc",
            "content_sha256",
        }
        or type(content_sha256) is not str
        or hashlib.sha256(_canonical_json_bytes(core)).hexdigest()
        != content_sha256
        or value["schema"] != RESERVATION_SCHEMA
        or value["status"] != "RESERVED_METADATA_PREFLIGHT_ONE_SHOT"
        or value["preregistration_commit"] != metrics.PREREGISTRATION_COMMIT
        or value["authority"] != dict(authority.binding)
        or value["repository_root"] != authority.repository_root
        or value["output_root"] != authority.output_root
        or type(value["created_utc"]) is not str
        or not value["created_utc"]
    ):
        raise MetadataPreflightError("attempt reservation identity changed")
    return value


def _validate_leaf(value: Any, *, scene_id: str) -> tuple[str, int, int]:
    if type(value) is not str:
        raise MetadataPreflightError("RGB metadata leaf is not a string")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or len(path.parts) != 3
        or path.parts[:2] != (scene_id, "rgb")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MetadataPreflightError("RGB metadata leaf escaped scene/rgb")
    match = _RGB_RE.fullmatch(path.name)
    if match is None:
        raise MetadataPreflightError("RGB metadata filename is not canonical")
    frame, environment = map(int, match.groups())
    if (
        not 0 <= frame < FRAME_COUNT_PER_SOURCE
        or not 0 <= environment < ENVIRONMENT_COUNT
        or frame % ENVIRONMENT_COUNT != environment
    ):
        raise MetadataPreflightError("RGB numeric metadata identity is invalid")
    return value, frame, environment


def _decode_row(value: Any, *, role: str, index: int) -> metrics.MetadataRow:
    if type(value) is not dict or set(value) != {
        "schema",
        "role",
        "family",
        "scene_id",
        "rgb",
        "actions",
    }:
        raise MetadataPreflightError(f"H6 row {index} fields changed")
    if value["schema"] != ROW_SCHEMA or value["role"] != role:
        raise MetadataPreflightError(f"H6 row {index} schema or role changed")
    family = value["family"]
    scene_id = value["scene_id"]
    if (
        type(family) is not str
        or family not in metrics.REGISTERED_FAMILIES
        or type(scene_id) is not str
        or _SCENE_RE.fullmatch(scene_id) is None
        or not scene_id.startswith(f"{family}_")
    ):
        raise MetadataPreflightError(f"H6 row {index} family or scene changed")
    rgb_values = value["rgb"]
    actions = value["actions"]
    if type(rgb_values) is not list or len(rgb_values) != 7:
        raise MetadataPreflightError(f"H6 row {index} must have seven RGB paths")
    if (
        type(actions) is not list
        or len(actions) != 6
        or any(
            type(action) is not int
            or not 0 <= action < metrics.ACTION_COUNT
            for action in actions
        )
    ):
        raise MetadataPreflightError(f"H6 row {index} must have six action IDs")
    leaves: list[str] = []
    frames: list[int] = []
    environments: list[int] = []
    for raw_leaf in rgb_values:
        leaf, frame, environment = _validate_leaf(raw_leaf, scene_id=scene_id)
        leaves.append(leaf)
        frames.append(frame)
        environments.append(environment)
    if (
        len(set(leaves)) != 7
        or len(set(environments)) != 1
        or any(
            right - left != CAUSAL_FRAME_DELTA
            for left, right in zip(frames, frames[1:])
        )
    ):
        raise MetadataPreflightError(f"H6 row {index} causal RGB sequence changed")
    return metrics.MetadataRow(
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
) -> tuple[metrics.MetadataRow, ...]:
    """Strictly decode canonical JSONL without following any referenced path."""

    if (
        type(raw) is not bytes
        or not raw
        or not raw.endswith(b"\n")
        or b"\r" in raw
        or role not in {"train", "val"}
        or type(expected_rows) is not int
        or expected_rows <= 0
    ):
        raise MetadataPreflightError("index bytes, role, or row count is invalid")
    rows: list[metrics.MetadataRow] = []
    row_hashes: set[str] = set()
    for index, line in enumerate(raw.splitlines(keepends=True)):
        if not line.endswith(b"\n") or line == b"\n":
            raise MetadataPreflightError(f"index row {index} is not canonical JSONL")
        body = line[:-1]
        value = _strict_json_loads(body)
        if _canonical_json_bytes(value) != body:
            raise MetadataPreflightError(f"index row {index} is not canonical JSON")
        row_hash = hashlib.sha256(body).hexdigest()
        if row_hash in row_hashes:
            raise MetadataPreflightError(f"duplicate H6 row at index {index}")
        row_hashes.add(row_hash)
        rows.append(_decode_row(value, role=role, index=index))
    if len(rows) != expected_rows:
        raise MetadataPreflightError("H6 index row count changed")
    return tuple(rows)


def _binding_check(raw: bytes, binding: IndexBinding) -> None:
    if (
        type(binding.path) is not str
        or not binding.path
        or type(raw) is not bytes
        or len(raw) != binding.byte_count
        or hashlib.sha256(raw).hexdigest() != binding.file_sha256
    ):
        raise MetadataPreflightError(f"{binding.role} index binding changed")


def _family_inventory(
    rows: Sequence[metrics.MetadataRow],
) -> tuple[dict[str, int], dict[str, int]]:
    return (
        {
            family: sum(row.family == family for row in rows)
            for family in metrics.REGISTERED_FAMILIES
        },
        {
            family: len(
                {row.scene_id for row in rows if row.family == family}
            )
            for family in metrics.REGISTERED_FAMILIES
        },
    )


def _visible_action_coverage(
    rows: Sequence[metrics.MetadataRow],
    indices: Sequence[int],
) -> dict[str, list[int]]:
    return {
        f"actions[{position}]": sorted(
            {rows[index].actions[position] for index in indices}
        )
        for position in range(3)
    }


def preflight_from_bytes(
    *,
    train_raw: bytes,
    validation_raw: bytes,
    authority: ValidatedAuthority,
    reservation: Mapping[str, Any],
    policy: PreflightPolicy = FROZEN_POLICY,
    metadata_index_open_count: int = 0,
) -> dict[str, Any]:
    """Run the complete preflight over already-read metadata-only JSONL."""

    if (
        not isinstance(policy, PreflightPolicy)
        or not isinstance(authority, ValidatedAuthority)
        or type(metadata_index_open_count) is not int
        or metadata_index_open_count not in {0, 2}
    ):
        raise MetadataPreflightError("preflight policy type changed")
    reserved = _validate_reservation(reservation, authority)
    _binding_check(train_raw, policy.train_binding)
    _binding_check(validation_raw, policy.validation_binding)
    train = decode_index_bytes(
        train_raw,
        role=policy.train_binding.role,
        expected_rows=policy.train_binding.row_count,
    )
    validation = decode_index_bytes(
        validation_raw,
        role=policy.validation_binding.role,
        expected_rows=policy.validation_binding.row_count,
    )

    train_family_rows, train_family_scenes = _family_inventory(train)
    validation_family_rows, validation_family_scenes = _family_inventory(validation)
    expected_train_family_rows = len(train) // len(metrics.REGISTERED_FAMILIES)
    expected_validation_family_rows = len(validation) // len(
        metrics.REGISTERED_FAMILIES
    )
    if (
        len(train) % len(metrics.REGISTERED_FAMILIES)
        or len(validation) % len(metrics.REGISTERED_FAMILIES)
        or any(
            count != expected_train_family_rows
            for count in train_family_rows.values()
        )
        or any(
            count != expected_validation_family_rows
            for count in validation_family_rows.values()
        )
    ):
        raise MetadataPreflightError("H6 role is not exactly family-balanced")
    if dict(train_family_scenes) != dict(policy.train_scene_counts):
        raise MetadataPreflightError("train family-scene inventory changed")
    if dict(validation_family_scenes) != dict(policy.validation_scene_counts):
        raise MetadataPreflightError("validation family-scene inventory changed")

    train_scenes = {row.scene_id for row in train}
    validation_scenes = {row.scene_id for row in validation}
    train_rgb = {leaf for row in train for leaf in row.rgb}
    validation_rgb = {leaf for row in validation for leaf in row.rgb}
    if not train_scenes.isdisjoint(validation_scenes):
        raise MetadataPreflightError("train and validation scenes overlap")
    if not train_rgb.isdisjoint(validation_rgb):
        raise MetadataPreflightError("train and validation RGB leaves overlap")

    schedule = metrics.build_training_schedule(
        train, rows_per_family=policy.train_rows_per_family
    )
    schedule_sha256 = metrics.canonical_json_sha256(schedule)
    if schedule_sha256 != policy.train_schedule_sha256:
        raise MetadataPreflightError("training schedule identity changed")
    coverage = _visible_action_coverage(train, schedule)
    if policy.require_all_visible_actions and any(
        values != list(range(metrics.ACTION_COUNT)) for values in coverage.values()
    ):
        raise MetadataPreflightError("training visible-action coverage changed")

    sentinel = metrics.build_sentinel_indices(
        validation, rows_per_family=policy.sentinel_rows_per_family
    )
    sentinel_sha256 = metrics.canonical_json_sha256(sentinel)
    if sentinel_sha256 != policy.sentinel_indices_sha256:
        raise MetadataPreflightError("sentinel identity changed")
    sentinel_scene_count = len({validation[index].scene_id for index in sentinel})
    if sentinel_scene_count != policy.sentinel_scene_count:
        raise MetadataPreflightError("sentinel scene count changed")

    full_donors = metrics.build_wrong_history_donor_indices(validation)
    sentinel_donors = metrics.build_wrong_history_donor_indices(
        validation, selected_indices=sentinel
    )
    full_donor_sha256 = metrics.canonical_json_sha256(full_donors)
    sentinel_donor_sha256 = metrics.canonical_json_sha256(sentinel_donors)
    if full_donor_sha256 != policy.full_donor_sha256:
        raise MetadataPreflightError("full wrong-history donors changed")
    if sentinel_donor_sha256 != policy.sentinel_donor_sha256:
        raise MetadataPreflightError("sentinel wrong-history donors changed")

    full_wrong_action = metrics.wrong_action_eligible_indices(validation)
    sentinel_wrong_action = metrics.wrong_action_eligible_indices(
        validation, selected_indices=sentinel
    )
    full_action_rows, full_action_scenes = metrics.family_row_scene_counts(
        validation, full_wrong_action
    )
    sentinel_action_rows, sentinel_action_scenes = metrics.family_row_scene_counts(
        validation, sentinel_wrong_action
    )
    if (
        full_action_rows != dict(policy.full_wrong_action_row_counts)
        or full_action_scenes != dict(policy.full_wrong_action_scene_counts)
        or sentinel_action_rows != dict(policy.sentinel_wrong_action_row_counts)
        or sentinel_action_scenes
        != dict(policy.sentinel_wrong_action_scene_counts)
    ):
        raise MetadataPreflightError("wrong-action eligibility changed")

    full_panel_identity = metrics.validation_panel_identity(
        validation,
        tuple(range(len(validation))),
        full_donors,
        full_wrong_action,
    )
    sentinel_panel_identity = metrics.validation_panel_identity(
        validation,
        sentinel,
        sentinel_donors,
        sentinel_wrong_action,
    )
    checks = {
        "bound_train_index": True,
        "bound_validation_index": True,
        "strict_canonical_jsonl": True,
        "train_validation_scenes_disjoint": True,
        "train_validation_rgb_leaves_disjoint": True,
        "training_schedule_exact": True,
        "sentinel_exact": True,
        "wrong_history_donors_exact": True,
        "wrong_action_eligibility_exact": True,
        "visible_action_coverage_exact": True,
        "rgb_open_count_zero": True,
        "checkpoint_open_count_zero": True,
        "navigation_heldout_sealed_open_count_zero": True,
    }
    return _with_content_sha256(
        {
            "schema": RECEIPT_SCHEMA,
            "status": "PASS_METADATA_PREFLIGHT",
            "preregistration_commit": metrics.PREREGISTRATION_COMMIT,
            "authority": dict(authority.binding),
            "reservation": {
                "schema": reserved["schema"],
                "status": reserved["status"],
                "content_sha256": reserved["content_sha256"],
                "created_utc": reserved["created_utc"],
            },
            "inputs": {
                "train": asdict(policy.train_binding),
                "validation": asdict(policy.validation_binding),
            },
            "train": {
                "row_count": len(train),
                "scene_count": len(train_scenes),
                "family_rows": train_family_rows,
                "family_scenes": train_family_scenes,
                "schedule_row_count": len(schedule),
                "schedule_indices_sha256": schedule_sha256,
                "visible_action_coverage": coverage,
            },
            "validation": {
                "row_count": len(validation),
                "scene_count": len(validation_scenes),
                "family_rows": validation_family_rows,
                "family_scenes": validation_family_scenes,
                "sentinel_row_count": len(sentinel),
                "sentinel_scene_count": sentinel_scene_count,
                "sentinel_indices_sha256": sentinel_sha256,
                "full_wrong_history_donors_sha256": full_donor_sha256,
                "sentinel_wrong_history_donors_sha256": sentinel_donor_sha256,
                "full_wrong_action_row_counts": full_action_rows,
                "full_wrong_action_scene_counts": full_action_scenes,
                "sentinel_wrong_action_row_counts": sentinel_action_rows,
                "sentinel_wrong_action_scene_counts": sentinel_action_scenes,
                "full_panel_identity_sha256": full_panel_identity,
                "sentinel_panel_identity_sha256": sentinel_panel_identity,
            },
            "access": {
                "metadata_index_open_count": metadata_index_open_count,
                "metadata_row_count": len(train) + len(validation),
                "rgb_path_string_count": 7 * (len(train) + len(validation)),
                "action_id_count": 6 * (len(train) + len(validation)),
                "rgb_open_count": 0,
                "checkpoint_open_count": 0,
                "navigation_open_count": 0,
                "held_out_or_sealed_opened": False,
            },
            "checks": checks,
        }
    )


def _read_bound_relative_file(repository_root: Path, binding: IndexBinding) -> bytes:
    root = Path(repository_root)
    if not root.is_absolute():
        raise MetadataPreflightError("repository root must be absolute")
    path = PurePosixPath(binding.path)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MetadataPreflightError("bound index path is not canonical relative")
    directory_fd = os.open(root, _DIR_FLAGS)
    file_fd: int | None = None
    try:
        for component in path.parts[:-1]:
            child_fd = os.open(component, _DIR_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        file_fd = os.open(path.parts[-1], _READ_FLAGS, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise MetadataPreflightError("bound index is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        raw = b"".join(chunks)
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
        ):
            raise MetadataPreflightError("bound index changed while open")
        _binding_check(raw, binding)
        return raw
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def _read_relative_file(repository_root: Path, relative_path: str) -> bytes:
    root = Path(repository_root)
    path = PurePosixPath(relative_path)
    if (
        not root.is_absolute()
        or type(relative_path) is not str
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise MetadataPreflightError("relative metadata source path changed")
    directory_fd = os.open(root, _DIR_FLAGS)
    file_fd: int | None = None
    try:
        for component in path.parts[:-1]:
            child_fd = os.open(component, _DIR_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = child_fd
        file_fd = os.open(path.parts[-1], _READ_FLAGS, dir_fd=directory_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise MetadataPreflightError("metadata source is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_fd)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise MetadataPreflightError("metadata source changed while open")
        return b"".join(chunks)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory_fd)


def _publish_receipt(path: Path, receipt: Mapping[str, Any]) -> None:
    output = Path(path)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise MetadataPreflightError("receipt output must be an absent absolute path")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise MetadataPreflightError(
            "receipt parent must be an existing real directory"
        )
    raw = _canonical_json_bytes(dict(receipt)) + b"\n"
    descriptor = os.open(
        output,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o444,
    )
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise MetadataPreflightError("receipt write did not make progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(output, 0o444, follow_symlinks=False)


def reserve_attempt_root(
    repository_root: Path,
    authority: ValidatedAuthority,
    *,
    created_utc: str,
) -> tuple[Path, dict[str, Any]]:
    """Atomically reserve the authority-selected initially absent attempt root."""

    root = Path(repository_root).resolve(strict=True)
    if (
        not isinstance(authority, ValidatedAuthority)
        or root.as_posix() != authority.repository_root
    ):
        raise MetadataPreflightError("authority repository root changed")
    relative = PurePosixPath(authority.output_root)
    output = root.joinpath(*relative.parts)
    if output.exists() or output.is_symlink():
        raise MetadataPreflightError("authorized attempt root is not absent")
    parent = output.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as error:
        raise MetadataPreflightError(
            "authorized attempt parent escaped repository root"
        ) from error
    parent_fd = os.open(parent, _DIR_FLAGS)
    try:
        os.mkdir(output.name, 0o700, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    reservation = build_reservation(authority, created_utc=created_utc)
    return output, reservation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--created-utc", required=True)
    arguments = parser.parse_args(argv)
    repository_root = arguments.repository_root.resolve(strict=True)
    authority_raw = _read_relative_file(repository_root, arguments.authority)
    authority = validate_authority_bytes(
        authority_raw,
        authority_path=arguments.authority,
    )
    if authority.repository_root != repository_root.as_posix():
        raise MetadataPreflightError("CLI and authority repository roots disagree")
    output, reservation = reserve_attempt_root(
        repository_root,
        authority,
        created_utc=arguments.created_utc,
    )
    try:
        train_raw = _read_bound_relative_file(
            repository_root, FROZEN_POLICY.train_binding
        )
        validation_raw = _read_bound_relative_file(
            repository_root, FROZEN_POLICY.validation_binding
        )
        receipt = preflight_from_bytes(
            train_raw=train_raw,
            validation_raw=validation_raw,
            authority=authority,
            reservation=reservation,
            metadata_index_open_count=2,
        )
        _publish_receipt(output / "receipt.json", receipt)
    except BaseException as error:
        failure = _with_content_sha256(
            {
                "schema": f"{metrics.SCHEMA_PREFIX}_metadata_preflight_failure_v1",
                "status": "FAIL_METADATA_PREFLIGHT_ATTEMPT_CONSUMED",
                "preregistration_commit": metrics.PREREGISTRATION_COMMIT,
                "authority": dict(authority.binding),
                "reservation_content_sha256": reservation["content_sha256"],
                "error_type": type(error).__name__,
                "rgb_open_count": 0,
                "checkpoint_open_count": 0,
                "held_out_or_sealed_opened": False,
            }
        )
        _publish_receipt(output / "failure.json", failure)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
