#!/usr/bin/env python3
"""Build the metadata-only memory-role place-triplet index V1.

Only the frozen raw-supervision endpoint index and exact derived-label JSONL
files for train/checkpoint-selection scenes are read.  Endpoint roles are
filtered before any referenced path is inspected.  RGB files are never opened.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks.go2_recurrent_jepa_main_pool_census import FAMILIES  # noqa: E402
from lewm.datasets.go2_memory_role_place_triplets_v1 import (  # noqa: E402
    ALLOWED_ROLES,
    MANIFEST_SCHEMA,
    RECEIPT_SCHEMA,
    SCHEMA,
    PlaceTripletContractError,
    canonical_json_bytes,
    canonical_json_sha256,
)


RAW_ROOT = Path(
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
RAW_MANIFEST_PATH = RAW_ROOT / "manifest.json"
RAW_MANIFEST_BYTES = 311_598
RAW_MANIFEST_SHA256 = "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360"
RAW_MANIFEST_CONTENT_SHA256 = (
    "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a"
)
RAW_MANIFEST_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_dataset_v1"
RAW_ENDPOINT_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_endpoint_index_v1"
RAW_ENDPOINT_ROWS = 9_460
EXPECTED_ENDPOINT_COUNTS = {"train": 7_777, "checkpoint_selection": 924}
EXPECTED_SCENE_COUNTS = {"train": 72, "checkpoint_selection": 8}
EXCLUDED_ROLE = "probability_calibration"
EXPECTED_EXCLUDED_ENDPOINTS = 759

DEFAULT_OUTPUT = Path(".generated/go2_memory_role_place_triplet_index_v1")
SELECTION_SEED = "lewm_go2_memory_role_place_triplet_v1"
POSITIVE_MIN_TIME_NS = 4_000_000_000
EXPECTED_YAW_BIN_COUNT = 8
TARGET_ROWS = {"train": 3_200, "checkpoint_selection": 320}
MEASURED_CANDIDATE_CAPACITIES = {
    "train": {
        "large_enclosed_maze": 242,
        "local_composite_motifs": 456,
        "loop_alias_stress": 325,
        "medium_enclosed_maze": 323,
        "open_obstacle_field": 755,
        "rough_local_dynamics": 734,
        "small_enclosed_maze": 387,
        "visual_sensor_stress": 412,
    },
    "checkpoint_selection": {
        "large_enclosed_maze": 39,
        "local_composite_motifs": 59,
        "loop_alias_stress": 34,
        "medium_enclosed_maze": 33,
        "open_obstacle_field": 77,
        "rough_local_dynamics": 79,
        "small_enclosed_maze": 20,
        "visual_sensor_stress": 30,
    },
}
DEFAULT_FAMILY_QUOTAS = {
    "train": {
        "large_enclosed_maze": 242,
        "local_composite_motifs": 456,
        "loop_alias_stress": 325,
        "medium_enclosed_maze": 323,
        "open_obstacle_field": 528,
        "rough_local_dynamics": 527,
        "small_enclosed_maze": 387,
        "visual_sensor_stress": 412,
    },
    "checkpoint_selection": {
        "large_enclosed_maze": 32,
        "local_composite_motifs": 48,
        "loop_alias_stress": 32,
        "medium_enclosed_maze": 32,
        "open_obstacle_field": 64,
        "rough_local_dynamics": 64,
        "small_enclosed_maze": 20,
        "visual_sensor_stress": 28,
    },
}

_RAW_ENDPOINT_FIELDS = frozenset(
    {
        "schema",
        "dataset_role",
        "family",
        "scene_id",
        "endpoint_identity_sha256",
        "plan_endpoint_content_sha256",
        "shard_row",
        "image_path_metadata_only",
        "image_sha256_commitment_only",
        "evidence_content_sha256",
        "raster_content_sha256",
        "content_sha256",
        "scene_shard",
    }
)
_FRAME_NAME = re.compile(r"^frame_([0-9]{6})_env_([0-9]{2})\.png$")
_CHUNK_NAME = re.compile(r"^(?:chunk_[0-9]{4}|chunk_backfill)$")
_SHA_FIELDS = (
    "endpoint_identity_sha256",
    "plan_endpoint_content_sha256",
    "image_sha256_commitment_only",
    "evidence_content_sha256",
    "raster_content_sha256",
    "content_sha256",
)
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NONBLOCK", 0)
)


@dataclass(frozen=True, slots=True)
class EndpointMetadata:
    role: str
    family: str
    scene_id: str
    endpoint_identity_sha256: str
    image_sha256: str
    rgb_path: str
    label_path: str
    frame_index: int
    env_index: int


@dataclass(frozen=True, slots=True)
class JoinedEndpoint:
    role: str
    family: str
    scene_id: str
    endpoint_identity_sha256: str
    image_sha256: str
    rgb_path: str
    frame_index: int
    env_index: int
    episode_id: str
    episode_step: int
    timestamp_ns: int
    cell_id: int
    yaw_bin: int


@dataclass(frozen=True, slots=True)
class TripletCandidate:
    anchor: JoinedEndpoint
    positive: JoinedEndpoint
    negative: JoinedEndpoint

    def to_row(self) -> dict[str, Any]:
        def rgb(endpoint: JoinedEndpoint) -> dict[str, str]:
            return {
                "endpoint_identity_sha256": endpoint.endpoint_identity_sha256,
                "rgb_path": endpoint.rgb_path,
                "image_sha256": endpoint.image_sha256,
            }

        def proof(endpoint: JoinedEndpoint) -> dict[str, Any]:
            return {
                "cell_id": endpoint.cell_id,
                "yaw_bin": endpoint.yaw_bin,
                "env_index": endpoint.env_index,
                "episode_id": endpoint.episode_id,
                "timestamp_ns": endpoint.timestamp_ns,
            }

        different_stream = (
            self.anchor.env_index,
            self.anchor.episode_id,
        ) != (self.positive.env_index, self.positive.episode_id)
        core = {
            "schema": SCHEMA,
            "role": self.anchor.role,
            "family": self.anchor.family,
            "scene_id": self.anchor.scene_id,
            "anchor": rgb(self.anchor),
            "positive": rgb(self.positive),
            "negative": rgb(self.negative),
            "selection_proof": {
                "anchor": proof(self.anchor),
                "positive": proof(self.positive),
                "negative": proof(self.negative),
                "positive_separation": (
                    "different_stream" if different_stream else "timestamp_gap_ge_4s"
                ),
            },
        }
        return {**core, "content_sha256": canonical_json_sha256(core)}


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_loads(raw: bytes, *, name: str) -> Any:
    def unique_object(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PlaceTripletContractError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                PlaceTripletContractError(f"{name} contains non-finite {value}")
            ),
        )
    except PlaceTripletContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PlaceTripletContractError(f"{name} is invalid UTF-8 JSON") from error


def _read_regular_file(path: Path) -> bytes:
    descriptor = os.open(path, _READ_FLAGS)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PlaceTripletContractError(f"not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise PlaceTripletContractError(f"file changed while reading: {path}")
    return b"".join(chunks)


def _exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or int(value) < minimum:
        raise PlaceTripletContractError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _exact_text(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise PlaceTripletContractError(f"{name} must be a nonempty string")
    return str(value)


def _parse_rgb_location(
    repo_root: Path,
    value: object,
    *,
    family: str,
    scene_id: str,
) -> tuple[str, int, int]:
    del family, scene_id
    text = _exact_text(value, name="image_path_metadata_only")
    path = Path(text)
    if not path.is_absolute() or os.path.normpath(text) != text or ".." in path.parts:
        raise PlaceTripletContractError("endpoint RGB metadata path is not canonical absolute")
    try:
        relative = path.relative_to(repo_root)
    except ValueError as error:
        raise PlaceTripletContractError("endpoint RGB metadata path escapes repository") from error
    parts = relative.parts
    if (
        len(parts) != 6
        or parts[:3]
        != (".generated", "go2_render_selected_v04", "scenes")
        or re.fullmatch(r"scene_[0-9a-f]{16}", parts[3]) is None
        or parts[4] != "rgb"
    ):
        raise PlaceTripletContractError("endpoint RGB metadata path left source layout")
    match = _FRAME_NAME.fullmatch(parts[5])
    if match is None:
        raise PlaceTripletContractError("endpoint RGB leaf name changed")
    frame_index, env_index = map(int, match.groups())
    return relative.as_posix(), frame_index, env_index


def _label_path_from_source_frames_inventory(
    repo_root: Path,
    value: object,
    *,
    family: str,
    scene_id: str,
) -> str:
    text = _exact_text(value, name="source_frames_jsonl path")
    path = Path(text)
    if not path.is_absolute() or os.path.normpath(text) != text or ".." in path.parts:
        raise PlaceTripletContractError("source-frames path is not canonical absolute")
    try:
        relative = path.relative_to(repo_root)
    except ValueError as error:
        raise PlaceTripletContractError("source-frames path escapes repository") from error
    parts = relative.parts
    expected_sequence = re.compile(rf"^[0-9]{{6}}_{re.escape(scene_id)}$")
    if (
        len(parts) != 9
        or parts[:3] != (".generated", "datagen_full", "rollout")
        or parts[3] not in {"train", "val", "test_id", "test_hard"}
        or parts[4] != family
        or _CHUNK_NAME.fullmatch(parts[5]) is None
        or parts[6] != "plan"
        or expected_sequence.fullmatch(parts[7]) is None
        or parts[8] != "frames.jsonl"
    ):
        raise PlaceTripletContractError("source-frames path left source layout")
    return Path(*parts[:6], "labels", scene_id, "labels.jsonl").as_posix()


def _decode_allowed_endpoint(
    repo_root: Path,
    value: Mapping[str, Any],
    *,
    role: str,
) -> EndpointMetadata:
    if set(value) != _RAW_ENDPOINT_FIELDS:
        raise PlaceTripletContractError("allowed raw endpoint fields changed")
    if value.get("schema") != RAW_ENDPOINT_SCHEMA or value.get("dataset_role") != role:
        raise PlaceTripletContractError("allowed raw endpoint schema or role changed")
    if any(not _is_sha256(value.get(field)) for field in _SHA_FIELDS):
        raise PlaceTripletContractError("allowed raw endpoint contains an invalid hash")
    core = dict(value)
    declared = core.pop("content_sha256")
    if canonical_json_sha256(core) != declared:
        raise PlaceTripletContractError("allowed raw endpoint content hash changed")
    family = _exact_text(value.get("family"), name="endpoint family")
    scene_id = _exact_text(value.get("scene_id"), name="endpoint scene_id")
    if family not in FAMILIES:
        raise PlaceTripletContractError("endpoint family left the eight-family contract")
    _exact_int(value.get("shard_row"), name="endpoint shard_row")
    shard = PurePosixPath(_exact_text(value.get("scene_shard"), name="scene_shard"))
    if shard.is_absolute() or len(shard.parts) != 3 or shard.parts[0] != "shards":
        raise PlaceTripletContractError("endpoint shard metadata path changed")
    rgb_path, frame_index, env_index = _parse_rgb_location(
        repo_root,
        value.get("image_path_metadata_only"),
        family=family,
        scene_id=scene_id,
    )
    return EndpointMetadata(
        role=role,
        family=family,
        scene_id=scene_id,
        endpoint_identity_sha256=str(value["endpoint_identity_sha256"]),
        image_sha256=str(value["image_sha256_commitment_only"]),
        rgb_path=rgb_path,
        label_path="",
        frame_index=frame_index,
        env_index=env_index,
    )


def decode_allowed_endpoint_bytes(
    repo_root: Path,
    raw: bytes,
    *,
    enforce_frozen_counts: bool,
) -> tuple[tuple[EndpointMetadata, ...], dict[str, Any]]:
    """Filter by role before validating or resolving any endpoint path."""

    if not raw or not raw.endswith(b"\n"):
        raise PlaceTripletContractError("raw endpoint index is not newline JSONL")
    allowed: list[EndpointMetadata] = []
    excluded = 0
    total = 0
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise PlaceTripletContractError("raw endpoint index contains a blank row")
        value = _strict_json_loads(line, name=f"raw endpoint line {line_number}")
        if type(value) is not dict:
            raise PlaceTripletContractError("raw endpoint row is not an object")
        total += 1
        role = value.get("dataset_role")
        if role == EXCLUDED_ROLE:
            excluded += 1
            continue
        if role not in ALLOWED_ROLES:
            raise PermissionError("raw endpoint row has a non-allowed role")
        allowed.append(_decode_allowed_endpoint(repo_root, value, role=str(role)))

    counts = Counter(endpoint.role for endpoint in allowed)
    scenes = {
        role: {endpoint.scene_id for endpoint in allowed if endpoint.role == role}
        for role in ALLOWED_ROLES
    }
    if len({endpoint.endpoint_identity_sha256 for endpoint in allowed}) != len(allowed):
        raise PlaceTripletContractError("allowed endpoint identity is duplicated")
    if len({endpoint.rgb_path for endpoint in allowed}) != len(allowed):
        raise PlaceTripletContractError("allowed endpoint RGB path is duplicated")
    if scenes["train"] & scenes["checkpoint_selection"]:
        raise PlaceTripletContractError("train/checkpoint scenes overlap")
    if enforce_frozen_counts and (
        total != RAW_ENDPOINT_ROWS
        or excluded != EXPECTED_EXCLUDED_ENDPOINTS
        or dict(counts) != EXPECTED_ENDPOINT_COUNTS
        or {role: len(value) for role, value in scenes.items()} != EXPECTED_SCENE_COUNTS
    ):
        raise PlaceTripletContractError("frozen endpoint role population changed")
    return tuple(allowed), {
        "endpoint_rows_read": total,
        "allowed_endpoint_counts": {role: counts[role] for role in ALLOWED_ROLES},
        "allowed_scene_counts": {role: len(scenes[role]) for role in ALLOWED_ROLES},
        "excluded_probability_calibration_endpoint_count": excluded,
        "excluded_role_referenced_path_dereference_count": 0,
    }


def bind_allowed_label_paths(
    repo_root: Path,
    endpoints: Sequence[EndpointMetadata],
    raw_manifest: Mapping[str, Any],
) -> tuple[tuple[EndpointMetadata, ...], dict[str, Any]]:
    """Bind each allowed scene to its original derived-label JSONL."""

    allowed = {
        (endpoint.role, endpoint.family, endpoint.scene_id) for endpoint in endpoints
    }
    allowed_by_scene: dict[str, tuple[str, str, str]] = {}
    for key in allowed:
        scene_id = key[2]
        if scene_id in allowed_by_scene and allowed_by_scene[scene_id] != key:
            raise PlaceTripletContractError("allowed scene crosses role or family")
        allowed_by_scene[scene_id] = key
    provenance = raw_manifest.get("input_provenance")
    inventory = (
        provenance.get("source_payload_inventory")
        if isinstance(provenance, Mapping)
        else None
    )
    if type(inventory) is not list:
        raise PlaceTripletContractError("source payload inventory is unavailable")

    labels: dict[tuple[str, str, str], str] = {}
    unselected = 0
    for record in inventory:
        if type(record) is not dict:
            raise PlaceTripletContractError("source payload inventory row is not an object")
        scene_id = record.get("scene_id")
        key = allowed_by_scene.get(scene_id) if isinstance(scene_id, str) else None
        if key is None:
            # The inventory does not publish a role field.  Scene membership is
            # therefore the fail-closed boundary: do not inspect purpose/path
            # for calibration or any other unselected scene.
            unselected += 1
            continue
        if record.get("purpose") != "source_frames_jsonl":
            continue
        if key in labels:
            raise PlaceTripletContractError("allowed scene repeats source_frames_jsonl")
        labels[key] = _label_path_from_source_frames_inventory(
            repo_root,
            record.get("path"),
            family=key[1],
            scene_id=key[2],
        )

    if set(labels) != allowed:
        raise PlaceTripletContractError(
            "allowed scene does not have exactly one source_frames_jsonl binding"
        )
    return (
        tuple(
            replace(
                endpoint,
                label_path=labels[(endpoint.role, endpoint.family, endpoint.scene_id)],
            )
            for endpoint in endpoints
        ),
        {
            "allowed_scene_source_frames_binding_count": len(labels),
            "unselected_scene_inventory_record_count": unselected,
            "excluded_role_referenced_path_dereference_count": 0,
        },
    )
def _parent_endpoint_bytes(
    repo_root: Path,
    raw_root: Path,
    *,
    enforce_frozen_binding: bool,
) -> tuple[bytes, dict[str, Any], Mapping[str, Any]]:
    directory = raw_root if raw_root.is_absolute() else repo_root / raw_root
    manifest_raw = _read_regular_file(directory / "manifest.json")
    manifest_file_sha = hashlib.sha256(manifest_raw).hexdigest()
    if enforce_frozen_binding and (
        len(manifest_raw) != RAW_MANIFEST_BYTES
        or manifest_file_sha != RAW_MANIFEST_SHA256
    ):
        raise PlaceTripletContractError("frozen raw-supervision manifest changed")
    manifest = _strict_json_loads(manifest_raw, name="raw-supervision manifest")
    if type(manifest) is not dict or manifest.get("schema") != RAW_MANIFEST_SCHEMA:
        raise PlaceTripletContractError("raw-supervision manifest schema changed")
    core = dict(manifest)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PlaceTripletContractError("raw-supervision manifest content hash changed")
    if enforce_frozen_binding and declared != RAW_MANIFEST_CONTENT_SHA256:
        raise PlaceTripletContractError("raw-supervision manifest content binding changed")
    binding = manifest.get("endpoint_index")
    if type(binding) is not dict or binding.get("path") != "endpoints.jsonl":
        raise PlaceTripletContractError("raw endpoint-index binding changed")
    row_count = _exact_int(binding.get("row_count"), name="endpoint row_count")
    expected_sha = binding.get("file_sha256")
    if not _is_sha256(expected_sha):
        raise PlaceTripletContractError("raw endpoint-index SHA-256 is invalid")
    endpoint_raw = _read_regular_file(directory / "endpoints.jsonl")
    if hashlib.sha256(endpoint_raw).hexdigest() != expected_sha:
        raise PlaceTripletContractError("raw endpoint-index bytes changed")
    if len(endpoint_raw.splitlines()) != row_count:
        raise PlaceTripletContractError("raw endpoint-index row count changed")
    return endpoint_raw, {
        "manifest_path": (raw_root / "manifest.json").as_posix(),
        "manifest_byte_count": len(manifest_raw),
        "manifest_file_sha256": manifest_file_sha,
        "manifest_content_sha256": str(declared),
        "endpoint_index_path": (raw_root / "endpoints.jsonl").as_posix(),
        "endpoint_index_byte_count": len(endpoint_raw),
        "endpoint_index_row_count": row_count,
        "endpoint_index_file_sha256": str(expected_sha),
    }, manifest


def _label_value(value: object, *, name: str, minimum: int = 0) -> int:
    return _exact_int(value, name=name, minimum=minimum)


def _join_one_label(
    endpoint: EndpointMetadata,
    label: Mapping[str, Any],
) -> JoinedEndpoint:
    episode_number = _label_value(label.get("episode_id"), name="label episode_id")
    episode_id = str(episode_number)
    env_index = _label_value(label.get("env_idx"), name="label env_idx")
    episode_step = _label_value(label.get("episode_step"), name="label episode_step")
    timestamp_ns = _label_value(label.get("timestamp_ns"), name="label timestamp_ns")
    cell_id = _label_value(label.get("cell_id"), name="label cell_id")
    yaw_bin = _label_value(label.get("yaw_bin"), name="label yaw_bin")
    yaw_bin_count = _label_value(label.get("yaw_bin_count"), name="label yaw_bin_count", minimum=1)
    if (
        label.get("scene_id") != endpoint.scene_id
        or env_index != endpoint.env_index
        or yaw_bin_count != EXPECTED_YAW_BIN_COUNT
        or yaw_bin >= yaw_bin_count
    ):
        raise PlaceTripletContractError("endpoint and derived-label row identity changed")
    identity = {
        "dataset_role": endpoint.role,
        "scene_id": endpoint.scene_id,
        "episode_id": episode_id,
        "env_index": env_index,
        "episode_step": episode_step,
        "frame_index": endpoint.frame_index,
        "timestamp_ns": timestamp_ns,
        "image_sha256": endpoint.image_sha256,
    }
    if canonical_json_sha256(identity) != endpoint.endpoint_identity_sha256:
        raise PlaceTripletContractError("derived label does not reproduce endpoint identity")
    return JoinedEndpoint(
        role=endpoint.role,
        family=endpoint.family,
        scene_id=endpoint.scene_id,
        endpoint_identity_sha256=endpoint.endpoint_identity_sha256,
        image_sha256=endpoint.image_sha256,
        rgb_path=endpoint.rgb_path,
        frame_index=endpoint.frame_index,
        env_index=env_index,
        episode_id=episode_id,
        episode_step=episode_step,
        timestamp_ns=timestamp_ns,
        cell_id=cell_id,
        yaw_bin=yaw_bin,
    )


def join_exact_derived_labels(
    repo_root: Path,
    endpoints: Sequence[EndpointMetadata],
) -> tuple[tuple[JoinedEndpoint, ...], tuple[dict[str, Any], ...]]:
    """Join allowed endpoints to direct row-index labels, without RGB access."""

    grouped: dict[str, list[EndpointMetadata]] = defaultdict(list)
    for endpoint in endpoints:
        if endpoint.role not in ALLOWED_ROLES:
            raise PermissionError("non-allowed endpoint reached label dereference")
        grouped[endpoint.label_path].append(endpoint)
    joined: list[JoinedEndpoint] = []
    receipts: list[dict[str, Any]] = []
    for relative in sorted(grouped):
        selected = grouped[relative]
        roles = {endpoint.role for endpoint in selected}
        scenes = {endpoint.scene_id for endpoint in selected}
        families = {endpoint.family for endpoint in selected}
        if len(roles) != 1 or len(scenes) != 1 or len(families) != 1:
            raise PlaceTripletContractError("one derived-label file crossed identities")
        wanted = {endpoint.frame_index: endpoint for endpoint in selected}
        if len(wanted) != len(selected):
            raise PlaceTripletContractError("label join repeats a frame index")
        path = repo_root / relative
        descriptor = os.open(path, _READ_FLAGS)
        digest = hashlib.sha256()
        byte_count = 0
        row_count = 0
        found: dict[int, JoinedEndpoint] = {}
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise PlaceTripletContractError("derived-label leaf is not regular")
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                for row_index, raw_line in enumerate(stream):
                    if not raw_line.endswith(b"\n"):
                        raise PlaceTripletContractError("derived-label JSONL is not newline terminated")
                    digest.update(raw_line)
                    byte_count += len(raw_line)
                    row_count += 1
                    endpoint = wanted.get(row_index)
                    if endpoint is None:
                        continue
                    value = _strict_json_loads(raw_line, name=f"derived label {relative}:{row_index}")
                    if type(value) is not dict:
                        raise PlaceTripletContractError("derived-label row is not an object")
                    found[row_index] = _join_one_label(endpoint, value)
                after = os.fstat(stream.fileno())
        except BaseException:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        if (
            set(found) != set(wanted)
            or (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or byte_count != before.st_size
        ):
            raise PlaceTripletContractError("derived-label join is incomplete or changed")
        joined.extend(found[index] for index in sorted(found))
        receipts.append(
            {
                "path": relative,
                "role": next(iter(roles)),
                "family": next(iter(families)),
                "scene_id": next(iter(scenes)),
                "row_count": row_count,
                "selected_endpoint_count": len(selected),
                "byte_count": byte_count,
                "sha256": digest.hexdigest(),
            }
        )
    if len(joined) != len(endpoints):
        raise PlaceTripletContractError("not every allowed endpoint joined exactly once")
    return tuple(joined), tuple(receipts)


def _donor_score(anchor: JoinedEndpoint, donor: JoinedEndpoint, kind: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        (
            SELECTION_SEED
            + "\0"
            + anchor.role
            + "\0"
            + anchor.endpoint_identity_sha256
            + "\0"
            + kind
            + "\0"
            + donor.endpoint_identity_sha256
        ).encode("ascii")
    ).hexdigest()
    return digest, donor.endpoint_identity_sha256


def _anchor_score(endpoint: JoinedEndpoint) -> tuple[str, str]:
    return (
        hashlib.sha256(
            (SELECTION_SEED + "\0anchor\0" + endpoint.endpoint_identity_sha256).encode(
                "ascii"
            )
        ).hexdigest(),
        endpoint.endpoint_identity_sha256,
    )


def _positive_allowed(anchor: JoinedEndpoint, candidate: JoinedEndpoint) -> bool:
    return (
        candidate.endpoint_identity_sha256 != anchor.endpoint_identity_sha256
        and candidate.rgb_path != anchor.rgb_path
        and (
            (candidate.env_index, candidate.episode_id)
            != (anchor.env_index, anchor.episode_id)
            or abs(candidate.timestamp_ns - anchor.timestamp_ns)
            >= POSITIVE_MIN_TIME_NS
        )
    )


def _unique_positive_matches(
    group: Sequence[JoinedEndpoint],
) -> dict[str, JoinedEndpoint]:
    """Return a deterministic maximum matching with unique positive donors."""

    ordered = sorted(group, key=_anchor_score)
    edges = {
        anchor.endpoint_identity_sha256: sorted(
            (candidate for candidate in ordered if _positive_allowed(anchor, candidate)),
            key=lambda item: _donor_score(anchor, item, "positive"),
        )
        for anchor in ordered
    }
    donor_to_anchor: dict[str, str] = {}
    endpoint_by_identity = {
        endpoint.endpoint_identity_sha256: endpoint for endpoint in ordered
    }

    def augment(anchor_identity: str, seen: set[str]) -> bool:
        for donor in edges[anchor_identity]:
            donor_identity = donor.endpoint_identity_sha256
            if donor_identity in seen:
                continue
            seen.add(donor_identity)
            previous = donor_to_anchor.get(donor_identity)
            if previous is None or augment(previous, seen):
                donor_to_anchor[donor_identity] = anchor_identity
                return True
        return False

    for anchor in ordered:
        augment(anchor.endpoint_identity_sha256, set())
    return {
        anchor_identity: endpoint_by_identity[donor_identity]
        for donor_identity, anchor_identity in donor_to_anchor.items()
    }


def construct_candidates(
    endpoints: Sequence[JoinedEndpoint],
) -> tuple[TripletCandidate, ...]:
    positives: dict[tuple[str, str, int, int], list[JoinedEndpoint]] = defaultdict(list)
    negatives: dict[tuple[str, str, int], list[JoinedEndpoint]] = defaultdict(list)
    for endpoint in endpoints:
        positives[(endpoint.role, endpoint.scene_id, endpoint.cell_id, endpoint.yaw_bin)].append(endpoint)
        negatives[(endpoint.role, endpoint.scene_id, endpoint.yaw_bin)].append(endpoint)
    positive_matches: dict[str, JoinedEndpoint] = {}
    for group in positives.values():
        positive_matches.update(_unique_positive_matches(group))
    candidates: list[TripletCandidate] = []
    for anchor in endpoints:
        positive = positive_matches.get(anchor.endpoint_identity_sha256)
        negative_pool = [
            item
            for item in negatives[(anchor.role, anchor.scene_id, anchor.yaw_bin)]
            if item.cell_id != anchor.cell_id and item.rgb_path != anchor.rgb_path
        ]
        if positive is None or not negative_pool:
            continue
        candidates.append(
            TripletCandidate(
                anchor=anchor,
                positive=positive,
                negative=min(negative_pool, key=lambda item: _donor_score(anchor, item, "negative")),
            )
        )
    return tuple(candidates)


def _select_family_round_robin(
    candidates: Sequence[TripletCandidate],
    *,
    target: int,
) -> list[TripletCandidate]:
    by_scene: dict[str, list[TripletCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_scene[candidate.anchor.scene_id].append(candidate)
    for values in by_scene.values():
        values.sort(key=lambda item: _anchor_score(item.anchor))
    scene_order = sorted(
        by_scene,
        key=lambda scene: (hashlib.sha256((SELECTION_SEED + "\0scene\0" + scene).encode()).hexdigest(), scene),
    )
    selected: list[TripletCandidate] = []
    offsets = {scene: 0 for scene in scene_order}
    while len(selected) < target:
        progressed = False
        for scene in scene_order:
            offset = offsets[scene]
            if offset < len(by_scene[scene]):
                selected.append(by_scene[scene][offset])
                offsets[scene] += 1
                progressed = True
                if len(selected) == target:
                    break
        if not progressed:
            raise PlaceTripletContractError("candidate support exhausted before target")
    return selected


def _resolve_family_quotas(
    families: Sequence[str],
    targets: Mapping[str, int],
    family_quotas: Mapping[str, Mapping[str, int]] | None,
) -> tuple[dict[str, dict[str, int]], str]:
    ordered = tuple(sorted(families))
    use_measured_default = (
        family_quotas is None
        and ordered == tuple(sorted(FAMILIES))
        and dict(targets) == TARGET_ROWS
    )
    if use_measured_default:
        source = DEFAULT_FAMILY_QUOTAS
        schedule = "measured_support_v1"
    elif family_quotas is None:
        source = {}
        schedule = "equal_custom"
        for role in ALLOWED_ROLES:
            target = _exact_int(targets.get(role), name=f"{role} target", minimum=1)
            if target % len(ordered):
                raise PlaceTripletContractError("custom role target is not family-balanced")
            source[role] = {family: target // len(ordered) for family in ordered}
    else:
        source = family_quotas
        schedule = "explicit_custom"

    resolved: dict[str, dict[str, int]] = {}
    for role in ALLOWED_ROLES:
        role_quotas = source.get(role)
        if not isinstance(role_quotas, Mapping) or set(role_quotas) != set(ordered):
            raise PlaceTripletContractError(f"{role} family-quota keys changed")
        resolved[role] = {
            family: _exact_int(
                role_quotas.get(family),
                name=f"{role}/{family} quota",
                minimum=1,
            )
            for family in ordered
        }
        target = _exact_int(targets.get(role), name=f"{role} target", minimum=1)
        if sum(resolved[role].values()) != target:
            raise PlaceTripletContractError(f"{role} family quotas do not sum to target")
    return resolved, schedule


def select_balanced_triplets(
    endpoints: Sequence[JoinedEndpoint],
    *,
    families: Sequence[str] = FAMILIES,
    targets: Mapping[str, int] = TARGET_ROWS,
    family_quotas: Mapping[str, Mapping[str, int]] | None = None,
    require_all_checkpoint_scenes: bool = True,
) -> tuple[dict[str, tuple[TripletCandidate, ...]], dict[str, Any]]:
    candidates = construct_candidates(endpoints)
    selected: dict[str, tuple[TripletCandidate, ...]] = {}
    support: dict[str, Any] = {}
    ordered_families = tuple(sorted(families))
    quotas, quota_schedule = _resolve_family_quotas(
        ordered_families, targets, family_quotas
    )
    for role in ALLOWED_ROLES:
        target = _exact_int(targets.get(role), name=f"{role} target", minimum=1)
        chosen_by_family: dict[str, list[TripletCandidate]] = {}
        candidate_counts: dict[str, int] = {}
        for family in ordered_families:
            quota = quotas[role][family]
            pool = [
                item
                for item in candidates
                if item.anchor.role == role and item.anchor.family == family
            ]
            candidate_counts[family] = len(pool)
            if len(pool) < quota:
                raise PlaceTripletContractError(
                    f"{role}/{family} has {len(pool)} candidates; needs {quota}"
                )
            chosen_by_family[family] = _select_family_round_robin(pool, target=quota)
        if quota_schedule == "measured_support_v1" and candidate_counts != (
            MEASURED_CANDIDATE_CAPACITIES[role]
        ):
            raise PlaceTripletContractError(
                f"{role} measured candidate capacities changed"
            )
        interleaved = tuple(
            chosen_by_family[family][offset]
            for offset in range(max(quotas[role].values()))
            for family in ordered_families
            if offset < quotas[role][family]
        )
        if len(interleaved) != target:
            raise PlaceTripletContractError("family quotas did not produce role target")
        selected_scenes = {item.anchor.scene_id for item in interleaved}
        input_scenes = {item.scene_id for item in endpoints if item.role == role}
        candidate_scene_counts = Counter(
            item.anchor.scene_id for item in candidates if item.anchor.role == role
        )
        selected_scene_counts = Counter(item.anchor.scene_id for item in interleaved)
        anchors = [item.anchor.endpoint_identity_sha256 for item in interleaved]
        positives = [item.positive.endpoint_identity_sha256 for item in interleaved]
        positive_scene_keys = [
            (item.anchor.scene_id, item.positive.endpoint_identity_sha256)
            for item in interleaved
        ]
        if len(set(anchors)) != len(anchors) or len(set(positive_scene_keys)) != len(
            positive_scene_keys
        ):
            raise PlaceTripletContractError("selected anchors or scene-positive keys repeat")
        if role == "checkpoint_selection" and require_all_checkpoint_scenes and (
            selected_scenes != input_scenes
            or set(candidate_scene_counts) != input_scenes
            or any(selected_scene_counts[scene] < 1 for scene in input_scenes)
        ):
            raise PlaceTripletContractError("checkpoint-selection support gate failed")
        selected[role] = interleaved
        support[role] = {
            "input_endpoint_count": sum(item.role == role for item in endpoints),
            "candidate_anchor_count": sum(candidate_counts.values()),
            "candidate_anchor_counts_by_family": candidate_counts,
            "selected_row_count": len(interleaved),
            "quota_schedule": quota_schedule,
            "selected_row_quotas_by_family": dict(quotas[role]),
            "selected_rows_by_family": dict(
                sorted(Counter(item.anchor.family for item in interleaved).items())
            ),
            "input_scene_count": len(input_scenes),
            "selected_scene_count": len(selected_scenes),
            "selected_scene_ids_sha256": canonical_json_sha256(sorted(selected_scenes)),
            "candidate_usable_anchor_counts_by_scene": dict(
                sorted(candidate_scene_counts.items())
            ),
            "selected_anchor_counts_by_scene": dict(
                sorted(selected_scene_counts.items())
            ),
            "selected_anchor_identity_unique": True,
            "selected_positive_identity_unique_within_scene": True,
            "selected_unique_positive_identity_count": len(set(positives)),
        }
    return selected, support


def build_index(
    repo_root: Path,
    *,
    raw_root: Path = RAW_ROOT,
    enforce_frozen_binding: bool = True,
    enforce_frozen_counts: bool = True,
    families: Sequence[str] = FAMILIES,
    targets: Mapping[str, int] = TARGET_ROWS,
    family_quotas: Mapping[str, Mapping[str, int]] | None = None,
    require_all_checkpoint_scenes: bool = True,
) -> tuple[dict[str, tuple[TripletCandidate, ...]], dict[str, Any]]:
    root = Path(repo_root).resolve(strict=True)
    endpoint_raw, parent, raw_manifest = _parent_endpoint_bytes(
        root, raw_root, enforce_frozen_binding=enforce_frozen_binding
    )
    endpoints, endpoint_audit = decode_allowed_endpoint_bytes(
        root, endpoint_raw, enforce_frozen_counts=enforce_frozen_counts
    )
    endpoints, inventory_audit = bind_allowed_label_paths(
        root, endpoints, raw_manifest
    )
    joined, label_receipts = join_exact_derived_labels(root, endpoints)
    selected, support = select_balanced_triplets(
        joined,
        families=families,
        targets=targets,
        family_quotas=family_quotas,
        require_all_checkpoint_scenes=require_all_checkpoint_scenes,
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "PASS",
        "selection": {
            "seed": SELECTION_SEED,
            "positive": (
                "same_scene_cell_yaw; distinct endpoint/path; different env-episode "
                "stream or absolute timestamp gap >=4s"
            ),
            "negative": "same_scene_yaw_and_different_cell",
            "family_quota_round_robin": True,
            "positive_donors_unique_within_scene": True,
            "targets": dict(targets),
            "quota_schedule": support["train"]["quota_schedule"],
            "family_quotas": {
                role: support[role]["selected_row_quotas_by_family"]
                for role in ALLOWED_ROLES
            },
            "measured_candidate_capacities": (
                MEASURED_CANDIDATE_CAPACITIES
                if support["train"]["quota_schedule"] == "measured_support_v1"
                else None
            ),
            "privileged_cell_yaw_are_selection_only": True,
        },
        "support": support,
        "source": {
            "raw_supervision": parent,
            "endpoint_filter": endpoint_audit,
            "source_inventory_label_binding": inventory_audit,
            "derived_label_files": list(label_receipts),
            "derived_label_file_count": len(label_receipts),
            "derived_label_ordered_binding_sha256": canonical_json_sha256(label_receipts),
        },
        "access_ledger": {
            "raw_manifest_byte_opens": 1,
            "raw_endpoint_index_byte_opens": 1,
            "allowed_derived_label_byte_opens": len(label_receipts),
            "probability_calibration_referenced_path_dereferences": 0,
            "rgb_byte_opens": 0,
            "rgb_decodes": 0,
            "scene_manifest_opens": 0,
            "raw_frames_jsonl_opens": 0,
            "heldout_or_sealed_opens": 0,
            "checkpoint_or_runtime_output_opens": 0,
            "gpu_use_count": 0,
        },
    }
    return selected, manifest


def _artifact_payload(rows: Sequence[TripletCandidate]) -> bytes:
    return b"".join(canonical_json_bytes(row.to_row()) + b"\n" for row in rows)


def _write_fresh(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def publish_index(
    output_dir: Path,
    selected: Mapping[str, Sequence[TripletCandidate]],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"fresh output directory already exists: {output_dir}")
    payloads = {f"{role}.jsonl": _artifact_payload(selected[role]) for role in ALLOWED_ROLES}
    artifacts = {
        name: {
            "path": name,
            "row_count": len(selected[name.removesuffix(".jsonl")]),
            "byte_count": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in payloads.items()
    }
    manifest_core = {**dict(manifest), "artifacts": artifacts}
    manifest_value = {
        **manifest_core,
        "content_sha256": canonical_json_sha256(manifest_core),
    }
    manifest_payload = canonical_json_bytes(manifest_value) + b"\n"
    receipt_core = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS",
        "manifest_file_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "manifest_content_sha256": manifest_value["content_sha256"],
        "artifacts": artifacts,
        "rgb_open_count": 0,
        "probability_calibration_referenced_path_dereference_count": 0,
    }
    receipt = {**receipt_core, "content_sha256": canonical_json_sha256(receipt_core)}
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        _write_fresh(output_dir / name, payload)
    _write_fresh(output_dir / "manifest.json", manifest_payload)
    _write_fresh(output_dir / "receipt.json", canonical_json_bytes(receipt) + b"\n")
    return receipt


def _source_bindings(repo_root: Path) -> list[dict[str, Any]]:
    paths = (
        Path("scripts/build_go2_memory_role_place_triplet_index_v1.py"),
        Path("lewm/datasets/go2_memory_role_place_triplets_v1.py"),
    )
    result: list[dict[str, Any]] = []
    for relative in paths:
        payload = _read_regular_file(repo_root / relative)
        result.append(
            {
                "path": relative.as_posix(),
                "byte_count": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve(strict=True)
    output = args.output_dir
    if not output.is_absolute():
        output = repo_root / output
    source_before = _source_bindings(repo_root)
    selected, manifest = build_index(repo_root)
    source_after = _source_bindings(repo_root)
    if source_after != source_before:
        raise PlaceTripletContractError("builder source changed during metadata pass")
    manifest["public_source_bindings"] = source_after
    receipt = publish_index(output, selected, manifest)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EndpointMetadata",
    "JoinedEndpoint",
    "TripletCandidate",
    "build_index",
    "bind_allowed_label_paths",
    "construct_candidates",
    "decode_allowed_endpoint_bytes",
    "join_exact_derived_labels",
    "publish_index",
    "select_balanced_triplets",
]
