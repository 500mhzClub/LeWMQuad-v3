"""Metadata-only plan for shared-JEPA V5 raw V4 supervision.

This module deliberately stops before geometry or RGB access.  It proves the
pair/endpoint/attitude joins for the three development roles and emits exact
records that a separately reviewed raycast builder can consume.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping, Sequence

from lewm.datasets import go2_attitude_sidecar as attitude_sidecar


PLAN_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_plan_v1"
PAIR_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_pair_v1"
ENDPOINT_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_endpoint_v1"
DEVELOPMENT_ROLES = (
    "train",
    "checkpoint_selection",
    "probability_calibration",
)
ALL_ROLES = (*DEVELOPMENT_ROLES, "g2_evaluation")
ROLE_PAIR_COUNTS = {
    "train": 4262,
    "checkpoint_selection": 495,
    "probability_calibration": 415,
}
ROLE_SCENE_COUNTS = {
    "train": 72,
    "checkpoint_selection": 8,
    "probability_calibration": 8,
}
ROLE_UNIQUE_ENDPOINT_COUNTS = {
    "train": 7777,
    "checkpoint_selection": 924,
    "probability_calibration": 759,
}
FULL_ROLE_PAIR_COUNTS = {**ROLE_PAIR_COUNTS, "g2_evaluation": 469}
PRIMITIVE_VOCABULARY = (
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

DATASET_MANIFEST_RELATIVE_PATH = (
    ".generated/go2_paired_navigation/geometry_v3_physical_v1/"
    "dataset/dataset_manifest.json"
)
DATASET_ROWS_RELATIVE_PATH = (
    ".generated/go2_paired_navigation/geometry_v3_physical_v1/dataset/rows.jsonl"
)
SIDECAR_MANIFEST_RELATIVE_PATH = (
    ".generated/go2_attitude_sidecar/dynamic_cartesian_v1/manifest.json"
)
SOURCE_INDEX_RELATIVE_PATH = (
    ".generated/go2_paired_navigation/geometry_v3_physical_v1/"
    "source_index/go2_navigation_sources_v04.jsonl"
)
DATASET_MANIFEST_FILE_SHA256 = (
    "ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180"
)
DATASET_ROWS_FILE_SHA256 = (
    "187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac"
)
ROLE_ASSIGNMENT_SHA256 = (
    "016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02"
)
SIDECAR_MANIFEST_FILE_SHA256 = (
    "6fafa417b4f724a0fdf32cfde5740025c3117e4c0b43231fe9ebe94bd9eff529"
)
SOURCE_INDEX_FILE_SHA256 = (
    "11b9a669324cc7630ba072138983f2dd0daf0d0a4e12596a1204f665eb208a6c"
)
SOURCE_INVENTORY_SHA256 = {
    "scene_role": "f967364a2869f9f87a4c2c1c0053616e263464be53691283909a8f910b94ed5b",
    "frames": "7512a041d2f163cc8978eee1a261951162b2ccd2020414325504e41eac9c623d",
    "manifests": "2bc5f468eeba3f44b1f428b0145c9e63ec84d08d1c21e8a2870227e12e0c44c5",
    "plans": "0359078471ac3f85aa704f44012a44ec9f3c1c2fd6f61ce1628533fc1c2a36e4",
    "summaries": "bd2b181973e3023df0825200657d0d2895f71804134100f60234363503be548a",
}
SIDECAR_ROLE_FILE_SHA256 = {
    "train": "6cd47d0d679ace897f5b5d8e5c2f11eabab01930904666161eec3792fd9ab6d6",
    "checkpoint_selection": (
        "4ed434d04afc94b7b82050f5e9fafc900cc03c33a2d847f9784410f8f76f65de"
    ),
    "probability_calibration": (
        "3e5c10e6c15969eb30fbf38bbdb7b47d5fafe25bf14c5547f07ac609b79d91ae"
    ),
}


class RawSupervisionPlanError(ValueError):
    """Raised when the frozen metadata cannot produce the exact plan."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        ensure_ascii=True,
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise RawSupervisionPlanError(f"{name} must be an exact integer")
    return value


def _exact_str(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise RawSupervisionPlanError(f"{name} must be a nonempty string")
    return value


def _exact_sha256(value: object, *, name: str) -> str:
    if not _is_sha256(value):
        raise RawSupervisionPlanError(f"{name} must be lowercase SHA-256")
    return str(value)


def _finite_vector(value: object, *, name: str, length: int) -> list[float]:
    if type(value) is not list or len(value) != length:
        raise RawSupervisionPlanError(f"{name} must have length {length}")
    result: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RawSupervisionPlanError(f"{name}[{index}] must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise RawSupervisionPlanError(f"{name}[{index}] must be finite")
        result.append(number)
    return result


def _with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json_bytes(core))
    return {**normalized, "content_sha256": canonical_json_sha256(normalized)}


def _read_regular_file(path: Path, expected_sha256: str, *, name: str) -> bytes:
    resolved = path.resolve(strict=True)
    if (
        path != resolved
        or path.is_symlink()
        or not stat.S_ISREG(path.stat(follow_symlinks=False).st_mode)
    ):
        raise PermissionError(f"{name} must be a regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RawSupervisionPlanError(f"{name} file SHA-256 changed")
    return payload


def _parse_jsonl(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise RawSupervisionPlanError(f"{name} must be nonempty newline JSONL")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(payload.splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RawSupervisionPlanError(f"{name}:{number} is invalid JSON") from error
        if type(value) is not dict:
            raise RawSupervisionPlanError(f"{name}:{number} is not an object")
        rows.append(value)
    return rows


def endpoint_identity(row: Mapping[str, Any], side: str) -> dict[str, Any]:
    """Return the frozen full endpoint identity; image hash alone is insufficient."""

    if side not in {"current", "next"}:
        raise RawSupervisionPlanError("endpoint side must be current or next")
    return {
        "dataset_role": _exact_str(row.get("dataset_role"), name="dataset_role"),
        "scene_id": _exact_str(row.get("scene_id"), name="scene_id"),
        "episode_id": _exact_str(row.get("episode_id"), name="episode_id"),
        "env_index": _exact_int(row.get("env_index"), name="env_index"),
        "episode_step": _exact_int(
            row.get(f"{side}_episode_step"), name=f"{side}_episode_step"
        ),
        "frame_index": _exact_int(
            row.get(f"{side}_frame_index"), name=f"{side}_frame_index"
        ),
        "timestamp_ns": _exact_int(
            row.get(f"{side}_timestamp_ns"), name=f"{side}_timestamp_ns"
        ),
        "image_sha256": _exact_sha256(
            row.get(f"{side}_image_sha256"), name=f"{side}_image_sha256"
        ),
    }


def _validate_sidecar_join(
    row: Mapping[str, Any],
    sidecar: Mapping[str, Any],
) -> None:
    global_row = _exact_int(row.get("global_row"), name="global_row")
    expected = {
        "global_row": global_row,
        "dataset_role": row.get("dataset_role"),
        "scene_id_sha256": hashlib.sha256(
            _exact_str(row.get("scene_id"), name="scene_id").encode("utf-8")
        ).hexdigest(),
        "frames_jsonl_sha256": row.get("frames_jsonl_sha256"),
        "env_index": row.get("env_index"),
        "current_frame_index": row.get("current_frame_index"),
        "next_frame_index": row.get("next_frame_index"),
        "current_timestamp_ns": row.get("current_timestamp_ns"),
        "next_timestamp_ns": row.get("next_timestamp_ns"),
        "row_identity_sha256": attitude_sidecar.row_identity_sha256(row),
    }
    for field, wanted in expected.items():
        if sidecar.get(field) != wanted:
            raise RawSupervisionPlanError(
                f"global row {global_row} sidecar join changed at {field}"
            )
    core = dict(sidecar)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise RawSupervisionPlanError(
            f"global row {global_row} sidecar content hash changed"
        )


def _endpoint_record(
    row: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    side: str,
) -> dict[str, Any]:
    identity = endpoint_identity(row, side)
    attitude = sidecar.get(side)
    if type(attitude) is not dict:
        raise RawSupervisionPlanError(f"sidecar {side} attitude is absent")
    quaternion = _finite_vector(
        attitude.get("base_quat_world_xyzw"),
        name=f"{side}.base_quat_world_xyzw",
        length=4,
    )
    yaw = _finite_vector(
        [attitude.get("stored_base_yaw_rad")],
        name=f"{side}.stored_base_yaw_rad",
        length=1,
    )[0]
    core = {
        "schema": ENDPOINT_SCHEMA,
        "identity": identity,
        "identity_sha256": canonical_json_sha256(identity),
        "image_path_metadata_only": _exact_str(
            row.get(f"{side}_image_path"), name=f"{side}_image_path"
        ),
        "frames_jsonl_sha256": _exact_sha256(
            row.get("frames_jsonl_sha256"), name="frames_jsonl_sha256"
        ),
        "scene_manifest_sha256": _exact_sha256(
            row.get("scene_manifest_sha256"), name="scene_manifest_sha256"
        ),
        "base_quat_world_xyzw": quaternion,
        "stored_base_yaw_rad": yaw,
    }
    return _with_content_sha256(core)


def _pair_record(
    row: Mapping[str, Any],
    sidecar: Mapping[str, Any],
    endpoint_hashes: Mapping[str, str],
) -> dict[str, Any]:
    primitive = _exact_str(row.get("primitive"), name="primitive")
    if primitive not in PRIMITIVE_VOCABULARY:
        raise RawSupervisionPlanError(f"unsupported primitive {primitive!r}")
    role = _exact_str(row.get("dataset_role"), name="dataset_role")
    core = {
        "schema": PAIR_SCHEMA,
        "dataset_role": role,
        "global_row": _exact_int(row.get("global_row"), name="global_row"),
        "scene_id": _exact_str(row.get("scene_id"), name="scene_id"),
        "family": _exact_str(row.get("family"), name="family"),
        "episode_id": _exact_str(row.get("episode_id"), name="episode_id"),
        "env_index": _exact_int(row.get("env_index"), name="env_index"),
        "reset_count": _exact_int(row.get("reset_count"), name="reset_count"),
        "source_split": _exact_str(row.get("source_split"), name="source_split"),
        "frames_jsonl_sha256": _exact_sha256(
            row.get("frames_jsonl_sha256"), name="frames_jsonl_sha256"
        ),
        "scene_manifest_sha256": _exact_sha256(
            row.get("scene_manifest_sha256"), name="scene_manifest_sha256"
        ),
        "primitive": primitive,
        "relative_se2_current_frame": _finite_vector(
            row.get("relative_se2_current_frame"),
            name="relative_se2_current_frame",
            length=3,
        ),
        "current_endpoint_sha256": endpoint_hashes["current"],
        "next_endpoint_sha256": endpoint_hashes["next"],
        "label_shard_path_metadata_only": _exact_str(
            row.get("label_shard_path"), name="label_shard_path"
        ),
        "label_shard_sha256": _exact_sha256(
            row.get("label_shard_sha256"), name="label_shard_sha256"
        ),
        "label_shard_row": _exact_int(
            row.get("label_shard_row"), name="label_shard_row"
        ),
        "sidecar_row_identity_sha256": _exact_sha256(
            sidecar.get("row_identity_sha256"), name="sidecar row identity"
        ),
    }
    return _with_content_sha256(core)


@dataclass(frozen=True)
class DevelopmentRawSupervisionPlan:
    value: Mapping[str, Any]
    pairs: tuple[Mapping[str, Any], ...]
    endpoints: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class DevelopmentSourceInventory:
    records: tuple[Mapping[str, Any], ...]
    hashes: Mapping[str, str]
    access_ledger: Mapping[str, Any]


def plan_development_raw_supervision(
    rows: Sequence[Mapping[str, Any]],
    sidecar_rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    input_bindings: Mapping[str, Any],
    access_ledger: Mapping[str, Any],
    enforce_frozen_counts: bool = True,
) -> DevelopmentRawSupervisionPlan:
    """Join exact pair metadata to development attitudes without payload access."""

    if set(sidecar_rows_by_role) != set(DEVELOPMENT_ROLES):
        raise RawSupervisionPlanError("sidecar roles must be exactly development roles")
    role_counts = Counter()
    role_scenes: dict[str, set[str]] = defaultdict(set)
    full_global_rows: set[int] = set()
    development_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if type(row) is not dict:
            raise RawSupervisionPlanError("paired row must be an exact object")
        role = _exact_str(row.get("dataset_role"), name="dataset_role")
        if role not in ALL_ROLES:
            raise RawSupervisionPlanError(f"unsupported dataset role {role!r}")
        global_row = _exact_int(row.get("global_row"), name="global_row")
        if global_row in full_global_rows:
            raise RawSupervisionPlanError("global row is duplicated")
        full_global_rows.add(global_row)
        role_counts[role] += 1
        if role in DEVELOPMENT_ROLES:
            development_rows.append(row)
            role_scenes[role].add(_exact_str(row.get("scene_id"), name="scene_id"))
    if enforce_frozen_counts and dict(role_counts) != FULL_ROLE_PAIR_COUNTS:
        raise RawSupervisionPlanError("frozen paired role counts changed")

    sidecars: dict[str, dict[int, Mapping[str, Any]]] = {}
    for role in DEVELOPMENT_ROLES:
        indexed: dict[int, Mapping[str, Any]] = {}
        for row in sidecar_rows_by_role[role]:
            if row.get("dataset_role") != role:
                raise RawSupervisionPlanError(f"{role} sidecar contains another role")
            global_row = _exact_int(row.get("global_row"), name="sidecar global_row")
            if global_row in indexed:
                raise RawSupervisionPlanError(f"{role} sidecar global row repeats")
            indexed[global_row] = row
        if enforce_frozen_counts and len(indexed) != ROLE_PAIR_COUNTS[role]:
            raise RawSupervisionPlanError(f"{role} sidecar count changed")
        sidecars[role] = indexed

    endpoint_by_identity: dict[str, dict[str, Any]] = {}
    pairs: list[dict[str, Any]] = []
    used_sidecars: dict[str, set[int]] = defaultdict(set)
    primitives: set[str] = set()
    for row in sorted(
        development_rows,
        key=lambda item: (
            DEVELOPMENT_ROLES.index(str(item["dataset_role"])),
            int(item["global_row"]),
        ),
    ):
        role = str(row["dataset_role"])
        global_row = int(row["global_row"])
        sidecar = sidecars[role].get(global_row)
        if sidecar is None:
            raise RawSupervisionPlanError(
                f"{role} global row {global_row} has no attitude sidecar"
            )
        _validate_sidecar_join(row, sidecar)
        used_sidecars[role].add(global_row)
        references: dict[str, str] = {}
        for side in ("current", "next"):
            endpoint = _endpoint_record(row, sidecar, side)
            identity_sha = str(endpoint["identity_sha256"])
            existing = endpoint_by_identity.get(identity_sha)
            if existing is not None and existing != endpoint:
                raise RawSupervisionPlanError(
                    "one exact endpoint identity maps to conflicting metadata"
                )
            endpoint_by_identity.setdefault(identity_sha, endpoint)
            references[side] = identity_sha
        pair = _pair_record(row, sidecar, references)
        primitives.add(str(pair["primitive"]))
        pairs.append(pair)
    for role in DEVELOPMENT_ROLES:
        if used_sidecars[role] != set(sidecars[role]):
            raise RawSupervisionPlanError(f"{role} sidecar contains an orphan row")
    if primitives != set(PRIMITIVE_VOCABULARY):
        raise RawSupervisionPlanError("development primitive vocabulary changed")

    endpoints = sorted(
        endpoint_by_identity.values(),
        key=lambda item: (
            DEVELOPMENT_ROLES.index(str(item["identity"]["dataset_role"])),
            str(item["identity_sha256"]),
        ),
    )
    endpoint_role_counts = Counter(
        str(item["identity"]["dataset_role"]) for item in endpoints
    )
    observed_pair_counts = Counter(str(item["dataset_role"]) for item in pairs)
    if enforce_frozen_counts and (
        dict(observed_pair_counts) != ROLE_PAIR_COUNTS
        or dict(endpoint_role_counts) != ROLE_UNIQUE_ENDPOINT_COUNTS
        or {role: len(role_scenes[role]) for role in DEVELOPMENT_ROLES}
        != ROLE_SCENE_COUNTS
    ):
        raise RawSupervisionPlanError("frozen development population changed")

    core = {
        "schema": PLAN_SCHEMA,
        "roles": list(DEVELOPMENT_ROLES),
        "primitive_vocabulary": list(PRIMITIVE_VOCABULARY),
        "pair_counts": dict(observed_pair_counts),
        "scene_counts": {
            role: len(role_scenes[role]) for role in DEVELOPMENT_ROLES
        },
        "endpoint_instance_count": 2 * len(pairs),
        "unique_endpoint_counts": dict(endpoint_role_counts),
        "ordered_pair_sha256": canonical_json_sha256(
            [item["content_sha256"] for item in pairs]
        ),
        "ordered_endpoint_sha256": canonical_json_sha256(
            [item["content_sha256"] for item in endpoints]
        ),
        "input_bindings": json.loads(canonical_json_bytes(input_bindings)),
        "access_ledger": json.loads(canonical_json_bytes(access_ledger)),
        "licenses": {
            "raw_raycast_build_authorized": False,
            "rgb_decode_authorized": False,
            "training_authorized": False,
            "selection_authorized": False,
            "calibration_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    return DevelopmentRawSupervisionPlan(
        value=_with_content_sha256(core),
        pairs=tuple(pairs),
        endpoints=tuple(endpoints),
    )


def load_frozen_development_metadata(repo_root: Path) -> DevelopmentRawSupervisionPlan:
    """Open only frozen pair metadata and the three allowed sidecar files."""

    root = Path(repo_root).resolve(strict=True)
    dataset_manifest_path = root / DATASET_MANIFEST_RELATIVE_PATH
    rows_path = root / DATASET_ROWS_RELATIVE_PATH
    sidecar_manifest_path = root / SIDECAR_MANIFEST_RELATIVE_PATH
    manifest_raw = _read_regular_file(
        dataset_manifest_path,
        DATASET_MANIFEST_FILE_SHA256,
        name="paired dataset manifest",
    )
    try:
        manifest = json.loads(manifest_raw)
    except json.JSONDecodeError as error:
        raise RawSupervisionPlanError("paired dataset manifest is invalid JSON") from error
    if type(manifest) is not dict:
        raise RawSupervisionPlanError("paired dataset manifest must be an object")
    scene_roles = manifest.get("scene_roles")
    if (
        type(scene_roles) is not dict
        or scene_roles.get("assignments_sha256") != ROLE_ASSIGNMENT_SHA256
    ):
        raise RawSupervisionPlanError("paired role assignment changed")
    rows_raw = _read_regular_file(
        rows_path,
        DATASET_ROWS_FILE_SHA256,
        name="paired row index",
    )
    rows = _parse_jsonl(rows_raw, name="paired row index")
    role_rows = attitude_sidecar.load_attitude_sidecar_roles(
        sidecar_manifest_path,
        roles=DEVELOPMENT_ROLES,
        expected_manifest_sha256=SIDECAR_MANIFEST_FILE_SHA256,
        contract=attitude_sidecar.FROZEN_BUILD_CONTRACT,
    )
    for role in DEVELOPMENT_ROLES:
        role_path = sidecar_manifest_path.parent / f"{role}.jsonl"
        if attitude_sidecar.sha256_file(role_path) != SIDECAR_ROLE_FILE_SHA256[role]:
            raise RawSupervisionPlanError(f"{role} sidecar file SHA-256 changed")
    input_bindings = {
        "dataset_manifest_path": DATASET_MANIFEST_RELATIVE_PATH,
        "dataset_manifest_file_sha256": DATASET_MANIFEST_FILE_SHA256,
        "dataset_rows_path": DATASET_ROWS_RELATIVE_PATH,
        "dataset_rows_file_sha256": DATASET_ROWS_FILE_SHA256,
        "role_assignment_sha256": ROLE_ASSIGNMENT_SHA256,
        "sidecar_manifest_path": SIDECAR_MANIFEST_RELATIVE_PATH,
        "sidecar_manifest_file_sha256": SIDECAR_MANIFEST_FILE_SHA256,
        "sidecar_role_file_sha256": SIDECAR_ROLE_FILE_SHA256,
    }
    access_ledger = {
        "measurement_scope": (
            "controlled_data_file_opens_excluding_interpreter_module_loading"
        ),
        "dataset_manifest_byte_opens": 1,
        "dataset_rows_byte_opens": 1,
        "g2_row_metadata_rows_read_for_exclusion": FULL_ROLE_PAIR_COUNTS[
            "g2_evaluation"
        ],
        "sidecar_manifest_byte_opens": 1,
        "sidecar_train_byte_opens": 2,
        "sidecar_checkpoint_selection_byte_opens": 2,
        "sidecar_probability_calibration_byte_opens": 2,
        "sidecar_g2_byte_opens": 0,
        "label_shard_payload_opens": 0,
        "rgb_byte_opens": 0,
        "rgb_decodes": 0,
        "g2_geometry_or_label_payload_opens": 0,
        "checkpoint_or_model_output_opens": 0,
        "runtime_or_navigation_result_opens": 0,
        "heldout_or_sealed_opens": 0,
        "hardware_or_production_opens": 0,
    }
    return plan_development_raw_supervision(
        rows,
        role_rows,
        input_bindings=input_bindings,
        access_ledger=access_ledger,
    )


def plan_development_source_inventory(
    plan: DevelopmentRawSupervisionPlan,
    source_rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    enforce_frozen_hashes: bool = True,
) -> DevelopmentSourceInventory:
    """Reduce source-index metadata to exact development scenes without opening paths."""

    if not isinstance(plan, DevelopmentRawSupervisionPlan):
        raise TypeError("plan must be DevelopmentRawSupervisionPlan")
    root = Path(repo_root).resolve(strict=True)
    scene_roles: dict[str, str] = {}
    scene_families: dict[str, str] = {}
    scene_splits: dict[str, str] = {}
    for pair in plan.pairs:
        scene_id = _exact_str(pair.get("scene_id"), name="pair scene_id")
        family = _exact_str(pair.get("family"), name="pair family")
        source_split = _exact_str(pair.get("source_split"), name="pair source_split")
        if scene_families.setdefault(scene_id, family) != family:
            raise RawSupervisionPlanError("one planned scene has multiple families")
        if scene_splits.setdefault(scene_id, source_split) != source_split:
            raise RawSupervisionPlanError("one planned scene has multiple source splits")
    for endpoint in plan.endpoints:
        identity = endpoint.get("identity")
        if not isinstance(identity, Mapping):
            raise RawSupervisionPlanError("planned endpoint identity is absent")
        scene_id = _exact_str(identity.get("scene_id"), name="endpoint scene_id")
        role = _exact_str(identity.get("dataset_role"), name="endpoint role")
        if role not in DEVELOPMENT_ROLES:
            raise RawSupervisionPlanError("planned endpoint crossed development roles")
        previous = scene_roles.setdefault(scene_id, role)
        if previous != role:
            raise RawSupervisionPlanError("one scene appears in two development roles")
    if enforce_frozen_hashes and len(scene_roles) != 88:
        raise RawSupervisionPlanError("development scene population changed")

    by_scene: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        if type(row) is not dict:
            raise RawSupervisionPlanError("source-index row must be an exact object")
        scene_id = _exact_str(row.get("scene_id"), name="source scene_id")
        if scene_id in by_scene:
            raise RawSupervisionPlanError("source index repeats a scene")
        by_scene[scene_id] = row
    if enforce_frozen_hashes and len(by_scene) != 96:
        raise RawSupervisionPlanError("source-index scene count changed")
    if not set(scene_roles) <= set(by_scene):
        raise RawSupervisionPlanError("source index lacks a development scene")
    if enforce_frozen_hashes and len(set(by_scene) - set(scene_roles)) != 8:
        raise RawSupervisionPlanError("source-index G2 exclusion count changed")

    def metadata_path(value: object, *, name: str) -> str:
        path = Path(_exact_str(value, name=name))
        if (
            not path.is_absolute()
            or str(path) != os.path.normpath(str(path))
            or ".." in path.parts
        ):
            raise RawSupervisionPlanError(f"{name} must be canonical and absolute")
        try:
            path.relative_to(root)
        except ValueError as error:
            raise PermissionError(f"{name} escapes the repository") from error
        return str(path)

    selected: list[dict[str, Any]] = []
    scene_role_records: list[dict[str, str]] = []
    frames: list[dict[str, str]] = []
    manifests: list[dict[str, str]] = []
    plans: list[dict[str, str]] = []
    summaries: list[dict[str, str]] = []
    for scene_id in sorted(scene_roles):
        row = by_scene[scene_id]
        hashes = row.get("hashes")
        if not isinstance(hashes, Mapping):
            raise RawSupervisionPlanError("source-index hashes are absent")
        role = scene_roles[scene_id]
        family = _exact_str(row.get("family"), name="source family")
        source_split = _exact_str(row.get("split"), name="source split")
        if (
            scene_families.get(scene_id) != family
            or scene_splits.get(scene_id) != source_split
        ):
            raise RawSupervisionPlanError(
                "source-index family/split differs from paired metadata"
            )
        scene_role_records.append({"scene_id": scene_id, "role": role})
        frame_record = {
            "scene_id": scene_id,
            "path": metadata_path(
                row.get("frames_jsonl_path"), name="frames_jsonl_path"
            ),
            "sha256": _exact_sha256(
                hashes.get("frames_jsonl_file_sha256"),
                name="frames_jsonl_file_sha256",
            ),
        }
        manifest_record = {
            "scene_id": scene_id,
            "path": metadata_path(
                row.get("scene_manifest_path"), name="scene_manifest_path"
            ),
            "file_sha256": _exact_sha256(
                hashes.get("scene_manifest_file_sha256"),
                name="scene_manifest_file_sha256",
            ),
            "content_sha256": _exact_sha256(
                hashes.get("scene_manifest_sha256"),
                name="scene_manifest_sha256",
            ),
        }
        plan_record = {
            "scene_id": scene_id,
            "path": metadata_path(
                row.get("render_plan_path"), name="render_plan_path"
            ),
            "sha256": _exact_sha256(
                hashes.get("render_plan_file_sha256"),
                name="render_plan_file_sha256",
            ),
        }
        summary_record = {
            "scene_id": scene_id,
            "path": metadata_path(
                row.get("render_summary_path"), name="render_summary_path"
            ),
            "sha256": _exact_sha256(
                hashes.get("render_summary_file_sha256"),
                name="render_summary_file_sha256",
            ),
        }
        frames.append(frame_record)
        manifests.append(manifest_record)
        plans.append(plan_record)
        summaries.append(summary_record)
        selected.append(
            {
                "scene_id": scene_id,
                "role": role,
                "family": family,
                "source_split": source_split,
                "frames": frame_record,
                "scene_manifest": manifest_record,
                "render_plan": plan_record,
                "render_summary": summary_record,
            }
        )
    observed_hashes = {
        "scene_role": canonical_json_sha256(scene_role_records),
        "frames": canonical_json_sha256(frames),
        "manifests": canonical_json_sha256(manifests),
        "plans": canonical_json_sha256(plans),
        "summaries": canonical_json_sha256(summaries),
    }
    if enforce_frozen_hashes and observed_hashes != SOURCE_INVENTORY_SHA256:
        raise RawSupervisionPlanError("development source inventory changed")
    return DevelopmentSourceInventory(
        records=tuple(selected),
        hashes=observed_hashes,
        access_ledger={
            "measurement_scope": "metadata_only_no_referenced_path_dereference",
            "source_index_metadata_rows_read": len(source_rows),
            "development_source_records_retained": len(selected),
            "g2_source_records_read_for_exclusion": len(source_rows) - len(selected),
            "source_frames_payload_opens": 0,
            "source_scene_manifest_payload_opens": 0,
            "render_plan_payload_opens": 0,
            "render_summary_payload_opens": 0,
            "g2_payload_opens": 0,
            "rgb_byte_opens": 0,
            "label_shard_payload_opens": 0,
        },
    )


def load_frozen_development_source_inventory(
    repo_root: Path,
    plan: DevelopmentRawSupervisionPlan | None = None,
) -> DevelopmentSourceInventory:
    root = Path(repo_root).resolve(strict=True)
    resolved_plan = plan or load_frozen_development_metadata(root)
    source_path = root / SOURCE_INDEX_RELATIVE_PATH
    source_raw = _read_regular_file(
        source_path,
        SOURCE_INDEX_FILE_SHA256,
        name="rendered source index",
    )
    source_rows = _parse_jsonl(source_raw, name="rendered source index")
    return plan_development_source_inventory(
        resolved_plan,
        source_rows,
        repo_root=root,
    )


__all__ = [
    "ALL_ROLES",
    "DATASET_MANIFEST_FILE_SHA256",
    "DATASET_ROWS_FILE_SHA256",
    "DEVELOPMENT_ROLES",
    "DevelopmentRawSupervisionPlan",
    "DevelopmentSourceInventory",
    "ENDPOINT_SCHEMA",
    "PAIR_SCHEMA",
    "PLAN_SCHEMA",
    "PRIMITIVE_VOCABULARY",
    "ROLE_ASSIGNMENT_SHA256",
    "RawSupervisionPlanError",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "endpoint_identity",
    "load_frozen_development_metadata",
    "load_frozen_development_source_inventory",
    "plan_development_raw_supervision",
    "plan_development_source_inventory",
]
