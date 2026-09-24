"""Fail-closed attitude metadata sidecar for the physical Go2 dataset.

The sidecar adds only deployment-available orientation to immutable dataset
rows. It never opens an image, label shard, depth file, or model artifact.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import secrets
import stat
from typing import Any, Iterable, Mapping, Sequence

from lewm.benchmarks.go2_dynamic_cell_square_projection import (
    compose_yaw_aligned_camera,
)


SIDECAR_MANIFEST_SCHEMA = "lewm_go2_attitude_sidecar_manifest_v1"
SIDECAR_ROW_SCHEMA = "lewm_go2_attitude_sidecar_row_v1"
SIDECAR_ROLE_SCHEMA = "lewm_go2_attitude_sidecar_role_v1"
SIDECAR_ACCESS_LEDGER_SCHEMA = "lewm_go2_attitude_sidecar_access_ledger_v1"
SIDECAR_G2_ATTEMPT_SCHEMA = "lewm_go2_attitude_sidecar_g2_attempt_v1"
SIDECAR_G2_RECEIPT_SCHEMA = "lewm_go2_attitude_sidecar_g2_receipt_v1"
SIDECAR_IMPLEMENTATION_MANIFEST_SCHEMA = (
    "lewm_go2_attitude_sidecar_implementation_manifest_v1"
)
DATASET_SCHEMA = "lewm_go2_paired_navigation_dataset_v3"
DATASET_ROW_SCHEMA = "lewm_go2_paired_navigation_row_v3"
SOURCE_SCHEMA = "lewm_go2_navigation_source_v2_sparse_rgb"
RENDER_AUDIT_SCHEMA = "lewm_go2_selected_render_audit_v1"

DATASET_ROLES = (
    "train",
    "checkpoint_selection",
    "probability_calibration",
    "g2_evaluation",
)
ROLE_FILE_NAMES = {
    role: f"{role}.jsonl" for role in DATASET_ROLES
}
SIDECAR_SOURCE_ROLES = (
    "binding",
    "builder",
    "dynamic_geometry",
    "implementation_manifest",
    "sidecar_library",
    "sidecar_test",
)
SIDECAR_PRECOMMITTED_SOURCE_ROLES = tuple(
    role for role in SIDECAR_SOURCE_ROLES if role != "implementation_manifest"
)

FROZEN_DATASET_MANIFEST_SHA256 = (
    "ed927cceaedb56ff68334af5109381466740850554048127bb72f04da59f7180"
)
FROZEN_DATASET_ROWS_SHA256 = (
    "187b92f0f311718cf3da098f252da89a992071ea800406bbfff382809085caac"
)
FROZEN_SOURCE_INDEX_SHA256 = (
    "11b9a669324cc7630ba072138983f2dd0daf0d0a4e12596a1204f665eb208a6c"
)
FROZEN_RENDER_AUDIT_SHA256 = (
    "9a045dff82fb82adbbb89d10cb4dc0063297805038b000e5f6cd53816e995a9a"
)
FROZEN_DYNAMIC_GEOMETRY_SHA256 = (
    "ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf"
)
FROZEN_ROLE_ASSIGNMENT_SHA256 = (
    "016c5f872c493065ee4c38fb612fb76958728b37a64987b80d7c0d2736616a02"
)
FROZEN_ROLE_COUNTS = {
    "train": 4262,
    "checkpoint_selection": 495,
    "probability_calibration": 415,
    "g2_evaluation": 469,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_THREAD_ENV_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SIDECAR_IMPLEMENTATION_TEST_COUNT = 39
SIDECAR_IMPLEMENTATION_TEST_COMMAND = (
    "env "
    f"PYTHONPATH={_REPOSITORY_ROOT}:{_REPOSITORY_ROOT / 'lewm_worlds'}:"
    "/home/andrewknowles/TinyQuadJEPA/lib/python3.12/site-packages "
    "OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "
    "NUMEXPR_NUM_THREADS=1 /usr/bin/pytest -q "
    "lewm/tests/test_go2_attitude_sidecar.py"
)
_ROW_KEYS = frozenset(
    {
        "schema",
        "global_row",
        "dataset_role",
        "row_identity_sha256",
        "scene_id_sha256",
        "frames_jsonl_sha256",
        "env_index",
        "current_frame_index",
        "next_frame_index",
        "current_timestamp_ns",
        "next_timestamp_ns",
        "current",
        "next",
        "content_sha256",
    }
)
_ATTITUDE_KEYS = frozenset(
    {"base_quat_world_xyzw", "stored_base_yaw_rad"}
)
_ROLE_ENTRY_KEYS = frozenset(
    {
        "schema",
        "dataset_role",
        "path",
        "file_sha256",
        "content_sha256",
        "row_count",
        "ordered_identity_sha256",
        "ordered_global_rows_sha256",
        "distribution_summary_emitted",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "binding",
        "dataset",
        "source_index",
        "render_audit",
        "dynamic_geometry",
        "role_assignment",
        "roles",
        "source_frames",
        "source_map",
        "construction",
        "access_ledger",
        "content_sha256",
    }
)
_G2_ATTEMPT_KEYS = frozenset(
    {
        "schema",
        "attempt_id_sha256",
        "attempt_marker_path",
        "created_at_utc",
        "intent",
        "sidecar_manifest",
        "dataset_manifest",
        "source_checkpoint",
        "g2_role",
        "status",
        "content_sha256",
    }
)


class AttitudeSidecarContractError(ValueError):
    """Raised when an input or sidecar violates the frozen contract."""


class AttitudeSidecarAccessError(PermissionError):
    """Raised before a caller can open a sidecar role it is not authorized for."""


@dataclass(frozen=True)
class AttitudeSidecarBuildContract:
    """Exact immutable identities required by one sidecar construction."""

    dataset_manifest_sha256: str
    dataset_rows_sha256: str
    source_index_sha256: str
    render_audit_sha256: str
    dynamic_geometry_sha256: str
    role_assignment_sha256: str
    role_counts: Mapping[str, int]
    binding_path: Path
    binding_sha256: str
    source_map_paths: Mapping[str, Path]

    def __post_init__(self) -> None:
        for name in (
            "dataset_manifest_sha256",
            "dataset_rows_sha256",
            "source_index_sha256",
            "render_audit_sha256",
            "dynamic_geometry_sha256",
            "role_assignment_sha256",
            "binding_sha256",
        ):
            value = getattr(self, name)
            if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
                raise AttitudeSidecarContractError(
                    f"{name} must be lowercase SHA-256"
                )
        counts = dict(self.role_counts)
        if set(counts) != set(DATASET_ROLES):
            raise AttitudeSidecarContractError("role counts must name exactly four roles")
        for role, count in counts.items():
            if type(count) is not int or count < 0:
                raise AttitudeSidecarContractError("role counts must be nonnegative")
        object.__setattr__(self, "role_counts", counts)
        binding_path = Path(self.binding_path)
        absolute_binding_path = Path(os.path.abspath(os.fspath(binding_path)))
        if binding_path != absolute_binding_path:
            raise AttitudeSidecarContractError(
                "binding_path must be a canonical absolute path"
            )
        object.__setattr__(self, "binding_path", absolute_binding_path)
        source_map_paths = dict(self.source_map_paths)
        if set(source_map_paths) != set(SIDECAR_SOURCE_ROLES):
            raise AttitudeSidecarContractError(
                "source_map_paths must name exactly the registered source roles"
            )
        normalized_source_paths: dict[str, Path] = {}
        for role in SIDECAR_SOURCE_ROLES:
            source_path = Path(source_map_paths[role])
            absolute_source_path = Path(os.path.abspath(os.fspath(source_path)))
            if source_path != absolute_source_path:
                raise AttitudeSidecarContractError(
                    f"source-map path must be canonical and absolute: {role}"
                )
            normalized_source_paths[role] = absolute_source_path
        if len(set(normalized_source_paths.values())) != len(normalized_source_paths):
            raise AttitudeSidecarContractError("source-map paths must be unique")
        object.__setattr__(self, "source_map_paths", normalized_source_paths)


FROZEN_BUILD_CONTRACT = AttitudeSidecarBuildContract(
    dataset_manifest_sha256=FROZEN_DATASET_MANIFEST_SHA256,
    dataset_rows_sha256=FROZEN_DATASET_ROWS_SHA256,
    source_index_sha256=FROZEN_SOURCE_INDEX_SHA256,
    render_audit_sha256=FROZEN_RENDER_AUDIT_SHA256,
    dynamic_geometry_sha256=FROZEN_DYNAMIC_GEOMETRY_SHA256,
    role_assignment_sha256=FROZEN_ROLE_ASSIGNMENT_SHA256,
    role_counts=FROZEN_ROLE_COUNTS,
    binding_path=(
        _REPOSITORY_ROOT
        / "docs/lewm_go2_dynamic_cartesian_n32_v1_binding_2026-07-11.md"
    ),
    binding_sha256=(
        "42687e80a16fb424be47d49782699bbc3ed549d7826a0ce6e78e92aa37188e1e"
    ),
    source_map_paths={
        "binding": (
            _REPOSITORY_ROOT
            / "docs/lewm_go2_dynamic_cartesian_n32_v1_binding_2026-07-11.md"
        ),
        "builder": _REPOSITORY_ROOT / "scripts/build_go2_attitude_sidecar.py",
        "dynamic_geometry": (
            _REPOSITORY_ROOT
            / "lewm/benchmarks/go2_dynamic_cell_square_projection.py"
        ),
        "implementation_manifest": (
            _REPOSITORY_ROOT
            / "docs/lewm_go2_attitude_sidecar_implementation_manifest_2026-07-11.json"
        ),
        "sidecar_library": _REPOSITORY_ROOT / "lewm/datasets/go2_attitude_sidecar.py",
        "sidecar_test": _REPOSITORY_ROOT / "lewm/tests/test_go2_attitude_sidecar.py",
    },
)


@dataclass(frozen=True)
class _SceneTask:
    scene_id: str
    dataset_role: str
    source_split: str
    scene_manifest_sha256: str
    frames_path: str
    frames_sha256: str
    rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _SceneResult:
    scene_id: str
    dataset_role: str
    rows: tuple[dict[str, Any], ...]
    source_record: dict[str, Any]


@dataclass
class _BuildAccessRecorder:
    records: dict[str, dict[str, Any]]

    def __init__(self) -> None:
        self.records = {}

    def record(
        self,
        *,
        path: Path | str,
        sha256: str,
        byte_count: int,
        purpose: str,
    ) -> None:
        canonical_path = str(
            _canonical_absolute_path(path, name="recorded access path")
        )
        digest = _exact_sha256(sha256, name="recorded access hash")
        count = _exact_int(byte_count, name="recorded access byte count")
        if count < 0:
            raise AttitudeSidecarContractError("recorded byte count is negative")
        purpose_value = _exact_str(purpose, name="recorded access purpose")
        record = self.records.setdefault(
            canonical_path,
            {
                "path": canonical_path,
                "sha256": digest,
                "purposes": set(),
                "open_count": 0,
                "byte_count": count,
                "total_bytes_read": 0,
            },
        )
        if record["sha256"] != digest or record["byte_count"] != count:
            raise AttitudeSidecarContractError(
                "one recorded path changed hash or size between byte opens"
            )
        record["purposes"].add(purpose_value)
        record["open_count"] += 1
        record["total_bytes_read"] += count

    def canonical_records(self) -> list[dict[str, Any]]:
        return [
            {
                **{key: value for key, value in record.items() if key != "purposes"},
                "purposes": sorted(record["purposes"]),
            }
            for _path, record in sorted(self.records.items())
        ]


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AttitudeSidecarContractError("value is not canonical JSON") from exc


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(
        _read_regular_file_bytes(Path(path), name="SHA-256 input")
    ).hexdigest()


def _exact_dict(value: Any, *, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise AttitudeSidecarContractError(f"{name} must be an exact JSON object")
    return value


def _exact_list(value: Any, *, name: str) -> list[Any]:
    if type(value) is not list:
        raise AttitudeSidecarContractError(f"{name} must be an exact JSON array")
    return value


def _exact_str(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise AttitudeSidecarContractError(f"{name} must be a nonempty string")
    return value


def _exact_int(value: Any, *, name: str) -> int:
    if type(value) is not int:
        raise AttitudeSidecarContractError(f"{name} must be an exact integer")
    return value


def _exact_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise AttitudeSidecarContractError(f"{name} must be an exact boolean")
    return value


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttitudeSidecarContractError(f"{name} must be a JSON number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise AttitudeSidecarContractError(f"{name} must be finite")
    return result


def _exact_sha256(value: Any, *, name: str) -> str:
    result = _exact_str(value, name=name)
    if _SHA256_RE.fullmatch(result) is None:
        raise AttitudeSidecarContractError(f"{name} must be lowercase SHA-256")
    return result


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, name: str) -> None:
    actual = set(value)
    wanted = set(expected)
    if actual != wanted:
        raise AttitudeSidecarContractError(
            f"{name} keys differ: missing={sorted(wanted - actual)}, "
            f"extra={sorted(actual - wanted)}"
        )


def _canonical_absolute_path(value: Path | str, *, name: str) -> Path:
    path = Path(value)
    absolute = Path(os.path.abspath(os.fspath(path)))
    if path != absolute:
        raise AttitudeSidecarContractError(f"{name} must be a canonical absolute path")
    return absolute


def _open_directory_no_symlinks(path: Path, *, name: str) -> int:
    path = _canonical_absolute_path(path, name=name)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(path.anchor, directory_flags)
    try:
        for component in path.parts[1:]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except OSError as exc:
        os.close(directory_fd)
        raise AttitudeSidecarContractError(
            f"{name} path is missing, aliased, or not a regular directory path"
        ) from exc
    return directory_fd


def _mkdir_parents_no_symlinks(path: Path, *, name: str) -> None:
    path = _canonical_absolute_path(path, name=name)
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_fd = os.open(path.anchor, directory_flags)
    try:
        for component in path.parts[1:]:
            try:
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=directory_fd)
                next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
    except BaseException:
        os.close(directory_fd)
        raise
    os.close(directory_fd)


def _make_staging_directory(parent: Path, *, prefix: str) -> Path:
    parent = _canonical_absolute_path(parent, name="staging parent")
    parent_fd = _open_directory_no_symlinks(parent, name="staging parent")
    try:
        for _attempt in range(128):
            basename = prefix + secrets.token_hex(12)
            try:
                os.mkdir(basename, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            return parent / basename
    finally:
        os.close(parent_fd)
    raise AttitudeSidecarContractError("could not allocate a unique staging directory")


def _read_regular_file_bytes(path: Path, *, name: str) -> bytes:
    path = _canonical_absolute_path(path, name=name)
    directory_fd = _open_directory_no_symlinks(path.parent, name=f"{name} parent")
    try:
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(path.name, file_flags, dir_fd=directory_fd)
    except OSError as exc:
        raise AttitudeSidecarContractError(
            f"{name} path is missing, aliased, or not a regular path"
        ) from exc
    finally:
        os.close(directory_fd)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AttitudeSidecarContractError(f"{name} must be a regular file")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(data) != before.st_size:
        raise AttitudeSidecarContractError(f"{name} changed while it was read")
    return data


def _regular_file_bytes(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
    recorder: _BuildAccessRecorder | None = None,
    purpose: str | None = None,
) -> bytes:
    data = _read_regular_file_bytes(path, name=name)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise AttitudeSidecarContractError(
            f"{name} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    if recorder is not None:
        recorder.record(
            path=path,
            sha256=actual,
            byte_count=len(data),
            purpose=purpose if purpose is not None else name,
        )
    return data


def _parse_json_bytes(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AttitudeSidecarContractError(f"invalid JSON in {name}") from exc
    return _exact_dict(value, name=name)


def _parse_jsonl_bytes(data: bytes, *, name: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise AttitudeSidecarContractError(f"invalid UTF-8 in {name}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line:
            raise AttitudeSidecarContractError(f"blank line in {name}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AttitudeSidecarContractError(
                f"invalid JSON in {name}:{line_number}"
            ) from exc
        result.append(_exact_dict(value, name=f"{name}:{line_number}"))
    if not result:
        raise AttitudeSidecarContractError(f"{name} is empty")
    return result


def _validate_content_hash(value: Mapping[str, Any], *, name: str) -> None:
    core = dict(value)
    declared = _exact_sha256(core.pop("content_sha256", None), name=f"{name} hash")
    if canonical_json_sha256(core) != declared:
        raise AttitudeSidecarContractError(f"{name} content SHA-256 mismatch")


def _scene_id_sha256(scene_id: str) -> str:
    return hashlib.sha256(scene_id.encode("utf-8")).hexdigest()


def row_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    scene_id = _exact_str(row.get("scene_id"), name="row.scene_id")
    role = _exact_str(row.get("dataset_role"), name="row.dataset_role")
    if role not in DATASET_ROLES:
        raise AttitudeSidecarContractError(f"unsupported dataset role {role!r}")
    return {
        "global_row": _exact_int(row.get("global_row"), name="row.global_row"),
        "scene_id": scene_id,
        "scene_id_sha256": _scene_id_sha256(scene_id),
        "dataset_role": role,
        "label_shard_row": _exact_int(
            row.get("label_shard_row"), name="row.label_shard_row"
        ),
        "label_shard_sha256": _exact_sha256(
            row.get("label_shard_sha256"), name="row.label_shard_sha256"
        ),
        "current_image_sha256": _exact_sha256(
            row.get("current_image_sha256"), name="row.current_image_sha256"
        ),
        "next_image_sha256": _exact_sha256(
            row.get("next_image_sha256"), name="row.next_image_sha256"
        ),
    }


def row_identity_sha256(row: Mapping[str, Any]) -> str:
    return canonical_json_sha256(row_identity(row))


def _attitude_from_frame(frame: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    quaternion = _exact_list(
        frame.get("base_quat_world_xyzw"), name=f"{name}.base_quat_world_xyzw"
    )
    if len(quaternion) != 4:
        raise AttitudeSidecarContractError(f"{name} quaternion must have four values")
    normalized = [
        _finite_number(value, name=f"{name}.quaternion[{index}]")
        for index, value in enumerate(quaternion)
    ]
    rpy = _exact_dict(frame.get("base_rpy_rad"), name=f"{name}.base_rpy_rad")
    yaw = _finite_number(rpy.get("yaw"), name=f"{name}.stored_base_yaw_rad")
    try:
        compose_yaw_aligned_camera(normalized, yaw)
    except (TypeError, ValueError) as exc:
        raise AttitudeSidecarContractError(f"invalid {name} quaternion/yaw: {exc}") from exc
    return {
        "base_quat_world_xyzw": normalized,
        "stored_base_yaw_rad": yaw,
    }


def _frame_exact_int(frame: Mapping[str, Any], field: str, *, name: str) -> int:
    return _exact_int(frame.get(field), name=f"{name}.{field}")


def _join_endpoint(
    row: Mapping[str, Any],
    frame: Mapping[str, Any],
    *,
    side: str,
    scene_id: str,
    source_episode_scene_id: int,
    source_split: str,
    scene_manifest_sha256: str,
) -> dict[str, Any]:
    name = f"global row {row['global_row']} {side}"
    if _exact_str(row.get("scene_id"), name="row.scene_id") != scene_id:
        raise AttitudeSidecarContractError(f"{name} dataset scene mismatch")
    expected_frame_index = _exact_int(
        row.get(f"{side}_frame_index"), name=f"row.{side}_frame_index"
    )
    expected_timestamp = _exact_int(
        row.get(f"{side}_timestamp_ns"), name=f"row.{side}_timestamp_ns"
    )
    expected_env = _exact_int(row.get("env_index"), name="row.env_index")
    if (
        _frame_exact_int(frame, "frame_index", name=name) != expected_frame_index
        or _frame_exact_int(frame, "env_index", name=name) != expected_env
        or _frame_exact_int(frame, "timestamp_ns", name=name) != expected_timestamp
    ):
        raise AttitudeSidecarContractError(f"{name} frame key/timestamp mismatch")
    episode = _exact_dict(frame.get("episode"), name=f"{name}.episode")
    if _exact_int(
        episode.get("scene_id"), name=f"{name}.episode.scene_id"
    ) != source_episode_scene_id:
        raise AttitudeSidecarContractError(f"{name} source scene mismatch")
    if _exact_sha256(
        episode.get("manifest_sha256"),
        name=f"{name}.episode.manifest_sha256",
    ) != scene_manifest_sha256 or _exact_sha256(
        row.get("scene_manifest_sha256"), name="row.scene_manifest_sha256"
    ) != scene_manifest_sha256:
        raise AttitudeSidecarContractError(f"{name} scene manifest mismatch")
    if _exact_str(episode.get("split"), name=f"{name}.episode.split") != (
        source_split
    ) or _exact_str(row.get("source_split"), name="row.source_split") != source_split:
        raise AttitudeSidecarContractError(f"{name} source split mismatch")
    source_episode_id = _exact_int(
        episode.get("episode_id"), name=f"{name}.episode.episode_id"
    )
    if str(source_episode_id) != _exact_str(
        row.get("episode_id"), name="row.episode_id"
    ):
        raise AttitudeSidecarContractError(f"{name} episode ID mismatch")
    if _exact_int(episode.get("reset_count"), name=f"{name}.episode.reset_count") != (
        _exact_int(row.get("reset_count"), name="row.reset_count")
    ):
        raise AttitudeSidecarContractError(f"{name} reset count mismatch")
    expected_step = _exact_int(
        row.get(f"{side}_episode_step"), name=f"row.{side}_episode_step"
    )
    if _exact_int(
        episode.get("episode_step"), name=f"{name}.episode.episode_step"
    ) != expected_step:
        raise AttitudeSidecarContractError(f"{name} episode step mismatch")
    return _attitude_from_frame(frame, name=name)


def _build_scene(task: _SceneTask) -> _SceneResult:
    data = _regular_file_bytes(
        Path(task.frames_path),
        expected_sha256=task.frames_sha256,
        name=f"frames metadata for {task.scene_id}",
    )
    frames = _parse_jsonl_bytes(data, name=f"frames metadata for {task.scene_id}")
    frame_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    source_episode_scene_ids: set[int] = set()
    for frame in frames:
        key = (
            _exact_int(frame.get("frame_index"), name="frame.frame_index"),
            _exact_int(frame.get("env_index"), name="frame.env_index"),
        )
        if key in frame_by_key:
            raise AttitudeSidecarContractError(
                f"duplicate source frame {key} in scene {task.scene_id}"
            )
        episode = _exact_dict(frame.get("episode"), name="frame.episode")
        source_episode_scene_ids.add(
            _exact_int(episode.get("scene_id"), name="frame.episode.scene_id")
        )
        if _exact_sha256(
            episode.get("manifest_sha256"), name="frame.episode.manifest_sha256"
        ) != task.scene_manifest_sha256:
            raise AttitudeSidecarContractError(
                f"source frame manifest mismatch in {task.scene_id}"
            )
        if _exact_str(episode.get("split"), name="frame.episode.split") != (
            task.source_split
        ):
            raise AttitudeSidecarContractError(
                f"source frame split mismatch in {task.scene_id}"
            )
        frame_by_key[key] = frame
    if len(source_episode_scene_ids) != 1:
        raise AttitudeSidecarContractError(
            f"source file has multiple numeric scene identities: {task.scene_id}"
        )
    source_episode_scene_id = next(iter(source_episode_scene_ids))
    if source_episode_scene_id < 0:
        raise AttitudeSidecarContractError(
            f"source file has a negative numeric scene identity: {task.scene_id}"
        )

    output: list[dict[str, Any]] = []
    join_keys: set[tuple[Any, ...]] = set()
    identities: set[str] = set()
    for row in task.rows:
        if _exact_str(row.get("scene_id"), name="row.scene_id") != task.scene_id:
            raise AttitudeSidecarContractError("scene task contains a foreign row")
        if _exact_str(row.get("dataset_role"), name="row.dataset_role") != task.dataset_role:
            raise AttitudeSidecarContractError("scene task contains a foreign role")
        if _exact_sha256(
            row.get("frames_jsonl_sha256"), name="row.frames_jsonl_sha256"
        ) != task.frames_sha256:
            raise AttitudeSidecarContractError("row/source frames hash mismatch")
        env_index = _exact_int(row.get("env_index"), name="row.env_index")
        current_index = _exact_int(
            row.get("current_frame_index"), name="row.current_frame_index"
        )
        next_index = _exact_int(row.get("next_frame_index"), name="row.next_frame_index")
        current = frame_by_key.get((current_index, env_index))
        next_frame = frame_by_key.get((next_index, env_index))
        if current is None or next_frame is None:
            raise AttitudeSidecarContractError("dataset row has a missing source endpoint")
        join_key = (
            task.scene_id,
            env_index,
            _exact_str(row.get("episode_id"), name="row.episode_id"),
            _exact_int(row.get("reset_count"), name="row.reset_count"),
            current_index,
            next_index,
        )
        if join_key in join_keys:
            raise AttitudeSidecarContractError("source transition join is non-injective")
        join_keys.add(join_key)
        identity_sha = row_identity_sha256(row)
        if identity_sha in identities:
            raise AttitudeSidecarContractError("row identity is duplicated")
        identities.add(identity_sha)
        core = {
            "schema": SIDECAR_ROW_SCHEMA,
            "global_row": _exact_int(row.get("global_row"), name="row.global_row"),
            "dataset_role": task.dataset_role,
            "row_identity_sha256": identity_sha,
            "scene_id_sha256": _scene_id_sha256(task.scene_id),
            "frames_jsonl_sha256": task.frames_sha256,
            "env_index": env_index,
            "current_frame_index": current_index,
            "next_frame_index": next_index,
            "current_timestamp_ns": _exact_int(
                row.get("current_timestamp_ns"), name="row.current_timestamp_ns"
            ),
            "next_timestamp_ns": _exact_int(
                row.get("next_timestamp_ns"), name="row.next_timestamp_ns"
            ),
            "current": _join_endpoint(
                row,
                current,
                side="current",
                scene_id=task.scene_id,
                source_episode_scene_id=source_episode_scene_id,
                source_split=task.source_split,
                scene_manifest_sha256=task.scene_manifest_sha256,
            ),
            "next": _join_endpoint(
                row,
                next_frame,
                side="next",
                scene_id=task.scene_id,
                source_episode_scene_id=source_episode_scene_id,
                source_split=task.source_split,
                scene_manifest_sha256=task.scene_manifest_sha256,
            ),
        }
        output.append({**core, "content_sha256": canonical_json_sha256(core)})
    output.sort(key=lambda item: item["global_row"])
    return _SceneResult(
        scene_id=task.scene_id,
        dataset_role=task.dataset_role,
        rows=tuple(output),
        source_record={
            "scene_id_sha256": _scene_id_sha256(task.scene_id),
            "dataset_role": task.dataset_role,
            "source_episode_scene_id": source_episode_scene_id,
            "source_split": task.source_split,
            "scene_manifest_sha256": task.scene_manifest_sha256,
            "path": task.frames_path,
            "sha256": task.frames_sha256,
            "frame_count": len(frames),
            "selected_row_count": len(output),
            "byte_count": len(data),
        },
    )


def _worker_initializer() -> None:
    for name in _THREAD_ENV_NAMES:
        os.environ[name] = "1"


def _validate_attitude(value: Any, *, name: str) -> dict[str, Any]:
    attitude = _exact_dict(value, name=name)
    _exact_keys(attitude, _ATTITUDE_KEYS, name=name)
    frame = {
        "base_quat_world_xyzw": attitude["base_quat_world_xyzw"],
        "base_rpy_rad": {"yaw": attitude["stored_base_yaw_rad"]},
    }
    _attitude_from_frame(frame, name=name)
    return attitude


def validate_sidecar_row(value: Any) -> dict[str, Any]:
    row = _exact_dict(value, name="sidecar row")
    _exact_keys(row, _ROW_KEYS, name="sidecar row")
    if row.get("schema") != SIDECAR_ROW_SCHEMA:
        raise AttitudeSidecarContractError("unsupported sidecar row schema")
    if _exact_int(row.get("global_row"), name="global_row") < 0:
        raise AttitudeSidecarContractError("global_row must be nonnegative")
    role = _exact_str(row.get("dataset_role"), name="dataset_role")
    if role not in DATASET_ROLES:
        raise AttitudeSidecarContractError("unsupported dataset role")
    for field in (
        "row_identity_sha256",
        "scene_id_sha256",
        "frames_jsonl_sha256",
        "content_sha256",
    ):
        _exact_sha256(row.get(field), name=field)
    for field in (
        "env_index",
        "current_frame_index",
        "next_frame_index",
        "current_timestamp_ns",
        "next_timestamp_ns",
    ):
        if _exact_int(row.get(field), name=field) < 0:
            raise AttitudeSidecarContractError(f"{field} must be nonnegative")
    _validate_attitude(row.get("current"), name="current")
    _validate_attitude(row.get("next"), name="next")
    _validate_content_hash(row, name="sidecar row")
    return row


def _canonical_role_rows(
    rows: Sequence[Mapping[str, Any]], *, role: str, expected_count: int
) -> list[dict[str, Any]]:
    if role not in DATASET_ROLES:
        raise AttitudeSidecarContractError(f"unsupported role {role!r}")
    normalized = [validate_sidecar_row(dict(row)) for row in rows]
    if len(normalized) != expected_count:
        raise AttitudeSidecarContractError(
            f"{role} sidecar count {len(normalized)} != {expected_count}"
        )
    global_rows = [_exact_int(row["global_row"], name="global_row") for row in normalized]
    if global_rows != sorted(global_rows) or len(global_rows) != len(set(global_rows)):
        raise AttitudeSidecarContractError(f"{role} rows are duplicated or reordered")
    identities = [str(row["row_identity_sha256"]) for row in normalized]
    if len(identities) != len(set(identities)):
        raise AttitudeSidecarContractError(f"{role} row identities are not injective")
    if any(str(row["dataset_role"]) != role for row in normalized):
        raise AttitudeSidecarContractError(f"{role} file contains a foreign role")
    return normalized


def _write_jsonl_exclusive(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    path = _canonical_absolute_path(path, name="JSONL output path")
    parent_fd = _open_directory_no_symlinks(path.parent, name="JSONL output parent")
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    with os.fdopen(descriptor, "wb") as stream:
        for row in rows:
            encoded = canonical_json_bytes(dict(row)) + b"\n"
            stream.write(encoded)
            digest.update(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return digest.hexdigest()


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    path = _canonical_absolute_path(path, name="JSON output path")
    parent_fd = _open_directory_no_symlinks(path.parent, name="JSON output parent")
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o644,
            dir_fd=parent_fd,
        )
    finally:
        os.close(parent_fd)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory_no_symlinks(path, name="fsync directory")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_identity(metadata: os.stat_result) -> tuple[int, int]:
    return (int(metadata.st_dev), int(metadata.st_ino))


def _assert_directory_entry_identity(
    parent_fd: int,
    basename: str,
    expected: tuple[int, int],
    *,
    name: str,
) -> None:
    metadata = os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise AttitudeSidecarContractError(f"{name} is no longer a regular directory")
    if _directory_identity(metadata) != expected:
        raise AttitudeSidecarContractError(f"{name} identity changed during publication")


def _publish_staged_no_replace(
    *,
    staging: Path,
    expected_staging_identity: tuple[int, int],
    output_dir: Path,
    role_names: Sequence[str],
) -> None:
    staging_fd = _open_directory_no_symlinks(staging, name="sidecar staging directory")
    if _directory_identity(os.fstat(staging_fd)) != expected_staging_identity:
        os.close(staging_fd)
        raise AttitudeSidecarContractError(
            "sidecar staging directory identity changed before publication"
        )
    parent_fd = _open_directory_no_symlinks(
        output_dir.parent,
        name="sidecar output parent",
    )
    try:
        os.mkdir(output_dir.name, mode=0o755, dir_fd=parent_fd)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        directory_fd = os.open(output_dir.name, flags, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        os.close(staging_fd)
        raise
    directory_metadata = os.fstat(directory_fd)
    directory_identity = _directory_identity(directory_metadata)
    created: dict[str, tuple[int, int]] = {}
    manifest_published = False
    try:
        for name in (*role_names, "manifest.json"):
            _assert_directory_entry_identity(
                parent_fd,
                output_dir.name,
                directory_identity,
                name="sidecar output directory",
            )
            source_metadata = os.stat(
                name,
                dir_fd=staging_fd,
                follow_symlinks=False,
            )
            if not stat.S_ISREG(source_metadata.st_mode):
                raise AttitudeSidecarContractError(
                    f"staged sidecar entry is not regular: {name}"
                )
            os.link(
                name,
                name,
                src_dir_fd=staging_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
            published = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(published.st_mode):
                raise AttitudeSidecarContractError(
                    f"published sidecar entry is not regular: {name}"
                )
            if _directory_identity(published) != _directory_identity(source_metadata):
                raise AttitudeSidecarContractError(
                    f"published sidecar inode differs from staging: {name}"
                )
            created[name] = _directory_identity(published)
            os.unlink(name, dir_fd=staging_fd)
            os.fsync(directory_fd)
            if name == "manifest.json":
                manifest_published = True
        _assert_directory_entry_identity(
            parent_fd,
            output_dir.name,
            directory_identity,
            name="sidecar output directory",
        )
    except BaseException:
        if not manifest_published:
            for name, expected in reversed(tuple(created.items())):
                try:
                    current = os.stat(
                        name,
                        dir_fd=directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    continue
                if _directory_identity(current) == expected:
                    os.unlink(name, dir_fd=directory_fd)
            os.fsync(directory_fd)
        raise
    finally:
        os.close(directory_fd)
        os.close(staging_fd)
        if not manifest_published:
            try:
                _assert_directory_entry_identity(
                    parent_fd,
                    output_dir.name,
                    directory_identity,
                    name="sidecar output directory",
                )
            except (FileNotFoundError, AttitudeSidecarContractError):
                pass
            else:
                try:
                    os.rmdir(output_dir.name, dir_fd=parent_fd)
                except OSError:
                    pass
        os.close(parent_fd)


def _cleanup_staging_directory(
    staging: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    allowed = set(ROLE_FILE_NAMES.values()) | {"manifest.json"}
    try:
        parent_fd = _open_directory_no_symlinks(
            staging.parent,
            name="staging cleanup parent",
        )
        metadata = os.stat(staging.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or _directory_identity(metadata) != expected_identity
        ):
            os.close(parent_fd)
            return
        staging_fd = os.open(
            staging.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        try:
            entries = os.listdir(staging_fd)
            if set(entries) - allowed:
                return
            for entry in entries:
                entry_metadata = os.stat(
                    entry,
                    dir_fd=staging_fd,
                    follow_symlinks=False,
                )
                if not stat.S_ISREG(entry_metadata.st_mode):
                    return
                os.unlink(entry, dir_fd=staging_fd)
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        current = os.stat(staging.name, dir_fd=parent_fd, follow_symlinks=False)
        if _directory_identity(current) == expected_identity:
            os.rmdir(staging.name, dir_fd=parent_fd)
    except (FileNotFoundError, AttitudeSidecarContractError, OSError):
        return
    finally:
        if "parent_fd" in locals():
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _source_map_record(
    source_map: Mapping[str, Path],
    *,
    contract: AttitudeSidecarBuildContract,
    recorder: _BuildAccessRecorder,
) -> dict[str, Any]:
    if set(source_map) != set(SIDECAR_SOURCE_ROLES):
        raise AttitudeSidecarContractError(
            "source map must name exactly the registered source roles"
        )
    entries = []
    seen_paths: set[str] = set()
    seen_inodes: set[tuple[int, int]] = set()
    for role, raw_path in sorted(source_map.items()):
        role_name = _exact_str(role, name="source-map role")
        path = _canonical_absolute_path(raw_path, name=f"source-map path {role_name}")
        if path != contract.source_map_paths[role_name]:
            raise AttitudeSidecarContractError(
                f"source-map path differs from the build contract: {role_name}"
            )
        path_string = str(path)
        if path_string in seen_paths:
            raise AttitudeSidecarContractError("source-map paths must be unique")
        seen_paths.add(path_string)
        inode = _directory_identity(path.stat(follow_symlinks=False))
        if inode in seen_inodes:
            raise AttitudeSidecarContractError("source-map paths use hardlink aliases")
        seen_inodes.add(inode)
        data = _read_regular_file_bytes(path, name=f"source-map file {role_name}")
        digest = hashlib.sha256(data).hexdigest()
        recorder.record(
            path=path,
            sha256=digest,
            byte_count=len(data),
            purpose=f"source_map:{role_name}",
        )
        entries.append({"role": role_name, "path": path_string, "sha256": digest})
    return {
        "entries": entries,
        "entry_count": len(entries),
        "source_map_sha256": canonical_json_sha256(entries),
    }


def validate_attitude_sidecar_implementation_manifest(
    manifest_path: Path,
    *,
    expected_sha256: str,
    contract: AttitudeSidecarBuildContract,
    recorder: _BuildAccessRecorder | None = None,
) -> dict[str, Any]:
    """Authenticate the externally frozen source set before data access."""

    manifest_path = _canonical_absolute_path(
        manifest_path, name="implementation manifest path"
    )
    if manifest_path != contract.source_map_paths["implementation_manifest"]:
        raise AttitudeSidecarContractError(
            "implementation manifest path differs from the build contract"
        )
    manifest_sha = _exact_sha256(
        expected_sha256, name="implementation manifest SHA-256"
    )
    data = _regular_file_bytes(
        manifest_path,
        expected_sha256=manifest_sha,
        name="implementation manifest",
        recorder=recorder,
        purpose="implementation_manifest_validation",
    )
    payload = _parse_json_bytes(data, name="implementation manifest")
    expected_bytes = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    if data != expected_bytes:
        raise AttitudeSidecarContractError(
            "implementation manifest JSON is not canonical"
        )
    _exact_keys(
        payload,
        {
            "schema",
            "binding",
            "sources",
            "tests",
            "resource_policy",
            "content_sha256",
        },
        name="implementation manifest",
    )
    if payload.get("schema") != SIDECAR_IMPLEMENTATION_MANIFEST_SCHEMA:
        raise AttitudeSidecarContractError(
            "unsupported sidecar implementation manifest schema"
        )
    _validate_content_hash(payload, name="implementation manifest")

    binding = _exact_dict(payload.get("binding"), name="implementation binding")
    _exact_keys(binding, {"path", "sha256"}, name="implementation binding")
    if (
        _canonical_absolute_path(binding.get("path"), name="implementation binding path")
        != contract.binding_path
        or _exact_sha256(
            binding.get("sha256"), name="implementation binding hash"
        )
        != contract.binding_sha256
    ):
        raise AttitudeSidecarContractError("implementation binding mismatch")

    sources = _exact_dict(payload.get("sources"), name="implementation sources")
    _exact_keys(
        sources,
        {"entries", "entry_count", "source_map_sha256"},
        name="implementation sources",
    )
    entries = _exact_list(sources.get("entries"), name="implementation source entries")
    if _exact_int(sources.get("entry_count"), name="source entry count") != len(
        entries
    ):
        raise AttitudeSidecarContractError("implementation source count mismatch")
    if canonical_json_sha256(entries) != _exact_sha256(
        sources.get("source_map_sha256"), name="implementation source-map hash"
    ):
        raise AttitudeSidecarContractError("implementation source-map hash mismatch")
    roles = []
    inodes: set[tuple[int, int]] = set()
    for raw_entry in entries:
        entry = _exact_dict(raw_entry, name="implementation source entry")
        _exact_keys(entry, {"role", "path", "sha256"}, name="source entry")
        role = _exact_str(entry.get("role"), name="source role")
        path = _canonical_absolute_path(entry.get("path"), name=f"source path {role}")
        digest = _exact_sha256(entry.get("sha256"), name=f"source hash {role}")
        if role not in SIDECAR_PRECOMMITTED_SOURCE_ROLES:
            raise AttitudeSidecarContractError(f"unexpected source role: {role}")
        if path != contract.source_map_paths[role]:
            raise AttitudeSidecarContractError(
                f"implementation source path mismatch: {role}"
            )
        _regular_file_bytes(
            path,
            expected_sha256=digest,
            name=f"source {role}",
            recorder=recorder,
            purpose=f"implementation_source_validation:{role}",
        )
        metadata = path.stat(follow_symlinks=False)
        inode = _directory_identity(metadata)
        if inode in inodes:
            raise AttitudeSidecarContractError("implementation sources use hardlink aliases")
        inodes.add(inode)
        roles.append(role)
    if roles != sorted(SIDECAR_PRECOMMITTED_SOURCE_ROLES):
        raise AttitudeSidecarContractError(
            "implementation source roles are incomplete or reordered"
        )

    tests = _exact_dict(payload.get("tests"), name="implementation tests")
    _exact_keys(tests, {"command", "passed"}, name="implementation tests")
    if (
        _exact_str(tests.get("command"), name="implementation test command")
        != SIDECAR_IMPLEMENTATION_TEST_COMMAND
        or _exact_int(tests.get("passed"), name="implementation passed tests")
        != SIDECAR_IMPLEMENTATION_TEST_COUNT
    ):
        raise AttitudeSidecarContractError("implementation test evidence mismatch")
    resource_policy = _exact_dict(
        payload.get("resource_policy"), name="implementation resource policy"
    )
    _exact_keys(
        resource_policy,
        {"cpu_workers_max", "thread_environment", "neural_device", "igpu"},
        name="implementation resource policy",
    )
    if _exact_int(resource_policy.get("cpu_workers_max"), name="CPU workers") != 6:
        raise AttitudeSidecarContractError("implementation CPU worker cap changed")
    thread_environment = _exact_dict(
        resource_policy.get("thread_environment"), name="implementation thread policy"
    )
    if thread_environment != {name: "1" for name in _THREAD_ENV_NAMES}:
        raise AttitudeSidecarContractError("implementation thread policy changed")
    if (
        resource_policy.get("neural_device") != "none_metadata_only"
        or resource_policy.get("igpu") != "forbidden"
    ):
        raise AttitudeSidecarContractError("implementation device policy changed")
    return payload


def _verified_build_access_records(
    recorder: _BuildAccessRecorder,
    *,
    contract: AttitudeSidecarBuildContract,
    dataset_manifest_path: Path,
    rows_path: Path,
    source_index_path: Path,
    render_audit_path: Path,
    source_frame_records: Sequence[Mapping[str, Any]],
    source_map_record: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}

    def add(
        path: Path | str,
        sha256: str,
        open_count: int,
        purposes: Sequence[str],
    ) -> None:
        canonical_path = str(
            _canonical_absolute_path(path, name="expected access path")
        )
        if canonical_path in expected:
            raise AttitudeSidecarContractError("expected access path is duplicated")
        expected[canonical_path] = {
            "sha256": _exact_sha256(sha256, name="expected access hash"),
            "open_count": _exact_int(open_count, name="expected open count"),
            "purposes": sorted(set(purposes)),
        }

    add(
        dataset_manifest_path,
        contract.dataset_manifest_sha256,
        1,
        ("dataset manifest",),
    )
    add(rows_path, contract.dataset_rows_sha256, 1, ("dataset_rows_metadata",))
    add(source_index_path, contract.source_index_sha256, 1, ("source_index",))
    add(render_audit_path, contract.render_audit_sha256, 1, ("render audit",))
    for record in source_frame_records:
        add(
            record["path"],
            record["sha256"],
            1,
            (f"source_frames_metadata:{record['dataset_role']}",),
        )
    for raw_entry in source_map_record["entries"]:
        entry = _exact_dict(raw_entry, name="source-map access entry")
        role = _exact_str(entry.get("role"), name="source-map access role")
        count = 3
        purposes = [f"source_map:{role}"]
        if role == "implementation_manifest":
            purposes.append("implementation_manifest_validation")
        else:
            purposes.append(f"implementation_source_validation:{role}")
        if role == "binding":
            count += 1
            purposes.append("execution_binding_contract")
        elif role == "dynamic_geometry":
            count += 1
            purposes.append("dynamic_geometry_contract")
        add(entry["path"], entry["sha256"], count, purposes)

    actual_records = recorder.canonical_records()
    actual = {record["path"]: record for record in actual_records}
    if set(actual) != set(expected):
        raise AttitudeSidecarContractError(
            "observed builder read paths differ from the frozen allowlist"
        )
    for path, expected_record in expected.items():
        actual_record = actual[path]
        for field in ("sha256", "open_count", "purposes"):
            if actual_record[field] != expected_record[field]:
                raise AttitudeSidecarContractError(
                    f"observed builder read {field} mismatch: {path}"
                )
        if actual_record["total_bytes_read"] != (
            actual_record["byte_count"] * actual_record["open_count"]
        ):
            raise AttitudeSidecarContractError(
                f"observed builder byte count does not reconcile: {path}"
            )
    return actual_records


def _read_record(
    path: Path,
    expected_sha256: str,
    *,
    name: str,
    recorder: _BuildAccessRecorder | None = None,
) -> dict[str, Any]:
    return _parse_json_bytes(
        _regular_file_bytes(
            path,
            expected_sha256=expected_sha256,
            name=name,
            recorder=recorder,
            purpose=name,
        ),
        name=name,
    )


def build_attitude_sidecar(
    *,
    dataset_manifest_path: Path,
    source_index_path: Path,
    render_audit_path: Path,
    dynamic_geometry_path: Path,
    output_dir: Path,
    source_map: Mapping[str, Path],
    implementation_manifest_path: Path,
    expected_implementation_manifest_sha256: str,
    contract: AttitudeSidecarBuildContract = FROZEN_BUILD_CONTRACT,
    workers: int = 6,
) -> dict[str, Any]:
    """Build and atomically publish the exact role-separated sidecar."""

    workers = _exact_int(workers, name="workers")
    if not 1 <= workers <= 6:
        raise AttitudeSidecarContractError("workers must lie in [1, 6]")
    for thread_variable in _THREAD_ENV_NAMES:
        os.environ[thread_variable] = "1"
    access_recorder = _BuildAccessRecorder()
    validate_attitude_sidecar_implementation_manifest(
        implementation_manifest_path,
        expected_sha256=expected_implementation_manifest_sha256,
        contract=contract,
        recorder=access_recorder,
    )
    dataset_manifest_path = _canonical_absolute_path(
        dataset_manifest_path, name="dataset manifest path"
    )
    source_index_path = _canonical_absolute_path(
        source_index_path, name="source index path"
    )
    render_audit_path = _canonical_absolute_path(
        render_audit_path, name="render audit path"
    )
    dynamic_geometry_path = _canonical_absolute_path(
        dynamic_geometry_path, name="dynamic geometry path"
    )
    output_dir = _canonical_absolute_path(output_dir, name="sidecar output path")
    if output_dir.exists():
        raise FileExistsError(f"sidecar output already exists: {output_dir}")

    manifest = _read_record(
        dataset_manifest_path,
        contract.dataset_manifest_sha256,
        name="dataset manifest",
        recorder=access_recorder,
    )
    if manifest.get("schema") != DATASET_SCHEMA:
        raise AttitudeSidecarContractError("sidecar requires physical dataset v3")
    if _exact_int(manifest.get("row_count"), name="dataset row_count") != sum(
        contract.role_counts.values()
    ):
        raise AttitudeSidecarContractError("dataset row count differs from contract")
    index = _exact_dict(manifest.get("index"), name="dataset index")
    rows_path = _canonical_absolute_path(
        _exact_str(index.get("path"), name="dataset index path"),
        name="dataset index path",
    )
    if _exact_sha256(index.get("sha256"), name="dataset index hash") != (
        contract.dataset_rows_sha256
    ):
        raise AttitudeSidecarContractError("dataset index hash differs from contract")
    rows_data = _regular_file_bytes(
        rows_path,
        expected_sha256=contract.dataset_rows_sha256,
        name="dataset rows",
        recorder=access_recorder,
        purpose="dataset_rows_metadata",
    )
    dataset_rows = _parse_jsonl_bytes(rows_data, name="dataset rows")
    global_rows = []
    for row in dataset_rows:
        if row.get("schema") != DATASET_ROW_SCHEMA:
            raise AttitudeSidecarContractError("dataset row schema mismatch")
        global_rows.append(_exact_int(row.get("global_row"), name="global_row"))
    if global_rows != list(range(len(dataset_rows))):
        raise AttitudeSidecarContractError("dataset rows are missing, duplicated, or reordered")

    role_contract = _exact_dict(manifest.get("scene_roles"), name="scene roles")
    if _exact_sha256(
        role_contract.get("assignments_sha256"), name="role assignment hash"
    ) != contract.role_assignment_sha256:
        raise AttitudeSidecarContractError("role assignment hash mismatch")
    declared_counts_raw = _exact_dict(
        role_contract.get("row_counts"), name="role counts"
    )
    if set(declared_counts_raw) != set(DATASET_ROLES):
        raise AttitudeSidecarContractError("dataset role row counts are incomplete")
    declared_counts = {
        role: _exact_int(declared_counts_raw[role], name=f"role count {role}")
        for role in DATASET_ROLES
    }
    if declared_counts != dict(contract.role_counts):
        raise AttitudeSidecarContractError("dataset role row counts mismatch")
    assignments = _exact_dict(role_contract.get("assignments"), name="role assignments")

    source_index_data = _regular_file_bytes(
        source_index_path,
        expected_sha256=contract.source_index_sha256,
        name="source index",
        recorder=access_recorder,
        purpose="source_index",
    )
    source_rows = _parse_jsonl_bytes(source_index_data, name="source index")
    source_by_scene: dict[str, dict[str, Any]] = {}
    for source in source_rows:
        if source.get("schema") != SOURCE_SCHEMA:
            raise AttitudeSidecarContractError("source index schema mismatch")
        scene_id = _exact_str(source.get("scene_id"), name="source.scene_id")
        if _exact_sha256(
            source.get("scene_id_sha256"), name="source.scene_id_sha256"
        ) != _scene_id_sha256(scene_id):
            raise AttitudeSidecarContractError("source index scene hash mismatch")
        if scene_id in source_by_scene:
            raise AttitudeSidecarContractError("source index scene is duplicated")
        source_by_scene[scene_id] = source

    audit = _read_record(
        render_audit_path,
        contract.render_audit_sha256,
        name="render audit",
        recorder=access_recorder,
    )
    if audit.get("schema") != RENDER_AUDIT_SCHEMA:
        raise AttitudeSidecarContractError("render audit schema mismatch")
    _validate_content_hash(audit, name="render audit")
    audited_index = _exact_dict(audit.get("output_source_index"), name="audited index")
    if (
        _canonical_absolute_path(
            _exact_str(audited_index.get("path"), name="audited index path"),
            name="audited index path",
        )
        != source_index_path
        or _exact_sha256(audited_index.get("sha256"), name="audited index hash")
        != contract.source_index_sha256
    ):
        raise AttitudeSidecarContractError("render audit binds a different source index")
    _regular_file_bytes(
        dynamic_geometry_path,
        expected_sha256=contract.dynamic_geometry_sha256,
        name="dynamic geometry source",
        recorder=access_recorder,
        purpose="dynamic_geometry_contract",
    )
    _regular_file_bytes(
        contract.binding_path,
        expected_sha256=contract.binding_sha256,
        name="execution binding",
        recorder=access_recorder,
        purpose="execution_binding_contract",
    )

    dataset_sources_raw = _exact_list(manifest.get("sources"), name="dataset sources")
    dataset_source_by_scene: dict[str, dict[str, Any]] = {}
    for raw in dataset_sources_raw:
        source = _exact_dict(raw, name="dataset source")
        scene_id = _exact_str(source.get("scene_id"), name="dataset source scene")
        if scene_id in dataset_source_by_scene:
            raise AttitudeSidecarContractError("dataset source scene is duplicated")
        dataset_source_by_scene[scene_id] = source
    if set(dataset_source_by_scene) != set(source_by_scene) or set(assignments) != set(
        source_by_scene
    ):
        raise AttitudeSidecarContractError("dataset/source/role scene sets differ")

    rows_by_scene: dict[str, list[dict[str, Any]]] = {
        scene_id: [] for scene_id in source_by_scene
    }
    role_counts = Counter()
    for row in dataset_rows:
        scene_id = _exact_str(row.get("scene_id"), name="row.scene_id")
        if scene_id not in rows_by_scene:
            raise AttitudeSidecarContractError("dataset row names an unknown scene")
        role = _exact_str(row.get("dataset_role"), name="row.dataset_role")
        if assignments.get(scene_id) != role:
            raise AttitudeSidecarContractError("dataset row role assignment mismatch")
        role_counts[role] += 1
        rows_by_scene[scene_id].append(row)
    if dict(role_counts) != dict(contract.role_counts):
        raise AttitudeSidecarContractError("observed role counts differ from contract")

    tasks: list[_SceneTask] = []
    for scene_id in sorted(source_by_scene):
        indexed = source_by_scene[scene_id]
        dataset_source = dataset_source_by_scene[scene_id]
        task_role = _exact_str(assignments[scene_id], name="scene role")
        if _exact_str(
            dataset_source.get("dataset_role"), name="dataset source role"
        ) != task_role:
            raise AttitudeSidecarContractError(
                "dataset source role assignment mismatch"
            )
        indexed_hashes = _exact_dict(indexed.get("hashes"), name="source hashes")
        dataset_hashes = _exact_dict(
            dataset_source.get("hashes"), name="dataset source hashes"
        )
        dataset_paths = _exact_dict(
            dataset_source.get("paths"), name="dataset source paths"
        )
        frames_path = _canonical_absolute_path(
            _exact_str(indexed.get("frames_jsonl_path"), name="frames path"),
            name="frames path",
        )
        frames_sha = _exact_sha256(
            indexed_hashes.get("frames_jsonl_file_sha256"), name="frames hash"
        )
        scene_manifest_sha = _exact_sha256(
            indexed_hashes.get("scene_manifest_sha256"),
            name="source scene manifest hash",
        )
        source_split = _exact_str(indexed.get("split"), name="source split")
        if (
            _canonical_absolute_path(
                _exact_str(
                    dataset_paths.get("frames_jsonl"),
                    name="dataset frames path",
                ),
                name="dataset frames path",
            )
            != frames_path
            or _exact_sha256(
                dataset_hashes.get("frames_jsonl_sha256"), name="dataset frames hash"
            )
            != frames_sha
            or _exact_sha256(
                dataset_hashes.get("scene_manifest_sha256"),
                name="dataset scene manifest hash",
            )
            != scene_manifest_sha
        ):
            raise AttitudeSidecarContractError("dataset/source frames provenance mismatch")
        tasks.append(
            _SceneTask(
                scene_id=scene_id,
                dataset_role=task_role,
                source_split=source_split,
                scene_manifest_sha256=scene_manifest_sha,
                frames_path=str(frames_path),
                frames_sha256=frames_sha,
                rows=tuple(rows_by_scene[scene_id]),
            )
        )

    if workers == 1:
        results = [_build_scene(task) for task in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_initializer,
        ) as executor:
            results = list(executor.map(_build_scene, tasks))
    if [result.scene_id for result in results] != [task.scene_id for task in tasks]:
        raise AttitudeSidecarContractError("worker result order/identity mismatch")

    rows_by_role: dict[str, list[dict[str, Any]]] = {
        role: [] for role in DATASET_ROLES
    }
    source_frame_records = []
    for result in results:
        rows_by_role[result.dataset_role].extend(result.rows)
        source_frame_records.append(result.source_record)
        access_recorder.record(
            path=result.source_record["path"],
            sha256=result.source_record["sha256"],
            byte_count=result.source_record["byte_count"],
            purpose=f"source_frames_metadata:{result.dataset_role}",
        )
    all_global_rows: list[int] = []
    for role in DATASET_ROLES:
        rows_by_role[role].sort(key=lambda item: int(item["global_row"]))
        rows_by_role[role] = _canonical_role_rows(
            rows_by_role[role],
            role=role,
            expected_count=contract.role_counts[role],
        )
        all_global_rows.extend(int(row["global_row"]) for row in rows_by_role[role])
    if sorted(all_global_rows) != list(range(len(dataset_rows))):
        raise AttitudeSidecarContractError("sidecar did not join every dataset row exactly once")

    source_map_record = _source_map_record(
        source_map,
        contract=contract,
        recorder=access_recorder,
    )
    _mkdir_parents_no_symlinks(
        output_dir.parent,
        name="sidecar output parent",
    )
    staging = _make_staging_directory(
        output_dir.parent,
        prefix=f".{output_dir.name}.tmp.",
    )
    staging_fd = _open_directory_no_symlinks(staging, name="staging directory")
    staging_identity = _directory_identity(os.fstat(staging_fd))
    os.close(staging_fd)
    try:
        role_entries: dict[str, dict[str, Any]] = {}
        for role in DATASET_ROLES:
            role_rows = rows_by_role[role]
            staged_path = staging / ROLE_FILE_NAMES[role]
            file_sha = _write_jsonl_exclusive(staged_path, role_rows)
            final_path = output_dir / ROLE_FILE_NAMES[role]
            role_entries[role] = {
                "schema": SIDECAR_ROLE_SCHEMA,
                "dataset_role": role,
                "path": str(final_path),
                "file_sha256": file_sha,
                "content_sha256": canonical_json_sha256(role_rows),
                "row_count": len(role_rows),
                "ordered_identity_sha256": canonical_json_sha256(
                    [row["row_identity_sha256"] for row in role_rows]
                ),
                "ordered_global_rows_sha256": canonical_json_sha256(
                    [row["global_row"] for row in role_rows]
                ),
                "distribution_summary_emitted": False,
            }

        validate_attitude_sidecar_implementation_manifest(
            implementation_manifest_path,
            expected_sha256=expected_implementation_manifest_sha256,
            contract=contract,
            recorder=access_recorder,
        )

        completed_read_events = _verified_build_access_records(
            access_recorder,
            contract=contract,
            dataset_manifest_path=dataset_manifest_path,
            rows_path=rows_path,
            source_index_path=source_index_path,
            render_audit_path=render_audit_path,
            source_frame_records=source_frame_records,
            source_map_record=source_map_record,
        )
        completed_role_write_events = [
            {
                "dataset_role": role,
                "logical_path": str(output_dir / ROLE_FILE_NAMES[role]),
                "file_sha256": role_entries[role]["file_sha256"],
                "byte_count": int((staging / ROLE_FILE_NAMES[role]).stat().st_size),
                "physical_mode": "private_staging_then_hardlink_noreplace",
            }
            for role in DATASET_ROLES
        ]
        access_ledger = {
            "schema": SIDECAR_ACCESS_LEDGER_SCHEMA,
            "ledger_semantics": (
                "observed_builder_wrapper_events_plus_worker_receipts_v1"
            ),
            "measurement_scope": (
                "builder_controlled_byte_opens_excluding_interpreter_module_loading"
            ),
            "completed_read_events": completed_read_events,
            "completed_role_write_events": completed_role_write_events,
            "logical_output_artifacts": [
                str(output_dir / ROLE_FILE_NAMES[role]) for role in DATASET_ROLES
            ]
            + [str(output_dir / "manifest.json")],
            "publication_mode": (
                "retained_staging_and_output_dirfds_hardlink_noreplace_manifest_last"
            ),
            "post_ledger_prepublication_source_guard": {
                "implementation_manifest_open_count": 1,
                "precommitted_source_open_count_each": 1,
                "occurs_after_manifest_staging_before_publication": True,
            },
            "manifest_hash_reopens_after_publication": 0,
            "image_byte_opens": 0,
            "label_shard_byte_opens": 0,
            "depth_byte_opens": 0,
            "model_artifact_opens": 0,
            "g2_distribution_summaries": 0,
            "g2_role_file_reopens_after_write": 0,
            "source_metadata_only": True,
        }
        core = {
            "schema": SIDECAR_MANIFEST_SCHEMA,
            "binding": {
                "path": str(contract.binding_path),
                "sha256": contract.binding_sha256,
            },
            "dataset": {
                "manifest_path": str(dataset_manifest_path),
                "manifest_sha256": contract.dataset_manifest_sha256,
                "rows_path": str(rows_path),
                "rows_sha256": contract.dataset_rows_sha256,
                "row_count": len(dataset_rows),
                "schema": DATASET_SCHEMA,
            },
            "source_index": {
                "path": str(source_index_path),
                "sha256": contract.source_index_sha256,
                "scene_count": len(source_rows),
            },
            "render_audit": {
                "path": str(render_audit_path),
                "sha256": contract.render_audit_sha256,
                "content_sha256": _exact_sha256(
                    audit.get("content_sha256"), name="render audit content hash"
                ),
            },
            "dynamic_geometry": {
                "path": str(dynamic_geometry_path),
                "sha256": contract.dynamic_geometry_sha256,
            },
            "role_assignment": {
                "sha256": contract.role_assignment_sha256,
                "row_counts": dict(contract.role_counts),
            },
            "roles": role_entries,
            "source_frames": sorted(
                source_frame_records, key=lambda item: item["scene_id_sha256"]
            ),
            "source_map": source_map_record,
            "construction": {
                "workers": workers,
                "worker_start_method": "serial" if workers == 1 else "spawn",
                "maximum_workers": 6,
                "native_threads_per_worker": 1,
                "thread_environment": {
                    name: os.environ.get(name) for name in _THREAD_ENV_NAMES
                },
                "scene_order": "scene_id_ascending",
                "row_order": "global_row_ascending_within_role",
                "publication": "exclusive_directory_create_manifest_last",
            },
            "access_ledger": access_ledger,
        }
        sidecar_manifest = {
            **core,
            "content_sha256": canonical_json_sha256(core),
        }
        _write_json_exclusive(staging / "manifest.json", sidecar_manifest)
        validate_attitude_sidecar_implementation_manifest(
            implementation_manifest_path,
            expected_sha256=expected_implementation_manifest_sha256,
            contract=contract,
        )
        _publish_staged_no_replace(
            staging=staging,
            expected_staging_identity=staging_identity,
            output_dir=output_dir,
            role_names=tuple(ROLE_FILE_NAMES[role] for role in DATASET_ROLES),
        )
        _fsync_directory(output_dir)
        _fsync_directory(output_dir.parent)
        _cleanup_staging_directory(
            staging,
            expected_identity=staging_identity,
        )
        return sidecar_manifest
    except BaseException:
        _cleanup_staging_directory(
            staging,
            expected_identity=staging_identity,
        )
        raise


def _validate_role_entry(
    value: Any, *, role: str, manifest_path: Path
) -> dict[str, Any]:
    entry = _exact_dict(value, name=f"role entry {role}")
    _exact_keys(entry, _ROLE_ENTRY_KEYS, name=f"role entry {role}")
    if entry.get("schema") != SIDECAR_ROLE_SCHEMA or entry.get("dataset_role") != role:
        raise AttitudeSidecarContractError(f"role entry identity mismatch: {role}")
    expected_path = _canonical_absolute_path(
        manifest_path.parent / ROLE_FILE_NAMES[role],
        name=f"expected {role} path",
    )
    if _canonical_absolute_path(
        _exact_str(entry.get("path"), name=f"{role} path"),
        name=f"{role} path",
    ) != expected_path:
        raise AttitudeSidecarContractError(f"role file path is noncanonical: {role}")
    for field in (
        "file_sha256",
        "content_sha256",
        "ordered_identity_sha256",
        "ordered_global_rows_sha256",
    ):
        _exact_sha256(entry.get(field), name=f"{role}.{field}")
    if _exact_int(entry.get("row_count"), name=f"{role}.row_count") < 0:
        raise AttitudeSidecarContractError("role row count must be nonnegative")
    if _exact_bool(
        entry.get("distribution_summary_emitted"),
        name=f"{role}.distribution_summary_emitted",
    ) is not False:
        raise AttitudeSidecarContractError("sidecar role must not contain a summary")
    return entry


def validate_attitude_sidecar_manifest(
    value: Any,
    *,
    manifest_path: Path,
    contract: AttitudeSidecarBuildContract | None = None,
) -> dict[str, Any]:
    manifest = _exact_dict(value, name="sidecar manifest")
    _exact_keys(manifest, _MANIFEST_KEYS, name="sidecar manifest")
    if manifest.get("schema") != SIDECAR_MANIFEST_SCHEMA:
        raise AttitudeSidecarContractError("unsupported sidecar manifest schema")
    _validate_content_hash(manifest, name="sidecar manifest")
    binding = _exact_dict(manifest.get("binding"), name="binding")
    _exact_keys(binding, {"path", "sha256"}, name="binding")
    _exact_str(binding.get("path"), name="binding.path")
    dataset = _exact_dict(manifest.get("dataset"), name="dataset")
    _exact_keys(
        dataset,
        {
            "manifest_path", "manifest_sha256", "rows_path", "rows_sha256",
            "row_count", "schema",
        },
        name="dataset",
    )
    source_index = _exact_dict(manifest.get("source_index"), name="source index")
    _exact_keys(source_index, {"path", "sha256", "scene_count"}, name="source index")
    render_audit = _exact_dict(manifest.get("render_audit"), name="render audit")
    _exact_keys(
        render_audit, {"path", "sha256", "content_sha256"}, name="render audit"
    )
    geometry = _exact_dict(manifest.get("dynamic_geometry"), name="dynamic geometry")
    _exact_keys(geometry, {"path", "sha256"}, name="dynamic geometry")
    assignment = _exact_dict(manifest.get("role_assignment"), name="role assignment")
    _exact_keys(assignment, {"sha256", "row_counts"}, name="role assignment")
    for record, fields in (
        (binding, ("sha256",)),
        (dataset, ("manifest_sha256", "rows_sha256")),
        (source_index, ("sha256",)),
        (render_audit, ("sha256", "content_sha256")),
        (geometry, ("sha256",)),
        (assignment, ("sha256",)),
    ):
        for field in fields:
            _exact_sha256(record.get(field), name=field)
    for record, fields in (
        (dataset, ("manifest_path", "rows_path")),
        (source_index, ("path",)),
        (render_audit, ("path",)),
        (geometry, ("path",)),
    ):
        for field in fields:
            _exact_str(record.get(field), name=field)
    _exact_int(dataset.get("row_count"), name="dataset.row_count")
    source_scene_count = _exact_int(
        source_index.get("scene_count"), name="source_index.scene_count"
    )
    if dataset.get("schema") != DATASET_SCHEMA:
        raise AttitudeSidecarContractError("manifest dataset schema mismatch")

    role_counts = _exact_dict(assignment.get("row_counts"), name="role counts")
    if set(role_counts) != set(DATASET_ROLES):
        raise AttitudeSidecarContractError("manifest role counts are incomplete")
    roles = _exact_dict(manifest.get("roles"), name="roles")
    if set(roles) != set(DATASET_ROLES):
        raise AttitudeSidecarContractError("manifest roles are incomplete")
    for role in DATASET_ROLES:
        entry = _validate_role_entry(roles[role], role=role, manifest_path=manifest_path)
        if entry["row_count"] != _exact_int(role_counts[role], name=f"count {role}"):
            raise AttitudeSidecarContractError(f"role count mismatch: {role}")
    if sum(int(value) for value in role_counts.values()) != dataset["row_count"]:
        raise AttitudeSidecarContractError("manifest role counts do not sum to row count")

    source_frames = _exact_list(manifest.get("source_frames"), name="source frames")
    if len(source_frames) != source_scene_count:
        raise AttitudeSidecarContractError(
            "source frame record count differs from source index scene count"
        )
    prior_scene_hash = ""
    seen_scene_hashes: set[str] = set()
    seen_source_paths: set[str] = set()
    selected_counts: Counter[str] = Counter()
    for raw in source_frames:
        record = _exact_dict(raw, name="source frame record")
        _exact_keys(
            record,
            {
                "scene_id_sha256", "dataset_role", "path", "sha256",
                "source_episode_scene_id", "source_split",
                "scene_manifest_sha256", "frame_count", "selected_row_count",
                "byte_count",
            },
            name="source frame record",
        )
        scene_hash = _exact_sha256(record.get("scene_id_sha256"), name="scene hash")
        if scene_hash <= prior_scene_hash or scene_hash in seen_scene_hashes:
            raise AttitudeSidecarContractError("source frame records are reordered or duplicate")
        prior_scene_hash = scene_hash
        seen_scene_hashes.add(scene_hash)
        if _exact_str(record.get("dataset_role"), name="source role") not in DATASET_ROLES:
            raise AttitudeSidecarContractError("source frame role is invalid")
        if _exact_int(
            record.get("source_episode_scene_id"),
            name="source episode scene ID",
        ) < 0:
            raise AttitudeSidecarContractError("source episode scene ID is negative")
        _exact_str(record.get("source_split"), name="source split")
        _exact_sha256(
            record.get("scene_manifest_sha256"), name="source scene manifest hash"
        )
        source_path = _exact_str(record.get("path"), name="source frame path")
        if source_path in seen_source_paths:
            raise AttitudeSidecarContractError("source frame path is duplicated")
        seen_source_paths.add(source_path)
        _exact_sha256(record.get("sha256"), name="source frame hash")
        for field in ("frame_count", "selected_row_count", "byte_count"):
            if _exact_int(record.get(field), name=field) < 0:
                raise AttitudeSidecarContractError(f"{field} must be nonnegative")
        selected_counts[str(record["dataset_role"])] += int(
            record["selected_row_count"]
        )
    if {role: int(selected_counts[role]) for role in DATASET_ROLES} != role_counts:
        raise AttitudeSidecarContractError("source-frame selected counts mismatch")

    source_map_record = _exact_dict(manifest.get("source_map"), name="source map")
    _exact_keys(
        source_map_record,
        {"entries", "entry_count", "source_map_sha256"},
        name="source map",
    )
    source_entries = _exact_list(source_map_record.get("entries"), name="source entries")
    if _exact_int(source_map_record.get("entry_count"), name="source entry count") != len(
        source_entries
    ):
        raise AttitudeSidecarContractError("source-map entry count mismatch")
    if canonical_json_sha256(source_entries) != _exact_sha256(
        source_map_record.get("source_map_sha256"), name="source-map hash"
    ):
        raise AttitudeSidecarContractError("source-map content hash mismatch")
    source_roles: list[str] = []
    source_paths: set[str] = set()
    for raw in source_entries:
        entry = _exact_dict(raw, name="source entry")
        _exact_keys(entry, {"role", "path", "sha256"}, name="source entry")
        source_role = _exact_str(entry.get("role"), name="source role")
        source_path = _exact_str(entry.get("path"), name="source path")
        if source_role in source_roles or source_path in source_paths:
            raise AttitudeSidecarContractError("source-map role/path is duplicated")
        source_roles.append(source_role)
        source_paths.add(source_path)
        _exact_sha256(entry.get("sha256"), name="source hash")
    if source_roles != sorted(SIDECAR_SOURCE_ROLES):
        raise AttitudeSidecarContractError(
            "source-map roles are incomplete, unexpected, or reordered"
        )
    if contract is not None:
        actual_source_paths = {
            str(entry["role"]): _canonical_absolute_path(
                entry["path"], name=f"source-map path {entry['role']}"
            )
            for entry in source_entries
        }
        if actual_source_paths != dict(contract.source_map_paths):
            raise AttitudeSidecarContractError(
                "source-map paths differ from the build contract"
            )

    construction = _exact_dict(manifest.get("construction"), name="construction")
    _exact_keys(
        construction,
        {
            "workers", "worker_start_method", "maximum_workers",
            "native_threads_per_worker", "thread_environment", "scene_order",
            "row_order", "publication",
        },
        name="construction",
    )
    worker_count = _exact_int(construction.get("workers"), name="workers")
    if not 1 <= worker_count <= 6:
        raise AttitudeSidecarContractError("manifest worker count is outside [1,6]")
    if _exact_int(construction.get("maximum_workers"), name="maximum_workers") != 6:
        raise AttitudeSidecarContractError("manifest maximum worker count changed")
    if _exact_int(
        construction.get("native_threads_per_worker"), name="native threads"
    ) != 1:
        raise AttitudeSidecarContractError("native worker thread count changed")
    thread_environment = _exact_dict(
        construction.get("thread_environment"), name="thread environment"
    )
    if thread_environment != {name: "1" for name in _THREAD_ENV_NAMES}:
        raise AttitudeSidecarContractError("native thread environment is not capped")
    if construction.get("worker_start_method") != (
        "serial" if worker_count == 1 else "spawn"
    ):
        raise AttitudeSidecarContractError("worker start method mismatch")
    if (
        construction.get("scene_order") != "scene_id_ascending"
        or construction.get("row_order") != "global_row_ascending_within_role"
        or construction.get("publication")
        != "exclusive_directory_create_manifest_last"
    ):
        raise AttitudeSidecarContractError("canonical construction policy changed")

    ledger = _exact_dict(manifest.get("access_ledger"), name="access ledger")
    _exact_keys(
        ledger,
        {
            "schema", "ledger_semantics", "measurement_scope",
            "completed_read_events", "completed_role_write_events",
            "logical_output_artifacts", "publication_mode",
            "post_ledger_prepublication_source_guard",
            "manifest_hash_reopens_after_publication", "image_byte_opens",
            "label_shard_byte_opens", "depth_byte_opens",
            "model_artifact_opens", "g2_distribution_summaries",
            "g2_role_file_reopens_after_write", "source_metadata_only",
        },
        name="access ledger",
    )
    if ledger.get("schema") != SIDECAR_ACCESS_LEDGER_SCHEMA:
        raise AttitudeSidecarContractError("access ledger schema mismatch")
    if (
        ledger.get("ledger_semantics")
        != "observed_builder_wrapper_events_plus_worker_receipts_v1"
        or ledger.get("measurement_scope")
        != "builder_controlled_byte_opens_excluding_interpreter_module_loading"
        or ledger.get("publication_mode")
        != "retained_staging_and_output_dirfds_hardlink_noreplace_manifest_last"
    ):
        raise AttitudeSidecarContractError("access-ledger semantics changed")
    source_guard = _exact_dict(
        ledger.get("post_ledger_prepublication_source_guard"),
        name="post-ledger source guard",
    )
    _exact_keys(
        source_guard,
        {
            "implementation_manifest_open_count",
            "precommitted_source_open_count_each",
            "occurs_after_manifest_staging_before_publication",
        },
        name="post-ledger source guard",
    )
    if (
        _exact_int(
            source_guard.get("implementation_manifest_open_count"),
            name="guard implementation opens",
        )
        != 1
        or _exact_int(
            source_guard.get("precommitted_source_open_count_each"),
            name="guard source opens",
        )
        != 1
        or _exact_bool(
            source_guard.get("occurs_after_manifest_staging_before_publication"),
            name="guard ordering",
        )
        is not True
    ):
        raise AttitudeSidecarContractError("post-ledger source guard changed")
    for field in (
        "manifest_hash_reopens_after_publication",
        "image_byte_opens", "label_shard_byte_opens", "depth_byte_opens",
        "model_artifact_opens", "g2_distribution_summaries",
        "g2_role_file_reopens_after_write",
    ):
        if _exact_int(ledger.get(field), name=field) != 0:
            raise AttitudeSidecarContractError(f"forbidden access counter is nonzero: {field}")
    if _exact_bool(ledger.get("source_metadata_only"), name="source_metadata_only") is not True:
        raise AttitudeSidecarContractError("sidecar construction was not metadata-only")
    completed_reads = _exact_list(
        ledger.get("completed_read_events"), name="completed read events"
    )
    normalized_reads: dict[str, dict[str, Any]] = {}
    read_paths_in_order: list[str] = []
    for raw in completed_reads:
        entry = _exact_dict(raw, name="completed read event")
        _exact_keys(
            entry,
            {
                "path", "sha256", "purposes", "open_count", "byte_count",
                "total_bytes_read",
            },
            name="completed read event",
        )
        path = str(
            _canonical_absolute_path(
                _exact_str(entry.get("path"), name="completed read path"),
                name="completed read path",
            )
        )
        digest = _exact_sha256(entry.get("sha256"), name="completed read hash")
        purposes = _exact_list(entry.get("purposes"), name="completed read purposes")
        if any(type(value) is not str for value in purposes) or purposes != sorted(
            set(purposes)
        ):
            raise AttitudeSidecarContractError("completed read purposes are noncanonical")
        open_count = _exact_int(entry.get("open_count"), name="open count")
        byte_count = _exact_int(entry.get("byte_count"), name="byte count")
        total_bytes = _exact_int(
            entry.get("total_bytes_read"), name="total bytes read"
        )
        if open_count <= 0 or byte_count <= 0 or total_bytes != open_count * byte_count:
            raise AttitudeSidecarContractError("completed read counts do not reconcile")
        if path in normalized_reads:
            raise AttitudeSidecarContractError("completed read path is duplicated")
        normalized_reads[path] = {
            "sha256": digest,
            "open_count": open_count,
            "purposes": purposes,
            "byte_count": byte_count,
        }
        read_paths_in_order.append(path)
    if read_paths_in_order != sorted(read_paths_in_order):
        raise AttitudeSidecarContractError("completed reads are reordered")
    required_reads = {
        str(_canonical_absolute_path(binding["path"], name="binding path")): {
            "sha256": binding["sha256"], "open_count": 4,
            "purposes": [
                "execution_binding_contract",
                "implementation_source_validation:binding",
                "source_map:binding",
            ],
        },
        str(
            _canonical_absolute_path(
                dataset["manifest_path"], name="dataset manifest path"
            )
        ): {
            "sha256": dataset["manifest_sha256"],
            "open_count": 1,
            "purposes": ["dataset manifest"],
        },
        str(
            _canonical_absolute_path(dataset["rows_path"], name="dataset rows path")
        ): {
            "sha256": dataset["rows_sha256"],
            "open_count": 1,
            "purposes": ["dataset_rows_metadata"],
        },
        str(
            _canonical_absolute_path(source_index["path"], name="source index path")
        ): {
            "sha256": source_index["sha256"],
            "open_count": 1,
            "purposes": ["source_index"],
        },
        str(
            _canonical_absolute_path(render_audit["path"], name="render audit path")
        ): {
            "sha256": render_audit["sha256"],
            "open_count": 1,
            "purposes": ["render audit"],
        },
        str(
            _canonical_absolute_path(geometry["path"], name="dynamic geometry path")
        ): {
            "sha256": geometry["sha256"],
            "open_count": 4,
            "purposes": [
                "dynamic_geometry_contract",
                "implementation_source_validation:dynamic_geometry",
                "source_map:dynamic_geometry",
            ],
        },
    }
    for record in source_frames:
        required_reads[
            str(_canonical_absolute_path(record["path"], name="source frame path"))
        ] = {
            "sha256": record["sha256"],
            "open_count": 1,
            "purposes": [f"source_frames_metadata:{record['dataset_role']}"],
            "byte_count": record["byte_count"],
        }
    for entry in source_entries:
        path = str(_canonical_absolute_path(entry["path"], name="source-map path"))
        if entry["role"] not in ("binding", "dynamic_geometry"):
            role = entry["role"]
            purposes = [f"source_map:{role}"]
            if role == "implementation_manifest":
                purposes.append("implementation_manifest_validation")
            else:
                purposes.append(f"implementation_source_validation:{role}")
            required_reads[path] = {
                "sha256": entry["sha256"],
                "open_count": 3,
                "purposes": sorted(purposes),
            }
    if set(normalized_reads) != set(required_reads):
        raise AttitudeSidecarContractError("completed-read graph paths are not exact")
    for path, expected_read in required_reads.items():
        actual_read = normalized_reads[path]
        for field in ("sha256", "open_count", "purposes"):
            if actual_read[field] != expected_read[field]:
                raise AttitudeSidecarContractError(
                    f"completed-read graph {field} is not exact: {path}"
                )
        if "byte_count" in expected_read and (
            actual_read["byte_count"] != expected_read["byte_count"]
        ):
            raise AttitudeSidecarContractError(
                f"completed-read byte count is not exact: {path}"
            )

    role_write_events = _exact_list(
        ledger.get("completed_role_write_events"), name="role write events"
    )
    if len(role_write_events) != len(DATASET_ROLES):
        raise AttitudeSidecarContractError("role write event count mismatch")
    for role, raw in zip(DATASET_ROLES, role_write_events, strict=True):
        event = _exact_dict(raw, name="role write event")
        _exact_keys(
            event,
            {"dataset_role", "logical_path", "file_sha256", "byte_count", "physical_mode"},
            name="role write event",
        )
        if (
            event.get("dataset_role") != role
            or _canonical_absolute_path(
                event.get("logical_path"), name="role logical path"
            )
            != manifest_path.parent / ROLE_FILE_NAMES[role]
            or _exact_sha256(event.get("file_sha256"), name="role write hash")
            != roles[role]["file_sha256"]
            or _exact_int(event.get("byte_count"), name="role write byte count") <= 0
            or event.get("physical_mode")
            != "private_staging_then_hardlink_noreplace"
        ):
            raise AttitudeSidecarContractError("role write event mismatch")

    logical_outputs = _exact_list(
        ledger.get("logical_output_artifacts"), name="logical output artifacts"
    )
    expected_writes = [
        str(
            _canonical_absolute_path(
                manifest_path.parent / ROLE_FILE_NAMES[role],
                name=f"expected {role} write path",
            )
        )
        for role in DATASET_ROLES
    ] + [str(_canonical_absolute_path(manifest_path, name="manifest write path"))]
    if logical_outputs != expected_writes:
        raise AttitudeSidecarContractError("logical output graph is not exact")

    if contract is not None:
        expected = {
            "binding": contract.binding_sha256,
            "dataset_manifest": contract.dataset_manifest_sha256,
            "dataset_rows": contract.dataset_rows_sha256,
            "source_index": contract.source_index_sha256,
            "render_audit": contract.render_audit_sha256,
            "dynamic_geometry": contract.dynamic_geometry_sha256,
            "role_assignment": contract.role_assignment_sha256,
        }
        actual = {
            "binding": binding["sha256"],
            "dataset_manifest": dataset["manifest_sha256"],
            "dataset_rows": dataset["rows_sha256"],
            "source_index": source_index["sha256"],
            "render_audit": render_audit["sha256"],
            "dynamic_geometry": geometry["sha256"],
            "role_assignment": assignment["sha256"],
        }
        if (
            actual != expected
            or role_counts != dict(contract.role_counts)
            or _canonical_absolute_path(binding["path"], name="binding path")
            != contract.binding_path
        ):
            raise AttitudeSidecarContractError("sidecar immutable binding mismatch")
    return manifest


def g2_sidecar_attempt_id(
    *,
    sidecar_manifest_sha256: str,
    dataset_manifest_sha256: str,
    source_checkpoint_sha256: str,
    g2_role_file_sha256: str,
) -> str:
    identity = {
        "schema": "lewm_go2_attitude_sidecar_g2_attempt_identity_v1",
        "sidecar_manifest_sha256": _exact_sha256(
            sidecar_manifest_sha256, name="sidecar manifest hash"
        ),
        "dataset_manifest_sha256": _exact_sha256(
            dataset_manifest_sha256, name="dataset manifest hash"
        ),
        "source_checkpoint_sha256": _exact_sha256(
            source_checkpoint_sha256, name="source checkpoint hash"
        ),
        "g2_role_file_sha256": _exact_sha256(
            g2_role_file_sha256, name="G2 role file hash"
        ),
    }
    return canonical_json_sha256(identity)


def g2_sidecar_attempt_path(
    source_checkpoint_path: Path,
    attempt_id_sha256: str,
) -> Path:
    checkpoint_path = _canonical_absolute_path(
        source_checkpoint_path, name="G2 source checkpoint path"
    )
    attempt_id = _exact_sha256(attempt_id_sha256, name="G2 attempt ID")
    return checkpoint_path.with_name(
        checkpoint_path.name + f".{attempt_id}.g2_sidecar_attempt.json"
    )


def g2_sidecar_receipt_path(
    source_checkpoint_path: Path,
    attempt_id_sha256: str,
) -> Path:
    checkpoint_path = _canonical_absolute_path(
        source_checkpoint_path, name="G2 source checkpoint path"
    )
    attempt_id = _exact_sha256(attempt_id_sha256, name="G2 attempt ID")
    return checkpoint_path.with_name(
        checkpoint_path.name + f".{attempt_id}.g2_sidecar_opened.json"
    )


def _validate_g2_attempt_marker(
    data: bytes,
    *,
    marker_path: Path,
    marker_sha256: str,
    manifest_path: Path,
    manifest_sha256: str,
    manifest: Mapping[str, Any],
    source_checkpoint_path: Path,
    source_checkpoint_sha256: str,
) -> dict[str, Any]:
    marker = _parse_json_bytes(data, name="G2 attempt marker")
    expected_bytes = (
        json.dumps(marker, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
        + b"\n"
    )
    if data != expected_bytes:
        raise AttitudeSidecarContractError("G2 attempt marker JSON is not canonical")
    _exact_keys(marker, _G2_ATTEMPT_KEYS, name="G2 attempt marker")
    if marker.get("schema") != SIDECAR_G2_ATTEMPT_SCHEMA:
        raise AttitudeSidecarContractError("unsupported G2 attempt marker schema")
    _validate_content_hash(marker, name="G2 attempt marker")
    created_at = _exact_str(marker.get("created_at_utc"), name="created_at_utc")
    try:
        parsed_time = datetime.fromisoformat(created_at)
    except ValueError as exc:
        raise AttitudeSidecarContractError("G2 attempt timestamp is invalid") from exc
    if parsed_time.tzinfo != timezone.utc:
        raise AttitudeSidecarContractError("G2 attempt timestamp must use UTC")
    if marker.get("intent") != "open_exact_untouched_g2_sidecar_once":
        raise AttitudeSidecarContractError("G2 attempt intent mismatch")
    if marker.get("status") != "committed_before_g2_sidecar_open":
        raise AttitudeSidecarContractError("G2 attempt status mismatch")

    sidecar_binding = _exact_dict(
        marker.get("sidecar_manifest"), name="attempt sidecar manifest"
    )
    dataset_binding = _exact_dict(
        marker.get("dataset_manifest"), name="attempt dataset manifest"
    )
    checkpoint_binding = _exact_dict(
        marker.get("source_checkpoint"), name="attempt source checkpoint"
    )
    role_binding = _exact_dict(marker.get("g2_role"), name="attempt G2 role")
    for name, value in (
        ("attempt sidecar manifest", sidecar_binding),
        ("attempt dataset manifest", dataset_binding),
        ("attempt source checkpoint", checkpoint_binding),
    ):
        _exact_keys(value, {"path", "sha256"}, name=name)
        _canonical_absolute_path(
            _exact_str(value.get("path"), name=f"{name}.path"),
            name=f"{name}.path",
        )
        _exact_sha256(value.get("sha256"), name=f"{name}.sha256")
    _exact_keys(
        role_binding,
        {"path", "file_sha256", "row_count", "ordered_global_rows_sha256"},
        name="attempt G2 role",
    )
    _canonical_absolute_path(
        _exact_str(role_binding.get("path"), name="attempt G2 role.path"),
        name="attempt G2 role.path",
    )
    _exact_sha256(
        role_binding.get("file_sha256"),
        name="attempt G2 role.file_sha256",
    )
    if _exact_int(role_binding.get("row_count"), name="attempt G2 role.row_count") < 0:
        raise AttitudeSidecarContractError("attempt G2 role row count is negative")
    _exact_sha256(
        role_binding.get("ordered_global_rows_sha256"),
        name="attempt G2 role.ordered_global_rows_sha256",
    )

    expected_sidecar = {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
    }
    expected_dataset = {
        "path": str(
            _canonical_absolute_path(
                manifest["dataset"]["manifest_path"],
                name="dataset manifest path",
            )
        ),
        "sha256": manifest["dataset"]["manifest_sha256"],
    }
    expected_checkpoint = {
        "path": str(source_checkpoint_path),
        "sha256": source_checkpoint_sha256,
    }
    g2_entry = manifest["roles"]["g2_evaluation"]
    expected_role = {
        "path": str(
            _canonical_absolute_path(g2_entry["path"], name="G2 role path")
        ),
        "file_sha256": g2_entry["file_sha256"],
        "row_count": g2_entry["row_count"],
        "ordered_global_rows_sha256": g2_entry["ordered_global_rows_sha256"],
    }
    expected_attempt_id = g2_sidecar_attempt_id(
        sidecar_manifest_sha256=manifest_sha256,
        dataset_manifest_sha256=manifest["dataset"]["manifest_sha256"],
        source_checkpoint_sha256=source_checkpoint_sha256,
        g2_role_file_sha256=g2_entry["file_sha256"],
    )
    if _exact_sha256(marker.get("attempt_id_sha256"), name="G2 attempt ID") != (
        expected_attempt_id
    ):
        raise AttitudeSidecarContractError("G2 attempt ID mismatch")
    expected_marker_path = g2_sidecar_attempt_path(
        source_checkpoint_path,
        expected_attempt_id,
    )
    if marker_path != expected_marker_path or _canonical_absolute_path(
        _exact_str(marker.get("attempt_marker_path"), name="attempt marker path"),
        name="attempt marker path",
    ) != expected_marker_path:
        raise AttitudeSidecarContractError("G2 attempt marker path is noncanonical")
    if sidecar_binding != expected_sidecar:
        raise AttitudeSidecarContractError("G2 attempt binds a different sidecar")
    if dataset_binding != expected_dataset:
        raise AttitudeSidecarContractError("G2 attempt binds a different dataset")
    if checkpoint_binding != expected_checkpoint:
        raise AttitudeSidecarContractError("G2 attempt binds a different checkpoint")
    if role_binding != expected_role:
        raise AttitudeSidecarContractError("G2 attempt binds a different G2 role")
    if sha256_file(source_checkpoint_path) != source_checkpoint_sha256:
        raise AttitudeSidecarContractError("G2 source checkpoint SHA-256 mismatch")

    receipt_core = {
        "schema": SIDECAR_G2_RECEIPT_SCHEMA,
        "attempt_marker": {"path": str(marker_path), "sha256": marker_sha256},
        "sidecar_manifest": expected_sidecar,
        "source_checkpoint": expected_checkpoint,
        "g2_role": expected_role,
        "status": "committed_before_g2_role_file_open",
    }
    receipt = {
        **receipt_core,
        "content_sha256": canonical_json_sha256(receipt_core),
    }
    receipt_path = g2_sidecar_receipt_path(
        source_checkpoint_path,
        expected_attempt_id,
    )
    try:
        _write_json_exclusive(receipt_path, receipt)
    except FileExistsError as exc:
        raise AttitudeSidecarAccessError(
            "G2 attempt was already consumed by a sidecar open"
        ) from exc
    _fsync_directory(receipt_path.parent)
    return marker


def load_attitude_sidecar_roles(
    manifest_path: Path,
    *,
    roles: Iterable[str],
    expected_manifest_sha256: str,
    contract: AttitudeSidecarBuildContract,
    g2_attempt_marker_path: Path | None = None,
    expected_g2_attempt_marker_sha256: str | None = None,
    g2_source_checkpoint_path: Path | None = None,
    expected_g2_source_checkpoint_sha256: str | None = None,
) -> dict[str, tuple[dict[str, Any], ...]]:
    """Load only authorized role files, requiring a marker before any G2 open."""

    requested = tuple(dict.fromkeys(str(role) for role in roles))
    if not requested or set(requested) - set(DATASET_ROLES):
        raise AttitudeSidecarAccessError(f"invalid sidecar role request: {requested}")
    manifest_path = _canonical_absolute_path(manifest_path, name="sidecar manifest path")
    manifest_sha = _exact_sha256(expected_manifest_sha256, name="manifest hash")
    manifest_data = _regular_file_bytes(
        manifest_path,
        expected_sha256=manifest_sha,
        name="sidecar manifest",
    )
    parsed_manifest = _parse_json_bytes(manifest_data, name="sidecar manifest")
    expected_manifest_bytes = (
        json.dumps(parsed_manifest, sort_keys=True, indent=2, allow_nan=False).encode(
            "utf-8"
        )
        + b"\n"
    )
    if manifest_data != expected_manifest_bytes:
        raise AttitudeSidecarContractError("sidecar manifest JSON is not canonical")
    manifest = validate_attitude_sidecar_manifest(
        parsed_manifest,
        manifest_path=manifest_path,
        contract=contract,
    )
    if "g2_evaluation" in requested:
        if (
            g2_attempt_marker_path is None
            or expected_g2_attempt_marker_sha256 is None
            or g2_source_checkpoint_path is None
            or expected_g2_source_checkpoint_sha256 is None
        ):
            raise AttitudeSidecarAccessError(
                "G2 sidecar access requires a bound checkpoint and attempt marker"
            )
        marker_sha = _exact_sha256(
            expected_g2_attempt_marker_sha256, name="G2 attempt marker hash"
        )
        marker_path = _canonical_absolute_path(
            g2_attempt_marker_path, name="G2 attempt marker path"
        )
        checkpoint_path = _canonical_absolute_path(
            g2_source_checkpoint_path, name="G2 source checkpoint path"
        )
        checkpoint_sha = _exact_sha256(
            expected_g2_source_checkpoint_sha256,
            name="G2 source checkpoint hash",
        )
        marker_data = _regular_file_bytes(
            marker_path,
            expected_sha256=marker_sha,
            name="G2 attempt marker",
        )
        _validate_g2_attempt_marker(
            marker_data,
            marker_path=marker_path,
            marker_sha256=marker_sha,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            manifest=manifest,
            source_checkpoint_path=checkpoint_path,
            source_checkpoint_sha256=checkpoint_sha,
        )

    loaded: dict[str, tuple[dict[str, Any], ...]] = {}
    for role in requested:
        entry = manifest["roles"][role]
        role_path = _canonical_absolute_path(entry["path"], name=f"{role} sidecar path")
        data = _regular_file_bytes(
            role_path,
            expected_sha256=entry["file_sha256"],
            name=f"{role} sidecar",
        )
        rows = _parse_jsonl_bytes(data, name=f"{role} sidecar")
        expected_bytes = b"".join(
            canonical_json_bytes(row) + b"\n" for row in rows
        )
        if data != expected_bytes:
            raise AttitudeSidecarContractError(f"{role} JSONL is not canonical")
        normalized = _canonical_role_rows(
            rows,
            role=role,
            expected_count=entry["row_count"],
        )
        if canonical_json_sha256(normalized) != entry["content_sha256"]:
            raise AttitudeSidecarContractError(f"{role} content hash mismatch")
        if canonical_json_sha256(
            [row["row_identity_sha256"] for row in normalized]
        ) != entry["ordered_identity_sha256"]:
            raise AttitudeSidecarContractError(f"{role} ordered identity hash mismatch")
        if canonical_json_sha256(
            [row["global_row"] for row in normalized]
        ) != entry["ordered_global_rows_sha256"]:
            raise AttitudeSidecarContractError(f"{role} global-row hash mismatch")
        loaded[role] = tuple(normalized)
    return loaded


__all__ = [
    "AttitudeSidecarAccessError",
    "AttitudeSidecarBuildContract",
    "AttitudeSidecarContractError",
    "DATASET_ROLES",
    "FROZEN_BUILD_CONTRACT",
    "ROLE_FILE_NAMES",
    "SIDECAR_MANIFEST_SCHEMA",
    "SIDECAR_ROW_SCHEMA",
    "SIDECAR_G2_ATTEMPT_SCHEMA",
    "SIDECAR_G2_RECEIPT_SCHEMA",
    "SIDECAR_IMPLEMENTATION_MANIFEST_SCHEMA",
    "SIDECAR_IMPLEMENTATION_TEST_COMMAND",
    "SIDECAR_IMPLEMENTATION_TEST_COUNT",
    "SIDECAR_PRECOMMITTED_SOURCE_ROLES",
    "SIDECAR_SOURCE_ROLES",
    "build_attitude_sidecar",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "g2_sidecar_attempt_id",
    "g2_sidecar_attempt_path",
    "g2_sidecar_receipt_path",
    "load_attitude_sidecar_roles",
    "row_identity",
    "row_identity_sha256",
    "sha256_file",
    "validate_attitude_sidecar_manifest",
    "validate_attitude_sidecar_implementation_manifest",
    "validate_sidecar_row",
]
