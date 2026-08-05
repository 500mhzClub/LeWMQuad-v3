"""Independent auditor for the Shared-JEPA V5 raw-supervision dataset.

The auditor deliberately does not import the Shared-JEPA V5 builder.  Its
filesystem, JSON, join, array, and sampling contracts are repeated literally
from the frozen preregistration.  Exact mode replays the precommitted sample
through the older reviewed V4 geometry/raycast implementation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import ctypes
import errno
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "lewm_worlds") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "lewm_worlds"))

from lewm.benchmarks import go2_dynamic_cell_square_projection as dynamic_projection
from lewm.benchmarks import go2_observable_camera_ray_evidence_v4 as evidence_v4
from lewm.datasets import go2_shared_jepa_v5_raw_supervision_plan_v5 as plan_v5
from lewm_worlds.manifest import manifest_sha256, parse_scene_manifest_dict
from scripts import audit_go2_n32_camera_frustum_observability as source_v4
from scripts import build_go2_observable_camera_ray_fit_v4 as raycast_v4


ROOT = _REPOSITORY_ROOT
CANONICAL_DATASET = (
    ROOT
    / ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "development_raw_supervision_v1"
)
CANONICAL_AUDIT_REPORT = CANONICAL_DATASET.with_name(
    CANONICAL_DATASET.name + ".audit.json"
)
CANONICAL_AUDIT_FAILURE = CANONICAL_DATASET.with_name(
    CANONICAL_DATASET.name + ".audit.failed.json"
)
DATASET_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_dataset_v1"
SHARD_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_shard_v1"
ENDPOINT_INDEX_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_endpoint_index_v1"
AUDIT_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_audit_v1"
MAX_WORKERS = 6
EXPECTED_SAMPLE_COUNT = 24

ACCESS_LEDGER_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_access_ledger_v1"
GEOMETRY_CONTRACT_PATH = ROOT / "config/go2_generalization_geometry_v2.json"
GEOMETRY_CONTRACT_FILE_SHA256 = (
    "e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52"
)
GEOMETRY_CONTRACT_CONTENT_SHA256 = (
    "e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca"
)
RENDER_AUDIT_PATH = ROOT / ".generated/go2_render_selected_v04/audit_report.json"
RENDER_AUDIT_FILE_SHA256 = (
    "9a045dff82fb82adbbb89d10cb4dc0063297805038b000e5f6cd53816e995a9a"
)
RENDER_AUDIT_CONTENT_SHA256 = (
    "c9280ed4cab9ff54f7d8684835b8448886209a8cc50eba3588519c34572a6358"
)
BUILD_AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_2026-07-13.json"
)
EXACT_PLAN_CONTENT_SHA256 = (
    "8004ab0d3aa6a2f5d576ba0ff4d6a75f50899152e542dc62b8d6e35f614921a3"
)
EXACT_ORDERED_PAIR_SHA256 = (
    "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
)
EXACT_ORDERED_ENDPOINT_SHA256 = (
    "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
)

FROZEN_PARENT_FILE_SHA256 = {
    "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_preregistration_2026-07-13.md": (
        "07a51661f7d86391bda8974799a881287ccace8083fadf396e5c01b6345ed3bb"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_builder_source_inventory_amendment_2026-07-13.md": (
        "39dd1eda32bdcac12a1573fbf3d7d2c7547fa4d7b0cd30e4da3b8a0d47aaf2f3"
    ),
    "lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan_v5.py": (
        "67c4d325ddab3ac3405e231b78681f4b9ef17b4833ca199395f24ed7a8b82921"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_author_handoff_2026-07-13.md": (
        "b362d26372f01e670a477dda5e7abb5e55370cc1d8d89052545afa229e7bba66"
    ),
    "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_plan_v5_independent_qa.py": (
        "8a50bcf5275d243f06b92264e017f355fd54faaca8f8e73aab1e3cc45dc51298"
    ),
    "docs/lewm_go2_shared_jepa_v5_raw_supervision_metadata_plan_v5_independent_review_2026-07-13.md": (
        "7d7344e423492a3cf36d1cd50ca09e6c7eb6eba17c25861c840531465aaf7706"
    ),
}
REVIEWED_V4_SOURCE_SHA256 = {
    "lewm/benchmarks/go2_observable_camera_ray_evidence_v4.py": (
        "708d368e461fe60aacb860dda5b0cbfd1acaf43e5cb3ae18a77bb48de739fb85"
    ),
    "scripts/build_go2_observable_camera_ray_fit_v4.py": (
        "4efb0517130df39a1953539755d82289b16e89b314bba5713d6d9d944acf1d16"
    ),
    "scripts/audit_go2_n32_camera_frustum_observability.py": (
        "f7e3a3e60937caabbe003ff41af6aec44248df137b0a53c383364272152f3079"
    ),
    "lewm/benchmarks/go2_dynamic_cell_square_projection.py": (
        "ce2bb0d38ed1436635cdd1468ba1dfe1a935fdafdd6dda5adcf37b97a32a74bf"
    ),
    "lewm_worlds/lewm_worlds/manifest.py": (
        "5679768016226e89e385ec7a7238616416248a9a1194b898ecb9078662f6a888"
    ),
    "lewm/benchmarks/go2_n32_camera_frustum_observability.py": (
        "ab97c34a8a07a93d6b49b5adb0b1a82bc66d38be206baab362b7b1f1b59f3cc3"
    ),
    "lewm/datasets/go2_paired_navigation.py": (
        "14df0cf59ab7554431b1be2ef91e3ab7229200be94bb9afa88127e3ea53c2c08"
    ),
    "lewm/planning/geometry_contract.py": (
        "6873a9550399a5decc90e4a31b2945e54074bdb56855a035924f49b4511c813b"
    ),
    "lewm_worlds/lewm_worlds/planning_grid.py": (
        "e6f7e26d584dfd7923493803fc95a75135122b37a1f95cb51f9267b284649510"
    ),
}

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
ACCELERATOR_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
)

ARRAY_LAYOUT = (
    ("camera_origin_body_m.f4", "<f4", (3,)),
    ("camera_basis_body_fru.f4", "<f4", (3, 3)),
    ("ground_plane_z_body_m.f4", "<f4", ()),
    ("ground_support_in_frustum.u1", "|u1", (128, 128, 5)),
    ("ground_support_clear_to_target.u1", "|u1", (128, 128, 5)),
    ("pixel_hit_mask.u1", "|u1", (84, 112)),
    ("pixel_first_hit_distance_m.f4", "<f4", (84, 112)),
    ("raster_labels.u1", "|u1", (64, 64)),
)

MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "status",
        "evidence_schema",
        "raster_schema",
        "roles",
        "pair_counts",
        "endpoint_instance_count",
        "unique_endpoint_counts",
        "scene_shard_count",
        "ordered_pair_sha256",
        "ordered_endpoint_sha256",
        "pair_index",
        "endpoint_index",
        "array_layout",
        "shards",
        "files",
        "input_provenance",
        "access_ledger",
        "independent_audit_precommit",
        "parallel_contract",
        "publication",
        "licenses",
        "content_sha256",
    }
)
SHARD_FIELDS = frozenset(
    {
        "schema",
        "dataset_role",
        "family",
        "scene_id",
        "scene_id_sha256",
        "endpoint_count",
        "ordered_endpoint_identity_sha256",
        "ordered_evidence_sha256",
        "ordered_raster_sha256",
        "files",
        "content_sha256",
    }
)
SHARD_INDEX_FIELDS = frozenset(
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
    }
)
TOP_ENDPOINT_FIELDS = frozenset((*SHARD_INDEX_FIELDS, "scene_shard"))
ROOT_FILE_FIELDS = frozenset({"path", "byte_count", "file_sha256"})
SHARD_FILE_FIELDS = frozenset(
    {"path", "byte_count", "file_sha256", "dtype", "shape"}
)

FALSE_LICENSE_FIELDS = frozenset(
    {
        "independent_audit_passed",
        "dataset_use_authorized",
        "rgb_decode_authorized",
        "training_authorized",
        "selection_authorized",
        "calibration_authorized",
        "g2_authorized",
        "heldout_authorized",
        "runtime_authorized",
        "hardware_authorized",
        "production_authorized",
        "promotion_authorized",
    }
)
FORBIDDEN_LEDGER_FRAGMENTS = (
    "rgb_byte_open",
    "rgb_decode",
    "label_shard_payload_open",
    "g2_payload_open",
    "g2_geometry",
    "g2_label",
    "g2_rgb",
    "checkpoint",
    "model_output",
    "runtime",
    "navigation_result",
    "heldout",
    "sealed",
    "hardware",
    "production",
)
EXACT_ACCESS_LEDGER_KEYS = frozenset(
    {
        "schema",
        "measurement_scope",
        "metadata_plan_first_pass",
        "metadata_source_inventory_first_pass",
        "metadata_plan_second_pass",
        "metadata_source_inventory_second_pass",
        "development_scene_workers",
        "unique_endpoint_raycasts",
        "pair_endpoint_references",
        "source_frames_jsonl_records_scanned",
        "source_frames_selected_records",
        "source_frames_byte_opens",
        "source_scene_manifest_byte_opens",
        "render_plan_byte_opens",
        "render_summary_byte_opens",
        "geometry_contract_byte_opens",
        "render_audit_byte_opens",
        "source_payload_first_pass_file_count",
        "source_payload_second_pass_file_count",
        "source_payload_total_byte_opens",
        "g2_source_index_rows_read_for_exclusion",
        "g2_sidecar_byte_opens",
        "g2_source_payload_opens",
        "g2_label_payload_opens",
        "g2_rgb_byte_opens",
        "rgb_byte_opens",
        "rgb_decodes",
        "parent_label_shard_payload_opens",
        "checkpoint_or_model_output_opens",
        "runtime_or_navigation_result_opens",
        "heldout_or_sealed_opens",
        "hardware_or_production_opens",
        "writes_outside_output_or_failure_namespace",
        "denied_or_unexpected_accesses",
    }
)
EXACT_INPUT_PROVENANCE_FIELDS = frozenset(
    {
        "authorization_file_sha256",
        "authorization_content_sha256",
        "authorization_source_map_sha256",
        "frozen_parent_file_sha256",
        "reviewed_v4_source_sha256",
        "metadata_plan_content_sha256",
        "metadata_ordered_pair_sha256",
        "metadata_ordered_endpoint_sha256",
        "source_inventory_sha256",
        "source_payload_inventory",
        "source_payload_inventory_sha256",
        "geometry_contract_file_sha256",
        "geometry_contract_content_sha256",
        "render_audit_file_sha256",
        "render_audit_content_sha256",
    }
)


class RawSupervisionAuditError(RuntimeError):
    """Raised when the immutable raw-supervision artifact is not exact."""


@dataclass(frozen=True)
class StoredEndpointEvidence:
    endpoint_identity_sha256: str
    arrays: tuple[np.ndarray, ...]
    evidence_content_sha256: str
    raster_content_sha256: str


@dataclass(frozen=True)
class AuditInputs:
    plan: plan_v5.DevelopmentRawSupervisionPlan
    inventory: plan_v5.DevelopmentSourceInventory


SampleRecomputer = Callable[
    [Sequence[Mapping[str, Any]], Mapping[str, Mapping[str, Any]], AuditInputs, int],
    Mapping[str, tuple[np.ndarray, ...]],
]


Fingerprint = tuple[int, int, int, int, int, int, int]


def _fingerprint(metadata: os.stat_result) -> Fingerprint:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _directory_open_flags() -> int:
    if not getattr(os, "O_DIRECTORY", 0) or not getattr(os, "O_NOFOLLOW", 0):
        raise RawSupervisionAuditError(
            "descriptor-relative no-follow directories are unavailable"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


@dataclass
class _RetainedDirectoryChain:
    absolute_path: Path
    descriptors: list[int]
    entries: list[tuple[int, str, int, Fingerprint]]
    anchor_fingerprint: Fingerprint

    @property
    def directory_fd(self) -> int:
        return self.descriptors[-1]

    def validate(self, *, allow_final_metadata_change: bool = False) -> None:
        if _fingerprint(os.fstat(self.descriptors[0])) != self.anchor_fingerprint:
            raise RawSupervisionAuditError("filesystem root changed during publication")
        final_index = len(self.entries) - 1
        for index, (parent_fd, component, child_fd, expected) in enumerate(self.entries):
            try:
                named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                opened = os.fstat(child_fd)
            except (FileNotFoundError, NotADirectoryError, OSError) as error:
                raise RawSupervisionAuditError(
                    "audit output directory chain changed"
                ) from error
            named_fingerprint = _fingerprint(named)
            opened_fingerprint = _fingerprint(opened)
            # Publishing a leaf legitimately changes only its immediate
            # directory's size/timestamps.  Keep identity, type, mode, and
            # link count stable there; retain complete fingerprints above it.
            expected_matches = (
                named_fingerprint[:4] == expected[:4]
                and opened_fingerprint[:4] == expected[:4]
                if index == final_index and allow_final_metadata_change
                else named_fingerprint == expected
                and opened_fingerprint == expected
            )
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or not expected_matches
            ):
                raise RawSupervisionAuditError("audit output directory chain changed")

    def close(self) -> None:
        for descriptor in reversed(self.descriptors):
            os.close(descriptor)
        self.descriptors.clear()

    def __enter__(self) -> "_RetainedDirectoryChain":
        self.validate()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _open_retained_directory_chain(path: Path) -> _RetainedDirectoryChain:
    absolute = Path(path).absolute()
    if not absolute.is_absolute() or not absolute.anchor:
        raise RawSupervisionAuditError("audit output parent must be absolute")
    descriptors: list[int] = []
    entries: list[tuple[int, str, int, Fingerprint]] = []
    try:
        filesystem_root = Path(absolute.anchor)
        anchor_before = filesystem_root.stat(follow_symlinks=False)
        anchor_fd = os.open(filesystem_root, _directory_open_flags())
        descriptors.append(anchor_fd)
        anchor_fingerprint = _fingerprint(anchor_before)
        if _fingerprint(os.fstat(anchor_fd)) != anchor_fingerprint:
            raise RawSupervisionAuditError("filesystem root changed during open")
        parent_fd = anchor_fd
        for component in absolute.parts[1:]:
            named = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(named.st_mode) or not stat.S_ISDIR(named.st_mode):
                raise PermissionError("audit output parent contains an alias")
            expected = _fingerprint(named)
            child_fd = os.open(component, _directory_open_flags(), dir_fd=parent_fd)
            descriptors.append(child_fd)
            if _fingerprint(os.fstat(child_fd)) != expected:
                raise RawSupervisionAuditError(
                    "audit output parent changed during descriptor open"
                )
            entries.append((parent_fd, component, child_fd, expected))
            parent_fd = child_fd
        chain = _RetainedDirectoryChain(
            absolute_path=absolute,
            descriptors=descriptors,
            entries=entries,
            anchor_fingerprint=anchor_fingerprint,
        )
        chain.validate()
        return chain
    except BaseException:
        for descriptor in reversed(descriptors):
            os.close(descriptor)
        raise


def _lstat_optional_at(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _read_absolute_bound_payload(
    path: Path,
    expected_sha256: str,
    *,
    repository_root: Path,
    name: str,
) -> bytes:
    """Read one allowlisted absolute file through a retained no-follow chain."""

    lexical = Path(path)
    root = Path(repository_root).absolute()
    if not lexical.is_absolute() or lexical != Path(os.path.normpath(str(lexical))):
        raise PermissionError(f"{name} path must be canonical and absolute")
    try:
        lexical.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"{name} escapes the repository") from error
    if not _is_sha256(expected_sha256):
        raise RawSupervisionAuditError(f"{name} SHA-256 is malformed")
    with _open_retained_directory_chain(lexical.parent) as chain:
        leaf_name = lexical.name
        before = os.stat(leaf_name, dir_fd=chain.directory_fd, follow_symlinks=False)
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or int(before.st_nlink) != 1
        ):
            raise PermissionError(f"{name} must be an unaliased regular file")
        expected_fingerprint = _fingerprint(before)
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(leaf_name, flags, dir_fd=chain.directory_fd)
        try:
            if _fingerprint(os.fstat(descriptor)) != expected_fingerprint:
                raise RawSupervisionAuditError(f"{name} changed during open")
            chain.validate()
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            chain.validate()
            named_after = os.stat(
                leaf_name,
                dir_fd=chain.directory_fd,
                follow_symlinks=False,
            )
            if (
                _fingerprint(named_after) != expected_fingerprint
                or _fingerprint(os.fstat(descriptor)) != expected_fingerprint
            ):
                raise RawSupervisionAuditError(f"{name} changed while read")
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
    if _sha256_bytes(payload) != expected_sha256:
        raise RawSupervisionAuditError(f"{name} file SHA-256 changed")
    return payload


def _rename_noreplace_at(
    parent_fd: int,
    source_name: str,
    destination_name: str,
) -> None:
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    if renameat2 is None:
        raise OSError(errno.ENOSYS, "renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        parent_fd,
        os.fsencode(source_name),
        parent_fd,
        os.fsencode(destination_name),
        1,
    ) != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), destination_name)
        raise OSError(number, os.strerror(number), destination_name)


class _ExclusiveAuditPublisher:
    """Publish one canonical leaf while retaining its complete parent chain."""

    def __init__(self, parent: Path) -> None:
        self._chain = _open_retained_directory_chain(parent)
        self._counter = 0

    @property
    def parent_fd(self) -> int:
        return self._chain.directory_fd

    def __enter__(self) -> "_ExclusiveAuditPublisher":
        self._chain.validate(allow_final_metadata_change=True)
        return self

    def __exit__(self, *_args: object) -> None:
        self._chain.close()

    def require_absent(self, *names: str) -> None:
        self._chain.validate(allow_final_metadata_change=True)
        for name in names:
            if _lstat_optional_at(self.parent_fd, name) is not None:
                raise FileExistsError(f"immutable audit leaf already exists: {name}")
        self._chain.validate(allow_final_metadata_change=True)

    def publish(self, name: str, value: Mapping[str, Any]) -> None:
        if Path(name).name != name or name in {"", ".", ".."}:
            raise RawSupervisionAuditError("audit publication leaf is not canonical")
        payload = canonical_json_bytes(value) + b"\n"
        self._counter += 1
        temporary = f".{name}.owned.{os.getpid()}.{self._counter}.tmp"
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(temporary, flags, 0o600, dir_fd=self.parent_fd)
        owned = _fingerprint(os.fstat(descriptor))
        owned_identity = owned[:2]
        renamed = False
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("audit publication write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            self._chain.validate(allow_final_metadata_change=True)
            if _lstat_optional_at(self.parent_fd, name) is not None:
                raise FileExistsError(f"immutable audit leaf already exists: {name}")
            _rename_noreplace_at(self.parent_fd, temporary, name)
            renamed = True
            os.fsync(self.parent_fd)
            published = os.stat(name, dir_fd=self.parent_fd, follow_symlinks=False)
            if _fingerprint(published)[:2] != owned_identity:
                raise RawSupervisionAuditError("published audit leaf identity changed")
            try:
                self._chain.validate(allow_final_metadata_change=True)
            except BaseException:
                current = _lstat_optional_at(self.parent_fd, name)
                if current is not None and _fingerprint(current)[:2] == owned_identity:
                    os.unlink(name, dir_fd=self.parent_fd)
                    os.fsync(self.parent_fd)
                raise
        finally:
            os.close(descriptor)
            if not renamed:
                current = _lstat_optional_at(self.parent_fd, temporary)
                if current is not None and _fingerprint(current)[:2] == owned_identity:
                    os.unlink(temporary, dir_fd=self.parent_fd)
                    os.fsync(self.parent_fd)


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            ensure_ascii=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise RawSupervisionAuditError("value is not strict JSON") from error


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RawSupervisionAuditError(f"{name} must be an exact integer")
    return value


def _object_pairs(name: str) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]:
    def decode(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise RawSupervisionAuditError(
                    f"{name} contains duplicate JSON key {key!r}"
                )
            result[key] = value
        return result

    return decode


def _decode_json(payload: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise RawSupervisionAuditError(f"{name} contains nonfinite {value}")

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_object_pairs(name),
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawSupervisionAuditError(f"{name} is invalid JSON") from error
    if type(value) is not dict:
        raise RawSupervisionAuditError(f"{name} must be an exact object")
    return value


def _validate_content_hash(value: Mapping[str, Any], *, name: str) -> None:
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise RawSupervisionAuditError(f"{name} content SHA-256 changed")


def _parse_canonical_jsonl(payload: bytes, *, name: str) -> list[dict[str, Any]]:
    if not payload or not payload.endswith(b"\n"):
        raise RawSupervisionAuditError(
            f"{name} must be nonempty newline-terminated JSONL"
        )
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(payload.splitlines(), start=1):
        row = _decode_json(line, name=f"{name}:{index}")
        if canonical_json_bytes(row) != line:
            raise RawSupervisionAuditError(f"{name}:{index} is not canonical JSON")
        _validate_content_hash(row, name=f"{name}:{index}")
        rows.append(row)
    return rows


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_real_directory(path: Path, *, name: str) -> Path:
    lexical = Path(path).absolute()
    try:
        metadata = lexical.stat(follow_symlinks=False)
        resolved = lexical.resolve(strict=True)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        raise RawSupervisionAuditError(f"{name} is absent") from error
    if (
        lexical != resolved
        or lexical.is_symlink()
        or not stat.S_ISDIR(metadata.st_mode)
    ):
        raise PermissionError(f"{name} must be a canonical real directory")
    return lexical


def _canonical_relative_path(value: object, *, name: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise RawSupervisionAuditError(f"{name} must be a nonempty path")
    path = Path(value)
    if (
        path.is_absolute()
        or str(path) != value
        or os.path.normpath(value) != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PermissionError(f"{name} must be canonical and relative")
    return path


def _resolve_regular_file(root: Path, relative: object, *, name: str) -> Path:
    rel = _canonical_relative_path(relative, name=name)
    current = root
    for component in rel.parts[:-1]:
        candidate = current / component
        metadata = candidate.stat(follow_symlinks=False)
        if candidate.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError(f"{name} crosses a non-directory or alias")
        current = candidate
    path = current / rel.parts[-1]
    metadata = path.stat(follow_symlinks=False)
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise PermissionError(f"{name} must be a regular file")
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise PermissionError(f"{name} escapes the dataset") from error
    return path


def _read_bound_file(
    root: Path,
    relative: object,
    *,
    expected_bytes: int,
    expected_sha256: str,
    name: str,
) -> bytes:
    path = _resolve_regular_file(root, relative, name=name)
    before = path.stat(follow_symlinks=False)
    payload = path.read_bytes()
    after = path.stat(follow_symlinks=False)
    fingerprint_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    fingerprint_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if (
        fingerprint_before != fingerprint_after
        or len(payload) != expected_bytes
        or _sha256_bytes(payload) != expected_sha256
    ):
        raise RawSupervisionAuditError(f"{name} bytes changed")
    return payload


def _tree_file_inventory(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        metadata = path.lstat()
        relative = str(path.relative_to(root))
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError(f"dataset tree contains symlink {relative}")
        if stat.S_ISREG(metadata.st_mode):
            result[relative] = path
        elif not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError(f"dataset tree contains special entry {relative}")
    return result


def _validate_root_file_inventory(
    root: Path,
    records: object,
) -> dict[str, Mapping[str, Any]]:
    if type(records) is not list:
        raise RawSupervisionAuditError("manifest files must be a list")
    indexed: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if type(record) is not dict or set(record) != ROOT_FILE_FIELDS:
            raise RawSupervisionAuditError(f"manifest file {index} fields changed")
        relative = str(_canonical_relative_path(record["path"], name="manifest file"))
        if relative == "manifest.json" or relative in indexed:
            raise RawSupervisionAuditError("manifest file inventory repeats/self-includes")
        _strict_int(record.get("byte_count"), name=f"{relative}.byte_count")
        if not _is_sha256(record.get("file_sha256")):
            raise RawSupervisionAuditError(f"{relative} SHA-256 is malformed")
        indexed[relative] = record
    if list(indexed) != sorted(indexed):
        raise RawSupervisionAuditError("manifest file inventory is not ordered")
    observed = _tree_file_inventory(root)
    if set(observed) != set(indexed) | {"manifest.json"}:
        raise RawSupervisionAuditError("manifest and filesystem inventories differ")
    for relative, record in indexed.items():
        payload = _read_bound_file(
            root,
            relative,
            expected_bytes=int(record["byte_count"]),
            expected_sha256=str(record["file_sha256"]),
            name=f"dataset file {relative}",
        )
        del payload
    return indexed


def _validate_access_boundary(ledger: object) -> None:
    if type(ledger) is not dict or not ledger:
        raise RawSupervisionAuditError("access ledger must be a nonempty object")
    for name, value in ledger.items():
        if any(fragment in str(name).lower() for fragment in FORBIDDEN_LEDGER_FRAGMENTS):
            if type(value) is not int or value != 0:
                raise PermissionError(f"forbidden access ledger field is nonzero: {name}")


def _validate_manifest_constants(manifest: Mapping[str, Any], *, exact: bool) -> None:
    if set(manifest) != MANIFEST_FIELDS:
        raise RawSupervisionAuditError("dataset manifest fields changed")
    _validate_content_hash(manifest, name="dataset manifest")
    if (
        manifest.get("schema") != DATASET_SCHEMA
        or manifest.get("status") != "complete_pending_independent_audit"
        or manifest.get("evidence_schema") != evidence_v4.EVIDENCE_SCHEMA
        or manifest.get("raster_schema") != evidence_v4.RASTER_SCHEMA
        or manifest.get("roles") != list(plan_v5.DEVELOPMENT_ROLES)
    ):
        raise RawSupervisionAuditError("dataset manifest identity changed")
    expected_layout = [
        {"path": path, "dtype": np.dtype(dtype).str, "trailing_shape": list(shape)}
        for path, dtype, shape in ARRAY_LAYOUT
    ]
    if manifest.get("array_layout") != expected_layout:
        raise RawSupervisionAuditError("dataset array layout changed")
    licenses = manifest.get("licenses")
    if (
        type(licenses) is not dict
        or set(licenses) != FALSE_LICENSE_FIELDS
        or any(value is not False for value in licenses.values())
    ):
        raise PermissionError("unaudited dataset grants authority")
    if manifest.get("parallel_contract") != {
        "worker_start_method": "spawn",
        "maximum_workers": 6,
        "native_threads_per_worker": 1,
        "gpu_visible_to_workers": False,
        "merge_order": "role_then_scene_then_endpoint_identity",
        "worker_count_does_not_change_artifact_bytes": True,
    }:
        raise RawSupervisionAuditError("parallel construction contract changed")
    if manifest.get("publication") != {
        "staging": "private_sibling_directory_mode_0700",
        "commit": "single_renameat2_RENAME_NOREPLACE",
        "manifest_self_inventory": "canonical_content_sha256",
        "file_inventory": "every_regular_file_except_manifest_self",
    }:
        raise RawSupervisionAuditError("publication contract changed")
    _validate_access_boundary(manifest.get("access_ledger"))
    if exact:
        expected_pairs = {"train": 4262, "checkpoint_selection": 495, "probability_calibration": 415}
        expected_endpoints = {"train": 7777, "checkpoint_selection": 924, "probability_calibration": 759}
        if (
            manifest.get("pair_counts") != expected_pairs
            or manifest.get("unique_endpoint_counts") != expected_endpoints
            or manifest.get("endpoint_instance_count") != 10344
            or manifest.get("scene_shard_count") != 88
        ):
            raise RawSupervisionAuditError("frozen development counts changed")


def _parse_manifest(
    root: Path,
    *,
    expected_manifest_file_sha256: str,
    exact: bool,
) -> dict[str, Any]:
    if not _is_sha256(expected_manifest_file_sha256):
        raise RawSupervisionAuditError("expected manifest SHA-256 is malformed")
    path = _resolve_regular_file(root, "manifest.json", name="dataset manifest")
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_manifest_file_sha256:
        raise RawSupervisionAuditError("dataset manifest file SHA-256 changed")
    if not payload.endswith(b"\n"):
        raise RawSupervisionAuditError("dataset manifest lacks terminal newline")
    manifest = _decode_json(payload, name="dataset manifest")
    if canonical_json_bytes(manifest) + b"\n" != payload:
        raise RawSupervisionAuditError("dataset manifest is not canonical")
    _validate_manifest_constants(manifest, exact=exact)
    return manifest


def _records_by_hash(
    records: Sequence[Mapping[str, Any]],
    *,
    field: str,
    name: str,
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        digest = record.get(field)
        if not _is_sha256(digest) or digest in result:
            raise RawSupervisionAuditError(f"{name} hashes are malformed or repeated")
        result[str(digest)] = record
    return result


def _validate_pair_and_endpoint_indexes(
    root: Path,
    manifest: Mapping[str, Any],
    inputs: AuditInputs,
    file_records: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    pair_record = manifest.get("pair_index")
    endpoint_record = manifest.get("endpoint_index")
    for name, record, expected_path in (
        ("pair index", pair_record, "pairs.jsonl"),
        ("endpoint index", endpoint_record, "endpoints.jsonl"),
    ):
        if type(record) is not dict or set(record) != {"path", "row_count", "file_sha256"}:
            raise RawSupervisionAuditError(f"{name} manifest record changed")
        if record.get("path") != expected_path or record.get("file_sha256") != file_records[expected_path]["file_sha256"]:
            raise RawSupervisionAuditError(f"{name} file binding changed")
    pair_payload = _read_bound_file(
        root,
        "pairs.jsonl",
        expected_bytes=int(file_records["pairs.jsonl"]["byte_count"]),
        expected_sha256=str(file_records["pairs.jsonl"]["file_sha256"]),
        name="pair index",
    )
    endpoint_payload = _read_bound_file(
        root,
        "endpoints.jsonl",
        expected_bytes=int(file_records["endpoints.jsonl"]["byte_count"]),
        expected_sha256=str(file_records["endpoints.jsonl"]["file_sha256"]),
        name="endpoint index",
    )
    pairs = _parse_canonical_jsonl(pair_payload, name="pair index")
    endpoints = _parse_canonical_jsonl(endpoint_payload, name="endpoint index")
    if int(pair_record["row_count"]) != len(pairs) or int(endpoint_record["row_count"]) != len(endpoints):
        raise RawSupervisionAuditError("index row counts changed")
    expected_pairs = [dict(item) for item in inputs.plan.pairs]
    if pairs != expected_pairs:
        raise RawSupervisionAuditError("published pair index differs from metadata V5")
    if canonical_json_sha256([item["content_sha256"] for item in pairs]) != manifest.get("ordered_pair_sha256"):
        raise RawSupervisionAuditError("ordered pair hash changed")
    plan_endpoints = _records_by_hash(
        inputs.plan.endpoints,
        field="identity_sha256",
        name="metadata-plan endpoints",
    )
    seen: set[str] = set()
    previous_key: tuple[int, str] | None = None
    for index, row in enumerate(endpoints):
        if set(row) != TOP_ENDPOINT_FIELDS:
            raise RawSupervisionAuditError(f"endpoint index row {index} fields changed")
        digest = str(row.get("endpoint_identity_sha256", ""))
        planned = plan_endpoints.get(digest)
        if planned is None or digest in seen:
            raise RawSupervisionAuditError("endpoint index is extra, missing, or repeated")
        seen.add(digest)
        identity = planned.get("identity")
        if not isinstance(identity, Mapping):
            raise RawSupervisionAuditError("planned endpoint identity is absent")
        key = (plan_v5.DEVELOPMENT_ROLES.index(str(row["dataset_role"])), digest)
        if previous_key is not None and key <= previous_key:
            raise RawSupervisionAuditError("endpoint index order changed")
        previous_key = key
        if (
            row["schema"] != ENDPOINT_INDEX_SCHEMA
            or row["dataset_role"] != identity.get("dataset_role")
            or row["scene_id"] != identity.get("scene_id")
            or row["plan_endpoint_content_sha256"] != planned.get("content_sha256")
            or row["image_path_metadata_only"] != planned.get("image_path_metadata_only")
            or row["image_sha256_commitment_only"] != identity.get("image_sha256")
            or not _is_sha256(row.get("evidence_content_sha256"))
            or not _is_sha256(row.get("raster_content_sha256"))
        ):
            raise RawSupervisionAuditError("endpoint index disagrees with metadata V5")
        _strict_int(row.get("shard_row"), name="endpoint shard_row")
        expected_shard = f"shards/{hashlib.sha256(str(row['scene_id']).encode()).hexdigest()[:16]}/shard.json"
        if row.get("scene_shard") != expected_shard:
            raise RawSupervisionAuditError("endpoint scene-shard binding changed")
    if seen != set(plan_endpoints):
        raise RawSupervisionAuditError("endpoint index does not cover metadata V5")
    if canonical_json_sha256([item["content_sha256"] for item in endpoints]) != manifest.get("ordered_endpoint_sha256"):
        raise RawSupervisionAuditError("ordered endpoint hash changed")
    endpoint_by_hash = {str(item["endpoint_identity_sha256"]): item for item in endpoints}
    uses = Counter()
    for pair in pairs:
        for side in ("current", "next"):
            digest = str(pair[f"{side}_endpoint_sha256"])
            endpoint = endpoint_by_hash.get(digest)
            if endpoint is None:
                raise RawSupervisionAuditError("pair references an absent endpoint")
            if (
                endpoint["dataset_role"] != pair["dataset_role"]
                or endpoint["scene_id"] != pair["scene_id"]
                or endpoint["family"] != pair["family"]
            ):
                raise RawSupervisionAuditError("pair crossed endpoint role/scene/family")
            uses[digest] += 1
    if set(uses) != set(endpoint_by_hash):
        raise RawSupervisionAuditError("endpoint index contains an orphan")
    return pairs, endpoints, plan_endpoints


def _sample_records(endpoint_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in endpoint_rows:
        grouped[(str(row["dataset_role"]), str(row["family"]))].append(row)
    records: list[dict[str, Any]] = []
    for role, family in sorted(grouped):
        scored = [
            (
                hashlib.sha256(
                    role.encode("utf-8")
                    + b"\0"
                    + family.encode("utf-8")
                    + b"\0"
                    + str(row["endpoint_identity_sha256"]).encode("ascii")
                ).hexdigest(),
                row,
            )
            for row in grouped[(role, family)]
        ]
        score, chosen = min(scored, key=lambda item: item[0])
        records.append(
            {
                "dataset_role": role,
                "family": family,
                "endpoint_identity_sha256": chosen["endpoint_identity_sha256"],
                "selection_sha256": score,
            }
        )
    return records


def _validate_sample_precommit(
    manifest: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    *,
    exact: bool,
) -> list[dict[str, Any]]:
    records = _sample_records(endpoints)
    expected = {
        "scheme": "minimum_sha256_role_nul_family_nul_endpoint_identity_v1",
        "one_endpoint_per_observed_role_family": True,
        "expected_exact_record_count": 24,
        "records": records,
        "records_sha256": canonical_json_sha256(records),
    }
    if manifest.get("independent_audit_precommit") != expected:
        raise RawSupervisionAuditError("independent audit sample precommit changed")
    if exact:
        role_groups = Counter(str(item["dataset_role"]) for item in records)
        if len(records) != EXPECTED_SAMPLE_COUNT or role_groups != Counter(
            {role: 8 for role in plan_v5.DEVELOPMENT_ROLES}
        ):
            raise RawSupervisionAuditError(
                "exact audit sample is not eight families in each development role"
            )
    return records


def _read_array(
    directory: Path,
    record: Mapping[str, Any],
    *,
    endpoint_count: int,
    trailing_shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    if set(record) != SHARD_FILE_FIELDS:
        raise RawSupervisionAuditError("shard array file record changed")
    expected_shape = (endpoint_count, *trailing_shape)
    if record.get("dtype") != np.dtype(dtype).str or record.get("shape") != list(expected_shape):
        raise RawSupervisionAuditError("shard array dtype/shape changed")
    payload = _read_bound_file(
        directory,
        record["path"],
        expected_bytes=int(record["byte_count"]),
        expected_sha256=str(record["file_sha256"]),
        name=f"shard array {record['path']}",
    )
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64)) * np.dtype(dtype).itemsize
    if len(payload) != expected_bytes:
        raise RawSupervisionAuditError("shard array byte count disagrees with shape")
    return np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(expected_shape)


def _stored_arrays_from_evidence(evidence: Any, raster: Any) -> tuple[np.ndarray, ...]:
    return (
        np.ascontiguousarray(evidence.camera_origin_body_m, dtype="<f4"),
        np.ascontiguousarray(evidence.camera_basis_body_fru, dtype="<f4"),
        np.asarray(evidence.ground_plane_z_body_m, dtype="<f4"),
        np.ascontiguousarray(evidence.ground_support_in_frustum, dtype=np.uint8),
        np.ascontiguousarray(evidence.ground_support_clear_to_target, dtype=np.uint8),
        np.ascontiguousarray(evidence.pixel_hit_mask, dtype=np.uint8),
        np.ascontiguousarray(evidence.pixel_first_hit_distance_m, dtype="<f4"),
        np.ascontiguousarray(raster.output_labels, dtype=np.uint8),
    )


def _validate_shards(
    root: Path,
    manifest: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    file_records: Mapping[str, Mapping[str, Any]],
    sample_hashes: set[str],
) -> dict[str, StoredEndpointEvidence]:
    shard_records = manifest.get("shards")
    if type(shard_records) is not list:
        raise RawSupervisionAuditError("manifest shards must be a list")
    endpoint_by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for endpoint in endpoints:
        endpoint_by_shard[str(endpoint["scene_shard"])].append(endpoint)
    seen_endpoints: set[str] = set()
    samples: dict[str, StoredEndpointEvidence] = {}
    seen_directories: set[str] = set()
    previous_scene = ""
    for shard_record in shard_records:
        if type(shard_record) is not dict or set(shard_record) != {
            "path", "dataset_role", "family", "scene_id", "endpoint_count", "content_sha256"
        }:
            raise RawSupervisionAuditError("manifest shard record fields changed")
        scene_id = str(shard_record["scene_id"])
        if scene_id <= previous_scene:
            raise RawSupervisionAuditError("manifest shard order changed")
        previous_scene = scene_id
        directory_name = hashlib.sha256(scene_id.encode("utf-8")).hexdigest()[:16]
        relative = f"shards/{directory_name}/shard.json"
        if shard_record["path"] != relative or directory_name in seen_directories:
            raise RawSupervisionAuditError("scene shard path collided or changed")
        seen_directories.add(directory_name)
        root_file = file_records.get(relative)
        if root_file is None:
            raise RawSupervisionAuditError("scene shard is absent from file inventory")
        payload = _read_bound_file(
            root,
            relative,
            expected_bytes=int(root_file["byte_count"]),
            expected_sha256=str(root_file["file_sha256"]),
            name=f"scene shard {scene_id}",
        )
        shard = _decode_json(payload, name=f"scene shard {scene_id}")
        if canonical_json_bytes(shard) + b"\n" != payload or set(shard) != SHARD_FIELDS:
            raise RawSupervisionAuditError("scene shard is noncanonical or fields changed")
        _validate_content_hash(shard, name=f"scene shard {scene_id}")
        if (
            shard["schema"] != SHARD_SCHEMA
            or shard["dataset_role"] != shard_record["dataset_role"]
            or shard["family"] != shard_record["family"]
            or shard["scene_id"] != scene_id
            or shard["scene_id_sha256"] != hashlib.sha256(scene_id.encode()).hexdigest()
            or shard["content_sha256"] != shard_record["content_sha256"]
        ):
            raise RawSupervisionAuditError("scene shard identity changed")
        count = _strict_int(shard["endpoint_count"], name="shard endpoint_count", minimum=1)
        if count != shard_record["endpoint_count"]:
            raise RawSupervisionAuditError("shard endpoint count changed")
        local_records = shard.get("files")
        if type(local_records) is not list:
            raise RawSupervisionAuditError("shard files must be a list")
        local_by_name: dict[str, Mapping[str, Any]] = {}
        for record in local_records:
            if type(record) is not dict or set(record) != SHARD_FILE_FIELDS:
                raise RawSupervisionAuditError("shard file record fields changed")
            name = str(_canonical_relative_path(record["path"], name="shard file"))
            if len(Path(name).parts) != 1 or name in local_by_name:
                raise RawSupervisionAuditError("shard file path repeats or is nested")
            local_by_name[name] = record
            root_relative = f"shards/{directory_name}/{name}"
            root_record = file_records.get(root_relative)
            if root_record is None or any(
                root_record[field] != record[field]
                for field in ("byte_count", "file_sha256")
            ):
                raise RawSupervisionAuditError("root/shard file inventories disagree")
        expected_names = {name for name, _dtype, _shape in ARRAY_LAYOUT} | {"index.jsonl"}
        if set(local_by_name) != expected_names:
            raise RawSupervisionAuditError("shard file inventory changed")
        directory = root / "shards" / directory_name
        arrays = {
            name: _read_array(
                directory,
                local_by_name[name],
                endpoint_count=count,
                trailing_shape=shape,
                dtype=dtype,
            )
            for name, dtype, shape in ARRAY_LAYOUT
        }
        for boolean_name in (
            "ground_support_in_frustum.u1",
            "ground_support_clear_to_target.u1",
            "pixel_hit_mask.u1",
        ):
            if not np.isin(arrays[boolean_name], (0, 1)).all():
                raise RawSupervisionAuditError("boolean evidence array contains another value")
        index_record = local_by_name["index.jsonl"]
        if index_record.get("dtype") != "canonical_jsonl" or index_record.get("shape") != [count]:
            raise RawSupervisionAuditError("shard index dtype/shape changed")
        index_payload = _read_bound_file(
            directory,
            "index.jsonl",
            expected_bytes=int(index_record["byte_count"]),
            expected_sha256=str(index_record["file_sha256"]),
            name=f"shard index {scene_id}",
        )
        rows = _parse_canonical_jsonl(index_payload, name=f"shard index {scene_id}")
        if len(rows) != count:
            raise RawSupervisionAuditError("shard index count changed")
        top_rows = endpoint_by_shard.get(relative, [])
        top_by_hash = {str(item["endpoint_identity_sha256"]): item for item in top_rows}
        if len(top_by_hash) != len(top_rows) or len(top_rows) != count:
            raise RawSupervisionAuditError("top endpoint/shard counts disagree")
        endpoint_hashes: list[str] = []
        evidence_hashes: list[str] = []
        raster_hashes: list[str] = []
        for row_index, row in enumerate(rows):
            if set(row) != SHARD_INDEX_FIELDS or row.get("shard_row") != row_index:
                raise RawSupervisionAuditError("shard index row fields/order changed")
            digest = str(row.get("endpoint_identity_sha256", ""))
            top = top_by_hash.get(digest)
            if top is None or digest in seen_endpoints:
                raise RawSupervisionAuditError("shard endpoint is absent or repeated")
            seen_endpoints.add(digest)
            expected_top = dict(row)
            expected_top.pop("content_sha256")
            expected_top["scene_shard"] = relative
            expected_top["content_sha256"] = canonical_json_sha256(expected_top)
            if top != expected_top:
                raise RawSupervisionAuditError("top endpoint differs from shard index")
            if (
                row["dataset_role"] != shard["dataset_role"]
                or row["family"] != shard["family"]
                or row["scene_id"] != scene_id
            ):
                raise RawSupervisionAuditError("shard index crossed role/family/scene")
            evidence = evidence_v4.ObservableCameraRayEvidenceV4(
                camera_origin_body_m=arrays["camera_origin_body_m.f4"][row_index],
                camera_basis_body_fru=arrays["camera_basis_body_fru.f4"][row_index],
                ground_plane_z_body_m=float(arrays["ground_plane_z_body_m.f4"][row_index]),
                ground_support_in_frustum=arrays["ground_support_in_frustum.u1"][row_index].astype(bool),
                ground_support_clear_to_target=arrays["ground_support_clear_to_target.u1"][row_index].astype(bool),
                pixel_hit_mask=arrays["pixel_hit_mask.u1"][row_index].astype(bool),
                pixel_first_hit_distance_m=arrays["pixel_first_hit_distance_m.f4"][row_index],
            )
            raster = evidence_v4.rasterize_observable_camera_ray_evidence_v4(evidence)
            if (
                evidence.content_sha256() != row["evidence_content_sha256"]
                or raster.content_sha256() != row["raster_content_sha256"]
                or not np.array_equal(
                    raster.output_labels,
                    arrays["raster_labels.u1"][row_index],
                )
            ):
                raise RawSupervisionAuditError("stored V4 evidence/raster changed")
            endpoint_hashes.append(digest)
            evidence_hashes.append(evidence.content_sha256())
            raster_hashes.append(raster.content_sha256())
            if digest in sample_hashes:
                samples[digest] = StoredEndpointEvidence(
                    endpoint_identity_sha256=digest,
                    arrays=tuple(
                        np.array(arrays[name][row_index], copy=True)
                        for name, _dtype, _shape in ARRAY_LAYOUT
                    ),
                    evidence_content_sha256=evidence.content_sha256(),
                    raster_content_sha256=raster.content_sha256(),
                )
        if (
            canonical_json_sha256(endpoint_hashes) != shard["ordered_endpoint_identity_sha256"]
            or canonical_json_sha256(evidence_hashes) != shard["ordered_evidence_sha256"]
            or canonical_json_sha256(raster_hashes) != shard["ordered_raster_sha256"]
        ):
            raise RawSupervisionAuditError("shard ordered hashes changed")
    if seen_endpoints != {str(item["endpoint_identity_sha256"]) for item in endpoints}:
        raise RawSupervisionAuditError("shards do not cover the endpoint index")
    if set(samples) != sample_hashes:
        raise RawSupervisionAuditError("not every precommitted sample is present")
    return samples


def _validate_one_shard_task(
    task: tuple[
        str,
        Mapping[str, Any],
        Sequence[Mapping[str, Any]],
        Mapping[str, Mapping[str, Any]],
        set[str],
    ],
) -> dict[str, StoredEndpointEvidence]:
    _set_worker_environment()
    root, shard_record, endpoints, file_records, sample_hashes = task
    return _validate_shards(
        Path(root),
        {"shards": [shard_record]},
        endpoints,
        file_records,
        sample_hashes,
    )


def _validate_shards_parallel(
    root: Path,
    manifest: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    file_records: Mapping[str, Mapping[str, Any]],
    sample_hashes: set[str],
    *,
    workers: int,
) -> dict[str, StoredEndpointEvidence]:
    if int(workers) == 1:
        return _validate_shards(
            root, manifest, endpoints, file_records, sample_hashes
        )
    endpoint_by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for endpoint in endpoints:
        endpoint_by_shard[str(endpoint["scene_shard"])].append(endpoint)
    tasks = []
    for shard_record in manifest["shards"]:
        path = str(shard_record["path"])
        directory_prefix = str(Path(path).parent) + "/"
        scoped_files = {
            name: record
            for name, record in file_records.items()
            if name == path or name.startswith(directory_prefix)
        }
        scoped_endpoints = endpoint_by_shard.get(path, [])
        scoped_sample = {
            str(item["endpoint_identity_sha256"])
            for item in scoped_endpoints
            if item["endpoint_identity_sha256"] in sample_hashes
        }
        tasks.append(
            (
                str(root),
                shard_record,
                scoped_endpoints,
                scoped_files,
                scoped_sample,
            )
        )
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=context,
        initializer=_set_worker_environment,
    ) as executor:
        partials = list(executor.map(_validate_one_shard_task, tasks))
    samples: dict[str, StoredEndpointEvidence] = {}
    for partial in partials:
        if set(samples) & set(partial):
            raise RawSupervisionAuditError("parallel shard audit repeated a sample")
        samples.update(partial)
    if set(samples) != sample_hashes:
        raise RawSupervisionAuditError("parallel shard audit missed a sample")
    return samples


def _compare_source_replay(
    sample: Sequence[Mapping[str, Any]],
    observed: Mapping[str, StoredEndpointEvidence],
    recomputed: Mapping[str, tuple[np.ndarray, ...]],
) -> list[dict[str, Any]]:
    wanted = {str(item["endpoint_identity_sha256"]) for item in sample}
    if set(recomputed) != wanted:
        raise RawSupervisionAuditError("source replay returned another endpoint set")
    results: list[dict[str, Any]] = []
    for record in sample:
        digest = str(record["endpoint_identity_sha256"])
        actual = observed[digest]
        replay = recomputed[digest]
        if len(replay) != len(ARRAY_LAYOUT):
            raise RawSupervisionAuditError("source replay array count changed")
        byte_hashes: list[str] = []
        for position, ((name, dtype, shape), expected, received) in enumerate(
            zip(ARRAY_LAYOUT, actual.arrays, replay)
        ):
            raw = np.asarray(received, dtype=np.dtype(dtype))
            normalized = (
                raw.reshape(())
                if shape == ()
                else np.ascontiguousarray(raw, dtype=np.dtype(dtype))
            )
            if normalized.shape != shape:
                raise RawSupervisionAuditError(
                    f"source replay {digest}:{name} shape changed at {position}"
                )
            if normalized.tobytes(order="C") != np.ascontiguousarray(expected).tobytes(order="C"):
                raise RawSupervisionAuditError(
                    f"source replay {digest}:{name} differs byte-for-byte"
                )
            byte_hashes.append(_sha256_bytes(normalized.tobytes(order="C")))
        results.append(
            {
                **dict(record),
                "array_byte_sha256": byte_hashes,
                "array_byte_sha256_set": canonical_json_sha256(byte_hashes),
                "passes": True,
            }
        )
    return results


def audit_dataset_v1(
    dataset_root: Path,
    *,
    expected_manifest_file_sha256: str,
    inputs: AuditInputs,
    sample_recomputer: SampleRecomputer,
    workers: int = 1,
    exact: bool = False,
) -> dict[str, Any]:
    """Audit one immutable artifact; exact callers use :func:`audit_exact_dataset_v1`."""

    if not isinstance(inputs, AuditInputs):
        raise TypeError("inputs must be AuditInputs")
    if isinstance(workers, bool) or not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f"workers must lie in [1,{MAX_WORKERS}]")
    root = _require_real_directory(Path(dataset_root), name="dataset root")
    manifest = _parse_manifest(
        root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        exact=exact,
    )
    file_records = _validate_root_file_inventory(root, manifest.get("files"))
    pairs, endpoints, plan_endpoints = _validate_pair_and_endpoint_indexes(
        root, manifest, inputs, file_records
    )
    del pairs
    sample = _validate_sample_precommit(manifest, endpoints, exact=exact)
    sample_hashes = {str(item["endpoint_identity_sha256"]) for item in sample}
    observed = _validate_shards_parallel(
        root,
        manifest,
        endpoints,
        file_records,
        sample_hashes,
        workers=int(workers),
    )
    recomputed = sample_recomputer(sample, plan_endpoints, inputs, int(workers))
    sample_results = _compare_source_replay(sample, observed, recomputed)
    _validate_root_file_inventory(root, manifest.get("files"))
    if _sha256_file(root / "manifest.json") != expected_manifest_file_sha256:
        raise RawSupervisionAuditError("dataset manifest changed during audit")
    core = {
        "schema": AUDIT_SCHEMA,
        "verdict": "PASS",
        "dataset_manifest_file_sha256": expected_manifest_file_sha256,
        "dataset_manifest_content_sha256": manifest["content_sha256"],
        "pair_count": len(inputs.plan.pairs),
        "unique_endpoint_count": len(endpoints),
        "scene_shard_count": len(manifest["shards"]),
        "sample_count": len(sample_results),
        "sample_results": sample_results,
        "sample_results_sha256": canonical_json_sha256(sample_results),
        "full_byte_inventory_revalidated": True,
        "pair_endpoint_joins_reconstructed": True,
        "all_stored_evidence_and_rasters_recomputed": True,
        "sample_original_geometry_recomputed": True,
        "dataset_use_authorized": False,
        "training_authorized": False,
        "g2_authorized": False,
        "production_authorized": False,
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _set_worker_environment() -> None:
    for name in THREAD_ENVIRONMENT:
        os.environ[name] = "1"
    for name in ACCELERATOR_ENVIRONMENT:
        os.environ[name] = ""


def _source_file_records(inventory: plan_v5.DevelopmentSourceInventory) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene in inventory.records:
        records.extend(
            (
                {"scene_id": scene["scene_id"], "kind": "frames", **scene["frames"]},
                {"scene_id": scene["scene_id"], "kind": "scene_manifest", **scene["scene_manifest"]},
                {"scene_id": scene["scene_id"], "kind": "render_plan", **scene["render_plan"]},
                {"scene_id": scene["scene_id"], "kind": "render_summary", **scene["render_summary"]},
            )
        )
    return records


def _hash_source_file(record: Mapping[str, Any]) -> dict[str, Any]:
    _set_worker_environment()
    path = Path(str(record["path"]))
    expected = str(record.get("sha256", record.get("file_sha256", "")))
    payload = _read_absolute_bound_payload(
        path,
        expected,
        repository_root=ROOT,
        name=f"allowed {record['kind']} source",
    )
    result = {
        "scene_id": record["scene_id"],
        "kind": record["kind"],
        "path": str(path),
        "byte_count": len(payload),
        "file_sha256": _sha256_bytes(payload),
    }
    if record["kind"] == "frames":
        if not payload or not payload.endswith(b"\n") or b"\n\n" in payload:
            raise RawSupervisionAuditError("allowed frames source is malformed JSONL")
        result["jsonl_record_count"] = len(payload.splitlines())
    return result


def _hash_complete_source_inventory(
    inventory: plan_v5.DevelopmentSourceInventory,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    records = _source_file_records(inventory)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=context,
        initializer=_set_worker_environment,
    ) as executor:
        results = list(executor.map(_hash_source_file, records))
    results.sort(key=lambda item: (item["scene_id"], item["kind"]))
    if len(results) != 352:
        raise RawSupervisionAuditError("allowed development source inventory is not 352 files")
    return results


def _parent_contract_receipts() -> list[dict[str, Any]]:
    geometry_raw = _read_absolute_bound_payload(
        GEOMETRY_CONTRACT_PATH,
        GEOMETRY_CONTRACT_FILE_SHA256,
        repository_root=ROOT,
        name="physical geometry contract",
    )
    render_raw = _read_absolute_bound_payload(
        RENDER_AUDIT_PATH,
        RENDER_AUDIT_FILE_SHA256,
        repository_root=ROOT,
        name="render audit contract",
    )
    geometry = _decode_json(geometry_raw, name="physical geometry contract")
    render = _decode_json(render_raw, name="render audit contract")
    if source_v4._geometry_semantic_sha256(geometry) != GEOMETRY_CONTRACT_CONTENT_SHA256:
        raise RawSupervisionAuditError("physical geometry semantic hash changed")
    source_v4._geometry_flags(geometry)
    source_v4._validate_render_audit_contract(
        render, expected_content_sha256=RENDER_AUDIT_CONTENT_SHA256
    )
    return [
        {
            "path": str(GEOMETRY_CONTRACT_PATH),
            "file_sha256": GEOMETRY_CONTRACT_FILE_SHA256,
            "byte_count": len(geometry_raw),
            "purpose": "geometry_contract",
        },
        {
            "path": str(RENDER_AUDIT_PATH),
            "file_sha256": RENDER_AUDIT_FILE_SHA256,
            "byte_count": len(render_raw),
            "purpose": "render_audit",
        },
    ]


def _builder_source_receipts(
    hashed_sources: Sequence[Mapping[str, Any]],
    parent_contracts: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    purpose_by_kind = {
        "frames": "source_frames_jsonl",
        "scene_manifest": "source_scene_manifest",
        "render_plan": "render_plan",
        "render_summary": "render_summary",
    }
    receipts = [dict(item) for item in parent_contracts]
    for item in hashed_sources:
        receipts.append(
            {
                "path": item["path"],
                "file_sha256": item["file_sha256"],
                "byte_count": item["byte_count"],
                "purpose": purpose_by_kind[str(item["kind"])],
                "scene_id": item["scene_id"],
            }
        )
    receipts.sort(
        key=lambda item: (
            str(item["path"]),
            str(item["purpose"]),
            str(item.get("scene_id", "")),
        )
    )
    if len(receipts) != 354:
        raise RawSupervisionAuditError("builder source receipt population changed")
    return receipts


def _validate_exact_access_ledger(
    value: object,
    *,
    inputs: AuditInputs,
    frames_scanned: int,
) -> None:
    if type(value) is not dict or set(value) != EXACT_ACCESS_LEDGER_KEYS:
        raise RawSupervisionAuditError("exact builder access-ledger fields changed")
    expected_scalars = {
        "schema": ACCESS_LEDGER_SCHEMA,
        "measurement_scope": (
            "controlled_data_opens_excluding_import_and_reviewed_source_hash_reads"
        ),
        "development_scene_workers": 88,
        "unique_endpoint_raycasts": 9460,
        "pair_endpoint_references": 10344,
        "source_frames_jsonl_records_scanned": int(frames_scanned),
        "source_frames_selected_records": 9460,
        "source_frames_byte_opens": 176,
        "source_scene_manifest_byte_opens": 176,
        "render_plan_byte_opens": 176,
        "render_summary_byte_opens": 176,
        "geometry_contract_byte_opens": 2,
        "render_audit_byte_opens": 2,
        "source_payload_first_pass_file_count": 354,
        "source_payload_second_pass_file_count": 354,
        "source_payload_total_byte_opens": 708,
        "g2_source_index_rows_read_for_exclusion": 8,
        "g2_sidecar_byte_opens": 0,
        "g2_source_payload_opens": 0,
        "g2_label_payload_opens": 0,
        "g2_rgb_byte_opens": 0,
        "rgb_byte_opens": 0,
        "rgb_decodes": 0,
        "parent_label_shard_payload_opens": 0,
        "checkpoint_or_model_output_opens": 0,
        "runtime_or_navigation_result_opens": 0,
        "heldout_or_sealed_opens": 0,
        "hardware_or_production_opens": 0,
        "writes_outside_output_or_failure_namespace": 0,
        "denied_or_unexpected_accesses": 0,
    }
    for name, expected in expected_scalars.items():
        if value.get(name) != expected:
            raise RawSupervisionAuditError(f"exact access ledger changed at {name}")
    if (
        value["metadata_plan_first_pass"] != inputs.plan.value["access_ledger"]
        or value["metadata_plan_second_pass"] != inputs.plan.value["access_ledger"]
        or value["metadata_source_inventory_first_pass"]
        != inputs.inventory.access_ledger
        or value["metadata_source_inventory_second_pass"]
        != inputs.inventory.access_ledger
    ):
        raise RawSupervisionAuditError("exact metadata access receipts changed")


def _validate_frozen_source_map(mapping: object, expected: Mapping[str, str], *, name: str) -> None:
    if mapping != dict(expected):
        raise RawSupervisionAuditError(f"{name} provenance map changed")
    for relative, digest in expected.items():
        payload = _read_absolute_bound_payload(
            ROOT / relative,
            digest,
            repository_root=ROOT,
            name=f"{name} {relative}",
        )
        del payload


def _validate_exact_input_provenance(
    value: object,
    *,
    inputs: AuditInputs,
    receipts: Sequence[Mapping[str, Any]],
) -> None:
    if type(value) is not dict or set(value) != EXACT_INPUT_PROVENANCE_FIELDS:
        raise RawSupervisionAuditError("exact input-provenance fields changed")
    if (
        value["metadata_plan_content_sha256"] != EXACT_PLAN_CONTENT_SHA256
        or value["metadata_plan_content_sha256"] != inputs.plan.value["content_sha256"]
        or value["metadata_ordered_pair_sha256"] != EXACT_ORDERED_PAIR_SHA256
        or value["metadata_ordered_pair_sha256"] != inputs.plan.value["ordered_pair_sha256"]
        or value["metadata_ordered_endpoint_sha256"] != EXACT_ORDERED_ENDPOINT_SHA256
        or value["metadata_ordered_endpoint_sha256"]
        != inputs.plan.value["ordered_endpoint_sha256"]
        or value["source_inventory_sha256"] != dict(inputs.inventory.hashes)
        or value["source_inventory_sha256"] != dict(plan_v5.SOURCE_INVENTORY_SHA256)
    ):
        raise RawSupervisionAuditError("exact metadata/source provenance changed")
    _validate_frozen_source_map(
        value["frozen_parent_file_sha256"],
        FROZEN_PARENT_FILE_SHA256,
        name="frozen parent",
    )
    _validate_frozen_source_map(
        value["reviewed_v4_source_sha256"],
        REVIEWED_V4_SOURCE_SHA256,
        name="reviewed V4 source",
    )
    if (
        value["geometry_contract_file_sha256"] != GEOMETRY_CONTRACT_FILE_SHA256
        or value["geometry_contract_content_sha256"]
        != GEOMETRY_CONTRACT_CONTENT_SHA256
        or value["render_audit_file_sha256"] != RENDER_AUDIT_FILE_SHA256
        or value["render_audit_content_sha256"] != RENDER_AUDIT_CONTENT_SHA256
        or value["source_payload_inventory"] != list(receipts)
        or value["source_payload_inventory_sha256"]
        != canonical_json_sha256(list(receipts))
    ):
        raise RawSupervisionAuditError("exact source payload provenance changed")
    _validate_exact_authorization(value)


def _validate_exact_authorization(value: Mapping[str, Any]) -> None:
    authorization_sha256 = value.get("authorization_file_sha256")
    if not _is_sha256(authorization_sha256):
        raise RawSupervisionAuditError("build authorization file hash is malformed")
    authorization_raw = _read_absolute_bound_payload(
        BUILD_AUTHORIZATION_PATH,
        str(authorization_sha256),
        repository_root=ROOT,
        name="build authorization",
    )
    authorization = _decode_json(authorization_raw, name="build authorization")
    _validate_content_hash(authorization, name="build authorization")
    source_map = authorization.get("source_map")
    if (
        authorization.get("schema")
        != "lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v1"
        or authorization.get("exact_build_authorized_after_independent_reviews")
        is not True
        or authorization.get("builder_review", {}).get("verdict") != "PASS"
        or authorization.get("auditor_review", {}).get("verdict") != "PASS"
        or authorization.get("content_sha256")
        != value["authorization_content_sha256"]
        or canonical_json_sha256(source_map)
        != value["authorization_source_map_sha256"]
    ):
        raise PermissionError("exact build authorization provenance changed")
    if type(source_map) is not list:
        raise RawSupervisionAuditError("build authorization source map is absent")
    required_roles = {
        "builder_source",
        "builder_cli",
        "builder_test",
        "builder_handoff",
        "builder_review",
        "auditor_source",
        "auditor_cli",
        "auditor_test",
        "auditor_review",
    }
    observed_roles: set[str] = set()
    for entry in source_map:
        if type(entry) is not dict or set(entry) != {"role", "path", "sha256"}:
            raise RawSupervisionAuditError(
                "build authorization source entry fields changed"
            )
        role = str(entry["role"])
        relative = _canonical_relative_path(
            entry["path"], name=f"authorized source {role}"
        )
        digest = str(entry["sha256"])
        if role in observed_roles or role not in required_roles or not _is_sha256(digest):
            raise RawSupervisionAuditError("build authorization source map changed")
        observed_roles.add(role)
        payload = _read_absolute_bound_payload(
            ROOT / relative,
            digest,
            repository_root=ROOT,
            name=f"authorized source {role}",
        )
        del payload
    if observed_roles != required_roles:
        raise RawSupervisionAuditError("build authorization source roles changed")


def _preflight_exact_authorization(
    dataset_root: Path,
    *,
    expected_manifest_file_sha256: str,
) -> dict[str, Any]:
    manifest = _parse_manifest(
        _require_real_directory(dataset_root, name="dataset root"),
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        exact=True,
    )
    provenance = manifest.get("input_provenance")
    if type(provenance) is not dict or set(provenance) != EXACT_INPUT_PROVENANCE_FIELDS:
        raise RawSupervisionAuditError("exact input-provenance fields changed")
    _validate_exact_authorization(provenance)
    return manifest


def _validate_exact_manifest_bindings(
    manifest: Mapping[str, Any],
    *,
    inputs: AuditInputs,
    hashed_sources: Sequence[Mapping[str, Any]],
    parent_contracts: Sequence[Mapping[str, Any]],
) -> None:
    frames_scanned = sum(
        int(item.get("jsonl_record_count", 0))
        for item in hashed_sources
        if item["kind"] == "frames"
    )
    receipts = _builder_source_receipts(hashed_sources, parent_contracts)
    _validate_exact_access_ledger(
        manifest.get("access_ledger"),
        inputs=inputs,
        frames_scanned=frames_scanned,
    )
    _validate_exact_input_provenance(
        manifest.get("input_provenance"), inputs=inputs, receipts=receipts
    )


def _read_exact_source_json(path: str, expected_sha256: str, *, name: str) -> dict[str, Any]:
    payload = _read_absolute_bound_payload(
        Path(path), expected_sha256, repository_root=ROOT, name=name
    )
    return _decode_json(payload, name=name)


def _source_record_for_endpoint(
    endpoint_digest: str,
    endpoint: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates: list[tuple[int, str, Mapping[str, Any]]] = []
    for pair in pairs:
        for side_rank, side in enumerate(("current", "next")):
            if pair.get(f"{side}_endpoint_sha256") == endpoint_digest:
                candidates.append((int(pair["global_row"]), side, pair))
    if not candidates:
        raise RawSupervisionAuditError("sample endpoint has no pair occurrence")
    _global, side, pair = min(candidates, key=lambda item: (item[0], item[1]))
    identity = endpoint["identity"]
    return {
        "family": pair["family"],
        "scene_id": identity["scene_id"],
        "global_row": pair["global_row"],
        "side": side,
        "image_path_metadata_only": endpoint["image_path_metadata_only"],
        "image_sha256": identity["image_sha256"],
        "label_shard_sha256": pair["label_shard_sha256"],
        "label_row": pair["label_shard_row"],
        "frame_index": identity["frame_index"],
        "env_index": identity["env_index"],
        "timestamp_ns": identity["timestamp_ns"],
        "episode_id": identity["episode_id"],
        "reset_count": pair["reset_count"],
        "episode_step": identity["episode_step"],
    }


def _summary_source_entry(summary: Mapping[str, Any], name: str) -> tuple[str, str]:
    source = summary.get("source")
    entry = source.get(name) if isinstance(source, Mapping) else None
    if type(entry) is not dict or set(entry) != {"path", "sha256"}:
        raise RawSupervisionAuditError(f"render summary source.{name} changed")
    if not _is_sha256(entry.get("sha256")):
        raise RawSupervisionAuditError(f"render summary source.{name} hash changed")
    return str(entry["path"]), str(entry["sha256"])


def _source_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _ensure_reviewed_source_semantics() -> None:
    if source_v4._SEMANTICS_LOADED:
        return
    from lewm.benchmarks import go2_n32_camera_frustum_observability as core
    from lewm.datasets import go2_paired_navigation as paired_navigation
    from lewm_worlds import manifest as manifest_semantics
    from lewm_worlds import planning_grid as planning_semantics

    source_v4._install_semantic_modules(
        core,
        paired_navigation,
        manifest_semantics,
        planning_semantics,
    )


def _validate_sample_render_contract(
    render_plan: Mapping[str, Any],
    summary: Mapping[str, Any],
    scene_manifest: Any,
    source_record: Mapping[str, Any],
) -> tuple[Any, ...]:
    camera = render_plan.get("camera")
    expected_camera_fields = {
        "native_resolution",
        "training_resolution",
        "fov_axis",
        "fov_deg",
        "near_m",
        "far_m",
        "encoding",
        "mount_body",
    }
    if (
        not isinstance(camera, Mapping)
        or set(camera) != expected_camera_fields
        or camera.get("fov_axis") != "horizontal"
        or not math.isclose(
            float(camera.get("fov_deg", math.nan)),
            78.323,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(camera.get("near_m", math.nan)),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or _source_path(render_plan.get("frames_jsonl"))
        != Path(str(source_record["frames"]["path"]))
    ):
        raise RawSupervisionAuditError("sample render-plan camera/source changed")
    projection = summary.get("camera_projection")
    expected_projection_fields = {
        "model",
        "renderer_fov_axis",
        "horizontal_fov_deg",
        "vertical_fov_deg",
        "near_m",
        "far_m",
        "runtime_rectification_required",
    }
    expected_vertical = math.degrees(
        2.0
        * math.atan(
            math.tan(math.radians(float(camera["fov_deg"])) * 0.5)
            * (168.0 / 224.0)
        )
    )
    if (
        not isinstance(projection, Mapping)
        or set(projection) != expected_projection_fields
        or projection.get("model") != "pinhole"
        or projection.get("renderer_fov_axis") != "vertical"
        or projection.get("runtime_rectification_required") is not False
        or summary.get("resolution_wh") != [224, 168]
        or not math.isclose(
            float(projection.get("horizontal_fov_deg", math.nan)),
            float(camera["fov_deg"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(projection.get("vertical_fov_deg", math.nan)),
            expected_vertical,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            float(projection.get("near_m", math.nan)),
            float(camera["near_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(projection.get("far_m", math.nan)),
            float(camera["far_m"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RawSupervisionAuditError("sample plan/summary projection changed")
    _ensure_reviewed_source_semantics()
    source_v4._validate_raw_scene_object_records(scene_manifest.to_dict())
    rendered_boxes = tuple(source_v4._rendered_boxes(scene_manifest))
    object_records = source_v4.labels_v3._render_object_records(scene_manifest)
    object_ids = sorted(str(item["object_id"]) for item in object_records)
    parity = summary.get("object_parity")
    if (
        not isinstance(parity, Mapping)
        or parity.get("schema") != "lewm_render_object_parity_v1"
        or parity.get("rendered_groups")
        != ["wall", "obstacle", "landmark", "distractor"]
        or parity.get("rendered_object_count") != len(object_records)
        or parity.get("rendered_object_ids") != object_ids
        or parity.get("rendered_object_ids_sha256")
        != source_v4.canonical_json_sha256(object_ids)
        or parity.get("rendered_object_records_sha256")
        != source_v4.canonical_json_sha256(object_records)
        or parity.get("collision_distractors_rendered") is not True
        or parity.get("full_box_roll_pitch_yaw_rendered") is not True
    ):
        raise RawSupervisionAuditError("sample full-RPY render parity changed")
    return rendered_boxes


def _find_source_frame(
    payload: bytes,
    record: Mapping[str, Any],
    *,
    plan_camera_mount: Mapping[str, Any],
) -> Mapping[str, Any]:
    matches: list[Mapping[str, Any]] = []
    if not payload or not payload.endswith(b"\n"):
        raise RawSupervisionAuditError("source frames JSONL is not newline terminated")
    for line_number, line in enumerate(payload.splitlines(), start=1):
        frame = _decode_json(line, name=f"source frames:{line_number}")
        if (
            frame.get("frame_index") == record["frame_index"]
            and frame.get("env_index") == record["env_index"]
            and frame.get("timestamp_ns") == record["timestamp_ns"]
        ):
            matches.append(frame)
    if len(matches) != 1:
        raise RawSupervisionAuditError("sample source frame did not match exactly once")
    return source_v4._extract_source_frame(
        matches[0], record, plan_camera_mount_body=plan_camera_mount
    )


def _recompute_one_exact_sample(
    *,
    endpoint_digest: str,
    endpoint: Mapping[str, Any],
    pair_record: Mapping[str, Any],
    source_record: Mapping[str, Any],
) -> tuple[np.ndarray, ...]:
    frames_info = source_record["frames"]
    manifest_info = source_record["scene_manifest"]
    plan_info = source_record["render_plan"]
    summary_info = source_record["render_summary"]
    frames_payload = _read_absolute_bound_payload(
        Path(str(frames_info["path"])),
        str(frames_info["sha256"]),
        repository_root=ROOT,
        name="sample frames file",
    )
    manifest_payload = _read_exact_source_json(
        str(manifest_info["path"]),
        str(manifest_info["file_sha256"]),
        name="sample scene manifest",
    )
    render_plan = _read_exact_source_json(
        str(plan_info["path"]), str(plan_info["sha256"]), name="sample render plan"
    )
    summary = _read_exact_source_json(
        str(summary_info["path"]),
        str(summary_info["sha256"]),
        name="sample render summary",
    )
    scene_id = str(source_record["scene_id"])
    if (
        render_plan.get("schema") != "lewm_render_replay_plan_v0"
        or render_plan.get("scene_id") != scene_id
        or summary.get("schema") != "lewm_rendered_vision_v04"
        or summary.get("scene_id") != scene_id
        or summary.get("family") != source_record["family"]
        or summary.get("render_status") != "complete"
    ):
        raise RawSupervisionAuditError("sample plan/summary scene identity changed")
    source_section = summary.get("source")
    if not isinstance(source_section, Mapping) or set(source_section) != {
        "plan", "frames_jsonl", "scene_manifest", "renderer_source"
    }:
        raise RawSupervisionAuditError("sample render-summary source inventory changed")
    for summary_name, inventory_name in (
        ("frames_jsonl", "frames"),
        ("scene_manifest", "scene_manifest"),
        ("plan", "render_plan"),
    ):
        path, digest = _summary_source_entry(summary, summary_name)
        inventory_entry = source_record[inventory_name]
        expected_digest = inventory_entry.get("sha256", inventory_entry.get("file_sha256"))
        summary_path = Path(path)
        if not summary_path.is_absolute():
            summary_path = ROOT / summary_path
        if (
            summary_path != Path(str(inventory_entry["path"]))
            or digest != expected_digest
        ):
            raise RawSupervisionAuditError("render summary source inventory changed")
    camera = render_plan.get("camera")
    if not isinstance(camera, Mapping):
        raise RawSupervisionAuditError("render plan camera is absent")
    plan_mount = source_v4._camera_mount_record(
        camera.get("mount_body"), label="sample render plan camera.mount_body"
    )
    source_v4._validate_summary_records(
        summary,
        [pair_record],
        summary_path=Path(str(summary_info["path"])),
    )
    frame = _find_source_frame(frames_payload, pair_record, plan_camera_mount=plan_mount)
    sidecar_quaternion, stored_yaw = raycast_v4._validated_sidecar_source_attitude(
        frame, endpoint
    )
    scene_manifest = parse_scene_manifest_dict(manifest_payload)
    if (
        scene_manifest.scene_id != scene_id
        or scene_manifest.family != source_record["family"]
        or manifest_sha256(scene_manifest) != manifest_info["content_sha256"]
    ):
        raise RawSupervisionAuditError("sample scene manifest semantic hash changed")
    raw_rendered_boxes = _validate_sample_render_contract(
        render_plan, summary, scene_manifest, source_record
    )
    position = frame["base_pose_world"]["position"]
    base_position = tuple(float(position[axis]) for axis in ("x", "y", "z"))
    composed = dynamic_projection.compose_yaw_aligned_camera(
        sidecar_quaternion, stored_yaw
    )
    basis = raycast_v4._normalized_camera_basis_fru(composed)
    rendered_boxes = tuple(
        raycast_v4._box_in_yaw_body(
            box,
            base_position_world=base_position,
            stored_yaw_rad=stored_yaw,
        )
        for box in raw_rendered_boxes
    )
    frame_input = raycast_v4.FrameBuildInputV4(
        frame_key={
            "dataset_role": endpoint["identity"]["dataset_role"],
            "family": source_record["family"],
            "scene_id": scene_id,
            "endpoint_identity_sha256": endpoint_digest,
        },
        camera_origin_body_m=tuple(composed.origin_xyz),
        camera_basis_body_fru=basis,
        ground_plane_z_body_m=-base_position[2],
        rendered_boxes_body=rendered_boxes,
        image_path_metadata_only=str(endpoint["image_path_metadata_only"]),
        image_sha256=str(endpoint["identity"]["image_sha256"]),
        sidecar_row_identity_sha256=str(pair_record["sidecar_row_identity_sha256"]),
    )
    evidence = raycast_v4.build_frame_evidence_v4(frame_input)
    raster = raycast_v4.rasterize_observable_camera_ray_evidence_v4(evidence)
    return _stored_arrays_from_evidence(evidence, raster)


def _recompute_exact_sample_task(
    task: tuple[str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
) -> tuple[str, tuple[np.ndarray, ...]]:
    _set_worker_environment()
    digest, endpoint, pair_record, source_record = task
    return digest, _recompute_one_exact_sample(
        endpoint_digest=digest,
        endpoint=endpoint,
        pair_record=pair_record,
        source_record=source_record,
    )


def _exact_sample_recomputer(
    sample: Sequence[Mapping[str, Any]],
    endpoints: Mapping[str, Mapping[str, Any]],
    inputs: AuditInputs,
    workers: int,
) -> Mapping[str, tuple[np.ndarray, ...]]:
    source_by_scene = {str(item["scene_id"]): item for item in inputs.inventory.records}
    tasks: list[
        tuple[str, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]
    ] = []
    for sample_record in sample:
        digest = str(sample_record["endpoint_identity_sha256"])
        endpoint = endpoints[digest]
        pair_record = _source_record_for_endpoint(digest, endpoint, inputs.plan.pairs)
        source = source_by_scene.get(str(endpoint["identity"]["scene_id"]))
        if source is None:
            raise RawSupervisionAuditError("sample scene is absent from source inventory")
        if source["role"] != endpoint["identity"]["dataset_role"] or source["family"] != pair_record["family"]:
            raise RawSupervisionAuditError("sample source crossed role/family")
        tasks.append((digest, endpoint, pair_record, source))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(workers),
        mp_context=context,
        initializer=_set_worker_environment,
    ) as executor:
        replayed = list(executor.map(_recompute_exact_sample_task, tasks))
    result = dict(replayed)
    if len(result) != len(tasks):
        raise RawSupervisionAuditError("exact sample replay repeated an endpoint")
    return result


def audit_exact_dataset_v1(
    repo_root: Path,
    dataset_root: Path,
    *,
    expected_manifest_file_sha256: str,
    workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    """Run the sealed exact audit without accepting caller-provided replay logic."""

    if isinstance(workers, bool) or not 1 <= int(workers) <= MAX_WORKERS:
        raise ValueError(f"workers must lie in [1,{MAX_WORKERS}]")
    root = _require_real_directory(Path(repo_root), name="repository root")
    if root != ROOT or Path(dataset_root).absolute() != CANONICAL_DATASET:
        raise PermissionError("exact audit paths are fixed to the canonical repository")
    manifest = _preflight_exact_authorization(
        Path(dataset_root),
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    plan = plan_v5.load_frozen_development_metadata(root)
    inventory = plan_v5.load_frozen_development_source_inventory(root, plan)
    before = _hash_complete_source_inventory(inventory, workers=int(workers))
    parent_contracts_before = _parent_contract_receipts()
    inputs = AuditInputs(plan=plan, inventory=inventory)
    _validate_exact_manifest_bindings(
        manifest,
        inputs=inputs,
        hashed_sources=before,
        parent_contracts=parent_contracts_before,
    )
    result = audit_dataset_v1(
        dataset_root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        inputs=inputs,
        sample_recomputer=_exact_sample_recomputer,
        workers=int(workers),
        exact=True,
    )
    after = _hash_complete_source_inventory(inventory, workers=int(workers))
    parent_contracts_after = _parent_contract_receipts()
    if after != before or parent_contracts_after != parent_contracts_before:
        raise RawSupervisionAuditError("development source inventory changed during audit")
    final_plan = plan_v5.load_frozen_development_metadata(root)
    final_inventory = plan_v5.load_frozen_development_source_inventory(root, final_plan)
    if (
        final_plan.value != plan.value
        or final_plan.pairs != plan.pairs
        or final_plan.endpoints != plan.endpoints
        or final_inventory.records != inventory.records
        or final_inventory.hashes != inventory.hashes
    ):
        raise RawSupervisionAuditError("metadata V5 changed during exact audit")
    core = dict(result)
    core.pop("content_sha256")
    core["source_file_count"] = len(before) + len(parent_contracts_before)
    complete_receipts = _builder_source_receipts(before, parent_contracts_before)
    core["source_inventory_before_after_sha256"] = canonical_json_sha256(
        complete_receipts
    )
    core["source_payload_opens"] = {
        "complete_inventory_hash_passes": 2,
        "permitted_source_files_per_pass": len(complete_receipts),
        "sample_endpoint_count": EXPECTED_SAMPLE_COUNT,
        "rgb_byte_opens": 0,
        "rgb_decodes": 0,
        "label_shard_payload_opens": 0,
        "g2_payload_opens": 0,
        "checkpoint_model_runtime_heldout_hardware_production_opens": 0,
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def execute_exact_audit_v1(
    *,
    expected_manifest_file_sha256: str,
    workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    """Audit the one canonical dataset and exclusively publish one terminal leaf."""

    report_name = CANONICAL_AUDIT_REPORT.name
    failure_name = CANONICAL_AUDIT_FAILURE.name
    with _ExclusiveAuditPublisher(CANONICAL_AUDIT_REPORT.parent) as publisher:
        publisher.require_absent(report_name, failure_name)
        try:
            result = audit_exact_dataset_v1(
                ROOT,
                CANONICAL_DATASET,
                expected_manifest_file_sha256=expected_manifest_file_sha256,
                workers=workers,
            )
        except BaseException as error:
            failure_core = {
                "schema": "lewm_go2_shared_jepa_v5_raw_supervision_audit_failure_v1",
                "status": "terminal_failed_no_dataset_authority",
                "dataset_manifest_file_sha256": expected_manifest_file_sha256,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "canonical_dataset_present": CANONICAL_DATASET.exists(),
                "audit_report_present": False,
                "dataset_use_authorized": False,
                "training_authorized": False,
                "g2_authorized": False,
                "production_authorized": False,
                "retry_authorized": False,
            }
            failure = {
                **failure_core,
                "content_sha256": canonical_json_sha256(failure_core),
            }
            publisher.publish(failure_name, failure)
            raise
        publisher.publish(report_name, result)
        return result


__all__ = [
    "ARRAY_LAYOUT",
    "AUDIT_SCHEMA",
    "AuditInputs",
    "CANONICAL_DATASET",
    "CANONICAL_AUDIT_FAILURE",
    "CANONICAL_AUDIT_REPORT",
    "DATASET_SCHEMA",
    "ENDPOINT_INDEX_SCHEMA",
    "MAX_WORKERS",
    "RawSupervisionAuditError",
    "SHARD_SCHEMA",
    "audit_dataset_v1",
    "audit_exact_dataset_v1",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "execute_exact_audit_v1",
]
