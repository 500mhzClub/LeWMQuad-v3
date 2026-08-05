"""Additive V3 auditor for the Shared-JEPA V5 raw-supervision artifact.

V1 remains frozen with its independent BLOCK.  This successor keeps V1's
independent byte, join, V4-raster, and source-replay checks while closing the
four blocked trust boundaries:

* callback-driven audits are synthetic-only and have no exact switch;
* exact population declarations are derived from the audited rows first;
* authorization uses a zero-target-open structural phase followed by a fixed
  nine-target rehash and duplicate-key-safe review parse; and
* every dataset leaf, including the manifest, is an unaliased descriptor-bound
  regular file with strict integer cardinalities.

The exact entry point accepts neither loaders nor replay callbacks.  It is
fixed to the canonical repository, dataset, authorization, and reviewed V4
replayer.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import stat
from typing import Any, Callable, Mapping, Sequence

from lewm.datasets.go2_shared_jepa_v5_raw_supervision_auditor_v1 import (
    ARRAY_LAYOUT,
    AuditInputs,
    CANONICAL_DATASET,
    DATASET_SCHEMA,
    ENDPOINT_INDEX_SCHEMA,
    EXACT_INPUT_PROVENANCE_FIELDS,
    EXACT_ORDERED_ENDPOINT_SHA256,
    EXACT_ORDERED_PAIR_SHA256,
    EXACT_PLAN_CONTENT_SHA256,
    EXPECTED_SAMPLE_COUNT,
    FROZEN_PARENT_FILE_SHA256,
    GEOMETRY_CONTRACT_CONTENT_SHA256,
    GEOMETRY_CONTRACT_FILE_SHA256,
    MAX_WORKERS,
    RENDER_AUDIT_CONTENT_SHA256,
    RENDER_AUDIT_FILE_SHA256,
    REVIEWED_V4_SOURCE_SHA256,
    ROOT_FILE_FIELDS,
    RawSupervisionAuditError,
    SHARD_SCHEMA,
    _builder_source_receipts,
    _canonical_relative_path,
    _decode_json,
    _parse_canonical_jsonl,
    _validate_exact_access_ledger,
    _validate_frozen_source_map,
    _validate_manifest_constants,
    canonical_json_bytes,
    canonical_json_sha256,
)
from lewm.datasets.go2_shared_jepa_v5_raw_supervision_plan_v5 import (
    DEVELOPMENT_ROLES,
    SOURCE_INVENTORY_SHA256,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_AUDIT_REPORT = CANONICAL_DATASET.with_name(
    CANONICAL_DATASET.name + ".audit_v3.json"
)
CANONICAL_AUDIT_FAILURE = CANONICAL_DATASET.with_name(
    CANONICAL_DATASET.name + ".audit_v3.failed.json"
)
BUILD_AUTHORIZATION_PATH = (
    ROOT
    / "docs/lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_2026-07-13.json"
)

AUDIT_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_audit_v3"
SYNTHETIC_AUDIT_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_synthetic_audit_v3"
)
AUDIT_FAILURE_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_audit_failure_v3"
AUTHORIZATION_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v3"
)
BUILDER_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_builder_v3_independent_review_v1"
)
AUDITOR_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_independent_review_v1"
)
BUILDER_IMPLEMENTATION_AUTHOR = "/root/raw_builder_arch"
AUDITOR_IMPLEMENTATION_AUTHOR = "/root/raw_auditor_author"
REVIEW_BINDING_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v3"
)

SOURCE_ROLE_PATHS: tuple[tuple[str, str], ...] = (
    (
        "builder_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v3.py",
    ),
    (
        "builder_cli",
        "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v3.py",
    ),
    (
        "builder_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v3.py",
    ),
    (
        "builder_handoff",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v3_"
        "author_handoff_2026-07-13.md",
    ),
    (
        "builder_review",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v3_"
        "independent_review_2026-07-13.json",
    ),
    (
        "auditor_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v3.py",
    ),
    (
        "auditor_cli",
        "scripts/audit_go2_shared_jepa_v5_raw_supervision_v3.py",
    ),
    (
        "auditor_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v3.py",
    ),
    (
        "auditor_review",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v3_"
        "independent_review_2026-07-13.json",
    ),
)
SOURCE_ROLES = tuple(role for role, _path in SOURCE_ROLE_PATHS)
SOURCE_PATH_BY_ROLE = dict(SOURCE_ROLE_PATHS)
BUILDER_CANDIDATE_ROLES = (
    "builder_source",
    "builder_cli",
    "builder_test",
    "builder_handoff",
)
AUDITOR_CANDIDATE_ROLES = (
    "auditor_source",
    "auditor_cli",
    "auditor_test",
)
AUTHORIZATION_FIELDS = frozenset(
    {
        "schema",
        "exact_build_authorized_after_independent_reviews",
        "builder_review",
        "auditor_review",
        "source_map",
        "content_sha256",
    }
)
SOURCE_ENTRY_FIELDS = frozenset({"role", "path", "sha256"})
REVIEW_BINDING_FIELDS = frozenset(
    {
        "schema",
        "review_schema",
        "verdict",
        "reviewer",
        "implementation_author",
        "path",
        "file_sha256",
        "content_sha256",
        "candidate",
    }
)
REVIEW_RECORD_FIELDS = frozenset(
    {
        "schema",
        "verdict",
        "reviewer",
        "implementation_author",
        "candidate",
        "authority",
        "content_sha256",
    }
)
REVIEW_AUTHORITY_FALSE_FIELDS = (
    "exact_build_authorized",
    "exact_audit_authorized",
    "dataset_use_authorized",
    "training_authorized",
    "selection_authorized",
    "calibration_authorized",
    "g2_authorized",
    "heldout_authorized",
    "runtime_authorized",
    "navigation_authorized",
    "hardware_authorized",
    "production_authorized",
    "promotion_authorized",
)

FROZEN_PAIR_COUNTS = {
    "train": 4262,
    "checkpoint_selection": 495,
    "probability_calibration": 415,
}
FROZEN_UNIQUE_ENDPOINT_COUNTS = {
    "train": 7777,
    "checkpoint_selection": 924,
    "probability_calibration": 759,
}
FROZEN_ENDPOINT_REFERENCE_COUNT = 10344
FROZEN_UNIQUE_ENDPOINT_COUNT = 9460
FROZEN_PAIR_COUNT = 5172
FROZEN_SCENE_SHARD_COUNT = 88
FROZEN_FAMILY_COUNT_PER_ROLE = 8


SampleRecomputer = Callable[
    [
        Sequence[Mapping[str, Any]],
        Mapping[str, Mapping[str, Any]],
        AuditInputs,
        int,
    ],
    Mapping[str, tuple[Any, ...]],
]


@dataclass(frozen=True)
class DatasetPreflightV3:
    root: Path
    manifest: dict[str, Any]
    manifest_payload: bytes
    leaf_payloads: Mapping[str, bytes]
    pairs: tuple[dict[str, Any], ...]
    endpoints: tuple[dict[str, Any], ...]
    population: Mapping[str, Any]


@dataclass(frozen=True)
class SourceBindingV3:
    role: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ReviewBindingV3:
    kind: str
    review_schema: str
    verdict: str
    reviewer: str
    implementation_author: str
    path: str
    file_sha256: str
    content_sha256: str
    candidate: tuple[SourceBindingV3, ...]


@dataclass(frozen=True)
class PhaseOneAuthorizationV3:
    authorization_file_sha256: str
    authorization_content_sha256: str
    source_map_sha256: str
    canonical_payload: bytes
    sources: tuple[SourceBindingV3, ...]
    builder_review: ReviewBindingV3
    auditor_review: ReviewBindingV3


def _strict_workers(value: object) -> int:
    if type(value) is not int or not 1 <= value <= MAX_WORKERS:
        raise ValueError(f"workers must be an exact integer in [1,{MAX_WORKERS}]")
    return value


def _strict_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise RawSupervisionAuditError(f"{name} must be an exact integer")
    return value


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_shape(value: object, *, name: str) -> tuple[int, ...]:
    if type(value) is not list:
        raise RawSupervisionAuditError(f"{name} must be an exact shape list")
    return tuple(
        _strict_int(component, name=f"{name}[{index}]")
        for index, component in enumerate(value)
    )


_INTEGER_FIELD_NAMES = frozenset(
    {
        "byte_count",
        "development_scene_workers",
        "denied_or_unexpected_accesses",
        "endpoint_count",
        "endpoint_instance_count",
        "env_index",
        "episode_step",
        "expected_exact_record_count",
        "frame_index",
        "g2_label_payload_opens",
        "g2_rgb_byte_opens",
        "g2_sidecar_byte_opens",
        "g2_source_index_rows_read_for_exclusion",
        "g2_source_payload_opens",
        "geometry_contract_byte_opens",
        "global_row",
        "hardware_or_production_opens",
        "heldout_or_sealed_opens",
        "label_shard_row",
        "maximum_workers",
        "native_threads_per_worker",
        "pair_endpoint_references",
        "parent_label_shard_payload_opens",
        "render_audit_byte_opens",
        "render_plan_byte_opens",
        "render_summary_byte_opens",
        "reset_count",
        "rgb_byte_opens",
        "rgb_decodes",
        "row_count",
        "runtime_or_navigation_result_opens",
        "scene_shard_count",
        "shard_row",
        "source_frames_byte_opens",
        "source_frames_jsonl_records_scanned",
        "source_frames_selected_records",
        "source_payload_first_pass_file_count",
        "source_payload_second_pass_file_count",
        "source_payload_total_byte_opens",
        "source_scene_manifest_byte_opens",
        "timestamp_ns",
        "unique_endpoint_raycasts",
        "writes_outside_output_or_failure_namespace",
    }
)


def _validate_integer_fields(value: object, *, name: str) -> None:
    """Reject bool/float coercions at every declared cardinality boundary."""

    if type(value) is dict:
        for key, child in value.items():
            child_name = f"{name}.{key}"
            if key in {"shape", "trailing_shape"}:
                _strict_shape(child, name=child_name)
            elif key in _INTEGER_FIELD_NAMES:
                _strict_int(child, name=child_name)
            else:
                _validate_integer_fields(child, name=child_name)
    elif type(value) is list:
        for index, child in enumerate(value):
            _validate_integer_fields(child, name=f"{name}[{index}]")


def _validate_access_ledger_integers(value: object, *, name: str) -> None:
    if type(value) is not dict:
        raise RawSupervisionAuditError(f"{name} must be an exact object")
    for key, child in value.items():
        child_name = f"{name}.{key}"
        if type(child) is dict:
            _validate_access_ledger_integers(child, name=child_name)
        elif isinstance(child, (bool, int, float)):
            _strict_int(child, name=child_name)


def _bound_dataset_leaf(
    root: Path,
    relative: str,
    *,
    expected_sha256: str,
    expected_bytes: int | None,
    name: str,
) -> bytes:
    rel = _v1._canonical_relative_path(relative, name=name)
    path = root / rel
    payload = _v1._read_absolute_bound_payload(
        path,
        expected_sha256,
        repository_root=root,
        name=name,
    )
    if expected_bytes is not None and len(payload) != expected_bytes:
        raise RawSupervisionAuditError(f"{name} byte count changed")
    return payload


def _read_manifest_bound_v3(
    dataset_root: Path,
    *,
    expected_manifest_file_sha256: str,
) -> tuple[Path, bytes, dict[str, Any]]:
    if not _is_sha256(expected_manifest_file_sha256):
        raise RawSupervisionAuditError("expected manifest SHA-256 is malformed")
    root = _v1._require_real_directory(Path(dataset_root), name="dataset root")
    payload = _bound_dataset_leaf(
        root,
        "manifest.json",
        expected_sha256=expected_manifest_file_sha256,
        expected_bytes=None,
        name="dataset manifest",
    )
    if not payload.endswith(b"\n"):
        raise RawSupervisionAuditError("dataset manifest lacks terminal newline")
    manifest = _v1._decode_json(payload, name="dataset manifest")
    if canonical_json_bytes(manifest) + b"\n" != payload:
        raise RawSupervisionAuditError("dataset manifest is not canonical")
    _validate_integer_fields(manifest, name="dataset manifest")
    for field in ("pair_counts", "unique_endpoint_counts"):
        counts = manifest.get(field)
        if type(counts) is not dict:
            raise RawSupervisionAuditError(f"manifest {field} must be an exact object")
        for role, count in counts.items():
            _strict_int(count, name=f"manifest {field}.{role}")
    _validate_access_ledger_integers(
        manifest.get("access_ledger"), name="dataset manifest.access_ledger"
    )
    _v1._validate_manifest_constants(manifest, exact=False)
    return root, payload, manifest


def _strict_file_inventory_v3(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    manifest_payload: bytes,
) -> dict[str, bytes]:
    records = manifest.get("files")
    if type(records) is not list:
        raise RawSupervisionAuditError("manifest files must be an exact list")
    expected: dict[str, Mapping[str, Any]] = {}
    for index, record in enumerate(records):
        if type(record) is not dict or set(record) != _v1.ROOT_FILE_FIELDS:
            raise RawSupervisionAuditError(f"manifest file {index} fields changed")
        relative = str(
            _v1._canonical_relative_path(record.get("path"), name="manifest file")
        )
        if relative == "manifest.json" or relative in expected:
            raise RawSupervisionAuditError("manifest file inventory repeats/self-includes")
        _strict_int(record.get("byte_count"), name=f"{relative}.byte_count")
        if not _is_sha256(record.get("file_sha256")):
            raise RawSupervisionAuditError(f"{relative} SHA-256 is malformed")
        expected[relative] = record
    if list(expected) != sorted(expected):
        raise RawSupervisionAuditError("manifest file inventory is not ordered")

    observed: set[str] = set()
    for path in sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root))):
        metadata = path.lstat()
        relative = str(path.relative_to(root))
        if stat.S_ISLNK(metadata.st_mode):
            raise PermissionError(f"dataset tree contains alias {relative}")
        if stat.S_ISREG(metadata.st_mode):
            if int(metadata.st_nlink) != 1:
                raise PermissionError(f"dataset tree contains hard-link alias {relative}")
            observed.add(relative)
        elif not stat.S_ISDIR(metadata.st_mode):
            raise PermissionError(f"dataset tree contains special entry {relative}")
    if observed != set(expected) | {"manifest.json"}:
        raise RawSupervisionAuditError("manifest and filesystem inventories differ")

    payloads = {"manifest.json": manifest_payload}
    for relative, record in expected.items():
        payloads[relative] = _bound_dataset_leaf(
            root,
            relative,
            expected_sha256=str(record["file_sha256"]),
            expected_bytes=_strict_int(
                record["byte_count"], name=f"{relative}.byte_count"
            ),
            name=f"dataset file {relative}",
        )
    return payloads


def _parse_jsonl_payload(payload: bytes, *, name: str) -> tuple[dict[str, Any], ...]:
    rows = tuple(_v1._parse_canonical_jsonl(payload, name=name))
    for index, row in enumerate(rows):
        _validate_integer_fields(row, name=f"{name}[{index}]")
    return rows


def _declared_index_payloads(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    pair_record = manifest.get("pair_index")
    endpoint_record = manifest.get("endpoint_index")
    for name, record, path in (
        ("pair index", pair_record, "pairs.jsonl"),
        ("endpoint index", endpoint_record, "endpoints.jsonl"),
    ):
        if type(record) is not dict or set(record) != {
            "path",
            "row_count",
            "file_sha256",
        }:
            raise RawSupervisionAuditError(f"{name} manifest record changed")
        if record.get("path") != path:
            raise RawSupervisionAuditError(f"{name} path changed")
        _strict_int(record.get("row_count"), name=f"{name} row_count")
    pairs = _parse_jsonl_payload(payloads["pairs.jsonl"], name="pair index")
    endpoints = _parse_jsonl_payload(
        payloads["endpoints.jsonl"], name="endpoint index"
    )
    if pair_record["row_count"] != len(pairs):
        raise RawSupervisionAuditError("pair index row count changed")
    if endpoint_record["row_count"] != len(endpoints):
        raise RawSupervisionAuditError("endpoint index row count changed")
    return pairs, endpoints


def _validate_shard_cardinalities_v3(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> None:
    records = manifest.get("shards")
    if type(records) is not list:
        raise RawSupervisionAuditError("manifest shards must be an exact list")
    for index, record in enumerate(records):
        if type(record) is not dict:
            raise RawSupervisionAuditError(f"manifest shard {index} must be an object")
        count = _strict_int(
            record.get("endpoint_count"),
            name=f"manifest shard {index}.endpoint_count",
            minimum=1,
        )
        relative = str(record.get("path"))
        payload = payloads.get(relative)
        if payload is None:
            raise RawSupervisionAuditError(f"manifest shard {index} is absent")
        shard = _v1._decode_json(payload, name=f"scene shard {index}")
        _validate_integer_fields(shard, name=f"scene shard {index}")
        if _strict_int(
            shard.get("endpoint_count"),
            name=f"scene shard {index}.endpoint_count",
            minimum=1,
        ) != count:
            raise RawSupervisionAuditError("manifest/shard endpoint counts differ")
        files = shard.get("files")
        if type(files) is not list:
            raise RawSupervisionAuditError("shard files must be an exact list")
        directory = str(Path(relative).parent)
        for file_index, file_record in enumerate(files):
            if type(file_record) is not dict:
                raise RawSupervisionAuditError("shard file record must be an object")
            _strict_int(
                file_record.get("byte_count"),
                name=f"scene shard {index}.files[{file_index}].byte_count",
            )
            shape = _strict_shape(
                file_record.get("shape"),
                name=f"scene shard {index}.files[{file_index}].shape",
            )
            if file_record.get("path") == "index.jsonl" and shape != (count,):
                raise RawSupervisionAuditError("shard index shape changed")
        index_payload = payloads.get(f"{directory}/index.jsonl")
        if index_payload is None:
            raise RawSupervisionAuditError("shard index payload is absent")
        rows = _parse_jsonl_payload(index_payload, name=f"scene shard {index} index")
        if len(rows) != count:
            raise RawSupervisionAuditError("shard index row count changed")
        for row_index, row in enumerate(rows):
            if _strict_int(
                row.get("shard_row"),
                name=f"scene shard {index} index[{row_index}].shard_row",
            ) != row_index:
                raise RawSupervisionAuditError("shard index row order changed")


def _derive_population_v3(
    manifest: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    endpoints: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive every exact population from audited rows before frozen checks."""

    pair_counts: Counter[str] = Counter()
    reference_counts: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()
    pair_families: dict[str, set[str]] = defaultdict(set)
    endpoint_families: dict[str, set[str]] = defaultdict(set)
    endpoint_by_hash: dict[str, Mapping[str, Any]] = {}
    for index, endpoint in enumerate(endpoints):
        digest = endpoint.get("endpoint_identity_sha256")
        if not _is_sha256(digest) or digest in endpoint_by_hash:
            raise RawSupervisionAuditError("endpoint population repeats or is malformed")
        role = endpoint.get("dataset_role")
        family = endpoint.get("family")
        if type(role) is not str or type(family) is not str:
            raise RawSupervisionAuditError(f"endpoint {index} role/family changed")
        endpoint_by_hash[str(digest)] = endpoint
        unique_counts[role] += 1
        endpoint_families[role].add(family)

    referenced: set[str] = set()
    for index, pair in enumerate(pairs):
        role = pair.get("dataset_role")
        family = pair.get("family")
        if type(role) is not str or type(family) is not str:
            raise RawSupervisionAuditError(f"pair {index} role/family changed")
        pair_counts[role] += 1
        pair_families[role].add(family)
        for side in ("current", "next"):
            digest = pair.get(f"{side}_endpoint_sha256")
            endpoint = endpoint_by_hash.get(str(digest))
            if endpoint is None:
                raise RawSupervisionAuditError("pair references an absent endpoint")
            if (
                endpoint.get("dataset_role") != role
                or endpoint.get("family") != family
                or endpoint.get("scene_id") != pair.get("scene_id")
            ):
                raise RawSupervisionAuditError(
                    "pair reference crossed role/family/scene"
                )
            reference_counts[role] += 1
            referenced.add(str(digest))
    if referenced != set(endpoint_by_hash):
        raise RawSupervisionAuditError("endpoint population contains an orphan")

    shard_records = manifest.get("shards")
    if type(shard_records) is not list:
        raise RawSupervisionAuditError("manifest shards must be an exact list")
    shard_families: dict[str, set[str]] = defaultdict(set)
    shard_scenes: set[str] = set()
    endpoints_per_scene = Counter(str(row.get("scene_id")) for row in endpoints)
    endpoint_scene_contract: dict[str, tuple[str, str]] = {}
    for endpoint in endpoints:
        scene = str(endpoint.get("scene_id"))
        contract = (str(endpoint.get("dataset_role")), str(endpoint.get("family")))
        previous = endpoint_scene_contract.setdefault(scene, contract)
        if previous != contract:
            raise RawSupervisionAuditError(
                "endpoint scene crosses role/family populations"
            )
    for index, record in enumerate(shard_records):
        if type(record) is not dict:
            raise RawSupervisionAuditError(f"manifest shard {index} changed")
        role = record.get("dataset_role")
        family = record.get("family")
        scene = record.get("scene_id")
        if type(role) is not str or type(family) is not str or type(scene) is not str:
            raise RawSupervisionAuditError("manifest shard role/family/scene changed")
        if scene in shard_scenes:
            raise RawSupervisionAuditError("manifest shard scene repeats")
        shard_scenes.add(scene)
        shard_families[role].add(family)
        if endpoint_scene_contract.get(scene) != (role, family):
            raise RawSupervisionAuditError(
                "shard scene crosses endpoint role/family populations"
            )
        if _strict_int(
            record.get("endpoint_count"), name=f"manifest shard {index}.endpoint_count"
        ) != endpoints_per_scene[scene]:
            raise RawSupervisionAuditError("shard count disagrees with endpoint rows")
    if shard_scenes != set(endpoints_per_scene):
        raise RawSupervisionAuditError("shard scenes do not cover endpoint rows")

    roles = tuple(manifest.get("roles", ()))
    actual_roles = set(pair_counts) | set(unique_counts) | set(reference_counts)
    if (
        list(roles) != list(plan_v5.DEVELOPMENT_ROLES)
        or not actual_roles
        or not actual_roles <= set(roles)
    ):
        raise RawSupervisionAuditError("actual role population changed")
    for role in roles:
        if not (
            pair_families[role]
            == endpoint_families[role]
            == shard_families[role]
        ):
            raise RawSupervisionAuditError(
                f"actual family population disagrees in role {role}"
            )

    declared_pairs = manifest.get("pair_counts")
    declared_unique = manifest.get("unique_endpoint_counts")
    observed_pairs = {role: pair_counts[role] for role in roles}
    observed_unique = {role: unique_counts[role] for role in roles}
    observed_references = {role: reference_counts[role] for role in roles}
    if declared_pairs != observed_pairs:
        raise RawSupervisionAuditError("declared pair counts differ from audited rows")
    if declared_unique != observed_unique:
        raise RawSupervisionAuditError(
            "declared unique-endpoint counts differ from audited rows"
        )
    reference_total = sum(observed_references.values())
    if manifest.get("endpoint_instance_count") != reference_total:
        raise RawSupervisionAuditError(
            "declared endpoint-reference count differs from audited rows"
        )
    if manifest.get("scene_shard_count") != len(shard_records):
        raise RawSupervisionAuditError(
            "declared scene-shard count differs from audited rows"
        )
    family_counts = {role: len(pair_families[role]) for role in roles}
    return {
        "pair_counts": observed_pairs,
        "pair_count": len(pairs),
        "endpoint_reference_counts": observed_references,
        "endpoint_reference_count": reference_total,
        "unique_endpoint_counts": observed_unique,
        "unique_endpoint_count": len(endpoints),
        "role_count": len(actual_roles),
        "family_counts": family_counts,
        "scene_shard_count": len(shard_records),
    }


def _validate_frozen_population_v3(population: Mapping[str, Any]) -> None:
    if population.get("pair_counts") != FROZEN_PAIR_COUNTS:
        raise RawSupervisionAuditError("frozen actual pair counts changed")
    if population.get("unique_endpoint_counts") != FROZEN_UNIQUE_ENDPOINT_COUNTS:
        raise RawSupervisionAuditError("frozen actual unique-endpoint counts changed")
    if population.get("pair_count") != FROZEN_PAIR_COUNT:
        raise RawSupervisionAuditError("frozen actual pair population changed")
    if population.get("unique_endpoint_count") != FROZEN_UNIQUE_ENDPOINT_COUNT:
        raise RawSupervisionAuditError("frozen actual endpoint population changed")
    if population.get("endpoint_reference_count") != FROZEN_ENDPOINT_REFERENCE_COUNT:
        raise RawSupervisionAuditError("frozen actual endpoint references changed")
    if population.get("scene_shard_count") != FROZEN_SCENE_SHARD_COUNT:
        raise RawSupervisionAuditError("frozen actual shard population changed")
    if population.get("role_count") != len(plan_v5.DEVELOPMENT_ROLES):
        raise RawSupervisionAuditError("frozen actual role population changed")
    if population.get("family_counts") != {
        role: FROZEN_FAMILY_COUNT_PER_ROLE for role in plan_v5.DEVELOPMENT_ROLES
    }:
        raise RawSupervisionAuditError("frozen actual family population changed")


def _preflight_dataset_v3(
    dataset_root: Path,
    *,
    expected_manifest_file_sha256: str,
    exact: bool,
    initial: tuple[Path, bytes, dict[str, Any]] | None = None,
) -> DatasetPreflightV3:
    root, manifest_payload, manifest = (
        initial
        if initial is not None
        else _read_manifest_bound_v3(
            dataset_root,
            expected_manifest_file_sha256=expected_manifest_file_sha256,
        )
    )
    payloads = _strict_file_inventory_v3(
        root, manifest, manifest_payload=manifest_payload
    )
    pairs, endpoints = _declared_index_payloads(manifest, payloads)
    _validate_shard_cardinalities_v3(manifest, payloads)
    population = _derive_population_v3(manifest, pairs, endpoints)
    if exact:
        _validate_frozen_population_v3(population)
    return DatasetPreflightV3(
        root=root,
        manifest=manifest,
        manifest_payload=manifest_payload,
        leaf_payloads=payloads,
        pairs=pairs,
        endpoints=endpoints,
        population=population,
    )


def _source_binding_v3(entry: object) -> SourceBindingV3:
    if type(entry) is not dict or set(entry) != SOURCE_ENTRY_FIELDS:
        raise RawSupervisionAuditError("authorization source entry is malformed")
    role = entry.get("role")
    path = entry.get("path")
    digest = entry.get("sha256")
    if type(role) is not str or not role or not _is_sha256(digest):
        raise RawSupervisionAuditError("authorization source entry is malformed")
    relative = str(
        _v1._canonical_relative_path(path, name=f"authorized source {role}")
    )
    return SourceBindingV3(role=role, path=relative, sha256=str(digest))


def _candidate_bindings_v3(
    value: object,
    *,
    roles: Sequence[str],
    source_by_role: Mapping[str, SourceBindingV3],
) -> tuple[SourceBindingV3, ...]:
    if type(value) is not list or len(value) != len(roles):
        raise RawSupervisionAuditError("authorization review candidate changed")
    candidate = tuple(_source_binding_v3(entry) for entry in value)
    if tuple(item.role for item in candidate) != tuple(roles):
        raise RawSupervisionAuditError("authorization review candidate roles changed")
    if candidate != tuple(source_by_role[role] for role in roles):
        raise RawSupervisionAuditError("authorization review candidate binding changed")
    return candidate


def _review_binding_v3(
    value: object,
    *,
    kind: str,
    review_role: str,
    review_schema: str,
    implementation_author: str,
    candidate_roles: Sequence[str],
    source_by_role: Mapping[str, SourceBindingV3],
) -> ReviewBindingV3:
    if type(value) is not dict or set(value) != REVIEW_BINDING_FIELDS:
        raise RawSupervisionAuditError(
            f"authorization {kind} review binding is malformed"
        )
    if (
        value.get("schema") != REVIEW_BINDING_SCHEMA
        or value.get("review_schema") != review_schema
        or value.get("verdict") != "PASS"
    ):
        raise PermissionError(f"authorization {kind} review is not a bound PASS")
    reviewer = value.get("reviewer")
    author = value.get("implementation_author")
    if (
        type(reviewer) is not str
        or not reviewer
        or type(author) is not str
        or author != implementation_author
        or reviewer == author
    ):
        raise PermissionError(f"authorization {kind} review lacks a distinct reviewer")
    review_source = source_by_role[review_role]
    path = str(
        _v1._canonical_relative_path(
            value.get("path"), name=f"authorization {kind} review"
        )
    )
    if (
        path != review_source.path
        or value.get("file_sha256") != review_source.sha256
        or not _is_sha256(value.get("content_sha256"))
    ):
        raise RawSupervisionAuditError(
            f"authorization {kind} review file binding changed"
        )
    candidate = _candidate_bindings_v3(
        value.get("candidate"),
        roles=candidate_roles,
        source_by_role=source_by_role,
    )
    return ReviewBindingV3(
        kind=kind,
        review_schema=review_schema,
        verdict="PASS",
        reviewer=reviewer,
        implementation_author=author,
        path=path,
        file_sha256=review_source.sha256,
        content_sha256=str(value["content_sha256"]),
        candidate=candidate,
    )


def _validate_authorization_phase_one_v3(
    payload: Mapping[str, Any],
    *,
    authorization_file_sha256: str,
    authorization_content_sha256: str,
    authorization_source_map_sha256: str,
) -> PhaseOneAuthorizationV3:
    """Validate complete V3 authority without opening any mapped target."""

    if type(payload) is not dict or set(payload) != AUTHORIZATION_FIELDS:
        raise RawSupervisionAuditError("build authorization object fields changed")
    if not _is_sha256(authorization_file_sha256):
        raise PermissionError("build authorization file hash is not frozen")
    if (
        payload.get("schema") != AUTHORIZATION_SCHEMA
        or payload.get("exact_build_authorized_after_independent_reviews") is not True
    ):
        raise PermissionError("raw-supervision exact audit is not authorized")
    core = dict(payload)
    declared = core.pop("content_sha256", None)
    if (
        not _is_sha256(declared)
        or canonical_json_sha256(core) != declared
        or declared != authorization_content_sha256
    ):
        raise RawSupervisionAuditError("build authorization content hash changed")

    source_map = payload.get("source_map")
    if type(source_map) is not list or len(source_map) != len(SOURCE_ROLE_PATHS):
        raise RawSupervisionAuditError("build authorization source map is incomplete")
    if (
        not _is_sha256(authorization_source_map_sha256)
        or canonical_json_sha256(source_map) != authorization_source_map_sha256
    ):
        raise RawSupervisionAuditError("build authorization source-map hash changed")
    sources = tuple(_source_binding_v3(entry) for entry in source_map)
    roles = tuple(item.role for item in sources)
    paths = tuple(item.path for item in sources)
    if len(set(roles)) != len(roles) or len(set(paths)) != len(paths):
        raise RawSupervisionAuditError("authorization source map repeats a role/path")
    if roles != SOURCE_ROLES:
        raise RawSupervisionAuditError("authorization source roles/order changed")
    if paths != tuple(path for _role, path in SOURCE_ROLE_PATHS):
        raise RawSupervisionAuditError("authorization role-to-path mapping changed")
    source_by_role = {item.role: item for item in sources}

    builder_review = _review_binding_v3(
        payload.get("builder_review"),
        kind="builder",
        review_role="builder_review",
        review_schema=BUILDER_REVIEW_SCHEMA,
        implementation_author=BUILDER_IMPLEMENTATION_AUTHOR,
        candidate_roles=BUILDER_CANDIDATE_ROLES,
        source_by_role=source_by_role,
    )
    auditor_review = _review_binding_v3(
        payload.get("auditor_review"),
        kind="auditor",
        review_role="auditor_review",
        review_schema=AUDITOR_REVIEW_SCHEMA,
        implementation_author=AUDITOR_IMPLEMENTATION_AUTHOR,
        candidate_roles=AUDITOR_CANDIDATE_ROLES,
        source_by_role=source_by_role,
    )
    if builder_review.implementation_author == auditor_review.implementation_author:
        raise PermissionError("builder and auditor implementations are not independent")
    if builder_review.reviewer == auditor_review.reviewer:
        raise PermissionError("builder and auditor reviews are not independent")
    return PhaseOneAuthorizationV3(
        authorization_file_sha256=authorization_file_sha256,
        authorization_content_sha256=str(declared),
        source_map_sha256=authorization_source_map_sha256,
        canonical_payload=canonical_json_bytes(payload),
        sources=sources,
        builder_review=builder_review,
        auditor_review=auditor_review,
    )


def _review_candidate_value_v3(
    candidate: Sequence[SourceBindingV3],
) -> list[dict[str, str]]:
    return [
        {"role": item.role, "path": item.path, "sha256": item.sha256}
        for item in candidate
    ]


def _expected_review_authority_v3(kind: str) -> dict[str, bool]:
    return {
        f"{kind}_source_approved": True,
        **{field: False for field in REVIEW_AUTHORITY_FALSE_FIELDS},
    }


def _validate_review_record_v3(raw: bytes, binding: ReviewBindingV3) -> None:
    review = _v1._decode_json(raw, name=f"{binding.kind} independent review")
    if raw != canonical_json_bytes(review) + b"\n":
        raise RawSupervisionAuditError(
            f"{binding.kind} independent review is not canonical JSON"
        )
    if set(review) != REVIEW_RECORD_FIELDS:
        raise RawSupervisionAuditError(
            f"{binding.kind} independent review fields changed"
        )
    core = dict(review)
    declared = core.pop("content_sha256", None)
    if (
        not _is_sha256(declared)
        or canonical_json_sha256(core) != declared
        or declared != binding.content_sha256
    ):
        raise RawSupervisionAuditError(
            f"{binding.kind} independent review content hash changed"
        )
    if (
        review.get("schema") != binding.review_schema
        or review.get("verdict") != "PASS"
        or review.get("reviewer") != binding.reviewer
        or review.get("implementation_author") != binding.implementation_author
        or review.get("candidate") != _review_candidate_value_v3(binding.candidate)
        or review.get("authority") != _expected_review_authority_v3(binding.kind)
    ):
        raise PermissionError(
            f"{binding.kind} independent review PASS binding changed"
        )


def _validate_authorization_phase_two_v3(
    phase_one: PhaseOneAuthorizationV3,
    *,
    repository_root: Path = ROOT,
) -> dict[str, bytes]:
    """Rehash exactly nine literal targets after completed phase one."""

    if type(phase_one) is not PhaseOneAuthorizationV3:
        raise TypeError("phase two requires a completed V3 phase-one result")
    embedded = _v1._decode_json(
        phase_one.canonical_payload + b"\n",
        name="phase-one authorization capsule",
    )
    if canonical_json_bytes(embedded) != phase_one.canonical_payload:
        raise PermissionError("phase-one authorization capsule was fabricated")
    revalidated = _validate_authorization_phase_one_v3(
        embedded,
        authorization_file_sha256=phase_one.authorization_file_sha256,
        authorization_content_sha256=phase_one.authorization_content_sha256,
        authorization_source_map_sha256=phase_one.source_map_sha256,
    )
    if revalidated != phase_one:
        raise PermissionError("phase-one authorization capsule was fabricated")
    payload_by_role: dict[str, bytes] = {}
    for source in phase_one.sources:
        payload_by_role[source.role] = _v1._read_absolute_bound_payload(
            repository_root / source.path,
            source.sha256,
            repository_root=repository_root,
            name=f"authorized source {source.role}",
        )
    _validate_review_record_v3(
        payload_by_role["builder_review"], phase_one.builder_review
    )
    _validate_review_record_v3(
        payload_by_role["auditor_review"], phase_one.auditor_review
    )
    return payload_by_role


def _require_exact_authorization_v3(
    expected_authorization_file_sha256: str,
) -> PhaseOneAuthorizationV3:
    """Validate fixed V3 authority before any dataset or metadata open."""

    if not _is_sha256(expected_authorization_file_sha256):
        raise RawSupervisionAuditError("build authorization file hash is malformed")
    raw = _v1._read_absolute_bound_payload(
        BUILD_AUTHORIZATION_PATH,
        expected_authorization_file_sha256,
        repository_root=ROOT,
        name="build authorization",
    )
    authorization = _v1._decode_json(raw, name="build authorization")
    if raw != canonical_json_bytes(authorization) + b"\n":
        raise RawSupervisionAuditError("build authorization is not canonical JSON")
    phase_one = _validate_authorization_phase_one_v3(
        authorization,
        authorization_file_sha256=expected_authorization_file_sha256,
        authorization_content_sha256=str(authorization.get("content_sha256", "")),
        authorization_source_map_sha256=canonical_json_sha256(
            authorization.get("source_map")
        ),
    )
    _validate_authorization_phase_two_v3(phase_one, repository_root=ROOT)
    return phase_one


def _validate_manifest_authorization_binding_v3(
    provenance: Mapping[str, Any], authorization: PhaseOneAuthorizationV3
) -> None:
    if (
        provenance.get("authorization_file_sha256")
        != authorization.authorization_file_sha256
        or provenance.get("authorization_content_sha256")
        != authorization.authorization_content_sha256
        or provenance.get("authorization_source_map_sha256")
        != authorization.source_map_sha256
    ):
        raise PermissionError("dataset manifest authorization binding changed")


def _validate_exact_authorization_v3(
    provenance: Mapping[str, Any],
) -> PhaseOneAuthorizationV3:
    """Compatibility helper: require fixed authority and bind a manifest record."""

    authorization_sha256 = provenance.get("authorization_file_sha256")
    if not _is_sha256(authorization_sha256):
        raise RawSupervisionAuditError("build authorization file hash is malformed")
    phase_one = _require_exact_authorization_v3(str(authorization_sha256))
    _validate_manifest_authorization_binding_v3(provenance, phase_one)
    return phase_one


def _validate_exact_manifest_bindings_v3(
    manifest: Mapping[str, Any],
    *,
    inputs: AuditInputs,
    hashed_sources: Sequence[Mapping[str, Any]],
    parent_contracts: Sequence[Mapping[str, Any]],
    authorization: PhaseOneAuthorizationV3,
) -> None:
    frames_scanned = sum(
        _strict_int(
            item.get("jsonl_record_count", 0),
            name="hashed source jsonl_record_count",
        )
        for item in hashed_sources
        if item.get("kind") == "frames"
    )
    _v1._validate_exact_access_ledger(
        manifest.get("access_ledger"),
        inputs=inputs,
        frames_scanned=frames_scanned,
    )
    receipts = _v1._builder_source_receipts(hashed_sources, parent_contracts)
    value = manifest.get("input_provenance")
    if type(value) is not dict or set(value) != _v1.EXACT_INPUT_PROVENANCE_FIELDS:
        raise RawSupervisionAuditError("exact input-provenance fields changed")
    if (
        value.get("authorization_file_sha256")
        != authorization.authorization_file_sha256
        or value.get("authorization_content_sha256")
        != authorization.authorization_content_sha256
        or value.get("authorization_source_map_sha256")
        != authorization.source_map_sha256
        or value.get("metadata_plan_content_sha256") != _v1.EXACT_PLAN_CONTENT_SHA256
        or value.get("metadata_plan_content_sha256")
        != inputs.plan.value.get("content_sha256")
        or value.get("metadata_ordered_pair_sha256") != _v1.EXACT_ORDERED_PAIR_SHA256
        or value.get("metadata_ordered_pair_sha256")
        != inputs.plan.value.get("ordered_pair_sha256")
        or value.get("metadata_ordered_endpoint_sha256")
        != _v1.EXACT_ORDERED_ENDPOINT_SHA256
        or value.get("metadata_ordered_endpoint_sha256")
        != inputs.plan.value.get("ordered_endpoint_sha256")
        or value.get("source_inventory_sha256") != dict(inputs.inventory.hashes)
        or value.get("source_inventory_sha256")
        != dict(plan_v5.SOURCE_INVENTORY_SHA256)
    ):
        raise RawSupervisionAuditError("exact metadata/authorization provenance changed")
    _v1._validate_frozen_source_map(
        value.get("frozen_parent_file_sha256"),
        _v1.FROZEN_PARENT_FILE_SHA256,
        name="frozen parent",
    )
    _v1._validate_frozen_source_map(
        value.get("reviewed_v4_source_sha256"),
        _v1.REVIEWED_V4_SOURCE_SHA256,
        name="reviewed V4 source",
    )
    if (
        value.get("geometry_contract_file_sha256")
        != _v1.GEOMETRY_CONTRACT_FILE_SHA256
        or value.get("geometry_contract_content_sha256")
        != _v1.GEOMETRY_CONTRACT_CONTENT_SHA256
        or value.get("render_audit_file_sha256") != _v1.RENDER_AUDIT_FILE_SHA256
        or value.get("render_audit_content_sha256")
        != _v1.RENDER_AUDIT_CONTENT_SHA256
        or value.get("source_payload_inventory") != list(receipts)
        or value.get("source_payload_inventory_sha256")
        != canonical_json_sha256(list(receipts))
    ):
        raise RawSupervisionAuditError("exact source payload provenance changed")


def _v3_report(
    v1_result: Mapping[str, Any],
    population: Mapping[str, Any],
    *,
    authoritative_exact: bool,
) -> dict[str, Any]:
    core = dict(v1_result)
    core.pop("content_sha256", None)
    core["schema"] = AUDIT_SCHEMA if authoritative_exact else SYNTHETIC_AUDIT_SCHEMA
    core["audit_scope"] = (
        "sealed_exact_fixed_loader_and_replay"
        if authoritative_exact
        else "synthetic_non_authoritative_callback"
    )
    core["observed_population"] = dict(population)
    core["strict_integer_cardinalities"] = True
    core["unaliased_descriptor_bound_dataset_leaves"] = True
    return {**core, "content_sha256": canonical_json_sha256(core)}


def audit_dataset_v3(
    dataset_root: Path,
    *,
    expected_manifest_file_sha256: str,
    inputs: AuditInputs,
    sample_recomputer: SampleRecomputer,
    workers: int = 1,
) -> dict[str, Any]:
    """Audit a synthetic fixture; this callback path can never be exact."""

    worker_count = _strict_workers(workers)
    if not isinstance(inputs, AuditInputs):
        raise TypeError("inputs must be AuditInputs")
    before = _preflight_dataset_v3(
        dataset_root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        exact=False,
    )
    result = _v1.audit_dataset_v1(
        before.root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        inputs=inputs,
        sample_recomputer=sample_recomputer,
        workers=worker_count,
        exact=False,
    )
    after = _preflight_dataset_v3(
        dataset_root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        exact=False,
    )
    if (
        after.manifest_payload != before.manifest_payload
        or after.population != before.population
        or set(after.leaf_payloads) != set(before.leaf_payloads)
        or any(
            after.leaf_payloads[name] != payload
            for name, payload in before.leaf_payloads.items()
        )
    ):
        raise RawSupervisionAuditError("dataset changed during V3 audit")
    return _v3_report(result, before.population, authoritative_exact=False)


audit_synthetic_dataset_v3 = audit_dataset_v3


def audit_exact_dataset_v3(
    repo_root: Path,
    dataset_root: Path,
    *,
    expected_manifest_file_sha256: str,
    expected_authorization_file_sha256: str,
    workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    """Run the sealed exact audit with fixed paths, loaders, and replay code."""

    worker_count = _strict_workers(workers)
    root = _v1._require_real_directory(Path(repo_root), name="repository root")
    if root != ROOT or Path(dataset_root).absolute() != CANONICAL_DATASET:
        raise PermissionError("exact audit paths are fixed to the canonical repository")

    authorization = _require_exact_authorization_v3(
        expected_authorization_file_sha256
    )

    initial = _read_manifest_bound_v3(
        dataset_root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
    )
    provenance = initial[2].get("input_provenance")
    if type(provenance) is not dict or set(provenance) != _v1.EXACT_INPUT_PROVENANCE_FIELDS:
        raise RawSupervisionAuditError("exact input-provenance fields changed")
    _validate_manifest_authorization_binding_v3(provenance, authorization)
    preflight = _preflight_dataset_v3(
        dataset_root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        exact=True,
        initial=initial,
    )

    plan = plan_v5.load_frozen_development_metadata(root)
    inventory = plan_v5.load_frozen_development_source_inventory(root, plan)
    before = _v1._hash_complete_source_inventory(inventory, workers=worker_count)
    parent_contracts_before = _v1._parent_contract_receipts()
    inputs = AuditInputs(plan=plan, inventory=inventory)
    _validate_exact_manifest_bindings_v3(
        preflight.manifest,
        inputs=inputs,
        hashed_sources=before,
        parent_contracts=parent_contracts_before,
        authorization=authorization,
    )
    v1_result = _v1.audit_dataset_v1(
        preflight.root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        inputs=inputs,
        sample_recomputer=_v1._exact_sample_recomputer,
        workers=worker_count,
        exact=False,
    )
    dataset_after = _preflight_dataset_v3(
        preflight.root,
        expected_manifest_file_sha256=expected_manifest_file_sha256,
        exact=True,
    )
    if (
        dataset_after.manifest_payload != preflight.manifest_payload
        or dataset_after.population != preflight.population
        or set(dataset_after.leaf_payloads) != set(preflight.leaf_payloads)
        or any(
            dataset_after.leaf_payloads[name] != payload
            for name, payload in preflight.leaf_payloads.items()
        )
    ):
        raise RawSupervisionAuditError("exact dataset changed during V3 audit")
    result = _v3_report(
        v1_result, preflight.population, authoritative_exact=True
    )
    after = _v1._hash_complete_source_inventory(inventory, workers=worker_count)
    parent_contracts_after = _v1._parent_contract_receipts()
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
    core.pop("content_sha256", None)
    core["source_file_count"] = len(before) + len(parent_contracts_before)
    receipts = _v1._builder_source_receipts(before, parent_contracts_before)
    core["source_inventory_before_after_sha256"] = canonical_json_sha256(receipts)
    core["authorization_v3"] = {
        "file_sha256": authorization.authorization_file_sha256,
        "content_sha256": authorization.authorization_content_sha256,
        "source_map_sha256": authorization.source_map_sha256,
        "phase_one_zero_target_opens": True,
        "phase_two_fixed_target_count": len(authorization.sources),
        "machine_pass_reviews_parsed": 2,
    }
    core["source_payload_opens"] = {
        "complete_inventory_hash_passes": 2,
        "permitted_source_files_per_pass": len(receipts),
        "sample_endpoint_count": _v1.EXPECTED_SAMPLE_COUNT,
        "rgb_byte_opens": 0,
        "rgb_decodes": 0,
        "label_shard_payload_opens": 0,
        "g2_payload_opens": 0,
        "checkpoint_model_runtime_heldout_hardware_production_opens": 0,
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def execute_exact_audit_v3(
    *,
    expected_manifest_file_sha256: str,
    expected_authorization_file_sha256: str,
    workers: int = MAX_WORKERS,
) -> dict[str, Any]:
    """Audit the canonical dataset and publish one immutable V3 terminal leaf."""

    report_name = CANONICAL_AUDIT_REPORT.name
    failure_name = CANONICAL_AUDIT_FAILURE.name
    with _v1._ExclusiveAuditPublisher(CANONICAL_AUDIT_REPORT.parent) as publisher:
        publisher.require_absent(report_name, failure_name)
        try:
            result = audit_exact_dataset_v3(
                ROOT,
                CANONICAL_DATASET,
                expected_manifest_file_sha256=expected_manifest_file_sha256,
                expected_authorization_file_sha256=(
                    expected_authorization_file_sha256
                ),
                workers=workers,
            )
        except BaseException as error:
            failure_core = {
                "schema": AUDIT_FAILURE_SCHEMA,
                "status": "terminal_failed_no_dataset_authority",
                "dataset_manifest_file_sha256": expected_manifest_file_sha256,
                "authorization_file_sha256": expected_authorization_file_sha256,
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
    "CANONICAL_AUDIT_FAILURE",
    "CANONICAL_AUDIT_REPORT",
    "CANONICAL_DATASET",
    "DATASET_SCHEMA",
    "ENDPOINT_INDEX_SCHEMA",
    "MAX_WORKERS",
    "RawSupervisionAuditError",
    "SHARD_SCHEMA",
    "SYNTHETIC_AUDIT_SCHEMA",
    "audit_dataset_v3",
    "audit_exact_dataset_v3",
    "audit_synthetic_dataset_v3",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "execute_exact_audit_v3",
]
