"""Two-phase authorization successor for the Shared-JEPA V5 raw builder.

V2 leaves the frozen V1 construction, evidence, layout, and publication engine
unchanged.  It replaces only the exact-build authority boundary.  Phase one is
pure validation: no metadata, reviewed-source, or referenced-source target is
opened.  Phase two rehashes nine literal reviewed implementation targets and
validates the two machine-readable PASS records before any parent metadata or
development source may be opened.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
import multiprocessing
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from lewm.datasets import go2_shared_jepa_v5_raw_supervision_builder_v1 as _v1


ROOT = _v1.ROOT
CANONICAL_OUTPUT = _v1.CANONICAL_OUTPUT
FAILURE_RECEIPT = _v1.FAILURE_RECEIPT
AUTHORIZATION_PATH = _v1.AUTHORIZATION_PATH

DATASET_SCHEMA = _v1.DATASET_SCHEMA
SHARD_SCHEMA = _v1.SHARD_SCHEMA
ENDPOINT_INDEX_SCHEMA = _v1.ENDPOINT_INDEX_SCHEMA
FAILURE_SCHEMA = _v1.FAILURE_SCHEMA
AUTHORIZATION_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_build_authorization_v2"
REVIEW_BINDING_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_implementation_review_binding_v2"
)
BUILDER_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_builder_v2_independent_review_v1"
)
AUDITOR_REVIEW_SCHEMA = (
    "lewm_go2_shared_jepa_v5_raw_supervision_auditor_v2_independent_review_v1"
)
BUILDER_IMPLEMENTATION_AUTHOR = "/root/raw_builder_arch"
AUDITOR_IMPLEMENTATION_AUTHOR = "/root/raw_auditor_author"

MAX_WORKERS = _v1.MAX_WORKERS
THREAD_ENVIRONMENT = _v1.THREAD_ENVIRONMENT
ACCELERATOR_ENVIRONMENT = _v1.ACCELERATOR_ENVIRONMENT
ARRAY_LAYOUT = _v1.ARRAY_LAYOUT
FROZEN_PARENT_HASHES = _v1.FROZEN_PARENT_HASHES
REVIEWED_V4_SOURCES = _v1.REVIEWED_V4_SOURCES
EXACT_ACCESS_LEDGER_KEYS = _v1.EXACT_ACCESS_LEDGER_KEYS

PreparedEndpointV1 = _v1.PreparedEndpointV1
PreparedSceneJobV1 = _v1.PreparedSceneJobV1
RawSupervisionBuildError = _v1.RawSupervisionBuildError
build_prepared_dataset_v1 = _v1.build_prepared_dataset_v1
canonical_json_bytes = _v1.canonical_json_bytes
canonical_json_sha256 = _v1.canonical_json_sha256

# Exposed as a module seam so adversarial tests can instrument every descriptor
# target read.  Production calls still use the frozen V1 no-follow reader.
_read_bound_regular_file = _v1._read_bound_regular_file


AUTHORIZED_ROLE_PATHS: tuple[tuple[str, str], ...] = (
    (
        "builder_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_builder_v2.py",
    ),
    (
        "builder_cli",
        "scripts/build_go2_shared_jepa_v5_development_raw_supervision_v2.py",
    ),
    (
        "builder_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_builder_v2.py",
    ),
    (
        "builder_handoff",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_"
        "author_handoff_2026-07-13.md",
    ),
    (
        "builder_review",
        "docs/lewm_go2_shared_jepa_v5_development_raw_supervision_builder_v2_"
        "independent_review_2026-07-13.json",
    ),
    (
        "auditor_source",
        "lewm/datasets/go2_shared_jepa_v5_raw_supervision_auditor_v2.py",
    ),
    (
        "auditor_cli",
        "scripts/audit_go2_shared_jepa_v5_raw_supervision_v2.py",
    ),
    (
        "auditor_test",
        "lewm/tests/test_go2_shared_jepa_v5_raw_supervision_auditor_v2.py",
    ),
    (
        "auditor_review",
        "docs/lewm_go2_shared_jepa_v5_raw_supervision_auditor_v2_"
        "independent_review_2026-07-13.json",
    ),
)
ROLE_PATH_BY_NAME = dict(AUTHORIZED_ROLE_PATHS)
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


@dataclass(frozen=True)
class SourceBindingV2:
    role: str
    path: str
    sha256: str


@dataclass(frozen=True)
class ReviewBindingV2:
    kind: str
    review_schema: str
    verdict: str
    reviewer: str
    implementation_author: str
    path: str
    file_sha256: str
    content_sha256: str
    candidate: tuple[SourceBindingV2, ...]


@dataclass(frozen=True)
class PhaseOneAuthorizationV2:
    authorization_file_sha256: str
    canonical_payload: bytes
    sources: tuple[SourceBindingV2, ...]
    builder_review: ReviewBindingV2
    auditor_review: ReviewBindingV2


def _is_sha256(value: object) -> bool:
    return _v1._is_sha256(value)


def _strict_json_object(raw: bytes, *, name: str) -> dict[str, Any]:
    duplicate: str | None = None

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        nonlocal duplicate
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result and duplicate is None:
                duplicate = key
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=object_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RawSupervisionBuildError(f"{name} is invalid JSON") from error
    if duplicate is not None:
        raise RawSupervisionBuildError(f"{name} has duplicate key {duplicate!r}")
    if type(value) is not dict:
        raise RawSupervisionBuildError(f"{name} is not an object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise RawSupervisionBuildError(f"{name} is not canonical JSON")
    return value


def _canonical_relative_path(value: object) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise RawSupervisionBuildError("authorization source path is noncanonical")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RawSupervisionBuildError("authorization source path is noncanonical")
    return value


def _source_binding(entry: object) -> SourceBindingV2:
    if type(entry) is not dict or set(entry) != SOURCE_ENTRY_FIELDS:
        raise RawSupervisionBuildError("authorization source entry is malformed")
    role = entry["role"]
    if type(role) is not str or not role:
        raise RawSupervisionBuildError("authorization source entry is malformed")
    path = _canonical_relative_path(entry["path"])
    digest = entry["sha256"]
    if not _is_sha256(digest):
        raise RawSupervisionBuildError("authorization source entry is malformed")
    return SourceBindingV2(role=role, path=path, sha256=digest)


def _candidate_bindings(
    value: object,
    *,
    roles: Sequence[str],
    source_by_role: Mapping[str, SourceBindingV2],
) -> tuple[SourceBindingV2, ...]:
    if type(value) is not list or len(value) != len(roles):
        raise RawSupervisionBuildError("authorization review candidate changed")
    candidate = tuple(_source_binding(entry) for entry in value)
    if tuple(item.role for item in candidate) != tuple(roles):
        raise RawSupervisionBuildError("authorization review candidate roles changed")
    if candidate != tuple(source_by_role[role] for role in roles):
        raise RawSupervisionBuildError("authorization review candidate binding changed")
    return candidate


def _review_binding(
    value: object,
    *,
    kind: str,
    review_role: str,
    review_schema: str,
    implementation_author: str,
    candidate_roles: Sequence[str],
    source_by_role: Mapping[str, SourceBindingV2],
) -> ReviewBindingV2:
    if type(value) is not dict or set(value) != REVIEW_BINDING_FIELDS:
        raise RawSupervisionBuildError(f"authorization {kind} review binding is malformed")
    if (
        value["schema"] != REVIEW_BINDING_SCHEMA
        or value["review_schema"] != review_schema
        or value["verdict"] != "PASS"
    ):
        raise PermissionError(f"authorization {kind} review is not a bound PASS")
    reviewer = value["reviewer"]
    author = value["implementation_author"]
    if (
        type(reviewer) is not str
        or not reviewer
        or type(author) is not str
        or not author
        or reviewer == author
    ):
        raise PermissionError(f"authorization {kind} review lacks a distinct reviewer")
    if author != implementation_author:
        raise PermissionError(
            f"authorization {kind} implementation author changed"
        )
    review_source = source_by_role[review_role]
    path = _canonical_relative_path(value["path"])
    if (
        path != review_source.path
        or value["file_sha256"] != review_source.sha256
        or not _is_sha256(value["content_sha256"])
    ):
        raise RawSupervisionBuildError(f"authorization {kind} review file binding changed")
    candidate = _candidate_bindings(
        value["candidate"], roles=candidate_roles, source_by_role=source_by_role
    )
    return ReviewBindingV2(
        kind=kind,
        review_schema=review_schema,
        verdict="PASS",
        reviewer=reviewer,
        implementation_author=author,
        path=path,
        file_sha256=review_source.sha256,
        content_sha256=value["content_sha256"],
        candidate=candidate,
    )


def _validate_authorization_phase_one(
    payload: Mapping[str, Any],
    *,
    authorization_file_sha256: str,
) -> PhaseOneAuthorizationV2:
    """Validate the complete authority without opening any target file."""

    if type(payload) is not dict or set(payload) != AUTHORIZATION_FIELDS:
        raise RawSupervisionBuildError("build authorization object fields changed")
    if not _is_sha256(authorization_file_sha256):
        raise PermissionError("build authorization file hash is not frozen")
    if (
        payload["schema"] != AUTHORIZATION_SCHEMA
        or payload["exact_build_authorized_after_independent_reviews"] is not True
    ):
        raise PermissionError("raw-supervision exact build is not authorized")
    core = dict(payload)
    declared = core.pop("content_sha256")
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise RawSupervisionBuildError("build authorization content hash changed")

    expected_paths = AUTHORIZED_ROLE_PATHS
    if (
        len(expected_paths) != 9
        or len({role for role, _path in expected_paths}) != 9
        or len({path for _role, path in expected_paths}) != 9
    ):
        raise AssertionError("production authorization role policy is not nine unique rows")
    source_map = payload["source_map"]
    if type(source_map) is not list:
        raise RawSupervisionBuildError("build authorization source map is absent")
    sources = tuple(_source_binding(entry) for entry in source_map)
    observed_roles = tuple(item.role for item in sources)
    if len(set(observed_roles)) != len(observed_roles):
        raise RawSupervisionBuildError("authorization source roles are duplicated")
    observed_paths = tuple(item.path for item in sources)
    if len(set(observed_paths)) != len(observed_paths):
        raise RawSupervisionBuildError("authorization source paths are duplicated")
    if observed_roles != tuple(role for role, _path in expected_paths):
        raise RawSupervisionBuildError("authorization source roles changed")
    if observed_paths != tuple(path for _role, path in expected_paths):
        raise RawSupervisionBuildError("authorization role-to-path mapping changed")
    source_by_role = {item.role: item for item in sources}

    builder_review = _review_binding(
        payload["builder_review"],
        kind="builder",
        review_role="builder_review",
        review_schema=BUILDER_REVIEW_SCHEMA,
        implementation_author=BUILDER_IMPLEMENTATION_AUTHOR,
        candidate_roles=BUILDER_CANDIDATE_ROLES,
        source_by_role=source_by_role,
    )
    auditor_review = _review_binding(
        payload["auditor_review"],
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

    normalized = json.loads(canonical_json_bytes(payload))
    return PhaseOneAuthorizationV2(
        authorization_file_sha256=authorization_file_sha256,
        canonical_payload=canonical_json_bytes(normalized),
        sources=sources,
        builder_review=builder_review,
        auditor_review=auditor_review,
    )


def _expected_review_authority(kind: str) -> dict[str, bool]:
    approval = f"{kind}_source_approved"
    return {
        approval: True,
        **{field: False for field in REVIEW_AUTHORITY_FALSE_FIELDS},
    }


def _review_candidate_value(
    candidate: Sequence[SourceBindingV2],
) -> list[dict[str, str]]:
    return [
        {"role": item.role, "path": item.path, "sha256": item.sha256}
        for item in candidate
    ]


def _validate_review_record(raw: bytes, binding: ReviewBindingV2) -> None:
    review = _strict_json_object(raw, name=f"{binding.kind} independent review")
    if set(review) != REVIEW_RECORD_FIELDS:
        raise RawSupervisionBuildError(
            f"{binding.kind} independent review fields changed"
        )
    core = dict(review)
    declared = core.pop("content_sha256")
    if (
        not _is_sha256(declared)
        or canonical_json_sha256(core) != declared
        or declared != binding.content_sha256
    ):
        raise RawSupervisionBuildError(
            f"{binding.kind} independent review content hash changed"
        )
    if (
        review["schema"] != binding.review_schema
        or review["verdict"] != "PASS"
        or review["reviewer"] != binding.reviewer
        or review["implementation_author"] != binding.implementation_author
        or review["candidate"] != _review_candidate_value(binding.candidate)
        or review["authority"] != _expected_review_authority(binding.kind)
    ):
        raise PermissionError(
            f"{binding.kind} independent review PASS binding changed"
        )


def _validate_authorization_phase_two(
    phase_one: PhaseOneAuthorizationV2,
    *,
    repository_root: Path = ROOT,
    reader: Callable[..., bytes] | None = None,
    rehash_frozen_parents: bool = True,
) -> dict[str, Any]:
    """Open fixed reviewed targets only after a complete phase-one result."""

    if type(phase_one) is not PhaseOneAuthorizationV2:
        raise TypeError("phase two requires a completed V2 phase-one result")
    embedded = _strict_json_object(
        phase_one.canonical_payload + b"\n",
        name="phase-one authorization capsule",
    )
    revalidated = _validate_authorization_phase_one(
        embedded,
        authorization_file_sha256=phase_one.authorization_file_sha256,
    )
    if revalidated != phase_one:
        raise PermissionError("phase-one authorization capsule was fabricated")
    read = _read_bound_regular_file if reader is None else reader
    payload_by_role: dict[str, bytes] = {}
    for source in phase_one.sources:
        payload_by_role[source.role] = read(
            repository_root=repository_root,
            path=(repository_root / source.path).absolute(),
            expected_sha256=source.sha256,
        )
    _validate_review_record(
        payload_by_role["builder_review"], phase_one.builder_review
    )
    _validate_review_record(
        payload_by_role["auditor_review"], phase_one.auditor_review
    )
    if rehash_frozen_parents:
        for relative, expected in {
            **FROZEN_PARENT_HASHES,
            **REVIEWED_V4_SOURCES,
        }.items():
            read(
                repository_root=repository_root,
                path=(repository_root / relative).absolute(),
                expected_sha256=expected,
            )
    return json.loads(phase_one.canonical_payload)


def _validate_authorization_payload(
    payload: Mapping[str, Any],
    *,
    authorization_file_sha256: str,
) -> dict[str, Any]:
    phase_one = _validate_authorization_phase_one(
        payload, authorization_file_sha256=authorization_file_sha256
    )
    return _validate_authorization_phase_two(phase_one)


def _require_exact_authority(authorization_sha256: str) -> dict[str, Any]:
    if not _is_sha256(authorization_sha256):
        raise PermissionError("build authorization file hash is not frozen")
    if not AUTHORIZATION_PATH.is_file():
        raise PermissionError("reviewed raw-supervision build authorization is absent")
    raw = _read_bound_regular_file(
        repository_root=ROOT,
        path=AUTHORIZATION_PATH,
        expected_sha256=authorization_sha256,
    )
    payload = _strict_json_object(raw, name="build authorization")
    phase_one = _validate_authorization_phase_one(
        payload, authorization_file_sha256=authorization_sha256
    )
    return _validate_authorization_phase_two(phase_one)


def _call_v1_load_exact_scene_job(
    source_record: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    authorization_sha256: str,
) -> dict[str, Any]:
    authority = _require_exact_authority(authorization_sha256)
    original = _v1._require_exact_authority

    def accepted(candidate_sha256: str) -> dict[str, Any]:
        if candidate_sha256 != authorization_sha256:
            raise PermissionError("worker authorization hash changed")
        return json.loads(canonical_json_bytes(authority))

    _v1._require_exact_authority = accepted
    try:
        return _v1._load_exact_scene_job(
            source_record, endpoints, contexts, authorization_sha256
        )
    finally:
        _v1._require_exact_authority = original


def _call_v1_revalidate_exact_scene_sources(
    source_record: Mapping[str, Any], authorization_sha256: str
) -> tuple[str, ...]:
    authority = _require_exact_authority(authorization_sha256)
    original = _v1._require_exact_authority

    def accepted(candidate_sha256: str) -> dict[str, Any]:
        if candidate_sha256 != authorization_sha256:
            raise PermissionError("worker authorization hash changed")
        return json.loads(canonical_json_bytes(authority))

    _v1._require_exact_authority = accepted
    try:
        return _v1._revalidate_exact_scene_sources(
            source_record, authorization_sha256
        )
    finally:
        _v1._require_exact_authority = original


def _call_v1_load_parent_contracts(
    authorization_sha256: str,
) -> tuple[dict[str, Any], ...]:
    authority = _require_exact_authority(authorization_sha256)
    original = _v1._require_exact_authority

    def accepted(candidate_sha256: str) -> dict[str, Any]:
        if candidate_sha256 != authorization_sha256:
            raise PermissionError("parent authorization hash changed")
        return json.loads(canonical_json_bytes(authority))

    _v1._require_exact_authority = accepted
    try:
        return _v1._load_parent_contracts(authorization_sha256)
    finally:
        _v1._require_exact_authority = original


def _initialize_worker() -> None:
    _v1._worker_environment()


def _load_exact_scene_job_v2(
    source_record: Mapping[str, Any],
    endpoints: Sequence[Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    authorization_sha256: str,
) -> dict[str, Any]:
    return _call_v1_load_exact_scene_job(
        source_record, endpoints, contexts, authorization_sha256
    )


def _revalidate_exact_scene_sources_v2(
    source_record: Mapping[str, Any], authorization_sha256: str
) -> tuple[str, ...]:
    return _call_v1_revalidate_exact_scene_sources(
        source_record, authorization_sha256
    )


def _run_authorized_scene_pool(
    function: Callable[..., Any],
    argument_rows: Sequence[tuple[Any, ...]],
    *,
    workers: int,
    authorization_sha256: str,
) -> list[Any]:
    previous = {
        name: os.environ.get(name)
        for name in (*THREAD_ENVIRONMENT, *ACCELERATOR_ENVIRONMENT)
    }
    try:
        _v1._worker_environment()
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=int(workers),
            mp_context=context,
            initializer=_initialize_worker,
        ) as executor:
            futures = [
                executor.submit(function, *arguments) for arguments in argument_rows
            ]
            return [future.result() for future in futures]
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_parent_contracts_v2(
    authorization_sha256: str,
) -> tuple[dict[str, Any], ...]:
    return _call_v1_load_parent_contracts(authorization_sha256)


def execute_exact_build_v2(
    *, authorization_sha256: str, workers: int
) -> dict[str, Any]:
    """Run the unchanged V1 exact construction after the V2 two-phase gate."""

    authority_verified = False
    try:
        if isinstance(workers, bool) or not 1 <= int(workers) <= MAX_WORKERS:
            raise ValueError(f"workers must lie in [1,{MAX_WORKERS}]")
        authority = _require_exact_authority(authorization_sha256)
        authority_verified = True
        _v1._ensure_exact_output_container()
        if FAILURE_RECEIPT.exists():
            raise PermissionError("a prior exact build failure is terminal")
        if CANONICAL_OUTPUT.exists():
            raise FileExistsError("the immutable exact raw-supervision dataset exists")

        plan = _v1.plan_v5.load_frozen_development_metadata(ROOT)
        inventory = _v1.plan_v5.load_frozen_development_source_inventory(ROOT, plan)
        _v1._validate_exact_plan_result(plan, inventory)
        contexts = _v1._pair_endpoint_contexts(plan)
        endpoints_by_scene: dict[str, list[Mapping[str, Any]]] = {}
        for endpoint in plan.endpoints:
            scene_id = str(endpoint["identity"]["scene_id"])
            endpoints_by_scene.setdefault(scene_id, []).append(endpoint)
        source_by_scene = {
            str(record["scene_id"]): record for record in inventory.records
        }
        if set(source_by_scene) != set(endpoints_by_scene):
            raise RawSupervisionBuildError(
                "source inventory and endpoint scenes differ"
            )

        parent_contract_receipts = list(_load_parent_contracts_v2(authorization_sha256))
        load_arguments = [
            (
                source_by_scene[scene_id],
                tuple(endpoints_by_scene[scene_id]),
                {
                    str(endpoint["identity_sha256"]): contexts[
                        str(endpoint["identity_sha256"])
                    ]
                    for endpoint in endpoints_by_scene[scene_id]
                },
                authorization_sha256,
            )
            for scene_id in sorted(source_by_scene)
        ]
        loaded = _run_authorized_scene_pool(
            _load_exact_scene_job_v2,
            load_arguments,
            workers=int(workers),
            authorization_sha256=authorization_sha256,
        )
        jobs = tuple(item["job"] for item in loaded)
        if len(jobs) != 88 or sum(len(job.endpoints) for job in jobs) != 9460:
            raise RawSupervisionBuildError("exact scene jobs changed population")
        source_receipts = parent_contract_receipts + [
            receipt for item in loaded for receipt in item["source_receipts"]
        ]
        if len(source_receipts) != 354:
            raise RawSupervisionBuildError("exact source payload inventory changed")
        source_receipts.sort(
            key=lambda item: (
                str(item["path"]),
                str(item["purpose"]),
                str(item.get("scene_id", "")),
            )
        )
        frames_scanned = sum(
            int(item["source_frames_jsonl_records_scanned"]) for item in loaded
        )
        access_ledger = _v1._exact_access_ledger(
            plan=plan,
            inventory=inventory,
            frames_scanned=frames_scanned,
        )
        input_provenance = {
            "authorization_file_sha256": authorization_sha256,
            "authorization_content_sha256": authority["content_sha256"],
            "authorization_source_map_sha256": canonical_json_sha256(
                authority["source_map"]
            ),
            "frozen_parent_file_sha256": dict(FROZEN_PARENT_HASHES),
            "reviewed_v4_source_sha256": dict(REVIEWED_V4_SOURCES),
            "metadata_plan_content_sha256": plan.value["content_sha256"],
            "metadata_ordered_pair_sha256": plan.value["ordered_pair_sha256"],
            "metadata_ordered_endpoint_sha256": plan.value[
                "ordered_endpoint_sha256"
            ],
            "source_inventory_sha256": dict(inventory.hashes),
            "source_payload_inventory": source_receipts,
            "source_payload_inventory_sha256": canonical_json_sha256(
                source_receipts
            ),
            "geometry_contract_file_sha256": _v1.GEOMETRY_CONTRACT_FILE_SHA256,
            "geometry_contract_content_sha256": (
                _v1.GEOMETRY_CONTRACT_CONTENT_SHA256
            ),
            "render_audit_file_sha256": _v1.RENDER_AUDIT_FILE_SHA256,
            "render_audit_content_sha256": _v1.RENDER_AUDIT_CONTENT_SHA256,
        }

        def revalidate_before_publication() -> None:
            second_plan = _v1.plan_v5.load_frozen_development_metadata(ROOT)
            second_inventory = _v1.plan_v5.load_frozen_development_source_inventory(
                ROOT, second_plan
            )
            _v1._validate_exact_plan_result(second_plan, second_inventory)
            if (
                canonical_json_bytes(second_plan.value)
                != canonical_json_bytes(plan.value)
                or canonical_json_bytes(second_plan.pairs)
                != canonical_json_bytes(plan.pairs)
                or canonical_json_bytes(second_plan.endpoints)
                != canonical_json_bytes(plan.endpoints)
                or canonical_json_bytes(second_inventory.records)
                != canonical_json_bytes(inventory.records)
            ):
                raise RawSupervisionBuildError(
                    "parent metadata changed before publication"
                )
            _load_parent_contracts_v2(authorization_sha256)
            revalidated = _run_authorized_scene_pool(
                _revalidate_exact_scene_sources_v2,
                [
                    (source_by_scene[scene_id], authorization_sha256)
                    for scene_id in sorted(source_by_scene)
                ],
                workers=int(workers),
                authorization_sha256=authorization_sha256,
            )
            if len(revalidated) != 88 or sum(map(len, revalidated)) != 352:
                raise RawSupervisionBuildError(
                    "source revalidation inventory changed"
                )

        return build_prepared_dataset_v1(
            jobs,
            plan.pairs,
            output_directory=CANONICAL_OUTPUT,
            workers=int(workers),
            input_provenance=input_provenance,
            access_ledger=access_ledger,
            exact_role_family_count=True,
            prepublication_validator=revalidate_before_publication,
        )
    except BaseException as error:
        if authority_verified and not FAILURE_RECEIPT.exists():
            _v1._write_failure_receipt(
                authorization_sha256=authorization_sha256,
                error=error,
            )
        raise


__all__ = [
    "ACCELERATOR_ENVIRONMENT",
    "ARRAY_LAYOUT",
    "AUTHORIZATION_PATH",
    "AUTHORIZED_ROLE_PATHS",
    "CANONICAL_OUTPUT",
    "DATASET_SCHEMA",
    "FAILURE_RECEIPT",
    "MAX_WORKERS",
    "PreparedEndpointV1",
    "PreparedSceneJobV1",
    "RawSupervisionBuildError",
    "THREAD_ENVIRONMENT",
    "build_prepared_dataset_v1",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "execute_exact_build_v2",
]
