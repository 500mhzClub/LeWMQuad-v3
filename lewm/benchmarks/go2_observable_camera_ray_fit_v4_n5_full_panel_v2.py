"""Additive V2 execution authority for the V4 N5 full-panel experiment.

The scientific experiment and its V1 implementation remain frozen.  V2 closes
only the independently reviewed authority, one-attempt, recovery, and directory
durability gaps.  Importing this module is stdlib-only and opens no experiment
data, RGB, checkpoint, model, accelerator, or output.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import threading
from typing import Any, Mapping, Sequence

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v1 as retained_v1,
)


ROOT = Path(__file__).resolve().parents[2]
IMPLEMENTATION_AUTHOR = "/root/v4_execution_successor_review"

POLICY_RELATIVE_PATH = (
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py"
)
LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py"
)
TRAINER_RELATIVE_PATH = (
    "scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py"
)
VERIFIER_RELATIVE_PATH = (
    "scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py"
)
FINALIZER_RELATIVE_PATH = (
    "scripts/finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v2.py"
)
SUCCESSOR_SOURCE_PATHS = (
    POLICY_RELATIVE_PATH,
    LAUNCHER_RELATIVE_PATH,
    TRAINER_RELATIVE_PATH,
    VERIFIER_RELATIVE_PATH,
    FINALIZER_RELATIVE_PATH,
)

SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_"
    "independent_review_2026-07-13.json"
)
CANONICAL_SOURCE_REVIEW_PATH = (ROOT / SOURCE_REVIEW_RELATIVE_PATH).resolve()

V1_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_"
    "independent_review_2026-07-13.md"
)
V1_REVIEW_FILE_SHA256 = (
    "11479b03ff9eac24dd5541d38faeda480739c8d17de7b2b658759e306ace2d5e"
)
V1_BLOCK_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_"
    "independent_review_block_2026-07-13.json"
)
V1_BLOCK_FILE_SHA256 = (
    "ccd8d97988d2ce165722703fbfcf813758ee42a5408e02d26bf7db38d8ea506e"
)
V1_BLOCK_CONTENT_SHA256 = (
    "99ded56d11b357ada724b238e750d1845bd0010d72a081f4819948b3e05163e7"
)
V1_EXPLOIT_TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_"
    "independent_review.py"
)
V1_EXPLOIT_TEST_FILE_SHA256 = (
    "387147a8dd6fe1a20184284a05c18df73419ca91c21054eb378e79a8194d5b3b"
)
V1_HANDOFF_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v1_"
    "implementation_handoff_2026-07-13.md"
)
V1_HANDOFF_FILE_SHA256 = (
    "8f4735a3ecd20a8c19bd729fdaf71ceb60a3a884de717423e8f84ef6ef2745f7"
)

RETAINED_V1_SOURCE_BINDINGS = {
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py": (
        "875edc86efbe25d246b24c2ef2467cc7956b1b3bb90e6d8d1e03e4a9c5b11d88"
    ),
    "scripts/launch_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py": (
        "3cb9ff782a15bc97dd3cca2cc25705e006d6af19a7dbef6d27dee893d9b570c8"
    ),
    "scripts/train_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py": (
        "48ac856c080906a8d73d5a9b97d1dcf7fe21f5bc99217cce669c43b9c091acca"
    ),
    "scripts/verify_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py": (
        "00c62cec39e1eb05bf23a96a9153aa8ff350235c2e5c6662f6148934ab9d85b0"
    ),
    "scripts/finalize_go2_observable_camera_ray_fit_v4_n5_full_panel_v1.py": (
        "1d4471381a6c3b29f0b077e44e3126f956281ff105d4e38aa8e0f6ba18675b8b"
    ),
}

# The scientific contract is deliberately identical to frozen V1.
PREREGISTRATION_RELATIVE_PATH = retained_v1.PREREGISTRATION_RELATIVE_PATH
PREREGISTRATION_FILE_SHA256 = retained_v1.PREREGISTRATION_FILE_SHA256
TRIGGER_AMENDMENT_RELATIVE_PATH = retained_v1.TRIGGER_AMENDMENT_RELATIVE_PATH
TRIGGER_AMENDMENT_FILE_SHA256 = retained_v1.TRIGGER_AMENDMENT_FILE_SHA256
TERMINAL_INVALIDATION_RELATIVE_PATH = retained_v1.TERMINAL_INVALIDATION_RELATIVE_PATH
TERMINAL_INVALIDATION_FILE_SHA256 = retained_v1.TERMINAL_INVALIDATION_FILE_SHA256
TERMINAL_INVALIDATION_CONTENT_SHA256 = retained_v1.TERMINAL_INVALIDATION_CONTENT_SHA256
DATASET_MANIFEST_RELATIVE_PATH = retained_v1.DATASET_MANIFEST_RELATIVE_PATH
DATASET_MANIFEST_FILE_SHA256 = retained_v1.DATASET_MANIFEST_FILE_SHA256
DATASET_MANIFEST_CONTENT_SHA256 = retained_v1.DATASET_MANIFEST_CONTENT_SHA256
AUDIT_RECEIPT_RELATIVE_PATH = retained_v1.AUDIT_RECEIPT_RELATIVE_PATH
AUDIT_RECEIPT_FILE_SHA256 = retained_v1.AUDIT_RECEIPT_FILE_SHA256
AUDIT_RECEIPT_CONTENT_SHA256 = retained_v1.AUDIT_RECEIPT_CONTENT_SHA256
TRAINER_AUTHORIZATION_RELATIVE_PATH = retained_v1.TRAINER_AUTHORIZATION_RELATIVE_PATH
TRAINER_AUTHORIZATION_FILE_SHA256 = retained_v1.TRAINER_AUTHORIZATION_FILE_SHA256
TRAINER_AUTHORIZATION_CONTENT_SHA256 = retained_v1.TRAINER_AUTHORIZATION_CONTENT_SHA256
TRAINER_REVIEW_RELATIVE_PATH = retained_v1.TRAINER_REVIEW_RELATIVE_PATH
TRAINER_REVIEW_FILE_SHA256 = retained_v1.TRAINER_REVIEW_FILE_SHA256
TRAINER_REVIEW_CONTENT_SHA256 = retained_v1.TRAINER_REVIEW_CONTENT_SHA256
RGB_RECEIPT_CONTENT_SHA256 = retained_v1.RGB_RECEIPT_CONTENT_SHA256
SUBSET_CONTENT_SHA256 = retained_v1.SUBSET_CONTENT_SHA256
TARGET_PARTITION_CONTENT_SHA256 = retained_v1.TARGET_PARTITION_CONTENT_SHA256
OUTPUT_ROOT_RELATIVE_PATH = retained_v1.OUTPUT_ROOT_RELATIVE_PATH
CANONICAL_OUTPUT_ROOT = retained_v1.CANONICAL_OUTPUT_ROOT
CANONICAL_ATTEMPT_PATH = retained_v1.CANONICAL_ATTEMPT_PATH
CANONICAL_METRIC_RECEIPT_PATH = retained_v1.CANONICAL_METRIC_RECEIPT_PATH
CANONICAL_GATE_PATH = retained_v1.CANONICAL_GATE_PATH
SCHEDULE_ALGORITHM = retained_v1.SCHEDULE_ALGORITHM
EXPECTED_SCHEDULE_SHA256 = retained_v1.EXPECTED_SCHEDULE_SHA256
LOSS_COMPONENTS = retained_v1.LOSS_COMPONENTS
LOSS_WEIGHTS = retained_v1.LOSS_WEIGHTS
LOSS_ABSOLUTE_TOLERANCE = retained_v1.LOSS_ABSOLUTE_TOLERANCE
THREAD_ENVIRONMENT = retained_v1.THREAD_ENVIRONMENT
EXPERIMENT = retained_v1.EXPERIMENT
AUTHORITY_BINDINGS = retained_v1.AUTHORITY_BINDINGS
LICENSES = retained_v1.LICENSES
FROZEN_SOURCE_BINDINGS = retained_v1.FROZEN_SOURCE_BINDINGS

SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_source_review_v1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_reservation_v1"
)
RESULT_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_result_v1"
COMPLETION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_completion_v1"
)
FAILURE_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_failure_v1"
)
METRIC_RECEIPT_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_metric_verification_v1"
)
GATE_SCHEMA = "lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v2_gate_v1"

AUTHORITY_PURPOSE_PATHS = {
    "exact_run": CANONICAL_ATTEMPT_PATH,
    "metric_verification": CANONICAL_METRIC_RECEIPT_PATH,
    "finalization": CANONICAL_GATE_PATH,
}


canonical_json_bytes = retained_v1.canonical_json_bytes
canonical_json_sha256 = retained_v1.canonical_json_sha256
is_sha256 = retained_v1.is_sha256
parse_json = retained_v1.parse_json
read_regular_bytes = retained_v1.read_regular_bytes
read_hashed_bytes = retained_v1.read_hashed_bytes
load_hashed_json = retained_v1.load_hashed_json
artifact_binding = retained_v1.artifact_binding
parse_bound_path = retained_v1.parse_bound_path
validate_evaluation_structure = retained_v1.validate_evaluation_structure


def _hash_file(relative: str, expected: str, *, name: str) -> bytes:
    raw = read_regular_bytes(ROOT / relative, name=name)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise PermissionError(f"{name} changed")
    return raw


def _validate_v1_block_evidence() -> None:
    _hash_file(V1_REVIEW_RELATIVE_PATH, V1_REVIEW_FILE_SHA256, name="V1 BLOCK review")
    raw = _hash_file(V1_BLOCK_RELATIVE_PATH, V1_BLOCK_FILE_SHA256, name="V1 BLOCK JSON")
    value = parse_json(raw, name="V1 BLOCK JSON")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if declared != V1_BLOCK_CONTENT_SHA256 or canonical_json_sha256(core) != declared:
        raise PermissionError("V1 BLOCK JSON content binding changed")
    _hash_file(
        V1_EXPLOIT_TEST_RELATIVE_PATH,
        V1_EXPLOIT_TEST_FILE_SHA256,
        name="V1 independent exploit tests",
    )
    _hash_file(V1_HANDOFF_RELATIVE_PATH, V1_HANDOFF_FILE_SHA256, name="V1 handoff")
    for relative, expected in RETAINED_V1_SOURCE_BINDINGS.items():
        _hash_file(relative, expected, name=f"retained V1 source {relative}")


def preflight_static_authority() -> dict[str, Any]:
    static = retained_v1.preflight_static_authority()
    _validate_v1_block_evidence()
    if EXPERIMENT != retained_v1.EXPERIMENT:
        raise PermissionError("V2 changed the frozen V1 experiment")
    return {
        **static,
        "v1_block_review_file_sha256": V1_REVIEW_FILE_SHA256,
        "v1_block_file_sha256": V1_BLOCK_FILE_SHA256,
        "v1_block_content_sha256": V1_BLOCK_CONTENT_SHA256,
        "v1_exploit_test_file_sha256": V1_EXPLOIT_TEST_FILE_SHA256,
    }


def expected_source_review_core(
    *,
    reviewer: str,
    successor_sources: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "schema": SOURCE_REVIEW_SCHEMA,
        "status": "approved_authority_and_reservation_successor",
        "implementation_author": IMPLEMENTATION_AUTHOR,
        "reviewer": reviewer,
        "review_completed": True,
        "source_closure_approved": True,
        "exact_attempt_authorized": True,
        "successor_sources": dict(successor_sources),
        "retained_v1_sources": RETAINED_V1_SOURCE_BINDINGS,
        "v1_block_evidence": {
            "review": {
                "path": V1_REVIEW_RELATIVE_PATH,
                "file_sha256": V1_REVIEW_FILE_SHA256,
            },
            "block": {
                "path": V1_BLOCK_RELATIVE_PATH,
                "file_sha256": V1_BLOCK_FILE_SHA256,
                "content_sha256": V1_BLOCK_CONTENT_SHA256,
            },
            "exploit_tests": {
                "path": V1_EXPLOIT_TEST_RELATIVE_PATH,
                "file_sha256": V1_EXPLOIT_TEST_FILE_SHA256,
            },
            "handoff": {
                "path": V1_HANDOFF_RELATIVE_PATH,
                "file_sha256": V1_HANDOFF_FILE_SHA256,
            },
        },
        "authority_contract": {
            "exact_live_object_registry": True,
            "immutable_issuance_digest": True,
            "copy_clone_reconstruction_rejected": True,
            "purpose_and_canonical_path_bound": True,
            "atomic_single_use_consumption": True,
            "canonical_source_revalidated_on_every_protected_use": True,
            "test_capability_separate_and_production_ineligible": True,
        },
        "reservation_contract": {
            "unique_private_staging": True,
            "process_death_safe_locking": True,
            "incomplete_complete_foreign_mutated_recovery_reviewed": True,
            "seed_parent_fsync_immediately_after_rename": True,
            "terminal_attempt_and_parent_fsync": True,
        },
        "authority_bindings": AUTHORITY_BINDINGS,
        "experiment": EXPERIMENT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "licenses": LICENSES,
    }


def _preflight_source_review_at(
    path: Path,
    file_sha256: str,
    *,
    required_path: Path,
) -> tuple[dict[str, Any], bytes]:
    resolved = Path(path).resolve(strict=True)
    if resolved != Path(required_path).resolve(strict=True):
        raise PermissionError("N5 full-panel V2 source review path is not bound")
    review, raw = load_hashed_json(
        resolved,
        file_sha256,
        name="N5 full-panel V2 different-agent source review",
    )
    reviewer = review.get("reviewer")
    sources = review.get("successor_sources")
    if (
        not isinstance(reviewer, str)
        or not reviewer.startswith("/root/")
        or reviewer == IMPLEMENTATION_AUTHOR
        or not isinstance(sources, Mapping)
        or set(sources) != set(SUCCESSOR_SOURCE_PATHS)
    ):
        raise PermissionError("N5 full-panel V2 review is not by a different agent")
    checked: dict[str, dict[str, str]] = {}
    for relative in SUCCESSOR_SOURCE_PATHS:
        binding = sources.get(relative)
        if (
            not isinstance(binding, Mapping)
            or binding.get("path") != relative
            or not is_sha256(binding.get("file_sha256"))
        ):
            raise PermissionError("N5 full-panel V2 source binding changed")
        source_raw = read_regular_bytes(ROOT / relative, name=f"V2 source {relative}")
        if hashlib.sha256(source_raw).hexdigest() != binding["file_sha256"]:
            raise PermissionError(f"N5 full-panel V2 source changed: {relative}")
        checked[relative] = dict(binding)
    expected = expected_source_review_core(reviewer=reviewer, successor_sources=checked)
    core = dict(review)
    declared = core.pop("content_sha256", None)
    if core != expected or canonical_json_sha256(core) != declared:
        raise PermissionError("N5 full-panel V2 source review contract changed")
    return review, raw


def preflight_source_review(path: Path, file_sha256: str) -> tuple[dict[str, Any], bytes]:
    return _preflight_source_review_at(
        path,
        file_sha256,
        required_path=CANONICAL_SOURCE_REVIEW_PATH,
    )


class VerifiedAuthorityV2:
    """Issuer-owned, immutable, exact-identity execution authority."""

    __slots__ = (
        "_purpose",
        "_target_path",
        "_review_path",
        "_review_file_sha256",
        "_review_content_sha256",
        "_review_raw",
        "_static_raw",
        "_test_only",
    )

    def __new__(cls, *args: object, **kwargs: object) -> "VerifiedAuthorityV2":
        del args, kwargs
        raise TypeError("verified authority is issued only by a reviewed preflight")

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise TypeError("verified authority is immutable")

    def __delattr__(self, name: str) -> None:
        del name
        raise TypeError("verified authority is immutable")

    def __copy__(self) -> "VerifiedAuthorityV2":
        raise TypeError("verified authority is an exact live noncopyable object")

    def __deepcopy__(self, memo: object) -> "VerifiedAuthorityV2":
        del memo
        raise TypeError("verified authority is an exact live noncopyable object")

    def __reduce_ex__(self, protocol: int) -> object:
        del protocol
        raise TypeError("verified authority cannot be serialized or reconstructed")

    @property
    def purpose(self) -> str:
        return object.__getattribute__(self, "_purpose")

    @property
    def target_path(self) -> Path:
        return Path(object.__getattribute__(self, "_target_path"))

    @property
    def source_review_file_sha256(self) -> str:
        return object.__getattribute__(self, "_review_file_sha256")

    @property
    def source_review_content_sha256(self) -> str:
        return object.__getattribute__(self, "_review_content_sha256")

    @property
    def test_only(self) -> bool:
        return bool(object.__getattribute__(self, "_test_only"))


@dataclass(frozen=True)
class _AuthorityRecord:
    authority: VerifiedAuthorityV2
    issuance_digest: str
    state: str


def _authority_digest(authority: VerifiedAuthorityV2) -> str:
    core = {
        "purpose": object.__getattribute__(authority, "_purpose"),
        "target_path": object.__getattribute__(authority, "_target_path"),
        "review_path": object.__getattribute__(authority, "_review_path"),
        "review_file_sha256": object.__getattribute__(
            authority, "_review_file_sha256"
        ),
        "review_content_sha256": object.__getattribute__(
            authority, "_review_content_sha256"
        ),
        "review_raw_sha256": hashlib.sha256(
            object.__getattribute__(authority, "_review_raw")
        ).hexdigest(),
        "static_raw_sha256": hashlib.sha256(
            object.__getattribute__(authority, "_static_raw")
        ).hexdigest(),
        "test_only": object.__getattribute__(authority, "_test_only"),
    }
    return canonical_json_sha256(core)


class TestAuthorityCapabilityV2:
    """Explicit, path-confined, permanently production-ineligible test issuer."""

    __slots__ = ()

    def __new__(cls, *args: object, **kwargs: object) -> "TestAuthorityCapabilityV2":
        del args, kwargs
        raise TypeError("test capability must be issued explicitly")

    def __copy__(self) -> "TestAuthorityCapabilityV2":
        raise TypeError("test capability is noncopyable")

    def __deepcopy__(self, memo: object) -> "TestAuthorityCapabilityV2":
        del memo
        raise TypeError("test capability is noncopyable")

    @property
    def production_eligible(self) -> bool:
        return False

    @property
    def root(self) -> Path:
        return _test_capability_root(self)

    def issue(
        self,
        review_path: Path,
        review_file_sha256: str,
        *,
        target_path: Path,
    ) -> VerifiedAuthorityV2:
        return _test_issue_authority(
            self,
            review_path=review_path,
            review_file_sha256=review_file_sha256,
            target_path=target_path,
        )

    def validate(
        self,
        authority: object,
        *,
        target_path: Path,
        allowed_states: Sequence[str],
    ) -> VerifiedAuthorityV2:
        return _test_validate_authority(
            self,
            authority,
            target_path=target_path,
            allowed_states=allowed_states,
        )

    def transition(
        self,
        authority: object,
        *,
        target_path: Path,
        from_states: Sequence[str],
        to_state: str,
    ) -> VerifiedAuthorityV2:
        return _test_transition_authority(
            self,
            authority,
            target_path=target_path,
            from_states=from_states,
            to_state=to_state,
        )


def _build_authority_api() -> tuple[object, ...]:
    """Keep lifecycle state out of authority/capability object graphs."""

    production_records: dict[int, _AuthorityRecord] = {}
    test_scopes: dict[
        int,
        tuple[
            TestAuthorityCapabilityV2,
            Path,
            dict[int, _AuthorityRecord],
        ],
    ] = {}
    lock = threading.RLock()

    def make_authority(
        *,
        review_path: Path,
        review_file_sha256: str,
        purpose: str,
        target_path: Path,
        test_only: bool,
    ) -> tuple[VerifiedAuthorityV2, _AuthorityRecord]:
        resolved_review = Path(review_path).resolve(strict=True)
        resolved_target = Path(target_path).resolve()
        if test_only:
            review, review_raw = _preflight_source_review_at(
                resolved_review,
                review_file_sha256,
                required_path=resolved_review,
            )
        else:
            review, review_raw = preflight_source_review(
                resolved_review,
                review_file_sha256,
            )
        static = preflight_static_authority()
        authority = object.__new__(VerifiedAuthorityV2)
        for name, value in (
            ("_purpose", purpose),
            ("_target_path", str(resolved_target)),
            ("_review_path", str(resolved_review)),
            ("_review_file_sha256", review_file_sha256),
            ("_review_content_sha256", str(review["content_sha256"])),
            ("_review_raw", bytes(review_raw)),
            ("_static_raw", canonical_json_bytes(static)),
            ("_test_only", test_only),
        ):
            object.__setattr__(authority, name, value)
        return authority, _AuthorityRecord(
            authority=authority,
            issuance_digest=_authority_digest(authority),
            state="issued",
        )

    def validate_record(
        authority: object,
        records: dict[int, _AuthorityRecord],
        *,
        purpose: str | None,
        target_path: Path | None,
        allowed_states: Sequence[str],
        test_only: bool,
    ) -> _AuthorityRecord:
        if type(authority) is not VerifiedAuthorityV2:
            raise PermissionError("protected work lacks an exact verified authority")
        record = records.get(id(authority))
        if record is None or record.authority is not authority:
            raise PermissionError(
                "verified authority is forged, cloned, reconstructed, or cross-issuer"
            )
        if record.issuance_digest != _authority_digest(authority):
            raise PermissionError("verified authority was mutated after issuance")
        if record.state not in set(allowed_states):
            raise PermissionError(
                "verified authority was consumed or replayed outside its one use"
            )
        if authority.test_only is not test_only:
            raise PermissionError("verified authority test/production role was mutated")
        if purpose is not None and authority.purpose != purpose:
            raise PermissionError("verified authority purpose binding changed")
        if target_path is not None and authority.target_path != Path(target_path).resolve():
            raise PermissionError("verified authority path binding changed")
        review_path = Path(object.__getattribute__(authority, "_review_path"))
        if test_only:
            _preflight_source_review_at(
                review_path,
                authority.source_review_file_sha256,
                required_path=review_path,
            )
        else:
            preflight_source_review(review_path, authority.source_review_file_sha256)
        preflight_static_authority()
        return record

    def verify(
        source_review_path: Path,
        source_review_file_sha256: str,
        *,
        purpose: str = "exact_run",
        require_unclaimed_output: bool = True,
    ) -> VerifiedAuthorityV2:
        if purpose not in AUTHORITY_PURPOSE_PATHS:
            raise PermissionError("N5 full-panel V2 authority purpose is not authorized")
        if (
            purpose == "exact_run"
            and require_unclaimed_output
            and CANONICAL_ATTEMPT_PATH.exists()
        ):
            raise FileExistsError("the sole N5 full-panel attempt is already claimed")
        authority, record = make_authority(
            review_path=source_review_path,
            review_file_sha256=source_review_file_sha256,
            purpose=purpose,
            target_path=AUTHORITY_PURPOSE_PATHS[purpose],
            test_only=False,
        )
        with lock:
            production_records[id(authority)] = record
        return authority

    def require(
        value: object,
        *,
        purpose: str | None = None,
        target_path: Path | None = None,
        allowed_states: Sequence[str] = ("issued", "active", "claiming", "claimed"),
    ) -> VerifiedAuthorityV2:
        with lock:
            return validate_record(
                value,
                production_records,
                purpose=purpose,
                target_path=target_path,
                allowed_states=allowed_states,
                test_only=False,
            ).authority

    def transition(
        value: object,
        *,
        purpose: str,
        target_path: Path,
        from_states: Sequence[str],
        to_state: str,
    ) -> VerifiedAuthorityV2:
        with lock:
            record = validate_record(
                value,
                production_records,
                purpose=purpose,
                target_path=target_path,
                allowed_states=from_states,
                test_only=False,
            )
            production_records[id(record.authority)] = _AuthorityRecord(
                authority=record.authority,
                issuance_digest=record.issuance_digest,
                state=to_state,
            )
            return record.authority

    def create_test(test_root: Path) -> TestAuthorityCapabilityV2:
        root = Path(test_root).resolve(strict=True)
        if root == ROOT or ROOT.is_relative_to(root):
            raise PermissionError("test authority root must not contain the repository")
        capability = object.__new__(TestAuthorityCapabilityV2)
        with lock:
            test_scopes[id(capability)] = (capability, root, {})
        return capability

    def test_scope(
        capability: object,
    ) -> tuple[TestAuthorityCapabilityV2, Path, dict[int, _AuthorityRecord]]:
        if type(capability) is not TestAuthorityCapabilityV2:
            raise PermissionError("test authority capability is forged")
        scope = test_scopes.get(id(capability))
        if scope is None or scope[0] is not capability:
            raise PermissionError("test authority capability is cloned or reconstructed")
        return scope

    def test_root(capability: object) -> Path:
        with lock:
            return test_scope(capability)[1]

    def test_issue(
        capability: object,
        review_path: Path,
        review_file_sha256: str,
        *,
        target_path: Path,
    ) -> VerifiedAuthorityV2:
        with lock:
            _cap, root, records = test_scope(capability)
            resolved_review = Path(review_path).resolve(strict=True)
            resolved_target = Path(target_path).resolve()
            if not resolved_review.is_relative_to(root) or not resolved_target.is_relative_to(
                root
            ):
                raise PermissionError("test authority escaped its private test root")
            if resolved_target == CANONICAL_ATTEMPT_PATH.resolve():
                raise PermissionError("test authority cannot target canonical output")
            authority, record = make_authority(
                review_path=resolved_review,
                review_file_sha256=review_file_sha256,
                purpose="test_exact_run",
                target_path=resolved_target,
                test_only=True,
            )
            records[id(authority)] = record
            return authority

    def test_validate(
        capability: object,
        authority: object,
        *,
        target_path: Path,
        allowed_states: Sequence[str],
    ) -> VerifiedAuthorityV2:
        with lock:
            _cap, _root, records = test_scope(capability)
            return validate_record(
                authority,
                records,
                purpose="test_exact_run",
                target_path=target_path,
                allowed_states=allowed_states,
                test_only=True,
            ).authority

    def test_transition(
        capability: object,
        authority: object,
        *,
        target_path: Path,
        from_states: Sequence[str],
        to_state: str,
    ) -> VerifiedAuthorityV2:
        with lock:
            _cap, _root, records = test_scope(capability)
            record = validate_record(
                authority,
                records,
                purpose="test_exact_run",
                target_path=target_path,
                allowed_states=from_states,
                test_only=True,
            )
            records[id(record.authority)] = _AuthorityRecord(
                authority=record.authority,
                issuance_digest=record.issuance_digest,
                state=to_state,
            )
            return record.authority

    return (
        verify,
        require,
        transition,
        create_test,
        test_root,
        test_issue,
        test_validate,
        test_transition,
    )


(
    verify_authority,
    require_verified_authority,
    transition_authority,
    create_test_authority_capability,
    _test_capability_root,
    _test_issue_authority,
    _test_validate_authority,
    _test_transition_authority,
) = _build_authority_api()
del _build_authority_api


def source_review_binding(
    authority: VerifiedAuthorityV2,
    *,
    test_capability: TestAuthorityCapabilityV2 | None = None,
) -> dict[str, str]:
    if test_capability is None:
        checked = require_verified_authority(authority)
        path = SOURCE_REVIEW_RELATIVE_PATH
    else:
        checked = test_capability.validate(
            authority,
            target_path=authority.target_path,
            allowed_states=("issued", "active", "claiming", "claimed"),
        )
        path = str(object.__getattribute__(checked, "_review_path"))
    return {
        "path": path,
        "file_sha256": checked.source_review_file_sha256,
        "content_sha256": checked.source_review_content_sha256,
    }


def _shadow_review(review: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(review)
    value["path"] = retained_v1.SOURCE_REVIEW_RELATIVE_PATH
    return value


def validate_reservation_structure(
    reservation: Mapping[str, Any],
    *,
    expected_source_review: Mapping[str, str],
) -> dict[str, Any]:
    core = dict(reservation)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError("N5 full-panel V2 reservation content SHA-256 changed")
    recovery = reservation.get("preclaim_recovery")
    if not isinstance(recovery, list) or any(
        not isinstance(item, Mapping) for item in recovery
    ):
        raise ValueError("N5 full-panel V2 preclaim recovery ledger is malformed")
    shadow = copy.deepcopy(dict(reservation))
    shadow.pop("preclaim_recovery", None)
    shadow["schema"] = retained_v1.RESERVATION_SCHEMA
    shadow["source_review"] = _shadow_review(shadow["source_review"])
    shadow_core = dict(shadow)
    shadow_core.pop("content_sha256")
    shadow["content_sha256"] = canonical_json_sha256(shadow_core)
    retained_v1.validate_reservation_structure(
        shadow,
        expected_source_review=_shadow_review(expected_source_review),
    )
    if reservation.get("schema") != RESERVATION_SCHEMA:
        raise ValueError("N5 full-panel V2 reservation schema changed")
    return dict(reservation)


def validate_result_structure(
    result: Mapping[str, Any],
    *,
    expected_source_review: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    core = dict(result)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise ValueError("N5 full-panel V2 result content SHA-256 changed")
    shadow = copy.deepcopy(dict(result))
    shadow["schema"] = retained_v1.RESULT_SCHEMA
    shadow["source_review"] = _shadow_review(shadow["source_review"])
    shadow_core = dict(shadow)
    shadow_core.pop("content_sha256")
    shadow["content_sha256"] = canonical_json_sha256(shadow_core)
    shadow_expected = (
        None if expected_source_review is None else _shadow_review(expected_source_review)
    )
    retained_v1.validate_result_structure(
        shadow,
        expected_source_review=shadow_expected,
    )
    if result.get("schema") != RESULT_SCHEMA:
        raise ValueError("N5 full-panel V2 result schema changed")
    return dict(result)


def write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path)
    if path not in {CANONICAL_METRIC_RECEIPT_PATH, CANONICAL_GATE_PATH}:
        raise PermissionError("N5 full-panel V2 output path is not canonical")
    if CANONICAL_OUTPUT_ROOT.is_symlink() or not CANONICAL_OUTPUT_ROOT.is_dir():
        raise PermissionError("N5 full-panel V2 output root is not a real directory")
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(exist_ok=True)
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise PermissionError("N5 full-panel V2 output parent is not a real directory")
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
        path.unlink(missing_ok=True)
        raise
    parent_descriptor = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    return artifact_binding(
        str(path.relative_to(CANONICAL_OUTPUT_ROOT)),
        payload,
        content_sha256=str(value["content_sha256"]),
    )


__all__ = [
    "AUTHORITY_BINDINGS",
    "AUTHORITY_PURPOSE_PATHS",
    "AUDIT_RECEIPT_CONTENT_SHA256",
    "AUDIT_RECEIPT_FILE_SHA256",
    "AUDIT_RECEIPT_RELATIVE_PATH",
    "CANONICAL_ATTEMPT_PATH",
    "CANONICAL_GATE_PATH",
    "CANONICAL_METRIC_RECEIPT_PATH",
    "CANONICAL_OUTPUT_ROOT",
    "CANONICAL_SOURCE_REVIEW_PATH",
    "COMPLETION_SCHEMA",
    "DATASET_MANIFEST_CONTENT_SHA256",
    "DATASET_MANIFEST_FILE_SHA256",
    "DATASET_MANIFEST_RELATIVE_PATH",
    "EXPERIMENT",
    "EXPECTED_SCHEDULE_SHA256",
    "FAILURE_SCHEMA",
    "FINALIZER_RELATIVE_PATH",
    "FROZEN_SOURCE_BINDINGS",
    "GATE_SCHEMA",
    "IMPLEMENTATION_AUTHOR",
    "LAUNCHER_RELATIVE_PATH",
    "LICENSES",
    "LOSS_COMPONENTS",
    "LOSS_WEIGHTS",
    "METRIC_RECEIPT_SCHEMA",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "POLICY_RELATIVE_PATH",
    "RESERVATION_SCHEMA",
    "RESULT_SCHEMA",
    "RETAINED_V1_SOURCE_BINDINGS",
    "RGB_RECEIPT_CONTENT_SHA256",
    "SCHEDULE_ALGORITHM",
    "SOURCE_REVIEW_RELATIVE_PATH",
    "SOURCE_REVIEW_SCHEMA",
    "SUBSET_CONTENT_SHA256",
    "SUCCESSOR_SOURCE_PATHS",
    "TARGET_PARTITION_CONTENT_SHA256",
    "TERMINAL_INVALIDATION_CONTENT_SHA256",
    "TERMINAL_INVALIDATION_FILE_SHA256",
    "THREAD_ENVIRONMENT",
    "TRAINER_AUTHORIZATION_CONTENT_SHA256",
    "TRAINER_AUTHORIZATION_FILE_SHA256",
    "TRAINER_AUTHORIZATION_RELATIVE_PATH",
    "TRAINER_RELATIVE_PATH",
    "TRAINER_REVIEW_CONTENT_SHA256",
    "TRAINER_REVIEW_FILE_SHA256",
    "TRAINER_REVIEW_RELATIVE_PATH",
    "TestAuthorityCapabilityV2",
    "VERIFIER_RELATIVE_PATH",
    "V1_BLOCK_CONTENT_SHA256",
    "V1_BLOCK_FILE_SHA256",
    "V1_EXPLOIT_TEST_FILE_SHA256",
    "V1_REVIEW_FILE_SHA256",
    "VerifiedAuthorityV2",
    "artifact_binding",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "create_test_authority_capability",
    "expected_source_review_core",
    "is_sha256",
    "load_hashed_json",
    "parse_bound_path",
    "parse_json",
    "preflight_source_review",
    "preflight_static_authority",
    "read_hashed_bytes",
    "read_regular_bytes",
    "require_verified_authority",
    "source_review_binding",
    "transition_authority",
    "validate_evaluation_structure",
    "validate_reservation_structure",
    "validate_result_structure",
    "verify_authority",
    "write_exclusive",
]
