"""Custody for the interrupted fixed-state source-reissue validation.

The clean ca09 successor reached the expensive pre-issuance validation for
the seven fixed scorer-fit state shards and was interrupted with SIGINT before
the wrapper issuance loop.  This module archives only the five exact ca09
successor authorities, retains the exact pre-identity validation artifact,
and records the absence of every fixed wrapper and the small-prefix reissue
receipt.  It grants no scientific, retry, resume, or wrapper authority.

This module intentionally does not import the performance-interruption module:
the successor performance lineage imports this transition module.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lewm.oracle import go2_scorer_state_selector_amendment_v2 as SELECTOR


ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "go2_scorer_fixed_reissue_validation_preoutcome_interruption_v1"
STATUS = "INTERRUPTED_PRE_ISSUANCE_VALIDATION_NO_WRITE"
SELF_DIGEST_KEY = (
    "preoutcome_fixed_reissue_validation_interruption_receipt_digest"
)

CORPUS_ROOT_RELATIVE = Path(".generated/go2_branch_corpus_v1_2")
SCORER_ROOT_RELATIVE = Path(".generated/go2_utility_scorer_v1_2")
SCORER_FIT_RELATIVE = CORPUS_ROOT_RELATIVE / "scorer_fit"
ARCHIVE_ROOT_RELATIVE = (
    SCORER_FIT_RELATIVE /
    "superseded_preoutcome_fixed_reissue_validation_v1"
)
SCORER_ARCHIVE_ROOT_RELATIVE = (
    SCORER_ROOT_RELATIVE /
    "superseded_preoutcome_fixed_reissue_validation_v1"
)
RECEIPT_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE /
    "preoutcome_fixed_reissue_validation_interruption_receipt_v1.json"
)

INTERRUPTED_SOURCE_REPOSITORY_COMMIT = (
    "ca09f5f004ed8280469edfeb0f2164f071b52a71"
)
INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST = (
    "40c253e07c55fbb8f12d333f7134562446bc6ee0d85a824749834855c1dc5433"
)
INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST = (
    "494eabc64d6fa64c0547231a03907ff1567acfccab666528b207d2d5ea6a732b"
)
INTERRUPTED_SCORER_CONTRACT_DIGEST = (
    "7e1793e16de1c5c3bef9f966a5231b3b94d9ef43ba69d235305d811bb3d12ffc"
)

PERFORMANCE_RECEIPT_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE /
    "preoutcome_small_search_performance_interruption_receipt_v1.json"
)
PROJECTION_RECEIPT_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE /
    "preoutcome_projection_fix_interruption_receipt_v1.json"
)
MIXED_DISPOSITION_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE /
    "preserved_state_mixed_precontract_disposition_reachability_v2.json"
)
SCORER_CONTRACT_RELATIVE_PATH = (
    SCORER_ROOT_RELATIVE / "scorer_contract_v1_2.json"
)
CLEAN_LAUNCH_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE / "clean_source_launch_receipt.json"
)
PREIDENTITY_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE / "pre_identity_allocation_validation.json"
)
SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH = (
    SCORER_FIT_RELATIVE / "small_fixed_prefix_source_reissue_receipt_v1.json"
)

PERFORMANCE_SCHEMA = (
    "go2_scorer_small_search_performance_preoutcome_interruption_v1"
)
PERFORMANCE_STATUS = "SUPERSEDED_PRE_OUTCOME_PERFORMANCE_INTERRUPTION"
PROJECTION_SCHEMA = "go2_scorer_projection_fix_preoutcome_interruption_v1"
PROJECTION_STATUS = "SUPERSEDED_PRE_OUTCOME_IMPLEMENTATION_INTERRUPTION"
MIXED_DISPOSITION_SCHEMA = (
    "go2_scorer_fit_preserved_state_mixed_precontract_"
    "disposition_reachability_v2"
)
MIXED_DISPOSITION_STATUS = (
    "PASS_PREOUTCOME_37_RETAINED_8_REPLACEMENT_DISPOSITION"
)
SCORER_CONTRACT_SCHEMA = "go2_utility_scorer_contract_v1_2_artifact"
CLEAN_LAUNCH_SCHEMA = "go2_utility_scorer_v1_2_clean_source_launch_receipt"
PREIDENTITY_SCHEMA = (
    "go2_candidate_allocation_v1_2_pre_identity_structural_validation"
)
DEVELOPMENT_STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"


def _authority_binding(
        *, label: str, managed_root: Path, active_path: Path,
        archive_root: Path, self_digest_key: str, self_digest: str,
        raw_sha256: str, byte_count: int) -> dict[str, Any]:
    return {
        "label": label,
        "managed_root": str(managed_root),
        "active_path": str(active_path),
        "archive_path": str(
            archive_root / "authorities" /
            f"{active_path.stem}.{self_digest[:8]}.json"
        ),
        "self_digest_key": self_digest_key,
        "self_digest": self_digest,
        "raw_sha256": raw_sha256,
        "byte_count": byte_count,
    }


INTERRUPTED_AUTHORITIES: dict[str, dict[str, Any]] = {
    "performance_interruption": _authority_binding(
        label="performance_interruption",
        managed_root=CORPUS_ROOT_RELATIVE,
        active_path=PERFORMANCE_RECEIPT_RELATIVE_PATH,
        archive_root=ARCHIVE_ROOT_RELATIVE,
        self_digest_key=(
            "preoutcome_small_search_performance_interruption_receipt_digest"
        ),
        self_digest=(
            "045d8622a1caee597ceabe4e38775ce0e3ce5602c1ffe401aaaa64e65d383eaf"
        ),
        raw_sha256=(
            "3ed53a03576bf31a5203016c4c10a15c6fb12bc26bb202fc1974071bba2ae090"
        ),
        byte_count=253_157,
    ),
    "projection_interruption": _authority_binding(
        label="projection_interruption",
        managed_root=CORPUS_ROOT_RELATIVE,
        active_path=PROJECTION_RECEIPT_RELATIVE_PATH,
        archive_root=ARCHIVE_ROOT_RELATIVE,
        self_digest_key=(
            "preoutcome_projection_fix_interruption_receipt_digest"
        ),
        self_digest=(
            "86d28a77d5749635c43ae2596340b6f1e6dfbdf81fb32192439581e2bb537faf"
        ),
        raw_sha256=(
            "1dc79857dc83ceebad6b7b8726f3c11b8b08a6318fb1768cf59edd2b43f4f330"
        ),
        byte_count=19_061,
    ),
    "mixed_disposition": _authority_binding(
        label="mixed_disposition",
        managed_root=CORPUS_ROOT_RELATIVE,
        active_path=MIXED_DISPOSITION_RELATIVE_PATH,
        archive_root=ARCHIVE_ROOT_RELATIVE,
        self_digest_key="mixed_precontract_disposition_receipt_digest",
        self_digest=(
            "5e5013b512a8562120fbe73292930f84ecf98850fa4cbfa8fd895ff1a281bd03"
        ),
        raw_sha256=(
            "a96344d951415cf90e94cc6f3a63c0e878025f56fa7e42c2ce60559e2248239b"
        ),
        byte_count=29_403,
    ),
    "scorer_contract": _authority_binding(
        label="scorer_contract",
        managed_root=SCORER_ROOT_RELATIVE,
        active_path=SCORER_CONTRACT_RELATIVE_PATH,
        archive_root=SCORER_ARCHIVE_ROOT_RELATIVE,
        self_digest_key="contract_artifact_digest",
        self_digest=(
            "7dec89d28635b2bbe8ba811d25be973db1d37f36144407e98e84042d38df8204"
        ),
        raw_sha256=(
            "5ebad4e8f4601274c67de0b2c9de136b49044edf6aeb6edf316dad1ea353fde2"
        ),
        byte_count=87_210,
    ),
    "clean_launch": _authority_binding(
        label="clean_launch",
        managed_root=CORPUS_ROOT_RELATIVE,
        active_path=CLEAN_LAUNCH_RELATIVE_PATH,
        archive_root=ARCHIVE_ROOT_RELATIVE,
        self_digest_key="clean_source_launch_receipt_digest",
        self_digest=(
            "b0665c4cdc124dadf417c556686d9eda5969af206a3aba3e6d79f49897eca44f"
        ),
        raw_sha256=(
            "49b89a1c496d9a41f5af9db3441cbb6a2111b795c3f7a223c2298947ba711478"
        ),
        byte_count=2_435,
    ),
}
AUTHORITY_LABELS = (
    "performance_interruption", "projection_interruption",
    "mixed_disposition", "scorer_contract", "clean_launch",
)
# These four canonical paths are intentionally reused by current-source
# successor authorities after the historical bytes have been archived.  The
# V1 performance receipt is different: V2 has a distinct path, so its old
# active name must remain absent.
SUCCESSOR_REUSABLE_ACTIVE_AUTHORITY_LABELS = frozenset({
    "projection_interruption", "mixed_disposition", "scorer_contract",
    "clean_launch",
})

RETAINED_PREIDENTITY_ARTIFACT = {
    "managed_root": str(CORPUS_ROOT_RELATIVE),
    "path": str(PREIDENTITY_RELATIVE_PATH),
    "self_digest_key": "pre_identity_validation_digest",
    "self_digest": (
        "46efa42e3bdcad6df6cd4e404c2e8a796a9a331109a433cfbfffcfa18bf60d"
    ),
    "raw_sha256": (
        "a7f23011cdfec1f7a1938bfff57b4e6aa5f32b4e69082236c57d37a9ffd50256"
    ),
    "byte_count": 190_128,
}
RETAINED_PREIDENTITY_PROOF_SOURCE_BINDINGS = (
    {
        "label": "candidate_allocator_source",
        "path": "lewm/oracle/go2_candidate_allocation_v1_2.py",
        "raw_sha256": (
            "9ebd494d979f73fe63863731740418e60ccd2d6d3f61d1f11171c879f598d0b7"
        ),
        "byte_count": 51_923,
    },
    {
        "label": "candidate_allocation_amendment",
        "path": (
            "docs/lewm_go2_shared_utility_scorer_v1_2_"
            "allocation_amendment_v1_2026-08-11.json"
        ),
        "raw_sha256": (
            "1790429d6c02deebc794aa255be3b8c93ac5278de9c8c94920ee13b877fb5f38"
        ),
        "byte_count": 3_046,
    },
)
RETAINED_PREIDENTITY_VALIDATION_MODE = (
    "EXACT_CA09_ARTIFACT_AND_UNCHANGED_ALLOCATOR_SOURCE_REUSE_NO_MILP"
)

FIXED_WRAPPER_ACTIVE_PATHS = (
    str(SCORER_FIT_RELATIVE /
        "active_mixed_state_shard_large_enclosed_maze_reachability_v2.json"),
    str(SCORER_FIT_RELATIVE /
        "active_mixed_state_shard_local_composite_motifs_reachability_v2.json"),
    str(SCORER_FIT_RELATIVE /
        "active_mixed_state_shard_loop_alias_stress_reachability_v2.json"),
    str(SCORER_FIT_RELATIVE / "state_shard_medium_enclosed_maze.json"),
    str(SCORER_FIT_RELATIVE / "state_shard_open_obstacle_field.json"),
    str(SCORER_FIT_RELATIVE / "state_shard_rough_local_dynamics.json"),
    str(SCORER_FIT_RELATIVE / "state_shard_visual_sensor_stress.json"),
)
ABSENT_SUCCESSOR_OUTPUT_PATHS = (
    *FIXED_WRAPPER_ACTIVE_PATHS,
    str(SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH),
)

ZERO_SCIENCE_FIELDS = {
    "candidate_outcomes_loaded": False,
    "branch_identities_created": False,
    "branches_attempted": 0,
    "frames_rendered": 0,
    "target_latents_encoded": 0,
    "scorer_training_started": False,
    "scorer_qualification_started": False,
    "predictor_checkpoints_opened": 0,
}
INTERPRETER_VERSION_KEYS = ("python", "numpy", "scipy")
INTERRUPTED_ARGV = (
    ".generated/venvs/genesis_render_vulkan/bin/python",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "--pool", "scorer_fit",
    "--stage", "reissue-performance-fixed-states",
)
INTERRUPTED_INTERPRETER_VERSIONS = {
    "python": "3.12.3",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
}
INTERRUPTED_CALL_CHAIN = (
    "_revalidate_performance_interrupted_fixed_shard",
    "_validate_state_resolution_scene_request",
    "_state_shard_bindings",
    "_load_pre_identity_allocation_validation",
    "ALLOC.validate_pre_identity_structural_validation",
    "build_pre_identity_structural_validation",
    "build_allocation_manifest",
    "_lexicographic_rotations",
    "scipy.optimize.milp",
)


class FixedReissueValidationInterruptionError(RuntimeError):
    """The fixed-reissue transition custody could not be established."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()


def _raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return bool(
        isinstance(value, str) and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_successor_source(
        source_repository_commit: str, clean_source_binding_digest: str,
        bound_implementations_digest: str) -> None:
    if not _is_hex(source_repository_commit, 40):
        raise FixedReissueValidationInterruptionError(
            "successor source commit is invalid")
    if not _is_hex(clean_source_binding_digest, 64):
        raise FixedReissueValidationInterruptionError(
            "successor clean-source digest is invalid")
    if not _is_hex(bound_implementations_digest, 64):
        raise FixedReissueValidationInterruptionError(
            "successor implementation digest is invalid")
    if source_repository_commit == INTERRUPTED_SOURCE_REPOSITORY_COMMIT:
        raise FixedReissueValidationInterruptionError(
            "transition requires a later clean source commit")


def _forbidden(path: Path) -> bool:
    return any(
        part == ".." or part == "sealed" or part == "sealed_test.json"
        or part.startswith("sealed_") for part in path.parts
    )


def _assert_no_symlink(path: Path) -> None:
    if _forbidden(path):
        raise FixedReissueValidationInterruptionError(
            "transition path crosses inaccessible custody")
    absolute = path if path.is_absolute() else Path.cwd() / path
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise FixedReissueValidationInterruptionError(
                "transition descendant contains a symlink")


def _pin_managed(
        relative_path: str | Path, *, root: Path,
        managed_root_relative: str | Path = CORPUS_ROOT_RELATIVE) -> Path:
    repository = Path(root)
    if not repository.is_absolute():
        repository = Path.cwd() / repository
    managed = repository / Path(managed_root_relative)
    logical = repository / Path(relative_path)
    if _forbidden(managed) or _forbidden(logical):
        raise FixedReissueValidationInterruptionError(
            "transition path crosses inaccessible custody")
    try:
        suffix = logical.relative_to(managed)
    except ValueError as exc:
        raise FixedReissueValidationInterruptionError(
            "transition path escaped its managed root") from exc
    if not suffix.parts:
        raise FixedReissueValidationInterruptionError(
            "transition path names only its managed root")
    _assert_no_symlink(managed.parent)
    if managed.is_symlink():
        raw_target = managed.readlink()
        target = raw_target if raw_target.is_absolute() \
            else managed.parent / raw_target
        if target.name != managed.name or _forbidden(target):
            raise FixedReissueValidationInterruptionError(
                "managed transition alias identity changed")
        _assert_no_symlink(target)
        try:
            canonical_root = target.resolve(strict=True)
        except OSError as exc:
            raise FixedReissueValidationInterruptionError(
                "managed transition root is missing") from exc
    else:
        if not managed.is_dir():
            raise FixedReissueValidationInterruptionError(
                "managed transition root is missing")
        canonical_root = managed.resolve(strict=True)
    if not canonical_root.is_dir() or canonical_root.name != managed.name:
        raise FixedReissueValidationInterruptionError(
            "managed transition root identity changed")
    _assert_no_symlink(canonical_root)
    pinned = canonical_root.joinpath(*suffix.parts)
    _assert_no_symlink(pinned)
    return pinned


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FixedReissueValidationInterruptionError(f"{label} is missing")
    try:
        payload = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FixedReissueValidationInterruptionError(
            f"{label} JSON is invalid") from exc
    if not isinstance(payload, dict):
        raise FixedReissueValidationInterruptionError(
            f"{label} is not an object")
    return payload


def _verify_self(payload: Mapping[str, Any], key: str, label: str) -> None:
    observed = payload.get(key)
    expected = _digest({
        name: value for name, value in payload.items() if name != key
    })
    if not _is_hex(observed, 64) or observed != expected:
        raise FixedReissueValidationInterruptionError(
            f"{label} self binding changed")


def _exact_file(path: Path, binding: Mapping[str, Any]) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    raw = path.read_bytes()
    return (
        len(raw) == binding["byte_count"]
        and _raw_sha256(raw) == binding["raw_sha256"]
    )


def _binding_paths(
        binding: Mapping[str, Any], *, root: Path) -> tuple[Path, Path]:
    managed = binding["managed_root"]
    return (
        _pin_managed(
            binding["active_path"], root=root,
            managed_root_relative=managed),
        _pin_managed(
            binding["archive_path"], root=root,
            managed_root_relative=managed),
    )


def _same_file(left: Path, right: Path) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _locate_authority(
        binding: Mapping[str, Any], *, root: Path, require_archived: bool,
        label: str, allow_same_inode_crash: bool = False,
        ) -> tuple[dict[str, Any], Path]:
    active, archive = _binding_paths(binding, root=root)
    active_exists = active.exists() or active.is_symlink()
    archive_exists = archive.exists() or archive.is_symlink()
    active_exact = _exact_file(active, binding)
    archive_exact = _exact_file(archive, binding)
    successor_replacement = (
        require_archived and archive_exact and active_exists
        and not active_exact
        and binding.get("label") in SUCCESSOR_REUSABLE_ACTIVE_AUTHORITY_LABELS
        and active.is_file() and not active.is_symlink()
    )
    if active_exists and not active_exact and not successor_replacement:
        raise FixedReissueValidationInterruptionError(
            f"{label} active path is a collision")
    if archive_exists and not archive_exact:
        raise FixedReissueValidationInterruptionError(
            f"{label} archive path is a collision")
    if active_exact and archive_exact:
        if allow_same_inode_crash and _same_file(active, archive):
            chosen = archive
        else:
            raise FixedReissueValidationInterruptionError(
                f"{label} active/archive collision")
    elif archive_exact:
        chosen = archive
    elif active_exact and not require_archived:
        chosen = active
    else:
        qualifier = "archived " if require_archived else ""
        raise FixedReissueValidationInterruptionError(
            f"exact {qualifier}{label} is unavailable")
    payload = _load_json(chosen, label)
    key = str(binding["self_digest_key"])
    _verify_self(payload, key, label)
    if payload.get(key) != binding["self_digest"]:
        raise FixedReissueValidationInterruptionError(
            f"{label} declared digest changed")
    return payload, chosen


def _load_retained_preidentity(*, root: Path) -> dict[str, Any]:
    binding = RETAINED_PREIDENTITY_ARTIFACT
    path = _pin_managed(
        binding["path"], root=root,
        managed_root_relative=binding["managed_root"])
    if not _exact_file(path, binding):
        raise FixedReissueValidationInterruptionError(
            "exact retained pre-identity artifact is unavailable")
    payload = _load_json(path, "retained pre-identity artifact")
    key = str(binding["self_digest_key"])
    _verify_self(payload, key, "retained pre-identity artifact")
    if payload.get(key) != binding["self_digest"]:
        raise FixedReissueValidationInterruptionError(
            "retained pre-identity declared digest changed")
    if (payload.get("schema") != PREIDENTITY_SCHEMA
            or payload.get("status") !=
            "PASS_PRE_IDENTITY_STRUCTURAL_VALIDATION"):
        raise FixedReissueValidationInterruptionError(
            "retained pre-identity artifact contract changed")
    global_counts = payload.get("global")
    if (not isinstance(global_counts, Mapping)
            or global_counts.get("state_slot_count") != 120
            or global_counts.get("candidate_slot_count") != 720):
        raise FixedReissueValidationInterruptionError(
            "retained pre-identity allocation cardinality changed")
    return payload


def _validate_preidentity_proof_sources(*, root: Path) -> None:
    repository = Path(root)
    if not repository.is_absolute():
        repository = Path.cwd() / repository
    _assert_no_symlink(repository)
    for binding in RETAINED_PREIDENTITY_PROOF_SOURCE_BINDINGS:
        relative = Path(binding["path"])
        if relative.is_absolute() or _forbidden(relative):
            raise FixedReissueValidationInterruptionError(
                "pre-identity proof source path is invalid")
        path = repository / relative
        try:
            path.relative_to(repository)
        except ValueError as exc:  # pragma: no cover - frozen constants
            raise FixedReissueValidationInterruptionError(
                "pre-identity proof source escaped repository") from exc
        _assert_no_symlink(path)
        if not path.is_file() or path.is_symlink():
            raise FixedReissueValidationInterruptionError(
                f"pre-identity proof source is missing: {binding['label']}")
        raw = path.read_bytes()
        if (len(raw) != binding["byte_count"]
                or _raw_sha256(raw) != binding["raw_sha256"]):
            raise FixedReissueValidationInterruptionError(
                f"pre-identity proof source changed: {binding['label']}")


def _receipt_style_binding(label: str) -> dict[str, Any]:
    binding = INTERRUPTED_AUTHORITIES[label]
    status = {
        "performance_interruption": PERFORMANCE_STATUS,
        "projection_interruption": PROJECTION_STATUS,
    }[label]
    return {
        "path": binding["active_path"],
        "receipt_digest": binding["self_digest"],
        "raw_sha256": binding["raw_sha256"],
        "byte_count": binding["byte_count"],
        "status": status,
    }


def _validate_zero_if_present(payload: Mapping[str, Any], label: str) -> None:
    for key, expected in ZERO_SCIENCE_FIELDS.items():
        if key in payload and payload.get(key) != expected:
            raise FixedReissueValidationInterruptionError(
                f"{label} is not outcome-free")


def _validate_authority_cross_bindings(
        payloads: Mapping[str, Mapping[str, Any]]) -> None:
    if set(payloads) != set(AUTHORITY_LABELS):
        raise FixedReissueValidationInterruptionError(
            "interrupted authority coverage changed")
    performance = payloads["performance_interruption"]
    projection = payloads["projection_interruption"]
    disposition = payloads["mixed_disposition"]
    contract = payloads["scorer_contract"]
    launch = payloads["clean_launch"]
    for payload, schema, status, label in (
        (performance, PERFORMANCE_SCHEMA, PERFORMANCE_STATUS,
         "performance interruption"),
        (projection, PROJECTION_SCHEMA, PROJECTION_STATUS,
         "projection interruption"),
    ):
        if (payload.get("schema") != schema or payload.get("status") != status
                or payload.get("superseding_source_repository_commit")
                != INTERRUPTED_SOURCE_REPOSITORY_COMMIT
                or payload.get("superseding_clean_source_binding_digest")
                != INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST
                or payload.get("superseding_bound_implementations_digest")
                != INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST):
            raise FixedReissueValidationInterruptionError(
                f"{label} ca09 source binding changed")
    if (disposition.get("schema") != MIXED_DISPOSITION_SCHEMA
            or disposition.get("status") != MIXED_DISPOSITION_STATUS
            or disposition.get("complete") is not True
            or disposition.get("source_repository_commit")
            != INTERRUPTED_SOURCE_REPOSITORY_COMMIT
            or disposition.get("clean_source_binding_digest")
            != INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST
            or disposition.get("bound_implementations_digest")
            != INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST
            or disposition.get("retained_predecessor_state_count") != 37
            or disposition.get("rejected_predecessor_state_count") != 8
            or disposition.get("replacement_slot_count") != 8):
        raise FixedReissueValidationInterruptionError(
            "mixed disposition ca09 binding changed")
    expected_performance = _receipt_style_binding(
        "performance_interruption")
    expected_projection = _receipt_style_binding(
        "projection_interruption")
    contract_binding = contract.get("clean_source_binding")
    if (contract.get("schema") != SCORER_CONTRACT_SCHEMA
            or contract.get("status") != DEVELOPMENT_STATUS
            or contract.get("complete") is not True
            or contract.get("source_repository_commit")
            != INTERRUPTED_SOURCE_REPOSITORY_COMMIT
            or contract.get("clean_source_binding_digest")
            != INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST
            or not isinstance(contract_binding, Mapping)
            or contract_binding.get("source_repository_commit")
            != INTERRUPTED_SOURCE_REPOSITORY_COMMIT
            or contract_binding.get("bound_implementations_digest")
            != INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST
            or contract.get("scorer_contract_v1_2_digest")
            != INTERRUPTED_SCORER_CONTRACT_DIGEST
            or contract.get("mixed_precontract_disposition_receipt_digest")
            != INTERRUPTED_AUTHORITIES["mixed_disposition"]["self_digest"]
            or contract.get("preoutcome_small_search_performance_interruption")
            != expected_performance
            or contract.get("preoutcome_projection_fix_interruption")
            != expected_projection):
        raise FixedReissueValidationInterruptionError(
            "scorer contract ca09 cross-binding changed")
    if (launch.get("schema") != CLEAN_LAUNCH_SCHEMA
            or launch.get("status") != DEVELOPMENT_STATUS
            or launch.get("complete") is not True
            or launch.get("source_repository_commit")
            != INTERRUPTED_SOURCE_REPOSITORY_COMMIT
            or launch.get("clean_source_binding_digest")
            != INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST
            or launch.get("bound_implementations_digest")
            != INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST
            or launch.get("scorer_contract_v1_2_digest")
            != INTERRUPTED_SCORER_CONTRACT_DIGEST
            or launch.get("scorer_contract_artifact_digest")
            != INTERRUPTED_AUTHORITIES["scorer_contract"]["self_digest"]
            or launch.get("scorer_contract_artifact_sha256")
            != INTERRUPTED_AUTHORITIES["scorer_contract"]["raw_sha256"]
            or launch.get("mixed_precontract_disposition_receipt_digest")
            != INTERRUPTED_AUTHORITIES["mixed_disposition"]["self_digest"]
            or launch.get("pre_identity_allocation_validation_digest")
            != RETAINED_PREIDENTITY_ARTIFACT["self_digest"]
            or launch.get("preoutcome_small_search_performance_interruption")
            != expected_performance
            or launch.get("preoutcome_projection_fix_interruption")
            != expected_projection):
        raise FixedReissueValidationInterruptionError(
            "clean launch ca09 cross-binding changed")
    for label, payload in payloads.items():
        _validate_zero_if_present(payload, label)


def _validate_outcome_attestation(
        attestation: Mapping[str, Any]) -> dict[str, Any]:
    try:
        SELECTOR.validate_phase1_outcome_surface_absence_attestation(attestation)
    except Exception as exc:
        raise FixedReissueValidationInterruptionError(
            "outcome-surface absence audit failed") from exc
    return copy.deepcopy(dict(attestation))


def _absence_rows(*, root: Path) -> list[dict[str, Any]]:
    rows = []
    for relative in ABSENT_SUCCESSOR_OUTPUT_PATHS:
        path = _pin_managed(relative, root=root)
        if path.exists() or path.is_symlink():
            raise FixedReissueValidationInterruptionError(
                f"pre-issuance output already exists: {relative}")
        rows.append({
            "path": relative,
            "exists": False,
            "kind": "absent",
            "artifact_absent": True,
        })
    return rows


def _execution_record(
        execution_argv: Sequence[str] | None,
        interpreter_versions: Mapping[str, str] | None) -> dict[str, Any]:
    if execution_argv is not None:
        if (isinstance(execution_argv, (str, bytes))
                or not isinstance(execution_argv, Sequence)
                or list(execution_argv) != list(INTERRUPTED_ARGV)):
            raise FixedReissueValidationInterruptionError(
                "execution argv differs from the frozen interrupted command")
    if interpreter_versions is not None:
        if (not isinstance(interpreter_versions, Mapping)
                or dict(interpreter_versions)
                != INTERRUPTED_INTERPRETER_VERSIONS):
            raise FixedReissueValidationInterruptionError(
                "interpreter versions differ from the frozen interruption")
    return {
        "stage": "reissue-performance-fixed-states",
        "argv": list(INTERRUPTED_ARGV),
        "interpreter_versions": dict(INTERRUPTED_INTERPRETER_VERSIONS),
        "pid_recorded": False,
        "timing_recorded": False,
        "exit_code": 130,
        "signal": "SIGINT",
        "exception": "KeyboardInterrupt",
        "preissuance_validation_entered": True,
        "interrupted_call_chain": list(INTERRUPTED_CALL_CHAIN),
        "wrapper_loop_entered": False,
        "wrapper_count_issued": 0,
        "small_prefix_reissue_receipt_issued": False,
        "genesis_started": False,
    }


def _receipt_payload(
        *, source_repository_commit: str, clean_source_binding_digest: str,
        bound_implementations_digest: str,
        outcome_attestation: Mapping[str, Any],
        absence_rows: Sequence[Mapping[str, Any]],
        execution: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "record_complete": True,
        "attempt_complete": False,
        "binding_receipt": False,
        "scientific_gate_input": False,
        "may_satisfy_selector_gate": False,
        "retry_authority": False,
        "resume_authority": False,
        "wrapper_issuance_authority": False,
        "interrupted_source_repository_commit":
            INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
        "interrupted_clean_source_binding_digest":
            INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST,
        "interrupted_bound_implementations_digest":
            INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST,
        "interrupted_scorer_contract_v1_2_digest":
            INTERRUPTED_SCORER_CONTRACT_DIGEST,
        "superseding_source_repository_commit": source_repository_commit,
        "superseding_clean_source_binding_digest":
            clean_source_binding_digest,
        "superseding_bound_implementations_digest":
            bound_implementations_digest,
        "execution": copy.deepcopy(dict(execution)),
        "interrupted_authorities": copy.deepcopy(INTERRUPTED_AUTHORITIES),
        "interrupted_authority_count": 5,
        "authority_archive_order": list(AUTHORITY_LABELS),
        "retained_preidentity_artifact":
            copy.deepcopy(RETAINED_PREIDENTITY_ARTIFACT),
        "retained_preidentity_proof_source_bindings":
            copy.deepcopy(list(RETAINED_PREIDENTITY_PROOF_SOURCE_BINDINGS)),
        "retained_preidentity_validation_mode":
            RETAINED_PREIDENTITY_VALIDATION_MODE,
        "preidentity_artifact_retained_not_archived": True,
        "preidentity_exact_proof_reuse_only": True,
        "preidentity_milp_validation_rerun": False,
        "new_candidate_allocation_performed": False,
        "absent_successor_outputs": [dict(row) for row in absence_rows],
        "fixed_wrapper_active_path_count": 7,
        "small_prefix_reissue_receipt_existed": False,
        "absence_checked_before_first_authority_archive": True,
        "absence_rechecked_before_receipt_install": True,
        "outcome_surface_absence_rechecked_before_receipt_install": True,
        "absence_is_an_issuance_time_fact": True,
        "live_absence_reopen_after_legitimate_successor_outputs": False,
        "outcome_surface_absence_attestation":
            copy.deepcopy(dict(outcome_attestation)),
        "outcome_surface_absence_attestation_digest":
            outcome_attestation["attestation_digest"],
        **ZERO_SCIENCE_FIELDS,
        "genesis_started": False,
        "fixed_wrapper_count_issued": 0,
        "small_prefix_reissue_receipt_issued": False,
        "five_authorities_archived_nonoverwriting": True,
        "retained_preidentity_bytes_unchanged": True,
    }
    payload[SELF_DIGEST_KEY] = _digest(payload)
    return payload


def _fsync_directories(*directories: Path) -> None:
    for directory in dict.fromkeys(directories):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _archive_one(
        binding: Mapping[str, Any], *, root: Path, label: str) -> None:
    _payload, source = _locate_authority(
        binding, root=root, require_archived=False, label=label,
        allow_same_inode_crash=True)
    active, archive = _binding_paths(binding, root=root)
    if source == archive:
        if active.exists() or active.is_symlink():
            if not _same_file(active, archive):
                raise FixedReissueValidationInterruptionError(
                    f"{label} active/archive collision")
            active.unlink()
            _fsync_directories(active.parent, archive.parent)
        return
    if source != active:
        raise FixedReissueValidationInterruptionError(
            f"{label} location changed")
    archive.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink(archive.parent)
    if archive.exists() or archive.is_symlink():
        raise FixedReissueValidationInterruptionError(
            f"{label} archive collision")
    os.link(active, archive, follow_symlinks=False)
    _fsync_directories(archive.parent)
    active.unlink()
    _fsync_directories(active.parent, archive.parent)


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink(path.parent)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FixedReissueValidationInterruptionError(
            "transition receipt temporary exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o644)
    installed = False
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write((json.dumps(
                payload, indent=2, sort_keys=True) + "\n").encode())
            sink.flush()
            os.fsync(sink.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise FixedReissueValidationInterruptionError(
                "transition receipt collision") from exc
        temporary.unlink()
        installed = True
        _fsync_directories(path.parent)
    finally:
        if not installed:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def validate_interruption_receipt(
        receipt: Mapping[str, Any], *,
        expected_source_repository_commit: str,
        expected_clean_source_binding_digest: str,
        expected_bound_implementations_digest: str,
        expected_execution_argv: Sequence[str] | None = None,
        expected_interpreter_versions: Mapping[str, str] | None = None,
        root: Path = ROOT) -> dict[str, Any]:
    """Read-only validation of the immutable transition and archived inputs."""

    _validate_successor_source(
        expected_source_repository_commit,
        expected_clean_source_binding_digest,
        expected_bound_implementations_digest)
    if not isinstance(receipt, Mapping):
        raise FixedReissueValidationInterruptionError(
            "transition receipt is not a mapping")
    _verify_self(receipt, SELF_DIGEST_KEY, "transition receipt")
    attestation = _validate_outcome_attestation(
        receipt.get("outcome_surface_absence_attestation", {}))
    execution = receipt.get("execution")
    if not isinstance(execution, Mapping):
        raise FixedReissueValidationInterruptionError(
            "transition execution binding is invalid")
    reconstructed_execution = _execution_record(None, None)
    if dict(execution) != reconstructed_execution:
        raise FixedReissueValidationInterruptionError(
            "transition execution binding changed")
    _execution_record(
        expected_execution_argv, expected_interpreter_versions)
    expected_absence = [{
        "path": relative,
        "exists": False,
        "kind": "absent",
        "artifact_absent": True,
    } for relative in ABSENT_SUCCESSOR_OUTPUT_PATHS]
    expected = _receipt_payload(
        source_repository_commit=expected_source_repository_commit,
        clean_source_binding_digest=expected_clean_source_binding_digest,
        bound_implementations_digest=expected_bound_implementations_digest,
        outcome_attestation=attestation,
        absence_rows=expected_absence,
        execution=reconstructed_execution,
    )
    if dict(receipt) != expected:
        raise FixedReissueValidationInterruptionError(
            "transition receipt differs from exact reconstruction")
    payloads = {
        label: _locate_authority(
            INTERRUPTED_AUTHORITIES[label], root=root,
            require_archived=True, label=f"archived {label}")[0]
        for label in AUTHORITY_LABELS
    }
    _validate_authority_cross_bindings(payloads)
    validate_retained_preidentity_artifact(receipt, root=root)
    return dict(receipt)


def load_and_validate_interruption_receipt(
        *, expected_source_repository_commit: str,
        expected_clean_source_binding_digest: str,
        expected_bound_implementations_digest: str,
        expected_execution_argv: Sequence[str] | None = None,
        expected_interpreter_versions: Mapping[str, str] | None = None,
        root: Path = ROOT) -> dict[str, Any]:
    path = _pin_managed(RECEIPT_RELATIVE_PATH, root=root)
    receipt = _load_json(path, "fixed-reissue validation interruption receipt")
    return validate_interruption_receipt(
        receipt,
        expected_source_repository_commit=expected_source_repository_commit,
        expected_clean_source_binding_digest=
            expected_clean_source_binding_digest,
        expected_bound_implementations_digest=
            expected_bound_implementations_digest,
        expected_execution_argv=expected_execution_argv,
        expected_interpreter_versions=expected_interpreter_versions,
        root=root,
    )


def issue_and_archive_interruption_receipt(
        *, source_repository_commit: str, clean_source_binding_digest: str,
        bound_implementations_digest: str,
        outcome_surface_absent: Callable[[], Mapping[str, Any]],
        execution_argv: Sequence[str] | None = None,
        interpreter_versions: Mapping[str, str] | None = None,
        root: Path = ROOT) -> dict[str, Any]:
    """Archive five exact ca09 authorities and issue one exclusive receipt."""

    _validate_successor_source(
        source_repository_commit, clean_source_binding_digest,
        bound_implementations_digest)
    receipt_path = _pin_managed(RECEIPT_RELATIVE_PATH, root=root)
    if receipt_path.exists() or receipt_path.is_symlink():
        return load_and_validate_interruption_receipt(
            expected_source_repository_commit=source_repository_commit,
            expected_clean_source_binding_digest=clean_source_binding_digest,
            expected_bound_implementations_digest=bound_implementations_digest,
            expected_execution_argv=execution_argv,
            expected_interpreter_versions=interpreter_versions,
            root=root,
        )

    attestation = _validate_outcome_attestation(outcome_surface_absent())
    execution = _execution_record(execution_argv, interpreter_versions)
    absence = _absence_rows(root=root)
    payloads: dict[str, dict[str, Any]] = {}
    # Complete validation precedes the first archive mutation.  The locator
    # accepts only the one recoverable hardlink-then-unlink crash state.
    for label in AUTHORITY_LABELS:
        payloads[label] = _locate_authority(
            INTERRUPTED_AUTHORITIES[label], root=root,
            require_archived=False, label=label,
            allow_same_inode_crash=True)[0]
    _validate_authority_cross_bindings(payloads)
    _load_retained_preidentity(root=root)
    _validate_preidentity_proof_sources(root=root)
    expected = _receipt_payload(
        source_repository_commit=source_repository_commit,
        clean_source_binding_digest=clean_source_binding_digest,
        bound_implementations_digest=bound_implementations_digest,
        outcome_attestation=attestation,
        absence_rows=absence,
        execution=execution,
    )

    for label in AUTHORITY_LABELS:
        _archive_one(
            INTERRUPTED_AUTHORITIES[label], root=root, label=label)
    _load_retained_preidentity(root=root)
    _validate_preidentity_proof_sources(root=root)
    if _absence_rows(root=root) != absence:
        raise FixedReissueValidationInterruptionError(
            "successor output absence changed during authority archive")
    if _validate_outcome_attestation(outcome_surface_absent()) != attestation:
        raise FixedReissueValidationInterruptionError(
            "outcome-surface absence changed during authority archive")
    _atomic_write(receipt_path, expected)
    return validate_interruption_receipt(
        expected,
        expected_source_repository_commit=source_repository_commit,
        expected_clean_source_binding_digest=clean_source_binding_digest,
        expected_bound_implementations_digest=bound_implementations_digest,
        expected_execution_argv=execution_argv,
        expected_interpreter_versions=interpreter_versions,
        root=root,
    )


def receipt_binding(
        receipt: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    validated = validate_interruption_receipt(
        receipt,
        expected_source_repository_commit=str(
            receipt.get("superseding_source_repository_commit", "")),
        expected_clean_source_binding_digest=str(
            receipt.get("superseding_clean_source_binding_digest", "")),
        expected_bound_implementations_digest=str(
            receipt.get("superseding_bound_implementations_digest", "")),
        root=root,
    )
    path = _pin_managed(RECEIPT_RELATIVE_PATH, root=root)
    on_disk = _load_json(path, "fixed-reissue validation interruption receipt")
    if on_disk != validated:
        raise FixedReissueValidationInterruptionError(
            "transition binding payload differs from disk")
    raw = path.read_bytes()
    return {
        "path": str(RECEIPT_RELATIVE_PATH),
        "receipt_digest": validated[SELF_DIGEST_KEY],
        "raw_sha256": _raw_sha256(raw),
        "byte_count": len(raw),
        "status": STATUS,
    }


def validate_retained_preidentity_artifact(
        receipt: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    """Reopen the exact ca09 artifact and unchanged proof sources, solve-free."""

    if not isinstance(receipt, Mapping):
        raise FixedReissueValidationInterruptionError(
            "transition receipt is not a mapping")
    _verify_self(receipt, SELF_DIGEST_KEY, "transition receipt")
    if (receipt.get("schema") != SCHEMA or receipt.get("status") != STATUS
            or receipt.get("retained_preidentity_artifact")
            != RETAINED_PREIDENTITY_ARTIFACT
            or receipt.get("retained_preidentity_proof_source_bindings")
            != list(RETAINED_PREIDENTITY_PROOF_SOURCE_BINDINGS)
            or receipt.get("retained_preidentity_validation_mode")
            != RETAINED_PREIDENTITY_VALIDATION_MODE
            or receipt.get("preidentity_artifact_retained_not_archived")
            is not True
            or receipt.get("preidentity_exact_proof_reuse_only") is not True
            or receipt.get("preidentity_milp_validation_rerun") is not False
            or receipt.get("new_candidate_allocation_performed") is not False):
        raise FixedReissueValidationInterruptionError(
            "retained pre-identity proof contract changed")
    _validate_preidentity_proof_sources(root=root)
    return copy.deepcopy(_load_retained_preidentity(root=root))


def load_archived_authority(
        label: str, receipt: Mapping[str, Any], *,
        root: Path = ROOT) -> dict[str, Any]:
    if label not in AUTHORITY_LABELS:
        raise FixedReissueValidationInterruptionError(
            "archived authority label is not registered")
    validate_interruption_receipt(
        receipt,
        expected_source_repository_commit=str(
            receipt.get("superseding_source_repository_commit", "")),
        expected_clean_source_binding_digest=str(
            receipt.get("superseding_clean_source_binding_digest", "")),
        expected_bound_implementations_digest=str(
            receipt.get("superseding_bound_implementations_digest", "")),
        root=root,
    )
    return _locate_authority(
        INTERRUPTED_AUTHORITIES[label], root=root, require_archived=True,
        label=f"archived {label}")[0]


def archived_authority_binding(
        label: str, receipt: Mapping[str, Any], *,
        root: Path = ROOT) -> dict[str, Any]:
    payload = load_archived_authority(label, receipt, root=root)
    binding = INTERRUPTED_AUTHORITIES[label]
    return {
        "path": binding["archive_path"],
        "receipt_digest": binding["self_digest"],
        "raw_sha256": binding["raw_sha256"],
        "byte_count": binding["byte_count"],
        "status": payload["status"],
    }


def load_archived_performance_receipt_v1(
        receipt: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    return load_archived_authority(
        "performance_interruption", receipt, root=root)


def archived_performance_receipt_binding_v1(
        receipt: Mapping[str, Any], *, root: Path = ROOT) -> dict[str, Any]:
    return archived_authority_binding(
        "performance_interruption", receipt, root=root)


def lineage_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "receipt_path": str(RECEIPT_RELATIVE_PATH),
        "interrupted_source_repository_commit":
            INTERRUPTED_SOURCE_REPOSITORY_COMMIT,
        "interrupted_clean_source_binding_digest":
            INTERRUPTED_CLEAN_SOURCE_BINDING_DIGEST,
        "interrupted_bound_implementations_digest":
            INTERRUPTED_BOUND_IMPLEMENTATIONS_DIGEST,
        "interrupted_scorer_contract_v1_2_digest":
            INTERRUPTED_SCORER_CONTRACT_DIGEST,
        "interrupted_authorities": copy.deepcopy(INTERRUPTED_AUTHORITIES),
        "interrupted_authority_count": 5,
        "successor_reusable_active_authority_labels": sorted(
            SUCCESSOR_REUSABLE_ACTIVE_AUTHORITY_LABELS),
        "archived_predecessor_validation_tolerates_registered_successor_paths":
            True,
        "retained_preidentity_artifact":
            copy.deepcopy(RETAINED_PREIDENTITY_ARTIFACT),
        "retained_preidentity_proof_source_bindings":
            copy.deepcopy(list(RETAINED_PREIDENTITY_PROOF_SOURCE_BINDINGS)),
        "retained_preidentity_validation_mode":
            RETAINED_PREIDENTITY_VALIDATION_MODE,
        "preidentity_exact_proof_reuse_only": True,
        "preidentity_milp_validation_rerun": False,
        "new_candidate_allocation_performed": False,
        "fixed_wrapper_active_paths": list(FIXED_WRAPPER_ACTIVE_PATHS),
        "fixed_wrapper_active_path_count": 7,
        "small_prefix_reissue_receipt_path":
            str(SMALL_PREFIX_REISSUE_RECEIPT_RELATIVE_PATH),
        "wrapper_loop_entered": False,
        "wrapper_count_issued": 0,
        "outcome_surface_absence_rechecked_before_receipt_install": True,
        "execution_exit_code": 130,
        "execution_signal": "SIGINT",
        "execution_argv": list(INTERRUPTED_ARGV),
        "interpreter_versions": dict(INTERRUPTED_INTERPRETER_VERSIONS),
        "pid_recorded": False,
        "timing_recorded": False,
        "genesis_started": False,
        "scientific_outcome_existed": False,
        "scientific_gate_input": False,
        "may_satisfy_selector_gate": False,
        "retry_authority": False,
        "resume_authority": False,
        "wrapper_issuance_authority": False,
        "archive_install": "hardlink_then_unlink_no_overwrite",
        "receipt_install": "exclusive_fsynced_hardlink_no_overwrite",
    }
