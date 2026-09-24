"""One-shot V2 contract for the pre-outcome small-completion benchmark.

V1 remains an immutable failed execution of its own contract.  V2 changes
only process-pool readiness: all 32 workers must cross one deterministic,
non-scientific readiness barrier before sample 0 is timed, and the same live
pool must continue from the three-sample benchmark into scientific search if
and only if both frozen timing gates pass.

This module is custody-only.  Importing it does not inspect generated state,
start a worker, construct a solver problem, or read a scientific outcome or
mask.  The only write API is :func:`issue_contract`, which can create the one
dedicated V2 contract path exactly once (or validate an identical existing
artifact).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]

SCHEMA = "go2_parallel_small_completion_benchmark_v2_contract"
STATUS = "ISSUED_ONE_SHOT_PREOUTCOME_COLD_START_ASYMMETRY_SUCCESSOR"
SELF_DIGEST_KEY = "benchmark_v2_contract_digest"
CONTRACT_RELATIVE_PATH = Path(
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "small_completion_parallel_prefix_benchmark_v2_contract.json"
)
GENERATED_ROOT_RELATIVE_PATH = Path(".generated/go2_branch_corpus_v1_2")
SCORER_FIT_RELATIVE_PATH = (
    GENERATED_ROOT_RELATIVE_PATH / "scorer_fit"
)

V2_RUNTIME_OUTPUT_PATHS = (
    ("worker_readiness_record", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_worker_readiness_v2.json", "file"),
    ("attempt_start_receipt", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_attempt_start_v2.json", "file"),
    ("benchmark_receipt", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_prefix_benchmark_v2.json", "file"),
    ("scientific_search_plan", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_search_plan_v2.json", "file"),
    ("scientific_search_checkpoint_root", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_search_v2", "directory"),
    ("terminal_result", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_terminal_result_v2.json", "file"),
    ("terminal_failure", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_terminal_failure_v2.json", "file"),
    ("same_pool_terminal_wrapper_shutdown_receipt",
     SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_pool_shutdown_v2.json", "file"),
)

V1_DOWNSTREAM_OUTPUT_PATHS = (
    ("scientific_search_plan", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_search_plan_v1.json", "file"),
    ("scientific_search_checkpoint_root", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_search_v1", "directory"),
    ("terminal_result", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_terminal_result_v1.json", "file"),
    ("terminal_failure", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_terminal_failure_v1.json", "file"),
    ("joint_receipt", SCORER_FIT_RELATIVE_PATH /
     "small_completion_parallel_joint_receipt_v1.json", "file"),
    ("small_terminal_state_shard", SCORER_FIT_RELATIVE_PATH /
     "state_shard_small_enclosed_maze.json", "file"),
)

V1_FAILURE_STATUS_DESCRIPTOR = (
    "IMMUTABLE_FAIL_COLD_START_INCLUDED_IN_FIRST_TIMED_WAVE"
)
V1_FAILURE_RECEIPT_RELATIVE_PATH = Path(
    ".generated/go2_branch_corpus_v1_2/scorer_fit/"
    "small_completion_parallel_prefix_benchmark_v1.json"
)
V1_FAILURE_RECEIPT_SCHEMA = (
    "go2_parallel_small_completion_search_v1_benchmark_receipt"
)
V1_FAILURE_RECEIPT_DIGEST = (
    "afb4c190cf7d2e93b678a546fc233340102c6f5260110b1471752bc54a0e88d6"
)
V1_FAILURE_RECEIPT_RAW_SHA256 = (
    "cc3b07b3ed470058dc395d0eb34d5d6cd83e8edc0140e4c18f249d4d4747fe5b"
)
V1_FAILURE_RECEIPT_BYTE_COUNT = 2_688
V1_FAILURE_SOURCE_REPOSITORY_COMMIT = (
    "d9d129e2bbea8519f7ed3186f3cfb3c661baba04"
)
V1_FAILURE_SOURCE_BINDING_DIGEST = (
    "2c72545b8abb503ba11ae03a98076afba2ebc478a308ff928023998b091c56f3"
)
V1_FAILURE_STATE_PROJECTION_DIGEST = (
    "53608bb6c5954917c449e14716a9a338f8d847c8d9e9b41be4b0d94267f33488"
)
V1_FAILURE_SOURCE_IDENTITY_MANIFEST_DIGEST = (
    "00f671d5c0f4cba4a4a71b9d313748b95ed31c4452876a87a1e1494b4922c044"
)
V1_FAILURE_SAMPLE_ROWS_DIGEST = (
    "0671161f2bf62075ca1ffdc20b7f2f79d3ec19b09108b5c9f230d56b80f30b75"
)
V1_FAILURE_MEDIAN_PARALLEL_FRACTION = 0.13736699377347508
V1_FAILURE_MAXIMUM_PARALLEL_FRACTION = 1.9944577548815676

WORKER_COUNT = 32
WORKER_THREADS = 1
ROTATION_COUNT = 12
PREFIX_STATE_COUNT = 120
SAMPLE_PREFIX_INDICES = (0, 1, 2)
MAXIMUM_PARALLEL_FRACTION = 0.5

SCIENTIFIC_INPUT_BINDINGS_SCHEMA = (
    "go2_parallel_small_completion_benchmark_v2_"
    "predecessor_scientific_input_bindings"
)
SCIENTIFIC_INPUT_BINDING_KEYS = frozenset({
    "schema",
    "provisional_search_plan_digest",
    "benchmark_source_binding_digest",
    "rank_zero_source_identity_manifest_digest",
    "rank_zero_state_projection_digest",
    "candidate_pool_scene_ids_digest",
    "fixed_state_projection_digest",
    "candidate_outcomes_consumed",
    "scientific_masks_accessed",
})

EXPECTED_V2_SOURCE_PATHS = (
    "lewm/oracle/go2_parallel_small_completion_benchmark_v2_contract.py",
    "lewm/oracle/go2_parallel_small_completion_search_v2.py",
    "lewm/oracle/go2_parallel_small_completion_search_v1.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_parallel_small_completion_search_v2.py",
)

BENCHMARK_CONTRACT: dict[str, Any] = {
    "benchmark_version": "V2_ONE_SHOT_EAGER_READY_POOL",
    "predecessor_benchmark_version": "V1",
    "scientific_input_change": "none",
    "sample_prefix_indices": list(SAMPLE_PREFIX_INDICES),
    "sample_prefix_count": len(SAMPLE_PREFIX_INDICES),
    "prefix_state_count": PREFIX_STATE_COUNT,
    "rotation_count": ROTATION_COUNT,
    "rotation_submission_order": list(range(ROTATION_COUNT)),
    "control_arm": (
        "the unchanged exact objective MILP C on the exact 120-state matrix, "
        "fixed warm prefix, integrality, bounds, constraints and plan-bound "
        "solver options"
    ),
    "fixed_arm": (
        "the unchanged twelve fixed-rotation MILPs F for rotations 0..11 on "
        "the same matrix, prefix, integrality, bounds, constraints and "
        "plan-bound solver options"
    ),
    "control_timing_boundary": (
        "time.monotonic immediately before scipy.optimize.milp through "
        "time.monotonic immediately after that call returns"
    ),
    "fixed_timing_boundary": (
        "time.monotonic immediately before submitting rotations 0..11 to "
        "the already-ready pool through time.monotonic immediately after all "
        "twelve future results have returned"
    ),
    "parallel_fraction_definition": "F_elapsed_seconds / C_elapsed_seconds",
    "median_gate": {
        "statistic": "median of the three exact sample F/C ratios",
        "operator": "<=",
        "threshold": MAXIMUM_PARALLEL_FRACTION,
    },
    "maximum_gate": {
        "statistic": "maximum of the three exact sample F/C ratios",
        "operator": "<=",
        "threshold": MAXIMUM_PARALLEL_FRACTION,
    },
    "overall_gate": "median_gate AND maximum_gate",
    "sample_zero_retained": True,
    "sample_substitution_permitted": False,
    "equivalence_requirement": (
        "each fixed-wave selected rotation equals the corresponding exact "
        "objective-MILP selected rotation"
    ),
    "timeout_policy": "unchanged from the bound provisional V1 search plan",
    "readiness_in_timed_samples": False,
    "startup_prewarm_in_fc_timing": False,
}

# The implementation names below are part of the source-bound protocol.  The
# V2 executor validates the same fields before it creates a pool.
EAGER_READINESS_PROCEDURE: dict[str, Any] = {
    "readiness_algorithm": (
        "spawn32_identical_manager_queue_event_barrier_scipy_highs_"
        "projected_constraint_input_no_solve_v2"
    ),
    "pool_implementation": "concurrent.futures.ProcessPoolExecutor",
    "process_start_method": "spawn",
    "worker_count": WORKER_COUNT,
    "worker_threads": WORKER_THREADS,
    "initial_pool_construction_count": 1,
    "maximum_full_pool_rebuilds_before_sample_zero": 1,
    "maximum_total_pool_constructions_before_sample_zero": 2,
    "full_pool_rebuild_condition": (
        "a worker fails before timed sample 0; discard the entire first pool "
        "and repeat the identical readiness procedure once"
    ),
    "readiness_task_scientific_role": "NON_SCIENTIFIC_READINESS_ONLY",
    "readiness_task_count": WORKER_COUNT,
    "readiness_task_identity": (
        "lewm.oracle.go2_parallel_small_completion_search_v2."
        "_v2_readiness_worker"
    ),
    "worker_initializer_identity": (
        "lewm.oracle.go2_parallel_small_completion_search_v2."
        "_v2_worker_initialise"
    ),
    "fixed_worker_identity": (
        "lewm.oracle.go2_parallel_small_completion_search_v2."
        "_v2_fixed_rotation_worker"
    ),
    "readiness_barrier": (
        "submit 32 identical readiness tasks; every task remains held until "
        "the coordinator has received 32 distinct immutable worker-instance "
        "identities, then release all tasks and require 32 successful returns"
    ),
    "distinct_worker_proof": (
        "exactly 32 distinct (pid, worker_instance_id) pairs and exactly one "
        "successful readiness return for each pair"
    ),
    "required_loaded_state": [
        "single-thread worker environment",
        "scipy.optimize.milp and HiGHS solver path",
        "exact immutable normalised 120-state benchmark input",
        "exact constraint system and base bounds",
        "exact plan-bound solver options",
        "exact immutable search-input digest, predecessor scientific input "
        "bindings digest, and V2 contract digest",
    ],
    "readiness_return": [
        "worker_pid",
        "worker_instance_id",
        "readiness_task_digest",
        "state_projection_digest",
        "source_identity_manifest_digest",
        "solver_options_digest",
        "immutable_search_input_digest",
        "predecessor_scientific_input_bindings_digest",
        "thread_environment",
        "solver_module/backend/version",
        "constraint/variable/bounds counts",
        "solver_imported=true",
        "immutable_inputs_loaded=true",
        "readiness_barrier_reached=true",
        "readiness_task_completed=true",
        "readiness_elapsed_s",
        "candidate_outcomes_consumed=false",
        "scientific_masks_accessed=false",
        "solver_call_count=0",
        "worker_readiness_digest",
    ],
    "identical_for_every_worker": True,
    "candidate_outcomes_consumed": False,
    "scientific_masks_accessed": False,
    "scientific_allocation_mutated": False,
    "scientific_candidate_mutated": False,
    "scientific_objective_mutated": False,
    "startup_cost_recorded_separately": True,
    "startup_cost_excluded_from_samples": list(SAMPLE_PREFIX_INDICES),
    "pool_destroyed_after_readiness": False,
}

LIVE_POOL_CONTINUITY_CONTRACT: dict[str, Any] = {
    "pool_identity_definition": (
        "canonical digest of the V2 contract digest, readiness-task digest, "
        "pool generation, and the sorted 32 immutable "
        "(worker_pid, worker_instance_id) rows"
    ),
    "benchmark_uses_ready_pool": True,
    "search_uses_benchmark_pool": True,
    "pool_teardown_between_readiness_and_benchmark": False,
    "pool_teardown_between_pass_and_search": False,
    "worker_restart_count_required": 0,
    "worker_replacement_after_sample_zero_permitted": False,
    "pool_reconstruction_after_sample_zero_permitted": False,
    "pool_integrity_failure_effect": (
        "issue immutable V2 failure receipt, start no new pool, and stop with "
        "no scientific search running"
    ),
}

PRE_GATE_PROHIBITIONS: dict[str, Any] = {
    "search_plan_issued": False,
    "candidate_outcomes_consumed": False,
    "scientific_masks_accessed": False,
    "completion_allocation_search_started": False,
    "branch_identities_created": False,
    "frames_rendered": 0,
    "target_latents_encoded": 0,
    "scorer_training_started": False,
    "scorer_qualification_started": False,
}

UNCHANGED_SCIENTIFIC_SEARCH_CONTRACT: dict[str, Any] = {
    "scene_and_state_pool": "unchanged",
    "benchmark_samples_and_rotations": "unchanged",
    "candidate_allocation": "unchanged",
    "milp_inputs": "unchanged",
    "milp_objective_and_lexicographic_priorities": "unchanged",
    "solver_settings_and_timeout_policy": "unchanged",
    "search_masks": "unchanged",
    "completion_enrichment_rules": "unchanged",
    "fit_and_calibration_quotas": "unchanged",
    "family_and_stratum_quotas": "unchanged",
    "scientific_selector_conditions": "unchanged",
}

ONE_SHOT_TERMINAL_POLICY: dict[str, Any] = {
    "attempt_count": 1,
    "v1_retry_permitted": False,
    "v1_delete_overwrite_or_reinterpret_permitted": False,
    "v1_mark_pass_permitted": False,
    "v2_retry_permitted": False,
    "automatic_v3_permitted": False,
    "failure_receipt_required_on_gate_or_pool_integrity_failure": True,
    "failure_stop_requirement": "nothing running",
    "failure_search_plan_issued": False,
    "failure_candidate_outcome_or_mask_operation_started": False,
    "pass_receipt_frozen_and_hashed_before_search_plan": True,
    "pass_search_plan_issued_without_approval_stop": True,
    "pass_same_live_pool_continues_directly_into_search": True,
}

_HEX = frozenset("0123456789abcdef")


class BenchmarkV2ContractError(RuntimeError):
    """The V2 pre-execution contract or a bound authority is malformed."""


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BenchmarkV2ContractError(
            "contract value is not canonical JSON") from exc


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str) and len(value) == length
        and all(character in _HEX for character in value)
    )


def _without_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in payload.items()
        if key != SELF_DIGEST_KEY
    }


def _pin_relative(root: Path, relative: str | Path, *, label: str) -> Path:
    repository = Path(root).resolve()
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise BenchmarkV2ContractError(f"{label} path is not repository-relative")
    if any(part == "sealed" or part.startswith("sealed_")
           for part in candidate_relative.parts):
        raise BenchmarkV2ContractError(f"{label} path enters sealed custody")
    candidate = repository / candidate_relative
    try:
        candidate.relative_to(repository)
    except ValueError as exc:  # pragma: no cover - resolve plus relative guards.
        raise BenchmarkV2ContractError(f"{label} path escaped repository") from exc
    cursor = repository
    for part in candidate_relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BenchmarkV2ContractError(f"{label} path is symlinked")
    return candidate


def _pin_generated(root: Path, relative: str | Path, *, label: str) -> Path:
    """Permit only the repository's one exact managed generated-root alias."""

    repository = Path(root).resolve()
    candidate_relative = Path(relative)
    if candidate_relative.is_absolute() or ".." in candidate_relative.parts:
        raise BenchmarkV2ContractError(f"{label} path is not repository-relative")
    try:
        suffix = candidate_relative.relative_to(GENERATED_ROOT_RELATIVE_PATH)
    except ValueError as exc:
        raise BenchmarkV2ContractError(
            f"{label} escaped the managed generated root") from exc
    if not suffix.parts or any(
            part == "sealed" or part == "sealed_test.json"
            or part.startswith("sealed_") for part in suffix.parts):
        raise BenchmarkV2ContractError(f"{label} path is inaccessible")
    generated_root = repository / GENERATED_ROOT_RELATIVE_PATH
    cursor = repository
    for part in GENERATED_ROOT_RELATIVE_PATH.parent.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BenchmarkV2ContractError(
                "managed generated-root parent is symlinked")
    if generated_root.is_symlink():
        raw_target = generated_root.readlink()
        target = (raw_target if raw_target.is_absolute()
                  else generated_root.parent / raw_target)
        if (target.name != generated_root.name
                or any(part == "sealed" or part == "sealed_test.json"
                       or part.startswith("sealed_") or part == ".."
                       for part in target.parts)):
            raise BenchmarkV2ContractError(
                "managed generated-root alias identity changed")
        target_absolute = target if target.is_absolute() else target.absolute()
        target_cursor = Path(target_absolute.anchor)
        for part in target_absolute.parts[1:]:
            target_cursor = target_cursor / part
            if target_cursor.is_symlink():
                raise BenchmarkV2ContractError(
                    "managed generated-root target is transitively symlinked")
        try:
            canonical_root = target.resolve(strict=True)
        except OSError as exc:
            raise BenchmarkV2ContractError(
                "managed generated-root alias target is missing") from exc
    else:
        if not generated_root.is_dir():
            raise BenchmarkV2ContractError(
                "managed generated root is unavailable")
        canonical_root = generated_root.resolve(strict=True)
    if (not canonical_root.is_dir()
            or canonical_root.name != generated_root.name):
        raise BenchmarkV2ContractError(
            "managed generated root identity changed")
    candidate = canonical_root.joinpath(*suffix.parts)
    cursor = canonical_root
    for part in suffix.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise BenchmarkV2ContractError(f"{label} path is symlinked")
    return candidate


def _require_contract_logical_path(path: Path, *, root: Path) -> Path:
    repository = Path(root).resolve()
    supplied = Path(path)
    supplied_absolute = supplied if supplied.is_absolute() else Path.cwd() / supplied
    expected_logical = repository / CONTRACT_RELATIVE_PATH
    if supplied_absolute.absolute() != expected_logical.absolute():
        raise BenchmarkV2ContractError("V2 contract path changed")
    return _pin_generated(root, CONTRACT_RELATIVE_PATH, label="V2 contract")


def _expected_absence_rows(
        registry: Sequence[tuple[str, Path, str]], *, lineage: str,
        ) -> list[dict[str, Any]]:
    return [{
        "lineage": lineage,
        "label": label,
        "path": str(relative),
        "expected_kind": kind,
        "exists": False,
        "symlink": False,
        "artifact_absent": True,
    } for label, relative, kind in registry]


def _audit_absent_outputs(
        registry: Sequence[tuple[str, Path, str]], *, lineage: str,
        root: Path,
        ) -> list[dict[str, Any]]:
    rows = _expected_absence_rows(registry, lineage=lineage)
    for row in rows:
        path = _pin_generated(root, row["path"],
                              label=f"{lineage} {row['label']}")
        if path.exists() or path.is_symlink():
            raise BenchmarkV2ContractError(
                f"{lineage} output predates V2 contract issuance: "
                f"{row['label']}")
    return rows


def _audit_issuance_output_absence(*, root: Path) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        _audit_absent_outputs(
            V2_RUNTIME_OUTPUT_PATHS, lineage="V2", root=root),
        _audit_absent_outputs(
            V1_DOWNSTREAM_OUTPUT_PATHS, lineage="V1", root=root),
    )


def validate_predecessor_scientific_input_bindings(
        payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise BenchmarkV2ContractError(
            "predecessor scientific input bindings are not a mapping")
    bindings = dict(payload)
    if (set(bindings) != SCIENTIFIC_INPUT_BINDING_KEYS
            or bindings.get("schema") != SCIENTIFIC_INPUT_BINDINGS_SCHEMA):
        raise BenchmarkV2ContractError(
            "predecessor scientific input binding surface changed")
    for key in SCIENTIFIC_INPUT_BINDING_KEYS - {
            "schema", "candidate_outcomes_consumed",
            "scientific_masks_accessed"}:
        if not _is_hex(bindings.get(key), 64):
            raise BenchmarkV2ContractError(
                f"predecessor scientific input digest is malformed: {key}")
    if (bindings.get("candidate_outcomes_consumed") is not False
            or bindings.get("scientific_masks_accessed") is not False):
        raise BenchmarkV2ContractError(
            "predecessor scientific input binding is not pre-outcome")
    return bindings


def _validate_v1_failure_payload(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise BenchmarkV2ContractError("V1 failure receipt is not a mapping")
    receipt = dict(payload)
    details = receipt.get("details")
    rows = details.get("sample_rows") if isinstance(details, Mapping) else None
    if (receipt.get("schema") != V1_FAILURE_RECEIPT_SCHEMA
            or receipt.get("benchmark_receipt_digest")
            != V1_FAILURE_RECEIPT_DIGEST
            or canonical_digest({
                key: value for key, value in receipt.items()
                if key != "benchmark_receipt_digest"
            }) != V1_FAILURE_RECEIPT_DIGEST
            or receipt.get("passes") is not False
            or receipt.get("maximum_parallel_fraction")
            != MAXIMUM_PARALLEL_FRACTION
            or receipt.get("source_binding_digest")
            != V1_FAILURE_SOURCE_BINDING_DIGEST
            or receipt.get("candidate_outcomes_consumed") is not False
            or not isinstance(details, Mapping)
            or details.get("sample_prefix_indices")
            != list(SAMPLE_PREFIX_INDICES)
            or details.get("sample_prefix_count") != len(SAMPLE_PREFIX_INDICES)
            or details.get("worker_count") != WORKER_COUNT
            or details.get("worker_threads") != WORKER_THREADS
            or details.get("state_projection_digest")
            != V1_FAILURE_STATE_PROJECTION_DIGEST
            or details.get("source_identity_manifest_digest")
            != V1_FAILURE_SOURCE_IDENTITY_MANIFEST_DIGEST
            or details.get("sample_rows_digest")
            != V1_FAILURE_SAMPLE_ROWS_DIGEST
            or canonical_digest(rows) != V1_FAILURE_SAMPLE_ROWS_DIGEST
            or details.get("median_parallel_fraction")
            != V1_FAILURE_MEDIAN_PARALLEL_FRACTION
            or details.get("maximum_parallel_fraction_observed")
            != V1_FAILURE_MAXIMUM_PARALLEL_FRACTION
            or V1_FAILURE_MEDIAN_PARALLEL_FRACTION
            > MAXIMUM_PARALLEL_FRACTION
            or V1_FAILURE_MAXIMUM_PARALLEL_FRACTION
            <= MAXIMUM_PARALLEL_FRACTION):
        raise BenchmarkV2ContractError(
            "immutable V1 failed benchmark binding changed")


def _expected_v1_failure_binding() -> dict[str, Any]:
    return {
        "path": str(V1_FAILURE_RECEIPT_RELATIVE_PATH),
        "schema": V1_FAILURE_RECEIPT_SCHEMA,
        "status_descriptor": V1_FAILURE_STATUS_DESCRIPTOR,
        "benchmark_receipt_digest": V1_FAILURE_RECEIPT_DIGEST,
        "raw_sha256": V1_FAILURE_RECEIPT_RAW_SHA256,
        "byte_count": V1_FAILURE_RECEIPT_BYTE_COUNT,
        "source_repository_commit": V1_FAILURE_SOURCE_REPOSITORY_COMMIT,
        "passes": False,
        "median_parallel_fraction": V1_FAILURE_MEDIAN_PARALLEL_FRACTION,
        "maximum_parallel_fraction": V1_FAILURE_MAXIMUM_PARALLEL_FRACTION,
        "median_gate_passes": True,
        "maximum_gate_passes": False,
        "overall_verdict": "FAIL",
        "sample_prefix_indices": list(SAMPLE_PREFIX_INDICES),
        "source_binding_digest": V1_FAILURE_SOURCE_BINDING_DIGEST,
        "state_projection_digest": V1_FAILURE_STATE_PROJECTION_DIGEST,
        "source_identity_manifest_digest":
            V1_FAILURE_SOURCE_IDENTITY_MANIFEST_DIGEST,
        "sample_rows_digest": V1_FAILURE_SAMPLE_ROWS_DIGEST,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
        "disposition": "preserve_complete_lineage_do_not_retry_or_overwrite",
    }


def load_v1_failure_binding(*, root: Path = ROOT) -> dict[str, Any]:
    path = _pin_generated(
        root, V1_FAILURE_RECEIPT_RELATIVE_PATH, label="V1 failure receipt")
    if not path.is_file() or path.is_symlink():
        raise BenchmarkV2ContractError(
            "immutable V1 failure receipt is unavailable")
    raw = path.read_bytes()
    if (len(raw) != V1_FAILURE_RECEIPT_BYTE_COUNT
            or hashlib.sha256(raw).hexdigest()
            != V1_FAILURE_RECEIPT_RAW_SHA256):
        raise BenchmarkV2ContractError(
            "immutable V1 failure receipt raw binding changed")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkV2ContractError(
            "immutable V1 failure receipt JSON is corrupt") from exc
    _validate_v1_failure_payload(payload)
    return _expected_v1_failure_binding()


def _read_source_bindings(*, root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in EXPECTED_V2_SOURCE_PATHS:
        path = _pin_relative(root, relative, label="V2 source")
        if not path.is_file() or path.is_symlink():
            raise BenchmarkV2ContractError(
                f"expected V2 source is unavailable: {relative}")
        raw = path.read_bytes()
        rows.append({
            "path": relative,
            "byte_count": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        })
    return rows


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BenchmarkV2ContractError(
            f"git source-custody check failed: {' '.join(args)}") from exc


def _clean_source_commit(*, root: Path) -> str:
    commit = _git("rev-parse", "HEAD", root=root).decode().strip()
    if not _is_hex(commit, 40):
        raise BenchmarkV2ContractError("source commit is malformed")
    status = _git(
        "status", "--porcelain=v1", "--untracked-files=all", root=root)
    if status:
        raise BenchmarkV2ContractError(
            "V2 contract requires a clean source commit")
    tracked = _git(
        "ls-files", "--error-unmatch", "--", *EXPECTED_V2_SOURCE_PATHS,
        root=root,
    ).decode().splitlines()
    if (len(tracked) != len(EXPECTED_V2_SOURCE_PATHS)
            or set(tracked) != set(EXPECTED_V2_SOURCE_PATHS)):
        raise BenchmarkV2ContractError(
            "expected V2 sources are not frozen at the source commit")
    for relative in EXPECTED_V2_SOURCE_PATHS:
        committed_raw = _git("show", f"HEAD:{relative}", root=root)
        live_raw = _pin_relative(root, relative, label="V2 source").read_bytes()
        if committed_raw != live_raw:
            raise BenchmarkV2ContractError(
                f"live V2 source differs from commit: {relative}")
    return commit


def _validate_source_binding_rows(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_V2_SOURCE_PATHS):
        raise BenchmarkV2ContractError("V2 source binding coverage changed")
    validated: list[dict[str, Any]] = []
    for expected_path, raw_row in zip(
            EXPECTED_V2_SOURCE_PATHS, rows, strict=True):
        if not isinstance(raw_row, Mapping):
            raise BenchmarkV2ContractError("V2 source binding is malformed")
        row = dict(raw_row)
        if (set(row) != {"path", "byte_count", "sha256"}
                or row.get("path") != expected_path
                or isinstance(row.get("byte_count"), bool)
                or not isinstance(row.get("byte_count"), int)
                or row["byte_count"] <= 0
                or not _is_hex(row.get("sha256"), 64)):
            raise BenchmarkV2ContractError("V2 source binding changed")
        validated.append(row)
    return validated


def build_contract(
        *, source_repository_commit: str,
        source_bindings: Sequence[Mapping[str, Any]],
        v1_failure_binding: Mapping[str, Any],
        predecessor_scientific_input_bindings: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Build the exact self-digested contract without reading or writing files."""

    if not _is_hex(source_repository_commit, 40):
        raise BenchmarkV2ContractError("source repository commit is malformed")
    rows = _validate_source_binding_rows(list(source_bindings))
    predecessor = validate_predecessor_scientific_input_bindings(
        predecessor_scientific_input_bindings)
    v1_binding = dict(v1_failure_binding)
    # The exact expected V1 envelope is reconstructed without file access.
    expected_v1 = _expected_v1_failure_binding()
    if v1_binding != expected_v1:
        raise BenchmarkV2ContractError("V1 failure envelope changed")
    runtime_absence = _expected_absence_rows(
        V2_RUNTIME_OUTPUT_PATHS, lineage="V2")
    v1_downstream_absence = _expected_absence_rows(
        V1_DOWNSTREAM_OUTPUT_PATHS, lineage="V1")
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "complete": True,
        "source_repository_commit": source_repository_commit,
        "source_bindings": rows,
        "source_binding_set_digest": canonical_digest(rows),
        "expected_v2_source_paths": list(EXPECTED_V2_SOURCE_PATHS),
        "immutable_v1_failure_receipt": v1_binding,
        "predecessor_scientific_input_bindings": predecessor,
        "predecessor_scientific_input_bindings_digest":
            canonical_digest(predecessor),
        "runtime_outputs_absent_at_issue": runtime_absence,
        "runtime_outputs_absent_at_issue_digest":
            canonical_digest(runtime_absence),
        "v1_downstream_outputs_absent_at_issue": v1_downstream_absence,
        "v1_downstream_outputs_absent_at_issue_digest":
            canonical_digest(v1_downstream_absence),
        "benchmark_contract": copy.deepcopy(BENCHMARK_CONTRACT),
        "eager_readiness_procedure": copy.deepcopy(EAGER_READINESS_PROCEDURE),
        "live_pool_continuity_contract":
            copy.deepcopy(LIVE_POOL_CONTINUITY_CONTRACT),
        "pre_gate_prohibitions": copy.deepcopy(PRE_GATE_PROHIBITIONS),
        "unchanged_scientific_search_contract":
            copy.deepcopy(UNCHANGED_SCIENTIFIC_SEARCH_CONTRACT),
        "one_shot_terminal_policy": copy.deepcopy(ONE_SHOT_TERMINAL_POLICY),
        "candidate_outcomes_consumed_at_issue": False,
        "scientific_masks_accessed_at_issue": False,
        "worker_pool_constructed_at_issue": False,
        "scientific_search_plan_issued_at_issue": False,
    }
    payload[SELF_DIGEST_KEY] = canonical_digest(payload)
    return payload


_CONTRACT_KEYS = frozenset({
    "schema", "status", "complete", "source_repository_commit",
    "source_bindings", "source_binding_set_digest",
    "expected_v2_source_paths", "immutable_v1_failure_receipt",
    "predecessor_scientific_input_bindings",
    "predecessor_scientific_input_bindings_digest",
    "runtime_outputs_absent_at_issue",
    "runtime_outputs_absent_at_issue_digest",
    "v1_downstream_outputs_absent_at_issue",
    "v1_downstream_outputs_absent_at_issue_digest", "benchmark_contract",
    "eager_readiness_procedure", "live_pool_continuity_contract",
    "pre_gate_prohibitions", "unchanged_scientific_search_contract",
    "one_shot_terminal_policy", "candidate_outcomes_consumed_at_issue",
    "scientific_masks_accessed_at_issue", "worker_pool_constructed_at_issue",
    "scientific_search_plan_issued_at_issue", SELF_DIGEST_KEY,
})


def validate_contract(
        payload: Mapping[str, Any], *,
        expected_predecessor_scientific_input_bindings: Mapping[str, Any],
        expected_source_repository_commit: str | None = None,
        root: Path = ROOT,
        validate_live_authorities: bool = True,
        ) -> dict[str, Any]:
    """Validate exact schema, self digest, lineage, and optionally live files."""

    if not isinstance(payload, Mapping):
        raise BenchmarkV2ContractError("V2 contract is not a mapping")
    contract = dict(payload)
    predecessor = validate_predecessor_scientific_input_bindings(
        expected_predecessor_scientific_input_bindings)
    rows = _validate_source_binding_rows(contract.get("source_bindings"))
    runtime_absence = _expected_absence_rows(
        V2_RUNTIME_OUTPUT_PATHS, lineage="V2")
    v1_downstream_absence = _expected_absence_rows(
        V1_DOWNSTREAM_OUTPUT_PATHS, lineage="V1")
    commit = contract.get("source_repository_commit")
    if (set(contract) != _CONTRACT_KEYS
            or contract.get("schema") != SCHEMA
            or contract.get("status") != STATUS
            or contract.get("complete") is not True
            or not _is_hex(commit, 40)
            or (expected_source_repository_commit is not None
                and commit != expected_source_repository_commit)
            or contract.get("source_binding_set_digest")
            != canonical_digest(rows)
            or contract.get("expected_v2_source_paths")
            != list(EXPECTED_V2_SOURCE_PATHS)
            or contract.get("predecessor_scientific_input_bindings")
            != predecessor
            or contract.get("predecessor_scientific_input_bindings_digest")
            != canonical_digest(predecessor)
            or contract.get("runtime_outputs_absent_at_issue")
            != runtime_absence
            or contract.get("runtime_outputs_absent_at_issue_digest")
            != canonical_digest(runtime_absence)
            or contract.get("v1_downstream_outputs_absent_at_issue")
            != v1_downstream_absence
            or contract.get("v1_downstream_outputs_absent_at_issue_digest")
            != canonical_digest(v1_downstream_absence)
            or contract.get("benchmark_contract") != BENCHMARK_CONTRACT
            or contract.get("eager_readiness_procedure")
            != EAGER_READINESS_PROCEDURE
            or contract.get("live_pool_continuity_contract")
            != LIVE_POOL_CONTINUITY_CONTRACT
            or contract.get("pre_gate_prohibitions") != PRE_GATE_PROHIBITIONS
            or contract.get("unchanged_scientific_search_contract")
            != UNCHANGED_SCIENTIFIC_SEARCH_CONTRACT
            or contract.get("one_shot_terminal_policy")
            != ONE_SHOT_TERMINAL_POLICY
            or contract.get("candidate_outcomes_consumed_at_issue") is not False
            or contract.get("scientific_masks_accessed_at_issue") is not False
            or contract.get("worker_pool_constructed_at_issue") is not False
            or contract.get("scientific_search_plan_issued_at_issue") is not False
            or contract.get(SELF_DIGEST_KEY)
            != canonical_digest(_without_digest(contract))):
        raise BenchmarkV2ContractError("V2 benchmark contract binding changed")
    # Reuse build_contract's exact immutable V1 envelope validation.
    rebuilt = build_contract(
        source_repository_commit=str(commit), source_bindings=rows,
        v1_failure_binding=contract.get("immutable_v1_failure_receipt", {}),
        predecessor_scientific_input_bindings=predecessor)
    if rebuilt != contract:
        raise BenchmarkV2ContractError("V2 benchmark contract is not exact")
    if validate_live_authorities:
        clean_commit = _clean_source_commit(root=root)
        if clean_commit != commit:
            raise BenchmarkV2ContractError("live clean source commit changed")
        if _read_source_bindings(root=root) != rows:
            raise BenchmarkV2ContractError("live V2 source binding changed")
        if (load_v1_failure_binding(root=root)
                != contract["immutable_v1_failure_receipt"]):
            raise BenchmarkV2ContractError("live V1 failure lineage changed")
    return contract


def _load_json_exact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise BenchmarkV2ContractError("V2 contract is unavailable")
    try:
        payload = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkV2ContractError("V2 contract JSON is corrupt") from exc
    if not isinstance(payload, dict):
        raise BenchmarkV2ContractError("V2 contract JSON root changed")
    return payload


def load_contract(
        path: Path, *,
        expected_predecessor_scientific_input_bindings: Mapping[str, Any],
        expected_source_repository_commit: str | None = None,
        root: Path = ROOT,
        validate_live_authorities: bool = True,
        ) -> dict[str, Any]:
    expected_path = _require_contract_logical_path(path, root=root)
    return validate_contract(
        _load_json_exact(expected_path),
        expected_predecessor_scientific_input_bindings=
            expected_predecessor_scientific_input_bindings,
        expected_source_repository_commit=expected_source_repository_commit,
        root=root, validate_live_authorities=validate_live_authorities)


def issue_contract(
        path: Path, *,
        predecessor_scientific_input_bindings: Mapping[str, Any],
        source_repository_commit: str | None = None,
        root: Path = ROOT,
        ) -> dict[str, Any]:
    """Issue the sole V2 contract, before any caller constructs a worker pool."""

    expected_path = _require_contract_logical_path(path, root=root)
    if not expected_path.parent.is_dir() or expected_path.parent.is_symlink():
        raise BenchmarkV2ContractError(
            "dedicated V2 contract parent is unavailable")
    if expected_path.exists() or expected_path.is_symlink():
        return load_contract(
            expected_path,
            expected_predecessor_scientific_input_bindings=
                predecessor_scientific_input_bindings,
            expected_source_repository_commit=source_repository_commit,
            root=root)
    initial_v2_absence, initial_v1_absence = \
        _audit_issuance_output_absence(root=root)
    live_commit = _clean_source_commit(root=root)
    if (source_repository_commit is not None
            and source_repository_commit != live_commit):
        raise BenchmarkV2ContractError("requested source commit is not live")
    expected = build_contract(
        source_repository_commit=live_commit,
        source_bindings=_read_source_bindings(root=root),
        v1_failure_binding=load_v1_failure_binding(root=root),
        predecessor_scientific_input_bindings=
            predecessor_scientific_input_bindings,
    )
    raw = json.dumps(
        expected, indent=2, sort_keys=True, ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    final_v2_absence, final_v1_absence = \
        _audit_issuance_output_absence(root=root)
    if (final_v2_absence != initial_v2_absence
            or final_v1_absence != initial_v1_absence
            or final_v2_absence
            != expected["runtime_outputs_absent_at_issue"]
            or final_v1_absence
            != expected["v1_downstream_outputs_absent_at_issue"]):
        raise BenchmarkV2ContractError(
            "issuance-time output absence binding changed before install")
    try:
        descriptor = os.open(expected_path, flags, 0o444)
    except FileExistsError:
        return load_contract(
            expected_path,
            expected_predecessor_scientific_input_bindings=
                predecessor_scientific_input_bindings,
            expected_source_repository_commit=live_commit, root=root)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        # Never delete or replace a partially issued path; fail closed.
        raise
    return load_contract(
        expected_path,
        expected_predecessor_scientific_input_bindings=
            predecessor_scientific_input_bindings,
        expected_source_repository_commit=live_commit, root=root)


__all__ = [
    "BENCHMARK_CONTRACT", "BenchmarkV2ContractError",
    "CONTRACT_RELATIVE_PATH", "EAGER_READINESS_PROCEDURE",
    "EXPECTED_V2_SOURCE_PATHS", "LIVE_POOL_CONTINUITY_CONTRACT",
    "MAXIMUM_PARALLEL_FRACTION", "ONE_SHOT_TERMINAL_POLICY",
    "PRE_GATE_PROHIBITIONS", "SAMPLE_PREFIX_INDICES", "SCHEMA",
    "SCIENTIFIC_INPUT_BINDINGS_SCHEMA", "SELF_DIGEST_KEY", "STATUS",
    "UNCHANGED_SCIENTIFIC_SEARCH_CONTRACT",
    "V1_DOWNSTREAM_OUTPUT_PATHS",
    "V1_FAILURE_RECEIPT_DIGEST", "V1_FAILURE_SOURCE_REPOSITORY_COMMIT",
    "V1_FAILURE_STATUS_DESCRIPTOR",
    "V2_RUNTIME_OUTPUT_PATHS", "WORKER_COUNT", "build_contract",
    "canonical_digest", "issue_contract",
    "load_contract", "load_v1_failure_binding", "validate_contract",
    "validate_predecessor_scientific_input_bindings",
]
