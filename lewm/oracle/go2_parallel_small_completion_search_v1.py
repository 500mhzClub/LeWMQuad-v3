"""Outcome-blind parallel executor for the scorer-fit small-last search.

This module changes only *how* the frozen lexicographic search is executed.
It does not change candidate ordering, state identities, the allocation
constraints, the lexicographically-smallest allocation rule, or the exact
completion-mask predicate.  The bounded objective-wave certificate proves the
canonical earliest winner without invoking the predecessor's duplicate,
unbounded allocation re-solve.

All durable records are operational pre-outcome evidence.  They deliberately
have no API for reading branch outcomes, frames, latents, oracle labels,
predictor checkpoints, or scorer weights.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from lewm.oracle import go2_candidate_allocation_v1_2 as ALLOC


SCHEMA_VERSION = "go2_parallel_small_completion_search_v1"
SEARCH_PLAN_SCHEMA = f"{SCHEMA_VERSION}_plan"
RANK_RECEIPT_SCHEMA = f"{SCHEMA_VERSION}_rank_receipt"
WAVE_RECEIPT_SCHEMA = f"{SCHEMA_VERSION}_prefix_wave_receipt"
COORDINATOR_RECEIPT_SCHEMA = f"{SCHEMA_VERSION}_coordinator_receipt"
BENCHMARK_RECEIPT_SCHEMA = f"{SCHEMA_VERSION}_benchmark_receipt"
SEARCH_RESULT_SCHEMA = f"{SCHEMA_VERSION}_terminal_search_receipt"
OBJECTIVE_WAVE_RECEIPT_SCHEMA = f"{SCHEMA_VERSION}_objective_wave_receipt"
WINNER_VALIDATION_RECEIPT_SCHEMA = (
    f"{SCHEMA_VERSION}_winner_objective_validation_receipt")
ALGORITHM_VERSION = (
    "itertools_lexicographic_rank_window_fixed_rotation_feasibility_v1"
)

ROTATION_COUNT = ALLOC.CANDIDATE_COUNT
COMBINATION_SIZE = 5
PREFIX_STATE_COUNT = 120
RANK_CLASSIFICATIONS = frozenset(
    ("NONPASS", "ALLOCATOR_INFEASIBLE", "MASK_FAIL", "PASS", "FATAL")
)
ORDINARY_NONPASS = frozenset(
    ("NONPASS", "ALLOCATOR_INFEASIBLE", "MASK_FAIL")
)
ROTATION_STATUSES = frozenset(("FEASIBLE", "INFEASIBLE", "FATAL"))
WAVE_STATUSES = frozenset(("SELECTED", "ALLOCATOR_INFEASIBLE", "FATAL"))
THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "HIGHS_THREADS": "1",
}
DEFAULT_MILP_TIME_LIMIT_S = 7_200.0
ALLOCATOR_IDENTITY_FIELDS = (
    "state_id", "state_identity_digest", "family", "stratum", "split_role",
    "goal_type",
)

_HEX = frozenset("0123456789abcdef")
_WORKER_THREAD_LIMITER: Any | None = None


class ParallelSearchError(RuntimeError):
    """The operational search proof is malformed or cannot be continued."""


class ParallelSearchFatal(ParallelSearchError):
    """A solver/evidence defect that the serial search would propagate."""


class _SpeculativeRankStopped(ParallelSearchError):
    """An operational rank lane stopped after the canonical frontier closed."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def project_allocator_identity_states(
        states: Sequence[Mapping[str, Any]],
        ) -> list[dict[str, str]]:
    """Return the canonical exact six-key allocator identity projection.

    Scientific selector rows contain additional pre-outcome fields required by
    mask callbacks.  The frozen allocator contract accepts exactly six fields,
    so every allocator-facing path must pass through this projection first.
    Its return value is identical to normalizing an already-projected legacy
    six-key input, including canonical digest ordering.
    """

    if isinstance(states, (str, bytes)) or not isinstance(states, Sequence):
        raise ParallelSearchError("allocator states must be a sequence")
    projected: list[dict[str, Any]] = []
    for index, state in enumerate(states):
        if not isinstance(state, Mapping):
            raise ParallelSearchError(
                f"allocator state {index} is not a mapping")
        missing = [field for field in ALLOCATOR_IDENTITY_FIELDS
                   if field not in state]
        if missing:
            raise ParallelSearchError(
                f"allocator state {index} lacks identity fields {missing}")
        projected.append({field: state[field]
                          for field in ALLOCATOR_IDENTITY_FIELDS})
    return ALLOC._normalise_identity_states(projected)


def _is_digest(value: Any, *, length: int = 64) -> bool:
    return (
        isinstance(value, str) and len(value) == length
        and all(character in _HEX for character in value)
    )


def _without_digest(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {name: value for name, value in payload.items() if name != key}


def combination_count(n: int, k: int = COMBINATION_SIZE) -> int:
    if isinstance(n, bool) or isinstance(k, bool):
        raise TypeError("combination dimensions must be integers")
    if not isinstance(n, int) or not isinstance(k, int):
        raise TypeError("combination dimensions must be integers")
    if n < 0 or k < 0 or k > n:
        raise ValueError("invalid combination dimensions")
    return math.comb(n, k)


def unrank_combination(rank: int, n: int,
                       k: int = COMBINATION_SIZE) -> tuple[int, ...]:
    """Return exactly rank ``rank`` from ``itertools.combinations(range(n), k)``."""

    total = combination_count(n, k)
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("combination rank must be an integer")
    if not 0 <= rank < total:
        raise ValueError("combination rank is outside the exact domain")
    remaining_rank = rank
    result: list[int] = []
    lower = 0
    for position in range(k):
        remaining_positions = k - position - 1
        for value in range(lower, n - remaining_positions):
            suffixes = math.comb(n - value - 1, remaining_positions)
            if remaining_rank < suffixes:
                result.append(value)
                lower = value + 1
                break
            remaining_rank -= suffixes
        else:  # pragma: no cover - guarded by exact arithmetic above.
            raise ParallelSearchError("combination unranking arithmetic failed")
    return tuple(result)


def rank_combination(combination: Sequence[int], n: int,
                     k: int = COMBINATION_SIZE) -> int:
    """Inverse of :func:`unrank_combination` in itertools lexical order."""

    combination_count(n, k)
    if isinstance(combination, (str, bytes)) or not isinstance(
            combination, Sequence):
        raise TypeError("combination must be an integer sequence")
    values = tuple(combination)
    if (len(values) != k or any(isinstance(value, bool)
                                or not isinstance(value, int)
                                for value in values)
            or tuple(sorted(set(values))) != values
            or any(value < 0 or value >= n for value in values)):
        raise ValueError("combination is not a canonical increasing k-tuple")
    rank = 0
    lower = 0
    for position, chosen in enumerate(values):
        remaining_positions = k - position - 1
        for value in range(lower, chosen):
            rank += math.comb(n - value - 1, remaining_positions)
        lower = chosen + 1
    return rank


def contiguous_partitions(total_rank_count: int,
                          worker_count: int) -> list[tuple[int, int]]:
    """Return complete, disjoint, balanced contiguous half-open ranges."""

    if (isinstance(total_rank_count, bool)
            or not isinstance(total_rank_count, int)
            or total_rank_count < 0):
        raise ValueError("total_rank_count must be a nonnegative integer")
    if (isinstance(worker_count, bool) or not isinstance(worker_count, int)
            or worker_count <= 0):
        raise ValueError("worker_count must be a positive integer")
    quotient, remainder = divmod(total_rank_count, worker_count)
    result: list[tuple[int, int]] = []
    cursor = 0
    for worker_index in range(worker_count):
        width = quotient + (1 if worker_index < remainder else 0)
        result.append((cursor, cursor + width))
        cursor += width
    assert cursor == total_rank_count
    return result


def build_search_plan(
        *, candidate_scene_ids: Sequence[str], combination_size: int,
        worker_count: int, source_repository_commit: str,
        clean_source_launch_receipt_digest: str,
        state_selector_amendment_digest: str,
        candidate_allocation_amendment_digest: str,
        fixed_state_projection_digest: str,
        resolver_cursor_scene_id: str,
        solver_identity: Mapping[str, Any],
        solver_options: Mapping[str, Any],
        bindings: Mapping[str, Any] | None = None,
        measured_benchmark_receipt_digest: str | None = None,
        active_rank_window: int | None = None,
        ) -> dict[str, Any]:
    """Build the immutable, outcome-free execution plan."""

    scenes = [str(value) for value in candidate_scene_ids]
    total = combination_count(len(scenes), combination_size)
    window = (max(1, math.ceil(worker_count / ROTATION_COUNT))
              if active_rank_window is None else active_rank_window)
    bound_solver_options = dict(solver_options)
    bound_solver_options.setdefault("time_limit", DEFAULT_MILP_TIME_LIMIT_S)
    payload: dict[str, Any] = {
        "schema": SEARCH_PLAN_SCHEMA,
        "algorithm_version": ALGORITHM_VERSION,
        "candidate_scene_ids": scenes,
        "candidate_pool_count": len(scenes),
        "candidate_pool_scene_ids_digest": canonical_digest(scenes),
        "combination_size": combination_size,
        "total_rank_count": total,
        "rank_partitions": [list(row) for row in
                            contiguous_partitions(total, worker_count)],
        "worker_count": worker_count,
        "active_rank_window": window,
        "source_repository_commit": source_repository_commit,
        "clean_source_launch_receipt_digest":
            clean_source_launch_receipt_digest,
        "state_selector_amendment_digest": state_selector_amendment_digest,
        "candidate_allocation_amendment_digest":
            candidate_allocation_amendment_digest,
        "candidate_allocation_contract_digest":
            ALLOC.allocation_contract_digest(),
        "fixed_state_projection_digest": fixed_state_projection_digest,
        "resolver_cursor_scene_id": resolver_cursor_scene_id,
        "solver_identity": dict(solver_identity),
        "solver_options": bound_solver_options,
        "thread_environment": dict(THREAD_ENVIRONMENT),
        "measured_benchmark_receipt_digest":
            measured_benchmark_receipt_digest,
        "bindings": {} if bindings is None else dict(bindings),
        "candidate_outcomes_consumed": False,
    }
    payload["search_plan_digest"] = canonical_digest(payload)
    validate_search_plan(payload)
    return payload


def validate_search_plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("search plan must be a mapping")
    plan = dict(payload)
    expected_keys = {
        "schema", "algorithm_version", "candidate_scene_ids",
        "candidate_pool_count", "candidate_pool_scene_ids_digest",
        "combination_size", "total_rank_count", "rank_partitions",
        "worker_count", "active_rank_window", "source_repository_commit",
        "clean_source_launch_receipt_digest",
        "state_selector_amendment_digest",
        "candidate_allocation_amendment_digest",
        "candidate_allocation_contract_digest",
        "fixed_state_projection_digest", "resolver_cursor_scene_id",
        "solver_identity", "solver_options", "thread_environment",
        "measured_benchmark_receipt_digest", "bindings",
        "candidate_outcomes_consumed", "search_plan_digest",
    }
    if set(plan) != expected_keys:
        raise ValueError("search plan key surface changed")
    scenes = plan.get("candidate_scene_ids")
    if (plan.get("schema") != SEARCH_PLAN_SCHEMA
            or plan.get("algorithm_version") != ALGORITHM_VERSION
            or not isinstance(scenes, list)
            or len(scenes) < COMBINATION_SIZE
            or any(not isinstance(value, str) or not value for value in scenes)
            or scenes != sorted(scenes)
            or len(set(scenes)) != len(scenes)
            or plan.get("candidate_pool_count") != len(scenes)
            or plan.get("candidate_pool_scene_ids_digest")
            != canonical_digest(scenes)
            or plan.get("combination_size") != COMBINATION_SIZE
            or plan.get("total_rank_count")
            != combination_count(len(scenes), COMBINATION_SIZE)
            or plan.get("rank_partitions") != [list(row) for row in
                contiguous_partitions(plan["total_rank_count"],
                                      plan.get("worker_count"))]
            or not isinstance(plan.get("active_rank_window"), int)
            or not 1 <= plan["active_rank_window"] <= plan["worker_count"]
            or plan.get("candidate_outcomes_consumed") is not False):
        raise ValueError("search plan candidate/rank contract changed")
    if (not _is_digest(plan.get("source_repository_commit"), length=40)
            or any(not _is_digest(plan.get(key)) for key in (
                "clean_source_launch_receipt_digest",
                "state_selector_amendment_digest",
                "candidate_allocation_amendment_digest",
                "candidate_allocation_contract_digest",
                "fixed_state_projection_digest"))
            or plan.get("candidate_allocation_contract_digest")
            != ALLOC.allocation_contract_digest()
            or not isinstance(plan.get("resolver_cursor_scene_id"), str)
            or not isinstance(plan.get("solver_identity"), dict)
            or not isinstance(plan.get("solver_options"), dict)
            or plan["solver_options"].get("threads") != 1
            or not isinstance(plan["solver_options"].get("time_limit"),
                              (int, float))
            or plan["solver_options"]["time_limit"] <= 0
            or plan.get("thread_environment") != THREAD_ENVIRONMENT
            or not isinstance(plan.get("bindings"), dict)):
        raise ValueError("search plan source/solver binding changed")
    benchmark = plan.get("measured_benchmark_receipt_digest")
    if benchmark is not None and not _is_digest(benchmark):
        raise ValueError("search plan benchmark digest is invalid")
    if plan.get("search_plan_digest") != canonical_digest(
            _without_digest(plan, "search_plan_digest")):
        raise ValueError("search plan digest mismatch")
    return plan


class OrderedFrontier:
    """Commit out-of-order rank results with exact serial semantics."""

    def __init__(self, *, total_rank_count: int) -> None:
        if total_rank_count < 0:
            raise ValueError("total rank count cannot be negative")
        self.total_rank_count = total_rank_count
        self.committed_rank_count = 0
        self.terminal = total_rank_count == 0
        self._pending: dict[int, str] = {}

    def record(self, *, rank: int, classification: str) -> dict[str, Any] | None:
        if self.terminal:
            raise ValueError("frontier is already terminal")
        if (not isinstance(rank, int) or not 0 <= rank < self.total_rank_count
                or classification not in RANK_CLASSIFICATIONS):
            raise ValueError("invalid rank classification")
        if rank < self.committed_rank_count or rank in self._pending:
            raise ValueError("duplicate rank classification")
        self._pending[rank] = classification
        event: dict[str, Any] | None = None
        while self.committed_rank_count in self._pending:
            current = self.committed_rank_count
            value = self._pending.pop(current)
            self.committed_rank_count += 1
            if value == "PASS":
                self.terminal = True
                event = {"status": "PASS", "rank": current}
                break
            if value == "FATAL":
                self.terminal = True
                event = {"status": "FATAL", "rank": current}
                break
            if value not in ORDINARY_NONPASS:
                raise ValueError("unknown ordinary rank classification")
        if (not self.terminal
                and self.committed_rank_count == self.total_rank_count):
            self.terminal = True
            event = {"status": "EXHAUSTED",
                     "rank_count": self.total_rank_count}
        return event


def fixed_rotation_lexicographic_rotations(
        *, rotation_count: int,
        solve_rotation: Callable[[int], str],
        completion_order: Sequence[int] | None = None,
        ) -> dict[str, Any]:
    """Pure ordered reduction for one fixed-prefix feasibility wave."""

    if rotation_count <= 0:
        raise ValueError("rotation_count must be positive")
    order = (list(range(rotation_count)) if completion_order is None
             else list(completion_order))
    if sorted(order) != list(range(rotation_count)):
        raise ValueError("completion order is not an exact rotation permutation")
    statuses: list[str | None] = [None] * rotation_count
    for rotation in order:
        status = solve_rotation(rotation)
        if status not in ROTATION_STATUSES:
            raise RuntimeError(f"rotation {rotation} returned {status!r}")
        statuses[rotation] = status
    assert all(value is not None for value in statuses)
    for rotation, status in enumerate(statuses):
        if status == "FATAL":
            raise RuntimeError(f"rotation {rotation} returned FATAL")
        if status == "FEASIBLE":
            result = {
                "selected_rotation": rotation,
                "statuses": statuses,
            }
            result["allocation_bytes"] = _json_bytes(result)
            return result
    raise ALLOC.CandidateAllocationInfeasible(
        "no fixed rotation admits a feasible allocation completion")


def _guard_regular_path(path: Path, *, for_write: bool = False) -> Path:
    raw = Path(path)
    if any(part in ("..", "sealed", "sealed_test.json")
           or part.startswith("sealed_") for part in raw.parts):
        raise RuntimeError("receipt custody path is inaccessible")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    cursor = Path(absolute.anchor)
    for part in absolute.parts[1:-1]:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError("symlinked receipt custody path is inaccessible")
    if absolute.is_symlink():
        raise RuntimeError("symlinked receipt custody path is inaccessible")
    if not for_write and not absolute.is_file():
        raise RuntimeError("receipt is not a regular file")
    return absolute


def _load_json(path: Path) -> dict[str, Any]:
    pinned = _guard_regular_path(path)
    try:
        payload = json.loads(pinned.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("receipt JSON is corrupt or invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("receipt JSON root is not a mapping")
    return payload


def _atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = _guard_regular_path(path, for_write=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    _guard_regular_path(target, for_write=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        with os.fdopen(descriptor, "w") as stream:
            descriptor = -1
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except FileExistsError:
            pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _signed(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(key, None)
    result[key] = canonical_digest(result)
    return result


def _validate_rank_receipt(
        payload: Mapping[str, Any], plan: Mapping[str, Any], *,
        expected_rank: int | None = None) -> dict[str, Any]:
    receipt = dict(payload)
    validate_search_plan(plan)
    required = {
        "schema", "search_plan_digest", "rank", "combination_indices",
        "selected_scene_ids", "projection_digest",
        "source_identity_manifest_digest", "completed_prefix_wave_count",
        "selected_rotations", "provisional_allocation_manifest_digest",
        "provisional_candidate_assignment_set_digest", "classification",
        "candidate_outcomes_consumed", "rank_receipt_digest",
    }
    if set(receipt) != required:
        raise RuntimeError("rank receipt key surface changed")
    rank = receipt.get("rank")
    if (receipt.get("schema") != RANK_RECEIPT_SCHEMA
            or receipt.get("search_plan_digest") != plan["search_plan_digest"]
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 <= rank < plan["total_rank_count"]
            or (expected_rank is not None
                and (isinstance(expected_rank, bool)
                     or not isinstance(expected_rank, int)
                     or rank != expected_rank))
            or receipt.get("combination_indices") != list(unrank_combination(
                rank, plan["candidate_pool_count"], plan["combination_size"]))
            or receipt.get("selected_scene_ids") != [
                plan["candidate_scene_ids"][index]
                for index in receipt["combination_indices"]]
            or not _is_digest(receipt.get("projection_digest"))
            or not _is_digest(receipt.get("source_identity_manifest_digest"))
            or isinstance(receipt.get("completed_prefix_wave_count"), bool)
            or not isinstance(receipt.get("completed_prefix_wave_count"), int)
            or not isinstance(receipt.get("selected_rotations"), list)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or not 0 <= value < ROTATION_COUNT
                   for value in receipt["selected_rotations"])
            or receipt.get("classification") not in RANK_CLASSIFICATIONS
            or receipt.get("candidate_outcomes_consumed") is not False
            or receipt.get("rank_receipt_digest") != canonical_digest(
                _without_digest(receipt, "rank_receipt_digest"))):
        raise RuntimeError("rank receipt digest/combination/outcome binding changed")
    completed = receipt["completed_prefix_wave_count"]
    rotations = receipt["selected_rotations"]
    classification = receipt["classification"]
    allocation_digest = receipt["provisional_allocation_manifest_digest"]
    assignment_digest = receipt["provisional_candidate_assignment_set_digest"]
    if (completed != len(rotations) or not 0 <= completed <= PREFIX_STATE_COUNT):
        raise RuntimeError("rank receipt prefix-wave/rotation count changed")
    if classification in ("PASS", "MASK_FAIL", "NONPASS"):
        if (completed != PREFIX_STATE_COUNT
                or not _is_digest(allocation_digest)
                or not _is_digest(assignment_digest)):
            raise RuntimeError("completed rank lacks exact allocation evidence")
    elif classification == "ALLOCATOR_INFEASIBLE":
        if (completed != 0 or rotations
                or allocation_digest is not None or assignment_digest is not None):
            raise RuntimeError("allocator-infeasible rank evidence changed")
    elif classification == "FATAL":
        if (allocation_digest is not None or assignment_digest is not None
                or completed >= PREFIX_STATE_COUNT):
            raise RuntimeError("fatal rank evidence changed")
    return receipt


def write_rank_receipt(path: Path, payload: Mapping[str, Any], *,
                       search_plan: Mapping[str, Any],
                       validate: bool = True) -> dict[str, Any]:
    signed = _signed(payload, "rank_receipt_digest")
    if validate:
        signed = _validate_rank_receipt(signed, search_plan)
    if Path(path).exists():
        existing = load_rank_receipt(
            path, search_plan=search_plan, expected_rank=signed["rank"])
        if existing != signed:
            raise RuntimeError("refusing to overwrite a different rank receipt")
        return existing
    _atomic_create_json(path, signed)
    if not Path(path).exists():
        raise RuntimeError("atomic rank receipt creation lost a race")
    if validate:
        return load_rank_receipt(
            path, search_plan=search_plan, expected_rank=signed["rank"])
    return signed


def load_rank_receipt(path: Path, *, search_plan: Mapping[str, Any],
                      expected_rank: int) -> dict[str, Any]:
    return _validate_rank_receipt(
        _load_json(path), search_plan, expected_rank=expected_rank)


def _validate_coordinator_receipt(payload: Mapping[str, Any],
                                  plan: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    required = {
        "schema", "search_plan_digest", "committed_frontier_rank",
        "terminal_status", "candidate_outcomes_consumed",
        "coordinator_receipt_digest",
    }
    if (set(receipt) != required
            or receipt.get("schema") != COORDINATOR_RECEIPT_SCHEMA
            or receipt.get("search_plan_digest") != plan["search_plan_digest"]
            or isinstance(receipt.get("committed_frontier_rank"), bool)
            or not isinstance(receipt.get("committed_frontier_rank"), int)
            or not 0 <= receipt["committed_frontier_rank"] <= plan[
                "total_rank_count"]
            or receipt.get("terminal_status") not in
            (None, "PASS", "FATAL", "EXHAUSTED")
            or receipt.get("candidate_outcomes_consumed") is not False
            or receipt.get("coordinator_receipt_digest") != canonical_digest(
                _without_digest(receipt, "coordinator_receipt_digest"))):
        raise RuntimeError("coordinator receipt plan/digest binding changed")
    return receipt


def write_coordinator_receipt(path: Path, payload: Mapping[str, Any], *,
                              search_plan: Mapping[str, Any]) -> dict[str, Any]:
    signed = _signed(payload, "coordinator_receipt_digest")
    signed = _validate_coordinator_receipt(signed, search_plan)
    if Path(path).exists():
        existing = load_coordinator_receipt(path, search_plan=search_plan)
        if existing != signed:
            raise RuntimeError("refusing to overwrite a different coordinator receipt")
        return existing
    _atomic_create_json(path, signed)
    return load_coordinator_receipt(path, search_plan=search_plan)


def load_coordinator_receipt(path: Path, *,
                             search_plan: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_coordinator_receipt(_load_json(path), search_plan)


def _rank_path(root: Path, rank: int) -> Path:
    return root / "ranks" / f"{rank:012d}.json"


def run_parallel_search(
        *, search_plan: Mapping[str, Any], output_root: Path,
        evaluate_rank: Callable[[int, tuple[int, ...]], Mapping[str, Any]],
        ) -> dict[str, Any]:
    """Generic resumable ordered reducer used by pure tests and audit tools.

    Scientific execution uses the wave-level functions below.  This generic
    reducer intentionally makes no claim about how ``evaluate_rank`` computes
    its outcome-blind classification.
    """

    plan = validate_search_plan(search_plan)
    root = Path(output_root)
    frontier = OrderedFrontier(total_rank_count=plan["total_rank_count"])
    classifications: dict[int, str] = {}

    def evaluate(rank: int) -> dict[str, Any]:
        path = _rank_path(root, rank)
        if path.exists():
            return load_rank_receipt(
                path, search_plan=plan, expected_rank=rank)
        combination = unrank_combination(
            rank, plan["candidate_pool_count"], plan["combination_size"])
        return write_rank_receipt(
            path, evaluate_rank(rank, combination), search_plan=plan)

    # Threads here are rank lanes, not solver workers.  Scientific rank lanes
    # share the bounded process pool used by ``parallel_lexicographic_rotations``.
    # Keeping this reducer concurrent makes the ordered-frontier semantics
    # directly testable without importing any simulator or scientific outcome.
    window = min(plan["active_rank_window"], plan["total_rank_count"])
    next_rank = 0
    in_flight: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, window))
    try:
        while next_rank < plan["total_rank_count"] and len(in_flight) < window:
            future = pool.submit(evaluate, next_rank)
            in_flight[future] = next_rank
            next_rank += 1
        while in_flight:
            done, _pending = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                rank = in_flight.pop(future)
                receipt = future.result()
                classification = str(receipt["classification"])
                classifications[rank] = classification
                event = frontier.record(rank=rank, classification=classification)
                if event is not None:
                    for speculative in in_flight:
                        speculative.cancel()
                    if event["status"] == "PASS":
                        return {
                            **event,
                            "combination_attempt_count": event["rank"] + 1,
                            "allocator_infeasible_combination_count": sum(
                                value == "ALLOCATOR_INFEASIBLE"
                                for key, value in classifications.items()
                                if key <= event["rank"]),
                        }
                    if event["status"] == "FATAL":
                        raise ParallelSearchFatal(
                            f"canonical rank {event['rank']} returned FATAL")
                    return {
                        **event,
                        "combination_attempt_count": plan["total_rank_count"],
                        "allocator_infeasible_combination_count": sum(
                            value == "ALLOCATOR_INFEASIBLE"
                            for value in classifications.values()),
                    }
                while (next_rank < plan["total_rank_count"]
                       and len(in_flight) < window):
                    replacement = pool.submit(evaluate, next_rank)
                    in_flight[replacement] = next_rank
                    next_rank += 1
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    raise ParallelSearchError("ordered search reducer ended without a terminal event")


def _worker_initialise() -> None:
    global _WORKER_THREAD_LIMITER
    for key, value in THREAD_ENVIRONMENT.items():
        os.environ[key] = value
    try:
        from threadpoolctl import threadpool_limits
        _WORKER_THREAD_LIMITER = threadpool_limits(limits=1)
    except ImportError:  # pragma: no cover - environment variables remain active.
        _WORKER_THREAD_LIMITER = None


def _classify_fixed_rotation_result(
        result: Any, *, constraints: Any, lower: np.ndarray,
        upper: np.ndarray, fixed_variable_index: int | None,
        integrality_tolerance: float = 1e-6,
        feasibility_tolerance: float = 1e-6,
        ) -> tuple[str, str]:
    """Fail-closed classification of one HiGHS fixed-rotation result.

    A limit/status error is operationally FATAL.  A successful result is
    accepted only after every binary variable, bound, and frozen linear
    constraint is independently checked within the same numerical tolerance
    used by the predecessor objective allocator.
    """

    if result.status == 2:
        return "INFEASIBLE", str(result.message)
    if not result.success or result.x is None or result.status != 0:
        return "FATAL", f"status={result.status}; message={result.message!r}"
    solution = np.asarray(result.x, dtype=np.float64)
    if (solution.shape != lower.shape or solution.shape != upper.shape
            or not np.all(np.isfinite(solution))):
        return "FATAL", "successful MILP returned a malformed solution vector"
    rounded = np.rint(solution)
    if np.max(np.abs(solution - rounded), initial=0.0) > integrality_tolerance:
        return "FATAL", "successful MILP returned a non-integral solution"
    if (np.any(solution < lower - feasibility_tolerance)
            or np.any(solution > upper + feasibility_tolerance)):
        return "FATAL", "successful MILP violated frozen variable bounds"
    activity = np.asarray(constraints.A @ solution, dtype=np.float64).reshape(-1)
    constraint_lower = np.asarray(constraints.lb, dtype=np.float64).reshape(-1)
    constraint_upper = np.asarray(constraints.ub, dtype=np.float64).reshape(-1)
    if (activity.shape != constraint_lower.shape
            or activity.shape != constraint_upper.shape
            or not np.all(np.isfinite(activity))
            or np.any(activity < constraint_lower - feasibility_tolerance)
            or np.any(activity > constraint_upper + feasibility_tolerance)):
        return "FATAL", "successful MILP violated frozen linear constraints"
    if (fixed_variable_index is not None
            and abs(float(solution[fixed_variable_index]) - 1.0)
            > integrality_tolerance):
        return "FATAL", "successful MILP violated its fixed rotation"
    return "FEASIBLE", str(result.message)


def _fixed_rotation_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Solve one fixed-prefix/fixed-current feasibility MILP."""

    started = time.monotonic()
    try:
        from scipy.optimize import Bounds, milp

        states = project_allocator_identity_states(list(task["states"]))
        prefix = [int(value) for value in task["prefix_rotations"]]
        state_index = int(task["state_index"])
        rotation = int(task["rotation"])
        constraints, base_bounds = ALLOC._constraint_system(states)
        variable_count = len(states) * ALLOC.CANDIDATE_COUNT
        lower = np.asarray(base_bounds.lb, dtype=np.float64).copy()
        upper = np.asarray(base_bounds.ub, dtype=np.float64).copy()
        for prefix_index, chosen in enumerate(prefix):
            start = prefix_index * ALLOC.CANDIDATE_COUNT
            lower[start:start + ALLOC.CANDIDATE_COUNT] = 0.0
            upper[start:start + ALLOC.CANDIDATE_COUNT] = 0.0
            lower[start + chosen] = 1.0
            upper[start + chosen] = 1.0
        start = state_index * ALLOC.CANDIDATE_COUNT
        lower[start:start + ALLOC.CANDIDATE_COUNT] = 0.0
        upper[start:start + ALLOC.CANDIDATE_COUNT] = 0.0
        lower[start + rotation] = 1.0
        upper[start + rotation] = 1.0
        options = dict(task["solver_options"])
        result = milp(
            c=np.zeros(variable_count, dtype=np.float64),
            integrality=np.ones(variable_count, dtype=np.uint8),
            bounds=Bounds(lower, upper), constraints=constraints,
            options=options,
        )
        status, message = _classify_fixed_rotation_result(
            result, constraints=constraints, lower=lower, upper=upper,
            fixed_variable_index=start + rotation)
    except BaseException as exc:  # returned to coordinator; never a nonpass.
        status = "FATAL"
        message = f"{type(exc).__name__}: {exc}"
    return {
        "rotation": int(task["rotation"]),
        "status": status,
        "message": message,
        "elapsed_s": time.monotonic() - started,
        "solver_call_count": 1,
        "worker_pid": os.getpid(),
        "thread_environment": {key: os.environ.get(key)
                               for key in THREAD_ENVIRONMENT},
    }


def _wave_path(root: Path, rank: int, state_index: int) -> Path:
    return root / "waves" / f"rank-{rank:012d}" / f"prefix-{state_index:03d}.json"


def _lexicographic_wave_decision(
        statuses: Sequence[str], *, state_index: int) -> tuple[str, int | None]:
    """Reduce fixed-rotation statuses to the exact objective-MILP decision.

    Rotation ``k`` is certified as the lexicographic minimum exactly when its
    fixed-r problem is feasible and every lower rotation is infeasible.  A
    timeout or solver error at a higher rotation is irrelevant to that proof;
    an error below ``k`` remains fatal because a lower feasible choice has not
    been excluded.
    """

    values = list(statuses)
    if len(values) != ROTATION_COUNT or any(
            value not in ROTATION_STATUSES for value in values):
        raise RuntimeError("fixed-rotation status surface changed")
    feasible = [index for index, value in enumerate(values)
                if value == "FEASIBLE"]
    if feasible:
        selected = min(feasible)
        if all(value == "INFEASIBLE" for value in values[:selected]):
            return "SELECTED", selected
        return "FATAL", None
    if "FATAL" in values:
        return "FATAL", None
    if state_index == 0:
        return "ALLOCATOR_INFEASIBLE", None
    # A previously feasible prefix cannot become globally infeasible after
    # fixing its own chosen variables.  Treat this as an internal fatal
    # contradiction without rewriting any worker result.
    return "FATAL", None


def _validate_wave_receipt(payload: Mapping[str, Any], *,
                           plan: Mapping[str, Any], rank: int,
                           projection_digest: str,
                           expected_prefix: Sequence[int]) -> dict[str, Any]:
    receipt = dict(payload)
    required = {
        "schema", "search_plan_digest", "rank", "state_index",
        "projection_digest", "prefix_rotations_before", "rotation_results",
        "wave_status", "selected_rotation", "solver_call_count", "wave_elapsed_s",
        "candidate_outcomes_consumed", "wave_receipt_digest",
    }
    results = receipt.get("rotation_results")
    if (set(receipt) != required or receipt.get("schema") != WAVE_RECEIPT_SCHEMA
            or receipt.get("search_plan_digest") != plan["search_plan_digest"]
            or receipt.get("rank") != rank
            or receipt.get("state_index") != len(expected_prefix)
            or receipt.get("projection_digest") != projection_digest
            or receipt.get("prefix_rotations_before") != list(expected_prefix)
            or not isinstance(results, list) or len(results) != ROTATION_COUNT
            or [row.get("rotation") for row in results]
            != list(range(ROTATION_COUNT))
            or any(set(row) != {
                "rotation", "status", "message", "elapsed_s",
                "solver_call_count", "worker_pid", "thread_environment",
            } for row in results)
            or any(row.get("status") not in ROTATION_STATUSES
                   or not isinstance(row.get("message"), str)
                   or not isinstance(row.get("elapsed_s"), (int, float))
                   or row["elapsed_s"] < 0
                   or row.get("solver_call_count") != 1
                   or not isinstance(row.get("worker_pid"), int)
                   or row.get("thread_environment") != THREAD_ENVIRONMENT
                   for row in results)
            or receipt.get("wave_status") not in WAVE_STATUSES
            or receipt.get("solver_call_count") != ROTATION_COUNT
            or not isinstance(receipt.get("wave_elapsed_s"), (int, float))
            or receipt["wave_elapsed_s"] < 0
            or receipt.get("candidate_outcomes_consumed") is not False
            or receipt.get("wave_receipt_digest") != canonical_digest(
                _without_digest(receipt, "wave_receipt_digest"))):
        raise RuntimeError("prefix-wave receipt binding changed")
    expected_status, expected_rotation = _lexicographic_wave_decision(
        [row["status"] for row in results], state_index=len(expected_prefix))
    if (receipt["wave_status"] != expected_status
            or receipt.get("selected_rotation") != expected_rotation):
        raise RuntimeError("prefix-wave lexicographic decision changed")
    return receipt


def _write_wave_receipt(path: Path, payload: Mapping[str, Any], *,
                        plan: Mapping[str, Any], rank: int,
                        projection_digest: str,
                        expected_prefix: Sequence[int]) -> dict[str, Any]:
    signed = _signed(payload, "wave_receipt_digest")
    _validate_wave_receipt(
        signed, plan=plan, rank=rank, projection_digest=projection_digest,
        expected_prefix=expected_prefix)
    if path.exists():
        existing = _validate_wave_receipt(
            _load_json(path), plan=plan, rank=rank,
            projection_digest=projection_digest,
            expected_prefix=expected_prefix)
        if existing != signed:
            raise RuntimeError("refusing to overwrite a different wave receipt")
        return existing
    _atomic_create_json(path, signed)
    return _validate_wave_receipt(
        _load_json(path), plan=plan, rank=rank,
        projection_digest=projection_digest,
        expected_prefix=expected_prefix)


def parallel_lexicographic_rotations(
        states: Sequence[Mapping[str, Any]], *, search_plan: Mapping[str, Any],
        rank: int, checkpoint_root: Path,
        executor: concurrent.futures.ProcessPoolExecutor | None = None,
        telemetry: Callable[[Mapping[str, Any]], None] | None = None,
        stop_event: threading.Event | None = None,
        wave_submission_lock: Any | None = None,
        ) -> tuple[list[int], dict[str, Any]]:
    """Compute/resume canonical prefix waves with cooperative rank stopping.

    Once a wave begins, all twelve solver futures are drained and its durable
    receipt is written.  A shared submission lock makes the next-wave boundary
    atomic with canonical-frontier termination, so a speculative lane never
    submits another wave after the coordinator closes the frontier.
    """

    plan = validate_search_plan(search_plan)
    normalised = project_allocator_identity_states(states)
    projection_digest = canonical_digest(normalised)
    prefix: list[int] = []
    cumulative_calls = 0
    cumulative_elapsed = 0.0
    owned = executor is None
    if owned:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=min(plan["worker_count"], ROTATION_COUNT),
            initializer=_worker_initialise)
    assert executor is not None
    try:
        for state_index in range(len(normalised)):
            if stop_event is not None and stop_event.is_set():
                raise _SpeculativeRankStopped(
                    f"rank {rank} stopped before prefix {state_index}")
            path = _wave_path(Path(checkpoint_root), rank, state_index)
            if path.exists():
                receipt = _validate_wave_receipt(
                    _load_json(path), plan=plan, rank=rank,
                    projection_digest=projection_digest,
                    expected_prefix=prefix)
            else:
                wave_started = time.monotonic()
                def submit_wave() -> list[Any]:
                    if stop_event is not None and stop_event.is_set():
                        raise _SpeculativeRankStopped(
                            f"rank {rank} stopped before prefix {state_index}")
                    return [executor.submit(_fixed_rotation_worker, {
                        "states": normalised,
                        "prefix_rotations": prefix,
                        "state_index": state_index,
                        "rotation": rotation,
                        "solver_options": plan["solver_options"],
                    }) for rotation in range(ROTATION_COUNT)]

                if wave_submission_lock is None:
                    futures = submit_wave()
                else:
                    with wave_submission_lock:
                        futures = submit_wave()
                results: list[dict[str, Any]] = []
                first_error: BaseException | None = None
                # Never abandon already-submitted native solver work.  Even if
                # one future faults, drain the other eleven before propagating.
                for future in futures:
                    try:
                        results.append(future.result())
                    except BaseException as exc:
                        if first_error is None:
                            first_error = exc
                if first_error is not None:
                    raise first_error
                results.sort(key=lambda row: int(row["rotation"]))
                wave_status, selected_rotation = \
                    _lexicographic_wave_decision(
                        [row["status"] for row in results],
                        state_index=state_index)
                receipt = _write_wave_receipt(path, {
                    "schema": WAVE_RECEIPT_SCHEMA,
                    "search_plan_digest": plan["search_plan_digest"],
                    "rank": rank,
                    "state_index": state_index,
                    "projection_digest": projection_digest,
                    "prefix_rotations_before": list(prefix),
                    "rotation_results": results,
                    "wave_status": wave_status,
                    "selected_rotation": selected_rotation,
                    "solver_call_count": ROTATION_COUNT,
                    "wave_elapsed_s": time.monotonic() - wave_started,
                    "candidate_outcomes_consumed": False,
                }, plan=plan, rank=rank, projection_digest=projection_digest,
                    expected_prefix=prefix)
            if receipt["wave_status"] == "ALLOCATOR_INFEASIBLE":
                raise ALLOC.CandidateAllocationInfeasible(
                    "the frozen identity/goal-type contingency has no allocation "
                    "satisfying all exact candidate margins")
            if receipt["wave_status"] == "FATAL":
                fatal = next(
                    (row for row in receipt["rotation_results"]
                     if row["status"] == "FATAL"),
                    {"message": (
                        "previously feasible allocation prefix became "
                        "infeasible")})
                raise ParallelSearchFatal(
                    f"rank {rank} prefix {state_index} rotation "
                    f"{fatal['rotation']}: {fatal['message']}")
            prefix.append(int(receipt["selected_rotation"]))
            cumulative_calls += int(receipt["solver_call_count"])
            cumulative_elapsed += float(receipt["wave_elapsed_s"])
            event = {
                "rank": rank,
                "completed_prefix_waves": len(prefix),
                "total_prefix_waves": len(normalised),
                "cumulative_solver_calls": cumulative_calls,
                "cumulative_wave_elapsed_s": cumulative_elapsed,
                "last_wave_elapsed_s": float(receipt["wave_elapsed_s"]),
                "operational_only_not_selector_input": True,
            }
            if telemetry is not None:
                telemetry(event)
        if stop_event is not None and stop_event.is_set():
            raise _SpeculativeRankStopped(
                f"rank {rank} stopped after its final prefix wave")
        if len(prefix) != PREFIX_STATE_COUNT:
            raise ParallelSearchError(
                f"parallel allocator resolved {len(prefix)}, expected 120 prefixes")
        return prefix, {
            "projection_digest": projection_digest,
            "completed_prefix_waves": len(prefix),
            "solver_call_count": cumulative_calls,
            "cumulative_wave_elapsed_s": cumulative_elapsed,
        }
    finally:
        if owned:
            executor.shutdown(wait=True, cancel_futures=False)


def materialize_allocation_manifest_single_solve(
        states: Sequence[Mapping[str, Any]], *,
        source_identity_manifest_digest: str,
        rotations: Sequence[int]) -> dict[str, Any]:
    """Materialize exact allocator bytes without a second canonical re-solve."""

    normalised = project_allocator_identity_states(states)
    if (len(rotations) != len(normalised)
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or not 0 <= value < ROTATION_COUNT for value in rotations)):
        raise ParallelSearchError("rotation vector is malformed")
    assignments = [{
        **state,
        "rotation_index": rotation,
        "candidate_indices": list(ALLOC.candidate_block(rotation)),
    } for state, rotation in zip(normalised, rotations, strict=True)]
    manifest: dict[str, Any] = {
        "schema": ALLOC.SCHEMA,
        "status": ALLOC.STATUS,
        "source_identity_manifest_digest": source_identity_manifest_digest,
        "pre_outcome_identity_digest": ALLOC.pre_outcome_identity_digest(normalised),
        "allocation_contract": ALLOC.algorithm_contract(),
        "allocation_contract_digest": ALLOC.allocation_contract_digest(),
        "allocation_amendment": ALLOC.allocation_amendment_contract(),
        "allocation_amendment_digest": ALLOC.allocation_amendment_digest(),
        "assignments": assignments,
        "contingency_tables": ALLOC._contingency_tables(assignments),
        "post_identity_pre_outcome_validation":
            ALLOC._post_identity_pre_outcome_validation(assignments),
    }
    manifest["allocation_manifest_digest"] = \
        ALLOC.allocation_manifest_digest(manifest)
    # Every deterministic structural and scientific balance check, excluding
    # only the predecessor validator's duplicate canonical re-solve.
    ALLOC._validate_counts(manifest)
    if manifest["post_identity_pre_outcome_validation"] != \
            ALLOC._post_identity_pre_outcome_validation(assignments):
        raise ParallelSearchError("single-solve allocation validation changed")
    return manifest


def _objective_wave_path(root: Path, rank: int, state_index: int) -> Path:
    return (root / "winner-objective" / f"rank-{rank:012d}"
            / f"prefix-{state_index:03d}.json")


def _validate_objective_wave_receipt(
        payload: Mapping[str, Any], *, plan: Mapping[str, Any], rank: int,
        projection_digest: str, source_digest: str,
        expected_prefix: Sequence[int], certified_rotation: int,
        ) -> dict[str, Any]:
    receipt = dict(payload)
    required = {
        "schema", "search_plan_digest", "rank", "state_index",
        "projection_digest", "source_identity_manifest_digest",
        "prefix_rotations_before", "certified_rotation", "solver_status",
        "selected_rotation", "objective_value", "solver_message",
        "solver_elapsed_s", "solver_call_count", "thread_environment",
        "candidate_outcomes_consumed", "objective_wave_receipt_digest",
    }
    if (set(receipt) != required
            or receipt.get("schema") != OBJECTIVE_WAVE_RECEIPT_SCHEMA
            or receipt.get("search_plan_digest") != plan["search_plan_digest"]
            or receipt.get("rank") != rank
            or receipt.get("state_index") != len(expected_prefix)
            or receipt.get("projection_digest") != projection_digest
            or receipt.get("source_identity_manifest_digest") != source_digest
            or receipt.get("prefix_rotations_before") != list(expected_prefix)
            or receipt.get("certified_rotation") != certified_rotation
            or receipt.get("solver_status") not in ("FEASIBLE", "FATAL")
            or not isinstance(receipt.get("solver_message"), str)
            or not isinstance(receipt.get("solver_elapsed_s"), (int, float))
            or receipt["solver_elapsed_s"] < 0
            or receipt.get("solver_call_count") != 1
            or receipt.get("thread_environment") != THREAD_ENVIRONMENT
            or receipt.get("candidate_outcomes_consumed") is not False
            or receipt.get("objective_wave_receipt_digest") != canonical_digest(
                _without_digest(receipt, "objective_wave_receipt_digest"))):
        raise ParallelSearchError("bounded objective-wave receipt binding changed")
    if receipt["solver_status"] == "FEASIBLE":
        if (receipt.get("selected_rotation") != certified_rotation
                or not isinstance(receipt.get("objective_value"), (int, float))
                or abs(float(receipt["objective_value"])
                           - certified_rotation) > 1e-6):
            raise ParallelSearchError(
                "bounded objective wave differs from certified rotation")
    elif (receipt.get("selected_rotation") is not None
          or receipt.get("objective_value") is not None):
        raise ParallelSearchError("fatal objective-wave evidence changed")
    return receipt


def _bounded_objective_rotations(
        states: Sequence[Mapping[str, Any]], *, plan: Mapping[str, Any],
        rank: int, checkpoint_root: Path,
        certified_rotations: Sequence[int] | None = None,
        source_identity_manifest_digest: str,
        telemetry: Callable[[Mapping[str, Any]], None] | None = None,
        ) -> tuple[list[int], list[dict[str, Any]]]:
    """Bounded exact predecessor objective algorithm with durable prefixes."""

    from scipy.optimize import Bounds, milp

    _worker_initialise()
    plan = validate_search_plan(plan)
    normalised = project_allocator_identity_states(states)
    projection_digest = canonical_digest(normalised)
    if certified_rotations is not None and (
            len(certified_rotations) != PREFIX_STATE_COUNT
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or not 0 <= value < ROTATION_COUNT
                   for value in certified_rotations)):
        raise ParallelSearchError("certified winner rotation vector is malformed")
    constraints, base_bounds = ALLOC._constraint_system(normalised)
    variable_count = len(normalised) * ROTATION_COUNT
    integrality = np.ones(variable_count, dtype=np.uint8)
    lower = np.asarray(base_bounds.lb, dtype=np.float64).copy()
    upper = np.asarray(base_bounds.ub, dtype=np.float64).copy()
    selected: list[int] = []
    receipts: list[dict[str, Any]] = []
    options = dict(plan["solver_options"])
    for state_index in range(len(normalised)):
        certified = (int(certified_rotations[state_index])
                     if certified_rotations is not None else -1)
        path = _objective_wave_path(Path(checkpoint_root), rank, state_index)
        if path.exists():
            if certified < 0:
                raise ParallelSearchError(
                    "unbound objective benchmark cannot resume winner receipts")
            receipt = _validate_objective_wave_receipt(
                _load_json(path), plan=plan, rank=rank,
                projection_digest=projection_digest,
                source_digest=source_identity_manifest_digest,
                expected_prefix=selected, certified_rotation=certified)
        else:
            started = time.monotonic()
            objective = np.zeros(variable_count, dtype=np.float64)
            start = state_index * ROTATION_COUNT
            objective[start:start + ROTATION_COUNT] = np.arange(
                ROTATION_COUNT, dtype=np.float64)
            result = milp(
                c=objective, integrality=integrality,
                bounds=Bounds(lower, upper), constraints=constraints,
                options=options)
            status, message = _classify_fixed_rotation_result(
                result, constraints=constraints, lower=lower, upper=upper,
                fixed_variable_index=None)
            rotation: int | None = None
            objective_value: float | None = None
            if status == "FEASIBLE":
                start = state_index * ROTATION_COUNT
                local = np.asarray(
                    result.x[start:start + ROTATION_COUNT], dtype=np.float64)
                rotation = int(np.argmax(local))
                objective_value = (
                    None if result.fun is None else float(result.fun))
                if (local[rotation] < 0.5 or objective_value is None
                        or abs(objective_value - rotation) > 1e-6):
                    status = "FATAL"
                    message = "objective disagrees with integral state choice"
                    rotation = None
                    objective_value = None
                elif certified >= 0 and rotation != certified:
                    status = "FATAL"
                    message = (
                        "bounded objective differs from fixed-rotation certificate")
                    rotation = None
                    objective_value = None
            else:
                # An infeasible or limited prefix after a certified feasible
                # fixed-r solution is an operational contradiction, never a
                # scientific nonpass.
                status = "FATAL"
            payload = {
                "schema": OBJECTIVE_WAVE_RECEIPT_SCHEMA,
                "search_plan_digest": plan["search_plan_digest"],
                "rank": rank,
                "state_index": state_index,
                "projection_digest": projection_digest,
                "source_identity_manifest_digest":
                    source_identity_manifest_digest,
                "prefix_rotations_before": list(selected),
                "certified_rotation": certified if certified >= 0 else rotation,
                "solver_status": status,
                "selected_rotation": rotation,
                "objective_value": objective_value,
                "solver_message": message,
                "solver_elapsed_s": time.monotonic() - started,
                "solver_call_count": 1,
                "thread_environment": dict(THREAD_ENVIRONMENT),
                "candidate_outcomes_consumed": False,
            }
            payload = _signed(payload, "objective_wave_receipt_digest")
            # Benchmarks use no certified vector and do not persist.  Winner
            # proof always has a certified vector and persists atomically.
            if certified >= 0:
                _validate_objective_wave_receipt(
                    payload, plan=plan, rank=rank,
                    projection_digest=projection_digest,
                    source_digest=source_identity_manifest_digest,
                    expected_prefix=selected, certified_rotation=certified)
                _atomic_create_json(path, payload)
                receipt = _validate_objective_wave_receipt(
                    _load_json(path), plan=plan, rank=rank,
                    projection_digest=projection_digest,
                    source_digest=source_identity_manifest_digest,
                    expected_prefix=selected, certified_rotation=certified)
            else:
                receipt = payload
        if receipt["solver_status"] != "FEASIBLE":
            raise ParallelSearchFatal(
                f"bounded objective prefix {state_index} failed: "
                f"{receipt['solver_message']}")
        selected.append(int(receipt["selected_rotation"]))
        start = state_index * ROTATION_COUNT
        lower[start:start + ROTATION_COUNT] = 0.0
        upper[start:start + ROTATION_COUNT] = 0.0
        lower[start + selected[-1]] = 1.0
        upper[start + selected[-1]] = 1.0
        receipts.append(receipt)
        if telemetry is not None:
            telemetry({
                "winner_objective_completed_prefix_waves": len(selected),
                "winner_objective_total_prefix_waves": PREFIX_STATE_COUNT,
                "last_wave_elapsed_s": receipt["solver_elapsed_s"],
                "operational_only_not_selector_input": True,
            })
    return selected, receipts


def validate_winner_allocation_bounded(
        *, states: Sequence[Mapping[str, Any]], allocation: Mapping[str, Any],
        search_plan: Mapping[str, Any], rank: int, checkpoint_root: Path,
        assignment_digest: str,
        telemetry: Callable[[Mapping[str, Any]], None] | None = None,
        ) -> dict[str, Any]:
    """Issue/reopen the one bounded full canonical winner proof."""

    plan = validate_search_plan(search_plan)
    manifest = dict(allocation)
    rotations = [int(row["rotation_index"])
                 for row in manifest.get("assignments", [])]
    source_digest = str(manifest.get("source_identity_manifest_digest", ""))
    expected_materialized = materialize_allocation_manifest_single_solve(
        states, source_identity_manifest_digest=source_digest,
        rotations=rotations)
    if expected_materialized != manifest:
        raise ParallelSearchFatal("winner allocation bytes changed before proof")
    objective_rotations, waves = _bounded_objective_rotations(
        states, plan=plan, rank=rank, checkpoint_root=checkpoint_root,
        certified_rotations=rotations,
        source_identity_manifest_digest=source_digest,
        telemetry=telemetry)
    if objective_rotations != rotations:
        raise ParallelSearchFatal("bounded objective winner vector changed")
    wave_rows = [{
        "state_index": row["state_index"],
        "objective_wave_receipt_digest": row["objective_wave_receipt_digest"],
    } for row in waves]
    payload = {
        "schema": WINNER_VALIDATION_RECEIPT_SCHEMA,
        "status": "PASS_BOUNDED_EXACT_CANONICAL_OBJECTIVE",
        "search_plan_digest": plan["search_plan_digest"],
        "rank": rank,
        "projection_digest": canonical_digest(
            project_allocator_identity_states(states)),
        "source_identity_manifest_digest": source_digest,
        "allocation_manifest_digest": manifest["allocation_manifest_digest"],
        "candidate_assignment_set_digest": assignment_digest,
        "selected_rotations": objective_rotations,
        "objective_wave_count": PREFIX_STATE_COUNT,
        "objective_wave_receipts": wave_rows,
        "solver_time_limit_s": plan["solver_options"]["time_limit"],
        "candidate_outcomes_consumed": False,
    }
    payload = _signed(payload, "winner_validation_receipt_digest")
    path = Path(checkpoint_root) / "winner-objective-validation.json"
    validation_kwargs = {
        "plan": plan,
        "rank": rank,
        "projection_digest": payload["projection_digest"],
        "source_digest": source_digest,
        "allocation_digest": manifest["allocation_manifest_digest"],
        "assignment_digest": assignment_digest,
        "rotations": objective_rotations,
        "objective_wave_rows": wave_rows,
    }
    if path.exists():
        existing = _validate_winner_validation_receipt(
            _load_json(path), **validation_kwargs)
        if existing != payload:
            raise ParallelSearchFatal(
                "winner objective validation receipt changed on resume")
    else:
        _atomic_create_json(path, payload)
        existing = _validate_winner_validation_receipt(
            _load_json(path), **validation_kwargs)
        if existing != payload:
            raise ParallelSearchFatal(
                "winner objective validation receipt changed during creation")
    return existing


def _rank_payload(
        *, plan: Mapping[str, Any], rank: int, projection_digest: str,
        source_digest: str, classification: str, rotations: Sequence[int],
        allocation: Mapping[str, Any] | None,
        assignment_digest: str | None) -> dict[str, Any]:
    combination = unrank_combination(
        rank, plan["candidate_pool_count"], plan["combination_size"])
    return {
        "schema": RANK_RECEIPT_SCHEMA,
        "search_plan_digest": plan["search_plan_digest"],
        "rank": rank,
        "combination_indices": list(combination),
        "selected_scene_ids": [plan["candidate_scene_ids"][index]
                               for index in combination],
        "projection_digest": projection_digest,
        "source_identity_manifest_digest": source_digest,
        "completed_prefix_wave_count": len(rotations),
        "selected_rotations": list(rotations),
        "provisional_allocation_manifest_digest": (
            None if allocation is None
            else allocation["allocation_manifest_digest"]),
        "provisional_candidate_assignment_set_digest": assignment_digest,
        "classification": classification,
        "candidate_outcomes_consumed": False,
    }


def _scientific_rank_lane(
        *, rank: int, plan: Mapping[str, Any], checkpoint_root: Path,
        shared_executor: concurrent.futures.ProcessPoolExecutor,
        prepare_rank: Callable[[int, tuple[int, ...]], Mapping[str, Any]],
        classify_mask: Callable[[Sequence[Mapping[str, Any]],
                                 Mapping[str, Any], Mapping[str, Any]], bool],
        telemetry: Callable[[Mapping[str, Any]], None] | None,
        stop_event: threading.Event,
        wave_submission_lock: Any,
        ) -> dict[str, Any]:
    """Evaluate one rank without committing it ahead of the ordered frontier."""

    combination = unrank_combination(
        rank, plan["candidate_pool_count"], plan["combination_size"])
    material = dict(prepare_rank(rank, combination))
    if set(material) != {
            "states", "source_identity_manifest_digest", "mask_context"}:
        raise ParallelSearchFatal("rank preparation surface changed")
    states = list(material["states"])
    source_digest = material["source_identity_manifest_digest"]
    mask_context = material["mask_context"]
    if not _is_digest(source_digest) or not isinstance(mask_context, Mapping):
        raise ParallelSearchFatal("rank preparation binding changed")
    projection_digest = canonical_digest(project_allocator_identity_states(states))
    if len(states) != PREFIX_STATE_COUNT:
        raise ParallelSearchFatal("scientific rank does not contain 120 states")
    try:
        rotations, evidence = parallel_lexicographic_rotations(
            states, search_plan=plan, rank=rank,
            checkpoint_root=checkpoint_root, executor=shared_executor,
            telemetry=telemetry, stop_event=stop_event,
            wave_submission_lock=wave_submission_lock)
    except ALLOC.CandidateAllocationInfeasible:
        return _rank_payload(
            plan=plan, rank=rank, projection_digest=projection_digest,
            source_digest=source_digest,
            classification="ALLOCATOR_INFEASIBLE", rotations=[],
            allocation=None, assignment_digest=None)
    allocation = materialize_allocation_manifest_single_solve(
        states, source_identity_manifest_digest=source_digest,
        rotations=rotations)
    assignment_digest = _candidate_assignment_set_digest(allocation)
    passes = classify_mask(states, allocation, mask_context)
    if not isinstance(passes, bool):
        raise ParallelSearchFatal("exact-mask classifier did not return a boolean")
    payload = _rank_payload(
        plan=plan, rank=rank, projection_digest=projection_digest,
        source_digest=source_digest,
        classification="PASS" if passes else "MASK_FAIL",
        rotations=rotations, allocation=allocation,
        assignment_digest=assignment_digest)
    payload["_allocation"] = allocation
    payload["_mask_context"] = dict(mask_context)
    payload["_operational_evidence"] = evidence
    return payload


def run_scientific_parallel_search(
        *, search_plan: Mapping[str, Any], checkpoint_root: Path,
        prepare_rank: Callable[[int, tuple[int, ...]], Mapping[str, Any]],
        classify_mask: Callable[[Sequence[Mapping[str, Any]],
                                 Mapping[str, Any], Mapping[str, Any]], bool],
        validate_winner: Callable[[int, Sequence[Mapping[str, Any]],
                                   Mapping[str, Any], Mapping[str, Any]], bool],
        telemetry: Callable[[Mapping[str, Any]], None] | None = print,
        executor_factory: Callable[..., Any] | None = None,
        process_executor: Any | None = None,
        ) -> dict[str, Any]:
    """Run/resume the actual outcome-blind search with ordered rank lanes.

    Exactly one process pool owns ``worker_count`` single-thread workers.  Rank
    lanes are threads which submit their 12 fixed-rotation wave tasks into that
    shared pool; no rank creates a nested pool.  Speculative rank results become
    scientific evidence only when the ordered frontier commits them.
    """

    plan = validate_search_plan(search_plan)
    root = Path(checkpoint_root)
    if process_executor is not None and executor_factory is not None:
        raise ValueError(
            "process_executor and executor_factory are mutually exclusive")
    owned_process_pool = process_executor is None
    if owned_process_pool:
        factory = (concurrent.futures.ProcessPoolExecutor
                   if executor_factory is None else executor_factory)
        process_pool = factory(
            max_workers=plan["worker_count"], initializer=_worker_initialise,
            mp_context=multiprocessing.get_context("spawn"))
    else:
        process_pool = process_executor
    lane_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=plan["active_rank_window"])
    frontier = OrderedFrontier(total_rank_count=plan["total_rank_count"])
    stop_event = threading.Event()
    wave_submission_lock = threading.Lock()
    completed: dict[int, dict[str, Any]] = {}
    in_flight: dict[concurrent.futures.Future[dict[str, Any]], int] = {}
    next_rank = 0

    def report(event: Mapping[str, Any]) -> None:
        if telemetry is not None:
            telemetry(dict(event))

    def submit(rank: int) -> None:
        path = _rank_path(root, rank)
        if path.exists():
            receipt = load_rank_receipt(
                path, search_plan=plan, expected_rank=rank)
            future: concurrent.futures.Future[dict[str, Any]] = \
                concurrent.futures.Future()
            future.set_result(receipt)
        else:
            future = lane_pool.submit(
                _scientific_rank_lane, rank=rank, plan=plan,
                checkpoint_root=root, shared_executor=process_pool,
                prepare_rank=prepare_rank, classify_mask=classify_mask,
                telemetry=report, stop_event=stop_event,
                wave_submission_lock=wave_submission_lock)
        in_flight[future] = rank

    def stop_speculation() -> None:
        # Serialize the stop with every lane's next-wave submission.  A wave
        # already inside the gate is allowed to submit all twelve tasks, drain
        # them, and persist its operational receipt; no later wave can start.
        with wave_submission_lock:
            stop_event.set()
        for speculative in tuple(in_flight):
            speculative.cancel()

    def drain_stopped_speculation() -> None:
        """Wait for every already-started wave before objective validation."""

        for speculative in tuple(in_flight):
            try:
                speculative.result()
            except BaseException:
                # These ranks are strictly above an already established
                # canonical frontier.  Their scientific result is excluded;
                # only draining their bounded operational work matters here.
                pass
            finally:
                in_flight.pop(speculative, None)

    try:
        while (next_rank < plan["total_rank_count"]
               and len(in_flight) < plan["active_rank_window"]):
            submit(next_rank)
            next_rank += 1
        while in_flight:
            done, _ = concurrent.futures.wait(
                in_flight, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                rank = in_flight.pop(future)
                try:
                    result = future.result()
                except BaseException as exc:
                    completed[rank] = {
                        "receipt": {"classification": "FATAL"},
                        "fatal_exception": exc,
                    }
                    event = frontier.record(rank=rank, classification="FATAL")
                    if event is not None:
                        stop_speculation()
                    report({
                        "completed_nonpass_rank_count": sum(
                            row["receipt"]["classification"] in ORDINARY_NONPASS
                            for row in completed.values()),
                        "committed_frontier_rank": frontier.committed_rank_count,
                        "active_speculative_rank_count": len(in_flight),
                        "last_completed_rank": rank,
                        "last_completed_classification": "FATAL",
                        "operational_only_not_selector_input": True,
                    })
                    if event is not None and event["status"] == "FATAL":
                        cause = completed[event["rank"]]["fatal_exception"]
                        raise ParallelSearchFatal(
                            f"canonical rank {event['rank']} failed fatally: "
                            f"{type(cause).__name__}: {cause}") from cause
                    while (next_rank < plan["total_rank_count"]
                           and len(in_flight) < plan["active_rank_window"]):
                        submit(next_rank)
                        next_rank += 1
                    continue
                allocation = result.pop("_allocation", None)
                mask_context = result.pop("_mask_context", None)
                evidence = result.pop("_operational_evidence", None)
                receipt = write_rank_receipt(
                    _rank_path(root, rank), result, search_plan=plan)
                completed[rank] = {
                    "receipt": receipt, "allocation": allocation,
                    "mask_context": mask_context, "evidence": evidence,
                }
                event = frontier.record(
                    rank=rank, classification=receipt["classification"])
                if event is not None:
                    stop_speculation()
                report({
                    "completed_nonpass_rank_count": sum(
                        row["receipt"]["classification"] in ORDINARY_NONPASS
                        for row in completed.values()),
                    "committed_frontier_rank": frontier.committed_rank_count,
                    "active_speculative_rank_count": len(in_flight),
                    "last_completed_rank": rank,
                    "operational_only_not_selector_input": True,
                })
                if event is not None:
                    if event["status"] == "FATAL":
                        raise ParallelSearchFatal(
                            f"canonical rank {event['rank']} returned FATAL")
                    if event["status"] == "EXHAUSTED":
                        return {
                            "schema": SEARCH_RESULT_SCHEMA,
                            "status": "EXHAUSTED",
                            "combination_attempt_count": plan["total_rank_count"],
                            "allocator_infeasible_combination_count": sum(
                                row["receipt"]["classification"]
                                == "ALLOCATOR_INFEASIBLE"
                                for row in completed.values()),
                            "search_plan_digest": plan["search_plan_digest"],
                            "candidate_outcomes_consumed": False,
                        }
                    winner_rank = event["rank"]
                    winner = completed[winner_rank]
                    # ``stop_speculation`` prevents a next wave, but native
                    # tasks in a wave already submitted to the shared pool must
                    # finish.  Drain their lane futures before the coordinator
                    # starts the one-thread bounded objective proof, preserving
                    # the plan's global 32-solver ceiling.
                    drain_stopped_speculation()
                    if winner["allocation"] is None:
                        # A resumed PASS reloads the 120 waves and rematerializes
                        # its allocation without solving them again.
                        material = dict(prepare_rank(
                            winner_rank, unrank_combination(
                                winner_rank, plan["candidate_pool_count"],
                                plan["combination_size"])))
                        rotations, _ = parallel_lexicographic_rotations(
                            material["states"], search_plan=plan,
                            rank=winner_rank, checkpoint_root=root,
                            executor=process_pool, telemetry=report)
                        winner["allocation"] = \
                            materialize_allocation_manifest_single_solve(
                                material["states"],
                                source_identity_manifest_digest=material[
                                    "source_identity_manifest_digest"],
                                rotations=rotations)
                        winner["mask_context"] = material["mask_context"]
                    material = dict(prepare_rank(
                        winner_rank, unrank_combination(
                            winner_rank, plan["candidate_pool_count"],
                            plan["combination_size"])))
                    winner_validation = validate_winner_allocation_bounded(
                        states=material["states"],
                        allocation=winner["allocation"], search_plan=plan,
                        rank=winner_rank, checkpoint_root=root,
                        assignment_digest=winner["receipt"][
                            "provisional_candidate_assignment_set_digest"],
                        telemetry=report)
                    if validate_winner(
                            winner_rank, material["states"],
                            winner["allocation"], winner["mask_context"]) is not True:
                        raise ParallelSearchFatal(
                            "solve-free structural/scientific winner callback "
                            "did not pass")
                    return {
                        "schema": SEARCH_RESULT_SCHEMA,
                        "status": "PASS",
                        "rank": winner_rank,
                        "combination_attempt_count": winner_rank + 1,
                        "allocator_infeasible_combination_count": sum(
                            completed[index]["receipt"]["classification"]
                            == "ALLOCATOR_INFEASIBLE"
                            for index in range(winner_rank + 1)),
                        "rank_receipt": winner["receipt"],
                        "allocation": winner["allocation"],
                        "winner_validation_receipt": winner_validation,
                        "search_plan_digest": plan["search_plan_digest"],
                        "candidate_outcomes_consumed": False,
                    }
                while (next_rank < plan["total_rank_count"]
                       and len(in_flight) < plan["active_rank_window"]):
                    submit(next_rank)
                    next_rank += 1
    finally:
        stop_speculation()
        lane_pool.shutdown(wait=True, cancel_futures=True)
        # Rank lanes have drained every wave they began.  Do not cancel any
        # queued process future: native solver work is always allowed to finish.
        if owned_process_pool:
            process_pool.shutdown(wait=True, cancel_futures=False)
    raise ParallelSearchError("scientific parallel search ended without terminal state")


def _candidate_assignment_set_digest(
        allocation: Mapping[str, Any]) -> str:
    assignments = allocation.get("assignments")
    if not isinstance(assignments, list):
        raise ParallelSearchError("allocation assignments are unavailable")
    try:
        projection = [{
            "state_id": row["state_id"],
            "state_identity_digest": row["state_identity_digest"],
            "candidate_rotation_index": row["rotation_index"],
            "candidate_indices": row["candidate_indices"],
        } for row in sorted(assignments, key=lambda value: (
            value["state_identity_digest"], value["state_id"]))]
    except (KeyError, TypeError) as exc:
        raise ParallelSearchError(
            "allocation assignment projection is malformed") from exc
    return canonical_digest(projection)


def _validate_winner_validation_receipt(
        payload: Mapping[str, Any], *, plan: Mapping[str, Any], rank: int,
        projection_digest: str, source_digest: str,
        allocation_digest: str, assignment_digest: str,
        rotations: Sequence[int], objective_wave_rows: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    receipt = dict(payload)
    required = {
        "schema", "status", "search_plan_digest", "rank",
        "projection_digest", "source_identity_manifest_digest",
        "allocation_manifest_digest", "candidate_assignment_set_digest",
        "selected_rotations", "objective_wave_count",
        "objective_wave_receipts", "solver_time_limit_s",
        "candidate_outcomes_consumed", "winner_validation_receipt_digest",
    }
    if (set(receipt) != required
            or receipt.get("schema") != WINNER_VALIDATION_RECEIPT_SCHEMA
            or receipt.get("status")
            != "PASS_BOUNDED_EXACT_CANONICAL_OBJECTIVE"
            or receipt.get("search_plan_digest") != plan["search_plan_digest"]
            or receipt.get("rank") != rank
            or receipt.get("projection_digest") != projection_digest
            or receipt.get("source_identity_manifest_digest") != source_digest
            or receipt.get("allocation_manifest_digest") != allocation_digest
            or receipt.get("candidate_assignment_set_digest")
            != assignment_digest
            or receipt.get("selected_rotations") != list(rotations)
            or receipt.get("objective_wave_count") != PREFIX_STATE_COUNT
            or receipt.get("objective_wave_receipts")
            != [dict(row) for row in objective_wave_rows]
            or receipt.get("solver_time_limit_s")
            != plan["solver_options"]["time_limit"]
            or receipt.get("candidate_outcomes_consumed") is not False
            or receipt.get("winner_validation_receipt_digest")
            != canonical_digest(_without_digest(
                receipt, "winner_validation_receipt_digest"))):
        raise ParallelSearchError(
            "winner objective-validation receipt binding changed")
    return receipt


def validate_terminal_search_result(
        *, terminal_result: Mapping[str, Any],
        search_plan: Mapping[str, Any], checkpoint_root: Path,
        prepare_rank: Callable[[int, tuple[int, ...]], Mapping[str, Any]],
        classify_mask: Callable[[Sequence[Mapping[str, Any]],
                                 Mapping[str, Any], Mapping[str, Any]], bool],
        validate_winner: Callable[[int, Sequence[Mapping[str, Any]],
                                   Mapping[str, Any], Mapping[str, Any]], bool],
        ) -> dict[str, Any]:
    """Reopen and certify a terminal PASS without performing any MILP solve.

    Every canonical rank through the winner is reconstructed from the frozen
    plan and caller-supplied pre-outcome preparation.  Durable fixed-rotation
    waves, the materialized allocation, exact-mask classification, and the
    bounded winner objective proof must all agree byte-for-byte.  The returned
    value is the certified allocation manifest.
    """

    plan = validate_search_plan(search_plan)
    result = dict(terminal_result)
    required = {
        "schema", "status", "rank", "combination_attempt_count",
        "allocator_infeasible_combination_count", "rank_receipt",
        "allocation", "winner_validation_receipt", "search_plan_digest",
        "candidate_outcomes_consumed",
    }
    winner_rank = result.get("rank")
    attempt_claim = result.get("combination_attempt_count")
    infeasible_claim = result.get("allocator_infeasible_combination_count")
    if (set(result) != required
            or result.get("schema") != SEARCH_RESULT_SCHEMA
            or result.get("status") != "PASS"
            or isinstance(winner_rank, bool)
            or not isinstance(winner_rank, int)
            or not 0 <= winner_rank < plan["total_rank_count"]
            or isinstance(attempt_claim, bool)
            or not isinstance(attempt_claim, int)
            or attempt_claim != winner_rank + 1
            or isinstance(infeasible_claim, bool)
            or not isinstance(infeasible_claim, int)
            or not 0 <= infeasible_claim <= winner_rank
            or result.get("search_plan_digest") != plan["search_plan_digest"]
            or result.get("candidate_outcomes_consumed") is not False):
        raise ParallelSearchError("terminal scientific PASS surface changed")

    root = Path(checkpoint_root)
    certified_allocation: dict[str, Any] | None = None
    certified_states: list[Mapping[str, Any]] | None = None
    certified_context: Mapping[str, Any] | None = None
    certified_receipt: dict[str, Any] | None = None
    certified_projection = ""
    certified_source = ""
    certified_rotations: list[int] = []
    allocator_infeasible_count = 0

    for rank in range(winner_rank + 1):
        receipt = load_rank_receipt(
            _rank_path(root, rank), search_plan=plan, expected_rank=rank)
        combination = unrank_combination(
            rank, plan["candidate_pool_count"], plan["combination_size"])
        material = dict(prepare_rank(rank, combination))
        if set(material) != {
                "states", "source_identity_manifest_digest", "mask_context"}:
            raise ParallelSearchError("rank preparation surface changed on replay")
        states = list(material["states"])
        source_digest = material["source_identity_manifest_digest"]
        mask_context = material["mask_context"]
        if (not _is_digest(source_digest)
                or not isinstance(mask_context, Mapping)):
            raise ParallelSearchError("rank preparation binding changed on replay")
        normalised = project_allocator_identity_states(states)
        projection_digest = canonical_digest(normalised)
        if (receipt["projection_digest"] != projection_digest
                or receipt["source_identity_manifest_digest"] != source_digest):
            raise ParallelSearchError(
                f"rank {rank} projection/source binding changed")

        classification = receipt["classification"]
        if classification == "ALLOCATOR_INFEASIBLE":
            allocator_infeasible_count += 1
            first_wave = _validate_wave_receipt(
                _load_json(_wave_path(root, rank, 0)), plan=plan, rank=rank,
                projection_digest=projection_digest, expected_prefix=[])
            if first_wave["wave_status"] != "ALLOCATOR_INFEASIBLE":
                raise ParallelSearchError(
                    f"rank {rank} lacks first-wave infeasibility evidence")
            if rank == winner_rank:
                raise ParallelSearchError("terminal winner is allocator-infeasible")
            continue
        if classification not in ("MASK_FAIL", "PASS"):
            raise ParallelSearchError(
                f"rank {rank} has non-scientific terminal classification")

        rotations: list[int] = []
        for state_index in range(PREFIX_STATE_COUNT):
            wave = _validate_wave_receipt(
                _load_json(_wave_path(root, rank, state_index)),
                plan=plan, rank=rank, projection_digest=projection_digest,
                expected_prefix=rotations)
            if wave["wave_status"] != "SELECTED":
                raise ParallelSearchError(
                    f"rank {rank} prefix {state_index} is not selected")
            rotations.append(int(wave["selected_rotation"]))
        if rotations != receipt["selected_rotations"]:
            raise ParallelSearchError(f"rank {rank} rotation vector changed")
        allocation = materialize_allocation_manifest_single_solve(
            states, source_identity_manifest_digest=source_digest,
            rotations=rotations)
        assignment_digest = _candidate_assignment_set_digest(allocation)
        if (allocation.get("allocation_manifest_digest")
                != receipt["provisional_allocation_manifest_digest"]
                or assignment_digest
                != receipt["provisional_candidate_assignment_set_digest"]):
            raise ParallelSearchError(
                f"rank {rank} allocation/assignment digest changed")
        mask_passes = classify_mask(states, allocation, mask_context)
        if not isinstance(mask_passes, bool):
            raise ParallelSearchError(
                "exact-mask classifier did not return a boolean on replay")
        expected_classification = "PASS" if mask_passes else "MASK_FAIL"
        if classification != expected_classification:
            raise ParallelSearchError(
                f"rank {rank} exact-mask classification changed")
        if rank < winner_rank and classification == "PASS":
            raise ParallelSearchError(
                f"rank {rank} is an earlier canonical PASS")
        if rank == winner_rank:
            if classification != "PASS":
                raise ParallelSearchError("terminal winner rank is not PASS")
            certified_allocation = allocation
            certified_states = states
            certified_context = mask_context
            certified_receipt = receipt
            certified_projection = projection_digest
            certified_source = source_digest
            certified_rotations = rotations

    if (certified_allocation is None or certified_states is None
            or certified_context is None or certified_receipt is None):
        raise ParallelSearchError("terminal PASS has no certified winner")
    if infeasible_claim != allocator_infeasible_count:
        raise ParallelSearchError("terminal infeasible-rank count changed")
    if result.get("rank_receipt") != certified_receipt:
        raise ParallelSearchError("embedded terminal rank receipt changed")
    if result.get("allocation") != certified_allocation:
        raise ParallelSearchError("embedded terminal allocation changed")

    objective_prefix: list[int] = []
    objective_wave_rows: list[dict[str, Any]] = []
    for state_index, certified_rotation in enumerate(certified_rotations):
        wave = _validate_objective_wave_receipt(
            _load_json(_objective_wave_path(root, winner_rank, state_index)),
            plan=plan, rank=winner_rank,
            projection_digest=certified_projection,
            source_digest=certified_source,
            expected_prefix=objective_prefix,
            certified_rotation=certified_rotation)
        if wave["solver_status"] != "FEASIBLE":
            raise ParallelSearchError(
                f"winner objective prefix {state_index} is not feasible")
        objective_prefix.append(int(wave["selected_rotation"]))
        objective_wave_rows.append({
            "state_index": state_index,
            "objective_wave_receipt_digest":
                wave["objective_wave_receipt_digest"],
        })
    assignment_digest = _candidate_assignment_set_digest(certified_allocation)
    winner_receipt = _validate_winner_validation_receipt(
        _load_json(root / "winner-objective-validation.json"),
        plan=plan, rank=winner_rank,
        projection_digest=certified_projection,
        source_digest=certified_source,
        allocation_digest=certified_allocation["allocation_manifest_digest"],
        assignment_digest=assignment_digest,
        rotations=certified_rotations,
        objective_wave_rows=objective_wave_rows)
    if result.get("winner_validation_receipt") != winner_receipt:
        raise ParallelSearchError(
            "embedded winner objective-validation receipt changed")
    if validate_winner(
            winner_rank, certified_states, certified_allocation,
            certified_context) is not True:
        raise ParallelSearchError(
            "solve-free structural/scientific winner callback did not pass")
    return certified_allocation


def validate_exhausted_search_result(
        *, terminal_result: Mapping[str, Any],
        search_plan: Mapping[str, Any], checkpoint_root: Path,
        prepare_rank: Callable[[int, tuple[int, ...]], Mapping[str, Any]],
        classify_mask: Callable[[Sequence[Mapping[str, Any]],
                                 Mapping[str, Any], Mapping[str, Any]], bool],
        ) -> list[dict[str, Any]]:
    """Reopen and certify complete search exhaustion without any MILP solve.

    Every plan rank must have a durable scientific rank receipt.  Feasible
    nonpasses reopen all 120 fixed-rotation waves, rematerialize the exact
    allocation bytes, and rerun the solve-free mask predicate.  Allocator
    infeasibility is accepted only from a validated all-infeasible first wave.
    The ordered certified rank receipts are returned for downstream failure
    evidence; a partial frontier can never certify as exhausted.
    """

    if not isinstance(terminal_result, Mapping):
        raise ParallelSearchError("terminal EXHAUSTED result is not a mapping")
    plan = validate_search_plan(search_plan)
    result = dict(terminal_result)
    required = {
        "schema", "status", "combination_attempt_count",
        "allocator_infeasible_combination_count", "search_plan_digest",
        "candidate_outcomes_consumed",
    }
    total_rank_count = plan["total_rank_count"]
    attempt_claim = result.get("combination_attempt_count")
    infeasible_claim = result.get("allocator_infeasible_combination_count")
    if (set(result) != required
            or result.get("schema") != SEARCH_RESULT_SCHEMA
            or result.get("status") != "EXHAUSTED"
            or isinstance(attempt_claim, bool)
            or not isinstance(attempt_claim, int)
            or attempt_claim != total_rank_count
            or isinstance(infeasible_claim, bool)
            or not isinstance(infeasible_claim, int)
            or not 0 <= infeasible_claim <= total_rank_count
            or result.get("search_plan_digest") != plan["search_plan_digest"]
            or result.get("candidate_outcomes_consumed") is not False):
        raise ParallelSearchError("terminal scientific EXHAUSTED surface changed")

    root = Path(checkpoint_root)
    certified_receipts: list[dict[str, Any]] = []
    allocator_infeasible_count = 0
    for rank in range(total_rank_count):
        receipt = load_rank_receipt(
            _rank_path(root, rank), search_plan=plan, expected_rank=rank)
        combination = unrank_combination(
            rank, plan["candidate_pool_count"], plan["combination_size"])
        material = dict(prepare_rank(rank, combination))
        if set(material) != {
                "states", "source_identity_manifest_digest", "mask_context"}:
            raise ParallelSearchError("rank preparation surface changed on replay")
        states = list(material["states"])
        source_digest = material["source_identity_manifest_digest"]
        mask_context = material["mask_context"]
        if (not _is_digest(source_digest)
                or not isinstance(mask_context, Mapping)):
            raise ParallelSearchError("rank preparation binding changed on replay")
        projection_digest = canonical_digest(
            project_allocator_identity_states(states))
        if (receipt["projection_digest"] != projection_digest
                or receipt["source_identity_manifest_digest"] != source_digest):
            raise ParallelSearchError(
                f"rank {rank} projection/source binding changed")

        classification = receipt["classification"]
        if classification == "ALLOCATOR_INFEASIBLE":
            allocator_infeasible_count += 1
            first_wave = _validate_wave_receipt(
                _load_json(_wave_path(root, rank, 0)), plan=plan, rank=rank,
                projection_digest=projection_digest, expected_prefix=[])
            if first_wave["wave_status"] != "ALLOCATOR_INFEASIBLE":
                raise ParallelSearchError(
                    f"rank {rank} lacks first-wave infeasibility evidence")
            certified_receipts.append(receipt)
            continue
        if classification != "MASK_FAIL":
            raise ParallelSearchError(
                f"rank {rank} is not an ordinary scientific nonpass")

        rotations: list[int] = []
        for state_index in range(PREFIX_STATE_COUNT):
            wave = _validate_wave_receipt(
                _load_json(_wave_path(root, rank, state_index)),
                plan=plan, rank=rank, projection_digest=projection_digest,
                expected_prefix=rotations)
            if wave["wave_status"] != "SELECTED":
                raise ParallelSearchError(
                    f"rank {rank} prefix {state_index} is not selected")
            rotations.append(int(wave["selected_rotation"]))
        if rotations != receipt["selected_rotations"]:
            raise ParallelSearchError(f"rank {rank} rotation vector changed")
        allocation = materialize_allocation_manifest_single_solve(
            states, source_identity_manifest_digest=source_digest,
            rotations=rotations)
        assignment_digest = _candidate_assignment_set_digest(allocation)
        if (allocation.get("allocation_manifest_digest")
                != receipt["provisional_allocation_manifest_digest"]
                or assignment_digest
                != receipt["provisional_candidate_assignment_set_digest"]):
            raise ParallelSearchError(
                f"rank {rank} allocation/assignment digest changed")
        mask_passes = classify_mask(states, allocation, mask_context)
        if not isinstance(mask_passes, bool):
            raise ParallelSearchError(
                "exact-mask classifier did not return a boolean on replay")
        if mask_passes:
            raise ParallelSearchError(
                f"rank {rank} is a canonical PASS, not an exhausted nonpass")
        certified_receipts.append(receipt)

    if len(certified_receipts) != total_rank_count:
        raise ParallelSearchError("terminal EXHAUSTED rank frontier is partial")
    if infeasible_claim != allocator_infeasible_count:
        raise ParallelSearchError("terminal infeasible-rank count changed")
    return certified_receipts


def benchmark_fixed_rotation_gate(
        *, serial_work_units: int | None = None,
        parallel_work_units: int | None = None,
        maximum_parallel_fraction: float = 0.5,
        serial_elapsed_s: float | None = None,
        parallel_elapsed_s: float | None = None,
        source_binding_digest: str | None = None,
        details: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Build a benchmark verdict; runtime authority requires measured seconds."""

    if serial_elapsed_s is not None or parallel_elapsed_s is not None:
        if (not isinstance(serial_elapsed_s, (int, float))
                or not isinstance(parallel_elapsed_s, (int, float))
                or serial_elapsed_s <= 0 or parallel_elapsed_s < 0):
            raise ValueError("measured benchmark seconds are invalid")
        ratio = float(parallel_elapsed_s) / float(serial_elapsed_s)
        measured = True
    else:
        if (not isinstance(serial_work_units, int)
                or not isinstance(parallel_work_units, int)
                or serial_work_units <= 0 or parallel_work_units < 0):
            raise ValueError("benchmark work units are invalid")
        ratio = parallel_work_units / serial_work_units
        measured = False
    payload = {
        "schema": BENCHMARK_RECEIPT_SCHEMA,
        "measured_wall_clock": measured,
        "serial_elapsed_s": serial_elapsed_s,
        "parallel_elapsed_s": parallel_elapsed_s,
        "serial_work_units": serial_work_units,
        "parallel_work_units": parallel_work_units,
        "observed_parallel_fraction": ratio,
        "maximum_parallel_fraction": maximum_parallel_fraction,
        "passes": ratio <= maximum_parallel_fraction,
        "source_binding_digest": source_binding_digest,
        "details": {} if details is None else dict(details),
        "candidate_outcomes_consumed": False,
    }
    payload["benchmark_receipt_digest"] = canonical_digest(payload)
    return payload


def run_measured_fixed_rotation_benchmark(
        *, states: Sequence[Mapping[str, Any]], search_plan: Mapping[str, Any],
        source_binding_digest: str,
        sample_prefix_indices: Sequence[int] = (0, 1, 2),
        maximum_parallel_fraction: float = 0.5,
        executor_factory: Callable[..., Any] | None = None,
        ) -> dict[str, Any]:
    """Bounded same-matrix prefix-wave microbenchmark for the runtime gate.

    The preregistered sample compares one old exact objective solve ``C`` with
    the corresponding 12 fixed-rotation solves executed concurrently ``F``.
    Both arms use the exact same 120-state constraint matrix, fixed warm prefix,
    and plan-bound options.  It neither emits selector evidence nor evaluates a
    combination mask.  The gate requires both median(F/C) and max(F/C) <= 0.5.
    """

    plan = validate_search_plan(search_plan)
    if not _is_digest(source_binding_digest):
        raise ValueError("benchmark source binding digest is invalid")
    samples = list(sample_prefix_indices)
    if (not samples or len(samples) > 3
            or any(isinstance(value, bool) or not isinstance(value, int)
                   or not 0 <= value < PREFIX_STATE_COUNT for value in samples)
            or samples != sorted(set(samples))):
        raise ValueError("benchmark prefix sample must be 1..3 sorted unique indices")
    normalised = project_allocator_identity_states(states)
    source_digest = ALLOC.pre_outcome_identity_digest(normalised)
    from scipy.optimize import Bounds, milp
    _worker_initialise()
    constraints, base_bounds = ALLOC._constraint_system(normalised)
    variable_count = len(normalised) * ROTATION_COUNT
    integrality = np.ones(variable_count, dtype=np.uint8)
    lower = np.asarray(base_bounds.lb, dtype=np.float64).copy()
    upper = np.asarray(base_bounds.ub, dtype=np.float64).copy()
    options = dict(plan["solver_options"])
    factory = (concurrent.futures.ProcessPoolExecutor
               if executor_factory is None else executor_factory)
    process_pool = factory(
        max_workers=plan["worker_count"], initializer=_worker_initialise,
        mp_context=multiprocessing.get_context("spawn"))
    sample_rows: list[dict[str, Any]] = []
    selected_prefix: list[int] = []
    try:
        for state_index in range(max(samples) + 1):
            start = state_index * ROTATION_COUNT
            objective = np.zeros(variable_count, dtype=np.float64)
            objective[start:start + ROTATION_COUNT] = np.arange(
                ROTATION_COUNT, dtype=np.float64)
            control_started = time.monotonic()
            result = milp(
                c=objective, integrality=integrality,
                bounds=Bounds(lower, upper), constraints=constraints,
                options=options)
            control_elapsed = time.monotonic() - control_started
            status, message = _classify_fixed_rotation_result(
                result, constraints=constraints, lower=lower, upper=upper,
                fixed_variable_index=None)
            if status != "FEASIBLE" or result.fun is None:
                raise ParallelSearchFatal(
                    f"benchmark objective prefix {state_index} failed: {message}")
            local = np.asarray(
                result.x[start:start + ROTATION_COUNT], dtype=np.float64)
            objective_rotation = int(np.argmax(local))
            if abs(float(result.fun) - objective_rotation) > 1e-6:
                raise ParallelSearchFatal("benchmark objective choice is inconsistent")
            if state_index in samples:
                parallel_started = time.monotonic()
                futures = [process_pool.submit(_fixed_rotation_worker, {
                    "states": normalised,
                    "prefix_rotations": selected_prefix,
                    "state_index": state_index,
                    "rotation": rotation,
                    "solver_options": options,
                }) for rotation in range(ROTATION_COUNT)]
                fixed_rows = sorted(
                    (future.result() for future in futures),
                    key=lambda row: int(row["rotation"]))
                parallel_elapsed = time.monotonic() - parallel_started
                fixed_status, fixed_rotation = _lexicographic_wave_decision(
                    [row["status"] for row in fixed_rows],
                    state_index=state_index)
                if fixed_status != "SELECTED":
                    fatal = next(
                        (row for row in fixed_rows
                         if row["status"] == "FATAL"),
                        {"message": "fixed-r wave has no certified minimum"})
                    raise ParallelSearchFatal(
                        f"benchmark fixed-r prefix {state_index} failed: "
                        f"{fatal['message']}")
                if fixed_rotation != objective_rotation:
                    raise ParallelSearchFatal(
                        "benchmark fixed-r choice differs from objective choice")
                ratio = parallel_elapsed / control_elapsed
                sample_rows.append({
                    "state_index": state_index,
                    "prefix_rotations_digest": canonical_digest(selected_prefix),
                    "objective_rotation": objective_rotation,
                    "fixed_rotation": fixed_rotation,
                    "objective_elapsed_s": control_elapsed,
                    "fixed_wave_elapsed_s": parallel_elapsed,
                    "parallel_fraction": ratio,
                    "fixed_solver_call_count": ROTATION_COUNT,
                    "objective_solver_call_count": 1,
                })
            selected_prefix.append(objective_rotation)
            lower[start:start + ROTATION_COUNT] = 0.0
            upper[start:start + ROTATION_COUNT] = 0.0
            lower[start + objective_rotation] = 1.0
            upper[start + objective_rotation] = 1.0
    finally:
        process_pool.shutdown(wait=True, cancel_futures=True)
    fractions = [float(row["parallel_fraction"]) for row in sample_rows]
    median_fraction = float(np.median(np.asarray(fractions)))
    maximum_fraction = max(fractions)
    serial_elapsed = sum(float(row["objective_elapsed_s"])
                         for row in sample_rows)
    parallel_elapsed = sum(float(row["fixed_wave_elapsed_s"])
                           for row in sample_rows)
    receipt = benchmark_fixed_rotation_gate(
        serial_elapsed_s=serial_elapsed,
        parallel_elapsed_s=parallel_elapsed,
        maximum_parallel_fraction=maximum_parallel_fraction,
        source_binding_digest=source_binding_digest,
        details={
            "sample_prefix_indices": samples,
            "sample_prefix_count": len(samples),
            "sample_rows": sample_rows,
            "sample_rows_digest": canonical_digest(sample_rows),
            "median_parallel_fraction": median_fraction,
            "maximum_parallel_fraction_observed": maximum_fraction,
            "state_projection_digest": canonical_digest(normalised),
            "source_identity_manifest_digest": source_digest,
            "worker_count": plan["worker_count"],
            "worker_threads": 1,
            "solver_time_limit_s": plan["solver_options"]["time_limit"],
            "allocation_vectors_equal": True,
            "temporary_receipts_removed": True,
            "operational_benchmark_not_selector_input": True,
        })
    # The aggregate ratio is descriptive; preregistered gate is median AND max.
    receipt["passes"] = (median_fraction <= maximum_parallel_fraction
                         and maximum_fraction <= maximum_parallel_fraction)
    receipt["benchmark_receipt_digest"] = canonical_digest(
        _without_digest(receipt, "benchmark_receipt_digest"))
    return receipt


def require_measured_benchmark_gate(payload: Mapping[str, Any], *,
                                    expected_source_binding_digest: str,
                                    maximum_parallel_fraction: float = 0.5
                                    ) -> dict[str, Any]:
    receipt = dict(payload)
    required = {
        "schema", "measured_wall_clock", "serial_elapsed_s",
        "parallel_elapsed_s", "serial_work_units", "parallel_work_units",
        "observed_parallel_fraction", "maximum_parallel_fraction", "passes",
        "source_binding_digest", "details", "candidate_outcomes_consumed",
        "benchmark_receipt_digest",
    }
    required_details = {
        "sample_prefix_indices", "sample_prefix_count", "sample_rows",
        "sample_rows_digest", "median_parallel_fraction",
        "maximum_parallel_fraction_observed", "state_projection_digest",
        "source_identity_manifest_digest", "worker_count", "worker_threads",
        "solver_time_limit_s", "allocation_vectors_equal",
        "temporary_receipts_removed",
        "operational_benchmark_not_selector_input",
    }
    required_row = {
        "state_index", "prefix_rotations_digest", "objective_rotation",
        "fixed_rotation", "objective_elapsed_s", "fixed_wave_elapsed_s",
        "parallel_fraction", "fixed_solver_call_count",
        "objective_solver_call_count",
    }
    serial_elapsed = receipt.get("serial_elapsed_s")
    parallel_elapsed = receipt.get("parallel_elapsed_s")
    details = receipt.get("details")
    rows = details.get("sample_rows") if isinstance(details, dict) else None
    row_contract_passes = isinstance(rows, list) and len(rows) == 3
    fractions: list[float] = []
    objective_seconds: list[float] = []
    fixed_seconds: list[float] = []
    expected_prefix: list[int] = []
    if row_contract_passes:
        for expected_index, row in zip((0, 1, 2), rows, strict=True):
            objective_elapsed = row.get("objective_elapsed_s") \
                if isinstance(row, dict) else None
            fixed_elapsed = row.get("fixed_wave_elapsed_s") \
                if isinstance(row, dict) else None
            fraction = row.get("parallel_fraction") \
                if isinstance(row, dict) else None
            objective_rotation = row.get("objective_rotation") \
                if isinstance(row, dict) else None
            fixed_rotation = row.get("fixed_rotation") \
                if isinstance(row, dict) else None
            numeric = (
                isinstance(objective_elapsed, (int, float))
                and not isinstance(objective_elapsed, bool)
                and math.isfinite(float(objective_elapsed))
                and objective_elapsed > 0
                and isinstance(fixed_elapsed, (int, float))
                and not isinstance(fixed_elapsed, bool)
                and math.isfinite(float(fixed_elapsed))
                and fixed_elapsed >= 0
                and isinstance(fraction, (int, float))
                and not isinstance(fraction, bool)
                and math.isfinite(float(fraction))
            )
            row_contract_passes = bool(
                row_contract_passes and isinstance(row, dict)
                and set(row) == required_row
                and row.get("state_index") == expected_index
                and row.get("prefix_rotations_digest")
                == canonical_digest(expected_prefix)
                and isinstance(objective_rotation, int)
                and not isinstance(objective_rotation, bool)
                and 0 <= objective_rotation < ROTATION_COUNT
                and fixed_rotation == objective_rotation
                and numeric
                and abs(float(fraction)
                        - float(fixed_elapsed) / float(objective_elapsed)) <= 1e-12
                and row.get("fixed_solver_call_count") == ROTATION_COUNT
                and row.get("objective_solver_call_count") == 1
            )
            if not row_contract_passes:
                break
            fractions.append(float(fraction))
            objective_seconds.append(float(objective_elapsed))
            fixed_seconds.append(float(fixed_elapsed))
            expected_prefix.append(int(objective_rotation))
    recomputed_median = (float(np.median(np.asarray(fractions)))
                         if len(fractions) == 3 else math.inf)
    recomputed_maximum = max(fractions) if len(fractions) == 3 else math.inf
    recomputed_serial = sum(objective_seconds)
    recomputed_parallel = sum(fixed_seconds)
    if (set(receipt) != required
            or receipt.get("schema") != BENCHMARK_RECEIPT_SCHEMA
            or receipt.get("measured_wall_clock") is not True
            or not isinstance(serial_elapsed, (int, float))
            or isinstance(serial_elapsed, bool)
            or not math.isfinite(float(serial_elapsed))
            or serial_elapsed <= 0
            or not isinstance(parallel_elapsed, (int, float))
            or isinstance(parallel_elapsed, bool)
            or not math.isfinite(float(parallel_elapsed))
            or parallel_elapsed < 0
            or receipt.get("serial_work_units") is not None
            or receipt.get("parallel_work_units") is not None
            or not isinstance(receipt.get("observed_parallel_fraction"),
                              (int, float))
            or isinstance(receipt.get("observed_parallel_fraction"), bool)
            or abs(receipt.get("observed_parallel_fraction", math.inf)
                   - parallel_elapsed / serial_elapsed) > 1e-12
            or receipt.get("source_binding_digest")
            != expected_source_binding_digest
            or receipt.get("maximum_parallel_fraction")
            != maximum_parallel_fraction
            or receipt.get("passes") is not True
            or not isinstance(details, dict)
            or set(details) != required_details
            or details.get("allocation_vectors_equal") is not True
            or details.get("temporary_receipts_removed") is not True
            or details.get("worker_threads") != 1
            or details.get("worker_count") != 32
            or details.get("sample_prefix_indices") != [0, 1, 2]
            or details.get("sample_prefix_count") != 3
            or not row_contract_passes
            or details.get("sample_rows_digest")
            != canonical_digest(rows)
            or not _is_digest(details.get("state_projection_digest"))
            or not _is_digest(details.get("source_identity_manifest_digest"))
            or not isinstance(details.get("solver_time_limit_s"), (int, float))
            or isinstance(details.get("solver_time_limit_s"), bool)
            or not math.isfinite(float(details.get("solver_time_limit_s", -1)))
            or details.get("solver_time_limit_s") <= 0
            or not isinstance(details.get("median_parallel_fraction"),
                              (int, float))
            or isinstance(details.get("median_parallel_fraction"), bool)
            or not math.isfinite(float(details["median_parallel_fraction"]))
            or abs(float(details["median_parallel_fraction"])
                   - recomputed_median) > 1e-12
            or not isinstance(details.get("maximum_parallel_fraction_observed"),
                              (int, float))
            or isinstance(details.get("maximum_parallel_fraction_observed"),
                          bool)
            or not math.isfinite(float(
                details["maximum_parallel_fraction_observed"]))
            or abs(float(details["maximum_parallel_fraction_observed"])
                   - recomputed_maximum) > 1e-12
            or abs(float(serial_elapsed) - recomputed_serial) > 1e-12
            or abs(float(parallel_elapsed) - recomputed_parallel) > 1e-12
            or recomputed_median > maximum_parallel_fraction
            or recomputed_maximum > maximum_parallel_fraction
            or details.get("operational_benchmark_not_selector_input") is not True
            or receipt.get("candidate_outcomes_consumed") is not False
            or receipt.get("benchmark_receipt_digest") != canonical_digest(
                _without_digest(receipt, "benchmark_receipt_digest"))):
        raise ParallelSearchError(
            "measured fixed-rotation benchmark gate is unavailable or failed")
    return receipt


__all__ = [
    "ALGORITHM_VERSION", "ALLOCATOR_IDENTITY_FIELDS",
    "BENCHMARK_RECEIPT_SCHEMA",
    "COORDINATOR_RECEIPT_SCHEMA", "OrderedFrontier", "ParallelSearchError",
    "DEFAULT_MILP_TIME_LIMIT_S", "OBJECTIVE_WAVE_RECEIPT_SCHEMA",
    "ParallelSearchFatal", "RANK_RECEIPT_SCHEMA", "SEARCH_PLAN_SCHEMA",
    "SEARCH_RESULT_SCHEMA", "WAVE_RECEIPT_SCHEMA",
    "WINNER_VALIDATION_RECEIPT_SCHEMA", "benchmark_fixed_rotation_gate",
    "build_search_plan", "canonical_digest", "combination_count",
    "contiguous_partitions", "fixed_rotation_lexicographic_rotations",
    "load_coordinator_receipt", "load_rank_receipt",
    "materialize_allocation_manifest_single_solve",
    "parallel_lexicographic_rotations",
    "project_allocator_identity_states",
    "rank_combination", "require_measured_benchmark_gate",
    "run_measured_fixed_rotation_benchmark", "run_parallel_search",
    "run_scientific_parallel_search", "unrank_combination",
    "validate_exhausted_search_result", "validate_search_plan",
    "validate_terminal_search_result",
    "validate_winner_allocation_bounded",
    "write_coordinator_receipt", "write_rank_receipt",
]
