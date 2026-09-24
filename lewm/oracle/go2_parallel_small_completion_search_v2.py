"""One-shot warm-pool successor for the V1 small-completion benchmark.

V2 changes only process-pool readiness.  Thirty-two identical, non-solving
tasks first import the SciPy/HiGHS path and construct the immutable projected
allocation constraint input.  Each task announces a distinct worker identity
and then blocks on one coordinator release event, so all 32 spawn workers must
be simultaneously ready before any timed sample can start.  The exact live
pool is retained for samples 0/1/2 and, only after both unchanged gates pass,
for the V1 scientific search.

This module has no branch-outcome or scientific-mask input in its readiness or
benchmark APIs.  Scientific callbacks are accepted only by the explicitly
PASS-gated continuation function.
"""
from __future__ import annotations

import concurrent.futures
import math
import multiprocessing
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np

from lewm.oracle import go2_parallel_small_completion_search_v1 as V1


SCHEMA_VERSION = "go2_parallel_small_completion_search_v2"
READINESS_RECORD_SCHEMA = f"{SCHEMA_VERSION}_readiness_record"
BENCHMARK_RECEIPT_SCHEMA = f"{SCHEMA_VERSION}_benchmark_receipt"
POOL_CONTINUATION_SCHEMA = f"{SCHEMA_VERSION}_pool_continuation"
BENCHMARK_CONTRACT_SCHEMA = "go2_parallel_small_completion_benchmark_v2_contract"
READINESS_ALGORITHM = (
    "spawn32_identical_manager_queue_event_barrier_scipy_highs_"
    "projected_constraint_input_no_solve_v2"
)
READINESS_TASK_SYMBOL = (
    "lewm.oracle.go2_parallel_small_completion_search_v2."
    "_v2_readiness_worker"
)
WORKER_INITIALIZER_SYMBOL = (
    "lewm.oracle.go2_parallel_small_completion_search_v2."
    "_v2_worker_initialise"
)
FIXED_WORKER_SYMBOL = (
    "lewm.oracle.go2_parallel_small_completion_search_v2."
    "_v2_fixed_rotation_worker"
)
V1_FAILURE_DISPOSITION = (
    "IMMUTABLE_FAIL_COLD_START_INCLUDED_IN_FIRST_TIMED_WAVE"
)
WORKER_COUNT = 32
SAMPLE_PREFIX_INDICES = (0, 1, 2)
MAXIMUM_PARALLEL_FRACTION = 0.5
DEFAULT_READINESS_TIMEOUT_S = 300.0

_V2_WORKER_INSTANCE_ID: str | None = None


class WorkerPoolIntegrityError(V1.ParallelSearchFatal):
    """The one live V2 worker generation changed after sample 0 began."""


class WorkerReadinessError(V1.ParallelSearchFatal):
    """A complete 32-worker readiness barrier could not be certified."""

    def __init__(self, message: str, *, attempt_rows: Sequence[Mapping[str, Any]]
                 | None = None) -> None:
        super().__init__(message)
        self.attempt_rows = [dict(row) for row in (attempt_rows or [])]


def _is_digest(value: Any, *, length: int = 64) -> bool:
    return V1._is_digest(value, length=length)


def immutable_search_input_digest(plan: Mapping[str, Any]) -> str:
    """Bind every plan field except the post-gate receipt and self digest."""

    validated = V1.validate_search_plan(plan)
    projection = {
        key: value for key, value in validated.items()
        if key not in {"measured_benchmark_receipt_digest", "search_plan_digest"}
    }
    return V1.canonical_digest(projection)


def _strict_projected_states(
        states: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    if (isinstance(states, (str, bytes))
            or not isinstance(states, Sequence)
            or len(states) != V1.PREFIX_STATE_COUNT):
        raise ValueError("V2 readiness requires exactly 120 projected states")
    expected = set(V1.ALLOCATOR_IDENTITY_FIELDS)
    for index, state in enumerate(states):
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError(
                f"V2 readiness state {index} is not the exact six-field "
                "outcome-free allocator projection")
    return V1.project_allocator_identity_states(states)


def _readiness_public_task(
        *, states: Sequence[Mapping[str, Any]], plan: Mapping[str, Any],
        benchmark_v2_contract_digest: str,
        predecessor_scientific_input_bindings_digest: str,
        ) -> dict[str, Any]:
    projected = _strict_projected_states(states)
    validated = V1.validate_search_plan(plan)
    if validated["worker_count"] != WORKER_COUNT:
        raise ValueError("V2 requires exactly 32 solver workers")
    if not _is_digest(benchmark_v2_contract_digest):
        raise ValueError("V2 benchmark contract digest is invalid")
    if not _is_digest(predecessor_scientific_input_bindings_digest):
        raise ValueError("predecessor scientific input binding is invalid")
    return {
        "readiness_algorithm": READINESS_ALGORITHM,
        "readiness_task_symbol": READINESS_TASK_SYMBOL,
        "worker_initializer_symbol": WORKER_INITIALIZER_SYMBOL,
        "fixed_worker_symbol": FIXED_WORKER_SYMBOL,
        "benchmark_v2_contract_digest": benchmark_v2_contract_digest,
        "predecessor_scientific_input_bindings_digest":
            predecessor_scientific_input_bindings_digest,
        "immutable_search_input_digest": immutable_search_input_digest(validated),
        "state_projection_digest": V1.canonical_digest(projected),
        "source_identity_manifest_digest":
            V1.ALLOC.pre_outcome_identity_digest(projected),
        "state_count": len(projected),
        "variable_count": len(projected) * V1.ROTATION_COUNT,
        "solver_options": dict(validated["solver_options"]),
        "solver_options_digest": V1.canonical_digest(
            validated["solver_options"]),
        "thread_environment": dict(V1.THREAD_ENVIRONMENT),
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
        "solver_call_count": 0,
    }


def _v2_worker_initialise() -> None:
    global _V2_WORKER_INSTANCE_ID
    V1._worker_initialise()
    _V2_WORKER_INSTANCE_ID = uuid.uuid4().hex


def _readiness_announcement(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "worker_pid", "worker_instance_id", "readiness_task_digest",
        "predecessor_scientific_input_bindings_digest",
        "state_projection_digest", "source_identity_manifest_digest",
        "solver_options_digest", "immutable_search_input_digest",
        "thread_environment", "solver_module", "solver_backend",
        "solver_version", "constraint_row_count", "variable_count",
        "bounds_variable_count", "solver_imported", "immutable_inputs_loaded",
        "readiness_barrier_reached", "candidate_outcomes_consumed",
        "scientific_masks_accessed", "solver_call_count",
    }
    announcement = {key: row[key] for key in fields}
    announcement["readiness_announcement_digest"] = V1.canonical_digest(
        announcement)
    return announcement


def _v2_readiness_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Import/build only, announce readiness, then block until all 32 arrive."""

    started = time.monotonic()
    expected_keys = {
        "public_task", "projected_states", "announcement_queue",
        "release_event", "readiness_timeout_s",
    }
    if set(task) != expected_keys:
        raise WorkerReadinessError("readiness worker task surface changed")
    public_task = dict(task["public_task"])
    projected = _strict_projected_states(task["projected_states"])
    if (public_task.get("readiness_algorithm") != READINESS_ALGORITHM
            or public_task.get("readiness_task_symbol")
            != READINESS_TASK_SYMBOL
            or public_task.get("worker_initializer_symbol")
            != WORKER_INITIALIZER_SYMBOL
            or public_task.get("fixed_worker_symbol") != FIXED_WORKER_SYMBOL
            or public_task.get("state_projection_digest")
            != V1.canonical_digest(projected)
            or public_task.get("source_identity_manifest_digest")
            != V1.ALLOC.pre_outcome_identity_digest(projected)
            or public_task.get("state_count") != V1.PREFIX_STATE_COUNT
            or public_task.get("variable_count")
            != V1.PREFIX_STATE_COUNT * V1.ROTATION_COUNT
            or public_task.get("solver_options_digest")
            != V1.canonical_digest(public_task.get("solver_options"))
            or public_task.get("thread_environment") != V1.THREAD_ENVIRONMENT
            or public_task.get("candidate_outcomes_consumed") is not False
            or public_task.get("scientific_masks_accessed") is not False
            or public_task.get("solver_call_count") != 0):
        raise WorkerReadinessError("readiness public input binding changed")
    if (_V2_WORKER_INSTANCE_ID is None
            or len(_V2_WORKER_INSTANCE_ID) != 32):
        raise WorkerReadinessError("V2 worker initializer did not run")

    import scipy
    from scipy.optimize import milp

    if not callable(milp):
        raise WorkerReadinessError("SciPy MILP entry point is unavailable")
    constraints, bounds = V1.ALLOC._constraint_system(projected)
    constraint_rows = int(np.asarray(constraints.lb).reshape(-1).shape[0])
    variable_count = int(np.asarray(bounds.lb).reshape(-1).shape[0])
    upper_count = int(np.asarray(bounds.ub).reshape(-1).shape[0])
    if (variable_count != public_task["variable_count"]
            or upper_count != variable_count):
        raise WorkerReadinessError("immutable constraint input shape changed")

    row: dict[str, Any] = {
        "worker_pid": os.getpid(),
        "worker_instance_id": _V2_WORKER_INSTANCE_ID,
        "readiness_task_digest": V1.canonical_digest(public_task),
        "predecessor_scientific_input_bindings_digest": public_task[
            "predecessor_scientific_input_bindings_digest"],
        "state_projection_digest": public_task["state_projection_digest"],
        "source_identity_manifest_digest":
            public_task["source_identity_manifest_digest"],
        "solver_options_digest": public_task["solver_options_digest"],
        "immutable_search_input_digest":
            public_task["immutable_search_input_digest"],
        "thread_environment": {
            key: os.environ.get(key) for key in V1.THREAD_ENVIRONMENT},
        "solver_module": str(milp.__module__),
        "solver_backend": "scipy.optimize.milp/HiGHS",
        "solver_version": str(scipy.__version__),
        "constraint_row_count": constraint_rows,
        "variable_count": variable_count,
        "bounds_variable_count": upper_count,
        "solver_imported": True,
        "immutable_inputs_loaded": True,
        "readiness_barrier_reached": True,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
        "solver_call_count": 0,
    }
    task["announcement_queue"].put(_readiness_announcement(row))
    timeout = task["readiness_timeout_s"]
    if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
            or not math.isfinite(float(timeout)) or timeout <= 0
            or not task["release_event"].wait(float(timeout))):
        raise WorkerReadinessError("32-worker readiness release timed out")
    row["readiness_task_completed"] = True
    row["readiness_elapsed_s"] = time.monotonic() - started
    row["worker_readiness_digest"] = V1.canonical_digest(row)
    return row


def _v2_fixed_rotation_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    if _V2_WORKER_INSTANCE_ID is None:
        raise WorkerPoolIntegrityError("fixed task ran outside V2 pool generation")
    result = V1._fixed_rotation_worker(task)
    return {
        "worker_pid": os.getpid(),
        "worker_instance_id": _V2_WORKER_INSTANCE_ID,
        "worker_result": result,
    }


class _ManagerCoordination:
    def __init__(self, context: Any) -> None:
        self.manager = context.Manager()
        self.announcement_queue = self.manager.Queue()
        self.release_event = self.manager.Event()

    def close(self) -> None:
        self.manager.shutdown()


def _process_snapshot(executor: Any) -> list[dict[str, Any]]:
    custom = getattr(executor, "v2_process_snapshot", None)
    if callable(custom):
        rows = custom()
    else:
        processes = getattr(executor, "_processes", None)
        if not isinstance(processes, dict):
            raise WorkerPoolIntegrityError(
                "process executor does not expose a certifiable PID set")
        rows = [{
            "worker_pid": process.pid,
            "is_alive": bool(process.is_alive()),
            "exitcode": process.exitcode,
        } for process in processes.values()]
    normalised = [dict(row) for row in rows]
    if any(set(row) != {"worker_pid", "is_alive", "exitcode"}
           or not isinstance(row["worker_pid"], int)
           or row["worker_pid"] <= 0
           or not isinstance(row["is_alive"], bool)
           for row in normalised):
        raise WorkerPoolIntegrityError("worker process snapshot is malformed")
    return sorted(normalised, key=lambda row: row["worker_pid"])


class _ValidatedFuture(concurrent.futures.Future[Any]):
    def __init__(self, inner: concurrent.futures.Future[Any],
                 pool: "ReadyWorkerPoolV2") -> None:
        super().__init__()
        self._inner = inner
        self._pool = pool
        inner.add_done_callback(self._finish)

    def _finish(self, future: concurrent.futures.Future[Any]) -> None:
        if self.cancelled():
            return
        if future.cancelled():
            super().cancel()
            return
        try:
            envelope = future.result()
            result = self._pool._validate_fixed_envelope(envelope)
        except WorkerPoolIntegrityError as exc:
            self._pool._integrity_failed = True
            self.set_exception(exc)
        except BaseException as exc:
            self._pool._integrity_failed = True
            failure = WorkerPoolIntegrityError(
                "V2 fixed task failed outside the unchanged worker result "
                "surface")
            failure.__cause__ = exc
            self.set_exception(failure)
        else:
            self.set_result(result)

    def cancel(self) -> bool:
        if self._inner.cancel():
            return super().cancel()
        return False


class ReadyWorkerPoolV2:
    """Externally owned, integrity-guarded live V2 process pool."""

    def __init__(self, *, executor: Any, readiness_record: Mapping[str, Any],
                 readiness_completed_monotonic: float,
                 projected_states: Sequence[Mapping[str, Any]],
                 process_snapshot: Sequence[Mapping[str, Any]]) -> None:
        self._executor = executor
        self.readiness_record = dict(readiness_record)
        self.readiness_completed_monotonic = readiness_completed_monotonic
        self.projected_states = [dict(row) for row in projected_states]
        self._baseline_snapshot = [dict(row) for row in process_snapshot]
        self._identities = {
            (row["worker_pid"], row["worker_instance_id"])
            for row in self.readiness_record["worker_rows"]
        }
        self._lock = threading.Lock()
        self._observed_pairs: list[tuple[int, str]] = []
        self.sample0_started = False
        self.search_started = False
        self.closed = False
        self.worker_restart_count = 0
        self._integrity_failed = False

    @property
    def worker_pool_identity(self) -> str:
        return self.readiness_record["worker_pool_identity"]

    @property
    def benchmark_v2_contract_digest(self) -> str:
        return self.readiness_record["benchmark_v2_contract_digest"]

    @property
    def immutable_search_input_digest(self) -> str:
        return self.readiness_record["immutable_search_input_digest"]

    @property
    def predecessor_scientific_input_bindings_digest(self) -> str:
        return self.readiness_record[
            "predecessor_scientific_input_bindings_digest"]

    @classmethod
    def create(
            cls, *, states: Sequence[Mapping[str, Any]],
            search_plan: Mapping[str, Any],
            benchmark_v2_contract_digest: str,
            predecessor_scientific_input_bindings_digest: str,
            executor_factory: Callable[..., Any] | None = None,
            coordination_factory: Callable[[Any], Any] | None = None,
            readiness_timeout_s: float = DEFAULT_READINESS_TIMEOUT_S,
            clock: Callable[[], float] = time.monotonic,
            ) -> "ReadyWorkerPoolV2":
        plan = V1.validate_search_plan(search_plan)
        projected = _strict_projected_states(states)
        public_task = _readiness_public_task(
            states=projected, plan=plan,
            benchmark_v2_contract_digest=benchmark_v2_contract_digest,
            predecessor_scientific_input_bindings_digest=
                predecessor_scientific_input_bindings_digest)
        if (not isinstance(readiness_timeout_s, (int, float))
                or isinstance(readiness_timeout_s, bool)
                or not math.isfinite(float(readiness_timeout_s))
                or readiness_timeout_s <= 0):
            raise ValueError("readiness timeout must be finite and positive")
        context = multiprocessing.get_context("spawn")
        factory = (concurrent.futures.ProcessPoolExecutor
                   if executor_factory is None else executor_factory)
        coordinate = (_ManagerCoordination
                      if coordination_factory is None else coordination_factory)
        attempts: list[dict[str, Any]] = []
        last_error: BaseException | None = None
        for generation in range(2):
            attempt_started = clock()
            executor: Any | None = None
            coordination: Any | None = None
            observed_pids: list[int] = []
            try:
                executor = factory(
                    max_workers=WORKER_COUNT, initializer=_v2_worker_initialise,
                    mp_context=context)
                coordination = coordinate(context)
                task = {
                    "public_task": public_task,
                    "projected_states": projected,
                    "announcement_queue": coordination.announcement_queue,
                    "release_event": coordination.release_event,
                    "readiness_timeout_s": float(readiness_timeout_s),
                }
                futures = [executor.submit(_v2_readiness_worker, task)
                           for _ in range(WORKER_COUNT)]
                announcements: list[dict[str, Any]] = []
                deadline = time.monotonic() + float(readiness_timeout_s)
                while len(announcements) < WORKER_COUNT:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise WorkerReadinessError(
                            "32-worker readiness announcements timed out")
                    try:
                        announcement = coordination.announcement_queue.get(
                            timeout=min(0.25, remaining))
                    except queue.Empty:
                        failures = [future.exception()
                                    for future in futures if future.done()
                                    and not future.cancelled()]
                        failures = [error for error in failures
                                    if error is not None]
                        if failures:
                            raise WorkerReadinessError(
                                "readiness worker failed before the barrier") \
                                from failures[0]
                        continue
                    checked = _validate_readiness_announcement(
                        announcement, public_task=public_task)
                    announcements.append(checked)
                    observed_pids.append(checked["worker_pid"])
                    pairs = {(row["worker_pid"], row["worker_instance_id"])
                             for row in announcements}
                    if len(pairs) != len(announcements):
                        raise WorkerReadinessError(
                            "readiness barrier did not reach distinct workers")
                coordination.release_event.set()
                worker_rows = [future.result(timeout=readiness_timeout_s)
                               for future in futures]
                worker_rows = sorted(
                    (_validate_worker_row(row, public_task=public_task)
                     for row in worker_rows),
                    key=lambda row: (row["worker_pid"],
                                     row["worker_instance_id"]))
                identities = {(row["worker_pid"], row["worker_instance_id"])
                              for row in worker_rows}
                announced = {(row["worker_pid"], row["worker_instance_id"])
                             for row in announcements}
                if len(identities) != WORKER_COUNT or identities != announced:
                    raise WorkerReadinessError(
                        "readiness return identities differ from announcements")
                snapshot = _process_snapshot(executor)
                snapshot_pids = {row["worker_pid"] for row in snapshot}
                if (len(snapshot) != WORKER_COUNT
                        or any(not row["is_alive"] or row["exitcode"] is not None
                               for row in snapshot)
                        or snapshot_pids != {pid for pid, _ in identities}):
                    raise WorkerReadinessError(
                        "ready worker identities differ from live PID set")
                completed = clock()
                attempts.append({
                    "pool_generation": generation,
                    "status": "READY",
                    "attempt_elapsed_s": completed - attempt_started,
                    "observed_worker_pids": sorted(snapshot_pids),
                    "error_type": None,
                    "error_message": None,
                })
                record = _build_readiness_record(
                    public_task=public_task, worker_rows=worker_rows,
                    process_snapshot=snapshot, attempt_rows=attempts,
                    successful_attempt_elapsed_s=completed - attempt_started,
                    total_elapsed_s=sum(row["attempt_elapsed_s"]
                                      for row in attempts),
                    generation=generation)
                return cls(
                    executor=executor, readiness_record=record,
                    readiness_completed_monotonic=completed,
                    projected_states=projected, process_snapshot=snapshot)
            except BaseException as exc:
                last_error = exc
                elapsed = clock() - attempt_started
                attempts.append({
                    "pool_generation": generation,
                    "status": "FAILED_READINESS",
                    "attempt_elapsed_s": elapsed,
                    "observed_worker_pids": sorted(set(observed_pids)),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })
                if coordination is not None:
                    try:
                        coordination.release_event.set()
                    except BaseException:
                        pass
                if executor is not None:
                    executor.shutdown(wait=True, cancel_futures=True)
            finally:
                if coordination is not None:
                    coordination.close()
        raise WorkerReadinessError(
            "V2 readiness failed after the single permitted pre-attempt rebuild",
            attempt_rows=attempts) from last_error

    def assert_integrity(self, stage: str) -> list[dict[str, Any]]:
        if self.closed:
            raise WorkerPoolIntegrityError("V2 worker pool is already closed")
        if self._integrity_failed:
            raise WorkerPoolIntegrityError(
                f"V2 worker pool previously failed before {stage}")
        snapshot = _process_snapshot(self._executor)
        expected_pids = {row["worker_pid"] for row in self._baseline_snapshot}
        actual_pids = {row["worker_pid"] for row in snapshot}
        if (len(snapshot) != WORKER_COUNT or actual_pids != expected_pids
                or any(not row["is_alive"] or row["exitcode"] is not None
                       for row in snapshot)):
            if self.sample0_started:
                self.worker_restart_count += len(actual_pids - expected_pids)
            self._integrity_failed = True
            raise WorkerPoolIntegrityError(
                f"V2 worker generation changed at {stage}")
        return snapshot

    def mark_sample0_started(self) -> list[dict[str, Any]]:
        if self.sample0_started:
            raise WorkerPoolIntegrityError("V2 timed sample 0 already began")
        snapshot = self.assert_integrity("immediately_before_sample_0")
        self.sample0_started = True
        return snapshot

    def _validate_fixed_envelope(self, envelope: Any) -> dict[str, Any]:
        if (not isinstance(envelope, Mapping)
                or set(envelope) != {
                    "worker_pid", "worker_instance_id", "worker_result"}):
            raise WorkerPoolIntegrityError("V2 fixed worker envelope changed")
        pair = (envelope["worker_pid"], envelope["worker_instance_id"])
        result = envelope["worker_result"]
        if (pair not in self._identities or not isinstance(result, Mapping)
                or result.get("worker_pid") != pair[0]):
            if not self._integrity_failed:
                self.worker_restart_count += 1
            self._integrity_failed = True
            raise WorkerPoolIntegrityError(
                "fixed task returned from an unregistered worker instance")
        with self._lock:
            self._observed_pairs.append(pair)
        return dict(result)

    def observation_cursor(self) -> int:
        with self._lock:
            return len(self._observed_pairs)

    def observations_since(self, cursor: int) -> list[tuple[int, str]]:
        with self._lock:
            return list(self._observed_pairs[cursor:])

    def submit(self, function: Callable[..., Any], task: Mapping[str, Any]
               ) -> concurrent.futures.Future[Any]:
        if self.closed:
            raise WorkerPoolIntegrityError("cannot submit to closed V2 pool")
        if function is not V1._fixed_rotation_worker:
            raise WorkerPoolIntegrityError(
                "V2 pool accepts only the unchanged V1 fixed-rotation task")
        inner = self._executor.submit(_v2_fixed_rotation_worker, task)
        return _ValidatedFuture(inner, self)

    def shutdown(self, *, wait: bool = True,
                 cancel_futures: bool = False) -> None:
        if not self.closed:
            self._executor.shutdown(
                wait=wait, cancel_futures=cancel_futures)
            self.closed = True


def _validate_readiness_announcement(
        payload: Mapping[str, Any], *, public_task: Mapping[str, Any]
        ) -> dict[str, Any]:
    row = dict(payload)
    required = {
        "worker_pid", "worker_instance_id", "readiness_task_digest",
        "predecessor_scientific_input_bindings_digest",
        "state_projection_digest", "source_identity_manifest_digest",
        "solver_options_digest", "immutable_search_input_digest",
        "thread_environment", "solver_module", "solver_backend",
        "solver_version", "constraint_row_count", "variable_count",
        "bounds_variable_count", "solver_imported", "immutable_inputs_loaded",
        "readiness_barrier_reached", "candidate_outcomes_consumed",
        "scientific_masks_accessed", "solver_call_count",
        "readiness_announcement_digest",
    }
    if (set(row) != required
            or not isinstance(row.get("worker_pid"), int)
            or isinstance(row.get("worker_pid"), bool)
            or row["worker_pid"] <= 0
            or not isinstance(row.get("worker_instance_id"), str)
            or len(row["worker_instance_id"]) != 32
            or not all(character in "0123456789abcdef"
                       for character in row["worker_instance_id"])
            or row.get("readiness_task_digest")
            != V1.canonical_digest(public_task)
            or row.get("predecessor_scientific_input_bindings_digest")
            != public_task[
                "predecessor_scientific_input_bindings_digest"]
            or row.get("state_projection_digest")
            != public_task["state_projection_digest"]
            or row.get("source_identity_manifest_digest")
            != public_task["source_identity_manifest_digest"]
            or row.get("solver_options_digest")
            != public_task["solver_options_digest"]
            or row.get("immutable_search_input_digest")
            != public_task["immutable_search_input_digest"]
            or row.get("thread_environment") != V1.THREAD_ENVIRONMENT
            or not isinstance(row.get("solver_module"), str)
            or not row["solver_module"]
            or row.get("solver_backend") != "scipy.optimize.milp/HiGHS"
            or not isinstance(row.get("solver_version"), str)
            or not row["solver_version"]
            or not isinstance(row.get("constraint_row_count"), int)
            or row["constraint_row_count"] < 0
            or row.get("variable_count") != public_task["variable_count"]
            or row.get("bounds_variable_count") != public_task["variable_count"]
            or row.get("solver_imported") is not True
            or row.get("immutable_inputs_loaded") is not True
            or row.get("readiness_barrier_reached") is not True
            or row.get("candidate_outcomes_consumed") is not False
            or row.get("scientific_masks_accessed") is not False
            or not isinstance(row.get("solver_call_count"), int)
            or isinstance(row.get("solver_call_count"), bool)
            or row.get("solver_call_count") != 0
            or row.get("readiness_announcement_digest")
            != V1.canonical_digest(V1._without_digest(
                row, "readiness_announcement_digest"))):
        raise WorkerReadinessError("readiness announcement binding changed")
    return row


def _validate_worker_row(payload: Mapping[str, Any], *,
                         public_task: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(payload)
    required_extra = {
        "readiness_task_completed", "readiness_elapsed_s",
        "worker_readiness_digest",
    }
    base_keys = {
        "worker_pid", "worker_instance_id", "readiness_task_digest",
        "predecessor_scientific_input_bindings_digest",
        "state_projection_digest", "source_identity_manifest_digest",
        "solver_options_digest", "immutable_search_input_digest",
        "thread_environment", "solver_module", "solver_backend",
        "solver_version", "constraint_row_count", "variable_count",
        "bounds_variable_count", "solver_imported", "immutable_inputs_loaded",
        "readiness_barrier_reached", "candidate_outcomes_consumed",
        "scientific_masks_accessed", "solver_call_count",
    }
    elapsed = row.get("readiness_elapsed_s")
    if (set(row) != base_keys | required_extra
            or row.get("readiness_task_completed") is not True
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool) or not math.isfinite(float(elapsed))
            or elapsed < 0
            or row.get("worker_readiness_digest")
            != V1.canonical_digest(V1._without_digest(
                row, "worker_readiness_digest"))):
        raise WorkerReadinessError("worker readiness row surface changed")
    announcement = _readiness_announcement(row)
    _validate_readiness_announcement(announcement, public_task=public_task)
    return row


def _build_readiness_record(
        *, public_task: Mapping[str, Any],
        worker_rows: Sequence[Mapping[str, Any]],
        process_snapshot: Sequence[Mapping[str, Any]],
        attempt_rows: Sequence[Mapping[str, Any]],
        successful_attempt_elapsed_s: float, total_elapsed_s: float,
        generation: int) -> dict[str, Any]:
    identities = [{
        "worker_pid": row["worker_pid"],
        "worker_instance_id": row["worker_instance_id"],
    } for row in worker_rows]
    pool_identity = V1.canonical_digest({
        "benchmark_v2_contract_digest":
            public_task["benchmark_v2_contract_digest"],
        "readiness_task_digest": V1.canonical_digest(public_task),
        "pool_generation": generation,
        "worker_identities": identities,
    })
    payload = {
        "schema": READINESS_RECORD_SCHEMA,
        "benchmark_v2_contract_digest":
            public_task["benchmark_v2_contract_digest"],
        "readiness_algorithm": READINESS_ALGORITHM,
        "readiness_task_symbol": READINESS_TASK_SYMBOL,
        "worker_initializer_symbol": WORKER_INITIALIZER_SYMBOL,
        "fixed_worker_symbol": FIXED_WORKER_SYMBOL,
        "readiness_task_digest": V1.canonical_digest(public_task),
        "predecessor_scientific_input_bindings_digest": public_task[
            "predecessor_scientific_input_bindings_digest"],
        "immutable_search_input_digest":
            public_task["immutable_search_input_digest"],
        "state_projection_digest": public_task["state_projection_digest"],
        "source_identity_manifest_digest":
            public_task["source_identity_manifest_digest"],
        "solver_options_digest": public_task["solver_options_digest"],
        "worker_count": WORKER_COUNT,
        "worker_rows": [dict(row) for row in worker_rows],
        "worker_rows_digest": V1.canonical_digest(worker_rows),
        "ready_process_snapshot": [dict(row) for row in process_snapshot],
        "ready_process_snapshot_digest": V1.canonical_digest(process_snapshot),
        "worker_pool_identity": pool_identity,
        "pool_generation": generation,
        "rebuild_count": generation,
        "pool_rebuilt_before_sample0": generation == 1,
        "pool_construction_attempts": [dict(row) for row in attempt_rows],
        "successful_startup_prewarm_wall_s": successful_attempt_elapsed_s,
        "total_pool_preparation_wall_s": total_elapsed_s,
        "worker_restart_count": 0,
        "sample0_started": False,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
        "solver_call_count": 0,
    }
    payload["readiness_record_digest"] = V1.canonical_digest(payload)
    return validate_readiness_record(
        payload,
        expected_benchmark_v2_contract_digest=public_task[
            "benchmark_v2_contract_digest"],
        expected_predecessor_scientific_input_bindings_digest=public_task[
            "predecessor_scientific_input_bindings_digest"])


def validate_readiness_record(
        payload: Mapping[str, Any], *,
        expected_benchmark_v2_contract_digest: str,
        expected_predecessor_scientific_input_bindings_digest: str,
        ) -> dict[str, Any]:
    record = dict(payload)
    required = {
        "schema", "benchmark_v2_contract_digest", "readiness_algorithm",
        "readiness_task_symbol", "worker_initializer_symbol",
        "fixed_worker_symbol", "readiness_task_digest",
        "predecessor_scientific_input_bindings_digest",
        "immutable_search_input_digest", "state_projection_digest",
        "source_identity_manifest_digest", "solver_options_digest",
        "worker_count", "worker_rows", "worker_rows_digest",
        "ready_process_snapshot", "ready_process_snapshot_digest",
        "worker_pool_identity", "pool_generation", "rebuild_count",
        "pool_rebuilt_before_sample0", "pool_construction_attempts",
        "successful_startup_prewarm_wall_s", "total_pool_preparation_wall_s",
        "worker_restart_count", "sample0_started",
        "candidate_outcomes_consumed", "scientific_masks_accessed",
        "solver_call_count", "readiness_record_digest",
    }
    rows = record.get("worker_rows")
    snapshot = record.get("ready_process_snapshot")
    attempts = record.get("pool_construction_attempts")
    generation = record.get("pool_generation")
    numeric = (record.get("successful_startup_prewarm_wall_s"),
               record.get("total_pool_preparation_wall_s"))
    if (set(record) != required
            or record.get("schema") != READINESS_RECORD_SCHEMA
            or record.get("benchmark_v2_contract_digest")
            != expected_benchmark_v2_contract_digest
            or not _is_digest(expected_benchmark_v2_contract_digest)
            or record.get("predecessor_scientific_input_bindings_digest")
            != expected_predecessor_scientific_input_bindings_digest
            or not _is_digest(
                expected_predecessor_scientific_input_bindings_digest)
            or record.get("readiness_algorithm") != READINESS_ALGORITHM
            or record.get("readiness_task_symbol") != READINESS_TASK_SYMBOL
            or record.get("worker_initializer_symbol")
            != WORKER_INITIALIZER_SYMBOL
            or record.get("fixed_worker_symbol") != FIXED_WORKER_SYMBOL
            or any(not _is_digest(record.get(key)) for key in (
                "readiness_task_digest", "immutable_search_input_digest",
                "predecessor_scientific_input_bindings_digest",
                "state_projection_digest", "source_identity_manifest_digest",
                "solver_options_digest", "worker_pool_identity"))
            or record.get("worker_count") != WORKER_COUNT
            or not isinstance(generation, int) or isinstance(generation, bool)
            or generation not in (0, 1)
            or not isinstance(record.get("rebuild_count"), int)
            or isinstance(record.get("rebuild_count"), bool)
            or record.get("rebuild_count") != generation
            or not isinstance(record.get("pool_rebuilt_before_sample0"), bool)
            or record["pool_rebuilt_before_sample0"] != (generation == 1)
            or not isinstance(rows, list) or len(rows) != WORKER_COUNT
            or record.get("worker_rows_digest") != V1.canonical_digest(rows)
            or not isinstance(snapshot, list) or len(snapshot) != WORKER_COUNT
            or record.get("ready_process_snapshot_digest")
            != V1.canonical_digest(snapshot)
            or not isinstance(attempts, list) or len(attempts) != generation + 1
            or any(not isinstance(value, (int, float))
                   or isinstance(value, bool) or not math.isfinite(float(value))
                   or value < 0 for value in numeric)
            or record["total_pool_preparation_wall_s"]
            < record["successful_startup_prewarm_wall_s"]
            or not isinstance(record.get("worker_restart_count"), int)
            or isinstance(record.get("worker_restart_count"), bool)
            or record.get("worker_restart_count") != 0
            or record.get("sample0_started") is not False
            or record.get("candidate_outcomes_consumed") is not False
            or record.get("scientific_masks_accessed") is not False
            or not isinstance(record.get("solver_call_count"), int)
            or isinstance(record.get("solver_call_count"), bool)
            or record.get("solver_call_count") != 0
            or record.get("readiness_record_digest")
            != V1.canonical_digest(V1._without_digest(
                record, "readiness_record_digest"))):
        raise WorkerReadinessError("V2 readiness record binding changed")
    public_stub = {
        "state_projection_digest": record["state_projection_digest"],
        "source_identity_manifest_digest":
            record["source_identity_manifest_digest"],
        "solver_options_digest": record["solver_options_digest"],
        "immutable_search_input_digest":
            record["immutable_search_input_digest"],
        "predecessor_scientific_input_bindings_digest": record[
            "predecessor_scientific_input_bindings_digest"],
        "variable_count": V1.PREFIX_STATE_COUNT * V1.ROTATION_COUNT,
    }
    # Full worker validation needs only the fields used by its strict checks;
    # the already-bound task digest is checked directly below.
    identities: set[tuple[int, str]] = set()
    constraint_counts: set[int] = set()
    versions: set[tuple[str, str, str]] = set()
    for row in rows:
        if (not isinstance(row, dict)
                or row.get("readiness_task_digest")
                != record["readiness_task_digest"]
                or row.get("predecessor_scientific_input_bindings_digest")
                != public_stub[
                    "predecessor_scientific_input_bindings_digest"]
                or row.get("state_projection_digest")
                != public_stub["state_projection_digest"]
                or row.get("source_identity_manifest_digest")
                != public_stub["source_identity_manifest_digest"]
                or row.get("solver_options_digest")
                != public_stub["solver_options_digest"]
                or row.get("immutable_search_input_digest")
                != public_stub["immutable_search_input_digest"]):
            raise WorkerReadinessError("V2 readiness worker binding changed")
        # Validate all structural/boolean/digest fields without reconstructing
        # the unavailable public task from its digest.
        if (set(row) != {
                "worker_pid", "worker_instance_id", "readiness_task_digest",
                "predecessor_scientific_input_bindings_digest",
                "state_projection_digest", "source_identity_manifest_digest",
                "solver_options_digest", "immutable_search_input_digest",
                "thread_environment", "solver_module", "solver_backend",
                "solver_version", "constraint_row_count", "variable_count",
                "bounds_variable_count", "solver_imported",
                "immutable_inputs_loaded", "readiness_barrier_reached",
                "candidate_outcomes_consumed", "scientific_masks_accessed",
                "solver_call_count", "readiness_task_completed",
                "readiness_elapsed_s", "worker_readiness_digest"}
                or not isinstance(row.get("worker_pid"), int)
                or isinstance(row.get("worker_pid"), bool)
                or row["worker_pid"] <= 0
                or not isinstance(row.get("worker_instance_id"), str)
                or len(row["worker_instance_id"]) != 32
                or not all(character in "0123456789abcdef"
                           for character in row["worker_instance_id"])
                or row.get("thread_environment") != V1.THREAD_ENVIRONMENT
                or row.get("solver_backend") != "scipy.optimize.milp/HiGHS"
                or not isinstance(row.get("solver_module"), str)
                or not row["solver_module"]
                or not isinstance(row.get("solver_version"), str)
                or not row["solver_version"]
                or not isinstance(row.get("constraint_row_count"), int)
                or row["constraint_row_count"] < 0
                or row.get("variable_count")
                != V1.PREFIX_STATE_COUNT * V1.ROTATION_COUNT
                or row.get("bounds_variable_count") != row["variable_count"]
                or row.get("solver_imported") is not True
                or row.get("immutable_inputs_loaded") is not True
                or row.get("readiness_barrier_reached") is not True
                or row.get("readiness_task_completed") is not True
                or row.get("candidate_outcomes_consumed") is not False
                or row.get("scientific_masks_accessed") is not False
                or not isinstance(row.get("solver_call_count"), int)
                or isinstance(row.get("solver_call_count"), bool)
                or row.get("solver_call_count") != 0
                or not isinstance(row.get("readiness_elapsed_s"), (int, float))
                or isinstance(row.get("readiness_elapsed_s"), bool)
                or not math.isfinite(float(row["readiness_elapsed_s"]))
                or row["readiness_elapsed_s"] < 0
                or row.get("worker_readiness_digest")
                != V1.canonical_digest(V1._without_digest(
                    row, "worker_readiness_digest"))):
            raise WorkerReadinessError("V2 readiness worker row changed")
        identities.add((row["worker_pid"], row["worker_instance_id"]))
        constraint_counts.add(row["constraint_row_count"])
        versions.add((row["solver_module"], row["solver_backend"],
                      row["solver_version"]))
    if rows != sorted(rows, key=lambda row: (
            row["worker_pid"], row["worker_instance_id"])):
        raise WorkerReadinessError("V2 readiness worker order changed")
    snapshot_pids = {row.get("worker_pid") for row in snapshot
                     if isinstance(row, dict)}
    if (len(identities) != WORKER_COUNT or len(constraint_counts) != 1
            or len(versions) != 1
            or snapshot_pids != {pid for pid, _ in identities}
            or any(set(row) != {"worker_pid", "is_alive", "exitcode"}
                   or row.get("is_alive") is not True
                   or row.get("exitcode") is not None for row in snapshot)):
        raise WorkerReadinessError("V2 ready worker generation changed")
    attempt_required = {
        "pool_generation", "status", "attempt_elapsed_s",
        "observed_worker_pids", "error_type", "error_message",
    }
    for index, row in enumerate(attempts):
        expected_status = "READY" if index == generation else "FAILED_READINESS"
        if (not isinstance(row, dict) or set(row) != attempt_required
                or not isinstance(row.get("pool_generation"), int)
                or isinstance(row.get("pool_generation"), bool)
                or row.get("pool_generation") != index
                or row.get("status") != expected_status
                or not isinstance(row.get("attempt_elapsed_s"), (int, float))
                or isinstance(row.get("attempt_elapsed_s"), bool)
                or not math.isfinite(float(row["attempt_elapsed_s"]))
                or row["attempt_elapsed_s"] < 0
                or not isinstance(row.get("observed_worker_pids"), list)
                or any(not isinstance(pid, int) or isinstance(pid, bool)
                       or pid <= 0 for pid in row["observed_worker_pids"])
                or row["observed_worker_pids"]
                != sorted(set(row["observed_worker_pids"]))
                or (expected_status == "READY"
                    and row["observed_worker_pids"]
                    != sorted(snapshot_pids))
                or (expected_status == "READY"
                    and (row.get("error_type") is not None
                         or row.get("error_message") is not None))
                or (expected_status == "FAILED_READINESS"
                    and (not isinstance(row.get("error_type"), str)
                         or not isinstance(row.get("error_message"), str)))):
            raise WorkerReadinessError("V2 readiness attempt lineage changed")
    if (abs(float(record["successful_startup_prewarm_wall_s"])
            - float(attempts[-1]["attempt_elapsed_s"])) > 1e-12
            or abs(float(record["total_pool_preparation_wall_s"])
                   - sum(float(row["attempt_elapsed_s"])
                         for row in attempts)) > 1e-12):
        raise WorkerReadinessError("V2 readiness timing lineage changed")
    expected_identity = V1.canonical_digest({
        "benchmark_v2_contract_digest": record["benchmark_v2_contract_digest"],
        "readiness_task_digest": record["readiness_task_digest"],
        "pool_generation": generation,
        "worker_identities": [{
            "worker_pid": row["worker_pid"],
            "worker_instance_id": row["worker_instance_id"],
        } for row in rows],
    })
    if record["worker_pool_identity"] != expected_identity:
        raise WorkerReadinessError("V2 worker pool identity digest changed")
    return record


def run_ready_fixed_rotation_benchmark_v2(
        *, states: Sequence[Mapping[str, Any]],
        search_plan: Mapping[str, Any], source_binding_digest: str,
        bound_v1_failure_receipt_digest: str,
        predecessor_scientific_input_bindings_digest: str,
        ready_pool: ReadyWorkerPoolV2,
        clock: Callable[[], float] = time.monotonic,
        ) -> dict[str, Any]:
    """Run exact V1 samples 0/1/2 on an already-certified live pool."""

    plan = V1.validate_search_plan(search_plan)
    projected = _strict_projected_states(states)
    if (not _is_digest(source_binding_digest)
            or not _is_digest(bound_v1_failure_receipt_digest)
            or not _is_digest(predecessor_scientific_input_bindings_digest)):
        raise ValueError("V2 benchmark source/predecessor digest is invalid")
    if (predecessor_scientific_input_bindings_digest
            != ready_pool.predecessor_scientific_input_bindings_digest):
        raise WorkerPoolIntegrityError(
            "predecessor scientific input binding differs from ready pool")
    if (V1.canonical_digest(projected)
            != ready_pool.readiness_record["state_projection_digest"]
            or immutable_search_input_digest(plan)
            != ready_pool.immutable_search_input_digest):
        raise WorkerPoolIntegrityError(
            "benchmark inputs differ from the ready worker pool")

    from scipy.optimize import Bounds, milp

    V1._worker_initialise()
    constraints, base_bounds = V1.ALLOC._constraint_system(projected)
    variable_count = len(projected) * V1.ROTATION_COUNT
    integrality = np.ones(variable_count, dtype=np.uint8)
    lower = np.asarray(base_bounds.lb, dtype=np.float64).copy()
    upper = np.asarray(base_bounds.ub, dtype=np.float64).copy()
    options = dict(plan["solver_options"])
    sample_rows: list[dict[str, Any]] = []
    selected_prefix: list[int] = []
    initial_snapshot = ready_pool.mark_sample0_started()
    for state_index in SAMPLE_PREFIX_INDICES:
        before_snapshot = (initial_snapshot if state_index == 0 else
                           ready_pool.assert_integrity(
                               f"immediately_before_sample_{state_index}"))
        start = state_index * V1.ROTATION_COUNT
        objective = np.zeros(variable_count, dtype=np.float64)
        objective[start:start + V1.ROTATION_COUNT] = np.arange(
            V1.ROTATION_COUNT, dtype=np.float64)
        control_started = clock()
        readiness_delay = (control_started
                           - ready_pool.readiness_completed_monotonic)
        result = milp(
            c=objective, integrality=integrality,
            bounds=Bounds(lower, upper), constraints=constraints,
            options=options)
        control_elapsed = clock() - control_started
        status, message = V1._classify_fixed_rotation_result(
            result, constraints=constraints, lower=lower, upper=upper,
            fixed_variable_index=None)
        if status != "FEASIBLE" or result.fun is None:
            raise V1.ParallelSearchFatal(
                f"V2 benchmark objective prefix {state_index} failed: {message}")
        local = np.asarray(
            result.x[start:start + V1.ROTATION_COUNT], dtype=np.float64)
        objective_rotation = int(np.argmax(local))
        if abs(float(result.fun) - objective_rotation) > 1e-6:
            raise V1.ParallelSearchFatal(
                "V2 benchmark objective choice is inconsistent")
        observation_cursor = ready_pool.observation_cursor()
        parallel_started = clock()
        futures = [ready_pool.submit(V1._fixed_rotation_worker, {
            "states": projected,
            "prefix_rotations": selected_prefix,
            "state_index": state_index,
            "rotation": rotation,
            "solver_options": options,
        }) for rotation in range(V1.ROTATION_COUNT)]
        fixed_rows = sorted(
            (future.result() for future in futures),
            key=lambda row: int(row["rotation"]))
        parallel_elapsed = clock() - parallel_started
        fixed_row_keys = {
            "rotation", "status", "message", "elapsed_s",
            "solver_call_count", "worker_pid", "thread_environment",
        }
        if (len(fixed_rows) != V1.ROTATION_COUNT
                or [row.get("rotation") for row in fixed_rows]
                != list(range(V1.ROTATION_COUNT))
                or any(not isinstance(row, dict) or set(row) != fixed_row_keys
                       or row.get("status") not in V1.ROTATION_STATUSES
                       or not isinstance(row.get("message"), str)
                       or not isinstance(row.get("elapsed_s"), (int, float))
                       or isinstance(row.get("elapsed_s"), bool)
                       or not math.isfinite(float(row["elapsed_s"]))
                       or row["elapsed_s"] < 0
                       or row.get("solver_call_count") != 1
                       or not isinstance(row.get("worker_pid"), int)
                       or isinstance(row.get("worker_pid"), bool)
                       or row["worker_pid"] <= 0
                       or row.get("thread_environment")
                       != V1.THREAD_ENVIRONMENT for row in fixed_rows)):
            raise WorkerPoolIntegrityError(
                "V2 fixed-rotation worker result surface changed")
        fixed_status, fixed_rotation = V1._lexicographic_wave_decision(
            [row["status"] for row in fixed_rows], state_index=state_index)
        timeout_free = not any(row["status"] == "FATAL"
                               for row in fixed_rows)
        if fixed_status != "SELECTED":
            fatal = next((row for row in fixed_rows
                          if row["status"] == "FATAL"),
                         {"message": "fixed-r wave has no certified minimum"})
            raise V1.ParallelSearchFatal(
                f"V2 benchmark fixed-r prefix {state_index} failed: "
                f"{fatal['message']}")
        if fixed_rotation != objective_rotation:
            raise V1.ParallelSearchFatal(
                "V2 benchmark fixed-r choice differs from objective choice")
        after_snapshot = ready_pool.assert_integrity(
            f"immediately_after_sample_{state_index}")
        observations = ready_pool.observations_since(observation_cursor)
        ratio = parallel_elapsed / control_elapsed
        sample_rows.append({
            "state_index": state_index,
            "prefix_rotations_digest": V1.canonical_digest(selected_prefix),
            "objective_rotation": objective_rotation,
            "fixed_rotation": fixed_rotation,
            "objective_elapsed_s": control_elapsed,
            "fixed_wave_elapsed_s": parallel_elapsed,
            "parallel_fraction": ratio,
            "fixed_solver_call_count": V1.ROTATION_COUNT,
            "objective_solver_call_count": 1,
            "equivalence_verdict": "PASS",
            "timeout_verdict": "PASS" if timeout_free else "FAIL",
            "worker_pool_identity": ready_pool.worker_pool_identity,
            "worker_pids_before": [row["worker_pid"]
                                   for row in before_snapshot],
            "worker_pids_after": [row["worker_pid"]
                                  for row in after_snapshot],
            "fixed_task_worker_pids": sorted({pid for pid, _ in observations}),
            "fixed_task_worker_instance_ids": sorted(
                {instance for _, instance in observations}),
            "worker_restarted": False,
            "worker_restart_count": ready_pool.worker_restart_count,
            "readiness_to_sample_delay_s": readiness_delay,
        })
        selected_prefix.append(objective_rotation)
        lower[start:start + V1.ROTATION_COUNT] = 0.0
        upper[start:start + V1.ROTATION_COUNT] = 0.0
        lower[start + objective_rotation] = 1.0
        upper[start + objective_rotation] = 1.0

    fractions = [float(row["parallel_fraction"]) for row in sample_rows]
    median_fraction = float(np.median(np.asarray(fractions)))
    maximum_fraction = max(fractions)
    median_passes = median_fraction <= MAXIMUM_PARALLEL_FRACTION
    maximum_passes = maximum_fraction <= MAXIMUM_PARALLEL_FRACTION
    serial_elapsed = sum(float(row["objective_elapsed_s"])
                         for row in sample_rows)
    parallel_elapsed = sum(float(row["fixed_wave_elapsed_s"])
                           for row in sample_rows)
    payload = {
        "schema": BENCHMARK_RECEIPT_SCHEMA,
        "benchmark_v2_contract_digest":
            ready_pool.benchmark_v2_contract_digest,
        "bound_v1_failure_receipt_digest": bound_v1_failure_receipt_digest,
        "predecessor_scientific_input_bindings_digest":
            predecessor_scientific_input_bindings_digest,
        "v1_failure_disposition": V1_FAILURE_DISPOSITION,
        "source_binding_digest": source_binding_digest,
        "readiness_record_digest":
            ready_pool.readiness_record["readiness_record_digest"],
        "worker_pool_identity": ready_pool.worker_pool_identity,
        "immutable_search_input_digest":
            ready_pool.immutable_search_input_digest,
        "state_projection_digest": V1.canonical_digest(projected),
        "source_identity_manifest_digest":
            V1.ALLOC.pre_outcome_identity_digest(projected),
        "sample_prefix_indices": list(SAMPLE_PREFIX_INDICES),
        "sample_rows": sample_rows,
        "sample_rows_digest": V1.canonical_digest(sample_rows),
        "serial_elapsed_s": serial_elapsed,
        "parallel_elapsed_s": parallel_elapsed,
        "observed_parallel_fraction": parallel_elapsed / serial_elapsed,
        "median_parallel_fraction": median_fraction,
        "maximum_parallel_fraction_observed": maximum_fraction,
        "maximum_parallel_fraction": MAXIMUM_PARALLEL_FRACTION,
        "median_gate_passes": median_passes,
        "maximum_gate_passes": maximum_passes,
        "passes": median_passes and maximum_passes,
        "allocation_vectors_equal": True,
        "all_samples_timeout_free": True,
        "worker_restart_count": ready_pool.worker_restart_count,
        "pool_rebuilt_after_sample0": False,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
    }
    payload["benchmark_receipt_digest"] = V1.canonical_digest(payload)
    return validate_benchmark_receipt_v2(
        payload,
        expected_benchmark_v2_contract_digest=
            ready_pool.benchmark_v2_contract_digest,
        expected_v1_failure_receipt_digest=bound_v1_failure_receipt_digest,
        expected_predecessor_scientific_input_bindings_digest=
            predecessor_scientific_input_bindings_digest,
        expected_source_binding_digest=source_binding_digest)


def validate_benchmark_receipt_v2(
        payload: Mapping[str, Any], *,
        expected_benchmark_v2_contract_digest: str,
        expected_v1_failure_receipt_digest: str,
        expected_predecessor_scientific_input_bindings_digest: str,
        expected_source_binding_digest: str,
        require_pass: bool = False,
        ) -> dict[str, Any]:
    receipt = dict(payload)
    required = {
        "schema", "benchmark_v2_contract_digest",
        "bound_v1_failure_receipt_digest",
        "predecessor_scientific_input_bindings_digest",
        "v1_failure_disposition",
        "source_binding_digest", "readiness_record_digest",
        "worker_pool_identity", "immutable_search_input_digest",
        "state_projection_digest", "source_identity_manifest_digest",
        "sample_prefix_indices", "sample_rows", "sample_rows_digest",
        "serial_elapsed_s", "parallel_elapsed_s",
        "observed_parallel_fraction", "median_parallel_fraction",
        "maximum_parallel_fraction_observed", "maximum_parallel_fraction",
        "median_gate_passes", "maximum_gate_passes", "passes",
        "allocation_vectors_equal", "all_samples_timeout_free",
        "worker_restart_count", "pool_rebuilt_after_sample0",
        "candidate_outcomes_consumed", "scientific_masks_accessed",
        "benchmark_receipt_digest",
    }
    rows = receipt.get("sample_rows")
    aggregate_numbers = (
        receipt.get("serial_elapsed_s"), receipt.get("parallel_elapsed_s"),
        receipt.get("observed_parallel_fraction"),
        receipt.get("median_parallel_fraction"),
        receipt.get("maximum_parallel_fraction_observed"),
    )
    if (set(receipt) != required
            or receipt.get("schema") != BENCHMARK_RECEIPT_SCHEMA
            or receipt.get("benchmark_v2_contract_digest")
            != expected_benchmark_v2_contract_digest
            or receipt.get("bound_v1_failure_receipt_digest")
            != expected_v1_failure_receipt_digest
            or receipt.get("predecessor_scientific_input_bindings_digest")
            != expected_predecessor_scientific_input_bindings_digest
            or receipt.get("v1_failure_disposition") != V1_FAILURE_DISPOSITION
            or receipt.get("source_binding_digest")
            != expected_source_binding_digest
            or any(not _is_digest(value) for value in (
                expected_benchmark_v2_contract_digest,
                expected_v1_failure_receipt_digest,
                expected_predecessor_scientific_input_bindings_digest,
                expected_source_binding_digest))
            or any(not _is_digest(receipt.get(key)) for key in (
                "benchmark_v2_contract_digest",
                "bound_v1_failure_receipt_digest",
                "predecessor_scientific_input_bindings_digest",
                "source_binding_digest",
                "readiness_record_digest", "worker_pool_identity",
                "immutable_search_input_digest", "state_projection_digest",
                "source_identity_manifest_digest"))
            or receipt.get("sample_prefix_indices")
            != list(SAMPLE_PREFIX_INDICES)
            or not isinstance(rows, list) or len(rows) != 3
            or receipt.get("sample_rows_digest") != V1.canonical_digest(rows)
            or any(not isinstance(value, (int, float))
                   or isinstance(value, bool)
                   or not math.isfinite(float(value))
                   for value in aggregate_numbers)
            or receipt["serial_elapsed_s"] <= 0
            or receipt["parallel_elapsed_s"] < 0
            or receipt.get("maximum_parallel_fraction")
            != MAXIMUM_PARALLEL_FRACTION
            or receipt.get("allocation_vectors_equal") is not True
            or receipt.get("all_samples_timeout_free") is not True
            or not isinstance(receipt.get("worker_restart_count"), int)
            or isinstance(receipt.get("worker_restart_count"), bool)
            or receipt.get("worker_restart_count") != 0
            or receipt.get("pool_rebuilt_after_sample0") is not False
            or receipt.get("candidate_outcomes_consumed") is not False
            or receipt.get("scientific_masks_accessed") is not False
            or receipt.get("benchmark_receipt_digest")
            != V1.canonical_digest(V1._without_digest(
                receipt, "benchmark_receipt_digest"))):
        raise V1.ParallelSearchError("V2 benchmark receipt binding changed")
    row_keys = {
        "state_index", "prefix_rotations_digest", "objective_rotation",
        "fixed_rotation", "objective_elapsed_s", "fixed_wave_elapsed_s",
        "parallel_fraction", "fixed_solver_call_count",
        "objective_solver_call_count", "equivalence_verdict",
        "timeout_verdict", "worker_pool_identity", "worker_pids_before",
        "worker_pids_after", "fixed_task_worker_pids",
        "fixed_task_worker_instance_ids", "worker_restarted",
        "worker_restart_count", "readiness_to_sample_delay_s",
    }
    fractions: list[float] = []
    objective_seconds: list[float] = []
    fixed_seconds: list[float] = []
    prefix: list[int] = []
    expected_pid_set: list[int] | None = None
    for index, row in zip(SAMPLE_PREFIX_INDICES, rows, strict=True):
        if not isinstance(row, dict) or set(row) != row_keys:
            raise V1.ParallelSearchError("V2 benchmark sample surface changed")
        control = row.get("objective_elapsed_s")
        fixed = row.get("fixed_wave_elapsed_s")
        ratio = row.get("parallel_fraction")
        before = row.get("worker_pids_before")
        after = row.get("worker_pids_after")
        if (row.get("state_index") != index
                or row.get("prefix_rotations_digest")
                != V1.canonical_digest(prefix)
                or not isinstance(row.get("objective_rotation"), int)
                or isinstance(row.get("objective_rotation"), bool)
                or not 0 <= row["objective_rotation"] < V1.ROTATION_COUNT
                or row.get("fixed_rotation") != row["objective_rotation"]
                or any(not isinstance(value, (int, float))
                       or isinstance(value, bool)
                       or not math.isfinite(float(value))
                       for value in (control, fixed, ratio))
                or control <= 0 or fixed < 0
                or abs(float(ratio) - float(fixed) / float(control)) > 1e-12
                or row.get("fixed_solver_call_count") != V1.ROTATION_COUNT
                or row.get("objective_solver_call_count") != 1
                or row.get("equivalence_verdict") != "PASS"
                or row.get("timeout_verdict") != "PASS"
                or row.get("worker_pool_identity")
                != receipt["worker_pool_identity"]
                or not isinstance(before, list) or len(before) != WORKER_COUNT
                or any(not isinstance(pid, int) or isinstance(pid, bool)
                       or pid <= 0 for pid in before)
                or before != sorted(set(before)) or after != before
                or (expected_pid_set is not None and before != expected_pid_set)
                or not isinstance(row.get("fixed_task_worker_pids"), list)
                or not 1 <= len(row["fixed_task_worker_pids"])
                <= V1.ROTATION_COUNT
                or row["fixed_task_worker_pids"]
                != sorted(set(row["fixed_task_worker_pids"]))
                or not set(row["fixed_task_worker_pids"]).issubset(set(before))
                or not isinstance(row.get("fixed_task_worker_instance_ids"), list)
                or not 1 <= len(row["fixed_task_worker_instance_ids"])
                <= V1.ROTATION_COUNT
                or row["fixed_task_worker_instance_ids"]
                != sorted(set(row["fixed_task_worker_instance_ids"]))
                or any(not isinstance(value, str) or len(value) != 32
                       or not all(character in "0123456789abcdef"
                                  for character in value)
                       for value in row["fixed_task_worker_instance_ids"])
                or row.get("worker_restarted") is not False
                or not isinstance(row.get("worker_restart_count"), int)
                or isinstance(row.get("worker_restart_count"), bool)
                or row.get("worker_restart_count") != 0
                or not isinstance(row.get("readiness_to_sample_delay_s"),
                                  (int, float))
                or isinstance(row.get("readiness_to_sample_delay_s"), bool)
                or not math.isfinite(float(row["readiness_to_sample_delay_s"]))
                or row["readiness_to_sample_delay_s"] < 0):
            raise V1.ParallelSearchError("V2 benchmark sample binding changed")
        expected_pid_set = list(before)
        prefix.append(row["objective_rotation"])
        fractions.append(float(ratio))
        objective_seconds.append(float(control))
        fixed_seconds.append(float(fixed))
    serial = sum(objective_seconds)
    parallel = sum(fixed_seconds)
    median = float(np.median(np.asarray(fractions)))
    maximum = max(fractions)
    median_passes = median <= MAXIMUM_PARALLEL_FRACTION
    maximum_passes = maximum <= MAXIMUM_PARALLEL_FRACTION
    if (abs(float(receipt.get("serial_elapsed_s", math.inf)) - serial) > 1e-12
            or abs(float(receipt.get("parallel_elapsed_s", math.inf))
                   - parallel) > 1e-12
            or abs(float(receipt.get("observed_parallel_fraction", math.inf))
                   - parallel / serial) > 1e-12
            or abs(float(receipt.get("median_parallel_fraction", math.inf))
                   - median) > 1e-12
            or abs(float(receipt.get(
                "maximum_parallel_fraction_observed", math.inf))
                   - maximum) > 1e-12
            or receipt.get("median_gate_passes") is not median_passes
            or receipt.get("maximum_gate_passes") is not maximum_passes
            or receipt.get("passes") is not (median_passes and maximum_passes)
            or (require_pass and receipt.get("passes") is not True)):
        raise V1.ParallelSearchError("V2 benchmark gate binding changed")
    return receipt


def run_scientific_parallel_search_v2(
        *, ready_pool: ReadyWorkerPoolV2,
        benchmark_receipt: Mapping[str, Any],
        expected_v1_failure_receipt_digest: str,
        expected_predecessor_scientific_input_bindings_digest: str,
        expected_source_binding_digest: str,
        search_plan: Mapping[str, Any], checkpoint_root: Path,
        prepare_rank: Callable[[int, tuple[int, ...]], Mapping[str, Any]],
        classify_mask: Callable[[Sequence[Mapping[str, Any]],
                                 Mapping[str, Any], Mapping[str, Any]], bool],
        validate_winner: Callable[[int, Sequence[Mapping[str, Any]],
                                   Mapping[str, Any], Mapping[str, Any]], bool],
        telemetry: Callable[[Mapping[str, Any]], None] | None = print,
        ) -> dict[str, Any]:
    """PASS-gated V1 search continuation on the unchanged live V2 pool."""

    receipt = validate_benchmark_receipt_v2(
        benchmark_receipt,
        expected_benchmark_v2_contract_digest=
            ready_pool.benchmark_v2_contract_digest,
        expected_v1_failure_receipt_digest=
            expected_v1_failure_receipt_digest,
        expected_predecessor_scientific_input_bindings_digest=
            expected_predecessor_scientific_input_bindings_digest,
        expected_source_binding_digest=expected_source_binding_digest,
        require_pass=True)
    plan = V1.validate_search_plan(search_plan)
    if (plan.get("measured_benchmark_receipt_digest")
            != receipt["benchmark_receipt_digest"]
            or immutable_search_input_digest(plan)
            != ready_pool.immutable_search_input_digest
            or expected_predecessor_scientific_input_bindings_digest
            != ready_pool.predecessor_scientific_input_bindings_digest
            or ready_pool.search_started):
        raise WorkerPoolIntegrityError(
            "V2 search plan/pool continuation binding changed")
    ready_pool.assert_integrity("between_benchmark_pass_and_search")
    ready_pool.search_started = True
    result = V1.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=checkpoint_root,
        prepare_rank=prepare_rank, classify_mask=classify_mask,
        validate_winner=validate_winner, telemetry=telemetry,
        process_executor=ready_pool)
    ready_pool.assert_integrity("after_scientific_search")
    return result


__all__ = [
    "BENCHMARK_CONTRACT_SCHEMA", "BENCHMARK_RECEIPT_SCHEMA",
    "DEFAULT_READINESS_TIMEOUT_S", "FIXED_WORKER_SYMBOL",
    "MAXIMUM_PARALLEL_FRACTION", "POOL_CONTINUATION_SCHEMA",
    "READINESS_ALGORITHM", "READINESS_RECORD_SCHEMA",
    "READINESS_TASK_SYMBOL", "ReadyWorkerPoolV2", "SAMPLE_PREFIX_INDICES",
    "SCHEMA_VERSION", "V1_FAILURE_DISPOSITION", "WORKER_COUNT",
    "WORKER_INITIALIZER_SYMBOL", "WorkerPoolIntegrityError",
    "WorkerReadinessError", "immutable_search_input_digest",
    "run_ready_fixed_rotation_benchmark_v2",
    "run_scientific_parallel_search_v2", "validate_benchmark_receipt_v2",
    "validate_readiness_record",
]
