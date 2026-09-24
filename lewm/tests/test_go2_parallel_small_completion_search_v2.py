"""Source-only tests for the one-shot warm-pool V2 executor.

Every pool, process, solver result, constraint matrix, and clock in this file
is synthetic.  The tests expose no scene, branch outcome, scientific mask,
frame, latent, checkpoint, predictor, or scorer material.
"""
from __future__ import annotations

import concurrent.futures
import copy
import queue
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.oracle import go2_parallel_small_completion_search_v1 as V1
from lewm.oracle import go2_parallel_small_completion_search_v2 as V2


def _identity_states() -> list[dict[str, str]]:
    goal_types = {
        "general": "landmark_red",
        "safety_enriched": "landmark_blue",
        "completion_enriched": "landmark_green",
    }
    rows = []
    for family_index in range(8):
        family = f"family-{family_index}"
        for stratum in V1.ALLOC.STRATA:
            for ordinal in range(5):
                state_id = f"{family}|{stratum}|{ordinal}"
                rows.append({
                    "state_id": state_id,
                    "state_identity_digest": V1.canonical_digest(state_id),
                    "family": family,
                    "stratum": stratum,
                    "split_role": "calibration" if ordinal == 0 else "fit",
                    "goal_type": goal_types[stratum],
                })
    assert len(rows) == V1.PREFIX_STATE_COUNT
    return rows


def _plan(*, measured_digest=None) -> dict:
    return V1.build_search_plan(
        candidate_scene_ids=[f"scene-{index:03d}" for index in range(8)],
        combination_size=5, worker_count=V2.WORKER_COUNT,
        source_repository_commit="a" * 40,
        clean_source_launch_receipt_digest="b" * 64,
        state_selector_amendment_digest="c" * 64,
        candidate_allocation_amendment_digest="d" * 64,
        fixed_state_projection_digest="e" * 64,
        resolver_cursor_scene_id="scene-before-pool",
        solver_identity={"name": "synthetic", "version": "1"},
        solver_options={"threads": 1, "mip_rel_gap": 0.0},
        active_rank_window=3,
        measured_benchmark_receipt_digest=measured_digest)


class _FakeCoordination:
    def __init__(self, _context):
        self.announcement_queue = queue.Queue()
        self.release_event = threading.Event()
        self.closed = False

    def close(self):
        self.closed = True


class _FakeProcessPool:
    """Thread-backed executor with 32 stable synthetic process identities."""

    def __init__(self, options, *, fail_readiness=False):
        self.options = dict(options)
        self.fail_readiness = fail_readiness
        self.pids = list(range(31_000, 31_000 + V2.WORKER_COUNT))
        self.instances = [f"{index:032x}" for index in range(V2.WORKER_COUNT)]
        self._threads = concurrent.futures.ThreadPoolExecutor(
            max_workers=V2.WORKER_COUNT)
        self._readiness_submissions = 0
        self._fixed_submissions = 0
        self.shutdown_calls = []
        self.force_pid_change = False

    @staticmethod
    def _readiness_row(task, pid, instance):
        public = task["public_task"]
        row = {
            "worker_pid": pid,
            "worker_instance_id": instance,
            "readiness_task_digest": V1.canonical_digest(public),
            "predecessor_scientific_input_bindings_digest": public[
                "predecessor_scientific_input_bindings_digest"],
            "state_projection_digest": public["state_projection_digest"],
            "source_identity_manifest_digest":
                public["source_identity_manifest_digest"],
            "solver_options_digest": public["solver_options_digest"],
            "immutable_search_input_digest":
                public["immutable_search_input_digest"],
            "thread_environment": dict(V1.THREAD_ENVIRONMENT),
            "solver_module": "scipy.optimize._milp",
            "solver_backend": "scipy.optimize.milp/HiGHS",
            "solver_version": "synthetic",
            "constraint_row_count": 17,
            "variable_count": public["variable_count"],
            "bounds_variable_count": public["variable_count"],
            "solver_imported": True,
            "immutable_inputs_loaded": True,
            "readiness_barrier_reached": True,
            "candidate_outcomes_consumed": False,
            "scientific_masks_accessed": False,
            "solver_call_count": 0,
        }
        task["announcement_queue"].put(V2._readiness_announcement(row))
        if not task["release_event"].wait(3.0):
            raise RuntimeError("synthetic readiness release timed out")
        row["readiness_task_completed"] = True
        row["readiness_elapsed_s"] = 0.25
        row["worker_readiness_digest"] = V1.canonical_digest(row)
        return row

    @staticmethod
    def _fixed_row(task, pid, instance):
        status = "FEASIBLE" if task["rotation"] == 0 else "INFEASIBLE"
        result = {
            "rotation": task["rotation"],
            "status": status,
            "message": f"synthetic {status.lower()}",
            "elapsed_s": 0.01,
            "solver_call_count": 1,
            "worker_pid": pid,
            "thread_environment": dict(V1.THREAD_ENVIRONMENT),
        }
        return {
            "worker_pid": pid,
            "worker_instance_id": instance,
            "worker_result": result,
        }

    def submit(self, function, task):
        if function is V2._v2_readiness_worker:
            slot = self._readiness_submissions
            self._readiness_submissions += 1
            if self.fail_readiness and slot == V2.WORKER_COUNT - 1:
                return self._threads.submit(
                    lambda: (_ for _ in ()).throw(
                        RuntimeError("synthetic pre-barrier failure")))
            return self._threads.submit(
                self._readiness_row, task, self.pids[slot],
                self.instances[slot])
        assert function is V2._v2_fixed_rotation_worker
        slot = self._fixed_submissions % V2.WORKER_COUNT
        self._fixed_submissions += 1
        return self._threads.submit(
            self._fixed_row, task, self.pids[slot], self.instances[slot])

    def v2_process_snapshot(self):
        pids = list(self.pids)
        if self.force_pid_change:
            pids[-1] += 500
        return [{"worker_pid": pid, "is_alive": True, "exitcode": None}
                for pid in pids]

    def shutdown(self, **options):
        self.shutdown_calls.append(dict(options))
        self._threads.shutdown(wait=True, cancel_futures=True)


class _FakePoolFactory:
    def __init__(self, *, fail_first=False):
        self.fail_first = fail_first
        self.pools = []

    def __call__(self, **options):
        pool = _FakeProcessPool(
            options, fail_readiness=self.fail_first and not self.pools)
        self.pools.append(pool)
        return pool


def _ready_pool(*, fail_first=False):
    factory = _FakePoolFactory(fail_first=fail_first)
    pool = V2.ReadyWorkerPoolV2.create(
        states=_identity_states(), search_plan=_plan(),
        benchmark_v2_contract_digest="8" * 64,
        predecessor_scientific_input_bindings_digest="6" * 64,
        executor_factory=factory,
        coordination_factory=_FakeCoordination,
        readiness_timeout_s=3.0)
    return pool, factory


def _patch_benchmark(monkeypatch):
    import scipy.optimize
    from scipy.optimize import Bounds, LinearConstraint

    variable_count = V1.PREFIX_STATE_COUNT * V1.ROTATION_COUNT
    constraints = LinearConstraint(
        np.zeros((0, variable_count), dtype=np.float64),
        np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64))
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    monkeypatch.setattr(
        V1.ALLOC, "_constraint_system", lambda _states: (constraints, bounds))

    def fake_milp(**options):
        objective = np.asarray(options["c"], dtype=np.float64)
        lower = np.asarray(options["bounds"].lb, dtype=np.float64)
        solution = lower.copy()
        nonzero = np.flatnonzero(objective)
        state_start = int(nonzero[0]) - 1
        solution[state_start] = 1.0
        return SimpleNamespace(
            status=0, success=True, x=solution, fun=0.0,
            message="synthetic optimal")

    monkeypatch.setattr(scipy.optimize, "milp", fake_milp)


def _benchmark_clock(*, fixed_elapsed=4.0):
    values = []
    cursor = 1.0
    for _ in V2.SAMPLE_PREFIX_INDICES:
        values.extend((cursor, cursor + 10.0))
        cursor += 10.0
        values.extend((cursor, cursor + fixed_elapsed))
        cursor += fixed_elapsed + 1.0
    iterator = iter(values)
    return lambda: next(iterator)


def _run_benchmark(monkeypatch, pool, *, fixed_elapsed=4.0):
    _patch_benchmark(monkeypatch)
    pool.readiness_completed_monotonic = 0.0
    return V2.run_ready_fixed_rotation_benchmark_v2(
        states=_identity_states(), search_plan=_plan(),
        source_binding_digest="9" * 64,
        bound_v1_failure_receipt_digest="7" * 64,
        predecessor_scientific_input_bindings_digest="6" * 64,
        ready_pool=pool,
        clock=_benchmark_clock(fixed_elapsed=fixed_elapsed))


def test_readiness_forces_exact_distinct_generation_and_keeps_pool_live():
    pool, factory = _ready_pool()
    try:
        record = pool.readiness_record
        assert V2.validate_readiness_record(
            record, expected_benchmark_v2_contract_digest="8" * 64,
            expected_predecessor_scientific_input_bindings_digest=
                "6" * 64) == record
        assert record["worker_count"] == 32
        assert len({row["worker_pid"] for row in record["worker_rows"]}) == 32
        assert len({row["worker_instance_id"]
                    for row in record["worker_rows"]}) == 32
        assert record["solver_call_count"] == 0
        assert record["candidate_outcomes_consumed"] is False
        assert record["scientific_masks_accessed"] is False
        assert factory.pools[0].shutdown_calls == []
    finally:
        pool.shutdown()


def test_readiness_allows_only_one_complete_pre_sample_rebuild():
    pool, factory = _ready_pool(fail_first=True)
    try:
        assert len(factory.pools) == 2
        assert factory.pools[0].shutdown_calls == [{
            "wait": True, "cancel_futures": True}]
        assert pool.readiness_record["pool_generation"] == 1
        assert pool.readiness_record["rebuild_count"] == 1
        assert [row["status"] for row in
                pool.readiness_record["pool_construction_attempts"]] == [
                    "FAILED_READINESS", "READY"]
    finally:
        pool.shutdown()


def test_second_readiness_failure_is_terminal_without_a_third_pool():
    class AlwaysFail(_FakePoolFactory):
        def __call__(self, **options):
            pool = _FakeProcessPool(options, fail_readiness=True)
            self.pools.append(pool)
            return pool

    factory = AlwaysFail()
    with pytest.raises(V2.WorkerReadinessError, match="single permitted") as error:
        V2.ReadyWorkerPoolV2.create(
            states=_identity_states(), search_plan=_plan(),
            benchmark_v2_contract_digest="8" * 64,
            predecessor_scientific_input_bindings_digest="6" * 64,
            executor_factory=factory,
            coordination_factory=_FakeCoordination,
            readiness_timeout_s=1.0)
    assert len(factory.pools) == 2
    assert [row["pool_generation"]
            for row in error.value.attempt_rows] == [0, 1]
    assert all(pool.shutdown_calls == [{
        "wait": True, "cancel_futures": True}] for pool in factory.pools)


def test_readiness_refuses_any_nonprojection_field_before_worker_start():
    states = _identity_states()
    states[0]["candidate_outcome"] = "forbidden"
    factory = _FakePoolFactory()
    with pytest.raises(ValueError, match="six-field|outcome-free"):
        V2.ReadyWorkerPoolV2.create(
            states=states, search_plan=_plan(),
            benchmark_v2_contract_digest="8" * 64,
            predecessor_scientific_input_bindings_digest="6" * 64,
            executor_factory=factory,
            coordination_factory=_FakeCoordination)
    assert factory.pools == []


def test_benchmark_uses_ready_pool_exact_samples_and_unchanged_gate(monkeypatch):
    pool, factory = _ready_pool()
    try:
        receipt = _run_benchmark(monkeypatch, pool)
        assert receipt["passes"] is True
        assert receipt["sample_prefix_indices"] == [0, 1, 2]
        assert receipt["median_parallel_fraction"] == pytest.approx(0.4)
        assert receipt["maximum_parallel_fraction_observed"] \
            == pytest.approx(0.4)
        assert [row["equivalence_verdict"]
                for row in receipt["sample_rows"]] == ["PASS"] * 3
        assert [row["timeout_verdict"]
                for row in receipt["sample_rows"]] == ["PASS"] * 3
        assert all(row["worker_pool_identity"] == pool.worker_pool_identity
                   for row in receipt["sample_rows"])
        assert factory.pools[0].shutdown_calls == []
        assert factory.pools[0]._fixed_submissions == 3 * V1.ROTATION_COUNT
        assert V2.validate_benchmark_receipt_v2(
            receipt,
            expected_benchmark_v2_contract_digest="8" * 64,
            expected_v1_failure_receipt_digest="7" * 64,
            expected_predecessor_scientific_input_bindings_digest="6" * 64,
            expected_source_binding_digest="9" * 64,
            require_pass=True) == receipt
    finally:
        pool.shutdown()


def test_benchmark_preserves_failed_max_gate_and_validator_rejects_tamper(
        monkeypatch):
    pool, _factory = _ready_pool()
    try:
        receipt = _run_benchmark(monkeypatch, pool, fixed_elapsed=6.0)
        assert receipt["median_gate_passes"] is False
        assert receipt["maximum_gate_passes"] is False
        assert receipt["passes"] is False
        with pytest.raises(V1.ParallelSearchError, match="gate"):
            V2.validate_benchmark_receipt_v2(
                receipt,
                expected_benchmark_v2_contract_digest="8" * 64,
                expected_v1_failure_receipt_digest="7" * 64,
                expected_predecessor_scientific_input_bindings_digest=
                    "6" * 64,
                expected_source_binding_digest="9" * 64,
                require_pass=True)

        tampered = copy.deepcopy(receipt)
        tampered["sample_rows"][0]["worker_restarted"] = True
        tampered["sample_rows_digest"] = V1.canonical_digest(
            tampered["sample_rows"])
        tampered["benchmark_receipt_digest"] = V1.canonical_digest({
            key: value for key, value in tampered.items()
            if key != "benchmark_receipt_digest"})
        with pytest.raises(V1.ParallelSearchError, match="sample|binding"):
            V2.validate_benchmark_receipt_v2(
                tampered,
                expected_benchmark_v2_contract_digest="8" * 64,
                expected_v1_failure_receipt_digest="7" * 64,
                expected_predecessor_scientific_input_bindings_digest=
                    "6" * 64,
                expected_source_binding_digest="9" * 64)
    finally:
        pool.shutdown()


def test_pid_change_after_sample0_is_terminal_and_never_rebuilds(monkeypatch):
    pool, factory = _ready_pool()
    _patch_benchmark(monkeypatch)
    pool.readiness_completed_monotonic = 0.0
    original = pool.assert_integrity
    calls = []

    def changing(stage):
        calls.append(stage)
        if stage == "immediately_after_sample_0":
            factory.pools[0].force_pid_change = True
        return original(stage)

    pool.assert_integrity = changing
    try:
        with pytest.raises(V2.WorkerPoolIntegrityError, match="generation"):
            V2.run_ready_fixed_rotation_benchmark_v2(
                states=_identity_states(), search_plan=_plan(),
                source_binding_digest="9" * 64,
                bound_v1_failure_receipt_digest="7" * 64,
                predecessor_scientific_input_bindings_digest="6" * 64,
                ready_pool=pool, clock=_benchmark_clock())
        assert pool.sample0_started is True
        assert pool.worker_restart_count == 1
        assert len(factory.pools) == 1
    finally:
        pool.shutdown()


def test_pass_gate_injects_exact_live_pool_into_scientific_search(
        monkeypatch, tmp_path):
    pool, factory = _ready_pool()
    try:
        receipt = _run_benchmark(monkeypatch, pool)
        final_plan = _plan(
            measured_digest=receipt["benchmark_receipt_digest"])
        calls = []

        def fake_search(**options):
            calls.append(options)
            assert options["process_executor"] is pool
            assert "executor_factory" not in options
            assert factory.pools[0].shutdown_calls == []
            return {"status": "EXHAUSTED", "synthetic": True}

        monkeypatch.setattr(V1, "run_scientific_parallel_search", fake_search)
        result = V2.run_scientific_parallel_search_v2(
            ready_pool=pool, benchmark_receipt=receipt,
            expected_v1_failure_receipt_digest="7" * 64,
            expected_predecessor_scientific_input_bindings_digest="6" * 64,
            expected_source_binding_digest="9" * 64,
            search_plan=final_plan, checkpoint_root=tmp_path,
            prepare_rank=pytest.fail, classify_mask=pytest.fail,
            validate_winner=pytest.fail, telemetry=None)
        assert result == {"status": "EXHAUSTED", "synthetic": True}
        assert len(calls) == 1
        assert pool.search_started is True
        assert factory.pools[0].shutdown_calls == []
    finally:
        pool.shutdown()


def test_v1_external_process_executor_is_never_owned_or_shutdown(monkeypatch,
                                                                 tmp_path):
    # A zero-rank domain is impossible through the public plan, so exercise the
    # external-ownership branch with a one-rank, immediately infeasible wave.
    plan = V1.build_search_plan(
        candidate_scene_ids=[f"scene-{index}" for index in range(5)],
        combination_size=5, worker_count=32,
        source_repository_commit="a" * 40,
        clean_source_launch_receipt_digest="b" * 64,
        state_selector_amendment_digest="c" * 64,
        candidate_allocation_amendment_digest="d" * 64,
        fixed_state_projection_digest="e" * 64,
        resolver_cursor_scene_id="cursor",
        solver_identity={"name": "synthetic"},
        solver_options={"threads": 1}, active_rank_window=1)

    class External:
        def __init__(self):
            self.shutdown_calls = []

        def submit(self, _function, task):
            future = concurrent.futures.Future()
            future.set_result({
                "rotation": task["rotation"], "status": "INFEASIBLE",
                "message": "synthetic", "elapsed_s": 0.0,
                "solver_call_count": 1, "worker_pid": 1,
                "thread_environment": dict(V1.THREAD_ENVIRONMENT),
            })
            return future

        def shutdown(self, **options):
            self.shutdown_calls.append(options)

    external = External()
    result = V1.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=tmp_path,
        prepare_rank=lambda _rank, _combination: {
            "states": _identity_states(),
            "source_identity_manifest_digest": "9" * 64,
            "mask_context": {},
        },
        classify_mask=lambda *_args: pytest.fail("mask accessed"),
        validate_winner=lambda *_args: pytest.fail("winner accessed"),
        telemetry=None, process_executor=external)
    assert result["status"] == "EXHAUSTED"
    assert external.shutdown_calls == []


def test_v1_rejects_ambiguous_external_executor_ownership():
    with pytest.raises(ValueError, match="mutually exclusive"):
        V1.run_scientific_parallel_search(
            search_plan=_plan(), checkpoint_root="synthetic-unused",
            prepare_rank=pytest.fail, classify_mask=pytest.fail,
            validate_winner=pytest.fail, telemetry=None,
            executor_factory=pytest.fail, process_executor=object())
