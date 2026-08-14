"""Pure tests for the parallel small-completion lexicographic search.

These tests deliberately use only synthetic integer combinations and a tiny
fixed-rotation solver abstraction.  They must never load a scene, simulator,
branch outcome, frame, latent, predictor, or scorer checkpoint.
"""
from __future__ import annotations

import copy
import concurrent.futures
import functools
import itertools
import json
import random
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.oracle import go2_parallel_small_completion_search_v1 as PSEARCH


def _all_combinations(n: int, k: int) -> list[tuple[int, ...]]:
    return list(itertools.combinations(range(n), k))


@pytest.mark.parametrize(
    ("n", "k"),
    [(5, 5), (6, 5), (8, 3), (9, 0), (12, 1)],
)
def test_rank_unrank_exactly_matches_itertools(n, k):
    expected = _all_combinations(n, k)
    assert PSEARCH.combination_count(n, k) == len(expected)
    for rank, combination in enumerate(expected):
        assert PSEARCH.unrank_combination(rank, n, k) == combination
        assert PSEARCH.rank_combination(combination, n, k) == rank


@pytest.mark.parametrize(
    ("rank", "n", "k"),
    [(-1, 6, 5), (6, 6, 5), (0, 4, 5)],
)
def test_unrank_rejects_out_of_range_or_impossible_domains(rank, n, k):
    with pytest.raises((TypeError, ValueError)):
        PSEARCH.unrank_combination(rank, n, k)


@pytest.mark.parametrize(
    ("combination", "n", "k"),
    [
        ((0, 1, 1), 6, 3),
        ((0, 3, 2), 6, 3),
        ((-1, 1, 2), 6, 3),
        ((0, 1, 6), 6, 3),
        ((0, 1), 6, 3),
    ],
)
def test_rank_rejects_noncanonical_combination(combination, n, k):
    with pytest.raises((TypeError, ValueError)):
        PSEARCH.rank_combination(combination, n, k)


@pytest.mark.parametrize("workers", [1, 2, 32])
def test_contiguous_partitions_are_complete_disjoint_and_balanced(workers):
    total = PSEARCH.combination_count(11, 5)
    partitions = PSEARCH.contiguous_partitions(total, workers)
    assert len(partitions) == workers
    assert partitions[0][0] == 0
    assert partitions[-1][1] == total
    assert all(start <= stop for start, stop in partitions)
    assert all(left[1] == right[0]
               for left, right in itertools.pairwise(partitions))
    flattened = [rank for start, stop in partitions
                 for rank in range(start, stop)]
    assert flattened == list(range(total))
    assert max(stop - start for start, stop in partitions) - min(
        stop - start for start, stop in partitions) <= 1


def test_contiguous_partitions_allow_more_workers_than_ranks():
    partitions = PSEARCH.contiguous_partitions(3, 8)
    assert len(partitions) == 8
    assert [rank for start, stop in partitions for rank in range(start, stop)] \
        == [0, 1, 2]
    assert sum(start == stop for start, stop in partitions) == 5


def _frontier(total: int):
    return PSEARCH.OrderedFrontier(total_rank_count=total)


def _record(frontier, rank: int, classification: str):
    return frontier.record(rank=rank, classification=classification)


def test_ordered_frontier_selects_earliest_pass_under_random_completion_order():
    results = {rank: "NONPASS" for rank in range(12)}
    results[4] = "PASS"
    results[9] = "PASS"
    completion_order = list(results)
    random.Random(7103).shuffle(completion_order)
    frontier = _frontier(len(results))
    selected = None
    for rank in completion_order:
        event = _record(frontier, rank, results[rank])
        if event is not None and event.get("status") == "PASS":
            selected = event
    assert selected is not None
    assert selected["rank"] == 4
    assert frontier.committed_rank_count == 5
    assert frontier.terminal is True


def test_ordered_frontier_does_not_surface_pass_before_lower_ranks_resolve():
    frontier = _frontier(8)
    assert _record(frontier, 5, "PASS") is None
    for rank in range(4):
        assert _record(frontier, rank, "NONPASS") is None
    event = _record(frontier, 4, "NONPASS")
    assert event == {"status": "PASS", "rank": 5}


def test_ordered_frontier_actions_fatal_only_when_it_reaches_frontier():
    frontier = _frontier(7)
    assert _record(frontier, 4, "FATAL") is None
    assert _record(frontier, 5, "PASS") is None
    for rank in range(3):
        assert _record(frontier, rank, "NONPASS") is None
    event = _record(frontier, 3, "NONPASS")
    assert event == {"status": "FATAL", "rank": 4}
    assert frontier.committed_rank_count == 5


def test_ordered_frontier_exhaustion_is_exact_and_duplicate_safe():
    frontier = _frontier(5)
    order = [4, 1, 3, 0, 2]
    events = [_record(frontier, rank, "NONPASS") for rank in order]
    assert events[-1] == {"status": "EXHAUSTED", "rank_count": 5}
    assert frontier.committed_rank_count == 5
    assert frontier.terminal is True
    with pytest.raises(ValueError, match="duplicate|terminal"):
        _record(frontier, 2, "NONPASS")


def test_wave_decision_ignores_only_fatal_rotations_above_proven_minimum():
    higher_fatal = ["INFEASIBLE"] * PSEARCH.ROTATION_COUNT
    higher_fatal[3] = "FEASIBLE"
    higher_fatal[8] = "FATAL"
    assert PSEARCH._lexicographic_wave_decision(
        higher_fatal, state_index=17) == ("SELECTED", 3)

    lower_fatal = list(higher_fatal)
    lower_fatal[1] = "FATAL"
    assert PSEARCH._lexicographic_wave_decision(
        lower_fatal, state_index=17) == ("FATAL", None)

    all_infeasible = ["INFEASIBLE"] * PSEARCH.ROTATION_COUNT
    assert PSEARCH._lexicographic_wave_decision(
        all_infeasible, state_index=0) == ("ALLOCATOR_INFEASIBLE", None)
    assert PSEARCH._lexicographic_wave_decision(
        all_infeasible, state_index=1) == ("FATAL", None)


def _plan(*, worker_count: int = 4,
          active_rank_window: int | None = None,
          candidate_count: int = 8) -> dict:
    return PSEARCH.build_search_plan(
        candidate_scene_ids=[
            f"scene-{index:03d}" for index in range(candidate_count)],
        combination_size=5,
        worker_count=worker_count,
        source_repository_commit="a" * 40,
        clean_source_launch_receipt_digest="b" * 64,
        state_selector_amendment_digest="c" * 64,
        candidate_allocation_amendment_digest="d" * 64,
        fixed_state_projection_digest="e" * 64,
        resolver_cursor_scene_id="scene-before-pool",
        solver_identity={"name": "synthetic", "version": "1"},
        solver_options={"threads": 1, "mip_rel_gap": 0.0},
        active_rank_window=active_rank_window,
    )


def _rank_receipt(plan: dict, *, rank: int = 0,
                  classification: str = "NONPASS") -> dict:
    combination = PSEARCH.unrank_combination(
        rank, plan["candidate_pool_count"], plan["combination_size"])
    completed = 0 if classification == "ALLOCATOR_INFEASIBLE" else 120
    rotations = [] if completed == 0 else [0] * completed
    allocation_digest = None if completed == 0 else "3" * 64
    assignment_digest = None if completed == 0 else "4" * 64
    return {
        "schema": PSEARCH.RANK_RECEIPT_SCHEMA,
        "search_plan_digest": plan["search_plan_digest"],
        "rank": rank,
        "combination_indices": list(combination),
        "selected_scene_ids": [plan["candidate_scene_ids"][index]
                               for index in combination],
        "projection_digest": "1" * 64,
        "source_identity_manifest_digest": "2" * 64,
        "completed_prefix_wave_count": completed,
        "selected_rotations": rotations,
        "provisional_allocation_manifest_digest": allocation_digest,
        "provisional_candidate_assignment_set_digest": assignment_digest,
        "classification": classification,
        "candidate_outcomes_consumed": False,
    }


def test_search_plan_is_deterministic_self_bound_and_one_threaded():
    first = _plan()
    second = _plan()
    assert first == second
    assert PSEARCH.validate_search_plan(first) == first
    assert first["solver_options"]["threads"] == 1
    changed = json.loads(json.dumps(first))
    changed["candidate_scene_ids"][0] = "scene-tamper"
    with pytest.raises(ValueError, match="digest|candidate"):
        PSEARCH.validate_search_plan(changed)


def test_rank_receipt_atomic_idempotence_and_tamper_rejection(tmp_path):
    plan = _plan()
    path = tmp_path / "ranks/000000000000.json"
    receipt = _rank_receipt(plan)
    first = PSEARCH.write_rank_receipt(path, receipt, search_plan=plan)
    second = PSEARCH.write_rank_receipt(path, receipt, search_plan=plan)
    assert first == second
    assert PSEARCH.load_rank_receipt(
        path, search_plan=plan, expected_rank=0) == first

    changed = dict(receipt)
    changed["classification"] = "PASS"
    with pytest.raises(RuntimeError, match="overwrite|different"):
        PSEARCH.write_rank_receipt(path, changed, search_plan=plan)

    raw = json.loads(path.read_text())
    raw["candidate_outcomes_consumed"] = True
    path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="digest|outcome"):
        PSEARCH.load_rank_receipt(path, search_plan=plan, expected_rank=0)

    moved_path = tmp_path / "ranks/000000000099.json"
    moved_path.write_text(json.dumps(PSEARCH._signed(
        _rank_receipt(plan, rank=1), "rank_receipt_digest")))
    with pytest.raises(RuntimeError, match="rank|binding"):
        PSEARCH.load_rank_receipt(
            moved_path, search_plan=plan, expected_rank=0)

    boolean_rank = _rank_receipt(plan, rank=1)
    boolean_rank["rank"] = True
    boolean_path = tmp_path / "boolean-rank.json"
    boolean_path.write_text(json.dumps(PSEARCH._signed(
        boolean_rank, "rank_receipt_digest")))
    with pytest.raises(RuntimeError, match="rank|binding"):
        PSEARCH.load_rank_receipt(
            boolean_path, search_plan=plan, expected_rank=1)


def test_rank_receipt_rejects_truncated_json_and_symlink(tmp_path):
    plan = _plan()
    truncated = tmp_path / "truncated.json"
    truncated.write_text('{"schema":')
    with pytest.raises(RuntimeError, match="JSON|corrupt|invalid"):
        PSEARCH.load_rank_receipt(
            truncated, search_plan=plan, expected_rank=0)

    target = tmp_path / "target.json"
    PSEARCH.write_rank_receipt(
        target, _rank_receipt(plan), search_plan=plan)
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink|custody"):
        PSEARCH.load_rank_receipt(alias, search_plan=plan, expected_rank=0)


def test_rank_receipt_requires_exact_120_rotations_and_allocation_digests(
        tmp_path):
    plan = _plan()
    valid = _rank_receipt(plan, classification="PASS")
    loaded = PSEARCH.write_rank_receipt(
        tmp_path / "valid.json", valid, search_plan=plan)
    assert loaded["completed_prefix_wave_count"] == 120
    assert len(loaded["selected_rotations"]) == 120
    assert loaded["provisional_allocation_manifest_digest"] == "3" * 64
    assert loaded["provisional_candidate_assignment_set_digest"] == "4" * 64

    for field, value in (
        ("selected_rotations", [0] * 119),
        ("provisional_allocation_manifest_digest", None),
        ("provisional_candidate_assignment_set_digest", None),
    ):
        malformed = _rank_receipt(plan, classification="PASS")
        malformed[field] = value
        with pytest.raises(RuntimeError, match="rotation|allocation|evidence"):
            PSEARCH.write_rank_receipt(
                tmp_path / f"malformed-{field}.json", malformed,
                search_plan=plan)


def test_allocator_infeasible_rank_receipt_has_zero_rotations_and_null_digests(
        tmp_path):
    plan = _plan()
    receipt = _rank_receipt(plan, classification="ALLOCATOR_INFEASIBLE")
    loaded = PSEARCH.write_rank_receipt(
        tmp_path / "infeasible.json", receipt, search_plan=plan)
    assert loaded["completed_prefix_wave_count"] == 0
    assert loaded["selected_rotations"] == []
    assert loaded["provisional_allocation_manifest_digest"] is None
    assert loaded["provisional_candidate_assignment_set_digest"] is None

    malformed = dict(receipt)
    malformed["selected_rotations"] = [0]
    with pytest.raises(RuntimeError, match="prefix-wave|infeasible"):
        PSEARCH.write_rank_receipt(
            tmp_path / "malformed-infeasible.json", malformed,
            search_plan=plan)


def _coordinator_receipt(plan: dict, *, frontier: int = 0) -> dict:
    return {
        "schema": PSEARCH.COORDINATOR_RECEIPT_SCHEMA,
        "search_plan_digest": plan["search_plan_digest"],
        "committed_frontier_rank": frontier,
        "terminal_status": None,
        "candidate_outcomes_consumed": False,
    }


def test_coordinator_receipt_atomic_idempotence_and_plan_binding(tmp_path):
    plan = _plan()
    path = tmp_path / "coordinator.json"
    receipt = _coordinator_receipt(plan, frontier=3)
    first = PSEARCH.write_coordinator_receipt(
        path, receipt, search_plan=plan)
    second = PSEARCH.write_coordinator_receipt(
        path, receipt, search_plan=plan)
    assert second == first
    assert PSEARCH.load_coordinator_receipt(path, search_plan=plan) == first

    changed_plan = _plan()
    changed_plan["search_plan_digest"] = "f" * 64
    with pytest.raises(RuntimeError, match="plan|digest"):
        PSEARCH.load_coordinator_receipt(path, search_plan=changed_plan)


def test_rank_and_coordinator_reject_self_resigned_legacy_schema_aliases(
        tmp_path):
    plan = _plan()
    rank = PSEARCH._signed(
        _rank_receipt(plan), "rank_receipt_digest")
    rank["schema"] = "go2_parallel_small_completion_rank_receipt_v1"
    rank = PSEARCH._signed(rank, "rank_receipt_digest")
    rank_path = tmp_path / "legacy-rank.json"
    rank_path.write_text(json.dumps(rank))
    with pytest.raises(RuntimeError, match="schema|binding|digest"):
        PSEARCH.load_rank_receipt(
            rank_path, search_plan=plan, expected_rank=0)

    coordinator = PSEARCH._signed(
        _coordinator_receipt(plan), "coordinator_receipt_digest")
    coordinator["schema"] = \
        "go2_parallel_small_completion_coordinator_receipt_v1"
    coordinator = PSEARCH._signed(
        coordinator, "coordinator_receipt_digest")
    coordinator_path = tmp_path / "legacy-coordinator.json"
    coordinator_path.write_text(json.dumps(coordinator))
    with pytest.raises(RuntimeError, match="plan|digest|binding"):
        PSEARCH.load_coordinator_receipt(
            coordinator_path, search_plan=plan)


def test_fixed_rotation_parallel_result_equals_serial_lowest_feasible():
    statuses = ["INFEASIBLE"] * 12
    statuses[7] = "FEASIBLE"
    statuses[10] = "FEASIBLE"

    def solve(rotation):
        return statuses[rotation]

    rotations = PSEARCH.fixed_rotation_lexicographic_rotations(
        rotation_count=12, solve_rotation=solve,
        completion_order=[10, 7, 11, 6, 5, 4, 3, 2, 1, 0, 8, 9])
    assert rotations["selected_rotation"] == 7
    assert rotations["statuses"] == statuses
    assert rotations["allocation_bytes"] == \
        PSEARCH.fixed_rotation_lexicographic_rotations(
            rotation_count=12, solve_rotation=solve,
            completion_order=list(range(12)))["allocation_bytes"]


def test_fixed_rotation_fatal_precedes_later_feasible():
    statuses = ["INFEASIBLE"] * 12
    statuses[3] = "FATAL"
    statuses[9] = "FEASIBLE"
    with pytest.raises(RuntimeError, match="rotation 3|FATAL"):
        PSEARCH.fixed_rotation_lexicographic_rotations(
            rotation_count=12,
            solve_rotation=lambda rotation: statuses[rotation],
            completion_order=list(reversed(range(12))))


def _identity_states() -> list[dict[str, str]]:
    material = {
        "general": "landmark_red",
        "safety_enriched": "landmark_blue",
        "completion_enriched": "landmark_green",
    }
    states = []
    for family_index in range(8):
        family = f"family-{family_index}"
        for stratum in PSEARCH.ALLOC.STRATA:
            for ordinal in range(5):
                state_id = f"{family}|{stratum}|{ordinal}"
                states.append({
                    "state_id": state_id,
                    "state_identity_digest":
                        PSEARCH.canonical_digest(state_id),
                    "family": family,
                    "stratum": stratum,
                    "split_role": (
                        "calibration" if ordinal == 0 else "fit"),
                    "goal_type": material[stratum],
                })
    return states


def _full_identity_states() -> list[dict]:
    """Synthetic full selector rows with the allocator identity embedded."""

    return [{
        **state,
        "scene_id": f"scene-for-{state['state_id']}",
        "source_state_digest": PSEARCH.canonical_digest({
            "source": state["state_identity_digest"],
        }),
        "snapshot_step": 17,
        "previous_applied_command": [0.0, 0.0, 0.0],
        "candidate_outcomes_loaded": False,
    } for state in _identity_states()]


def _valid_identity_rotations(states) -> list[int]:
    """A hand-constructed feasible balance witness; this performs no solve."""

    calibration = (
        (11, 2, 9, 5, 10, 8, 4, 3),
        (5, 4, 8, 11, 1, 2, 7, 10),
        (6, 3, 0, 4, 7, 1, 10, 9),
    )
    strata = ("general", "safety_enriched", "completion_enriched")
    fit = (0, 6, 1, 7)
    rotations = []
    for state in PSEARCH.project_allocator_identity_states(states):
        family_index = int(state["family"].rsplit("-", 1)[1])
        ordinal = int(state["state_id"].rsplit("|", 1)[1])
        stratum_index = strata.index(state["stratum"])
        rotations.append(
            calibration[stratum_index][family_index]
            if ordinal == 0 else fit[ordinal - 1])
    return rotations


def test_full_rows_project_to_identical_allocator_and_manifest_bytes():
    identity_only = _identity_states()
    full_rows = _full_identity_states()
    projected = PSEARCH.project_allocator_identity_states(full_rows)
    assert projected == PSEARCH.project_allocator_identity_states(identity_only)
    assert all(set(row) == set(PSEARCH.ALLOCATOR_IDENTITY_FIELDS)
               for row in projected)

    rotations = _valid_identity_rotations(projected)
    legacy_manifest = PSEARCH.materialize_allocation_manifest_single_solve(
        identity_only, source_identity_manifest_digest="9" * 64,
        rotations=rotations)
    full_manifest = PSEARCH.materialize_allocation_manifest_single_solve(
        full_rows, source_identity_manifest_digest="9" * 64,
        rotations=rotations)
    assert full_manifest == legacy_manifest
    assert PSEARCH._json_bytes(full_manifest) == \
        PSEARCH._json_bytes(legacy_manifest)


@functools.lru_cache(maxsize=1)
def _canonical_identity_allocation() -> dict:
    return PSEARCH.ALLOC.build_allocation_manifest(
        _identity_states(), source_identity_manifest_digest="9" * 64)


class _SyntheticFuture:
    def __init__(self, evaluate):
        self._evaluate = evaluate

    def result(self):
        return self._evaluate()


class _SyntheticExecutor:
    def __init__(self, handler):
        self.handler = handler
        self.tasks: list[dict] = []

    def submit(self, _function, task):
        captured = copy.deepcopy(task)
        self.tasks.append(captured)
        return _SyntheticFuture(lambda: self.handler(captured))


class _RecordingPool:
    """Immediate Future-compatible stand-in for one shared process pool."""

    def __init__(self, handler, options):
        self.handler = handler
        self.options = dict(options)
        self.tasks: list[dict] = []
        self.shutdown_calls: list[dict] = []
        self._lock = threading.Lock()

    def submit(self, _function, task):
        captured = copy.deepcopy(task)
        with self._lock:
            self.tasks.append(captured)
        future = concurrent.futures.Future()
        try:
            future.set_result(self.handler(captured))
        except BaseException as exc:
            future.set_exception(exc)
        return future

    def shutdown(self, **options):
        self.shutdown_calls.append(dict(options))


class _RecordingPoolFactory:
    def __init__(self, handler):
        self.handler = handler
        self.pools: list[_RecordingPool] = []

    def __call__(self, **options):
        pool = _RecordingPool(self.handler, options)
        self.pools.append(pool)
        return pool


def _worker_row(task, status):
    return {
        "rotation": task["rotation"],
        "status": status,
        "message": f"synthetic {status.lower()}",
        "elapsed_s": 0.0,
        "solver_call_count": 1,
        "worker_pid": 101,
        "thread_environment": dict(PSEARCH.THREAD_ENVIRONMENT),
    }


def _one_feasible_rotation(task, selected=0):
    status = "FEASIBLE" if task["rotation"] == selected else "INFEASIBLE"
    return _worker_row(task, status)


def test_first_wave_all_infeasible_is_durable_and_zero_new_on_resume(tmp_path):
    states = _identity_states()
    plan = _plan(worker_count=12)
    checkpoint_root = tmp_path / "checkpoints"
    first = _SyntheticExecutor(
        lambda task: _worker_row(task, "INFEASIBLE"))
    with pytest.raises(PSEARCH.ALLOC.CandidateAllocationInfeasible):
        PSEARCH.parallel_lexicographic_rotations(
            states, search_plan=plan, rank=3,
            checkpoint_root=checkpoint_root, executor=first)
    assert len(first.tasks) == 12
    durable = sorted(checkpoint_root.glob("waves/rank-*/prefix-*.json"))
    assert len(durable) == 1
    frozen_bytes = durable[0].read_bytes()
    payload = json.loads(frozen_bytes)
    assert payload["wave_status"] == "ALLOCATOR_INFEASIBLE"
    assert payload["selected_rotation"] is None
    assert [row["status"] for row in payload["rotation_results"]] == \
        ["INFEASIBLE"] * 12

    must_not_run = _SyntheticExecutor(
        lambda _task: pytest.fail("durable infeasible wave was recomputed"))
    with pytest.raises(PSEARCH.ALLOC.CandidateAllocationInfeasible):
        PSEARCH.parallel_lexicographic_rotations(
            states, search_plan=plan, rank=3,
            checkpoint_root=checkpoint_root, executor=must_not_run)
    assert must_not_run.tasks == []
    assert durable[0].read_bytes() == frozen_bytes


def test_mid_rank_wave_crash_resumes_at_first_missing_wave_and_rejects_tamper(
        tmp_path):
    states = _identity_states()
    plan = _plan(worker_count=12)
    checkpoint_root = tmp_path / "checkpoints"
    crash_prefix = 7

    def crash(task):
        if task["state_index"] == crash_prefix and task["rotation"] == 3:
            raise KeyboardInterrupt("synthetic mid-rank crash")
        return _one_feasible_rotation(task)

    interrupted = _SyntheticExecutor(crash)
    with pytest.raises(KeyboardInterrupt, match="mid-rank crash"):
        PSEARCH.parallel_lexicographic_rotations(
            states, search_plan=plan, rank=11,
            checkpoint_root=checkpoint_root, executor=interrupted)
    durable = sorted(checkpoint_root.glob("waves/rank-*/prefix-*.json"))
    assert len(durable) == crash_prefix

    resumed = _SyntheticExecutor(_one_feasible_rotation)
    rotations, evidence = PSEARCH.parallel_lexicographic_rotations(
        states, search_plan=plan, rank=11,
        checkpoint_root=checkpoint_root, executor=resumed)
    assert rotations == [0] * 120
    assert evidence["completed_prefix_waves"] == 120
    assert min(task["state_index"] for task in resumed.tasks) == crash_prefix
    assert len(resumed.tasks) == (120 - crash_prefix) * 12

    tamper_path = sorted(
        checkpoint_root.glob("waves/rank-*/prefix-*.json"))[4]
    tampered = json.loads(tamper_path.read_text())
    tampered["prefix_rotations_before"][0] = 1
    tampered["wave_receipt_digest"] = PSEARCH.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "wave_receipt_digest"
    })
    tamper_path.write_text(json.dumps(tampered))
    must_not_run = _SyntheticExecutor(
        lambda _task: pytest.fail("tampered durable prefix was resumed"))
    with pytest.raises(RuntimeError, match="binding"):
        PSEARCH.parallel_lexicographic_rotations(
            states, search_plan=plan, rank=11,
            checkpoint_root=checkpoint_root, executor=must_not_run)
    assert must_not_run.tasks == []


def test_worker_count_1_2_32_produces_exact_same_rotations_and_allocation(
        tmp_path):
    states = _identity_states()
    source_digest = "9" * 64
    serial = PSEARCH.ALLOC.build_allocation_manifest(
        states, source_identity_manifest_digest=source_digest)
    expected_rotations = [
        row["rotation_index"] for row in serial["assignments"]]
    observed = []
    for worker_count in (1, 2, 32):
        plan = _plan(worker_count=worker_count)

        def solve(task):
            return _one_feasible_rotation(
                task, selected=expected_rotations[task["state_index"]])

        rotations, _evidence = PSEARCH.parallel_lexicographic_rotations(
            states, search_plan=plan, rank=0,
            checkpoint_root=tmp_path / f"workers-{worker_count}",
            executor=_SyntheticExecutor(solve))
        materialized = PSEARCH.materialize_allocation_manifest_single_solve(
            states, source_identity_manifest_digest=source_digest,
            rotations=rotations)
        observed.append((rotations, materialized))
    assert all(rotations == expected_rotations for rotations, _ in observed)
    assert all(manifest == serial for _, manifest in observed)
    assert len({manifest["allocation_manifest_digest"]
                for _, manifest in observed}) == 1


def _worker_task() -> dict:
    return {
        "states": _identity_states(),
        "prefix_rotations": [],
        "state_index": 0,
        "rotation": 0,
        "solver_options": {"threads": 1, "time_limit": 0.01},
    }


def _patch_worker_constraint_system(monkeypatch, *, required_sum=1.0):
    from scipy.optimize import Bounds, LinearConstraint

    variable_count = PSEARCH.PREFIX_STATE_COUNT * PSEARCH.ROTATION_COUNT
    constraints = LinearConstraint(
        np.ones((1, variable_count), dtype=np.float64),
        np.asarray([required_sum]), np.asarray([required_sum]))
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    monkeypatch.setattr(
        PSEARCH.ALLOC, "_constraint_system",
        lambda _states: (constraints, bounds))


def test_solver_timeout_or_limit_is_fatal_never_infeasible(
        monkeypatch):
    import scipy.optimize

    _patch_worker_constraint_system(monkeypatch)
    monkeypatch.setattr(
        scipy.optimize, "milp",
        lambda **_kwargs: SimpleNamespace(
            status=1, success=False, x=None,
            message="Time limit reached"))
    result = PSEARCH._fixed_rotation_worker(_worker_task())
    assert result["status"] == "FATAL"
    assert "status=1" in result["message"]


@pytest.mark.parametrize("tamper", ["fractional", "constraint_residual"])
def test_worker_rejects_integrality_and_constraint_residual_tamper(
        monkeypatch, tamper):
    import scipy.optimize

    solution = np.zeros(
        PSEARCH.PREFIX_STATE_COUNT * PSEARCH.ROTATION_COUNT,
        dtype=np.float64)
    if tamper == "fractional":
        _patch_worker_constraint_system(monkeypatch)
        solution[0] = 0.75
        solution[1] = 0.25
    else:
        # The fixed rotation satisfies the frozen variable bounds but violates
        # this deliberately inconsistent synthetic linear equality.
        _patch_worker_constraint_system(monkeypatch, required_sum=0.0)
        solution[0] = 1.0
    monkeypatch.setattr(
        scipy.optimize, "milp",
        lambda **_kwargs: SimpleNamespace(
            status=0, success=True, x=solution,
            message="synthetic corrupted success"))
    result = PSEARCH._fixed_rotation_worker(_worker_task())
    assert result["status"] == "FATAL"
    expected = "non-integral" if tamper == "fractional" else "linear constraint"
    assert expected in result["message"].lower()


def test_run_parallel_search_crash_resume_skips_only_valid_rank_receipts(
        tmp_path):
    plan = _plan()
    root = tmp_path / "search"
    calls: list[int] = []

    def evaluate(rank, combination):
        calls.append(rank)
        if len(calls) == 3:
            raise KeyboardInterrupt("synthetic crash")
        return _rank_receipt(plan, rank=rank, classification="NONPASS")

    with pytest.raises(KeyboardInterrupt, match="synthetic crash"):
        PSEARCH.run_parallel_search(
            search_plan=plan, output_root=root, evaluate_rank=evaluate)
    durable = sorted((root / "ranks").glob("*.json"))
    assert len(durable) == 2

    resumed_calls: list[int] = []

    def resume(rank, combination):
        resumed_calls.append(rank)
        classification = "PASS" if rank == 4 else "NONPASS"
        return _rank_receipt(plan, rank=rank, classification=classification)

    result = PSEARCH.run_parallel_search(
        search_plan=plan, output_root=root, evaluate_rank=resume)
    assert result["status"] == "PASS"
    assert result["rank"] == 4
    assert not ({0, 1} & set(resumed_calls))
    assert result["combination_attempt_count"] == 5


def test_run_parallel_search_rejects_resigned_wrong_input_binding(tmp_path):
    plan = _plan()
    root = tmp_path / "search"
    rank_path = root / "ranks/000000000000.json"
    wrong = _rank_receipt(plan)
    wrong["selected_scene_ids"][0] = "wrong-scene"
    PSEARCH.write_rank_receipt(rank_path, wrong, search_plan=plan,
                               validate=False)
    with pytest.raises(RuntimeError, match="scene|combination|binding"):
        PSEARCH.run_parallel_search(
            search_plan=plan, output_root=root,
            evaluate_rank=lambda *_args: pytest.fail("must fail pre-evaluate"))


def test_rank_lane_worker_counts_are_output_identical_with_randomized_delays(
        tmp_path):
    results = []
    completion_orders = {}
    for worker_count in (1, 2, 32):
        plan = _plan(worker_count=worker_count)
        barrier = threading.Barrier(3) if worker_count == 32 else None
        completed: list[int] = []
        lock = threading.Lock()

        def evaluate(rank, _combination):
            if barrier is not None and rank < 3:
                barrier.wait(timeout=3.0)
            # Ranks finish in an order different from the canonical commit
            # order when a three-rank lane is available.
            time.sleep({0: 0.03, 1: 0.02, 2: 0.01}.get(rank, 0.0))
            with lock:
                completed.append(rank)
            classification = "PASS" if rank == 4 else "MASK_FAIL"
            return _rank_receipt(
                plan, rank=rank, classification=classification)

        result = PSEARCH.run_parallel_search(
            search_plan=plan,
            output_root=tmp_path / f"rank-lanes-{worker_count}",
            evaluate_rank=evaluate)
        results.append(result)
        completion_orders[worker_count] = completed

    assert results == [{
        "status": "PASS",
        "rank": 4,
        "combination_attempt_count": 5,
        "allocator_infeasible_combination_count": 0,
    }] * 3
    assert completion_orders[32].index(2) < completion_orders[32].index(1) \
        < completion_orders[32].index(0)


def test_benchmark_gate_uses_deterministic_work_units_not_wall_clock():
    assert PSEARCH.benchmark_fixed_rotation_gate(
        serial_work_units=240, parallel_work_units=130,
        maximum_parallel_fraction=0.60)["passes"] is True
    assert PSEARCH.benchmark_fixed_rotation_gate(
        serial_work_units=240, parallel_work_units=150,
        maximum_parallel_fraction=0.60)["passes"] is False


def _fake_allocation(states, *, source_identity_manifest_digest, rotations):
    normalised = PSEARCH.project_allocator_identity_states(states)
    assignments = [{
        **state,
        "rotation_index": int(rotation),
        "candidate_indices": list(PSEARCH.ALLOC.candidate_block(rotation)),
    } for state, rotation in zip(normalised, rotations, strict=True)]
    result = {
        "schema": "synthetic_source_only_allocation",
        "source_identity_manifest_digest": source_identity_manifest_digest,
        "assignments": assignments,
    }
    result["allocation_manifest_digest"] = PSEARCH.canonical_digest(result)
    return result


def _fake_winner_proof(*, states, allocation, search_plan, rank,
                       checkpoint_root, assignment_digest, telemetry=None):
    del telemetry
    normalised = PSEARCH.project_allocator_identity_states(states)
    projection_digest = PSEARCH.canonical_digest(normalised)
    source_digest = allocation["source_identity_manifest_digest"]
    rotations = [row["rotation_index"] for row in allocation["assignments"]]
    prefix = []
    wave_rows = []
    for state_index, rotation in enumerate(rotations):
        payload = {
            "schema": PSEARCH.OBJECTIVE_WAVE_RECEIPT_SCHEMA,
            "search_plan_digest": search_plan["search_plan_digest"],
            "rank": rank,
            "state_index": state_index,
            "projection_digest": projection_digest,
            "source_identity_manifest_digest": source_digest,
            "prefix_rotations_before": list(prefix),
            "certified_rotation": rotation,
            "solver_status": "FEASIBLE",
            "selected_rotation": rotation,
            "objective_value": float(rotation),
            "solver_message": "synthetic source-only objective proof",
            "solver_elapsed_s": 0.01,
            "solver_call_count": 1,
            "thread_environment": dict(PSEARCH.THREAD_ENVIRONMENT),
            "candidate_outcomes_consumed": False,
        }
        payload = PSEARCH._signed(
            payload, "objective_wave_receipt_digest")
        path = PSEARCH._objective_wave_path(
            checkpoint_root, rank, state_index)
        PSEARCH._atomic_create_json(path, payload)
        loaded = PSEARCH._load_json(path)
        wave_rows.append({
            "state_index": state_index,
            "objective_wave_receipt_digest":
                loaded["objective_wave_receipt_digest"],
        })
        prefix.append(rotation)
    receipt = {
        "schema": PSEARCH.WINNER_VALIDATION_RECEIPT_SCHEMA,
        "status": "PASS_BOUNDED_EXACT_CANONICAL_OBJECTIVE",
        "search_plan_digest": search_plan["search_plan_digest"],
        "rank": rank,
        "projection_digest": projection_digest,
        "source_identity_manifest_digest": source_digest,
        "allocation_manifest_digest": allocation["allocation_manifest_digest"],
        "candidate_assignment_set_digest": assignment_digest,
        "selected_rotations": rotations,
        "objective_wave_count": PSEARCH.PREFIX_STATE_COUNT,
        "objective_wave_receipts": wave_rows,
        "solver_time_limit_s": search_plan["solver_options"]["time_limit"],
        "candidate_outcomes_consumed": False,
    }
    receipt = PSEARCH._signed(
        receipt, "winner_validation_receipt_digest")
    path = checkpoint_root / "winner-objective-validation.json"
    PSEARCH._atomic_create_json(path, receipt)
    return PSEARCH._load_json(path)


def _science_callbacks(*, ordered=False):
    rank_two_done = threading.Event()
    rank_one_done = threading.Event()

    def prepare_rank(rank, _combination):
        return {
            "states": _identity_states(),
            "source_identity_manifest_digest": "9" * 64,
            "mask_context": {"rank": rank},
        }

    def classify_mask(_states, _allocation, context):
        rank = context["rank"]
        if ordered and rank == 2:
            rank_two_done.set()
            return True
        if ordered and rank == 1:
            assert rank_two_done.wait(timeout=5.0)
            time.sleep(0.04)
            rank_one_done.set()
            return True
        if ordered and rank == 0:
            assert rank_one_done.wait(timeout=5.0)
            time.sleep(0.04)
            return False
        return rank == 1

    def validate_winner(rank, _states, _allocation, context):
        return rank == 1 and context["rank"] == 1

    return prepare_rank, classify_mask, validate_winner


def _install_fake_science(monkeypatch):
    monkeypatch.setattr(
        PSEARCH, "materialize_allocation_manifest_single_solve",
        _fake_allocation)
    monkeypatch.setattr(
        PSEARCH, "validate_winner_allocation_bounded",
        _fake_winner_proof)


def test_scientific_search_uses_one_shared_32_worker_pool_and_ordered_frontier(
        monkeypatch, tmp_path):
    _install_fake_science(monkeypatch)
    plan = _plan(worker_count=32, active_rank_window=3)
    prepare_rank, classify_mask, validate_winner = _science_callbacks(
        ordered=True)
    factory = _RecordingPoolFactory(_one_feasible_rotation)
    telemetry = []

    result = PSEARCH.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=tmp_path / "scientific",
        prepare_rank=prepare_rank, classify_mask=classify_mask,
        validate_winner=validate_winner, telemetry=telemetry.append,
        executor_factory=factory)

    assert result["status"] == "PASS"
    assert result["rank"] == 1
    assert result["combination_attempt_count"] == 2
    assert result["candidate_outcomes_consumed"] is False
    assert len(factory.pools) == 1
    pool = factory.pools[0]
    assert pool.options["max_workers"] == 32
    assert pool.options["initializer"] is PSEARCH._worker_initialise
    assert pool.options["mp_context"].get_start_method() == "spawn"
    assert pool.shutdown_calls == [{"wait": True, "cancel_futures": False}]
    assert len(pool.tasks) >= 3 * PSEARCH.PREFIX_STATE_COUNT \
        * PSEARCH.ROTATION_COUNT
    assert all(task["solver_options"]["threads"] == 1
               and task["solver_options"]["time_limit"]
               == PSEARCH.DEFAULT_MILP_TIME_LIMIT_S
               for task in pool.tasks)
    completion_order = [row["last_completed_rank"] for row in telemetry
                        if "last_completed_rank" in row]
    assert completion_order.index(2) < completion_order.index(1) \
        < completion_order.index(0)


def test_rank_zero_pass_stops_speculative_lanes_after_current_wave(
        monkeypatch, tmp_path):
    """Canonical PASS drains, but does not extend, speculative wave zero."""

    _install_fake_science(monkeypatch)
    plan = _plan(worker_count=32, active_rank_window=3)
    captured_stop_event = {}
    original_rank_lane = PSEARCH._scientific_rank_lane

    def capture_rank_lane(**options):
        captured_stop_event["event"] = options["stop_event"]
        return original_rank_lane(**options)

    monkeypatch.setattr(PSEARCH, "_scientific_rank_lane", capture_rank_lane)

    class StopAwareFuture:
        def __init__(self, pool, task, rank):
            self.pool = pool
            self.task = task
            self.rank = rank

        def result(self):
            if self.rank == 0 and self.task["state_index"] == 0 \
                    and self.task["rotation"] == 0:
                with self.pool.condition:
                    ready = self.pool.condition.wait_for(
                        lambda: self.pool.task_counts.get(1, 0) == 12
                        and self.pool.task_counts.get(2, 0) == 12,
                        timeout=5.0)
                assert ready, "speculative lanes did not begin their first wave"
            elif self.rank in (1, 2):
                stop_event = captured_stop_event.get("event")
                assert stop_event is not None
                assert stop_event.wait(timeout=5.0), \
                    "speculative solver future was not drained after PASS"
            return _one_feasible_rotation(self.task)

    class StopAwarePool:
        def __init__(self, options):
            self.options = dict(options)
            self.condition = threading.Condition()
            self.task_counts = {}
            self.tasks = []
            self.shutdown_calls = []

        def submit(self, _function, task):
            captured = copy.deepcopy(task)
            marker = captured["states"][0]["goal_type"]
            rank = int(marker.rsplit("-rank-", 1)[1])
            with self.condition:
                self.tasks.append(captured)
                self.task_counts[rank] = self.task_counts.get(rank, 0) + 1
                self.condition.notify_all()
            return StopAwareFuture(self, captured, rank)

        def shutdown(self, **options):
            self.shutdown_calls.append(dict(options))

    class StopAwarePoolFactory:
        def __init__(self):
            self.pools = []

        def __call__(self, **options):
            pool = StopAwarePool(options)
            self.pools.append(pool)
            return pool

    def prepare_rank(rank, _combination):
        states = copy.deepcopy(_identity_states())
        for state in states:
            state["goal_type"] = f"{state['goal_type']}-rank-{rank}"
        return {
            "states": states,
            "source_identity_manifest_digest": "9" * 64,
            "mask_context": {"rank": rank},
        }

    def classify_mask(_states, _allocation, context):
        return context["rank"] == 0

    def validate_winner(rank, _states, _allocation, context):
        return rank == 0 and context["rank"] == 0

    factory = StopAwarePoolFactory()
    root = tmp_path / "early-pass"
    result = PSEARCH.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=root,
        prepare_rank=prepare_rank, classify_mask=classify_mask,
        validate_winner=validate_winner, telemetry=None,
        executor_factory=factory)

    assert result["status"] == "PASS"
    assert result["rank"] == 0
    pool = factory.pools[0]
    assert pool.task_counts == {
        0: PSEARCH.PREFIX_STATE_COUNT * PSEARCH.ROTATION_COUNT,
        1: PSEARCH.ROTATION_COUNT,
        2: PSEARCH.ROTATION_COUNT,
    }
    assert pool.shutdown_calls == [{"wait": True, "cancel_futures": False}]
    for rank in (1, 2):
        wave_paths = sorted(
            (root / "waves" / f"rank-{rank:012d}").glob("prefix-*.json"))
        assert [path.name for path in wave_paths] == ["prefix-000.json"]
        assert not PSEARCH._rank_path(root, rank).exists()


def test_scientific_search_resumes_nonpass_and_pass_without_new_solver_tasks(
        monkeypatch, tmp_path):
    _install_fake_science(monkeypatch)
    plan = _plan(worker_count=32, active_rank_window=1)
    prepare_rank, classify_mask, validate_winner = _science_callbacks()
    root = tmp_path / "resume-scientific"
    first_factory = _RecordingPoolFactory(_one_feasible_rotation)
    first = PSEARCH.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=root, prepare_rank=prepare_rank,
        classify_mask=classify_mask, validate_winner=validate_winner,
        telemetry=None, executor_factory=first_factory)
    rank_bytes = {rank: PSEARCH._rank_path(root, rank).read_bytes()
                  for rank in (0, 1)}

    resumed_factory = _RecordingPoolFactory(
        lambda _task: pytest.fail("resumed rank submitted a solver task"))
    resumed = PSEARCH.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=root, prepare_rank=prepare_rank,
        classify_mask=classify_mask, validate_winner=validate_winner,
        telemetry=None, executor_factory=resumed_factory)

    assert resumed == first
    assert len(resumed_factory.pools) == 1
    assert resumed_factory.pools[0].tasks == []
    assert {rank: PSEARCH._rank_path(root, rank).read_bytes()
            for rank in (0, 1)} == rank_bytes


def test_exhausted_search_validator_replays_full_rows_with_zero_solves(
        monkeypatch, tmp_path):
    _install_fake_science(monkeypatch)
    plan = _plan(worker_count=32, active_rank_window=1, candidate_count=6)
    full_states = _full_identity_states()
    mask_calls = []

    def prepare_rank(rank, _combination):
        assert 0 <= rank < plan["total_rank_count"]
        return {
            "states": copy.deepcopy(full_states),
            "source_identity_manifest_digest": "9" * 64,
            "mask_context": {"rank": rank},
        }

    def classify_mask(states, _allocation, context):
        assert all("source_state_digest" in state for state in states)
        mask_calls.append(context["rank"])
        return False

    def projected_solver(task):
        assert all(set(state) == set(PSEARCH.ALLOCATOR_IDENTITY_FIELDS)
                   for state in task["states"])
        return _one_feasible_rotation(task)

    root = tmp_path / "exhausted-mask-fail"
    result = PSEARCH.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=root,
        prepare_rank=prepare_rank, classify_mask=classify_mask,
        validate_winner=lambda *_args: pytest.fail(
            "EXHAUSTED search invoked winner validation"),
        telemetry=None,
        executor_factory=_RecordingPoolFactory(projected_solver))
    assert result == {
        "schema": PSEARCH.SEARCH_RESULT_SCHEMA,
        "status": "EXHAUSTED",
        "combination_attempt_count": 6,
        "allocator_infeasible_combination_count": 0,
        "search_plan_digest": plan["search_plan_digest"],
        "candidate_outcomes_consumed": False,
    }

    import scipy.optimize
    monkeypatch.setattr(
        scipy.optimize, "milp",
        lambda **_kwargs: pytest.fail("EXHAUSTED replay performed a MILP"))
    certified = PSEARCH.validate_exhausted_search_result(
        terminal_result=result, search_plan=plan, checkpoint_root=root,
        prepare_rank=prepare_rank, classify_mask=classify_mask)
    assert len(certified) == plan["total_rank_count"] == 6
    assert [row["rank"] for row in certified] == list(range(6))
    assert {row["classification"] for row in certified} == {"MASK_FAIL"}
    assert mask_calls == list(range(6)) * 2

    changed_surface = {**result, "rank_receipts": certified}
    with pytest.raises(PSEARCH.ParallelSearchError, match="surface"):
        PSEARCH.validate_exhausted_search_result(
            terminal_result=changed_surface, search_plan=plan,
            checkpoint_root=root, prepare_rank=prepare_rank,
            classify_mask=classify_mask)
    with pytest.raises(PSEARCH.ParallelSearchError, match="PASS|nonpass"):
        PSEARCH.validate_exhausted_search_result(
            terminal_result=result, search_plan=plan, checkpoint_root=root,
            prepare_rank=prepare_rank,
            classify_mask=lambda _states, _allocation, _context: True)

    rank_path = PSEARCH._rank_path(root, 5)
    frozen_rank = rank_path.read_bytes()
    rank_path.unlink()
    with pytest.raises(RuntimeError, match="regular file|receipt"):
        PSEARCH.validate_exhausted_search_result(
            terminal_result=result, search_plan=plan, checkpoint_root=root,
            prepare_rank=prepare_rank, classify_mask=classify_mask)
    rank_path.write_bytes(frozen_rank)

    wave_path = PSEARCH._wave_path(root, 3, 7)
    wave = json.loads(wave_path.read_text())
    wave["selected_rotation"] = 1
    wave["wave_receipt_digest"] = PSEARCH.canonical_digest({
        key: value for key, value in wave.items()
        if key != "wave_receipt_digest"
    })
    wave_path.write_text(json.dumps(wave))
    with pytest.raises(RuntimeError, match="wave|rotation|feasible"):
        PSEARCH.validate_exhausted_search_result(
            terminal_result=result, search_plan=plan, checkpoint_root=root,
            prepare_rank=prepare_rank, classify_mask=classify_mask)


def test_exhausted_search_validator_recounts_first_wave_infeasibility(
        monkeypatch, tmp_path):
    _install_fake_science(monkeypatch)
    plan = _plan(worker_count=32, active_rank_window=1, candidate_count=5)

    def prepare_rank(rank, _combination):
        return {
            "states": _identity_states(),
            "source_identity_manifest_digest": "9" * 64,
            "mask_context": {"rank": rank},
        }

    root = tmp_path / "exhausted-infeasible"
    result = PSEARCH.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=root,
        prepare_rank=prepare_rank,
        classify_mask=lambda *_args: pytest.fail(
            "allocator infeasibility invoked the mask"),
        validate_winner=lambda *_args: pytest.fail(
            "allocator infeasibility invoked winner validation"),
        telemetry=None, executor_factory=_RecordingPoolFactory(
            lambda task: _worker_row(task, "INFEASIBLE")))
    assert result["status"] == "EXHAUSTED"
    assert result["allocator_infeasible_combination_count"] == 1
    certified = PSEARCH.validate_exhausted_search_result(
        terminal_result=result, search_plan=plan, checkpoint_root=root,
        prepare_rank=prepare_rank,
        classify_mask=lambda *_args: pytest.fail(
            "infeasible replay invoked the mask"))
    assert [row["classification"] for row in certified] == \
        ["ALLOCATOR_INFEASIBLE"]

    wrong_count = {**result, "allocator_infeasible_combination_count": 0}
    with pytest.raises(PSEARCH.ParallelSearchError, match="count"):
        PSEARCH.validate_exhausted_search_result(
            terminal_result=wrong_count, search_plan=plan,
            checkpoint_root=root, prepare_rank=prepare_rank,
            classify_mask=lambda *_args: False)
    boolean_attempt_count = {**result, "combination_attempt_count": True}
    with pytest.raises(PSEARCH.ParallelSearchError, match="surface"):
        PSEARCH.validate_exhausted_search_result(
            terminal_result=boolean_attempt_count, search_plan=plan,
            checkpoint_root=root, prepare_rank=prepare_rank,
            classify_mask=lambda *_args: False)


def test_terminal_search_validator_is_zero_solve_and_rejects_lower_rank_omission(
        monkeypatch, tmp_path):
    _install_fake_science(monkeypatch)
    plan = _plan(worker_count=32, active_rank_window=1)
    prepare_rank, classify_mask, validate_winner = _science_callbacks()
    root = tmp_path / "terminal"
    result = PSEARCH.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=root, prepare_rank=prepare_rank,
        classify_mask=classify_mask, validate_winner=validate_winner,
        telemetry=None,
        executor_factory=_RecordingPoolFactory(_one_feasible_rotation))

    import scipy.optimize
    monkeypatch.setattr(
        scipy.optimize, "milp",
        lambda **_kwargs: pytest.fail("terminal validation performed a MILP"))
    certified = PSEARCH.validate_terminal_search_result(
        terminal_result=result, search_plan=plan, checkpoint_root=root,
        prepare_rank=prepare_rank, classify_mask=classify_mask,
        validate_winner=validate_winner)
    assert certified == result["allocation"]

    for field in ("combination_attempt_count",
                  "allocator_infeasible_combination_count"):
        boolean_count = {**result, field: True}
        with pytest.raises(PSEARCH.ParallelSearchError, match="surface"):
            PSEARCH.validate_terminal_search_result(
                terminal_result=boolean_count, search_plan=plan,
                checkpoint_root=root, prepare_rank=prepare_rank,
                classify_mask=classify_mask,
                validate_winner=validate_winner)

    PSEARCH._rank_path(root, 0).unlink()
    with pytest.raises(RuntimeError, match="regular file|receipt"):
        PSEARCH.validate_terminal_search_result(
            terminal_result=result, search_plan=plan, checkpoint_root=root,
            prepare_rank=prepare_rank, classify_mask=classify_mask,
            validate_winner=validate_winner)


def test_terminal_search_validator_rejects_self_resigned_objective_tamper(
        monkeypatch, tmp_path):
    _install_fake_science(monkeypatch)
    plan = _plan(worker_count=32, active_rank_window=1)
    prepare_rank, classify_mask, validate_winner = _science_callbacks()
    root = tmp_path / "terminal-tamper"
    result = PSEARCH.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=root, prepare_rank=prepare_rank,
        classify_mask=classify_mask, validate_winner=validate_winner,
        telemetry=None,
        executor_factory=_RecordingPoolFactory(_one_feasible_rotation))
    path = PSEARCH._objective_wave_path(root, 1, 7)
    tampered = json.loads(path.read_text())
    tampered["selected_rotation"] = 1
    tampered["objective_value"] = 1.0
    tampered["objective_wave_receipt_digest"] = PSEARCH.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "objective_wave_receipt_digest"
    })
    path.write_text(json.dumps(tampered))

    with pytest.raises(PSEARCH.ParallelSearchError, match="objective|certified"):
        PSEARCH.validate_terminal_search_result(
            terminal_result=result, search_plan=plan, checkpoint_root=root,
            prepare_rank=prepare_rank, classify_mask=classify_mask,
            validate_winner=validate_winner)


def test_terminal_search_validator_rejects_self_resigned_winner_receipt_tamper(
        monkeypatch, tmp_path):
    _install_fake_science(monkeypatch)
    plan = _plan(worker_count=32, active_rank_window=1)
    prepare_rank, classify_mask, validate_winner = _science_callbacks()
    root = tmp_path / "winner-receipt-tamper"
    result = PSEARCH.run_scientific_parallel_search(
        search_plan=plan, checkpoint_root=root, prepare_rank=prepare_rank,
        classify_mask=classify_mask, validate_winner=validate_winner,
        telemetry=None,
        executor_factory=_RecordingPoolFactory(_one_feasible_rotation))
    path = root / "winner-objective-validation.json"
    tampered = json.loads(path.read_text())
    tampered["objective_wave_count"] = PSEARCH.PREFIX_STATE_COUNT - 1
    tampered["winner_validation_receipt_digest"] = PSEARCH.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "winner_validation_receipt_digest"
    })
    path.write_text(json.dumps(tampered))

    with pytest.raises(PSEARCH.ParallelSearchError, match="winner|binding"):
        PSEARCH.validate_terminal_search_result(
            terminal_result=result, search_plan=plan, checkpoint_root=root,
            prepare_rank=prepare_rank, classify_mask=classify_mask,
            validate_winner=validate_winner)


def _patch_benchmark_matrix_and_solver(monkeypatch, *, fatal_control=False):
    import scipy.optimize
    from scipy.optimize import Bounds, LinearConstraint

    variable_count = PSEARCH.PREFIX_STATE_COUNT * PSEARCH.ROTATION_COUNT
    constraints = LinearConstraint(
        np.zeros((0, variable_count), dtype=np.float64),
        np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64))
    bounds = Bounds(np.zeros(variable_count), np.ones(variable_count))
    monkeypatch.setattr(
        PSEARCH.ALLOC, "_constraint_system",
        lambda _states: (constraints, bounds))
    calls = []

    def fake_milp(**options):
        calls.append(options)
        assert options["options"]["threads"] == 1
        assert options["options"]["time_limit"] \
            == PSEARCH.DEFAULT_MILP_TIME_LIMIT_S
        if fatal_control:
            return SimpleNamespace(
                status=1, success=False, x=None, fun=None,
                message="synthetic time limit reached")
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
    return calls


def _benchmark_clock(monkeypatch, *, fixed_elapsed_s):
    values = []
    cursor = 0.0
    for _sample in range(3):
        values.extend((cursor, cursor + 10.0))
        cursor += 10.0
        values.extend((cursor, cursor + fixed_elapsed_s))
        cursor += fixed_elapsed_s
    iterator = iter(values)
    monkeypatch.setattr(PSEARCH.time, "monotonic", lambda: next(iterator))


def test_measured_benchmark_samples_0_1_2_exact_choices_and_passes_gate(
        monkeypatch):
    calls = _patch_benchmark_matrix_and_solver(monkeypatch)
    _benchmark_clock(monkeypatch, fixed_elapsed_s=4.0)
    plan = _plan(worker_count=32)
    factory = _RecordingPoolFactory(_one_feasible_rotation)
    source_binding = "8" * 64

    receipt = PSEARCH.run_measured_fixed_rotation_benchmark(
        states=_identity_states(), search_plan=plan,
        source_binding_digest=source_binding,
        executor_factory=factory)

    assert receipt["passes"] is True
    assert receipt["details"]["sample_prefix_indices"] == [0, 1, 2]
    assert [row["objective_rotation"] for row in
            receipt["details"]["sample_rows"]] == [0, 0, 0]
    assert [row["fixed_rotation"] for row in
            receipt["details"]["sample_rows"]] == [0, 0, 0]
    assert receipt["details"]["median_parallel_fraction"] == pytest.approx(0.4)
    assert receipt["details"]["maximum_parallel_fraction_observed"] \
        == pytest.approx(0.4)
    assert PSEARCH.require_measured_benchmark_gate(
        receipt, expected_source_binding_digest=source_binding) == receipt
    assert len(calls) == 3
    assert len(factory.pools) == 1
    assert factory.pools[0].options["max_workers"] == 32
    assert len(factory.pools[0].tasks) == 3 * PSEARCH.ROTATION_COUNT


def test_measured_benchmark_failed_ratio_is_not_runtime_authority(monkeypatch):
    _patch_benchmark_matrix_and_solver(monkeypatch)
    _benchmark_clock(monkeypatch, fixed_elapsed_s=6.0)
    source_binding = "8" * 64
    receipt = PSEARCH.run_measured_fixed_rotation_benchmark(
        states=_identity_states(), search_plan=_plan(worker_count=32),
        source_binding_digest=source_binding,
        executor_factory=_RecordingPoolFactory(_one_feasible_rotation))
    assert receipt["passes"] is False
    with pytest.raises(PSEARCH.ParallelSearchError, match="gate|failed"):
        PSEARCH.require_measured_benchmark_gate(
            receipt, expected_source_binding_digest=source_binding)


def test_measured_benchmark_rejects_choice_mismatch_and_timeout(monkeypatch):
    _patch_benchmark_matrix_and_solver(monkeypatch)
    _benchmark_clock(monkeypatch, fixed_elapsed_s=4.0)
    with pytest.raises(PSEARCH.ParallelSearchFatal, match="choice differs"):
        PSEARCH.run_measured_fixed_rotation_benchmark(
            states=_identity_states(), search_plan=_plan(worker_count=32),
            source_binding_digest="8" * 64,
            executor_factory=_RecordingPoolFactory(
                lambda task: _one_feasible_rotation(task, selected=1)))

    _patch_benchmark_matrix_and_solver(monkeypatch, fatal_control=True)
    _benchmark_clock(monkeypatch, fixed_elapsed_s=4.0)
    with pytest.raises(PSEARCH.ParallelSearchFatal, match="failed|status=1"):
        PSEARCH.run_measured_fixed_rotation_benchmark(
            states=_identity_states(), search_plan=_plan(worker_count=32),
            source_binding_digest="8" * 64,
            executor_factory=_RecordingPoolFactory(_one_feasible_rotation))


def test_measured_benchmark_gate_rejects_self_resigned_row_tamper(monkeypatch):
    _patch_benchmark_matrix_and_solver(monkeypatch)
    _benchmark_clock(monkeypatch, fixed_elapsed_s=4.0)
    source_binding = "8" * 64
    receipt = PSEARCH.run_measured_fixed_rotation_benchmark(
        states=_identity_states(), search_plan=_plan(worker_count=32),
        source_binding_digest=source_binding,
        executor_factory=_RecordingPoolFactory(_one_feasible_rotation))
    tampered = copy.deepcopy(receipt)
    tampered["details"]["sample_rows"][1]["fixed_rotation"] = 1
    tampered["details"]["sample_rows_digest"] = PSEARCH.canonical_digest(
        tampered["details"]["sample_rows"])
    tampered["benchmark_receipt_digest"] = PSEARCH.canonical_digest({
        key: value for key, value in tampered.items()
        if key != "benchmark_receipt_digest"
    })
    with pytest.raises(PSEARCH.ParallelSearchError, match="gate|failed"):
        PSEARCH.require_measured_benchmark_gate(
            tampered, expected_source_binding_digest=source_binding)
