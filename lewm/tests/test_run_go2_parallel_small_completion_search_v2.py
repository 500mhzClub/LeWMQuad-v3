"""Solver-free orchestration tests for the one-shot warm-pool V2 runner.

All authorities, inputs, pools, timings, receipts, masks, callbacks, and
search results in this module are synthetic.  The tests never read the real
generated corpus and never construct a process, simulator, or solver.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts import run_go2_parallel_small_completion_search_v2 as runner


_CONTRACT_DIGEST = "a" * 64
_SOURCE_BINDING_DIGEST = "c" * 64
_READINESS_DIGEST = "d" * 64
_BENCHMARK_DIGEST = "e" * 64


class _FakePool:
    def __init__(self) -> None:
        self.readiness_record = {
            "readiness_record_digest": _READINESS_DIGEST,
            "successful_startup_prewarm_wall_s": 1.25,
        }
        self.worker_pool_identity = "f" * 64
        self.worker_restart_count = 0
        self.closed = False
        self.shutdown_calls: list[dict[str, Any]] = []

    def shutdown(self, **options: Any) -> None:
        self.shutdown_calls.append(dict(options))
        self.closed = True


def _benchmark(*, passes: bool) -> dict[str, Any]:
    return {
        "benchmark_receipt_digest": _BENCHMARK_DIGEST,
        "passes": passes,
        "median_gate_passes": passes,
        "maximum_gate_passes": passes,
        "median_parallel_fraction": 0.25 if passes else 0.75,
        "maximum_parallel_fraction_observed": 0.4 if passes else 0.9,
        "sample_rows": [
            {
                "sample_prefix_index": index,
                "fixed_elapsed_s": 4.0,
                "candidate_elapsed_s": 10.0,
                "parallel_fraction": 0.4,
            }
            for index in (0, 1, 2)
        ],
        "worker_pool_identity": "f" * 64,
        "worker_restart_count": 0,
    }


@pytest.fixture
def isolated_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect every runner artifact and replace all scientific inputs."""

    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    runtime_paths = {
        label: runtime_root / f"{label}.json"
        for label in runner._RUNTIME_PATHS
    }
    # The checkpoint is the sole directory-shaped runtime artifact.
    runtime_paths["scientific_search_checkpoint_root"] = \
        runtime_root / "checkpoint"
    v1_root = tmp_path / "v1-downstream"
    v1_root.mkdir()
    v1_paths = {
        label: v1_root / f"{label}.json"
        for label in runner._V1_DOWNSTREAM_PATHS
    }
    contract_path = tmp_path / "contract.json"

    monkeypatch.setattr(runner, "_RUNTIME_PATHS", runtime_paths)
    monkeypatch.setattr(runner, "_V1_DOWNSTREAM_PATHS", v1_paths)
    monkeypatch.setattr(runner, "CONTRACT_PATH", contract_path)
    monkeypatch.setattr(runner, "_pin_generated", lambda path: Path(path))

    events: list[str] = []
    def recording_write(path, payload, *, label):
        del label
        target = Path(path)
        events.append(f"write:{target.name}")
        with target.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return dict(payload)

    monkeypatch.setattr(runner, "_exclusive_json", recording_write)

    predecessor = {
        "schema": "synthetic_predecessor_bindings",
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
    }
    predecessor_digest = runner.SEARCH_V1.canonical_digest(predecessor)
    inputs = {
        "predecessor_scientific_input_bindings": predecessor,
        "predecessor_v1_benchmark_source_binding_digest":
            _SOURCE_BINDING_DIGEST,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
    }
    contract = {
        "status": "ISSUED_BEFORE_WORKER_POOL",
        "source_repository_commit": "1" * 40,
        runner.CONTRACT.SELF_DIGEST_KEY: _CONTRACT_DIGEST,
        "predecessor_scientific_input_bindings_digest": predecessor_digest,
    }

    monkeypatch.setattr(
        runner.BUILDER, "build_v2_predecessor_scientific_input_bindings",
        lambda: dict(predecessor))
    monkeypatch.setattr(
        runner.CONTRACT, "load_contract",
        lambda *_args, **_kwargs: dict(contract))
    monkeypatch.setattr(
        runner.BUILDER, "load_v2_parallel_small_benchmark_inputs",
        lambda **_kwargs: dict(inputs))
    monkeypatch.setattr(runner, "_rank_zero_projected_states",
                        lambda _inputs: [{"synthetic": "state"}])
    monkeypatch.setattr(
        runner.SEARCH_V2, "immutable_search_input_digest",
        lambda _plan: "2" * 64)

    plan_calls: list[str | None] = []

    def build_plan(_inputs, **options):
        measured = options["measured_benchmark_receipt_digest"]
        plan_calls.append(measured)
        events.append("build:provisional" if measured is None
                      else "build:final-plan")
        return {
            "search_plan_digest": "3" * 64 if measured is None else "4" * 64,
            "measured_benchmark_receipt_digest": measured,
        }

    monkeypatch.setattr(
        runner.BUILDER, "build_v2_parallel_search_plan", build_plan)

    return {
        "runtime_paths": runtime_paths,
        "contract_path": contract_path,
        "events": events,
        "plan_calls": plan_calls,
        "predecessor": predecessor,
        "inputs": inputs,
        "contract": contract,
        "predecessor_digest": predecessor_digest,
    }


def _configure_pool_and_readiness(
        monkeypatch: pytest.MonkeyPatch, pool: _FakePool,
        events: list[str]) -> None:
    def create(**_options):
        events.append("pool:create")
        return pool

    monkeypatch.setattr(
        runner.SEARCH_V2.ReadyWorkerPoolV2, "create", staticmethod(create))

    def validate(payload, **_options):
        assert payload is pool.readiness_record
        events.append("readiness:validate")
        return dict(payload)

    monkeypatch.setattr(
        runner.SEARCH_V2, "validate_readiness_record", validate)


def test_issue_contract_never_constructs_pool(
        isolated_runner, monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, Any]] = []

    def issue(path, **options):
        calls.append({"path": path, **options})
        return dict(isolated_runner["contract"])

    monkeypatch.setattr(runner.CONTRACT, "issue_contract", issue)
    monkeypatch.setattr(
        runner.SEARCH_V2.ReadyWorkerPoolV2, "create",
        staticmethod(lambda **_kwargs: pytest.fail(
            "contract issuance constructed a worker pool")))

    result = runner.issue_contract()

    assert result == isolated_runner["contract"]
    assert calls == [{
        "path": isolated_runner["contract_path"],
        "predecessor_scientific_input_bindings":
            isolated_runner["predecessor"],
        "root": runner.ROOT,
    }]


def test_gate_fail_closes_pool_and_freezes_failure_without_science(
        isolated_runner, monkeypatch: pytest.MonkeyPatch):
    pool = _FakePool()
    events = isolated_runner["events"]
    _configure_pool_and_readiness(monkeypatch, pool, events)

    def run_benchmark(**options):
        assert options["ready_pool"] is pool
        events.append("benchmark:run")
        return _benchmark(passes=False)

    monkeypatch.setattr(
        runner.SEARCH_V2, "run_ready_fixed_rotation_benchmark_v2",
        run_benchmark)
    monkeypatch.setattr(
        runner.BUILDER, "attach_v2_parallel_search_mask_context",
        lambda *_args, **_kwargs: pytest.fail("FAIL gate opened masks"))
    monkeypatch.setattr(
        runner.SEARCH_V2, "run_scientific_parallel_search_v2",
        lambda **_kwargs: pytest.fail("FAIL gate started search"))

    assert runner.run_one_shot() == 2

    paths = isolated_runner["runtime_paths"]
    assert pool.closed is True
    assert pool.shutdown_calls == [{"wait": True, "cancel_futures": False}]
    assert paths["worker_readiness_record"].is_file()
    assert paths["benchmark_receipt"].is_file()
    assert paths["terminal_failure"].is_file()
    assert not paths["scientific_search_plan"].exists()
    assert not paths["scientific_search_checkpoint_root"].exists()
    failure = json.loads(paths["terminal_failure"].read_text())
    assert failure["failure_kind"] == "BENCHMARK_GATE_FAIL"
    assert failure["scientific_search_plan_issued"] is False
    assert failure["scientific_search_started"] is False
    assert failure["scientific_masks_accessed"] is False
    assert failure["pool_closed_nothing_running"] is True
    assert isolated_runner["plan_calls"] == [None]


def test_pass_freezes_benchmark_then_plan_then_masks_and_same_pool_search(
        isolated_runner, monkeypatch: pytest.MonkeyPatch):
    pool = _FakePool()
    events = isolated_runner["events"]
    _configure_pool_and_readiness(monkeypatch, pool, events)
    benchmark = _benchmark(passes=True)

    def run_benchmark(**options):
        assert options["ready_pool"] is pool
        events.append("benchmark:run")
        return dict(benchmark)

    monkeypatch.setattr(
        runner.SEARCH_V2, "run_ready_fixed_rotation_benchmark_v2",
        run_benchmark)

    attached = {"synthetic": "attached-mask-context"}

    def attach(inputs, *, v2_pass_receipt):
        assert inputs == isolated_runner["inputs"]
        assert v2_pass_receipt == benchmark
        events.append("masks:attach")
        return attached

    monkeypatch.setattr(
        runner.BUILDER, "attach_v2_parallel_search_mask_context", attach)

    callbacks = (lambda *_args: None,) * 3

    def make_callbacks(inputs):
        assert inputs is attached
        events.append("callbacks:create")
        return callbacks

    monkeypatch.setattr(
        runner.BUILDER, "_parallel_search_callbacks", make_callbacks)

    search_calls: list[dict[str, Any]] = []

    def run_search(**options):
        events.append("search:run")
        search_calls.append(options)
        return {"status": "COMPLETE", "synthetic": True}

    monkeypatch.setattr(
        runner.SEARCH_V2, "run_scientific_parallel_search_v2", run_search)

    assert runner.run_one_shot() == 0

    paths = isolated_runner["runtime_paths"]
    assert len(search_calls) == 1
    assert search_calls[0]["ready_pool"] is pool
    assert search_calls[0]["benchmark_receipt"] == benchmark
    assert search_calls[0]["prepare_rank"] is callbacks[0]
    assert search_calls[0]["classify_mask"] is callbacks[1]
    assert search_calls[0]["validate_winner"] is callbacks[2]
    benchmark_write = events.index(
        f"write:{paths['benchmark_receipt'].name}")
    final_plan_build = events.index("build:final-plan")
    plan_write = events.index(
        f"write:{paths['scientific_search_plan'].name}")
    masks_attach = events.index("masks:attach")
    search_run = events.index("search:run")
    assert benchmark_write < final_plan_build < plan_write < masks_attach < \
        search_run
    assert isolated_runner["plan_calls"] == [None, _BENCHMARK_DIGEST]
    assert pool.closed is True
    assert pool.shutdown_calls == [{"wait": True, "cancel_futures": False}]
    assert paths["terminal_result"].is_file()
    assert not paths["terminal_failure"].exists()
    terminal = json.loads(paths["terminal_result"].read_text())
    assert terminal["same_pool_continued_from_readiness_through_search"] is True
    assert terminal["scientific_masks_accessed_only_after_v2_pass"] is True


def test_readiness_exception_freezes_immutable_failure_without_plan(
        isolated_runner, monkeypatch: pytest.MonkeyPatch):
    events = isolated_runner["events"]
    readiness_attempts = [{
        "attempt_index": 0,
        "status": "FAILED_BEFORE_SAMPLE_ZERO",
    }]

    def fail_create(**_options):
        events.append("pool:create")
        raise runner.SEARCH_V2.WorkerReadinessError(
            "synthetic readiness barrier failed",
            attempt_rows=readiness_attempts)

    monkeypatch.setattr(
        runner.SEARCH_V2.ReadyWorkerPoolV2, "create",
        staticmethod(fail_create))
    monkeypatch.setattr(
        runner.SEARCH_V2, "run_ready_fixed_rotation_benchmark_v2",
        lambda **_kwargs: pytest.fail("benchmark ran after readiness failure"))
    monkeypatch.setattr(
        runner.BUILDER, "attach_v2_parallel_search_mask_context",
        lambda *_args, **_kwargs: pytest.fail(
            "readiness failure opened masks"))

    assert runner.run_one_shot() == 3

    paths = isolated_runner["runtime_paths"]
    assert paths["attempt_start_receipt"].is_file()
    assert paths["terminal_failure"].is_file()
    assert not paths["worker_readiness_record"].exists()
    assert not paths["benchmark_receipt"].exists()
    assert not paths["scientific_search_plan"].exists()
    failure = json.loads(paths["terminal_failure"].read_text())
    assert failure["failure_kind"] == "WORKER_READINESS_FAILURE"
    assert failure["readiness_failure_attempts"] == readiness_attempts
    assert failure["scientific_search_plan_issued"] is False
    assert failure["scientific_masks_accessed"] is False
    assert failure["v2_retry_permitted"] is False


def test_preexisting_attempt_receipt_blocks_a_second_run_before_pool(
        isolated_runner, monkeypatch: pytest.MonkeyPatch):
    attempt_path = isolated_runner["runtime_paths"]["attempt_start_receipt"]
    attempt_path.write_text("immutable first attempt\n")
    monkeypatch.setattr(
        runner.SEARCH_V2.ReadyWorkerPoolV2, "create",
        staticmethod(lambda **_kwargs: pytest.fail(
            "second run constructed a worker pool")))

    with pytest.raises(runner.OneShotV2Error, match="already exists"):
        runner.run_one_shot()

    assert attempt_path.read_text() == "immutable first attempt\n"


def test_exclusive_json_installs_once_and_never_rewrites(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "receipt.json"
    monkeypatch.setattr(runner, "_pin_generated", lambda value: Path(value))

    assert runner._exclusive_json(
        path, {"schema": "synthetic", "value": 1}, label="receipt") == {
            "schema": "synthetic", "value": 1}
    original = path.read_bytes()

    with pytest.raises(runner.OneShotV2Error, match="already exists"):
        runner._exclusive_json(
            path, {"schema": "synthetic", "value": 2}, label="receipt")
    assert path.read_bytes() == original
