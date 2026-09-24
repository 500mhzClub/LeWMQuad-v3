#!/usr/bin/env python3
"""Issue and execute the one-shot warm-pool small-completion V2.

The two explicit stages enforce the authority boundary:

* ``issue-contract`` writes the source-bound V2 contract while no worker
  exists and while every V2 runtime path is absent.
* ``run-one-shot`` claims the sole attempt, constructs and readies one
  32-worker pool, runs the exact three-sample benchmark, and either freezes a
  terminal failure or (only after PASS) writes the search plan, opens the
  preserved mask context, and continues directly on that same live pool.

No import or contract-only path starts a worker, reads a scientific mask, or
performs a MILP solve.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT, ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from scripts import build_go2_branch_corpus_v1_2 as BUILDER
from lewm.oracle import (
    go2_parallel_small_completion_benchmark_v2_contract as CONTRACT,
)
from lewm.oracle import go2_parallel_small_completion_search_v1 as SEARCH_V1
from lewm.oracle import go2_parallel_small_completion_search_v2 as SEARCH_V2


ATTEMPT_SCHEMA = "go2_parallel_small_completion_benchmark_v2_attempt_start"
FAILURE_SCHEMA = "go2_parallel_small_completion_benchmark_v2_terminal_failure"
TERMINAL_SCHEMA = "go2_parallel_small_completion_search_v2_terminal_wrapper"
SHUTDOWN_SCHEMA = "go2_parallel_small_completion_search_v2_pool_shutdown"

ATTEMPT_SELF_KEY = "attempt_start_receipt_digest"
FAILURE_SELF_KEY = "terminal_failure_receipt_digest"
TERMINAL_SELF_KEY = "terminal_wrapper_digest"
SHUTDOWN_SELF_KEY = "pool_shutdown_receipt_digest"

_RUNTIME_PATHS = {
    label: ROOT / relative
    for label, relative, _kind in CONTRACT.V2_RUNTIME_OUTPUT_PATHS
}
_V1_DOWNSTREAM_PATHS = {
    label: ROOT / relative
    for label, relative, _kind in CONTRACT.V1_DOWNSTREAM_OUTPUT_PATHS
}
CONTRACT_PATH = ROOT / CONTRACT.CONTRACT_RELATIVE_PATH


class OneShotV2Error(RuntimeError):
    """The one-shot runner cannot safely proceed or freeze its terminal state."""


def _with_digest(payload: Mapping[str, Any], self_key: str) -> dict[str, Any]:
    result = dict(payload)
    if self_key in result:
        raise OneShotV2Error(f"{self_key} was supplied before self-digesting")
    result[self_key] = SEARCH_V1.canonical_digest(result)
    return result


def _pin_generated(raw_path: Path) -> Path:
    return BUILDER._pin_generated_path(raw_path, raw_path)


def _require_absent(raw_path: Path, *, label: str) -> None:
    pinned = _pin_generated(raw_path)
    if pinned.exists() or pinned.is_symlink():
        raise OneShotV2Error(f"one-shot path already exists: {label}")


def _require_runtime_namespace_absent() -> None:
    for label, raw_path in _RUNTIME_PATHS.items():
        _require_absent(raw_path, label=f"V2 {label}")
    for label, raw_path in _V1_DOWNSTREAM_PATHS.items():
        _require_absent(raw_path, label=f"V1 downstream {label}")


def _exclusive_json(raw_path: Path, payload: Mapping[str, Any], *,
                    label: str) -> dict[str, Any]:
    """Install one immutable JSON artifact with O_EXCL and reopen exactly."""

    # Runtime receipts are deliberately restricted to ordinary JSON-native
    # mappings.  Encoding with allow_nan=False is the fail-closed type gate.
    expected = dict(payload)
    pinned = _pin_generated(raw_path)
    if not pinned.parent.is_dir() or pinned.parent.is_symlink():
        raise OneShotV2Error(f"{label} parent is unavailable")
    encoded = (json.dumps(
        expected, indent=2, sort_keys=True, ensure_ascii=True,
        allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(pinned, flags, 0o444)
    except FileExistsError as exc:
        raise OneShotV2Error(f"{label} already exists") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(
            pinned.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        # The exclusive path is never deleted or replaced after installation.
        raise
    if (not pinned.is_file() or pinned.is_symlink()
            or pinned.read_bytes() != encoded):
        raise OneShotV2Error(f"{label} exact reopen changed")
    return dict(expected)


def _contract_and_inputs() -> tuple[
        dict[str, Any], dict[str, Any], dict[str, Any], str]:
    predecessor = BUILDER.build_v2_predecessor_scientific_input_bindings()
    contract = CONTRACT.load_contract(
        CONTRACT_PATH,
        expected_predecessor_scientific_input_bindings=predecessor,
        root=ROOT,
    )
    predecessor_digest = SEARCH_V1.canonical_digest(predecessor)
    if (contract["predecessor_scientific_input_bindings_digest"]
            != predecessor_digest):
        raise OneShotV2Error("contract predecessor input digest changed")
    inputs = BUILDER.load_v2_parallel_small_benchmark_inputs(
        predecessor_scientific_input_bindings=predecessor)
    if (inputs.get("candidate_outcomes_consumed") is not False
            or inputs.get("scientific_masks_accessed") is not False
            or "preserved_vectors" in inputs):
        raise OneShotV2Error("pre-gate input loader crossed a scientific gate")
    return contract, predecessor, inputs, predecessor_digest


def _provisional_plan(*, contract: Mapping[str, Any], inputs: Mapping[str, Any],
                      predecessor_digest: str) -> dict[str, Any]:
    return BUILDER.build_v2_parallel_search_plan(
        inputs,
        source_repository_commit=str(contract["source_repository_commit"]),
        benchmark_v2_contract_digest=str(
            contract[CONTRACT.SELF_DIGEST_KEY]),
        predecessor_scientific_input_bindings_digest=predecessor_digest,
        measured_benchmark_receipt_digest=None,
    )


def _rank_zero_projected_states(inputs: Mapping[str, Any]) -> list[
        dict[str, str]]:
    combination = SEARCH_V1.unrank_combination(
        0, len(inputs["raw_candidates"]), 5)
    material = BUILDER._parallel_rank_identity_material(
        inputs, 0, combination)
    projected = SEARCH_V1.project_allocator_identity_states(
        material["states"])
    envelope = inputs["predecessor_scientific_input_bindings"]
    if (SEARCH_V1.canonical_digest(projected)
            != envelope["rank_zero_state_projection_digest"]
            or SEARCH_V1.ALLOC.pre_outcome_identity_digest(projected)
            != envelope["rank_zero_source_identity_manifest_digest"]):
        raise OneShotV2Error("rank-zero V2 benchmark identity changed")
    return projected


def _attempt_receipt(*, contract: Mapping[str, Any], plan: Mapping[str, Any],
                     predecessor_digest: str) -> dict[str, Any]:
    return _with_digest({
        "schema": ATTEMPT_SCHEMA,
        "status": "CLAIMED_SINGLE_V2_ATTEMPT_BEFORE_POOL_CONSTRUCTION",
        "attempt_count": 1,
        "benchmark_v2_contract_digest": contract[CONTRACT.SELF_DIGEST_KEY],
        "source_repository_commit": contract["source_repository_commit"],
        "bound_v1_failure_receipt_digest":
            CONTRACT.V1_FAILURE_RECEIPT_DIGEST,
        "v1_failure_disposition": CONTRACT.V1_FAILURE_STATUS_DESCRIPTOR,
        "predecessor_scientific_input_bindings_digest": predecessor_digest,
        "provisional_search_plan_digest": plan["search_plan_digest"],
        "immutable_search_input_digest":
            SEARCH_V2.immutable_search_input_digest(plan),
        "worker_count": SEARCH_V2.WORKER_COUNT,
        "worker_pool_constructed": False,
        "timed_sample_zero_started": False,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": False,
        "scientific_search_plan_issued": False,
        "scientific_search_started": False,
    }, ATTEMPT_SELF_KEY)


def _shutdown_receipt(*, contract_digest: str,
                      attempt_digest: str,
                      pool: SEARCH_V2.ReadyWorkerPoolV2 | None,
                      reason: str,
                      readiness_digest: str | None,
                      benchmark_digest: str | None,
                      search_plan_digest: str | None,
                      search_started: bool,
                      search_result_digest: str | None,
                      masks_accessed: bool) -> dict[str, Any]:
    return _with_digest({
        "schema": SHUTDOWN_SCHEMA,
        "status": "POOL_CLOSED_NOTHING_RUNNING",
        "reason": reason,
        "benchmark_v2_contract_digest": contract_digest,
        "attempt_start_receipt_digest": attempt_digest,
        "readiness_record_digest": readiness_digest,
        "benchmark_receipt_digest": benchmark_digest,
        "search_plan_digest": search_plan_digest,
        "search_result_digest": search_result_digest,
        "worker_pool_constructed": pool is not None,
        "worker_pool_identity": (
            None if pool is None else pool.worker_pool_identity),
        "worker_count": 0 if pool is None else SEARCH_V2.WORKER_COUNT,
        "worker_restart_count": (
            0 if pool is None else pool.worker_restart_count),
        "pool_closed": True,
        "scientific_search_started": search_started,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": masks_accessed,
    }, SHUTDOWN_SELF_KEY)


def _failure_receipt(*, contract: Mapping[str, Any], attempt: Mapping[str, Any],
                     predecessor_digest: str, failure_kind: str,
                     message: str, error_type: str,
                     readiness: Mapping[str, Any] | None,
                     readiness_attempts: list[dict[str, Any]],
                     benchmark: Mapping[str, Any] | None,
                     shutdown: Mapping[str, Any], plan_issued: bool,
                     search_started: bool, masks_accessed: bool) -> dict[str, Any]:
    return _with_digest({
        "schema": FAILURE_SCHEMA,
        "status": "IMMUTABLE_ONE_SHOT_V2_FAILURE",
        "failure_kind": failure_kind,
        "error_type": error_type,
        "error_message": message,
        "benchmark_v2_contract_digest": contract[CONTRACT.SELF_DIGEST_KEY],
        "source_repository_commit": contract["source_repository_commit"],
        "attempt_start_receipt_digest": attempt[ATTEMPT_SELF_KEY],
        "bound_v1_failure_receipt_digest":
            CONTRACT.V1_FAILURE_RECEIPT_DIGEST,
        "v1_failure_disposition": CONTRACT.V1_FAILURE_STATUS_DESCRIPTOR,
        "predecessor_scientific_input_bindings_digest": predecessor_digest,
        "readiness_record": None if readiness is None else dict(readiness),
        "readiness_record_digest": (
            None if readiness is None
            else readiness["readiness_record_digest"]),
        "readiness_failure_attempts": readiness_attempts,
        "benchmark_receipt": None if benchmark is None else dict(benchmark),
        "benchmark_receipt_digest": (
            None if benchmark is None
            else benchmark["benchmark_receipt_digest"]),
        "median_gate_passes": (
            None if benchmark is None else benchmark["median_gate_passes"]),
        "maximum_gate_passes": (
            None if benchmark is None else benchmark["maximum_gate_passes"]),
        "overall_v2_passes": (
            None if benchmark is None else benchmark["passes"]),
        "pool_shutdown_receipt_digest": shutdown[SHUTDOWN_SELF_KEY],
        "worker_pool_identity": shutdown["worker_pool_identity"],
        "worker_restart_count": shutdown["worker_restart_count"],
        "pool_closed_nothing_running": True,
        "scientific_search_plan_issued": plan_issued,
        "scientific_search_started": search_started,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed": masks_accessed,
        "v2_retry_permitted": False,
        "automatic_v3_permitted": False,
    }, FAILURE_SELF_KEY)


def _terminal_receipt(*, contract: Mapping[str, Any],
                      attempt: Mapping[str, Any],
                      predecessor_digest: str,
                      readiness: Mapping[str, Any],
                      benchmark: Mapping[str, Any],
                      plan: Mapping[str, Any],
                      search_result: Mapping[str, Any],
                      shutdown: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(search_result)
    return _with_digest({
        "schema": TERMINAL_SCHEMA,
        "status": "COMPLETE_AUTHORISED_SEARCH_AFTER_V2_PASS",
        "benchmark_v2_contract_digest": contract[CONTRACT.SELF_DIGEST_KEY],
        "source_repository_commit": contract["source_repository_commit"],
        "attempt_start_receipt_digest": attempt[ATTEMPT_SELF_KEY],
        "bound_v1_failure_receipt_digest":
            CONTRACT.V1_FAILURE_RECEIPT_DIGEST,
        "v1_failure_disposition": CONTRACT.V1_FAILURE_STATUS_DESCRIPTOR,
        "predecessor_scientific_input_bindings_digest": predecessor_digest,
        "readiness_record_digest": readiness["readiness_record_digest"],
        "benchmark_receipt_digest": benchmark["benchmark_receipt_digest"],
        "search_plan_digest": plan["search_plan_digest"],
        "worker_pool_identity": benchmark["worker_pool_identity"],
        "same_pool_continued_from_readiness_through_search": True,
        "worker_restart_count": benchmark["worker_restart_count"],
        "search_result": result,
        "search_result_digest": SEARCH_V1.canonical_digest(result),
        "search_status": result.get("status"),
        "pool_shutdown_receipt_digest": shutdown[SHUTDOWN_SELF_KEY],
        "pool_closed_nothing_running": True,
        "candidate_outcomes_consumed": False,
        "scientific_masks_accessed_only_after_v2_pass": True,
    }, TERMINAL_SELF_KEY)


def issue_contract() -> dict[str, Any]:
    predecessor = BUILDER.build_v2_predecessor_scientific_input_bindings()
    contract = CONTRACT.issue_contract(
        CONTRACT_PATH,
        predecessor_scientific_input_bindings=predecessor,
        root=ROOT,
    )
    print(json.dumps({
        "status": contract["status"],
        "source_repository_commit": contract["source_repository_commit"],
        "benchmark_v2_contract_digest": contract[CONTRACT.SELF_DIGEST_KEY],
        "bound_v1_failure_receipt_digest":
            CONTRACT.V1_FAILURE_RECEIPT_DIGEST,
        "v1_failure_disposition": CONTRACT.V1_FAILURE_STATUS_DESCRIPTOR,
        "predecessor_scientific_input_bindings_digest": contract[
            "predecessor_scientific_input_bindings_digest"],
        "worker_pool_constructed": False,
    }, indent=2, sort_keys=True))
    return contract


def _telemetry(event: Mapping[str, Any]) -> None:
    print(json.dumps({"scientific_search_telemetry": dict(event)},
                     sort_keys=True), flush=True)


def run_one_shot() -> int:
    _require_runtime_namespace_absent()
    contract, _predecessor, inputs, predecessor_digest = \
        _contract_and_inputs()
    provisional = _provisional_plan(
        contract=contract, inputs=inputs,
        predecessor_digest=predecessor_digest)
    projected_states = _rank_zero_projected_states(inputs)
    attempt = _attempt_receipt(
        contract=contract, plan=provisional,
        predecessor_digest=predecessor_digest)
    _exclusive_json(
        _RUNTIME_PATHS["attempt_start_receipt"], attempt,
        label="V2 attempt-start receipt")

    pool: SEARCH_V2.ReadyWorkerPoolV2 | None = None
    readiness: dict[str, Any] | None = None
    benchmark: dict[str, Any] | None = None
    final_plan: dict[str, Any] | None = None
    masks_accessed = False
    plan_written = False
    search_started = False
    terminal_written = False
    shutdown_written = False
    shutdown_receipt: dict[str, Any] | None = None
    try:
        pool = SEARCH_V2.ReadyWorkerPoolV2.create(
            states=projected_states,
            search_plan=provisional,
            benchmark_v2_contract_digest=contract[CONTRACT.SELF_DIGEST_KEY],
            predecessor_scientific_input_bindings_digest=predecessor_digest,
        )
        readiness = SEARCH_V2.validate_readiness_record(
            pool.readiness_record,
            expected_benchmark_v2_contract_digest=
                contract[CONTRACT.SELF_DIGEST_KEY],
            expected_predecessor_scientific_input_bindings_digest=
                predecessor_digest,
        )
        _exclusive_json(
            _RUNTIME_PATHS["worker_readiness_record"], readiness,
            label="V2 worker-readiness record")

        benchmark = SEARCH_V2.run_ready_fixed_rotation_benchmark_v2(
            states=projected_states,
            search_plan=provisional,
            source_binding_digest=inputs[
                "predecessor_v1_benchmark_source_binding_digest"],
            bound_v1_failure_receipt_digest=
                CONTRACT.V1_FAILURE_RECEIPT_DIGEST,
            predecessor_scientific_input_bindings_digest=predecessor_digest,
            ready_pool=pool,
        )
        _exclusive_json(
            _RUNTIME_PATHS["benchmark_receipt"], benchmark,
            label="V2 benchmark receipt")

        if benchmark["passes"] is not True:
            pool.shutdown(wait=True, cancel_futures=False)
            shutdown = _shutdown_receipt(
                contract_digest=contract[CONTRACT.SELF_DIGEST_KEY],
                attempt_digest=attempt[ATTEMPT_SELF_KEY], pool=pool,
                reason="V2_BENCHMARK_GATE_FAIL",
                readiness_digest=readiness["readiness_record_digest"],
                benchmark_digest=benchmark["benchmark_receipt_digest"],
                search_plan_digest=None, search_started=False,
                search_result_digest=None, masks_accessed=False)
            _exclusive_json(
                _RUNTIME_PATHS[
                    "same_pool_terminal_wrapper_shutdown_receipt"],
                shutdown, label="V2 pool-shutdown receipt")
            shutdown_written = True
            shutdown_receipt = shutdown
            failure = _failure_receipt(
                contract=contract, attempt=attempt,
                predecessor_digest=predecessor_digest,
                failure_kind="BENCHMARK_GATE_FAIL", message=(
                    "unchanged median/max V2 benchmark gate did not pass"),
                error_type="BenchmarkGateFailure", readiness=readiness,
                readiness_attempts=[], benchmark=benchmark,
                shutdown=shutdown, plan_issued=False,
                search_started=False, masks_accessed=False)
            _exclusive_json(
                _RUNTIME_PATHS["terminal_failure"], failure,
                label="V2 terminal failure receipt")
            print(json.dumps({
                "status": failure["status"],
                "failure_kind": failure["failure_kind"],
                "benchmark_v2_contract_digest":
                    contract[CONTRACT.SELF_DIGEST_KEY],
                "readiness_record_digest":
                    readiness["readiness_record_digest"],
                "startup_prewarm_wall_s": readiness[
                    "successful_startup_prewarm_wall_s"],
                "sample_rows": benchmark["sample_rows"],
                "median_parallel_fraction":
                    benchmark["median_parallel_fraction"],
                "maximum_parallel_fraction": benchmark[
                    "maximum_parallel_fraction_observed"],
                "median_gate_passes": benchmark["median_gate_passes"],
                "maximum_gate_passes": benchmark["maximum_gate_passes"],
                "passes": False,
                "worker_restart_count": benchmark["worker_restart_count"],
                "benchmark_receipt_digest":
                    benchmark["benchmark_receipt_digest"],
                "terminal_failure_receipt_digest":
                    failure[FAILURE_SELF_KEY],
                "scientific_search_plan_issued": False,
                "candidate_outcomes_consumed": False,
                "scientific_masks_accessed": False,
                "pool_closed_nothing_running": True,
            }, indent=2, sort_keys=True), flush=True)
            return 2

        # Freeze PASS before issuing the plan or opening any scientific mask.
        final_plan = BUILDER.build_v2_parallel_search_plan(
            inputs,
            source_repository_commit=str(contract["source_repository_commit"]),
            benchmark_v2_contract_digest=contract[CONTRACT.SELF_DIGEST_KEY],
            predecessor_scientific_input_bindings_digest=predecessor_digest,
            measured_benchmark_receipt_digest=benchmark[
                "benchmark_receipt_digest"],
        )
        _exclusive_json(
            _RUNTIME_PATHS["scientific_search_plan"], final_plan,
            label="V2 scientific search plan")
        plan_written = True
        # Any failure inside the sole post-PASS bridge is conservatively
        # recorded as scientific-mask access, even if it fails mid-validation.
        masks_accessed = True
        attached_inputs = BUILDER.attach_v2_parallel_search_mask_context(
            inputs, v2_pass_receipt=benchmark)
        prepare_rank, classify_mask, validate_winner = \
            BUILDER._parallel_search_callbacks(attached_inputs)
        raw_checkpoint = _RUNTIME_PATHS["scientific_search_checkpoint_root"]
        checkpoint = _pin_generated(raw_checkpoint)
        if checkpoint.exists() or checkpoint.is_symlink():
            raise OneShotV2Error("V2 checkpoint root predates PASS search")
        search_started = True
        search_result = SEARCH_V2.run_scientific_parallel_search_v2(
            ready_pool=pool,
            benchmark_receipt=benchmark,
            expected_v1_failure_receipt_digest=
                CONTRACT.V1_FAILURE_RECEIPT_DIGEST,
            expected_predecessor_scientific_input_bindings_digest=
                predecessor_digest,
            expected_source_binding_digest=inputs[
                "predecessor_v1_benchmark_source_binding_digest"],
            search_plan=final_plan,
            checkpoint_root=checkpoint,
            prepare_rank=prepare_rank,
            classify_mask=classify_mask,
            validate_winner=validate_winner,
            telemetry=_telemetry,
        )
        result_digest = SEARCH_V1.canonical_digest(search_result)
        pool.shutdown(wait=True, cancel_futures=False)
        shutdown = _shutdown_receipt(
            contract_digest=contract[CONTRACT.SELF_DIGEST_KEY],
            attempt_digest=attempt[ATTEMPT_SELF_KEY], pool=pool,
            reason="AUTHORISED_SCIENTIFIC_SEARCH_COMPLETE",
            readiness_digest=readiness["readiness_record_digest"],
            benchmark_digest=benchmark["benchmark_receipt_digest"],
            search_plan_digest=final_plan["search_plan_digest"],
            search_started=True, search_result_digest=result_digest,
            masks_accessed=True)
        _exclusive_json(
            _RUNTIME_PATHS[
                "same_pool_terminal_wrapper_shutdown_receipt"],
            shutdown, label="V2 pool-shutdown receipt")
        shutdown_written = True
        shutdown_receipt = shutdown
        terminal = _terminal_receipt(
            contract=contract, attempt=attempt,
            predecessor_digest=predecessor_digest, readiness=readiness,
            benchmark=benchmark, plan=final_plan,
            search_result=search_result, shutdown=shutdown)
        _exclusive_json(
            _RUNTIME_PATHS["terminal_result"], terminal,
            label="V2 terminal result")
        terminal_written = True
        print(json.dumps({
            "status": terminal["status"],
            "benchmark_v2_contract_digest":
                contract[CONTRACT.SELF_DIGEST_KEY],
            "readiness_record_digest": readiness["readiness_record_digest"],
            "startup_prewarm_wall_s": readiness[
                "successful_startup_prewarm_wall_s"],
            "sample_rows": benchmark["sample_rows"],
            "median_parallel_fraction": benchmark[
                "median_parallel_fraction"],
            "maximum_parallel_fraction": benchmark[
                "maximum_parallel_fraction_observed"],
            "median_gate_passes": True,
            "maximum_gate_passes": True,
            "passes": True,
            "worker_restart_count": benchmark["worker_restart_count"],
            "benchmark_receipt_digest": benchmark[
                "benchmark_receipt_digest"],
            "scientific_search_plan_issued": True,
            "same_pool_continued_into_search": True,
            "search_status": terminal["search_status"],
            "terminal_wrapper_digest": terminal[TERMINAL_SELF_KEY],
            "pool_closed_nothing_running": True,
        }, indent=2, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        # A frozen gate failure follows its dedicated branch above.  Every
        # exceptional path closes the only pool before installing failure.
        if terminal_written:
            raise
        if pool is not None and not pool.closed:
            pool.shutdown(wait=True, cancel_futures=False)
        readiness_attempts = (
            [dict(row) for row in exc.attempt_rows]
            if isinstance(exc, SEARCH_V2.WorkerReadinessError) else [])
        if isinstance(exc, SEARCH_V2.WorkerReadinessError):
            kind = "WORKER_READINESS_FAILURE"
        elif isinstance(exc, SEARCH_V2.WorkerPoolIntegrityError):
            kind = "WORKER_POOL_INTEGRITY_FAILURE"
        elif benchmark is not None and benchmark.get("passes") is True:
            kind = "SCIENTIFIC_SEARCH_FATAL_AFTER_V2_PASS"
        else:
            kind = "BENCHMARK_EXECUTION_FATAL"
        shutdown = shutdown_receipt or _shutdown_receipt(
            contract_digest=contract[CONTRACT.SELF_DIGEST_KEY],
            attempt_digest=attempt[ATTEMPT_SELF_KEY], pool=pool,
            reason=kind,
            readiness_digest=(None if readiness is None else
                              readiness["readiness_record_digest"]),
            benchmark_digest=(None if benchmark is None else
                              benchmark["benchmark_receipt_digest"]),
            search_plan_digest=(None if not plan_written else
                                final_plan["search_plan_digest"]),
            search_started=search_started,
            search_result_digest=None,
            masks_accessed=masks_accessed)
        if not shutdown_written:
            _exclusive_json(
                _RUNTIME_PATHS[
                    "same_pool_terminal_wrapper_shutdown_receipt"],
                shutdown, label="V2 pool-shutdown receipt")
            shutdown_written = True
        failure = _failure_receipt(
            contract=contract, attempt=attempt,
            predecessor_digest=predecessor_digest,
            failure_kind=kind, message=str(exc),
            error_type=type(exc).__name__, readiness=readiness,
            readiness_attempts=readiness_attempts, benchmark=benchmark,
            shutdown=shutdown, plan_issued=plan_written,
            search_started=search_started, masks_accessed=masks_accessed)
        _exclusive_json(
            _RUNTIME_PATHS["terminal_failure"], failure,
            label="V2 terminal failure receipt")
        print(json.dumps({
            "status": failure["status"],
            "failure_kind": failure["failure_kind"],
            "error_type": failure["error_type"],
            "error_message": failure["error_message"],
            "benchmark_v2_contract_digest":
                contract[CONTRACT.SELF_DIGEST_KEY],
            "terminal_failure_receipt_digest": failure[FAILURE_SELF_KEY],
            "worker_restart_count": shutdown["worker_restart_count"],
            "scientific_search_plan_issued": plan_written,
            "scientific_search_started": search_started,
            "candidate_outcomes_consumed": False,
            "scientific_masks_accessed": masks_accessed,
            "pool_closed_nothing_running": True,
        }, indent=2, sort_keys=True), flush=True)
        return 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True,
        choices=("issue-contract", "run-one-shot"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "issue-contract":
        issue_contract()
        return 0
    return run_one_shot()


if __name__ == "__main__":
    raise SystemExit(main())
