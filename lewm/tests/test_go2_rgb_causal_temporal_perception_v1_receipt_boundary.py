from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import stat
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str) -> Any:
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = _load(
    "_test_temporal_v1_receipt_contract",
    "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py",
)
runner = _load(
    "_test_temporal_v1_receipt_runner",
    "scripts/run_go2_rgb_causal_temporal_perception_v1.py",
)


def _reservation(output: Path) -> tuple[dict[str, Any], bytes]:
    return runner._publish_json(output / "reservation.json", {
        "schema": contract.RESERVATION_SCHEMA,
        "status": "RESERVED_SYNTHETIC_TEST",
        "attempt_identity": "a" * 64,
    })


def _reservation_binding(
    reservation: dict[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    return contract.artifact_binding(
        "reservation.json",
        raw,
        content_sha256=reservation["content_sha256"],
    )


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    core = deepcopy(value)
    core.pop("content_sha256")
    return contract.with_content_sha256(core)


@pytest.mark.parametrize(
    ("injected", "boundary", "header_durable"),
    [
        ("ledger_before_header", "before_header_publication", False),
        (
            "ledger_after_durable_header",
            "after_durable_header_before_constructor_acceptance",
            True,
        ),
    ],
)
def test_actual_pre_ledger_failures_publish_distinct_valid_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected: str,
    boundary: str,
    header_durable: bool,
) -> None:
    output = tmp_path / "attempt"
    output.mkdir(mode=0o700)
    reservation, reservation_raw = _reservation(output)

    def fail(name: str) -> None:
        if name == injected:
            raise RuntimeError(f"synthetic {injected}")

    monkeypatch.setattr(runner, "_failure_boundary", fail)
    with pytest.raises(RuntimeError, match=f"synthetic {injected}"):
        runner._execute_after_reservation(
            review={},
            review_raw=b"",
            authorization={},
            authorization_raw=b"",
            sources={},
            reservation=reservation,
            reservation_raw=reservation_raw,
            preflight={},
            output_root=output,
        )

    failed = contract.parse_canonical_json(
        (output / "failed.json").read_bytes(),
        name="synthetic temporal pre-ledger failure",
    )
    binding = _reservation_binding(reservation, reservation_raw)
    contract.validate_pre_ledger_failure_receipt(
        failed,
        reservation_binding=binding,
        attempt_identity=reservation["attempt_identity"],
    )
    assert failed["status"] == (
        "TERMINAL_CAUSAL_TEMPORAL_V1_POST_RESERVATION_PRE_LEDGER_"
        "FAILURE_NO_RETRY"
    )
    assert failed["failure_stage"] == {
        "name": "partial_access_ledger_initialization",
        "boundary": boundary,
    }
    assert failed["prior_runtime_output_open_count"] == 0
    assert failed["operation_counts"] == contract.empty_partial_operation_counts()
    assert failed["ledger_state"]["standard_ledger_complete"] is False
    assert failed["ledger_state"][
        "standard_failure_validator_applicable"
    ] is False

    header_path = output / runner.PartialAccessLedger.RELATIVE_PATH
    assert header_path.exists() is header_durable
    if header_durable:
        header_raw = header_path.read_bytes()
        assert contract.validate_pre_ledger_header(
            header_raw,
            reservation_binding=binding,
            attempt_identity=reservation["attempt_identity"],
        ) == failed["ledger_state"]["header"]
        with pytest.raises(PermissionError, match="ledger is incomplete"):
            contract.parse_partial_access_ledger(header_raw)
    else:
        assert failed["ledger_state"]["header"] is None

    assert all(
        stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == 0o444
        for path in output.iterdir()
        if path.is_file()
    )
    assert stat.S_IMODE(
        output.stat(follow_symlinks=False).st_mode
    ) == 0o555


def test_hash_chain_accepts_valid_ledger_and_rejects_relinked_record(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    output = repository / "attempt"
    output.mkdir(parents=True, mode=0o700)
    reservation, reservation_raw = _reservation(output)
    ledger = runner._initialize_partial_access_ledger(
        output,
        reservation=reservation,
        reservation_raw=reservation_raw,
        repository_root=repository,
    )
    runtime_input = (
        repository / contract.RAW_ROOT_RELATIVE_PATH / "synthetic.bin"
    )
    runtime_input.parent.mkdir(parents=True)
    runtime_input.write_bytes(b"synthetic runtime input")
    raw = runtime_input.read_bytes()
    assert ledger.read_regular(
        runtime_input,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_byte_count=len(raw),
        kind="raw_supervision",
        stage="synthetic_runtime_load",
        role="train",
        purpose="runtime_load",
    ) == raw
    ledger.append_terminal(
        record_type="ATTEMPT_TERMINATING",
        stage={
            "name": "synthetic_terminal",
            "update": None,
            "microbatch": None,
            "checkpoint_update": None,
            "role": "train",
        },
        operation_counts=contract.empty_partial_operation_counts(),
        error=RuntimeError("synthetic terminal"),
    )
    ledger.binding()
    ledger_raw = b"".join(ledger.raw_parts)
    records = contract.parse_partial_access_ledger(ledger_raw)
    assert [record["record_type"] for record in records] == [
        "LEDGER_OPENED",
        "OPEN_ATTEMPTED",
        "OPEN_OUTCOME",
        "ATTEMPT_TERMINATING",
    ]

    relinked = deepcopy(records)
    relinked[1]["previous_record_content_sha256"] = "f" * 64
    relinked[1] = _rehash(relinked[1])
    relinked_raw = b"".join(
        contract.canonical_json_bytes(record) + b"\n"
        for record in relinked
    )
    with pytest.raises(PermissionError, match="ledger chain changed"):
        contract.parse_partial_access_ledger(relinked_raw)


def test_standard_failure_receipt_rejects_generic_prior_output_count_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "attempt"
    output.mkdir(mode=0o700)
    reservation, reservation_raw = _reservation(output)
    ledger = runner._initialize_partial_access_ledger(
        output,
        reservation=reservation,
        reservation_raw=reservation_raw,
        repository_root=tmp_path,
    )
    progress = runner.OperationProgress()
    progress.enter("synthetic_standard_failure", role="authority")
    runner._terminal_failure(
        output,
        reservation,
        reservation_raw,
        ledger,
        progress,
        error=RuntimeError("synthetic standard failure"),
    )

    failed = contract.parse_canonical_json(
        (output / "failed.json").read_bytes(),
        name="synthetic temporal standard failure",
    )
    binding = _reservation_binding(reservation, reservation_raw)
    contract.validate_failure_receipt(
        failed,
        reservation_binding=binding,
    )
    assert failed["status"] == (
        "TERMINAL_CAUSAL_TEMPORAL_V1_OPERATIONAL_OR_INTEGRITY_"
        "FAILURE_NO_RETRY"
    )
    assert failed["prior_runtime_output_open_count"] == 0

    mutated = deepcopy(failed)
    mutated["prior_runtime_output_open_count"] = 1
    mutated = _rehash(mutated)
    with pytest.raises(PermissionError, match="failure receipt changed"):
        contract.validate_failure_receipt(
            mutated,
            reservation_binding=binding,
        )


def test_metric_sidecar_publishes_and_rejects_mutation(
    tmp_path: Path,
) -> None:
    update = 100
    checkpoint = {
        "path": f"checkpoints/update_{update}.pt",
        "file_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "byte_count": 1,
        "state_sha256": "3" * 64,
        "frozen_state_sha256": "4" * 64,
    }
    evaluation = {
        "complete_physical_scope_count": 0,
        "margin_count": contract.MARGIN_COUNT,
        "passed_margin_count": 0,
        "total_shortfall": 100.0,
        "worst_margin": -1.0,
        "rough_motion": {
            "pixel_balanced_accuracy": 0.0,
            "ground_balanced_accuracy": 0.0,
            "depth_p95_m": 10.0,
        },
    }
    metric = {
        "update": update,
        "role": "checkpoint_selection",
        "pair_count": contract.SELECTION_ROLE_COUNTS["pairs"],
        "unique_endpoint_count":
            contract.SELECTION_ROLE_COUNTS["unique_endpoints"],
        "temporal_population": {},
        "scopes": {scope: {} for scope in contract.SCOPES},
        "warm_scopes_informational_only": {
            scope: {} for scope in contract.SCOPES
        },
        "aggregate_complete_v4_tail_depth_loss": 1.0,
        "evaluation": evaluation,
        "integrity_pass": True,
        "state_sha256_before": "5" * 64,
        "state_sha256_after": "5" * 64,
        "frozen_state_sha256_before_and_after": "4" * 64,
        "state_mutation_count": 0,
    }
    sidecar_binding, control = runner._publish_metric_sidecar(
        tmp_path,
        update=update,
        checkpoint=checkpoint,
        metric=metric,
    )
    sidecar = contract.parse_canonical_json(
        (tmp_path / contract.metric_sidecar_relative_path(update)).read_bytes(),
        name="synthetic temporal metric sidecar",
    )
    contract.validate_metric_sidecar(sidecar, update=update)
    assert sidecar["continuation"] == control
    assert sidecar_binding["path"] == contract.metric_sidecar_relative_path(update)

    mutated = deepcopy(sidecar)
    mutated["state_mutation_count"] = 1
    mutated = _rehash(mutated)
    with pytest.raises(PermissionError, match="metric sidecar changed"):
        contract.validate_metric_sidecar(mutated, update=update)
