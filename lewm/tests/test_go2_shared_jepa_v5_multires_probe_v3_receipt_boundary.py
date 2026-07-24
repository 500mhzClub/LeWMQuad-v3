from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from lewm.models import observable_camera_ray_evidence_v4
from lewm.models import shared_observable_camera_ray_jepa_v5
from lewm.models import shared_observable_camera_ray_jepa_v5_multires_v1


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
    "_test_go2_multires_probe_v3_receipt_contract",
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py",
)
runner = _load(
    "_test_go2_multires_probe_v3_receipt_runner",
    "scripts/run_go2_shared_jepa_v5_multires_probe_v3.py",
)


@pytest.fixture(scope="module")
def real_initialization() -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    """Exercise the frozen V1 producer and the real V3 consumer on CPU."""
    fit = observable_camera_ray_evidence_v4.ObservableCameraRayEvidenceV4Model()
    model, receipt = (
        shared_observable_camera_ray_jepa_v5_multires_v1
        .SharedObservableCameraRayJepaV5MultiresV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=(
                shared_observable_camera_ray_jepa_v5_multires_v1
                .N320_CHECKPOINT_FILE_SHA256
            ),
            n320_checkpoint_content_sha256=(
                shared_observable_camera_ray_jepa_v5_multires_v1
                .N320_CHECKPOINT_CONTENT_SHA256
            ),
        )
    )
    runtime = SimpleNamespace(
        torch=torch,
        model_module=shared_observable_camera_ray_jepa_v5,
    )
    normalized = runner._validate_migration_receipt(
        runtime,
        shared_observable_camera_ray_jepa_v5_multires_v1,
        model,
        fit,
        receipt,
    )
    return runtime, model, fit, receipt, normalized


def test_real_v1_receipt_normalizes_through_to_dict_before_dataclass(
    real_initialization: tuple[Any, Any, Any, Any, dict[str, Any]],
) -> None:
    _runtime, _model, _fit, receipt, normalized = real_initialization
    assert runner._receipt_dict(receipt) == normalized
    assert type(normalized["copied_state_keys"]) is list
    assert normalized["copied_state_keys"] == sorted(
        normalized["copied_state_keys"]
    )
    assert len(normalized["copied_state_keys"]) == 84
    assert len(set(normalized["copied_state_keys"])) == 84
    assert not any(
        "dense_decoder" in name
        for name in normalized["copied_state_keys"]
    )
    assert normalized["hard_sync_count"] == 1


def _mutate_receipt(
    name: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    value = deepcopy(baseline)
    if name == "schema":
        value["schema"] = "changed"
    elif name == "missing_field":
        value.pop("torch_version")
    elif name == "extra_field":
        value["unexpected"] = False
    elif name == "reversed_copy_order":
        value["copied_state_keys"].reverse()
    elif name == "duplicate_copy":
        value["copied_state_keys"][-1] = value["copied_state_keys"][0]
    elif name == "copy_count":
        value["copied_state_entry_count"] = 83
    elif name == "state_hash":
        value["decoder_state_sha256"] = "0" * 64
    elif name == "hard_sync":
        value["hard_sync_count"] = 0
    elif name == "dense_decoder_copy":
        value["copied_state_keys"][-1] = (
            "evidence_head.dense_decoder.stages.0.0.weight"
        )
    else:
        raise AssertionError(f"unknown mutation: {name}")
    return value


@pytest.mark.parametrize(
    "mutation",
    [
        "schema",
        "missing_field",
        "extra_field",
        "reversed_copy_order",
        "duplicate_copy",
        "copy_count",
        "state_hash",
        "hard_sync",
        "dense_decoder_copy",
    ],
)
def test_real_runner_rejects_receipt_mutations(
    mutation: str,
    real_initialization: tuple[Any, Any, Any, Any, dict[str, Any]],
) -> None:
    runtime, model, fit, _receipt, normalized = real_initialization
    with pytest.raises(PermissionError, match="initialization receipt changed"):
        runner._validate_migration_receipt(
            runtime,
            shared_observable_camera_ray_jepa_v5_multires_v1,
            model,
            fit,
            _mutate_receipt(mutation, normalized),
        )


@pytest.mark.parametrize(
    "component",
    ["model_state", "frozen_target_state"],
)
def test_real_runner_rejects_live_state_mutations(
    component: str,
    real_initialization: tuple[Any, Any, Any, Any, dict[str, Any]],
) -> None:
    runtime, model, fit, _receipt, normalized = real_initialization
    module = (
        model.evidence_head.dense_decoder
        if component == "model_state"
        else model.target_encoder
    )
    parameter = next(module.parameters())
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.view(-1)[0].add_(1.0)
        with pytest.raises(
            PermissionError, match="initialization receipt changed"
        ):
            runner._validate_migration_receipt(
                runtime,
                shared_observable_camera_ray_jepa_v5_multires_v1,
                model,
                fit,
                normalized,
            )
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def test_receipt_normalization_rejects_non_mapping_adapters() -> None:
    class BadAdapter:
        def to_dict(self) -> tuple[str, ...]:
            return ("not", "a", "dict")

    with pytest.raises(TypeError, match="normalize to a dict"):
        runner._receipt_dict(BadAdapter())
    with pytest.raises(TypeError, match="not structured"):
        runner._receipt_dict(object())


def _reservation(output: Path) -> tuple[dict[str, Any], bytes]:
    return runner._publish_json(output / "reservation.json", {
        "schema": contract.RESERVATION_SCHEMA,
        "status": "RESERVED_TEST",
        "attempt_identity": "a" * 64,
    })


@pytest.mark.parametrize(
    ("injected", "expected_boundary", "header_durable"),
    [
        ("ledger_before_header", "before_header_publication", False),
        (
            "ledger_after_durable_header",
            "after_durable_header_before_constructor_acceptance",
            True,
        ),
    ],
)
def test_post_reservation_pre_ledger_failures_are_distinct_and_sealed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    injected: str,
    expected_boundary: str,
    header_durable: bool,
) -> None:
    output = tmp_path / "attempt"
    output.mkdir(mode=0o700)
    reservation, reservation_raw = _reservation(output)
    seen: list[str] = []

    def inject(name: str) -> None:
        seen.append(name)
        if name == injected:
            raise RuntimeError(f"synthetic {injected}")

    monkeypatch.setattr(runner, "_failure_boundary", inject)
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

    expected_seen = (
        ["ledger_before_header", "ledger_after_durable_header"]
        if header_durable
        else ["ledger_before_header"]
    )
    assert seen == expected_seen
    failed_raw = (output / "failed.json").read_bytes()
    failed = contract.parse_canonical_json(
        failed_raw, name="synthetic V3 pre-ledger failure"
    )
    reservation_binding = contract.artifact_binding(
        "reservation.json",
        reservation_raw,
        content_sha256=reservation["content_sha256"],
    )
    contract.validate_pre_ledger_failure_receipt(
        failed,
        reservation_binding=reservation_binding,
        attempt_identity=reservation["attempt_identity"],
    )
    with pytest.raises(PermissionError, match="failure-receipt fields"):
        contract.validate_failure_receipt(
            failed,
            reservation_binding=reservation_binding,
        )
    assert failed["failure_stage"] == {
        "name": "partial_access_ledger_initialization",
        "boundary": expected_boundary,
    }
    assert (
        failed["operation_counts"]
        == contract.empty_partial_operation_counts()
    )
    assert failed["ledger_state"]["runtime_input_open_count"] == 0
    assert failed["ledger_state"]["header_prefix"] is None
    assert failed["ledger_state"]["standard_ledger_complete"] is False
    assert failed["ledger_state"][
        "standard_failure_validator_applicable"
    ] is False
    assert failed["scientific_result"] is None
    assert failed["retry_authorized"] is False
    assert failed["v1_runtime_output_open_count"] == 0
    assert failed["v2_runtime_output_open_count"] == 0
    assert not (output / "result.json").exists()
    assert not (output / "completed.json").exists()
    assert not (output / "access.json").exists()

    expected_files = {"failed.json", "reservation.json"}
    if header_durable:
        expected_files.add("partial_access.jsonl")
        header_raw = (output / "partial_access.jsonl").read_bytes()
        header = contract.validate_pre_ledger_header(
            header_raw,
            reservation_binding=reservation_binding,
            attempt_identity=reservation["attempt_identity"],
        )
        assert header == failed["ledger_state"]["header"]
        with pytest.raises(PermissionError, match="ledger is incomplete"):
            contract.parse_partial_access_ledger(header_raw)
    else:
        assert not (output / "partial_access.jsonl").exists()
        assert failed["ledger_state"]["header"] is None

    files = {path.name for path in output.iterdir() if path.is_file()}
    assert files == expected_files
    assert all(
        stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == 0o444
        for path in output.iterdir()
        if path.is_file()
    )
    assert stat.S_IMODE(
        output.stat(follow_symlinks=False).st_mode
    ) == 0o555
    assert hashlib.sha256(failed_raw).hexdigest() == hashlib.sha256(
        (output / "failed.json").read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("publication", "matches_expected_header"),
    [
        ("exact_then_error", True),
        ("partial_then_error", False),
    ],
)
def test_writer_errors_with_a_published_prefix_still_receipt_and_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication: str,
    matches_expected_header: bool,
) -> None:
    output = tmp_path / "attempt"
    output.mkdir(mode=0o700)
    reservation, reservation_raw = _reservation(output)
    original_write = runner._write_exclusive
    original_read = runner._read_regular

    def fail_during_header(path: Path, raw: bytes, *, mode: int = 0o644) -> None:
        if path.name != runner.PartialAccessLedger.RELATIVE_PATH:
            original_write(path, raw, mode=mode)
            return
        if publication == "exact_then_error":
            original_write(path, raw, mode=mode)
        else:
            path.write_bytes(raw[:17])
            path.chmod(mode)
        raise OSError(f"synthetic {publication}")

    monkeypatch.setattr(runner, "_write_exclusive", fail_during_header)

    def fail_general_prefix_observation(
        path: Path,
        *,
        expected_sha256: str | None = None,
    ) -> bytes:
        if path.name == runner.PartialAccessLedger.RELATIVE_PATH:
            raise OSError("synthetic secondary prefix observation failure")
        return original_read(path, expected_sha256=expected_sha256)

    monkeypatch.setattr(runner, "_read_regular", fail_general_prefix_observation)
    with pytest.raises(OSError, match=f"synthetic {publication}"):
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
        name="writer-error V3 pre-ledger failure",
    )
    reservation_binding = contract.artifact_binding(
        "reservation.json",
        reservation_raw,
        content_sha256=reservation["content_sha256"],
    )
    contract.validate_pre_ledger_failure_receipt(
        failed,
        reservation_binding=reservation_binding,
        attempt_identity=reservation["attempt_identity"],
    )
    assert failed["failure_stage"]["boundary"] == (
        "during_header_publication_unaccepted_prefix"
    )
    assert failed["ledger_state"]["status"] == "UNACCEPTED_HEADER_PREFIX"
    assert failed["ledger_state"]["header"] is None
    assert failed["ledger_state"]["header_prefix"][
        "matches_expected_header"
    ] is matches_expected_header
    assert failed["ledger_state"]["runtime_input_open_count"] == 0
    assert failed["operation_counts"] == (
        contract.empty_partial_operation_counts()
    )
    assert failed["retry_authorized"] is False
    assert sorted(path.name for path in output.iterdir()) == [
        "failed.json",
        "partial_access.jsonl",
        "reservation.json",
    ]
    assert stat.S_IMODE(output.stat(follow_symlinks=False).st_mode) == 0o555
