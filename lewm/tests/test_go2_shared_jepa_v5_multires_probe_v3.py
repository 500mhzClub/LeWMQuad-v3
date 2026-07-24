from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
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
    "_test_go2_multires_probe_v3_contract",
    "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v3.py",
)
runner = _load(
    "_test_go2_multires_probe_v3_runner",
    "scripts/run_go2_shared_jepa_v5_multires_probe_v3.py",
)
launcher = _load(
    "_test_go2_multires_probe_v3_launcher",
    "scripts/launch_go2_shared_jepa_v5_multires_probe_v3.py",
)


def _reservation(output: Path) -> tuple[dict[str, Any], bytes]:
    return runner._publish_json(output / "reservation.json", {
        "schema": contract.RESERVATION_SCHEMA,
        "status": "RESERVED_TEST",
        "attempt_identity": "a" * 64,
    })


def _ledger(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any], bytes, Any]:
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
    return repository, output, reservation, reservation_raw, ledger


def _write_input(repository: Path, name: str, raw: bytes) -> Path:
    path = repository / "runtime" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _read(
    ledger: Any,
    path: Path,
    *,
    kind: str,
    stage: str,
) -> bytes:
    raw = path.read_bytes()
    return ledger.read_regular(
        path,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        expected_byte_count=len(raw),
        kind=kind,
        stage=stage,
        role="authority",
        purpose="synthetic_test",
    )


def _full_counts() -> dict[str, Any]:
    value = contract.empty_partial_operation_counts()
    value.update({
        "training_entered": True,
        "optimizer_construction_attempt_count": 1,
        "optimizer_construction_completion_count": 1,
        "optimizer_update_attempt_count": 1_000,
        "complete_optimizer_updates": 1_000,
        "pair_index_presentations_attempted": 16_000,
        "pair_index_presentations_materialized": 16_000,
        "microbatch_attempt_count": 4_000,
        "microbatch_completion_count": 4_000,
        "camera_objective_attempt_count": 4_000,
        "camera_objective_completion_count": 4_000,
        "finite_camera_objective_count": 4_000,
        "backward_attempt_count": 4_000,
        "backward_completion_count": 4_000,
        "head_clip_attempt_count": 1_000,
        "head_clip_completion_count": 1_000,
        "encoder_clip_attempt_count": 1_000,
        "encoder_clip_completion_count": 1_000,
        "optimizer_step_attempt_count": 1_000,
        "optimizer_step_completion_count": 1_000,
        "checkpoint_snapshot_completion_count": 3,
        "checkpoint_selection_evaluation_attempt_count": 3,
        "checkpoint_selection_evaluation_completion_count": 3,
        "metric_sidecar_publication_count": 3,
        "checkpoint_selection_evaluation_updates_attempted": [100, 400, 1_000],
        "checkpoint_selection_evaluation_updates_completed": [100, 400, 1_000],
    })
    return value


def _first_backward_counts() -> dict[str, Any]:
    value = contract.empty_partial_operation_counts()
    value.update({
        "training_entered": True,
        "optimizer_construction_attempt_count": 1,
        "optimizer_construction_completion_count": 1,
        "optimizer_update_attempt_count": 1,
        "pair_index_presentations_attempted": 4,
        "pair_index_presentations_materialized": 4,
        "microbatch_attempt_count": 1,
        "microbatch_completion_count": 1,
        "camera_objective_attempt_count": 1,
        "camera_objective_completion_count": 1,
        "finite_camera_objective_count": 1,
        "backward_attempt_count": 1,
        "backward_completion_count": 1,
    })
    return value


def _first_evaluation_counts() -> dict[str, Any]:
    value = contract.empty_partial_operation_counts()
    value.update({
        "training_entered": True,
        "optimizer_construction_attempt_count": 1,
        "optimizer_construction_completion_count": 1,
        "optimizer_update_attempt_count": 100,
        "complete_optimizer_updates": 100,
        "pair_index_presentations_attempted": 1_600,
        "pair_index_presentations_materialized": 1_600,
        "microbatch_attempt_count": 400,
        "microbatch_completion_count": 400,
        "camera_objective_attempt_count": 400,
        "camera_objective_completion_count": 400,
        "finite_camera_objective_count": 400,
        "backward_attempt_count": 400,
        "backward_completion_count": 400,
        "head_clip_attempt_count": 100,
        "head_clip_completion_count": 100,
        "encoder_clip_attempt_count": 100,
        "encoder_clip_completion_count": 100,
        "optimizer_step_attempt_count": 100,
        "optimizer_step_completion_count": 100,
        "checkpoint_snapshot_completion_count": 1,
        "checkpoint_selection_evaluation_attempt_count": 1,
        "checkpoint_selection_evaluation_updates_attempted": [100],
    })
    return value


def test_v3_identity_and_fresh_operational_envelope() -> None:
    assert contract.SCHEMA_PREFIX.endswith("_v3")
    assert contract.PREREGISTRATION_COMMIT == (
        "7e6e539370c8f9d9d228da5ef4bc9ea4d10569a2"
    )
    assert contract.PREREGISTRATION_FILE_SHA256 == (
        "a8a5d870382ad505edd907f96dfae8a6ed737caf7ff424d2b52f8e4bc020e5d5"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "rgb_multiresolution_perception_probe_v3_retry1"
    )
    assert contract.V1_OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "rgb_multiresolution_perception_probe_v1"
    )
    assert contract.V2_OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "rgb_multiresolution_perception_probe_v2"
    )
    assert contract.MODEL_RUNTIME_VERSION.endswith("_v1_model_runtime_v1")
    assert contract.science_contract()["model_family"].endswith("_multires_v1")
    assert contract.canonical_json_sha256(contract.science_contract()) == (
        "e181381c00585fa5df41a71fff918b5599acc955d59283ce397ba6dd530dc23f"
    )
    assert contract.MAXIMUM_PRESENTATIONS == 16_000


def test_v1_implementation_bytes_remain_frozen() -> None:
    expected = {
        "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v1.py":
            "ffdeb2b6b3a03a1b1b65e2fe3961a8561717c8ced4d800c640f03710af40fa3b",
        "scripts/run_go2_shared_jepa_v5_multires_probe_v1.py":
            "c84604df4933a04939c297fa68e765ec6c00e68d360da0c6ed8de5a56ba87e41",
        "scripts/launch_go2_shared_jepa_v5_multires_probe_v1.py":
            "adf97ed861c2f37960db1fbc171c91913847d2f4a98e553ea903d9371419f42e",
        "lewm/tests/test_go2_shared_jepa_v5_multires_probe_v1.py":
            "dba0954f9eed9d700bfe808b6911466cce8728cef247788fbcfe00b65798de0b",
        "scripts/check_go2_multires_probe_source_closure.py":
            "ac9fcaa9107ad43201b5082581c0743ebb46653ff8b51a6f09c33fc992142911",
        "lewm/tests/test_go2_multires_probe_source_closure.py":
            "fb09c98b0f008eb863622dab1b4204535b719734eaf9293adb6eaefd3417f846",
    }
    assert {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in expected
    } == expected
    assert hashlib.sha256(
        (ROOT / contract.MODEL_RELATIVE_PATH).read_bytes()
    ).hexdigest() == contract.MODEL_FILE_SHA256


def test_partial_access_ledger_is_hash_chained_and_pairs_every_open(
    tmp_path: Path,
) -> None:
    repository, _output, _reservation_value, _raw, ledger = _ledger(tmp_path)
    accepted = _write_input(repository, "accepted.bin", b"accepted")
    assert _read(
        ledger, accepted, kind="bound_schedule", stage="schedule_phase_a"
    ) == b"accepted"
    missing = repository / "runtime/missing.bin"
    with pytest.raises(FileNotFoundError):
        ledger.read_regular(
            missing,
            expected_sha256="b" * 64,
            kind="n320_gate",
            stage="n320_gate",
            role="authority",
            purpose="synthetic_test",
        )
    ledger.append_terminal(
        record_type="ATTEMPT_TERMINATING",
        stage={"name": "synthetic"},
        operation_counts=contract.empty_partial_operation_counts(),
        error=RuntimeError("synthetic"),
    )
    binding = ledger.binding()
    opens = ledger.runtime_opens()
    assert [item["outcome"] for item in opens] == ["ACCEPTED", "OPEN_FAILED"]
    assert binding["attempted_open_count"] == 2
    assert binding["descriptor_opened_count"] == 1
    assert binding["read_completed_count"] == 1
    assert binding["accepted_open_count"] == 1
    previous = None
    for sequence, record in enumerate(ledger.records):
        assert record["sequence"] == sequence
        assert record["previous_record_content_sha256"] == previous
        core = dict(record)
        declared = core.pop("content_sha256")
        assert contract.canonical_json_sha256(core) == declared
        previous = declared


def test_injected_open_failure_is_durably_paired_before_rethrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _output, _reservation_value, _raw, ledger = _ledger(tmp_path)
    schedule = _write_input(repository, "schedule.json", b"schedule")

    def fail(name: str) -> None:
        assert name == "schedule"
        on_disk = ledger.path.read_text(encoding="ascii").splitlines()
        assert json.loads(on_disk[-1])["record_type"] == "OPEN_ATTEMPTED"
        raise RuntimeError("injected schedule boundary")

    monkeypatch.setattr(runner, "_failure_boundary", fail)
    with pytest.raises(RuntimeError, match="injected schedule boundary"):
        _read(ledger, schedule, kind="bound_schedule", stage="schedule_phase_a")
    assert ledger.runtime_opens()[0]["outcome"] == "OPEN_FAILED"
    assert json.loads(
        ledger.path.read_text(encoding="ascii").splitlines()[-1]
    )["record_type"] == "OPEN_OUTCOME"
    ledger.close()


def test_partial_read_failure_records_exact_bytes_before_rethrow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, _output, _reservation_value, _raw, ledger = _ledger(tmp_path)
    runtime_input = _write_input(repository, "partial.bin", b"abcdefgh")
    original_read = os.read
    read_count = 0

    def fail_after_first_chunk(descriptor: int, _size: int) -> bytes:
        nonlocal read_count
        read_count += 1
        if read_count == 1:
            return original_read(descriptor, 3)
        raise OSError("synthetic partial-read failure")

    monkeypatch.setattr(runner.os, "read", fail_after_first_chunk)
    with pytest.raises(OSError, match="synthetic partial-read failure"):
        _read(
            ledger,
            runtime_input,
            kind="synthetic_partial_read",
            stage="synthetic",
        )
    outcome = ledger.runtime_opens()[0]
    assert outcome["outcome"] == "READ_FAILED"
    assert outcome["descriptor_opened"] is True
    assert outcome["read_completed"] is False
    assert outcome["partial_byte_count"] == 3
    assert json.loads(
        ledger.path.read_text(encoding="ascii").splitlines()[-1]
    )["record_type"] == "OPEN_OUTCOME"
    ledger.close()


def test_ledger_rejects_symlinked_parent_components(tmp_path: Path) -> None:
    repository, _output, _reservation_value, _raw, ledger = _ledger(tmp_path)
    real_parent = repository / "real-runtime"
    real_parent.mkdir()
    runtime_input = real_parent / "input.bin"
    runtime_input.write_bytes(b"bound bytes")
    alias_parent = repository / "runtime-alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    alias_input = alias_parent / runtime_input.name

    with pytest.raises(PermissionError, match="symlinked parent"):
        ledger.read_regular(
            alias_input,
            expected_sha256=hashlib.sha256(b"bound bytes").hexdigest(),
            expected_byte_count=len(b"bound bytes"),
            kind="synthetic_parent_symlink",
            stage="synthetic",
            role="authority",
            purpose="synthetic_test",
        )
    assert ledger.runtime_opens() == []
    ledger.close()


@pytest.mark.parametrize(
    "prior_root",
    [
        contract.V1_OUTPUT_ROOT_RELATIVE_PATH,
        contract.V2_OUTPUT_ROOT_RELATIVE_PATH,
    ],
)
def test_ledger_denies_any_prior_attempt_output_access(
    tmp_path: Path,
    prior_root: str,
) -> None:
    repository, _output, _reservation_value, _raw, ledger = _ledger(tmp_path)
    forbidden = repository / prior_root / "result.json"
    with pytest.raises(PermissionError, match="prior multires runtime output"):
        ledger.read_regular(
            forbidden,
            expected_sha256="c" * 64,
            kind="forbidden_prior_output",
            stage="synthetic",
            role=None,
            purpose="synthetic_test",
        )
    assert ledger.runtime_opens() == []
    ledger.close()


@pytest.mark.parametrize("value", ["a//b", "a/./b", "./a", "a/"])
def test_safe_relative_path_rejects_noncanonical_spellings(value: str) -> None:
    with pytest.raises(ValueError, match="safe relative path"):
        contract.safe_relative_path(value, name="synthetic path")
    assert contract.safe_relative_path("a/b", name="synthetic path") == "a/b"


def test_partial_operation_counts_are_exact_and_capped() -> None:
    assert contract.validate_partial_operation_counts(
        _first_backward_counts()
    )["backward_completion_count"] == 1
    assert contract.validate_partial_operation_counts(
        _first_evaluation_counts()
    )["checkpoint_selection_evaluation_attempt_count"] == 1
    assert contract.validate_partial_operation_counts(
        _full_counts()
    )["complete_optimizer_updates"] == 1_000
    invalid = _full_counts()
    invalid["pair_index_presentations_attempted"] = 16_001
    with pytest.raises(PermissionError):
        contract.validate_partial_operation_counts(invalid)


@pytest.mark.parametrize(
    ("stage", "counts_factory"),
    [
        ("schedule_phase_a", contract.empty_partial_operation_counts),
        ("n320_gate", contract.empty_partial_operation_counts),
        ("n320_checkpoint", contract.empty_partial_operation_counts),
        ("raw_authority", contract.empty_partial_operation_counts),
        ("raw_indexes", contract.empty_partial_operation_counts),
        ("model_preparation", contract.empty_partial_operation_counts),
        ("training_backward", _first_backward_counts),
        ("checkpoint_selection_evaluation", _first_evaluation_counts),
        ("result_publication", _full_counts),
        ("completion_publication", _full_counts),
    ],
)
def test_every_required_boundary_publishes_complete_sealed_failure(
    tmp_path: Path,
    stage: str,
    counts_factory: Any,
) -> None:
    repository, output, reservation, reservation_raw, ledger = _ledger(tmp_path)
    schedule = _write_input(repository, "schedule.json", b"schedule")
    _read(ledger, schedule, kind="bound_schedule", stage="schedule_phase_a")
    progress = runner.OperationProgress()
    progress.counts = counts_factory()
    progress.enter(
        stage,
        update=(
            100
            if stage == "checkpoint_selection_evaluation"
            else 1_000
            if stage in {"result_publication", "completion_publication"}
            else 1
            if stage == "training_backward"
            else None
        ),
        microbatch=0 if stage == "training_backward" else None,
        checkpoint_update=(
            100 if stage == "checkpoint_selection_evaluation" else None
        ),
    )
    runner._terminal_failure(
        output,
        reservation,
        reservation_raw,
        ledger,
        progress,
        error=RuntimeError(f"synthetic {stage} failure"),
    )
    failed_path = output / "failed.json"
    failed = contract.parse_canonical_json(
        failed_path.read_bytes(), name="synthetic V3 failure"
    )
    contract.validate_failure_receipt(
        failed,
        reservation_binding=contract.artifact_binding(
            "reservation.json",
            reservation_raw,
            content_sha256=reservation["content_sha256"],
        ),
    )
    contract.parse_partial_access_ledger(
        (output / "partial_access.jsonl").read_bytes()
    )
    assert failed["failure_stage"]["name"] == stage
    assert failed["reservation"] == contract.artifact_binding(
        "reservation.json",
        reservation_raw,
        content_sha256=reservation["content_sha256"],
    )
    assert failed["partial_access_ledger"]["accepted_open_count"] == 1
    assert failed["runtime_opens"][0]["kind"] == "bound_schedule"
    assert failed["operation_counts"] == contract.validate_partial_operation_counts(
        counts_factory()
    )
    assert failed["retry_authorized"] is False
    assert failed["v1_runtime_output_open_count"] == 0
    assert failed["v2_runtime_output_open_count"] == 0
    files = [
        item for item in output.rglob("*") if item.is_file()
    ]
    directories = [
        output,
        *(item for item in output.rglob("*") if item.is_dir()),
    ]
    assert all(stat.S_IMODE(item.stat().st_mode) == 0o444 for item in files)
    assert all(
        stat.S_IMODE(item.stat().st_mode) == 0o555
        for item in directories
    )


def test_schedule_adapter_calls_are_in_the_required_custody_order() -> None:
    source = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text("utf-8")
    execute = source[source.index("def _execute_after_reservation("):]
    positions = [
        execute.index("schedule_state = _load_schedule_phase_a("),
        execute.index("matched._camera_model_after_reservation("),
        execute.index("inputs = matched.RawInputs("),
        execute.index("_finalize_schedule_train_identity("),
        execute.index("model, head, encoder, frozen, partition = _prepare_model("),
        execute.index("training = _train("),
    ]
    assert positions == sorted(positions)
    assert "validate_bound_schedule_phase_a(" in source
    assert "finalize_train_identity(" in source
    assert "def _load_schedule(" not in source
    assert set(runner.SYNTHETIC_FAILURE_BOUNDARIES) == {
        "ledger_before_header",
        "ledger_after_durable_header",
        "schedule",
        "gate",
        "n320_checkpoint",
        "raw_authority",
        "raw_indexes",
        "model_preparation",
        "training",
        "evaluation",
        "result_publication",
        "completion_publication",
    }
    for boundary in runner.SYNTHETIC_FAILURE_BOUNDARIES:
        assert f'_failure_boundary("{boundary}")' in source or boundary in {
            "schedule",
            "gate",
            "n320_checkpoint",
            "raw_authority",
            "raw_indexes",
        }


def test_launcher_still_orders_authority_preflight_then_immediate_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        launcher,
        "_load_authority_before_hardware",
        lambda _args: events.append("authority") or {},
    )
    monkeypatch.setattr(
        launcher,
        "_run_no_tensor_preflight",
        lambda _environment: events.append("preflight") or {},
    )
    monkeypatch.setattr(
        launcher,
        "_preflight_receipt",
        lambda _value, _authority: (
            events.append("receipt") or {"content_sha256": "0" * 64},
            b"receipt\n",
        ),
    )

    def execute(*_args: Any, **_kwargs: Any) -> None:
        events.append("exec")
        raise RuntimeError("synthetic exec")

    monkeypatch.setattr(launcher, "_exec_runner", execute)
    args = type("Args", (), {
        "review_sha256": "a" * 64,
        "authorization_sha256": "b" * 64,
    })()
    with pytest.raises(RuntimeError, match="synthetic exec"):
        launcher._launch(args, {})
    assert events == ["authority", "preflight", "receipt", "exec"]


def test_v3_source_imports_do_not_import_torch() -> None:
    program = f"""
import importlib.util
import pathlib
import sys
root = pathlib.Path({str(ROOT)!r})
for index, relative in enumerate({
    (
        contract.CONTRACT_RELATIVE_PATH,
        contract.RUNNER_RELATIVE_PATH,
        contract.LAUNCHER_RELATIVE_PATH,
    )
!r}):
    spec = importlib.util.spec_from_file_location(f"_source_only_{{index}}", root / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
raise SystemExit(1 if "torch" in sys.modules else 0)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert contract.REVIEW_RELATIVE_PATH.endswith(
        "v3_source_review_2026-07-24.json"
    )
    assert contract.AUTHORIZATION_RELATIVE_PATH.endswith(
        "v3_execution_authorization_2026-07-24.json"
    )
