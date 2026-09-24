from __future__ import annotations

import builtins
import hashlib
import importlib.util
from pathlib import Path
import stat
from types import SimpleNamespace
import sys
from unittest import mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "scripts/run_go2_rgb_causal_motion_alignment_v1.py"
LAUNCHER_PATH = ROOT / "scripts/launch_go2_rgb_causal_motion_alignment_v1.py"


def _load_runner(name: str):
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _failing_evaluation(module) -> dict[str, object]:
    return {
        "scope_evaluations": {
            scope: {"physical_margins": [], "passes": False}
            for scope in module.contract.SCOPES
        },
        "complete_physical_scope_count": 0,
        "margin_count": module.contract.MARGIN_COUNT,
        "passed_margin_count": 0,
        "total_shortfall": 100.0,
        "worst_margin": -1.0,
        "rough_motion": {
            "pixel_balanced_accuracy": 0.0,
            "ground_balanced_accuracy": 0.0,
            "depth_p95_m": 10.0,
        },
    }


def _provisional_metric(module, update: int) -> dict[str, object]:
    return {
        "update": update,
        "role": "checkpoint_selection",
        "pair_count": module.contract.SELECTION_ROLE_COUNTS["pairs"],
        "unique_endpoint_count":
            module.contract.SELECTION_ROLE_COUNTS["unique_endpoints"],
        "temporal_population": {},
        "scopes": {scope: {} for scope in module.contract.SCOPES},
        "warm_scopes_informational_only": {
            scope: {} for scope in module.contract.SCOPES
        },
        "aggregate_complete_v4_tail_depth_loss": 1.0,
        "evaluation": _failing_evaluation(module),
        "preledger_model_state_checks_pass": True,
        "state_sha256_before": "5" * 64,
        "state_sha256_after": "5" * 64,
        "frozen_state_sha256_before_and_after": "4" * 64,
        "state_mutation_count": 0,
    }


def _checkpoint(update: int) -> dict[str, object]:
    return {
        "path": f"checkpoints/update_{update}.pt",
        "file_sha256": "1" * 64,
        "content_sha256": "2" * 64,
        "byte_count": 1,
        "state_sha256": "3" * 64,
        "frozen_state_sha256": "4" * 64,
    }


def _reservation(module, output: Path) -> tuple[dict, bytes]:
    return module._BASE._publish_json(output / "reservation.json", {
        "schema": module.contract.RESERVATION_SCHEMA,
        "status": "RESERVED_SYNTHETIC_SOURCE_TEST",
        "attempt_identity": "a" * 64,
    })


def _ledger(module, output: Path, repository: Path):
    reservation, reservation_raw = _reservation(module, output)
    ledger = module._BASE._initialize_partial_access_ledger(
        output,
        reservation=reservation,
        reservation_raw=reservation_raw,
        repository_root=repository,
    )
    return reservation, reservation_raw, ledger


def _append_context_invalid_schedule_open(module, ledger) -> None:
    path = module.contract.SCHEDULE_RELATIVE_PATH
    payload = b"synthetic schedule"
    expected = {
        "path": path,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": None,
        "byte_count": len(payload),
    }
    ledger._append({
        "record_type": "OPEN_ATTEMPTED",
        "open_id": 1,
        "stage": "synthetic_runtime_load",
        "kind": "bound_schedule",
        "role": "train",
        "purpose": "runtime_load",
        "expected_binding": expected,
    })
    ledger._append({
        "record_type": "OPEN_OUTCOME",
        "open_id": 1,
        "stage": "synthetic_runtime_load",
        "kind": "bound_schedule",
        "outcome": "ACCEPTED",
        "descriptor_opened": True,
        "read_completed": True,
        "binding_accepted": True,
        "observed_binding": {
            "path": path,
            "file_sha256": expected["file_sha256"],
            "byte_count": len(payload),
        },
        "partial_byte_count": len(payload),
        "error": None,
    })


def _append_near_miss_render_open(module, ledger) -> None:
    path = (
        ".generated/go2_render_selected_v04/scenes/"
        "scene_0123456789abcdef/rgb/frame_000123_env_04.jpg"
    )
    payload = b"synthetic near-miss render"
    expected = {
        "path": path,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": None,
        "byte_count": len(payload),
    }
    ledger._append({
        "record_type": "OPEN_ATTEMPTED",
        "open_id": 1,
        "stage": "synthetic_runtime_load",
        "kind": "development_rgb",
        "role": "train",
        "purpose": "runtime_load",
        "expected_binding": expected,
    })
    ledger._append({
        "record_type": "OPEN_OUTCOME",
        "open_id": 1,
        "stage": "synthetic_runtime_load",
        "kind": "development_rgb",
        "outcome": "ACCEPTED",
        "descriptor_opened": True,
        "read_completed": True,
        "binding_accepted": True,
        "observed_binding": {
            "path": path,
            "file_sha256": expected["file_sha256"],
            "byte_count": len(payload),
        },
        "partial_byte_count": len(payload),
        "error": None,
    })


def _assert_sealed(output: Path) -> None:
    assert stat.S_IMODE(
        output.stat(follow_symlinks=False).st_mode
    ) == 0o555
    assert all(
        stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) == 0o444
        for path in output.rglob("*")
        if path.is_file()
    )


def test_runner_import_is_source_only_and_uses_new_identity() -> None:
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {"torch", "numpy", "PIL"}:
            raise AssertionError(f"source-only runner imported {name}")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded):
        module = _load_runner("_motion_alignment_source_only_import")
    assert module.PREFLIGHT_ENVIRONMENT_KEY == (
        "LEWM_CAUSAL_MOTION_ALIGNMENT_V1_PREFLIGHT_JSON"
    )
    assert module.contract.MODEL_FAMILY.endswith("motion_alignment_v1")
    runner_source = RUNNER_PATH.read_text(encoding="utf-8")
    launcher_source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "go2_rgb_causal_motion_alignment_v1.py" in runner_source
    assert "go2_rgb_causal_motion_alignment_v1.py" in launcher_source


def _train_rows(module) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for primitive_index, primitive in enumerate(
        module.PRIMITIVE_VOCABULARY
    ):
        for sample in (0, 1):
            base = float(primitive_index * 10 + sample * 2)
            rows.append({
                "dataset_role": "train",
                "primitive": primitive,
                "relative_se2_current_frame": [
                    base,
                    base + 1.0,
                    base + 2.0,
                ],
            })
    return rows


def test_nominal_table_is_train_only_float32_median_and_aggregate_only() -> None:
    module = _load_runner("_motion_alignment_nominal_table")
    runtime = SimpleNamespace(torch=torch)
    vocabulary, table, receipt = module._build_nominal_command_table(
        runtime, _train_rows(module)
    )
    assert vocabulary == module.PRIMITIVE_VOCABULARY
    assert table.dtype == torch.float32
    assert table.shape == (9, 3)
    assert torch.equal(table[0], torch.tensor([1.0, 2.0, 3.0]))
    assert receipt["selection_rows_contributed"] == 0
    assert receipt["per_sample_realized_se2_model_batch_count"] == 0
    assert receipt["aggregation"] == "float32_torch_quantile_0.5_dim_0"

    crossed = _train_rows(module)
    crossed[0]["dataset_role"] = "checkpoint_selection"
    with pytest.raises(PermissionError, match="crossed dataset roles"):
        module._build_nominal_command_table(runtime, crossed)


def test_training_batch_passes_nominal_delta_but_no_realized_row_motion() -> None:
    module = _load_runner("_motion_alignment_batch")
    vocabulary, table, _receipt = module._build_nominal_command_table(
        SimpleNamespace(torch=torch), _train_rows(module)
    )
    pairs = [
        {
            "dataset_role": "train",
            "primitive": vocabulary[0],
            "relative_se2_current_frame": [999.0, 999.0, 999.0],
            "current_endpoint_sha256": "current-0",
            "next_endpoint_sha256": "next-0",
        },
        {
            "dataset_role": "train",
            "primitive": vocabulary[1],
            "relative_se2_current_frame": [-999.0, -999.0, -999.0],
            "current_endpoint_sha256": "current-1",
            "next_endpoint_sha256": "next-1",
        },
    ]

    def frame(endpoint_id, *, role, arm, stage):
        assert role == "train"
        assert arm == "causal_motion_alignment_v1"
        assert stage == "camera_gradient"
        code = float(endpoint_id.endswith("1"))
        if endpoint_id.startswith("next"):
            code += 2.0
        return {
            "image": torch.full((3, 4, 4), code),
            "camera_origin": torch.full((3,), code),
            "camera_basis": torch.full((3, 3), code),
            "ground": torch.tensor(code),
        }

    trainer = SimpleNamespace(
        r=SimpleNamespace(torch=torch),
        inputs=SimpleNamespace(frame=frame),
        supervision=lambda frames, _device: SimpleNamespace(
            count=len(frames)
        ),
        _motion_primitive_to_index={
            primitive: index for index, primitive in enumerate(vocabulary)
        },
        _motion_nominal_table=table,
    )
    batch = module._visual_only_batch(
        trainer,
        pairs,
        [0, 1],
        torch.device("cpu"),
        role="train",
        arm="causal_temporal_perception_v1",
        stage="camera_gradient",
    )
    forward = batch["forward"]
    assert "nominal_delta_current_frame" in forward
    assert not any(
        "realized" in name or "relative_se2" in name for name in forward
    )
    assert torch.equal(
        forward["nominal_delta_current_frame"],
        table[:2],
    )
    assert not torch.any(
        forward["nominal_delta_current_frame"] == 999.0
    )


def test_camera_pair_forwards_only_nominal_motion_not_primitive_or_realized() -> None:
    module = _load_runner("_motion_alignment_camera_pair")
    batch_size = 2
    previous = SimpleNamespace(bev=torch.zeros(batch_size, 3, 4, 4))
    current = SimpleNamespace(bev=torch.ones(batch_size, 3, 4, 4))

    class Model:
        received = None

        def forward_camera_pair(self, **kwargs):
            self.received = kwargs
            return previous, current

    forward = {
        "current_image": torch.zeros(batch_size, 3, 4, 4),
        "next_image": torch.ones(batch_size, 3, 4, 4),
        "nominal_delta_current_frame": torch.full((batch_size, 3), 0.25),
        "current_camera_origin_body_m": torch.zeros(batch_size, 3),
        "current_camera_basis_body_fru": torch.zeros(batch_size, 3, 3),
        "current_ground_plane_z_body_m": torch.zeros(batch_size),
        "next_camera_origin_body_m": torch.ones(batch_size, 3),
        "next_camera_basis_body_fru": torch.ones(batch_size, 3, 3),
        "next_ground_plane_z_body_m": torch.ones(batch_size),
    }
    model = Model()
    runtime = SimpleNamespace(
        torch=torch,
        model_module=SimpleNamespace(
            SharedTrainingPairV5=lambda **kwargs: SimpleNamespace(**kwargs)
        ),
    )
    pair = module._camera_pair(runtime, model, {"forward": forward})
    assert model.received is not None
    assert set(model.received) == {
        "previous_image",
        "current_image",
        "previous_camera_origin_body_m",
        "previous_camera_basis_body_fru",
        "previous_ground_plane_z_body_m",
        "current_camera_origin_body_m",
        "current_camera_basis_body_fru",
        "current_ground_plane_z_body_m",
        "nominal_delta_current_frame",
    }
    assert not any(
        "primitive" in name or "realized" in name
        for name in model.received
    )
    assert pair.jepa is None


def test_provisional_sidecar_withholds_integrity_and_pass_fail(
    tmp_path: Path,
) -> None:
    module = _load_runner("_motion_alignment_provisional_sidecar")
    update = module.contract.MAXIMUM_UPDATE
    binding, internal_control = module._publish_metric_sidecar(
        tmp_path,
        update=update,
        checkpoint=_checkpoint(update),
        metric=_provisional_metric(module, update),
    )
    sidecar_path = tmp_path / module.contract.metric_sidecar_relative_path(
        update
    )
    raw = sidecar_path.read_bytes()
    sidecar = module.contract.parse_canonical_json(
        raw, name="synthetic provisional sidecar"
    )

    module.contract.validate_metric_sidecar(sidecar, update=update)
    assert binding["path"] == module.contract.metric_sidecar_relative_path(update)
    assert internal_control["action"] == module.contract.CONTROL_FAIL
    assert sidecar["scientifically_admissible"] is False
    for forbidden in (
        b'"integrity_pass"',
        b"PASS_BOUNDED_FALSIFICATION",
        b"FAIL_TERMINAL_NO_RETRY",
    ):
        assert forbidden not in raw


def test_provisional_sidecar_rejects_forbidden_nested_claims_before_write(
    tmp_path: Path,
) -> None:
    module = _load_runner("_motion_alignment_provisional_rejection")
    update = module.contract.MAXIMUM_UPDATE

    class HiddenDict(dict):
        pass

    class HiddenStr(str):
        pass

    mutations = (
        lambda metric: metric.__setitem__("integrity_pass", True),
        lambda metric: metric["evaluation"].__setitem__(
            "integrity_pass", True
        ),
        lambda metric: metric["evaluation"]["scope_evaluations"][
            module.contract.SCOPES[0]
        ].__setitem__(
            "physical_margins", [module.contract.CONTROL_PASS]
        ),
        lambda metric: metric["temporal_population"].__setitem__(
            "nested", (module.contract.CONTROL_FAIL,)
        ),
        lambda metric: metric["temporal_population"].__setitem__(
            "nested",
            HiddenDict({
                "integrity_pass": True,
                "action": module.contract.CONTROL_PASS,
            }),
        ),
        lambda metric: metric["temporal_population"].__setitem__(
            "nested", HiddenStr(module.contract.CONTROL_PASS)
        ),
    )
    for index, mutate in enumerate(mutations):
        output = tmp_path / str(index)
        metric = _provisional_metric(module, update)
        mutate(metric)
        with pytest.raises(PermissionError):
            module._publish_metric_sidecar(
                output,
                update=update,
                checkpoint=_checkpoint(update),
                metric=metric,
            )
        assert not (
            output / module.contract.metric_sidecar_relative_path(update)
        ).exists()


def test_collated_checkpoint_metrics_remain_preledger_inadmissible(
    tmp_path: Path,
) -> None:
    module = _load_runner("_motion_alignment_provisional_collation")
    updates = module.contract.CHECKPOINT_UPDATES
    training = {
        "trace": [],
        "metrics": [
            _provisional_metric(module, update) for update in updates
        ],
        "sidecars": [],
        "controls": [
            module.contract.provisional_checkpoint_control(update)
            for update in updates
        ],
    }
    module._publish_training_records(tmp_path, training)
    raw = (tmp_path / "checkpoint_metrics.json").read_bytes()
    value = module.contract.parse_canonical_json(
        raw, name="synthetic provisional checkpoint collation"
    )
    assert value["scientifically_admissible"] is False
    assert value["integrity_or_pass_fail_control_emitted"] is False
    for forbidden in (
        b'"integrity_pass"',
        b"PASS_BOUNDED_FALSIFICATION",
        b"FAIL_TERMINAL_NO_RETRY",
    ):
        assert forbidden not in raw


def test_collated_metrics_reject_nested_control_before_checkpoint_write(
    tmp_path: Path,
) -> None:
    module = _load_runner("_motion_alignment_collation_rejection")
    updates = module.contract.CHECKPOINT_UPDATES
    metrics = [_provisional_metric(module, update) for update in updates]
    metrics[-1]["evaluation"]["scope_evaluations"][
        module.contract.SCOPES[0]
    ]["physical_margins"] = [module.contract.CONTROL_FAIL]
    with pytest.raises(PermissionError):
        module._publish_training_records(tmp_path, {
            "trace": [],
            "metrics": metrics,
            "sidecars": [],
            "controls": [
                module.contract.provisional_checkpoint_control(update)
                for update in updates
            ],
        })
    assert not (tmp_path / "checkpoint_metrics.json").exists()


def test_training_trace_rejects_preledger_control_before_write(
    tmp_path: Path,
) -> None:
    module = _load_runner("_motion_alignment_trace_rejection")
    with pytest.raises(PermissionError):
        module._publish_training_records(tmp_path, {
            "trace": [{"nested": (module.contract.CONTROL_PASS,)}],
            "metrics": [],
            "sidecars": [],
            "controls": [],
        })
    assert not (tmp_path / "training_trace.jsonl").exists()
    assert not (tmp_path / "checkpoint_metrics.json").exists()


def test_terminal_control_requires_then_uses_finalized_parse_receipt() -> None:
    module = _load_runner("_motion_alignment_deferred_terminal_control")
    evaluation = _failing_evaluation(module)
    control = module._DeferredTerminalControl(evaluation)

    with pytest.raises(
        PermissionError, match="before finalized-ledger parse"
    ):
        control["action"]
    assert not control

    module._FINALIZED_LEDGER_PARSE_RECEIPT = {
        "corrected_parser_pass": True,
        "full_on_disk_ledger_checked": True,
        "ledger_file_sha256": "1" * 64,
        "record_count": 1,
        "last_record_content_sha256": "2" * 64,
        "terminal_record_type": "RUNTIME_INPUT_ACCESS_FINALIZED",
    }
    expected = module.contract.checkpoint_control_decision(
        update=module.contract.MAXIMUM_UPDATE,
        evaluation=evaluation,
        integrity_pass=True,
    )
    assert control["action"] == module.contract.CONTROL_FAIL
    assert dict(control) == expected


def test_contract_invalid_ledger_publishes_terminal_successor_receipt(
    tmp_path: Path,
) -> None:
    module = _load_runner("_motion_alignment_contract_invalid_failure")
    output = tmp_path / "attempt"
    output.mkdir(mode=0o700)
    reservation, reservation_raw, ledger = _ledger(
        module, output, tmp_path
    )
    _append_context_invalid_schedule_open(module, ledger)
    progress = module.OperationProgress()
    progress.enter("synthetic_contract_invalid_failure", role="authority")

    module._terminal_failure(
        output,
        reservation,
        reservation_raw,
        ledger,
        progress,
        error=RuntimeError("synthetic triggering failure"),
    )

    failed_raw = (output / "failed.json").read_bytes()
    failed = module.contract.parse_canonical_json(
        failed_raw, name="synthetic contract-invalid failure"
    )
    ledger_raw = (output / module.PartialAccessLedger.RELATIVE_PATH).read_bytes()
    module.contract.validate_contract_invalid_ledger_failure_receipt(
        failed,
        reservation_binding=module._BASE._binding(
            "reservation.json", reservation, reservation_raw
        ),
        ledger_raw=ledger_raw,
    )
    assert failed["status"] == (
        module.contract.CONTRACT_INVALID_LEDGER_FAILURE_STATUS
    )
    assert failed["retry_authorized"] is False
    assert failed["error"]["type"] == "RuntimeError"
    assert failed["ledger_parser_failure"]["accepted"] is False
    assert failed["ledger_parser_failure"]["error"]["type"] == (
        "PermissionError"
    )
    assert "CAUSAL_TEMPORAL_V1" not in failed_raw.decode("utf-8")
    assert not (output / "result.json").exists()
    assert not (output / "completed.json").exists()
    _assert_sealed(output)


def test_near_miss_binding_parser_failure_still_publishes_receipt(
    tmp_path: Path,
) -> None:
    module = _load_runner("_motion_alignment_near_miss_failure")
    output = tmp_path / "attempt"
    output.mkdir(mode=0o700)
    reservation, reservation_raw, ledger = _ledger(
        module, output, tmp_path
    )
    _append_near_miss_render_open(module, ledger)
    progress = module.OperationProgress()
    progress.enter("synthetic_near_miss_failure", role="authority")

    module._terminal_failure(
        output,
        reservation,
        reservation_raw,
        ledger,
        progress,
        error=RuntimeError("synthetic near-miss failure"),
    )

    failed = module.contract.parse_canonical_json(
        (output / "failed.json").read_bytes(),
        name="synthetic near-miss contract-invalid failure",
    )
    ledger_raw = (
        output / module.PartialAccessLedger.RELATIVE_PATH
    ).read_bytes()
    module.contract.validate_contract_invalid_ledger_failure_receipt(
        failed,
        reservation_binding=module._BASE._binding(
            "reservation.json", reservation, reservation_raw
        ),
        ledger_raw=ledger_raw,
    )
    assert failed["status"] == (
        module.contract.CONTRACT_INVALID_LEDGER_FAILURE_STATUS
    )
    assert failed["ledger_parser_failure"]["accepted"] is False
    assert failed["retry_authorized"] is False
    _assert_sealed(output)


def test_parser_valid_failure_uses_successor_normal_status(
    tmp_path: Path,
) -> None:
    module = _load_runner("_motion_alignment_normal_failure")
    output = tmp_path / "attempt"
    output.mkdir(mode=0o700)
    reservation, reservation_raw, ledger = _ledger(
        module, output, tmp_path
    )
    progress = module.OperationProgress()
    progress.enter("synthetic_normal_failure", role="authority")

    module._terminal_failure(
        output,
        reservation,
        reservation_raw,
        ledger,
        progress,
        error=RuntimeError("synthetic normal failure"),
    )

    failed_raw = (output / "failed.json").read_bytes()
    failed = module.contract.parse_canonical_json(
        failed_raw, name="synthetic normal failure"
    )
    module.contract.validate_failure_receipt(
        failed,
        reservation_binding=module._BASE._binding(
            "reservation.json", reservation, reservation_raw
        ),
    )
    assert failed["status"] == module.contract.NORMAL_FAILURE_STATUS
    assert failed["retry_authorized"] is False
    assert "CAUSAL_TEMPORAL_V1" not in failed_raw.decode("utf-8")
    _assert_sealed(output)


def test_preledger_failure_uses_successor_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_runner("_motion_alignment_preledger_failure")
    output = tmp_path / "attempt"
    output.mkdir(mode=0o700)
    reservation, reservation_raw = _reservation(module, output)

    def fail_before_header(name: str) -> None:
        if name == "ledger_before_header":
            raise RuntimeError("synthetic preledger failure")

    monkeypatch.setattr(
        module._BASE, "_failure_boundary", fail_before_header
    )
    with pytest.raises(RuntimeError, match="synthetic preledger failure"):
        module._BASE._execute_after_reservation(
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

    failed_raw = (output / "failed.json").read_bytes()
    failed = module.contract.parse_canonical_json(
        failed_raw, name="synthetic preledger failure"
    )
    module.contract.validate_pre_ledger_failure_receipt(
        failed,
        reservation_binding=module._BASE._binding(
            "reservation.json", reservation, reservation_raw
        ),
        attempt_identity=reservation["attempt_identity"],
    )
    assert failed["status"] == module.contract.PRE_LEDGER_FAILURE_STATUS
    assert failed["retry_authorized"] is False
    assert "CAUSAL_TEMPORAL_V1" not in failed_raw.decode("utf-8")
    _assert_sealed(output)
