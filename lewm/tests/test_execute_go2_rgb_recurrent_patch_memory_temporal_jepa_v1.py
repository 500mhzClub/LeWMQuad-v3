from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import stat
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from lewm.benchmarks import (
    go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as metrics,
)
from scripts import (
    execute_go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as executor,
)


FULL_IDENTITY = "1" * 64
SENTINEL_IDENTITY = "2" * 64


def _authority(preflight_binding: Mapping[str, Any] | None = None) -> dict:
    return executor._content_bound(
        {
            "schema": f"{executor.SCHEMA_PREFIX}_future_execution_authority_v1",
            "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT",
            "scientific_payload_authorized": True,
            "one_shot": True,
            "maximum_updates": 400,
            "maximum_presentations": 16_000,
            "retry_authorized": False,
            "resume_authorized": False,
            "preregistration_commit": executor.PREREGISTRATION_COMMIT,
            "pinned_source_and_review_commit": "a" * 40,
            "certified_source_root": executor.CERTIFIED_SOURCE_ROOT,
            "output_root": executor.OUTPUT_ROOT_RELATIVE_PATH,
            "rgb_root_relative_path": executor.RGB_ROOT_RELATIVE_PATH,
            "output_root_absent_at_authorization": True,
            "device": "cuda:0",
            "runtime_data_root": executor.RUNTIME_DATA_ROOT,
            "selectors": {
                "executor_module": executor.__name__,
                "model_module": executor.MODEL_MODULE_NAME,
                "model_class": executor.MODEL_CLASS_NAME,
                "training_module": executor.TRAINING_MODULE_NAME,
                "evaluation_module": executor.EVALUATION_MODULE_NAME,
                "metrics_module": executor.METRICS_MODULE_NAME,
                "metadata_preflight_module": executor.PREFLIGHT_MODULE_NAME,
            },
            "clean_export_certification": {
                "path": executor.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
                "file_sha256": "b" * 64,
                "content_sha256": "c" * 64,
                "byte_count": 1,
            },
            "metadata_preflight_receipt": dict(
                preflight_binding
                or {
                    "path": executor.METADATA_PREFLIGHT_RECEIPT_RELATIVE_PATH,
                    "file_sha256": "d" * 64,
                    "content_sha256": "e" * 64,
                    "byte_count": 1,
                }
            ),
            "runtime_inputs": {
                name: dict(value)
                for name, value in executor.RUNTIME_INPUT_BINDINGS.items()
            },
        }
    )


def _synthetic_environment_attestation(
    authority: Mapping[str, Any],
) -> Any:
    return executor._build_execution_environment_attestation_v1(
        authority["certified_source_root"],
        authority,
        source_receipt={
            "status": "PASS_CERTIFIED_SOURCE_REHASH",
            "validated_path_count": 1,
            "bindings_sha256": "3" * 64,
            "certification_content_sha256": "4" * 64,
        },
        gpu_receipt={
            "status": "PASS_EXACTLY_ONE_VISIBLE_AMD_R9700",
            "visible_device_count": 1,
            "visible_device_name": "AMD Radeon AI PRO R9700",
            "torch_hip_version": "synthetic",
            "tensor_allocation_count": 0,
            "dataset_open_count": 0,
            "checkpoint_open_count": 0,
        },
    )


def _preflight_receipt() -> dict:
    expected = executor.RUNTIME_INPUT_BINDINGS
    return executor._content_bound(
        {
            "schema": f"{executor.SCHEMA_PREFIX}_metadata_preflight_receipt_v1",
            "status": "PASS_METADATA_PREFLIGHT",
            "preregistration_commit": executor.PREREGISTRATION_COMMIT,
            "authority": {"path": "docs/synthetic.json"},
            "reservation": {"status": "synthetic"},
            "inputs": {
                "train": {
                    **dict(expected["h6_train_index"]),
                    "role": "train",
                    "row_count": 16_000,
                },
                "validation": {
                    **dict(expected["h6_validation_index"]),
                    "role": "val",
                    "row_count": 2_048,
                },
            },
            "train": {
                "schedule_indices_sha256": metrics.TRAIN_SCHEDULE_SHA256,
            },
            "validation": {
                "sentinel_indices_sha256": metrics.SENTINEL_INDICES_SHA256,
                "full_wrong_history_donors_sha256": (
                    metrics.FULL_WRONG_HISTORY_DONORS_SHA256
                ),
                "sentinel_wrong_history_donors_sha256": (
                    metrics.SENTINEL_WRONG_HISTORY_DONORS_SHA256
                ),
                "full_panel_identity_sha256": FULL_IDENTITY,
                "sentinel_panel_identity_sha256": SENTINEL_IDENTITY,
            },
            "access": {
                "metadata_index_open_count": 2,
                "rgb_open_count": 0,
                "checkpoint_open_count": 0,
                "navigation_open_count": 0,
                "held_out_or_sealed_opened": False,
            },
            "checks": {
                "bound_indices": True,
                "schedule_and_controls": True,
                "zero_payload_access": True,
            },
        }
    )


def _write_content_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    raw = executor._canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": path.relative_to(path.parents[len(path.parts) - 2]).as_posix(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": value["content_sha256"],
        "byte_count": len(raw),
    }


def _control(ratio: float, families: int = 8) -> metrics.ControlSummary:
    return metrics.ControlSummary(
        correct_macro_mean=ratio,
        control_macro_mean=1.0,
        primary_ratio=ratio,
        advantage_macro_mean=1.0 - ratio,
        advantage_bootstrap_lower_95=0.01,
        positive_family_count=families,
        correct_by_scene={},
        control_by_scene={},
        advantage_by_scene={},
        advantage_by_family={},
    )


class _FakeModel(torch.nn.Module):
    def __init__(self, state: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1, bias=False)
        self.encoder.weight.data.copy_(state["encoder.weight"])
        self.predictor_mask_token = torch.nn.Parameter(
            state["predictor_mask_token"].clone()
        )
        self.action_embedding = torch.nn.Embedding(9, 1)
        self.time_embedding = torch.nn.Embedding(3, 1)
        self.temporal_gru = torch.nn.GRU(1, 1, batch_first=True)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)
        self.register_buffer(
            "ema_update_count",
            torch.zeros((), dtype=torch.long),
        )
        self.target_encoder.eval()

    def to(self, *_args: Any, **_kwargs: Any) -> "_FakeModel":
        return self

    def train(self, mode: bool = True) -> "_FakeModel":
        super().train(mode)
        self.target_encoder.requires_grad_(False)
        self.target_encoder.eval()
        return self


def _accounting(update: int) -> dict[str, int]:
    return executor._expected_training_accounting(update)


class _FakeTraining:
    @staticmethod
    def partition_parameters_v1(model: _FakeModel) -> Any:
        return SimpleNamespace(
            encoder=tuple(model.encoder.parameters()),
            predictor=(model.predictor_mask_token,),
            memory=(
                *tuple(model.action_embedding.parameters()),
                *tuple(model.time_embedding.parameters()),
                *tuple(model.temporal_gru.parameters()),
            ),
            target=tuple(model.target_encoder.parameters()),
        )

    @staticmethod
    def parameter_inventory_v1(_model: _FakeModel) -> dict[str, Any]:
        return {
            "schema": "synthetic",
            "target_optimizer_excluded": True,
        }

    @staticmethod
    def validate_optimizer_v1(_optimizer: Any, _partition: Any) -> None:
        return None

    @staticmethod
    def build_optimizer_v1(_model: _FakeModel) -> object:
        return SimpleNamespace(state_dict=lambda: {})

    @staticmethod
    def training_update_v1(
        model: _FakeModel,
        _optimizer: object,
        _context: Any,
        _actions: Any,
        _future: Any,
        rows: Any,
        *,
        expected_row_indices: Any,
        schedule_offset: int,
        accounting: Any,
    ) -> Any:
        update = 1 if accounting is None else accounting["updates"] + 1
        assert schedule_offset == 10 * (update - 1)
        flattened = tuple(row for batch in rows for row in batch)
        assert flattened == tuple(expected_row_indices)
        model.ema_update_count.fill_(update)
        return SimpleNamespace(
            accounting=_accounting(update),
            mean_jepa_loss=1.0 / update,
            gradient_receipt={
                "sole_future_jepa_route": True,
                "all_gradient_receipts_finite": True,
                "encoder_missing_gradient_tensor_count": 0,
                "predictor_missing_gradient_tensor_count": 0,
                "memory_missing_gradient_tensor_count": 0,
                "encoder_gradient_norm_before_clip": 1.0,
                "predictor_gradient_norm_before_clip": 1.0,
                "memory_gradient_norm_before_clip": 1.0,
            },
            target_gradient_tensor_count=0,
            optimizer_steps_this_update=1,
            ema_steps_this_update=1,
            row_indices_sha256=f"{update:064x}"[-64:],
            target_indices_sha256="f" * 64,
        )

    @staticmethod
    def checkpoint_payload_v1(
        model: _FakeModel,
        _optimizer: object,
        accounting: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": f"{executor.SCHEMA_PREFIX}_checkpoint_v1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "accounting": dict(accounting),
        }


def _empty_access() -> dict[str, int]:
    result = {
        "rgb_tensor_request_count": 0,
        "rgb_open_attempt_count": 0,
        "rgb_open_success_count": 0,
        "rgb_decode_success_count": 0,
        "rgb_byte_count": 0,
        "denied_rgb_position_request_count": 0,
    }
    for role in ("train", "val"):
        for kind in ("factual", "donor"):
            for position in range(7):
                for operation in (
                    "request",
                    "open_attempt",
                    "open_success",
                    "decode_success",
                ):
                    result[
                        f"{role}_{kind}_rgb_position_{position}_{operation}_count"
                    ] = 0
    return result


class _FakeRuntime:
    def __init__(self) -> None:
        self.training_schedule = tuple(range(4_000))
        self.sentinel_indices = tuple(range(256))
        self._access = _empty_access()
        self.requested_updates: list[int] = []
        self.closed = False

    def access_snapshot(self) -> dict[str, int]:
        return dict(self._access)

    def load_training_microbatches(
        self,
        schedule_slice: Any,
        _device: Any,
    ) -> tuple[Any, ...]:
        selected = tuple(schedule_slice)
        self.requested_updates.append(selected[0] // 10 + 1)
        for name in (
            "rgb_tensor_request_count",
            "rgb_open_attempt_count",
            "rgb_open_success_count",
            "rgb_decode_success_count",
        ):
            self._access[name] += 40
        for position in range(4):
            for operation in (
                "request",
                "open_attempt",
                "open_success",
                "decode_success",
            ):
                self._access[
                    f"train_factual_rgb_position_{position}_{operation}_count"
                ] += 10
        return tuple(
            SimpleNamespace(
                row_indices=selected[offset : offset + 2],
                context_rgb=torch.zeros(2, 3, 3, 1, 1),
                action_sequence=torch.zeros(2, 3, dtype=torch.long),
                target_rgb=torch.zeros(2, 3, 1, 1),
            )
            for offset in range(0, 10, 2)
        )

    def access_audit(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "passed": True,
            "forbidden_rgb_open_count": 0,
            "counters": dict(self._access),
        }

    def close(self) -> None:
        self.closed = True


def _runtime_audit() -> dict[str, Any]:
    bindings = executor.RUNTIME_INPUT_BINDINGS
    return {
        "train": dict(bindings["h6_train_index"]),
        "validation": dict(bindings["h6_validation_index"]),
        "panels": {
            "training_schedule_indices_sha256": metrics.TRAIN_SCHEDULE_SHA256,
            "sentinel_indices_sha256": metrics.SENTINEL_INDICES_SHA256,
            "wrong_history_donor_indices_sha256": (
                metrics.FULL_WRONG_HISTORY_DONORS_SHA256
            ),
            "sentinel_wrong_history_donor_indices_sha256": (
                metrics.SENTINEL_WRONG_HISTORY_DONORS_SHA256
            ),
        },
        "rgb_open_count": 0,
        "checkpoint_open_count": 0,
        "gpu_tensor_allocation_count": 0,
    }


def _predecessor_panel(update: int) -> dict[str, Any]:
    bindings = executor.RUNTIME_INPUT_BINDINGS
    controls = {
        name: _control(0.80).to_dict()
        for name in executor.PREDECESSOR_CONTROL_NAMES
    }
    raw_health = {
        branch: {
            "image_count": 2_048,
            "token_count": 256,
            "feature_dimension": 192,
            "effective_rank": 20.0,
            "cross_sample_variance": 1.0,
            "within_image_spatial_diversity": 1.0,
            "finite": True,
        }
        for branch in ("online", "target")
    }
    return {
        "schema": f"{executor.SCHEMA_PREFIX}_predecessor_retention_panel_v1",
        "temporal_update": update,
        "underlying_spatial_evaluator_update": 0,
        "underlying_spatial_evaluator_schema": "synthetic",
        "runtime_audit": {
            "train": dict(bindings["h6_train_index"]),
            "validation": dict(bindings["h6_validation_index"]),
            "place": {
                "manifest_file_sha256": bindings["place_triplet_manifest"][
                    "file_sha256"
                ],
                "index_file_sha256": bindings[
                    "place_triplet_checkpoint_selection_index"
                ]["file_sha256"],
            },
        },
        "evaluation": {
            "schema": "synthetic_spatial_evaluation",
            "update": 0,
            "controls": controls,
            "raw_health": raw_health,
            "place": {
                "retrieval": {"chance_multiple": 3.0},
                "target_place_key_effective_rank": 20.0,
            },
            "access": {
                "future_rgb_tensor_count": 0,
                "action_tensor_count": 0,
            },
            "integrity": {"passed": True},
        },
    }


def _temporal_record(
    model: _FakeModel,
    update: int,
    *,
    full: bool,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = {0: 1.10, 50: 0.98, 100: 0.94, 200: 0.96, 400: 0.90}
    ratio = ratios[update]
    controls = {name: _control(ratio) for name in metrics.CONTROL_NAMES}
    row_count = 2_048 if full else 256
    health = metrics.RepresentationHealth(
        row_count=row_count,
        token_count=256,
        feature_dimension=192,
        effective_rank=10.0,
        cross_sample_variance=1.0,
        finite=True,
    )
    target_health = metrics.RepresentationHealth(
        row_count=row_count,
        token_count=64,
        feature_dimension=192,
        effective_rank=10.0,
        cross_sample_variance=1.0,
        finite=True,
    )
    prediction_health = metrics.RepresentationHealth(
        row_count=row_count,
        token_count=64,
        feature_dimension=192,
        effective_rank=8.0,
        cross_sample_variance=0.8,
        finite=True,
    )
    predecessor = baseline.get("predecessor_controls")
    observation = metrics.TemporalObservation(
        update=update,
        panel_kind="full" if full else "sentinel",
        panel_identity_sha256=FULL_IDENTITY if full else SENTINEL_IDENTITY,
        controls=controls,
        recurrent_health=health,
        prediction_health=prediction_health,
        target_health=target_health,
        integrity=metrics.IntegrityFacts(
            access_and_accounting_exact=True,
            all_evaluated_finite=True,
            target_frozen_eval=True,
            target_gradient_tensor_count=0,
            ema_count=int(model.ema_update_count),
            latest_training_receipt_pass=baseline.get(
                "latest_training_receipt_pass"
            ),
            baseline_health_noncollapsed=True,
        ),
        predecessor_controls=(
            None
            if predecessor is None
            else {
                name: metrics.ControlSummary(**value)
                for name, value in predecessor.items()
            }
        ),
        raw_health_retentions=baseline.get("raw_health_retentions"),
        place_chance_multiple_retention=baseline.get(
            "place_chance_multiple_retention"
        ),
        target_place_rank_retention=baseline.get(
            "target_place_rank_retention"
        ),
    )
    checks = metrics.observation_survival_checks(observation)
    if full and update in (200, 400):
        checks = metrics.qualification_checks(observation)
    return {
        "schema": f"{executor.SCHEMA_PREFIX}_checkpoint_evaluation_v1",
        "update": update,
        "panel_kind": observation.panel_kind,
        "panel_identity_sha256": observation.panel_identity_sha256,
        "row_count": row_count,
        "controls": {
            name: value.to_dict() for name, value in controls.items()
        },
        "health": {
            "recurrent": health.to_dict(),
            "prediction": prediction_health.to_dict(),
            "target": target_health.to_dict(),
        },
        "diagnostics": {"recurrent_temporal_change": 1.0},
        "integrity": asdict(observation.integrity),
        "access": {},
        "access_provenance": {
            "source_panel_row_count": row_count,
            "derived_from_source_panel": False,
            "additional_rgb_open_count": None,
            "additional_model_call_count": None,
        },
        "gate": {"checks": checks, "passed": all(checks.values())},
    }


def _fake_apis(*, fail_update: int | None = None) -> Any:
    runtime = _FakeRuntime()
    predecessor_state = {
        "encoder.weight": torch.ones(1, 1),
        "predictor_mask_token": torch.ones(1, 1, 1),
        "target_encoder.weight": torch.full((1, 1), 7.0),
        "ema_update_count": torch.tensor(91, dtype=torch.long),
    }

    def evaluate_checkpoint(
        model: _FakeModel,
        _runtime: _FakeRuntime,
        update: int,
        _device: Any,
        *,
        full: bool,
        baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        if update == fail_update:
            raise RuntimeError("synthetic observation failure")
        return _temporal_record(
            model,
            update,
            full=full,
            baseline=baseline,
        )

    def evaluate_update_zero(
        model: _FakeModel,
        _runtime: _FakeRuntime,
        _device: Any,
        *,
        baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        full = _temporal_record(model, 0, full=True, baseline=baseline)
        sentinel = _temporal_record(model, 0, full=False, baseline=baseline)
        sentinel["access_provenance"] = {
            "source_panel_row_count": 2_048,
            "derived_from_source_panel": True,
            "additional_rgb_open_count": 0,
            "additional_model_call_count": 0,
        }
        return {
            "schema": f"{executor.SCHEMA_PREFIX}_update_zero_full_and_sentinel_v1",
            "update": 0,
            "single_temporal_rgb_and_model_pass": True,
            "full": full,
            "sentinel": sentinel,
        }

    preflight = _preflight_receipt()
    return SimpleNamespace(
        torch=torch,
        metrics=metrics,
        model_class=_FakeModel,
        training=_FakeTraining,
        runtime=runtime,
        predecessor_calls=[],
        load_preflight=lambda *_args: (
            preflight,
            {
                "receipt_open_count": 1,
                "rgb_open_count": 0,
                "checkpoint_open_count": 0,
                "passed": True,
            },
        ),
        load_predecessor=lambda *_args: (
            predecessor_state,
            {
                "checkpoint_open_count": 1,
                "checkpoint_deserialize_count": 1,
                "post_initialization_checkpoint_reopen_count": 0,
                "passed": True,
            },
        ),
        open_runtime=lambda *_args, **_kwargs: (runtime, _runtime_audit()),
        evaluate_update_zero=evaluate_update_zero,
        evaluate_checkpoint=evaluate_checkpoint,
        evaluate_predecessor=lambda _model, _root, update, _device: (
            _predecessor_panel(update)
        ),
    )


def test_authority_reservation_and_metadata_preflight_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "RUNTIME_DATA_ROOT", str(tmp_path.resolve()))
    receipt = _preflight_receipt()
    receipt_path = tmp_path / executor.METADATA_PREFLIGHT_RECEIPT_RELATIVE_PATH
    raw = executor._canonical_json_bytes(receipt) + b"\n"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(raw)
    binding = {
        "path": executor.METADATA_PREFLIGHT_RECEIPT_RELATIVE_PATH,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": receipt["content_sha256"],
        "byte_count": len(raw),
    }
    authority = _authority(binding)
    assert executor.validate_future_execution_prerequisites_v1(authority) == authority
    loaded, access = executor.load_metadata_preflight_receipt_v1(
        tmp_path,
        authority,
    )
    assert loaded == receipt
    assert access["receipt_open_count"] == 1
    output = tmp_path / executor.OUTPUT_ROOT_RELATIVE_PATH
    with pytest.raises(TypeError, match="environment_attestation"):
        executor.reserve_attempt_v1(
            tmp_path,
            authority,
            created_utc="2026-07-31T00:00:00Z",
        )
    assert not output.exists()
    environment = _synthetic_environment_attestation(authority)
    reservation = executor.reserve_attempt_v1(
        tmp_path,
        authority,
        environment_attestation=environment,
        created_utc="2026-07-31T00:00:00Z",
    )
    assert executor.validate_attempt_reservation_v1(reservation) == reservation
    assert reservation[
        "environment_attestation_content_sha256"
    ] == environment.receipt["content_sha256"]
    assert stat.S_IMODE((output / "reservation.json").stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        executor.reserve_attempt_v1(
            tmp_path,
            authority,
            environment_attestation=environment,
            created_utc="2026-07-31T00:00:01Z",
        )


def test_synthetic_predecessor_checkpoint_opens_and_deserializes_once(
    tmp_path: Path,
) -> None:
    state = {
        "encoder.weight": torch.ones(1, 1),
        "predictor_mask_token": torch.ones(1, 1, 1),
        "target_encoder.weight": torch.full((1, 1), 9.0),
        "ema_update_count": torch.tensor(1_000, dtype=torch.long),
    }
    names_sha = hashlib.sha256(
        executor._canonical_json_bytes(tuple(state))
    ).hexdigest()
    checkpoint = {
        "schema": (
            "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
            "checkpoint_v1"
        ),
        "model_state_dict": state,
        "optimizer_state_dict": {},
        "accounting": {
            "updates": 1_000,
            "presentations": 16_000,
            "mask_rows": 16_000,
            "online_frame_encodings": 16_000,
            "ema_target_frame_encodings": 16_000,
            "microbatch_graphs": 4_000,
            "backward_calls": 4_000,
            "global_gradient_clips": 1_000,
            "optimizer_steps": 1_000,
            "ema_steps": 1_000,
        },
        "model_state_inventory": {
            "state_tensor_count": 4,
            "state_names_sha256": names_sha,
            "ema_update_count": 1_000,
        },
        "training_contract": {},
        "update": 1_000,
        "authority_sha256": "a" * 64,
        "rng": {},
        "complete_continuation_state": True,
        "same_attempt_reopen_count": 0,
        "retry_authorized": False,
        "resume_authorized": False,
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    checkpoint_raw = buffer.getvalue()
    checkpoint_path = tmp_path / "runtime/predecessor.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.write_bytes(checkpoint_raw)
    checkpoint_binding = {
        "path": "runtime/predecessor.pt",
        "file_sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
        "byte_count": len(checkpoint_raw),
    }
    selected = {
        **checkpoint_binding,
        "path": "snapshots/update_1000.pt",
        "update": 1_000,
        "retry_authorized": False,
        "resume_authorized": False,
    }
    result = executor._content_bound(
        {
            "schema": (
                "lewm_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1_"
                "scientific_result_v1"
            ),
            "status": "PASS_PERCEPTION_QUALIFIED",
            "passed": True,
            "selected_update": 1_000,
            "terminal_artifacts": {"selected_checkpoint": selected},
        }
    )
    success = executor._content_bound(
        {
            "status": "PASS_PERCEPTION_QUALIFIED",
            "selected_update": 1_000,
            "selected_checkpoint": selected,
        }
    )
    bindings: dict[str, Any] = {"predecessor_checkpoint": checkpoint_binding}
    for name, value in (
        ("predecessor_scientific_result", result),
        ("predecessor_success", success),
    ):
        path = tmp_path / f"runtime/{name}.json"
        raw = executor._canonical_json_bytes(value) + b"\n"
        path.write_bytes(raw)
        bindings[name] = {
            "path": f"runtime/{name}.json",
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": value["content_sha256"],
            "byte_count": len(raw),
        }

    calls = 0

    class TorchProxy:
        Tensor = torch.Tensor
        uint8 = torch.uint8
        isfinite = staticmethod(torch.isfinite)

        @staticmethod
        def load(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            assert isinstance(args[0], io.BytesIO)
            return torch.load(*args, **kwargs)

    loaded, receipt = executor.load_predecessor_model_state_v1(
        tmp_path,
        {"runtime_inputs": bindings},
        TorchProxy,
    )
    assert calls == 1
    assert tuple(loaded) == tuple(state)
    assert receipt["checkpoint_open_count"] == 1
    assert receipt["checkpoint_deserialize_count"] == 1
    assert receipt["migration"]["accepted_state_tensor_count"] == 2
    assert receipt["migration"]["rejected_state_tensor_count"] == 2


def test_synthetic_engine_runs_exact_schedule_and_selects_update_400(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "RUNTIME_DATA_ROOT", str(tmp_path.resolve()))
    authority = _authority()
    environment = _synthetic_environment_attestation(authority)
    reservation = executor.reserve_attempt_v1(
        tmp_path,
        authority,
        environment_attestation=environment,
        created_utc="2026-07-31T00:00:00Z",
    )
    apis = _fake_apis()
    predecessor_calls: list[int] = []
    original = apis.evaluate_predecessor

    def record_predecessor(*args: Any) -> dict[str, Any]:
        update = int(args[2])
        predecessor_calls.append(update)
        return original(*args)

    apis.evaluate_predecessor = record_predecessor
    with pytest.raises(PermissionError, match="guard-issued"):
        executor.run_authorized_engine_v1(
            authority=authority,
            reservation=reservation,
            environment_attestation={},
            repository_root=tmp_path,
            runtime_data_root=tmp_path,
            device="cuda:0",
            apis=apis,
        )
    assert apis.runtime.requested_updates == []
    assert predecessor_calls == []
    result = executor.run_authorized_engine_v1(
        authority=authority,
        reservation=reservation,
        environment_attestation=environment,
        repository_root=tmp_path,
        runtime_data_root=tmp_path,
        device="cuda:0",
        apis=apis,
    )
    output = tmp_path / executor.OUTPUT_ROOT_RELATIVE_PATH
    assert result["status"] == "PASS_TEMPORAL_PERCEPTION_QUALIFIED"
    assert result["selected_update"] == 400
    assert result["accounting"]["logical_rgb_presentations"] == 16_000
    assert apis.runtime.requested_updates == list(range(1, 401))
    assert predecessor_calls == [0, 200, 400]
    assert [value["update"] for value in result["checkpoints"]] == [200, 400]
    assert all(
        value["predecessor_checkpoint_reopen_count"] == 0
        for value in result["checkpoints"]
    )
    assert apis.runtime.closed is True
    assert (output / "metrics/update_0_full.json").is_file()
    assert (output / "metrics/update_0_sentinel.json").is_file()
    assert (output / "metrics/update_50_sentinel.json").is_file()
    assert (output / "metrics/update_200_full.json").is_file()
    assert (output / "snapshots/update_400.pt").is_file()
    assert stat.S_IMODE((output / "success.json").stat().st_mode) == 0o444
    assert len((output / "trace.jsonl").read_bytes().splitlines()) == 400
    terminal_access = executor._decode_content_bound_json(
        (output / "receipts/terminal_access.json").read_bytes(),
        name="synthetic terminal access",
    )
    schedule_prefix = terminal_access["schedule_prefix"]
    assert schedule_prefix["consumed_schedule_row_count"] == 4_000
    assert schedule_prefix["consumed_schedule_rows_equal_runtime_prefix"] is True
    assert schedule_prefix["consumed_schedule_rows_unique"] is True
    assert schedule_prefix["runtime_training_schedule_row_count"] == 4_000
    assert schedule_prefix["runtime_training_schedule_unique"] is True
    assert schedule_prefix[
        "consumed_schedule_rows_canonical_sha256"
    ] == hashlib.sha256(
        executor._canonical_json_bytes(tuple(range(4_000)))
    ).hexdigest()


def test_observation_exception_consumes_attempt_with_exact_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "RUNTIME_DATA_ROOT", str(tmp_path.resolve()))
    authority = _authority()
    environment = _synthetic_environment_attestation(authority)
    reservation = executor.reserve_attempt_v1(
        tmp_path,
        authority,
        environment_attestation=environment,
        created_utc="2026-07-31T00:00:00Z",
    )
    apis = _fake_apis(fail_update=50)
    result = executor.run_authorized_engine_v1(
        authority=authority,
        reservation=reservation,
        environment_attestation=environment,
        repository_root=tmp_path,
        runtime_data_root=tmp_path,
        device="cuda:0",
        apis=apis,
    )
    assert result["status"] == "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME"
    assert result["stage"] == "observe_update_50_temporal"
    assert result["last_completed_update"] == 50
    assert result["retry_authorized"] is False
    assert result["resume_authorized"] is False
    assert result["predecessor_checkpoint_reopen_count"] == 0
    assert result["access"]["predecessor"]["checkpoint_open_count"] == 1
    assert result["access"]["schedule_prefix"][
        "consumed_schedule_row_count"
    ] == 500
    assert result["access"]["schedule_prefix"][
        "consumed_schedule_rows_equal_runtime_prefix"
    ] is True
    assert apis.runtime.closed is True


def test_nonunique_runtime_schedule_terminalizes_before_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "RUNTIME_DATA_ROOT", str(tmp_path.resolve()))
    authority = _authority()
    environment = _synthetic_environment_attestation(authority)
    reservation = executor.reserve_attempt_v1(
        tmp_path,
        authority,
        environment_attestation=environment,
        created_utc="2026-07-31T00:00:00Z",
    )
    apis = _fake_apis()
    apis.runtime.training_schedule = (
        tuple(range(3_999)) + (3_998,)
    )
    result = executor.run_authorized_engine_v1(
        authority=authority,
        reservation=reservation,
        environment_attestation=environment,
        repository_root=tmp_path,
        runtime_data_root=tmp_path,
        device="cuda:0",
        apis=apis,
    )
    assert result["status"] == "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME"
    assert result["stage"] == "open_temporal_runtime"
    assert result["last_completed_update"] == 0
    assert result["access"]["schedule_prefix"][
        "runtime_training_schedule_row_count"
    ] == 4_000
    assert result["access"]["schedule_prefix"][
        "runtime_training_schedule_unique"
    ] is False
    assert apis.runtime.closed is True


def test_direct_executor_source_guard_fails_before_gpu_or_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    source_root.mkdir()
    runtime_root.mkdir()
    monkeypatch.setattr(
        executor,
        "CERTIFIED_SOURCE_ROOT",
        str(source_root.resolve()),
    )
    monkeypatch.setattr(
        executor,
        "RUNTIME_DATA_ROOT",
        str(runtime_root.resolve()),
    )
    authority = _authority()
    events: list[str] = []

    def reject_source(_root: Path, _authority: Mapping[str, Any]) -> dict:
        events.append("source")
        raise PermissionError("synthetic source rehash failure")

    def unexpected_gpu(_torch: Any) -> dict:
        events.append("gpu")
        raise AssertionError("GPU guard ran after failed source guard")

    def unexpected_reservation(*_args: Any, **_kwargs: Any) -> dict:
        events.append("reservation")
        raise AssertionError("reservation ran after failed source guard")

    monkeypatch.setattr(executor, "validate_certified_source_v1", reject_source)
    monkeypatch.setattr(executor, "validate_gpu_v1", unexpected_gpu)
    monkeypatch.setattr(executor, "reserve_attempt_v1", unexpected_reservation)
    with pytest.raises(PermissionError, match="source rehash"):
        executor.execute_authorized_v1(source_root, authority)
    assert events == ["source"]
    assert not (
        runtime_root / executor.OUTPUT_ROOT_RELATIVE_PATH
    ).exists()


def test_direct_executor_gpu_guard_fails_before_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source"
    runtime_root = tmp_path / "runtime"
    source_root.mkdir()
    runtime_root.mkdir()
    monkeypatch.setattr(
        executor,
        "CERTIFIED_SOURCE_ROOT",
        str(source_root.resolve()),
    )
    monkeypatch.setattr(
        executor,
        "RUNTIME_DATA_ROOT",
        str(runtime_root.resolve()),
    )
    authority = _authority()
    events: list[str] = []

    def accept_source(_root: Path, _authority: Mapping[str, Any]) -> dict:
        events.append("source")
        return {"passed": True}

    def reject_gpu(_torch: Any) -> dict:
        events.append("gpu")
        raise RuntimeError("synthetic AMD R9700 failure")

    def unexpected_reservation(*_args: Any, **_kwargs: Any) -> dict:
        events.append("reservation")
        raise AssertionError("reservation ran after failed GPU guard")

    monkeypatch.setattr(executor, "validate_certified_source_v1", accept_source)
    monkeypatch.setattr(executor, "validate_gpu_v1", reject_gpu)
    monkeypatch.setattr(executor, "reserve_attempt_v1", unexpected_reservation)
    with pytest.raises(RuntimeError, match="AMD R9700"):
        executor.execute_authorized_v1(source_root, authority)
    assert events == ["source", "gpu"]
    assert not (
        runtime_root / executor.OUTPUT_ROOT_RELATIVE_PATH
    ).exists()


def test_nonfinite_optimizer_state_fails_before_update_zero_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(executor, "RUNTIME_DATA_ROOT", str(tmp_path.resolve()))
    authority = _authority()
    environment = _synthetic_environment_attestation(authority)
    reservation = executor.reserve_attempt_v1(
        tmp_path,
        authority,
        environment_attestation=environment,
        created_utc="2026-07-31T00:00:00Z",
    )
    apis = _fake_apis()

    class BadOptimizer:
        @staticmethod
        def state_dict() -> dict[str, Any]:
            return {"state": {"moment": torch.tensor(float("nan"))}}

    class BadTraining(_FakeTraining):
        @staticmethod
        def build_optimizer_v1(_model: _FakeModel) -> BadOptimizer:
            return BadOptimizer()

    apis.training = BadTraining
    result = executor.run_authorized_engine_v1(
        authority=authority,
        reservation=reservation,
        environment_attestation=environment,
        repository_root=tmp_path,
        runtime_data_root=tmp_path,
        device="cuda:0",
        apis=apis,
    )
    assert result["status"] == "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME"
    assert result["stage"] == "initialize_model"
    assert result["last_completed_update"] == 0
