from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest
import torch

from scripts import (
    execute_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
    as v13,
)


ROOT = Path(__file__).resolve().parents[2]


def _future_authority() -> dict[str, object]:
    runtime_inputs = {
        name: {
            "path": f"bound/{name}",
            "file_sha256": f"{index + 1:064x}",
            "byte_count": index + 1,
        }
        for index, name in enumerate(v13.RUNTIME_INPUT_BINDING_NAMES)
    }
    return {
        "schema": f"{v13.SCHEMA_PREFIX}_future_execution_authority_v1",
        "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT",
        "preregistration_commit": v13.PREREGISTRATION_COMMIT,
        "frozen_source_and_review_commit": "a" * 40,
        "recursive_source_closure_manifest_sha256": "b" * 64,
        "independent_source_review_sha256": "e" * 64,
        "recursive_source_closure_reviewed": True,
        "exact_path_sha256_byte_count_review_passed": True,
        "custody_clean_export_exception_committed": True,
        "clean_export_certification_sha256": "c" * 64,
        "exported_paths_frozen_commit_validated": True,
        "execution_binding_commit": "d" * 40,
        "scientific_payload_authorized": True,
        "one_shot": True,
        "retry_authorized": False,
        "resume_authorized": False,
        "maximum_updates": 1_000,
        "maximum_presentations": 16_000,
        "output_root": v13.OUTPUT_ROOT_RELATIVE_PATH,
        "certified_source_root": "/certified/v13-source",
        "runtime_data_root": "/authorized/v13-runtime-data",
        "runtime_inputs": runtime_inputs,
        "authorized_roles": {
            "train": {"pairs": 4_262, "scenes": 72, "unique_endpoints": 7_777},
            "checkpoint_selection": {
                "pairs": 495,
                "scenes": 8,
                "unique_endpoints": 924,
            },
            "probability_calibration_open_count": 0,
        },
        "hardware": {
            "visible_device_count": 1,
            "name": "AMD Radeon AI PRO R9700",
            "total_memory_bytes": 34_208_743_424,
            "isolated_python": True,
        },
        "runtime": dict(v13.EXPECTED_RUNTIME_FINGERPRINT),
    }


def _physical_metrics() -> dict[str, object]:
    return {
        "pixel_first_hit_balanced_accuracy": 0.96,
        "ground_clear_balanced_accuracy": 0.96,
        "derived_raster_balanced_accuracy": 0.96,
        "wrong_rgb_pixel_balanced_accuracy_drop": 0.13,
        "wrong_rgb_depth_median_error_increase_m": 0.13,
        "wrong_rgb_depth_p95_error_increase_m": 0.21,
        "wrong_rgb_ground_balanced_accuracy_drop": 0.13,
        "wrong_rgb_raster_nll_increase": 0.13,
        "wrong_rgb_raster_balanced_accuracy_drop": 0.13,
        "depth_median_error_m": 0.09,
        "depth_p95_error_m": 0.24,
        "derived_raster_nll": 0.14,
        "distance_group_balanced_accuracy": [0.93] * 6,
        "present_class_recall": {
            "free": 0.96,
            "occupied": 0.96,
            "unknown": 0.96,
        },
    }


def _summary(
    *,
    passed: int,
    shortfall: float,
    pixel: float,
    ground: float,
    depth: float,
    complete: int = 0,
) -> dict[str, object]:
    return {
        "complete_physical_scope_count": complete,
        "margin_count": 189,
        "passed_margin_count": passed,
        "total_shortfall": shortfall,
        "worst_margin": -0.5 if shortfall else 0.0,
        "rough_motion": {
            "pixel_balanced_accuracy": pixel,
            "ground_balanced_accuracy": ground,
            "depth_p95_m": depth,
        },
    }


def _positive_controls() -> dict[str, dict[str, bool]]:
    return {
        name: {check: True for check in v13.CONTROL_CHECK_NAMES}
        for name in v13.CONTROL_NAMES
    }


def _weak_physical_metrics() -> dict[str, object]:
    metrics = _physical_metrics()
    metrics.update(
        {
            "pixel_first_hit_balanced_accuracy": 0.70,
            "ground_clear_balanced_accuracy": 0.60,
            "derived_raster_balanced_accuracy": 0.70,
            "wrong_rgb_pixel_balanced_accuracy_drop": 0.01,
            "wrong_rgb_depth_median_error_increase_m": 0.01,
            "wrong_rgb_depth_p95_error_increase_m": 0.01,
            "wrong_rgb_ground_balanced_accuracy_drop": 0.01,
            "wrong_rgb_raster_nll_increase": 0.01,
            "wrong_rgb_raster_balanced_accuracy_drop": 0.01,
            "depth_median_error_m": 0.80,
            "depth_p95_error_m": 1.10,
            "derived_raster_nll": 0.50,
            "distance_group_balanced_accuracy": [0.70] * 6,
            "present_class_recall": {
                "free": 0.70,
                "occupied": 0.70,
                "unknown": 0.70,
            },
        }
    )
    return metrics


@dataclass(frozen=True)
class _FakeConfig:
    architecture_version: int = 13


class _FakeParameter:
    def __init__(self, count: int, *, requires_grad: bool = True) -> None:
        self._count = count
        self.requires_grad = requires_grad
        self.grad = None

    def numel(self) -> int:
        return self._count


class _FakeTargetModule:
    training = False

    def __init__(self) -> None:
        self._parameter = _FakeParameter(
            v13.MODEL_REQUIRED_CONSTANTS["TARGET_BOTTLENECK_PARAMETER_COUNT_V13"],
            requires_grad=False,
        )

    def parameters(self) -> tuple[_FakeParameter, ...]:
        return (self._parameter,)


class GeometryAnchoredSweptProgressSurvivalJointJepaV13:
    def __init__(self) -> None:
        self.config = _FakeConfig()
        self.target_hard_sync_count = torch.tensor(1, dtype=torch.long)
        self.ema_update_count = torch.tensor(0, dtype=torch.long)
        self._target = _FakeTargetModule()
        self.target_collapse_update: int | None = None
        self._groups = tuple(
            ((name, _FakeParameter(count)),)
            for name, count in (
                ("encoder.weight", 3_105_513),
                ("semantic_head.weight", 22_020),
                ("predictor.weight", 259_073),
            )
        )
        self.bev_lift = SimpleNamespace(
            evidence_head=SimpleNamespace(
                ground_query_geometry=self._ground_query_geometry
            )
        )

    @staticmethod
    def _ground_query_geometry(
        camera_origin_body_m: torch.Tensor,
        _camera_basis_body_fru: torch.Tensor,
        _ground_plane_z_body_m: torch.Tensor,
    ) -> SimpleNamespace:
        batch = int(camera_origin_body_m.shape[0])
        return SimpleNamespace(
            in_frustum=torch.ones((batch, 1, 1, 1), dtype=torch.bool),
            target_distance_m=torch.ones((batch, 1, 1, 1)),
        )

    @staticmethod
    def _latent(rgb: torch.Tensor) -> torch.Tensor:
        pattern = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        mean = rgb.float().mean(dim=(1, 2, 3), keepdim=True)
        return pattern + mean

    @staticmethod
    def _nominal_evidence(batch: int) -> SimpleNamespace:
        scalar = torch.zeros((batch, 1, 1, 1))
        return SimpleNamespace(
            pixel_first_hit_hazard_logits=scalar,
            pixel_within_bin_offset_m=scalar,
            ground_clear_to_target_logits=scalar,
            ground_query_in_frustum=torch.ones_like(scalar, dtype=torch.bool),
            ground_query_uv_px=torch.zeros((batch, 1, 1, 1, 2)),
            ground_target_distance_m=scalar,
        )

    def encode_online_with_evidence(self, rgb: torch.Tensor) -> SimpleNamespace:
        return SimpleNamespace(
            latent=self._latent(rgb),
            nominal_evidence=self._nominal_evidence(int(rgb.shape[0])),
        )

    def encode_online_with_auxiliary_evidence(
        self,
        rgb: torch.Tensor,
        **_geometry: torch.Tensor,
    ) -> SimpleNamespace:
        return self.encode_online_with_evidence(rgb)

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        return self._latent(rgb)

    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        if (
            self.target_collapse_update is not None
            and int(self.ema_update_count.item()) >= self.target_collapse_update
        ):
            return torch.zeros_like(self._latent(rgb)).detach()
        return self._latent(rgb).detach()

    def encode_online_training(self, rgb: torch.Tensor, **_kwargs: object) -> torch.Tensor:
        return self.encode_online(rgb)

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.cat((latent, -latent, torch.zeros_like(latent)), dim=1)

    def online_state(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.encode_online(rgb)

    def predict_all_actions_with_survival(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        return torch.zeros(1)

    def update_target_ema_after_optimizer_step(self) -> None:
        self.ema_update_count.add_(1)

    def trainable_parameter_groups_v13(self) -> tuple[tuple[tuple[str, _FakeParameter], ...], ...]:
        return self._groups

    def target_modules(self) -> tuple[_FakeTargetModule, ...]:
        return (self._target,)

    def named_parameters(self) -> tuple[tuple[str, _FakeParameter], ...]:
        return tuple(row for group in self._groups for row in group)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "online.synthetic": torch.tensor(
                [float(self.ema_update_count.item()), 13.0], dtype=torch.float32
            ),
            "target.synthetic": torch.tensor([1.0, 2.0], dtype=torch.float32),
        }


def _fake_model_module() -> SimpleNamespace:
    values: dict[str, object] = {
        v13.MODEL_CLASS_NAME: GeometryAnchoredSweptProgressSurvivalJointJepaV13,
        **v13.MODEL_REQUIRED_CONSTANTS,
        "SHARED_PARAMETER_PREFIXES_V13": ("encoder.", "bev_lift.evidence_head."),
        "REPRESENTATION_PARAMETER_PREFIXES_V13": (
            "bev_lift.free_projection.",
            "bev_lift.occupied_projection.",
            "semantic_head.",
        ),
        "PREDICTOR_PARAMETER_PREFIXES_V13": ("predictor.",),
        "TARGET_PARAMETER_PREFIXES_V13": (
            "target_encoder.",
            "target_bev_lift.evidence_head.",
            "target_bev_lift.free_projection.",
            "target_bev_lift.occupied_projection.",
        ),
    }
    return SimpleNamespace(**values)


def _fake_training_module() -> SimpleNamespace:
    def update(
        model: GeometryAnchoredSweptProgressSurvivalJointJepaV13,
        _optimizer: object,
        _microbatches: object,
        *,
        accounting: object,
    ) -> SimpleNamespace:
        del accounting
        model.update_target_ema_after_optimizer_step()
        current = int(model.ema_update_count.item())
        route = {
            "preclip_l2": 1.0,
            "applied_scale": 1.0,
            "parameter_tensor_count": 1,
            "absent_tensor_gradient_count": 0,
        }
        return SimpleNamespace(
            accounting={
                name: current * multiplier
                for name, multiplier in v13.ACCOUNTING_MULTIPLIERS.items()
            },
            mean_losses={
                "S": 1.0,
                "P": 1.0,
                "U": 1.0,
                "R": 1.0,
                "O": 0.5,
                "N": 4.5,
                "C": 0.25,
                "L": 4.75,
            },
            gradient_routes={
                name: dict(route)
                for name in (
                    "camera_shared",
                    "joint_shared",
                    "representation",
                    "predictor",
                )
            },
            ranking_active_microbatches=4,
            ranking_eligible_pairs=16,
            survival_supervised_decisions=16,
            target_gradient_tensor_count=0,
            optimizer_steps_this_update=1,
            ema_steps_this_update=1,
        )

    no_op = lambda *_args, **_kwargs: None
    return SimpleNamespace(
        MICROBATCH_SIZE=4,
        MICROBATCHES_PER_UPDATE=4,
        PRESENTATIONS_PER_UPDATE=16,
        MAXIMUM_UPDATES=1_000,
        MAXIMUM_PRESENTATIONS=16_000,
        REQUIRED_BATCH_KEYS=v13.TRAINING_REQUIRED_BATCH_KEYS,
        partition_parameters_v13=no_op,
        build_frozen_optimizer_v13=no_op,
        validate_optimizer_v13=no_op,
        joint_training_update_v13=update,
        validate_accounting_v13=no_op,
        _validate_microbatches_v13=no_op,
    )


class _FakeRuntime:
    torch = torch
    train_pair_count = 4_262

    def __init__(self, schedule: list[int], *, scenario: str) -> None:
        self.schedule = schedule
        self.scenario = scenario
        self.model_module = _fake_model_module()
        self.training_module = _fake_training_module()
        self.model = GeometryAnchoredSweptProgressSurvivalJointJepaV13()
        if scenario == "collapsed_target":
            self.model.target_collapse_update = 400
        self.initialize_calls = 0
        self.access_calls = 0
        self.terminal_access_calls = 0
        self.close_calls = 0
        self.observed_updates: list[int] = []

    def initialize_model_v13(self) -> tuple[object, object, dict[str, object]]:
        self.initialize_calls += 1
        return self.model, object(), {
            "n320_gate_open_count": 1,
            "n320_checkpoint_open_count": 1,
            "n320_gate_passed": True,
            "payload_access_after_reservation": True,
            "probability_calibration_open_count": 0,
            "constructor_initialization_seed": v13.CONSTRUCTOR_INITIALIZATION_SEED,
            "projection_initialization_seed": v13.PROJECTION_INITIALIZATION_SEED,
        }

    def structural_probe_inputs_v13(self) -> dict[str, torch.Tensor]:
        return {
            "rgb": torch.zeros((1, 3, 2, 2)),
            "wrong_rgb": torch.ones((1, 3, 2, 2)),
            "camera_origin_a": torch.zeros((1, 3)),
            "camera_origin_b": torch.ones((1, 3)),
            "camera_basis": torch.eye(3)[None],
            "ground_plane_z": torch.zeros(1),
        }

    def build_microbatches_v13(
        self, indices: list[int], *, update: int
    ) -> list[dict[str, torch.Tensor]]:
        assert len(indices) == 16
        assert update == int(self.model.ema_update_count.item()) + 1
        batches = []
        for _ in range(4):
            batch = {
                name: torch.zeros(4) for name in v13.TRAINING_REQUIRED_BATCH_KEYS
            }
            for prefix in ("current", "next"):
                batch[f"{prefix}_rgb"] = torch.zeros((4, 3, 2, 2))
                batch[f"{prefix}_camera_origin_body_m"] = torch.zeros((4, 3))
                batch[f"{prefix}_camera_basis_body_fru"] = torch.eye(3).repeat(4, 1, 1)
                batch[f"{prefix}_ground_plane_z_body_m"] = torch.zeros(4)
                batch[f"{prefix}_ground_support_in_frustum"] = torch.ones(
                    (4, 1, 1, 1), dtype=torch.bool
                )
            batches.append(batch)
        return batches

    def observe_v13(
        self,
        _model: object,
        *,
        update: int,
        physical_endpoint_updater: object,
    ) -> dict[str, object]:
        assert physical_endpoint_updater is v13.update_physical_accumulator_from_rgb_v13
        self.observed_updates.append(update)
        weak = update in (0, 100) or (self.scenario == "stop400" and update == 400)
        metrics = _weak_physical_metrics() if weak else _physical_metrics()
        checks = {name: True for name in v13.V12_GATE_CHECK_NAMES}
        if self.scenario == "fail1000" and update == 1_000:
            checks[v13.V12_GATE_CHECK_NAMES[0]] = False
        provenance = {
            "target_endpoint_count": 924,
            "matched_nominal_call_count": 924,
            "wrong_nominal_call_count": 924,
            "qualifying_updater_call_count": 1_848,
            "qualifying_updater_name": "update_physical_accumulator_from_rgb_v13",
            "auxiliary_logits_used": False,
            "old_camera_raster_used": False,
            "target_query_identity_pass": True,
            "wrong_rgb_dependence_nonzero": True,
        }
        if self.scenario == "bad_provenance":
            provenance["qualifying_updater_call_count"] = 1_847
        return {
            "physical_scopes": {
                scope: dict(metrics) for scope in v13.SCOPES
            },
            "v12_gate": {"passed": all(checks.values()), "checks": checks},
            "controls": _positive_controls(),
            "physical_provenance": provenance,
        }

    def access_receipt_v13(self) -> dict[str, object]:
        self.access_calls += 1
        return {
            "forbidden_input_count": 0,
            "probability_calibration_open_count": 0,
            "opened_roles": (
                "authority",
                "index",
                "train",
                "checkpoint_selection",
            ),
            "receipt_kind": "lightweight_in_memory",
        }

    def terminal_access_receipt_v13(self) -> dict[str, object]:
        self.terminal_access_calls += 1
        records_sha256 = "f" * 64
        certified_bindings = [
            {
                "path": "scripts/runtime.py",
                "file_sha256": "1" * 64,
                "byte_count": 1,
            },
            {
                "path": "lewm/model.py",
                "file_sha256": "2" * 64,
                "byte_count": 2,
            },
        ]
        bound_parent_sources = {
            "validated_path_count": len(v13.BOUND_PARENT_SOURCES),
            "execution_authority_granted": False,
        }
        return {
            "forbidden_input_count": 0,
            "probability_calibration_open_count": 0,
            "opened_roles": (
                "authority",
                "index",
                "train",
                "checkpoint_selection",
            ),
            "receipt_kind": "complete_rehash",
            "terminal_full_rehash_count": self.terminal_access_calls,
            "raw_consumed_inputs_rehashed": True,
            "raw_consumed_file_rehash_count": 12,
            "raw_consumed_records_sha256": records_sha256,
            "label_source_rehash_count": 3,
            "label_sources_rehashed": [
                "labels/manifest.json",
                "labels/train.jsonl",
                "labels/checkpoint_selection.jsonl",
            ],
            "bound_parent_source_rehash_count": len(v13.BOUND_PARENT_SOURCES),
            "bound_parent_sources": bound_parent_sources,
            "certified_source_rehash_count": len(certified_bindings),
            "certified_source_bindings_sha256": v13._canonical_value_sha256(
                certified_bindings
            ),
            "certified_source_bindings": certified_bindings,
            "all_consumed_inputs_rehashed": True,
            "source_root": "/certified/v13-source",
            "runtime_data_root": "/authorized/v13-runtime-data",
            "runtime_fingerprint": dict(v13.EXPECTED_RUNTIME_FINGERPRINT),
            "raw_inputs": {
                "unique_file_count": 12,
                "records_sha256": records_sha256,
                "all_consumed_files_rehashed": True,
            },
        }

    def close_v13(self) -> None:
        self.close_calls += 1


class _MemoryPublisher:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    @staticmethod
    def _binding(path: str, raw: bytes) -> dict[str, object]:
        return {
            "path": path,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }

    def publish_json(
        self, relative_path: str, core: dict[str, object]
    ) -> dict[str, object]:
        if relative_path in self.files:
            raise FileExistsError(relative_path)
        value = v13._content_bound(core)
        raw = v13._canonical_json_bytes(value) + b"\n"
        self.files[relative_path] = raw
        return {"value": value, "binding": self._binding(relative_path, raw)}

    def publish_bytes(self, relative_path: str, raw: bytes) -> dict[str, object]:
        if relative_path in self.files:
            raise FileExistsError(relative_path)
        self.files[relative_path] = raw
        return self._binding(relative_path, raw)


def _frozen_schedule(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    schedule = [index % 4_262 for index in range(16_000)]
    monkeypatch.setattr(
        v13,
        "CHECKPOINT_SCHEDULE_PREFIX_SHA256",
        {
            update: v13._canonical_value_sha256(schedule[: update * 16])
            for update in (100, 400, 1_000)
        },
    )
    return schedule


def _reservation(authority: dict[str, object]) -> dict[str, object]:
    return v13._content_bound(
        {
            "schema": f"{v13.SCHEMA_PREFIX}_attempt_reservation_v1",
            "status": "RESERVED_BEFORE_SCIENTIFIC_PAYLOAD",
            "created_utc": "2026-07-29T12:00:00Z",
            "authority_sha256": hashlib.sha256(
                v13._canonical_json_bytes(authority)
            ).hexdigest(),
            "one_shot_attempt": 1,
            "attempt_consumed": True,
            "maximum_updates": 1_000,
            "maximum_presentations": 16_000,
            "retry_authorized": False,
            "resume_authorized": False,
        }
    )


def test_source_shell_is_denied_and_parent_bindings_validate(capsys: pytest.CaptureFixture[str]) -> None:
    assert v13.CURRENT_EXECUTION_AUTHORIZED is False
    with pytest.raises(PermissionError, match="source closure"):
        v13.execute_v13()
    assert v13.main([]) == 4
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["status"] == "DENIED_SOURCE_ONLY"
    assert receipt["scientific_payload_opened"] is False
    assert receipt["reservation_created"] is False
    assert receipt["attempt_consumed"] is False
    assert v13.validate_content_bound_v13(receipt) == receipt

    validation = v13.validate_bound_sources_v13(ROOT)
    assert validation["validated_path_count"] == len(v13.BOUND_PARENT_SOURCES)
    assert validation["execution_authority_granted"] is False


def test_model_and_training_public_apis_are_exact() -> None:
    from lewm.models import (
        geometry_anchored_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
        as model_module,
    )
    from scripts import (
        run_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck
        as training_module,
    )

    model = v13.validate_model_api_v13(model_module)
    training = v13.validate_training_api_v13(training_module)
    assert model["online_trainable_parameter_count"] == 3_386_606
    assert training["required_batch_key_count"] == 21
    assert training["presentations_per_update"] == 16


def test_future_authority_is_conjunctive_and_receipts_are_write_once(
    tmp_path: Path,
) -> None:
    authority = _future_authority()
    assert v13.validate_future_execution_prerequisites_v13(authority) == authority
    denied = dict(authority)
    denied["recursive_source_closure_reviewed"] = False
    with pytest.raises(PermissionError, match="conjunctively"):
        v13.validate_future_execution_prerequisites_v13(denied)
    with pytest.raises(PermissionError, match="forbidden role or input"):
        v13._validate_access_receipt_v13(
            {
                "forbidden_input_count": 0,
                "probability_calibration_open_count": 0,
                "opened_roles": ["authority", "index", "train"],
            },
            terminal=True,
        )

    export_root = tmp_path / "certified_export"
    export_root.mkdir()
    output = export_root / v13.OUTPUT_ROOT_RELATIVE_PATH
    output.parent.mkdir(parents=True)
    reservation = v13.reserve_attempt_v13(
        export_root,
        authority,
        created_utc="2026-07-29T12:00:00Z",
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "reservation.json").stat().st_mode) == 0o444
    assert reservation["attempt_consumed"] is True
    with pytest.raises(FileExistsError):
        v13.reserve_attempt_v13(
            export_root,
            authority,
            created_utc="2026-07-29T12:00:01Z",
        )

    failure = v13.terminalize_failure_v13(
        output,
        reservation,
        stage="synthetic_source_test",
        error=RuntimeError("synthetic detail is hash-bound, not published"),
        created_utc="2026-07-29T12:00:02Z",
    )
    assert failure["status"] == "FAIL_TERMINAL_NO_RETRY_NO_RESUME"
    assert "synthetic detail" not in (output / "failure.json").read_text()
    assert stat.S_IMODE((output / "failure.json").stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        v13.terminalize_failure_v13(
            output,
            reservation,
            stage="second_failure",
            error=RuntimeError("no retry"),
            created_utc="2026-07-29T12:00:03Z",
        )


def test_nominal_metric_adapter_changes_only_target_metadata() -> None:
    from lewm.benchmarks.go2_observable_camera_ray_fit_v4_metrics import (
        ObservableCameraRayFitV4MetricAccumulator,
    )
    from lewm.models.observable_camera_ray_evidence_v4 import (
        ObservableCameraRayEvidenceV4RawOutput,
    )
    from lewm.models.observable_camera_ray_evidence_v4_training import (
        ObservableCameraRayEvidenceV4Targets,
    )

    hazards = torch.tensor(
        [[[[5.0, -5.0]], [[-5.0, -5.0]]]], dtype=torch.float32
    )
    offsets = torch.zeros_like(hazards)
    ground_logits = torch.tensor([[[[2.0, -2.0, 2.0, -2.0, 2.0]]]])
    original_validity = torch.zeros((1, 1, 1, 5), dtype=torch.bool)
    original_uv = torch.zeros((1, 1, 1, 5, 2), dtype=torch.float32)
    original_distance = torch.zeros((1, 1, 1, 5), dtype=torch.float32)
    nominal = ObservableCameraRayEvidenceV4RawOutput(
        pixel_first_hit_hazard_logits=hazards,
        pixel_within_bin_offset_m=offsets,
        ground_clear_to_target_logits=ground_logits,
        ground_query_in_frustum=original_validity,
        ground_query_uv_px=original_uv,
        ground_target_distance_m=original_distance,
    )
    target_validity = torch.tensor([[[[True, True, True, True, True]]]])
    target_distance = torch.tensor([[[[0.5, 1.5, 2.5, 3.5, 4.5]]]])
    adapted = v13.adapt_nominal_logits_with_target_metadata_v13(
        nominal,
        target_ground_in_frustum=target_validity,
        target_ground_distance_m=target_distance,
    )
    assert adapted.pixel_first_hit_hazard_logits is hazards
    assert adapted.pixel_within_bin_offset_m is offsets
    assert adapted.ground_clear_to_target_logits is ground_logits
    assert adapted.ground_query_uv_px is original_uv
    assert adapted.ground_query_in_frustum is target_validity
    assert adapted.ground_target_distance_m is target_distance

    targets = ObservableCameraRayEvidenceV4Targets(
        pixel_in_range_hit_mask=torch.tensor([[[True, False]]]),
        pixel_no_hit_mask=torch.tensor([[[False, True]]]),
        pixel_hit_bin_index=torch.tensor([[[0, 0]]], dtype=torch.long),
        pixel_within_bin_offset_m=torch.zeros((1, 1, 2)),
        ground_in_frustum=target_validity,
        ground_clear_to_target=torch.tensor([[[[True, False, True, False, True]]]]),
    )
    probabilities = torch.tensor(
        [[[[0.8, 0.1], [0.1, 0.1]], [[0.1, 0.8], [0.1, 0.1]], [[0.1, 0.1], [0.8, 0.8]]]]
    )
    labels = torch.tensor([[[0, 1], [2, 2]]], dtype=torch.long)

    class MetricModel:
        def __init__(self) -> None:
            self.calls = 0
            self.bev_lift = SimpleNamespace(
                evidence_head=SimpleNamespace(
                    ground_query_geometry=lambda *_args: SimpleNamespace(
                        in_frustum=target_validity,
                        target_distance_m=target_distance,
                    )
                )
            )

        def encode_online_with_evidence(self, _rgb: torch.Tensor) -> SimpleNamespace:
            self.calls += 1
            return SimpleNamespace(
                nominal_evidence=nominal,
                latent=torch.zeros((1, 1, 2, 2)),
            )

        def semantic_logits_from_latent(self, _latent: torch.Tensor) -> torch.Tensor:
            return probabilities.log()

    model = MetricModel()
    accumulator = ObservableCameraRayFitV4MetricAccumulator()
    provenance = v13.update_physical_accumulator_from_rgb_v13(
        model,
        accumulator,
        selected_rgb=torch.zeros((1, 3, 2, 2)),
        target_camera_origin_body_m=torch.zeros((1, 3)),
        target_camera_basis_body_fru=torch.eye(3)[None],
        target_ground_plane_z_body_m=torch.zeros(1),
        targets=targets,
        target_raster_labels=labels,
        families=("synthetic",),
    )
    assert model.calls == 1
    assert provenance["auxiliary_logits_used"] is False
    assert provenance["old_camera_raster_used"] is False
    finalized = accumulator.finalize()
    assert finalized["frame_count"] == 1
    assert finalized["derived_raster"]["balanced_accuracy"] == 1.0


def test_exact_189_margin_evaluator_and_wrong_rgb_rotation() -> None:
    scopes = {scope: _physical_metrics() for scope in v13.SCOPES}
    evaluated = v13.evaluate_physical_scopes_v13(scopes)
    assert evaluated["margin_count"] == 189
    assert evaluated["passed_margin_count"] == 189
    assert evaluated["complete_physical_scope_count"] == 9
    assert evaluated["total_shortfall"] == 0.0

    malformed = _physical_metrics()
    malformed["distance_group_balanced_accuracy"] = [0.93] * 5
    with pytest.raises(ValueError, match="groups changed"):
        v13.physical_margins_v13(malformed)

    endpoints = [
        {
            "endpoint_sha256": f"{index:064x}",
            "family": v13.REGISTERED_FAMILIES[index % len(v13.REGISTERED_FAMILIES)],
        }
        for index in range(924)
    ]
    mapping = v13.registered_wrong_rgb_mapping_v13(endpoints)
    assert len(mapping) == 924
    first_family_ids = sorted(
        row["endpoint_sha256"]
        for row in endpoints
        if row["family"] == v13.REGISTERED_FAMILIES[0]
    )
    assert mapping[first_family_ids[0]] == first_family_ids[1]
    assert mapping[first_family_ids[-1]] == first_family_ids[0]


def test_update400_gate_requires_directional_improvement_and_all_controls() -> None:
    before = _summary(passed=90, shortfall=40.0, pixel=0.70, ground=0.60, depth=1.2)
    after = _summary(passed=100, shortfall=30.0, pixel=0.71, ground=0.61, depth=1.2)
    controls = _positive_controls()
    decision = v13.evaluate_update400_gate_v13(
        before, after, controls, integrity_pass=True
    )
    assert decision["passed"] is True
    assert len(decision["causal_control_checks"]) == 12
    assert decision["next_update"] == 1_000

    controls["wrong_rgb"]["positive_bootstrap_lower_95"] = False
    decision = v13.evaluate_update400_gate_v13(
        before, after, controls, integrity_pass=True
    )
    assert decision["passed"] is False
    assert decision["action"] == "FAIL_TERMINAL_NO_RETRY_NO_RESUME"

    equal_count = _summary(
        passed=90, shortfall=30.0, pixel=0.71, ground=0.61, depth=1.2
    )
    assert v13.evaluate_update400_gate_v13(
        before, equal_count, _positive_controls(), integrity_pass=True
    )["passed"] is False
    assert v13.evaluate_update400_gate_v13(
        before, after, _positive_controls(), integrity_pass=False
    )["passed"] is False


def test_final_gate_has_unchanged_24_checks_and_strict_physical_thresholds() -> None:
    v12_gate = {
        "passed": True,
        "checks": {name: True for name in v13.V12_GATE_CHECK_NAMES},
    }
    passing = _summary(
        passed=112,
        shortfall=33.0,
        complete=1,
        pixel=0.82,
        ground=0.65,
        depth=0.97,
    )
    decision = v13.evaluate_final_gate_v13(
        v12_gate, passing, integrity_pass=True
    )
    assert decision["passed"] is True
    assert decision["physical_adapter_preregistration_eligible"] is True
    assert decision["probability_calibration_authorized"] is False
    assert decision["g2_authorized"] is False

    equality = dict(passing)
    equality["total_shortfall"] = 33.05143763708337
    assert v13.evaluate_final_gate_v13(
        v12_gate, equality, integrity_pass=True
    )["passed"] is False
    broken_v12 = {"passed": False, "checks": dict(v12_gate["checks"])}
    broken_v12["checks"][v13.V12_GATE_CHECK_NAMES[0]] = False
    assert v13.evaluate_final_gate_v13(
        broken_v12, passing, integrity_pass=True
    )["passed"] is False
    assert v13.evaluate_final_gate_v13(
        v12_gate, passing, integrity_pass=False
    )["passed"] is False


@pytest.mark.parametrize("terminal_update", [400, 1_000])
def test_terminal_accounting_is_exact_and_capped(terminal_update: int) -> None:
    accounting = {
        name: terminal_update * multiplier
        for name, multiplier in v13.ACCOUNTING_MULTIPLIERS.items()
    }
    assert v13.validate_terminal_accounting_v13(
        accounting, terminal_update=terminal_update
    ) == accounting
    accounting["ema_steps"] -= 1
    with pytest.raises(RuntimeError, match="inconsistent"):
        v13.validate_terminal_accounting_v13(
            accounting, terminal_update=terminal_update
        )


def test_engine_stops_once_at_update400_without_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _future_authority()
    runtime = _FakeRuntime(_frozen_schedule(monkeypatch), scenario="stop400")
    publisher = _MemoryPublisher()
    result = v13.run_future_authorized_engine_v13(
        authority=authority,
        reservation=_reservation(authority),
        runtime=runtime,
        publisher=publisher,
    )
    assert result["status"] == "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"
    assert result["terminal_update"] == 400
    assert result["checkpoint_published"] is False
    assert runtime.observed_updates == [0, 100, 400]
    assert runtime.access_calls == 401
    assert runtime.terminal_access_calls == 1
    assert runtime.close_calls == 1
    assert v13.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH not in publisher.files
    assert v13.SUCCESS_RELATIVE_PATH not in publisher.files
    assert v13.SCIENTIFIC_FAILURE_RELATIVE_PATH in publisher.files
    assert v13.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH in publisher.files


def test_engine_rejects_post_initialization_collapsed_ema_target_at_update400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _future_authority()
    runtime = _FakeRuntime(_frozen_schedule(monkeypatch), scenario="collapsed_target")
    publisher = _MemoryPublisher()
    result = v13.run_future_authorized_engine_v13(
        authority=authority,
        reservation=_reservation(authority),
        runtime=runtime,
        publisher=publisher,
    )
    assert result["status"] == "FAIL_SCIENTIFIC_UPDATE400_GATE_TERMINAL"
    assert runtime.observed_updates == [0, 100, 400]
    update400 = json.loads(publisher.files[v13.METRIC_RELATIVE_PATHS[400]])
    assert update400["target_integrity"]["passed"] is False
    assert (
        update400["target_integrity"]["checks"]["target_is_noncollapsed"]
        is False
    )
    assert update400["integrity_pass"] is False
    assert v13.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH not in publisher.files
    assert v13.SUCCESS_RELATIVE_PATH not in publisher.files
    assert v13.SCIENTIFIC_FAILURE_RELATIVE_PATH in publisher.files
    access = json.loads(
        publisher.files[v13.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH]
    )
    assert access["receipt"]["certified_source_rehash_count"] == 2


def test_engine_passes_update1000_and_publishes_only_bound_development_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _future_authority()
    runtime = _FakeRuntime(_frozen_schedule(monkeypatch), scenario="pass1000")
    publisher = _MemoryPublisher()
    result = v13.run_future_authorized_engine_v13(
        authority=authority,
        reservation=_reservation(authority),
        runtime=runtime,
        publisher=publisher,
    )
    assert result["status"] == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL"
    assert result["terminal_update"] == 1_000
    assert result["decision"]["passed"] is True
    assert result["probability_calibration_authorized"] is False
    assert result["g2_authorized"] is False
    assert runtime.observed_updates == [0, 100, 400, 1_000]
    assert runtime.access_calls == 1_001
    assert runtime.terminal_access_calls == 1
    assert runtime.close_calls == 1
    assert v13.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH in publisher.files
    assert v13.DEVELOPMENT_CHECKPOINT_BINDING_RELATIVE_PATH in publisher.files
    assert v13.SUCCESS_RELATIVE_PATH in publisher.files
    assert v13.SCIENTIFIC_FAILURE_RELATIVE_PATH not in publisher.files
    assert v13.TERMINAL_ACCESS_RECEIPT_RELATIVE_PATH in publisher.files
    trace = publisher.files[v13.TRACE_RELATIVE_PATH].decode("utf-8").splitlines()
    assert json.loads(trace[-1])["event"] == "update1000_final_gate"


def test_engine_update1000_scientific_failure_never_serializes_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _future_authority()
    runtime = _FakeRuntime(_frozen_schedule(monkeypatch), scenario="fail1000")
    publisher = _MemoryPublisher()
    result = v13.run_future_authorized_engine_v13(
        authority=authority,
        reservation=_reservation(authority),
        runtime=runtime,
        publisher=publisher,
    )
    assert result["status"] == "FAIL_SCIENTIFIC_UPDATE1000_GATE_TERMINAL"
    assert result["terminal_update"] == 1_000
    assert result["decision"]["passed"] is False
    assert runtime.observed_updates == [0, 100, 400, 1_000]
    assert runtime.terminal_access_calls == 1
    assert v13.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH not in publisher.files
    assert v13.SUCCESS_RELATIVE_PATH not in publisher.files


def test_engine_schedule_mismatch_fails_before_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _future_authority()
    schedule = _frozen_schedule(monkeypatch)
    schedule[0] = 4_261
    runtime = _FakeRuntime(schedule, scenario="pass1000")
    publisher = _MemoryPublisher()
    result = v13.run_future_authorized_engine_v13(
        authority=authority,
        reservation=_reservation(authority),
        runtime=runtime,
        publisher=publisher,
    )
    assert result["status"] == "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME"
    assert result["stage"] == "validate_deferred_runtime_and_schedule"
    assert runtime.initialize_calls == 0
    assert runtime.observed_updates == []
    assert v13.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH not in publisher.files


def test_engine_rejects_unverifiable_physical_provenance_at_update0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _future_authority()
    runtime = _FakeRuntime(_frozen_schedule(monkeypatch), scenario="bad_provenance")
    publisher = _MemoryPublisher()
    result = v13.run_future_authorized_engine_v13(
        authority=authority,
        reservation=_reservation(authority),
        runtime=runtime,
        publisher=publisher,
    )
    assert result["status"] == "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME"
    assert result["stage"] == "observe_update_0"
    assert runtime.observed_updates == [0]
    assert v13.DEVELOPMENT_CHECKPOINT_RELATIVE_PATH not in publisher.files
    assert v13.METRIC_RELATIVE_PATHS[0] not in publisher.files


def test_executor_source_has_no_runtime_discovery_or_accelerator_path() -> None:
    source = (
        ROOT
        / "scripts/execute_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck.py"
    ).read_text()
    for forbidden in (
        "torch.cuda",
        "torch.load",
        "os.walk",
        ".rglob(",
        ".glob(",
        "SoftObservableCameraRayRasterV4(",
    ):
        assert forbidden not in source
