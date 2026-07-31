from __future__ import annotations

import copy
from pathlib import Path
import stat
from types import SimpleNamespace

import pytest
import torch

from scripts import (
    execute_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1 as executor,
)


def _authority() -> dict:
    return executor._content_bound(
        {
            "schema": f"{executor.SCHEMA_PREFIX}_future_execution_authority_v1",
            "status": "AUTHORIZED_CERTIFIED_NARROW_EXPORT_ONE_SHOT",
            "scientific_payload_authorized": True,
            "one_shot": True,
            "maximum_updates": 1_000,
            "maximum_presentations": 16_000,
            "retry_authorized": False,
            "resume_authorized": False,
            "preregistration_commit": executor.PREREGISTRATION_COMMIT,
            "certified_source_root": executor.CERTIFIED_SOURCE_ROOT,
            "output_root": executor.OUTPUT_ROOT_RELATIVE_PATH,
            "rgb_root_relative_path": executor.RGB_ROOT_RELATIVE_PATH,
            "output_root_absent_at_authorization": True,
            "device": "cuda:0",
            "runtime_data_root": "/home/andrewknowles/Workspace/LeWMQuad-v3",
            "selectors": {
                "executor_module": executor.__name__,
                "model_module": executor.MODEL_MODULE_NAME,
                "model_class": executor.MODEL_CLASS_NAME,
                "training_module": executor.TRAINING_MODULE_NAME,
                "evaluation_module": executor.EVALUATION_MODULE_NAME,
            },
            "clean_export_certification": {
                "path": executor.CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
                "file_sha256": "a" * 64,
                "content_sha256": "b" * 64,
                "byte_count": 1,
            },
            "runtime_inputs": {
                name: dict(value)
                for name, value in executor.RUNTIME_INPUT_BINDINGS.items()
            },
        }
    )


def _raw_observation(
    update: int,
    *,
    ratio: float,
    positive_families: int,
    health_scale: float = 1.0,
) -> dict:
    controls = {
        name: {
            "correct_macro_mean": ratio,
            "control_macro_mean": 1.0,
            "primary_ratio": ratio,
            "advantage_macro_mean": 1.0 - ratio,
            "advantage_bootstrap_lower_95": 0.01,
            "positive_family_count": positive_families,
        }
        for name in executor.CONTROL_NAMES
    }
    health = {
        branch: {
            "image_count": 2_048,
            "token_count": 256,
            "feature_dimension": 192,
            "effective_rank": 10.0 * health_scale,
            "cross_sample_variance": 2.0 * health_scale,
            "within_image_spatial_diversity": 3.0 * health_scale,
            "finite": True,
        }
        for branch in ("online", "target")
    }
    return {
        "schema": f"{executor.SCHEMA_PREFIX}_checkpoint_evaluation_v1",
        "update": update,
        "controls": controls,
        "raw_health": health,
        "place": {
            "retrieval": {
                "chance_multiple": 2.0,
                "scene_count_at_least_1_5x_chance": 6,
            },
            "target_place_key_effective_rank": 3.0,
        },
        "access": {"h6_current_rgb": {"open_count": 6_144}},
        "integrity": {"checks": {"target_frozen": True}, "passed": True},
    }


class _FakeModel(torch.nn.Module):
    def __init__(self, _state: object) -> None:
        super().__init__()
        self.encoder = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.encoder.weight.fill_(1.0)
        self.predictor = torch.nn.Parameter(torch.ones(()))
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))
        self.target_encoder.eval()

    def train(self, mode: bool = True) -> "_FakeModel":
        super().train(mode)
        self.target_encoder.eval()
        self.target_encoder.requires_grad_(False)
        return self


def _accounting(update: int) -> dict[str, int]:
    return {
        "updates": update,
        "presentations": 16 * update,
        "mask_rows": 16 * update,
        "online_frame_encodings": 16 * update,
        "ema_target_frame_encodings": 16 * update,
        "microbatch_graphs": 4 * update,
        "backward_calls": 4 * update,
        "global_gradient_clips": update,
        "optimizer_steps": update,
        "ema_steps": update,
    }


class _FakeTraining:
    @staticmethod
    def build_optimizer_v1(model: _FakeModel) -> object:
        return torch.optim.AdamW(
            [model.encoder.weight, model.predictor], lr=1.0e-4
        )

    @staticmethod
    def parameter_inventory_v1(_model: _FakeModel) -> dict:
        return {
            "target_optimizer_excluded": True,
            "encoder_tensor_count": 1,
            "predictor_tensor_count": 1,
            "target_tensor_count": 1,
        }

    @staticmethod
    def training_update_v1(
        model: _FakeModel,
        _optimizer: object,
        _rgb: object,
        rows: object,
        *,
        accounting: object,
    ) -> object:
        del accounting
        update = int(model.ema_update_count) + 1
        model.ema_update_count.add_(1)
        flattened = tuple(index for batch in rows for index in batch)
        assert flattened == tuple(range(16 * (update - 1), 16 * update))
        return SimpleNamespace(
            accounting=_accounting(update),
            mean_jepa_loss=1.0 / update,
            gradient_receipt={
                "sole_jepa_route": True,
                "all_gradient_receipts_finite": True,
            },
            target_gradient_tensor_count=0,
            optimizer_steps_this_update=1,
            ema_steps_this_update=1,
            row_indices_sha256=f"{update:064x}"[-64:],
            target_indices_sha256="b" * 64,
            visible_indices_sha256="c" * 64,
        )

    @staticmethod
    def checkpoint_payload_v1(
        model: _FakeModel, optimizer: object, accounting: object
    ) -> dict:
        return {
            "schema": f"{executor.SCHEMA_PREFIX}_checkpoint_v1",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "accounting": dict(accounting),
        }


class _FakeLoader:
    closed = False

    def close(self) -> None:
        self.closed = True


class _FakeRuntime:
    def __init__(self) -> None:
        self.loader = _FakeLoader()
        self.requested_updates: list[int] = []

    def train_rows_for_update(self, update: int) -> tuple[object, ...]:
        self.requested_updates.append(update)
        first = 16 * (update - 1)
        return tuple(
            SimpleNamespace(
                row_indices=tuple(range(first + 4 * batch, first + 4 * batch + 4)),
                rgb=torch.zeros(4, 3, 1, 1),
            )
            for batch in range(4)
        )


def _audit() -> dict:
    bindings = executor.RUNTIME_INPUT_BINDINGS
    return {
        "train": {
            key: bindings["h6_train_index"][key]
            for key in ("path", "file_sha256", "byte_count")
        },
        "validation": {
            key: bindings["h6_validation_index"][key]
            for key in ("path", "file_sha256", "byte_count")
        },
        "place": {
            "manifest_file_sha256": bindings["place_triplet_manifest"][
                "file_sha256"
            ],
            "index_file_sha256": bindings[
                "place_triplet_checkpoint_selection_index"
            ]["file_sha256"],
        },
        "future_rgb_tensor_count": 0,
        "action_tensor_count": 0,
    }


def _fake_apis(ratios: dict[int, tuple[float, int]]) -> object:
    runtime = _FakeRuntime()

    def evaluate(
        _model: object, _runtime: object, update: int, _device: object
    ) -> dict:
        ratio, families = ratios[update]
        return _raw_observation(
            update, ratio=ratio, positive_families=families
        )

    return SimpleNamespace(
        torch=torch,
        model_class=_FakeModel,
        training=_FakeTraining,
        load_n320=lambda *_args: (
            {"weight": torch.ones(1, 1)},
            {
                "gate_open_count": 1,
                "checkpoint_open_count": 1,
                "checkpoint_deserialize_count": 1,
                "passed": True,
            },
        ),
        open_runtime=lambda *_args, **_kwargs: (runtime, _audit()),
        evaluate=evaluate,
        runtime=runtime,
    )


def test_authority_and_absent_root_reservation_are_exact(
    tmp_path: Path,
) -> None:
    authority = _authority()
    assert executor.validate_future_execution_prerequisites_v1(authority) == authority
    reservation = executor.reserve_attempt_v1(
        tmp_path, authority, created_utc="2026-07-31T00:00:00Z"
    )
    output = tmp_path / executor.OUTPUT_ROOT_RELATIVE_PATH
    assert executor.validate_attempt_reservation_v1(reservation) == reservation
    assert stat.S_IMODE((output / "reservation.json").stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        executor.reserve_attempt_v1(
            tmp_path, authority, created_utc="2026-07-31T00:00:01Z"
        )
    changed = executor._content_bound({**authority, "device": "cpu"})
    with pytest.raises(PermissionError):
        executor.validate_future_execution_prerequisites_v1(changed)


def test_execute_wrapper_keeps_outputs_in_runtime_data_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "certified_source"
    data = tmp_path / "runtime_data"
    source.mkdir()
    data.mkdir()
    authority = {
        "certified_source_root": str(source),
        "runtime_data_root": str(data),
        "device": "cuda:0",
    }
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        executor,
        "validate_future_execution_prerequisites_v1",
        lambda value: dict(value),
    )

    def reserve(root: Path, _authority: object, *, created_utc: str) -> dict:
        calls["reserve_root"] = root
        assert created_utc.endswith("Z")
        return {"reservation": True}

    def run(**kwargs: object) -> dict:
        calls.update(kwargs)
        return {"status": "synthetic"}

    monkeypatch.setattr(executor, "reserve_attempt_v1", reserve)
    monkeypatch.setattr(executor, "run_authorized_engine_v1", run)
    assert executor.execute_authorized_v1(source, authority) == {
        "status": "synthetic"
    }
    assert calls["reserve_root"] == data.resolve()
    assert calls["repository_root"] == data.resolve()
    assert calls["runtime_data_root"] == data.resolve()


def test_gate_math_collapse_continuation_and_qualification() -> None:
    baseline = _raw_observation(0, ratio=1.10, positive_families=0)
    update250 = _raw_observation(250, ratio=1.05, positive_families=2)
    first = executor.evaluate_observation_gate_v1(
        update250, baseline, baseline
    )
    assert first["continue_training"] is True
    update500 = _raw_observation(500, ratio=1.0495, positive_families=2)
    stalled = executor.evaluate_observation_gate_v1(
        update500, baseline, update250
    )
    assert stalled["improves_from_preceding_observation"] is False
    assert stalled["continue_training"] is False
    collapsed = _raw_observation(
        250, ratio=0.8, positive_families=8, health_scale=0.249
    )
    collapse_gate = executor.evaluate_observation_gate_v1(
        collapsed, baseline, baseline
    )
    assert collapse_gate["catastrophic_representation_collapse"] is True
    qualified = _raw_observation(1_000, ratio=0.89, positive_families=6)
    pass_gate = executor.evaluate_observation_gate_v1(
        qualified, baseline, update500
    )
    assert pass_gate["perception_qualified"] is True


def test_encoder_only_n320_extraction_checks_tensor_manifest() -> None:
    state = {
        "encoder.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "evidence_head.weight": torch.ones(1, 2),
    }
    manifest = executor._tensor_manifest(torch, state)
    semantic = {
        "schema": "lewm_go2_observable_camera_ray_fit_v4_development_checkpoint_v2",
        "model_class": "ObservableCameraRayEvidenceV4Model",
        "state_manifest": manifest,
        "metadata": {"seed": 20_260_710},
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
    }
    content = executor.hashlib.sha256(
        executor._canonical_json_bytes(semantic)
    ).hexdigest()
    checkpoint = {
        **semantic,
        "state_dict": state,
        "content_sha256": content,
    }
    extracted = executor.extract_n320_encoder_state_v1(
        torch, checkpoint, expected_content_sha256=content
    )
    assert tuple(extracted) == ("weight",)
    assert torch.equal(extracted["weight"], state["encoder.weight"])
    checkpoint["authoritative"] = True
    with pytest.raises(PermissionError):
        executor.extract_n320_encoder_state_v1(
            torch, checkpoint, expected_content_sha256=content
        )


def test_fake_engine_runs_update_zero_then_exact_cap_and_qualifies(
    tmp_path: Path,
) -> None:
    authority = _authority()
    reservation = executor.reserve_attempt_v1(
        tmp_path, authority, created_utc="2026-07-31T00:00:00Z"
    )
    apis = _fake_apis(
        {
            0: (1.10, 0),
            250: (1.05, 2),
            500: (0.99, 4),
            750: (0.94, 5),
            1_000: (0.85, 6),
        }
    )
    result = executor.run_authorized_engine_v1(
        authority=authority,
        reservation=reservation,
        repository_root=tmp_path,
        runtime_data_root=Path(authority["runtime_data_root"]),
        device="cpu",
        apis=apis,
    )
    output = tmp_path / executor.OUTPUT_ROOT_RELATIVE_PATH
    assert result["status"] == "PASS_PERCEPTION_QUALIFIED"
    assert result["selected_update"] == 1_000
    assert result["accounting"]["presentations"] == 16_000
    assert apis.runtime.requested_updates == list(range(1, 1_001))
    assert apis.runtime.loader.closed is True
    assert len(result["checkpoints"]) == 4
    assert all(item["same_attempt_reopen_count"] == 0 for item in result["checkpoints"])
    assert (output / "metrics/update_0.json").is_file()
    assert (output / "snapshots/update_1000.pt").is_file()
    assert stat.S_IMODE((output / "success.json").stat().st_mode) == 0o444


def test_fake_engine_stops_at_nonimproving_update500(
    tmp_path: Path,
) -> None:
    authority = _authority()
    reservation = executor.reserve_attempt_v1(
        tmp_path, authority, created_utc="2026-07-31T00:00:00Z"
    )
    apis = _fake_apis(
        {
            0: (1.10, 0),
            250: (1.05, 2),
            500: (1.05, 2),
        }
    )
    result = executor.run_authorized_engine_v1(
        authority=authority,
        reservation=reservation,
        repository_root=tmp_path,
        runtime_data_root=Path(authority["runtime_data_root"]),
        device="cpu",
        apis=apis,
    )
    assert result["status"] == "FAIL_SCIENTIFIC_CONTINUATION_GATE_NOT_MET"
    assert result["terminal_update"] == 500
    assert result["accounting"]["presentations"] == 8_000
    assert len(result["checkpoints"]) == 2


def test_update_zero_exception_gets_complete_terminal_receipt(
    tmp_path: Path,
) -> None:
    authority = _authority()
    reservation = executor.reserve_attempt_v1(
        tmp_path, authority, created_utc="2026-07-31T00:00:00Z"
    )
    apis = _fake_apis({0: (1.0, 0)})

    def fail(*_args: object) -> dict:
        raise RuntimeError("synthetic update-zero failure")

    apis.evaluate = fail
    result = executor.run_authorized_engine_v1(
        authority=authority,
        reservation=reservation,
        repository_root=tmp_path,
        runtime_data_root=Path(authority["runtime_data_root"]),
        device="cpu",
        apis=apis,
    )
    assert result["status"] == "FAIL_EXCEPTION_TERMINAL_NO_RETRY_NO_RESUME"
    assert result["stage"] == "observe_update_0"
    assert result["retry_authorized"] is False
    assert result["resume_authorized"] is False
    assert result["access"]["n320"]["checkpoint_open_count"] == 1
    assert apis.runtime.loader.closed is True
