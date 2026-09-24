from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract
from lewm.benchmarks.go2_categorical_radial_factorization import (
    build_radial_factorization,
)
from lewm.datasets.go2_attitude_sidecar import row_identity_sha256
from scripts import run_go2_dynamic_categorical_radial_fit as runner


def _digest(value: object) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()


def _panel_row(index: int) -> dict[str, object]:
    return {
        "schema": "lewm_go2_physical_micro_overfit_row_v1",
        "scene_id": f"scene-{index // 4}",
        "family": contract.FAMILIES[index % 5],
        "dataset_role": "train",
        "global_row": index,
        "env_index": index % 16,
        "episode_id": "1",
        "reset_count": 0,
        "current_episode_step": index * 2,
        "next_episode_step": index * 2 + 1,
        "current_frame_index": index * 20,
        "next_frame_index": index * 20 + 1,
        "current_timestamp_ns": index * 200,
        "next_timestamp_ns": index * 200 + 100,
        "primitive": "hold",
        "relative_se2_current_frame": [0.0, 0.0, 0.0],
        "label_shard_path": f"/synthetic/shard-{index // 8}.npz",
        "label_shard_sha256": _digest(f"shard-{index // 8}"),
        "label_shard_row": index % 8,
        "current_image_path": f"/synthetic/current-{index}.png",
        "current_image_sha256": _digest(f"current-{index}"),
        "next_image_path": f"/synthetic/next-{index}.png",
        "next_image_sha256": _digest(f"next-{index}"),
    }


def _sidecar_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "global_row": row["global_row"],
        "dataset_role": "train",
        "row_identity_sha256": row_identity_sha256(row),
        "scene_id_sha256": hashlib.sha256(str(row["scene_id"]).encode()).hexdigest(),
        "env_index": row["env_index"],
        "current_frame_index": row["current_frame_index"],
        "next_frame_index": row["next_frame_index"],
        "current_timestamp_ns": row["current_timestamp_ns"],
        "next_timestamp_ns": row["next_timestamp_ns"],
        "current": {
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": 0.0,
        },
        "next": {
            "base_quat_world_xyzw": [0.0, 0.0, 0.0, 1.0],
            "stored_base_yaw_rad": 0.0,
        },
    }


def _padded_sidecar(rows: list[dict[str, object]]) -> tuple[dict[str, object], ...]:
    selected = [_sidecar_row(row) for row in rows]
    selected.extend({"global_row": index} for index in range(len(rows), 4262))
    return tuple(selected)


def test_fit_loader_joins_only_fit_rows_and_only_train_sidecar(
    monkeypatch,
) -> None:
    rows = [_panel_row(index) for index in range(160)]
    sidecar = _padded_sidecar(rows)
    requested_roles = []
    panel = {
        "content_sha256": runner.PANEL_CONTENT_SHA256,
        "panels": {"fit": {"rows_sha256": runner.FIT_ROWS_SHA256}},
    }
    monkeypatch.setattr(runner.backend, "_read_json", lambda *_args, **_kwargs: panel)
    monkeypatch.setattr(runner, "validate_panel_manifest", lambda _panel: {"fit": rows})

    def load(_path, *, roles, **_kwargs):
        requested_roles.extend(roles)
        return {"train": sidecar}

    monkeypatch.setattr(runner, "load_attitude_sidecar_roles", load)
    monkeypatch.setattr(
        runner,
        "sidecar_json_sha256",
        lambda _rows: runner.SIDECAR_TRAIN_CONTENT_SHA256,
    )
    records, audit = runner._load_fit_records()
    assert requested_roles == ["train"]
    assert len(records) == 320
    assert audit["transition_count"] == 160
    assert [record["side"] for record in records[:4]] == [
        "current", "next", "current", "next"
    ]


def test_fit_loader_rejects_sidecar_identity_drift(monkeypatch) -> None:
    rows = [_panel_row(index) for index in range(160)]
    sidecar = list(_padded_sidecar(rows))
    sidecar[0]["current_timestamp_ns"] = 999
    panel = {
        "content_sha256": runner.PANEL_CONTENT_SHA256,
        "panels": {"fit": {"rows_sha256": runner.FIT_ROWS_SHA256}},
    }
    monkeypatch.setattr(runner.backend, "_read_json", lambda *_args, **_kwargs: panel)
    monkeypatch.setattr(runner, "validate_panel_manifest", lambda _panel: {"fit": rows})
    monkeypatch.setattr(
        runner,
        "load_attitude_sidecar_roles",
        lambda *_args, **_kwargs: {"train": tuple(sidecar)},
    )
    monkeypatch.setattr(
        runner,
        "sidecar_json_sha256",
        lambda _rows: runner.SIDECAR_TRAIN_CONTENT_SHA256,
    )
    with pytest.raises(ValueError, match="attitude join mismatch"):
        runner._load_fit_records()


def test_faithful_schedule_is_batch80_complete_epochs_and_deterministic() -> None:
    first = runner._faithful_schedule(20260710, 8)
    second = runner._faithful_schedule(20260710, 8)
    assert first == second
    assert all(len(batch) == 80 for batch in first)
    for start in (0, 4):
        assert sorted(index for batch in first[start : start + 4] for index in batch) == list(range(320))


def test_faithful_cosine_has_exact_registered_endpoints() -> None:
    assert runner._faithful_learning_rate(1, 2000) == pytest.approx(2e-4)
    assert runner._faithful_learning_rate(2000, 2000) == pytest.approx(1e-5)
    assert [runner._faithful_learning_rate(step, 3) for step in (1, 2, 3)] == pytest.approx([2e-4, 1.05e-4, 1e-5])


def test_ceiling_schedule_is_batch4_and_fixed_5000() -> None:
    schedule = runner._schedule(20260710, "ceiling_optimizer", False)
    assert len(schedule) == 5000
    assert all(len(batch) == 4 for batch in schedule)
    assert contract.validate_minibatch_schedule(
        schedule, seed=20260710, branch="ceiling_optimizer"
    ) == contract.SCHEDULE_SHA256[(20260710, "ceiling_optimizer")]


class _EvalDataset:
    def __init__(self) -> None:
        self.events = Counter()

    def snapshot(self):
        return {name: int(self.events[name]) for name in runner.EVENT_FIELDS}

    def delta(self, before):
        return {
            name: int(self.events[name]) - int(before.get(name, 0))
            for name in runner.EVENT_FIELDS
        }

    def evaluation_batch(self, indices):
        count = len(indices)
        self.events.update(
            image_requests=count * 3,
            target_requests=count,
            attitude_requests=count,
        )
        return {
            **{
                condition: torch.zeros(count, 3, 2, 2)
                for condition in contract.CONDITIONS
            },
            "labels": torch.zeros(count, 64, 64, dtype=torch.long),
            "mask": torch.ones(count, 64, 64, dtype=torch.bool),
            "base_quat_world_xyzw": torch.arange(count, dtype=torch.float32)[
                :, None
            ].repeat(1, 4),
            "stored_base_yaw_rad": torch.arange(count, dtype=torch.float32),
        }


class _CaptureModel:
    def __init__(self) -> None:
        self.attitudes = []

    def eval(self):
        return self

    def __call__(self, images, quaternions, yaws):
        self.attitudes.append((quaternions.clone(), yaws.clone()))
        return torch.zeros(images.shape[0], 3, 64, 64)


def test_wrong_rgb_controls_keep_target_attitude(monkeypatch) -> None:
    monkeypatch.setattr(runner, "empty_raw_accumulator", lambda: {})
    monkeypatch.setattr(runner, "update_raw_accumulator", lambda *_args: None)
    monkeypatch.setattr(
        runner, "finalize_raw_accumulator", lambda _value: {"synthetic": True}
    )
    monkeypatch.setattr(runner.contract, "fit_panel_gate_report", lambda _report: {"passes": False})
    monkeypatch.setattr(runner.contract, "validate_panel_report", lambda *_args, **_kwargs: None)
    records = [
        {"family": contract.FAMILIES[index % 5]} for index in range(4)
    ]
    controls = {"role_global_shuffle": {"seed": 20260710}}
    model = _CaptureModel()
    report, access = runner.evaluate_fit(
        model,
        _EvalDataset(),
        records,
        device=torch.device("cpu"),
        controls=controls,
    )
    quaternions, yaws = model.attitudes[0]
    assert torch.equal(quaternions[:4], quaternions[4:8])
    assert torch.equal(quaternions[:4], quaternions[8:12])
    assert torch.equal(yaws[:4], yaws[4:8])
    assert report["wrong_rgb_uses_target_attitude"] is True
    assert access["model_output_frames"] == 12


def test_target_support_audit_accepts_unknown_but_rejects_known_outside() -> None:
    factorization = build_radial_factorization()
    support = np.asarray(factorization.representable_mask, dtype=bool)
    labels = np.zeros((1, 64, 64), dtype=np.int64)
    masks = np.ones((1, 64, 64), dtype=bool)
    shard = {
        "current_labels": labels,
        "current_supervision_mask": masks,
    }
    records = [
        {"label_shard_path": "shard", "label_shard_row": 0, "side": "current"}
    ]
    report = runner.audit_fit_target_support(records, {"shard": shard})
    assert report["supervised_outside_support_occurrences"] > 0
    assert report["known_outside_support_occurrences"] == 0
    outside = np.argwhere(~support)[0]
    labels[(0, *outside)] = 1
    with pytest.raises(ValueError, match="KNOWN outside"):
        runner.audit_fit_target_support(records, {"shard": shard})


def test_actual_fit_targets_never_supervise_unrepresentable_known_cells() -> None:
    records, _audit = runner._load_fit_records()
    shard_contract = sorted(
        {
            (str(record["label_shard_path"]), str(record["label_shard_sha256"]))
            for record in records
        }
    )
    shards = {
        path: runner.backend._decode_shard(path, digest)
        for path, digest in shard_contract
    }
    report = runner.audit_fit_target_support(records, shards)
    assert report["frame_count"] == 320
    assert report["known_outside_support_occurrences"] == 0
    assert report["all_supervised_known_cells_representable"] is True


def test_parser_enforces_canonical_immutable_smoke_and_full_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    outputs = {
        smoke: (tmp_path / f"{smoke}.json").resolve() for smoke in (False, True)
    }
    monkeypatch.setattr(
        runner, "_canonical_output", lambda _seed, smoke: outputs[smoke]
    )
    assert runner._parse_args(["--output", str(outputs[False])]).smoke is False
    assert runner._parse_args(
        ["--output", str(outputs[True]), "--smoke"]
    ).smoke is True
    with pytest.raises(SystemExit):
        runner._parse_args(["--output", str(tmp_path / "other.json")])
    outputs[False].write_text("occupied")
    with pytest.raises(SystemExit):
        runner._parse_args(["--output", str(outputs[False])])


def test_result_schema_is_permanently_development_fit_only() -> None:
    source = Path(runner.__file__).read_text()
    assert '"authoritative": False' in source
    assert '"development_only": True' in source
    assert '"fit_only": True' in source
    assert '"holdouts": None' in source
    assert '"g2": None' in source
    assert '"non_fit_image_payload_byte_opens": 0' in source
    assert '"g2_payload_byte_opens": 0' in source


class _TinyDynamic(torch.nn.Module):
    batch_sizes: list[int] = []

    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, images, quaternions, yaws):
        del quaternions, yaws
        self.batch_sizes.append(int(images.shape[0]))
        return self.value.expand(images.shape[0], 3, 64, 64)


def test_new_model_moves_numpy_backed_buffers_to_requested_device() -> None:
    model = runner._new_model(torch.device("meta"))
    assert {parameter.device.type for parameter in model.parameters()} == {"meta"}
    assert {buffer.device.type for buffer in model.buffers()} == {"meta"}


def test_development_determinism_declares_rocm_grid_sampler_exception(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        runner.backend,
        "_configure_determinism",
        lambda seed: {"seed": seed, "warn_only": False},
    )
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only: calls.append((enabled, warn_only)),
    )
    record = runner._configure_development_determinism(20260710)
    assert calls == [(True, True)]
    assert record["warn_only"] is True
    assert record["known_kernel_exception"] == (
        "ROCm grid_sampler_2d_backward_cuda"
    )
    assert record["replication_required_before_any_promotion"] is True


class _StageDataset:
    def __init__(self) -> None:
        self.events = Counter()

    def snapshot(self):
        return {name: int(self.events[name]) for name in runner.EVENT_FIELDS}

    def delta(self, before):
        return {
            name: int(self.events[name]) - int(before.get(name, 0))
            for name in runner.EVENT_FIELDS
        }

    def training_batch(self, indices):
        count = len(indices)
        self.events.update(
            image_requests=count,
            target_requests=count,
            attitude_requests=count,
        )
        return {
            "image": torch.zeros(count, 3, 2, 2),
            "base_quat_world_xyzw": torch.zeros(count, 4),
            "stored_base_yaw_rad": torch.zeros(count),
            "labels": torch.zeros(count, 64, 64, dtype=torch.long),
            "mask": torch.ones(count, 64, 64, dtype=torch.bool),
        }


def _patch_stage_evaluation(monkeypatch) -> None:
    def evaluate(_model, dataset, _records, **_kwargs):
        access = {
            "image_requests": 960,
            "target_requests": 320,
            "attitude_requests": 320,
            "image_decode_events": 0,
            "label_shard_npz_open_events": 0,
            "model_calls": 80,
            "model_output_frames": 960,
            "model_attitude_frames": 960,
        }
        dataset.events.update(
            {name: value for name, value in access.items() if name in (
                "image_requests", "target_requests", "attitude_requests"
            )}
        )
        return {"fit_gate": {"passes": False}}, access

    monkeypatch.setattr(runner, "evaluate_fit", evaluate)
    monkeypatch.setattr(
        runner.contract,
        "terminal_fit_gate_summary",
        lambda curve, updates, interval: {
            "passes": False,
            "steps": [point["step"] for point in curve],
            "updates": updates,
            "interval": interval,
        },
    )


def test_three_update_faithful_smoke_uses_batch80_cosine_and_finite_steps(
    monkeypatch,
) -> None:
    _TinyDynamic.batch_sizes = []
    monkeypatch.setattr(runner, "_new_model", lambda _device: _TinyDynamic())
    _patch_stage_evaluation(monkeypatch)
    initial = _TinyDynamic().state_dict()
    initial_hash = runner.backend._state_dict_sha256(initial)
    stage, _model = runner._run_stage(
        branch="production_faithful",
        smoke=True,
        initial_state=initial,
        initial_state_sha256=initial_hash,
        dataset=_StageDataset(),
        records=[{}] * 320,
        controls={},
        device=torch.device("cpu"),
        seed=20260710,
    )
    assert _TinyDynamic.batch_sizes == [80, 80, 80]
    assert [point["learning_rate"] for point in stage["learning_curve"]] == pytest.approx([2e-4, 1.05e-4, 1e-5])
    assert stage["training_access"]["model_output_frames"] == 240
    assert stage["fit_evaluation_access"]["model_output_frames"] == 2880


def test_smoke_fails_closed_on_nonfinite_gradient_norm(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_new_model", lambda _device: _TinyDynamic())
    monkeypatch.setattr(
        torch.nn.utils, "clip_grad_norm_", lambda *_args, **_kwargs: torch.tensor(float("inf"))
    )
    initial = _TinyDynamic().state_dict()
    with pytest.raises(FloatingPointError, match="gradient norm"):
        runner._run_stage(
            branch="production_faithful",
            smoke=True,
            initial_state=initial,
            initial_state_sha256=runner.backend._state_dict_sha256(initial),
            dataset=_StageDataset(),
            records=[{}] * 320,
            controls={},
            device=torch.device("cpu"),
            seed=20260710,
        )
