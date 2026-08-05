from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from lewm.benchmarks.go2_categorical_radial_n32 import HOLDOUT_PANELS
from scripts import run_go2_categorical_radial_n32_v2 as runner


class _TinyModel(torch.nn.Module):
    batch_sizes: list[int] = []

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 3, 64, 64))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        type(self).batch_sizes.append(int(image.shape[0]))
        return self.logits.expand(image.shape[0], -1, -1, -1)


def test_bound_metadata_and_failed_v1_validate_without_payload_access() -> None:
    panel, panels, ladder, v3_result, reference, v1_result = (
        runner._load_bound_inputs()
    )
    assert panel["content_sha256"] == runner.PANEL_CONTENT_SHA256
    assert ladder["content_sha256"] == runner.LADDER_CONTENT_SHA256
    assert v3_result["content_sha256"] == runner.V3_RESULT_CONTENT_SHA256
    assert v1_result["content_sha256"] == runner.V1_RESULT_CONTENT_SHA256
    assert v1_result["decision"]["classification"] == "fit_gate_failed"
    assert {name: len(rows) for name, rows in panels.items()} == {
        "fit": 160,
        "same_scene_holdout": 160,
        "cross_scene_holdout": 160,
    }
    assert set(reference["panels"]) == set(HOLDOUT_PANELS)


def test_v2_binding_or_v1_result_hash_drift_fails_before_deserialization(
    monkeypatch,
) -> None:
    original = runner._sha256_file

    def changed(path: Path) -> str:
        if path.resolve() == runner.V1_RESULT_PATH.resolve():
            return "0" * 64
        return original(path)

    monkeypatch.setattr(runner, "_sha256_file", changed)
    with pytest.raises(ValueError, match="evidence SHA-256"):
        runner._load_bound_inputs()


def test_batch80_schedule_is_exact_full_epoch_randperm() -> None:
    expected_hashes = {
        20260710: "79b6e66d4e90246f9eb045675f2a06eb25ae28d26f0997392b6780518e668156",
        20260711: "f621b85716607b7e7b8e1ba931d19cf552eb944feca48d099a2c1a3b8ef801c6",
    }
    for seed, expected_hash in expected_hashes.items():
        schedule = runner.frozen_minibatch_schedule(320, 2000, seed)
        assert len(schedule) == 2000
        assert all(len(batch) == 80 for batch in schedule)
        for start in (0, 4, 1996):
            epoch = [index for batch in schedule[start : start + 4] for index in batch]
            assert sorted(epoch) == list(range(320))
        assert runner.v1.v3.canonical_json_sha256(schedule) == expected_hash
    assert runner.frozen_minibatch_schedule(320, 3, 20260710) != (
        runner.frozen_minibatch_schedule(320, 3, 20260711)
    )


def test_cosine_schedule_has_exact_endpoints_midpoint_and_no_warmup() -> None:
    assert runner.cosine_learning_rate(1, 2000) == 2e-4
    assert runner.cosine_learning_rate(2000, 2000) == 1e-5
    assert runner.cosine_learning_rate(2, 3) == pytest.approx(1.05e-4)
    contract = runner.learning_rate_schedule_contract(2000)
    assert contract["warmup_updates"] == 0
    assert contract["one_indexed"] is True
    assert contract["assignment_timing"] == "immediately_before_optimizer_step"


class _StageDataset:
    def __init__(self) -> None:
        self.events: Counter[str] = Counter()

    def snapshot(self):
        return dict(self.events)

    def delta(self, before):
        return {
            key: int(self.events[key]) - int(before.get(key, 0))
            for key in set(before) | set(self.events)
        }

    def training_batch(self, indices):
        count = len(indices)
        self.events["image_requests"] += count
        self.events["target_requests"] += count
        return {
            "image": torch.zeros(count, 3, 1, 1),
            "labels": torch.zeros(count, 64, 64, dtype=torch.long),
            "mask": torch.ones(count, 64, 64, dtype=torch.bool),
        }


def test_stage_uses_three_direct_batch80_calls_and_clip_once(
    monkeypatch,
) -> None:
    monkeypatch.setattr(runner, "CategoricalRadialPerceptionFullRay", _TinyModel)
    model = _TinyModel()
    initial = runner.v1.v3.v2.v1._clone_state(model.state_dict())
    initial_hash = runner.v1.v3.v2.v1._state_dict_sha256(initial)
    dataset = _StageDataset()
    _TinyModel.batch_sizes = []

    def evaluate(_model, target, _records, **_kwargs):
        target.events["image_requests"] += 960
        target.events["target_requests"] += 320
        return {}, {
            "image_requests": 960,
            "target_requests": 320,
            "model_calls": 80,
            "model_output_frames": 960,
        }

    monkeypatch.setattr(runner, "evaluate_panel", evaluate)
    monkeypatch.setattr(
        runner,
        "terminal_fit_gate_summary",
        lambda curve, updates, interval: {
            "passes": False,
            "steps": [point["step"] for point in curve],
            "updates": updates,
            "interval": interval,
        },
    )
    stage, _trained = runner._run_stage(
        config=runner.SMOKE_CONFIG,
        initial_state=initial,
        initial_state_sha256=initial_hash,
        dataset=dataset,
        records=[{}] * 320,
        controls={},
        device=torch.device("cpu"),
        seed=20260710,
        evaluation_interval=1,
    )
    assert _TinyModel.batch_sizes == [80, 80, 80]
    assert stage["one_direct_forward_backward_per_update"] is True
    assert stage["gradient_accumulation_or_microbatching"] is False
    assert stage["optimizer"]["gradient_clip_applications"] == 3
    assert [point["learning_rate"] for point in stage["learning_curve"]] == (
        pytest.approx([2e-4, 1.05e-4, 1e-5])
    )
    assert stage["training_access"]["image_requests"] == 240
    assert stage["training_access"]["model_output_frames"] == 240
    assert stage["fit_evaluation_access"]["model_output_frames"] == 2880


class _FakePanelDataset:
    def __init__(self, _records, panel):
        self.panel = panel

    def snapshot(self):
        return {name: 0 for name in runner.EVENT_FIELDS}


class _ReconcileDataset:
    def snapshot(self):
        return {
            "image_requests": 179200,
            "target_requests": 166400,
            "image_decode_events": 320,
            "label_shard_npz_open_events": 20,
        }


def test_real_access_reconciliation_separates_data_and_model_counters() -> None:
    stage = {
        "completed_steps": 2000,
        "learning_curve": [{}] * 20,
        "training_access": {
            "image_requests": 160000,
            "target_requests": 160000,
            "image_decode_events": 320,
            "label_shard_npz_open_events": 20,
            "model_calls": 2000,
            "model_output_frames": 160000,
        },
        "fit_evaluation_access": {
            "image_requests": 19200,
            "target_requests": 6400,
            "image_decode_events": 0,
            "label_shard_npz_open_events": 0,
            "model_calls": 1600,
            "model_output_frames": 19200,
        },
    }
    panel_access = {
        panel: {
            "authorized": False,
            "dataset_access": {name: 0 for name in runner.EVENT_FIELDS},
        }
        for panel in HOLDOUT_PANELS
    }
    runner._reconcile_access(
        _ReconcileDataset(), stage, panel_access, holdouts=None
    )


def _patch_main(monkeypatch, tmp_path: Path, *, fit_passes: bool):
    calls = []
    reference = {
        "panels": {panel: {"panel": panel} for panel in HOLDOUT_PANELS}
    }
    monkeypatch.setattr(
        runner,
        "_load_bound_inputs",
        lambda: (
            {},
            {"fit": [], **{panel: [] for panel in HOLDOUT_PANELS}},
            {},
            {},
            reference,
            {"decision": {"classification": "fit_gate_failed"}},
        ),
    )
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"stable": {}})
    monkeypatch.setattr(runner.v1.v3.v2.v1, "_git_snapshot", lambda: {})
    monkeypatch.setattr(
        runner.v1.v3.v2.v1,
        "_resolve_device",
        lambda _value: torch.device("cpu"),
    )
    monkeypatch.setattr(
        runner.v1.v3.v2.v1,
        "_configure_determinism",
        lambda seed: {"seed": seed},
    )
    monkeypatch.setattr(runner, "CategoricalRadialPerceptionFullRay", _TinyModel)
    monkeypatch.setattr(
        runner,
        "REGISTERED_PARAMETER_COUNT",
        sum(parameter.numel() for parameter in _TinyModel().parameters()),
    )
    initial_hash = runner.v1.v3.v2.v1._state_dict_sha256(
        _TinyModel().state_dict()
    )
    monkeypatch.setattr(
        runner,
        "EXPECTED_INITIAL_STATE_SHA256",
        {20260710: initial_hash, 20260711: initial_hash},
    )

    def canonical(_rows, *, seed, panel):
        calls.append(panel)
        return ([{"panel": panel}] * 320, {"seed": seed, "panel": panel})

    monkeypatch.setattr(runner, "_canonical_panel_records", canonical)

    def artifacts(_records, panel):
        count = runner.EXPECTED_PANEL_ARTIFACT_COUNTS[panel]
        return (
            {f"image-{index}": "x" for index in range(count["images"])},
            {f"shard-{index}": "x" for index in range(count["shards"])},
        )

    monkeypatch.setattr(runner, "_artifact_contract", artifacts)
    monkeypatch.setattr(runner, "_verify_artifacts", lambda *_args: None)
    monkeypatch.setattr(runner, "PanelFrameDataset", _FakePanelDataset)

    def stage_run(**kwargs):
        assert kwargs["config"]["batch_size"] == 80
        assert kwargs["config"]["updates"] == 3
        return {
            "stage": runner.STAGE_NAME,
            "terminal_fit_gate": {"passes": fit_passes},
            "holdouts_evaluated": False,
        }, _TinyModel()

    monkeypatch.setattr(runner, "_run_stage", stage_run)
    monkeypatch.setattr(
        runner,
        "evaluate_panel",
        lambda _model, _dataset, _records, **kwargs: (
            {"panel": kwargs["panel"]},
            {name: 0 for name in runner.EVENT_FIELDS},
        ),
    )
    monkeypatch.setattr(
        runner,
        "categorical_holdout_checks",
        lambda candidate, _reference: {
            "panel": candidate["panel"],
            "passes": True,
        },
    )
    monkeypatch.setattr(runner, "_reconcile_access", lambda *_args: None)
    evidence = {
        str(path.resolve()): digest
        for path, digest in {
            runner.PANEL_PATH: runner.PANEL_FILE_SHA256,
            runner.LADDER_PATH: runner.LADDER_FILE_SHA256,
            runner.V3_RESULT_PATH: runner.V3_RESULT_FILE_SHA256,
            runner.PATCH7_RESULT_PATH: runner.PATCH7_RESULT_FILE_SHA256,
            runner.PROTOCOL_PATH: runner.PROTOCOL_SHA256,
            runner.V1_CONTRACT_PATH: runner.V1_CONTRACT_SHA256,
            runner.V1_RESULT_PATH: runner.V1_RESULT_FILE_SHA256,
            runner.V1_RESULT_NOTE_PATH: runner.V1_RESULT_NOTE_SHA256,
            runner.CONTRACT_PATH: runner.EXECUTION_BINDING_SHA256,
        }.items()
    }
    monkeypatch.setattr(
        runner,
        "_sha256_file",
        lambda path: evidence.get(str(Path(path).resolve()), "unused"),
    )
    return calls


def test_smoke_fit_failure_is_one_stage_and_never_touches_holdouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _patch_main(monkeypatch, tmp_path, fit_passes=False)
    output = tmp_path / "smoke.json"
    assert runner.main(
        ["--output", str(output), "--device", "cpu", "--non-authoritative-smoke"]
    ) == 0
    result = json.loads(output.read_text())
    assert calls == ["fit"]
    assert result["schema"] == runner.SMOKE_RESULT_SCHEMA
    assert result["aggregation_eligible"] is False
    assert set(result["stages"]) == {runner.STAGE_NAME}
    assert result["holdouts"] is None
    assert all(
        result["access_ledger"]["panels"][panel]["authorized"] is False
        for panel in HOLDOUT_PANELS
    )


def test_smoke_fit_pass_opens_each_holdout_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _patch_main(monkeypatch, tmp_path, fit_passes=True)
    output = tmp_path / "smoke-pass.json"
    runner.main(
        ["--output", str(output), "--device", "cpu", "--non-authoritative-smoke"]
    )
    result = json.loads(output.read_text())
    assert calls == ["fit", *HOLDOUT_PANELS]
    assert set(result["holdouts"]) == set(HOLDOUT_PANELS)
    assert result["stages"][runner.STAGE_NAME]["holdouts_evaluated"] is True
    assert result["decision"]["favorable"] is True
    assert result["decision"]["aggregation_eligible"] is False


def test_seed11_authorization_fails_before_model_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_main(monkeypatch, tmp_path, fit_passes=False)
    primary = runner._canonical_output(20260710)
    monkeypatch.setattr(runner, "_canonical_output", lambda _seed: tmp_path / "out.json")
    constructed = False

    class ForbiddenModel:
        def __init__(self):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(runner, "CategoricalRadialPerceptionFullRay", ForbiddenModel)
    monkeypatch.setattr(
        runner,
        "_validate_primary_authorization",
        lambda *_args: (_ for _ in ()).throw(ValueError("authorization rejected")),
    )
    with pytest.raises(ValueError, match="authorization rejected"):
        runner.main(
            [
                "--output",
                str(tmp_path / "out.json"),
                "--device",
                "cpu",
                "--seed",
                "20260711",
                "--seed-20260710-result",
                str(primary),
                "--expected-seed-20260710-sha256",
                "a" * 64,
            ]
        )
    assert constructed is False


def test_authoritative_output_path_is_versioned_v2() -> None:
    assert runner._canonical_output(20260710) == (
        runner.REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_n32/v2/seed_20260710_result.json"
    ).resolve()
