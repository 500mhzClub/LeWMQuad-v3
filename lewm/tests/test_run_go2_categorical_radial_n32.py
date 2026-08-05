from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from lewm.benchmarks.go2_categorical_radial_n32 import (
    FAMILIES,
    HOLDOUT_PANELS,
)
from scripts import run_go2_categorical_radial_n32 as runner


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 3, 64, 64))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.logits.expand(image.shape[0], -1, -1, -1)


def test_bound_metadata_inputs_validate_without_payload_access() -> None:
    panel, panels, ladder, v3_result, reference = runner._load_bound_inputs()
    assert panel["content_sha256"] == runner.PANEL_CONTENT_SHA256
    assert ladder["content_sha256"] == runner.LADDER_CONTENT_SHA256
    assert v3_result["content_sha256"] == runner.V3_RESULT_CONTENT_SHA256
    assert {name: len(rows) for name, rows in panels.items()} == {
        "fit": 160,
        "same_scene_holdout": 160,
        "cross_scene_holdout": 160,
    }
    assert set(reference["panels"]) == set(HOLDOUT_PANELS)


def test_bound_input_hash_drift_fails_before_deserialization(monkeypatch) -> None:
    original = runner._sha256_file

    def changed(path: Path) -> str:
        if path.resolve() == runner.PANEL_PATH.resolve():
            return "0" * 64
        return original(path)

    monkeypatch.setattr(runner, "_sha256_file", changed)
    with pytest.raises(ValueError, match="evidence SHA-256"):
        runner._load_bound_inputs()


def test_frame_level_schedule_and_branch_prefix_are_exact() -> None:
    faithful = runner.frozen_minibatch_schedule(320, 2000, 20260710)
    ceiling = runner.frozen_minibatch_schedule(320, 5000, 20260710)
    assert ceiling[:2000] == faithful
    assert len(faithful) == 2000
    assert all(len(batch) == 4 for batch in faithful)
    first_epoch = [index for batch in faithful[:80] for index in batch]
    assert sorted(first_epoch) == list(range(320))
    assert runner.frozen_minibatch_schedule(320, 3, 20260711) != (
        runner.frozen_minibatch_schedule(320, 3, 20260710)
    )


def test_training_loss_is_direct_and_not_legacy_double_scaled() -> None:
    generator = torch.Generator().manual_seed(19)
    logits = torch.randn(2, 3, 64, 64, generator=generator)
    labels = torch.arange(64).remainder(3)[None, :, None]
    labels = labels.expand(2, 64, 64).long()
    mask = torch.ones_like(labels, dtype=torch.bool)
    direct = runner.direct_hierarchical_loss(logits, labels, mask)
    frozen = runner.v3.v2.v1.hierarchical_occupancy_loss(logits, labels, mask)
    assert torch.equal(direct, frozen)
    assert not torch.equal(direct, 2.0 * frozen)


def test_seeded_initialization_restarts_identically_without_forward() -> None:
    runner.v3.v2.v1._configure_determinism(20260710)
    first = runner.CategoricalRadialPerceptionFullRay()
    first_hash = runner.v3.v2.v1._state_dict_sha256(first.state_dict())
    runner.v3.v2.v1._configure_determinism(20260710)
    second = runner.CategoricalRadialPerceptionFullRay()
    second_hash = runner.v3.v2.v1._state_dict_sha256(second.state_dict())
    assert first_hash == second_hash
    assert sum(parameter.numel() for parameter in first.parameters()) == (
        runner.REGISTERED_PARAMETER_COUNT
    )
    assert runner.EXPECTED_SEED10_INITIAL_STATE_SHA256 == (
        "8b149b57ae4bb305a2306a4dde2cab5f57a46f1c3760837593ed4d9862491278"
    )


def test_complete_panel_controls_are_once_created_and_zero_match(monkeypatch) -> None:
    records = [
        {
            "scene_id": f"scene_{index // 20:02d}",
            "family": FAMILIES[index % len(FAMILIES)],
            "global_row": index,
            "side": "current",
            "image_path": f"/train/{index}.png",
            "image_sha256": f"{index:064x}",
            "label_shard_path": f"/train/{index // 16}.npz",
            "label_shard_sha256": f"{1000 + index // 16:064x}",
            "label_shard_row": index,
        }
        for index in range(320)
    ]
    calls = []

    def frozen_records(_rows):
        calls.append("frame_records")
        return [dict(record) for record in records]

    monkeypatch.setattr(runner, "frame_records", frozen_records)
    attached, controls = runner._canonical_panel_records(
        [], seed=20260710, panel="fit"
    )
    assert calls == ["frame_records"]
    assert len(attached) == 320
    assert controls["role_global_shuffle"]["same_image_pairs"] == 0
    assert controls["role_global_shuffle"]["same_scene_pairs"] == 0
    assert controls["role_global_shuffle"]["same_transition_pairs"] == 0
    assert controls["same_scene_wrong_view"]["same_image_pairs"] == 0
    assert controls["same_scene_wrong_view"]["same_transition_pairs"] == 0
    assert controls["same_scene_wrong_view"]["different_scene_pairs"] == 0


def test_panel_dataset_caches_images_shards_and_uses_target_labels(
    tmp_path: Path,
) -> None:
    labels = np.zeros((4, 64, 64), dtype=np.uint8)
    labels[:, 21:42] = 1
    labels[:, 42:] = 2
    masks = np.ones_like(labels, dtype=bool)
    shard = tmp_path / "labels.npz"
    np.savez(
        shard,
        current_labels=labels,
        current_supervision_mask=masks,
    )
    records = []
    for index in range(4):
        path = tmp_path / f"{index}.png"
        Image.fromarray(np.full((12, 16, 3), index * 20, dtype=np.uint8)).save(path)
        records.append(
            {
                "image_path": str(path),
                "control_image_path": str(tmp_path / f"{(index + 1) % 4}.png"),
                "same_scene_control_image_path": str(
                    tmp_path / f"{(index + 2) % 4}.png"
                ),
                "label_shard_path": str(shard),
                "label_shard_row": index,
                "side": "current",
            }
        )
    dataset = runner.PanelFrameDataset(records, "fit")
    training = dataset.training_batch(range(4))
    evaluation = dataset.evaluation_batch(range(4))
    assert training["image"].shape == (4, 3, 112, 112)
    assert evaluation["role_global_shuffled_rgb"].shape == (4, 3, 112, 112)
    assert torch.equal(training["labels"], evaluation["labels"])
    assert dataset.events["image_decode_events"] == 4
    assert dataset.events["label_shard_npz_open_events"] == 1


class _EvaluationDataset:
    def __init__(self) -> None:
        self.events = {}

    def snapshot(self):
        return dict(self.events)

    def delta(self, _before):
        return {}

    def evaluation_batch(self, _indices):
        labels = torch.zeros(4, 64, 64, dtype=torch.long)
        labels[:, 21:42] = 1
        labels[:, 42:] = 2
        images = torch.zeros(4, 3, 112, 112)
        return {
            "correct_rgb": images,
            "role_global_shuffled_rgb": images.clone(),
            "same_scene_wrong_view_rgb": images.clone(),
            "labels": labels,
            "mask": torch.ones_like(labels, dtype=torch.bool),
        }


class _EvaluationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        logits = torch.full((1, 3, 64, 64), -5.0)
        logits[:, 0, :21] = 5.0
        logits[:, 1, 21:42] = 5.0
        logits[:, 2, 42:] = 5.0
        self.register_buffer("logits", logits)
        self.batch_sizes = []

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(image.shape[0]))
        return self.logits.expand(image.shape[0], -1, -1, -1)


def test_panel_evaluator_uses_combined_12_and_target_family_slices() -> None:
    records = [
        {"family": family}
        for family in FAMILIES
        for _index in range(4)
    ]
    model = _EvaluationModel()
    report, access = runner.evaluate_panel(
        model,
        _EvaluationDataset(),
        records,
        device=torch.device("cpu"),
        panel="same_scene_holdout",
        controls={},
    )
    assert model.batch_sizes == [12] * 5
    assert report["model_call_dtype"] == "float32"
    assert report["metric_accumulator_dtype"] == "float64"
    assert report["families"][FAMILIES[0]]["conditions"]["correct_rgb"][
        "cell_count"
    ] == 4 * 64 * 64
    assert access["model_calls"] == 5
    assert access["model_output_frames"] == 60


class _FakePanelDataset:
    def __init__(self, _records, panel):
        self.panel = panel

    def snapshot(self):
        return {
            "image_decode_events": 320,
            "label_shard_npz_open_events": 20,
        }


def _patch_main(
    monkeypatch,
    tmp_path: Path,
    *,
    faithful_pass: bool,
    ceiling_pass: bool = False,
):
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
        ),
    )
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"stable": {}})
    monkeypatch.setattr(runner.v3.v2.v1, "_git_snapshot", lambda: {})
    monkeypatch.setattr(
        runner.v3.v2.v1,
        "_resolve_device",
        lambda _value: torch.device("cpu"),
    )
    monkeypatch.setattr(
        runner.v3.v2.v1,
        "_configure_determinism",
        lambda seed: {"seed": seed, "warn_only": True},
    )
    monkeypatch.setattr(runner, "CategoricalRadialPerceptionFullRay", _TinyModel)
    monkeypatch.setattr(
        runner,
        "REGISTERED_PARAMETER_COUNT",
        sum(parameter.numel() for parameter in _TinyModel().parameters()),
    )
    tiny_hash = runner.v3.v2.v1._state_dict_sha256(_TinyModel().state_dict())
    monkeypatch.setattr(runner, "EXPECTED_SEED10_INITIAL_STATE_SHA256", tiny_hash)

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
    branch_passes = [faithful_pass, ceiling_pass]

    def stage_run(**kwargs):
        passes = branch_passes.pop(0)
        updates = int(kwargs["config"]["updates"])
        stage = {
            "stage": kwargs["stage_name"],
            "initial_state_sha256": kwargs["initial_state_sha256"],
            "terminal_fit_gate": {"passes": passes},
            "minibatch_indices": [[0, 1, 2, 3]] * updates,
            "holdouts_evaluated": False,
        }
        return stage, _TinyModel()

    monkeypatch.setattr(runner, "_run_stage", stage_run)
    monkeypatch.setattr(
        runner,
        "evaluate_panel",
        lambda _model, _dataset, _records, **kwargs: (
            {"panel": kwargs["panel"]},
            {"model_calls": 80, "model_output_frames": 960},
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
            runner.CONTRACT_PATH: runner.EXECUTION_BINDING_SHA256,
        }.items()
    }
    monkeypatch.setattr(
        runner,
        "_sha256_file",
        lambda path: evidence.get(str(Path(path).resolve()), "unused"),
    )
    return calls


def test_smoke_failed_fit_never_touches_holdouts_and_is_nonaggregatable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _patch_main(
        monkeypatch,
        tmp_path,
        faithful_pass=False,
        ceiling_pass=False,
    )
    output = tmp_path / "smoke.json"
    assert runner.main(
        ["--output", str(output), "--device", "cpu", "--non-authoritative-smoke"]
    ) == 0
    result = json.loads(output.read_text())
    assert calls == ["fit"]
    assert result["schema"] == runner.SMOKE_RESULT_SCHEMA
    assert result["aggregation_eligible"] is False
    assert result["decision"]["aggregation_eligible"] is False
    assert result["holdouts"] is None
    assert result["stages"]["ceiling_optimizer"] is not None
    assert result["categorical_radial_full_train_candidate_licensed"] is False


def test_passing_faithful_skips_ceiling_and_opens_both_holdouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _patch_main(monkeypatch, tmp_path, faithful_pass=True)
    output = tmp_path / "smoke_pass.json"
    runner.main(
        ["--output", str(output), "--device", "cpu", "--non-authoritative-smoke"]
    )
    result = json.loads(output.read_text())
    assert calls == ["fit", *HOLDOUT_PANELS]
    assert result["stages"]["ceiling_optimizer"] is None
    assert set(result["holdouts"]) == set(HOLDOUT_PANELS)
    assert result["stages"]["production_faithful"]["holdouts_evaluated"] is True


def test_passing_ceiling_restarts_prefix_and_opens_both_holdouts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _patch_main(
        monkeypatch,
        tmp_path,
        faithful_pass=False,
        ceiling_pass=True,
    )
    output = tmp_path / "smoke_ceiling.json"
    runner.main(
        ["--output", str(output), "--device", "cpu", "--non-authoritative-smoke"]
    )
    result = json.loads(output.read_text())
    assert calls == ["fit", *HOLDOUT_PANELS]
    assert result["stages"]["ceiling_optimizer"] is not None
    assert result["stages"]["ceiling_optimizer"]["holdouts_evaluated"] is True
    assert result["decision"]["qualifying_optimizer_stage"] == (
        "ceiling_optimizer"
    )


def test_seed11_authorization_blocks_before_model_construction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_main(monkeypatch, tmp_path, faithful_pass=False)
    primary = tmp_path / "primary.json"
    primary.write_text("{}")
    monkeypatch.setattr(
        runner,
        "_canonical_output",
        lambda _seed: tmp_path / "out.json",
    )
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


def test_source_drift_aborts_before_result_publication(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _patch_main(monkeypatch, tmp_path, faithful_pass=False, ceiling_pass=False)
    calls = 0

    def drifting_sources():
        nonlocal calls
        calls += 1
        return {"stable": {"sha256": "a" if calls == 1 else "b"}}

    monkeypatch.setattr(runner, "_source_hashes", drifting_sources)
    output = tmp_path / "drift.json"
    with pytest.raises(RuntimeError, match="sources changed"):
        runner.main(
            [
                "--output",
                str(output),
                "--device",
                "cpu",
                "--non-authoritative-smoke",
            ]
        )
    assert not output.exists()
