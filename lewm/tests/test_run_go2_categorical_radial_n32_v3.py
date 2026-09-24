from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from lewm.benchmarks.go2_categorical_radial_n32 import HOLDOUT_PANELS
from scripts import run_go2_categorical_radial_n32_v3 as runner


class _TinyModel(torch.nn.Module):
    batch_sizes: list[int] = []

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 3, 64, 64))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        type(self).batch_sizes.append(int(image.shape[0]))
        return self.logits.expand(image.shape[0], -1, -1, -1)


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


def test_registered_comparable_initialization_for_both_seeds(monkeypatch) -> None:
    for seed in (20260710, 20260711):
        runner.v2.v1.v3.v2.v1._configure_determinism(seed)
        state = torch.get_rng_state().clone()
        reference, candidate_model = (
            runner.build_comparable_width24_and_token32_models(state)
        )
        local_reference_hash = runner.v2.v1.v3.v2.v1._state_dict_sha256(
            reference.state_dict()
        )
        local_candidate_hash = runner.v2.v1.v3.v2.v1._state_dict_sha256(
            candidate_model.state_dict()
        )
        monkeypatch.setitem(
            runner.EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256,
            seed,
            local_reference_hash,
        )
        monkeypatch.setitem(
            runner.EXPECTED_INITIAL_STATE_SHA256, seed, local_candidate_hash
        )
        candidate, digest, proof = runner._build_comparable_initialization(
            seed=seed, cpu_rng_state=state
        )
        assert digest == local_candidate_hash
        assert proof["v2_reference_initial_state_sha256"] == (
            local_reference_hash
        )
        assert proof["same_shape_entry_count"] == 130
        assert proof["same_shape_entries_bit_identical"] is True
        assert proof["only_shape_changed_state_keys"] == sorted(
            runner.REGISTERED_SHAPE_CHANGED_STATE_KEYS
        )
        assert proof["candidate_parameter_count"] == runner.REGISTERED_PARAMETER_COUNT
        assert runner.v2.v1.v3.v2.v1._state_dict_sha256(candidate) == digest


def test_schedule_and_cosine_are_exact_v2_exposure_transfer() -> None:
    expected = {
        20260710: "79b6e66d4e90246f9eb045675f2a06eb25ae28d26f0997392b6780518e668156",
        20260711: "f621b85716607b7e7b8e1ba931d19cf552eb944feca48d099a2c1a3b8ef801c6",
    }
    for seed, digest in expected.items():
        schedule = runner.frozen_minibatch_schedule(320, 2000, seed)
        assert runner.v2.v1.v3.canonical_json_sha256(schedule) == digest
        for start in (0, 4, 1996):
            epoch = [
                index
                for batch in schedule[start : start + 4]
                for index in batch
            ]
            assert sorted(epoch) == list(range(320))
    assert runner.cosine_learning_rate(1, 2000) == 2e-4
    assert runner.cosine_learning_rate(2000, 2000) == 1e-5


def test_stage_uses_direct_batch80_and_one_clip_per_update(monkeypatch) -> None:
    monkeypatch.setattr(runner, "CategoricalRadialPerceptionFullRayToken32", _TinyModel)
    model = _TinyModel()
    initial = runner.v2.v1.v3.v2.v1._clone_state(model.state_dict())
    initial_hash = runner.v2.v1.v3.v2.v1._state_dict_sha256(initial)
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
    assert stage["optimizer"]["gradient_clip_applications"] == 3
    assert stage["gradient_accumulation_or_microbatching"] is False
    assert [point["learning_rate"] for point in stage["learning_curve"]] == (
        pytest.approx([2e-4, 1.05e-4, 1e-5])
    )
    assert stage["training_access"]["model_output_frames"] == 240
    assert stage["fit_evaluation_access"]["model_output_frames"] == 2880


class _ReconcileDataset:
    def snapshot(self):
        return {
            "image_requests": 179200,
            "target_requests": 166400,
            "image_decode_events": 320,
            "label_shard_npz_open_events": 20,
        }


def test_strict_smoke_access_reconciliation() -> None:
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
    runner._reconcile_access(_ReconcileDataset(), stage, panel_access, None)
    panel_access["same_scene_holdout"]["dataset_access"]["model_output_frames"] = 1
    with pytest.raises(RuntimeError, match="unauthorized"):
        runner._reconcile_access(_ReconcileDataset(), stage, panel_access, None)


class _FakePanelDataset:
    def __init__(self, _records, _panel):
        pass

    def snapshot(self):
        return {name: 0 for name in runner.EVENT_FIELDS}


def _patch_smoke_main(monkeypatch, *, fit_passes: bool = False) -> list[str]:
    calls: list[str] = []
    reference = {"panels": {panel: {} for panel in HOLDOUT_PANELS}}
    monkeypatch.setattr(
        runner,
        "_load_bound_inputs",
        lambda: (
            {},
            {"fit": [], **{panel: [] for panel in HOLDOUT_PANELS}},
            {},
            {},
            reference,
            {},
            {},
        ),
    )
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"stable": {}})
    monkeypatch.setattr(runner.v2.v1.v3.v2.v1, "_git_snapshot", lambda: {})
    monkeypatch.setattr(
        runner.v2.v1.v3.v2.v1,
        "_resolve_device",
        lambda _value: torch.device("cpu"),
    )
    monkeypatch.setattr(
        runner.v2.v1.v3.v2.v1,
        "_configure_determinism",
        lambda seed: {
            "seed": seed,
            "requested": "strict_deterministic_algorithms",
            "effective": "strict_where_supported_warn_on_unsupported",
            "warn_only": True,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        },
    )
    tiny = _TinyModel().state_dict()
    digest = runner.v2.v1.v3.v2.v1._state_dict_sha256(tiny)
    monkeypatch.setattr(
        runner,
        "_build_comparable_initialization",
        lambda **_kwargs: (
            tiny,
            digest,
            {"candidate_parameter_count": runner.REGISTERED_PARAMETER_COUNT},
        ),
    )

    def canonical(_rows, *, seed, panel):
        calls.append(panel)
        return ([{}] * 320, {"seed": seed, "panel": panel})

    monkeypatch.setattr(runner, "_canonical_panel_records", canonical)

    def artifacts(_records, panel):
        counts = runner.EXPECTED_PANEL_ARTIFACT_COUNTS[panel]
        return (
            {f"image-{index}": "x" for index in range(counts["images"])},
            {f"shard-{index}": "x" for index in range(counts["shards"])},
        )

    monkeypatch.setattr(runner, "_artifact_contract", artifacts)
    monkeypatch.setattr(runner, "_verify_artifacts", lambda *_args: None)
    monkeypatch.setattr(runner, "PanelFrameDataset", _FakePanelDataset)
    monkeypatch.setattr(
        runner,
        "_run_stage",
        lambda **_kwargs: (
            {
                "stage": runner.STAGE_NAME,
                "terminal_fit_gate": {"passes": fit_passes},
                "holdouts_evaluated": False,
            },
            _TinyModel(),
        ),
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
            runner.V2_CONTRACT_PATH: runner.V2_CONTRACT_SHA256,
            runner.V2_RESULT_PATH: runner.V2_RESULT_FILE_SHA256,
            runner.V2_RESULT_NOTE_PATH: runner.V2_RESULT_NOTE_SHA256,
            runner.CONTRACT_PATH: runner.EXECUTION_BINDING_SHA256,
        }.items()
    }
    monkeypatch.setattr(
        runner,
        "_sha256_file",
        lambda path: evidence.get(str(Path(path).resolve()), "unused"),
    )
    return calls


def test_synthetic_smoke_failure_never_opens_holdouts(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _patch_smoke_main(monkeypatch)
    output = tmp_path / "smoke.json"
    assert runner.main(
        ["--output", str(output), "--device", "cpu", "--non-authoritative-smoke"]
    ) == 0
    payload = json.loads(output.read_text())
    assert calls == ["fit"]
    assert payload["schema"] == runner.SMOKE_RESULT_SCHEMA
    assert payload["holdouts"] is None
    assert all(
        payload["access_ledger"]["panels"][panel]["authorized"] is False
        for panel in HOLDOUT_PANELS
    )


def test_synthetic_smoke_pass_still_never_opens_holdouts(
    tmp_path: Path, monkeypatch
) -> None:
    calls = _patch_smoke_main(monkeypatch, fit_passes=True)
    output = tmp_path / "smoke-pass.json"

    assert runner.main(
        ["--output", str(output), "--device", "cpu", "--non-authoritative-smoke"]
    ) == 0
    payload = json.loads(output.read_text())
    assert calls == ["fit"]
    assert payload["holdouts"] is None
    assert payload["holdout_checks"] is None
    assert payload["decision"]["classification"] == "non_authoritative_smoke"
    assert payload["decision"]["token_width_32_fit_passes"] is True
    assert payload["decision"]["favorable"] is False
    assert payload["decision"]["aggregation_eligible"] is False
    assert all(
        payload["access_ledger"]["panels"][panel]["authorized"] is False
        for panel in HOLDOUT_PANELS
    )


@pytest.mark.parametrize("canonical_seed", (20260710, 20260711))
def test_smoke_rejects_both_canonical_result_paths(
    canonical_seed: int, tmp_path: Path, monkeypatch
) -> None:
    canonical = {
        seed: tmp_path / f"n32-v3-seed-{seed}.json"
        for seed in (20260710, 20260711)
    }
    monkeypatch.setattr(
        runner,
        "_canonical_output",
        lambda seed: canonical[seed],
    )

    with pytest.raises(SystemExit) as raised:
        runner._parse_args(
            [
                "--output",
                str(canonical[canonical_seed]),
                "--non-authoritative-smoke",
            ]
        )
    assert raised.value.code == 2


def test_seed11_authorization_precedes_device_and_initialization(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_smoke_main(monkeypatch)
    output = tmp_path / "seed11.json"
    primary = tmp_path / "seed10.json"
    monkeypatch.setattr(runner, "_canonical_output", lambda _seed: output)
    monkeypatch.setattr(
        runner,
        "_validate_primary_authorization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("not favorable")),
    )
    reached_device = False

    def forbidden_device(_value):
        nonlocal reached_device
        reached_device = True
        return torch.device("cpu")

    monkeypatch.setattr(runner.v2.v1.v3.v2.v1, "_resolve_device", forbidden_device)
    with pytest.raises(ValueError, match="not favorable"):
        runner.main(
            [
                "--output",
                str(output),
                "--seed",
                "20260711",
                "--seed-20260710-result",
                str(primary),
                "--expected-seed-20260710-sha256",
                "a" * 64,
            ]
        )
    assert reached_device is False
