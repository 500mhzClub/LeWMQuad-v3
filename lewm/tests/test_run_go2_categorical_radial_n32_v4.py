from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
import torch

from lewm.benchmarks.go2_categorical_radial_n32 import HOLDOUT_PANELS
from scripts import run_go2_categorical_radial_n32_v4 as runner


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


def test_frozen_identity_and_additive_source_map() -> None:
    assert runner.EXECUTION_BINDING_SHA256 == (
        "bb691c787af0b90f813ced4e5e521f1b15b70b75c836147cd69275c50df6b5d3"
    )
    assert runner.RESULT_SCHEMA == "lewm_go2_categorical_radial_n32_v4_result_v1"
    assert runner.SMOKE_RESULT_SCHEMA == (
        "lewm_go2_categorical_radial_n32_v4_smoke_result_v1"
    )
    assert runner.STAGE_SCHEMA == "lewm_go2_categorical_radial_n32_v4_stage_v1"
    assert runner.STAGE_NAME == (
        "explicit_hierarchical_output_exposure_matched_v3_cosine"
    )
    assert runner.REGISTERED_PARAMETER_COUNT == 2_887_002
    assert runner.REGISTERED_STATE_ENTRY_COUNT == 133
    assert runner.REGISTERED_SAME_SHAPE_STATE_ENTRY_COUNT == 131
    assert runner.POSE_ROLE_NAMESPACE_AMENDMENT_SHA256 == (
        "ae17eb856c5329e8c5dfa5e4339306ef19e60c53c5f67d43746b268be9cc3370"
    )
    paths = runner._source_paths()
    assert "v3_n32_runner" in paths
    assert "runner" in paths
    assert set(paths) >= {
        "n32_v4_binding",
        "n32_v4_pure",
        "n32_v4_hierarchical_model",
        "n32_v4_model_test",
        "n32_v4_pure_test",
        "n32_v4_runner_test",
        "n32_v4_finalizer_test",
        "v4_finalizer",
    }


def test_registered_comparable_initialization_for_both_seeds() -> None:
    expected = {
        20260710: (
            "8b149b57ae4bb305a2306a4dde2cab5f57a46f1c3760837593ed4d9862491278",
            "0e82e8832eb2c27dc9ef2ea4c6ff35a83dcca181cb1d4172830fb6b2811a9c5e",
        ),
        20260711: (
            "989e2db491d199bc544fabe2df40443a39f3ffc6e936f0d28c24625e7bd0ce13",
            "55ae2bbeecbe3913c7e886c11a3a14a5c4c435673a6067df45a2cca6d12fbc99",
        ),
    }
    for seed, (reference_hash, candidate_hash) in expected.items():
        runner._backend._configure_determinism(seed)
        state = torch.get_rng_state().clone()
        candidate, digest, proof = runner._build_comparable_initialization(
            seed=seed,
            cpu_rng_state=state,
        )
        assert digest == candidate_hash
        assert proof["v2_reference_initial_state_sha256"] == reference_hash
        assert proof["candidate_initial_state_sha256"] == candidate_hash
        assert proof["same_shape_entry_count"] == 131
        assert proof["same_shape_entries_bit_identical"] is True
        assert proof["only_shape_changed_state_keys"] == [
            "polar_head.bias",
            "polar_head.weight",
        ]
        assert proof[
            "shape_changed_head_tensors_left_at_deterministic_pytorch_default"
        ] is True
        assert proof["class_prior_bias_matching_applied"] is False
        assert proof["analytic_v2_head_transform_applied"] is False
        assert proof["zero_initialization_applied"] is False
        assert proof["trained_v2_weight_loaded"] is False
        assert proof["trained_v3_weight_loaded"] is False
        assert proof["candidate_parameter_count"] == runner.REGISTERED_PARAMETER_COUNT
        assert runner._backend._state_dict_sha256(candidate) == candidate_hash


def test_schedule_and_cosine_are_exact_v2_exposure_transfer() -> None:
    for seed, digest in runner.EXPECTED_SCHEDULE_SHA256.items():
        schedule = runner.frozen_minibatch_schedule(320, 2000, seed)
        assert runner._canonical_json_sha256(schedule) == digest
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
    monkeypatch.setattr(
        runner,
        "CategoricalRadialPerceptionFullRayHierarchical",
        _TinyModel,
    )
    model = _TinyModel()
    initial = runner._backend._clone_state(model.state_dict())
    initial_hash = runner._backend._state_dict_sha256(initial)
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
    assert stage["schema"] == runner.STAGE_SCHEMA
    assert stage["stage"] == runner.STAGE_NAME
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


def test_strict_access_reconciliation_is_unchanged() -> None:
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
            {},
            {},
        ),
    )
    monkeypatch.setattr(runner, "_source_hashes", lambda: {"stable": {}})
    monkeypatch.setattr(runner, "_evidence_files", lambda: {})
    monkeypatch.setattr(runner._backend, "_git_snapshot", lambda: {})
    monkeypatch.setattr(
        runner._backend,
        "_resolve_device",
        lambda _value: torch.device("cpu"),
    )
    monkeypatch.setattr(
        runner._backend,
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
    digest = runner._backend._state_dict_sha256(tiny)
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
    return calls


@pytest.mark.parametrize("fit_passes", (False, True))
def test_synthetic_smoke_is_unconditionally_fit_only(
    fit_passes: bool,
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _patch_smoke_main(monkeypatch, fit_passes=fit_passes)
    output = tmp_path / f"smoke-{fit_passes}.json"
    assert runner.main(
        ["--output", str(output), "--device", "cpu", "--non-authoritative-smoke"]
    ) == 0
    payload = json.loads(output.read_text())
    assert calls == ["fit"]
    assert payload["schema"] == runner.SMOKE_RESULT_SCHEMA
    assert payload["authoritative"] is False
    assert payload["aggregation_eligible"] is False
    assert payload["holdouts"] is None
    assert payload["holdout_checks"] is None
    assert payload["decision"]["classification"] == "non_authoritative_smoke"
    assert payload["decision"]["explicit_hierarchical_output_fit_passes"] is (
        fit_passes
    )
    assert payload["decision"]["favorable"] is False
    assert payload["model"]["factor_output_contract"] == (
        runner.FACTOR_OUTPUT_CONTRACT
    )
    assert payload["inputs"]["pose_projection_role_namespace_amendment"] == {
        "path": str(runner.POSE_ROLE_NAMESPACE_AMENDMENT_PATH.resolve()),
        "sha256": runner.POSE_ROLE_NAMESPACE_AMENDMENT_SHA256,
    }
    assert payload["access_ledger"]["dataset_role_policy"] == {
        "current_physical_dataset_role": "train",
        "current_physical_dataset_role_governs_access": True,
        "legacy_rollout_split_is_provenance_only": True,
        "legacy_rollout_split_used_to_filter_rank_calibrate_or_select": False,
    }
    assert all(
        payload["access_ledger"]["panels"][panel]["authorized"] is False
        for panel in HOLDOUT_PANELS
    )


@pytest.mark.parametrize("canonical_seed", (20260710, 20260711))
def test_smoke_rejects_both_canonical_result_paths(
    canonical_seed: int,
    tmp_path: Path,
    monkeypatch,
) -> None:
    canonical = {
        seed: tmp_path / f"n32-v4-seed-{seed}.json"
        for seed in runner.EXPECTED_INITIAL_STATE_SHA256
    }
    monkeypatch.setattr(runner, "_canonical_output", lambda seed: canonical[seed])
    with pytest.raises(SystemExit) as raised:
        runner._parse_args(
            [
                "--output",
                str(canonical[canonical_seed]),
                "--non-authoritative-smoke",
            ]
        )
    assert raised.value.code == 2


def test_authoritative_output_requires_exact_canonical_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    canonical = tmp_path / "canonical.json"
    monkeypatch.setattr(runner, "_canonical_output", lambda _seed: canonical)
    with pytest.raises(SystemExit) as raised:
        runner._parse_args(["--output", str(tmp_path / "wrong.json")])
    assert raised.value.code == 2
    assert runner._parse_args(["--output", str(canonical)]).output == canonical


def test_bound_evidence_hash_drift_fails_before_deserialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    evidence = tmp_path / "evidence.bin"
    evidence.write_bytes(b"synthetic metadata")
    monkeypatch.setattr(runner, "_evidence_files", lambda: {evidence: "0" * 64})
    deserialized = False

    def forbidden():
        nonlocal deserialized
        deserialized = True
        raise AssertionError("must not deserialize")

    monkeypatch.setattr(runner.v3, "_load_bound_inputs", forbidden)
    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        runner._load_bound_inputs()
    assert deserialized is False


def test_seed11_authorization_precedes_device_and_initialization(
    tmp_path: Path,
    monkeypatch,
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

    monkeypatch.setattr(runner._backend, "_resolve_device", forbidden_device)
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
