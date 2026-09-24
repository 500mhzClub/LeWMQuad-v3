from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from lewm.tests.test_run_go2_categorical_radial_ladder import (
    _TinyCategoricalRadial,
    _write_synthetic_manifest,
)
from scripts import run_go2_categorical_radial_ladder_v2 as v2
from scripts import run_go2_categorical_radial_ladder_v3 as runner


def _tiny_initial_state_sha256() -> str:
    return v2.v1._state_dict_sha256(_TinyCategoricalRadial().state_dict())


def _configure_synthetic_main(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, str]:
    manifest_path, manifest_sha256 = _write_synthetic_manifest(tmp_path)
    tiny_hash = _tiny_initial_state_sha256()
    monkeypatch.setattr(runner, "FROZEN_LADDER_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        runner,
        "EXPECTED_LADDER_FILE_SHA256",
        manifest_sha256,
    )
    monkeypatch.setattr(
        runner,
        "V2CategoricalRadialPerception",
        _TinyCategoricalRadial,
    )
    monkeypatch.setattr(
        runner,
        "CategoricalRadialPerceptionFullRay",
        _TinyCategoricalRadial,
    )
    monkeypatch.setattr(
        runner,
        "REGISTERED_PARAMETER_COUNT",
        sum(
            parameter.numel()
            for parameter in _TinyCategoricalRadial().parameters()
        ),
    )
    monkeypatch.setattr(runner, "EXPECTED_V2_INITIAL_STATE_SHA256", tiny_hash)
    monkeypatch.setattr(
        runner,
        "validate_v3_preregistration",
        lambda: {"model": {"initial_state_sha256": tiny_hash}},
    )
    monkeypatch.setattr(
        runner,
        "full_ray_architecture_report",
        lambda _model: {
            "direct_reachability_all_true": True,
            "circular_range_wrap": False,
        },
    )
    return manifest_path, manifest_sha256


def test_v3_inherits_the_exact_v2_schedule_and_stage_contracts() -> None:
    assert runner.AUTHORITATIVE_STAGES is v2.AUTHORITATIVE_STAGES
    assert runner.SMOKE_STAGES is v2.SMOKE_STAGES
    assert runner.TRAINING_WEIGHTS is v2.TRAINING_WEIGHTS
    assert runner.WEIGHT_DECAY == v2.WEIGHT_DECAY
    assert runner.GRADIENT_CLIP == v2.GRADIENT_CLIP
    for updates in (1000, 1500, 2000):
        assert v2.learning_rate_schedule_contract(updates) == (
            runner.v2.learning_rate_schedule_contract(updates)
        )
        assert v2.cosine_learning_rate(1, updates) == 2e-4
        assert v2.cosine_learning_rate(updates, updates) == 1e-5
    assert set(runner.AUTHORITATIVE_STAGES) == {1, 4, 16}
    assert 32 not in runner.AUTHORITATIVE_STAGES


def test_v3_learning_rate_is_assigned_before_every_optimizer_step(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, _manifest_sha256 = _write_synthetic_manifest(tmp_path)
    selected = v2.v1.validate_ladder_manifest(json.loads(manifest_path.read_text()))
    monkeypatch.setattr(
        runner,
        "CategoricalRadialPerceptionFullRay",
        _TinyCategoricalRadial,
    )
    observed_learning_rates = []
    base_adamw = torch.optim.AdamW

    class ObservedAdamW(base_adamw):
        def step(self, closure=None):
            observed_learning_rates.append(float(self.param_groups[0]["lr"]))
            return super().step(closure)

    monkeypatch.setattr(runner.torch.optim, "AdamW", ObservedAdamW)
    initial_state = v2.v1._clone_state(_TinyCategoricalRadial().state_dict())
    stage = runner._train_stage(
        selected[:1],
        initial_state=initial_state,
        initial_state_sha256=v2.v1._state_dict_sha256(initial_state),
        device=torch.device("cpu"),
        seed=20260710,
        updates=3,
        batch_size=1,
        evaluation_interval=1,
    )

    expected = [v2.cosine_learning_rate(update, 3) for update in range(1, 4)]
    assert observed_learning_rates == expected
    assert [point["learning_rate_for_update"] for point in stage["curve"]] == expected
    assert stage["optimizer"]["learning_rate_schedule"] == (
        v2.learning_rate_schedule_contract(3)
    )


def test_preregistration_binds_exact_v2_evidence_and_all_source_layers() -> None:
    result = runner.validate_v3_preregistration()
    sources = runner._source_hashes()

    assert result["content_sha256"] == runner.EXPECTED_V2_RESULT_CONTENT_SHA256
    assert result["model"]["initial_state_sha256"] == (
        runner.EXPECTED_V2_INITIAL_STATE_SHA256
    )
    assert sources["v3_amendment"]["sha256"] == (
        runner.EXPECTED_AMENDMENT_SHA256
    )
    assert sources["v2_result"]["sha256"] == (
        runner.EXPECTED_V2_RESULT_FILE_SHA256
    )
    assert sources["v2_runner"]["sha256"] == (
        result["source_hashes"]["runner"]["sha256"]
    )
    assert Path(sources["model_full_ray"]["path"]).resolve() == (
        Path(runner.__file__).resolve().parents[1]
        / "lewm/models/categorical_radial_perception_full_ray.py"
    )
    assert Path(sources["runner"]["path"]) == Path(runner.__file__).resolve()
    assert sources["runner"]["sha256"] != sources["v2_runner"]["sha256"]


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("EXPECTED_AMENDMENT_SHA256", "amendment"),
        ("EXPECTED_V2_RESULT_FILE_SHA256", "result file"),
    ],
)
def test_preregistration_refuses_wrong_file_hashes(
    monkeypatch,
    attribute: str,
    message: str,
) -> None:
    monkeypatch.setattr(runner, attribute, "0" * 64)
    with pytest.raises(ValueError, match=message):
        runner.validate_v3_preregistration()


def test_preregistration_refuses_wrong_v2_content_hash(monkeypatch) -> None:
    monkeypatch.setattr(runner, "EXPECTED_V2_RESULT_CONTENT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="content"):
        runner.validate_bound_v2_result()


def test_preregistration_refuses_wrong_v2_initial_hash(monkeypatch) -> None:
    monkeypatch.setattr(runner, "EXPECTED_V2_INITIAL_STATE_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="initial-state"):
        runner.validate_bound_v2_result()


def test_preregistration_refuses_v1_or_v2_source_drift(monkeypatch) -> None:
    current = v2._source_hashes()
    changed = {name: dict(record) for name, record in current.items()}
    changed["v1_runner"]["sha256"] = "0" * 64
    monkeypatch.setattr(v2, "_source_hashes", lambda: changed)
    with pytest.raises(ValueError, match="provenance"):
        runner.validate_bound_v2_result()


def test_common_initialization_report_rejects_any_common_tensor_change() -> None:
    torch.manual_seed(11)
    first = _TinyCategoricalRadial()
    torch.manual_seed(11)
    second = _TinyCategoricalRadial()
    report = runner.common_initialization_report(first, second)
    assert report["all_common_tensors_exactly_equal"] is True

    with torch.no_grad():
        second.logits[0, 0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="common initialization"):
        runner.common_initialization_report(first, second)


def test_execution_refuses_v3_model_or_runner_source_drift(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, manifest_sha256 = _configure_synthetic_main(
        tmp_path,
        monkeypatch,
    )
    stable_sources = runner._source_hashes()
    source_calls = 0

    def drifting_sources():
        nonlocal source_calls
        source_calls += 1
        result = {
            name: dict(record) for name, record in stable_sources.items()
        }
        if source_calls > 1:
            result["model_full_ray"]["sha256"] = "0" * 64
            result["runner"]["sha256"] = "1" * 64
        return result

    def failed_stage(records, **kwargs):
        return {
            "frame_count": len(records),
            "initial_state_sha256": kwargs["initial_state_sha256"],
            "final_fit_gate_passes": False,
            "access_ledger": {},
        }

    monkeypatch.setattr(runner, "_source_hashes", drifting_sources)
    monkeypatch.setattr(runner, "_train_stage", failed_stage)
    with pytest.raises(RuntimeError, match="sources changed"):
        runner.main(
            [
                "--ladder-manifest",
                str(manifest_path),
                "--expected-ladder-sha256",
                manifest_sha256,
                "--output",
                str(tmp_path / "must_not_exist.json"),
                "--device",
                "cpu",
                "--seed",
                "20260710",
                "--non-authoritative-smoke",
            ]
        )


def test_cli_refuses_nonfrozen_manifest_hash_and_second_seed(tmp_path: Path) -> None:
    base = [
        "--ladder-manifest",
        str(tmp_path / "ladder.json"),
        "--output",
        str(tmp_path / "result.json"),
    ]
    with pytest.raises(SystemExit):
        runner._parse_args(
            [*base, "--expected-ladder-sha256", "0" * 64]
        )
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                *base,
                "--expected-ladder-sha256",
                runner.EXPECTED_LADDER_FILE_SHA256,
                "--seed",
                "20260711",
            ]
        )


def test_smoke_visits_all_stages_and_records_zero_nontrain_access(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, manifest_sha256 = _configure_synthetic_main(
        tmp_path,
        monkeypatch,
    )
    output_path = tmp_path / "v3_smoke_result.json"
    status = runner.main(
        [
            "--ladder-manifest",
            str(manifest_path),
            "--expected-ladder-sha256",
            manifest_sha256,
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--seed",
            "20260710",
            "--non-authoritative-smoke",
        ]
    )
    result = json.loads(output_path.read_text())

    assert status == 0
    assert result["schema"] == runner.SMOKE_RESULT_SCHEMA
    assert result["authoritative"] is False
    assert result["promotion_eligible"] is False
    assert result["g2_evaluated"] is False
    assert result["v3_scope"]["bundled_effect_not_receptive_field_only"] is True
    assert result["v3_scope"]["n32_or_full_dataset_attempted"] is False
    assert [stage["frame_count"] for stage in result["stages"]] == [1, 4, 16]
    assert all(stage["schema"] == runner.STAGE_SCHEMA for stage in result["stages"])
    assert all(stage["completed_updates"] == 3 for stage in result["stages"])
    assert all(stage["fixed_budget_consumed"] for stage in result["stages"])
    expected_rates = [v2.cosine_learning_rate(update, 3) for update in range(1, 4)]
    for stage in result["stages"]:
        assert [
            point["learning_rate_for_update"] for point in stage["curve"]
        ] == expected_rates
        assert all(
            point["evaluation"]["schema"] == runner.EVALUATION_SCHEMA
            for point in stage["curve"]
        )
    assert result["decision"]["smoke_exercised_all_stage_paths"] is True
    assert result["decision"]["n32_diagnostic_construction_licensed"] is False
    assert result["decision"]["n32_attempted"] is False
    assert result["model"]["common_initialization_audit"][
        "all_common_tensors_exactly_equal"
    ] is True
    assert result["model"]["full_ray_architecture_audit"][
        "direct_reachability_all_true"
    ] is True
    assert result["model"]["stage_restart_initial_hashes_equal"] is True
    assert result["execution"]["determinism"]["warn_only"] is True

    stages = {stage["frame_count"]: stage for stage in result["stages"]}
    assert stages[1]["access_ledger"]["target_requests"] == 6
    assert stages[4]["access_ledger"]["target_requests"] == 24
    assert stages[16]["access_ledger"]["target_requests"] == 60
    for size in (4, 16):
        for point in stages[size]["curve"]:
            control = point["evaluation"]["wrong_view_control"]
            assert control["same_scene_pairs"] == 0
            assert control["same_image_pairs"] == 0
    ledger = result["artifact_access_ledger"]
    assert ledger["selected_train_image_hash_byte_open_events"] == 32
    assert ledger["selected_train_label_shard_hash_byte_open_events"] == 2
    assert ledger["checkpoint_selection"]["model_outputs"] == 0
    assert ledger["probability_calibration"]["model_outputs"] == 0
    assert ledger["g2_evaluation"]["model_outputs"] == 0
    assert ledger["non_train_image_opens"] == 0
    assert ledger["non_train_label_shard_opens"] == 0
    assert result["content_sha256"] == v2.canonical_json_sha256(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )


def test_authoritative_v3_stops_on_first_failed_terminal_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, manifest_sha256 = _configure_synthetic_main(
        tmp_path,
        monkeypatch,
    )
    output_path = tmp_path / "v3_authoritative_result.json"
    calls = []

    def failed_stage(records, **kwargs):
        calls.append((len(records), kwargs["updates"], kwargs["batch_size"]))
        return {
            "frame_count": len(records),
            "initial_state_sha256": kwargs["initial_state_sha256"],
            "final_fit_gate_passes": False,
            "access_ledger": {},
        }

    monkeypatch.setattr(runner, "_train_stage", failed_stage)
    status = runner.main(
        [
            "--ladder-manifest",
            str(manifest_path),
            "--expected-ladder-sha256",
            manifest_sha256,
            "--output",
            str(output_path),
            "--device",
            "cpu",
            "--seed",
            "20260710",
        ]
    )
    result = json.loads(output_path.read_text())

    assert status == 0
    assert calls == [(1, 1000, 1)]
    assert result["schema"] == runner.RESULT_SCHEMA
    assert result["decision"]["attempted_frame_counts"] == [1]
    assert result["decision"]["stopped_on_first_failed_stage"] is True
