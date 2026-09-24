from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from lewm.tests.test_run_go2_categorical_radial_ladder import (
    _TinyCategoricalRadial,
    _write_synthetic_manifest,
)
from scripts import run_go2_categorical_radial_ladder as v1
from scripts import run_go2_categorical_radial_ladder_v2 as runner


def _tiny_initial_state_sha256() -> str:
    return v1._state_dict_sha256(_TinyCategoricalRadial().state_dict())


def _configure_synthetic_main(
    tmp_path: Path,
    monkeypatch,
) -> tuple[Path, str]:
    manifest_path, manifest_sha256 = _write_synthetic_manifest(tmp_path)
    monkeypatch.setattr(runner, "FROZEN_LADDER_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(
        runner,
        "EXPECTED_LADDER_FILE_SHA256",
        manifest_sha256,
    )
    monkeypatch.setattr(
        runner,
        "CategoricalRadialPerception",
        _TinyCategoricalRadial,
    )
    monkeypatch.setattr(
        runner,
        "validate_v2_preregistration",
        lambda: {
            "model": {
                "initial_state_sha256": _tiny_initial_state_sha256(),
            }
        },
    )
    return manifest_path, manifest_sha256


def test_cosine_schedule_has_exact_endpoints_and_registered_midpoint() -> None:
    assert runner.cosine_learning_rate(1, 1000) == 2e-4
    assert runner.cosine_learning_rate(1000, 1000) == 1e-5
    assert runner.cosine_learning_rate(1, 1500) == 2e-4
    assert runner.cosine_learning_rate(1500, 1500) == 1e-5
    assert runner.cosine_learning_rate(1, 2000) == 2e-4
    assert runner.cosine_learning_rate(2000, 2000) == 1e-5

    expected_midpoint = 1e-5 + 0.5 * (2e-4 - 1e-5)
    assert runner.cosine_learning_rate(3, 5) == pytest.approx(expected_midpoint)
    values = [runner.cosine_learning_rate(update, 9) for update in range(1, 10)]
    assert all(left > right for left, right in zip(values, values[1:]))
    with pytest.raises(ValueError, match="at least two"):
        runner.cosine_learning_rate(1, 1)
    with pytest.raises(ValueError, match="outside"):
        runner.cosine_learning_rate(0, 3)


def test_schedule_contract_freezes_semantics_and_excludes_n32() -> None:
    contract = runner.learning_rate_schedule_contract(1500)
    assert contract["first_update_learning_rate"] == 2e-4
    assert contract["final_update_learning_rate"] == 1e-5
    assert contract["warmup_updates"] == 0
    assert contract["stage_local_restart"] is True
    assert contract["library_scheduler_used"] is False
    assert contract["assignment_timing"] == "immediately_before_optimizer_step"
    assert contract["scope"] == "n1_n4_n16_train_only_ladder"
    assert contract["must_not_apply_to_n32_or_full_dataset"] is True
    assert contract["best_step_selection"] is False
    assert runner.AUTHORITATIVE_STAGES is v1.AUTHORITATIVE_STAGES
    assert set(runner.AUTHORITATIVE_STAGES) == {1, 4, 16}
    assert 32 not in runner.AUTHORITATIVE_STAGES


def test_learning_rate_is_assigned_before_each_optimizer_step(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, _manifest_sha256 = _write_synthetic_manifest(tmp_path)
    selected = v1.validate_ladder_manifest(json.loads(manifest_path.read_text()))
    monkeypatch.setattr(
        runner,
        "CategoricalRadialPerception",
        _TinyCategoricalRadial,
    )
    observed_learning_rates = []
    base_adamw = torch.optim.AdamW

    class ObservedAdamW(base_adamw):
        def step(self, closure=None):
            observed_learning_rates.append(float(self.param_groups[0]["lr"]))
            return super().step(closure)

    monkeypatch.setattr(runner.torch.optim, "AdamW", ObservedAdamW)
    initial_state = v1._clone_state(_TinyCategoricalRadial().state_dict())
    stage = runner._train_stage(
        selected[:1],
        initial_state=initial_state,
        initial_state_sha256=v1._state_dict_sha256(initial_state),
        device=torch.device("cpu"),
        seed=20260710,
        updates=3,
        batch_size=1,
        evaluation_interval=1,
    )

    expected = [
        runner.cosine_learning_rate(update, 3) for update in range(1, 4)
    ]
    assert observed_learning_rates == expected
    assert [point["learning_rate_for_update"] for point in stage["curve"]] == expected
    assert stage["optimizer"]["learning_rate_schedule"] == (
        runner.learning_rate_schedule_contract(3)
    )


def test_preregistration_binds_exact_amendment_and_v1_result() -> None:
    result = runner.validate_v2_preregistration()
    sources = runner._source_hashes()

    assert result["content_sha256"] == runner.EXPECTED_V1_RESULT_CONTENT_SHA256
    assert sources["amendment"]["sha256"] == runner.EXPECTED_AMENDMENT_SHA256
    assert sources["v1_result"]["sha256"] == (
        runner.EXPECTED_V1_RESULT_FILE_SHA256
    )
    assert sources["v1_runner"]["sha256"] == (
        result["source_hashes"]["runner"]["sha256"]
    )
    assert Path(sources["v1_runner"]["path"]) == v1.SOURCE_PATHS[
        "runner"
    ].resolve()
    assert Path(sources["runner"]["path"]) == Path(runner.__file__).resolve()
    assert sources["runner"]["sha256"] != sources["v1_runner"]["sha256"]


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("EXPECTED_AMENDMENT_SHA256", "amendment"),
        ("EXPECTED_V1_RESULT_FILE_SHA256", "result file"),
    ],
)
def test_preregistration_refuses_wrong_file_hashes(
    monkeypatch,
    attribute: str,
    message: str,
) -> None:
    monkeypatch.setattr(runner, attribute, "0" * 64)
    with pytest.raises(ValueError, match=message):
        runner.validate_v2_preregistration()


def test_preregistration_refuses_wrong_v1_content_hash(monkeypatch) -> None:
    monkeypatch.setattr(runner, "EXPECTED_V1_RESULT_CONTENT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="content"):
        runner.validate_bound_v1_result()


def test_cli_refuses_any_nonfrozen_ladder_hash(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--ladder-manifest",
                str(tmp_path / "ladder.json"),
                "--expected-ladder-sha256",
                "0" * 64,
                "--output",
                str(tmp_path / "result.json"),
            ]
        )


def test_smoke_exercises_all_stages_with_v2_schemas_and_access_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, manifest_sha256 = _configure_synthetic_main(
        tmp_path,
        monkeypatch,
    )
    output_path = tmp_path / "v2_smoke_result.json"
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
    assert result["v2_scope"] == {
        "adaptive_post_hoc_train_only_schedule": True,
        "schedule_sweep_permitted": False,
        "applies_only_to_n1_n4_n16": True,
        "n32_optimizer_branches_unchanged": True,
        "full_dataset_optimizer_unchanged": True,
        "rocm_grid_sample_backward_warn_only_nondeterminism": True,
    }
    assert [stage["frame_count"] for stage in result["stages"]] == [1, 4, 16]
    assert all(stage["schema"] == runner.STAGE_SCHEMA for stage in result["stages"])
    assert all(stage["completed_updates"] == 3 for stage in result["stages"])
    assert all(stage["fixed_budget_consumed"] for stage in result["stages"])
    expected_rates = [
        runner.cosine_learning_rate(update, 3) for update in range(1, 4)
    ]
    for stage in result["stages"]:
        assert [
            point["learning_rate_for_update"] for point in stage["curve"]
        ] == expected_rates
        assert all(
            point["evaluation"]["schema"] == runner.EVALUATION_SCHEMA
            for point in stage["curve"]
        )
    assert result["decision"]["smoke_exercised_all_stage_paths"] is True
    assert result["decision"]["n32_fit_panel_diagnostic_licensed"] is False
    assert result["decision"]["n32_attempted"] is False
    assert result["model"]["matches_v1_initial_state"] is True
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
    assert result["content_sha256"] == v1.canonical_json_sha256(
        {key: value for key, value in result.items() if key != "content_sha256"}
    )


def test_authoritative_v2_stops_after_first_failed_terminal_stage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path, manifest_sha256 = _configure_synthetic_main(
        tmp_path,
        monkeypatch,
    )
    output_path = tmp_path / "v2_authoritative_result.json"
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
    assert result["decision"][
        "authoritative_first_failure_stop_policy_enforced"
    ] is True
