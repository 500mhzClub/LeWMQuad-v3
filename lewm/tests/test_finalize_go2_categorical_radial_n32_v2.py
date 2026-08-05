from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lewm.benchmarks.go2_categorical_radial_n32_v2 import (
    STAGE_NAME,
    STAGE_SCHEMA,
    per_seed_decision,
)
from scripts import finalize_go2_categorical_radial_n32_v2 as finalizer
from scripts import run_go2_categorical_radial_n32_v2 as runner


def _terminal(*, passes: bool = False) -> dict:
    steps = list(range(100, 2001, 100))
    values = [False] * 20
    if passes:
        values[-3:] = [True, True, True]
    return {
        "schema": "lewm_go2_categorical_radial_n32_terminal_fit_gate_v1",
        "maximum_steps": 2000,
        "evaluation_interval": 100,
        "evaluation_steps": steps,
        "evaluation_passes": values,
        "terminal_evaluation_steps": steps[-3:],
        "terminal_evaluation_passes": values[-3:],
        "requires_exact_final_three": True,
        "first_single_fit_gate_step": 1800 if passes else None,
        "first_three_consecutive_fit_gate_step": 2000 if passes else None,
        "passes": passes,
    }


def _stage(seed: int = 20260710, *, passes: bool = False) -> dict:
    curve = [
        {
            "step": step,
            "learning_rate": finalizer._cosine_learning_rate(step, 2000),
            "batch_loss": 0.1,
            "gradient_norm_before_clip": 0.2,
            "fit_panel": {},
        }
        for step in range(100, 2001, 100)
    ]
    schedule = runner.frozen_minibatch_schedule(320, 2000, seed)
    return {
        "schema": STAGE_SCHEMA,
        "stage": STAGE_NAME,
        "maximum_steps": 2000,
        "completed_steps": 2000,
        "batch_size": 80,
        "batches_per_epoch": 4,
        "effective_epochs": 500.0,
        "frame_presentations": 160000,
        "presentations_per_fit_frame": 500.0,
        "evaluation_interval": 100,
        "optimizer": finalizer._expected_optimizer(),
        "one_direct_forward_backward_per_update": True,
        "gradient_accumulation_or_microbatching": False,
        "fixed_update_budget_consumed": True,
        "initial_state_sha256": finalizer.EXPECTED_INITIAL_STATE_SHA256[seed],
        "final_state_sha256": "f" * 64,
        "minibatch_indices": schedule,
        "minibatch_indices_sha256": finalizer._canonical_json_sha256(schedule),
        "learning_curve": curve,
        "terminal_fit_gate": _terminal(passes=passes),
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
        "holdouts_evaluated": passes,
    }


def _access(*, authorized: bool = False) -> dict:
    panels = {
        "fit": {
            "authorized": True,
            "artifact_hash_passes": 2,
            "image_hash_byte_open_events": 640,
            "shard_hash_byte_open_events": 40,
        }
    }
    for panel in finalizer.HOLDOUT_PANELS:
        shards = finalizer.EXPECTED_PANEL_ARTIFACT_COUNTS[panel]["shards"]
        if authorized:
            panels[panel] = {
                "authorized": True,
                "artifact_hash_passes": 2,
                "image_hash_byte_open_events": 640,
                "shard_hash_byte_open_events": 2 * shards,
                "dataset_access": {
                    "image_requests": 960,
                    "target_requests": 320,
                    "image_decode_events": 320,
                    "label_shard_npz_open_events": shards,
                    "model_calls": 80,
                    "model_output_frames": 960,
                },
            }
        else:
            panels[panel] = {
                "authorized": False,
                "artifact_hash_passes": 0,
                "image_hash_byte_open_events": 0,
                "shard_hash_byte_open_events": 0,
                "dataset_access": {
                    name: 0 for name in sorted(finalizer.EVENT_FIELDS)
                },
            }
    return {
        "panels": panels,
        "fit_dataset_totals": {
            "image_requests": 179200,
            "target_requests": 166400,
            "image_decode_events": 320,
            "label_shard_npz_open_events": 20,
        },
        "checkpoint_selection": {
            "image_byte_opens": 0,
            "label_shard_byte_opens": 0,
            "model_outputs": 0,
        },
        "probability_calibration": {
            "image_byte_opens": 0,
            "label_shard_byte_opens": 0,
            "model_outputs": 0,
        },
        "g2_evaluation": {
            "image_byte_opens": 0,
            "label_shard_byte_opens": 0,
            "model_outputs": 0,
        },
        "non_train_image_opens": 0,
        "non_train_label_shard_opens": 0,
        "non_train_model_outputs": 0,
    }


def _inputs(seed: int) -> dict:
    result = {}
    path_by_name = {
        "panel": finalizer.base.PANEL_PATH,
        "ladder_manifest": finalizer.base.LADDER_PATH,
        "v3_result": finalizer.base.V3_RESULT_PATH,
        "patch7_reference_result": finalizer.base.PATCH7_RESULT_PATH,
    }
    for name, raw in finalizer.EXPECTED_INPUTS.items():
        if name in finalizer.base.EXPECTED_INPUTS:
            file_hash, content_hash = raw
            path = path_by_name[name]
        else:
            path, file_hash, content_hash = raw
        result[name] = {"path": str(path.resolve()), "sha256": file_hash}
        if content_hash is not None:
            result[name]["content_sha256"] = content_hash
    result["seed_20260710_authorization"] = (
        None
        if seed == 20260710
        else {
            "path": str(finalizer.CANONICAL_RESULT_PATHS[20260710]),
            "sha256": "a" * 64,
        }
    )
    return result


def test_bound_evidence_and_transitive_runner_sources_validate() -> None:
    bound = finalizer._load_bound_evidence()
    assert len(bound["panels"]["fit"]) == 160
    assert bound["patch7_reference"]["schema"].endswith("patch7_reference_v1")
    assert finalizer._runner_source_hashes() == runner._source_hashes()


def test_finalizer_is_torch_free_and_does_not_import_the_runner() -> None:
    source = Path(finalizer.__file__).read_text()
    assert "import torch" not in source
    assert "from scripts import run_go2_categorical_radial_n32_v2" not in source


def test_exact_schedule_commitments_and_complete_epochs() -> None:
    for seed in finalizer.EXPECTED_SEEDS:
        schedule = runner.frozen_minibatch_schedule(320, 2000, seed)
        assert finalizer._validate_minibatches(schedule, seed=seed) == schedule
        assert finalizer._canonical_json_sha256(schedule) == (
            finalizer.SCHEDULE_SHA256[seed]
        )
    changed = runner.frozen_minibatch_schedule(320, 2000, 20260710)
    changed[0][0], changed[0][1] = changed[0][1], changed[0][0]
    with pytest.raises(ValueError, match="exact seeded"):
        finalizer._validate_minibatches(changed, seed=20260710)


def test_stage_validator_enforces_batch80_cosine_and_access(
    monkeypatch,
) -> None:
    stage = _stage()
    monkeypatch.setattr(finalizer.base, "_validate_panel_report", lambda *a, **k: {})
    monkeypatch.setattr(
        finalizer,
        "terminal_fit_gate_summary",
        lambda *_args: _terminal(),
    )
    validated, terminal = finalizer._validate_stage(
        stage,
        seed=20260710,
        expected_initial_state=finalizer.EXPECTED_INITIAL_STATE_SHA256[20260710],
        expected_controls={},
    )
    assert validated["batch_size"] == 80
    assert terminal["passes"] is False
    changed = copy.deepcopy(stage)
    changed["optimizer"]["learning_rate_schedule"]["final_learning_rate"] = 2e-5
    with pytest.raises(ValueError, match="optimizer/schedule"):
        finalizer._validate_stage(
            changed,
            seed=20260710,
            expected_initial_state=finalizer.EXPECTED_INITIAL_STATE_SHA256[20260710],
            expected_controls={},
        )
    changed = copy.deepcopy(stage)
    changed["optimizer"]["amsgrad"] = 0
    with pytest.raises(ValueError, match="JSON boolean"):
        finalizer._validate_stage(
            changed,
            seed=20260710,
            expected_initial_state=finalizer.EXPECTED_INITIAL_STATE_SHA256[20260710],
            expected_controls={},
        )
    changed = copy.deepcopy(stage)
    changed["optimizer"]["learning_rate_schedule"]["one_indexed"] = 1
    with pytest.raises(ValueError, match="JSON boolean"):
        finalizer._validate_stage(
            changed,
            seed=20260710,
            expected_initial_state=finalizer.EXPECTED_INITIAL_STATE_SHA256[20260710],
            expected_controls={},
        )


def test_access_validator_rejects_nonzero_or_noninteger_unauthorized_contact() -> None:
    access = _access(authorized=False)
    finalizer._validate_access(access, holdouts_authorized=False)
    changed = copy.deepcopy(access)
    changed["panels"]["same_scene_holdout"]["dataset_access"][
        "model_output_frames"
    ] = 1
    with pytest.raises(ValueError, match="conditional access"):
        finalizer._validate_access(changed, holdouts_authorized=False)
    changed = copy.deepcopy(access)
    changed["panels"]["cross_scene_holdout"]["dataset_access"][
        "image_requests"
    ] = False
    with pytest.raises(ValueError, match="JSON integer"):
        finalizer._validate_access(changed, holdouts_authorized=False)
    changed = copy.deepcopy(access)
    changed["panels"]["same_scene_holdout"]["authorized"] = 0
    with pytest.raises(ValueError, match="JSON boolean"):
        finalizer._validate_access(changed, holdouts_authorized=False)


def _failed_artifact(monkeypatch) -> tuple[dict, dict]:
    reference = {"bound": "reference"}
    sources = {"source": {"path": "/source", "sha256": "1" * 64}}
    monkeypatch.setattr(finalizer.base, "_validate_panel_report", lambda *a, **k: {})
    monkeypatch.setattr(
        finalizer,
        "terminal_fit_gate_summary",
        lambda *_args: _terminal(),
    )
    monkeypatch.setattr(
        finalizer.base, "_validate_patch7_reference", lambda value: value
    )
    monkeypatch.setattr(finalizer, "_runner_source_hashes", lambda: sources)
    monkeypatch.setattr(finalizer, "_validate_artifact_verification", lambda _v: None)
    stage = _stage()
    decision = per_seed_decision(stage, None)
    core = {
        "schema": finalizer.RESULT_SCHEMA,
        "authoritative": True,
        "aggregation_eligible": True,
        "promotion_eligible": False,
        "seed": 20260710,
        "created_at_utc": "2026-07-11T00:00:00+00:00",
        "completed_at_utc": "2026-07-11T00:01:00+00:00",
        "invocation": ["runner"],
        "execution": {
            "device": "cuda",
            "device_name": "test",
            "determinism": {
                "seed": 20260710,
                "requested": "strict_deterministic_algorithms",
                "effective": "strict_where_supported_warn_on_unsupported",
                "warn_only": True,
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
            "batch_size_frames": 80,
            "evaluation_target_batch_size": 4,
            "evaluation_combined_model_batch_size": 12,
            "evaluation_interval": 100,
            "stage_config": finalizer.AUTHORITATIVE_CONFIG,
            "effective_epochs": 500.0,
            "fp32_no_autocast_amp_compile_or_quantization": True,
        },
        "contract": {
            "path": str(finalizer.CONTRACT_PATH.resolve()),
            "sha256": finalizer.EXECUTION_BINDING_SHA256,
        },
        "inputs": _inputs(20260710),
        "source_hashes": sources,
        "git": {},
        "model": {
            "class": "CategoricalRadialPerceptionFullRay",
            "parameter_count": finalizer.REGISTERED_PARAMETER_COUNT,
            "initial_state_sha256": finalizer.EXPECTED_INITIAL_STATE_SHA256[
                20260710
            ],
        },
        "stages": {STAGE_NAME: stage},
        "patch7_reference": reference,
        "holdouts": None,
        "holdout_checks": None,
        "decision": decision,
        "artifact_verification": {},
        "access_ledger": _access(authorized=False),
        "categorical_radial_full_train_candidate_licensed": False,
    }
    return {**core, "content_sha256": finalizer._canonical_json_sha256(core)}, reference


def test_authoritative_validator_recomputes_failed_decision_and_strict_types(
    monkeypatch,
) -> None:
    artifact, reference = _failed_artifact(monkeypatch)
    validated = finalizer._validate_authoritative_result(
        artifact,
        expected_seed=20260710,
        expected_controls={"fit": {}},
        expected_patch7_reference=reference,
    )
    assert validated["decision"]["classification"] == "fit_gate_failed"
    changed = copy.deepcopy(artifact)
    changed["decision"]["favorable"] = 0
    core = dict(changed)
    core.pop("content_sha256")
    changed["content_sha256"] = finalizer._canonical_json_sha256(core)
    with pytest.raises(ValueError, match="JSON boolean"):
        finalizer._validate_authoritative_result(
            changed,
            expected_seed=20260710,
            expected_controls={"fit": {}},
            expected_patch7_reference=reference,
        )


def test_primary_authorization_rehashes_and_requires_favorable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "seed10.json"
    path.write_text(json.dumps({"value": 1}))
    digest = finalizer._sha256_file(path)
    monkeypatch.setitem(finalizer.CANONICAL_RESULT_PATHS, 20260710, path.resolve())
    monkeypatch.setattr(
        finalizer,
        "_load_bound_evidence",
        lambda: {
            "controls": {20260710: {}},
            "patch7_reference": {"reference": 1},
        },
    )
    monkeypatch.setattr(
        finalizer,
        "_validate_authoritative_result",
        lambda *_args, **_kwargs: {
            "decision": {"favorable": True},
            "source_hashes": {"source": 1},
            "patch7_reference": {"reference": 1},
        },
    )
    result = finalizer.validate_seed10_authorization(
        path,
        digest,
        expected_runner_sources={"source": 1},
        expected_patch7_reference={"reference": 1},
    )
    assert result["sha256"] == digest


def test_two_seed_aggregation_only_licenses_two_favorable_results() -> None:
    favorable = {"decision": {"favorable": True}}
    failed = {"decision": {"favorable": False}}
    assert finalizer._aggregate(favorable, favorable)[
        "categorical_radial_full_train_candidate_licensed"
    ] is True
    replication_failed = finalizer._aggregate(favorable, failed)
    assert replication_failed["classification"] == "two_seed_replication_failed"
    assert replication_failed[
        "categorical_radial_full_train_candidate_licensed"
    ] is False
