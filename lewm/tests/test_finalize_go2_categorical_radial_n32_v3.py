from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lewm.benchmarks.go2_categorical_radial_n32_v3 import (
    STAGE_NAME,
    STAGE_SCHEMA,
    per_seed_decision,
)
from scripts import finalize_go2_categorical_radial_n32_v3 as finalizer
from scripts import run_go2_categorical_radial_n32_v3 as runner


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
        "optimizer": finalizer.v2._expected_optimizer(),
        "one_direct_forward_backward_per_update": True,
        "gradient_accumulation_or_microbatching": False,
        "fixed_update_budget_consumed": True,
        "initial_state_sha256": finalizer.EXPECTED_INITIAL_STATE_SHA256[seed],
        "final_state_sha256": "f" * 64,
        "minibatch_indices": schedule,
        "minibatch_indices_sha256": finalizer._canonical_json_sha256(schedule),
        "learning_curve": [
            {
                "step": step,
                "learning_rate": finalizer.v2._cosine_learning_rate(step, 2000),
                "batch_loss": 0.1,
                "gradient_norm_before_clip": 0.2,
                "fit_panel": {},
            }
            for step in range(100, 2001, 100)
        ],
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


def _proof(seed: int = 20260710) -> dict:
    return {
        "construction": (
            "save_cpu_rng_after_determinism_then_replay_for_width24_and_width32_"
            "and_copy_every_same_shape_entry"
        ),
        "v2_reference_initial_state_sha256": (
            finalizer.EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256[seed]
        ),
        "candidate_initial_state_sha256": (
            finalizer.EXPECTED_INITIAL_STATE_SHA256[seed]
        ),
        "v2_reference_parameter_count": finalizer.REGISTERED_V2_PARAMETER_COUNT,
        "candidate_parameter_count": finalizer.REGISTERED_PARAMETER_COUNT,
        "state_key_sets_identical": True,
        "same_shape_entry_count": 130,
        "same_shape_entries_bit_identical": True,
        "only_shape_changed_state_keys": sorted(finalizer.EXPECTED_SHAPE_CHANGES),
        "shape_changes": copy.deepcopy(finalizer.EXPECTED_SHAPE_CHANGES),
        "trained_v2_weight_loaded": False,
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
        panels[panel] = (
            {
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
            if authorized
            else {
                "authorized": False,
                "artifact_hash_passes": 0,
                "image_hash_byte_open_events": 0,
                "shard_hash_byte_open_events": 0,
                "dataset_access": {
                    name: 0 for name in sorted(finalizer.EVENT_FIELDS)
                },
            }
        )
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
    for name, (path, file_hash, content_hash) in finalizer.EXPECTED_INPUTS.items():
        record = {"path": str(path.resolve()), "sha256": file_hash}
        if content_hash is not None:
            record["content_sha256"] = content_hash
        result[name] = record
    result["seed_20260710_authorization"] = (
        None
        if seed == 20260710
        else {
            "path": str(finalizer.CANONICAL_RESULT_PATHS[20260710]),
            "sha256": "a" * 64,
        }
    )
    return result


def _failed_artifact(monkeypatch) -> tuple[dict, dict]:
    reference = {"bound": "reference"}
    sources = {"source": {"path": "/source", "sha256": "1" * 64}}
    monkeypatch.setattr(finalizer.base, "_validate_panel_report", lambda *a, **k: {})
    monkeypatch.setattr(
        finalizer.v2,
        "terminal_fit_gate_summary",
        lambda *_args: _terminal(),
    )
    monkeypatch.setattr(
        finalizer.base, "_validate_patch7_reference", lambda value: value
    )
    monkeypatch.setattr(finalizer, "_runner_source_hashes", lambda: sources)
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
            "cpu_rng_state_captured_immediately_after_determinism": True,
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
            "class": "CategoricalRadialPerceptionFullRayToken32",
            "parameter_count": finalizer.REGISTERED_PARAMETER_COUNT,
            "token_feature_dim": 32,
            "context_dim": 64,
            "parameter_delta_from_v2": 4104,
            "initial_state_sha256": finalizer.EXPECTED_INITIAL_STATE_SHA256[
                20260710
            ],
            "initialization_comparability": _proof(),
        },
        "stages": {STAGE_NAME: stage},
        "patch7_reference": reference,
        "holdouts": None,
        "holdout_checks": None,
        "decision": decision,
        "artifact_verification": {
            "fit_verified_before_access": True,
            "holdouts_verified_only_after_terminal_fit_pass": True,
            "evidence_hashes": {
                str(path.resolve()): digest
                for path, digest in finalizer.BOUND_EVIDENCE.items()
            },
        },
        "access_ledger": _access(),
        "shared_jepa_full_train_candidate_licensed": False,
        "runtime_ready": False,
        "g2_licensed": False,
        "g3_licensed": False,
    }
    return {
        **core,
        "content_sha256": finalizer._canonical_json_sha256(core),
    }, reference


def _rehash(artifact: dict) -> None:
    core = dict(artifact)
    core.pop("content_sha256", None)
    artifact["content_sha256"] = finalizer._canonical_json_sha256(core)


def test_runner_and_torch_free_finalizer_bind_the_same_source_map() -> None:
    assert finalizer._runner_source_hashes() == runner._source_hashes()
    source = Path(finalizer.__file__).read_text()
    assert "import torch" not in source
    assert "from scripts import run_go2_categorical_radial_n32_v3" not in source


def test_stage_validator_enforces_v3_identity_and_exact_v2_schedule(
    monkeypatch,
) -> None:
    monkeypatch.setattr(finalizer.base, "_validate_panel_report", lambda *a, **k: {})
    monkeypatch.setattr(
        finalizer.v2, "terminal_fit_gate_summary", lambda *_args: _terminal()
    )
    stage = _stage()
    validated, terminal = finalizer._validate_stage(
        stage,
        seed=20260710,
        expected_initial_state=finalizer.EXPECTED_INITIAL_STATE_SHA256[20260710],
        expected_controls={},
    )
    assert validated["stage"] == STAGE_NAME
    assert terminal["passes"] is False
    changed = copy.deepcopy(stage)
    changed["schema"] = finalizer.v2.STAGE_SCHEMA
    with pytest.raises(ValueError, match="identity"):
        finalizer._validate_stage(
            changed,
            seed=20260710,
            expected_initial_state=finalizer.EXPECTED_INITIAL_STATE_SHA256[20260710],
            expected_controls={},
        )
    changed = copy.deepcopy(stage)
    changed["minibatch_indices"][0][0], changed["minibatch_indices"][0][1] = (
        changed["minibatch_indices"][0][1],
        changed["minibatch_indices"][0][0],
    )
    with pytest.raises(ValueError, match="exact seeded"):
        finalizer._validate_stage(
            changed,
            seed=20260710,
            expected_initial_state=finalizer.EXPECTED_INITIAL_STATE_SHA256[20260710],
            expected_controls={},
        )


def test_initialization_proof_rejects_each_fairness_escape() -> None:
    proof = _proof()
    assert finalizer._validate_initialization_proof(proof, seed=20260710) == proof
    mutations = []
    changed = copy.deepcopy(proof)
    changed["same_shape_entries_bit_identical"] = False
    mutations.append(changed)
    changed = copy.deepcopy(proof)
    changed["only_shape_changed_state_keys"].append("extra.weight")
    mutations.append(changed)
    changed = copy.deepcopy(proof)
    changed["shape_changes"]["token_projection.bias"]["v3_shape"] = [31]
    mutations.append(changed)
    changed = copy.deepcopy(proof)
    changed["trained_v2_weight_loaded"] = True
    mutations.append(changed)
    changed = copy.deepcopy(proof)
    changed["candidate_initial_state_sha256"] = "0" * 64
    mutations.append(changed)
    for mutation in mutations:
        with pytest.raises(ValueError, match="proof"):
            finalizer._validate_initialization_proof(mutation, seed=20260710)
    changed = copy.deepcopy(proof)
    changed["trained_v2_weight_loaded"] = 0
    with pytest.raises(ValueError, match="JSON boolean"):
        finalizer._validate_initialization_proof(changed, seed=20260710)


def test_access_rejects_any_unauthorized_holdout_contact() -> None:
    access = _access()
    finalizer._validate_access(access, holdouts_authorized=False)
    access["panels"]["cross_scene_holdout"]["dataset_access"][
        "model_output_frames"
    ] = 1
    with pytest.raises(ValueError, match="conditional access"):
        finalizer._validate_access(access, holdouts_authorized=False)


def test_authoritative_validator_recomputes_decision_and_rejects_licenses(
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
    changed["model"]["initialization_comparability"][
        "same_shape_entries_bit_identical"
    ] = False
    _rehash(changed)
    with pytest.raises(ValueError, match="proof"):
        finalizer._validate_authoritative_result(
            changed,
            expected_seed=20260710,
            expected_controls={"fit": {}},
            expected_patch7_reference=reference,
        )
    changed = copy.deepcopy(artifact)
    changed["runtime_ready"] = True
    _rehash(changed)
    with pytest.raises(ValueError, match="forbidden license"):
        finalizer._validate_authoritative_result(
            changed,
            expected_seed=20260710,
            expected_controls={"fit": {}},
            expected_patch7_reference=reference,
        )
    changed = copy.deepcopy(artifact)
    changed["decision"]["favorable"] = 0
    _rehash(changed)
    with pytest.raises(ValueError, match="JSON boolean"):
        finalizer._validate_authoritative_result(
            changed,
            expected_seed=20260710,
            expected_controls={"fit": {}},
            expected_patch7_reference=reference,
        )
    changed = copy.deepcopy(artifact)
    changed["source_hashes"]["source"]["sha256"] = "2" * 64
    _rehash(changed)
    with pytest.raises(ValueError, match="source provenance"):
        finalizer._validate_authoritative_result(
            changed,
            expected_seed=20260710,
            expected_controls={"fit": {}},
            expected_patch7_reference=reference,
        )


def test_two_seed_aggregation_only_licenses_shared_jepa_candidate() -> None:
    favorable = {"decision": {"favorable": True}}
    failed = {"decision": {"favorable": False}}
    passed = finalizer._aggregate(favorable, favorable)
    assert passed["shared_jepa_full_train_candidate_licensed"] is True
    assert passed["runtime_ready"] is False
    assert passed["g2_licensed"] is False
    replication_failed = finalizer._aggregate(favorable, failed)
    assert replication_failed["classification"] == "two_seed_replication_failed"
    assert replication_failed["shared_jepa_full_train_candidate_licensed"] is False
