from __future__ import annotations

import copy
from pathlib import Path

import pytest

from lewm.benchmarks.go2_categorical_radial_n32_v4 import (
    FACTOR_OUTPUT_CONTRACT,
    FACTOR_OUTPUT_CONTRACT_SHA256,
    STAGE_NAME,
    STAGE_SCHEMA,
    per_seed_decision,
)
from scripts import finalize_go2_categorical_radial_n32_v4 as finalizer
from scripts import run_go2_categorical_radial_n32_v4 as runner


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
            "save_cpu_rng_after_determinism_then_replay_for_v2_and_v4_copy_131_"
            "same_shape_entries_leave_two_changed_head_tensors_at_pytorch_defaults"
        ),
        "v2_reference_initial_state_sha256": (
            finalizer.EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256[seed]
        ),
        "candidate_initial_state_sha256": finalizer.EXPECTED_INITIAL_STATE_SHA256[seed],
        "v2_reference_parameter_count": finalizer.REGISTERED_V2_PARAMETER_COUNT,
        "candidate_parameter_count": finalizer.REGISTERED_PARAMETER_COUNT,
        "state_key_sets_identical": True,
        "same_shape_entry_count": 131,
        "same_shape_entries_bit_identical": True,
        "only_shape_changed_state_keys": sorted(finalizer.EXPECTED_SHAPE_CHANGES),
        "shape_changes": copy.deepcopy(finalizer.EXPECTED_SHAPE_CHANGES),
        "shape_changed_head_tensors_left_at_deterministic_pytorch_default": True,
        "class_prior_bias_matching_applied": False,
        "analytic_v2_head_transform_applied": False,
        "zero_initialization_applied": False,
        "trained_v2_weight_loaded": False,
        "trained_v3_weight_loaded": False,
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
        "dataset_role_policy": {
            "current_physical_dataset_role": "train",
            "current_physical_dataset_role_governs_access": True,
            "legacy_rollout_split_is_provenance_only": True,
            "legacy_rollout_split_used_to_filter_rank_calibrate_or_select": False,
        },
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
            "class": finalizer.MODEL_CLASS,
            "parameter_count": finalizer.REGISTERED_PARAMETER_COUNT,
            "token_feature_dim": finalizer.REGISTERED_TOKEN_FEATURE_DIM,
            "context_dim": finalizer.REGISTERED_CONTEXT_DIM,
            "parameter_delta_from_v2": finalizer.REGISTERED_PARAMETER_DELTA,
            "factor_order": FACTOR_OUTPUT_CONTRACT["raw_factor_order"],
            "output_semantics": finalizer.MODEL_OUTPUT_SEMANTICS,
            "factor_output_contract": FACTOR_OUTPUT_CONTRACT,
            "factor_output_contract_sha256": FACTOR_OUTPUT_CONTRACT_SHA256,
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
        "categorical_radial_full_train_candidate_licensed": False,
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
    assert "from scripts import run_go2_categorical_radial_n32_v4" not in source


def test_stage_validator_enforces_v4_identity_and_exact_v2_schedule(
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
    for name, value in (
        ("same_shape_entries_bit_identical", False),
        ("shape_changed_head_tensors_left_at_deterministic_pytorch_default", False),
        ("class_prior_bias_matching_applied", True),
        ("analytic_v2_head_transform_applied", True),
        ("zero_initialization_applied", True),
        ("trained_v2_weight_loaded", True),
        ("trained_v3_weight_loaded", True),
    ):
        changed = copy.deepcopy(proof)
        changed[name] = value
        mutations.append(changed)
    changed = copy.deepcopy(proof)
    changed["only_shape_changed_state_keys"].append("extra.weight")
    mutations.append(changed)
    changed = copy.deepcopy(proof)
    changed["shape_changes"]["polar_head.bias"]["v4_shape"] = [3]
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


def test_access_rejects_holdout_or_role_policy_tampering() -> None:
    access = _access()
    finalizer._validate_access(access, holdouts_authorized=False)
    changed = copy.deepcopy(access)
    changed["panels"]["cross_scene_holdout"]["dataset_access"][
        "model_output_frames"
    ] = 1
    with pytest.raises(ValueError, match="conditional access"):
        finalizer._validate_access(changed, holdouts_authorized=False)
    changed = copy.deepcopy(access)
    changed["dataset_role_policy"][
        "legacy_rollout_split_used_to_filter_rank_calibrate_or_select"
    ] = True
    with pytest.raises(ValueError, match="policy drift"):
        finalizer._validate_access(changed, holdouts_authorized=False)


def test_authoritative_validator_recomputes_decision_and_rejects_tampering(
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
    mutations = []
    changed = copy.deepcopy(artifact)
    changed["model"]["initialization_comparability"][
        "same_shape_entries_bit_identical"
    ] = False
    mutations.append((changed, "proof"))
    changed = copy.deepcopy(artifact)
    changed["model"]["factor_output_contract_sha256"] = "0" * 64
    mutations.append((changed, "factors"))
    changed = copy.deepcopy(artifact)
    changed["runtime_ready"] = True
    mutations.append((changed, "forbidden license"))
    changed = copy.deepcopy(artifact)
    changed["decision"]["favorable"] = 0
    mutations.append((changed, "JSON boolean"))
    changed = copy.deepcopy(artifact)
    changed["source_hashes"]["source"]["sha256"] = "2" * 64
    mutations.append((changed, "source provenance"))
    for changed, message in mutations:
        _rehash(changed)
        with pytest.raises(ValueError, match=message):
            finalizer._validate_authoritative_result(
                changed,
                expected_seed=20260710,
                expected_controls={"fit": {}},
                expected_patch7_reference=reference,
            )


def test_smoke_schema_is_rejected_even_if_other_fields_claim_authority(
    monkeypatch,
) -> None:
    artifact, reference = _failed_artifact(monkeypatch)
    artifact["schema"] = finalizer.SMOKE_RESULT_SCHEMA
    _rehash(artifact)
    with pytest.raises(ValueError, match="rejects smoke"):
        finalizer._validate_authoritative_result(
            artifact,
            expected_seed=20260710,
            expected_controls={"fit": {}},
            expected_patch7_reference=reference,
        )


def test_two_seed_aggregation_only_licenses_categorical_candidate() -> None:
    favorable = {"decision": {"favorable": True}}
    failed = {"decision": {"favorable": False}}
    passed = finalizer._aggregate(favorable, favorable)
    assert passed["categorical_radial_full_train_candidate_licensed"] is True
    assert passed["runtime_ready"] is False
    assert passed["g2_licensed"] is False
    replication_failed = finalizer._aggregate(favorable, failed)
    assert replication_failed["classification"] == "two_seed_replication_failed"
    assert replication_failed["categorical_radial_full_train_candidate_licensed"] is False
