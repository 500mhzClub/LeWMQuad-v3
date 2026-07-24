from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py"
)
_MODULES_BEFORE = set(sys.modules)
_SPEC = importlib.util.spec_from_file_location(
    "_test_go2_rgb_causal_temporal_perception_v1_contract",
    CONTRACT_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)
_MODULES_IMPORTED_BY_CONTRACT = set(sys.modules) - _MODULES_BEFORE


def _evaluation(
    *,
    complete: int = 1,
    passed: int = 98,
    shortfall: float = 41.0,
    pixel: float = 0.82,
    ground: float = 0.65,
    depth: float = 0.97,
) -> dict[str, Any]:
    return {
        "complete_physical_scope_count": complete,
        "margin_count": 189,
        "passed_margin_count": passed,
        "total_shortfall": shortfall,
        "worst_margin": -1.0,
        "rough_motion": {
            "pixel_balanced_accuracy": pixel,
            "ground_balanced_accuracy": ground,
            "depth_p95_m": depth,
        },
    }


def _self_hashed(core: dict[str, Any]) -> dict[str, Any]:
    return contract.with_content_sha256(core)


def _runtime_leaf(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": contract.RUNTIME_FILE_SHA256[path],
        "content_sha256": contract.RUNTIME_CONTENT_SHA256[path],
        "byte_count": contract.RUNTIME_BYTE_COUNTS.get(path, 1),
    }


def _runtime_inputs() -> dict[str, Any]:
    return {
        "raw": {
            "root": contract.RAW_ROOT_RELATIVE_PATH,
            "manifest": _runtime_leaf(contract.RAW_MANIFEST_RELATIVE_PATH),
            "audit": _runtime_leaf(contract.RAW_AUDIT_RELATIVE_PATH),
            "role_counts": {
                "train": dict(contract.TRAIN_ROLE_COUNTS),
                "checkpoint_selection":
                    dict(contract.SELECTION_ROLE_COUNTS),
            },
            "grant": {
                "allowed_roles": ["train", "checkpoint_selection"],
                "allowed_operations": [
                    "development_rgb_decode",
                    "causal_temporal_perception_training",
                    "physical_checkpoint_selection",
                ],
                "calibration_g2_navigation_heldout_or_production_use": False,
            },
        },
        "camera": {
            "root": contract.N320_ROOT_RELATIVE_PATH,
            "gate": _runtime_leaf(contract.N320_GATE_RELATIVE_PATH),
            "checkpoint":
                _runtime_leaf(contract.N320_CHECKPOINT_RELATIVE_PATH),
            "seed": 20_260_710,
            "fit_size": 320,
            "updates": 40_000,
            "gate_must_pass_all_checks": 26,
        },
        "schedule": _runtime_leaf(contract.SCHEDULE_RELATIVE_PATH),
    }


def _expected_review_sources() -> dict[str, str]:
    paths = {
        *contract.SOURCE_PATHS,
        *contract.SOURCE_REVIEW_ADDITIONAL_PATHS,
    }
    result = {path: "1" * 64 for path in sorted(paths)}
    result.update(contract.FROZEN_SOURCE_SHA256)
    result[contract.PREREGISTRATION_RELATIVE_PATH] = (
        contract.PREREGISTRATION_FILE_SHA256
    )
    result[contract.PREREGISTRATION_REVIEW_RELATIVE_PATH] = (
        contract.PREREGISTRATION_REVIEW_FILE_SHA256
    )
    return result


def _review_value(
    expected_sources: dict[str, str],
    *,
    reviewer: str = "/root/independent_source_reviewer",
) -> dict[str, Any]:
    return _self_hashed({
        "schema": contract.REVIEW_SCHEMA,
        "status": "PASS_SOURCE_ONLY",
        "implementation_author": contract.CONTRACT_AUTHOR,
        "reviewer": reviewer,
        "reviewed_sources": dict(expected_sources),
        "preregistration": contract.preregistration_binding(),
        "frozen_source_bindings": dict(contract.FROZEN_SOURCE_SHA256),
        "science_contract": contract.science_contract(),
        "lifecycle_contract": contract.lifecycle_contract(),
        "source_only": True,
        "deferred_runtime_inputs_opened": [],
        "findings": [],
        "authority": dict(contract.REVIEW_AUTHORITY),
    })


def _review_binding(review: dict[str, Any]) -> dict[str, Any]:
    raw = contract.canonical_json_bytes(review) + b"\n"
    return contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        raw,
        content_sha256=review["content_sha256"],
    )


def _authorization_value(
    review_binding: dict[str, Any],
    *,
    authorizer: str = "/root/independent_execution_authorizer",
) -> dict[str, Any]:
    return _self_hashed({
        "schema": contract.AUTHORIZATION_SCHEMA,
        "status": "AUTHORIZED_ONE_EXACT_BOUNDED_PROBE",
        "authorizer": authorizer,
        "independent_source_review": dict(review_binding),
        "preregistration": contract.preregistration_binding(),
        "runtime_inputs": _runtime_inputs(),
        "hardware": contract.hardware_contract(),
        "experiment": contract.science_contract(),
        "lifecycle": contract.lifecycle_contract(),
        "authority": dict(contract.EXECUTION_AUTHORITY),
    })


def _source_manifest_value() -> dict[str, Any]:
    paths = sorted({
        *contract.SOURCE_PATHS,
        *contract.SOURCE_MANIFEST_ENTRYPOINTS,
        *contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES,
    })
    bindings = [
        {
            "path": path,
            "file_sha256":
                contract.FROZEN_SOURCE_SHA256.get(path, "2" * 64),
            "byte_count": 1,
        }
        for path in paths
    ]
    return _self_hashed({
        "schema": contract.SOURCE_MANIFEST_SCHEMA,
        "date": "2026-07-24",
        "status": "SOURCE_ONLY_RECURSIVE_CLOSURE",
        "authority":
            "source_closure_only_no_generated_input_checkpoint_training_gpu_"
            "qualification_g2_navigation_heldout_production_or_promotion_"
            "authority",
        "entrypoints": list(contract.SOURCE_MANIFEST_ENTRYPOINTS),
        "forced_dynamic_sources":
            list(contract.SOURCE_MANIFEST_FORCED_DYNAMIC_SOURCES),
        "source_count": len(paths),
        "source_paths": paths,
        "source_bindings": bindings,
        "source_bindings_sha256":
            contract.canonical_json_sha256(bindings),
        "excluded_runtime_categories":
            list(contract.SOURCE_MANIFEST_EXCLUDED_RUNTIME_CATEGORIES),
        "consumed_adaptation_runner_source_count": 0,
        "generated_input_open_count": 0,
        "tensor_checkpoint_open_count": 0,
        "sealed_or_heldout_open_count": 0,
        "whole_tree_export_authorized": False,
    })


def test_import_is_stdlib_only_and_preregistration_bytes_are_exact() -> None:
    imported_roots = {
        name.partition(".")[0] for name in _MODULES_IMPORTED_BY_CONTRACT
    }
    assert imported_roots.isdisjoint({
        "torch",
        "numpy",
        "PIL",
        "cv2",
        "jax",
        "tensorflow",
    })
    assert contract.PREREGISTRATION_COMMIT == (
        "3e30b8ae9dbdfeafd0f62bfc4243cece7a885d95"
    )
    preregistration = ROOT / contract.PREREGISTRATION_RELATIVE_PATH
    review = ROOT / contract.PREREGISTRATION_REVIEW_RELATIVE_PATH
    preregistration_raw = preregistration.read_bytes()
    review_raw = review.read_bytes()
    assert len(preregistration_raw) == contract.PREREGISTRATION_BYTE_COUNT
    assert hashlib.sha256(preregistration_raw).hexdigest() == (
        contract.PREREGISTRATION_FILE_SHA256
    )
    assert len(review_raw) == contract.PREREGISTRATION_REVIEW_BYTE_COUNT
    assert hashlib.sha256(review_raw).hexdigest() == (
        contract.PREREGISTRATION_REVIEW_FILE_SHA256
    )
    review_value = json.loads(review_raw)
    review_core = dict(review_value)
    declared = review_core.pop("content_sha256")
    assert declared == contract.PREREGISTRATION_REVIEW_CONTENT_SHA256
    assert contract.canonical_json_sha256(review_core) == declared
    assert review_value["verdict"] == "PASS"
    assert contract.preregistration_binding()["independent_review"][
        "verdict"
    ] == "PASS"


def test_additive_paths_and_fresh_output_root_are_exact() -> None:
    assert contract.ADDITIVE_SOURCE_PATHS == (
        "lewm/benchmarks/go2_rgb_causal_temporal_perception_v1.py",
        "lewm/models/"
        "shared_observable_camera_ray_jepa_v5_multires_temporal_v1.py",
        "scripts/run_go2_rgb_causal_temporal_perception_v1.py",
        "scripts/launch_go2_rgb_causal_temporal_perception_v1.py",
        "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_contract.py",
        "lewm/tests/"
        "test_shared_observable_camera_ray_jepa_v5_multires_temporal_v1.py",
        "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_runner.py",
        "lewm/tests/test_go2_rgb_causal_temporal_perception_v1_evaluator.py",
        "lewm/tests/"
        "test_go2_rgb_causal_temporal_perception_v1_receipt_boundary.py",
        "scripts/check_go2_rgb_causal_temporal_perception_v1_source_closure.py",
        "lewm/tests/"
        "test_go2_rgb_causal_temporal_perception_v1_source_closure.py",
    )
    assert contract.MODEL_FAMILY == (
        "shared_observable_camera_ray_jepa_v5_multires_temporal_v1"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "/rgb_causal_temporal_perception_probe_v1"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH not in (
        contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
    )
    assert len(contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS) == 11
    assert len(set(contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS)) == 11
    assert sum(
        "rgb_multiresolution_perception_probe" in item
        for item in contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
    ) == 4
    assert sum(
        "protected_camera_adaptation" in item
        for item in contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
    ) == 7
    assert all(
        item.startswith(
            ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        )
        for item in contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
    )


def test_source_freeze_is_external_noncyclic_and_fails_closed_when_absent(
    tmp_path: Path,
) -> None:
    for cyclic_name in (
        "MODEL_FILE_SHA256",
        "SOURCE_MANIFEST_FILE_SHA256",
        "SOURCE_REVIEW_FILE_SHA256",
        "EXECUTION_AUTHORIZATION_FILE_SHA256",
    ):
        assert cyclic_name not in vars(contract)
    assert set(contract.FROZEN_SOURCE_SHA256) == set(
        contract.REUSED_SOURCE_PATHS
    )
    assert set(contract.FROZEN_SOURCE_SHA256).isdisjoint(
        contract.ADDITIVE_SOURCE_PATHS
    )
    assert contract.SOURCE_REVIEW_ADDITIONAL_PATHS == (
        contract.SOURCE_MANIFEST_RELATIVE_PATH,
        contract.PREREGISTRATION_RELATIVE_PATH,
        contract.PREREGISTRATION_REVIEW_RELATIVE_PATH,
    )
    status = contract.source_freeze_status(tmp_path)
    assert status["ready"] is False
    assert status["manifest_present"] is False
    assert status["manifest_error"] == "absent"
    with pytest.raises(PermissionError, match="freeze is incomplete"):
        contract.require_source_freeze(tmp_path)
    with pytest.raises(FileNotFoundError):
        contract.current_source_bindings(tmp_path)
    assert set(contract.SOURCE_ONLY_AUTHORITY.values()) == {False}
    assert set(contract.DOWNSTREAM_DENIALS.values()) == {False}


def test_external_source_manifest_is_exact_and_rejects_missing_forced_root() -> None:
    manifest = _source_manifest_value()
    raw = json.dumps(manifest, ensure_ascii=True).encode("ascii")
    assert contract.validate_source_manifest(raw) == manifest

    tampered = dict(manifest)
    omitted = contract.SOURCE_CLOSURE_BASE_CHECKER_RELATIVE_PATH
    tampered["source_paths"] = [
        path for path in manifest["source_paths"] if path != omitted
    ]
    tampered["source_bindings"] = [
        row for row in manifest["source_bindings"] if row["path"] != omitted
    ]
    tampered["source_count"] -= 1
    tampered["source_bindings_sha256"] = contract.canonical_json_sha256(
        tampered["source_bindings"]
    )
    tampered.pop("content_sha256")
    tampered = _self_hashed(tampered)
    with pytest.raises(PermissionError, match="path order or closure"):
        contract.validate_source_manifest(
            json.dumps(tampered, ensure_ascii=True).encode("ascii")
        )


def test_review_and_authorization_validate_exact_external_bindings() -> None:
    sources = _expected_review_sources()
    review = _review_value(sources)
    assert contract.validate_review(
        review,
        expected_sources=sources,
    ) == review
    binding = _review_binding(review)
    authorization = _authorization_value(binding)
    assert contract.validate_authorization(
        authorization,
        review_binding=binding,
        reviewer=review["reviewer"],
    ) == authorization

    conflicted = dict(review)
    conflicted["reviewer"] = contract.CONTRACT_AUTHOR
    conflicted.pop("content_sha256")
    conflicted = _self_hashed(conflicted)
    with pytest.raises(PermissionError, match="did not pass"):
        contract.validate_review(conflicted, expected_sources=sources)

    expanded = json.loads(json.dumps(authorization))
    expanded["runtime_inputs"]["raw"]["grant"][
        "calibration_g2_navigation_heldout_or_production_use"
    ] = True
    expanded.pop("content_sha256")
    expanded = _self_hashed(expanded)
    with pytest.raises(PermissionError, match="raw runtime authority"):
        contract.validate_authorization(
            expanded,
            review_binding=binding,
            reviewer=review["reviewer"],
        )


def test_exact_temporal_mechanism_and_parameter_identity() -> None:
    science = contract.science_contract()
    temporal = science["temporal_mechanism"]
    assert science["one_science_delta"] == (
        "pure_visual_fixed_lag_token_difference_residual_only"
    )
    assert temporal["inputs"] == [
        "previous_raw_visual_tokens_at_fixed_lag",
        "current_raw_visual_tokens",
        "history_valid",
    ]
    assert {
        "requested_primitive",
        "median_commanded_delta",
        "executed_command",
        "realized_simulator_se2",
        "exact_simulator_pose",
    }.issubset(temporal["forbidden_inputs"])
    assert temporal["lag_s"] == 0.5
    assert temporal["tick_count"] == 5
    assert temporal["tick_s"] == 0.10
    assert temporal["same_environment_episode_and_reset_required"] is True
    assert temporal["missing_irregular_or_reset_history_fails_cold"] is True
    assert temporal["persistent_history_buffer_state"] is False
    assert temporal["state_prefix"] == "evidence_head.temporal_residual."
    assert science["initialization"] == {
        "base_seed": 20_260_712,
        "decoder_local_cpu_seed": 20_260_724,
        "temporal_local_cpu_seed": 20_260_725,
        "n320_only_tensor_input": True,
        "permitted_copies": ["encoder", "pixel_head", "ground_head"],
        "temporal_entry_copy_count": 0,
        "predecessor_dense_decoder_copy_count": 0,
        "temporal_final_projection_zero": True,
        "caller_cpu_rng_restored": True,
        "hard_sync_count": 1,
        "rejected_checkpoint_open_count": 0,
    }
    assert contract.EXPECTED_PARAMETER_COUNTS == {
        "evidence_head": 355_849,
        "encoder": 2_747_520,
    }
    assert contract.EXPECTED_PARAMETER_TENSOR_COUNTS == {
        "evidence_head": 31,
        "encoder": 78,
    }
    assert contract.TEMPORAL_PARAMETER_COUNT == 3_160
    assert contract.TEMPORAL_PARAMETER_TENSOR_COUNT == 5
    assert contract.EXPECTED_TOTAL_TRAINABLE_PARAMETER_COUNT == 3_103_369
    assert contract.EXPECTED_TOTAL_TRAINABLE_PARAMETER_TENSOR_COUNT == 109
    assert (
        contract.EVIDENCE_HEAD_PARAMETER_CEILING
        - contract.EXPECTED_PARAMETER_COUNTS["evidence_head"]
        == 2_144
    )


def test_roles_pair_aware_population_and_no_calibration_or_jepa() -> None:
    assert contract.TRAIN_ROLE_COUNTS == {
        "pairs": 4_262,
        "unique_endpoints": 7_777,
        "scenes": 72,
    }
    assert contract.SELECTION_ROLE_COUNTS == {
        "pairs": 495,
        "unique_endpoints": 924,
        "warm_endpoints": 495,
        "cold_endpoints": 429,
        "both_roles": 66,
        "ambiguous_predecessors": 0,
        "scenes": 8,
    }
    selection = contract.SELECTION_ROLE_COUNTS
    assert (
        selection["warm_endpoints"] + selection["cold_endpoints"]
        == selection["unique_endpoints"]
    )
    science = contract.science_contract()
    assert science["evaluation"]["primary_population"] == (
        "924_unique_endpoints"
    )
    assert science["evaluation"]["primary_margin_count"] == 189
    assert science["evaluation"]["scope_count"] == 9
    assert science["evaluation"]["warm_only_view"] == "informational_only"
    assert science["evaluation"]["warm_only_may_control_checkpoint"] is False
    assert science["data"]["probability_calibration_open_count"] == 0
    assert science["calibration_authorized"] is False
    assert science["jepa_objective_count"] == 0
    assert science["jepa_backward_count"] == 0
    assert science["prior_runtime_output_open_count"] == 0


def test_unchanged_physical_evaluator_produces_nine_scopes_189_margins() -> None:
    metrics = {
        **{
            name: 1.0
            for name in contract.PHYSICAL_LOWER_THRESHOLDS
        },
        **{
            name: 0.0
            for name in contract.PHYSICAL_UPPER_THRESHOLDS
        },
        "distance_group_balanced_accuracy": [1.0] * 6,
        "present_class_recall": {
            "free": 1.0,
            "occupied": 1.0,
            "unknown": 1.0,
        },
    }
    scopes = {
        scope: {
            **metrics,
            "distance_group_balanced_accuracy":
                list(metrics["distance_group_balanced_accuracy"]),
            "present_class_recall":
                dict(metrics["present_class_recall"]),
        }
        for scope in contract.SCOPES
    }
    evaluation = contract.evaluate_physical_scopes(scopes)
    assert tuple(evaluation["scope_evaluations"]) == contract.SCOPES
    assert evaluation["complete_physical_scope_count"] == 9
    assert evaluation["margin_count"] == 189
    assert evaluation["passed_margin_count"] == 189
    assert evaluation["total_shortfall"] == 0.0


def test_schedule_and_operation_caps_are_exact() -> None:
    assert contract.CHECKPOINT_UPDATES == (100, 400, 1_000)
    assert contract.MAXIMUM_PRESENTATIONS == 16_000
    assert contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256 == {
        100:
            "9000f08c11dd5fb4feef72370e9fbcd2ae9b9858162529fa118eb289d9645c51",
        400:
            "6e7e5cc766c0a768b5771181cfaf2583598c1c22e5d4fc19e6ff1b245a5c8f92",
        1_000:
            "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528",
    }
    assert contract.operation_counts(
        1_000,
        (100, 400, 1_000),
    ) == {
        "maximum_optimizer_updates": 1_000,
        "complete_optimizer_updates": 1_000,
        "maximum_pair_index_presentations": 16_000,
        "pair_index_presentations": 16_000,
        "microbatch_size": 4,
        "microbatches_per_update": 4,
        "camera_objective_count": 4_000,
        "backward_call_count": 4_000,
        "head_clip_invocation_count": 1_000,
        "encoder_clip_invocation_count": 1_000,
        "global_clip_invocation_count": 0,
        "optimizer_construction_count": 1,
        "checkpoint_selection_evaluation_count": 3,
        "checkpoint_selection_evaluation_updates": [100, 400, 1_000],
        "observer_evaluation_rerun_count": 0,
        "jepa_objective_count": 0,
        "jepa_backward_count": 0,
        "ema_update_count_after_initial_hard_sync": 0,
        "probability_calibration_open_count": 0,
        "prior_runtime_output_open_count": 0,
    }
    empty = contract.empty_partial_operation_counts()
    assert contract.validate_partial_operation_counts(empty) == empty
    assert empty["probability_calibration_open_count"] == 0
    assert empty["prior_runtime_output_open_count"] == 0


def test_six_part_terminal_gate_and_equality_behavior_are_exact() -> None:
    assert contract.PASS_THRESHOLDS == {
        "complete_physical_scope_count_minimum": 1,
        "passed_margin_count_minimum": 98,
        "total_shortfall_strictly_less_than": 41.01776266878769,
        "rough_pixel_balanced_accuracy_strictly_greater_than":
            0.8198594673963917,
        "rough_ground_balanced_accuracy_strictly_greater_than":
            0.647134926562893,
        "rough_depth_p95_m_strictly_less_than": 0.9777327477931971,
    }
    passing = contract.checkpoint_control_decision(
        update=1_000,
        evaluation=_evaluation(),
        integrity_pass=True,
    )
    assert passing["action"] == contract.CONTROL_PASS
    assert passing["qualifies_probe"] is True
    assert len(passing["conjuncts"]) == 6
    informational = contract.checkpoint_control_decision(
        update=100,
        evaluation=_evaluation(
            complete=0,
            passed=0,
            shortfall=100.0,
            pixel=0.0,
            ground=0.0,
            depth=10.0,
        ),
        integrity_pass=True,
    )
    assert informational["action"] == contract.CONTROL_CONTINUE
    assert informational["next_update"] == 400
    equality = contract.checkpoint_control_decision(
        update=1_000,
        evaluation=_evaluation(
            shortfall=contract.PASS_THRESHOLDS[
                "total_shortfall_strictly_less_than"
            ],
            pixel=contract.PASS_THRESHOLDS[
                "rough_pixel_balanced_accuracy_strictly_greater_than"
            ],
            ground=contract.PASS_THRESHOLDS[
                "rough_ground_balanced_accuracy_strictly_greater_than"
            ],
            depth=contract.PASS_THRESHOLDS[
                "rough_depth_p95_m_strictly_less_than"
            ],
        ),
        integrity_pass=True,
    )
    assert equality["action"] == contract.CONTROL_FAIL
    assert equality["retry_authorized"] is False
    assert equality["threshold_equality_passes"] is False
    integrity_failure = contract.checkpoint_control_decision(
        update=100,
        evaluation=_evaluation(),
        integrity_pass=False,
    )
    assert integrity_failure["action"] == contract.CONTROL_INTEGRITY_FAIL
    assert integrity_failure["terminal"] is True


def test_science_contract_is_canonical_and_source_only() -> None:
    first = contract.science_contract()
    second = contract.science_contract()
    assert first == second
    assert json.loads(contract.canonical_json_bytes(first)) == first
    assert contract.canonical_json_sha256(first) == (
        "a94f3b78dfe69f1b42bce0f8bc93736b06e7111ef1ca40f852050a43c1824bc9"
    )
    lifecycle = contract.lifecycle_contract()
    assert lifecycle["source_freeze_complete"] is True
    assert lifecycle["source_preparation_may_reserve_attempt"] is False
    assert lifecycle["maximum_attempts"] == 1
    assert (
        lifecycle["retry_resume_repair_second_seed_extension_or_rerun"]
        is False
    )
    assert lifecycle["probability_calibration_open_count"] == 0
    assert lifecycle["prior_runtime_output_open_count"] == 0
    assert lifecycle["prohibited_runtime_output_roots"] == list(
        contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS
    )
    assert set(lifecycle["downstream_authority"].values()) == {False}
