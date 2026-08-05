#!/usr/bin/env python3
"""Validate and aggregate two immutable explicit-hierarchy N32 V4 results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from lewm.benchmarks.go2_categorical_radial_n32 import (  # noqa: E402
    HOLDOUT_PANELS,
    categorical_holdout_checks,
)
from lewm.benchmarks.go2_categorical_radial_n32_v4 import (  # noqa: E402
    EXECUTION_BINDING_SHA256,
    FACTOR_OUTPUT_CONTRACT,
    FACTOR_OUTPUT_CONTRACT_SHA256,
    RESULT_SCHEMA,
    SMOKE_RESULT_SCHEMA,
    STAGE_NAME,
    STAGE_SCHEMA,
    TWO_SEED_RESULT_SCHEMA,
    per_seed_decision,
)
from scripts import finalize_go2_categorical_radial_n32_v3 as v3  # noqa: E402


v2 = v3.v2
base = v3.base
EXPECTED_SEEDS = (20260710, 20260711)
REGISTERED_PARAMETER_COUNT = 2_887_002
REGISTERED_V2_PARAMETER_COUNT = 2_887_067
REGISTERED_TOKEN_FEATURE_DIM = 24
REGISTERED_CONTEXT_DIM = 64
REGISTERED_PARAMETER_DELTA = -65
MODEL_CLASS = "CategoricalRadialPerceptionFullRayHierarchical"
MODEL_OUTPUT_SEMANTICS = (
    "normalized_joint_log_probabilities_before_unchanged_cartesian_gather"
)
EXPECTED_INITIAL_STATE_SHA256 = {
    20260710: "0e82e8832eb2c27dc9ef2ea4c6ff35a83dcca181cb1d4172830fb6b2811a9c5e",
    20260711: "55ae2bbeecbe3913c7e886c11a3a14a5c4c435673a6067df45a2cca6d12fbc99",
}
EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256 = dict(v2.EXPECTED_INITIAL_STATE_SHA256)
SCHEDULE_SHA256 = dict(v2.SCHEDULE_SHA256)
AUTHORITATIVE_CONFIG = dict(v2.AUTHORITATIVE_CONFIG)
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_n32_v4_hierarchical_binding_2026-07-11.md"
)
V3_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_n32/v3/seed_20260710_result.json"
)
V3_RESULT_FILE_SHA256 = (
    "0f3eb212afe54a38d7a81a1fc51ca544dfab667a94a836be742d3ea3e2298d85"
)
V3_RESULT_CONTENT_SHA256 = (
    "ec8dd8450fb34bee3a5ba1c5a5b532339d281241560c8ed9ac07a48d2c2bea4e"
)
V3_RESULT_NOTE_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_categorical_radial_n32_v3_result_2026-07-11.md"
)
V3_RESULT_NOTE_SHA256 = (
    "a346ecb9b909d897f839067e409bccd906f61223cec8e746da6b54c531f44fca"
)
KNOWN_BIAS_PROOF_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_n32_known_bias_impossibility_2026-07-11.md"
)
KNOWN_BIAS_PROOF_SHA256 = (
    "e214bb80bcccf9ae5051231d90f7a5d8c2bfa33ca799e7db3eb969698fa2108a"
)
POSE_AUDIT_RESULT_PATH = (
    REPOSITORY_ROOT / ".generated/go2_n32_pose_projection_audit/v1/result.json"
)
POSE_AUDIT_RESULT_FILE_SHA256 = (
    "2c7efba897054ea0067db58f020e70dc5f3c5804785c74cbda4a8b76e0210b9d"
)
POSE_AUDIT_RESULT_CONTENT_SHA256 = (
    "6a9d05a0fb92289334cf39bb6947a2022a05a7c1892e8bb1c5a7156f9ca227f4"
)
POSE_AUDIT_REPORT_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_n32_pose_projection_audit_result_2026-07-11.md"
)
POSE_AUDIT_REPORT_SHA256 = (
    "e1a0c7e8c161827c5d8a1e2088135d8d986cbce9f9f7c02aa43d78d37a0be5e8"
)
POSE_ROLE_NAMESPACE_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_n32_pose_projection_role_namespace_amendment_2026-07-11.md"
)
POSE_ROLE_NAMESPACE_SHA256 = (
    "ae17eb856c5329e8c5dfa5e4339306ef19e60c53c5f67d43746b268be9cc3370"
)
CANONICAL_RESULT_PATHS = {
    seed: (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_n32/v4/"
        f"seed_{seed}_result.json"
    ).resolve()
    for seed in EXPECTED_SEEDS
}
EXPECTED_INPUTS = {
    **v3.EXPECTED_INPUTS,
    "failed_v3_n32_result": (
        V3_RESULT_PATH,
        V3_RESULT_FILE_SHA256,
        V3_RESULT_CONTENT_SHA256,
    ),
    "failed_v3_n32_result_note": (
        V3_RESULT_NOTE_PATH,
        V3_RESULT_NOTE_SHA256,
        None,
    ),
    "known_bias_impossibility_proof": (
        KNOWN_BIAS_PROOF_PATH,
        KNOWN_BIAS_PROOF_SHA256,
        None,
    ),
    "pose_projection_audit_result": (
        POSE_AUDIT_RESULT_PATH,
        POSE_AUDIT_RESULT_FILE_SHA256,
        POSE_AUDIT_RESULT_CONTENT_SHA256,
    ),
    "pose_projection_audit_report": (
        POSE_AUDIT_REPORT_PATH,
        POSE_AUDIT_REPORT_SHA256,
        None,
    ),
    "pose_projection_role_namespace_amendment": (
        POSE_ROLE_NAMESPACE_PATH,
        POSE_ROLE_NAMESPACE_SHA256,
        None,
    ),
}
BOUND_EVIDENCE = {
    **v3.BOUND_EVIDENCE,
    V3_RESULT_PATH: V3_RESULT_FILE_SHA256,
    V3_RESULT_NOTE_PATH: V3_RESULT_NOTE_SHA256,
    KNOWN_BIAS_PROOF_PATH: KNOWN_BIAS_PROOF_SHA256,
    POSE_AUDIT_RESULT_PATH: POSE_AUDIT_RESULT_FILE_SHA256,
    POSE_AUDIT_REPORT_PATH: POSE_AUDIT_REPORT_SHA256,
    POSE_ROLE_NAMESPACE_PATH: POSE_ROLE_NAMESPACE_SHA256,
    CONTRACT_PATH: EXECUTION_BINDING_SHA256,
}
EXPECTED_PANEL_ARTIFACT_COUNTS = base.EXPECTED_PANEL_ARTIFACT_COUNTS
EVENT_FIELDS = set(v2.EVENT_FIELDS)
DATA_EVENT_FIELDS = set(v2.DATA_EVENT_FIELDS)
TOP_LEVEL_FIELDS = {
    "schema",
    "authoritative",
    "aggregation_eligible",
    "promotion_eligible",
    "seed",
    "created_at_utc",
    "completed_at_utc",
    "invocation",
    "execution",
    "contract",
    "inputs",
    "source_hashes",
    "git",
    "model",
    "stages",
    "patch7_reference",
    "holdouts",
    "holdout_checks",
    "decision",
    "artifact_verification",
    "access_ledger",
    "categorical_radial_full_train_candidate_licensed",
    "runtime_ready",
    "g2_licensed",
    "g3_licensed",
    "content_sha256",
}
EXPECTED_SHAPE_CHANGES = {
    "polar_head.weight": {
        "v2_shape": [3, 64, 1, 1],
        "v4_shape": [2, 64, 1, 1],
        "v2_dtype": "torch.float32",
        "v4_dtype": "torch.float32",
    },
    "polar_head.bias": {
        "v2_shape": [3],
        "v4_shape": [2],
        "v2_dtype": "torch.float32",
        "v4_dtype": "torch.float32",
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _runner_source_paths() -> dict[str, Path]:
    shared = {
        ("v3_n32_runner" if name == "runner" else name): path
        for name, path in v3._runner_source_paths().items()
    }
    return {
        **shared,
        "n32_v4_binding": CONTRACT_PATH,
        "n32_v4_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32_v4.py"
        ),
        "n32_v4_hierarchical_model": (
            REPOSITORY_ROOT
            / "lewm/models/categorical_radial_perception_full_ray_hierarchical.py"
        ),
        "n32_v4_model_test": (
            REPOSITORY_ROOT
            / "lewm/tests/test_categorical_radial_perception_full_ray_hierarchical.py"
        ),
        "n32_v4_pure_test": (
            REPOSITORY_ROOT / "lewm/tests/test_go2_categorical_radial_n32_v4.py"
        ),
        "n32_v4_runner_test": (
            REPOSITORY_ROOT / "lewm/tests/test_run_go2_categorical_radial_n32_v4.py"
        ),
        "n32_v4_finalizer_test": (
            REPOSITORY_ROOT / "lewm/tests/test_finalize_go2_categorical_radial_n32_v4.py"
        ),
        "v4_finalizer": Path(__file__).resolve(),
        "runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_n32_v4.py",
    }


def _runner_source_hashes() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in sorted(_runner_source_paths().items())
    }


def _source_hashes() -> dict[str, dict[str, str]]:
    paths = {
        "execution_binding": CONTRACT_PATH,
        "finalizer": Path(__file__).resolve(),
        "n32_v4_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32_v4.py"
        ),
        "n32_v4_hierarchical_model": (
            REPOSITORY_ROOT
            / "lewm/models/categorical_radial_perception_full_ray_hierarchical.py"
        ),
        "runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_n32_v4.py",
    }
    result = {
        name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        for name, path in sorted(paths.items())
    }
    result.update(
        {
            f"runner_bound_{name}": record
            for name, record in _runner_source_hashes().items()
        }
    )
    return dict(sorted(result.items()))


def _validate_pose_result(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != "lewm_go2_n32_pose_projection_audit_v1":
        raise ValueError("bound pose-projection audit schema drift")
    core = dict(payload)
    declared = core.pop("content_sha256", None)
    if (
        declared != POSE_AUDIT_RESULT_CONTENT_SHA256
        or _canonical_json_sha256(core) != declared
    ):
        raise ValueError("bound pose-projection audit content drift")
    decision = base._mapping(
        payload.get("ordering_decision"), context="pose ordering decision"
    )
    expected = {
        "schema": "lewm_go2_n32_pose_projection_ordering_decision_v1",
        "estimand": "median_of_per_frame_p50_token_displacement",
        "rough_threshold_token": 0.5,
        "rough_minus_non_rough_threshold_token": 0.25,
        "rough_local_dynamics_median_token": 0.2553285781074101,
        "pooled_non_rough_median_token": 0.2814419707514776,
        "rough_minus_non_rough_median_token": -0.02611339264406748,
        "rough_threshold_passes": False,
        "contrast_threshold_passes": False,
        "material_dynamic_pose_mismatch": False,
        "next_intervention": "explicit_hierarchical_output",
    }
    for name in (
        "rough_threshold_passes",
        "contrast_threshold_passes",
        "material_dynamic_pose_mismatch",
    ):
        base._strict_bool(decision.get(name), context=f"pose decision/{name}")
    for name in (
        "rough_threshold_token",
        "rough_minus_non_rough_threshold_token",
        "rough_local_dynamics_median_token",
        "pooled_non_rough_median_token",
        "rough_minus_non_rough_median_token",
    ):
        base._finite_number(decision.get(name), context=f"pose decision/{name}")
    if decision != expected:
        raise ValueError("bound pose-projection ordering decision drift")


def _load_bound_evidence() -> dict[str, Any]:
    pre_hashes = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    expected = {
        str(path.resolve()): digest for path, digest in BOUND_EVIDENCE.items()
    }
    if pre_hashes != expected:
        raise ValueError("bound N32 V4 evidence file SHA-256 drift")
    inherited = v3._load_bound_evidence()
    v3_payload, _v3_ledger = base._load_expected_json(
        V3_RESULT_PATH,
        expected_sha256=V3_RESULT_FILE_SHA256,
    )
    v3_validated = v3._validate_authoritative_result(
        v3_payload,
        expected_seed=20260710,
        expected_controls=inherited["controls"][20260710],
        expected_patch7_reference=inherited["patch7_reference"],
    )
    if (
        v3_payload.get("content_sha256") != V3_RESULT_CONTENT_SHA256
        or v3_validated["decision"].get("classification") != "fit_gate_failed"
        or v3_validated["decision"].get("favorable") is not False
        or v3_validated["decision"].get("token_width_32_fit_passes") is not False
    ):
        raise ValueError("bound failed N32 V3 result identity drift")
    pose_payload, _pose_ledger = base._load_expected_json(
        POSE_AUDIT_RESULT_PATH,
        expected_sha256=POSE_AUDIT_RESULT_FILE_SHA256,
    )
    _validate_pose_result(pose_payload)
    post_hashes = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    if post_hashes != pre_hashes:
        raise RuntimeError("bound N32 V4 evidence changed during parsing")
    return {
        "pre_hashes": pre_hashes,
        "post_parse_hashes": post_hashes,
        "panels": inherited["panels"],
        "patch7_reference": inherited["patch7_reference"],
        "controls": inherited["controls"],
    }


def _validate_minibatches(value: object, *, seed: int) -> list[list[int]]:
    batches = v2._validate_minibatches(value, seed=seed)
    if _canonical_json_sha256(batches) != SCHEDULE_SHA256[seed]:
        raise ValueError("N32 V4 exact seeded minibatch schedule drift")
    return batches


def _validate_stage(
    value: object,
    *,
    seed: int,
    expected_initial_state: str,
    expected_controls: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    stage = base._mapping(value, context="N32 V4 stage")
    if stage.get("schema") != STAGE_SCHEMA or stage.get("stage") != STAGE_NAME:
        raise ValueError("N32 V4 stage identity drift")
    normalized = dict(stage)
    normalized["schema"] = v2.STAGE_SCHEMA
    normalized["stage"] = v2.STAGE_NAME
    _validated, terminal = v2._validate_stage(
        normalized,
        seed=seed,
        expected_initial_state=expected_initial_state,
        expected_controls=expected_controls,
    )
    _validate_minibatches(stage.get("minibatch_indices"), seed=seed)
    return stage, terminal


def _validate_inputs(value: object, *, seed: int) -> Mapping[str, Any]:
    inputs = base._mapping(value, context="N32 V4 inputs")
    if set(inputs) != {*EXPECTED_INPUTS, "seed_20260710_authorization"}:
        raise ValueError("N32 V4 immutable input keys drift")
    for name, (path, file_hash, content_hash) in EXPECTED_INPUTS.items():
        record = base._mapping(inputs[name], context=f"input/{name}")
        expected = {"path": str(path.resolve()), "sha256": file_hash}
        if content_hash is not None:
            expected["content_sha256"] = content_hash
        if record != expected:
            raise ValueError(f"N32 V4 immutable input drift: {name}")
    authorization = inputs["seed_20260710_authorization"]
    if seed == 20260710 and authorization is not None:
        raise ValueError("N32 V4 primary must not carry authorization")
    if seed == 20260711:
        record = base._mapping(authorization, context="primary authorization")
        if (
            set(record) != {"path", "sha256"}
            or Path(str(record.get("path", ""))).resolve()
            != CANONICAL_RESULT_PATHS[20260710]
            or not base._is_sha256(record.get("sha256"))
        ):
            raise ValueError("N32 V4 replication authorization drift")
    return inputs


def _validate_artifact_verification(value: object) -> None:
    verification = base._mapping(value, context="artifact verification")
    if set(verification) != {
        "fit_verified_before_access",
        "holdouts_verified_only_after_terminal_fit_pass",
        "evidence_hashes",
    }:
        raise ValueError("N32 V4 artifact verification fields drift")
    if (
        verification.get("fit_verified_before_access") is not True
        or verification.get("holdouts_verified_only_after_terminal_fit_pass")
        is not True
    ):
        raise ValueError("N32 V4 artifact verification ordering drift")
    evidence = base._mapping(
        verification.get("evidence_hashes"), context="evidence hashes"
    )
    expected = {str(path.resolve()): digest for path, digest in BOUND_EVIDENCE.items()}
    if evidence != expected:
        raise ValueError("N32 V4 bound evidence mapping drift")


def _validate_access(value: object, *, holdouts_authorized: bool) -> None:
    access = base._mapping(value, context="N32 V4 access ledger")
    if "dataset_role_policy" not in access:
        raise ValueError("N32 V4 dataset-role policy is missing")
    inherited = dict(access)
    policy = base._mapping(
        inherited.pop("dataset_role_policy"), context="dataset role policy"
    )
    expected = {
        "current_physical_dataset_role": "train",
        "current_physical_dataset_role_governs_access": True,
        "legacy_rollout_split_is_provenance_only": True,
        "legacy_rollout_split_used_to_filter_rank_calibrate_or_select": False,
    }
    for name in (
        "current_physical_dataset_role_governs_access",
        "legacy_rollout_split_is_provenance_only",
        "legacy_rollout_split_used_to_filter_rank_calibrate_or_select",
    ):
        base._strict_bool(policy.get(name), context=f"dataset role policy/{name}")
    if policy != expected:
        raise ValueError("N32 V4 dataset-role policy drift")
    v2._validate_access(inherited, holdouts_authorized=holdouts_authorized)


def _validate_initialization_proof(
    value: object,
    *,
    seed: int,
) -> Mapping[str, Any]:
    proof = base._mapping(value, context="initialization comparability")
    expected_fields = {
        "construction",
        "v2_reference_initial_state_sha256",
        "candidate_initial_state_sha256",
        "v2_reference_parameter_count",
        "candidate_parameter_count",
        "state_key_sets_identical",
        "same_shape_entry_count",
        "same_shape_entries_bit_identical",
        "only_shape_changed_state_keys",
        "shape_changes",
        "shape_changed_head_tensors_left_at_deterministic_pytorch_default",
        "class_prior_bias_matching_applied",
        "analytic_v2_head_transform_applied",
        "zero_initialization_applied",
        "trained_v2_weight_loaded",
        "trained_v3_weight_loaded",
    }
    if set(proof) != expected_fields:
        raise ValueError("N32 V4 initialization proof fields drift")
    for name in (
        "state_key_sets_identical",
        "same_shape_entries_bit_identical",
        "shape_changed_head_tensors_left_at_deterministic_pytorch_default",
        "class_prior_bias_matching_applied",
        "analytic_v2_head_transform_applied",
        "zero_initialization_applied",
        "trained_v2_weight_loaded",
        "trained_v3_weight_loaded",
    ):
        base._strict_bool(proof.get(name), context=f"initialization/{name}")
    for name in (
        "v2_reference_parameter_count",
        "candidate_parameter_count",
        "same_shape_entry_count",
    ):
        base._strict_int(proof.get(name), context=f"initialization/{name}")
    expected = {
        "construction": (
            "save_cpu_rng_after_determinism_then_replay_for_v2_and_v4_copy_131_"
            "same_shape_entries_leave_two_changed_head_tensors_at_pytorch_defaults"
        ),
        "v2_reference_initial_state_sha256": (
            EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256[seed]
        ),
        "candidate_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256[seed],
        "v2_reference_parameter_count": REGISTERED_V2_PARAMETER_COUNT,
        "candidate_parameter_count": REGISTERED_PARAMETER_COUNT,
        "state_key_sets_identical": True,
        "same_shape_entry_count": 131,
        "same_shape_entries_bit_identical": True,
        "only_shape_changed_state_keys": sorted(EXPECTED_SHAPE_CHANGES),
        "shape_changes": EXPECTED_SHAPE_CHANGES,
        "shape_changed_head_tensors_left_at_deterministic_pytorch_default": True,
        "class_prior_bias_matching_applied": False,
        "analytic_v2_head_transform_applied": False,
        "zero_initialization_applied": False,
        "trained_v2_weight_loaded": False,
        "trained_v3_weight_loaded": False,
    }
    if proof != expected:
        raise ValueError("N32 V4 initialization comparability proof drift")
    return proof


def _validate_execution(value: object, *, seed: int) -> Mapping[str, Any]:
    execution = base._mapping(value, context="execution")
    fields = {
        "device",
        "device_name",
        "determinism",
        "cpu_rng_state_captured_immediately_after_determinism",
        "batch_size_frames",
        "evaluation_target_batch_size",
        "evaluation_combined_model_batch_size",
        "evaluation_interval",
        "stage_config",
        "effective_epochs",
        "fp32_no_autocast_amp_compile_or_quantization",
    }
    if set(execution) != fields:
        raise ValueError("N32 V4 execution fields drift")
    if (
        base._strict_bool(
            execution.get("cpu_rng_state_captured_immediately_after_determinism"),
            context="execution/cpu_rng_capture",
        )
        is not True
        or base._strict_int(execution.get("batch_size_frames"), context="train batch")
        != 80
        or base._strict_int(
            execution.get("evaluation_target_batch_size"), context="eval target batch"
        )
        != 4
        or base._strict_int(
            execution.get("evaluation_combined_model_batch_size"),
            context="eval model batch",
        )
        != 12
        or base._strict_int(
            execution.get("evaluation_interval"), context="eval interval"
        )
        != 100
        or execution.get("stage_config") != AUTHORITATIVE_CONFIG
        or base._finite_number(execution.get("effective_epochs"), context="epochs")
        != 500.0
        or execution.get("fp32_no_autocast_amp_compile_or_quantization") is not True
    ):
        raise ValueError("N32 V4 execution contract drift")
    stage_config = base._mapping(execution.get("stage_config"), context="stage config")
    for name in ("updates", "batch_size"):
        base._strict_int(stage_config.get(name), context=f"stage config/{name}")
    for name in ("learning_rate_start", "learning_rate_end", "weight_decay"):
        base._finite_number(stage_config.get(name), context=f"stage config/{name}")
    determinism = base._mapping(execution.get("determinism"), context="determinism")
    expected = {
        "seed": seed,
        "requested": "strict_deterministic_algorithms",
        "effective": "strict_where_supported_warn_on_unsupported",
        "warn_only": True,
        "torch_deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }
    base._strict_int(determinism.get("seed"), context="determinism/seed")
    for name in (
        "warn_only",
        "torch_deterministic_algorithms",
        "cudnn_benchmark",
        "cudnn_deterministic",
    ):
        base._strict_bool(determinism.get(name), context=f"determinism/{name}")
    if determinism != expected:
        raise ValueError("N32 V4 determinism contract drift")
    return execution


def _validate_authoritative_result(
    artifact: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_controls: Mapping[str, Mapping[str, Any]],
    expected_patch7_reference: Mapping[str, Any],
) -> dict[str, Any]:
    if expected_seed not in EXPECTED_SEEDS or set(artifact) != TOP_LEVEL_FIELDS:
        raise ValueError("N32 V4 top-level result fields drift")
    if artifact.get("schema") == SMOKE_RESULT_SCHEMA:
        raise ValueError("N32 V4 finalizer rejects smoke results")
    if (
        artifact.get("schema") != RESULT_SCHEMA
        or artifact.get("authoritative") is not True
        or artifact.get("aggregation_eligible") is not True
        or artifact.get("promotion_eligible") is not False
        or base._strict_int(artifact.get("seed"), context="result seed")
        != expected_seed
    ):
        raise ValueError("N32 V4 result is not authoritative")
    for name in (
        "categorical_radial_full_train_candidate_licensed",
        "runtime_ready",
        "g2_licensed",
        "g3_licensed",
    ):
        if base._strict_bool(artifact.get(name), context=name) is not False:
            raise ValueError("single-seed N32 V4 result grants a forbidden license")
    core = dict(artifact)
    declared = core.pop("content_sha256", None)
    if not base._is_sha256(declared) or _canonical_json_sha256(core) != declared:
        raise ValueError("N32 V4 content hash mismatch")
    for name in ("created_at_utc", "completed_at_utc"):
        if not isinstance(artifact.get(name), str) or not artifact[name]:
            raise ValueError("N32 V4 timestamps are missing")
    invocation = artifact.get("invocation")
    if not isinstance(invocation, list) or not invocation or any(
        not isinstance(item, str) for item in invocation
    ):
        raise ValueError("N32 V4 invocation provenance is missing")
    _validate_execution(artifact.get("execution"), seed=expected_seed)
    contract = base._mapping(artifact.get("contract"), context="contract")
    if contract != {
        "path": str(CONTRACT_PATH.resolve()),
        "sha256": EXECUTION_BINDING_SHA256,
    }:
        raise ValueError("N32 V4 execution binding drift")
    inputs = _validate_inputs(artifact.get("inputs"), seed=expected_seed)
    sources = base._mapping(artifact.get("source_hashes"), context="source hashes")
    if sources != _runner_source_hashes():
        raise ValueError("N32 V4 transitive source provenance drift")
    base._mapping(artifact.get("git"), context="git provenance")

    model = base._mapping(artifact.get("model"), context="model")
    fields = {
        "class",
        "parameter_count",
        "token_feature_dim",
        "context_dim",
        "parameter_delta_from_v2",
        "factor_order",
        "output_semantics",
        "factor_output_contract",
        "factor_output_contract_sha256",
        "initial_state_sha256",
        "initialization_comparability",
    }
    if set(model) != fields:
        raise ValueError("N32 V4 model fields drift")
    for name in ("parameter_count", "token_feature_dim", "context_dim"):
        base._strict_int(model.get(name), context=f"model/{name}")
    base._strict_int(
        model.get("parameter_delta_from_v2"),
        context="model/parameter_delta_from_v2",
        minimum=-REGISTERED_V2_PARAMETER_COUNT,
    )
    proof = _validate_initialization_proof(
        model.get("initialization_comparability"), seed=expected_seed
    )
    expected_model = {
        "class": MODEL_CLASS,
        "parameter_count": REGISTERED_PARAMETER_COUNT,
        "token_feature_dim": REGISTERED_TOKEN_FEATURE_DIM,
        "context_dim": REGISTERED_CONTEXT_DIM,
        "parameter_delta_from_v2": REGISTERED_PARAMETER_DELTA,
        "factor_order": FACTOR_OUTPUT_CONTRACT["raw_factor_order"],
        "output_semantics": MODEL_OUTPUT_SEMANTICS,
        "factor_output_contract": FACTOR_OUTPUT_CONTRACT,
        "factor_output_contract_sha256": FACTOR_OUTPUT_CONTRACT_SHA256,
        "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256[expected_seed],
        "initialization_comparability": proof,
    }
    if model != expected_model:
        raise ValueError("N32 V4 model identity, factors, or initialization drift")

    reference = base._validate_patch7_reference(artifact.get("patch7_reference"))
    if reference != expected_patch7_reference:
        raise ValueError("N32 V4 patch7 reference drift")
    stages = base._mapping(artifact.get("stages"), context="stages")
    if set(stages) != {STAGE_NAME}:
        raise ValueError("N32 V4 must contain exactly one optimizer stage")
    stage, terminal = _validate_stage(
        stages[STAGE_NAME],
        seed=expected_seed,
        expected_initial_state=EXPECTED_INITIAL_STATE_SHA256[expected_seed],
        expected_controls=expected_controls["fit"],
    )
    holdouts = artifact.get("holdouts")
    checks = artifact.get("holdout_checks")
    if terminal["passes"]:
        holdout_map = base._mapping(holdouts, context="holdouts")
        check_map = base._mapping(checks, context="holdout checks")
        if set(holdout_map) != set(HOLDOUT_PANELS) or set(check_map) != set(
            HOLDOUT_PANELS
        ):
            raise ValueError("N32 V4 authorized holdouts are incomplete")
        recomputed_checks = {}
        for panel in HOLDOUT_PANELS:
            report = base._validate_panel_report(
                holdout_map[panel],
                panel=panel,
                seed=expected_seed,
                require_fit_gate=False,
                expected_controls=expected_controls[panel],
            )
            recomputed_checks[panel] = categorical_holdout_checks(
                report, reference["panels"][panel]
            )
            base._validate_gate_boolean_types(
                check_map[panel], context=f"{panel} holdout checks"
            )
            for name in (
                "strictly_favorable_family_count",
                "strictly_favorable_family_requirement",
            ):
                base._strict_int(
                    check_map[panel].get(name), context=f"{panel}/{name}"
                )
        if check_map != recomputed_checks or stage.get("holdouts_evaluated") is not True:
            raise ValueError("N32 V4 holdout decision or stage flag drift")
    else:
        if holdouts is not None or checks is not None or stage.get(
            "holdouts_evaluated"
        ) is not False:
            raise ValueError("N32 V4 unauthorized holdout payload/access flag exists")
        recomputed_checks = None
    recomputed_decision = per_seed_decision(stage, recomputed_checks)
    stored_decision = base._mapping(artifact.get("decision"), context="decision")
    for name in (
        "explicit_hierarchical_output_fit_passes",
        "favorable",
        "aggregation_eligible",
        "categorical_radial_full_train_candidate_licensed",
        "runtime_ready",
        "g2_licensed",
        "g3_licensed",
        "promotion_licensed",
    ):
        base._strict_bool(stored_decision.get(name), context=f"decision/{name}")
    holdout_passes = stored_decision.get("holdout_passes")
    if holdout_passes is not None:
        holdout_passes = base._mapping(holdout_passes, context="decision holdouts")
        if set(holdout_passes) != set(HOLDOUT_PANELS):
            raise ValueError("N32 V4 decision holdout keys drift")
        for panel in HOLDOUT_PANELS:
            base._strict_bool(holdout_passes.get(panel), context=f"decision/{panel}")
    if stored_decision != recomputed_decision:
        raise ValueError("N32 V4 stored decision does not recompute")
    _validate_artifact_verification(artifact.get("artifact_verification"))
    _validate_access(artifact.get("access_ledger"), holdouts_authorized=terminal["passes"])
    return {
        "content_sha256": declared,
        "inputs": inputs,
        "source_hashes": sources,
        "contract": contract,
        "model_mechanism": {
            "class": model["class"],
            "parameter_count": model["parameter_count"],
            "token_feature_dim": model["token_feature_dim"],
            "context_dim": model["context_dim"],
            "parameter_delta_from_v2": model["parameter_delta_from_v2"],
            "factor_output_contract_sha256": model["factor_output_contract_sha256"],
        },
        "patch7_reference": reference,
        "decision": recomputed_decision,
    }


def validate_seed10_authorization(
    path: Path,
    expected_sha256: str,
    *,
    expected_runner_sources: Mapping[str, Any] | None = None,
    expected_patch7_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fully validate the favorable primary before seed-11 construction."""

    if path.resolve() != CANONICAL_RESULT_PATHS[20260710]:
        raise ValueError("seed-20260710 V4 authorization path is not canonical")
    bound = _load_bound_evidence()
    payload, ledger = base._load_expected_json(
        path.resolve(), expected_sha256=str(expected_sha256)
    )
    validated = _validate_authoritative_result(
        payload,
        expected_seed=20260710,
        expected_controls=bound["controls"][20260710],
        expected_patch7_reference=bound["patch7_reference"],
    )
    if validated["decision"].get("favorable") is not True:
        raise ValueError("seed-20260710 V4 result is not favorable")
    if expected_runner_sources is not None and validated[
        "source_hashes"
    ] != expected_runner_sources:
        raise ValueError("seed-20260710 V4 source authorization drift")
    if expected_patch7_reference is not None and validated[
        "patch7_reference"
    ] != expected_patch7_reference:
        raise ValueError("seed-20260710 V4 reference authorization drift")
    final_hash = _sha256_file(path.resolve())
    if final_hash != ledger["pre_deserialization_sha256"]:
        raise RuntimeError("seed-20260710 V4 authorization changed during validation")
    return {"validated": validated, "artifact": payload, "sha256": final_hash}


def _aggregate(
    primary: Mapping[str, Any], replication: Mapping[str, Any]
) -> dict[str, Any]:
    first = base._mapping(primary["decision"], context="primary decision")
    second = base._mapping(replication["decision"], context="replication decision")
    both_favorable = bool(first["favorable"]) and bool(second["favorable"])
    return {
        "classification": (
            "two_seed_favorable" if both_favorable else "two_seed_replication_failed"
        ),
        "seeds": list(EXPECTED_SEEDS),
        "both_seeds_favorable": both_favorable,
        "qualifying_optimizer_stage": STAGE_NAME if both_favorable else None,
        "seed_decisions": {
            str(EXPECTED_SEEDS[0]): dict(first),
            str(EXPECTED_SEEDS[1]): dict(second),
        },
        "categorical_radial_full_train_candidate_licensed": both_favorable,
        "runtime_ready": False,
        "promotion_licensed": False,
        "g2_licensed": False,
        "g3_licensed": False,
    }


_atomic_write_json_exclusive = v2._atomic_write_json_exclusive


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-20260710-result", type=Path, required=True)
    parser.add_argument("--expected-seed-20260710-result-sha256", required=True)
    parser.add_argument("--seed-20260711-result", type=Path, required=True)
    parser.add_argument("--expected-seed-20260711-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; N32 V4 finalization is immutable")
    for seed, path in (
        (20260710, args.seed_20260710_result),
        (20260711, args.seed_20260711_result),
    ):
        if path.resolve() != CANONICAL_RESULT_PATHS[seed]:
            parser.error(f"seed {seed} V4 result path is not canonical")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    bound = _load_bound_evidence()
    source_start = _source_hashes()
    paths = {
        20260710: args.seed_20260710_result.resolve(),
        20260711: args.seed_20260711_result.resolve(),
    }
    expected_hashes = {
        20260710: str(args.expected_seed_20260710_result_sha256),
        20260711: str(args.expected_seed_20260711_result_sha256),
    }
    payloads, ledgers, validated = {}, {}, {}
    for seed in EXPECTED_SEEDS:
        payloads[seed], ledgers[seed] = base._load_expected_json(
            paths[seed], expected_sha256=expected_hashes[seed]
        )
        validated[seed] = _validate_authoritative_result(
            payloads[seed],
            expected_seed=seed,
            expected_controls=bound["controls"][seed],
            expected_patch7_reference=bound["patch7_reference"],
        )
    for name in ("source_hashes", "contract", "model_mechanism", "patch7_reference"):
        if _canonical_json(validated[20260710][name]) != _canonical_json(
            validated[20260711][name]
        ):
            raise ValueError(f"two N32 V4 seeds disagree on common {name}")
    for name in EXPECTED_INPUTS:
        if validated[20260710]["inputs"][name] != validated[20260711]["inputs"][name]:
            raise ValueError(f"two N32 V4 seeds disagree on input {name}")
    authorization = base._mapping(
        validated[20260711]["inputs"]["seed_20260710_authorization"],
        context="replication authorization",
    )
    if (
        authorization.get("sha256") != expected_hashes[20260710]
        or validated[20260710]["decision"].get("favorable") is not True
    ):
        raise ValueError("N32 V4 replication lacks favorable primary authorization")
    aggregation = _aggregate(validated[20260710], validated[20260711])
    for seed in EXPECTED_SEEDS:
        final_hash = _sha256_file(paths[seed])
        if final_hash != ledgers[seed]["pre_deserialization_sha256"]:
            raise RuntimeError(f"seed {seed} V4 result changed during finalization")
        ledgers[seed]["post_validation_sha256"] = final_hash
        ledgers[seed]["post_validation_unchanged"] = True
        ledgers[seed]["content_sha256"] = validated[seed]["content_sha256"]
        ledgers[seed]["decision_recomputed_exactly"] = True
    if _source_hashes() != source_start:
        raise RuntimeError("N32 V4 finalizer sources changed during execution")
    evidence_end = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    if evidence_end != bound["pre_hashes"]:
        raise RuntimeError("bound N32 V4 evidence changed during finalization")
    core = {
        "schema": TWO_SEED_RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authoritative_inputs_only": True,
        "aggregation_eligible_inputs_only": True,
        "input_hash_verification": [ledgers[seed] for seed in EXPECTED_SEEDS],
        "bound_evidence_hash_verification": {
            "pre_deserialization": bound["pre_hashes"],
            "post_deserialization": bound["post_parse_hashes"],
            "post_finalization": evidence_end,
            "unchanged": True,
        },
        "common_provenance_validated": {
            "immutable_inputs": True,
            "source_hashes": True,
            "execution_binding": True,
            "model_mechanism": True,
            "patch7_reference": True,
        },
        "stored_seed_decisions_recomputed_from_raw_metrics": True,
        "aggregation": aggregation,
        "source_hashes": source_start,
        "categorical_radial_full_train_candidate_licensed": aggregation[
            "categorical_radial_full_train_candidate_licensed"
        ],
        "runtime_ready": False,
        "g2_licensed": False,
        "g3_licensed": False,
    }
    result = {**core, "content_sha256": _canonical_json_sha256(core)}
    _atomic_write_json_exclusive(args.output.resolve(), result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "content_sha256": result["content_sha256"],
                "classification": aggregation["classification"],
                "categorical_radial_full_train_candidate_licensed": result[
                    "categorical_radial_full_train_candidate_licensed"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
