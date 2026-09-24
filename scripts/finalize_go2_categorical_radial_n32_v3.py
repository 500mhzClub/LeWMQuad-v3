#!/usr/bin/env python3
"""Validate and aggregate two immutable token-width-32 N32 V3 results."""
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
from lewm.benchmarks.go2_categorical_radial_n32_v3 import (  # noqa: E402
    EXECUTION_BINDING_SHA256,
    RESULT_SCHEMA,
    SMOKE_RESULT_SCHEMA,
    STAGE_NAME,
    STAGE_SCHEMA,
    TWO_SEED_RESULT_SCHEMA,
    per_seed_decision,
)
from scripts import finalize_go2_categorical_radial_n32_v2 as v2  # noqa: E402


base = v2.base
EXPECTED_SEEDS = (20260710, 20260711)
REGISTERED_PARAMETER_COUNT = 2_891_171
REGISTERED_V2_PARAMETER_COUNT = 2_887_067
REGISTERED_TOKEN_FEATURE_DIM = 32
REGISTERED_CONTEXT_DIM = 64
REGISTERED_PARAMETER_DELTA = 4_104
EXPECTED_INITIAL_STATE_SHA256 = {
    20260710: "ddb8f6dbfa54a7445c2b4363d9978b0a99a86e6d88a28f480840c5d8d128804b",
    20260711: "fa9601fb5f658b640c43b50c28587c5129c6f42f8fd4fb09866983130e4954ee",
}
EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256 = dict(v2.EXPECTED_INITIAL_STATE_SHA256)
SCHEDULE_SHA256 = dict(v2.SCHEDULE_SHA256)
AUTHORITATIVE_CONFIG = dict(v2.AUTHORITATIVE_CONFIG)
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_n32_v3_token_width_binding_2026-07-11.md"
)
V2_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_n32/v2/seed_20260710_result.json"
)
V2_RESULT_FILE_SHA256 = (
    "0a5f8a822d7fec8287a30103125fca1a4927f0413e2f0906db431cef54ec2265"
)
V2_RESULT_CONTENT_SHA256 = (
    "e070cc96d69b76e1f85f533fa1d94221225963a2b66a491f0c2a867c008b97ef"
)
V2_RESULT_NOTE_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_categorical_radial_n32_v2_result_2026-07-11.md"
)
V2_RESULT_NOTE_SHA256 = (
    "d5e5748db8177d925990b5c31e23c45d43e16c62e0aac4f389ab47b1fa6547e0"
)
CANONICAL_RESULT_PATHS = {
    seed: (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_n32/v3/"
        f"seed_{seed}_result.json"
    ).resolve()
    for seed in EXPECTED_SEEDS
}
EXPECTED_INPUTS = {
    "panel": (
        base.PANEL_PATH,
        base.EXPECTED_INPUTS["panel"][0],
        base.EXPECTED_INPUTS["panel"][1],
    ),
    "ladder_manifest": (
        base.LADDER_PATH,
        base.EXPECTED_INPUTS["ladder_manifest"][0],
        base.EXPECTED_INPUTS["ladder_manifest"][1],
    ),
    "v3_result": (
        base.V3_RESULT_PATH,
        base.EXPECTED_INPUTS["v3_result"][0],
        base.EXPECTED_INPUTS["v3_result"][1],
    ),
    "patch7_reference_result": (
        base.PATCH7_RESULT_PATH,
        base.EXPECTED_INPUTS["patch7_reference_result"][0],
        base.EXPECTED_INPUTS["patch7_reference_result"][1],
    ),
    "failed_v1_n32_result": (
        v2.V1_RESULT_PATH,
        v2.V1_RESULT_FILE_SHA256,
        v2.V1_RESULT_CONTENT_SHA256,
    ),
    "failed_v1_n32_result_note": (
        v2.V1_RESULT_NOTE_PATH,
        v2.V1_RESULT_NOTE_SHA256,
        None,
    ),
    "failed_v2_n32_result": (
        V2_RESULT_PATH,
        V2_RESULT_FILE_SHA256,
        V2_RESULT_CONTENT_SHA256,
    ),
    "failed_v2_n32_result_note": (
        V2_RESULT_NOTE_PATH,
        V2_RESULT_NOTE_SHA256,
        None,
    ),
}
BOUND_EVIDENCE = {
    **v2.BOUND_EVIDENCE,
    V2_RESULT_PATH: V2_RESULT_FILE_SHA256,
    V2_RESULT_NOTE_PATH: V2_RESULT_NOTE_SHA256,
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
    "shared_jepa_full_train_candidate_licensed",
    "runtime_ready",
    "g2_licensed",
    "g3_licensed",
    "content_sha256",
}
EXPECTED_SHAPE_CHANGES = {
    "token_projection.weight": {
        "v2_shape": [24, 192, 1, 1],
        "v3_shape": [32, 192, 1, 1],
        "v2_dtype": "torch.float32",
        "v3_dtype": "torch.float32",
    },
    "token_projection.bias": {
        "v2_shape": [24],
        "v3_shape": [32],
        "v2_dtype": "torch.float32",
        "v3_dtype": "torch.float32",
    },
    "context_stem.0.weight": {
        "v2_shape": [64, 154, 1, 1],
        "v3_shape": [64, 194, 1, 1],
        "v2_dtype": "torch.float32",
        "v3_dtype": "torch.float32",
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
        ("v2_n32_runner" if name == "runner" else name): path
        for name, path in v2._runner_source_paths().items()
    }
    return {
        **shared,
        "n32_v3_binding": CONTRACT_PATH,
        "n32_v3_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32_v3.py"
        ),
        "n32_v3_token32_model": (
            REPOSITORY_ROOT
            / "lewm/models/categorical_radial_perception_full_ray_token32.py"
        ),
        "n32_v3_model_test": (
            REPOSITORY_ROOT
            / "lewm/tests/test_categorical_radial_perception_full_ray_token32.py"
        ),
        "n32_v3_pure_test": (
            REPOSITORY_ROOT / "lewm/tests/test_go2_categorical_radial_n32_v3.py"
        ),
        "n32_v3_runner_test": (
            REPOSITORY_ROOT / "lewm/tests/test_run_go2_categorical_radial_n32_v3.py"
        ),
        "n32_v3_finalizer_test": (
            REPOSITORY_ROOT
            / "lewm/tests/test_finalize_go2_categorical_radial_n32_v3.py"
        ),
        "v3_finalizer": Path(__file__).resolve(),
        "runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_n32_v3.py",
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
        "n32_v3_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32_v3.py"
        ),
        "n32_v3_token32_model": (
            REPOSITORY_ROOT
            / "lewm/models/categorical_radial_perception_full_ray_token32.py"
        ),
        "runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_n32_v3.py",
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


def _load_bound_evidence() -> dict[str, Any]:
    pre_hashes = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    expected = {
        str(path.resolve()): digest for path, digest in BOUND_EVIDENCE.items()
    }
    if pre_hashes != expected:
        raise ValueError("bound N32 V3 evidence file SHA-256 drift")
    inherited = v2._load_bound_evidence()
    payload, _ledger = base._load_expected_json(
        V2_RESULT_PATH,
        expected_sha256=V2_RESULT_FILE_SHA256,
    )
    validated = v2._validate_authoritative_result(
        payload,
        expected_seed=20260710,
        expected_controls=inherited["controls"][20260710],
        expected_patch7_reference=inherited["patch7_reference"],
    )
    if (
        payload.get("content_sha256") != V2_RESULT_CONTENT_SHA256
        or validated["decision"].get("classification") != "fit_gate_failed"
        or validated["decision"].get("favorable") is not False
        or validated["decision"].get(
            "exposure_matched_v3_cosine_fit_passes"
        ) is not False
    ):
        raise ValueError("bound failed N32 V2 result identity drift")
    post_hashes = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    if post_hashes != pre_hashes:
        raise RuntimeError("bound N32 V3 evidence changed during parsing")
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
        raise ValueError("N32 V3 exact seeded minibatch schedule drift")
    return batches


def _validate_stage(
    value: object,
    *,
    seed: int,
    expected_initial_state: str,
    expected_controls: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    stage = base._mapping(value, context="N32 V3 stage")
    if stage.get("schema") != STAGE_SCHEMA or stage.get("stage") != STAGE_NAME:
        raise ValueError("N32 V3 stage identity drift")
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
    inputs = base._mapping(value, context="N32 V3 inputs")
    if set(inputs) != {*EXPECTED_INPUTS, "seed_20260710_authorization"}:
        raise ValueError("N32 V3 immutable input keys drift")
    for name, (path, file_hash, content_hash) in EXPECTED_INPUTS.items():
        record = base._mapping(inputs[name], context=f"input/{name}")
        expected = {"path": str(path.resolve()), "sha256": file_hash}
        if content_hash is not None:
            expected["content_sha256"] = content_hash
        if record != expected:
            raise ValueError(f"N32 V3 immutable input drift: {name}")
    authorization = inputs["seed_20260710_authorization"]
    if seed == 20260710 and authorization is not None:
        raise ValueError("N32 V3 primary must not carry authorization")
    if seed == 20260711:
        record = base._mapping(authorization, context="primary authorization")
        if (
            set(record) != {"path", "sha256"}
            or Path(str(record.get("path", ""))).resolve()
            != CANONICAL_RESULT_PATHS[20260710]
            or not base._is_sha256(record.get("sha256"))
        ):
            raise ValueError("N32 V3 replication authorization drift")
    return inputs


def _validate_artifact_verification(value: object) -> None:
    verification = base._mapping(value, context="artifact verification")
    if set(verification) != {
        "fit_verified_before_access",
        "holdouts_verified_only_after_terminal_fit_pass",
        "evidence_hashes",
    }:
        raise ValueError("N32 V3 artifact verification fields drift")
    if (
        verification.get("fit_verified_before_access") is not True
        or verification.get("holdouts_verified_only_after_terminal_fit_pass")
        is not True
    ):
        raise ValueError("N32 V3 artifact verification ordering drift")
    evidence = base._mapping(
        verification.get("evidence_hashes"), context="evidence hashes"
    )
    expected = {str(path.resolve()): digest for path, digest in BOUND_EVIDENCE.items()}
    if evidence != expected:
        raise ValueError("N32 V3 bound evidence mapping drift")


def _validate_access(value: object, *, holdouts_authorized: bool) -> None:
    v2._validate_access(value, holdouts_authorized=holdouts_authorized)


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
        "trained_v2_weight_loaded",
    }
    if set(proof) != expected_fields:
        raise ValueError("N32 V3 initialization proof fields drift")
    for name in (
        "state_key_sets_identical",
        "same_shape_entries_bit_identical",
        "trained_v2_weight_loaded",
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
            "save_cpu_rng_after_determinism_then_replay_for_width24_and_width32_"
            "and_copy_every_same_shape_entry"
        ),
        "v2_reference_initial_state_sha256": (
            EXPECTED_V2_REFERENCE_INITIAL_STATE_SHA256[seed]
        ),
        "candidate_initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256[seed],
        "v2_reference_parameter_count": REGISTERED_V2_PARAMETER_COUNT,
        "candidate_parameter_count": REGISTERED_PARAMETER_COUNT,
        "state_key_sets_identical": True,
        "same_shape_entry_count": 130,
        "same_shape_entries_bit_identical": True,
        "only_shape_changed_state_keys": sorted(EXPECTED_SHAPE_CHANGES),
        "shape_changes": EXPECTED_SHAPE_CHANGES,
        "trained_v2_weight_loaded": False,
    }
    if proof != expected:
        raise ValueError("N32 V3 initialization comparability proof drift")
    return proof


def _validate_authoritative_result(
    artifact: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_controls: Mapping[str, Mapping[str, Any]],
    expected_patch7_reference: Mapping[str, Any],
) -> dict[str, Any]:
    if expected_seed not in EXPECTED_SEEDS or set(artifact) != TOP_LEVEL_FIELDS:
        raise ValueError("N32 V3 top-level result fields drift")
    if artifact.get("schema") == SMOKE_RESULT_SCHEMA:
        raise ValueError("N32 V3 finalizer rejects smoke results")
    if (
        artifact.get("schema") != RESULT_SCHEMA
        or artifact.get("authoritative") is not True
        or artifact.get("aggregation_eligible") is not True
        or artifact.get("promotion_eligible") is not False
        or base._strict_int(artifact.get("seed"), context="result seed")
        != expected_seed
    ):
        raise ValueError("N32 V3 result is not authoritative")
    for name in (
        "shared_jepa_full_train_candidate_licensed",
        "runtime_ready",
        "g2_licensed",
        "g3_licensed",
    ):
        if base._strict_bool(artifact.get(name), context=name) is not False:
            raise ValueError("single-seed N32 V3 result grants a forbidden license")
    core = dict(artifact)
    declared = core.pop("content_sha256", None)
    if not base._is_sha256(declared) or _canonical_json_sha256(core) != declared:
        raise ValueError("N32 V3 content hash mismatch")
    for name in ("created_at_utc", "completed_at_utc"):
        if not isinstance(artifact.get(name), str) or not artifact[name]:
            raise ValueError("N32 V3 timestamps are missing")
    invocation = artifact.get("invocation")
    if not isinstance(invocation, list) or not invocation or any(
        not isinstance(item, str) for item in invocation
    ):
        raise ValueError("N32 V3 invocation provenance is missing")

    execution = base._mapping(artifact.get("execution"), context="execution")
    expected_execution_fields = {
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
    if set(execution) != expected_execution_fields:
        raise ValueError("N32 V3 execution fields drift")
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
        raise ValueError("N32 V3 execution contract drift")
    stage_config = base._mapping(execution.get("stage_config"), context="stage config")
    if set(stage_config) != set(AUTHORITATIVE_CONFIG):
        raise ValueError("N32 V3 stage config fields drift")
    for name in ("updates", "batch_size"):
        base._strict_int(stage_config.get(name), context=f"stage config/{name}")
    for name in ("learning_rate_start", "learning_rate_end", "weight_decay"):
        base._finite_number(stage_config.get(name), context=f"stage config/{name}")
    determinism = base._mapping(execution.get("determinism"), context="determinism")
    expected_determinism = {
        "seed": expected_seed,
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
    if determinism != expected_determinism:
        raise ValueError("N32 V3 determinism contract drift")

    contract = base._mapping(artifact.get("contract"), context="contract")
    if contract != {
        "path": str(CONTRACT_PATH.resolve()),
        "sha256": EXECUTION_BINDING_SHA256,
    }:
        raise ValueError("N32 V3 execution binding drift")
    inputs = _validate_inputs(artifact.get("inputs"), seed=expected_seed)
    sources = base._mapping(artifact.get("source_hashes"), context="source hashes")
    if sources != _runner_source_hashes():
        raise ValueError("N32 V3 transitive source provenance drift")
    base._mapping(artifact.get("git"), context="git provenance")

    model = base._mapping(artifact.get("model"), context="model")
    expected_model_fields = {
        "class",
        "parameter_count",
        "token_feature_dim",
        "context_dim",
        "parameter_delta_from_v2",
        "initial_state_sha256",
        "initialization_comparability",
    }
    if set(model) != expected_model_fields:
        raise ValueError("N32 V3 model fields drift")
    for name in (
        "parameter_count",
        "token_feature_dim",
        "context_dim",
        "parameter_delta_from_v2",
    ):
        base._strict_int(model.get(name), context=f"model/{name}")
    proof = _validate_initialization_proof(
        model.get("initialization_comparability"), seed=expected_seed
    )
    if model != {
        "class": "CategoricalRadialPerceptionFullRayToken32",
        "parameter_count": REGISTERED_PARAMETER_COUNT,
        "token_feature_dim": REGISTERED_TOKEN_FEATURE_DIM,
        "context_dim": REGISTERED_CONTEXT_DIM,
        "parameter_delta_from_v2": REGISTERED_PARAMETER_DELTA,
        "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256[expected_seed],
        "initialization_comparability": proof,
    }:
        raise ValueError("N32 V3 model identity or initialization drift")

    reference = base._validate_patch7_reference(artifact.get("patch7_reference"))
    if reference != expected_patch7_reference:
        raise ValueError("N32 V3 patch7 reference drift")
    stages = base._mapping(artifact.get("stages"), context="stages")
    if set(stages) != {STAGE_NAME}:
        raise ValueError("N32 V3 must contain exactly one optimizer stage")
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
            raise ValueError("N32 V3 authorized holdouts are incomplete")
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
            raise ValueError("N32 V3 holdout decision or stage flag drift")
    else:
        if holdouts is not None or checks is not None or stage.get(
            "holdouts_evaluated"
        ) is not False:
            raise ValueError("N32 V3 unauthorized holdout payload/access flag exists")
        recomputed_checks = None
    recomputed_decision = per_seed_decision(stage, recomputed_checks)
    stored_decision = base._mapping(artifact.get("decision"), context="decision")
    for name in (
        "token_width_32_fit_passes",
        "favorable",
        "aggregation_eligible",
        "shared_jepa_full_train_candidate_licensed",
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
            raise ValueError("N32 V3 decision holdout keys drift")
        for panel in HOLDOUT_PANELS:
            base._strict_bool(holdout_passes.get(panel), context=f"decision/{panel}")
    if stored_decision != recomputed_decision:
        raise ValueError("N32 V3 stored decision does not recompute")
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
    """Fully validate the favorable primary before seed-11 device construction."""

    if path.resolve() != CANONICAL_RESULT_PATHS[20260710]:
        raise ValueError("seed-20260710 V3 authorization path is not canonical")
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
        raise ValueError("seed-20260710 V3 result is not favorable")
    if expected_runner_sources is not None and validated[
        "source_hashes"
    ] != expected_runner_sources:
        raise ValueError("seed-20260710 V3 source authorization drift")
    if expected_patch7_reference is not None and validated[
        "patch7_reference"
    ] != expected_patch7_reference:
        raise ValueError("seed-20260710 V3 reference authorization drift")
    final_hash = _sha256_file(path.resolve())
    if final_hash != ledger["pre_deserialization_sha256"]:
        raise RuntimeError("seed-20260710 V3 authorization changed during validation")
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
        "shared_jepa_full_train_candidate_licensed": both_favorable,
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
        parser.error("output already exists; N32 V3 finalization is immutable")
    for seed, path in (
        (20260710, args.seed_20260710_result),
        (20260711, args.seed_20260711_result),
    ):
        if path.resolve() != CANONICAL_RESULT_PATHS[seed]:
            parser.error(f"seed {seed} V3 result path is not canonical")
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
            raise ValueError(f"two N32 V3 seeds disagree on common {name}")
    for name in EXPECTED_INPUTS:
        if validated[20260710]["inputs"][name] != validated[20260711]["inputs"][name]:
            raise ValueError(f"two N32 V3 seeds disagree on input {name}")
    authorization = base._mapping(
        validated[20260711]["inputs"]["seed_20260710_authorization"],
        context="replication authorization",
    )
    if (
        authorization.get("sha256") != expected_hashes[20260710]
        or validated[20260710]["decision"].get("favorable") is not True
    ):
        raise ValueError("N32 V3 replication lacks favorable primary authorization")
    aggregation = _aggregate(validated[20260710], validated[20260711])
    for seed in EXPECTED_SEEDS:
        final_hash = _sha256_file(paths[seed])
        if final_hash != ledgers[seed]["pre_deserialization_sha256"]:
            raise RuntimeError(f"seed {seed} V3 result changed during finalization")
        ledgers[seed]["post_validation_sha256"] = final_hash
        ledgers[seed]["post_validation_unchanged"] = True
        ledgers[seed]["content_sha256"] = validated[seed]["content_sha256"]
        ledgers[seed]["decision_recomputed_exactly"] = True
    if _source_hashes() != source_start:
        raise RuntimeError("N32 V3 finalizer sources changed during execution")
    evidence_end = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    if evidence_end != bound["pre_hashes"]:
        raise RuntimeError("bound N32 V3 evidence changed during finalization")
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
        "shared_jepa_full_train_candidate_licensed": aggregation[
            "shared_jepa_full_train_candidate_licensed"
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
                "shared_jepa_full_train_candidate_licensed": result[
                    "shared_jepa_full_train_candidate_licensed"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
