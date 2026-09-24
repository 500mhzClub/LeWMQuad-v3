#!/usr/bin/env python3
"""Validate and aggregate two immutable exposure-matched N32 V2 results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from lewm.benchmarks.go2_categorical_radial_n32 import (  # noqa: E402
    HOLDOUT_PANELS,
    categorical_holdout_checks,
    terminal_fit_gate_summary,
)
from lewm.benchmarks.go2_categorical_radial_n32_v2 import (  # noqa: E402
    EXECUTION_BINDING_SHA256,
    RESULT_SCHEMA,
    SMOKE_RESULT_SCHEMA,
    STAGE_NAME,
    STAGE_SCHEMA,
    TWO_SEED_RESULT_SCHEMA,
    per_seed_decision,
)
from scripts import finalize_go2_categorical_radial_n32 as base  # noqa: E402


EXPECTED_SEEDS = (20260710, 20260711)
REGISTERED_PARAMETER_COUNT = 2_887_067
EXPECTED_INITIAL_STATE_SHA256 = {
    20260710: "8b149b57ae4bb305a2306a4dde2cab5f57a46f1c3760837593ed4d9862491278",
    20260711: "989e2db491d199bc544fabe2df40443a39f3ffc6e936f0d28c24625e7bd0ce13",
}
CONTRACT_PATH = (
    REPOSITORY_ROOT
    / "docs/lewm_go2_categorical_radial_n32_v2_exposure_binding_2026-07-11.md"
)
V1_RESULT_PATH = (
    REPOSITORY_ROOT
    / ".generated/go2_categorical_radial_n32/v1/seed_20260710_result.json"
)
V1_RESULT_NOTE_PATH = (
    REPOSITORY_ROOT / "docs/lewm_go2_categorical_radial_n32_v1_result_2026-07-11.md"
)
V1_RESULT_FILE_SHA256 = (
    "2f079925000ebbcd06843c413f4dcfd07fce93358482dd05512735af69cbc946"
)
V1_RESULT_CONTENT_SHA256 = (
    "ef023faff0e49888ca673cfab5fca0c1110852e49312ce339ecb7f03ab3a8d5b"
)
V1_RESULT_NOTE_SHA256 = (
    "4848c61e72be3b81bb4fe4ad0e545f9c3e6031df353c9d25f15a9dcd5109ddfd"
)
AUTHORITATIVE_CONFIG = {
    "updates": 2000,
    "batch_size": 80,
    "learning_rate_start": 2e-4,
    "learning_rate_end": 1e-5,
    "weight_decay": 1e-4,
}
SCHEDULE_SHA256 = {
    20260710: "79b6e66d4e90246f9eb045675f2a06eb25ae28d26f0997392b6780518e668156",
    20260711: "f621b85716607b7e7b8e1ba931d19cf552eb944feca48d099a2c1a3b8ef801c6",
}
CANONICAL_RESULT_PATHS = {
    seed: (
        REPOSITORY_ROOT
        / ".generated/go2_categorical_radial_n32/v2/"
        f"seed_{seed}_result.json"
    ).resolve()
    for seed in EXPECTED_SEEDS
}
EXPECTED_INPUTS = {
    **base.EXPECTED_INPUTS,
    "failed_v1_n32_result": (
        V1_RESULT_PATH,
        V1_RESULT_FILE_SHA256,
        V1_RESULT_CONTENT_SHA256,
    ),
    "failed_v1_n32_result_note": (
        V1_RESULT_NOTE_PATH,
        V1_RESULT_NOTE_SHA256,
        None,
    ),
}
BOUND_EVIDENCE = {
    **base.BOUND_EVIDENCE,
    V1_RESULT_PATH: V1_RESULT_FILE_SHA256,
    V1_RESULT_NOTE_PATH: V1_RESULT_NOTE_SHA256,
    CONTRACT_PATH: EXECUTION_BINDING_SHA256,
}
EXPECTED_PANEL_ARTIFACT_COUNTS = base.EXPECTED_PANEL_ARTIFACT_COUNTS
EVENT_FIELDS = {
    "image_requests",
    "target_requests",
    "image_decode_events",
    "label_shard_npz_open_events",
    "model_calls",
    "model_output_frames",
}
DATA_EVENT_FIELDS = {
    "image_requests",
    "target_requests",
    "image_decode_events",
    "label_shard_npz_open_events",
}
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
    "content_sha256",
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


def _cosine_learning_rate(update: int, total_updates: int) -> float:
    low, high = 1e-5, 2e-4
    return low + 0.5 * (high - low) * (
        1.0 + math.cos(math.pi * (update - 1) / (total_updates - 1))
    )


def _expected_schedule_contract(total_updates: int) -> dict[str, Any]:
    return {
        "name": "deterministic_stage_local_cosine_no_warmup_v3",
        "formula": (
            "1e-5 + 0.5 * (2e-4 - 1e-5) * "
            "(1 + cos(pi * (u - 1) / (U - 1)))"
        ),
        "one_indexed": True,
        "total_updates": total_updates,
        "first_learning_rate": 2e-4,
        "final_learning_rate": 1e-5,
        "warmup_updates": 0,
        "assignment_timing": "immediately_before_optimizer_step",
    }


def _expected_optimizer() -> dict[str, Any]:
    return {
        "name": "AdamW",
        "weight_decay": 1e-4,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "amsgrad": False,
        "gradient_clip": 1.0,
        "gradient_clip_applications": 2000,
        "learning_rate_schedule": _expected_schedule_contract(2000),
    }


def _runner_source_paths() -> dict[str, Path]:
    shared = {
        ("v1_n32_runner" if name == "runner" else name): path
        for name, path in base.RUNNER_SOURCE_PATHS.items()
    }
    return {
        **shared,
        "n32_v2_binding": CONTRACT_PATH,
        "n32_v2_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32_v2.py"
        ),
        "v1_n32_finalizer": (
            REPOSITORY_ROOT / "scripts/finalize_go2_categorical_radial_n32.py"
        ),
        "v2_finalizer": Path(__file__).resolve(),
        "runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_n32_v2.py",
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
        "n32_v2_pure": (
            REPOSITORY_ROOT / "lewm/benchmarks/go2_categorical_radial_n32_v2.py"
        ),
        "runner": REPOSITORY_ROOT / "scripts/run_go2_categorical_radial_n32_v2.py",
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
    if pre_hashes != {
        str(path.resolve()): digest for path, digest in BOUND_EVIDENCE.items()
    }:
        raise ValueError("bound N32 V2 evidence file SHA-256 drift")
    inherited = base._load_bound_evidence()
    v1_result, _ledger = base._load_expected_json(
        V1_RESULT_PATH,
        expected_sha256=V1_RESULT_FILE_SHA256,
    )
    controls = {
        seed: base._expected_controls(inherited["panels"], seed=seed)
        for seed in EXPECTED_SEEDS
    }
    validated_v1 = base._validate_authoritative_result(
        v1_result,
        expected_seed=20260710,
        expected_controls=controls[20260710],
    )
    if (
        v1_result.get("content_sha256") != V1_RESULT_CONTENT_SHA256
        or validated_v1["decision"].get("classification") != "fit_gate_failed"
        or validated_v1["decision"].get("favorable") is not False
    ):
        raise ValueError("bound failed N32 V1 result identity drift")
    post_hashes = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    if post_hashes != pre_hashes:
        raise RuntimeError("bound N32 V2 evidence changed during parsing")
    return {
        "pre_hashes": pre_hashes,
        "post_parse_hashes": post_hashes,
        "panels": inherited["panels"],
        "patch7_reference": inherited["patch7_reference"],
        "controls": controls,
    }


def _validate_minibatches(value: object, *, seed: int) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != 2000:
        raise ValueError("N32 V2 minibatch schedule has the wrong length")
    batches = []
    for batch in value:
        if (
            not isinstance(batch, list)
            or len(batch) != 80
            or any(type(index) is not int for index in batch)
            or len(set(batch)) != 80
            or any(not 0 <= index < 320 for index in batch)
        ):
            raise ValueError("N32 V2 minibatch schedule is malformed")
        batches.append(list(batch))
    for start in range(0, 2000, 4):
        epoch = [index for batch in batches[start : start + 4] for index in batch]
        if sorted(epoch) != list(range(320)):
            raise ValueError("N32 V2 epoch is not a complete frame permutation")
    if _canonical_json_sha256(value) != SCHEDULE_SHA256[seed]:
        raise ValueError("N32 V2 exact seeded minibatch schedule drift")
    return batches


def _validate_stage(
    value: object,
    *,
    seed: int,
    expected_initial_state: str,
    expected_controls: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    stage = base._mapping(value, context="N32 V2 stage")
    expected_fields = {
        "schema",
        "stage",
        "maximum_steps",
        "completed_steps",
        "batch_size",
        "batches_per_epoch",
        "effective_epochs",
        "frame_presentations",
        "presentations_per_fit_frame",
        "evaluation_interval",
        "optimizer",
        "one_direct_forward_backward_per_update",
        "gradient_accumulation_or_microbatching",
        "fixed_update_budget_consumed",
        "initial_state_sha256",
        "final_state_sha256",
        "minibatch_indices",
        "minibatch_indices_sha256",
        "learning_curve",
        "terminal_fit_gate",
        "training_access",
        "fit_evaluation_access",
        "holdouts_evaluated",
    }
    if set(stage) != expected_fields:
        raise ValueError("N32 V2 stage fields drift")
    if (
        stage.get("schema") != STAGE_SCHEMA
        or stage.get("stage") != STAGE_NAME
        or base._strict_int(stage.get("maximum_steps"), context="maximum steps")
        != 2000
        or base._strict_int(stage.get("completed_steps"), context="completed steps")
        != 2000
        or base._strict_int(stage.get("batch_size"), context="batch size") != 80
        or base._strict_int(stage.get("batches_per_epoch"), context="batches/epoch")
        != 4
        or base._finite_number(stage.get("effective_epochs"), context="epochs")
        != 500.0
        or base._strict_int(stage.get("frame_presentations"), context="presentations")
        != 160000
        or base._finite_number(
            stage.get("presentations_per_fit_frame"), context="per-frame exposure"
        )
        != 500.0
        or base._strict_int(stage.get("evaluation_interval"), context="eval interval")
        != 100
        or stage.get("one_direct_forward_backward_per_update") is not True
        or stage.get("gradient_accumulation_or_microbatching") is not False
        or stage.get("fixed_update_budget_consumed") is not True
        or stage.get("initial_state_sha256") != expected_initial_state
        or not base._is_sha256(stage.get("final_state_sha256"))
        or type(stage.get("holdouts_evaluated")) is not bool
    ):
        raise ValueError("N32 V2 exact stage contract drift")
    optimizer = base._mapping(stage.get("optimizer"), context="N32 V2 optimizer")
    for name in ("weight_decay", "epsilon", "gradient_clip"):
        base._finite_number(optimizer.get(name), context=f"optimizer/{name}")
    betas = optimizer.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("N32 V2 optimizer betas drift")
    for index, beta in enumerate(betas):
        base._unit_interval(beta, context=f"optimizer/betas/{index}")
    base._strict_bool(optimizer.get("amsgrad"), context="optimizer/amsgrad")
    base._strict_int(
        optimizer.get("gradient_clip_applications"),
        context="optimizer/gradient_clip_applications",
        minimum=1,
    )
    schedule_contract = base._mapping(
        optimizer.get("learning_rate_schedule"), context="learning-rate schedule"
    )
    base._strict_bool(schedule_contract.get("one_indexed"), context="schedule/one_indexed")
    base._strict_int(
        schedule_contract.get("total_updates"),
        context="schedule/total_updates",
        minimum=2,
    )
    base._strict_int(schedule_contract.get("warmup_updates"), context="schedule/warmup")
    for name in ("first_learning_rate", "final_learning_rate"):
        base._finite_number(schedule_contract.get(name), context=f"schedule/{name}")
    if optimizer != _expected_optimizer():
        raise ValueError("N32 V2 optimizer/schedule contract drift")
    batches = _validate_minibatches(stage.get("minibatch_indices"), seed=seed)
    if stage.get("minibatch_indices_sha256") != _canonical_json_sha256(batches):
        raise ValueError("N32 V2 stored schedule hash drift")
    curve = stage.get("learning_curve")
    if not isinstance(curve, list) or len(curve) != 20:
        raise ValueError("N32 V2 fit curve is incomplete")
    for point in curve:
        record = base._mapping(point, context="N32 V2 curve point")
        if set(record) != {
            "step",
            "learning_rate",
            "batch_loss",
            "gradient_norm_before_clip",
            "fit_panel",
        }:
            raise ValueError("N32 V2 curve point fields drift")
        step = base._strict_int(record.get("step"), context="curve step", minimum=1)
        if base._finite_number(
            record.get("learning_rate"), context="curve learning rate"
        ) != _cosine_learning_rate(step, 2000):
            raise ValueError("N32 V2 curve learning rate drift")
        for name in ("batch_loss", "gradient_norm_before_clip"):
            if base._finite_number(record.get(name), context=f"curve/{name}") < 0.0:
                raise ValueError(f"N32 V2 curve {name} must be nonnegative")
        base._validate_panel_report(
            record.get("fit_panel"),
            panel="fit",
            seed=seed,
            require_fit_gate=True,
            expected_controls=expected_controls,
        )
    terminal = terminal_fit_gate_summary(curve, 2000, 100)
    base._validate_terminal_summary(stage.get("terminal_fit_gate"), context="V2")
    if stage.get("terminal_fit_gate") != terminal:
        raise ValueError("N32 V2 terminal gate does not recompute")
    training = base._mapping(stage.get("training_access"), context="training access")
    evaluation = base._mapping(
        stage.get("fit_evaluation_access"), context="evaluation access"
    )
    if set(training) != EVENT_FIELDS or set(evaluation) != EVENT_FIELDS:
        raise ValueError("N32 V2 stage access fields drift")
    expected_training = {
        "image_requests": 160000,
        "target_requests": 160000,
        "image_decode_events": 320,
        "label_shard_npz_open_events": 20,
        "model_calls": 2000,
        "model_output_frames": 160000,
    }
    expected_evaluation = {
        "image_requests": 19200,
        "target_requests": 6400,
        "image_decode_events": 0,
        "label_shard_npz_open_events": 0,
        "model_calls": 1600,
        "model_output_frames": 19200,
    }
    for name, expected in expected_training.items():
        if base._strict_int(training.get(name), context=f"training/{name}") != expected:
            raise ValueError("N32 V2 direct-training access drift")
    for name, expected in expected_evaluation.items():
        if base._strict_int(evaluation.get(name), context=f"evaluation/{name}") != expected:
            raise ValueError("N32 V2 fit-evaluation access drift")
    return stage, terminal


def _validate_inputs(value: object, *, seed: int) -> Mapping[str, Any]:
    inputs = base._mapping(value, context="N32 V2 inputs")
    expected_keys = {*EXPECTED_INPUTS, "seed_20260710_authorization"}
    if set(inputs) != expected_keys:
        raise ValueError("N32 V2 immutable input keys drift")
    path_by_name = {
        "panel": base.PANEL_PATH,
        "ladder_manifest": base.LADDER_PATH,
        "v3_result": base.V3_RESULT_PATH,
        "patch7_reference_result": base.PATCH7_RESULT_PATH,
    }
    for name, raw in EXPECTED_INPUTS.items():
        if name in base.EXPECTED_INPUTS:
            file_hash, content_hash = raw
            path = path_by_name[name]
        else:
            path, file_hash, content_hash = raw
        record = base._mapping(inputs[name], context=f"input/{name}")
        expected = {"path": str(path.resolve()), "sha256": file_hash}
        if content_hash is not None:
            expected["content_sha256"] = content_hash
        if record != expected:
            raise ValueError(f"N32 V2 immutable input drift: {name}")
    authorization = inputs["seed_20260710_authorization"]
    if seed == 20260710 and authorization is not None:
        raise ValueError("N32 V2 primary must not carry authorization")
    if seed == 20260711:
        record = base._mapping(authorization, context="primary authorization")
        if (
            set(record) != {"path", "sha256"}
            or Path(str(record.get("path", ""))).resolve()
            != CANONICAL_RESULT_PATHS[20260710]
            or not base._is_sha256(record.get("sha256"))
        ):
            raise ValueError("N32 V2 replication authorization drift")
    return inputs


def _validate_artifact_verification(value: object) -> None:
    verification = base._mapping(value, context="artifact verification")
    if set(verification) != {
        "fit_verified_before_access",
        "holdouts_verified_only_after_terminal_fit_pass",
        "evidence_hashes",
    }:
        raise ValueError("N32 V2 artifact verification fields drift")
    if (
        verification.get("fit_verified_before_access") is not True
        or verification.get("holdouts_verified_only_after_terminal_fit_pass")
        is not True
    ):
        raise ValueError("N32 V2 artifact verification ordering drift")
    evidence = base._mapping(
        verification.get("evidence_hashes"), context="evidence hashes"
    )
    expected = {str(path.resolve()): digest for path, digest in BOUND_EVIDENCE.items()}
    if evidence != expected:
        raise ValueError("N32 V2 bound evidence mapping drift")


def _validate_access(value: object, *, holdouts_authorized: bool) -> None:
    access = base._mapping(value, context="N32 V2 access ledger")
    expected_top = {
        "panels",
        "fit_dataset_totals",
        "checkpoint_selection",
        "probability_calibration",
        "g2_evaluation",
        "non_train_image_opens",
        "non_train_label_shard_opens",
        "non_train_model_outputs",
    }
    if set(access) != expected_top:
        raise ValueError("N32 V2 access ledger fields drift")
    totals = base._mapping(access["fit_dataset_totals"], context="fit totals")
    expected_totals = {
        "image_requests": 179200,
        "target_requests": 166400,
        "image_decode_events": 320,
        "label_shard_npz_open_events": 20,
    }
    if set(totals) != DATA_EVENT_FIELDS:
        raise ValueError("N32 V2 fit total fields drift")
    for name, expected in expected_totals.items():
        if base._strict_int(totals.get(name), context=f"fit totals/{name}") != expected:
            raise ValueError("N32 V2 fit totals do not reconcile")
    panels = base._mapping(access["panels"], context="panel access")
    if set(panels) != {"fit", *HOLDOUT_PANELS}:
        raise ValueError("N32 V2 panel access keys drift")
    fit = base._mapping(panels["fit"], context="fit access")
    if fit.get("authorized") is not True:
        raise ValueError("N32 V2 fit authorization type drift")
    for name in (
        "artifact_hash_passes",
        "image_hash_byte_open_events",
        "shard_hash_byte_open_events",
    ):
        base._strict_int(fit.get(name), context=f"fit access/{name}")
    if fit != {
        "authorized": True,
        "artifact_hash_passes": 2,
        "image_hash_byte_open_events": 640,
        "shard_hash_byte_open_events": 40,
    }:
        raise ValueError("N32 V2 fit artifact access drift")
    for panel in HOLDOUT_PANELS:
        record = base._mapping(panels[panel], context=f"{panel} access")
        expected_shards = EXPECTED_PANEL_ARTIFACT_COUNTS[panel]["shards"]
        base._strict_bool(record.get("authorized"), context=f"{panel}/authorized")
        for name in (
            "artifact_hash_passes",
            "image_hash_byte_open_events",
            "shard_hash_byte_open_events",
        ):
            base._strict_int(record.get(name), context=f"{panel}/{name}")
        dataset_access = base._mapping(
            record.get("dataset_access"), context=f"{panel}/dataset_access"
        )
        if set(dataset_access) != EVENT_FIELDS:
            raise ValueError(f"N32 V2 {panel} dataset access fields drift")
        for name in EVENT_FIELDS:
            base._strict_int(dataset_access.get(name), context=f"{panel}/{name}")
        if holdouts_authorized:
            expected = {
                "authorized": True,
                "artifact_hash_passes": 2,
                "image_hash_byte_open_events": 640,
                "shard_hash_byte_open_events": 2 * expected_shards,
                "dataset_access": {
                    "image_requests": 960,
                    "target_requests": 320,
                    "image_decode_events": 320,
                    "label_shard_npz_open_events": expected_shards,
                    "model_calls": 80,
                    "model_output_frames": 960,
                },
            }
        else:
            expected = {
                "authorized": False,
                "artifact_hash_passes": 0,
                "image_hash_byte_open_events": 0,
                "shard_hash_byte_open_events": 0,
                "dataset_access": {name: 0 for name in sorted(EVENT_FIELDS)},
            }
        if record != expected:
            raise ValueError(f"N32 V2 {panel} conditional access drift")
    for role in ("checkpoint_selection", "probability_calibration", "g2_evaluation"):
        base._validate_zero_contact(access.get(role), context=f"{role} ledger")
    for name in (
        "non_train_image_opens",
        "non_train_label_shard_opens",
        "non_train_model_outputs",
    ):
        if base._strict_int(access.get(name), context=name) != 0:
            raise ValueError("N32 V2 records forbidden non-train access")


def _validate_authoritative_result(
    artifact: Mapping[str, Any],
    *,
    expected_seed: int,
    expected_controls: Mapping[str, Mapping[str, Any]],
    expected_patch7_reference: Mapping[str, Any],
) -> dict[str, Any]:
    if expected_seed not in EXPECTED_SEEDS or set(artifact) != TOP_LEVEL_FIELDS:
        raise ValueError("N32 V2 top-level result fields drift")
    if artifact.get("schema") == SMOKE_RESULT_SCHEMA:
        raise ValueError("N32 V2 finalizer rejects smoke results")
    if (
        artifact.get("schema") != RESULT_SCHEMA
        or artifact.get("authoritative") is not True
        or artifact.get("aggregation_eligible") is not True
        or artifact.get("promotion_eligible") is not False
        or base._strict_int(artifact.get("seed"), context="result seed")
        != expected_seed
        or artifact.get("categorical_radial_full_train_candidate_licensed")
        is not False
    ):
        raise ValueError("N32 V2 result is not authoritative")
    core = dict(artifact)
    declared = core.pop("content_sha256", None)
    if not base._is_sha256(declared) or _canonical_json_sha256(core) != declared:
        raise ValueError("N32 V2 content hash mismatch")
    for name in ("created_at_utc", "completed_at_utc"):
        if not isinstance(artifact.get(name), str) or not artifact[name]:
            raise ValueError("N32 V2 timestamps are missing")
    invocation = artifact.get("invocation")
    if not isinstance(invocation, list) or not invocation or any(
        not isinstance(item, str) for item in invocation
    ):
        raise ValueError("N32 V2 invocation provenance is missing")
    execution = base._mapping(artifact.get("execution"), context="execution")
    expected_execution_fields = {
        "device",
        "device_name",
        "determinism",
        "batch_size_frames",
        "evaluation_target_batch_size",
        "evaluation_combined_model_batch_size",
        "evaluation_interval",
        "stage_config",
        "effective_epochs",
        "fp32_no_autocast_amp_compile_or_quantization",
    }
    if set(execution) != expected_execution_fields:
        raise ValueError("N32 V2 execution fields drift")
    stage_config = base._mapping(execution.get("stage_config"), context="stage config")
    if set(stage_config) != set(AUTHORITATIVE_CONFIG):
        raise ValueError("N32 V2 stage config fields drift")
    for name in ("updates", "batch_size"):
        base._strict_int(stage_config.get(name), context=f"stage config/{name}")
    for name in ("learning_rate_start", "learning_rate_end", "weight_decay"):
        base._finite_number(stage_config.get(name), context=f"stage config/{name}")
    if (
        base._strict_int(execution.get("batch_size_frames"), context="train batch")
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
        or base._strict_int(execution.get("evaluation_interval"), context="eval interval")
        != 100
        or execution.get("stage_config") != AUTHORITATIVE_CONFIG
        or base._finite_number(execution.get("effective_epochs"), context="epochs")
        != 500.0
        or execution.get("fp32_no_autocast_amp_compile_or_quantization") is not True
    ):
        raise ValueError("N32 V2 execution contract drift")
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
        raise ValueError("N32 V2 determinism contract drift")
    contract = base._mapping(artifact.get("contract"), context="contract")
    if contract != {
        "path": str(CONTRACT_PATH.resolve()),
        "sha256": EXECUTION_BINDING_SHA256,
    }:
        raise ValueError("N32 V2 execution binding drift")
    inputs = _validate_inputs(artifact.get("inputs"), seed=expected_seed)
    sources = base._mapping(artifact.get("source_hashes"), context="source hashes")
    if sources != _runner_source_hashes():
        raise ValueError("N32 V2 transitive source provenance drift")
    base._mapping(artifact.get("git"), context="git provenance")
    model = base._mapping(artifact.get("model"), context="model")
    base._strict_int(model.get("parameter_count"), context="model/parameter_count")
    if model != {
        "class": "CategoricalRadialPerceptionFullRay",
        "parameter_count": REGISTERED_PARAMETER_COUNT,
        "initial_state_sha256": EXPECTED_INITIAL_STATE_SHA256[expected_seed],
    }:
        raise ValueError("N32 V2 model identity or initialization drift")
    reference = base._validate_patch7_reference(artifact.get("patch7_reference"))
    if reference != expected_patch7_reference:
        raise ValueError("N32 V2 patch7 reference drift")
    stages = base._mapping(artifact.get("stages"), context="stages")
    if set(stages) != {STAGE_NAME}:
        raise ValueError("N32 V2 must contain exactly one optimizer stage")
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
            raise ValueError("N32 V2 authorized holdouts are incomplete")
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
            raise ValueError("N32 V2 holdout decision or stage flag drift")
    else:
        if holdouts is not None or checks is not None or stage.get(
            "holdouts_evaluated"
        ) is not False:
            raise ValueError("N32 V2 unauthorized holdout payload/access flag exists")
        recomputed_checks = None
    recomputed_decision = per_seed_decision(stage, recomputed_checks)
    stored_decision = base._mapping(artifact.get("decision"), context="decision")
    for name in (
        "exposure_matched_v3_cosine_fit_passes",
        "favorable",
        "aggregation_eligible",
        "categorical_radial_full_train_candidate_licensed",
        "promotion_licensed",
    ):
        base._strict_bool(stored_decision.get(name), context=f"decision/{name}")
    holdout_passes = stored_decision.get("holdout_passes")
    if holdout_passes is not None:
        holdout_passes = base._mapping(holdout_passes, context="decision holdouts")
        if set(holdout_passes) != set(HOLDOUT_PANELS):
            raise ValueError("N32 V2 decision holdout keys drift")
        for panel in HOLDOUT_PANELS:
            base._strict_bool(holdout_passes.get(panel), context=f"decision/{panel}")
    if stored_decision != recomputed_decision:
        raise ValueError("N32 V2 stored decision does not recompute")
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
        raise ValueError("seed-20260710 V2 authorization path is not canonical")
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
        raise ValueError("seed-20260710 V2 result is not favorable")
    if expected_runner_sources is not None and validated[
        "source_hashes"
    ] != expected_runner_sources:
        raise ValueError("seed-20260710 V2 source authorization drift")
    if expected_patch7_reference is not None and validated[
        "patch7_reference"
    ] != expected_patch7_reference:
        raise ValueError("seed-20260710 V2 reference authorization drift")
    final_hash = _sha256_file(path.resolve())
    if final_hash != ledger["pre_deserialization_sha256"]:
        raise RuntimeError("seed-20260710 V2 authorization changed during validation")
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
        "promotion_licensed": False,
        "g2_licensed": False,
    }


def _atomic_write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"output already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-20260710-result", type=Path, required=True)
    parser.add_argument("--expected-seed-20260710-result-sha256", required=True)
    parser.add_argument("--seed-20260711-result", type=Path, required=True)
    parser.add_argument("--expected-seed-20260711-result-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; N32 V2 finalization is immutable")
    for seed, path in (
        (20260710, args.seed_20260710_result),
        (20260711, args.seed_20260711_result),
    ):
        if path.resolve() != CANONICAL_RESULT_PATHS[seed]:
            parser.error(f"seed {seed} V2 result path is not canonical")
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
            raise ValueError(f"two N32 V2 seeds disagree on common {name}")
    common_input_names = set(EXPECTED_INPUTS)
    for name in common_input_names:
        if validated[20260710]["inputs"][name] != validated[20260711]["inputs"][name]:
            raise ValueError(f"two N32 V2 seeds disagree on input {name}")
    authorization = base._mapping(
        validated[20260711]["inputs"]["seed_20260710_authorization"],
        context="replication authorization",
    )
    if (
        authorization.get("sha256") != expected_hashes[20260710]
        or validated[20260710]["decision"].get("favorable") is not True
    ):
        raise ValueError("N32 V2 replication lacks favorable primary authorization")
    aggregation = _aggregate(validated[20260710], validated[20260711])
    for seed in EXPECTED_SEEDS:
        final_hash = _sha256_file(paths[seed])
        if final_hash != ledgers[seed]["pre_deserialization_sha256"]:
            raise RuntimeError(f"seed {seed} V2 result changed during finalization")
        ledgers[seed]["post_validation_sha256"] = final_hash
        ledgers[seed]["post_validation_unchanged"] = True
        ledgers[seed]["content_sha256"] = validated[seed]["content_sha256"]
        ledgers[seed]["decision_recomputed_exactly"] = True
    if _source_hashes() != source_start:
        raise RuntimeError("N32 V2 finalizer sources changed during execution")
    evidence_end = {str(path.resolve()): _sha256_file(path) for path in BOUND_EVIDENCE}
    if evidence_end != bound["pre_hashes"]:
        raise RuntimeError("bound N32 V2 evidence changed during finalization")
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
