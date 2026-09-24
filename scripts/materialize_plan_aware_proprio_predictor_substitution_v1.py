#!/usr/bin/env python3
"""Conditional P1/PR materialisation for PLAN_AWARE_MONOTONE_JEPA_COST_V1.

This helper is deliberately outside the Stage-A route-ranker path.  It refuses
to reconstruct proprioception, open a proprioceptive predictor checkpoint, or
run predictor inference unless a self-digested Stage-B gate receipt explicitly
records that ``TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL`` passed.  Stage-C donor-input
reruns require a second receipt recording the prospective proprioception-
contribution decision.

The CPU phase replays the already frozen 40-block state prefix and records the
same deployment-valid sensed channels used by ``dev_proprio_predictor_v1``:
projected gravity, body gyro, relative joint position, and joint velocity.  It
does not create a state, candidate, target, or route outcome.  The GPU phase
loads the frozen P1/PR checkpoints in evaluation mode and uses
``dev_proprio_predictor_v1.unroll``.  Its learned absence token is therefore the
only representation of future proprioceptive slots.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.  NO TRAINING.  NO NAVIGATION.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import plan_aware_monotone_jepa_cost_v1_contract as CONTRACT  # noqa: E402


class MaterialisationError(RuntimeError):
    """Fail-closed conditional-materialisation error."""


EXPERIMENT_ID = "PLAN_AWARE_MONOTONE_JEPA_COST_V1"
STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
SOURCE_COMMIT = "1d799eb24d8171cb6d90bc0d0e375d9e1b0cc4f0"
STATE_COUNT = 48
CANDIDATE_COUNT = 12
HORIZONS = (1, 2, 3)
TENSOR_SHAPE = (768, 1024)
PREDICTION_STATE_SHAPE = (CANDIDATE_COUNT, len(HORIZONS), *TENSOR_SHAPE)
PROPRIO_SHAPE = (3, 5, 30)
FROZEN_WORKERS = 24

CPU_INTERPRETER = ROOT / ".generated/venvs/genesis_render_vulkan/bin/python"
GPU_INTERPRETER = Path("/home/andrewknowles/TinyQuadJEPA/bin/python")
SELF = Path(__file__).resolve()
SCRUBBED_PYTHON_ENVIRONMENT_KEYS = (
    "PYTHONPATH",
    "PYTHONHOME",
    "PYTHONUSERBASE",
    "PYTHONSTARTUP",
)

PREDECESSOR_ROOT = Path(CONTRACT.PREDECESSOR_TENSOR_PACKAGE["root"])
PREDECESSOR_CONTEXT_INDEX = (
    PREDECESSOR_ROOT
    / CONTRACT.PREDECESSOR_TENSOR_PACKAGE["context_reconstruction_index"]["path"]
)
PREDECESSOR_TENSOR_INDEX = (
    PREDECESSOR_ROOT / CONTRACT.PREDECESSOR_TENSOR_PACKAGE["tensor_index"]["path"]
)
STATE_MANIFEST = ROOT / CONTRACT.PANEL_BINDINGS["state_manifest"]["path"]
SPLIT_PATH = ROOT / CONTRACT.PANEL_BINDINGS["split"]["path"]

STAGE_B_GATE_SCHEMA = "plan_aware_monotone_jepa_cost_v1.stage_b_gate.v1"
STAGE_C_GATE_SCHEMA = "plan_aware_monotone_jepa_cost_v1.stage_c_gate.v1"
STAGE_A_GATE_EVIDENCE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.stage_a_gate_evidence.v1"
)
STAGE_B_GATE_EVIDENCE_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.stage_b_gate_evidence.v1"
)
CONTEXT_STATE_SCHEMA = "plan_aware_monotone_jepa_cost_v1.proprio_context_state.v1"
CONTEXT_INDEX_SCHEMA = "plan_aware_monotone_jepa_cost_v1.proprio_context_index.v1"
PREDICTION_INDEX_SCHEMA = "plan_aware_monotone_jepa_cost_v1.prediction_index.v1"
MATERIALISATION_RECEIPT_SCHEMA = (
    "plan_aware_monotone_jepa_cost_v1.proprio_predictor_materialisation.v1"
)

STAGE_C_ABLATIONS = (
    "PR_VISUAL_CONTEXT_DERANGED",
    "PR_PROPRIO_HISTORY_DERANGED",
    "PR_CONTROL_HISTORY_DERANGED",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def content_digest(value: Mapping[str, Any]) -> str:
    core = copy.deepcopy(dict(value))
    core.pop("content_digest", None)
    return hashlib.sha256(canonical_bytes(core)[:-1]).hexdigest()


def attach_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    output["content_digest"] = content_digest(output)
    return output


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise MaterialisationError(f"JSON object required: {path}")
    return value


def atomic_bytes(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(path, canonical_bytes(value))


def atomic_npy(path: Path, array: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.asarray(array), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def artifact_binding(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = Path(path)
    reference: Path = path
    if root is not None:
        try:
            reference = path.relative_to(root)
        except ValueError:
            pass
    return {
        "path": str(reference),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def verify_file_binding(path: Path, binding: Mapping[str, Any], label: str) -> None:
    path = Path(path)
    if not path.is_file():
        raise MaterialisationError(f"{label} is missing: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise MaterialisationError(f"{label} byte count drift")
    observed = sha256_file(path)
    if observed != str(binding["sha256"]):
        raise MaterialisationError(f"{label} SHA-256 drift: {observed}")


def resolve_artifact(reference: str, root: Path) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else Path(root) / path


def _validate_self_digest(value: Mapping[str, Any], label: str) -> None:
    observed = value.get("content_digest")
    if not isinstance(observed, str) or observed != content_digest(value):
        raise MaterialisationError(f"{label} self-digest mismatch")


def _validate_bound_artifact(
    record: Mapping[str, Any], *, base: Path, label: str
) -> Path:
    required = {"path", "sha256", "bytes"}
    if not required.issubset(record):
        raise MaterialisationError(f"{label} binding is incomplete")
    path = resolve_artifact(str(record["path"]), base)
    verify_file_binding(path, record, label)
    return path


def build_stage_b_gate_receipt(
    *,
    contract_freeze_commit: str,
    evaluation_contract_path: Path,
    stage_a_gate_evidence_path: Path,
    latent_ranker_checkpoint_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build, but do not write, the explicit Stage-B authorisation receipt."""

    return attach_digest(
        {
            "schema": STAGE_B_GATE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "contract_sha256": CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
            "contract_freeze_commit": str(contract_freeze_commit),
            "true_future_gate": {
                "classification": "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL",
                "pass": True,
            },
            "evaluation_contract": artifact_binding(
                evaluation_contract_path, root=output_root
            ),
            "stage_a_gate_evidence": artifact_binding(
                stage_a_gate_evidence_path, root=output_root
            ),
            "latent_ranker_checkpoint": artifact_binding(
                latent_ranker_checkpoint_path, root=output_root
            ),
            "stage_b_authorised": True,
            "predictor_training_authorised": False,
            "new_state_or_candidate_generation_authorised": False,
        }
    )


def validate_stage_b_gate_receipt(path: Path, *, output_root: Path) -> dict[str, Any]:
    value = load_json(path)
    _validate_self_digest(value, "Stage-B gate receipt")
    required = {
        "schema",
        "experiment_id",
        "contract_sha256",
        "contract_freeze_commit",
        "true_future_gate",
        "evaluation_contract",
        "stage_a_gate_evidence",
        "latent_ranker_checkpoint",
        "stage_b_authorised",
        "predictor_training_authorised",
        "new_state_or_candidate_generation_authorised",
        "content_digest",
    }
    if set(value) != required:
        raise MaterialisationError("Stage-B gate receipt key-set drift")
    if value["schema"] != STAGE_B_GATE_SCHEMA or value["experiment_id"] != EXPERIMENT_ID:
        raise MaterialisationError("Stage-B gate receipt identity drift")
    if value["contract_sha256"] != CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256:
        raise MaterialisationError("Stage-B contract digest drift")
    gate = value["true_future_gate"]
    if gate != {
        "classification": "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL",
        "pass": True,
    }:
        raise MaterialisationError("true-future gate did not explicitly pass")
    if value["stage_b_authorised"] is not True:
        raise MaterialisationError("Stage B is not explicitly authorised")
    if value["predictor_training_authorised"] is not False:
        raise MaterialisationError("gate receipt unexpectedly authorises predictor training")
    if value["new_state_or_candidate_generation_authorised"] is not False:
        raise MaterialisationError("gate receipt unexpectedly authorises panel generation")
    for key in ("evaluation_contract", "stage_a_gate_evidence", "latent_ranker_checkpoint"):
        _validate_bound_artifact(value[key], base=output_root, label=key)
    evaluation = load_json(
        resolve_artifact(str(value["evaluation_contract"]["path"]), output_root)
    )
    _validate_self_digest(evaluation, "evaluation contract")
    if (
        evaluation.get("experiment_contract_digest")
        != CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        or evaluation.get("final_epoch_only") is not True
        or evaluation.get("predictor_inference_authorised_before_true_gate") is not False
        or evaluation.get("pass") is not True
    ):
        raise MaterialisationError("evaluation contract does not authorise conditional Stage B")
    checkpoint = evaluation.get("checkpoint_bindings", {}).get(
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"
    )
    if not isinstance(checkpoint, Mapping):
        raise MaterialisationError("evaluation contract lacks the frozen latent ranker")
    checkpoint_path = resolve_artifact(str(checkpoint.get("path")), output_root)
    gate_checkpoint_path = resolve_artifact(
        str(value["latent_ranker_checkpoint"]["path"]), output_root
    )
    if checkpoint_path.resolve() != gate_checkpoint_path.resolve() or any(
        checkpoint.get(key) != value["latent_ranker_checkpoint"].get(key)
        for key in ("sha256", "bytes")
    ):
        raise MaterialisationError("Stage-B gate binds a different latent ranker")
    evidence = load_json(
        resolve_artifact(str(value["stage_a_gate_evidence"]["path"]), output_root)
    )
    _validate_self_digest(evidence, "Stage-A gate evidence")
    if (
        evidence.get("schema") != STAGE_A_GATE_EVIDENCE_SCHEMA
        or evidence.get("experiment_id") != EXPERIMENT_ID
        or evidence.get("contract_sha256")
        != CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        or evidence.get("source_freeze_commit") != value["contract_freeze_commit"]
    ):
        raise MaterialisationError("Stage-A gate evidence identity drift")
    decision = evidence.get("true_future_gate")
    if not isinstance(decision, Mapping) or (
        decision.get("pass") is not True
        or decision.get("classification") != "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL"
    ):
        raise MaterialisationError("bound Stage-A evidence does not pass the true-future gate")
    if evidence.get("pass") is not True:
        raise MaterialisationError("Stage-A gate evidence lacks its explicit pass")
    if evidence.get("evaluation_contract_content_digest") != evaluation.get(
        "content_digest"
    ):
        raise MaterialisationError("Stage-A evidence/evaluation-contract binding drift")
    return value


def validate_execution_correction_replay_receipt(
    path: Path, *, output_root: Path
) -> dict[str, Any]:
    """Validate the durable replay barrier without reopening replay artifacts."""

    expected_path = (output_root / "receipts/execution_correction_replay.json").resolve()
    if path.resolve() != expected_path or not expected_path.is_file():
        raise MaterialisationError("execution-correction replay receipt path drift")
    value = load_json(expected_path)
    _validate_self_digest(value, "execution-correction replay receipt")
    expected_byte_rows = [
        copy.deepcopy(row)
        for row in CONTRACT.EXECUTION_CORRECTION_ARCHIVE_INVENTORY_ROWS
        if row["path"] in CONTRACT.EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS
    ]
    expected_byte_rows.sort(
        key=lambda row: CONTRACT.EXECUTION_CORRECTION_BYTE_EXACT_REPLAY_PATHS.index(
            row["path"]
        )
    )
    expected_normalized_rows = [
        {
            "path": relative_path,
            "excluded_paths": list(
                CONTRACT.EXECUTION_CORRECTION_NORMALIZED_REPLAY_EXCLUSIONS[
                    relative_path
                ]
            ),
            "scientific_content_digest": expected_digest,
        }
        for relative_path, expected_digest in (
            CONTRACT.EXECUTION_CORRECTION_NORMALIZED_REPLAY_DIGESTS.items()
        )
    ]
    if (
        value.get("schema") != CONTRACT.EXECUTION_CORRECTION_REPLAY_SCHEMA_VERSION
        or value.get("experiment_id") != EXPERIMENT_ID
        or value.get("amendment")
        != CONTRACT.EXECUTION_CORRECTION_AMENDMENT_BINDING
        or value.get("failed_archive")
        != str(CONTRACT.EXECUTION_CORRECTION_FAILED_ARCHIVE)
        or Path(str(value.get("fresh_attempt", ""))).resolve()
        != output_root.resolve()
        or value.get("files_reused") != 0
        or value.get("byte_exact_replay") != expected_byte_rows
        or value.get("normalized_scientific_replay") != expected_normalized_rows
        or value.get("stage_b_started_before_replay_gate") is not False
        or value.get("stage_c_started_before_replay_gate") is not False
        or value.get("pass") is not True
    ):
        raise MaterialisationError("execution-correction replay receipt drift")
    return value


def build_stage_c_gate_receipt(
    *,
    contract_freeze_commit: str,
    stage_b_gate_receipt_path: Path,
    stage_b_metrics_path: Path,
    evaluation_contract_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build the independent Stage-C input-derangement authorisation."""

    return attach_digest(
        {
            "schema": STAGE_C_GATE_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "contract_sha256": CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
            "contract_freeze_commit": str(contract_freeze_commit),
            "proprioceptive_route_contribution": {
                "classification": "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION",
                "pass": True,
            },
            "stage_b_gate_receipt": artifact_binding(
                stage_b_gate_receipt_path, root=output_root
            ),
            "stage_b_metrics": artifact_binding(stage_b_metrics_path, root=output_root),
            "evaluation_contract": artifact_binding(
                evaluation_contract_path, root=output_root
            ),
            "stage_c_authorised": True,
            "training_authorised": False,
        }
    )


def validate_stage_c_gate_receipt(
    path: Path, *, output_root: Path, stage_b_gate_path: Path
) -> dict[str, Any]:
    value = load_json(path)
    _validate_self_digest(value, "Stage-C gate receipt")
    if value.get("schema") != STAGE_C_GATE_SCHEMA:
        raise MaterialisationError("Stage-C gate receipt schema drift")
    if value.get("experiment_id") != EXPERIMENT_ID:
        raise MaterialisationError("Stage-C gate receipt identity drift")
    if value.get("contract_sha256") != CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256:
        raise MaterialisationError("Stage-C contract digest drift")
    if value.get("proprioceptive_route_contribution") != {
        "classification": "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION",
        "pass": True,
    }:
        raise MaterialisationError("proprioceptive route contribution was not supported")
    if value.get("stage_c_authorised") is not True or value.get("training_authorised") is not False:
        raise MaterialisationError("Stage C is not explicitly inference-only")
    for key in ("stage_b_gate_receipt", "stage_b_metrics", "evaluation_contract"):
        if not isinstance(value.get(key), Mapping):
            raise MaterialisationError(f"Stage-C {key} binding absent")
        _validate_bound_artifact(value[key], base=output_root, label=f"Stage-C {key}")
    bound_b = resolve_artifact(
        str(value["stage_b_gate_receipt"]["path"]), output_root
    ).resolve()
    if bound_b != Path(stage_b_gate_path).resolve():
        raise MaterialisationError("Stage-C receipt binds a different Stage-B gate")
    validate_stage_b_gate_receipt(stage_b_gate_path, output_root=output_root)
    bound_stage_b = load_json(stage_b_gate_path)
    if value["evaluation_contract"] != bound_stage_b["evaluation_contract"]:
        raise MaterialisationError(
            "Stage-C receipt does not preserve the Stage-B evaluation contract"
        )
    evaluation = load_json(
        resolve_artifact(str(value["evaluation_contract"]["path"]), output_root)
    )
    _validate_self_digest(evaluation, "Stage-C evaluation contract")
    evidence = load_json(
        resolve_artifact(str(value["stage_b_metrics"]["path"]), output_root)
    )
    _validate_self_digest(evidence, "Stage-B gate evidence")
    if (
        evidence.get("schema") != STAGE_B_GATE_EVIDENCE_SCHEMA
        or evidence.get("experiment_id") != EXPERIMENT_ID
        or evidence.get("contract_sha256")
        != CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
        or evidence.get("source_freeze_commit") != value["contract_freeze_commit"]
    ):
        raise MaterialisationError("Stage-B gate evidence identity drift")
    decision = evidence.get("proprioception_gate")
    if not isinstance(decision, Mapping) or (
        decision.get("pass") is not True
        or decision.get("classification") != "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION"
    ):
        raise MaterialisationError("bound Stage-B evidence does not support proprioception")
    if evidence.get("pass") is not True:
        raise MaterialisationError("Stage-B gate evidence lacks its explicit pass")
    return value


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def validate_runtime_authority(gate: Mapping[str, Any]) -> None:
    head = _git("rev-parse", "HEAD")
    if head != str(gate["contract_freeze_commit"]):
        raise MaterialisationError("runtime HEAD is not the gate-bound freeze commit")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise MaterialisationError("scientific source commit is not an ancestor")
    if _git("status", "--porcelain"):
        raise MaterialisationError("conditional scientific execution requires a clean tree")


def validate_frozen_inputs() -> None:
    verify_file_binding(
        STATE_MANIFEST, CONTRACT.PANEL_BINDINGS["state_manifest"], "state manifest"
    )
    verify_file_binding(SPLIT_PATH, CONTRACT.PANEL_BINDINGS["split"], "split")
    verify_file_binding(
        PREDECESSOR_CONTEXT_INDEX,
        CONTRACT.PREDECESSOR_TENSOR_PACKAGE["context_reconstruction_index"],
        "predecessor context index",
    )
    verify_file_binding(
        PREDECESSOR_TENSOR_INDEX,
        CONTRACT.PREDECESSOR_TENSOR_PACKAGE["tensor_index"],
        "predecessor tensor index",
    )
    normalisation = CONTRACT.PROPRIO_INPUT_BINDINGS["normalisation"]
    verify_file_binding(Path(normalisation["path"]), normalisation, "proprio normalisation")


def _state_manifest_rows() -> list[dict[str, Any]]:
    payload = load_json(STATE_MANIFEST)
    rows = payload.get("state_candidates")
    if not isinstance(rows, list) or len(rows) != STATE_COUNT:
        raise MaterialisationError("state manifest cardinality drift")
    identities = [str(row.get("state_id")) for row in rows]
    if len(set(identities)) != STATE_COUNT:
        raise MaterialisationError("state manifest identities are not unique")
    return [dict(row) for row in rows]


def _context_records() -> dict[str, dict[str, Any]]:
    payload = load_json(PREDECESSOR_CONTEXT_INDEX)
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != STATE_COUNT:
        raise MaterialisationError("context index cardinality drift")
    output = {str(row.get("state_id")): dict(row) for row in rows}
    if len(output) != STATE_COUNT:
        raise MaterialisationError("context identities are not unique")
    return output


def _tensor_records() -> dict[tuple[str, str, int | None, int | None], dict[str, Any]]:
    payload = load_json(PREDECESSOR_TENSOR_INDEX)
    rows = payload.get("records")
    expected = CONTRACT.PREDECESSOR_TENSOR_PACKAGE["tensor_index"]["logical_records"]
    if not isinstance(rows, list) or len(rows) != expected:
        raise MaterialisationError("predecessor tensor index cardinality drift")
    output: dict[tuple[str, str, int | None, int | None], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("kind")),
            str(row.get("state_id")),
            row.get("candidate_index_or_null"),
            row.get("horizon_or_null"),
        )
        if key in output:
            raise MaterialisationError(f"duplicate tensor identity {key}")
        output[key] = dict(row)
    return output


def _normalisation_stats() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    binding = CONTRACT.PROPRIO_INPUT_BINDINGS["normalisation"]
    path = Path(binding["path"])
    value = load_json(path)
    if value.get("sha256") != binding["stats_sha256"]:
        raise MaterialisationError("proprio normalisation logical digest drift")
    mean = np.asarray(value.get("mean"), dtype=np.float32)
    std = np.asarray(value.get("std"), dtype=np.float32)
    if mean.shape != (30,) or std.shape != (30,) or np.any(std <= 0):
        raise MaterialisationError("proprio normalisation tensor contract drift")
    if not np.array_equal(mean[:3], np.zeros(3, np.float32)):
        raise MaterialisationError("projected-gravity mean is not the frozen identity")
    if not np.array_equal(std[:3], np.ones(3, np.float32)):
        raise MaterialisationError("projected-gravity scale is not the frozen identity")
    return mean, std, value


def sensed_proprio_feature(deployment_value: np.ndarray) -> np.ndarray:
    """Convert frozen 42-D deployment observation to the trained 30-D contract."""

    value = np.asarray(deployment_value, dtype=np.float32)
    if value.shape != (42,) or not np.isfinite(value).all():
        raise MaterialisationError("deployment proprioception must be finite [42]")
    sensed = value[:30].copy()
    # The training manifest stores g_body - (0, 0, -1), not raw g_body.
    sensed[:3] -= np.asarray((0.0, 0.0, -1.0), dtype=np.float32)
    if not np.isfinite(sensed).all():
        raise MaterialisationError("derived sensed proprioception is non-finite")
    return sensed


def normalize_proprio_history(
    raw: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != PROPRIO_SHAPE or not np.isfinite(raw).all():
        raise MaterialisationError("raw proprioception must be finite [3,5,30]")
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (30,) or std.shape != (30,) or np.any(std <= 0):
        raise MaterialisationError("normalisation vectors must be [30] with positive std")
    output = (raw - mean) / std
    if output.shape != PROPRIO_SHAPE or not np.isfinite(output).all():
        raise MaterialisationError("normalised proprioception contract failure")
    return output.astype(np.float32, copy=False)


def _control_history_from_blocks(executed: Mapping[int, np.ndarray]) -> np.ndarray:
    try:
        rows = np.concatenate(
            [np.asarray(executed[index], np.float32) for index in (37, 38, 39, 40)],
            axis=0,
        )
    except KeyError as exc:
        raise MaterialisationError("replay lacks block 37..40") from exc
    if rows.shape != (20, 3):
        raise MaterialisationError("warmup command history shape drift")
    return rows[4:19, (0, 2)].reshape(3, 5, 2)


def _capture_prefix_state(
    state_index: int, *, output_root: Path, gate_digest: str
) -> dict[str, Any]:
    """Run one exact frozen prefix and persist only observed proprioception."""

    # Simulator imports remain inside the CPU-only worker.
    from scripts import materialize_deployment_valid_dense_proprioception_v1 as DP
    from scripts import run_go2_oracle_branch_pilot_v1_2 as V

    started = time.time()
    rows = _state_manifest_rows()
    if not 0 <= state_index < len(rows):
        raise MaterialisationError("state index outside frozen panel")
    state = rows[state_index]
    state_id = str(state["state_id"])
    context = _context_records()[state_id]
    if str(context.get("family")) != str(state.get("family")):
        raise MaterialisationError(f"{state_id}: family drift")

    shared = V.V1._load_shared("cpu")
    ctx = V.V1.build_context(
        Path(str(state["scene_dir"])), seed=int(state["seed"]), backend="cpu", shared=shared
    )
    ctx.begin_episode()
    samples: list[np.ndarray] = []
    sample_rows: list[dict[str, Any]] = []
    executed_by_block: dict[int, np.ndarray] = {}
    steps_per_tick = int(ctx.runner._policy_steps_per_command_tick)

    for block_index in range(1, 41):
        original_execute = ctx.runner.execute_requested_block
        capture = block_index in (38, 39, 40)

        def execute_with_capture(
            requested_block: np.ndarray,
            *,
            after_policy_step: Any | None = None,
            _capture: bool = capture,
            _block_index: int = block_index,
        ) -> Any:
            clipped = ctx.runner._clip_block(
                np.asarray(requested_block, dtype=np.float32)
            )
            executed = np.asarray(clipped.executed, dtype=np.float32)

            def combined(tick_index: int, policy_step_index: int) -> None:
                if after_policy_step is not None:
                    after_policy_step(tick_index, policy_step_index)
                if not _capture or policy_step_index != steps_per_tick - 1:
                    return
                command = executed[:, tick_index]
                observed = DP.deployment_proprio(ctx.runner, command)
                sample_index = len(samples)
                samples.append(sensed_proprio_feature(observed))
                sample_rows.append(
                    {
                        "slot": sample_index // 5,
                        "slot_tick": sample_index % 5,
                        "replay_block": _block_index,
                        "block_tick": int(tick_index),
                        "observed_command_tick_index": (
                            (_block_index - 1) * 5 + int(tick_index) + 1
                        ),
                        "physics_observed": True,
                        "future_proprioception": False,
                        "executed_active_command_vx_yaw": [
                            float(command[0, 0]),
                            float(command[0, 2]),
                        ],
                        "sim_time_ns": int(ctx.runner._sim_time_ns),
                    }
                )

            return original_execute(
                requested_block, after_policy_step=combined if _capture else after_policy_step
            )

        if capture:
            ctx.runner.execute_requested_block = execute_with_capture
        try:
            block = ctx.drive_one_block()
        finally:
            if capture:
                ctx.runner.execute_requested_block = original_execute
        block_executed = np.asarray(block.executed, dtype=np.float32)
        if block_executed.shape != (1, 5, 3):
            raise MaterialisationError(f"{state_id}: replay command block shape drift")
        executed_by_block[block_index] = block_executed[0].copy()
        if capture and ctx.reset_in_last_block:
            raise MaterialisationError(f"{state_id}: frozen prefix unexpectedly reset")

    if len(samples) != 15:
        raise MaterialisationError(f"{state_id}: captured {len(samples)} samples, expected 15")
    raw = np.stack(samples).reshape(PROPRIO_SHAPE).astype(np.float32, copy=False)
    mean, std, stats = _normalisation_stats()
    normalized = normalize_proprio_history(raw, mean, std)

    # Verify the same post-block-40 state identity used by the predecessor.
    topology = V.link_topology(ctx)
    eligible = V.eligible_here(ctx, topology)
    if isinstance(eligible, str):
        raise MaterialisationError(f"{state_id}: replay eligibility changed: {eligible}")
    goal_record, _field = eligible
    snapshot = V.V1.capture_branch_state(
        ctx,
        goal=dict(goal_record["goal"]),
        identity={
            "state_id": state_id,
            "scene_id": str(state["scene_id"]),
            "family": str(state["family"]),
        },
    )
    expected_snapshot = context.get("replay_snapshot_digest")
    if expected_snapshot is None:
        expected_snapshot = context.get("branch_snapshot_digest")
    if not isinstance(expected_snapshot, str) or snapshot.digest != expected_snapshot:
        raise MaterialisationError(f"{state_id}: replay snapshot identity drift")

    control_raw = _control_history_from_blocks(executed_by_block)
    expected_control = np.asarray(context["control_history_raw_3x5x2"], np.float32)
    if expected_control.shape != (3, 5, 2) or not np.array_equal(control_raw, expected_control):
        raise MaterialisationError(f"{state_id}: frozen control history mismatch")
    control_mean = np.asarray(stats.get("control_mean"), np.float32)
    control_std = np.asarray(stats.get("control_std"), np.float32)
    if control_mean.shape != (2,) or control_std.shape != (2,) or np.any(control_std <= 0):
        raise MaterialisationError("frozen control normalisation contract drift")
    control_normalized = (control_raw - control_mean) / control_std
    expected_control_normalized = np.asarray(
        context["control_history_normalized_3x5x2"], np.float32
    )
    if (
        expected_control_normalized.shape != (3, 5, 2)
        or not np.array_equal(control_normalized, expected_control_normalized)
    ):
        raise MaterialisationError(f"{state_id}: normalized control history mismatch")

    shard = output_root / "stage_b/proprio_context/states" / f"{state_id}.npz"
    shard.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{shard.name}.tmp-", dir=shard.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(
                handle,
                raw=raw,
                normalized=normalized,
                control_history_raw=control_raw.astype(np.float32, copy=False),
                control_history_normalized=control_normalized.astype(np.float32, copy=False),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, shard)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    record = attach_digest(
        {
            "schema": CONTEXT_STATE_SCHEMA,
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "stage_b_gate_digest": gate_digest,
            "state_index": state_index,
            "state_id": state_id,
            "family": str(state["family"]),
            "split_role": str(context["role"]),
            "replay": {
                "mode": "FROZEN_STATE_RECONSTRUCTION_REPLAY",
                "blocks": 40,
                "prefix_blocks_only": True,
                "state_or_candidates_created": 0,
                "snapshot_digest": snapshot.digest,
                "predecessor_snapshot_digest": expected_snapshot,
                "snapshot_exact": True,
            },
            "observed_history": {
                "shape": list(PROPRIO_SHAPE),
                "dtype": "float32",
                "slots": 3,
                "samples_per_slot": 5,
                "channels": 30,
                "sample_rows": sample_rows,
                "future_slots_read": 0,
                "future_proprioception_forbidden": True,
            },
            "normalisation": {
                **artifact_binding(Path(CONTRACT.PROPRIO_INPUT_BINDINGS["normalisation"]["path"])),
                "stats_sha256": stats["sha256"],
                "gravity_offset_only": [0.0, 0.0, -1.0],
            },
            "control_history_exact_predecessor_match": True,
            "shard": artifact_binding(shard, root=output_root),
            "runtime_s": time.time() - started,
            "training_executed": False,
            "route_outcomes_opened": False,
        }
    )
    receipt_path = output_root / "stage_b/proprio_context/receipts" / f"{state_id}.json"
    atomic_json(receipt_path, record)
    return record


def _load_context_index(output_root: Path, gate_digest: str) -> dict[str, dict[str, Any]]:
    path = output_root / "stage_b/proprio_context/index.json"
    value = load_json(path)
    _validate_self_digest(value, "proprio context index")
    if (
        value.get("schema") != CONTEXT_INDEX_SCHEMA
        or value.get("complete") is not True
        or value.get("stage_b_gate_digest") != gate_digest
        or value.get("records") != STATE_COUNT
    ):
        raise MaterialisationError("proprio context index contract drift")
    records = value.get("state_records")
    if not isinstance(records, list) or len(records) != STATE_COUNT:
        raise MaterialisationError("proprio context state records are incomplete")
    output: dict[str, dict[str, Any]] = {}
    for summary in records:
        receipt_path = resolve_artifact(str(summary["receipt"]["path"]), output_root)
        verify_file_binding(receipt_path, summary["receipt"], "proprio state receipt")
        record = load_json(receipt_path)
        _validate_self_digest(record, "proprio state receipt")
        state_id = str(record["state_id"])
        if state_id in output:
            raise MaterialisationError("duplicate proprio state receipt")
        output[state_id] = record
    return output


def _write_context_index(output_root: Path, gate_digest: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for state in _state_manifest_rows():
        state_id = str(state["state_id"])
        receipt_path = output_root / "stage_b/proprio_context/receipts" / f"{state_id}.json"
        record = load_json(receipt_path)
        _validate_self_digest(record, f"{state_id} proprio receipt")
        if record.get("stage_b_gate_digest") != gate_digest:
            raise MaterialisationError(f"{state_id}: Stage-B gate binding drift")
        shard = resolve_artifact(str(record["shard"]["path"]), output_root)
        verify_file_binding(shard, record["shard"], f"{state_id} proprio shard")
        records.append(
            {
                "state_id": state_id,
                "family": str(record["family"]),
                "split_role": str(record["split_role"]),
                "receipt": artifact_binding(receipt_path, root=output_root),
                "shard": copy.deepcopy(record["shard"]),
            }
        )
    value = attach_digest(
        {
            "schema": CONTEXT_INDEX_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "stage_b_gate_digest": gate_digest,
            "complete": True,
            "records": len(records),
            "shape_per_state": list(PROPRIO_SHAPE),
            "dtype": "float32",
            "state_records": records,
            "reconstruction_prefix_blocks": STATE_COUNT * 40,
            "new_states": 0,
            "new_candidates": 0,
            "future_proprioception_read_count": 0,
            "training_executed": False,
        }
    )
    atomic_json(output_root / "stage_b/proprio_context/index.json", value)
    return value


def _load_context_arrays(
    state_id: str,
    *,
    output_root: Path,
    contexts: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    record = contexts[state_id]
    shard = resolve_artifact(str(record["shard"]["path"]), output_root)
    verify_file_binding(shard, record["shard"], f"{state_id} proprio shard")
    with np.load(shard, allow_pickle=False) as archive:
        raw = np.asarray(archive["raw"], np.float32)
        normalized = np.asarray(archive["normalized"], np.float32)
        control = np.asarray(archive["control_history_normalized"], np.float32)
    if raw.shape != PROPRIO_SHAPE or normalized.shape != PROPRIO_SHAPE:
        raise MaterialisationError(f"{state_id}: proprio shard shape drift")
    if control.shape != (3, 5, 2):
        raise MaterialisationError(f"{state_id}: control shard shape drift")
    if not all(np.isfinite(value).all() for value in (raw, normalized, control)):
        raise MaterialisationError(f"{state_id}: non-finite context input")
    return raw, normalized, control, artifact_binding(shard, root=output_root)


def _load_predecessor_context_tokens(
    state_id: str,
    tensor_rows: Mapping[tuple[str, str, int | None, int | None], Mapping[str, Any]],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    arrays: list[np.ndarray] = []
    bindings: list[dict[str, Any]] = []
    for horizon in (-2, -1, 0):
        try:
            record = tensor_rows[("CONTEXT", state_id, None, horizon)]
        except KeyError as exc:
            raise MaterialisationError(f"{state_id}: context latent h{horizon} absent") from exc
        path = resolve_artifact(str(record["path"]), PREDECESSOR_ROOT)
        verify_file_binding(path, record, f"{state_id} context latent h{horizon}")
        array = np.load(path, allow_pickle=False)
        if array.shape != TENSOR_SHAPE or array.dtype != np.float16:
            raise MaterialisationError(f"{state_id}: context latent contract drift")
        arrays.append(array)
        bindings.append(
            {
                "slot": horizon,
                "path": str(path),
                "sha256": str(record["sha256"]),
                "bytes": int(record["bytes"]),
            }
        )
    value = np.stack(arrays)
    if value.shape != (3, *TENSOR_SHAPE):
        raise MaterialisationError(f"{state_id}: context stack drift")
    return value, bindings


def _parameter_digest(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _load_frozen_predictor(source_id: str, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from scripts import dev_proprio_predictor_v1 as P

    if source_id not in ("P1_PROPRIO_ONE_STEP", "PR_PROPRIO_ROLLOUT"):
        raise MaterialisationError(f"unsupported proprio predictor source {source_id}")
    binding = CONTRACT.CHECKPOINT_BINDINGS[source_id]
    path = Path(binding["path"])
    verify_file_binding(path, binding, f"{source_id} checkpoint")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "cell": "proprio_one_step" if source_id.startswith("P1_") else "proprio_rollout",
        "use_proprio": True,
        "rollout": source_id.startswith("PR_"),
        "width": 384,
    }
    if payload.get("model_config") != expected:
        raise MaterialisationError(f"{source_id}: checkpoint model config drift")
    model = P.build_paired(
        CONTRACT.PREDICTOR_SEED, use_proprio=True, width=384, depth=6, heads=6
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise MaterialisationError(f"{source_id}: checkpoint did not freeze")
    parameter_digest = _parameter_digest(model)
    return model, {
        "source_id": source_id,
        "checkpoint": copy.deepcopy(binding),
        "model_config": expected,
        "strict_state_dict_load": True,
        "eval_mode": True,
        "requires_grad_all_false": True,
        "parameter_digest_before": parameter_digest,
        "checkpoint_optimizer_state_deserialized_but_ignored": (
            "optimizer_state_dict" in payload
        ),
        "optimizer_state_loaded_into_an_optimizer": False,
        "optimizer_steps": 0,
    }


def donor_input_sources(
    *,
    recipient: str,
    ablation: str | None,
    mapping: Mapping[str, str] | None,
) -> dict[str, str]:
    output = {"visual": recipient, "proprio": recipient, "control": recipient}
    if ablation is None:
        if mapping is not None:
            raise MaterialisationError("normal inference cannot carry a donor mapping")
        return output
    if ablation not in STAGE_C_ABLATIONS or mapping is None:
        raise MaterialisationError("Stage-C ablation or donor mapping is invalid")
    donor = mapping.get(recipient)
    if not isinstance(donor, str) or donor == recipient:
        raise MaterialisationError(f"{recipient}: Stage-C donor is not a derangement")
    component = {
        "PR_VISUAL_CONTEXT_DERANGED": "visual",
        "PR_PROPRIO_HISTORY_DERANGED": "proprio",
        "PR_CONTROL_HISTORY_DERANGED": "control",
    }[ablation]
    output[component] = donor
    return output


def _stage_c_mapping(
    *,
    ablation: str,
    evaluation_contract: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    all_maps = evaluation_contract.get("stage_c_input_donor_mappings")
    if not isinstance(all_maps, Mapping) or not isinstance(all_maps.get(ablation), Mapping):
        raise MaterialisationError("evaluation contract lacks Stage-C donor mapping")
    mapping = {str(key): str(value) for key, value in all_maps[ablation].items()}
    if set(mapping) != set(states):
        raise MaterialisationError("Stage-C donor mapping does not cover all states")
    for recipient, donor in mapping.items():
        if donor == recipient or donor not in states:
            raise MaterialisationError("Stage-C donor mapping is not a derangement")
        if str(states[recipient]["family"]) != str(states[donor]["family"]):
            raise MaterialisationError("Stage-C donor crosses maze family")
        if str(contexts[recipient]["split_role"]) != str(contexts[donor]["split_role"]):
            raise MaterialisationError("Stage-C donor crosses frozen split")
    return mapping


def _predict_source(
    source_id: str,
    *,
    output_root: Path,
    gate_digest: str,
    stage_c_gate_digest: str | None = None,
    ablation: str | None = None,
    donor_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    import torch
    from scripts import dev_proprio_predictor_v1 as P
    from scripts import run_dev_v03_temporal_action_jepa_v1 as T

    started = time.time()
    if ablation is not None and source_id != "PR_PROPRIO_ROLLOUT":
        raise MaterialisationError("Stage-C ablations use only frozen PR")
    if (ablation is None) != (stage_c_gate_digest is None):
        raise MaterialisationError(
            "Stage-C gate digest must be present exactly for an ablation"
        )
    contexts = _load_context_index(output_root, gate_digest)
    predecessor_context = _context_records()
    tensor_rows = _tensor_records()
    state_rows = {str(row["state_id"]): row for row in _state_manifest_rows()}
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise MaterialisationError("frozen predictor inference requires cuda:0")
    model, custody = _load_frozen_predictor(source_id, device)

    output_source = source_id if ablation is None else ablation
    logical_records: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []
    calls = 0
    for state_id in sorted(state_rows, key=lambda value: int(value.rsplit("-", 1)[1])):
        donors = donor_input_sources(
            recipient=state_id, ablation=ablation, mapping=donor_mapping
        )
        visual_np, visual_bindings = _load_predecessor_context_tokens(
            donors["visual"], tensor_rows
        )
        _raw, proprio_np, _control_from_shard, proprio_binding = _load_context_arrays(
            donors["proprio"], output_root=output_root, contexts=contexts
        )
        _raw_c, _proprio_c, control_np, control_binding = _load_context_arrays(
            donors["control"], output_root=output_root, contexts=contexts
        )
        recipient_context = predecessor_context[state_id]
        actions_np = np.asarray(
            recipient_context["action_blocks_raw_3x10_by_candidate"], np.float32
        )
        if actions_np.shape != (CANDIDATE_COUNT, 3, 10):
            raise MaterialisationError(f"{state_id}: candidate action contract drift")
        context = torch.from_numpy(visual_np).to(device=device, dtype=torch.float32)
        context = T.normalise(context).unsqueeze(0).repeat(CANDIDATE_COUNT, 1, 1, 1)
        actions = [
            torch.from_numpy(actions_np[:, horizon]).to(device=device, dtype=torch.float32)
            for horizon in range(3)
        ]
        proprio = (
            torch.from_numpy(proprio_np)
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(CANDIDATE_COUNT, 1, 1, 1)
        )
        control = (
            torch.from_numpy(control_np)
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(CANDIDATE_COUNT, 1, 1, 1)
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            outputs = P.unroll(
                model,
                context,
                actions,
                proprio=proprio,
                control=control,
                max_h=3,
            )
        values = np.stack(
            [value.float().cpu().numpy().astype(np.float16) for value in outputs], axis=1
        )
        if values.shape != PREDICTION_STATE_SHAPE or not np.isfinite(values).all():
            raise MaterialisationError(f"{state_id}: predictor output contract drift")
        path = output_root / "stage_b/predictions" / output_source / f"{state_id}.npy"
        atomic_npy(path, values)
        path_binding = artifact_binding(path, root=output_root)
        action_digest = hashlib.sha256(actions_np.tobytes()).hexdigest()
        state_records.append(
            {
                "state_id": state_id,
                "family": str(state_rows[state_id]["family"]),
                "split_role": str(contexts[state_id]["split_role"]),
                "source_id": output_source,
                "predictor_source_id": source_id,
                "ablation_or_null": ablation,
                "input_state_ids": donors,
                "visual_context_bindings": visual_bindings,
                "proprio_shard_binding": proprio_binding,
                "control_shard_binding": control_binding,
                "candidate_action_sha256": action_digest,
                "prediction_state_tensor": path_binding,
            }
        )
        for candidate in range(CANDIDATE_COUNT):
            for horizon_index, horizon in enumerate(HORIZONS):
                logical_records.append(
                    {
                        "kind": output_source,
                        "state_id": state_id,
                        "candidate_index_or_null": candidate,
                        "horizon_or_null": horizon,
                        "path": path_binding["path"],
                        "sha256": path_binding["sha256"],
                        "bytes": path_binding["bytes"],
                        "array_index": [candidate, horizon_index],
                        "logical_shape": list(TENSOR_SHAPE),
                        "logical_dtype": "float16",
                        "input_state_ids": donors,
                    }
                )
        calls += 1

    after = _parameter_digest(model)
    if after != custody["parameter_digest_before"]:
        raise MaterialisationError(f"{source_id}: predictor parameters changed")
    custody["parameter_digest_after"] = after
    custody["parameter_state_unchanged"] = True
    custody["state_batch_calls"] = calls
    custody["model_forward_calls"] = calls * 3
    custody["future_proprioception_read_count"] = 0
    custody["observed_slot_validity_by_horizon"] = {
        str(horizon): P.rollout_validity(horizon) for horizon in HORIZONS
    }

    value = attach_digest(
        {
            "schema": PREDICTION_INDEX_SCHEMA,
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "stage_b_gate_digest": gate_digest,
            "stage_c_gate_digest_or_null": stage_c_gate_digest,
            "source_id": output_source,
            "predictor_source_id": source_id,
            "ablation_or_null": ablation,
            "complete": True,
            "states": STATE_COUNT,
            "candidates_per_state": CANDIDATE_COUNT,
            "horizons": list(HORIZONS),
            "logical_records_count": len(logical_records),
            "logical_records": logical_records,
            "state_records": state_records,
            "prediction_shape_per_state": list(PREDICTION_STATE_SHAPE),
            "dtype": "float16",
            "predictor_custody": custody,
            "future_proprioception": {
                "values_available_to_predictor": False,
                "masked_by_frozen_absence_mechanism": True,
                "read_count": 0,
            },
            "route_outcomes_opened": False,
            "training_executed": False,
            "runtime_s": time.time() - started,
        }
    )
    path = output_root / "stage_b/predictions" / output_source / "index.json"
    atomic_json(path, value)
    return value


def load_logical_prediction(record: Mapping[str, Any], *, output_root: Path) -> np.ndarray:
    """Load one logical `[768,1024]` tensor without model inference."""

    path = resolve_artifact(str(record["path"]), output_root)
    verify_file_binding(path, record, "logical prediction backing tensor")
    state_tensor = np.load(path, mmap_mode="r", allow_pickle=False)
    if state_tensor.shape != PREDICTION_STATE_SHAPE or state_tensor.dtype != np.float16:
        raise MaterialisationError("prediction backing tensor contract drift")
    index = record.get("array_index")
    if (
        not isinstance(index, list)
        or len(index) != 2
        or not 0 <= int(index[0]) < CANDIDATE_COUNT
        or not 0 <= int(index[1]) < len(HORIZONS)
    ):
        raise MaterialisationError("logical prediction array index drift")
    value = np.asarray(state_tensor[int(index[0]), int(index[1])])
    if value.shape != TENSOR_SHAPE or value.dtype != np.float16:
        raise MaterialisationError("logical prediction shape/dtype drift")
    return value


def _run_checked(command: Sequence[str], *, environment: Mapping[str, str]) -> None:
    completed = subprocess.run(
        list(command),
        cwd=ROOT,
        env=dict(environment),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise MaterialisationError(
            f"child failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout[-8000:]}"
        )


def build_child_environment(interpreter: Path) -> dict[str, str]:
    """Return an interpreter-scoped environment with no inherited Python path."""

    # Preserve the venv launcher path: resolving its `python -> python3`
    # symlink would silently select the system prefix and defeat venv custody.
    interpreter = Path(interpreter).absolute()
    policy = CONTRACT.EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
        "only_authorised_environment_change"
    ]
    matching = [
        (label, row)
        for label, row in policy["per_interpreter"].items()
        if row["interpreter"] == str(interpreter)
    ]
    if len(matching) != 1:
        raise MaterialisationError(
            "conditional child interpreter is outside amendment authority"
        )
    _label, interpreter_policy = matching[0]
    if (
        list(SCRUBBED_PYTHON_ENVIRONMENT_KEYS) != policy["remove"]
        or policy["interpreter_flags"] != ["-E", "-s"]
    ):
        raise MaterialisationError("conditional child environment authority drift")
    environment = dict(os.environ)
    for key in SCRUBBED_PYTHON_ENVIRONMENT_KEYS:
        environment.pop(key, None)
    environment["VIRTUAL_ENV"] = str(interpreter_policy["VIRTUAL_ENV"])
    environment.update(
        {
            str(key): str(value)
            for key, value in policy["set_shared"].items()
        }
    )
    path_rows = [
        row
        for row in environment.get("PATH", "").split(os.pathsep)
        if row
        and row != str(CPU_INTERPRETER.parent)
        and row != str(GPU_INTERPRETER.parent)
    ]
    environment["PATH"] = os.pathsep.join(
        [str(interpreter_policy["PATH_prepend"]), *path_rows]
    )
    environment.update(CONTRACT.NUMERICAL_THREAD_ENV)
    return environment


def child_environment_contract(interpreter: Path) -> dict[str, Any]:
    environment = build_child_environment(interpreter)
    interpreter = Path(interpreter).absolute()
    policy = CONTRACT.EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
        "only_authorised_environment_change"
    ]
    label, interpreter_policy = next(
        (label, row)
        for label, row in policy["per_interpreter"].items()
        if row["interpreter"] == str(interpreter)
    )
    return {
        "authority_id": label,
        "interpreter": str(interpreter),
        "isolated_python_environment_flags": copy.deepcopy(
            policy["interpreter_flags"]
        ),
        "scrubbed_keys": copy.deepcopy(policy["remove"]),
        "scrubbed_keys_absent": {
            key: key not in environment for key in SCRUBBED_PYTHON_ENVIRONMENT_KEYS
        },
        "virtual_env": str(interpreter_policy["VIRTUAL_ENV"]),
        "python_no_user_site": str(policy["set_shared"]["PYTHONNOUSERSITE"]),
        "path_first_entry": str(interpreter_policy["PATH_prepend"]),
        "numerical_thread_environment": copy.deepcopy(CONTRACT.NUMERICAL_THREAD_ENV),
    }


def probe_child_interpreter(interpreter: Path, *, require_genesis: bool) -> dict[str, Any]:
    """Outcome-blind import probe for the exact conditional child interpreter."""

    interpreter = Path(interpreter).absolute()
    probe = (
        "import json,sys,typing_extensions;"
        "payload={'executable':sys.executable,'typing_extensions_path':"
        "typing_extensions.__file__,'sentinel_available':"
        "hasattr(typing_extensions,'Sentinel')};"
        + (
            "import pydantic_core,genesis;payload['pydantic_core_path']=pydantic_core.__file__;"
            "payload['pydantic_core_version']=pydantic_core.__version__;"
            "payload['genesis_path']=genesis.__file__;payload['genesis_version']=genesis.__version__;"
            "payload['torch_path']=None;payload['torch_version']=None;"
            if require_genesis else
            "import torch;payload['pydantic_core_path']=None;payload['pydantic_core_version']=None;"
            "payload['genesis_path']=None;payload['genesis_version']=None;"
            "payload['torch_path']=torch.__file__;payload['torch_version']=torch.__version__;"
        )
        + "print(json.dumps(payload,sort_keys=True))"
    )
    environment = build_child_environment(interpreter)
    completed = subprocess.run(
        [str(interpreter), "-E", "-s", "-c", probe],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode:
        raise MaterialisationError(
            f"conditional child import preflight failed ({completed.returncode}): "
            f"{completed.stdout[-4000:]}"
        )
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise MaterialisationError("conditional child import probe output drift") from exc
    venv_root = interpreter.parent.parent.resolve()
    typing_path = Path(str(payload.get("typing_extensions_path", ""))).resolve()
    if (
        Path(str(payload.get("executable", ""))).absolute() != interpreter
        or payload.get("sentinel_available") is not True
        or not typing_path.is_relative_to(venv_root)
    ):
        raise MaterialisationError("conditional child interpreter import custody drift")
    pydantic_path_value = payload.get("pydantic_core_path")
    pydantic_path = (
        None
        if pydantic_path_value is None
        else Path(str(pydantic_path_value)).resolve()
    )
    if require_genesis and (
        pydantic_path is None or not pydantic_path.is_relative_to(venv_root)
    ):
        raise MaterialisationError("pydantic-core did not resolve inside the CPU venv")
    genesis_path = payload.get("genesis_path")
    if require_genesis and (
        not isinstance(genesis_path, str)
        or not Path(genesis_path).resolve().is_relative_to(venv_root)
    ):
        raise MaterialisationError("Genesis import did not resolve inside the CPU venv")
    torch_path_value = payload.get("torch_path")
    torch_path = (
        None if torch_path_value is None else Path(str(torch_path_value)).resolve()
    )
    if not require_genesis and (
        torch_path is None or not torch_path.is_relative_to(venv_root)
    ):
        raise MaterialisationError("torch import did not resolve inside the GPU venv")
    expected = CONTRACT.EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
        "required_preflight"
    ]["cpu_child" if require_genesis else "gpu_child"]

    def exact_module_binding(
        label: str, observed_path: Path, *, version: str | None = None
    ) -> dict[str, Any]:
        expected_binding = expected[label]
        expected_path = Path(str(expected_binding["path"]))
        if observed_path != expected_path.resolve():
            raise MaterialisationError(
                f"conditional child {label} path differs from its amendment binding"
            )
        digest = hashlib.sha256(observed_path.read_bytes()).hexdigest()
        size = observed_path.stat().st_size
        if (
            digest != expected_binding["sha256"]
            or size != expected_binding["bytes"]
            or (
                "version" in expected_binding
                and version != expected_binding["version"]
            )
        ):
            raise MaterialisationError(
                f"conditional child {label} module binding drift"
            )
        result: dict[str, Any] = {
            "path": str(expected_path),
            "resolved_path": str(observed_path),
            "sha256": digest,
            "bytes": size,
        }
        if "version" in expected_binding:
            result["version"] = version
        if "Sentinel_present" in expected_binding:
            result["Sentinel_present"] = True
        return result

    typing_binding = exact_module_binding("typing_extensions", typing_path)
    pydantic_binding = (
        exact_module_binding(
            "pydantic_core",
            pydantic_path,
            version=str(payload.get("pydantic_core_version")),
        )
        if require_genesis and pydantic_path is not None
        else None
    )
    genesis_binding = (
        exact_module_binding(
            "genesis",
            Path(str(genesis_path)).resolve(),
            version=str(payload.get("genesis_version")),
        )
        if require_genesis and genesis_path is not None
        else None
    )
    torch_binding = (
        exact_module_binding(
            "torch",
            torch_path,
            version=str(payload.get("torch_version")),
        )
        if not require_genesis and torch_path is not None
        else None
    )
    return {
        "environment_contract": child_environment_contract(interpreter),
        "typing_extensions": typing_binding,
        "pydantic_core_or_null": pydantic_binding,
        "genesis_or_null": genesis_binding,
        "torch_or_null": torch_binding,
        "sentinel_available": True,
        "pass": True,
    }


def _run_context_workers(
    *,
    output_root: Path,
    gate_path: Path,
    replay_receipt_path: Path,
    workers: int,
) -> dict[str, Any]:
    if workers != FROZEN_WORKERS:
        raise MaterialisationError(f"worker topology must remain frozen at {FROZEN_WORKERS}")
    states = _state_manifest_rows()
    environment = build_child_environment(CPU_INTERPRETER)
    commands = [
        [
            str(CPU_INTERPRETER),
            "-E",
            "-s",
            str(SELF),
            "context-state",
            "--stage-b-authorised",
            "--gate-receipt",
            str(gate_path),
            "--execution-correction-replay-receipt",
            str(replay_receipt_path),
            "--output-root",
            str(output_root),
            "--state-index",
            str(index),
        ]
        for index in range(len(states))
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_checked, command, environment=environment): command for command in commands}
        for future in as_completed(futures):
            future.result()
    gate = validate_stage_b_gate_receipt(gate_path, output_root=output_root)
    return _write_context_index(output_root, str(gate["content_digest"]))


def _run_gpu_source(
    source_id: str,
    *,
    output_root: Path,
    gate_path: Path,
    replay_receipt_path: Path,
    stage_c_gate_path: Path | None = None,
    ablation: str | None = None,
) -> None:
    command = [
        str(GPU_INTERPRETER),
        "-E",
        "-s",
        str(SELF),
        "predict-source",
        "--stage-b-authorised",
        "--gate-receipt",
        str(gate_path),
        "--execution-correction-replay-receipt",
        str(replay_receipt_path),
        "--output-root",
        str(output_root),
        "--source-id",
        source_id,
    ]
    if ablation is not None:
        if stage_c_gate_path is None:
            raise MaterialisationError("Stage-C ablation lacks its gate receipt")
        command.extend(
            [
                "--stage-c-authorised",
                "--stage-c-gate-receipt",
                str(stage_c_gate_path),
                "--ablation",
                ablation,
            ]
        )
    _run_checked(command, environment=build_child_environment(GPU_INTERPRETER))


def _finalize_stage_b(
    output_root: Path,
    gate: Mapping[str, Any],
    gate_path: Path,
    replay_receipt_path: Path,
) -> dict[str, Any]:
    context_path = output_root / "stage_b/proprio_context/index.json"
    sources: dict[str, Any] = {}
    for source_id in ("P1_PROPRIO_ONE_STEP", "PR_PROPRIO_ROLLOUT"):
        path = output_root / "stage_b/predictions" / source_id / "index.json"
        value = load_json(path)
        _validate_self_digest(value, f"{source_id} prediction index")
        if value.get("logical_records_count") != STATE_COUNT * CANDIDATE_COUNT * len(HORIZONS):
            raise MaterialisationError(f"{source_id} prediction index incomplete")
        sources[source_id] = artifact_binding(path, root=output_root)
    receipt = attach_digest(
        {
            "schema": MATERIALISATION_RECEIPT_SCHEMA,
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "stage_b_gate": {
                **artifact_binding(gate_path, root=output_root),
                "content_digest": gate["content_digest"],
                "contract_freeze_commit": gate["contract_freeze_commit"],
            },
            "execution_correction_replay": artifact_binding(
                replay_receipt_path, root=output_root
            ),
            "proprio_context_index": artifact_binding(context_path, root=output_root),
            "prediction_indexes": sources,
            "source_paths": {
                "helper": artifact_binding(SELF),
                "predictor": artifact_binding(ROOT / "scripts/dev_proprio_predictor_v1.py"),
                "predictor_base": artifact_binding(
                    ROOT / "scripts/run_dev_v03_temporal_action_jepa_v1.py"
                ),
                "proprio_manifest_semantics": artifact_binding(
                    ROOT / "scripts/build_dev_v03_proprio_action_manifest_v1.py"
                ),
                "deployment_proprio": artifact_binding(
                    ROOT / "scripts/materialize_deployment_valid_dense_proprioception_v1.py"
                ),
                "state_replay": artifact_binding(
                    ROOT / "scripts/run_go2_oracle_branch_pilot_v1.py"
                ),
                "state_replay_v12": artifact_binding(
                    ROOT / "scripts/run_go2_oracle_branch_pilot_v1_2.py"
                ),
                "experiment_contract": artifact_binding(
                    ROOT / "lewm/safety/plan_aware_monotone_jepa_cost_v1_contract.py"
                ),
            },
            "counts": {
                "states_replayed": STATE_COUNT,
                "prefix_blocks": STATE_COUNT * 40,
                "candidate_actions_executed": 0,
                "new_states": 0,
                "new_candidates": 0,
                "predictor_checkpoints_opened": 2,
                "predictor_state_batch_calls": STATE_COUNT * 2,
                "logical_predictions": STATE_COUNT * CANDIDATE_COUNT * len(HORIZONS) * 2,
                "future_proprioception_reads": 0,
                "route_outcome_reads": 0,
            },
            "workers": FROZEN_WORKERS,
            "numerical_thread_environment": copy.deepcopy(CONTRACT.NUMERICAL_THREAD_ENV),
            "training_executed": False,
            "fresh_panel_collected": False,
            "navigation_executed": False,
            "nothing_running_at_receipt_write": True,
        }
    )
    atomic_json(
        output_root / "receipts/stage_b_proprio_predictor_materialisation.json", receipt
    )
    return receipt


def _load_evaluation_contract_from_gate(
    gate: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    path = resolve_artifact(str(gate["evaluation_contract"]["path"]), output_root)
    return load_json(path)


def run_stage_b(args: argparse.Namespace) -> int:
    if not args.stage_b_authorised:
        raise MaterialisationError("--stage-b-authorised is required")
    output_root = Path(args.output_root).resolve()
    gate_path = Path(args.gate_receipt).resolve()
    gate = validate_stage_b_gate_receipt(gate_path, output_root=output_root)
    replay_receipt_path = Path(args.execution_correction_replay_receipt).resolve()
    validate_execution_correction_replay_receipt(
        replay_receipt_path, output_root=output_root
    )
    validate_runtime_authority(gate)
    validate_frozen_inputs()
    _run_context_workers(
        output_root=output_root,
        gate_path=gate_path,
        replay_receipt_path=replay_receipt_path,
        workers=args.workers,
    )
    for source_id in ("P1_PROPRIO_ONE_STEP", "PR_PROPRIO_ROLLOUT"):
        _run_gpu_source(
            source_id,
            output_root=output_root,
            gate_path=gate_path,
            replay_receipt_path=replay_receipt_path,
        )
    _finalize_stage_b(output_root, gate, gate_path, replay_receipt_path)
    return 0


def run_context_state(args: argparse.Namespace) -> int:
    if not args.stage_b_authorised:
        raise MaterialisationError("--stage-b-authorised is required")
    output_root = Path(args.output_root).resolve()
    gate = validate_stage_b_gate_receipt(
        Path(args.gate_receipt).resolve(), output_root=output_root
    )
    validate_execution_correction_replay_receipt(
        Path(args.execution_correction_replay_receipt).resolve(),
        output_root=output_root,
    )
    validate_runtime_authority(gate)
    validate_frozen_inputs()
    _capture_prefix_state(
        args.state_index,
        output_root=output_root,
        gate_digest=str(gate["content_digest"]),
    )
    return 0


def run_predict_source(args: argparse.Namespace) -> int:
    if not args.stage_b_authorised:
        raise MaterialisationError("--stage-b-authorised is required")
    output_root = Path(args.output_root).resolve()
    gate_path = Path(args.gate_receipt).resolve()
    gate = validate_stage_b_gate_receipt(gate_path, output_root=output_root)
    validate_execution_correction_replay_receipt(
        Path(args.execution_correction_replay_receipt).resolve(),
        output_root=output_root,
    )
    validate_runtime_authority(gate)
    validate_frozen_inputs()
    donor_mapping = None
    stage_c_gate_digest = None
    if args.ablation is not None:
        if not args.stage_c_authorised or args.stage_c_gate_receipt is None:
            raise MaterialisationError("Stage-C inference lacks explicit authorisation")
        stage_c_gate = validate_stage_c_gate_receipt(
            Path(args.stage_c_gate_receipt).resolve(),
            output_root=output_root,
            stage_b_gate_path=gate_path,
        )
        stage_c_gate_digest = str(stage_c_gate["content_digest"])
        evaluation = _load_evaluation_contract_from_gate(gate, output_root)
        contexts = _load_context_index(output_root, str(gate["content_digest"]))
        states = {str(row["state_id"]): row for row in _state_manifest_rows()}
        donor_mapping = _stage_c_mapping(
            ablation=args.ablation,
            evaluation_contract=evaluation,
            states=states,
            contexts=contexts,
        )
    _predict_source(
        args.source_id,
        output_root=output_root,
        gate_digest=str(gate["content_digest"]),
        stage_c_gate_digest=stage_c_gate_digest,
        ablation=args.ablation,
        donor_mapping=donor_mapping,
    )
    return 0


def run_stage_c(args: argparse.Namespace) -> int:
    if not args.stage_b_authorised or not args.stage_c_authorised:
        raise MaterialisationError("Stage C requires both explicit gate flags")
    output_root = Path(args.output_root).resolve()
    gate_path = Path(args.gate_receipt).resolve()
    stage_c_path = Path(args.stage_c_gate_receipt).resolve()
    replay_receipt_path = Path(args.execution_correction_replay_receipt).resolve()
    validate_execution_correction_replay_receipt(
        replay_receipt_path, output_root=output_root
    )
    gate = validate_stage_b_gate_receipt(gate_path, output_root=output_root)
    validate_stage_c_gate_receipt(
        stage_c_path, output_root=output_root, stage_b_gate_path=gate_path
    )
    validate_runtime_authority(gate)
    for ablation in STAGE_C_ABLATIONS:
        _run_gpu_source(
            "PR_PROPRIO_ROLLOUT",
            output_root=output_root,
            gate_path=gate_path,
            replay_receipt_path=replay_receipt_path,
            stage_c_gate_path=stage_c_path,
            ablation=ablation,
        )
    receipt = attach_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.stage_c_materialisation.v1",
            "status": "PASS",
            "experiment_id": EXPERIMENT_ID,
            "stage_b_gate_digest": gate["content_digest"],
            "execution_correction_replay": artifact_binding(
                replay_receipt_path, root=output_root
            ),
            "stage_c_gate": artifact_binding(stage_c_path, root=output_root),
            "prediction_indexes": {
                ablation: artifact_binding(
                    output_root / "stage_b/predictions" / ablation / "index.json",
                    root=output_root,
                )
                for ablation in STAGE_C_ABLATIONS
            },
            "checkpoint": copy.deepcopy(CONTRACT.CHECKPOINT_BINDINGS["PR_PROPRIO_ROLLOUT"]),
            "training_executed": False,
            "outcome_informed_donor_selection": False,
            "nothing_running_at_receipt_write": True,
        }
    )
    atomic_json(output_root / "receipts/stage_c_input_derangement_materialisation.json", receipt)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--stage-b-authorised", action="store_true")
    common.add_argument("--gate-receipt", required=True)
    common.add_argument("--execution-correction-replay-receipt", required=True)
    common.add_argument("--output-root", required=True)

    run = subparsers.add_parser("run-stage-b", parents=[common])
    run.add_argument("--workers", type=int, default=FROZEN_WORKERS)
    run.set_defaults(function=run_stage_b)

    state = subparsers.add_parser("context-state", parents=[common])
    state.add_argument("--state-index", type=int, required=True)
    state.set_defaults(function=run_context_state)

    predict = subparsers.add_parser("predict-source", parents=[common])
    predict.add_argument(
        "--source-id",
        required=True,
        choices=("P1_PROPRIO_ONE_STEP", "PR_PROPRIO_ROLLOUT"),
    )
    predict.add_argument("--stage-c-authorised", action="store_true")
    predict.add_argument("--stage-c-gate-receipt")
    predict.add_argument("--ablation", choices=STAGE_C_ABLATIONS)
    predict.set_defaults(function=run_predict_source)

    stage_c = subparsers.add_parser("run-stage-c", parents=[common])
    stage_c.add_argument("--stage-c-authorised", action="store_true")
    stage_c.add_argument("--stage-c-gate-receipt", required=True)
    stage_c.set_defaults(function=run_stage_c)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
