#!/usr/bin/env python3
"""Execute PLAN_AWARE_MONOTONE_JEPA_COST_V1 under a split-open barrier.

The experiment is development-only and non-claim-bearing.  It trains two
small route-ordering residuals over the already frozen 48-state local-waypoint
panel.  No simulator, renderer, encoder, or predictor is entered in Stage A.
Only fit outcome shards are opened before both final-epoch route-ranker
checkpoints and the evaluation contract are durably published.

Stage B is conditional on the prospectively frozen true-future gate.  If it is
entered, the existing R1/RR payloads are reused and P1/PR are materialised from
the frozen predictors without training.  Stage C is conditional on the frozen
proprioceptive-contribution gate.  Nothing in this file implements navigation.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import jepa_local_waypoint_planning_cost_metrics_v1 as OLD_METRICS  # noqa: E402
from lewm.safety import plan_aware_monotone_jepa_cost_metrics_v1 as METRICS  # noqa: E402
from lewm.safety import plan_aware_monotone_jepa_cost_v1 as MODEL  # noqa: E402
from lewm.safety import plan_aware_monotone_jepa_cost_v1_contract as CONTRACT  # noqa: E402


class QualificationError(RuntimeError):
    """Fail-closed qualification error."""


SOURCE_COMMIT = "1d799eb24d8171cb6d90bc0d0e375d9e1b0cc4f0"
REQUIRED_ANCESTOR = "b29eae1929725a4cc26a35d95662b545daee4553"
PREDECESSOR_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "jepa_local_waypoint_planning_cost_qualification_v1"
)
V1_ROOT = ROOT / ".generated/safe_local_waypoint_purpose_built_v1"
V2_ROOT = ROOT / ".generated/safe_local_waypoint_route_intent_v2"
DENSE_ROOT = ROOT / ".generated/dense_temporal_true_future_safety_observability_v1"
STATE_MANIFEST = V1_ROOT / "state_manifest.json"
SPLIT_PATH = V1_ROOT / "split.json"
BRANCH_LEDGER = V1_ROOT / "branch_labels.jsonl"
ROUTE_LABELS = V2_ROOT / "route_intent_labels.jsonl"
TARGET_LATENT_INDEX = V2_ROOT / "target_latent_index.json"
CONTEXT_INDEX = PREDECESSOR_ROOT / "materialization/context_reconstruction_index.json"
FANOUT_INDEX = PREDECESSOR_ROOT / "materialization/oracle_admissibility_fanout_index.json"
GOAL_INDEX = PREDECESSOR_ROOT / "goal_views/index.json"
LATENT_INDEX = PREDECESSOR_ROOT / "latents/tensor_index.json"
PREDECESSOR_METRICS = PREDECESSOR_ROOT / "aggregates/metrics.json"
PREDECESSOR_RESULT = PREDECESSOR_ROOT / "result.json"
PREDECESSOR_PERSISTENCE = PREDECESSOR_ROOT / "receipts/persistence.json"

FIT, CALIBRATION, HELDOUT = "fit", "calibration", "heldout"
SPLIT_ROLES = (FIT, CALIBRATION, HELDOUT)
STATE_COUNT = 48
CANDIDATE_COUNT = 12
HORIZONS = (1, 2, 3)
TENSOR_SHAPE = (768, 1024)
TENSOR_DTYPE = np.dtype(np.float16)
MACRO_TO_PRIMITIVE = (3, 2, 1, 7, 8, 5, 6, 5, 6, 2, 4, 0)
ROUTE_LINE_STATE = re.compile(rb'"state_id"\s*:\s*"([^"]+)"')
ROUTE_LINE_CANDIDATE = re.compile(rb'"candidate_index"\s*:\s*([0-9]+)')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(path, canonical_bytes(value))


def atomic_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            for row in rows:
                zipped.write(canonical_bytes(row))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise QualificationError(f"JSON object required: {path}")
    return value


def numeric_state_key(state_id: str) -> tuple[str, int]:
    prefix, separator, suffix = state_id.rpartition("-")
    if not separator or not suffix.isdigit():
        raise QualificationError(f"invalid state identity: {state_id!r}")
    return prefix, int(suffix)


def binding(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    reference = path
    if relative_to is not None:
        try:
            reference = path.relative_to(relative_to)
        except ValueError:
            reference = path
    return {
        "path": str(reference),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def verify_binding(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise QualificationError(f"{label} is missing: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise QualificationError(
            f"{label} SHA drift: {observed} != {expected_sha256}"
        )


def verify_exact_file_record(
    path: Path, record: Mapping[str, Any], *, label: str
) -> None:
    verify_binding(path, str(record["sha256"]), label=label)
    if path.stat().st_size != int(record["bytes"]):
        raise QualificationError(f"{label} byte-size drift")


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_source_authority(*, runtime_commit: str | None = None) -> str:
    head = git_output("rev-parse", "HEAD")
    expected = SOURCE_COMMIT if runtime_commit is None else runtime_commit
    if head != expected:
        raise QualificationError(f"source HEAD {head} != expected {expected}")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", REQUIRED_ANCESTOR, head],
        cwd=ROOT,
        check=False,
    ).returncode:
        raise QualificationError("protected-contact requirements result is not an ancestor")
    if git_output("status", "--porcelain"):
        raise QualificationError("scientific execution requires a clean worktree")
    return head


def split_ids() -> dict[str, list[str]]:
    split = load_json(SPLIT_PATH)
    output = {role: [str(value) for value in split[role]] for role in SPLIT_ROLES}
    if {role: len(output[role]) for role in SPLIT_ROLES} != {
        FIT: 32,
        CALIBRATION: 8,
        HELDOUT: 8,
    }:
        raise QualificationError("frozen split cardinality drift")
    flat = [state for role in SPLIT_ROLES for state in output[role]]
    if len(set(flat)) != STATE_COUNT:
        raise QualificationError("frozen split identities are not disjoint and complete")
    return output


def route_line_index() -> dict[tuple[str, int], dict[str, Any]]:
    """Byte-index the mixed route ledger without parsing scientific fields."""

    output: dict[tuple[str, int], dict[str, Any]] = {}
    with ROUTE_LABELS.open("rb") as handle:
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            state_match = ROUTE_LINE_STATE.search(line)
            candidate_match = ROUTE_LINE_CANDIDATE.search(line)
            if state_match is None or candidate_match is None:
                raise QualificationError("route ledger identity cannot be byte-indexed")
            state_id = state_match.group(1).decode("utf-8")
            candidate = int(candidate_match.group(1))
            key = (state_id, candidate)
            if key in output:
                raise QualificationError(f"duplicate route identity {key}")
            output[key] = {
                "offset": offset,
                "bytes": len(line),
                "sha256": hashlib.sha256(line).hexdigest(),
            }
    if len(output) != STATE_COUNT * CANDIDATE_COUNT:
        raise QualificationError("route byte index cardinality drift")
    return output


def read_route_state(
    state_id: str,
    line_index: Mapping[tuple[str, int], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Parse exactly one already-authorised state's twelve route rows."""

    rows: list[dict[str, Any]] = []
    with ROUTE_LABELS.open("rb") as handle:
        for candidate in range(CANDIDATE_COUNT):
            record = line_index[(state_id, candidate)]
            handle.seek(int(record["offset"]))
            line = handle.read(int(record["bytes"]))
            if hashlib.sha256(line).hexdigest() != record["sha256"]:
                raise QualificationError(f"route row bytes drift: {state_id}:{candidate}")
            row = json.loads(line)
            if (
                str(row.get("state_id")) != state_id
                or int(row.get("candidate_index", -1)) != candidate
            ):
                raise QualificationError("route row identity drift after authorised open")
            rows.append(row)
    return rows


def role_one_hot(role: str) -> list[float]:
    roles = ("translational", "alignment", "hold_abstain")
    if role not in roles:
        raise QualificationError(f"unknown route role {role!r}")
    return [float(role == value) for value in roles]


def _index_by_state(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != STATE_COUNT:
        raise QualificationError(f"state index cardinality drift: {path}")
    output = {str(row["state_id"]): dict(row) for row in rows}
    if len(output) != STATE_COUNT:
        raise QualificationError(f"state index duplicate identity: {path}")
    return output


def tensor_index() -> dict[tuple[str, str, int | None, int | None], dict[str, Any]]:
    payload = load_json(LATENT_INDEX)
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != 5424:
        raise QualificationError("predecessor latent index cardinality drift")
    output: dict[tuple[str, str, int | None, int | None], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["kind"]),
            str(row["state_id"]),
            row["candidate_index_or_null"],
            row["horizon_or_null"],
        )
        if key in output:
            raise QualificationError(f"duplicate latent tensor identity {key}")
        output[key] = dict(row)
    return output


def _validate_true_future_index_reconciliation(
    target_index_value: Mapping[str, Any],
    tensor_index_value: Mapping[str, Any],
    *,
    expected_index_sha256: str,
    expected_count: int = 1_728,
) -> None:
    entries = target_index_value.get("entries")
    records = tensor_index_value.get("records")
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise QualificationError("frozen true-future target-index cardinality drift")
    if not isinstance(records, list):
        raise QualificationError("predecessor tensor-index records are absent")
    target_by_identity = {
        (
            str(row["state_id"]),
            int(row["candidate_index"]),
            int(row["horizon"]),
        ): row
        for row in entries
    }
    true_records = [row for row in records if row.get("kind") == "TRUE_FUTURE"]
    if len(target_by_identity) != expected_count or len(true_records) != expected_count:
        raise QualificationError("true-future identity cardinality drift")
    seen: set[tuple[str, int, int]] = set()
    for record in true_records:
        identity = (
            str(record["state_id"]),
            int(record["candidate_index_or_null"]),
            int(record["horizon_or_null"]),
        )
        if identity in seen or identity not in target_by_identity:
            raise QualificationError(f"true-future identity drift: {identity}")
        seen.add(identity)
        target = target_by_identity[identity]
        external = record.get("external_existing_artifact")
        if not isinstance(external, Mapping) or (
            record.get("shape") != target.get("shape")
            or record.get("dtype") != target.get("dtype")
            or record.get("sha256") != target.get("sha256")
            or external.get("path") != target.get("latent_path")
            or external.get("sha256") != target.get("sha256")
            or external.get("index_sha256") != expected_index_sha256
            or external.get("array_equality") is not True
        ):
            raise QualificationError(f"true-future metadata reconciliation drift: {identity}")
    if seen != set(target_by_identity):
        raise QualificationError("true-future reconciliation is incomplete")


def resolve_predecessor_artifact(reference: str) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else PREDECESSOR_ROOT / path


def load_tensor(row: Mapping[str, Any]) -> np.ndarray:
    path = resolve_predecessor_artifact(str(row["path"]))
    if not path.is_file() or path.stat().st_size != int(row["bytes"]):
        raise QualificationError(f"tensor path/byte drift: {path}")
    if sha256_file(path) != row["sha256"]:
        raise QualificationError(f"tensor SHA drift: {path}")
    value = np.load(path, allow_pickle=False)
    if value.shape != TENSOR_SHAPE or value.dtype != TENSOR_DTYPE:
        raise QualificationError(f"tensor shape/dtype drift: {path}")
    if not np.isfinite(value).all():
        raise QualificationError(f"tensor contains non-finite values: {path}")
    return value


def _fanout_path(state_id: str) -> Path:
    return (
        PREDECESSOR_ROOT
        / "materialization/states"
        / state_id
        / "oracle_contact_fanout.npz"
    )


def load_oracle_population(state_id: str) -> list[dict[str, Any]]:
    path = _fanout_path(state_id)
    if not path.is_file():
        raise QualificationError(f"oracle fanout shard missing: {state_id}")
    with np.load(path, allow_pickle=False) as archive:
        current = np.asarray(archive["current_contact_bitset"], dtype=np.bool_)
        successor = np.asarray(archive["successor_contact_bitset"], dtype=np.bool_)
    if current.shape != (9, 250) or successor.shape != (9, 9, 250):
        raise QualificationError(f"oracle fanout shape drift: {state_id}")
    output = []
    for candidate, primitive in enumerate(MACRO_TO_PRIMITIVE):
        immediate = bool(np.any(current[primitive]))
        safe_count = int(np.sum(~np.any(successor[primitive], axis=1)))
        output.append(
            {
                "candidate_index": candidate,
                "immediate_contact_h1": immediate,
                "successor_safe_action_count": safe_count,
                "successor_viable": safe_count > 0,
                "oracle_viability_admissible": (not immediate and safe_count > 0),
            }
        )
    return output


def dense_state(state_id: str) -> dict[str, Any]:
    path = DENSE_ROOT / "dense_replay" / f"{state_id}.json"
    value = load_json(path)
    if (
        value.get("schema") != "dense_route_intent_true_future_state_v1"
        or value.get("status") != "PASS"
        or str(value.get("state_id")) != state_id
        or len(value.get("branches", [])) != CANDIDATE_COUNT
    ):
        raise QualificationError(f"dense replay state drift: {state_id}")
    return value


def state_manifest_map() -> dict[str, dict[str, Any]]:
    value = load_json(STATE_MANIFEST)
    rows = value.get("state_candidates")
    if not isinstance(rows, list) or len(rows) != STATE_COUNT:
        raise QualificationError("state manifest cardinality drift")
    output = {str(row["state_id"]): dict(row) for row in rows}
    if len(output) != STATE_COUNT:
        raise QualificationError("state manifest duplicate identities")
    return output


def build_state_rows(
    state_id: str,
    *,
    expected_role: str,
    line_index: Mapping[tuple[str, int], Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    goals: Mapping[str, Mapping[str, Any]],
    route_roles: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Open and construct exactly one authorised state's scalar rows."""

    route_rows = read_route_state(state_id, line_index)
    try:
        role = str(route_roles[state_id])
    except KeyError as exc:
        raise QualificationError(f"frozen route-role authority lacks {state_id}") from exc
    if role not in ("translational", "alignment", "hold_abstain"):
        raise QualificationError(f"frozen route-role authority is invalid: {state_id}")
    oracle = load_oracle_population(state_id)
    dense = dense_state(state_id)
    dense_by = {int(row["candidate_index"]): row for row in dense["branches"]}
    if set(dense_by) != set(range(CANDIDATE_COUNT)):
        raise QualificationError(f"dense candidate identities drift: {state_id}")
    context = contexts[state_id]
    if context.get("role") != expected_role:
        raise QualificationError(f"context split role drift: {state_id}")
    goal = goals[state_id]["goal_body_dx_dy_sin_dyaw_cos_dyaw"]
    dx, dy, sin_heading, cos_heading = [float(value) for value in goal]
    heading = math.atan2(sin_heading, cos_heading)
    waypoint = [
        dx,
        dy,
        math.hypot(dx, dy),
        heading,
        sin_heading,
        cos_heading,
    ]
    role_vector = role_one_hot(role)
    raw_control = np.asarray(context["control_history_raw_3x5x2"], np.float64)
    if raw_control.shape != (3, 5, 2) or not np.isfinite(raw_control).all():
        raise QualificationError(f"control history shape drift: {state_id}")
    previous_active = raw_control[-1, -1]
    previous_full = [float(previous_active[0]), 0.0, float(previous_active[1])]
    nominal: list[dict[str, Any]] = []
    for candidate in range(CANDIDATE_COUNT):
        applied = np.asarray(
            context["applied_action_blocks_raw_3x5x3_by_candidate"][candidate],
            np.float64,
        )
        outcome = OLD_METRICS.kinematic_nominal_outcome(
            applied,
            waypoint[:2],
            route_heading_rad=heading,
        )
        nominal.append({"candidate_index": candidate, **outcome})
    rank_cost = OLD_METRICS.kinematic_rank_costs(nominal)
    output: list[dict[str, Any]] = []
    for candidate in range(CANDIDATE_COUNT):
        route = route_rows[candidate]["horizons"]["3"]
        oracle_row = oracle[candidate]
        requested = np.asarray(
            context["requested_action_blocks_raw_3x5x3_by_candidate"][candidate],
            np.float64,
        )
        applied = np.asarray(
            context["applied_action_blocks_raw_3x5x3_by_candidate"][candidate],
            np.float64,
        )
        active_plan = np.asarray(
            context["action_blocks_raw_3x10_by_candidate"][candidate], np.float64
        )
        if requested.shape != (3, 5, 3) or applied.shape != (3, 5, 3):
            raise QualificationError(f"full action tape shape drift: {state_id}:{candidate}")
        if active_plan.shape != (3, 10):
            raise QualificationError(f"predictor action plan shape drift: {state_id}:{candidate}")
        kin = nominal[candidate]
        anchor = -float(rank_cost[candidate])
        kin_features = [
            float(kin["x_m"]),
            float(kin["y_m"]),
            float(kin["yaw_rad"]),
            float(kin["nominal_p_d"]),
            float(kin["nominal_p_theta"]),
            anchor,
        ]
        base_features = np.asarray(
            [
                *waypoint,
                *role_vector,
                *requested.reshape(-1).tolist(),
                *applied.reshape(-1).tolist(),
                *previous_full,
                *raw_control.reshape(-1).tolist(),
                *kin_features,
            ],
            np.float32,
        )
        query_features = np.asarray(
            [
                *waypoint,
                *role_vector,
                *active_plan.reshape(-1).tolist(),
                float(previous_active[0]),
                float(previous_active[1]),
                *raw_control.reshape(-1).tolist(),
            ],
            np.float32,
        )
        if base_features.shape != (138,) or query_features.shape != (71,):
            raise QualificationError("feature vector dimension drift")
        ticks = dense_by[candidate]["ticks"]
        if len(ticks) != 15:
            raise QualificationError(f"dense tick count drift: {state_id}:{candidate}")
        contacts = [bool(ticks[index]["cumulative_contact"]) for index in (4, 9, 14)]
        output.append(
            {
                "state_id": state_id,
                "family": str(context["family"]),
                "split": expected_role,
                "route_intent_role": role,
                "candidate_index": candidate,
                "waypoint_features": waypoint,
                "requested_action_blocks": requested.astype(float).tolist(),
                "applied_action_blocks": applied.astype(float).tolist(),
                "predictor_candidate_action_plan_3x10": active_plan.astype(
                    float
                ).tolist(),
                "previous_applied_command": previous_full,
                "control_history": raw_control.astype(float).tolist(),
                "kinematic_features": kin_features,
                "kinematic_anchor": anchor,
                "base_features": base_features,
                "query_features": query_features,
                "p_d": float(route["p_d"]),
                "p_theta": float(route["p_theta_rad"]),
                "completed": bool(route["completed"]),
                "stuck": bool(ticks[14]["cumulative_stuck"]),
                "descriptive_contact_h1": contacts[0],
                "descriptive_contact_h2": contacts[1],
                "descriptive_contact_h3": contacts[2],
                **oracle_row,
            }
        )
    return output


def authorised_dataset(
    role: str,
    *,
    ids: Mapping[str, Sequence[str]],
    line_index: Mapping[tuple[str, int], Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, Any]],
    goals: Mapping[str, Mapping[str, Any]],
    route_roles: Mapping[str, str],
) -> dict[str, list[dict[str, Any]]]:
    if role not in SPLIT_ROLES:
        raise QualificationError(f"unknown split role {role!r}")
    return {
        state_id: build_state_rows(
            state_id,
            expected_role=role,
            line_index=line_index,
            contexts=contexts,
            goals=goals,
            route_roles=route_roles,
        )
        for state_id in sorted(ids[role], key=numeric_state_key)
    }


def route_utility(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return OLD_METRICS.margin_borda_utility(rows)


def admissible_positions(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if bool(row["oracle_viability_admissible"])
    ]


def state_training_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positions = admissible_positions(rows)
    subset = [rows[index] for index in positions]
    try:
        optimization_status = CONTRACT.fit_state_optimization_status(len(positions))
    except CONTRACT.ContractError as exc:
        raise QualificationError(str(exc)) from exc
    utility = (
        route_utility(subset).astype(np.float32)
        if subset
        else np.empty((0,), dtype=np.float32)
    )
    pairwise_targets = np.zeros((len(subset), len(subset)), dtype=np.float32)
    for left in range(len(subset)):
        for right in range(left + 1, len(subset)):
            preference = CONTRACT.margin_borda_pairwise_target(
                float(utility[left]), float(utility[right])
            )
            pairwise_targets[left, right] = float(preference)
            pairwise_targets[right, left] = float(-preference)
    if not np.array_equal(pairwise_targets, -pairwise_targets.T):
        raise QualificationError(
            "conditioned margin-Borda pairwise target matrix is not antisymmetric"
        )
    payload: dict[str, Any] = {
        "positions": positions,
        "utility": utility,
        "pairwise_targets": pairwise_targets,
        "optimization_status": optimization_status,
        "contributes_optimizer_step": optimization_status == "CONTRIBUTING",
    }
    if optimization_status == "CONTRIBUTING":
        payload.update(
            {
                "base_features": np.stack(
                    [rows[index]["base_features"] for index in positions]
                ),
                "query_features": np.stack(
                    [rows[index]["query_features"] for index in positions]
                ),
                "anchors": np.asarray(
                    [rows[index]["kinematic_anchor"] for index in positions],
                    np.float32,
                ),
            }
        )
    return payload


def _fit_optimization_summary(
    fit: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    by_status: dict[str, list[str]] = {
        "CONTRIBUTING": [],
        "SKIPPED_ZERO_ADMISSIBLE": [],
        "SKIPPED_SINGLETON_ADMISSIBLE": [],
    }
    total_ids: list[str] = []
    for state_id in sorted(fit, key=numeric_state_key):
        count = len(admissible_positions(fit[state_id]))
        try:
            status = CONTRACT.fit_state_optimization_status(count)
        except CONTRACT.ContractError as exc:
            raise QualificationError(str(exc)) from exc
        if status not in by_status:
            raise QualificationError(f"unknown fit optimization status {status!r}")
        by_status[status].append(state_id)
        total_ids.append(state_id)
    summary = {
        "fit_states_total": len(fit),
        "fit_state_ids_total": total_ids,
        "fit_states_contributing": len(by_status["CONTRIBUTING"]),
        "fit_state_ids_contributing": by_status["CONTRIBUTING"],
        "fit_states_skipped_zero_admissible": len(
            by_status["SKIPPED_ZERO_ADMISSIBLE"]
        ),
        "fit_state_ids_skipped_zero_admissible": by_status[
            "SKIPPED_ZERO_ADMISSIBLE"
        ],
        "fit_states_skipped_singleton_admissible": len(
            by_status["SKIPPED_SINGLETON_ADMISSIBLE"]
        ),
        "fit_state_ids_skipped_singleton_admissible": by_status[
            "SKIPPED_SINGLETON_ADMISSIBLE"
        ],
        "epoch_average_denominator": len(by_status["CONTRIBUTING"]),
    }
    _validate_fit_optimization_summary(summary, expected_total=len(fit))
    return summary


def _validate_fit_optimization_summary(
    summary: Mapping[str, Any], *, expected_total: int | None = None
) -> None:
    required = set(CONTRACT.FIT_OPTIMIZATION_RECEIPT_FIELDS)
    if set(summary) != required:
        raise QualificationError("fit optimization summary schema drift")
    total = int(summary["fit_states_total"])
    if expected_total is not None and total != expected_total:
        raise QualificationError("fit optimization total-state drift")
    groups = (
        ("fit_states_contributing", "fit_state_ids_contributing"),
        (
            "fit_states_skipped_zero_admissible",
            "fit_state_ids_skipped_zero_admissible",
        ),
        (
            "fit_states_skipped_singleton_admissible",
            "fit_state_ids_skipped_singleton_admissible",
        ),
    )
    all_ids: list[str] = []
    for count_key, ids_key in groups:
        ids = list(summary[ids_key])
        if int(summary[count_key]) != len(ids) or ids != sorted(
            ids, key=numeric_state_key
        ):
            raise QualificationError(f"fit optimization {ids_key} drift")
        all_ids.extend(ids)
    total_ids = list(summary["fit_state_ids_total"])
    if (
        len(total_ids) != total
        or total_ids != sorted(total_ids, key=numeric_state_key)
        or len(set(all_ids)) != len(all_ids)
        or set(all_ids) != set(total_ids)
        or sum(int(summary[count_key]) for count_key, _ in groups) != total
        or int(summary["epoch_average_denominator"])
        != int(summary["fit_states_contributing"])
    ):
        raise QualificationError("fit optimization partition/denominator drift")


def _validate_training_history(
    history: Any, fit_optimization: Mapping[str, Any]
) -> None:
    if not isinstance(history, list) or len(history) != CONTRACT.TRAINING["epochs"]:
        raise QualificationError("training history epoch cardinality drift")
    required = {
        "epoch",
        "optimizer_steps",
        "fit_states_total",
        "fit_states_contributing",
        "fit_states_skipped_zero_admissible",
        "fit_states_skipped_singleton_admissible",
        "epoch_average_denominator",
        "loss",
        "pair",
        "list",
        "residual",
    }
    for expected_epoch, row in enumerate(history, 1):
        if (
            not isinstance(row, Mapping)
            or set(row) != required
            or row["epoch"] != expected_epoch
            or row["optimizer_steps"]
            != fit_optimization["fit_states_contributing"]
            or row["fit_states_total"] != fit_optimization["fit_states_total"]
            or row["fit_states_contributing"]
            != fit_optimization["fit_states_contributing"]
            or row["fit_states_skipped_zero_admissible"]
            != fit_optimization["fit_states_skipped_zero_admissible"]
            or row["fit_states_skipped_singleton_admissible"]
            != fit_optimization["fit_states_skipped_singleton_admissible"]
            or row["epoch_average_denominator"]
            != fit_optimization["epoch_average_denominator"]
            or any(
                not math.isfinite(float(row[key]))
                for key in ("loss", "pair", "list", "residual")
            )
        ):
            raise QualificationError(
                f"training history state-accounting drift at epoch {expected_epoch}"
            )


def _latent_kind(source: str) -> str:
    try:
        return {
            "TRUE": "TRUE_FUTURE",
            "R1": "ONE_STEP_PREDICTED",
            "RR": "TWO_STEP_PREDICTED",
            "P1": "P1_PROPRIO_ONE_STEP",
            "PR": "PR_PROPRIO_ROLLOUT",
            "PR_VISUAL_CONTEXT_DERANGED": "PR_VISUAL_CONTEXT_DERANGED",
            "PR_PROPRIO_HISTORY_DERANGED": "PR_PROPRIO_HISTORY_DERANGED",
            "PR_CONTROL_HISTORY_DERANGED": "PR_CONTROL_HISTORY_DERANGED",
        }[source]
    except KeyError as exc:
        raise QualificationError(f"unknown latent source {source!r}") from exc


def _latent_binding_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    required = ("path", "sha256", "bytes")
    if any(key not in record for key in required):
        raise QualificationError("latent artifact binding is incomplete")
    output = {key: copy.deepcopy(record[key]) for key in required}
    if "array_index" in record:
        output["array_index"] = copy.deepcopy(record["array_index"])
    return output


def _candidate_latent_bindings(
    tensor_rows: Mapping[tuple[str, str, int | None, int | None], Mapping[str, Any]],
    *,
    state_id: str,
    candidate: int,
    source: str,
    deranged_candidate: int | None = None,
) -> dict[str, Any]:
    kind = _latent_kind(source)
    current = _latent_binding_fields(
        tensor_rows[("CURRENT", state_id, None, None)]
    )
    futures = {
        f"H{horizon}": _latent_binding_fields(
            tensor_rows[(kind, state_id, candidate, horizon)]
        )
        for horizon in HORIZONS
    }
    output: dict[str, Any] = {
        "current": current,
        "candidate_future": futures,
    }
    if deranged_candidate is not None:
        output["deranged_candidate_future"] = {
            f"H{horizon}": _latent_binding_fields(
                tensor_rows[(kind, state_id, deranged_candidate, horizon)]
            )
            for horizon in HORIZONS
        }
    return output


def _tensor_rows_for_state(
    tensor_rows: Mapping[tuple[str, str, int | None, int | None], Mapping[str, Any]],
    state_id: str,
    positions: Sequence[int],
    *,
    source: str,
    logical_output_root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    current = load_tensor(tensor_rows[("CURRENT", state_id, None, None)])
    kind = _latent_kind(source)

    first = tensor_rows[(kind, state_id, int(positions[0]), HORIZONS[0])]
    if "array_index" in first:
        if logical_output_root is None:
            raise QualificationError("logical Stage-B tensor lacks its output root")
        from scripts import materialize_plan_aware_proprio_predictor_substitution_v1 as stage_b_impl

        path = stage_b_impl.resolve_artifact(str(first["path"]), logical_output_root)
        try:
            stage_b_impl.verify_file_binding(path, first, f"{state_id}:{kind}")
        except stage_b_impl.MaterialisationError as exc:
            raise QualificationError(str(exc)) from exc
        state_tensor = np.load(path, mmap_mode="r", allow_pickle=False)
        if (
            state_tensor.shape != stage_b_impl.PREDICTION_STATE_SHAPE
            or state_tensor.dtype != np.float16
        ):
            raise QualificationError(f"Stage-B state tensor drift: {state_id}:{kind}")
        for candidate in range(CANDIDATE_COUNT):
            for horizon_index, horizon in enumerate(HORIZONS):
                record = tensor_rows[(kind, state_id, candidate, horizon)]
                if (
                    record.get("path") != first.get("path")
                    or record.get("sha256") != first.get("sha256")
                    or record.get("bytes") != first.get("bytes")
                    or record.get("array_index") != [candidate, horizon_index]
                ):
                    raise QualificationError(
                        f"logical Stage-B index drift: {state_id}:{candidate}:h{horizon}"
                    )
        futures = np.asarray(state_tensor[np.asarray(positions)]).copy()
    else:
        futures = np.stack(
            [
                np.stack(
                    [
                        load_tensor(tensor_rows[(kind, state_id, candidate, horizon)])
                        for horizon in HORIZONS
                    ],
                    axis=0,
                )
                for candidate in positions
            ],
            axis=0,
        )
    if current.shape != TENSOR_SHAPE or futures.shape != (
        len(positions),
        3,
        *TENSOR_SHAPE,
    ):
        raise QualificationError(f"latent trajectory tensor shape drift: {state_id}:{source}")
    return current, futures


def deterministic_epoch_order(state_ids: Sequence[str], epoch: int) -> list[str]:
    prefix = f"{CONTRACT.ROUTE_COST_SEED}:epoch:{epoch}:".encode("utf-8")
    return sorted(
        state_ids,
        key=lambda state: (hashlib.sha256(prefix + state.encode()).digest(), state),
    )


def _parameter_digest(model: Any) -> str:
    import torch

    digest = hashlib.sha256()
    for name, parameter in sorted(model.state_dict().items()):
        value = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch_save(path: Path, value: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    torch.save(dict(value), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run_training_smoke(
    fit: Mapping[str, Sequence[Mapping[str, Any]]],
    tensor_rows: Mapping[tuple[str, str, int | None, int | None], Mapping[str, Any]],
    *,
    ids: Mapping[str, Sequence[str]],
    output_root: Path,
) -> dict[str, Any]:
    """Run the registered fit-only implementation smoke before full training."""

    import torch

    if set(fit) != set(ids[FIT]) or set(fit) & (
        set(ids[CALIBRATION]) | set(ids[HELDOUT])
    ):
        raise QualificationError("training smoke split exclusion failed")
    optimization = _fit_optimization_summary(fit)
    contributing_ids = optimization["fit_state_ids_contributing"]
    if not contributing_ids:
        raise QualificationError("training smoke has no optimizer-contributing fit state")
    state_id = contributing_ids[0]
    rows = list(fit[state_id])
    waypoint_identities = {
        tuple(float(value) for value in row["waypoint_features"]) for row in rows
    }
    if len(waypoint_identities) != 1:
        raise QualificationError("state candidates do not share one waypoint")
    payload = state_training_payload(rows)
    current_np, future_np = _tensor_rows_for_state(
        tensor_rows, state_id, payload["positions"], source="TRUE"
    )
    # The readout consumes the exact row-major flattening of the 24x32 token
    # grid.  It intentionally adds no invented spatial reordering.
    sentinel = np.arange(768).reshape(24, 32)
    if not np.array_equal(sentinel.reshape(-1), np.arange(768)):
        raise QualificationError("24x32 spatial token order drift")
    device = torch.device("cuda:0")
    no_latent, latent = MODEL.build_matched_rankers(CONTRACT.ROUTE_COST_SEED)
    MODEL.assert_matched_initialisation(no_latent, latent)
    no_latent.to(device)
    latent.to(device)
    # Smoke copies receive deterministic non-zero readout weights solely to
    # establish input sensitivity before scientific training.  They are never
    # persisted as experiment checkpoints.
    with torch.no_grad():
        no_latent.base_residual.layers[-1].weight.fill_(0.01)
        latent.base_residual.layers[-1].weight.fill_(0.01)
        latent.latent_residual_mlp[-1].weight.fill_(0.01)
    base = torch.from_numpy(payload["base_features"]).to(device)
    query = torch.from_numpy(payload["query_features"]).to(device)
    anchors = torch.from_numpy(payload["anchors"]).to(device)
    utility = torch.from_numpy(payload["utility"]).to(device)
    pairwise_targets = torch.from_numpy(payload["pairwise_targets"]).to(device)
    current = torch.from_numpy(current_np.astype(np.float32)).to(device)
    future = torch.from_numpy(future_np.astype(np.float32)).to(device)
    first_no = no_latent(base, anchors)
    first_components = latent.score_components(base, query, anchors, current, future)
    repeated = latent.score_components(base, query, anchors, current, future)
    deterministic = torch.equal(first_components.score, repeated.score)
    changed_base = base.clone()
    changed_query = query.clone()
    changed_base[0, 9] += 0.25
    changed_query[0, 9] += 0.25
    action_score = latent.score_components(
        changed_base, changed_query, anchors, current, future
    ).score
    action_sensitive = not torch.equal(first_components.score, action_score)
    changed_future = future.clone()
    changed_future[0, 0, 0, 0] += 1.0
    changed_latent = latent.score_components(
        base, query, anchors, current, changed_future
    ).score
    latent_sensitive = not torch.equal(first_components.score, changed_latent)
    no_latent_invariant = torch.equal(first_no, no_latent(base, anchors))
    zero_branch_exact = torch.equal(
        first_components.kinematic_plus_base_score,
        anchors + latent.base_residual(base),
    )
    losses = MODEL.route_only_loss(
        scores=first_components.score.unsqueeze(0),
        route_utilities=utility.unsqueeze(0),
        kinematic_anchor=anchors.unsqueeze(0),
        pairwise_targets=pairwise_targets.unsqueeze(0),
    )
    losses.total.backward()
    finite_loss_gradient = bool(torch.isfinite(losses.total)) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in latent.parameters()
    )
    smoke_path = output_root / "staging/smoke_checkpoint.pt"
    _atomic_torch_save(
        smoke_path,
        {"state_dict": latent.state_dict(), "schema": "plan_aware_smoke_checkpoint_v1"},
    )
    reloaded = MODEL.build_matched_rankers(CONTRACT.ROUTE_COST_SEED)[1]
    checkpoint = torch.load(smoke_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(checkpoint["state_dict"], strict=True)
    reload_exact = _parameter_digest(reloaded) == _parameter_digest(latent)
    smoke_path.unlink()
    checks = {
        "fit_only_split_exclusion": True,
        "no_calibration_or_heldout_rows_opened": True,
        "candidate_waypoint_invariant": True,
        "spatial_24x32_row_major_order_preserved": True,
        "action_change_alters_latent_condition_score": action_sensitive,
        "latent_change_alters_latent_condition_score": latent_sensitive,
        "latent_change_cannot_enter_no_latent_condition": no_latent_invariant,
        "latent_branch_zero_exact_kinematic_plus_base": zero_branch_exact,
        "finite_loss_and_gradients": finite_loss_gradient,
        "checkpoint_save_reload_exact": reload_exact,
        "deterministic_repeated_inference": deterministic,
        "model_forward_inputs_exclude_outcome_fields": True,
    }
    if not all(checks.values()):
        raise QualificationError(f"training smoke failed: {checks}")
    receipt = attach_digest(
        {
            "schema": "plan_aware_monotone_jepa_training_smoke_v1",
            "state_id": state_id,
            "candidate_count": len(payload["positions"]),
            "fit_optimization": optimization,
            "checks": checks,
            "calibration_rows_opened": 0,
            "heldout_rows_opened": 0,
            "scientific_hyperparameters_changed": False,
            "pass": True,
        }
    )
    atomic_json(output_root / "receipts/training_smoke.json", receipt)
    return receipt


def train_rankers(
    fit: Mapping[str, Sequence[Mapping[str, Any]]],
    tensor_rows: Mapping[tuple[str, str, int | None, int | None], Mapping[str, Any]],
    *,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Train exactly the two frozen final-epoch route residuals."""

    import torch

    if not torch.cuda.is_available():
        raise QualificationError("route-ranker training requires frozen GPU availability")
    device = torch.device("cuda:0")
    torch.manual_seed(CONTRACT.ROUTE_COST_SEED)
    np.random.seed(CONTRACT.ROUTE_COST_SEED % (2**32))
    os.environ.update(CONTRACT.NUMERICAL_THREAD_ENV)
    torch.set_num_threads(1)
    no_latent, latent = MODEL.build_matched_rankers(CONTRACT.ROUTE_COST_SEED)
    MODEL.assert_matched_initialisation(no_latent, latent)
    counts = {
        "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": MODEL.parameter_count(no_latent),
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL": MODEL.parameter_count(latent),
    }
    if counts["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"] >= 250_000 or counts[
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"
    ] >= 500_000:
        raise QualificationError("route-ranker parameter budget exceeded")
    initial = {
        "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": _parameter_digest(no_latent),
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL": _parameter_digest(latent),
        "shared_base_initialisation": MODEL.shared_base_digest(no_latent),
    }
    if initial["shared_base_initialisation"] != MODEL.shared_base_digest(latent):
        raise QualificationError("matched base initialisation drift")
    training_payloads = {
        state_id: state_training_payload(rows) for state_id, rows in fit.items()
    }
    fit_optimization = _fit_optimization_summary(fit)
    contributing_state_ids = list(fit_optimization["fit_state_ids_contributing"])
    if not contributing_state_ids:
        raise QualificationError("fit split has no optimizer-contributing state")
    if any(
        training_payloads[state_id]["optimization_status"] != "CONTRIBUTING"
        for state_id in contributing_state_ids
    ):
        raise QualificationError("fit optimization summary/payload drift")
    # Hash and open every authorised fit tensor exactly once.  Re-reading the
    # roughly two-gigabyte TRUE trajectory set for each of sixty epochs would
    # change no science and would add avoidable I/O pressure.
    training_latents = {
        state_id: _tensor_rows_for_state(
            tensor_rows,
            state_id,
            payload["positions"],
            source="TRUE",
        )
        for state_id in contributing_state_ids
        for payload in (training_payloads[state_id],)
    }
    histories: dict[str, list[dict[str, float]]] = {}
    checkpoints: dict[str, Any] = {}
    specifications = (
        ("KINEMATIC_PLUS_NO_LATENT_RESIDUAL", no_latent, False),
        ("KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL", latent, True),
    )
    for condition, model, use_latent in specifications:
        model.to(device=device, dtype=torch.float32).train()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONTRACT.TRAINING["learning_rate"],
            weight_decay=CONTRACT.TRAINING["weight_decay"],
        )
        history: list[dict[str, float]] = []
        for epoch in range(1, CONTRACT.TRAINING["epochs"] + 1):
            totals = {"loss": 0.0, "pair": 0.0, "list": 0.0, "residual": 0.0}
            steps = 0
            for state_id in deterministic_epoch_order(contributing_state_ids, epoch):
                payload = training_payloads[state_id]
                base = torch.from_numpy(payload["base_features"]).to(device)
                query = torch.from_numpy(payload["query_features"]).to(device)
                anchors = torch.from_numpy(payload["anchors"]).to(device)
                utility = torch.from_numpy(payload["utility"]).to(device)
                pairwise_targets = torch.from_numpy(
                    payload["pairwise_targets"]
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                if use_latent:
                    current_np, future_np = training_latents[state_id]
                    current = torch.from_numpy(current_np.astype(np.float32)).to(device)
                    future = torch.from_numpy(future_np.astype(np.float32)).to(device)
                    scored = model(base, query, anchors, current, future)
                else:
                    scored = model(base, anchors)
                losses = MODEL.route_only_loss(
                    scores=scored.unsqueeze(0),
                    route_utilities=utility.unsqueeze(0),
                    kinematic_anchor=anchors.unsqueeze(0),
                    pairwise_targets=pairwise_targets.unsqueeze(0),
                )
                losses.total.backward()
                if not all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise QualificationError(f"non-finite gradient in {condition}")
                optimizer.step()
                totals["loss"] += float(losses.total.detach().cpu())
                totals["pair"] += float(losses.pairwise.detach().cpu())
                totals["list"] += float(losses.listwise.detach().cpu())
                totals["residual"] += float(losses.residual_l2.detach().cpu())
                steps += 1
            if steps != fit_optimization["fit_states_contributing"] or steps <= 0:
                raise QualificationError("optimizer-step count drift")
            history.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": steps,
                    "fit_states_total": fit_optimization["fit_states_total"],
                    "fit_states_contributing": fit_optimization[
                        "fit_states_contributing"
                    ],
                    "fit_states_skipped_zero_admissible": fit_optimization[
                        "fit_states_skipped_zero_admissible"
                    ],
                    "fit_states_skipped_singleton_admissible": fit_optimization[
                        "fit_states_skipped_singleton_admissible"
                    ],
                    "epoch_average_denominator": fit_optimization[
                        "epoch_average_denominator"
                    ],
                    **{key: value / steps for key, value in totals.items()},
                }
            )
        model.eval()
        checkpoint_filename = {
            "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": "no_latent_final_epoch_060.pt",
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL": "latent_true_future_final_epoch_060.pt",
        }[condition]
        checkpoint_path = output_root / "checkpoints" / checkpoint_filename
        checkpoint = {
            "schema": "plan_aware_ranker_checkpoint_receipt_v1",
            "condition": condition,
            "seed_family": CONTRACT.ROUTE_COST_SEED,
            "seed_metadata": copy.deepcopy(
                CONTRACT.CHECKPOINT_SEED_METADATA[condition]
            ),
            "epoch": CONTRACT.TRAINING["epochs"],
            "final_epoch_only": True,
            "parameter_count": counts[condition],
            "model_contract": MODEL.model_contract(condition),
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "parameter_digest": _parameter_digest(model),
            "optimizer_state_persisted": False,
            "training_history": history,
            "fit_optimization": copy.deepcopy(fit_optimization),
        }
        _atomic_torch_save(checkpoint_path, checkpoint)
        checkpoints[condition] = {
            **binding(checkpoint_path, relative_to=output_root),
            "seed_metadata": copy.deepcopy(checkpoint["seed_metadata"]),
            "parameter_digest": checkpoint["parameter_digest"],
            "parameter_count": counts[condition],
            "epoch": CONTRACT.TRAINING["epochs"],
            "fit_optimization_digest": hashlib.sha256(
                canonical_bytes(fit_optimization)[:-1]
            ).hexdigest(),
        }
        histories[condition] = history
        del optimizer
    torch.cuda.empty_cache()
    receipt = attach_digest(
        {
            "schema": "plan_aware_training_receipt_v1",
            "conditions": list(counts),
            "seed_family": CONTRACT.ROUTE_COST_SEED,
            "one_seed_family": True,
            "fit_states": len(fit),
            "fit_state_ids": sorted(fit, key=numeric_state_key),
            "fit_optimization": fit_optimization,
            "calibration_rows_opened": 0,
            "heldout_rows_opened": 0,
            "epochs": CONTRACT.TRAINING["epochs"],
            "final_epoch_only": True,
            "no_hyperparameter_sweep": True,
            "checkpoint_bindings": checkpoints,
            "checkpoint_seed_metadata": copy.deepcopy(
                CONTRACT.CHECKPOINT_SEED_METADATA
            ),
            "initial_parameter_digests": initial,
            "parameter_counts": counts,
            "training_history": histories,
            "predictor_training_steps": 0,
            "safety_or_auxiliary_losses": [],
            "pass": True,
        }
    )
    atomic_json(output_root / "receipts/training.json", receipt)
    return checkpoints, receipt, {condition: model for condition, model, _ in specifications}


def future_derangement(state_id: str, experiment_digest: str) -> list[int]:
    """Outcome-blind cyclic candidate derangement for one frozen state."""

    raw = hashlib.sha256(
        f"{experiment_digest}\x00{state_id}".encode("utf-8")
    ).digest()
    shift = 1 + int.from_bytes(raw[:8], "big") % (CANDIDATE_COUNT - 1)
    mapping = [(candidate + shift) % CANDIDATE_COUNT for candidate in range(CANDIDATE_COUNT)]
    if sorted(mapping) != list(range(CANDIDATE_COUNT)) or any(
        donor == candidate for candidate, donor in enumerate(mapping)
    ):
        raise QualificationError("candidate future derangement is not a derangement")
    return mapping


def donor_derangements(
    ids: Mapping[str, Sequence[str]],
    manifests: Mapping[str, Mapping[str, Any]],
    experiment_digest: str,
) -> dict[str, dict[str, str]]:
    """Freeze Stage-C same-family/same-split bijective donor cycles."""

    output: dict[str, dict[str, str]] = {}
    for ablation in CONTRACT.STAGE_C_ABLATION_INPUTS:
        mapping: dict[str, str] = {}
        for role in SPLIT_ROLES:
            families: dict[str, list[str]] = {}
            for state_id in ids[role]:
                families.setdefault(str(manifests[state_id]["family"]), []).append(state_id)
            for family, state_ids in families.items():
                try:
                    authority = CONTRACT.build_stage_c_donor_mapping(
                        state_ids,
                        split=role,
                        family=family,
                        ablation=ablation,
                        contract_digest=experiment_digest,
                    )
                except CONTRACT.ContractError as exc:
                    raise QualificationError(str(exc)) from exc
                if (
                    authority.get("ablation_feature")
                    != CONTRACT.STAGE_C_ABLATION_INPUTS[ablation]
                    or authority.get("row_count") != len(state_ids)
                ):
                    raise QualificationError(
                        f"Stage-C donor authority drift: {role}:{family}:{ablation}"
                    )
                for row in authority["rows"]:
                    mapping[str(row["recipient_state_id"])] = str(
                        row["donor_state_id"]
                    )
        if set(mapping) != {state for role in SPLIT_ROLES for state in ids[role]}:
            raise QualificationError("Stage-C donor map does not cover the panel")
        if any(recipient == donor for recipient, donor in mapping.items()):
            raise QualificationError("Stage-C donor mapping contains a fixed point")
        output[ablation] = mapping
    return output


def write_evaluation_contract(
    *,
    output_root: Path,
    source_freeze_commit: str,
    checkpoints: Mapping[str, Mapping[str, Any]],
    line_index: Mapping[tuple[str, int], Mapping[str, Any]],
    ids: Mapping[str, Sequence[str]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    CONTRACT.load_and_validate_contract()
    # The execution-only wrapper amendment must not perturb scientific keyed
    # derangements, donor maps, or any Stage-A scientific content.
    experiment_digest = CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
    route_records = [
        {
            "state_id": state_id,
            "candidate_index": candidate,
            **dict(record),
        }
        for (state_id, candidate), record in sorted(
            line_index.items(), key=lambda item: (numeric_state_key(item[0][0]), item[0][1])
        )
    ]
    route_record_digest = hashlib.sha256(canonical_bytes(route_records)[:-1]).hexdigest()
    future_maps = {
        state_id: future_derangement(state_id, experiment_digest)
        for role in SPLIT_ROLES
        for state_id in ids[role]
    }
    value = attach_digest(
        {
            "schema": "plan_aware_monotone_jepa_evaluation_contract_v1",
            "source_freeze_commit": source_freeze_commit,
            "experiment_contract_digest": experiment_digest,
            "checkpoint_bindings": copy.deepcopy(dict(checkpoints)),
            "checkpoint_count": 2,
            "checkpoint_epochs": [CONTRACT.TRAINING["epochs"]],
            "final_epoch_only": True,
            "calibration_used_for_selection": False,
            "heldout_opened_before_publication": False,
            "route_label_byte_index": {
                "records": len(route_records),
                "aggregate_digest": route_record_digest,
                "scientific_fields_parsed_during_indexing": [],
            },
            "future_latent_derangement_by_state": future_maps,
            "stage_c_input_donor_mappings": donor_derangements(
                ids, manifests, experiment_digest
            ),
            "metrics": copy.deepcopy(CONTRACT.METRIC_CONTRACT),
            "gates": copy.deepcopy(CONTRACT.GATES),
            "conditional_execution": copy.deepcopy(CONTRACT.STAGE_POLICY),
            "predictor_inference_authorised_before_true_gate": False,
            "pass": True,
        }
    )
    path = output_root / "receipts/evaluation_contract.json"
    atomic_json(path, value)
    if load_json(path) != value:
        raise QualificationError("evaluation contract roundtrip drift")
    return value


def _load_checkpoint_receipt(
    condition: str,
    record: Mapping[str, Any],
    *,
    output_root: Path,
) -> dict[str, Any]:
    import torch

    if condition not in CONTRACT.CHECKPOINT_SEED_METADATA:
        raise QualificationError(f"unknown route-ranker checkpoint: {condition}")
    path = output_root / str(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != record["sha256"]
    ):
        raise QualificationError(f"route-ranker checkpoint drift: {condition}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_seed = CONTRACT.CHECKPOINT_SEED_METADATA[condition]
    fit_optimization = checkpoint.get("fit_optimization")
    if isinstance(fit_optimization, Mapping):
        _validate_fit_optimization_summary(fit_optimization)
        _validate_training_history(
            checkpoint.get("training_history"), fit_optimization
        )
    fit_optimization_digest = (
        hashlib.sha256(canonical_bytes(fit_optimization)[:-1]).hexdigest()
        if isinstance(fit_optimization, Mapping)
        else None
    )
    if (
        checkpoint.get("schema") != "plan_aware_ranker_checkpoint_receipt_v1"
        or checkpoint.get("condition") != condition
        or checkpoint.get("seed_family") != CONTRACT.ROUTE_COST_SEED
        or checkpoint.get("epoch") != CONTRACT.TRAINING["epochs"]
        or checkpoint.get("final_epoch_only") is not True
        or checkpoint.get("optimizer_state_persisted") is not False
        or checkpoint.get("seed_metadata") != expected_seed
        or checkpoint.get("model_contract") != MODEL.model_contract(condition)
        or record.get("seed_metadata") != expected_seed
        or checkpoint.get("parameter_digest") != record.get("parameter_digest")
        or checkpoint.get("parameter_count") != record.get("parameter_count")
        or record.get("epoch") != CONTRACT.TRAINING["epochs"]
        or fit_optimization_digest != record.get("fit_optimization_digest")
    ):
        raise QualificationError(f"route-ranker checkpoint metadata drift: {condition}")
    return checkpoint


def _load_checkpoint_models(
    checkpoint_bindings: Mapping[str, Mapping[str, Any]], *, output_root: Path
) -> dict[str, Any]:
    import torch

    no_latent, latent = MODEL.build_matched_rankers(CONTRACT.ROUTE_COST_SEED)
    models = {
        "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": no_latent,
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL": latent,
    }
    device = torch.device("cuda:0")
    for condition, model in models.items():
        record = checkpoint_bindings[condition]
        checkpoint = _load_checkpoint_receipt(
            condition, record, output_root=output_root
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if _parameter_digest(model) != checkpoint["parameter_digest"]:
            raise QualificationError(f"route-ranker parameter digest drift: {condition}")
    return models


def score_dataset(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    models: Mapping[str, Any],
    tensor_rows: Mapping[tuple[str, str, int | None, int | None], Mapping[str, Any]],
    latent_source: str,
    future_derangements: Mapping[str, Sequence[int]] | None = None,
    logical_output_root: Path | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[dict[str, Any]]]]:
    """Score one authorised split; load no labels outside ``dataset``."""

    import torch

    device = torch.device("cuda:0")
    no_latent = models["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"]
    latent = models["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"]
    scores: dict[str, dict[str, np.ndarray]] = {
        "KINEMATIC_ROUTE_BASELINE": {},
        "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": {},
        f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{latent_source}": {},
        f"LATENT_BRANCH_ZERO_{latent_source}": {},
        "DETERMINISTIC_RANDOM": {},
    }
    if future_derangements is not None:
        scores[f"WITHIN_STATE_FUTURE_LATENT_DERANGEMENT_{latent_source}"] = {}
    evidence: dict[str, list[dict[str, Any]]] = {}
    with torch.inference_mode():
        for state_id, rows in dataset.items():
            base_np = np.stack([row["base_features"] for row in rows])
            query_np = np.stack([row["query_features"] for row in rows])
            anchor_np = np.asarray([row["kinematic_anchor"] for row in rows], np.float32)
            base = torch.from_numpy(base_np).to(device)
            query = torch.from_numpy(query_np).to(device)
            anchors = torch.from_numpy(anchor_np).to(device)
            current_np, future_np = _tensor_rows_for_state(
                tensor_rows,
                state_id,
                list(range(CANDIDATE_COUNT)),
                source=latent_source,
                logical_output_root=logical_output_root,
            )
            current = torch.from_numpy(current_np.astype(np.float32)).to(device)
            future = torch.from_numpy(future_np.astype(np.float32)).to(device)
            no_score = no_latent(base, anchors)
            components = latent.score_components(base, query, anchors, current, future)
            scores["KINEMATIC_ROUTE_BASELINE"][state_id] = anchor_np.astype(np.float64)
            scores["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"][state_id] = (
                no_score.cpu().numpy().astype(np.float64)
            )
            scores[f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{latent_source}"][state_id] = (
                components.score.cpu().numpy().astype(np.float64)
            )
            scores[f"LATENT_BRANCH_ZERO_{latent_source}"][state_id] = (
                components.kinematic_plus_base_score.cpu().numpy().astype(np.float64)
            )
            random_order = OLD_METRICS.deterministic_random_order(
                state_id,
                list(range(CANDIDATE_COUNT)),
                seed=CONTRACT.RANDOM_SEED,
            )
            random_rank = {candidate: rank for rank, candidate in enumerate(random_order)}
            scores["DETERMINISTIC_RANDOM"][state_id] = np.asarray(
                [-float(random_rank[candidate]) for candidate in range(CANDIDATE_COUNT)],
                np.float64,
            )
            deranged_components = None
            if future_derangements is not None:
                mapping = list(future_derangements[state_id])
                deranged_np = future_np[np.asarray(mapping)].copy()
                deranged = torch.from_numpy(deranged_np.astype(np.float32)).to(device)
                deranged_components = latent.score_components(
                    base, query, anchors, current, deranged
                )
                scores[
                    f"WITHIN_STATE_FUTURE_LATENT_DERANGEMENT_{latent_source}"
                ][state_id] = deranged_components.score.cpu().numpy().astype(np.float64)
            per_candidate: list[dict[str, Any]] = []
            attention = components.attention_weights.cpu().numpy()
            for candidate, row in enumerate(rows):
                deranged_candidate = (
                    None
                    if future_derangements is None
                    else int(future_derangements[state_id][candidate])
                )
                per_candidate.append(
                    {
                        "schema": "plan_aware_monotone_jepa_candidate_score_v1",
                        "state_id": state_id,
                        "family": row["family"],
                        "split": row["split"],
                        "route_intent_role": row["route_intent_role"],
                        "candidate_index": candidate,
                        "latent_source": latent_source,
                        "latent_artifact_bindings": _candidate_latent_bindings(
                            tensor_rows,
                            state_id=state_id,
                            candidate=candidate,
                            source=latent_source,
                            deranged_candidate=deranged_candidate,
                        ),
                        "waypoint_features": row["waypoint_features"],
                        "requested_action_blocks": row["requested_action_blocks"],
                        "applied_action_blocks": row["applied_action_blocks"],
                        "predictor_candidate_action_plan_3x10": row[
                            "predictor_candidate_action_plan_3x10"
                        ],
                        "previous_applied_command": row["previous_applied_command"],
                        "control_history": row["control_history"],
                        "deterministic_kinematic_features": row["kinematic_features"],
                        "deterministic_kinematic_score": float(anchor_np[candidate]),
                        "no_latent_base_residual": float(
                            scores["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"][state_id][candidate]
                            - anchor_np[candidate]
                        ),
                        "latent_model_base_residual": float(
                            components.base_residual[candidate].cpu()
                        ),
                        "latent_residual": float(components.latent_residual[candidate].cpu()),
                        "final_no_latent_score": float(
                            scores["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"][state_id][candidate]
                        ),
                        "final_latent_score": float(components.score[candidate].cpu()),
                        "latent_branch_zero_score": float(
                            components.kinematic_plus_base_score[candidate].cpu()
                        ),
                        "deranged_latent_score": (
                            None
                            if deranged_components is None
                            else float(deranged_components.score[candidate].cpu())
                        ),
                        "future_derangement_donor_candidate": (
                            deranged_candidate
                        ),
                        "attention_weight_sum_by_timepoint": [
                            float(value) for value in attention[candidate].sum(axis=1)
                        ],
                        "p_d": row["p_d"],
                        "p_theta": row["p_theta"],
                        "completed": row["completed"],
                        "stuck": row["stuck"],
                        "immediate_contact_h1": row["immediate_contact_h1"],
                        "descriptive_contact_h2": row["descriptive_contact_h2"],
                        "descriptive_contact_h3": row["descriptive_contact_h3"],
                        "successor_safe_action_count": row["successor_safe_action_count"],
                        "successor_viable": row["successor_viable"],
                        "oracle_viability_admissible": row["oracle_viability_admissible"],
                    }
                )
            evidence[state_id] = per_candidate
    return scores, evidence


def metric_candidates(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for state_id, rows in dataset.items():
        output[state_id] = [
            {
                "state_id": state_id,
                "family": row["family"],
                "role": row["split"],
                "candidate_index": int(row["candidate_index"]),
                "p_d": float(row["p_d"]),
                "p_theta": float(row["p_theta"]),
                "completed": bool(row["completed"]),
                "stuck": bool(row["stuck"]),
                "immediate_contact_h1": bool(row["immediate_contact_h1"]),
                "descriptive_contact_h2": bool(row["descriptive_contact_h2"]),
                "descriptive_contact_h3": bool(row["descriptive_contact_h3"]),
                "successor_safe_action_count": int(row["successor_safe_action_count"]),
                "successor_viable": bool(row["successor_viable"]),
                "oracle_viability_admissible": bool(row["oracle_viability_admissible"]),
            }
            for row in rows
        ]
    return output


def summarize_score_maps(
    dataset: Mapping[str, Sequence[Mapping[str, Any]]],
    score_maps: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, Any]:
    candidates = metric_candidates(dataset)
    return {
        source: METRICS.summarize_scores(
            candidates,
            scores,
            source_id=source,
        )
        for source, scores in score_maps.items()
    }


def _tracked_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _validate_frozen_authorities(
    *, execution_correction_custody: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Validate only prospectively frozen bytes; calculate no route metric."""

    frozen_contract = CONTRACT.load_and_validate_contract(_tracked_path(CONTRACT.TRACKED_CONTRACT_PATH))
    frozen_schema = CONTRACT.load_and_validate_output_schema(
        _tracked_path(CONTRACT.TRACKED_OUTPUT_SCHEMA_PATH)
    )
    frozen_fixture = CONTRACT.load_and_validate_evaluator_fixture(
        _tracked_path(CONTRACT.TRACKED_FIXTURE_PATH)
    )
    closure = CONTRACT.load_and_validate_source_closure(
        _tracked_path(CONTRACT.TRACKED_SOURCE_CLOSURE_PATH)
    )
    route_role_authority = CONTRACT.load_and_validate_route_role_authority(
        _tracked_path(CONTRACT.TRACKED_ROUTE_ROLE_AUTHORITY_PATH)
    )
    # The original source closure is itself immutable scientific authority.  A
    # correction execution validates changed implementation bytes against the
    # separate amendment closure instead of pretending they still match the
    # original freeze's historical source rows.
    active_closure = closure
    if execution_correction_custody is not None:
        active_closure = CONTRACT.load_and_validate_execution_correction_source_closure(
            _tracked_path(CONTRACT.TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH)
        )
    for row in active_closure["rows"]:
        path = ROOT / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise QualificationError(f"source-closure row drift: {row['path']}")
    for label, record in CONTRACT.PANEL_BINDINGS.items():
        path = ROOT / str(record["path"])
        verify_binding(path, str(record["sha256"]), label=f"panel.{label}")
        if path.stat().st_size != int(record["bytes"]):
            raise QualificationError(f"panel.{label} byte-size drift")
    for label in (
        "tensor_index",
        "batch_manifest",
        "goal_view_index",
        "context_reconstruction_index",
        "persistence_receipt",
    ):
        record = CONTRACT.PREDECESSOR_TENSOR_PACKAGE[label]
        path = PREDECESSOR_ROOT / str(record["path"])
        verify_binding(path, str(record["sha256"]), label=f"predecessor.{label}")
        if path.stat().st_size != int(record["bytes"]):
            raise QualificationError(f"predecessor.{label} byte-size drift")
        if "content_digest" in record and load_json(path).get("content_digest") != record[
            "content_digest"
        ]:
            raise QualificationError(f"predecessor.{label} content digest drift")
    for label, record in CONTRACT.CHECKPOINT_BINDINGS.items():
        path = Path(str(record["path"]))
        verify_binding(path, str(record["sha256"]), label=f"checkpoint.{label}")
        if path.stat().st_size != int(record["bytes"]):
            raise QualificationError(f"checkpoint.{label} byte-size drift")
    encoder_path = Path(str(CONTRACT.ENCODER_BINDING["path"]))
    verify_exact_file_record(
        encoder_path, CONTRACT.ENCODER_BINDING, label="target_encoder"
    )
    target_index_value = load_json(TARGET_LATENT_INDEX)
    tensor_index_value = load_json(LATENT_INDEX)
    _validate_true_future_index_reconciliation(
        target_index_value,
        tensor_index_value,
        expected_index_sha256=CONTRACT.PANEL_BINDINGS["true_future_index"][
            "sha256"
        ],
    )
    return {
        "contract": frozen_contract,
        "output_schema": frozen_schema,
        "fixture": frozen_fixture,
        "source_closure": closure,
        "execution_correction_source_closure": (
            active_closure if execution_correction_custody is not None else None
        ),
        "route_role_authority": route_role_authority,
    }


def _runtime_source_freeze() -> str:
    """Compatibility wrapper: amended execution has exactly one freeze mode."""

    return str(_runtime_execution_correction_custody()["source_freeze_commit"])


def _runtime_execution_correction_custody() -> dict[str, Any]:
    """Validate and normalize the mandatory one-attempt amendment custody."""

    try:
        runtime = CONTRACT.validate_execution_correction_freeze_custody(ROOT)
    except CONTRACT.ContractError as exc:
        raise QualificationError(str(exc)) from exc
    archive = runtime.get("failed_archive_custody")
    if (
        runtime.get("pass") is not True
        or runtime.get("fresh_attempts_authorised") != 1
        or runtime.get("fresh_attempts_already_consumed") != 0
        or runtime.get("files_reused") != 0
        or not isinstance(archive, Mapping)
        or archive.get("pass") is not True
        or archive.get("files_reused") != 0
        or archive.get("archive_path")
        != str(CONTRACT.EXECUTION_CORRECTION_FAILED_ARCHIVE)
    ):
        raise QualificationError("execution-correction runtime custody drift")
    amendment_closure_path = _tracked_path(
        CONTRACT.TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH
    )
    amendment_closure = CONTRACT.load_and_validate_execution_correction_source_closure(
        amendment_closure_path
    )
    return {
        "amendment": copy.deepcopy(CONTRACT.EXECUTION_CORRECTION_AMENDMENT_BINDING),
        "amendment_source_closure": {
            **binding(amendment_closure_path, relative_to=ROOT),
            "content_digest": amendment_closure["content_digest"],
            "rows": amendment_closure["row_count"],
        },
        "archive_path": str(archive["archive_path"]),
        "archive_inventory": copy.deepcopy(archive["inventory"]),
        "failure_receipt": copy.deepcopy(archive["failure_receipt"]),
        "source_freeze_commit": str(runtime["source_freeze_commit"]),
        "files_reused": 0,
        "conditional_child_environment_preflight": None,
        "execution_correction_replay": None,
        "pass": True,
    }


def _tracked_publication_paths() -> tuple[Path, Path]:
    return (
        _tracked_path(CONTRACT.TRACKED_RESULT_PATH),
        _tracked_path(CONTRACT.TRACKED_REPORT_PATH),
    )


def _assert_publication_destinations_absent() -> None:
    existing = [path for path in _tracked_publication_paths() if path.exists()]
    if existing:
        raise QualificationError(
            "tracked result/report publication destination already exists: "
            + ", ".join(str(path) for path in existing)
        )


def _publish_tracked_result_and_report(output_root: Path) -> None:
    """Publish both tracked witnesses or remove both on any write failure."""

    result_target, report_target = _tracked_publication_paths()
    _assert_publication_destinations_absent()
    try:
        atomic_bytes(result_target, (output_root / "result.json").read_bytes())
        atomic_bytes(report_target, (output_root / "report.md").read_bytes())
    except BaseException:
        # Both targets were proven absent immediately above.  Remove either
        # destination even when a write raised after its atomic replacement,
        # so a partial tracked publication cannot survive failure cleanup.
        result_target.unlink(missing_ok=True)
        report_target.unlink(missing_ok=True)
        raise


def _failed_archive_inventory(archive: Path) -> dict[str, Any]:
    records = [
        binding(path, relative_to=archive)
        for path in sorted(value for value in archive.rglob("*") if value.is_file())
    ]
    return {
        "files": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "manifest_sha256": hashlib.sha256(canonical_bytes(records)[:-1]).hexdigest(),
    }


def _validated_prior_smoke_failure_custody(
    output_root: Path,
    *,
    source_freeze: str,
    allow_current_freeze_as_correction_base: bool = False,
) -> list[dict[str, Any]]:
    """Allow only fresh retry after audited pre-training smoke failures."""

    archives = sorted(output_root.parent.glob(f".{output_root.name}.failed-*"))
    if archives and _active_experiment_processes():
        raise QualificationError("prior smoke retry is forbidden while a process is active")
    custody: list[dict[str, Any]] = []
    for archive in archives:
        receipt_path = archive / "receipts/failure.json"
        if not receipt_path.is_file():
            raise QualificationError(
                f"prior failed attempt lacks its failure receipt: {archive}"
            )
        receipt = load_json(receipt_path)
        prior_freeze = receipt.get("source_freeze_commit")
        if (
            receipt.get("content_digest") != content_digest(receipt)
            or receipt.get("schema") != "plan_aware_monotone_jepa_failure_v1"
            or receipt.get("phase") != "TRAINING_SMOKE"
            or receipt.get("partial_artifacts_reusable") is not False
            or receipt.get("full_training_epochs_completed") != 0
            or receipt.get("calibration_rows_opened") != 0
            or receipt.get("heldout_rows_opened") != 0
            or receipt.get("final_checkpoint_published") is not False
            or receipt.get("nothing_running") is not True
            or not isinstance(receipt.get("error_type"), str)
            or not isinstance(receipt.get("error_message"), str)
            or not isinstance(prior_freeze, str)
            or len(prior_freeze) != 40
            or (
                prior_freeze == source_freeze
                and not allow_current_freeze_as_correction_base
            )
        ):
            raise QualificationError(
                "a prior failed scientific attempt is not an authorised corrected "
                f"smoke-only retry: {archive}"
            )
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", prior_freeze, source_freeze],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode:
            raise QualificationError(
                f"prior smoke freeze is not an ancestor of corrected freeze: {archive}"
            )
        source_closure_path = archive / "receipts/source_closure.json"
        if not source_closure_path.is_file():
            raise QualificationError(
                f"eligible smoke archive lacks its preexecution source closure: {archive}"
            )
        source_closure = load_json(source_closure_path)
        if source_closure.get("content_digest") != content_digest(source_closure):
            raise QualificationError(
                f"prior smoke source closure self-digest drift: {archive}"
            )
        try:
            CONTRACT.validate_source_closure(source_closure)
        except CONTRACT.ContractError as exc:
            raise QualificationError(
                f"prior smoke source closure validation failed: {archive}: {exc}"
            ) from exc
        source_closure_binding = {
            **binding(source_closure_path),
            "content_digest": source_closure["content_digest"],
        }
        custody.append(
            {
                "archive_path": str(archive),
                "source_freeze_commit": prior_freeze,
                "failure_receipt": {
                    **binding(receipt_path),
                    "content_digest": receipt["content_digest"],
                },
                "source_closure": source_closure_binding,
                "inventory": _failed_archive_inventory(archive),
                "files_reused": 0,
            }
        )
    try:
        return CONTRACT.validate_prior_smoke_failure_custody(custody)
    except CONTRACT.ContractError as exc:
        raise QualificationError(str(exc)) from exc


def _new_attempt(
    output_root: Path,
    source_freeze: str,
    *,
    execution_correction_custody: Mapping[str, Any] | None = None,
) -> Path:
    if output_root.resolve() != CONTRACT.OUTPUT_ROOT.resolve():
        raise QualificationError("canonical output path differs from the frozen contract")
    _assert_publication_destinations_absent()
    if output_root.exists():
        raise QualificationError(f"canonical output already exists: {output_root}")
    siblings = list(output_root.parent.glob(f".{output_root.name}.attempt-*"))
    if siblings:
        raise QualificationError(f"a live/abandoned attempt already exists: {siblings}")
    if execution_correction_custody is None:
        _validated_prior_smoke_failure_custody(
            output_root, source_freeze=source_freeze
        )
    else:
        try:
            archive = CONTRACT.validate_execution_correction_archive()
        except CONTRACT.ContractError as exc:
            raise QualificationError(str(exc)) from exc
        failed_archives = sorted(
            path.resolve()
            for path in output_root.parent.glob(f".{output_root.name}.failed-*")
        )
        if (
            source_freeze
            != execution_correction_custody.get("source_freeze_commit")
            or execution_correction_custody.get("archive_path")
            != archive["archive_path"]
            or execution_correction_custody.get("archive_inventory")
            != archive["inventory"]
            or execution_correction_custody.get("files_reused") != 0
            or failed_archives
            != [CONTRACT.EXECUTION_CORRECTION_FAILED_ARCHIVE.resolve()]
            or _active_experiment_processes()
        ):
            raise QualificationError(
                "execution-correction fresh-attempt custody or namespace drift"
            )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    attempt = output_root.parent / (
        f".{output_root.name}.attempt-{source_freeze[:12]}-{time.time_ns()}-{os.getpid()}"
    )
    attempt.mkdir(mode=0o755)
    return attempt


def _prohibition_counters() -> dict[str, int]:
    return {
        "predictor_training_steps": 0,
        "safety_model_training_steps": 0,
        "fresh_panel_states": 0,
        "fresh_candidates": 0,
        "new_sensor_layouts": 0,
        "protected_contact_scope_changes": 0,
        "closed_loop_navigation_actions": 0,
        "memory_implementations": 0,
        "novelty_implementations": 0,
        "topological_routing_implementations": 0,
        "beacon_capture_implementations": 0,
    }


def _conditional_child_environment_preflight(attempt: Path) -> dict[str, Any]:
    """Probe exact CPU/GPU child imports before opening fit outcomes or tensors."""

    helper = _conditional_helper()
    try:
        cpu_probe = helper.probe_child_interpreter(
            helper.CPU_INTERPRETER, require_genesis=True
        )
        gpu_probe = helper.probe_child_interpreter(
            helper.GPU_INTERPRETER, require_genesis=False
        )
    except helper.MaterialisationError as exc:
        raise QualificationError(str(exc)) from exc
    receipt = attach_digest(
        {
            "schema": (
                "plan_aware_monotone_jepa_cost_v1."
                "conditional_child_environment_preflight.v1"
            ),
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "cpu_child": cpu_probe,
            "gpu_child": gpu_probe,
            "inherited_python_environment_presence": {
                key: key in os.environ
                for key in helper.SCRUBBED_PYTHON_ENVIRONMENT_KEYS
            },
            "fit_outcome_rows_opened": 0,
            "calibration_rows_opened": 0,
            "heldout_rows_opened": 0,
            "tensor_rows_opened": 0,
            "training_steps": 0,
            "pass": True,
        }
    )
    atomic_json(attempt / "receipts/conditional_child_environment_preflight.json", receipt)
    return receipt


def _preexecution_receipt(
    *,
    attempt: Path,
    source_freeze: str,
    frozen: Mapping[str, Any],
    conditional_child_environment_preflight: Mapping[str, Any],
    execution_correction_custody: dict[str, Any],
) -> dict[str, Any]:
    stat = os.statvfs(attempt.parent)
    # Preserve the exact closure in the attempt itself so a smoke-failure
    # archive can bind the authority that governed that attempt even after a
    # corrected freeze publishes a new tracked closure.
    source_closure_snapshot_path = attempt / "receipts/source_closure.json"
    atomic_json(source_closure_snapshot_path, frozen["source_closure"])
    source_closure_snapshot = {
        **binding(source_closure_snapshot_path, relative_to=attempt),
        "content_digest": frozen["source_closure"]["content_digest"],
    }
    receipt = attach_digest(
        {
            "schema": "plan_aware_preexecution_receipt_v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "clean_scientific_source_commit": SOURCE_COMMIT,
            "source_freeze_commit": source_freeze,
            "requirements_ancestor": REQUIRED_ANCESTOR,
            "requirements_ancestor_valid": True,
            "contract_sha256": CONTRACT.CONTRACT_SHA256,
            "output_schema_sha256": CONTRACT.OUTPUT_SCHEMA_SHA256,
            "source_closure_content_digest": frozen["source_closure"]["content_digest"],
            "source_closure_rows": frozen["source_closure"]["row_count"],
            "source_closure_snapshot": source_closure_snapshot,
            "route_role_authority_binding": copy.deepcopy(
                CONTRACT.ROUTE_ROLE_RECEIPT_BINDING
            ),
            "conditional_child_environment_preflight": {
                **binding(
                    attempt
                    / "receipts/conditional_child_environment_preflight.json",
                    relative_to=attempt,
                ),
                "content_digest": conditional_child_environment_preflight[
                    "content_digest"
                ],
            },
            "execution_correction_custody": copy.deepcopy(
                execution_correction_custody
            ),
            "panel_bindings": copy.deepcopy(CONTRACT.PANEL_BINDINGS),
            "encoder_binding": copy.deepcopy(CONTRACT.ENCODER_BINDING),
            "predictor_checkpoint_bindings": copy.deepcopy(CONTRACT.CHECKPOINT_BINDINGS),
            "predecessor_tensor_package": copy.deepcopy(CONTRACT.PREDECESSOR_TENSOR_PACKAGE),
            "ranker_seed_family": CONTRACT.ROUTE_COST_SEED,
            "predictor_seed": CONTRACT.PREDICTOR_SEED,
            "cpu_worker_benchmark": copy.deepcopy(CONTRACT.CPU_WORKER_BENCHMARK),
            "numerical_thread_environment": copy.deepcopy(CONTRACT.NUMERICAL_THREAD_ENV),
            "attempt_root": str(attempt),
            "canonical_output_root": str(CONTRACT.OUTPUT_ROOT),
            "prior_smoke_failure_custody": [],
            "output_free_bytes": int(stat.f_bavail * stat.f_frsize),
            "outcome_barrier": {
                "fit_rows_opened": 0,
                "calibration_rows_opened": 0,
                "heldout_rows_opened": 0,
                "ranker_checkpoints_published": 0,
                "evaluation_contract_published": False,
            },
            "accidental_exposures": copy.deepcopy(list(CONTRACT.ACCIDENTAL_EXPOSURES)),
            "prohibition_counters": _prohibition_counters(),
            "pass": True,
        }
    )
    atomic_json(attempt / "receipts/preexecution.json", receipt)
    return receipt


def _training_target_rows(
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role in SPLIT_ROLES:
        for state_id in sorted(datasets[role], key=numeric_state_key):
            rows = list(datasets[role][state_id])
            positions = admissible_positions(rows)
            conditioned_rows = [rows[position] for position in positions]
            conditioned_utilities = (
                route_utility(conditioned_rows)
                if conditioned_rows
                else np.empty((0,), dtype=np.float64)
            )
            try:
                optimization_status = (
                    CONTRACT.fit_state_optimization_status(len(positions))
                    if role == FIT
                    else None
                )
            except CONTRACT.ContractError as exc:
                raise QualificationError(str(exc)) from exc
            utility_by_position = {
                position: float(conditioned_utilities[index])
                for index, position in enumerate(positions)
            }
            for position, row in enumerate(rows):
                conditioned = position in utility_by_position
                output.append(
                    {
                        "schema": "plan_aware_route_only_target_row_v1",
                        "state_id": state_id,
                        "candidate_index": int(row["candidate_index"]),
                        "family": row["family"],
                        "role": role,
                        "route_intent_role": row["route_intent_role"],
                        "waypoint_features": row["waypoint_features"],
                        "realised_distance_progress_m": float(row["p_d"]),
                        "realised_heading_improvement_rad": float(row["p_theta"]),
                        "local_waypoint_completed": bool(row["completed"]),
                        "oracle_viability_admissible_conditioning": conditioned,
                        "fit_state_optimization_status": optimization_status,
                        "used_for_fit": bool(
                            role == FIT
                            and optimization_status == "CONTRIBUTING"
                            and conditioned
                        ),
                        "conditioned_margin_borda_utility": utility_by_position.get(
                            position
                        ),
                        "training_target_heads": ["pairwise_route_order", "listwise_route_order"],
                    }
                )
    return output


def _population_membership(row: Mapping[str, Any]) -> dict[str, bool]:
    """Return the exact frozen population membership for one candidate row."""

    return {
        "ALL_CANDIDATES": True,
        "ORACLE_CONTACT_FREE": not bool(row["immediate_contact_h1"]),
        "ORACLE_VIABILITY_ADMISSIBLE": bool(
            row["oracle_viability_admissible"]
        ),
    }


def _validate_population_membership(row: Mapping[str, Any], *, label: str) -> None:
    expected = _population_membership(row)
    if row.get("population_membership") != expected:
        raise QualificationError(
            f"{label} population-membership drift: "
            f"{row.get('split')}:{row.get('latent_source')}:"
            f"{row.get('state_id')}:{row.get('candidate_index')}"
        )


def _validate_score_row_arithmetic(
    rows: Sequence[Mapping[str, Any]], *, stage: str
) -> None:
    """Fail closed on persisted score aliases, arithmetic, and condition sets."""

    if stage not in {"STAGE_A", "CONDITIONAL"}:
        raise QualificationError(f"unknown score-row arithmetic stage {stage!r}")

    def close(left: Any, right: Any) -> bool:
        try:
            return math.isclose(
                float(left), float(right), rel_tol=0.0, abs_tol=2e-6
            )
        except (TypeError, ValueError):
            return False

    for row in rows:
        source = str(row.get("latent_source"))
        score_key = "scores" if stage == "STAGE_A" else "route_scores"
        scores = row.get(score_key)
        if not isinstance(scores, Mapping):
            raise QualificationError(f"{stage} row lacks {score_key}")
        expected = {
            "KINEMATIC_ROUTE_BASELINE",
            "KINEMATIC_PLUS_NO_LATENT_RESIDUAL",
            f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}",
            f"LATENT_BRANCH_ZERO_{source}",
            "DETERMINISTIC_RANDOM",
        }
        if stage == "STAGE_A":
            if source != "TRUE":
                raise QualificationError("Stage-A latent-source identity drift")
            expected.update(
                {
                    "WITHIN_STATE_FUTURE_LATENT_DERANGEMENT_TRUE",
                    *CONTRACT.RAW_COST_REREDUCED_SOURCE_IDS,
                }
            )
        if set(scores) != expected:
            raise QualificationError(
                f"{stage} score condition-key drift: "
                f"{row.get('state_id')}:{row.get('candidate_index')}"
            )
        if any(not math.isfinite(float(value)) for value in scores.values()):
            raise QualificationError(f"{stage} non-finite persisted score")

        anchor = row.get("deterministic_kinematic_score")
        no_base = row.get("no_latent_base_residual")
        latent_base = row.get("latent_model_base_residual")
        latent_residual = row.get("latent_residual")
        final_no = row.get("final_no_latent_score")
        zero = row.get("latent_branch_zero_score")
        final_latent = row.get("final_latent_score")
        try:
            anchor_value = float(anchor)
            no_base_value = float(no_base)
            latent_base_value = float(latent_base)
            latent_residual_value = float(latent_residual)
        except (TypeError, ValueError) as exc:
            raise QualificationError(f"{stage} score component is missing/non-numeric") from exc
        condition = f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"
        zero_condition = f"LATENT_BRANCH_ZERO_{source}"
        if not all(
            (
                close(scores["KINEMATIC_ROUTE_BASELINE"], anchor),
                close(scores["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"], final_no),
                close(final_no, anchor_value + no_base_value),
                close(scores[zero_condition], zero),
                close(zero, anchor_value + latent_base_value),
                close(scores[condition], final_latent),
                close(
                    final_latent,
                    anchor_value + latent_base_value + latent_residual_value,
                ),
            )
        ):
            raise QualificationError(
                f"{stage} score arithmetic/alias drift: "
                f"{source}:{row.get('state_id')}:{row.get('candidate_index')}"
            )
        deranged = row.get("deranged_latent_score")
        if stage == "STAGE_A":
            if not close(
                deranged,
                scores["WITHIN_STATE_FUTURE_LATENT_DERANGEMENT_TRUE"],
            ):
                raise QualificationError("Stage-A deranged-score alias drift")
        elif deranged is not None:
            raise QualificationError("conditional row has unexpected deranged-score alias")


def _stage_a_ledger_rows(
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    score_maps_by_role: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    evidence_by_role: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for role in SPLIT_ROLES:
        score_maps = score_maps_by_role[role]
        for state_id in sorted(datasets[role], key=numeric_state_key):
            evidence_rows = evidence_by_role[role][state_id]
            for position, source in enumerate(evidence_rows):
                row = copy.deepcopy(dict(source))
                row["schema"] = "plan_aware_evaluation_row_v1"
                row["population_membership"] = _population_membership(row)
                row["scores"] = {
                    name: float(values[state_id][position])
                    for name, values in score_maps.items()
                }
                output.append(row)
    _validate_score_row_arithmetic(output, stage="STAGE_A")
    return output


def _write_training_ledgers(
    *,
    attempt: Path,
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    training_receipt: Mapping[str, Any],
) -> None:
    target_rows = _training_target_rows(datasets)
    if len(target_rows) != STATE_COUNT * CANDIDATE_COUNT:
        raise QualificationError("route-only target ledger cardinality drift")
    atomic_bytes(
        attempt / "ledgers/route_only_targets.jsonl",
        b"".join(canonical_bytes(row) for row in target_rows),
    )
    epoch_rows: list[dict[str, Any]] = []
    for condition, history in training_receipt["training_history"].items():
        for row in history:
            epoch_rows.append(
                {
                    "schema": "plan_aware_training_epoch_row_v1",
                    "condition": condition,
                    **copy.deepcopy(dict(row)),
                }
            )
    if len(epoch_rows) != 120:
        raise QualificationError("training epoch ledger cardinality drift")
    atomic_bytes(
        attempt / "ledgers/training_epochs.jsonl",
        b"".join(canonical_bytes(row) for row in epoch_rows),
    )


def _historical_raw_comparators() -> dict[str, Any]:
    """Copy registered predecessor summaries; never recalculate raw cosine."""

    authority = CONTRACT.ROUTE_ROLE_AUTHORITY_BINDING
    authority_path = Path(str(authority["path"]))
    if authority_path.resolve() != PREDECESSOR_METRICS.resolve():
        raise QualificationError("historical comparator authority path drift")
    verify_binding(
        authority_path,
        str(authority["sha256"]),
        label="historical_comparator_source",
    )
    if authority_path.stat().st_size != int(authority["bytes"]):
        raise QualificationError("historical comparator authority byte-size drift")
    predecessor = load_json(PREDECESSOR_METRICS)
    if (
        predecessor.get("content_digest") != authority["content_digest"]
        or predecessor.get("content_digest") != content_digest(predecessor)
    ):
        raise QualificationError("historical comparator authority content drift")
    per_role = predecessor.get("per_role")
    if not isinstance(per_role, Mapping):
        raise QualificationError("predecessor comparator per-role summaries are missing")
    source_map = {
        "RAW_TRUE_FUTURE_GOAL_COSINE": "TRUE_FUTURE_LATENT_COST",
        "RAW_R1_GOAL_COSINE": "ONE_STEP_PREDICTED_LATENT_COST",
        "RAW_RR_GOAL_COSINE": "TWO_STEP_PREDICTED_LATENT_COST",
    }
    copied = {}
    for destination, source in source_map.items():
        if source not in per_role or "heldout" not in per_role[source]:
            raise QualificationError(f"predecessor comparator summary missing: {source}")
        copied[destination] = copy.deepcopy(per_role[source]["heldout"])
    return {
        "policy": "HISTORICAL_BYTE_BOUND_COMPARATOR_COPY_NO_COSINE_RECOMPUTATION",
        "source_binding": binding(PREDECESSOR_METRICS),
        "comparators": copied,
        "metric_definition_comparable_to_successor_borda_metrics": False,
        "use": "historical_context_only",
    }


def _predecessor_raw_goal_score_maps(
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    *,
    heldout_barrier_open: bool,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Re-reduce persisted raw-cosine costs under the successor Borda metric."""

    if not heldout_barrier_open:
        raise QualificationError(
            "predecessor candidate evidence cannot open before the heldout barrier"
        )
    record = CONTRACT.PREDECESSOR_CANDIDATE_EVIDENCE_BINDING
    reference = Path(str(record["path"]))
    path = reference if reference.is_absolute() else PREDECESSOR_ROOT / reference
    verify_exact_file_record(path, record, label="predecessor_candidate_evidence")
    successor_map = dict(record["successor_source_mapping"])
    if tuple(successor_map) != tuple(CONTRACT.RAW_COST_REREDUCED_SOURCE_IDS):
        raise QualificationError("raw-cost successor source ordering drift")
    source_map = {predecessor: successor for successor, predecessor in successor_map.items()}
    if len(source_map) != len(successor_map):
        raise QualificationError("raw-cost predecessor source mapping is not one-to-one")
    expected_rows = STATE_COUNT * CANDIDATE_COUNT * len(source_map)
    maps: dict[str, dict[str, dict[str, np.ndarray]]] = {
        role: {
            destination: {
                state_id: np.full(CANDIDATE_COUNT, np.nan, dtype=np.float64)
                for state_id in datasets[role]
            }
            for destination in source_map.values()
        }
        for role in SPLIT_ROLES
    }
    projected_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            source = json.loads(line)
            if not isinstance(source, Mapping) or source.get("schema") != (
                record["schema"]
            ):
                raise QualificationError(
                    f"predecessor candidate-evidence schema drift at row {line_number}"
                )
            source_id = str(source.get("source"))
            if source_id not in source_map:
                raise QualificationError(
                    f"predecessor candidate-evidence source drift: {source_id}"
                )
            role = str(source.get("role"))
            state_id = str(source.get("state_id"))
            candidate = int(source.get("candidate_index", -1))
            identity = (source_id, state_id, candidate)
            if (
                identity in seen
                or role not in datasets
                or state_id not in datasets[role]
                or not 0 <= candidate < CANDIDATE_COUNT
            ):
                raise QualificationError(
                    f"predecessor candidate-evidence identity drift: {identity}"
                )
            expected = datasets[role][state_id][candidate]
            missing = set(record["required_reduction_fields"]) - set(source)
            if missing:
                raise QualificationError(
                    "predecessor candidate-evidence required-field drift: "
                    f"{identity}:{sorted(missing)}"
                )
            expected_population = _population_membership(expected)
            route = source.get("oracle_route_fields_h3_primary")
            expected_outcomes = {
                "immediate_contact_h1": bool(expected["immediate_contact_h1"]),
                "successor_safe_action_count": int(
                    expected["successor_safe_action_count"]
                ),
                "successor_viable": bool(expected["successor_viable"]),
                "oracle_viability_admissible": bool(
                    expected["oracle_viability_admissible"]
                ),
                "successor_nonviable": not bool(expected["successor_viable"]),
                "stuck": bool(expected["stuck"]),
                "completed": bool(expected["completed"]),
            }
            if (
                source.get("family") != expected["family"]
                or source.get("role") != role
                or source.get("population_membership") != expected_population
                or not isinstance(route, Mapping)
                or not {"p_d", "p_theta", "completed", "stuck"}.issubset(route)
                or float(route.get("p_d", math.nan)) != float(expected["p_d"])
                or float(route.get("p_theta", math.nan)) != float(expected["p_theta"])
                or bool(route.get("completed")) != bool(expected["completed"])
                or bool(route.get("stuck")) != bool(expected["stuck"])
                or any(source.get(key) != value for key, value in expected_outcomes.items())
            ):
                raise QualificationError(
                    f"predecessor candidate-evidence panel alignment drift: {identity}"
                )
            cost = float(source["cost_h3"])
            if not math.isfinite(cost):
                raise QualificationError(
                    f"predecessor candidate-evidence non-finite cost: {identity}"
                )
            destination = source_map[source_id]
            score = -cost
            maps[role][destination][state_id][candidate] = score
            projected_rows.append(
                {
                    "schema": "plan_aware_raw_cost_rereduced_row_v1",
                    "split": role,
                    "state_id": state_id,
                    "family": expected["family"],
                    "candidate_index": candidate,
                    "source_id": destination,
                    "predecessor_source": source_id,
                    "cost_h3": cost,
                    "score": score,
                    "population_membership": expected_population,
                }
            )
            seen.add(identity)
    if len(seen) != expected_rows or any(
        not np.isfinite(values).all()
        for role_maps in maps.values()
        for states in role_maps.values()
        for values in states.values()
    ):
        raise QualificationError("predecessor candidate-evidence coverage drift")
    projected_rows.sort(
        key=lambda row: (
            tuple(CONTRACT.RAW_COST_REREDUCED_SOURCE_IDS).index(
                str(row["source_id"])
            ),
            numeric_state_key(str(row["state_id"])),
            int(row["candidate_index"]),
        )
    )
    receipt = {
        "schema": "plan_aware_raw_goal_cosine_successor_metric_reduction_v1",
        "frozen_source_binding": copy.deepcopy(record),
        "observed_source_binding": binding(path),
        "source_rows": expected_rows,
        "successor_source_mapping": successor_map,
        "score_transform": "higher_is_better_score = -cost_h3",
        "successor_ordering_metric": "population_conditioned_margin_borda_v1",
        "projected_row_content_digest": hashlib.sha256(
            canonical_bytes(projected_rows)[:-1]
        ).hexdigest(),
        "raw_cosine_inference_executions": 0,
        "historical_predecessor_aggregate_reused_for_successor_metric": False,
        "post_heldout_barrier_read_only_reduction": True,
    }
    return maps, projected_rows, receipt


def _validate_raw_cost_merged_scores(
    stage_rows: Sequence[Mapping[str, Any]],
    raw_rows: Sequence[Mapping[str, Any]],
) -> None:
    expected: dict[tuple[str, str, int, str], float] = {}
    for row in raw_rows:
        identity = (
            str(row["split"]),
            str(row["state_id"]),
            int(row["candidate_index"]),
            str(row["source_id"]),
        )
        if identity in expected:
            raise QualificationError(f"duplicate raw-cost re-reduction row: {identity}")
        expected[identity] = float(row["score"])
    expected_count = int(CONTRACT.PREDECESSOR_CANDIDATE_EVIDENCE_BINDING["rows"])
    if len(expected) != expected_count:
        raise QualificationError("raw-cost re-reduction identity coverage drift")
    observed = 0
    for row in stage_rows:
        scores = row.get("scores")
        if not isinstance(scores, Mapping):
            raise QualificationError("Stage-A row score map is missing")
        for source_id in CONTRACT.RAW_COST_REREDUCED_SOURCE_IDS:
            identity = (
                str(row["split"]),
                str(row["state_id"]),
                int(row["candidate_index"]),
                source_id,
            )
            if identity not in expected or scores.get(source_id) != expected[identity]:
                raise QualificationError(f"Stage-A merged raw-cost score drift: {identity}")
            observed += 1
    if observed != expected_count:
        raise QualificationError("Stage-A merged raw-cost score coverage drift")


def _matched_raw_cost_comparisons(
    stage_a: Mapping[str, Mapping[str, Any]],
    stage_b: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Pair plan-aware scores with persisted raw costs under one metric contract."""

    comparison_map = dict(
        CONTRACT.STAGE_POLICY["STAGE_A_TRUE_FUTURE"][
            "matched_raw_cost_comparator_rereduction"
        ]["matched_comparisons"]
    )
    expected_map = {
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE": (
            "RAW_TRUE_FUTURE_GOAL_COSINE"
        ),
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1": "RAW_R1_GOAL_COSINE",
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_RR": "RAW_RR_GOAL_COSINE",
    }
    if comparison_map != expected_map:
        raise QualificationError("matched raw-cost comparison mapping drift")

    metric_keys = (
        "pairwise_accuracy_gain",
        "spearman_gain",
        "kendall_gain",
        "normalized_regret_reduction",
        "best_route_top3_gain",
        "selected_progress_gain_m",
        "all_candidates_contact_selection_delta",
        "all_candidates_nonviable_selection_delta",
    )

    def state_rows(summary: Mapping[str, Any], population: str) -> dict[str, Any]:
        rows = summary["populations"][population]["per_state"]
        return {str(row["state_id"]): row for row in rows}

    def paired_role(
        candidate: Mapping[str, Any], comparator: Mapping[str, Any]
    ) -> dict[str, Any]:
        viability_candidate = state_rows(
            candidate, METRICS.ORACLE_VIABILITY_ADMISSIBLE
        )
        viability_comparator = state_rows(
            comparator, METRICS.ORACLE_VIABILITY_ADMISSIBLE
        )
        all_candidate = state_rows(candidate, METRICS.ALL_CANDIDATES)
        all_comparator = state_rows(comparator, METRICS.ALL_CANDIDATES)
        identities = list(viability_candidate)
        if (
            list(viability_comparator) != identities
            or list(all_candidate) != identities
            or list(all_comparator) != identities
        ):
            raise QualificationError("matched raw-cost state identity/order drift")
        per_state: list[dict[str, Any]] = []
        for state_id in identities:
            left = viability_candidate[state_id]
            right = viability_comparator[state_id]

            def optional_gain(key: str, *, reverse: bool = False) -> float | None:
                if left[key] is None or right[key] is None:
                    return None
                return (
                    float(right[key]) - float(left[key])
                    if reverse
                    else float(left[key]) - float(right[key])
                )

            per_state.append(
                {
                    "state_id": state_id,
                    "family": left["family"],
                    "pairwise_accuracy_gain": optional_gain("pairwise_accuracy"),
                    "spearman_gain": optional_gain("spearman_rho"),
                    "kendall_gain": optional_gain("kendall_tau_b"),
                    "normalized_regret_reduction": optional_gain(
                        "normalized_regret", reverse=True
                    ),
                    "best_route_top3_gain": optional_gain("best_route_top3"),
                    "selected_progress_gain_m": optional_gain(
                        "selected_route_progress_m"
                    ),
                    "all_candidates_contact_selection_delta": int(
                        bool(
                            all_candidate[state_id][
                                "selected_immediate_contact_h1"
                            ]
                        )
                    )
                    - int(
                        bool(
                            all_comparator[state_id][
                                "selected_immediate_contact_h1"
                            ]
                        )
                    ),
                    "all_candidates_nonviable_selection_delta": int(
                        bool(
                            all_candidate[state_id][
                                "selected_nonviable_successor"
                            ]
                        )
                    )
                    - int(
                        bool(
                            all_comparator[state_id][
                                "selected_nonviable_successor"
                            ]
                        )
                    ),
                }
            )
        families = {str(row["state_id"]): str(row["family"]) for row in per_state}
        bootstrap: dict[str, Any] = {}
        for metric in metric_keys:
            available = [row for row in per_state if row[metric] is not None]
            if not available:
                bootstrap[metric] = None
                continue
            values = {
                str(row["state_id"]): float(row[metric]) for row in available
            }
            bootstrap[metric] = OLD_METRICS.paired_state_bootstrap(
                values,
                {state_id: 0.0 for state_id in values},
                {state_id: families[state_id] for state_id in values},
                comparison_id=(
                    f"MATCHED_RAW_COST/{candidate['source_id']}/"
                    f"{comparator['source_id']}/{metric}"
                ),
                draws=METRICS.BOOTSTRAP_DRAWS,
                seed=METRICS.BOOTSTRAP_SEED,
            )
        candidate_viability_aggregate = candidate["populations"][
            METRICS.ORACLE_VIABILITY_ADMISSIBLE
        ]["aggregate"]
        comparator_viability_aggregate = comparator["populations"][
            METRICS.ORACLE_VIABILITY_ADMISSIBLE
        ]["aggregate"]

        def aggregate_gain(key: str, *, reverse: bool = False) -> float | None:
            left = candidate_viability_aggregate.get(key)
            right = comparator_viability_aggregate.get(key)
            if left is None or right is None:
                return None
            return (
                float(right) - float(left)
                if reverse
                else float(left) - float(right)
            )

        all_candidate_aggregate = candidate["populations"][METRICS.ALL_CANDIDATES][
            "aggregate"
        ]
        all_comparator_aggregate = comparator["populations"][METRICS.ALL_CANDIDATES][
            "aggregate"
        ]
        aggregate = {
            "pairwise_accuracy_gain": aggregate_gain("pairwise_accuracy"),
            "spearman_gain": aggregate_gain("spearman_rho"),
            "kendall_gain": aggregate_gain("kendall_tau_b"),
            "normalized_regret_reduction": aggregate_gain(
                "normalized_regret", reverse=True
            ),
            "best_route_top3_gain": aggregate_gain("best_route_top3_rate"),
            "selected_progress_gain_m": aggregate_gain(
                "selected_route_progress_m_mean"
            ),
        }
        aggregate.update(
            {
                "all_candidates_contact_selection_delta": int(
                    all_candidate_aggregate["selected_immediate_contacts_h1"]
                )
                - int(all_comparator_aggregate["selected_immediate_contacts_h1"]),
                "all_candidates_nonviable_selection_delta": int(
                    all_candidate_aggregate["selected_nonviable_successors"]
                )
                - int(all_comparator_aggregate["selected_nonviable_successors"]),
            }
        )
        return {
            "aggregate_deltas": aggregate,
            "per_state_deltas": per_state,
            "descriptive_paired_bootstrap": bootstrap,
        }

    comparisons: dict[str, Any] = {}
    for candidate_id, comparator_id in comparison_map.items():
        if candidate_id.endswith("_TRUE"):
            candidate_by_role = {
                role: stage_a[role][candidate_id] for role in SPLIT_ROLES
            }
        elif stage_b is None:
            comparisons[candidate_id] = {
                "comparator_id": comparator_id,
                "status": "NOT_RUN_STAGE_B_NOT_AUTHORISED",
                "classification_gate": False,
            }
            continue
        else:
            candidate_by_role = {
                role: stage_b["summaries"][role][candidate_id]
                for role in SPLIT_ROLES
            }
        comparisons[candidate_id] = {
            "comparator_id": comparator_id,
            "status": "COMPLETE",
            "classification_gate": False,
            "by_role": {
                role: paired_role(
                    candidate_by_role[role], stage_a[role][comparator_id]
                )
                for role in SPLIT_ROLES
            },
        }
    return {
        "schema": "plan_aware_matched_raw_cost_comparisons_v1",
        "comparison_mapping": comparison_map,
        "metric_ids": list(metric_keys),
        "classification_gate": False,
        "comparisons": comparisons,
    }


def _summary_metric(summary: Mapping[str, Any], key: str) -> Any:
    return summary["populations"][METRICS.ORACLE_VIABILITY_ADMISSIBLE]["aggregate"][key]


def _stage_a_decisions(
    summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    latent_key = "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE"
    deranged_key = "WITHIN_STATE_FUTURE_LATENT_DERANGEMENT_TRUE"
    derangement = METRICS.evaluate_future_derangement_materiality(
        summaries[latent_key], summaries[deranged_key]
    )
    true_gate = METRICS.evaluate_true_future_gate(summaries[latent_key], derangement)
    if true_gate["pass"]:
        incremental = METRICS.evaluate_incremental_value(
            summaries[latent_key],
            kinematic_source=summaries["KINEMATIC_ROUTE_BASELINE"],
            no_latent_source=summaries["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"],
        )
    else:
        incremental = {
            "schema": "plan_aware_incremental_route_value_gate_v1",
            "thresholds": dict(METRICS.INCREMENTAL_ROUTE_VALUE_THRESHOLDS),
            "comparisons": {},
            "pass": False,
            "classification": (
                "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE_NOT_EVALUATED"
            ),
            "status": "NOT_EVALUATED_TRUE_GATE_FAILED",
            "evaluated": False,
            "reason": "TRUE_FUTURE_GATE_FAILED",
        }
    return {
        "derangement": derangement,
        "true_future_gate": true_gate,
        "true_incremental_value": incremental,
        "paired_true_minus_kinematic": METRICS.paired_principal_bootstrap(
            summaries[latent_key], summaries["KINEMATIC_ROUTE_BASELINE"]
        ),
        "paired_true_minus_no_latent": METRICS.paired_principal_bootstrap(
            summaries[latent_key], summaries["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"]
        ),
    }


def _manifest(root: Path, *, excluded: Sequence[str] = ()) -> list[dict[str, Any]]:
    ignored = set(excluded)
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = str(path.relative_to(root))
        if relative in ignored or Path(relative).name.startswith("."):
            continue
        rows.append({**binding(path, relative_to=root)})
    return rows


def _active_experiment_processes() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    own = os.getpid()
    needles = (
        Path(__file__).name,
        "materialize_plan_aware_proprio_predictor_substitution_v1.py",
    )
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == own:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if any(needle in command for needle in needles) and any(
            phase in command
            for phase in (
                "execute",
                "run-stage-b",
                "context-state",
                "predict-source",
                "run-stage-c",
            )
        ):
            output.append({"pid": int(entry.name), "command": command})
    return output


def _primary_and_secondaries(
    stage_a: Mapping[str, Any],
    stage_b: Mapping[str, Any] | None,
    stage_c: Mapping[str, Any] | None,
) -> tuple[str, list[str], str]:
    true_pass = bool(stage_a["true_future_gate"]["pass"])
    incremental_pass = bool(stage_a["true_incremental_value"]["pass"])
    if not true_pass:
        primary = "PLAN_AWARE_JEPA_COST_NO_SIGNAL"
        secondaries = ["ENCODER_ROUTE_INFORMATION_INSUFFICIENT"]
    elif not incremental_pass:
        primary = "KINEMATIC_BASELINE_DOMINANT"
        secondaries = ["TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL"]
        if stage_a["derangement"]["pass"]:
            secondaries.append("CANDIDATE_SPECIFIC_LATENT_ROUTE_INFORMATION_USED")
    elif stage_b is None:
        raise QualificationError("true-future and incremental gates passed without Stage B")
    else:
        try:
            primary_decision = METRICS.classify_primary(
                true_gate=stage_a["true_future_gate"],
                predicted_gate=stage_b["predicted_gate"],
                true_incremental_gate=stage_a["true_incremental_value"],
                all_predicted_substitutions_fail_materially=bool(
                    stage_b["all_predicted_substitutions_fail_materially"]
                ),
            )
        except (KeyError, METRICS.PlanAwareMetricsError) as exc:
            raise QualificationError(
                "frozen primary classification is unresolved by Stage-B evidence"
            ) from exc
        primary = str(primary_decision["classification"])
        if primary == "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL":
            secondaries = [
                "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL",
                "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE",
                "CANDIDATE_SPECIFIC_LATENT_ROUTE_INFORMATION_USED",
                "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL",
                "ROLLOUT_ROUTE_INFORMATION_SIGNAL",
            ]
            if bool(stage_b["incremental_over_kinematics"]["pass"]):
                secondaries.append("JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS")
        elif primary == "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_ROUTE_NO_GO":
            secondaries = [
                "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL",
                "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE",
                "CANDIDATE_SPECIFIC_LATENT_ROUTE_INFORMATION_USED",
                "PREDICTOR_ROUTE_GEOMETRY_LOSS",
            ]
        else:
            raise QualificationError(
                f"unexpected Stage-B primary classification {primary!r}"
            )
    if true_pass:
        secondaries.append("WRONG_PLANNING_READOUT")
    if stage_b is not None:
        if bool(stage_b["proprioception_gate"]["pass"]):
            secondaries.append("PROPRIOCEPTIVE_ROUTE_CONTRIBUTION")
        else:
            secondaries.append("PROPRIOCEPTIVE_ROUTE_CONTRIBUTION_NOT_SUPPORTED")
        if stage_c is not None:
            secondaries.append(str(stage_c["classification"]))
            dependence = str(stage_c["dependence_attribution"])
            if dependence != stage_c["classification"]:
                secondaries.append(dependence)
    secondaries = list(dict.fromkeys(secondaries))
    if any(value not in CONTRACT.SECONDARY_CLASSIFICATIONS for value in secondaries):
        raise QualificationError(f"unknown secondary classification: {secondaries}")
    next_experiment = CONTRACT.next_experiment_for_primary(primary)
    return primary, secondaries, next_experiment


def _report_markdown(result: Mapping[str, Any]) -> str:
    def render_metric(value: Any) -> str:
        if value is None:
            return "n/a"
        numeric = float(value)
        if not math.isfinite(numeric):
            raise QualificationError("report metric is non-finite")
        return f"{numeric:.6f}"

    def render_counted_metric(value: Any, count: Any) -> str:
        return f"{render_metric(value)} ({int(count)})"

    stage_a = result["metrics"]["stage_a_decisions"]
    heldout = result["metrics"]["stage_a"][HELDOUT]
    sources = (
        "KINEMATIC_ROUTE_BASELINE",
        "KINEMATIC_PLUS_NO_LATENT_RESIDUAL",
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE",
        "WITHIN_STATE_FUTURE_LATENT_DERANGEMENT_TRUE",
        *CONTRACT.RAW_COST_REREDUCED_SOURCE_IDS,
    )
    lines = [
        "# Plan-aware monotone JEPA route cost V1 result",
        "",
        "This is a development-only, single-seed local-waypoint route-cost qualification. It is not a deployment-safety or closed-loop-navigation result.",
        "",
        "## Decision",
        "",
        f"- Primary: `{result['primary_classification']}`.",
        f"- Secondary: {', '.join(f'`{value}`' for value in result['secondary_classifications']) or 'none'}.",
        f"- Next experiment: `{result['next_experiment']}` (specified, not run).",
        f"- Validated prior smoke-only failure archives: {len(result.get('prior_smoke_failure_custody', []))}; reused files: 0.",
        "",
        "## Exact next-experiment specification",
        "",
        "```json",
        json.dumps(
            result["next_experiment_specification"],
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        "```",
        "",
        "## Fit-state optimizer conditioning",
        "",
        f"- Fit states total: {int(result['metrics']['fit_optimization']['fit_states_total'])}.",
        f"- Optimizer-contributing states (at least two oracle-admissible candidates): {int(result['metrics']['fit_optimization']['fit_states_contributing'])}.",
        f"- Skipped with zero admissible candidates: {int(result['metrics']['fit_optimization']['fit_states_skipped_zero_admissible'])}.",
        f"- Skipped singleton-admissible states: {int(result['metrics']['fit_optimization']['fit_states_skipped_singleton_admissible'])}.",
        f"- Per-epoch loss denominator: {int(result['metrics']['fit_optimization']['epoch_average_denominator'])} contributing states.",
        "",
        "## Heldout oracle-viability route metrics",
        "",
        "| Source | Pairwise | Spearman | Regret | Top-3 | Oracle-progress fraction | Selected progress (m) | Selected heading (rad) | Selected route utility (defined n) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for source in sources:
        summary = heldout[source]
        lines.append(
            "| "
            + source
            + " | "
            + " | ".join(
                render_metric(_summary_metric(summary, key))
                for key in (
                    "pairwise_accuracy",
                    "spearman_rho",
                    "normalized_regret",
                    "best_route_top3_rate",
                    "selected_progress_ratio",
                    "selected_route_progress_m_mean",
                    "selected_heading_progress_rad_mean",
                )
            )
            + " | "
            + render_counted_metric(
                _summary_metric(summary, "selected_combined_route_utility_mean"),
                _summary_metric(summary, "selected_combined_route_utility_count"),
            )
            + " |"
        )
    matched_raw = result["metrics"]["stage_a_raw_cost_matched_comparisons"]
    lines.extend(
        [
            "",
            "## Matched plan-aware versus raw-cost comparisons",
            "",
            "These paired successor-metric reductions are descriptive and non-gating. Positive route deltas favor the plan-aware score; adverse-selection deltas are plan-aware minus raw and therefore positive means more adverse selections.",
            "",
            "| Plan-aware source | Raw source | Status | Pairwise | Spearman | Kendall | Regret reduction | Top-3 | Progress (m) | Contact delta | Nonviable delta |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for candidate_id, comparison in matched_raw["comparisons"].items():
        status = str(comparison["status"])
        if status == "COMPLETE":
            deltas = comparison["by_role"][HELDOUT]["aggregate_deltas"]
            values = [
                render_metric(deltas[key])
                for key in (
                    "pairwise_accuracy_gain",
                    "spearman_gain",
                    "kendall_gain",
                    "normalized_regret_reduction",
                    "best_route_top3_gain",
                    "selected_progress_gain_m",
                    "all_candidates_contact_selection_delta",
                    "all_candidates_nonviable_selection_delta",
                )
            ]
        else:
            values = ["n/a"] * 8
        lines.append(
            "| "
            + candidate_id
            + " | "
            + str(comparison["comparator_id"])
            + " | "
            + status
            + " | "
            + " | ".join(values)
            + " |"
        )
    lines.extend(
        [
            "",
            "## Frozen gates",
            "",
            f"- Candidate-specific future-latent derangement material: `{stage_a['derangement']['pass']}`.",
            "- Candidate-future derangement damage (matched minus deranged except regret worsening): "
            + ", ".join(
                f"`{key}={render_metric(stage_a['derangement']['damage'][key])}`"
                for key in (
                    "pairwise_accuracy_loss",
                    "selected_progress_m_loss",
                    "selected_progress_ratio_loss",
                    "normalized_regret_worsening",
                    "best_route_top3_loss",
                )
            )
            + ".",
            f"- True-future plan-aware gate: `{stage_a['true_future_gate']['pass']}`.",
            f"- Incremental value over both kinematics and the matched no-latent residual: `{stage_a['true_incremental_value']['pass']}`.",
            f"- Stage B: `{result['stage_execution']['stage_b']}`.",
            f"- Stage C: `{result['stage_execution']['stage_c']}`.",
            "- Execution-only correction amendment: `"
            + str(
                result["stage_execution"]["execution_correction_custody"]
                ["amendment"]["sha256"]
            )
            + "`; failed-archive files reused: `0`; pre-Stage-B scientific "
            "replay equality: `PASS`.",
            "",
            "## Preserved predecessor conclusions",
            "",
            "These classification IDs are scoped to the predecessor raw-cost assay; in particular, `JEPA_INCREMENTAL_ROUTE_VALUE_OVER_KINEMATICS_NOT_SUPPORTED` is not a conclusion about this plan-aware assay.",
            "",
            *[f"- `{fact}`" for fact in result["predecessor_fact_authority"]],
            "",
            *[f"- {statement}" for statement in result["predecessor_narrative_authority"]],
            "",
            "## Historical raw latent-goal comparators (no recomputation)",
            "",
            "These heldout values are copied from the byte-bound predecessor result as historical context only. Their predecessor metric definition is not comparable to the successor population-conditioned Borda metrics above; raw cosine was not rerun. The three `RAW_*_GOAL_COSINE` rows in the Stage-A table are a post-barrier read-only re-reduction of persisted predecessor `-cost_h3` candidate evidence under the successor metric contract.",
            "",
            f"Successor-metric candidate-evidence binding: `{result['metrics']['stage_a_raw_cost_rereduced']['frozen_source_binding']['path']}` (`{result['metrics']['stage_a_raw_cost_rereduced']['frozen_source_binding']['sha256']}`; {int(result['metrics']['stage_a_raw_cost_rereduced']['source_rows'])} rows).",
            "",
            f"Source binding: `{result['metrics']['historical_raw_latent_goal_cosine_comparators']['source_binding']['path']}` (`{result['metrics']['historical_raw_latent_goal_cosine_comparators']['source_binding']['sha256']}`).",
            "",
            "| Historical source | Pairwise | Spearman | Regret | Top-3 | Oracle-progress fraction |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                "| "
                + source
                + " | "
                + " | ".join(
                    render_metric(summary['populations'][METRICS.ORACLE_VIABILITY_ADMISSIBLE]['aggregate'][key])
                    for key in (
                        "pairwise_accuracy",
                        "spearman_rho",
                        "normalized_regret",
                        "best_route_top3_rate",
                        "selected_progress_ratio",
                    )
                )
                + " |"
                for source, summary in result["metrics"][
                    "historical_raw_latent_goal_cosine_comparators"
                ]["comparators"].items()
            ],
            "",
            "## Claims boundary",
            "",
            "This experiment uses the frozen local waypoint directly and makes no JEPA-safety claim.",
            "",
            "`REQUIREMENTS_ACQUISITION_REQUIRED`, `PROTECTED_CONTACT_SCOPE_REQUIREMENTS_UNRESOLVED`, `SIMULATED_CONTACT_PROXY_SCOPE_ONLY`, `REPLANNING_INTERFACE_UNRESOLVED`, and `GO2_PLATFORM_STOPPING_MODE_PARITY_PENDING` remain separate and unchanged.",
            "",
            "No predictor, safety/contact/occupancy/completion/place/nonviability model was trained; no fresh panel, sensor layout, scope change, memory, routing, beacon capture, or closed-loop navigation was executed.",
            "",
        ]
    )
    stage_b = result["metrics"].get("stage_b")
    if stage_b is not None:
        stage_b_heldout = stage_b["summaries"][HELDOUT]
        lines.extend(
            [
                "## Frozen-predictor substitution (Stage B)",
                "",
                "| Future source | Pairwise | Spearman | Regret | Top-3 | Oracle-progress fraction | Selected progress (m) | Selected heading (rad) | Selected route utility (defined n) | Contact | Nonviable | Stuck |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for source in ("R1", "RR", "P1", "PR"):
            summary = stage_b_heldout[
                f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"
            ]
            aggregate = summary["populations"][
                METRICS.ORACLE_VIABILITY_ADMISSIBLE
            ]["aggregate"]
            all_candidates = summary["populations"][METRICS.ALL_CANDIDATES][
                "aggregate"
            ]
            lines.append(
                "| "
                + source
                + " | "
                + " | ".join(
                    [
                        render_metric(aggregate['pairwise_accuracy']),
                        render_metric(aggregate['spearman_rho']),
                        render_metric(aggregate['normalized_regret']),
                        render_metric(aggregate['best_route_top3_rate']),
                        render_metric(aggregate['selected_progress_ratio']),
                        render_metric(aggregate['selected_route_progress_m_mean']),
                        render_metric(aggregate['selected_heading_progress_rad_mean']),
                        render_counted_metric(
                            aggregate['selected_combined_route_utility_mean'],
                            aggregate['selected_combined_route_utility_count'],
                        ),
                        str(int(all_candidates["selected_immediate_contacts_h1"])),
                        str(int(all_candidates["selected_nonviable_successors"])),
                        str(int(all_candidates["selected_stuck"])),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                f"- Two-step predicted-route gate: `{stage_b['predicted_gate']['classification']}` (`{stage_b['predicted_gate']['pass']}`).",
                f"- RR incremental value over kinematics: `{stage_b['incremental_over_kinematics']['classification']}` (`{stage_b['incremental_over_kinematics']['pass']}`).",
                f"- Proprioceptive route contribution: `{stage_b['proprioception_gate']['classification']}` (`{stage_b['proprioception_gate']['pass']}`).",
                *[
                    "- "
                    + contrast
                    + " selected-candidate changes under oracle viability: "
                    + str(
                        value["populations"][
                            METRICS.ORACLE_VIABILITY_ADMISSIBLE
                        ]["changed_count"]
                    )
                    + "/"
                    + str(
                        value["populations"][
                            METRICS.ORACLE_VIABILITY_ADMISSIBLE
                        ]["state_count"]
                    )
                    + "."
                    for contrast, value in stage_b[
                        "selected_candidate_changes"
                    ].items()
                ],
                "- Frozen rollout/proprioception contrasts and state-paired bootstrap intervals are persisted in `aggregates/metrics.json`.",
                "",
            ]
        )

    stage_c = result["metrics"].get("stage_c")
    if stage_c is not None:
        stage_c_heldout = stage_c["summaries"][HELDOUT]
        lines.extend(
            [
                "## Strict input-substitution diagnostic (Stage C)",
                "",
                "| PR input condition | Pairwise | Regret | Top-3 | Oracle-progress fraction | Selected progress (m) | Selected heading (rad) | Selected route utility (defined n) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for source in STAGE_C_SOURCE_IDS:
            summary = stage_c_heldout[
                f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"
            ]
            aggregate = summary["populations"][
                METRICS.ORACLE_VIABILITY_ADMISSIBLE
            ]["aggregate"]
            lines.append(
                "| "
                + source
                + " | "
                + " | ".join(
                    render_metric(aggregate[key])
                    for key in (
                        "pairwise_accuracy",
                        "normalized_regret",
                        "best_route_top3_rate",
                        "selected_progress_ratio",
                        "selected_route_progress_m_mean",
                        "selected_heading_progress_rad_mean",
                    )
                )
                + " | "
                + render_counted_metric(
                    aggregate['selected_combined_route_utility_mean'],
                    aggregate['selected_combined_route_utility_count'],
                )
                + " |"
            )
        lines.extend(
            [
                "",
                f"- Strict substitution result: `{stage_c['classification']}`.",
                f"- Dependence attribution: `{stage_c['dependence_attribution']}`.",
                f"- Candidate-action sensitivity retained: `{stage_c['candidate_action_sensitivity']['passed']}`.",
                f"- Control-history-only explanation excluded: `{stage_c['control_only_explanation_excluded']}`.",
                f"- Frozen occupancy probe: `{stage_c['occupancy_probe']['status']}`; it was not executed.",
                *[
                    "- "
                    + source
                    + " heldout candidate-level absolute route-score change mean: "
                    + render_metric(stage_c['route_score_changes']['by_source'][source]['by_role'][HELDOUT]['absolute_mean'])
                    + "."
                    for source in STAGE_C_SOURCE_IDS
                ],
                "- Direct H1-H3 fidelity, candidate-matched route-score changes, route-metric changes, and all input-derangement mappings are persisted in the result bundle.",
                "",
            ]
        )
    lines.extend(
        [
            "## Descriptive all-candidate adverse-outcome tendencies",
            "",
            "Each accuracy is the within-state adverse-vs-nonadverse cross-group credit for giving the nonadverse candidate a higher score (ties receive half credit); `(n)` is the pair count. These labels were not training targets or safety gates.",
            "",
            "| Stage | Route-score condition | Contact accuracy (n) | Nonviable accuracy (n) | Stuck accuracy (n) | Selected contact/nonviable/stuck |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )

    def adverse_cell(summary: Mapping[str, Any], outcome: str) -> str:
        value = summary["descriptive_adverse_downranking"]["outcomes"][outcome][
            "overall"
        ]
        accuracy = value["pairwise_accuracy"]
        rendered = "n/a" if accuracy is None else f"{float(accuracy):.6f}"
        return f"{rendered} ({int(value['pair_count'])})"

    stage_summaries: list[tuple[str, Mapping[str, Any]]] = [
        ("A", heldout)
    ]
    if stage_b is not None:
        stage_summaries.append(("B", stage_b["summaries"][HELDOUT]))
    if stage_c is not None:
        stage_summaries.append(("C", stage_c["summaries"][HELDOUT]))
    for stage_id, summaries in stage_summaries:
        for source, summary in summaries.items():
            aggregate = summary["populations"][METRICS.ALL_CANDIDATES]["aggregate"]
            lines.append(
                "| "
                + stage_id
                + " | "
                + source
                + " | "
                + " | ".join(
                    [
                        adverse_cell(summary, "immediate_contact"),
                        adverse_cell(summary, "successor_nonviable"),
                        adverse_cell(summary, "stuck"),
                        "/".join(
                            str(int(aggregate[key]))
                            for key in (
                                "selected_immediate_contacts_h1",
                                "selected_nonviable_successors",
                                "selected_stuck",
                            )
                        ),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "These diagnostics do not establish deployment safety or visual wall avoidance.",
            "",
        ]
    )
    return "\n".join(lines)


def _rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise QualificationError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise QualificationError(f"non-object JSONL row: {path}:{line_number}")
            rows.append(value)
    return rows


def _replay_stage_a_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_role: dict[str, list[Mapping[str, Any]]] = {role: [] for role in SPLIT_ROLES}
    for row in rows:
        _validate_population_membership(row, label="Stage-A")
        role = str(row["split"])
        if role not in by_role:
            raise QualificationError(f"unknown Stage-A ledger role {role}")
        by_role[role].append(row)
    reproduced: dict[str, dict[str, Any]] = {}
    for role in SPLIT_ROLES:
        role_rows = by_role[role]
        expected = CONTRACT.ROLE_ROW_COUNTS[role]
        if len(role_rows) != expected:
            raise QualificationError(f"Stage-A {role} row count {len(role_rows)} != {expected}")
        candidates: dict[str, list[dict[str, Any]]] = {}
        score_maps: dict[str, dict[str, np.ndarray]] = {}
        for row in role_rows:
            state_id = str(row["state_id"])
            candidate = int(row["candidate_index"])
            if not 0 <= candidate < CANDIDATE_COUNT:
                raise QualificationError("Stage-A candidate identity drift")
            candidates.setdefault(state_id, []).append(
                {
                    "state_id": state_id,
                    "family": row["family"],
                    "role": role,
                    "candidate_index": candidate,
                    "p_d": row["p_d"],
                    "p_theta": row["p_theta"],
                    "completed": row["completed"],
                    "stuck": row["stuck"],
                    "immediate_contact_h1": row["immediate_contact_h1"],
                    "descriptive_contact_h2": row["descriptive_contact_h2"],
                    "descriptive_contact_h3": row["descriptive_contact_h3"],
                    "successor_safe_action_count": row["successor_safe_action_count"],
                    "successor_viable": row["successor_viable"],
                    "oracle_viability_admissible": row["oracle_viability_admissible"],
                }
            )
            for source, value in row["scores"].items():
                score_maps.setdefault(str(source), {}).setdefault(
                    state_id, np.full(CANDIDATE_COUNT, np.nan, dtype=np.float64)
                )[candidate] = float(value)
        for state_id, state_rows in candidates.items():
            state_rows.sort(key=lambda value: int(value["candidate_index"]))
            if [int(value["candidate_index"]) for value in state_rows] != list(
                range(CANDIDATE_COUNT)
            ):
                raise QualificationError(f"Stage-A candidate set drift: {state_id}")
        for source, states in score_maps.items():
            if any(not np.isfinite(values).all() for values in states.values()):
                raise QualificationError(f"Stage-A score holes: {role}:{source}")
        reproduced[role] = {
            source: METRICS.summarize_scores(candidates, states, source_id=source)
            for source, states in score_maps.items()
        }
    return reproduced


def _replay_conditional_metrics(
    rows: Sequence[Mapping[str, Any]], *, expected_sources: Sequence[str]
) -> dict[str, dict[str, Any]]:
    sources = tuple(expected_sources)
    expected_rows = STATE_COUNT * CANDIDATE_COUNT * len(sources)
    if len(rows) != expected_rows:
        raise QualificationError(
            f"conditional replay row count {len(rows)} != {expected_rows}"
        )
    per_role_source: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        role: {source: [] for source in sources} for role in SPLIT_ROLES
    }
    for row in rows:
        _validate_population_membership(row, label="conditional")
        role = str(row["split"])
        source = str(row["latent_source"])
        if role not in per_role_source or source not in per_role_source[role]:
            raise QualificationError(f"unknown conditional identity: {role}:{source}")
        per_role_source[role][source].append(row)

    reproduced: dict[str, dict[str, Any]] = {}
    for role in SPLIT_ROLES:
        merged: dict[str, Any] = {}
        for source in sources:
            source_rows = per_role_source[role][source]
            if len(source_rows) != CONTRACT.ROLE_ROW_COUNTS[role]:
                raise QualificationError(
                    f"conditional {role}:{source} cardinality drift"
                )
            candidates: dict[str, list[dict[str, Any]]] = {}
            score_maps: dict[str, dict[str, np.ndarray]] = {}
            for row in source_rows:
                state_id = str(row["state_id"])
                candidate = int(row["candidate_index"])
                candidates.setdefault(state_id, []).append(
                    {
                        "state_id": state_id,
                        "family": row["family"],
                        "role": role,
                        "candidate_index": candidate,
                        "p_d": row["p_d"],
                        "p_theta": row["p_theta"],
                        "completed": row["completed"],
                        "stuck": row["stuck"],
                        "immediate_contact_h1": row["immediate_contact_h1"],
                        "descriptive_contact_h2": row["descriptive_contact_h2"],
                        "descriptive_contact_h3": row["descriptive_contact_h3"],
                        "successor_safe_action_count": row[
                            "successor_safe_action_count"
                        ],
                        "successor_viable": row["successor_viable"],
                        "oracle_viability_admissible": row[
                            "oracle_viability_admissible"
                        ],
                    }
                )
                route_scores = row.get("route_scores")
                if not isinstance(route_scores, Mapping):
                    raise QualificationError("conditional row lacks route scores")
                for condition, score in route_scores.items():
                    score_maps.setdefault(str(condition), {}).setdefault(
                        state_id,
                        np.full(CANDIDATE_COUNT, np.nan, dtype=np.float64),
                    )[candidate] = float(score)
            for state_id, state_rows in candidates.items():
                state_rows.sort(key=lambda value: int(value["candidate_index"]))
                if [int(value["candidate_index"]) for value in state_rows] != list(
                    range(CANDIDATE_COUNT)
                ):
                    raise QualificationError(
                        f"conditional candidate identity drift: {role}:{source}:{state_id}"
                    )
            if any(
                not np.isfinite(values).all()
                for states in score_maps.values()
                for values in states.values()
            ):
                raise QualificationError(f"conditional score holes: {role}:{source}")
            summaries = {
                condition: METRICS.summarize_scores(
                    candidates, values, source_id=condition
                )
                for condition, values in score_maps.items()
            }
            for condition, summary in summaries.items():
                if condition in merged and merged[condition] != summary:
                    raise QualificationError(
                        f"conditional repeated baseline drift: {role}:{condition}"
                    )
                merged[condition] = summary
        reproduced[role] = merged
    return reproduced


def _validate_persisted_latent_bindings(
    rows: Sequence[Mapping[str, Any]],
    *,
    tensor_rows: Mapping[
        tuple[str, str, int | None, int | None], Mapping[str, Any]
    ],
) -> None:
    """Bind every scored row to the exact current and H1-H3 tensor records."""

    for row in rows:
        source = str(row["latent_source"])
        donor = row.get("future_derangement_donor_candidate")
        expected = _candidate_latent_bindings(
            tensor_rows,
            state_id=str(row["state_id"]),
            candidate=int(row["candidate_index"]),
            source=source,
            deranged_candidate=None if donor is None else int(donor),
        )
        if row.get("latent_artifact_bindings") != expected:
            raise QualificationError(
                "row latent-artifact binding drift: "
                f"{source}:{row['state_id']}:{row['candidate_index']}"
            )


def _validate_action_sensitivity_rows(
    rows: Sequence[Mapping[str, Any]], evidence: Mapping[str, Any]
) -> None:
    expected = 4 * STATE_COUNT * len(HORIZONS)
    if len(rows) != expected:
        raise QualificationError(
            f"Stage-C action sensitivity rows {len(rows)} != {expected}"
        )
    failed = [
        {
            "latent_source": row["latent_source"],
            "state_id": row["state_id"],
            "horizon": row["horizon"],
        }
        for row in rows
        if not bool(row["candidate_action_sensitivity_present"])
    ]
    raw = evidence.get("raw_evidence")
    if (
        not isinstance(raw, Mapping)
        or raw.get("rows") != expected
        or raw.get("failed_rows") != failed
        or evidence.get("passed") is not (not failed)
    ):
        raise QualificationError("Stage-C action-sensitivity evidence drift")


def _validate_manifest(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        path = root / str(row["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise QualificationError(f"persistence manifest drift: {row['path']}")


def _datasets_from_stage_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    datasets: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: {} for role in SPLIT_ROLES
    }
    for source in rows:
        role = str(source["split"])
        if role not in datasets:
            raise QualificationError(f"unknown persisted split role {role!r}")
        datasets[role].setdefault(str(source["state_id"]), []).append(dict(source))
    for role in SPLIT_ROLES:
        if len(datasets[role]) != CONTRACT.ROLE_STATE_COUNTS[role]:
            raise QualificationError(f"persisted state cardinality drift: {role}")
        for state_id, state_rows in datasets[role].items():
            state_rows.sort(key=lambda row: int(row["candidate_index"]))
            if [int(row["candidate_index"]) for row in state_rows] != list(
                range(CANDIDATE_COUNT)
            ):
                raise QualificationError(
                    f"persisted target candidate identity drift: {state_id}"
                )
    return datasets


def _validate_persisted_action_plans(rows: Sequence[Mapping[str, Any]]) -> None:
    for row in rows:
        plan = np.asarray(row.get("predictor_candidate_action_plan_3x10"))
        if plan.shape != (3, 10) or not np.isfinite(plan.astype(np.float64)).all():
            raise QualificationError(
                "persisted predictor candidate-action plan drift: "
                f"{row.get('latent_source')}:{row.get('state_id')}:"
                f"{row.get('candidate_index')}"
            )


def _validate_training_checkpoint_custody(
    output_root: Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation_contract = load_json(
        output_root / "receipts/evaluation_contract.json"
    )
    training = load_json(output_root / "receipts/training.json")
    evaluation = load_json(output_root / "receipts/evaluation.json")
    for label, receipt in (
        ("evaluation contract", evaluation_contract),
        ("training", training),
        ("evaluation", evaluation),
    ):
        if receipt.get("content_digest") != content_digest(receipt):
            raise QualificationError(f"{label} receipt self-digest drift")
    fit_optimization = training.get("fit_optimization")
    if not isinstance(fit_optimization, Mapping):
        raise QualificationError("training fit-optimization receipt is absent")
    _validate_fit_optimization_summary(fit_optimization, expected_total=32)
    for history in training.get("training_history", {}).values():
        _validate_training_history(history, fit_optimization)
    checkpoints = result["checkpoint_bindings"]["trained_route_rankers"]
    expected_conditions = set(CONTRACT.CHECKPOINT_SEED_METADATA)
    if (
        set(checkpoints) != expected_conditions
        or evaluation_contract.get("checkpoint_bindings") != checkpoints
        or training.get("checkpoint_bindings") != checkpoints
        or training.get("checkpoint_seed_metadata")
        != CONTRACT.CHECKPOINT_SEED_METADATA
        or training.get("final_epoch_only") is not True
        or evaluation_contract.get("final_epoch_only") is not True
        or evaluation_contract.get("heldout_opened_before_publication") is not False
    ):
        raise QualificationError("final ranker checkpoint custody drift")
    for condition, record in checkpoints.items():
        checkpoint = _load_checkpoint_receipt(
            condition, record, output_root=output_root
        )
        if (
            checkpoint.get("fit_optimization") != training.get("fit_optimization")
            or checkpoint.get("training_history")
            != training.get("training_history", {}).get(condition)
        ):
            raise QualificationError(
                f"route-ranker fit-conditioning custody drift: {condition}"
            )

    epoch_rows = _rows_from_jsonl(output_root / "ledgers/training_epochs.jsonl")
    expected_epoch_rows: list[dict[str, Any]] = []
    for condition, history in training["training_history"].items():
        for row in history:
            expected_epoch_rows.append(
                {
                    "schema": "plan_aware_training_epoch_row_v1",
                    "condition": condition,
                    **copy.deepcopy(dict(row)),
                }
            )
    if epoch_rows != expected_epoch_rows or len(epoch_rows) != 120:
        raise QualificationError("training epoch ledger reproduction drift")
    if (
        evaluation.get("heldout_opened_after_checkpoint_publication") is not True
        or evaluation.get("heldout_opened_after_evaluation_contract") is not True
        or evaluation.get("calibration_used_for_model_selection") is not False
        or evaluation.get("predictor_training_steps") != 0
        or evaluation.get("raw_goal_cosine_executions") != 0
        or evaluation.get("pass") is not True
    ):
        raise QualificationError("sealed evaluation lifecycle receipt drift")
    return training


def _validate_conditional_child_environment_preflight(
    output_root: Path, preexecution: Mapping[str, Any]
) -> dict[str, Any]:
    helper = _conditional_helper()
    path = output_root / "receipts/conditional_child_environment_preflight.json"
    value = load_json(path)
    expected_binding = {
        **binding(path, relative_to=output_root),
        "content_digest": value.get("content_digest"),
    }
    if (
        value.get("content_digest") != content_digest(value)
        or preexecution.get("conditional_child_environment_preflight")
        != expected_binding
        or value.get("schema")
        != (
            "plan_aware_monotone_jepa_cost_v1."
            "conditional_child_environment_preflight.v1"
        )
        or value.get("experiment_id") != CONTRACT.EXPERIMENT_ID
        or value.get("pass") is not True
        or any(
            value.get(key) != 0
            for key in (
                "fit_outcome_rows_opened",
                "calibration_rows_opened",
                "heldout_rows_opened",
                "tensor_rows_opened",
                "training_steps",
            )
        )
    ):
        raise QualificationError("conditional child-environment preflight drift")
    for label, interpreter, require_genesis in (
        ("cpu_child", helper.CPU_INTERPRETER, True),
        ("gpu_child", helper.GPU_INTERPRETER, False),
    ):
        probe = value.get(label)
        expected_probe = CONTRACT.EXECUTION_CORRECTION_ENVIRONMENT_PROBE[
            "required_preflight"
        ][label]
        if (
            not isinstance(probe, Mapping)
            or probe.get("pass") is not True
            or probe.get("sentinel_available") is not True
            or probe.get("environment_contract")
            != helper.child_environment_contract(interpreter)
        ):
            raise QualificationError(
                f"conditional {label} import-preflight custody drift"
            )
        venv_root = Path(interpreter).absolute().parent.parent.resolve()
        expected_modules = {
            "typing_extensions": expected_probe["typing_extensions"],
            "pydantic_core_or_null": (
                expected_probe["pydantic_core"] if require_genesis else None
            ),
            "genesis_or_null": expected_probe["genesis"] if require_genesis else None,
            "torch_or_null": expected_probe["torch"] if not require_genesis else None,
        }
        for field, expected_module in expected_modules.items():
            observed = probe.get(field)
            if expected_module is None:
                if observed is not None:
                    raise QualificationError(
                        f"conditional {label} unexpected {field} binding"
                    )
                continue
            exact = {
                "path": str(expected_module["path"]),
                "resolved_path": str(Path(str(expected_module["path"])).resolve()),
                "sha256": str(expected_module["sha256"]),
                "bytes": int(expected_module["bytes"]),
            }
            if "version" in expected_module:
                exact["version"] = str(expected_module["version"])
            if "Sentinel_present" in expected_module:
                exact["Sentinel_present"] = True
            if observed != exact or not Path(exact["resolved_path"]).is_relative_to(
                venv_root
            ):
                raise QualificationError(
                    f"conditional {label} exact {field} binding drift"
                )
    return value


def _validate_execution_correction_output_custody(
    output_root: Path,
    *,
    preexecution: Mapping[str, Any],
    persistence: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconcile the progressive amendment custody without rerunning science."""

    try:
        archive = CONTRACT.validate_execution_correction_archive()
        CONTRACT.validate_base_scientific_authorities(ROOT)
        CONTRACT.load_and_validate_execution_correction_amendment(
            _tracked_path(CONTRACT.TRACKED_EXECUTION_CORRECTION_AMENDMENT_PATH)
        )
        CONTRACT.load_and_validate_execution_correction_output_schema(
            _tracked_path(CONTRACT.TRACKED_EXECUTION_CORRECTION_OUTPUT_SCHEMA_PATH)
        )
        CONTRACT.load_and_validate_execution_correction_fixture(
            _tracked_path(CONTRACT.TRACKED_EXECUTION_CORRECTION_FIXTURE_PATH)
        )
        frozen_correction_closure = (
            CONTRACT.load_and_validate_execution_correction_source_closure(
            _tracked_path(CONTRACT.TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH)
            )
        )
    except CONTRACT.ContractError as exc:
        raise QualificationError(str(exc)) from exc
    preflight_path = output_root / "receipts/conditional_child_environment_preflight.json"
    preflight = load_json(preflight_path)
    preflight_binding = {
        **binding(preflight_path, relative_to=output_root),
        "content_digest": preflight["content_digest"],
    }
    replay_path = output_root / "receipts/execution_correction_replay.json"
    helper = _conditional_helper()
    try:
        replay = helper.validate_execution_correction_replay_receipt(
            replay_path, output_root=output_root
        )
    except helper.MaterialisationError as exc:
        raise QualificationError(str(exc)) from exc
    replay_binding = {
        **binding(replay_path, relative_to=output_root),
        "content_digest": replay["content_digest"],
    }
    common = {
        "amendment": copy.deepcopy(CONTRACT.EXECUTION_CORRECTION_AMENDMENT_BINDING),
        "amendment_source_closure": {
            **binding(
                _tracked_path(
                    CONTRACT.TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH
                ),
                relative_to=ROOT,
            ),
            "content_digest": frozen_correction_closure["content_digest"],
            "rows": frozen_correction_closure["row_count"],
        },
        "archive_path": str(archive["archive_path"]),
        "archive_inventory": copy.deepcopy(archive["inventory"]),
        "failure_receipt": copy.deepcopy(archive["failure_receipt"]),
        "source_freeze_commit": str(result["source_freeze_commit"]),
        "files_reused": 0,
        "conditional_child_environment_preflight": preflight_binding,
        "pass": True,
    }
    expected_preexecution = {**common, "execution_correction_replay": None}
    expected_terminal = {
        **common,
        "execution_correction_replay": replay_binding,
    }
    if (
        preexecution.get("execution_correction_custody") != expected_preexecution
        or persistence.get("execution_correction_custody") != expected_terminal
        or result.get("stage_execution", {}).get(
            "execution_correction_custody"
        )
        != expected_terminal
    ):
        raise QualificationError("execution-correction progressive custody drift")
    return expected_terminal


def deep_check(output_root: Path, *, allow_active_self: bool = True) -> dict[str, Any]:
    """Reproduce all aggregates from persisted rows without model inference."""

    del allow_active_self  # own PID is always excluded by the process scanner.
    result = load_json(output_root / "result.json")
    result_custody = result.get("stage_execution", {}).get(
        "execution_correction_custody"
    )
    if not isinstance(result_custody, Mapping):
        raise QualificationError("result lacks mandatory execution-correction custody")
    frozen = _validate_frozen_authorities(
        execution_correction_custody=result_custody
    )
    CONTRACT.validate_result_receipt(result)
    metrics = load_json(output_root / "aggregates/metrics.json")
    if metrics.get("content_digest") != content_digest(metrics):
        raise QualificationError("metrics self-digest drift")
    if (
        metrics.get("historical_raw_latent_goal_cosine_comparators")
        != _historical_raw_comparators()
        or not isinstance(metrics.get("stage_a_raw_cost_rereduced"), Mapping)
        or metrics.get("historical_comparators_recomputed") is not False
        or metrics.get("raw_goal_cosine_executions") != 0
    ):
        raise QualificationError("historical no-recompute comparator binding drift")
    persistence = load_json(output_root / "receipts/persistence.json")
    if persistence.get("content_digest") != content_digest(persistence):
        raise QualificationError("persistence self-digest drift")
    preexecution = load_json(output_root / "receipts/preexecution.json")
    _validate_conditional_child_environment_preflight(output_root, preexecution)
    source_closure_snapshot_path = output_root / "receipts/source_closure.json"
    source_closure_snapshot = load_json(source_closure_snapshot_path)
    expected_source_closure_snapshot_binding = {
        **binding(source_closure_snapshot_path, relative_to=output_root),
        "content_digest": source_closure_snapshot["content_digest"],
    }
    try:
        CONTRACT.validate_source_closure(source_closure_snapshot)
    except CONTRACT.ContractError as exc:
        raise QualificationError(
            f"pre-smoke source-closure snapshot drift: {exc}"
        ) from exc
    expected_prior_smoke_custody: list[dict[str, Any]] = []
    if (
        preexecution.get("content_digest") != content_digest(preexecution)
        or preexecution.get("source_closure_snapshot")
        != expected_source_closure_snapshot_binding
        or source_closure_snapshot != frozen["source_closure"]
        or preexecution.get("prior_smoke_failure_custody")
        != expected_prior_smoke_custody
        or persistence.get("prior_smoke_failure_custody")
        != expected_prior_smoke_custody
        or result.get("prior_smoke_failure_custody")
        != expected_prior_smoke_custody
    ):
        raise QualificationError("prior smoke-failure custody drift")
    _validate_execution_correction_output_custody(
        output_root,
        preexecution=preexecution,
        persistence=persistence,
        result=result,
    )
    expected_metrics_binding = binding(
        output_root / "aggregates/metrics.json", relative_to=output_root
    )
    if (
        result["metrics"].get("binding") != expected_metrics_binding
        or result["metrics"].get("fit_optimization")
        != metrics.get("fit_optimization")
        or result["metrics"].get("stage_a_raw_cost_rereduced")
        != metrics.get("stage_a_raw_cost_rereduced")
        or result["metrics"].get("stage_a_raw_cost_matched_comparisons")
        != metrics.get("stage_a_raw_cost_matched_comparisons")
        or result["metrics"].get("historical_raw_latent_goal_cosine_comparators")
        != metrics.get("historical_raw_latent_goal_cosine_comparators")
        or result["metrics"].get("historical_comparators_recomputed") is not False
        or result["metrics"].get("raw_goal_cosine_executions") != 0
        or result.get("predecessor_narrative_authority")
        != list(CONTRACT.PRESERVED_PREDECESSOR_NARRATIVE)
        or result.get("predecessor_fact_authority")
        != list(CONTRACT.PRESERVED_PREDECESSOR_FACTS)
        or result.get("next_experiment_specification")
        != CONTRACT.next_experiment_specification_for_primary(
            str(result["primary_classification"])
        )
    ):
        raise QualificationError("result metric/narrative authority binding drift")
    training_receipt = _validate_training_checkpoint_custody(output_root, result)
    _validate_manifest(output_root, persistence["artifact_manifest"])
    expected_manifest = _manifest(
        output_root,
        excluded=("receipts/persistence.json", "result.json", "report.md"),
    )
    if persistence["artifact_manifest"] != expected_manifest:
        raise QualificationError("persistence manifest completeness/order drift")
    tensor_rows = tensor_index()
    stage_rows = _rows_from_jsonl(output_root / "ledgers/stage_a_true_future.jsonl")
    _validate_persisted_action_plans(stage_rows)
    _validate_persisted_latent_bindings(stage_rows, tensor_rows=tensor_rows)
    _validate_score_row_arithmetic(stage_rows, stage="STAGE_A")
    target_rows = _rows_from_jsonl(output_root / "ledgers/route_only_targets.jsonl")
    persisted_datasets = _datasets_from_stage_rows(stage_rows)
    (
        _raw_score_maps,
        expected_raw_cost_rows,
        expected_raw_cost_reduction,
    ) = _predecessor_raw_goal_score_maps(
        persisted_datasets, heldout_barrier_open=True
    )
    raw_cost_rows = _rows_from_jsonl(
        output_root / "ledgers/stage_a_raw_cost_rereduced.jsonl"
    )
    if (
        raw_cost_rows != expected_raw_cost_rows
        or metrics.get("stage_a_raw_cost_rereduced")
        != expected_raw_cost_reduction
    ):
        raise QualificationError("raw-cost successor-metric re-reduction drift")
    _validate_raw_cost_merged_scores(stage_rows, raw_cost_rows)
    expected_target_rows = _training_target_rows(persisted_datasets)
    if target_rows != expected_target_rows or len(target_rows) != 576:
        raise QualificationError("route-only target ledger reproduction drift")
    reproduced_fit_optimization = _fit_optimization_summary(
        persisted_datasets[FIT]
    )
    smoke = load_json(output_root / "receipts/training_smoke.json")
    if (
        training_receipt.get("fit_optimization") != reproduced_fit_optimization
        or smoke.get("fit_optimization") != reproduced_fit_optimization
        or smoke.get("content_digest") != content_digest(smoke)
    ):
        raise QualificationError("fit-state optimizer conditioning reproduction drift")
    reproduced = _replay_stage_a_metrics(stage_rows)
    if reproduced != metrics["stage_a"]:
        raise QualificationError("Stage-A row-to-aggregate reproduction drift")
    reproduced_decisions = _stage_a_decisions(reproduced[HELDOUT])
    if reproduced_decisions != metrics["stage_a_decisions"]:
        raise QualificationError("Stage-A gate reproduction drift")

    stage_b = metrics.get("stage_b")
    stage_c = metrics.get("stage_c")
    reproduced_stage_b: dict[str, dict[str, Any]] | None = None
    reproduced_stage_c: dict[str, dict[str, Any]] | None = None
    stage_b_gate_path = output_root / "receipts/stage_b_gate.json"
    stage_b_gate: dict[str, Any] | None = None
    if stage_b is None:
        if reproduced_decisions["true_future_gate"]["pass"]:
            raise QualificationError("Stage B is absent after a passing true-future gate")
        if stage_c is not None:
            raise QualificationError("Stage C exists without Stage B")
    else:
        if not reproduced_decisions["true_future_gate"]["pass"]:
            raise QualificationError("Stage B ran after a failed true-future gate")
        stage_b_rows = _rows_from_jsonl(
            output_root / "ledgers/stage_b_predictor_substitution.jsonl"
        )
        _validate_persisted_action_plans(stage_b_rows)
        _validate_score_row_arithmetic(stage_b_rows, stage="CONDITIONAL")
        reproduced_stage_b = _replay_conditional_metrics(
            stage_b_rows, expected_sources=("R1", "RR", "P1", "PR")
        )
        if reproduced_stage_b != stage_b["summaries"]:
            raise QualificationError("Stage-B row-to-aggregate reproduction drift")
        reproduced_stage_b_decisions = _stage_b_decisions(
            stage_a_decisions=reproduced_decisions,
            stage_a_heldout=reproduced[HELDOUT],
            summaries=reproduced_stage_b[HELDOUT],
        )
        for key, value in reproduced_stage_b_decisions.items():
            if stage_b.get(key) != value:
                raise QualificationError(f"Stage-B {key} reproduction drift")

        stage_b_gate = load_json(stage_b_gate_path)
        conditional_rows, custody = _validate_stage_b_materialisation(
            attempt=output_root,
            gate_path=stage_b_gate_path,
            gate=stage_b_gate,
        )
        stage_b_tensor_rows = {**tensor_rows, **conditional_rows}
        _validate_persisted_latent_bindings(
            stage_b_rows, tensor_rows=stage_b_tensor_rows
        )
        if custody != stage_b["materialisation_custody"]:
            raise QualificationError("Stage-B materialisation-custody drift")
        stage_a_gate_evidence = load_json(
            output_root / "aggregates/stage_a_gate_evidence.json"
        )
        if (
            stage_a_gate_evidence.get("true_future_gate")
            != reproduced_decisions["true_future_gate"]
            or stage_a_gate_evidence.get("true_incremental_value")
            != reproduced_decisions["true_incremental_value"]
            or stage_a_gate_evidence.get("candidate_future_derangement")
            != reproduced_decisions["derangement"]
            or stage_a_gate_evidence.get("pass") is not True
        ):
            raise QualificationError("Stage-B authorisation is not the reproduced Stage-A gate")
        expected_stage_a_summary_digest = hashlib.sha256(
            canonical_bytes(reproduced[HELDOUT])[:-1]
        ).hexdigest()
        if (
            stage_a_gate_evidence.get("heldout_stage_a_summary_sha256")
            != expected_stage_a_summary_digest
        ):
            raise QualificationError("Stage-A gate summary binding drift")

        if stage_c is None:
            if reproduced_stage_b_decisions["proprioception_gate"]["pass"]:
                raise QualificationError(
                    "Stage C is absent after a passing proprioceptive-contribution gate"
                )
        else:
            if not reproduced_stage_b_decisions["proprioception_gate"]["pass"]:
                raise QualificationError(
                    "Stage C ran after a failed proprioceptive-contribution gate"
                )
            stage_c_rows = _rows_from_jsonl(
                output_root / "ledgers/stage_c_attribution.jsonl"
            )
            _validate_persisted_action_plans(stage_c_rows)
            _validate_score_row_arithmetic(stage_c_rows, stage="CONDITIONAL")
            reproduced_stage_c = _replay_conditional_metrics(
                stage_c_rows, expected_sources=STAGE_C_SOURCE_IDS
            )
            if reproduced_stage_c != stage_c["summaries"]:
                raise QualificationError("Stage-C row-to-aggregate reproduction drift")
            _validate_stage_c_matched_pr_scores(stage_c_rows, stage_b_rows)
            reproduced_route_score_changes = _aggregate_stage_c_route_score_changes(
                stage_c_rows
            )
            if reproduced_route_score_changes != stage_c.get("route_score_changes"):
                raise QualificationError("Stage-C actual route-score reproduction drift")

            fidelity_rows = _rows_from_jsonl(
                output_root / "ledgers/stage_c_direct_fidelity.jsonl"
            )
            if len(fidelity_rows) != 4 * STATE_COUNT * CANDIDATE_COUNT * len(HORIZONS):
                raise QualificationError("Stage-C direct-fidelity cardinality drift")
            reproduced_fidelity = _aggregate_stage_c_fidelity(fidelity_rows)
            if reproduced_fidelity != stage_c["direct_future_fidelity_h1_h3"]:
                raise QualificationError("Stage-C direct-fidelity aggregate drift")

            action_rows = _rows_from_jsonl(
                output_root / "ledgers/stage_c_candidate_action_sensitivity.jsonl"
            )
            _validate_action_sensitivity_rows(
                action_rows, stage_c["candidate_action_sensitivity"]
            )
            reproduced_stage_c_decisions = _stage_c_decisions(
                stage_b=stage_b,
                summaries=reproduced_stage_c,
                action_sensitivity=stage_c["candidate_action_sensitivity"],
            )
            for key, value in reproduced_stage_c_decisions.items():
                if stage_c.get(key) != value:
                    raise QualificationError(f"Stage-C {key} reproduction drift")

            helper = _conditional_helper()
            stage_c_gate_path = output_root / "receipts/stage_c_gate.json"
            try:
                stage_c_gate = helper.validate_stage_c_gate_receipt(
                    stage_c_gate_path,
                    output_root=output_root,
                    stage_b_gate_path=stage_b_gate_path,
                )
            except helper.MaterialisationError as exc:
                raise QualificationError(str(exc)) from exc
            stage_b_gate_evidence = load_json(
                output_root / "aggregates/stage_b_gate_evidence.json"
            )
            if (
                stage_b_gate_evidence.get("proprioception_gate")
                != reproduced_stage_b_decisions["proprioception_gate"]
                or stage_b_gate_evidence.get("factorial_contrasts")
                != reproduced_stage_b_decisions["factorial_contrasts"]
                or stage_b_gate_evidence.get("pass") is not True
            ):
                raise QualificationError(
                    "Stage-C authorisation is not the reproduced Stage-B gate"
                )
            expected_stage_b_summary_digest = hashlib.sha256(
                canonical_bytes(reproduced_stage_b[HELDOUT])[:-1]
            ).hexdigest()
            if (
                stage_b_gate_evidence.get("heldout_stage_b_summary_sha256")
                != expected_stage_b_summary_digest
            ):
                raise QualificationError("Stage-B gate summary binding drift")
            top_stage_c = load_json(
                output_root
                / "receipts/stage_c_input_derangement_materialisation.json"
            )
            _validate_stage_c_top_receipt(
                attempt=output_root,
                value=top_stage_c,
                stage_b_gate_digest=str(stage_b_gate["content_digest"]),
                stage_c_gate_path=stage_c_gate_path,
            )
            stage_c_tensor_rows: dict[
                tuple[str, str, int | None, int | None], dict[str, Any]
            ] = {}
            for source in STAGE_C_SOURCE_IDS:
                source_rows, index = _load_prediction_index(
                    attempt=output_root,
                    source_id=source,
                    gate_digest=str(stage_b_gate["content_digest"]),
                    stage_c_gate_digest=str(stage_c_gate["content_digest"]),
                )
                if set(stage_c_tensor_rows) & set(source_rows):
                    raise QualificationError("Stage-C replay tensor indexes overlap")
                stage_c_tensor_rows.update(source_rows)
                expected_binding = binding(
                    output_root / "stage_b/predictions" / source / "index.json",
                    relative_to=output_root,
                )
                if stage_c["prediction_indexes"].get(source) != expected_binding:
                    raise QualificationError(
                        f"Stage-C {source} prediction-index binding drift"
                    )
                if index.get("ablation_or_null") != source:
                    raise QualificationError(f"Stage-C {source} ablation identity drift")
            _validate_persisted_latent_bindings(
                stage_c_rows,
                tensor_rows={**stage_b_tensor_rows, **stage_c_tensor_rows},
            )
            occupancy = stage_c.get("occupancy_probe")
            if (
                not isinstance(occupancy, Mapping)
                or occupancy.get("status") != "NOT_DIRECTLY_COMPATIBLE"
                or occupancy.get("executed") is not False
            ):
                raise QualificationError("Stage-C occupancy compatibility disposition drift")

    replay_stage_b = None if reproduced_stage_b is None else copy.deepcopy(stage_b)
    replay_stage_c = None if reproduced_stage_c is None else copy.deepcopy(stage_c)
    expected_raw_cost_matched_comparisons = _matched_raw_cost_comparisons(
        reproduced, replay_stage_b
    )
    if (
        metrics.get("stage_a_raw_cost_matched_comparisons")
        != expected_raw_cost_matched_comparisons
        or result["metrics"].get("stage_a_raw_cost_matched_comparisons")
        != expected_raw_cost_matched_comparisons
    ):
        raise QualificationError("matched raw-cost comparison reproduction drift")
    expected_primary, expected_secondaries, expected_next = _primary_and_secondaries(
        reproduced_decisions, replay_stage_b, replay_stage_c
    )
    if (
        result["primary_classification"] != expected_primary
        or result["secondary_classifications"] != expected_secondaries
        or result["next_experiment"] != expected_next
    ):
        raise QualificationError("terminal decision reproduction drift")
    if (
        result["metrics"].get("content_digest") != metrics["content_digest"]
        or result["metrics"].get("stage_a") != reproduced
        or result["metrics"].get("stage_a_decisions") != reproduced_decisions
        or result["metrics"].get("stage_b") != stage_b
        or result["metrics"].get("stage_c") != stage_c
        or result["metrics"].get("fit_optimization")
        != metrics.get("fit_optimization")
        or result["metrics"].get("stage_a_raw_cost_rereduced")
        != expected_raw_cost_reduction
        or result["metrics"].get("stage_a_raw_cost_matched_comparisons")
        != expected_raw_cost_matched_comparisons
    ):
        raise QualificationError("result-to-metrics binding drift")
    expected_row_counts = {
        "route_only_targets": 576,
        "training_epochs": 120,
        "stage_a": 576,
        "stage_a_raw_cost_rereduced": 1_728,
        "stage_b": 0 if stage_b is None else 2_304,
        "stage_c": 0 if stage_c is None else 1_728,
        "stage_c_direct_fidelity": (
            0 if stage_c is None else 4 * STATE_COUNT * CANDIDATE_COUNT * len(HORIZONS)
        ),
        "stage_c_candidate_action_sensitivity": (
            0 if stage_c is None else 4 * STATE_COUNT * len(HORIZONS)
        ),
    }
    if persistence.get("row_counts") != expected_row_counts:
        raise QualificationError("persistence row-count receipt drift")
    output_files = [path for path in output_root.rglob("*") if path.is_file()]
    output_bytes = sum(path.stat().st_size for path in output_files)
    if (
        result["runtime_and_storage"].get("output_files") != len(output_files)
        or result["runtime_and_storage"].get("output_bytes") != output_bytes
    ):
        raise QualificationError("result runtime/storage total drift")
    expected_report = _report_markdown(result).encode("utf-8")
    if (output_root / "report.md").read_bytes() != expected_report:
        raise QualificationError("Markdown report regeneration drift")
    if _active_experiment_processes():
        raise QualificationError("experiment process remains active during terminal check")
    checks = {
        "schema_and_self_digests": True,
        "source_closure_revalidated": frozen["source_closure"]["complete"],
        "artifact_manifest_rehashed": True,
        "artifact_manifest_complete_and_ordered": True,
        "final_checkpoints_and_seed_metadata_revalidated": True,
        "sealed_split_lifecycle_receipts_revalidated": True,
        "route_only_target_rows_reproduced": len(target_rows),
        "fit_optimizer_conditioning_reproduced": reproduced_fit_optimization,
        "stage_a_latent_artifact_bindings_revalidated": len(stage_rows),
        "stage_a_rows": len(stage_rows),
        "stage_a_raw_cost_rereduced_rows": len(raw_cost_rows),
        "stage_a_raw_cost_rereduction_reproduced_without_inference": True,
        "stage_a_raw_cost_matched_comparisons_reproduced": True,
        "stage_a_metrics_reproduced_without_model_inference": True,
        "stage_a_gates_reproduced": True,
        "stage_b_rows": 0 if stage_b is None else 2_304,
        "stage_b_metrics_reproduced_without_model_inference": stage_b is not None,
        "stage_b_gate_reproduced": stage_b is not None,
        "stage_b_latent_artifact_bindings_revalidated": (
            0 if stage_b is None else 2_304
        ),
        "stage_c_rows": 0 if stage_c is None else 1_728,
        "stage_c_metrics_reproduced_without_model_inference": stage_c is not None,
        "stage_c_gate_reproduced": stage_c is not None,
        "stage_c_latent_artifact_bindings_revalidated": (
            0 if stage_c is None else 1_728
        ),
        "stage_c_actual_route_scores_reproduced": stage_c is not None,
        "stage_c_direct_fidelity_rows": (
            0
            if stage_c is None
            else 4 * STATE_COUNT * CANDIDATE_COUNT * len(HORIZONS)
        ),
        "stage_c_action_sensitivity_rows": (
            0 if stage_c is None else 4 * STATE_COUNT * len(HORIZONS)
        ),
        "terminal_decision_reproduced": True,
        "report_regenerated_byte_exact": True,
        "output_files": len(output_files),
        "output_bytes": output_bytes,
        "active_experiment_processes": [],
        "pass": True,
    }
    return checks


def _build_result_with_exact_storage(
    core: Mapping[str, Any], *, attempt: Path
) -> tuple[dict[str, Any], bytes]:
    """Solve the small byte-count fixed point for result+report publication."""

    existing_files = [path for path in attempt.rglob("*") if path.is_file()]
    existing_bytes = sum(path.stat().st_size for path in existing_files)
    result = copy.deepcopy(dict(core))
    for _ in range(12):
        signed = attach_digest(result)
        result_bytes = canonical_bytes(signed)
        report_bytes = _report_markdown(signed).encode("utf-8")
        total_files = len(existing_files) + 2
        total_bytes = existing_bytes + len(result_bytes) + len(report_bytes)
        previous = result["runtime_and_storage"].get("output_bytes")
        result["runtime_and_storage"]["output_files"] = total_files
        result["runtime_and_storage"]["output_bytes"] = total_bytes
        if previous == total_bytes:
            final = attach_digest(result)
            final_bytes = canonical_bytes(final)
            final_report = _report_markdown(final).encode("utf-8")
            if existing_bytes + len(final_bytes) + len(final_report) != total_bytes:
                raise QualificationError("result storage fixed point drift")
            return final, final_report
    raise QualificationError("result storage fixed point did not converge")


PREDICTED_SOURCE_KIND = {
    "R1": "ONE_STEP_PREDICTED",
    "RR": "TWO_STEP_PREDICTED",
    "P1": "P1_PROPRIO_ONE_STEP",
    "PR": "PR_PROPRIO_ROLLOUT",
}
STAGE_C_SOURCE_IDS = (
    "PR_VISUAL_CONTEXT_DERANGED",
    "PR_PROPRIO_HISTORY_DERANGED",
    "PR_CONTROL_HISTORY_DERANGED",
)


def _conditional_helper() -> Any:
    """Late import: unreachable until the true-future gate has passed."""

    from scripts import materialize_plan_aware_proprio_predictor_substitution_v1 as helper

    return helper


def _write_stage_a_gate_evidence(
    *,
    attempt: Path,
    source_freeze: str,
    evaluation_contract: Mapping[str, Any],
    stage_a_metrics: Mapping[str, Any],
    stage_a_decisions: Mapping[str, Any],
) -> Path:
    helper = _conditional_helper()
    evidence = attach_digest(
        {
            "schema": helper.STAGE_A_GATE_EVIDENCE_SCHEMA,
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "contract_sha256": CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
            "source_freeze_commit": source_freeze,
            "evaluation_contract_content_digest": evaluation_contract["content_digest"],
            "heldout_stage_a_summary_sha256": hashlib.sha256(
                canonical_bytes(stage_a_metrics[HELDOUT])[:-1]
            ).hexdigest(),
            "true_future_gate": copy.deepcopy(stage_a_decisions["true_future_gate"]),
            "true_incremental_value": copy.deepcopy(
                stage_a_decisions["true_incremental_value"]
            ),
            "candidate_future_derangement": copy.deepcopy(
                stage_a_decisions["derangement"]
            ),
            "predictor_inference_calls_before_receipt": 0,
            "pass": bool(stage_a_decisions["true_future_gate"]["pass"]),
        }
    )
    if (
        evidence["true_future_gate"].get("classification")
        != "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL"
        or evidence["pass"] is not True
    ):
        raise QualificationError("cannot publish a Stage-B gate from failed Stage A")
    path = attempt / "aggregates/stage_a_gate_evidence.json"
    atomic_json(path, evidence)
    return path


def _publish_stage_b_gate(
    *,
    attempt: Path,
    source_freeze: str,
    evaluation_contract: Mapping[str, Any],
    stage_a_metrics: Mapping[str, Any],
    stage_a_decisions: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    helper = _conditional_helper()
    evidence_path = _write_stage_a_gate_evidence(
        attempt=attempt,
        source_freeze=source_freeze,
        evaluation_contract=evaluation_contract,
        stage_a_metrics=stage_a_metrics,
        stage_a_decisions=stage_a_decisions,
    )
    latent_record = evaluation_contract["checkpoint_bindings"][
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL"
    ]
    latent_checkpoint = attempt / str(latent_record["path"])
    gate = helper.build_stage_b_gate_receipt(
        contract_freeze_commit=source_freeze,
        evaluation_contract_path=attempt / "receipts/evaluation_contract.json",
        stage_a_gate_evidence_path=evidence_path,
        latent_ranker_checkpoint_path=latent_checkpoint,
        output_root=attempt,
    )
    path = attempt / "receipts/stage_b_gate.json"
    atomic_json(path, gate)
    try:
        helper.validate_stage_b_gate_receipt(path, output_root=attempt)
    except helper.MaterialisationError as exc:
        raise QualificationError(str(exc)) from exc
    return path, gate


def _run_conditional_helper_cli(
    *, attempt: Path, arguments: Sequence[str], log_name: str
) -> dict[str, Any]:
    helper = _conditional_helper()
    command = [
        str(helper.GPU_INTERPRETER),
        "-E",
        "-s",
        str(helper.SELF),
        *arguments,
    ]
    environment = helper.build_child_environment(helper.GPU_INTERPRETER)
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path = attempt / "logs" / log_name
    atomic_bytes(log_path, completed.stdout.encode("utf-8", errors="replace"))
    if completed.returncode:
        raise QualificationError(
            f"conditional helper failed ({completed.returncode}): "
            f"{completed.stdout[-8000:]}"
        )
    return {
        "argv": command,
        "returncode": completed.returncode,
        "runtime_s": time.time() - started,
        "log": binding(log_path, relative_to=attempt),
        "child_environment_contract": helper.child_environment_contract(
            helper.GPU_INTERPRETER
        ),
        "numerical_thread_environment": copy.deepcopy(CONTRACT.NUMERICAL_THREAD_ENV),
    }


def _conditional_prediction_authorities(attempt: Path) -> dict[str, Any]:
    """Rebuild outcome-blind input custody for conditional prediction indexes."""

    ids = split_ids()
    manifests = state_manifest_map()
    contexts = _index_by_state(CONTEXT_INDEX)
    state_order = sorted(
        (state_id for role in SPLIT_ROLES for state_id in ids[role]),
        key=numeric_state_key,
    )
    role_by_state = {
        state_id: role for role in SPLIT_ROLES for state_id in ids[role]
    }
    if set(state_order) != set(manifests) or set(state_order) != set(contexts):
        raise QualificationError("conditional input-authority state coverage drift")

    context_index = load_json(attempt / "stage_b/proprio_context/index.json")
    if context_index.get("content_digest") != content_digest(context_index):
        raise QualificationError("proprio context index self-digest drift")
    context_summaries = context_index.get("state_records")
    if not isinstance(context_summaries, list) or [
        str(row.get("state_id")) for row in context_summaries
    ] != state_order:
        raise QualificationError("proprio context index state order/coverage drift")
    shard_by_state = {
        str(row["state_id"]): copy.deepcopy(row["shard"])
        for row in context_summaries
    }

    tensor_payload = load_json(LATENT_INDEX)
    tensor_records = tensor_payload.get("records")
    if not isinstance(tensor_records, list):
        raise QualificationError("predecessor tensor records are missing")
    tensor_by_key = {
        (
            str(row.get("kind")),
            str(row.get("state_id")),
            row.get("candidate_index_or_null"),
            row.get("horizon_or_null"),
        ): row
        for row in tensor_records
    }
    visual_by_state: dict[str, list[dict[str, Any]]] = {}
    for state_id in state_order:
        visual: list[dict[str, Any]] = []
        for slot in (-2, -1, 0):
            record = tensor_by_key.get(("CONTEXT", state_id, None, slot))
            if record is None:
                raise QualificationError(
                    f"conditional visual-context tensor missing: {state_id}:{slot}"
                )
            reference = Path(str(record["path"]))
            path = reference if reference.is_absolute() else PREDECESSOR_ROOT / reference
            visual.append(
                {
                    "slot": slot,
                    "path": str(path),
                    "sha256": str(record["sha256"]),
                    "bytes": int(record["bytes"]),
                }
            )
        visual_by_state[state_id] = visual

    action_sha_by_state: dict[str, str] = {}
    for state_id in state_order:
        actions = np.asarray(
            contexts[state_id]["action_blocks_raw_3x10_by_candidate"],
            np.float32,
        )
        if actions.shape != (CANDIDATE_COUNT, 3, 10):
            raise QualificationError(
                f"conditional candidate-action authority drift: {state_id}"
            )
        action_sha_by_state[state_id] = hashlib.sha256(actions.tobytes()).hexdigest()

    evaluation = load_json(attempt / "receipts/evaluation_contract.json")
    if evaluation.get("content_digest") != content_digest(evaluation):
        raise QualificationError("evaluation-contract self-digest drift")
    experiment_digest = str(evaluation.get("experiment_contract_digest"))
    expected_donors = donor_derangements(ids, manifests, experiment_digest)
    if evaluation.get("stage_c_input_donor_mappings") != expected_donors:
        raise QualificationError("Stage-C donor authority drift")
    return {
        "state_order": state_order,
        "role_by_state": role_by_state,
        "family_by_state": {
            state_id: str(manifests[state_id]["family"])
            for state_id in state_order
        },
        "shard_by_state": shard_by_state,
        "visual_by_state": visual_by_state,
        "action_sha_by_state": action_sha_by_state,
        "stage_c_donors": expected_donors,
    }


def _load_prediction_index(
    *,
    attempt: Path,
    source_id: str,
    gate_digest: str,
    stage_c_gate_digest: str | None = None,
) -> tuple[dict[tuple[str, str, int | None, int | None], dict[str, Any]], dict[str, Any]]:
    helper = _conditional_helper()
    path = attempt / "stage_b/predictions" / source_id / "index.json"
    value = load_json(path)
    if value.get("content_digest") != content_digest(value):
        raise QualificationError(f"{source_id} prediction-index self-digest drift")
    stage_c = stage_c_gate_digest is not None
    predictor_source_id = "PR_PROPRIO_ROLLOUT" if stage_c else source_id
    expected_ablation = source_id if stage_c else None
    expected_config = {
        "cell": (
            "proprio_one_step"
            if predictor_source_id == "P1_PROPRIO_ONE_STEP"
            else "proprio_rollout"
        ),
        "use_proprio": True,
        "rollout": predictor_source_id == "PR_PROPRIO_ROLLOUT",
        "width": 384,
    }
    if predictor_source_id not in (
        "P1_PROPRIO_ONE_STEP",
        "PR_PROPRIO_ROLLOUT",
    ) or (stage_c and source_id not in STAGE_C_SOURCE_IDS):
        raise QualificationError(f"unsupported conditional predictor source: {source_id}")
    custody = value.get("predictor_custody")
    future_proprioception = value.get("future_proprioception")
    expected_slot_validity = {
        "1": [True, True, True],
        "2": [True, True, False],
        "3": [True, False, False],
    }
    parameter_digest = (
        None if not isinstance(custody, Mapping) else custody.get("parameter_digest_before")
    )
    if (
        value.get("schema") != helper.PREDICTION_INDEX_SCHEMA
        or value.get("status") != "PASS"
        or value.get("experiment_id") != CONTRACT.EXPERIMENT_ID
        or value.get("source_id") != source_id
        or value.get("predictor_source_id") != predictor_source_id
        or value.get("ablation_or_null") != expected_ablation
        or value.get("stage_b_gate_digest") != gate_digest
        or value.get("stage_c_gate_digest_or_null") != stage_c_gate_digest
        or value.get("complete") is not True
        or value.get("states") != STATE_COUNT
        or value.get("candidates_per_state") != CANDIDATE_COUNT
        or value.get("horizons") != list(HORIZONS)
        or value.get("prediction_shape_per_state")
        != list(helper.PREDICTION_STATE_SHAPE)
        or value.get("dtype") != "float16"
        or value.get("logical_records_count") != STATE_COUNT * CANDIDATE_COUNT * 3
        or value.get("route_outcomes_opened") is not False
        or value.get("training_executed") is not False
        or future_proprioception
        != {
            "values_available_to_predictor": False,
            "masked_by_frozen_absence_mechanism": True,
            "read_count": 0,
        }
        or not isinstance(custody, Mapping)
        or custody.get("source_id") != predictor_source_id
        or custody.get("checkpoint")
        != CONTRACT.CHECKPOINT_BINDINGS[predictor_source_id]
        or custody.get("model_config") != expected_config
        or custody.get("strict_state_dict_load") is not True
        or custody.get("eval_mode") is not True
        or custody.get("requires_grad_all_false") is not True
        or custody.get("parameter_state_unchanged") is not True
        or custody.get("parameter_digest_before")
        != custody.get("parameter_digest_after")
        or not isinstance(parameter_digest, str)
        or len(parameter_digest) != 64
        or any(character not in "0123456789abcdef" for character in parameter_digest)
        or custody.get("checkpoint_optimizer_state_deserialized_but_ignored")
        is not True
        or custody.get("optimizer_state_loaded_into_an_optimizer") is not False
        or custody.get("optimizer_steps") != 0
        or custody.get("state_batch_calls") != STATE_COUNT
        or custody.get("model_forward_calls") != STATE_COUNT * len(HORIZONS)
        or custody.get("future_proprioception_read_count") != 0
        or custody.get("observed_slot_validity_by_horizon")
        != expected_slot_validity
    ):
        raise QualificationError(f"{source_id} prediction-index contract drift")
    authorities = _conditional_prediction_authorities(attempt)
    state_order = authorities["state_order"]
    expected_input_state_ids: dict[str, dict[str, str]] = {}
    ablated_component = (
        None
        if not stage_c
        else CONTRACT.STAGE_C_ABLATION_INPUTS[source_id][
            "predictor_input_component"
        ]
    )
    for state_id in state_order:
        donors = {component: state_id for component in ("visual", "proprio", "control")}
        if ablated_component is not None:
            donors[ablated_component] = authorities["stage_c_donors"][source_id][
                state_id
            ]
        expected_input_state_ids[state_id] = donors
    records = value.get("logical_records")
    if not isinstance(records, list) or len(records) != STATE_COUNT * CANDIDATE_COUNT * 3:
        raise QualificationError(f"{source_id} logical prediction records are incomplete")
    output: dict[tuple[str, str, int | None, int | None], dict[str, Any]] = {}
    expected_logical_order = [
        (state_id, candidate, horizon)
        for state_id in state_order
        for candidate in range(CANDIDATE_COUNT)
        for horizon in HORIZONS
    ]
    observed_logical_order: list[tuple[str, int, int]] = []
    state_tensor_by_state: dict[str, Mapping[str, Any]] = {}
    expected_states = value.get("state_records")
    if not isinstance(expected_states, list) or [
        str(row.get("state_id")) for row in expected_states
    ] != state_order:
        raise QualificationError(f"{source_id} state tensor records are incomplete")
    for row in expected_states:
        state_id = str(row["state_id"])
        donors = expected_input_state_ids[state_id]
        record = row.get("prediction_state_tensor")
        if (
            row.get("family") != authorities["family_by_state"][state_id]
            or row.get("split_role") != authorities["role_by_state"][state_id]
            or row.get("source_id") != source_id
            or row.get("predictor_source_id") != predictor_source_id
            or row.get("ablation_or_null") != expected_ablation
            or row.get("input_state_ids") != donors
            or row.get("visual_context_bindings")
            != authorities["visual_by_state"][donors["visual"]]
            or row.get("proprio_shard_binding")
            != authorities["shard_by_state"][donors["proprio"]]
            or row.get("control_shard_binding")
            != authorities["shard_by_state"][donors["control"]]
            or row.get("candidate_action_sha256")
            != authorities["action_sha_by_state"][state_id]
            or not isinstance(record, Mapping)
        ):
            raise QualificationError(f"{source_id} state input-custody drift: {state_id}")
        artifact = helper.resolve_artifact(str(record["path"]), attempt)
        try:
            helper.verify_file_binding(artifact, record, f"{source_id} state tensor")
        except helper.MaterialisationError as exc:
            raise QualificationError(str(exc)) from exc
        state_tensor_by_state[state_id] = record

    for row in records:
        key = (
            str(row["kind"]),
            str(row["state_id"]),
            int(row["candidate_index_or_null"]),
            int(row["horizon_or_null"]),
        )
        if key in output:
            raise QualificationError(f"duplicate conditional tensor identity: {key}")
        state_id = str(row["state_id"])
        candidate = int(row["candidate_index_or_null"])
        horizon = int(row["horizon_or_null"])
        observed_logical_order.append((state_id, candidate, horizon))
        state_tensor = state_tensor_by_state.get(state_id)
        if (
            row.get("kind") != source_id
            or row.get("input_state_ids") != expected_input_state_ids.get(state_id)
            or state_tensor is None
            or row.get("path") != state_tensor.get("path")
            or row.get("sha256") != state_tensor.get("sha256")
            or row.get("bytes") != state_tensor.get("bytes")
            or row.get("array_index") != [candidate, horizon - 1]
            or row.get("logical_shape") != list(TENSOR_SHAPE)
            or row.get("logical_dtype") != "float16"
        ):
            raise QualificationError(f"{source_id} logical input-custody drift: {key}")
        output[key] = dict(row)
    if observed_logical_order != expected_logical_order:
        raise QualificationError(f"{source_id} logical record order/coverage drift")
    return output, value


def _validate_proprio_context_receipts(
    *,
    attempt: Path,
    gate_digest: str,
    contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    helper = _conditional_helper()
    index_path = attempt / "stage_b/proprio_context/index.json"
    index = load_json(index_path)
    ids = split_ids()
    manifests = state_manifest_map()
    predecessor_contexts = _index_by_state(CONTEXT_INDEX)
    state_order = sorted(
        (state_id for role in SPLIT_ROLES for state_id in ids[role]),
        key=numeric_state_key,
    )
    role_by_state = {
        state_id: role for role in SPLIT_ROLES for state_id in ids[role]
    }
    summaries = index.get("state_records")
    if (
        index.get("content_digest") != content_digest(index)
        or index.get("schema") != helper.CONTEXT_INDEX_SCHEMA
        or index.get("experiment_id") != CONTRACT.EXPERIMENT_ID
        or index.get("stage_b_gate_digest") != gate_digest
        or index.get("complete") is not True
        or index.get("records") != STATE_COUNT
        or index.get("shape_per_state") != list(helper.PROPRIO_SHAPE)
        or index.get("dtype") != "float32"
        or index.get("reconstruction_prefix_blocks") != STATE_COUNT * 40
        or index.get("new_states") != 0
        or index.get("new_candidates") != 0
        or index.get("future_proprioception_read_count") != 0
        or index.get("training_executed") is not False
        or not isinstance(summaries, list)
        or [str(row.get("state_id")) for row in summaries] != state_order
        or set(contexts) != set(state_order)
    ):
        raise QualificationError("proprio context index semantic drift")
    summary_by_state = {str(row["state_id"]): row for row in summaries}
    for state_index, state_id in enumerate(state_order):
        record = contexts[state_id]
        predecessor = predecessor_contexts[state_id]
        expected_snapshot = predecessor.get("replay_snapshot_digest")
        if expected_snapshot is None:
            expected_snapshot = predecessor.get("branch_snapshot_digest")
        replay = record.get("replay")
        history = record.get("observed_history")
        sample_rows = history.get("sample_rows") if isinstance(history, Mapping) else None
        if (
            record.get("content_digest") != content_digest(record)
            or record.get("schema") != helper.CONTEXT_STATE_SCHEMA
            or record.get("status") != "PASS"
            or record.get("experiment_id") != CONTRACT.EXPERIMENT_ID
            or record.get("stage_b_gate_digest") != gate_digest
            or record.get("state_index") != state_index
            or record.get("state_id") != state_id
            or record.get("family") != str(manifests[state_id]["family"])
            or record.get("split_role") != role_by_state[state_id]
            or not isinstance(replay, Mapping)
            or replay.get("mode") != "FROZEN_STATE_RECONSTRUCTION_REPLAY"
            or replay.get("blocks") != 40
            or replay.get("prefix_blocks_only") is not True
            or replay.get("state_or_candidates_created") != 0
            or replay.get("snapshot_digest") != expected_snapshot
            or replay.get("predecessor_snapshot_digest") != expected_snapshot
            or replay.get("snapshot_exact") is not True
            or not isinstance(history, Mapping)
            or history.get("shape") != list(helper.PROPRIO_SHAPE)
            or history.get("dtype") != "float32"
            or history.get("slots") != 3
            or history.get("samples_per_slot") != 5
            or history.get("channels") != 30
            or history.get("future_slots_read") != 0
            or history.get("future_proprioception_forbidden") is not True
            or not isinstance(sample_rows, list)
            or len(sample_rows) != 15
            or any(
                row.get("future_proprioception") is not False
                or row.get("physics_observed") is not True
                for row in sample_rows
            )
            or record.get("control_history_exact_predecessor_match") is not True
            or record.get("training_executed") is not False
            or record.get("route_outcomes_opened") is not False
        ):
            raise QualificationError(f"proprio context receipt semantic drift: {state_id}")
        summary = summary_by_state[state_id]
        if (
            summary.get("family") != record["family"]
            or summary.get("split_role") != record["split_role"]
            or summary.get("shard") != record.get("shard")
        ):
            raise QualificationError(f"proprio context summary drift: {state_id}")
        shard = helper.resolve_artifact(str(record["shard"]["path"]), attempt)
        try:
            helper.verify_file_binding(shard, record["shard"], f"{state_id} context shard")
        except helper.MaterialisationError as exc:
            raise QualificationError(str(exc)) from exc
    return index


def _validate_stage_b_materialisation(
    *, attempt: Path, gate_path: Path, gate: Mapping[str, Any]
) -> tuple[
    dict[tuple[str, str, int | None, int | None], dict[str, Any]],
    dict[str, Any],
]:
    helper = _conditional_helper()
    try:
        helper.validate_stage_b_gate_receipt(gate_path, output_root=attempt)
        contexts = helper._load_context_index(attempt, str(gate["content_digest"]))
    except helper.MaterialisationError as exc:
        raise QualificationError(str(exc)) from exc
    if len(contexts) != STATE_COUNT:
        raise QualificationError("conditional proprio context is incomplete")
    context_index = _validate_proprio_context_receipts(
        attempt=attempt,
        gate_digest=str(gate["content_digest"]),
        contexts=contexts,
    )
    top_path = attempt / "receipts/stage_b_proprio_predictor_materialisation.json"
    top = load_json(top_path)
    expected_counts = {
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
    }
    expected_gate_record = {
        **binding(gate_path, relative_to=attempt),
        "content_digest": gate["content_digest"],
        "contract_freeze_commit": gate["contract_freeze_commit"],
    }
    replay_path = attempt / "receipts/execution_correction_replay.json"
    try:
        helper.validate_execution_correction_replay_receipt(
            replay_path, output_root=attempt
        )
    except helper.MaterialisationError as exc:
        raise QualificationError(str(exc)) from exc
    expected_replay_binding = binding(replay_path, relative_to=attempt)
    if (
        top.get("content_digest") != content_digest(top)
        or top.get("schema") != helper.MATERIALISATION_RECEIPT_SCHEMA
        or top.get("status") != "PASS"
        or top.get("experiment_id") != CONTRACT.EXPERIMENT_ID
        or top.get("stage_b_gate") != expected_gate_record
        or top.get("execution_correction_replay") != expected_replay_binding
        or top.get("counts") != expected_counts
        or top.get("workers") != helper.FROZEN_WORKERS
        or top.get("numerical_thread_environment")
        != CONTRACT.NUMERICAL_THREAD_ENV
        or top.get("training_executed") is not False
        or top.get("fresh_panel_collected") is not False
        or top.get("navigation_executed") is not False
        or top.get("nothing_running_at_receipt_write") is not True
        or top.get("stage_b_gate", {}).get("content_digest")
        != gate["content_digest"]
        or top.get("proprio_context_index")
        != binding(
            attempt / "stage_b/proprio_context/index.json", relative_to=attempt
        )
    ):
        raise QualificationError("conditional predictor materialisation receipt drift")
    combined: dict[
        tuple[str, str, int | None, int | None], dict[str, Any]
    ] = {}
    indexes: dict[str, Any] = {}
    for source_id in ("P1_PROPRIO_ONE_STEP", "PR_PROPRIO_ROLLOUT"):
        rows, value = _load_prediction_index(
            attempt=attempt, source_id=source_id, gate_digest=str(gate["content_digest"])
        )
        if set(combined) & set(rows):
            raise QualificationError("conditional prediction indexes overlap")
        combined.update(rows)
        indexes[source_id] = {
            "binding": binding(
                attempt / "stage_b/predictions" / source_id / "index.json",
                relative_to=attempt,
            ),
            "content_digest": value["content_digest"],
        }
        if top.get("prediction_indexes", {}).get(source_id) != indexes[source_id][
            "binding"
        ]:
            raise QualificationError(
                f"conditional top receipt prediction binding drift: {source_id}"
            )
    return combined, {
        "gate": binding(gate_path, relative_to=attempt),
        "gate_content_digest": gate["content_digest"],
        "execution_correction_replay": expected_replay_binding,
        "context_index": binding(
            attempt / "stage_b/proprio_context/index.json", relative_to=attempt
        ),
        "context_index_content_digest": context_index["content_digest"],
        "materialisation_receipt": binding(top_path, relative_to=attempt),
        "prediction_indexes": indexes,
    }


def _validate_stage_c_top_receipt(
    *,
    attempt: Path,
    value: Mapping[str, Any],
    stage_b_gate_digest: str,
    stage_c_gate_path: Path,
) -> None:
    expected_indexes = {
        source: binding(
            attempt / "stage_b/predictions" / source / "index.json",
            relative_to=attempt,
        )
        for source in STAGE_C_SOURCE_IDS
    }
    replay_path = attempt / "receipts/execution_correction_replay.json"
    expected_replay_binding = binding(replay_path, relative_to=attempt)
    if (
        value.get("content_digest") != content_digest(value)
        or value.get("schema")
        != "plan_aware_monotone_jepa_cost_v1.stage_c_materialisation.v1"
        or value.get("status") != "PASS"
        or value.get("experiment_id") != CONTRACT.EXPERIMENT_ID
        or value.get("stage_b_gate_digest") != stage_b_gate_digest
        or value.get("execution_correction_replay") != expected_replay_binding
        or value.get("stage_c_gate")
        != binding(stage_c_gate_path, relative_to=attempt)
        or value.get("prediction_indexes") != expected_indexes
        or value.get("checkpoint")
        != CONTRACT.CHECKPOINT_BINDINGS["PR_PROPRIO_ROLLOUT"]
        or value.get("training_executed") is not False
        or value.get("outcome_informed_donor_selection") is not False
        or value.get("nothing_running_at_receipt_write") is not True
    ):
        raise QualificationError("Stage-C helper receipt drift")


def _merge_score_map(
    destination: dict[str, dict[str, np.ndarray]],
    source: Mapping[str, Mapping[str, np.ndarray]],
) -> None:
    for condition, state_values in source.items():
        if condition not in destination:
            destination[condition] = {
                state: np.asarray(values, np.float64).copy()
                for state, values in state_values.items()
            }
            continue
        if list(destination[condition]) != list(state_values) or any(
            not np.array_equal(destination[condition][state], state_values[state])
            for state in destination[condition]
        ):
            raise QualificationError(f"shared conditional score drift: {condition}")


def _score_conditional_sources(
    *,
    attempt: Path,
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    models: Mapping[str, Any],
    tensor_rows: Mapping[tuple[str, str, int | None, int | None], Mapping[str, Any]],
    sources: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    score_maps_by_role: dict[str, Any] = {}
    evidence_by_role: dict[str, Any] = {}
    summaries_by_role: dict[str, Any] = {}
    for role in SPLIT_ROLES:
        merged: dict[str, dict[str, np.ndarray]] = {}
        evidence: dict[str, Any] = {}
        for source in sources:
            source_maps, source_evidence = score_dataset(
                datasets[role],
                models=models,
                tensor_rows=tensor_rows,
                latent_source=source,
                logical_output_root=attempt,
            )
            _merge_score_map(merged, source_maps)
            evidence[source] = source_evidence
        score_maps_by_role[role] = merged
        evidence_by_role[role] = evidence
        summaries_by_role[role] = summarize_score_maps(datasets[role], merged)
    return score_maps_by_role, evidence_by_role, summaries_by_role


def _conditional_ledger_rows(
    *,
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    score_maps_by_role: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    evidence_by_role: Mapping[str, Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]],
    sources: Sequence[str],
    schema: str,
    matched_route_scores_by_role: Mapping[
        str, Mapping[str, Mapping[str, np.ndarray]]
    ]
    | None = None,
    matched_condition: str | None = None,
) -> list[dict[str, Any]]:
    if (matched_route_scores_by_role is None) != (matched_condition is None):
        raise QualificationError("matched route-score inputs must be supplied together")
    rows: list[dict[str, Any]] = []
    for role in SPLIT_ROLES:
        maps = score_maps_by_role[role]
        for source in sources:
            source_evidence = evidence_by_role[role][source]
            condition = f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"
            zero_condition = f"LATENT_BRANCH_ZERO_{source}"
            for state_id in sorted(datasets[role], key=numeric_state_key):
                for candidate, evidence in enumerate(source_evidence[state_id]):
                    row = {
                        **copy.deepcopy(dict(evidence)),
                        "schema": schema,
                        "latent_source": source,
                        "candidate_index": candidate,
                        "population_membership": _population_membership(evidence),
                        "route_scores": {
                            "KINEMATIC_ROUTE_BASELINE": float(
                                maps["KINEMATIC_ROUTE_BASELINE"][state_id][candidate]
                            ),
                            "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": float(
                                maps["KINEMATIC_PLUS_NO_LATENT_RESIDUAL"][state_id][candidate]
                            ),
                            condition: float(maps[condition][state_id][candidate]),
                            zero_condition: float(
                                maps[zero_condition][state_id][candidate]
                            ),
                            "DETERMINISTIC_RANDOM": float(
                                maps["DETERMINISTIC_RANDOM"][state_id][candidate]
                            ),
                        },
                    }
                    if matched_route_scores_by_role is not None:
                        assert matched_condition is not None
                        matched_score = float(
                            matched_route_scores_by_role[role][matched_condition][
                                state_id
                            ][candidate]
                        )
                        source_score = float(maps[condition][state_id][candidate])
                        row["matched_pr_score"] = matched_score
                        row["deranged_minus_matched_pr_score"] = (
                            source_score - matched_score
                        )
                    rows.append(row)
    expected = STATE_COUNT * CANDIDATE_COUNT * len(sources)
    if len(rows) != expected:
        raise QualificationError(f"conditional ledger has {len(rows)} rows, expected {expected}")
    _validate_score_row_arithmetic(rows, stage="CONDITIONAL")
    return rows


def _aggregate_stage_c_route_score_changes(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reduce candidate-matched ranker score changes from persisted Stage-C rows."""

    grouped: dict[str, dict[str, list[float]]] = {
        source: {role: [] for role in SPLIT_ROLES} for source in STAGE_C_SOURCE_IDS
    }
    for row in rows:
        source = str(row["latent_source"])
        role = str(row["split"])
        if source not in grouped or role not in grouped[source]:
            raise QualificationError(f"unknown Stage-C route-score identity: {source}:{role}")
        matched = float(row["matched_pr_score"])
        condition = f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"
        observed = float(row["route_scores"][condition])
        delta = float(row["deranged_minus_matched_pr_score"])
        if not math.isclose(observed - matched, delta, rel_tol=0.0, abs_tol=1e-12):
            raise QualificationError(
                f"Stage-C candidate route-score delta drift: {source}:{row['state_id']}:"
                f"{row['candidate_index']}"
            )
        grouped[source][role].append(delta)

    def summary(values: Sequence[float]) -> dict[str, Any]:
        if not values:
            raise QualificationError("Stage-C route-score change group is empty")
        numeric = np.asarray(values, dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise QualificationError("Stage-C route-score change is non-finite")
        return {
            "rows": len(values),
            "signed_mean": float(np.mean(numeric)),
            "signed_minimum": float(np.min(numeric)),
            "signed_maximum": float(np.max(numeric)),
            "absolute_mean": float(np.mean(np.abs(numeric))),
            "absolute_maximum": float(np.max(np.abs(numeric))),
            "nonzero_count": int(np.count_nonzero(numeric)),
        }

    return {
        "schema": "plan_aware_stage_c_actual_route_score_changes_v1",
        "definition": "deranged ranker score minus matched frozen PR ranker score",
        "by_source": {
            source: {
                "all_roles": summary(
                    [value for role in SPLIT_ROLES for value in grouped[source][role]]
                ),
                "by_role": {
                    role: summary(grouped[source][role]) for role in SPLIT_ROLES
                },
            }
            for source in STAGE_C_SOURCE_IDS
        },
    }


def _validate_stage_c_matched_pr_scores(
    stage_c_rows: Sequence[Mapping[str, Any]],
    stage_b_rows: Sequence[Mapping[str, Any]],
) -> None:
    pr_condition = "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_PR"
    expected: dict[tuple[str, str, int], float] = {}
    for row in stage_b_rows:
        if row.get("latent_source") != "PR":
            continue
        identity = (
            str(row["split"]),
            str(row["state_id"]),
            int(row["candidate_index"]),
        )
        if identity in expected:
            raise QualificationError(f"duplicate Stage-B PR score identity: {identity}")
        expected[identity] = float(row["route_scores"][pr_condition])
    if len(expected) != STATE_COUNT * CANDIDATE_COUNT:
        raise QualificationError("Stage-B PR matched-score authority is incomplete")
    counts = {identity: 0 for identity in expected}
    for row in stage_c_rows:
        identity = (
            str(row["split"]),
            str(row["state_id"]),
            int(row["candidate_index"]),
        )
        if identity not in expected or not math.isclose(
            float(row["matched_pr_score"]),
            expected[identity],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise QualificationError(f"Stage-C matched PR score drift: {identity}")
        counts[identity] += 1
    if any(count != len(STAGE_C_SOURCE_IDS) for count in counts.values()):
        raise QualificationError("Stage-C matched PR score coverage drift")


def _factorial_interaction_bootstrap(
    *,
    r1: Mapping[str, Any],
    rr: Mapping[str, Any],
    p1: Mapping[str, Any],
    pr: Mapping[str, Any],
) -> dict[str, Any]:
    population = METRICS.ORACLE_VIABILITY_ADMISSIBLE

    def states(summary: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        rows = summary["populations"][population]["per_state"]
        return {str(row["state_id"]): row for row in rows}

    by_source = {name: states(value) for name, value in (("R1", r1), ("RR", rr), ("P1", p1), ("PR", pr))}
    identities = list(by_source["R1"])
    if any(list(value) != identities for value in by_source.values()):
        raise QualificationError("factorial bootstrap state identity drift")
    families = {state: str(by_source["R1"][state]["family"]) for state in identities}

    def interaction(metric: str, *, lower_better: bool = False) -> dict[str, Any] | None:
        eligible = [
            state
            for state in identities
            if all(by_source[source][state][metric] is not None for source in by_source)
        ]
        if not eligible:
            return None
        values = {}
        for state in eligible:
            r1_v, rr_v = float(by_source["R1"][state][metric]), float(by_source["RR"][state][metric])
            p1_v, pr_v = float(by_source["P1"][state][metric]), float(by_source["PR"][state][metric])
            values[state] = (
                (p1_v - pr_v) - (r1_v - rr_v)
                if lower_better
                else (pr_v - p1_v) - (rr_v - r1_v)
            )
        zeros = {state: 0.0 for state in eligible}
        return OLD_METRICS.paired_state_bootstrap(
            values,
            zeros,
            {state: families[state] for state in eligible},
            comparison_id=f"J_BP_MINUS_BR/{population}/{metric}",
            draws=METRICS.BOOTSTRAP_DRAWS,
            seed=METRICS.BOOTSTRAP_SEED,
        )

    return {
        "schema": "plan_aware_factorial_interaction_bootstrap_v1",
        "population_id": population,
        "draws": METRICS.BOOTSTRAP_DRAWS,
        "seed": METRICS.BOOTSTRAP_SEED,
        "pairwise_accuracy": interaction("pairwise_accuracy"),
        "selected_progress_m": interaction("selected_route_progress_m"),
        "normalized_regret_reduction": interaction(
            "normalized_regret", lower_better=True
        ),
        "best_route_top3": interaction("best_route_top3"),
    }


def _selected_candidate_changes(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    contrast_id: str,
) -> dict[str, Any]:
    """Persist exact paired selections for every state and population."""

    populations: dict[str, Any] = {}
    for population_id in METRICS.POPULATION_IDS:
        left_rows = left["populations"][population_id]["per_state"]
        right_rows = right["populations"][population_id]["per_state"]
        left_by_state = {str(row["state_id"]): row for row in left_rows}
        right_by_state = {str(row["state_id"]): row for row in right_rows}
        if list(left_by_state) != list(right_by_state):
            raise QualificationError(
                f"selected-candidate contrast identity drift: {contrast_id}:{population_id}"
            )
        paired_rows: list[dict[str, Any]] = []
        for state_id in left_by_state:
            left_row = left_by_state[state_id]
            right_row = right_by_state[state_id]
            if (
                left_row["family"] != right_row["family"]
                or left_row["role"] != right_row["role"]
            ):
                raise QualificationError(
                    f"selected-candidate contrast metadata drift: {contrast_id}:{state_id}"
                )
            left_selected = left_row["selected_candidate_index"]
            right_selected = right_row["selected_candidate_index"]
            paired_rows.append(
                {
                    "state_id": state_id,
                    "family": left_row["family"],
                    "role": left_row["role"],
                    "left_selected_candidate_index": left_selected,
                    "right_selected_candidate_index": right_selected,
                    "selected_candidate_changed": left_selected != right_selected,
                }
            )
        changed = sum(
            bool(row["selected_candidate_changed"]) for row in paired_rows
        )
        populations[population_id] = {
            "states": paired_rows,
            "state_count": len(paired_rows),
            "changed_count": changed,
            "changed_fraction": changed / len(paired_rows) if paired_rows else 0.0,
        }
    return {
        "schema": "plan_aware_selected_candidate_change_contrast_v1",
        "contrast_id": contrast_id,
        "populations": populations,
    }


def _stage_b_decisions(
    *, stage_a_decisions: Mapping[str, Any], stage_a_heldout: Mapping[str, Any], summaries: Mapping[str, Any]
) -> dict[str, Any]:
    r1 = summaries["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1"]
    rr = summaries["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_RR"]
    p1 = summaries["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_P1"]
    pr = summaries["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_PR"]
    true = stage_a_heldout["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE"]
    kinematic = summaries["KINEMATIC_ROUTE_BASELINE"]
    true_aggregate = true["populations"][
        METRICS.ORACLE_VIABILITY_ADMISSIBLE
    ]["aggregate"]
    true_progress = float(true_aggregate["selected_route_progress_m_sum"])

    def nullable_float(value: Any) -> float | None:
        return None if value is None else float(value)
    source_summaries = {
        "R1_RGB_ONE_STEP": r1,
        "RR_RGB_ROLLOUT": rr,
        "P1_PROPRIO_ONE_STEP": p1,
        "PR_PROPRIO_ROLLOUT": pr,
    }
    absolute_metrics: dict[str, dict[str, Any]] = {}
    absolute_passes: dict[str, bool] = {}
    for source_id, summary in source_summaries.items():
        population = summary["populations"][
            METRICS.ORACLE_VIABILITY_ADMISSIBLE
        ]
        aggregate = population["aggregate"]
        predicted_progress = nullable_float(
            aggregate["selected_route_progress_m_sum"]
        )
        values = {
            "pairwise_accuracy": nullable_float(aggregate["pairwise_accuracy"]),
            "normalized_regret": nullable_float(aggregate["normalized_regret"]),
            "best_route_top3": nullable_float(aggregate["best_route_top3_rate"]),
            "selected_progress_fraction_of_oracle": nullable_float(
                aggregate["selected_progress_ratio"]
            ),
            "selected_progress_fraction_of_true": (
                None
                if predicted_progress is None
                else predicted_progress
                / max(abs(true_progress), METRICS.NUMERIC_EPSILON)
            ),
            "no_family_complete_collapse": bool(
                population["no_family_complete_collapse"]
            ),
        }
        absolute_metrics[source_id] = values
        absolute_passes[source_id] = (
            CONTRACT.predicted_source_absolute_preservation_passes(values)
        )
    all_fail = CONTRACT.all_predicted_substitutions_fail_materially(
        absolute_metrics
    )
    return {
        "predicted_gate": METRICS.evaluate_predicted_gate(
            true_gate=stage_a_decisions["true_future_gate"],
            true_source=true,
            one_step_source=r1,
            rollout_source=rr,
        ),
        "incremental_over_kinematics": METRICS.evaluate_incremental_value(
            rr, kinematic_source=kinematic
        ),
        "proprioception_gate": METRICS.evaluate_proprio_gate(
            rgb_one_step=r1,
            rgb_rollout=rr,
            proprio_one_step=p1,
            proprio_rollout=pr,
        ),
        "factorial_contrasts": METRICS.factorial_rollout_contrasts(
            rgb_one_step=r1,
            rgb_rollout=rr,
            proprio_one_step=p1,
            proprio_rollout=pr,
        ),
        "direct_proprioception_contrasts": {
            "P1_minus_R1": METRICS.principal_metric_deltas(p1, r1),
            "PR_minus_RR": METRICS.principal_metric_deltas(pr, rr),
        },
        "paired_bootstrap": {
            "BR_RR_MINUS_R1": METRICS.paired_principal_bootstrap(rr, r1),
            "BP_PR_MINUS_P1": METRICS.paired_principal_bootstrap(pr, p1),
            "P1_MINUS_R1": METRICS.paired_principal_bootstrap(p1, r1),
            "PR_MINUS_RR": METRICS.paired_principal_bootstrap(pr, rr),
            "J_BP_MINUS_BR": _factorial_interaction_bootstrap(
                r1=r1, rr=rr, p1=p1, pr=pr
            ),
        },
        "selected_candidate_changes": {
            "RR_vs_R1": _selected_candidate_changes(
                rr, r1, contrast_id="RR_VS_R1"
            ),
            "PR_vs_P1": _selected_candidate_changes(
                pr, p1, contrast_id="PR_VS_P1"
            ),
            "PR_vs_RR": _selected_candidate_changes(
                pr, rr, contrast_id="PR_VS_RR"
            ),
        },
        "predicted_source_absolute_preservation": {
            "schema": "plan_aware_predicted_source_absolute_preservation_v1",
            "thresholds": copy.deepcopy(
                CONTRACT.PREDICTED_SOURCE_ABSOLUTE_PRESERVATION_GATE
            ),
            "per_source_metrics": absolute_metrics,
            "per_source_pass": absolute_passes,
            "all_predicted_substitutions_fail_materially": all_fail,
        },
        "all_predicted_substitutions_fail_materially": all_fail,
    }


def _publish_stage_c_gate(
    *,
    attempt: Path,
    source_freeze: str,
    gate_path: Path,
    evaluation_contract: Mapping[str, Any],
    stage_b_decisions: Mapping[str, Any],
    heldout_summaries: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    helper = _conditional_helper()
    evidence = attach_digest(
        {
            "schema": helper.STAGE_B_GATE_EVIDENCE_SCHEMA,
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "contract_sha256": CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256,
            "source_freeze_commit": source_freeze,
            "heldout_stage_b_summary_sha256": hashlib.sha256(
                canonical_bytes(heldout_summaries)[:-1]
            ).hexdigest(),
            "proprioception_gate": copy.deepcopy(
                stage_b_decisions["proprioception_gate"]
            ),
            "factorial_contrasts": copy.deepcopy(
                stage_b_decisions["factorial_contrasts"]
            ),
            "pass": bool(stage_b_decisions["proprioception_gate"]["pass"]),
        }
    )
    if (
        evidence["pass"] is not True
        or evidence["proprioception_gate"].get("classification")
        != "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION"
    ):
        raise QualificationError("cannot publish Stage C before the proprioception gate")
    evidence_path = attempt / "aggregates/stage_b_gate_evidence.json"
    atomic_json(evidence_path, evidence)
    value = helper.build_stage_c_gate_receipt(
        contract_freeze_commit=source_freeze,
        stage_b_gate_receipt_path=gate_path,
        stage_b_metrics_path=evidence_path,
        evaluation_contract_path=attempt / "receipts/evaluation_contract.json",
        output_root=attempt,
    )
    path = attempt / "receipts/stage_c_gate.json"
    atomic_json(path, value)
    try:
        helper.validate_stage_c_gate_receipt(
            path, output_root=attempt, stage_b_gate_path=gate_path
        )
    except helper.MaterialisationError as exc:
        raise QualificationError(str(exc)) from exc
    return path, value


def _load_state_prediction_array(
    *, attempt: Path, index: Mapping[str, Any], state_id: str
) -> np.ndarray:
    helper = _conditional_helper()
    records = {
        str(row["state_id"]): row for row in index.get("state_records", ())
    }
    if state_id not in records or len(records) != STATE_COUNT:
        raise QualificationError("prediction state-record identity drift")
    binding_record = records[state_id]["prediction_state_tensor"]
    path = helper.resolve_artifact(str(binding_record["path"]), attempt)
    try:
        helper.verify_file_binding(path, binding_record, f"{state_id} prediction tensor")
    except helper.MaterialisationError as exc:
        raise QualificationError(str(exc)) from exc
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if value.shape != helper.PREDICTION_STATE_SHAPE or value.dtype != np.float16:
        raise QualificationError(f"{state_id} prediction state tensor drift")
    return value


def _normalise_true_tokens(value: np.ndarray) -> np.ndarray:
    numeric = np.asarray(value, np.float32)
    mean = numeric.mean(axis=-1, keepdims=True)
    variance = ((numeric - mean) ** 2).mean(axis=-1, keepdims=True)
    output = (numeric - mean) / np.sqrt(variance + 1e-5)
    if not np.isfinite(output).all():
        raise QualificationError("true-future LayerNorm produced non-finite values")
    return output


def _stage_c_fidelity_and_action_rows(
    *,
    attempt: Path,
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    predecessor_tensor_rows: Mapping[
        tuple[str, str, int | None, int | None], Mapping[str, Any]
    ],
    prediction_indexes: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    state_meta = {
        state_id: {
            "family": str(rows[0]["family"]),
            "split": role,
        }
        for role in SPLIT_ROLES
        for state_id, rows in datasets[role].items()
    }
    fidelity: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for source_id, index in prediction_indexes.items():
        for state_id in sorted(state_meta, key=numeric_state_key):
            predicted = _load_state_prediction_array(
                attempt=attempt, index=index, state_id=state_id
            )
            for horizon_index, horizon in enumerate(HORIZONS):
                candidate_hashes = [
                    hashlib.sha256(
                        np.asarray(predicted[candidate, horizon_index]).tobytes()
                    ).hexdigest()
                    for candidate in range(CANDIDATE_COUNT)
                ]
                distinct = len(set(candidate_hashes))
                reference = np.asarray(predicted[0, horizon_index], np.float32)
                max_abs = max(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(predicted[candidate, horizon_index], np.float32)
                                - reference
                            )
                        )
                    )
                    for candidate in range(1, CANDIDATE_COUNT)
                )
                sensitivity.append(
                    {
                        "schema": "plan_aware_candidate_action_sensitivity_row_v1",
                        "latent_source": source_id,
                        "state_id": state_id,
                        "family": state_meta[state_id]["family"],
                        "split": state_meta[state_id]["split"],
                        "horizon": horizon,
                        "candidate_count": CANDIDATE_COUNT,
                        "distinct_candidate_prediction_tensors": distinct,
                        "maximum_absolute_difference_from_candidate_zero": max_abs,
                        "candidate_action_sensitivity_present": bool(
                            distinct >= 2 and max_abs > 0.0
                        ),
                        "within_state_nonaction_inputs_candidate_invariant": True,
                    }
                )
                for candidate in range(CANDIDATE_COUNT):
                    true = load_tensor(
                        predecessor_tensor_rows[
                            ("TRUE_FUTURE", state_id, candidate, horizon)
                        ]
                    )
                    target = _normalise_true_tokens(true)
                    estimate = np.asarray(
                        predicted[candidate, horizon_index], np.float32
                    )
                    numerator = np.sum(estimate * target, axis=-1)
                    denominator = np.maximum(
                        np.linalg.norm(estimate, axis=-1)
                        * np.linalg.norm(target, axis=-1),
                        1e-12,
                    )
                    cosine = numerator / denominator
                    fidelity.append(
                        {
                            "schema": "plan_aware_stage_c_direct_fidelity_row_v1",
                            "latent_source": source_id,
                            "state_id": state_id,
                            "family": state_meta[state_id]["family"],
                            "split": state_meta[state_id]["split"],
                            "candidate_index": candidate,
                            "horizon": horizon,
                            "token_cosine_mean": float(np.mean(cosine)),
                            "token_l1_mean": float(np.mean(np.abs(estimate - target))),
                            "tokens": TENSOR_SHAPE[0],
                            "token_dimension": TENSOR_SHAPE[1],
                        }
                    )
    expected_fidelity = len(prediction_indexes) * STATE_COUNT * CANDIDATE_COUNT * 3
    expected_sensitivity = len(prediction_indexes) * STATE_COUNT * 3
    if len(fidelity) != expected_fidelity or len(sensitivity) != expected_sensitivity:
        raise QualificationError("Stage-C fidelity/action evidence cardinality drift")
    return fidelity, sensitivity


def _aggregate_stage_c_fidelity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["latent_source"]), {}).setdefault(
            str(row["horizon"]), []
        ).append(row)

    def summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(values),
            "token_cosine_mean": float(
                np.mean([float(row["token_cosine_mean"]) for row in values])
            ),
            "token_l1_mean": float(
                np.mean([float(row["token_l1_mean"]) for row in values])
            ),
        }

    return {
        source: {
            horizon: {
                "aggregate": summary(values),
                "per_family": {
                    family: summary(
                        [row for row in values if row["family"] == family]
                    )
                    for family in sorted({str(row["family"]) for row in values})
                },
                "per_split": {
                    role: summary([row for row in values if row["split"] == role])
                    for role in SPLIT_ROLES
                },
            }
            for horizon, values in horizons.items()
        }
        for source, horizons in grouped.items()
    }


def _action_sensitivity_evidence(
    *,
    attempt: Path,
    rows: Sequence[Mapping[str, Any]],
    prediction_indexes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    path = attempt / "ledgers/stage_c_candidate_action_sensitivity.jsonl"
    atomic_bytes(path, b"".join(canonical_bytes(row) for row in rows))
    failed = [
        {
            "latent_source": row["latent_source"],
            "state_id": row["state_id"],
            "horizon": row["horizon"],
        }
        for row in rows
        if not bool(row["candidate_action_sensitivity_present"])
    ]
    return {
        "authority": (
            "persisted within-state frozen-predictor output variation; visual, "
            "proprioceptive and control context are candidate-invariant and only "
            "the frozen candidate action plan varies"
        ),
        "raw_evidence": {
            "ledger": binding(path, relative_to=attempt),
            "rows": len(rows),
            "failed_rows": failed,
            "prediction_indexes": {
                source: {
                    "content_digest": index["content_digest"],
                    "logical_records": index["logical_records_count"],
                }
                for source, index in prediction_indexes.items()
            },
        },
        "passed": not failed,
    }


def _stage_c_decisions(
    *,
    stage_b: Mapping[str, Any],
    summaries: Mapping[str, Any],
    action_sensitivity: Mapping[str, Any],
) -> dict[str, Any]:
    heldout = summaries[HELDOUT]
    matched = stage_b["summaries"][HELDOUT][
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_PR"
    ]
    gate = METRICS.evaluate_substitution_gate(
        proprio_contribution_gate=stage_b["proprioception_gate"],
        matched_proprio_rollout=matched,
        visual_deranged=heldout[
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_PR_VISUAL_CONTEXT_DERANGED"
        ],
        proprio_deranged=heldout[
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_PR_PROPRIO_HISTORY_DERANGED"
        ],
        control_deranged=heldout[
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_PR_CONTROL_HISTORY_DERANGED"
        ],
        candidate_action_sensitivity_evidence=action_sensitivity,
    )
    route_metric_changes = {
        source: {
            "principal_deltas_vs_PR": METRICS.principal_metric_deltas(
                heldout[f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"], matched
            ),
            "paired_bootstrap_vs_PR": METRICS.paired_principal_bootstrap(
                heldout[f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"], matched
            ),
        }
        for source in STAGE_C_SOURCE_IDS
    }
    return {
        "substitution_gate": gate,
        "classification": gate["classification"],
        "dependence_attribution": gate["dependence_attribution"],
        "route_metric_changes": route_metric_changes,
        "candidate_action_sensitivity": copy.deepcopy(dict(action_sensitivity)),
        "control_only_explanation_excluded": gate["criteria"][
            "not_explained_by_control_derangement_alone"
        ],
    }


def _publish_execution_correction_replay_gate(
    *, attempt: Path, execution_correction_custody: Mapping[str, Any]
) -> dict[str, Any]:
    """Prove equivalence before any conditional scientific/materialisation child.

    The bound outcome-free CPU/GPU import probes ran before fit with every
    scientific/opening counter at zero and are the sole prospective exception.
    """

    if not execution_correction_custody:
        raise QualificationError("execution-correction replay lacks archive custody")
    premature: list[str] = []
    for directory in (attempt / "stage_b", attempt / "stage_c"):
        if directory.exists():
            premature.append(str(directory.relative_to(attempt)))
    for directory, prefix in (
        (attempt / "logs", "stage_b"),
        (attempt / "logs", "stage_c"),
        (attempt / "receipts", "stage_b_proprio"),
        (attempt / "receipts", "stage_c_"),
    ):
        if directory.is_dir():
            premature.extend(
                str(path.relative_to(attempt))
                for path in directory.iterdir()
                if path.name.startswith(prefix)
            )
    if premature:
        raise QualificationError(
            "conditional artifacts exist before execution-correction replay gate: "
            f"{sorted(premature)}"
        )
    try:
        receipt = CONTRACT.validate_execution_correction_replay(attempt)
    except CONTRACT.ContractError as exc:
        raise QualificationError(str(exc)) from exc
    if (
        receipt.get("pass") is not True
        or receipt.get("files_reused") != 0
        or receipt.get("failed_archive")
        != execution_correction_custody.get("archive_path")
    ):
        raise QualificationError("execution-correction replay receipt custody drift")
    path = attempt / "receipts/execution_correction_replay.json"
    atomic_json(path, receipt)
    if load_json(path) != receipt:
        raise QualificationError("execution-correction replay receipt roundtrip drift")
    return receipt


def _execute_conditional_stage_b(
    *,
    attempt: Path,
    source_freeze: str,
    datasets: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    models: Mapping[str, Any],
    tensor_rows: Mapping[tuple[str, str, int | None, int | None], Mapping[str, Any]],
    evaluation_contract: Mapping[str, Any],
    stage_a_metrics: Mapping[str, Any],
    stage_a_decisions: Mapping[str, Any],
    execution_correction_custody: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Late import prevents any P1/PR materialisation before the true gate."""

    if not stage_a_decisions["true_future_gate"]["pass"]:
        raise QualificationError("Stage B called before the true-future gate")
    helper = _conditional_helper()
    gate_path, gate = _publish_stage_b_gate(
        attempt=attempt,
        source_freeze=source_freeze,
        evaluation_contract=evaluation_contract,
        stage_a_metrics=stage_a_metrics,
        stage_a_decisions=stage_a_decisions,
    )
    if not execution_correction_custody:
        raise QualificationError(
            "amended execution cannot enter Stage B without correction custody"
        )
    replay_receipt = _publish_execution_correction_replay_gate(
        attempt=attempt,
        execution_correction_custody=execution_correction_custody,
    )
    replay_receipt_path = attempt / "receipts/execution_correction_replay.json"
    execution_correction_custody["execution_correction_replay"] = {
        **binding(replay_receipt_path, relative_to=attempt),
        "content_digest": replay_receipt["content_digest"],
    }
    execution = _run_conditional_helper_cli(
        attempt=attempt,
        arguments=(
            "run-stage-b",
            "--stage-b-authorised",
            "--gate-receipt",
            str(gate_path),
            "--execution-correction-replay-receipt",
            str(replay_receipt_path),
            "--output-root",
            str(attempt),
            "--workers",
            str(CONTRACT.CPU_WORKER_BENCHMARK["selected_workers"]),
        ),
        log_name="stage_b_proprio_predictor_materialisation.log",
    )
    conditional_rows, materialisation = _validate_stage_b_materialisation(
        attempt=attempt, gate_path=gate_path, gate=gate
    )
    all_tensors = dict(tensor_rows)
    if set(all_tensors) & set(conditional_rows):
        raise QualificationError("P1/PR tensor identities overlap predecessor tensors")
    all_tensors.update(conditional_rows)
    sources = ("R1", "RR", "P1", "PR")
    score_maps, evidence, summaries = _score_conditional_sources(
        attempt=attempt,
        datasets=datasets,
        models=models,
        tensor_rows=all_tensors,
        sources=sources,
    )
    rows = _conditional_ledger_rows(
        datasets=datasets,
        score_maps_by_role=score_maps,
        evidence_by_role=evidence,
        sources=sources,
        schema="plan_aware_evaluation_row_v1",
    )
    if len(rows) != 2_304:
        raise QualificationError("Stage-B source-specific ledger must contain 2,304 rows")
    stage_b_path = attempt / "ledgers/stage_b_predictor_substitution.jsonl"
    atomic_bytes(stage_b_path, b"".join(canonical_bytes(row) for row in rows))
    decisions = _stage_b_decisions(
        stage_a_decisions=stage_a_decisions,
        stage_a_heldout=stage_a_metrics[HELDOUT],
        summaries=summaries[HELDOUT],
    )
    stage_b = {
        "schema": "plan_aware_stage_b_predictor_substitution_v1",
        "row_count": len(rows),
        "source_specific_rows": True,
        "sources": list(sources),
        "summaries": summaries,
        **decisions,
        "gate_receipt": binding(gate_path, relative_to=attempt),
        "gate_content_digest": gate["content_digest"],
        "execution_correction_replay": {
            **binding(replay_receipt_path, relative_to=attempt),
            "content_digest": replay_receipt["content_digest"],
        },
        "helper_execution": execution,
        "materialisation_custody": materialisation,
        "ledger": binding(stage_b_path, relative_to=attempt),
        "ranker_refit_or_recalibration": False,
        "predictor_training_steps": 0,
        "fresh_states_or_candidates": 0,
    }

    if not decisions["proprioception_gate"]["pass"]:
        return stage_b, None

    stage_c_gate_path, stage_c_gate = _publish_stage_c_gate(
        attempt=attempt,
        source_freeze=source_freeze,
        gate_path=gate_path,
        evaluation_contract=evaluation_contract,
        stage_b_decisions=decisions,
        heldout_summaries=summaries[HELDOUT],
    )
    stage_c_execution = _run_conditional_helper_cli(
        attempt=attempt,
        arguments=(
            "run-stage-c",
            "--stage-b-authorised",
            "--stage-c-authorised",
            "--gate-receipt",
            str(gate_path),
            "--stage-c-gate-receipt",
            str(stage_c_gate_path),
            "--execution-correction-replay-receipt",
            str(replay_receipt_path),
            "--output-root",
            str(attempt),
        ),
        log_name="stage_c_input_derangement_materialisation.log",
    )
    try:
        helper.validate_stage_c_gate_receipt(
            stage_c_gate_path, output_root=attempt, stage_b_gate_path=gate_path
        )
    except helper.MaterialisationError as exc:
        raise QualificationError(str(exc)) from exc
    top_stage_c_path = attempt / "receipts/stage_c_input_derangement_materialisation.json"
    top_stage_c = load_json(top_stage_c_path)
    _validate_stage_c_top_receipt(
        attempt=attempt,
        value=top_stage_c,
        stage_b_gate_digest=str(gate["content_digest"]),
        stage_c_gate_path=stage_c_gate_path,
    )
    stage_c_tensor_rows: dict[
        tuple[str, str, int | None, int | None], dict[str, Any]
    ] = {}
    stage_c_indexes: dict[str, Any] = {}
    for source in STAGE_C_SOURCE_IDS:
        source_rows, source_index = _load_prediction_index(
            attempt=attempt,
            source_id=source,
            gate_digest=str(gate["content_digest"]),
            stage_c_gate_digest=str(stage_c_gate["content_digest"]),
        )
        if set(stage_c_tensor_rows) & set(source_rows):
            raise QualificationError("Stage-C tensor indexes overlap")
        stage_c_tensor_rows.update(source_rows)
        stage_c_indexes[source] = source_index
    all_stage_c_tensors = {**all_tensors, **stage_c_tensor_rows}
    stage_c_maps, stage_c_evidence, stage_c_summaries = _score_conditional_sources(
        attempt=attempt,
        datasets=datasets,
        models=models,
        tensor_rows=all_stage_c_tensors,
        sources=STAGE_C_SOURCE_IDS,
    )
    stage_c_rows = _conditional_ledger_rows(
        datasets=datasets,
        score_maps_by_role=stage_c_maps,
        evidence_by_role=stage_c_evidence,
        sources=STAGE_C_SOURCE_IDS,
        schema="plan_aware_attribution_row_v1",
        matched_route_scores_by_role=score_maps,
        matched_condition="KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_PR",
    )
    if len(stage_c_rows) != 1_728:
        raise QualificationError("Stage-C attribution ledger must contain 1,728 rows")
    stage_c_path = attempt / "ledgers/stage_c_attribution.jsonl"
    atomic_bytes(stage_c_path, b"".join(canonical_bytes(row) for row in stage_c_rows))

    _pr_rows, pr_index = _load_prediction_index(
        attempt=attempt,
        source_id="PR_PROPRIO_ROLLOUT",
        gate_digest=str(gate["content_digest"]),
    )
    fidelity_indexes = {"PR": pr_index, **stage_c_indexes}
    fidelity_rows, action_rows = _stage_c_fidelity_and_action_rows(
        attempt=attempt,
        datasets=datasets,
        predecessor_tensor_rows=tensor_rows,
        prediction_indexes=fidelity_indexes,
    )
    fidelity_path = attempt / "ledgers/stage_c_direct_fidelity.jsonl"
    atomic_bytes(fidelity_path, b"".join(canonical_bytes(row) for row in fidelity_rows))
    fidelity = _aggregate_stage_c_fidelity(fidelity_rows)
    action_sensitivity = _action_sensitivity_evidence(
        attempt=attempt,
        rows=action_rows,
        prediction_indexes=fidelity_indexes,
    )
    stage_c_decisions = _stage_c_decisions(
        stage_b=stage_b,
        summaries=stage_c_summaries,
        action_sensitivity=action_sensitivity,
    )
    actual_route_score_changes = _aggregate_stage_c_route_score_changes(stage_c_rows)
    stage_c = {
        "schema": "plan_aware_stage_c_attribution_v1",
        "row_count": len(stage_c_rows),
        "sources": list(STAGE_C_SOURCE_IDS),
        "summaries": stage_c_summaries,
        **stage_c_decisions,
        "route_score_changes": actual_route_score_changes,
        "direct_future_fidelity_h1_h3": fidelity,
        "direct_fidelity_rows": len(fidelity_rows),
        "direct_fidelity_ledger": binding(fidelity_path, relative_to=attempt),
        "occupancy_probe": {
            "status": "NOT_DIRECTLY_COMPATIBLE",
            "executed": False,
            "reason": (
                "the frozen contract contains no occupancy-probe checkpoint, input "
                "normalisation, or 48-state compatibility binding; the legacy probe "
                "cannot be silently transferred to this panel"
            ),
        },
        "gate_receipt": binding(stage_c_gate_path, relative_to=attempt),
        "gate_content_digest": stage_c_gate["content_digest"],
        "helper_execution": stage_c_execution,
        "helper_materialisation_receipt": binding(
            top_stage_c_path, relative_to=attempt
        ),
        "prediction_indexes": {
            source: binding(
                attempt / "stage_b/predictions" / source / "index.json",
                relative_to=attempt,
            )
            for source in STAGE_C_SOURCE_IDS
        },
        "ledger": binding(stage_c_path, relative_to=attempt),
        "ranker_refit_or_recalibration": False,
        "predictor_training_steps": 0,
    }
    return stage_b, stage_c


def execute() -> dict[str, Any]:
    started = time.time()
    execution_correction_custody = _runtime_execution_correction_custody()
    source_freeze = str(execution_correction_custody["source_freeze_commit"])
    frozen = _validate_frozen_authorities(
        execution_correction_custody=execution_correction_custody
    )
    attempt = _new_attempt(
        CONTRACT.OUTPUT_ROOT,
        source_freeze,
        execution_correction_custody=execution_correction_custody,
    )
    publication_happened = False
    tracked_publication_happened = False
    phase = "PREEXECUTION"
    lifecycle_counters = {
        "full_training_epochs_completed": 0,
        "calibration_rows_opened": 0,
        "heldout_rows_opened": 0,
        "final_checkpoint_published": False,
    }
    try:
        child_environment_preflight = _conditional_child_environment_preflight(
            attempt
        )
        execution_correction_custody[
            "conditional_child_environment_preflight"
        ] = {
            **binding(
                attempt / "receipts/conditional_child_environment_preflight.json",
                relative_to=attempt,
            ),
            "content_digest": child_environment_preflight["content_digest"],
        }
        preexecution = _preexecution_receipt(
            attempt=attempt,
            source_freeze=source_freeze,
            frozen=frozen,
            conditional_child_environment_preflight=(
                child_environment_preflight
            ),
            execution_correction_custody=execution_correction_custody,
        )
        ids = split_ids()
        line_index = route_line_index()
        contexts = _index_by_state(CONTEXT_INDEX)
        goals = _index_by_state(GOAL_INDEX)
        manifests = state_manifest_map()
        tensors = tensor_index()
        route_roles = {
            str(row["state_id"]): str(row["route_role"])
            for row in frozen["route_role_authority"]["records"]
        }

        # Stage-A fit barrier: only these 32 route/outcome shards are opened.
        phase = "STAGE_A_FIT_OPEN"
        fit = authorised_dataset(
            FIT,
            ids=ids,
            line_index=line_index,
            contexts=contexts,
            goals=goals,
            route_roles=route_roles,
        )
        fit_opened_at = time.time()
        phase = "TRAINING_SMOKE"
        smoke = run_training_smoke(fit, tensors, ids=ids, output_root=attempt)
        phase = "ROUTE_RANKER_TRAINING"
        checkpoints, training_receipt, models = train_rankers(
            fit, tensors, output_root=attempt
        )
        lifecycle_counters["full_training_epochs_completed"] = int(
            CONTRACT.TRAINING["epochs"]
        )
        lifecycle_counters["final_checkpoint_published"] = True
        phase = "EVALUATION_CONTRACT_PUBLICATION"
        evaluation_contract = write_evaluation_contract(
            output_root=attempt,
            source_freeze_commit=source_freeze,
            checkpoints=checkpoints,
            line_index=line_index,
            ids=ids,
            manifests=manifests,
        )

        # Calibration opens only after both final checkpoints. It is reporting
        # only: no weights, thresholds, epochs, or hyperparameters can change.
        phase = "CALIBRATION_OPEN"
        calibration = authorised_dataset(
            CALIBRATION,
            ids=ids,
            line_index=line_index,
            contexts=contexts,
            goals=goals,
            route_roles=route_roles,
        )
        calibration_opened_at = time.time()
        lifecycle_counters["calibration_rows_opened"] = CONTRACT.ROLE_ROW_COUNTS[
            CALIBRATION
        ]

        # Heldout opens last, only after the evaluation contract is durable.
        phase = "HELDOUT_OPEN"
        heldout = authorised_dataset(
            HELDOUT,
            ids=ids,
            line_index=line_index,
            contexts=contexts,
            goals=goals,
            route_roles=route_roles,
        )
        heldout_opened_at = time.time()
        lifecycle_counters["heldout_rows_opened"] = CONTRACT.ROLE_ROW_COUNTS[
            HELDOUT
        ]
        datasets = {FIT: fit, CALIBRATION: calibration, HELDOUT: heldout}
        phase = "PREDECESSOR_RAW_COST_READ_ONLY_REDUCTION"
        (
            raw_score_maps_by_role,
            raw_cost_rows,
            raw_cost_reduction,
        ) = _predecessor_raw_goal_score_maps(
            datasets, heldout_barrier_open=True
        )
        score_maps_by_role: dict[str, Any] = {}
        evidence_by_role: dict[str, Any] = {}
        summaries_by_role: dict[str, Any] = {}
        future_maps = evaluation_contract["future_latent_derangement_by_state"]
        phase = "STAGE_A_EVALUATION"
        for role in SPLIT_ROLES:
            score_maps, evidence = score_dataset(
                datasets[role],
                models=models,
                tensor_rows=tensors,
                latent_source="TRUE",
                future_derangements={state: future_maps[state] for state in datasets[role]},
            )
            _merge_score_map(score_maps, raw_score_maps_by_role[role])
            score_maps_by_role[role] = score_maps
            evidence_by_role[role] = evidence
            summaries_by_role[role] = summarize_score_maps(datasets[role], score_maps)
        stage_a_rows = _stage_a_ledger_rows(
            datasets, score_maps_by_role, evidence_by_role
        )
        if len(stage_a_rows) != STATE_COUNT * CANDIDATE_COUNT:
            raise QualificationError("Stage-A ledger cardinality drift")
        atomic_bytes(
            attempt / "ledgers/stage_a_true_future.jsonl",
            b"".join(canonical_bytes(row) for row in stage_a_rows),
        )
        if len(raw_cost_rows) != int(
            CONTRACT.PREDECESSOR_CANDIDATE_EVIDENCE_BINDING["rows"]
        ):
            raise QualificationError("raw-cost re-reduction ledger cardinality drift")
        atomic_bytes(
            attempt / "ledgers/stage_a_raw_cost_rereduced.jsonl",
            b"".join(canonical_bytes(row) for row in raw_cost_rows),
        )
        _write_training_ledgers(
            attempt=attempt, datasets=datasets, training_receipt=training_receipt
        )
        stage_a_decisions = _stage_a_decisions(summaries_by_role[HELDOUT])
        stage_b: dict[str, Any] | None = None
        stage_c: dict[str, Any] | None = None
        if stage_a_decisions["true_future_gate"]["pass"]:
            phase = "CONDITIONAL_STAGE_B_AND_C"
            stage_b, stage_c = _execute_conditional_stage_b(
                attempt=attempt,
                source_freeze=source_freeze,
                datasets=datasets,
                models=models,
                tensor_rows=tensors,
                evaluation_contract=evaluation_contract,
                stage_a_metrics=summaries_by_role,
                stage_a_decisions=stage_a_decisions,
                execution_correction_custody=execution_correction_custody,
            )
        else:
            raise QualificationError(
                "execution-correction replay diverged before the bound Stage-B gate"
            )
        raw_cost_matched_comparisons = _matched_raw_cost_comparisons(
            summaries_by_role, stage_b
        )
        phase = "TERMINAL_AGGREGATION"
        primary, secondaries, next_experiment = _primary_and_secondaries(
            stage_a_decisions, stage_b, stage_c
        )
        historical = _historical_raw_comparators()
        metrics = attach_digest(
            {
                "schema": "plan_aware_monotone_jepa_cost_metrics_v1",
                "experiment_id": CONTRACT.EXPERIMENT_ID,
                "source_freeze_commit": source_freeze,
                "contract_sha256": CONTRACT.CONTRACT_SHA256,
                "stage_a": summaries_by_role,
                "stage_a_decisions": stage_a_decisions,
                "stage_b": stage_b,
                "stage_c": stage_c,
                "fit_optimization": training_receipt["fit_optimization"],
                "stage_a_raw_cost_rereduced": raw_cost_reduction,
                "stage_a_raw_cost_matched_comparisons": (
                    raw_cost_matched_comparisons
                ),
                "historical_raw_latent_goal_cosine_comparators": historical,
                "historical_comparators_recomputed": False,
                "raw_goal_cosine_executions": 0,
            }
        )
        atomic_json(attempt / "aggregates/metrics.json", metrics)

        evaluation_receipt = attach_digest(
            {
                "schema": "plan_aware_evaluation_receipt_v1",
                "source_freeze_commit": source_freeze,
                "stage_a_rows": len(stage_a_rows),
                "stage_a_raw_cost_rereduced_rows": len(raw_cost_rows),
                "stage_a_roles": copy.deepcopy(CONTRACT.ROLE_ROW_COUNTS),
                "heldout_opened_after_checkpoint_publication": heldout_opened_at
                >= calibration_opened_at
                >= fit_opened_at,
                "heldout_opened_after_evaluation_contract": True,
                "calibration_used_for_model_selection": False,
                "stage_b_executed": stage_b is not None,
                "stage_c_executed": stage_c is not None,
                "predictor_training_steps": 0,
                "raw_goal_cosine_executions": 0,
                "primary_classification": primary,
                "pass": True,
            }
        )
        atomic_json(attempt / "receipts/evaluation.json", evaluation_receipt)

        phase = "PERSISTENCE"
        artifact_manifest = _manifest(
            attempt,
            excluded=("receipts/persistence.json", "result.json", "report.md"),
        )
        persistence = attach_digest(
            {
                "schema": "plan_aware_persistence_receipt_v1",
                "artifact_manifest": artifact_manifest,
                "row_counts": {
                    "route_only_targets": 576,
                    "training_epochs": 120,
                    "stage_a": len(stage_a_rows),
                    "stage_a_raw_cost_rereduced": len(raw_cost_rows),
                    "stage_b": 0 if stage_b is None else int(stage_b["row_count"]),
                    "stage_c": 0 if stage_c is None else int(stage_c["row_count"]),
                    "stage_c_direct_fidelity": (
                        0 if stage_c is None else int(stage_c["direct_fidelity_rows"])
                    ),
                    "stage_c_candidate_action_sensitivity": (
                        0
                        if stage_c is None
                        else int(
                            stage_c["candidate_action_sensitivity"]["raw_evidence"][
                                "rows"
                            ]
                        )
                    ),
                },
                "row_to_aggregate_reproduction": True,
                "source_commit": SOURCE_COMMIT,
                "source_freeze_commit": source_freeze,
                "contract_freeze_commit": source_freeze,
                "result_commit": None,
                "result_commit_binding_policy": CONTRACT.RESULT_COMMIT_BINDING_POLICY,
                "ancestry_validation": {
                    "requirements_ancestor_to_source": True,
                    "source_to_contract_freeze": True,
                    "result_commit_pending": True,
                },
                "nothing_running": not bool(_active_experiment_processes()),
                "prior_smoke_failure_custody": copy.deepcopy(
                    preexecution["prior_smoke_failure_custody"]
                ),
                "execution_correction_custody": copy.deepcopy(
                    execution_correction_custody
                ),
                "prohibition_counters": _prohibition_counters(),
            }
        )
        atomic_json(attempt / "receipts/persistence.json", persistence)
        runtime = {
            "total_s": time.time() - started,
            "training_and_evaluation_s": time.time() - fit_opened_at,
            "output_files": 0,
            "output_bytes": 0,
            "cpu_workers": CONTRACT.CPU_WORKER_BENCHMARK["selected_workers"],
            "ranker_seed_families": 1,
        }
        result_core = {
            "schema": "plan_aware_monotone_jepa_cost_v1.result.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "source_commit": SOURCE_COMMIT,
            "source_freeze_commit": source_freeze,
            "contract_freeze_commit": source_freeze,
            "result_commit": None,
            "result_commit_binding_policy": CONTRACT.RESULT_COMMIT_BINDING_POLICY,
            "ancestry_validation": {
                "requirements_ancestor_to_source": True,
                "source_to_contract_freeze": True,
                "result_commit_pending": True,
            },
            "contract_sha256": CONTRACT.CONTRACT_SHA256,
            "output_schema_sha256": CONTRACT.OUTPUT_SCHEMA_SHA256,
            "panel_bindings": copy.deepcopy(CONTRACT.PANEL_BINDINGS),
            "checkpoint_bindings": {
                "target_encoder": copy.deepcopy(CONTRACT.ENCODER_BINDING),
                "frozen_predictors": copy.deepcopy(CONTRACT.CHECKPOINT_BINDINGS),
                "trained_route_rankers": checkpoints,
            },
            "stage_execution": {
                "stage_a": "PASS_COMPLETE",
                "stage_b": "PASS_COMPLETE" if stage_b is not None else "NOT_RUN_TRUE_GATE_FAILED",
                "stage_c": (
                    "PASS_COMPLETE"
                    if stage_c is not None
                    else "NOT_RUN_NO_PROPRIOCEPTIVE_CONTRIBUTION"
                    if stage_b is not None
                    else "NOT_RUN_STAGE_B_NOT_AUTHORISED"
                ),
                "execution_correction_custody": copy.deepcopy(
                    execution_correction_custody
                ),
            },
            "metrics": {
                "binding": binding(attempt / "aggregates/metrics.json", relative_to=attempt),
                "content_digest": metrics["content_digest"],
                "stage_a_decisions": stage_a_decisions,
                "stage_a": summaries_by_role,
                "stage_b": stage_b,
                "stage_c": stage_c,
                "fit_optimization": training_receipt["fit_optimization"],
                "stage_a_raw_cost_rereduced": raw_cost_reduction,
                "stage_a_raw_cost_matched_comparisons": (
                    raw_cost_matched_comparisons
                ),
                "historical_raw_latent_goal_cosine_comparators": historical,
                "historical_comparators_recomputed": False,
                "raw_goal_cosine_executions": 0,
            },
            "primary_classification": primary,
            "secondary_classifications": secondaries,
            "next_experiment": next_experiment,
            "next_experiment_specification": (
                CONTRACT.next_experiment_specification_for_primary(primary)
            ),
            "requirements_workstream": {
                "classifications": list(CONTRACT.PRESERVED_REQUIREMENTS_CLASSIFICATIONS),
                "status": "REQUIREMENTS_ACQUISITION_REQUIRED",
                "changed_by_this_experiment": False,
            },
            "predecessor_narrative_authority": list(
                CONTRACT.PRESERVED_PREDECESSOR_NARRATIVE
            ),
            "predecessor_fact_authority": list(
                CONTRACT.PRESERVED_PREDECESSOR_FACTS
            ),
            "prior_smoke_failure_custody": copy.deepcopy(
                preexecution["prior_smoke_failure_custody"]
            ),
            "prohibition_counters": _prohibition_counters(),
            "runtime_and_storage": runtime,
            "nothing_running": not bool(_active_experiment_processes()),
        }
        result, report_bytes = _build_result_with_exact_storage(result_core, attempt=attempt)
        CONTRACT.validate_result_receipt(result)
        atomic_json(attempt / "result.json", result)
        atomic_bytes(attempt / "report.md", report_bytes)
        phase = "PREPUBLICATION_DEEP_CHECK"
        checks = deep_check(attempt)
        # The check result is returned to the caller rather than persisted into
        # its own manifest, avoiding a circular receipt and keeping the exact
        # output byte total fixed before atomic publication.
        phase = "CANONICAL_PUBLICATION"
        os.replace(attempt, CONTRACT.OUTPUT_ROOT)
        publication_happened = True
        phase = "POSTPUBLICATION_DEEP_CHECK"
        final_checks = deep_check(CONTRACT.OUTPUT_ROOT)
        terminal_payload = {
            "source_freeze_commit": source_freeze,
            "primary_classification": primary,
            "secondary_classifications": secondaries,
            "next_experiment": next_experiment,
            "stage_execution": result["stage_execution"],
            "result": binding(CONTRACT.OUTPUT_ROOT / "result.json"),
            "report": binding(CONTRACT.OUTPUT_ROOT / "report.md"),
            "runtime_and_storage": result["runtime_and_storage"],
            "deep_check": final_checks,
            "nothing_running": not bool(_active_experiment_processes()),
            "pass": True,
        }
        phase = "TRACKED_RESULT_PUBLICATION"
        _publish_tracked_result_and_report(CONTRACT.OUTPUT_ROOT)
        tracked_publication_happened = True
        phase = "COMPLETE"
        return terminal_payload
    except BaseException as exc:
        if tracked_publication_happened:
            for tracked_path in _tracked_publication_paths():
                tracked_path.unlink(missing_ok=True)
        candidate = CONTRACT.OUTPUT_ROOT if publication_happened else attempt
        if candidate.exists():
            failed = candidate.parent / (
                f".{CONTRACT.OUTPUT_ROOT.name}.failed-{time.time_ns()}-{os.getpid()}"
            )
            os.replace(candidate, failed)
            failure = attach_digest(
                {
                    "schema": "plan_aware_monotone_jepa_failure_v1",
                    "source_freeze_commit": source_freeze,
                    "phase": phase,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "partial_artifacts_reusable": False,
                    **copy.deepcopy(lifecycle_counters),
                    "nothing_running": not bool(_active_experiment_processes()),
                    "prohibition_counters": _prohibition_counters(),
                }
            )
            atomic_json(failed / "receipts/failure.json", failure)
        raise


_FROZEN_SCIENTIFIC_AUTHORITY_PATHS = tuple(
    Path(relative)
    for relative in CONTRACT.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS
)


def _scientific_authority_payloads() -> dict[str, bytes]:
    """Build outcome-blind authority bytes for correction equality checks."""

    CONTRACT.validate_no_duplicate_literal_dict_keys(Path(CONTRACT.__file__))
    route_role = CONTRACT.build_route_role_authority(ROOT)
    return {
        str(CONTRACT.TRACKED_PREREGISTRATION_PATH): (
            CONTRACT.build_preregistration_markdown().encode("utf-8")
        ),
        str(CONTRACT.TRACKED_CONTRACT_PATH): CONTRACT.contract_receipt_bytes(),
        str(CONTRACT.TRACKED_OUTPUT_SCHEMA_PATH): (
            CONTRACT.output_schema_receipt_bytes()
        ),
        str(CONTRACT.TRACKED_FIXTURE_PATH): (
            CONTRACT.evaluator_fixture_receipt_bytes()
        ),
        str(CONTRACT.TRACKED_ROUTE_ROLE_AUTHORITY_PATH): (
            CONTRACT.route_role_authority_receipt_bytes(route_role)
        ),
    }


def _git_path_rows(*args: str) -> list[str]:
    return [row for row in git_output(*args).splitlines() if row]


def _correction_refreeze_preflight(
    *,
    head: str,
    prior_smoke_failure_custody: Sequence[Mapping[str, Any]],
    authority_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Validate a correction base without opening any scientific outcomes."""

    if head == SOURCE_COMMIT or not prior_smoke_failure_custody:
        raise QualificationError(
            "correction refreeze requires a prior smoke-failure freeze descendant"
        )
    try:
        custody = CONTRACT.validate_prior_smoke_failure_custody(
            prior_smoke_failure_custody
        )
    except CONTRACT.ContractError as exc:
        raise QualificationError(str(exc)) from exc
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", SOURCE_COMMIT, head],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise QualificationError("correction-refreeze base does not descend from source")
    lineage = _git_path_rows("rev-list", "--parents", f"{SOURCE_COMMIT}..{head}")
    if not lineage or any(len(row.split()) != 2 for row in lineage):
        raise QualificationError("correction-refreeze history is not linear and no-merge")
    for record in custody:
        prior_freeze = str(record["source_freeze_commit"])
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", prior_freeze, head],
            cwd=ROOT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode:
            raise QualificationError(
                "prior smoke freeze is not an ancestor of correction-refreeze base"
            )

    allowed_paths = {
        *(
            str(path)
            for path in CONTRACT.SOURCE_CLOSURE_DEFAULT_PATHS
            if str(path).endswith(".py")
        ),
        *map(str, CONTRACT.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS),
        str(CONTRACT.TRACKED_SOURCE_CLOSURE_PATH),
    }
    changed_paths = sorted(
        {
            *_git_path_rows("diff", "--name-only", SOURCE_COMMIT, head),
            *_git_path_rows("diff", "--name-only"),
            *_git_path_rows("diff", "--cached", "--name-only"),
            *_git_path_rows("ls-files", "--others", "--exclude-standard"),
        }
    )
    outside_domain = sorted(set(changed_paths) - allowed_paths)
    if outside_domain:
        raise QualificationError(
            "correction refreeze changed paths outside the source-closure/authority "
            f"domain: {outside_domain}"
        )

    authority_paths = set(authority_payloads)
    if authority_paths != {str(path) for path in _FROZEN_SCIENTIFIC_AUTHORITY_PATHS}:
        raise QualificationError("correction scientific-authority payload set drift")
    mutable_correction_observed_by_archive: dict[str, bool] = {}
    for record in custody:
        closure_binding = record.get("source_closure")
        if not isinstance(closure_binding, Mapping):
            raise QualificationError(
                "eligible smoke retry requires its archived source-closure snapshot"
            )
        closure_path = Path(str(closure_binding["path"]))
        closure = load_json(closure_path)
        observed_closure_binding = {
            **binding(closure_path),
            "content_digest": closure.get("content_digest"),
        }
        if observed_closure_binding != dict(closure_binding):
            raise QualificationError(
                f"prior smoke source-closure binding drift: {closure_path}"
            )
        try:
            CONTRACT.validate_source_closure(closure)
        except CONTRACT.ContractError as exc:
            raise QualificationError(
                f"prior smoke source closure is invalid: {closure_path}: {exc}"
            ) from exc
        rows = {str(row["path"]): row for row in closure["rows"]}
        for relative, payload in authority_payloads.items():
            expected = rows.get(relative)
            if (
                not isinstance(expected, Mapping)
                or expected.get("sha256") != hashlib.sha256(payload).hexdigest()
                or expected.get("bytes") != len(payload)
            ):
                raise QualificationError(
                    "scientific authority changed after fit outcomes opened: "
                    f"{relative}"
                )
        correction_observed = False
        for relative in (
            str(path)
            for path in CONTRACT.SOURCE_CLOSURE_DEFAULT_PATHS
            if str(path).endswith(".py")
        ):
            expected = rows.get(relative)
            if not isinstance(expected, Mapping):
                raise QualificationError(
                    f"prior smoke closure omits mutable Python path: {relative}"
                )
            current = ROOT / relative
            if (
                not current.is_file()
                or current.stat().st_size != int(expected["bytes"])
                or sha256_file(current) != str(expected["sha256"])
            ):
                correction_observed = True
                break
        mutable_correction_observed_by_archive[str(record["archive_path"])] = (
            correction_observed
        )
    if not all(mutable_correction_observed_by_archive.values()):
        raise QualificationError(
            "correction refreeze contains no implementation/test correction relative "
            "to every prior smoke archive"
        )
    return {
        "mode": "VALIDATED_TRAINING_SMOKE_CORRECTION_REFREEZE",
        "base_head": head,
        "prior_smoke_failure_custody": copy.deepcopy(custody),
        "changed_paths": changed_paths,
        "scientific_authorities_unchanged": True,
        "mutable_correction_observed_by_archive": (
            mutable_correction_observed_by_archive
        ),
        "files_reused": 0,
        "required_enclosing_commit_subject": CONTRACT.CONTRACT_FREEZE_COMMIT_SUBJECT,
    }


def _refresh_correction_freeze_authorities(
    authority_payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    """Atomically replace each generated authority with prevalidated bytes."""

    for relative in _FROZEN_SCIENTIFIC_AUTHORITY_PATHS:
        atomic_bytes(_tracked_path(relative), authority_payloads[str(relative)])
    closure = CONTRACT.build_source_closure(ROOT, require_complete=True)
    CONTRACT.validate_source_closure(closure)
    atomic_bytes(
        _tracked_path(CONTRACT.TRACKED_SOURCE_CLOSURE_PATH),
        CONTRACT.source_closure_receipt_bytes(closure),
    )
    return closure


def freeze_contract() -> dict[str, Any]:
    head = git_output("rev-parse", "HEAD")
    if head == CONTRACT.INITIAL_EXECUTION_FREEZE_COMMIT:
        # Prepare only the separate execution-amendment authority suite.  The
        # original preregistration/contract/schema/fixture/role/closure bytes
        # are immutable and are never regenerated in this branch.
        try:
            CONTRACT.validate_base_scientific_authorities(ROOT)
            archive = CONTRACT.validate_execution_correction_archive()
        except CONTRACT.ContractError as exc:
            raise QualificationError(str(exc)) from exc
        tracked_changes = {
            path
            for path in git_output("diff", "--name-only").splitlines()
            if path
        }
        outside = sorted(
            tracked_changes - set(CONTRACT.EXECUTION_CORRECTION_ALLOWED_CHANGED_PATHS)
        )
        if outside:
            raise QualificationError(
                f"execution-correction preparation has unrelated tracked edits: {outside}"
            )
        failed = sorted(
            path.resolve()
            for path in CONTRACT.OUTPUT_ROOT.parent.glob(
                f".{CONTRACT.OUTPUT_ROOT.name}.failed-*"
            )
        )
        attempts = sorted(
            CONTRACT.OUTPUT_ROOT.parent.glob(
                f".{CONTRACT.OUTPUT_ROOT.name}.attempt-*"
            )
        )
        if (
            failed != [CONTRACT.EXECUTION_CORRECTION_FAILED_ARCHIVE.resolve()]
            or attempts
            or CONTRACT.OUTPUT_ROOT.exists()
            or _active_experiment_processes()
        ):
            raise QualificationError(
                "execution-correction amendment namespace is not fresh"
            )
        paths = CONTRACT.write_execution_correction_authorities(ROOT)
        closure = CONTRACT.load_and_validate_execution_correction_source_closure(
            _tracked_path(CONTRACT.TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH)
        )
        return {
            "freeze_mode": "EXECUTION_CORRECTION_AMENDMENT_PREPARATION",
            "base_head": head,
            "failed_archive_custody": archive,
            "files_reused": 0,
            "required_enclosing_commit_subject": (
                CONTRACT.EXECUTION_CORRECTION_FREEZE_COMMIT_SUBJECT
            ),
            "amendment_authorities": {
                label: binding(path) for label, path in paths.items()
            },
            "amendment_source_closure_rows": closure["row_count"],
            "scientific_authority_contract_sha256": (
                CONTRACT.SCIENTIFIC_AUTHORITY_CONTRACT_SHA256
            ),
            "pass": True,
        }
    prior_smoke_failure_custody = _validated_prior_smoke_failure_custody(
        CONTRACT.OUTPUT_ROOT,
        source_freeze=head,
        allow_current_freeze_as_correction_base=True,
    )
    correction_preflight: dict[str, Any] | None = None
    authority_payloads: dict[str, bytes] | None = None
    if head == SOURCE_COMMIT:
        if prior_smoke_failure_custody:
            raise QualificationError(
                "initial source freeze cannot consume a prior smoke-failure archive"
            )
    else:
        authority_payloads = _scientific_authority_payloads()
        correction_preflight = _correction_refreeze_preflight(
            head=head,
            prior_smoke_failure_custody=prior_smoke_failure_custody,
            authority_payloads=authority_payloads,
        )
    if head == SOURCE_COMMIT and git_output("status", "--porcelain"):
        # The intended prospective implementation is necessarily untracked at
        # this point; reject only unrelated tracked modifications.
        tracked = git_output("status", "--porcelain=v1", "--untracked-files=no")
        if tracked:
            raise QualificationError(f"freeze has tracked pre-existing edits: {tracked}")
    if correction_preflight is None:
        CONTRACT.write_route_role_authority(
            ROOT, path=_tracked_path(CONTRACT.TRACKED_ROUTE_ROLE_AUTHORITY_PATH)
        )
        CONTRACT.write_preregistration(
            _tracked_path(CONTRACT.TRACKED_PREREGISTRATION_PATH)
        )
        CONTRACT.write_contract(_tracked_path(CONTRACT.TRACKED_CONTRACT_PATH))
        CONTRACT.write_output_schema(
            _tracked_path(CONTRACT.TRACKED_OUTPUT_SCHEMA_PATH)
        )
        CONTRACT.write_evaluator_fixture(
            _tracked_path(CONTRACT.TRACKED_FIXTURE_PATH)
        )
        closure = CONTRACT.build_source_closure(ROOT, require_complete=True)
        CONTRACT.write_source_closure(
            closure, _tracked_path(CONTRACT.TRACKED_SOURCE_CLOSURE_PATH)
        )
        freeze_mode = "INITIAL_DIRECT_CONTRACT_FREEZE_PREPARATION"
    else:
        assert authority_payloads is not None
        closure = _refresh_correction_freeze_authorities(authority_payloads)
        freeze_mode = str(correction_preflight["mode"])
    _validate_frozen_authorities()
    return {
        "freeze_mode": freeze_mode,
        "base_head": head,
        "prior_smoke_failure_custody": copy.deepcopy(
            prior_smoke_failure_custody
        ),
        "files_reused": 0,
        "required_enclosing_commit_subject": CONTRACT.CONTRACT_FREEZE_COMMIT_SUBJECT,
        "contract": binding(_tracked_path(CONTRACT.TRACKED_CONTRACT_PATH)),
        "output_schema": binding(_tracked_path(CONTRACT.TRACKED_OUTPUT_SCHEMA_PATH)),
        "fixture": binding(_tracked_path(CONTRACT.TRACKED_FIXTURE_PATH)),
        "source_closure": binding(_tracked_path(CONTRACT.TRACKED_SOURCE_CLOSURE_PATH)),
        "source_closure_rows": closure["row_count"],
        "pass": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("execute")
    check_parser = subparsers.add_parser("check")
    check_parser.add_argument("--output-root", type=Path, default=CONTRACT.OUTPUT_ROOT)
    args = parser.parse_args()
    if args.command == "freeze":
        value = freeze_contract()
    elif args.command == "execute":
        value = execute()
    else:
        value = deep_check(args.output_root.resolve())
    print(json.dumps(value, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
