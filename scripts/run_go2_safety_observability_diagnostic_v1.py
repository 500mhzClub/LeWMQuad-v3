#!/usr/bin/env python3
"""Build the exact historical safety-observability diagnostic panel.

This development-only workflow has one narrow purpose: retain the physical
10 Hz safety evidence that the four RGB target instants do not themselves
encode as labels.  It never changes a scientific label, state, source frame,
latent, scorer, or predecessor branch row.

The historical 24 x 12 panel is fixed before execution.  All 288 branches are
replayed once into a dedicated namespace because the predecessor trace lacks
deterministic contact type.  The twelve earlier v1.3 calibration traces are
bound as lineage and compared on their shared fields, but are not substituted
for the new diagnostic records.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.oracle import go2_scorer_failure_attribution_v1_contract as CONTRACT  # noqa: E402
from lewm.oracle import go2_scorer_fit_oracle_v1_3_contract as V13_CONTRACT  # noqa: E402
from lewm.oracle import go2_branch_oracle_v1_2 as ORACLE_V12  # noqa: E402
from scripts import build_go2_branch_corpus_v1_2 as BUILDER  # noqa: E402
from scripts import run_go2_scorer_fit_oracle_v1_3 as V13  # noqa: E402


OUT_ROOT = ROOT / CONTRACT.GENERATED_ROOT
CONTRACT_NAME = "diagnostic_contract.json"
SAFETY_SUBDIR = CONTRACT.SAFETY_OBSERVABILITY_ROOT.relative_to(
    CONTRACT.GENERATED_ROOT)
PLAN_NAME = SAFETY_SUBDIR / "plan.json"
ATTEMPTS_NAME = SAFETY_SUBDIR / "attempts"
TRACE_ROWS_NAME = SAFETY_SUBDIR / "trace_rows"
TERMINAL_NAME = SAFETY_SUBDIR / "terminal.json"
AUDIT_NAME = SAFETY_SUBDIR / "audit.json"

PLAN_SCHEMA = "go2_safety_observability_diagnostic_v1_plan_v1"
PLAN_SELF_KEY = "historical_calibration_trace_plan_digest"
ATTEMPT_SCHEMA = "go2_safety_observability_diagnostic_v1_attempt_v1"
ATTEMPT_SELF_KEY = "attempt_digest"
TRACE_ROW_SCHEMA = "go2_safety_observability_diagnostic_v1_trace_row_v1"
TRACE_ROW_SELF_KEY = "diagnostic_trace_row_digest"
TERMINAL_SCHEMA = "go2_safety_observability_diagnostic_v1_trace_manifest_v1"
TERMINAL_SELF_KEY = "diagnostic_trace_manifest_digest"
AUDIT_SCHEMA = "go2_safety_observability_diagnostic_v1_audit_v1"
AUDIT_SELF_KEY = "safety_observability_audit_digest"

EXPECTED_STATES = CONTRACT.EXPECTED_STATES
EXPECTED_CANDIDATES = tuple(range(12))
EXPECTED_BRANCHES = CONTRACT.EXPECTED_BRANCHES
EXPECTED_PRIOR_LINEAGE_TRACES = CONTRACT.SAFETY_OBSERVABILITY_CONTRACT[
    "prior_trace_references_compared_as_lineage"]
EXPECTED_NEW_REPLAYS = CONTRACT.REPLAY_TRACES
TICKS = tuple(CONTRACT.POLICY_TICKS)
SAMPLED_ENDPOINT_TICKS = tuple(CONTRACT.HORIZON_SAMPLE_TICKS)
PRE_H1_TICKS = frozenset(range(0, 4))
STRICTLY_BETWEEN_SAMPLED_HORIZON_TICKS = frozenset({
    *range(5, 9), *range(10, 14), *range(15, 19),
})
CONTACT_CATEGORIES = (
    "FOOT_NONGROUND_ENVIRONMENT",
    "NONFOOT_GROUND",
    "NONFOOT_NONGROUND_ENVIRONMENT",
)
COMPONENTS = ("contact", "clearance", "stuck", "unsafe_termination")
TIMING_COMPONENTS = (*COMPONENTS, "completion")
STATUS = CONTRACT.STATUS

AGGREGATE_LABEL_FIELDS = (
    "contact_fraction", "clearance_cost", "stuck_fraction", "fall",
    "safety", "completion",
)
AGGREGATE_ABS_TOLERANCES = {
    "contact_fraction": 0.0,
    "clearance_cost": V13.HORIZON_POSE_ATOL,
    "stuck_fraction": 0.0,
    "fall": 0.0,
    "safety": V13.HORIZON_POSE_ATOL,
    "completion": 0.0,
}
AGGREGATE_FORMULA_ATOL = 1e-12
V2_LABEL_SOURCE = "FROZEN_V2_FINITE_LABEL_PROJECTION"
V13_LABEL_SOURCE = "FROZEN_V1_3_REPLAY_OVERLAY_LABEL_PROJECTION"
EXPECTED_V2_LABEL_REFERENCES = EXPECTED_BRANCHES - EXPECTED_PRIOR_LINEAGE_TRACES
EXPECTED_V13_LABEL_REFERENCES = EXPECTED_PRIOR_LINEAGE_TRACES


class DiagnosticError(RuntimeError):
    """The frozen diagnostic identity, evidence, or attempt contract changed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            result.update(block)
    return result.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticError(message)


def is_digest(value: Any) -> bool:
    return bool(
        isinstance(value, str) and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def signed(payload: Mapping[str, Any], self_key: str) -> dict[str, Any]:
    value = copy.deepcopy(dict(payload))
    require(self_key not in value, f"{self_key} already exists")
    value[self_key] = digest(value)
    return value


def validate_signed(value: Mapping[str, Any], self_key: str,
                    label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} is not an object")
    result = copy.deepcopy(dict(value))
    claimed = result.pop(self_key, None)
    require(is_digest(claimed) and claimed == digest(result),
            f"{label} self digest changed")
    result[self_key] = claimed
    return result


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"{label} is absent or not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"{label} is invalid JSON") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def _managed_root(out_root: Path = OUT_ROOT) -> Path:
    root = out_root.absolute()
    if root == OUT_ROOT.absolute():
        require(root.is_symlink(), "registered diagnostic output alias is absent")
        raw = root.readlink()
        target = raw if raw.is_absolute() else root.parent / raw
        require(
            target == CONTRACT.REGISTERED_GENERATED_TARGET_ROOT
            and target.is_dir() and not target.is_symlink(),
            "registered diagnostic output target changed",
        )
    else:
        require(root.is_dir() and not root.is_symlink(),
                "synthetic diagnostic output root is invalid")
    return root


def _output_path(relative: str | Path, *, out_root: Path = OUT_ROOT) -> Path:
    relative = Path(relative)
    require(not relative.is_absolute() and relative.parts
            and ".." not in relative.parts,
            "diagnostic output must be a relative descendant")
    root = _managed_root(out_root)
    target = (root / relative).absolute()
    require(root in target.parents, "diagnostic output escaped its root")
    cursor = target.parent
    while cursor != root.parent and cursor.exists():
        require(cursor == root or not cursor.is_symlink(),
                "nested diagnostic output ancestor is a symlink")
        if cursor == root:
            break
        cursor = cursor.parent
    return target


def _atomic_json(path: Path, value: Mapping[str, Any], *, out_root: Path,
                 idempotent: bool) -> None:
    try:
        relative = path.absolute().relative_to(out_root.absolute())
    except ValueError as exc:
        raise DiagnosticError("diagnostic write escaped its root") from exc
    target = _output_path(relative, out_root=out_root)
    raw = _json_bytes(value)
    if target.exists() or target.is_symlink():
        require(idempotent and target.is_file() and not target.is_symlink()
                and target.read_bytes() == raw,
                f"refusing to replace diagnostic artifact {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.parent.is_symlink(),
            "diagnostic output parent is a symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        require(not target.exists() and not target.is_symlink(),
                "diagnostic output appeared concurrently")
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def plan_path(out_root: Path = OUT_ROOT) -> Path:
    return out_root / PLAN_NAME


def attempt_path(identity: str, out_root: Path = OUT_ROOT) -> Path:
    require(is_digest(identity), "attempt identity is malformed")
    return out_root / ATTEMPTS_NAME / f"{identity}.json"


def trace_row_path(identity: str, out_root: Path = OUT_ROOT) -> Path:
    require(is_digest(identity), "trace-row identity is malformed")
    return out_root / TRACE_ROWS_NAME / f"{identity}.json"


def terminal_path(out_root: Path = OUT_ROOT) -> Path:
    return out_root / TERMINAL_NAME


def audit_path(out_root: Path = OUT_ROOT) -> Path:
    return out_root / AUDIT_NAME


def _repo_relative(path: Path) -> str:
    try:
        return str(path.absolute().relative_to(ROOT.absolute()))
    except ValueError as exc:
        raise DiagnosticError(f"source artifact escapes repository: {path}") from exc


def _git_output(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=root, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DiagnosticError(f"cannot bind diagnostic source: {exc}") from exc
    return result.stdout.strip()


def build_source_closure(*, root: Path = ROOT) -> dict[str, Any]:
    """Bind only the contract's explicit clean, committed eight-file closure."""

    status = _git_output(root, "status", "--porcelain=v1")
    require(status == "", "diagnostic execution requires clean committed source")
    commit = _git_output(root, "rev-parse", "HEAD")
    files: dict[str, Any] = {}
    for relative in CONTRACT.SOURCE_CLOSURE_PATHS:
        path = root / relative
        require(path.is_file() and not path.is_symlink(),
                f"source-closure path changed: {relative}")
        files[relative] = {
            "path": relative,
            "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }
    unsigned = {
        "schema": CONTRACT.SOURCE_CLOSURE_SCHEMA,
        "source_repository_commit": commit,
        "source_repository_clean": True,
        "git_status_porcelain_v1": "",
        "files": files,
    }
    return CONTRACT.validate_source_closure({
        **unsigned,
        CONTRACT.SOURCE_CLOSURE_SELF_KEY: CONTRACT.canonical_digest(unsigned),
    })


def load_bound_contract(*, out_root: Path = OUT_ROOT,
                        root: Path = ROOT) -> dict[str, Any]:
    """Load the installed contract and require equality to current source bytes."""

    _managed_root(out_root)
    value = CONTRACT.validate_contract(_read_json(
        out_root / CONTRACT_NAME, "failure-attribution diagnostic contract"))
    require(value["source_closure"] == build_source_closure(root=root),
            "installed contract does not bind current clean source bytes")
    return value


def _historical_state_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: state.get(key) for key in (
            "state_identity_digest", "state_id", "scene_id", "family", "stratum"
        )
    }


def _validate_exact_historical_inventory(
        corpus: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states = [dict(row) for row in corpus["state_manifest"]["states"]
              if row.get("split_role") == "calibration"]
    rows = [dict(row) for row in corpus["rows"]
            if row.get("split_role") == "calibration"]
    expected_states = sorted(
        (asdict(row) for row in V13_CONTRACT.HISTORICAL_CALIBRATION_STATES),
        key=lambda row: row["state_identity_digest"],
    )
    observed_states = sorted(
        (_historical_state_projection(row) for row in states),
        key=lambda row: row["state_identity_digest"],
    )
    require(observed_states == expected_states and len(states) == EXPECTED_STATES,
            "historical calibration state inventory changed")
    by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row.get("state_identity_digest"))].append(row)
    require(
        len(rows) == EXPECTED_BRANCHES
        and len({row.get("branch_identity_digest") for row in rows})
        == EXPECTED_BRANCHES
        and all(is_digest(row.get("branch_identity_digest")) for row in rows)
        and set(by_state)
        == {row["state_identity_digest"] for row in expected_states}
        and all(sorted(int(item["candidate_index"]) for item in bank)
                == list(EXPECTED_CANDIDATES) for bank in by_state.values()),
        "historical calibration branch inventory is not exact 24x12",
    )
    rows.sort(key=lambda row: row["branch_identity_digest"])
    states.sort(key=lambda row: row["state_identity_digest"])
    return states, rows


def _trace_required_fields() -> frozenset[str]:
    return frozenset({
        "global_tick", "block_index", "tick_in_block", "episode_id",
        "episode_step", "timestamp_ns", "requested_command",
        "post_slew_command", "position_world_xyz_m", "quaternion_world_wxyz",
        "rpy_world_rad", "xy", "yaw", "z", "nearest_cell_id",
        "nearest_cell_distance_m", "located", "accepted_cell_id", "cell_id",
        "goal_cell_id", "raw_bfs_to_goal", "masked_bfs_to_goal", "geodesic_m",
        "graph_status", "at_goal_cell", "clearance_m", "clearance_deficit",
        "stuck", "disallowed_contacts", "disallowed_contact", "termination",
        "terminated", "nan",
    })


def validate_trace_structure(trace: Mapping[str, Any], *,
                             require_contact_types: bool) -> dict[str, Any]:
    require(isinstance(trace, Mapping)
            and trace.get("schema") == V13_CONTRACT.TRACE_SCHEMA
            and trace.get("blocks_completed") == 4
            and trace.get("nan") is False
            and isinstance(trace.get("start"), Mapping)
            and isinstance(trace.get("ticks"), list)
            and len(trace["ticks"]) == len(TICKS),
            "diagnostic trace is not one complete v1.3 horizon")
    requested = np.asarray(trace.get("requested"), dtype=np.float64)
    post_slew = np.asarray(trace.get("post_slew"), dtype=np.float64)
    require(isinstance(trace.get("candidate"), str)
            and isinstance(trace.get("primitives"), list)
            and len(trace["primitives"]) == 4
            and requested.shape == post_slew.shape == (4, 5, 3)
            and np.all(np.isfinite(requested))
            and np.all(np.isfinite(post_slew)),
            "diagnostic trace action surface changed")
    start = trace["start"]
    require(
        _trace_required_fields() <= set(start)
        and start.get("global_tick") is None
        and start.get("block_index") is None
        and start.get("tick_in_block") is None
        and type(start.get("episode_id")) is int
        and type(start.get("episode_step")) is int
        and type(start.get("timestamp_ns")) is int
        and start.get("nan") is False,
        "diagnostic trace start sample changed",
    )
    completion = False
    for expected, row in enumerate(trace["ticks"]):
        if isinstance(row, Mapping):
            completion = completion or bool(row.get("at_goal_cell"))
        require(isinstance(row, Mapping)
                and _trace_required_fields() <= set(row)
                and type(row.get("global_tick")) is int
                and type(row.get("block_index")) is int
                and type(row.get("tick_in_block")) is int
                and type(row.get("episode_id")) is int
                and type(row.get("episode_step")) is int
                and type(row.get("timestamp_ns")) is int
                and row.get("global_tick") == expected
                and row.get("block_index") == expected // 5
                and row.get("tick_in_block") == expected % 5
                and row.get("completion_latched") in {True, False}
                and row.get("completion_latched") is completion
                and row.get("episode_id") == start["episode_id"]
                and row.get("episode_step") == start["episode_step"] + expected + 1
                and row.get("timestamp_ns")
                == start["timestamp_ns"] + (expected + 1) * 100_000_000
                and row.get("nan") is False
                and isinstance(row.get("disallowed_contacts"), int)
                and not isinstance(row.get("disallowed_contacts"), bool)
                and row["disallowed_contacts"] >= 0
                and row.get("disallowed_contact")
                is bool(row["disallowed_contacts"] > 0)
                and isinstance(row.get("termination"), Mapping)
                and set(row["termination"])
                == {"fall", "out_of_bounds", "tipped", "nan"},
                f"diagnostic tick {expected} changed")
        require(
            all(type(row["termination"][key]) is bool
                for key in row["termination"])
            and type(row.get("stuck")) is bool
            and type(row.get("terminated")) is bool
            and row["terminated"] is bool(
                row["termination"]["fall"]
                or row["termination"]["out_of_bounds"]
                or row["termination"]["tipped"])
            and isinstance(row.get("clearance_m"), (int, float))
            and not isinstance(row.get("clearance_m"), bool)
            and math.isfinite(float(row["clearance_m"]))
            and isinstance(row.get("clearance_deficit"), (int, float))
            and not isinstance(row.get("clearance_deficit"), bool)
            and math.isfinite(float(row["clearance_deficit"]))
            and math.isclose(
                float(row["clearance_deficit"]),
                ORACLE_V12.clearance_deficit(float(row["clearance_m"])),
                rel_tol=0.0, abs_tol=1e-12,
            ),
            f"diagnostic safety scalars changed at tick {expected}",
        )
        if require_contact_types:
            details = row.get("disallowed_contact_details")
            categories = row.get("disallowed_contact_types")
            require(row.get("contact_type_evidence_status") == "COMPLETE"
                    and isinstance(details, list)
                    and len(details) == row["disallowed_contacts"]
                    and isinstance(categories, list)
                    and all(isinstance(item, Mapping) for item in details)
                    and categories == sorted({item.get("category") for item in details})
                    and all(
                        isinstance(item, Mapping)
                        and set(item) == {
                            "category", "robot_link_id", "robot_link_name",
                            "environment_link_id", "environment_link_name",
                            "force_magnitude_n",
                        }
                        and item.get("category") in CONTACT_CATEGORIES
                        and type(item.get("robot_link_id")) is int
                        and type(item.get("environment_link_id")) is int
                        and (item.get("robot_link_name") is None
                             or isinstance(item.get("robot_link_name"), str))
                        and (item.get("environment_link_name") is None
                             or isinstance(item.get("environment_link_name"), str))
                        and (item.get("force_magnitude_n") is None or (
                            isinstance(item.get("force_magnitude_n"), (int, float))
                            and not isinstance(item.get("force_magnitude_n"), bool)
                            and math.isfinite(float(item["force_magnitude_n"]))
                            and float(item["force_magnitude_n"])
                            > ORACLE_V12.CONTACT_FORCE_THRESHOLD_N
                        ))
                        for item in details
                    ),
                    f"diagnostic contact types changed at tick {expected}")
    canonical_bytes(trace)
    return copy.deepcopy(dict(trace))


def _finite_aggregate_label_projection(
        value: Mapping[str, Any], label: str) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{label} is not an object")
    result: dict[str, float] = {}
    for field in AGGREGATE_LABEL_FIELDS:
        item = value.get(field)
        require(isinstance(item, (int, float)) and not isinstance(item, bool)
                and math.isfinite(float(item))
                and 0.0 <= float(item) <= 1.0,
                f"{label} {field} is not a finite target in [0,1]")
        result[field] = float(item)
    require(result["fall"] in {0.0, 1.0}
            and result["completion"] in {0.0, 1.0},
            f"{label} binary targets changed")
    expected_safety = max(
        result["fall"],
        (result["contact_fraction"] + result["clearance_cost"]
         + result["stuck_fraction"]) / 3.0,
    )
    require(math.isclose(
        result["safety"], expected_safety,
        rel_tol=0.0, abs_tol=AGGREGATE_FORMULA_ATOL,
    ), f"{label} no longer satisfies the frozen safety formula")
    canonical_bytes(result)
    return result


def _raw_v2_aggregate_label_projection(
        row: Mapping[str, Any]) -> dict[str, float | None]:
    projection = {field: row.get(field) for field in AGGREGATE_LABEL_FIELDS}
    missing = [field for field, value in projection.items() if value is None]
    require(not missing or len(missing) == len(AGGREGATE_LABEL_FIELDS),
            "V2 aggregate label projection is partially missing")
    if not missing:
        return _finite_aggregate_label_projection(
            projection, "frozen V2 aggregate label projection")
    require(row.get("valid") is False
            and row.get("invalid_reason")
            == "unlocatable_or_unreachable_geodesic",
            "null V2 aggregate projection is not an exact registered refusal")
    canonical_bytes(projection)
    return projection


def _load_prior_calibration_traces(
        *, v13_root: Path = V13.OUT_ROOT,
        ) -> dict[str, dict[str, Any]]:
    manifest_path = v13_root / "replay_overlay_manifest.json"
    manifest = _read_json(manifest_path, "v1.3 replay overlay manifest")
    V13._validate_self_digest(manifest, V13.REPLAY_OVERLAY_MANIFEST_SELF_KEY)
    require(manifest.get("historical_calibration_overlay_count")
            == EXPECTED_PRIOR_LINEAGE_TRACES,
            "prior calibration overlay count changed")
    calibration_identities = {
        row.branch_identity_digest
        for row in V13_CONTRACT.FAILED_BRANCH_IDENTITIES
        if row.split_role == "calibration"
    }
    require(len(calibration_identities) == EXPECTED_PRIOR_LINEAGE_TRACES,
            "frozen prior calibration identity set changed")
    result: dict[str, dict[str, Any]] = {}
    for entry in manifest.get("rows", []):
        identity = str(entry.get("source_branch_identity_digest", ""))
        require(is_digest(identity), "prior overlay identity is malformed")
        if identity not in calibration_identities:
            continue
        expected_path = v13_root / "replay_overlays" / f"{identity}.json"
        path = ROOT / str(entry.get("path", ""))
        require(path.absolute() == expected_path.absolute()
                and file_sha256(path) == entry.get("sha256"),
                "prior overlay path or bytes changed")
        overlay = _read_json(path, "prior v1.3 replay overlay")
        V13._validate_self_digest(overlay, V13.REPLAY_OVERLAY_SELF_KEY)
        if overlay.get("split_role") != "calibration":
            continue
        trace = validate_trace_structure(
            overlay.get("trace", {}), require_contact_types=False)
        target = _finite_aggregate_label_projection(
            overlay.get("labels", {}),
            "frozen v1.3 replay-overlay aggregate label projection",
        )
        marker = _read_json(
            v13_root / "replay_attempts" / f"{identity}.json",
            "prior v1.3 attempt marker",
        )
        V13._validate_self_digest(marker, "attempt_digest")
        require(marker.get("identity_digest") == identity
                and marker.get("kind") == "replay"
                and overlay.get("attempt_digest") == marker.get("attempt_digest"),
                "prior replay attempt binding changed")
        result[identity] = {
            "path": _repo_relative(path),
            "sha256": entry["sha256"],
            "replay_overlay_digest": overlay[V13.REPLAY_OVERLAY_SELF_KEY],
            "attempt_digest": marker["attempt_digest"],
            "trace_digest": digest(trace),
            "contact_type_evidence": "NOT_RETAINED_IN_V1_3_TRACE",
            "aggregate_label_projection": target,
            "aggregate_label_projection_digest": digest(target),
            "trace": trace,
        }
    require(len(result) == EXPECTED_PRIOR_LINEAGE_TRACES,
            "prior calibration trace identity set changed")
    return result


def _prebranch_witness(row: Mapping[str, Any]) -> dict[str, Any]:
    return V13._normalised_prebranch_witness(
        proprio=row.get("proprio"), control=row.get("control"),
        action_context_blocks=row.get("action_context_blocks"),
        previous_applied_command=row.get("previous_applied_command"),
    )


def _plan_entry(row: Mapping[str, Any],
                prior: Mapping[str, Any] | None) -> dict[str, Any]:
    context_poses = [frame.get("camera_pose_world")
                     for frame in row.get("context_frames", [])]
    horizon_poses = [frame.get("camera_pose_world")
                     for frame in row.get("horizon_frames", [])]
    witness = _prebranch_witness(row)
    raw_v2_labels = _raw_v2_aggregate_label_projection(row)
    if all(value is not None for value in raw_v2_labels.values()):
        require(prior is None,
                "finite V2 target unexpectedly has a v1.3 refusal overlay")
        target_source = V2_LABEL_SOURCE
        target_labels = _finite_aggregate_label_projection(
            raw_v2_labels, "finite V2 replay target")
    else:
        require(prior is not None,
                "null V2 target lacks its frozen v1.3 replay overlay")
        target_source = V13_LABEL_SOURCE
        target_labels = _finite_aggregate_label_projection(
            prior.get("aggregate_label_projection", {}),
            "v1.3 replay-overlay replay target",
        )
        require(digest(target_labels)
                == prior.get("aggregate_label_projection_digest"),
                "v1.3 replay-overlay target digest changed")
    payload = {
        key: copy.deepcopy(row.get(key)) for key in (
            "branch_identity_digest", "branch_row_digest",
            "assignment_identity_digest", "state_id", "state_identity_digest",
            "scene_id", "family", "stratum", "candidate_index", "candidate",
            "primitives", "goal", "snapshot_digest",
            "realised_requested_prefix", "post_slew",
        )
    }
    payload.update({
        "source_context_camera_poses": context_poses,
        "source_context_pose_digest": digest(context_poses),
        "source_horizon_camera_poses": horizon_poses,
        "source_horizon_pose_digest": digest(horizon_poses),
        "source_prebranch_witness": witness,
        "source_prebranch_witness_digest": digest(witness),
        "source_v2_aggregate_label_projection": raw_v2_labels,
        "source_v2_aggregate_label_projection_digest": digest(raw_v2_labels),
        "frozen_replay_target_source_kind": target_source,
        "frozen_replay_target_projection": target_labels,
        "frozen_replay_target_projection_digest": digest(target_labels),
        "execution_disposition": "NEW_DIAGNOSTIC_REPLAY_REQUIRED",
        "prior_v1_3_trace": (
            None if prior is None else {
                key: prior[key] for key in (
                    "path", "sha256", "replay_overlay_digest", "attempt_digest",
                    "trace_digest", "contact_type_evidence",
                    "aggregate_label_projection_digest",
                )
            }
        ),
    })
    return payload


def build_plan(corpus: Mapping[str, Any],
               prior_traces: Mapping[str, Mapping[str, Any]],
               diagnostic_contract: Mapping[str, Any]) -> dict[str, Any]:
    states, rows = _validate_exact_historical_inventory(corpus)
    row_identities = {row["branch_identity_digest"] for row in rows}
    require(set(prior_traces) <= row_identities
            and len(prior_traces) == EXPECTED_PRIOR_LINEAGE_TRACES,
            "prior traces are not an exact historical subset")
    entries = [_plan_entry(row, prior_traces.get(row["branch_identity_digest"]))
               for row in rows]
    label_source_counts = Counter(
        entry["frozen_replay_target_source_kind"] for entry in entries)
    require(label_source_counts == Counter({
        V2_LABEL_SOURCE: EXPECTED_V2_LABEL_REFERENCES,
        V13_LABEL_SOURCE: EXPECTED_V13_LABEL_REFERENCES,
    }), "frozen replay-target source split changed")
    contract_value = CONTRACT.validate_contract(diagnostic_contract)
    contract_digest = contract_value[CONTRACT.CONTRACT_SELF_KEY]
    payload = {
        "schema": PLAN_SCHEMA,
        "status": STATUS,
        "contract_digest": contract_digest,
        "source_closure_digest": contract_value["source_closure"][
            CONTRACT.SOURCE_CLOSURE_SELF_KEY],
        "contract": contract_value,
        "source_v2_corpus_digest": V13_CONTRACT.FROZEN_CORPUS_DIGEST,
        "source_state_manifest_digest": V13_CONTRACT.FROZEN_STATE_MANIFEST_DIGEST,
        "source_assignment_manifest_digest":
            V13_CONTRACT.FROZEN_ASSIGNMENT_MANIFEST_DIGEST,
        "source_branch_rows_sha256": V13_CONTRACT.FROZEN_BRANCH_ROWS_SHA256,
        "source_branch_identity_set_digest":
            V13_CONTRACT.FROZEN_BRANCH_IDENTITY_SET_DIGEST,
        "historical_calibration_identity_projection_digest":
            V13_CONTRACT.HISTORICAL_CALIBRATION_IDENTITY_PROJECTION_DIGEST,
        "candidate_bank_digest": V13_CONTRACT.CANDIDATE_BANK_DIGEST,
        "oracle_v1_3_digest": V13_CONTRACT.ORACLE.oracle_digest(),
        "state_count": len(states),
        "branch_count": len(entries),
        "candidate_indices": list(EXPECTED_CANDIDATES),
        "prior_lineage_trace_count": len(prior_traces),
        "new_replay_count": len(entries),
        "branch_identity_digest": digest(sorted(row_identities)),
        "entries": entries,
        "selection": {
            "rule": "exact frozen historical calibration role x all 12 candidates",
            "outcome_fields_used_for_selection": [],
            "label_bearing_v2_rows_materialized_by_frozen_loader": True,
            "scientific_label_fields_consulted_for_selection": False,
            "frozen_target_labels_bound_after_identity_selection": True,
            "prior_trace_availability_used_for_selection": False,
            "prior_overlay_target_used_only_for_null_v2_projection_count":
                EXPECTED_V13_LABEL_REFERENCES,
        },
        "replay_aggregate_equality_contract": {
            "fields": list(AGGREGATE_LABEL_FIELDS),
            "absolute_tolerances": dict(AGGREGATE_ABS_TOLERANCES),
            "formula_absolute_tolerance": AGGREGATE_FORMULA_ATOL,
            "source_counts": dict(sorted(label_source_counts.items())),
            "raw_null_v2_projection_count": EXPECTED_V13_LABEL_REFERENCES,
            "raw_null_v2_lineage_is_preserved": True,
            "all_288_compare_to_a_finite_frozen_scorer_target": True,
        },
        "execution": {
            "backend": "cpu",
            "attempts_per_branch": 1,
            "retry_or_replacement": False,
            "all_288_replayed": True,
            "prior_12_are_not_substituted": True,
            "renders": 0,
            "frame_writes": 0,
            "scientific_corpus_labels_computed_or_replaced": False,
            "diagnostic_replay_aggregates_recomputed_for_equality": True,
            "tick_physical_evidence_sampled": True,
            "scientific_corpus_label_writes": 0,
            "diagnostic_aggregate_receipt_writes": EXPECTED_BRANCHES,
            "state_writes": 0,
            "latent_access": 0,
        },
        "contact_type_contract": {
            "categories": list(CONTACT_CATEGORIES),
            "force_threshold_n": ORACLE_V12.CONTACT_FORCE_THRESHOLD_N,
            "ordinary_foot_ground_is_allowed": True,
            "prior_v1_3_limitation": "NOT_RETAINED_IN_V1_3_TRACE",
        },
        "required_tick_field_mapping": {
            "contact_indicator": "disallowed_contact",
            "contact_type": "disallowed_contact_details[].category",
            "clearance_m": "clearance_m",
            "normalized_clearance_deficit": "clearance_deficit",
            "stuck": "stuck",
            "fall_or_unsafe_termination": "termination + terminated",
            "completion": "completion_latched",
        },
        "sampled_endpoint_ticks": list(SAMPLED_ENDPOINT_TICKS),
    }
    require(payload["state_count"] == EXPECTED_STATES
            and payload["branch_count"] == EXPECTED_BRANCHES
            and payload["prior_lineage_trace_count"]
            == EXPECTED_PRIOR_LINEAGE_TRACES
            and payload["new_replay_count"] == EXPECTED_NEW_REPLAYS,
            "diagnostic plan counts changed")
    return signed(payload, PLAN_SELF_KEY)


def issue_plan(*, out_root: Path = OUT_ROOT,
               v2_root: Path = V13.V2_ROOT,
               v13_root: Path = V13.OUT_ROOT) -> dict[str, Any]:
    _managed_root(out_root)
    diagnostic_contract = load_bound_contract(out_root=out_root)
    corpus = V13.load_v2_corpus(v2_root=v2_root)
    prior = _load_prior_calibration_traces(v13_root=v13_root)
    plan = build_plan(corpus, prior, diagnostic_contract)
    _atomic_json(plan_path(out_root), plan, out_root=out_root, idempotent=True)
    return plan


def load_plan(*, out_root: Path = OUT_ROOT,
              v2_root: Path = V13.V2_ROOT,
              v13_root: Path = V13.OUT_ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan = validate_signed(
        _read_json(plan_path(out_root), "diagnostic replay plan"),
        PLAN_SELF_KEY, "diagnostic replay plan",
    )
    diagnostic_contract = load_bound_contract(out_root=out_root)
    corpus = V13.load_v2_corpus(v2_root=v2_root)
    prior = _load_prior_calibration_traces(v13_root=v13_root)
    require(plan == build_plan(corpus, prior, diagnostic_contract),
            "diagnostic replay plan changed")
    return plan, corpus, prior


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    try:
        return value.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(value)


def classify_disallowed_contact_details(
        contacts: Mapping[str, Any], *, robot_link_range: tuple[int, int],
        foot_link_indices: Iterable[int], ground_link_indices: Iterable[int],
        link_names: Mapping[int, str] | None = None,
        ) -> list[dict[str, Any]]:
    """Return deterministic categories for the same contacts counted by v1.2."""

    names = {int(key): str(value) for key, value in (link_names or {}).items()}
    feet = frozenset(int(value) for value in foot_link_indices)
    ground = frozenset(int(value) for value in ground_link_indices)
    low, high = (int(value) for value in robot_link_range)
    link_a = [int(value) for value in _to_numpy(contacts.get("link_a", ())).ravel()]
    link_b = [int(value) for value in _to_numpy(contacts.get("link_b", ())).ravel()]
    require(len(link_a) == len(link_b), "contact link arrays differ in length")
    force = contacts.get("force_a")
    magnitudes: list[float] | None = None
    if force is not None:
        array = _to_numpy(force)
        if array.size:
            require(array.size % 3 == 0,
                    "contact force array cannot form 3-vectors")
            magnitudes = np.linalg.norm(array.reshape(-1, 3), axis=-1).tolist()
    details: list[dict[str, Any]] = []
    for index, (a, b) in enumerate(zip(link_a, link_b, strict=True)):
        a_robot = low <= a < high
        b_robot = low <= b < high
        if a_robot == b_robot:
            continue
        robot_link = a if a_robot else b
        environment_link = b if a_robot else a
        magnitude = (
            None if magnitudes is None or index >= len(magnitudes)
            else float(magnitudes[index])
        )
        if magnitude is not None and magnitude <= ORACLE_V12.CONTACT_FORCE_THRESHOLD_N:
            continue
        robot_is_foot = robot_link in feet
        environment_is_ground = environment_link in ground
        if robot_is_foot and environment_is_ground:
            continue
        if robot_is_foot:
            category = "FOOT_NONGROUND_ENVIRONMENT"
        elif environment_is_ground:
            category = "NONFOOT_GROUND"
        else:
            category = "NONFOOT_NONGROUND_ENVIRONMENT"
        details.append({
            "category": category,
            "robot_link_id": robot_link,
            "robot_link_name": names.get(robot_link),
            "environment_link_id": environment_link,
            "environment_link_name": names.get(environment_link),
            "force_magnitude_n": magnitude,
        })
    details.sort(key=lambda row: (
        row["category"], row["robot_link_id"], row["environment_link_id"],
        -1.0 if row["force_magnitude_n"] is None else row["force_magnitude_n"],
    ))
    canonical_bytes(details)
    return details


def _link_name_map(ctx: Any) -> dict[int, str]:
    result: dict[int, str] = {}
    entities = getattr(ctx.build.scene, "entities", ())
    for entity in entities:
        for link in getattr(entity, "links", ()):
            try:
                result[int(link.idx)] = str(link.name)
            except (AttributeError, TypeError, ValueError):
                continue
    for link in getattr(ctx.build.robot, "links", ()):
        try:
            result[int(link.idx)] = str(link.name)
        except (AttributeError, TypeError, ValueError):
            continue
    return result


def _contact_details_now(ctx: Any, topology: Mapping[str, Any],
                         names: Mapping[int, str]) -> list[dict[str, Any]]:
    contacts = ctx.build.robot.get_contacts()
    if not contacts:
        return []
    return classify_disallowed_contact_details(
        contacts,
        robot_link_range=topology["robot_link_range"],
        foot_link_indices=topology["foot_link_indices"],
        ground_link_indices=topology["ground_link_indices"],
        link_names=names,
    )


def execute_diagnostic_trace(
        ctx: Any, snapshot: Any, candidate: tuple[str, tuple[str, ...]], *,
        field: Any, topology: Mapping[str, Any], on_block_end: Any = None,
        ) -> dict[str, Any]:
    """Use the v1.3 executor while augmenting each exact sample with contact type."""

    original_sample = V13._trace_sample
    names = _link_name_map(ctx)

    def sample_with_contact_type(*args: Any, **kwargs: Any) -> dict[str, Any]:
        row = original_sample(*args, **kwargs)
        details = _contact_details_now(ctx, topology, names)
        require(len(details) == row["disallowed_contacts"],
                "diagnostic contact categorisation differs from v1.2 count")
        row["contact_type_evidence_status"] = "COMPLETE"
        row["disallowed_contact_types"] = sorted({
            item["category"] for item in details
        })
        row["disallowed_contact_details"] = details
        return row

    # The frozen executor calls this module function synchronously.  Restore it
    # even after a technical failure; this workflow is deliberately single-threaded.
    V13._trace_sample = sample_with_contact_type
    try:
        trace = V13.execute_branch_trace_v13(
            ctx, snapshot, candidate, field=field, topology=topology,
            on_block_end=on_block_end,
        )
    finally:
        V13._trace_sample = original_sample
    return validate_trace_structure(trace, require_contact_types=True)


def begin_attempt_once(identity: str, plan: Mapping[str, Any], *,
                       out_root: Path = OUT_ROOT) -> dict[str, Any]:
    marker = signed({
        "schema": ATTEMPT_SCHEMA,
        "status": "ATTEMPT_STARTED_NO_RETRY_AUTHORITY",
        "contract_digest": plan["contract_digest"],
        "plan_digest": plan[PLAN_SELF_KEY],
        "branch_identity_digest": identity,
        "attempt_number": 1,
        "maximum_attempts_for_identity": 1,
        "retry_or_replacement": False,
    }, ATTEMPT_SELF_KEY)
    _atomic_json(
        attempt_path(identity, out_root), marker,
        out_root=out_root, idempotent=False,
    )
    return marker


def _prior_shared_field_agreement(
        prior_trace: Mapping[str, Any], current_trace: Mapping[str, Any],
        ) -> dict[str, Any]:
    old = validate_trace_structure(prior_trace, require_contact_types=False)
    new = validate_trace_structure(current_trace, require_contact_types=True)
    require(
        old["schema"] == new["schema"]
        and old["candidate"] == new["candidate"]
        and old["primitives"] == new["primitives"]
        and old["requested"] == new["requested"]
        and old["blocks_completed"] == new["blocks_completed"] == 4
        and old["nan"] is new["nan"] is False,
        "new diagnostic replay differs from prior top-level trace evidence",
    )
    old_post = np.asarray(old["post_slew"], dtype=np.float64)
    new_post = np.asarray(new["post_slew"], dtype=np.float64)
    require(old_post.shape == new_post.shape
            and np.all(np.isfinite(old_post))
            and np.all(np.isfinite(new_post)),
            "prior/new top-level post-slew trace is malformed")
    post_slew_error = float(np.max(np.abs(old_post - new_post)))
    require(post_slew_error <= V13.ACTION_ATOL,
            "new diagnostic replay differs from prior post-slew trace evidence")
    exact_fields = (
        "global_tick", "block_index", "tick_in_block", "episode_id",
        "episode_step", "timestamp_ns", "nearest_cell_id", "located",
        "accepted_cell_id", "cell_id", "goal_cell_id", "raw_bfs_to_goal",
        "masked_bfs_to_goal", "graph_status", "at_goal_cell", "stuck",
        "disallowed_contacts", "disallowed_contact", "termination", "terminated",
        "nan", "completion_latched",
    )
    numeric_fields = (
        "requested_command", "post_slew_command", "position_world_xyz_m",
        "quaternion_world_wxyz", "rpy_world_rad", "xy", "yaw", "z",
        "nearest_cell_distance_m", "geodesic_m", "clearance_m",
        "clearance_deficit",
    )
    maximum = 0.0
    old_samples = [old["start"], *old["ticks"]]
    new_samples = [new["start"], *new["ticks"]]
    for old_row, new_row in zip(old_samples, new_samples, strict=True):
        require(all(old_row.get(key) == new_row.get(key) for key in exact_fields),
                "new diagnostic replay differs from prior discrete trace evidence")
        for key in numeric_fields:
            old_value, new_value = old_row.get(key), new_row.get(key)
            if old_value is None or new_value is None:
                require(old_value is new_value,
                        "new diagnostic replay differs in missing numeric evidence")
                continue
            left = np.asarray(old_value, dtype=np.float64)
            right = np.asarray(new_value, dtype=np.float64)
            require(left.shape == right.shape and np.all(np.isfinite(left))
                    and np.all(np.isfinite(right)),
                    "prior/new shared numeric trace is malformed")
            maximum = max(maximum, float(np.max(np.abs(left - right))))
    require(maximum <= V13.HORIZON_POSE_ATOL,
            "new diagnostic replay exceeds prior shared-field tolerance")
    return {
        "status": "PASS_SHARED_FIELDS_AGREE",
        "prior_trace_digest": digest(old),
        "new_trace_shared_field_max_abs_error": maximum,
        "numeric_atol": V13.HORIZON_POSE_ATOL,
        "top_level_post_slew_max_abs_error": post_slew_error,
        "top_level_post_slew_atol": V13.ACTION_ATOL,
        "contact_type_evidence_in_prior": "NOT_RETAINED_IN_V1_3_TRACE",
        "contact_type_evidence_in_new": "COMPLETE",
    }


def replay_aggregate_label_projection(
        trace: Mapping[str, Any]) -> dict[str, float]:
    """Recompute only the six frozen scorer targets from twenty trace ticks."""

    value = validate_trace_structure(trace, require_contact_types=True)
    ticks = value["ticks"]
    contact = sum(bool(row["disallowed_contact"]) for row in ticks) / 20.0
    clearance = sum(float(row["clearance_deficit"]) for row in ticks) / 20.0
    stuck = sum(bool(row["stuck"]) for row in ticks) / 20.0
    fall = 1.0 if any(bool(row["terminated"]) for row in ticks) else 0.0
    safety = max(fall, (contact + clearance + stuck) / 3.0)
    completion = 1.0 if any(bool(row["completion_latched"])
                            for row in ticks) else 0.0
    return _finite_aggregate_label_projection({
        "contact_fraction": contact,
        "clearance_cost": clearance,
        "stuck_fraction": stuck,
        "fall": fall,
        "safety": safety,
        "completion": completion,
    }, "diagnostic replay aggregate label projection")


def replay_aggregate_equality(
        entry: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, Any]:
    raw_candidate = entry.get("source_v2_aggregate_label_projection")
    require(isinstance(raw_candidate, Mapping)
            and set(raw_candidate) == set(AGGREGATE_LABEL_FIELDS),
            "planned raw V2 aggregate-label projection changed")
    if all(value is None for value in raw_candidate.values()):
        raw_v2: dict[str, float | None] = dict(raw_candidate)
    else:
        require(all(value is not None for value in raw_candidate.values()),
                "planned raw V2 aggregate-label projection is partially missing")
        raw_v2 = _finite_aggregate_label_projection(
            raw_candidate, "planned raw V2 aggregate-label projection")
    require(digest(raw_v2)
            == entry.get("source_v2_aggregate_label_projection_digest"),
            "planned raw V2 aggregate-label binding changed")
    source_kind = entry.get("frozen_replay_target_source_kind")
    require(source_kind in {V2_LABEL_SOURCE, V13_LABEL_SOURCE},
            "planned replay-target source kind changed")
    if source_kind == V2_LABEL_SOURCE:
        require(all(value is not None for value in raw_v2.values()),
                "finite V2 replay target became null")
    else:
        require(all(value is None for value in raw_v2.values())
                and entry.get("prior_v1_3_trace") is not None,
                "v1.3 overlay target lost its null V2 lineage")
    target = _finite_aggregate_label_projection(
        entry.get("frozen_replay_target_projection", {}),
        "planned frozen replay target",
    )
    require(digest(target) == entry.get("frozen_replay_target_projection_digest"),
            "planned frozen replay-target digest changed")
    replay = replay_aggregate_label_projection(trace)
    comparisons: dict[str, Any] = {}
    for field in AGGREGATE_LABEL_FIELDS:
        error = abs(replay[field] - target[field])
        tolerance = AGGREGATE_ABS_TOLERANCES[field]
        require(error <= tolerance,
                f"diagnostic replay aggregate differs from frozen target: {field}")
        comparisons[field] = {
            "frozen_target": target[field],
            "diagnostic_replay": replay[field],
            "absolute_error": error,
            "absolute_tolerance": tolerance,
            "within_tolerance": True,
        }
    return {
        "status": "PASS_FROZEN_SCORER_TARGET_AGREEMENT",
        "source_kind": source_kind,
        "raw_v2_projection": raw_v2,
        "raw_v2_projection_digest": digest(raw_v2),
        "frozen_target_projection": target,
        "frozen_target_projection_digest": digest(target),
        "diagnostic_replay_projection": replay,
        "diagnostic_replay_projection_digest": digest(replay),
        "absolute_tolerances": dict(AGGREGATE_ABS_TOLERANCES),
        "formula_absolute_tolerance": AGGREGATE_FORMULA_ATOL,
        "field_comparisons": comparisons,
    }


def _trace_row_payload(
        *, entry: Mapping[str, Any], trace: Mapping[str, Any],
        equality: Mapping[str, Any], attempt: Mapping[str, Any],
        prior: Mapping[str, Any] | None, plan: Mapping[str, Any],
        ) -> dict[str, Any]:
    agreement = None if prior is None else _prior_shared_field_agreement(
        prior["trace"], trace)
    aggregate_equality = replay_aggregate_equality(entry, trace)
    return signed({
        "schema": TRACE_ROW_SCHEMA,
        "status": STATUS,
        "contract_digest": plan["contract_digest"],
        "plan_digest": plan[PLAN_SELF_KEY],
        "source_v2_corpus_digest": V13_CONTRACT.FROZEN_CORPUS_DIGEST,
        **{key: copy.deepcopy(entry[key]) for key in (
            "branch_identity_digest", "branch_row_digest",
            "assignment_identity_digest", "state_id", "state_identity_digest",
            "scene_id", "family", "stratum", "candidate_index", "candidate",
        )},
        "attempt_digest": attempt[ATTEMPT_SELF_KEY],
        "replay_equality": copy.deepcopy(dict(equality)),
        "prior_v1_3_trace_reference": copy.deepcopy(entry["prior_v1_3_trace"]),
        "prior_v1_3_shared_field_agreement": agreement,
        "replay_aggregate_equality": aggregate_equality,
        "trace": copy.deepcopy(dict(trace)),
        "scientific_corpus_labels_computed_or_replaced": False,
        "diagnostic_replay_aggregates_recomputed_for_equality": True,
        "tick_physical_evidence_sampled": True,
        "scientific_corpus_labels_written": False,
        "diagnostic_aggregate_receipt_written": True,
        "frames_rendered": 0,
        "frames_written": 0,
        "states_written": 0,
        "latents_accessed": 0,
        "source_v2_modified": False,
    }, TRACE_ROW_SELF_KEY)


def _validate_trace_row(
        row: Mapping[str, Any], *, entry: Mapping[str, Any],
        old_row: Mapping[str, Any], prior: Mapping[str, Any] | None,
        plan: Mapping[str, Any], out_root: Path,
        ) -> dict[str, Any]:
    value = validate_signed(row, TRACE_ROW_SELF_KEY, "diagnostic trace row")
    identity = entry["branch_identity_digest"]
    marker = validate_signed(
        _read_json(attempt_path(identity, out_root), "diagnostic attempt marker"),
        ATTEMPT_SELF_KEY, "diagnostic attempt marker",
    )
    trace = validate_trace_structure(value.get("trace", {}),
                                     require_contact_types=True)
    equality = value.get("replay_equality")
    require(isinstance(equality, Mapping),
            "diagnostic replay equality witness is malformed")
    replay_witness = equality.get("replay_prebranch_witness", {})
    expected_pre = V13.validate_replay_preexecution(
        old_row=old_row,
        replay_snapshot_digest=equality.get("replay_snapshot_digest"),
        context_camera_poses=equality.get("replay_context_camera_poses", []),
        proprio=replay_witness.get("proprio", []),
        control=replay_witness.get("control", []),
        action_context_blocks=replay_witness.get("action_context_blocks", []),
        previous_applied_command=replay_witness.get("previous_applied_command", []),
    )
    expected_equality = V13.validate_replay_equality(
        old_row=old_row, branch=trace,
        horizon_camera_poses=equality.get("replay_horizon_camera_poses", []),
        preexecution=expected_pre,
    )
    expected_agreement = None if prior is None else _prior_shared_field_agreement(
        prior["trace"], trace)
    expected_aggregate_equality = replay_aggregate_equality(entry, trace)
    require(
        equality == expected_equality
        and value.get("schema") == TRACE_ROW_SCHEMA
        and value.get("status") == STATUS
        and value.get("contract_digest") == plan.get("contract_digest")
        and value.get("plan_digest") == plan.get(PLAN_SELF_KEY)
        and value.get("source_v2_corpus_digest")
        == V13_CONTRACT.FROZEN_CORPUS_DIGEST
        and marker.get("schema") == ATTEMPT_SCHEMA
        and marker.get("status") == "ATTEMPT_STARTED_NO_RETRY_AUTHORITY"
        and marker.get("contract_digest") == plan.get("contract_digest")
        and marker.get("branch_identity_digest") == identity
        and marker.get("plan_digest") == plan[PLAN_SELF_KEY]
        and marker.get("attempt_number") == 1
        and marker.get("maximum_attempts_for_identity") == 1
        and marker.get("retry_or_replacement") is False
        and value.get("attempt_digest") == marker[ATTEMPT_SELF_KEY]
        and all(value.get(key) == entry.get(key) for key in (
            "branch_identity_digest", "branch_row_digest",
            "assignment_identity_digest", "state_id", "state_identity_digest",
            "scene_id", "family", "stratum", "candidate_index", "candidate",
        ))
        and value.get("prior_v1_3_trace_reference")
        == entry.get("prior_v1_3_trace")
        and value.get("prior_v1_3_shared_field_agreement") == expected_agreement
        and value.get("replay_aggregate_equality")
        == expected_aggregate_equality
        and value.get("scientific_corpus_labels_computed_or_replaced") is False
        and value.get("diagnostic_replay_aggregates_recomputed_for_equality") is True
        and value.get("tick_physical_evidence_sampled") is True
        and value.get("scientific_corpus_labels_written") is False
        and value.get("diagnostic_aggregate_receipt_written") is True
        and value.get("frames_rendered") == value.get("frames_written") == 0
        and value.get("states_written") == value.get("latents_accessed") == 0
        and value.get("source_v2_modified") is False,
        "diagnostic trace-row binding changed",
    )
    return value


def _artifact_directory_identities(relative: Path, *,
                                   out_root: Path) -> set[str]:
    directory = out_root / relative
    if not directory.exists() and not directory.is_symlink():
        return set()
    require(directory.is_dir() and not directory.is_symlink(),
            f"diagnostic artifact directory changed: {relative}")
    identities: set[str] = set()
    for path in directory.iterdir():
        require(path.is_file() and not path.is_symlink()
                and path.suffix == ".json" and is_digest(path.stem),
                f"unexpected diagnostic artifact: {path.name}")
        identities.add(path.stem)
    return identities


def _existing_execution_inventory(
        plan: Mapping[str, Any], corpus: Mapping[str, Any],
        prior: Mapping[str, Mapping[str, Any]], *, out_root: Path,
        ) -> list[dict[str, Any]]:
    old_by_identity = {
        row["branch_identity_digest"]: row for row in corpus["rows"]
    }
    expected_identities = {
        row["branch_identity_digest"] for row in plan["entries"]
    }
    marker_identities = _artifact_directory_identities(
        ATTEMPTS_NAME, out_root=out_root)
    row_identities = _artifact_directory_identities(
        TRACE_ROWS_NAME, out_root=out_root)
    require(marker_identities <= expected_identities
            and row_identities <= expected_identities,
            "diagnostic namespace contains an unregistered branch artifact")
    complete: list[dict[str, Any]] = []
    for entry in plan["entries"]:
        identity = entry["branch_identity_digest"]
        marker_file = attempt_path(identity, out_root)
        row_file = trace_row_path(identity, out_root)
        marker_exists = marker_file.exists() or marker_file.is_symlink()
        row_exists = row_file.exists() or row_file.is_symlink()
        if marker_exists and not row_exists:
            raise DiagnosticError(
                f"orphan diagnostic attempt is terminal for {identity}"
            )
        require(not row_exists or marker_exists,
                "diagnostic trace exists without its attempt marker")
        if row_exists:
            complete.append(_validate_trace_row(
                _read_json(trace_row_path(identity, out_root),
                           "diagnostic trace row"),
                entry=entry, old_row=old_by_identity[identity],
                prior=prior.get(identity), plan=plan, out_root=out_root,
            ))
    return complete


def execute_replays(*, backend: str = "cpu", out_root: Path = OUT_ROOT,
                    v2_root: Path = V13.V2_ROOT,
                    v13_root: Path = V13.OUT_ROOT) -> dict[str, Any]:
    require(backend == "cpu", "diagnostic replay is frozen to CPU")
    plan, corpus, prior = load_plan(
        out_root=out_root, v2_root=v2_root, v13_root=v13_root)
    completed = _existing_execution_inventory(
        plan, corpus, prior, out_root=out_root)
    complete_ids = {row["branch_identity_digest"] for row in completed}
    missing_entries = [entry for entry in plan["entries"]
                       if entry["branch_identity_digest"] not in complete_ids]
    if not missing_entries:
        return compile_terminal_manifest(
            out_root=out_root, v2_root=v2_root, v13_root=v13_root)

    V13.require_genesis_runtime()
    state_by_id = {
        row["state_id"]: row for row in corpus["state_manifest"]["states"]
    }
    old_by_identity = {
        row["branch_identity_digest"]: row for row in corpus["rows"]
    }
    entries_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in missing_entries:
        entries_by_state[entry["state_id"]].append(entry)
    shared = V13.V1._load_shared(backend)
    for state_id in sorted(entries_by_state):
        state = state_by_id[state_id]
        scene_dir = Path(state["scene_dir"])
        scene_manifest = scene_dir / "manifest.json"
        require(
            BUILDER.file_sha256(scene_manifest) == state["scene_manifest_sha256"]
            and scene_manifest.stat().st_size == state["scene_manifest_byte_count"],
            "diagnostic replay scene manifest changed",
        )
        ctx = V13.V1.build_context(
            scene_dir, seed=int(state["drive_seed"]), backend=backend,
            shared=shared,
        )
        topology = V13.V12.link_topology(ctx)
        ctx.begin_episode()
        proprio_log: list[list[float]] = []
        control_log: list[list[float]] = []
        action_context_blocks: list[list[float]] = []
        context_poses: list[Any] = []
        warmup_blocks = int(state["warmup_blocks"])

        def probe(_tick: int, previous: Sequence[float],
                  _ctx: Any = ctx) -> None:
            proprio_log.append(BUILDER.proprio_sample(_ctx))
            control_log.append(BUILDER.control_sample(previous))

        for block_index in range(warmup_blocks):
            driven = BUILDER.drive_block_with_probe(ctx, probe)
            if block_index >= warmup_blocks - BUILDER.CONTEXT_SLOTS:
                action_context_blocks.append(BUILDER.action_block_10d(
                    np.asarray(driven.executed, dtype=np.float64)[0]
                ))
                context_poses.append(BUILDER.capture_base_pose(ctx))
        verdict = BUILDER.classify_state(
            ctx, topology, requested_stratum=state["stratum"])
        require(not isinstance(verdict, str),
                f"diagnostic state redrive failed: {verdict}")
        record, field, _strata = verdict
        mismatch = BUILDER._redrive_mismatch(
            state, record, ctx, full_bank_v2=True)
        require(mismatch is None, f"diagnostic state redrive mismatch: {mismatch}")
        snapshot = V13.V1.capture_branch_state(
            ctx, goal=state["goal"], identity={
                "state_id": state_id,
                "state_identity_digest": state["state_identity_digest"],
                "scene_id": state["scene_id"], "family": state["family"],
                "split": state["split"], "block_index": state["warmup_blocks"],
                "source_step": state["source_step"],
                "episode_id": state["episode_id"],
            },
        )
        context_camera_poses = [
            V13._pose_to_camera_pose(ctx, pose) for pose in context_poses
        ]
        proprio = np.asarray(
            proprio_log[-BUILDER.PROPRIO_HISTORY:], dtype=np.float32)
        control = np.asarray(
            control_log[-BUILDER.PROPRIO_HISTORY:], dtype=np.float32)
        previous = np.asarray(
            ctx.runner._last_executed, dtype=np.float64)[0].tolist()
        preexecution: dict[str, dict[str, Any]] = {}
        for entry in entries_by_state[state_id]:
            old_row = old_by_identity[entry["branch_identity_digest"]]
            preexecution[entry["branch_identity_digest"]] = \
                V13.validate_replay_preexecution(
                    old_row=old_row, replay_snapshot_digest=snapshot.digest,
                    context_camera_poses=context_camera_poses,
                    proprio=proprio.tolist(), control=control.tolist(),
                    action_context_blocks=action_context_blocks,
                    previous_applied_command=previous,
                )
        for entry in sorted(entries_by_state[state_id],
                            key=lambda row: row["candidate_index"]):
            identity = entry["branch_identity_digest"]
            horizon_poses: list[Any] = []
            # This marker is the irreversible boundary and is deliberately
            # adjacent to execution.  Pre-execution validation happens first.
            marker = begin_attempt_once(identity, plan, out_root=out_root)
            trace = execute_diagnostic_trace(
                ctx, snapshot,
                V13.V1.CANDIDATE_BANK[int(entry["candidate_index"])],
                field=field, topology=topology,
                on_block_end=lambda _index, _ctx=ctx: horizon_poses.append(
                    BUILDER.capture_base_pose(_ctx)),
            )
            old_row = old_by_identity[identity]
            equality = V13.validate_replay_equality(
                old_row=old_row, branch=trace,
                horizon_camera_poses=[
                    V13._pose_to_camera_pose(ctx, pose) for pose in horizon_poses
                ],
                preexecution=preexecution[identity],
            )
            row = _trace_row_payload(
                entry=entry, trace=trace, equality=equality, attempt=marker,
                prior=prior.get(identity), plan=plan,
            )
            _atomic_json(
                trace_row_path(identity, out_root), row,
                out_root=out_root, idempotent=False,
            )
        del ctx
        gc.collect()
    return compile_terminal_manifest(
        out_root=out_root, v2_root=v2_root, v13_root=v13_root)


def _terminal_payload(
        plan: Mapping[str, Any], inventory: Sequence[Mapping[str, Any]],
        ) -> dict[str, Any]:
    rows = [copy.deepcopy(dict(row)) for row in inventory]
    entries = plan.get("entries")
    require(isinstance(entries, list) and len(entries) == EXPECTED_BRANCHES,
            "terminal plan is not exact 24x12")
    expected_identities = {
        row.get("branch_identity_digest") for row in entries
    }
    entry_by_identity = {
        row.get("branch_identity_digest"): row for row in entries
    }
    expected_prior = {
        row.get("branch_identity_digest") for row in entries
        if row.get("prior_v1_3_trace") is not None
    }
    candidates_by_state: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        candidates_by_state[str(row.get("state_identity_digest"))].append(
            int(row.get("candidate_index", -1)))
    require(
        len(rows) == EXPECTED_BRANCHES
        and len({row.get("branch_identity_digest") for row in rows})
        == EXPECTED_BRANCHES
        and {row.get("branch_identity_digest") for row in rows}
        == expected_identities
        and all(all(
            row.get(key) == entry_by_identity[
                row.get("branch_identity_digest")].get(key)
            for key in ("state_identity_digest", "family", "stratum",
                        "candidate_index")
        ) for row in rows)
        and len(candidates_by_state) == EXPECTED_STATES
        and all(sorted(values) == list(EXPECTED_CANDIDATES)
                for values in candidates_by_state.values())
        and all(row.get("source_kind") == "NEW_DIAGNOSTIC_REPLAY"
                for row in rows)
        and Counter(row.get("frozen_replay_target_source_kind") for row in rows)
        == Counter({
            V2_LABEL_SOURCE: EXPECTED_V2_LABEL_REFERENCES,
            V13_LABEL_SOURCE: EXPECTED_V13_LABEL_REFERENCES,
        })
        and sum(row.get("prior_v1_3_trace_bound") is True for row in rows)
        == EXPECTED_PRIOR_LINEAGE_TRACES,
        "terminal diagnostic inventory changed",
    )
    require({
        row["branch_identity_digest"] for row in rows
        if row["prior_v1_3_trace_bound"] is True
    } == expected_prior, "terminal prior-lineage identity set changed")
    rows.sort(key=lambda row: row["branch_identity_digest"])
    return signed({
        "schema": TERMINAL_SCHEMA,
        "status": STATUS,
        "complete": True,
        "contract_digest": plan["contract_digest"],
        "plan_digest": plan[PLAN_SELF_KEY],
        "state_count": EXPECTED_STATES,
        "branch_count": EXPECTED_BRANCHES,
        "new_replay_count": EXPECTED_NEW_REPLAYS,
        "prior_lineage_trace_count": EXPECTED_PRIOR_LINEAGE_TRACES,
        "adopted_as_final_count": CONTRACT.ADOPTED_TRACES,
        "prior_v1_3_contact_type_evidence":
            "NOT_RETAINED_IN_V1_3_TRACE",
        "final_diagnostic_contact_type_evidence":
            "COMPLETE_FROM_ALL_288_NEW_REPLAYS",
        "frozen_replay_target_source_counts": {
            V2_LABEL_SOURCE: EXPECTED_V2_LABEL_REFERENCES,
            V13_LABEL_SOURCE: EXPECTED_V13_LABEL_REFERENCES,
        },
        "all_replay_aggregates_match_frozen_scorer_targets": True,
        "replay_aggregate_absolute_tolerances":
            dict(AGGREGATE_ABS_TOLERANCES),
        "sampled_endpoint_ticks": list(SAMPLED_ENDPOINT_TICKS),
        "rows": rows,
        "scientific_corpus_labels_computed_or_replaced": False,
        "diagnostic_replay_aggregates_recomputed_for_equality": True,
        "tick_physical_evidence_sampled": True,
        "scientific_corpus_labels_written": False,
        "diagnostic_aggregate_receipt_count": EXPECTED_BRANCHES,
        "frames_rendered": 0,
        "frames_written": 0,
        "states_written": 0,
        "latents_accessed": 0,
    }, TERMINAL_SELF_KEY)


def compile_terminal_manifest(*, out_root: Path = OUT_ROOT,
                              v2_root: Path = V13.V2_ROOT,
                              v13_root: Path = V13.OUT_ROOT) -> dict[str, Any]:
    plan, corpus, prior = load_plan(
        out_root=out_root, v2_root=v2_root, v13_root=v13_root)
    rows = _existing_execution_inventory(
        plan, corpus, prior, out_root=out_root)
    require(len(rows) == EXPECTED_BRANCHES,
            "diagnostic replay panel is incomplete")
    inventory = []
    for row in rows:
        identity = row["branch_identity_digest"]
        path = trace_row_path(identity, out_root)
        inventory.append({
            "branch_identity_digest": identity,
            "state_identity_digest": row["state_identity_digest"],
            "family": row["family"], "stratum": row["stratum"],
            "candidate_index": row["candidate_index"],
            "source_kind": "NEW_DIAGNOSTIC_REPLAY",
            "frozen_replay_target_source_kind": row[
                "replay_aggregate_equality"]["source_kind"],
            "prior_v1_3_trace_bound": row["prior_v1_3_trace_reference"] is not None,
            "path": str(path.absolute().relative_to(out_root.absolute())),
            "sha256": file_sha256(path),
            TRACE_ROW_SELF_KEY: row[TRACE_ROW_SELF_KEY],
        })
    terminal = _terminal_payload(plan, inventory)
    _atomic_json(
        terminal_path(out_root), terminal, out_root=out_root, idempotent=True)
    return terminal


def _component_timing(active_ticks: Sequence[int]) -> dict[str, Any]:
    ticks = sorted({int(value) for value in active_ticks})
    require(all(value in TICKS for value in ticks),
            "component timing contains an out-of-horizon tick")
    runs: list[list[int]] = []
    for tick in ticks:
        if not runs or tick != runs[-1][-1] + 1:
            runs.append([tick])
        else:
            runs[-1].append(tick)
    run_records = [{
        "first_tick": run[0],
        "last_tick": run[-1],
        "tick_count": len(run),
        "duration_s": len(run) / 10.0,
    } for run in runs]
    endpoint = [value for value in ticks if value in SAMPLED_ENDPOINT_TICKS]
    present = bool(ticks)
    return {
        "present": present,
        "first_tick": None if not ticks else ticks[0],
        "last_tick": None if not ticks else ticks[-1],
        "active_tick_count": len(ticks),
        "active_duration_s": len(ticks) / 10.0,
        "span_tick_count": 0 if not ticks else ticks[-1] - ticks[0] + 1,
        "span_duration_s": 0.0 if not ticks else (ticks[-1] - ticks[0] + 1) / 10.0,
        "contiguous_event_runs": run_records,
        "contiguous_event_run_count": len(run_records),
        "longest_contiguous_event_run_ticks": max(
            (row["tick_count"] for row in run_records), default=0),
        "sampled_endpoint_ticks_active": endpoint,
        "sampled_endpoint_overlap": bool(endpoint),
        "only_at_non_sample_ticks": present and not endpoint,
        "strictly_between_sampled_horizons_only": (
            present and set(ticks) <= STRICTLY_BETWEEN_SAMPLED_HORIZON_TICKS),
        "any_pre_h1": bool(set(ticks) & PRE_H1_TICKS),
        "pre_h1_only": present and set(ticks) <= PRE_H1_TICKS,
        "final_sample_active": 19 in ticks,
        "interpretation": (
            "physical label-evidence timing only; endpoint coincidence is not "
            "a claim that RGB makes the event visually observable"
        ),
    }


def _tick_activations(row: Mapping[str, Any]) -> dict[str, bool]:
    termination = row["termination"]
    unsafe = bool(
        termination["fall"] or termination["out_of_bounds"]
        or termination["tipped"]
    )
    require(row["terminated"] is unsafe,
            "tick termination summary differs from its predicates")
    return {
        "contact": bool(row["disallowed_contact"]),
        "clearance": float(row["clearance_deficit"]) > 0.0,
        "stuck": bool(row["stuck"]),
        "unsafe_termination": unsafe,
        "completion": bool(row["completion_latched"]),
    }


def branch_safety_observability(
        identity: Mapping[str, Any], trace: Mapping[str, Any],
        ) -> dict[str, Any]:
    value = validate_trace_structure(trace, require_contact_types=True)
    ticks = value["ticks"]
    activations = [_tick_activations(row) for row in ticks]
    contact_fraction = sum(row["contact"] for row in activations) / 20.0
    clearance_cost = sum(float(row["clearance_deficit"]) for row in ticks) / 20.0
    stuck_fraction = sum(row["stuck"] for row in activations) / 20.0
    fall = 1.0 if any(row["unsafe_termination"] for row in activations) else 0.0
    base = (contact_fraction + clearance_cost + stuck_fraction) / 3.0
    safety = max(fall, base)
    contract_masses = CONTRACT.safety_mass_attribution(
        contact=contact_fraction, clearance=clearance_cost,
        stuck=stuck_fraction, fall=fall, safety=safety,
    )
    masses = {
        "contact_mass": contract_masses["contact"],
        "clearance_mass": contract_masses["clearance"],
        "stuck_mass": contract_masses["stuck"],
        "fall_residual_mass": contract_masses["fall"],
    }
    require(math.isclose(sum(masses.values()), safety,
                         rel_tol=0.0, abs_tol=1e-12),
            "safety attribution does not sum to physical safety")
    component_ticks = {
        component: [index for index, row in enumerate(activations)
                    if row[component]]
        for component in TIMING_COMPONENTS
    }
    pairwise = {}
    for left, right in itertools.combinations(COMPONENTS, 2):
        overlap = sorted(set(component_ticks[left]) & set(component_ticks[right]))
        pairwise[f"{left}+{right}"] = {
            "overlap_present": bool(overlap),
            "overlap_tick_count": len(overlap),
            "overlap_ticks": overlap,
        }
    contact_types = Counter(
        category for row in ticks for category in row["disallowed_contact_types"]
    )
    safety_ticks = sorted(set().union(*(
        set(component_ticks[component]) for component in COMPONENTS
    )))
    endpoint_safety_ticks = [
        tick for tick in safety_ticks if tick in SAMPLED_ENDPOINT_TICKS
    ]
    result = {
        **{key: copy.deepcopy(identity[key]) for key in (
            "branch_identity_digest", "state_identity_digest", "family",
            "stratum", "candidate_index",
        )},
        "raw_means": {
            "disallowed_contact_count_per_tick": sum(
                int(row["disallowed_contacts"]) for row in ticks) / 20.0,
            "disallowed_contact_presence": contact_fraction,
            "clearance_m": sum(float(row["clearance_m"]) for row in ticks) / 20.0,
            "clearance_deficit": clearance_cost,
            "stuck_presence": stuck_fraction,
            "unsafe_termination_presence": sum(
                row["unsafe_termination"] for row in activations) / 20.0,
            "completion_latched_presence": sum(
                row["completion"] for row in activations) / 20.0,
            "completion_at_final_tick": float(activations[-1]["completion"]),
        },
        "safety_components": {
            "contact_fraction": contact_fraction,
            "clearance_cost": clearance_cost,
            "stuck_fraction": stuck_fraction,
            "fall": fall,
            "graded_base": base,
            "safety": safety,
        },
        "safety_mass_attribution": {
            **masses,
            "sum": sum(masses.values()),
            "equals_safety": True,
        },
        "component_timing": {
            component: _component_timing(component_ticks[component])
            for component in TIMING_COMPONENTS
        },
        "branch_wide_safety_timing": {
            "safety_positive": bool(safety_ticks),
            "positive_ticks": safety_ticks,
            "sampled_endpoint_ticks_positive": endpoint_safety_ticks,
            "sampled_endpoint_activity": {
                str(tick): tick in safety_ticks
                for tick in SAMPLED_ENDPOINT_TICKS
            },
            "any_evidence_at_sampled_endpoint": bool(endpoint_safety_ticks),
            "all_positive_evidence_only_at_non_sample_ticks": (
                bool(safety_ticks) and not endpoint_safety_ticks),
            "all_safety_evidence_strictly_between_sampled_horizons": (
                bool(safety_ticks)
                and set(safety_ticks)
                <= STRICTLY_BETWEEN_SAMPLED_HORIZON_TICKS),
            "any_pre_h1_safety_evidence": bool(
                set(safety_ticks) & PRE_H1_TICKS),
            "pre_h1_only_safety_evidence": (
                bool(safety_ticks) and set(safety_ticks) <= PRE_H1_TICKS),
            "positive_at_h4": 19 in safety_ticks,
        },
        "pairwise_physical_overlap": pairwise,
        "contact_type_tick_prevalence": {
            category: contact_types[category] / 20.0
            for category in CONTACT_CATEGORIES
        },
        "sampled_endpoint_ticks": list(SAMPLED_ENDPOINT_TICKS),
        "physical_timing_not_visual_observability": True,
    }
    canonical_bytes(result)
    return result


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _group_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    require(count > 0, "cannot aggregate an empty diagnostic group")
    components = {}
    for component in TIMING_COMPONENTS:
        timing = [row["component_timing"][component] for row in records]
        positive = [row for row in timing if row["present"]]
        raw_key = {
            "contact": "contact_fraction",
            "clearance": "clearance_cost",
            "stuck": "stuck_fraction",
            "unsafe_termination": "fall",
        }.get(component)
        raw_values = (
            [float(row["raw_means"]["completion_latched_presence"])
             for row in records]
            if component == "completion" else
            [float(row["safety_components"][raw_key]) for row in records]
        )
        run_ticks = [
            int(run["tick_count"])
            for row in timing for run in row["contiguous_event_runs"]
        ]
        run_histogram = {
            str(duration): run_ticks.count(duration)
            for duration in sorted(set(run_ticks))
        }
        components[component] = {
            "branch_prevalence": len(positive) / count,
            "raw_component_mean": _mean(raw_values),
            "positive_branch_count": len(positive),
            "mean_first_tick_among_positive": _mean([
                float(row["first_tick"]) for row in positive]),
            "mean_last_tick_among_positive": _mean([
                float(row["last_tick"]) for row in positive]),
            "mean_active_tick_count_among_positive": _mean([
                float(row["active_tick_count"]) for row in positive]),
            "mean_active_duration_s_among_positive": _mean([
                float(row["active_duration_s"]) for row in positive]),
            "contiguous_event_run_count": len(run_ticks),
            "contiguous_event_run_duration_ticks": run_ticks,
            "contiguous_event_run_duration_s": [
                value / 10.0 for value in run_ticks],
            "contiguous_event_run_duration_tick_histogram": run_histogram,
            "sampled_endpoint_overlap_prevalence": sum(
                row["sampled_endpoint_overlap"] for row in timing) / count,
            "sampled_endpoint_overlap_given_positive": (
                None if not positive else sum(
                    row["sampled_endpoint_overlap"] for row in positive
                ) / len(positive)
            ),
            "per_sampled_endpoint_branch_prevalence": {
                str(endpoint): sum(
                    endpoint in row["sampled_endpoint_ticks_active"]
                    for row in timing) / count
                for endpoint in SAMPLED_ENDPOINT_TICKS
            },
            "per_sampled_endpoint_given_positive": {
                str(endpoint): (
                    None if not positive else sum(
                        endpoint in row["sampled_endpoint_ticks_active"]
                        for row in positive) / len(positive)
                )
                for endpoint in SAMPLED_ENDPOINT_TICKS
            },
            "only_at_non_sample_ticks_prevalence": sum(
                row["only_at_non_sample_ticks"] for row in timing
            ) / count,
            "only_at_non_sample_ticks_given_positive": (
                None if not positive else sum(
                    row["only_at_non_sample_ticks"]
                    for row in positive) / len(positive)
            ),
            "strictly_between_sampled_horizons_only_prevalence": sum(
                row["strictly_between_sampled_horizons_only"] for row in timing
            ) / count,
            "strictly_between_sampled_horizons_only_given_positive": (
                None if not positive else sum(
                    row["strictly_between_sampled_horizons_only"]
                    for row in positive) / len(positive)
            ),
            "any_pre_h1_prevalence": sum(
                row["any_pre_h1"] for row in timing) / count,
            "pre_h1_only_prevalence": sum(
                row["pre_h1_only"] for row in timing) / count,
            "final_sample_active_prevalence": sum(
                row["final_sample_active"] for row in timing) / count,
        }
    pairwise = {}
    for pair in records[0]["pairwise_physical_overlap"]:
        values = [row["pairwise_physical_overlap"][pair] for row in records]
        pairwise[pair] = {
            "branch_prevalence": sum(row["overlap_present"] for row in values) / count,
            "mean_overlap_tick_count": _mean([
                float(row["overlap_tick_count"]) for row in values]),
        }
    raw_keys = tuple(records[0]["raw_means"])
    mass_keys = ("contact_mass", "clearance_mass", "stuck_mass",
                 "fall_residual_mass", "sum")
    branch_wide = [row["branch_wide_safety_timing"] for row in records]
    safety_positive = [row for row in branch_wide if row["safety_positive"]]
    mass_sums = {key: sum(
        float(row["safety_mass_attribution"][key]) for row in records
    ) for key in mass_keys[:-1]}
    total_mass = sum(mass_sums.values())
    summary = {
        "branch_count": count,
        "component_statistics": components,
        "raw_means": {key: _mean([
            float(row["raw_means"][key]) for row in records
        ]) for key in raw_keys},
        "pairwise_physical_overlap": pairwise,
        "contact_type_statistics": {
            category: {
                "branch_prevalence": sum(
                    row["contact_type_tick_prevalence"][category] > 0.0
                    for row in records) / count,
                "mean_tick_prevalence": _mean([
                    float(row["contact_type_tick_prevalence"][category])
                    for row in records
                ]),
            }
            for category in CONTACT_CATEGORIES
        },
        "mean_safety_mass_attribution": {key: _mean([
            float(row["safety_mass_attribution"][key]) for row in records
        ]) for key in mass_keys},
        "summed_safety_mass_attribution": {
            **mass_sums, "total_safety_mass": total_mass,
        },
        "component_share_of_summed_total_safety_mass": {
            key: None if total_mass == 0.0 else value / total_mass
            for key, value in mass_sums.items()
        },
        "mean_safety": _mean([
            float(row["safety_components"]["safety"]) for row in records]),
        "safety_positive_branch_timing": {
            "safety_positive_branch_count": len(safety_positive),
            "safety_positive_branch_prevalence": len(safety_positive) / count,
            "any_evidence_at_sampled_endpoint_given_safety_positive": (
                None if not safety_positive else sum(
                    row["any_evidence_at_sampled_endpoint"]
                    for row in safety_positive) / len(safety_positive)
            ),
            "all_positive_evidence_only_at_non_sample_ticks_given_safety_positive": (
                None if not safety_positive else sum(
                    row["all_positive_evidence_only_at_non_sample_ticks"]
                    for row in safety_positive) / len(safety_positive)
            ),
            "positive_at_h4_given_safety_positive": (
                None if not safety_positive else sum(
                    row["positive_at_h4"] for row in safety_positive
                ) / len(safety_positive)
            ),
            "all_safety_evidence_strictly_between_sampled_horizons_given_safety_positive": (
                None if not safety_positive else sum(
                    row["all_safety_evidence_strictly_between_sampled_horizons"]
                    for row in safety_positive) / len(safety_positive)
            ),
            "any_pre_h1_safety_evidence_given_safety_positive": (
                None if not safety_positive else sum(
                    row["any_pre_h1_safety_evidence"]
                    for row in safety_positive) / len(safety_positive)
            ),
            "pre_h1_only_safety_evidence_given_safety_positive": (
                None if not safety_positive else sum(
                    row["pre_h1_only_safety_evidence"]
                    for row in safety_positive) / len(safety_positive)
            ),
            "per_sampled_endpoint_given_safety_positive": {
                str(endpoint): (
                    None if not safety_positive else sum(
                        row["sampled_endpoint_activity"][str(endpoint)]
                        for row in safety_positive) / len(safety_positive)
                )
                for endpoint in SAMPLED_ENDPOINT_TICKS
            },
        },
    }
    require(math.isclose(
        sum(float(summary["mean_safety_mass_attribution"][key]) for key in mass_keys[:-1]),
        float(summary["mean_safety"]), rel_tol=0.0, abs_tol=1e-12,
    ), "group safety mass does not equal mean safety")
    shares = summary["component_share_of_summed_total_safety_mass"]
    require(total_mass == 0.0 or math.isclose(
        sum(float(value) for value in shares.values()),
        1.0, rel_tol=0.0, abs_tol=1e-12,
    ), "group safety mass shares do not sum to one")
    return summary


def build_audit(terminal: Mapping[str, Any],
                trace_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    manifest = validate_signed(
        terminal, TERMINAL_SELF_KEY, "diagnostic terminal manifest")
    require(manifest.get("complete") is True
            and manifest.get("branch_count") == EXPECTED_BRANCHES
            and isinstance(manifest.get("rows"), list)
            and len(manifest["rows"]) == EXPECTED_BRANCHES
            and len(trace_rows) == EXPECTED_BRANCHES,
            "audit input panel is incomplete")
    branch_records = []
    seen: set[str] = set()
    for row in trace_rows:
        identity = str(row.get("branch_identity_digest"))
        require(identity not in seen, "audit input duplicates a branch")
        seen.add(identity)
        branch_records.append(branch_safety_observability(row, row["trace"]))
    require(seen == {
        str(row.get("branch_identity_digest")) for row in manifest["rows"]
    }, "audit trace identity set differs from terminal manifest")
    branch_records.sort(key=lambda row: row["branch_identity_digest"])
    families = sorted({row["family"] for row in branch_records})
    strata = sorted({row["stratum"] for row in branch_records})
    payload = {
        "schema": AUDIT_SCHEMA,
        "status": STATUS,
        "complete": True,
        "contract_digest": manifest["contract_digest"],
        "terminal_manifest_digest": manifest[TERMINAL_SELF_KEY],
        "branch_count": len(branch_records),
        "sampled_endpoint_ticks": list(SAMPLED_ENDPOINT_TICKS),
        "physical_timing_contract": {
            "tick_hz": 10,
            "ticks": list(TICKS),
            "component_activation_is_physical_label_evidence": True,
            "sampled_endpoint_overlap_is_temporal_coincidence_only": True,
            "non_sample_only_excludes_4_9_14_19_but_may_include_pre_h1": True,
            "pre_h1_ticks": sorted(PRE_H1_TICKS),
            "strictly_between_sampled_horizon_ticks": sorted(
                STRICTLY_BETWEEN_SAMPLED_HORIZON_TICKS),
        },
        "visual_observability_limitation": (
            "No RGB pixels or latents are opened. A physical component active at a "
            "sampled endpoint is not thereby proven visually observable, and a "
            "component confined to ticks 5-8, 10-13, or 15-18 is not thereby "
            "proven visually invisible."
        ),
        "prior_lineage_evidence_limitation": {
            "prior_trace_count": EXPECTED_PRIOR_LINEAGE_TRACES,
            "contact_type_evidence": "NOT_RETAINED_IN_V1_3_TRACE",
            "prior_used_as_final_diagnostic_row_count": CONTRACT.ADOPTED_TRACES,
            "new_complete_contact_type_trace_count": EXPECTED_NEW_REPLAYS,
        },
        "overall": _group_summary(branch_records),
        "by_family": {
            family: _group_summary([
                row for row in branch_records if row["family"] == family
            ]) for family in families
        },
        "by_stratum": {
            stratum: _group_summary([
                row for row in branch_records if row["stratum"] == stratum
            ]) for stratum in strata
        },
        "by_family_stratum": {
            f"{family}/{stratum}": _group_summary([
                row for row in branch_records
                if row["family"] == family and row["stratum"] == stratum
            ])
            for family in families for stratum in strata
            if any(row["family"] == family and row["stratum"] == stratum
                   for row in branch_records)
        },
        "branches": branch_records,
        "frozen_scorer_target_fields_consulted": list(AGGREGATE_LABEL_FIELDS),
        "frozen_scorer_target_source_counts": copy.deepcopy(
            manifest["frozen_replay_target_source_counts"]),
        "all_replay_aggregates_match_frozen_scorer_targets": True,
        "scientific_corpus_labels_modified": False,
        "tick_physical_evidence_read": True,
        "rgb_pixels_read": 0,
        "latents_accessed": 0,
        "scorer_outputs_read": 0,
    }
    return signed(payload, AUDIT_SELF_KEY)


def issue_audit(*, out_root: Path = OUT_ROOT,
                v2_root: Path = V13.V2_ROOT,
                v13_root: Path = V13.OUT_ROOT) -> dict[str, Any]:
    plan, corpus, prior = load_plan(
        out_root=out_root, v2_root=v2_root, v13_root=v13_root)
    terminal = validate_signed(
        _read_json(terminal_path(out_root), "diagnostic terminal manifest"),
        TERMINAL_SELF_KEY, "diagnostic terminal manifest",
    )
    require(terminal == compile_terminal_manifest(
        out_root=out_root, v2_root=v2_root, v13_root=v13_root),
        "diagnostic terminal manifest changed")
    old_by_identity = {
        row["branch_identity_digest"]: row for row in corpus["rows"]
    }
    entry_by_identity = {
        row["branch_identity_digest"]: row for row in plan["entries"]
    }
    trace_rows = []
    for item in terminal["rows"]:
        identity = item["branch_identity_digest"]
        path = out_root / item["path"]
        require(file_sha256(path) == item["sha256"],
                "terminal trace-row bytes changed")
        trace_rows.append(_validate_trace_row(
            _read_json(path, "diagnostic trace row"),
            entry=entry_by_identity[identity], old_row=old_by_identity[identity],
            prior=prior.get(identity), plan=plan, out_root=out_root,
        ))
    audit = build_audit(terminal, trace_rows)
    _atomic_json(audit_path(out_root), audit, out_root=out_root, idempotent=True)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("issue-plan", "replay", "compile", "audit"))
    parser.add_argument("--backend", default="cpu", choices=("cpu",))
    args = parser.parse_args(argv)
    if args.stage == "issue-plan":
        result = issue_plan()
    elif args.stage == "replay":
        result = execute_replays(backend=args.backend)
    elif args.stage == "compile":
        result = compile_terminal_manifest()
    else:
        result = issue_audit()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
