#!/usr/bin/env python3
"""Read-only diagnosis of the 18 frozen scorer-fit V2 graph-label refusals.

This tool never imports Genesis, opens a predictor/checkpoint, or writes below
``.generated``.  It validates the completed branch corpus, reconstructs the
five preserved block-boundary base poses from camera receipts, and replays the
frozen locate/geodesic operations against the bound scene manifests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
from collections import Counter, deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCORER_FIT_RELATIVE = Path(".generated/go2_branch_corpus_v1_2/scorer_fit")

AUDITED_SOURCE_COMMIT = "5c67135ad83b9206e6520e507f1ecaf980fd3d8d"
EXPECTED_CORPUS_DIGEST = (
    "5216e2182a4e165a673714fcccbd6b769d01fa565a69a466b3cab066ab01ccc3"
)
EXPECTED_ATTEMPTED = 1_440
EXPECTED_VALID = 1_422
EXPECTED_INVALID = 18
EXPECTED_STATE_COUNT = 120
EXPECTED_FIT_STATES = 96
EXPECTED_CALIBRATION_STATES = 24
EXPECTED_CANDIDATES = tuple(range(12))
EXPECTED_INVALID_REASON = "unlocatable_or_unreachable_geodesic"

LOCATE_MAX_DISTANCE_M = 2.0
CAMERA_MOUNT_XYZ_BODY_M = np.asarray([0.326, 0.0, 0.043], dtype=np.float64)
CAMERA_MOUNT_RPY_BODY_RAD = (0.0, 0.0, 0.0)
BLOCK_ENDPOINT_TICKS = (4, 9, 14, 19)

CATEGORIES = {
    "LOCATOR_IMPLEMENTATION_DEFECT",
    "OFF_NAVIGABLE_GRAPH_OUTCOME",
    "LOCATABLE_GOAL_UNREACHABLE_OUTCOME",
    "INSUFFICIENT_TRACE_FOR_LABEL",
    "OTHER",
}

SOURCE_PATHS = (
    "scripts/diagnose_go2_scorer_fit_v2_graph_label_failures.py",
    "lewm/tests/test_diagnose_go2_scorer_fit_v2_graph_label_failures.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/run_go2_oracle_branch_pilot_v1_2.py",
    "lewm/oracle/go2_branch_oracle_v1_2.py",
    "lewm/oracle/go2_textured_v03_renderer.py",
    "lewm_worlds/lewm_worlds/scene_graph.py",
    "lewm_genesis/lewm_genesis/render_replay.py",
    "config/go2_platform_manifest.yaml",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def builder_digest(value: Any) -> str:
    """Digest dialect used by the frozen corpus producer."""

    return hashlib.sha256(
        json.dumps(_jsonable(value), sort_keys=True).encode()
    ).hexdigest()


def audit_digest(value: Any) -> str:
    """Compact canonical digest dialect used only by this diagnostic."""

    return hashlib.sha256(
        json.dumps(
            _jsonable(value), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _sha256_regular_file(path: Path) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    noatime = getattr(os, "O_NOATIME", 0)
    try:
        descriptor = os.open(path, flags | noatime)
    except PermissionError:
        descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise RuntimeError(f"file changed while hashing: {path}")
        return digest.hexdigest(), int(after.st_size)
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> list[float]:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2, 1] - rotation[1, 2]) / scale
        qy = (rotation[0, 2] - rotation[2, 0]) / scale
        qz = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rotation[0, 1] + rotation[1, 0]) / scale
            qz = (rotation[0, 2] + rotation[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / scale
            qx = (rotation[0, 1] + rotation[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / scale
            qx = (rotation[0, 2] + rotation[2, 0]) / scale
            qy = (rotation[1, 2] + rotation[2, 1]) / scale
            qz = 0.25 * scale
    quaternion = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return [float(value) for value in quaternion]


def recover_base_pose(camera_pose_world: Mapping[str, Any]) -> dict[str, Any]:
    """Invert the frozen zero-RPY camera mount at persisted float32 precision."""

    camera_position = np.asarray(camera_pose_world["position"], dtype=np.float64)
    lookat = np.asarray(camera_pose_world["lookat"], dtype=np.float64)
    up_raw = np.asarray(camera_pose_world["up"], dtype=np.float64)
    if camera_position.shape != (3,) or lookat.shape != (3,) or up_raw.shape != (3,):
        raise RuntimeError("camera pose vectors must be 3D")
    forward = lookat - camera_position
    forward_norm = float(np.linalg.norm(forward))
    if not math.isfinite(forward_norm) or forward_norm <= 0.0:
        raise RuntimeError("camera forward vector is invalid")
    forward /= forward_norm
    up = up_raw - forward * float(np.dot(up_raw, forward))
    up_norm = float(np.linalg.norm(up))
    if not math.isfinite(up_norm) or up_norm <= 0.0:
        raise RuntimeError("camera up vector is invalid")
    up /= up_norm
    body_y = np.cross(up, forward)
    body_y /= np.linalg.norm(body_y)
    rotation = np.column_stack((forward, body_y, up))
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-6):
        raise RuntimeError("reconstructed base rotation is not orthonormal")
    base_position = camera_position - rotation @ CAMERA_MOUNT_XYZ_BODY_M

    pitch = math.asin(max(-1.0, min(1.0, -float(rotation[2, 0]))))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return {
        "position_world_xyz_m": [float(value) for value in base_position],
        "quaternion_world_wxyz": _rotation_matrix_to_quaternion_wxyz(rotation),
        "roll_pitch_yaw_world_rad": [roll, pitch, yaw],
        "provenance": (
            "inverse of persisted camera_pose_world under frozen mount "
            "xyz=[0.326,0,0.043], rpy=[0,0,0]; precision is limited to the "
            "persisted float32 forward transform"
        ),
    }


def _independent_bfs_distance(
    graph: Any,
    start: int,
    goal: int,
    blocked: Iterable[int] = (),
) -> int | None:
    start, goal = int(start), int(goal)
    if start == goal:
        return 0
    blocked_set = frozenset(int(value) for value in blocked)
    visited = {start}
    frontier: deque[tuple[int, int]] = deque([(start, 0)])
    while frontier:
        node, depth = frontier.popleft()
        if node != start and node in blocked_set:
            continue
        for neighbour in graph.neighbors(node):
            neighbour = int(neighbour)
            if neighbour in visited:
                continue
            if neighbour == goal:
                return depth + 1
            visited.add(neighbour)
            frontier.append((neighbour, depth + 1))
    return None


def _world_contains(bounds: Sequence[Sequence[float]], xy: Sequence[float]) -> bool:
    return bool(
        float(bounds[0][0]) <= float(xy[0]) <= float(bounds[1][0])
        and float(bounds[0][1]) <= float(xy[1]) <= float(bounds[1][1])
    )


def graph_point_evidence(
    graph: Any,
    field: Any,
    pose: Mapping[str, Any],
    *,
    goal_cell: int,
    tick: int | None,
    label: str,
) -> dict[str, Any]:
    xy = tuple(float(value) for value in pose["position_world_xyz_m"][:2])
    hit = graph.locate(xy)
    distances = [
        (math.hypot(xy[0] - float(center[0]), xy[1] - float(center[1])), index)
        for index, center in enumerate(graph.node_xy)
    ]
    distances.sort(key=lambda item: (item[0], item[1]))
    brute_distance, brute_cell = distances[0]
    locator_matches_bruteforce = bool(
        int(hit.cell_id) == int(brute_cell)
        and math.isclose(float(hit.distance_m), float(brute_distance), rel_tol=0.0, abs_tol=1e-12)
    )
    if not locator_matches_bruteforce:
        raise RuntimeError("SceneGraph.locate disagrees with brute-force nearest node")

    located = bool(float(hit.distance_m) <= LOCATE_MAX_DISTANCE_M)
    remaining = (
        float(field.remaining_distance(xy, int(hit.cell_id))) if located else math.inf
    )
    raw_hops = _independent_bfs_distance(graph, int(hit.cell_id), int(goal_cell))
    blocked_hops = _independent_bfs_distance(
        graph, int(hit.cell_id), int(goal_cell), graph.nav_blocked_cells
    )
    graph_raw_hops = graph.bfs_distance(int(hit.cell_id), int(goal_cell))
    graph_blocked_hops = graph.bfs_distance(
        int(hit.cell_id), int(goal_cell), transit_blocked=graph.nav_blocked_cells
    )
    if raw_hops != graph_raw_hops or blocked_hops != graph_blocked_hops:
        raise RuntimeError("independent BFS disagrees with SceneGraph.bfs_distance")
    if located and (math.isfinite(remaining) != (blocked_hops is not None)):
        raise RuntimeError("GeodesicField reachability disagrees with blocked BFS")

    nearest_nodes = []
    for distance, cell in distances[:5]:
        center = graph.cell_center(cell)
        nearest_nodes.append({
            "cell_id": int(cell),
            "center_xy_m": [float(center[0]), float(center[1])],
            "distance_m": float(distance),
            "nav_blocked": bool(cell in graph.nav_blocked_cells),
            "raw_goal_hops": _independent_bfs_distance(graph, cell, goal_cell),
            "blocked_goal_hops": _independent_bfs_distance(
                graph, cell, goal_cell, graph.nav_blocked_cells
            ),
        })

    node_x = np.asarray(graph.node_xy[:, 0], dtype=np.float64)
    node_y = np.asarray(graph.node_xy[:, 1], dtype=np.float64)
    return {
        "label": label,
        "global_tick": tick,
        "base_pose": copy.deepcopy(dict(pose)),
        "nearest_cell_id": int(hit.cell_id),
        "nearest_cell_distance_m": float(hit.distance_m),
        "locator_matches_bruteforce_nearest_node": locator_matches_bruteforce,
        "located_under_frozen_2m_rule": located,
        "geodesic_distance_m": float(remaining) if math.isfinite(remaining) else None,
        "raw_graph_goal_hops": raw_hops,
        "nav_blocked_goal_hops": blocked_hops,
        "goal_reachable_under_frozen_mask": blocked_hops is not None,
        "inside_scene_world_bounds": _world_contains(
            graph.manifest.world_bounds_xy_m, xy
        ),
        "inside_graph_node_center_bounds": bool(
            float(node_x.min()) <= xy[0] <= float(node_x.max())
            and float(node_y.min()) <= xy[1] <= float(node_y.max())
        ),
        "clearance_to_walls_and_obstacles_m": float(
            graph.clearance_to_walls(xy, include_landmarks=False)
        ),
        "nearest_graph_nodes": nearest_nodes,
    }


def classify_final_evidence(final: Mapping[str, Any]) -> tuple[str, str]:
    if not bool(final["locator_matches_bruteforce_nearest_node"]):
        return "LOCATOR_IMPLEMENTATION_DEFECT", "locate"
    if not bool(final["located_under_frozen_2m_rule"]):
        return "OFF_NAVIGABLE_GRAPH_OUTCOME", "locate"
    if final["geodesic_distance_m"] is None:
        return "LOCATABLE_GOAL_UNREACHABLE_OUTCOME", "bfs_distance/geodesic"
    return "OTHER", "none"


def _first_unavailable(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    last_valid_tick: int | None = None
    for point in points:
        available = bool(
            point["located_under_frozen_2m_rule"]
            and point["geodesic_distance_m"] is not None
        )
        if available:
            last_valid_tick = point["global_tick"]
            continue
        observed_tick = point["global_tick"]
        return {
            "exact_first_unavailable_tick": None,
            "exact_tick_status": "NOT_PRESERVED_IN_V2_ROW",
            "first_observed_unavailable_block_endpoint_tick": observed_tick,
            "possible_first_unavailable_tick_range_inclusive": [0, int(observed_tick)],
            "range_caveat": (
                "block-end validity does not exclude an earlier transient failure "
                "followed by recovery"
            ),
            "exact_last_valid_tick": None,
            "last_observed_valid_block_endpoint_tick": last_valid_tick,
        }
    return {
        "exact_first_unavailable_tick": None,
        "exact_tick_status": "NO_UNAVAILABLE_PRESERVED_ENDPOINT",
        "first_observed_unavailable_block_endpoint_tick": None,
        "possible_first_unavailable_tick_range_inclusive": None,
        "range_caveat": "no unavailable preserved endpoint",
        "exact_last_valid_tick": None,
        "last_observed_valid_block_endpoint_tick": last_valid_tick,
    }


def _digest_fields(row: Mapping[str, Any]) -> dict[str, str]:
    result = {}
    for key, value in row.items():
        if (
            isinstance(value, str)
            and len(value) == 64
            and (key.endswith("_digest") or key.endswith("_sha256"))
        ):
            result[str(key)] = value
    return dict(sorted(result.items()))


def _validate_manifest_self_digest(payload: Mapping[str, Any], key: str) -> str:
    expected = payload.get(key)
    body = dict(payload)
    body.pop(key, None)
    actual = builder_digest(body)
    if expected != actual:
        raise RuntimeError(f"{key} self-digest mismatch")
    return actual


def _load_corpus_and_verify(
    root: Path, scorer_fit: Path
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    state_manifest_path = scorer_fit / "state_manifest_v2.json"
    assignment_manifest_path = scorer_fit / "full_bank_assignment_manifest_v2.json"
    receipt_path = scorer_fit / "corpus_receipt_v2.json"
    ledger_path = scorer_fit / "branch_rows_v2.jsonl"
    state_manifest = _read_json(state_manifest_path)
    assignment_manifest = _read_json(assignment_manifest_path)
    receipt = _read_json(receipt_path)
    _validate_manifest_self_digest(state_manifest, "state_manifest_digest")
    _validate_manifest_self_digest(
        assignment_manifest, "full_bank_assignment_manifest_digest"
    )
    identity_payload = receipt.get("corpus_digest_payload")
    if not isinstance(identity_payload, Mapping):
        raise RuntimeError("corpus identity payload is missing")
    if receipt.get("corpus_digest") != builder_digest(identity_payload):
        raise RuntimeError("corpus receipt digest mismatch")
    if receipt["corpus_digest"] != EXPECTED_CORPUS_DIGEST:
        raise RuntimeError("unexpected corpus digest")
    expected_counts = (
        receipt.get("attempted_branches"),
        receipt.get("valid_branches"),
        receipt.get("invalid_branches"),
        receipt.get("states"),
    )
    if expected_counts != (
        EXPECTED_ATTEMPTED,
        EXPECTED_VALID,
        EXPECTED_INVALID,
        EXPECTED_STATE_COUNT,
    ):
        raise RuntimeError(f"unexpected corpus counts: {expected_counts}")
    expected_identity = {
        "state_manifest_digest": state_manifest["state_manifest_digest"],
        "full_bank_assignment_manifest_digest": assignment_manifest[
            "full_bank_assignment_manifest_digest"
        ],
        "state_count": EXPECTED_STATE_COUNT,
        "attempted_branch_count": EXPECTED_ATTEMPTED,
        "valid_branch_count": EXPECTED_VALID,
        "invalid_branch_count": EXPECTED_INVALID,
        "complete": True,
    }
    for key, value in expected_identity.items():
        if identity_payload.get(key) != value:
            raise RuntimeError(f"corpus identity payload mismatch: {key}")
    if receipt.get("state_manifest_digest") != expected_identity[
        "state_manifest_digest"
    ]:
        raise RuntimeError("receipt/state-manifest binding mismatch")
    if receipt.get("full_bank_assignment_manifest_digest") != expected_identity[
        "full_bank_assignment_manifest_digest"
    ]:
        raise RuntimeError("receipt/assignment-manifest binding mismatch")

    ledger_sha, ledger_bytes = _sha256_regular_file(ledger_path)
    if (
        ledger_sha != receipt["branch_rows_sha256"]
        or ledger_sha != identity_payload.get("branch_rows_sha256")
    ):
        raise RuntimeError("ledger raw digest mismatch")
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != EXPECTED_ATTEMPTED:
        raise RuntimeError("ledger row count mismatch")
    if sum(row.get("valid") is True for row in rows) != EXPECTED_VALID:
        raise RuntimeError("valid row count mismatch")
    invalid_rows = [row for row in rows if row.get("valid") is False]
    if len(invalid_rows) != EXPECTED_INVALID or any(
        row.get("invalid_reason") != EXPECTED_INVALID_REASON for row in invalid_rows
    ):
        raise RuntimeError("invalid row inventory mismatch")
    if [row.get("branch_row_digest") for row in rows] != identity_payload.get(
        "branch_row_digests"
    ):
        raise RuntimeError("ordered row digests do not match corpus identity payload")
    branch_identity_set_digest = builder_digest(sorted(
        str(row.get("branch_identity_digest")) for row in rows
    ))
    if branch_identity_set_digest != identity_payload.get(
        "branch_identity_set_digest"
    ):
        raise RuntimeError("branch identity set does not match corpus identity payload")
    bound_digests = identity_payload.get("bound_digests")
    if not isinstance(bound_digests, Mapping) or any(
        receipt.get(key) != value for key, value in bound_digests.items()
    ):
        raise RuntimeError("receipt lineage does not match corpus identity payload")
    if any(
        row.get(key) != value
        for row in rows
        for key, value in bound_digests.items()
    ):
        raise RuntimeError("row lineage does not match corpus identity payload")

    state_by_id = {state["state_id"]: state for state in state_manifest["states"]}
    state_rows: dict[str, list[dict[str, Any]]] = {
        state_id: [] for state_id in state_by_id
    }
    row_record_inventory = []
    frame_bindings: dict[str, dict[str, Any]] = {}
    frame_reference_count = 0
    for row in rows:
        digest = row.get("branch_row_digest")
        body = dict(row)
        body.pop("branch_row_digest", None)
        if digest != builder_digest(body):
            raise RuntimeError(f"branch row self-digest mismatch: {row.get('branch_identity_digest')}")
        state = state_by_id.get(row.get("state_id"))
        if state is None or row.get("state_identity_digest") != state["state_identity_digest"]:
            raise RuntimeError("row/state identity join mismatch")
        state_rows[row["state_id"]].append(row)
        row_path = scorer_fit / "row_records_v2" / f"{row['branch_identity_digest']}.json"
        row_raw_sha, row_raw_bytes = _sha256_regular_file(row_path)
        if _read_json(row_path) != row:
            raise RuntimeError(f"ledger/row-record mismatch: {row_path}")
        row_record_inventory.append({
            "branch_identity_digest": row["branch_identity_digest"],
            "sha256": row_raw_sha,
            "byte_count": row_raw_bytes,
        })
        for frame in [*row.get("context_frames", []), *row.get("horizon_frames", [])]:
            frame_reference_count += 1
            path = str(frame["path"])
            binding = {
                "path": path,
                "sha256": frame["sha256"],
                "byte_count": int(frame["byte_count"]),
            }
            if path in frame_bindings and frame_bindings[path] != binding:
                raise RuntimeError(f"shared frame binding mismatch: {path}")
            frame_bindings[path] = binding

    state_bank_projection = []
    for state in state_manifest["states"]:
        bank = sorted(state_rows[state["state_id"]], key=lambda row: row["candidate_index"])
        if [row["candidate_index"] for row in bank] != list(EXPECTED_CANDIDATES):
            raise RuntimeError(f"candidate bank incomplete: {state['state_id']}")
        prebranch_projections = [{
            "snapshot_digest": row["snapshot_digest"],
            "context_frames": row["context_frames"],
            "proprio": row["proprio"],
            "control": row["control"],
            "goal": row["goal"],
            "previous_applied_command": row["previous_applied_command"],
        } for row in bank]
        first_projection_digest = audit_digest(prebranch_projections[0])
        if any(audit_digest(value) != first_projection_digest for value in prebranch_projections):
            raise RuntimeError(f"prebranch trace projection differs within state: {state['state_id']}")
        state_bank_projection.append({
            "state_id": state["state_id"],
            "state_identity_digest": state["state_identity_digest"],
            "split_role": state["split_role"],
            "prebranch_trace_projection_digest": first_projection_digest,
            "branch_row_digests": [row["branch_row_digest"] for row in bank],
        })

    frame_inventory = []
    for relative, binding in sorted(frame_bindings.items()):
        digest, byte_count = _sha256_regular_file(scorer_fit / relative)
        if digest != binding["sha256"] or byte_count != binding["byte_count"]:
            raise RuntimeError(f"frame binding mismatch: {relative}")
        frame_inventory.append(binding)

    roles = Counter(state["split_role"] for state in state_manifest["states"])
    if roles != Counter({"fit": EXPECTED_FIT_STATES, "calibration": EXPECTED_CALIBRATION_STATES}):
        raise RuntimeError(f"state role counts changed: {roles}")

    integrity = {
        "state_count": len(state_manifest["states"]),
        "fit_state_count": roles["fit"],
        "calibration_state_count": roles["calibration"],
        "branch_row_count": len(rows),
        "valid_branch_row_count": sum(row["valid"] is True for row in rows),
        "invalid_branch_row_count": sum(row["valid"] is False for row in rows),
        "row_record_count": len(row_record_inventory),
        "frame_reference_count": frame_reference_count,
        "unique_frame_file_count": len(frame_inventory),
        "all_row_self_digests_valid": True,
        "all_row_records_equal_compiled_ledger": True,
        "corpus_identity_payload_cross_bindings_valid": True,
        "ordered_branch_row_digests_equal_corpus_payload": True,
        "branch_identity_set_digest": branch_identity_set_digest,
        "all_frame_raw_digests_and_byte_counts_valid": True,
        "all_state_candidate_banks_complete_0_through_11": True,
        "all_state_prebranch_trace_projections_consistent": True,
        "ledger_sha256": ledger_sha,
        "ledger_byte_count": ledger_bytes,
        "row_record_inventory_digest": audit_digest(row_record_inventory),
        "frame_inventory_digest": audit_digest(frame_inventory),
        "state_trace_bank_digest": audit_digest(state_bank_projection),
    }
    return state_manifest, assignment_manifest, invalid_rows, integrity


def _scene_path(root: Path, state: Mapping[str, Any]) -> Path:
    value = Path(str(state["scene_dir"]))
    if value.is_absolute():
        try:
            value = root / value.relative_to(root)
        except ValueError:
            pass
    return value / "manifest.json"


def _state_identity_projection(states: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [{
        "state_index": int(state["state_index"]),
        "state_id": state["state_id"],
        "state_identity_digest": state["state_identity_digest"],
        "split_role": state["split_role"],
        "family": state["family"],
        "stratum": state["stratum"],
        "scene_id": state["scene_id"],
    } for state in states]


def _source_bindings(root: Path) -> dict[str, dict[str, Any]]:
    bindings = {}
    for relative in SOURCE_PATHS:
        digest, byte_count = _sha256_regular_file(root / relative)
        bindings[relative] = {"sha256": digest, "byte_count": byte_count}
    return bindings


def build_report(root: Path = ROOT) -> dict[str, Any]:
    scorer_fit = root / SCORER_FIT_RELATIVE
    platform = yaml.safe_load((root / "config/go2_platform_manifest.yaml").read_text())
    camera = platform["camera"]
    if (
        [float(value) for value in camera["xyz_body_m"]]
        != CAMERA_MOUNT_XYZ_BODY_M.tolist()
        or tuple(float(value) for value in camera["rpy_body_rad"])
        != CAMERA_MOUNT_RPY_BODY_RAD
    ):
        raise RuntimeError("frozen camera mount changed")

    # Imports remain inside the read-only operation so importing this module in
    # a synthetic test cannot initialize any simulator/runtime package.
    import sys
    for package_root in (root, root / "lewm_worlds"):
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
    from lewm.oracle.go2_branch_oracle_v1_2 import GeodesicField
    from lewm_worlds.manifest import parse_scene_manifest_dict
    from lewm_worlds.scene_graph import SceneGraph

    state_manifest, assignment_manifest, invalid_rows, integrity = (
        _load_corpus_and_verify(root, scorer_fit)
    )
    state_by_id = {state["state_id"]: state for state in state_manifest["states"]}
    records = []
    scene_cache: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
    for row in sorted(invalid_rows, key=lambda value: (value["state_index"], value["candidate_index"])):
        state = state_by_id[row["state_id"]]
        scene_path = _scene_path(root, state)
        scene_sha, scene_bytes = _sha256_regular_file(scene_path)
        if scene_sha != state["scene_manifest_sha256"] or scene_bytes != state["scene_manifest_byte_count"]:
            raise RuntimeError(f"scene manifest binding mismatch: {scene_path}")
        if row["scene_id"] not in scene_cache:
            raw_scene = _read_json(scene_path)
            manifest = parse_scene_manifest_dict(raw_scene)
            graph = SceneGraph(manifest)
            scene_cache[row["scene_id"]] = (manifest, graph, raw_scene)
        manifest, graph, _raw_scene = scene_cache[row["scene_id"]]
        goal_cell = int(row["goal"]["landmark_cell"])
        field = GeodesicField(
            graph, goal_cell, transit_blocked=graph.nav_blocked_cells
        )
        camera_receipts = [row["context_frames"][-1], *row["horizon_frames"]]
        point_labels = ["snapshot_start", "h1", "h2", "h3", "h4_final"]
        point_ticks: list[int | None] = [None, *BLOCK_ENDPOINT_TICKS]
        points = [
            graph_point_evidence(
                graph,
                field,
                recover_base_pose(frame["camera_pose_world"]),
                goal_cell=goal_cell,
                tick=tick,
                label=label,
            )
            for frame, tick, label in zip(camera_receipts, point_ticks, point_labels)
        ]
        start = points[0]
        if (
            start["nearest_cell_id"] != state["cell_id"]
            or not start["located_under_frozen_2m_rule"]
            or start["geodesic_distance_m"] is None
            or not math.isclose(
                float(start["geodesic_distance_m"]),
                float(state["goal"]["start_geodesic_m"]),
                rel_tol=0.0,
                abs_tol=5e-6,
            )
        ):
            raise RuntimeError(f"camera-derived snapshot start disagrees with state: {row['state_id']}")
        final = points[-1]
        category, source_operation = classify_final_evidence(final)
        if category not in CATEGORIES or category == "OTHER":
            raise RuntimeError(f"unexplained invalid row: {row['branch_identity_digest']}")
        row_path = scorer_fit / "row_records_v2" / f"{row['branch_identity_digest']}.json"
        row_raw_sha, row_raw_bytes = _sha256_regular_file(row_path)
        first_unavailable = _first_unavailable(points)
        records.append({
            "record_index": len(records),
            "split_role": row["split_role"],
            "split": row["split"],
            "family": row["family"],
            "stratum": row["stratum"],
            "scene_id": row["scene_id"],
            "state_id": row["state_id"],
            "state_index": int(row["state_index"]),
            "state_identity_digest": row["state_identity_digest"],
            "candidate_index": int(row["candidate_index"]),
            "candidate": row["candidate"],
            "primitives": row["primitives"],
            "branch_identity_digest": row["branch_identity_digest"],
            "assignment_identity_digest": row["assignment_identity_digest"],
            "branch_row_digest": row["branch_row_digest"],
            "row_record": {
                "logical_path": str(
                    SCORER_FIT_RELATIVE / "row_records_v2" / f"{row['branch_identity_digest']}.json"
                ),
                "sha256": row_raw_sha,
                "byte_count": row_raw_bytes,
            },
            "snapshot_digest": row["snapshot_digest"],
            "designated_goal": copy.deepcopy(row["goal"]),
            "snapshot_start_graph_evidence": {
                "frozen_state_cell_id": int(state["cell_id"]),
                "frozen_state_start_geodesic_m": float(state["goal"]["start_geodesic_m"]),
                "camera_derived": start,
            },
            "offline_block_endpoint_graph_replay": points[1:],
            "first_unavailable_evidence": first_unavailable,
            "final_continuous_robot_pose": copy.deepcopy(final["base_pose"]),
            "final_graph_result": {
                key: copy.deepcopy(final[key])
                for key in (
                    "nearest_cell_id",
                    "nearest_cell_distance_m",
                    "located_under_frozen_2m_rule",
                    "geodesic_distance_m",
                    "raw_graph_goal_hops",
                    "nav_blocked_goal_hops",
                    "goal_reachable_under_frozen_mask",
                    "inside_scene_world_bounds",
                    "inside_graph_node_center_bounds",
                    "clearance_to_walls_and_obstacles_m",
                    "nearest_graph_nodes",
                )
            },
            "failure_source_operation": source_operation,
            "primary_category": category,
            "category_explanation": (
                "final pose is inside scene bounds but exceeds the frozen 2.0 m "
                "nearest-node coverage radius"
                if category == "OFF_NAVIGABLE_GRAPH_OUTCOME"
                else "final pose has an exact nearest graph cell, but every raw path "
                "to the goal transits a cell blocked by the frozen nav mask"
            ),
            "execution_and_trace_evidence": {
                "invalid_reason": row["invalid_reason"],
                "blocks_completed": int(row["blocks_completed"]),
                "ticks_per_block": int(row["timing"]["ticks_per_block"]),
                "branch_otherwise_completed_all_twenty_ticks": bool(
                    row["blocks_completed"] == 4 and row["truncated_at_block"] is None
                ),
                "truncated_at_block": row["truncated_at_block"],
                "rgb_block_endpoint_receipts_complete": bool(
                    len(row["horizon_frames"]) == 4
                    and row["masks"]["target_rgb_valid"] == [True] * 4
                ),
                "rgb_complete_twenty_tick_trace_preserved": False,
                "observed_prebranch_proprio_shape": [len(row["proprio"]), len(row["proprio"][0])],
                "future_proprioception_preserved": False,
                "full_twenty_tick_base_pose_trace_preserved": False,
                "recoverable_pose_samples": "snapshot plus block-end ticks 4,9,14,19",
                "contact_trace_preserved": False,
                "clearance_trace_preserved": False,
                "stuck_trace_preserved": False,
                "fall_or_termination_trace_preserved": False,
                "completion_event_trace_preserved": False,
                "path_event_status": {
                    "contact_events": "UNKNOWN_NOT_PRESERVED",
                    "clearance_sequence": "UNKNOWN_NOT_PRESERVED",
                    "stuck_events": "UNKNOWN_NOT_PRESERVED",
                    "fall_events": "UNKNOWN_NOT_PRESERVED",
                    "termination_events": "UNKNOWN_NOT_PRESERVED",
                    "completion_events": "UNKNOWN_NOT_PRESERVED",
                    "block_end_truncation_observed": False,
                },
                "oracle_aggregate_fields": {
                    key: row[key]
                    for key in (
                        "start_geodesic_m",
                        "final_geodesic_m",
                        "progress",
                        "contact_fraction",
                        "clearance_cost",
                        "stuck_fraction",
                        "fall",
                        "safety",
                        "completion",
                        "utility",
                        "min_clearance_m",
                        "evaluation_points",
                    )
                },
            },
            "requested_and_post_slew_action": {
                "previous_applied_command": row["previous_applied_command"],
                "requested": row["requested"],
                "realised_requested_prefix": row["realised_requested_prefix"],
                "post_slew": row["post_slew"],
                "candidate_post_slew_plan": row["candidate_post_slew_plan"],
            },
            "source_digests": {
                **_digest_fields(row),
                "scene_manifest_sha256": scene_sha,
            },
        })

    def counted(key: str) -> dict[str, int]:
        return dict(sorted(Counter(str(record[key]) for record in records).items()))

    observed_tick_counts = Counter(
        str(record["first_unavailable_evidence"]["first_observed_unavailable_block_endpoint_tick"])
        for record in records
    )
    possible_range_counts = Counter(
        "-".join(
            str(value)
            for value in record["first_unavailable_evidence"][
                "possible_first_unavailable_tick_range_inclusive"
            ]
        )
        for record in records
    )
    states = state_manifest["states"]
    fit_states = [state for state in states if state["split_role"] == "fit"]
    calibration_states = [state for state in states if state["split_role"] == "calibration"]
    fit_projection = _state_identity_projection(fit_states)
    calibration_projection = _state_identity_projection(calibration_states)

    report: dict[str, Any] = {
        "schema": "go2_scorer_fit_v2_graph_label_failure_diagnostic_v1",
        "status": "DIAGNOSTIC_ONLY_NO_RELABEL_NO_EXECUTION",
        "audited_source_commit": AUDITED_SOURCE_COMMIT,
        "corpus": {
            "logical_root": str(SCORER_FIT_RELATIVE),
            "corpus_digest": EXPECTED_CORPUS_DIGEST,
            "attempted_branch_records": EXPECTED_ATTEMPTED,
            "oracle_labelled_branch_records": EXPECTED_VALID,
            "unlocatable_or_unreachable_records": EXPECTED_INVALID,
            "mutated_by_diagnostic": False,
        },
        "source_bindings": _source_bindings(root),
        "artifact_bindings": {
            "state_manifest_digest": state_manifest["state_manifest_digest"],
            "full_bank_assignment_manifest_digest": assignment_manifest[
                "full_bank_assignment_manifest_digest"
            ],
            "corpus_digest": EXPECTED_CORPUS_DIGEST,
            "branch_rows_sha256": integrity["ledger_sha256"],
        },
        "integrity_verification": integrity,
        "identity_verification": {
            "fit_state_count": len(fit_projection),
            "calibration_state_count": len(calibration_projection),
            "fit_identity_projection_digest": audit_digest(fit_projection),
            "calibration_identity_projection_digest": audit_digest(calibration_projection),
            "all_identity_projection_digest": audit_digest(
                _state_identity_projection(states)
            ),
            "fit_identities": fit_projection,
            "calibration_identities": calibration_projection,
        },
        "calibration_disposition": {
            "historical_identity_role_preserved": "calibration",
            "prospective_status_for_all_24": "DEVELOPMENT_ONLY_PENDING_NEXT_DECISION",
            "untouched_qualification_data_claim_allowed": False,
            "future_qualification_or_final_evaluation_eligibility": False,
            "discarded": False,
            "identity_replaced_or_reclassified_in_frozen_artifacts": False,
            "reason": (
                "all 24 calibration identities participated in diagnosis of an "
                "outcome-dependent label boundary"
            ),
        },
        "failure_inventory": records,
        "counts": {
            "by_family": counted("family"),
            "by_candidate_index_and_name": dict(sorted(Counter(
                f"{record['candidate_index']}:{record['candidate']}" for record in records
            ).items())),
            "by_fit_or_calibration": counted("split_role"),
            "by_stratum": counted("stratum"),
            "by_primary_category": {
                category: Counter(
                    record["primary_category"] for record in records
                )[category]
                for category in sorted(CATEGORIES)
            },
            "by_failure_source_operation": counted("failure_source_operation"),
            "by_exact_first_failure_tick": {"NOT_PRESERVED_IN_V2_ROW": 18},
            "by_first_observed_unavailable_block_endpoint_tick": dict(
                sorted(observed_tick_counts.items())
            ),
            "by_possible_first_failure_tick_range_inclusive": dict(
                sorted(possible_range_counts.items())
            ),
        },
        "classification_summary": {
            "single_mechanism": False,
            "mechanisms": [
                {
                    "category": "OFF_NAVIGABLE_GRAPH_OUTCOME",
                    "count": 4,
                    "finding": (
                        "in-bounds continuous poses outside the frozen sparse graph's "
                        "2.0 m location coverage"
                    ),
                },
                {
                    "category": "LOCATABLE_GOAL_UNREACHABLE_OUTCOME",
                    "count": 14,
                    "finding": (
                        "nearest cells are raw-graph connected to the goal but unreachable "
                        "under the frozen transit-blocked mask"
                    ),
                },
            ],
            "graph_localisation_implementation_defect_exists": False,
            "bfs_or_dijkstra_implementation_defect_exists": False,
            "diagnostic_observability_implementation_defect_exists": True,
            "diagnostic_observability_explanation": (
                "the producer retained start/tick locate, geodesic, pose, contact, "
                "clearance, stuck, termination, and completion evidence only in memory; "
                "the durable row collapses locate and reachability failures into one code"
            ),
            "physically_meaningful_off_graph_outcomes_exist": True,
            "off_graph_nuance": (
                "the four poses remain inside scene bounds and in positive-clearance free "
                "space; 'off graph' means outside represented graph coverage, not outside "
                "the simulated world"
            ),
        },
        "relabel_sufficiency": {
            "applies_to_categories": {
                "OFF_NAVIGABLE_GRAPH_OUTCOME": {
                    "progress": "possible only after a future versioned off-graph rule",
                    "safety": "not determinable: 20-tick contact/clearance/stuck/fall trace absent",
                    "completion": "not determinable: at-or-before-horizon tick trace absent",
                    "composite_utility": "not determinable",
                },
                "LOCATABLE_GOAL_UNREACHABLE_OUTCOME": {
                    "progress": "possible only after a future versioned reachability rule",
                    "safety": "not determinable: 20-tick contact/clearance/stuck/fall trace absent",
                    "completion": "not determinable: at-or-before-horizon tick trace absent",
                    "composite_utility": "not determinable",
                },
            },
            "last_finite_geodesic_before_failure_exact_tick_preserved": False,
            "exact_first_off_graph_or_unreachable_tick_preserved": False,
            "complete_path_level_safety_evidence_preserved": False,
            "completion_state_through_horizon_preserved": False,
            "deterministic_full_relabelling_from_preserved_evidence": False,
            "new_oracle_rule_frozen_by_this_diagnostic": False,
        },
        "recommendations_not_implemented": [
            (
                "Version any future oracle semantics explicitly: either treat graph-coverage/"
                "reachability loss as a terminal outcome or revise graph coverage/transit masks."
            ),
            (
                "For any future corpus, persist the 20-tick pose, locate-distance, cell, "
                "geodesic-finiteness, contact, clearance, stuck, termination, and completion trace."
            ),
            (
                "Do not relabel these 18 rows under V2; generate a separately versioned corpus "
                "if complete utilities are required."
            ),
            (
                "Treat the 24 historical calibration identities as development-only pending "
                "the next decision; do not present them later as untouched qualification data."
            ),
        ],
        "prohibited_actions_confirmed_absent": {
            "branches_rerun": False,
            "latents_encoded": False,
            "scorer_trained_or_qualified": False,
            "predictor_checkpoint_opened": False,
            "state_or_candidate_replaced": False,
            "invalid_record_deleted": False,
            "utility_assigned_to_invalid_record": False,
            "fresh_calibration_generated": False,
        },
    }
    report["audit_digest"] = audit_digest(report)
    return report


def validate_report(report: Mapping[str, Any]) -> None:
    body = copy.deepcopy(dict(report))
    expected = body.pop("audit_digest", None)
    if expected != audit_digest(body):
        raise RuntimeError("diagnostic audit digest mismatch")
    if len(report["failure_inventory"]) != EXPECTED_INVALID:
        raise RuntimeError("diagnostic inventory cardinality mismatch")
    if report["counts"]["by_primary_category"] != {
        "INSUFFICIENT_TRACE_FOR_LABEL": 0,
        "LOCATABLE_GOAL_UNREACHABLE_OUTCOME": 14,
        "LOCATOR_IMPLEMENTATION_DEFECT": 0,
        "OFF_NAVIGABLE_GRAPH_OUTCOME": 4,
        "OTHER": 0,
    }:
        raise RuntimeError("diagnostic category counts changed")
    if any(
        record["primary_category"] not in CATEGORIES
        for record in report["failure_inventory"]
    ):
        raise RuntimeError("unknown diagnostic category")


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.root.resolve())
    validate_report(report)
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        output = args.output
        if not output.is_absolute():
            output = args.root.resolve() / output
        _write_report(output, report)
        print(json.dumps({
            "audit_digest": report["audit_digest"],
            "output": str(output),
            "records": len(report["failure_inventory"]),
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
