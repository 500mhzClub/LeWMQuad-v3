#!/usr/bin/env python3
"""Independent persisted-evidence reducer for topological-belief V1.

This module is deliberately usable by the system Python.  It imports no model,
encoder, simulator, or runner code.  Production reduction imports only the
pure metrics module, reads the registered ordinary files through one retained
directory descriptor with ``O_NOFOLLOW``, rebuilds every metric and gate from
the raw ledgers, and emits its receipt outside the scientific output root.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
from typing import Any


EXPERIMENT_ID = "OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1"
ROOT = Path(__file__).resolve().parents[1]
REDUCER_SOURCE_PATH = "scripts/evaluate_occluded_goal_topological_belief_v1.py"
METRICS_SOURCE_PATH = "lewm/safety/occluded_goal_topological_belief_metrics_v1.py"
METRICS_MODULE = "lewm.safety.occluded_goal_topological_belief_metrics_v1"

GRAPH_FILE = "graph_manifest.json"
QUERY_FILE = "query_ledger.jsonl"
CALIBRATION_FILE = "calibration.json"
STAGE_A_BELIEFS_FILE = "stage_a_beliefs.jsonl"
STAGE_A_METRICS_FILE = "stage_a_metrics.json"
STAGE_B_TRACE_FILE = "stage_b_trace.jsonl"
STAGE_B_METRICS_FILE = "stage_b_metrics.json"

AUTHORITY_SCHEMA = "occluded_goal_topological_belief_v1.reducer_authority.v1"
RECEIPT_SCHEMA = "occluded_goal_topological_belief_v1.regeneration_receipt.v1"
EXPECTED_HELDOUT_QUERY_COUNT = 128
EXPECTED_QUERY_COUNT = 768
EXPECTED_QUERY_ROLE_COUNTS = {
    "FIT": 512,
    "CALIBRATION": 128,
    "DEVELOPMENT_HELDOUT": 128,
}
EXPECTED_REGISTERED_CONDITION_COUNT = 6
EXPECTED_ABLATION_COUNT = 3
EXPECTED_REGISTERED_CONDITION_IDS = (
    "CURRENT_FRAME_NEAREST_NODE",
    "FIXED_WINDOW_SEQUENCE",
    "MAP_FILTER",
    "TOP_K_BELIEF",
    "FULL_BELIEF",
    "ORACLE_PLACE_IDENTITY",
)
EXPECTED_ABLATION_IDS = (
    "NO_ACTION_CONSISTENCY",
    "SHUFFLED_ACTION_HISTORY",
    "NO_OBSERVATION_LIKELIHOOD",
)
EXPECTED_HELDOUT_ROLE = "DEVELOPMENT_HELDOUT"
PROBABILITY_SUM_ABS_TOLERANCE = 1.0e-8

AUTHORITY_FIELDS = {
    "schema",
    "experiment_id",
    "graph_manifest",
    "query_rows",
    "belief_rows",
    "calibration",
    "stage_b_trace",
    "stage_a_authorization_path",
    "content_digest",
}


class RegenerationError(ValueError):
    """Raised when persisted evidence is not exact and independently reducible."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise RegenerationError(
            "value contains a non-finite or unsupported canonical JSON value"
        ) from exc


def canonical_document_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _reject_json_constant(value: str) -> None:
    raise RegenerationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RegenerationError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if value is None or isinstance(value, (str, int, bool)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RegenerationError(f"{label} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite(item, label=f"{label}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nonfinite(item, label=f"{label}.{key}")
        return
    raise RegenerationError(f"{label} contains an unsupported JSON value")


def parse_canonical_json(raw: bytes, *, label: str) -> Any:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RegenerationError(f"{label} is not canonical UTF-8 JSON") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise RegenerationError(f"{label} is not valid JSON") from exc
    if canonical_document_bytes(value) != raw:
        raise RegenerationError(f"{label} is not canonical JSON with one LF")
    _reject_nonfinite(value, label=label)
    return value


def _parse_jsonl(raw: bytes, *, label: str) -> list[dict[str, Any]]:
    if not raw or not raw.endswith(b"\n"):
        raise RegenerationError(f"{label} must be nonempty and LF terminated")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw[:-1].split(b"\n"), start=1):
        if not line:
            raise RegenerationError(f"{label} contains an empty line")
        value = parse_canonical_json(line + b"\n", label=f"{label}:{line_number}")
        if not isinstance(value, dict):
            raise RegenerationError(f"{label}:{line_number} must be an object")
        rows.append(value)
    return rows


def _safe_leaf(leaf: str) -> None:
    if not leaf or leaf in {".", ".."} or "/" in leaf or "\\" in leaf:
        raise RegenerationError(f"unsafe reducer input leaf: {leaf!r}")


def _read_regular_at(root_fd: int, leaf: str, *, optional: bool = False) -> bytes | None:
    _safe_leaf(leaf)
    try:
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=root_fd)
    except FileNotFoundError:
        if optional:
            return None
        raise RegenerationError(f"required reducer input is absent: {leaf}")
    except OSError as exc:
        raise RegenerationError(f"reducer input cannot be opened safely: {leaf}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RegenerationError(
                f"reducer input is not a single-link regular file: {leaf}"
            )
        blocks: list[bytes] = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            blocks.append(block)
        raw = b"".join(blocks)
        after = os.fstat(fd)
        projection = lambda row: (
            row.st_dev,
            row.st_ino,
            row.st_mode,
            row.st_uid,
            row.st_gid,
            row.st_nlink,
            row.st_size,
            row.st_mtime_ns,
            row.st_ctime_ns,
        )
        if projection(before) != projection(after) or len(raw) != before.st_size:
            raise RegenerationError(f"reducer input changed while read: {leaf}")
        return raw
    finally:
        os.close(fd)


def _binding(path: str, raw: bytes, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegenerationError(f"{label} must be a nonempty string")
    return value


def _strings(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
    unique: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or (not value and not allow_empty):
        raise RegenerationError(f"{label} must be a {'possibly empty' if allow_empty else 'nonempty'} sequence")
    output = tuple(value)
    if any(not isinstance(item, str) or not item for item in output):
        raise RegenerationError(f"{label} must contain nonempty strings")
    if unique and len(output) != len(set(output)):
        raise RegenerationError(f"{label} contains duplicate identities")
    return output


def validate_reducer_authority(value: Any, module: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != AUTHORITY_FIELDS:
        raise RegenerationError("reducer authority field set drift")
    if value["schema"] != AUTHORITY_SCHEMA:
        raise RegenerationError("reducer authority schema drift")
    if value["experiment_id"] != EXPERIMENT_ID:
        raise RegenerationError("reducer authority experiment drift")
    core = dict(value)
    declared_digest = core.pop("content_digest")
    if declared_digest != canonical_digest(core):
        raise RegenerationError("reducer authority content digest drift")
    expected_sections = {
        "graph_manifest": _call(module, "graph_manifest_authority"),
        "query_rows": _call(module, "query_row_authority"),
        "belief_rows": _call(module, "belief_row_authority"),
        "calibration": _call(module, "calibration_authority"),
        "stage_b_trace": _call(module, "stage_b_trace_authority"),
    }
    for key, expected in expected_sections.items():
        if value[key] != expected:
            raise RegenerationError(f"nested reducer authority drift: {key}")
    if value["stage_a_authorization_path"] != "decision.stage_b_authorized":
        raise RegenerationError("Stage-A authorization path drift")
    condition_ids = tuple(_call(module, "stage_a_condition_ids"))
    if condition_ids != EXPECTED_REGISTERED_CONDITION_IDS + EXPECTED_ABLATION_IDS:
        raise RegenerationError("ordered nine-condition Stage-A authority drift")
    _reject_nonfinite(value, label="reducer_authority")
    return dict(value)


def _graph_nodes(graph_manifest: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    graphs = graph_manifest.get("graphs")
    if not isinstance(graphs, list) or not graphs:
        raise RegenerationError("graph_manifest graph collection is absent or empty")
    result: dict[str, tuple[str, ...]] = {}
    for index, graph in enumerate(graphs):
        if not isinstance(graph, Mapping):
            raise RegenerationError(f"graph_manifest graph {index} must be an object")
        graph_id = _string(graph.get("graph_id"), label=f"graph[{index}].id")
        if graph_id in result:
            raise RegenerationError(f"duplicate graph identity: {graph_id}")
        nodes = graph.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise RegenerationError(f"graph {graph_id} has no nodes")
        node_ids: list[str] = []
        for node_index, node in enumerate(nodes):
            if not isinstance(node, Mapping):
                raise RegenerationError(f"graph {graph_id} node {node_index} is not an object")
            node_ids.append(
                _string(
                    node.get("node_id"),
                    label=f"graph {graph_id} node {node_index} identity",
                )
            )
        if len(node_ids) != len(set(node_ids)):
            raise RegenerationError(f"graph {graph_id} contains duplicate node identities")
        result[graph_id] = tuple(node_ids)
    return result


def _validate_queries(
    rows: Sequence[Mapping[str, Any]],
    graphs: Mapping[str, tuple[str, ...]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, tuple[str, ...]],
    dict[str, int],
]:
    all_queries: dict[str, dict[str, Any]] = {}
    heldout_queries: dict[str, dict[str, Any]] = {}
    candidates: dict[str, tuple[str, ...]] = {}
    role_counts: dict[str, int] = {role: 0 for role in EXPECTED_QUERY_ROLE_COUNTS}
    for index, row in enumerate(rows):
        query_id = _string(row.get("query_id"), label=f"query[{index}].id")
        if query_id in all_queries:
            raise RegenerationError(f"duplicate query identity: {query_id}")
        role = row.get("role")
        if role not in role_counts:
            raise RegenerationError(f"query role drift: {query_id}")
        role_counts[str(role)] += 1
        graph_id = _string(row.get("graph_id"), label=f"query {query_id}.graph")
        if graph_id not in graphs:
            raise RegenerationError(f"query {query_id} references an unknown graph")
        candidate_ids = _strings(
            row.get("candidate_node_ids"),
            label=f"query {query_id}.candidate_ids",
        )
        unknown = set(candidate_ids) - set(graphs[graph_id])
        if unknown:
            raise RegenerationError(
                f"query {query_id} contains candidates outside its graph: {sorted(unknown)}"
            )
        if candidate_ids != graphs[graph_id]:
            raise RegenerationError(
                f"query {query_id} candidate identities do not cover the exact ordered graph"
            )
        all_queries[query_id] = dict(row)
        if role == EXPECTED_HELDOUT_ROLE:
            heldout_queries[query_id] = dict(row)
            candidates[query_id] = candidate_ids
    if len(all_queries) != EXPECTED_QUERY_COUNT:
        raise RegenerationError(
            f"query_ledger must contain exactly {EXPECTED_QUERY_COUNT} queries"
        )
    if role_counts != EXPECTED_QUERY_ROLE_COUNTS:
        raise RegenerationError(f"query role counts drift: {role_counts}")
    if len(heldout_queries) != EXPECTED_HELDOUT_QUERY_COUNT:
        raise RegenerationError(
            f"query_ledger must contain exactly {EXPECTED_HELDOUT_QUERY_COUNT} heldout queries"
        )
    return heldout_queries, candidates, role_counts


def _belief_vectors_and_selection(
    row: Mapping[str, Any],
    *,
    graph_nodes: tuple[str, ...],
    label: str,
) -> None:
    observed_nodes = _strings(row.get("node_ids"), label=f"{label}.node_ids")
    if observed_nodes != graph_nodes:
        raise RegenerationError(f"{label} does not cover the exact ordered graph node vector")
    vector_fields = (
        ("observation_similarities", False),
        ("observation_likelihoods", True),
        ("transition_prior_probabilities", True),
        ("preprojection_probabilities", True),
        ("posterior_probabilities", True),
    )
    for field, normalized in vector_fields:
        values = row.get(field)
        if not isinstance(values, list) or len(values) != len(graph_nodes):
            raise RegenerationError(f"{label} {field} vector length drift")
        numeric: list[float] = []
        for index, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RegenerationError(f"{label}.{field}[{index}] is not numeric")
            number = float(value)
            if not math.isfinite(number):
                raise RegenerationError(f"{label}.{field}[{index}] is not finite")
            if normalized and not 0.0 <= number <= 1.0:
                raise RegenerationError(f"{label}.{field}[{index}] is not a probability")
            numeric.append(number)
        if normalized and not math.isclose(
            math.fsum(numeric),
            1.0,
            rel_tol=0.0,
            abs_tol=PROBABILITY_SUM_ABS_TOLERANCE,
        ):
            raise RegenerationError(f"{label} {field} values do not sum to one")
    selected_node = _string(row.get("selected_node_id"), label=f"{label}.selected_node_id")
    if selected_node not in graph_nodes:
        raise RegenerationError(f"{label} selected node is outside its graph")
    _string(row.get("selected_edge_id"), label=f"{label}.selected_edge_id")
    _string(row.get("selected_port_label"), label=f"{label}.selected_port_label")
    entropy = row.get("normalized_entropy")
    if (
        isinstance(entropy, bool)
        or not isinstance(entropy, (int, float))
        or not math.isfinite(float(entropy))
        or not 0.0 <= float(entropy) <= 1.0
    ):
        raise RegenerationError(f"{label}.normalized_entropy must be finite in [0,1]")
    if type(row.get("abstained")) is not bool:
        raise RegenerationError(f"{label}.abstained must be Boolean")


def _validate_stage_a_beliefs(
    rows: Sequence[Mapping[str, Any]],
    *,
    queries: Mapping[str, Mapping[str, Any]],
    candidates: Mapping[str, tuple[str, ...]],
) -> dict[str, Any]:
    condition_ids = EXPECTED_REGISTERED_CONDITION_IDS + EXPECTED_ABLATION_IDS
    expected = {(query_id, condition_id) for query_id in queries for condition_id in condition_ids}
    observed: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        query_id = _string(row.get("query_id"), label=f"stage_a[{index}].query")
        condition_id = _string(
            row.get("condition_id"), label=f"stage_a[{index}].condition"
        )
        identity = (query_id, condition_id)
        if identity not in expected:
            raise RegenerationError(f"unexpected Stage-A belief identity: {identity}")
        if identity in observed:
            raise RegenerationError(f"duplicate Stage-A belief identity: {identity}")
        observed.add(identity)
        _belief_vectors_and_selection(
            row,
            graph_nodes=candidates[query_id],
            label=f"Stage-A {query_id}/{condition_id}",
        )
    missing = expected - observed
    if missing:
        raise RegenerationError(f"Stage-A belief coverage incomplete: {len(missing)} missing")
    return {
        "rows": len(rows),
        "expected_rows": EXPECTED_HELDOUT_QUERY_COUNT * len(condition_ids),
        "queries": len(queries),
        "registered_conditions": len(EXPECTED_REGISTERED_CONDITION_IDS),
        "ablations": len(EXPECTED_ABLATION_IDS),
        "row_identity_digest": canonical_digest(sorted([list(item) for item in observed])),
        "full_graph_probability_vectors_validated": True,
        "candidate_identity_coverage_validated": True,
    }


def _validate_calibration_evidence(
    calibration: Mapping[str, Any],
    *,
    query_rows: Sequence[Mapping[str, Any]],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    calibration_query_ids = {
        _string(row.get("query_id"), label="calibration query identity")
        for row in query_rows
        if row.get("role") == "CALIBRATION"
    }
    if len(calibration_query_ids) != EXPECTED_QUERY_ROLE_COUNTS["CALIBRATION"]:
        raise RegenerationError("calibration query identity count drift")
    grid_count = authority.get("grid_results")
    expected_row_count = authority.get("grid_query_results")
    if (
        isinstance(grid_count, bool)
        or not isinstance(grid_count, int)
        or grid_count <= 0
        or expected_row_count != grid_count * len(calibration_query_ids)
    ):
        raise RegenerationError("calibration raw-cube authority drift")
    raw_rows = calibration.get("grid_query_results")
    if not isinstance(raw_rows, list) or len(raw_rows) != expected_row_count:
        raise RegenerationError(
            "calibration must contain the exact grid-by-query raw outcome cube"
        )
    expected = {
        (grid_index, query_id)
        for grid_index in range(grid_count)
        for query_id in calibration_query_ids
    }
    observed: set[tuple[int, str]] = set()
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise RegenerationError(f"calibration raw row {index} is not an object")
        grid_index = row.get("grid_index")
        query_id = row.get("query_id")
        if (
            isinstance(grid_index, bool)
            or not isinstance(grid_index, int)
            or not isinstance(query_id, str)
        ):
            raise RegenerationError("calibration raw row identity type drift")
        identity = (grid_index, query_id)
        if identity not in expected or identity in observed:
            raise RegenerationError(f"calibration raw identity coverage drift: {identity}")
        observed.add(identity)
        for field in (
            "edge_correct",
            "localisation_top3",
            "false_confident",
            "abstained",
        ):
            if type(row.get(field)) is not bool:
                raise RegenerationError(f"calibration raw {field} must be Boolean")
        regret = row.get("normalized_regret")
        if (
            isinstance(regret, bool)
            or not isinstance(regret, (int, float))
            or not math.isfinite(float(regret))
            or not 0.0 <= float(regret) <= 1.0
        ):
            raise RegenerationError(
                "calibration raw normalized_regret must be finite in [0,1]"
            )
    if observed != expected:
        raise RegenerationError("calibration raw outcome cube coverage is incomplete")
    return {
        "grid_points": grid_count,
        "queries": len(calibration_query_ids),
        "rows": len(raw_rows),
        "row_identity_digest": canonical_digest(
            sorted([list(identity) for identity in observed])
        ),
        "raw_outcome_cube_exactly_covered": True,
    }


def _validate_stage_b_trace(
    rows: Sequence[Mapping[str, Any]],
    *,
    module: Any,
    graph_manifest: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
    stage_a_metrics: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    validated = _call(
        module,
        "validate_stage_b_trace_rows",
        graph_manifest,
        tuple(query_rows),
        tuple(rows),
        stage_a_metrics,
    )
    if not isinstance(validated, list) or len(validated) != len(rows):
        raise RegenerationError("Stage-B strict validator returned row-count drift")
    if authority != _call(module, "stage_b_trace_authority"):
        raise RegenerationError("Stage-B trace authority changed during reduction")
    fields = authority.get("fields")
    identity_fields = authority.get("identity")
    condition_ids = authority.get("condition_ids")
    episodes_per_condition = authority.get("episodes_per_condition")
    execution_policy = authority.get("execution_policy")
    terminal_reasons = authority.get("terminal_reasons")
    if (
        not isinstance(fields, list)
        or not fields
        or len(fields) != len(set(fields))
        or any(not isinstance(field, str) or not field for field in fields)
    ):
        raise RegenerationError("Stage-B field authority drift")
    if not isinstance(identity_fields, list) or not identity_fields:
        raise RegenerationError("Stage-B identity authority drift")
    conditions = _strings(condition_ids, label="Stage-B condition authority")
    terminals = _strings(terminal_reasons, label="Stage-B terminal authority")
    if (
        isinstance(episodes_per_condition, bool)
        or not isinstance(episodes_per_condition, int)
        or episodes_per_condition <= 0
        or not isinstance(execution_policy, Mapping)
    ):
        raise RegenerationError("Stage-B execution authority drift")
    decision_budget = execution_policy.get("decision_budget")
    if (
        isinstance(decision_budget, bool)
        or not isinstance(decision_budget, int)
        or decision_budget <= 0
    ):
        raise RegenerationError("Stage-B decision-budget authority drift")

    # These are evidence categories, not a copied trace schema.  The exact
    # schema remains dynamic and is byte-bound above; this independent pass
    # additionally refuses an authority that omits the evidence needed to
    # reconstruct a macro transition, carried belief, state, or ranker call.
    required_evidence = {
        "macro": {
            "constituent_edge_ids", "constituent_node_ids",
            "constituent_action_labels", "constituent_edge_costs_m",
            "constituent_observation_ids",
            "constituent_observation_pixel_sha256s",
            "executed_choice_edge_id", "executed_port_label",
            "executed_distance_m",
        },
        "state": {
            "actual_node_id_before", "actual_node_id_after", "query_id",
            "query_index", "prior_visited_node_ids", "observation_id",
            "observation_pixel_sha256", "reobservation_id",
            "reobservation_pixel_sha256", "terminal", "terminal_reason",
            "goal_reached", "decision_budget", "decision_index",
            "execution_id", "step_id",
            "prelude_distance_m",
        },
        "belief": {
            "belief_node_ids", "belief_probabilities", "selected_node_id",
            "belief_proposed_port_label", "belief_selected_edge_id",
            "selected_port_label", "normalized_entropy", "entropy_threshold",
            "filter_update_observation_ids", "filter_update_pixel_sha256s",
            "filter_update_node_ids", "filter_update_incoming_action_labels",
            "filter_update_observation_similarities",
            "filter_update_observation_likelihoods",
            "filter_update_transition_prior_probabilities",
            "filter_update_preprojection_probabilities",
            "filter_update_posterior_probabilities",
        },
        "ranker": {
            "ranker_called", "local_candidate_scores",
            "local_oracle_admissible_candidate_indices",
            "local_candidate_index", "local_candidate_name",
            "local_candidate_score", "ranker_latency_ms",
            "planning_latency_ms", "stored_relative_waypoint",
            "action_disposition", "local_execution_success",
            "oracle_admissible_execution",
            "ranker_previous_applied_command", "ranker_control_history",
            "selected_candidate_applied_commands",
            "post_decision_previous_applied_command",
            "post_decision_control_history",
        },
    }
    field_set = set(fields)
    for category, required in required_evidence.items():
        missing = required - field_set
        if missing:
            raise RegenerationError(
                f"Stage-B authority omits {category} evidence: {sorted(missing)}"
            )

    graph_rows = graph_manifest.get("graphs")
    if not isinstance(graph_rows, list) or not graph_rows:
        raise RegenerationError("Stage-B graph evidence is absent")
    graphs_by_episode: dict[str, Mapping[str, Any]] = {}
    for index, graph in enumerate(graph_rows):
        if not isinstance(graph, Mapping):
            raise RegenerationError(f"Stage-B graph {index} is not an object")
        episode_id = _string(graph.get("episode_id"), label=f"graph[{index}].episode")
        if episode_id in graphs_by_episode:
            raise RegenerationError(f"duplicate Stage-B graph episode: {episode_id}")
        graphs_by_episode[episode_id] = graph
    heldout_queries: dict[str, Mapping[str, Any]] = {}
    heldout_episodes: set[str] = set()
    for index, query in enumerate(query_rows):
        if query.get("role") != EXPECTED_HELDOUT_ROLE:
            continue
        query_id = _string(query.get("query_id"), label=f"Stage-B query[{index}].id")
        episode_id = _string(
            query.get("episode_id"), label=f"Stage-B query[{index}].episode"
        )
        if query_id in heldout_queries or episode_id not in graphs_by_episode:
            raise RegenerationError("Stage-B heldout query identity/binding drift")
        heldout_queries[query_id] = query
        heldout_episodes.add(episode_id)
    if len(heldout_episodes) != episodes_per_condition:
        raise RegenerationError("Stage-B heldout episode count differs from authority")

    def finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RegenerationError(f"{label} must be numeric")
        number = float(value)
        if not math.isfinite(number) or (nonnegative and number < 0.0):
            raise RegenerationError(f"{label} has an invalid finite range")
        return number

    def probability_vector(value: Any, size: int, label: str) -> list[float]:
        if not isinstance(value, list) or len(value) != size:
            raise RegenerationError(f"{label} probability shape drift")
        result = [finite(item, f"{label}[]") for item in value]
        if any(item < 0.0 or item > 1.0 for item in result) or not math.isclose(
            math.fsum(result), 1.0, rel_tol=0.0,
            abs_tol=PROBABILITY_SUM_ABS_TOLERANCE,
        ):
            raise RegenerationError(f"{label} is not a normalized probability vector")
        return result

    def sha256_text(value: Any, label: str) -> str:
        text = _string(value, label=label)
        if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
            raise RegenerationError(f"{label} is not lowercase SHA-256")
        return text

    identities: list[list[Any]] = []
    identity_encodings: set[bytes] = set()
    step_ids: set[str] = set()
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for index, row in enumerate(validated):
        if not isinstance(row, Mapping):
            raise RegenerationError(f"Stage-B validated row {index} is not an object")
        if set(row) != field_set:
            raise RegenerationError(f"Stage-B row {index} differs from dynamic field authority")
        identity = [row.get(field) for field in identity_fields]
        encoded_identity = canonical_json_bytes(identity)
        if encoded_identity in identity_encodings:
            raise RegenerationError(f"duplicate Stage-B dynamic identity: {identity}")
        identity_encodings.add(encoded_identity)
        identities.append(identity)
        condition = _string(row.get("condition_id"), label=f"Stage-B[{index}].condition")
        episode = _string(row.get("episode_id"), label=f"Stage-B[{index}].episode")
        if condition not in conditions or episode not in heldout_episodes:
            raise RegenerationError("Stage-B condition/episode coverage drift")
        query_id = _string(row.get("query_id"), label=f"Stage-B[{index}].query")
        query = heldout_queries.get(query_id)
        if query is None or query.get("episode_id") != episode:
            raise RegenerationError("Stage-B row does not bind a heldout query")
        graph = graphs_by_episode[episode]
        nodes = graph.get("nodes")
        edges = graph.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise RegenerationError("Stage-B graph lacks node/edge evidence")
        node_by_id = {
            _string(node.get("node_id"), label="Stage-B graph node"): node
            for node in nodes
            if isinstance(node, Mapping)
        }
        graph_node_order = tuple(
            _string(node.get("node_id"), label="Stage-B graph node")
            for node in nodes
            if isinstance(node, Mapping)
        )
        edge_by_id = {
            _string(edge.get("edge_id"), label="Stage-B graph edge"): edge
            for edge in edges
            if isinstance(edge, Mapping)
        }
        if len(node_by_id) != len(nodes) or len(edge_by_id) != len(edges):
            raise RegenerationError("Stage-B graph node/edge identities are not unique")

        belief_nodes = _strings(
            row.get("belief_node_ids"), label=f"Stage-B[{index}].belief nodes"
        )
        if belief_nodes != graph_node_order:
            raise RegenerationError("Stage-B belief does not cover its exact ordered graph")
        belief = probability_vector(
            row.get("belief_probabilities"), len(belief_nodes),
            f"Stage-B[{index}].belief",
        )
        selected_node = _string(
            row.get("selected_node_id"), label=f"Stage-B[{index}].selected node"
        )
        if selected_node not in node_by_id:
            raise RegenerationError("Stage-B selected node lies outside its graph")

        update_ids = _strings(
            row.get("filter_update_observation_ids"),
            label=f"Stage-B[{index}].filter observation IDs",
        )
        update_nodes = _strings(
            row.get("filter_update_node_ids"),
            label=f"Stage-B[{index}].filter nodes", allow_empty=False, unique=False,
        )
        update_shas_raw = row.get("filter_update_pixel_sha256s")
        update_actions = row.get("filter_update_incoming_action_labels")
        if (
            not isinstance(update_shas_raw, list)
            or not isinstance(update_actions, list)
            or len(update_ids) != len(update_nodes)
            or len(update_ids) != len(update_shas_raw)
            or len(update_ids) != len(update_actions)
        ):
            raise RegenerationError("Stage-B filter-update occurrence alignment drift")
        update_shas = [
            sha256_text(value, f"Stage-B[{index}].filter pixel SHA")
            for value in update_shas_raw
        ]
        if any(node not in node_by_id for node in update_nodes):
            raise RegenerationError("Stage-B filter update references an unknown node")
        if any(
            node_by_id[node].get("pixel_sha256") != pixel_sha
            for node, pixel_sha in zip(update_nodes, update_shas)
        ):
            raise RegenerationError("Stage-B filter pixel evidence differs from its node")
        for action in update_actions:
            if action is not None:
                _string(action, label=f"Stage-B[{index}].filter action")
        matrix_names = (
            "filter_update_observation_similarities",
            "filter_update_observation_likelihoods",
            "filter_update_transition_prior_probabilities",
            "filter_update_preprojection_probabilities",
            "filter_update_posterior_probabilities",
        )
        matrices: dict[str, list[list[float]]] = {}
        for name in matrix_names:
            matrix = row.get(name)
            if not isinstance(matrix, list) or len(matrix) != len(update_ids):
                raise RegenerationError(f"Stage-B {name} update-count drift")
            vectors: list[list[float]] = []
            for update_index, vector in enumerate(matrix):
                if name == "filter_update_observation_similarities":
                    if not isinstance(vector, list) or len(vector) != len(belief_nodes):
                        raise RegenerationError("Stage-B similarity-vector shape drift")
                    values = [
                        finite(item, f"Stage-B {name}[{update_index}]")
                        for item in vector
                    ]
                    if any(value < -1.0 or value > 1.0 for value in values):
                        raise RegenerationError("Stage-B similarity is outside [-1,1]")
                else:
                    values = probability_vector(
                        vector, len(belief_nodes),
                        f"Stage-B {name}[{update_index}]",
                    )
                vectors.append(values)
            matrices[name] = vectors
        if any(
            not math.isclose(a, b, rel_tol=0.0, abs_tol=PROBABILITY_SUM_ABS_TOLERANCE)
            for a, b in zip(
                belief, matrices["filter_update_posterior_probabilities"][-1]
            )
        ):
            raise RegenerationError("Stage-B belief differs from the final carried posterior")
        if (
            row.get("observation_id") != update_ids[-1]
            or row.get("observation_pixel_sha256") != update_shas[-1]
        ):
            raise RegenerationError("Stage-B decision observation is not the final filter update")
        witness_edge_id = row.get("belief_selected_edge_id")
        if witness_edge_id is None:
            if row.get("stored_relative_waypoint") is not None:
                raise RegenerationError("Stage-B route-free belief retained a waypoint")
        else:
            witness_edge = edge_by_id.get(
                _string(witness_edge_id, label="Stage-B belief-selected edge")
            )
            if witness_edge is None or row.get("stored_relative_waypoint") != witness_edge.get(
                "relative_waypoint"
            ):
                raise RegenerationError("Stage-B stored waypoint differs from belief edge")

        edge_ids_raw = row.get("constituent_edge_ids")
        node_ids_raw = row.get("constituent_node_ids")
        actions_raw = row.get("constituent_action_labels")
        costs_raw = row.get("constituent_edge_costs_m")
        occurrence_ids_raw = row.get("constituent_observation_ids")
        occurrence_shas_raw = row.get("constituent_observation_pixel_sha256s")
        if not all(
            isinstance(value, list)
            for value in (
                edge_ids_raw, node_ids_raw, actions_raw, costs_raw,
                occurrence_ids_raw, occurrence_shas_raw,
            )
        ):
            raise RegenerationError("Stage-B macro evidence must use JSON arrays")
        assert isinstance(edge_ids_raw, list) and isinstance(node_ids_raw, list)
        assert isinstance(actions_raw, list) and isinstance(costs_raw, list)
        assert isinstance(occurrence_ids_raw, list) and isinstance(occurrence_shas_raw, list)
        macro_length = len(edge_ids_raw)
        if not (
            len(actions_raw) == len(costs_raw) == len(occurrence_ids_raw)
            == len(occurrence_shas_raw) == macro_length
            and len(node_ids_raw) == macro_length + 1
        ):
            raise RegenerationError("Stage-B constituent macro vector alignment drift")
        macro_nodes = [
            _string(value, label=f"Stage-B[{index}].constituent node")
            for value in node_ids_raw
        ]
        if (
            row.get("actual_node_id_before") != macro_nodes[0]
            or row.get("actual_node_id_after") != macro_nodes[-1]
            or any(node not in node_by_id for node in macro_nodes)
        ):
            raise RegenerationError("Stage-B macro does not bind actual before/after state")
        if node_by_id[macro_nodes[-1]].get("pixel_sha256") != row.get(
            "reobservation_pixel_sha256"
        ):
            raise RegenerationError("Stage-B terminal reobservation differs from actual state")
        macro_costs: list[float] = []
        for macro_index, edge_id_value in enumerate(edge_ids_raw):
            edge_id = _string(edge_id_value, label="Stage-B constituent edge")
            action = _string(actions_raw[macro_index], label="Stage-B constituent action")
            cost = finite(
                costs_raw[macro_index], "Stage-B constituent edge cost", nonnegative=True
            )
            _string(occurrence_ids_raw[macro_index], label="Stage-B constituent observation")
            pixel_sha = sha256_text(
                occurrence_shas_raw[macro_index], "Stage-B constituent pixel SHA"
            )
            edge = edge_by_id.get(edge_id)
            if edge is None or (
                edge.get("source_node_id") != macro_nodes[macro_index]
                or edge.get("target_node_id") != macro_nodes[macro_index + 1]
                or edge.get("executed_action_label") != action
            ):
                raise RegenerationError("Stage-B macro is not an exact graph-edge chain")
            graph_cost = finite(edge.get("edge_cost"), "Stage-B graph edge cost", nonnegative=True)
            if not math.isclose(cost, graph_cost, rel_tol=0.0, abs_tol=1e-12):
                raise RegenerationError("Stage-B constituent cost differs from graph")
            target = node_by_id[macro_nodes[macro_index + 1]]
            if target.get("pixel_sha256") != pixel_sha:
                raise RegenerationError("Stage-B constituent observation pixel differs from node")
            macro_costs.append(cost)
        executed_distance = finite(
            row.get("executed_distance_m"), "Stage-B executed distance", nonnegative=True
        )
        if not math.isclose(
            executed_distance, math.fsum(macro_costs), rel_tol=0.0, abs_tol=1e-10
        ):
            raise RegenerationError("Stage-B executed distance differs from macro costs")

        outcomes = query.get("stage_b_local_candidate_outcomes")
        if not isinstance(outcomes, list) or not outcomes:
            raise RegenerationError("Stage-B query lacks preregistered local candidate outcomes")
        outcome_by_index: dict[int, Mapping[str, Any]] = {}
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                raise RegenerationError("Stage-B local candidate outcome is not an object")
            candidate_index = outcome.get("candidate_index")
            if (
                isinstance(candidate_index, bool)
                or not isinstance(candidate_index, int)
                or candidate_index in outcome_by_index
            ):
                raise RegenerationError("Stage-B local candidate index drift")
            outcome_by_index[candidate_index] = outcome
        if set(outcome_by_index) != set(range(len(outcomes))):
            raise RegenerationError("Stage-B local candidate indices are not contiguous")
        mask_raw = row.get("local_oracle_admissible_candidate_indices")
        if (
            not isinstance(mask_raw, list)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in mask_raw)
            or len(mask_raw) != len(set(mask_raw))
            or any(value not in outcome_by_index for value in mask_raw)
        ):
            raise RegenerationError("Stage-B ranker mask identity drift")
        expected_mask = [
            candidate_index
            for candidate_index in range(len(outcomes))
            if outcome_by_index[candidate_index].get("oracle_admissible") is True
        ]
        if mask_raw != expected_mask:
            raise RegenerationError("Stage-B ranker mask differs from preregistered outcomes")
        ranker_called = row.get("ranker_called")
        if type(ranker_called) is not bool:
            raise RegenerationError("Stage-B ranker_called must be Boolean")
        scores_raw = row.get("local_candidate_scores")
        if ranker_called:
            if not isinstance(scores_raw, list) or len(scores_raw) != len(outcomes) or not mask_raw:
                raise RegenerationError("Stage-B ranker call lacks its full score vector/mask")
            scores = [finite(value, "Stage-B candidate score") for value in scores_raw]
            chosen = min(mask_raw, key=lambda value: (-scores[value], value))
            outcome = outcome_by_index[chosen]
            if (
                row.get("local_candidate_index") != chosen
                or row.get("local_candidate_name") != outcome.get("candidate_name")
                or not math.isclose(
                    finite(row.get("local_candidate_score"), "Stage-B selected score"),
                    scores[chosen], rel_tol=0.0, abs_tol=1e-12,
                )
                or row.get("executed_choice_edge_id") != outcome.get("actual_edge_id")
                or row.get("executed_port_label") != outcome.get("actual_port_label")
                or macro_length == 0
                or edge_ids_raw[0] != row.get("executed_choice_edge_id")
            ):
                raise RegenerationError("Stage-B ranker selection/macro binding drift")
        else:
            if (
                scores_raw != []
                or any(
                    row.get(name) is not None
                    for name in (
                        "local_candidate_index", "local_candidate_name",
                        "local_candidate_score", "executed_choice_edge_id",
                        "executed_port_label",
                    )
                )
                or macro_length != 0
            ):
                raise RegenerationError("Stage-B non-ranker row contains ranker/macro evidence")
        ranker_latency = finite(
            row.get("ranker_latency_ms"), "Stage-B ranker latency", nonnegative=True
        )
        planning_latency = finite(
            row.get("planning_latency_ms"), "Stage-B planning latency", nonnegative=True
        )
        if planning_latency < ranker_latency or (not ranker_called and ranker_latency != 0.0):
            raise RegenerationError("Stage-B ranker/planning latency evidence drift")

        def numeric_vector(value: Any, size: int, label: str) -> list[float]:
            if not isinstance(value, list) or len(value) != size:
                raise RegenerationError(f"{label} shape drift")
            return [finite(item, f"{label}[]") for item in value]

        previous_command = numeric_vector(
            row.get("ranker_previous_applied_command"), 3,
            "Stage-B previous applied command",
        )
        post_command = numeric_vector(
            row.get("post_decision_previous_applied_command"), 3,
            "Stage-B post-decision applied command",
        )
        histories: dict[str, list[list[float]]] = {}
        for name in ("ranker_control_history", "post_decision_control_history"):
            history = row.get(name)
            if not isinstance(history, list) or len(history) != 15:
                raise RegenerationError(f"Stage-B {name} shape drift")
            histories[name] = [
                numeric_vector(vector, 2, f"Stage-B {name}[]")
                for vector in history
            ]
        applied_commands = row.get("selected_candidate_applied_commands")
        if ranker_called:
            if not isinstance(applied_commands, list) or len(applied_commands) != 5:
                raise RegenerationError(
                    "Stage-B ranker call lacks five selected applied commands"
                )
            for vector in applied_commands:
                numeric_vector(vector, 3, "Stage-B selected applied command")
        elif applied_commands != []:
            raise RegenerationError(
                "Stage-B non-ranker row persisted selected applied commands"
            )
        if not ranker_called and (
            post_command != previous_command
            or histories["post_decision_control_history"]
            != histories["ranker_control_history"]
        ):
            raise RegenerationError("Stage-B NO_OP changed command/control state")
        finite(row.get("prelude_distance_m"), "Stage-B prelude distance", nonnegative=True)
        for field in (
            "local_execution_success", "oracle_admissible_execution", "terminal",
            "goal_reached", "abstained",
        ):
            if type(row.get(field)) is not bool:
                raise RegenerationError(f"Stage-B {field} must be Boolean")
        if row.get("ranker_called") != (row.get("action_disposition") == "EXECUTED_CANDIDATE"):
            raise RegenerationError("Stage-B action disposition differs from ranker evidence")
        if row.get("decision_budget") != decision_budget:
            raise RegenerationError("Stage-B row decision budget differs from authority")
        terminal_reason = row.get("terminal_reason")
        if terminal_reason is not None and terminal_reason not in terminals:
            raise RegenerationError("Stage-B terminal reason differs from authority")
        sha256_text(row.get("observation_pixel_sha256"), "Stage-B observation SHA")
        sha256_text(row.get("reobservation_pixel_sha256"), "Stage-B reobservation SHA")
        _string(row.get("reobservation_id"), label="Stage-B reobservation ID")
        step_id = _string(row.get("step_id"), label="Stage-B step ID")
        if step_id in step_ids:
            raise RegenerationError("duplicate Stage-B step ID")
        step_ids.add(step_id)
        grouped.setdefault((condition, episode), []).append(row)

    expected_execution_keys = {
        (condition, episode)
        for condition in conditions
        for episode in heldout_episodes
    }
    if set(grouped) != expected_execution_keys:
        raise RegenerationError("Stage-B condition/episode cross-product is incomplete")
    terminal_rows_per_execution = authority.get("terminal_rows_per_execution")
    if terminal_rows_per_execution != 1:
        raise RegenerationError("Stage-B terminal-row authority drift")
    for key, execution in grouped.items():
        ordered = sorted(execution, key=lambda row: row.get("decision_index"))
        indices = [row.get("decision_index") for row in ordered]
        if indices != list(range(len(ordered))) or len(ordered) > decision_budget:
            raise RegenerationError(f"Stage-B execution decision coverage drift: {key}")
        if len({_string(row.get("execution_id"), label="Stage-B execution ID") for row in ordered}) != 1:
            raise RegenerationError(f"Stage-B execution identity changed: {key}")
        if sum(row.get("terminal") is True for row in ordered) != 1 or not ordered[-1].get("terminal"):
            raise RegenerationError(f"Stage-B execution terminal coverage drift: {key}")
        if any(row.get("terminal") for row in ordered[:-1]):
            raise RegenerationError(f"Stage-B execution terminated before its final row: {key}")
        for previous, current in zip(ordered, ordered[1:]):
            if current.get("actual_node_id_before") != previous.get("actual_node_id_after"):
                raise RegenerationError(f"Stage-B actual-state continuity drift: {key}")
            if (
                current.get("ranker_previous_applied_command")
                != previous.get("post_decision_previous_applied_command")
                or current.get("ranker_control_history")
                != previous.get("post_decision_control_history")
            ):
                raise RegenerationError(f"Stage-B command/control continuity drift: {key}")
    return {
        "rows": len(rows),
        "conditions": len(conditions),
        "episodes_per_condition": episodes_per_condition,
        "executions": len(grouped),
        "row_identity_digest": canonical_digest(sorted(identities)),
        "strict_trace_validation_passed": True,
        "dynamic_field_authority_validated": True,
        "exact_macro_evidence_validated": True,
        "exact_state_evidence_validated": True,
        "exact_belief_evidence_validated": True,
        "exact_ranker_evidence_validated": True,
    }


def _load_metrics_module() -> Any:
    return importlib.import_module(METRICS_MODULE)


def _metrics_module_name(module: Any) -> str:
    name = getattr(module, "__name__", None)
    return name if isinstance(name, str) and name else module.__class__.__name__


def _git_bytes(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RegenerationError(
            f"source-freeze Git observation failed: {' '.join(arguments)}"
        ) from exc
    return completed.stdout


def _git_text(*arguments: str) -> str:
    try:
        return _git_bytes(*arguments).decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RegenerationError("source-freeze Git output is not ASCII") from exc


def _read_repository_source(relative: str) -> bytes:
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise RegenerationError(f"unsafe repository source path: {relative!r}")
    fds: list[int] = []
    try:
        current_fd = os.open(ROOT, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        fds.append(current_fd)
        for component in parts[:-1]:
            current_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            info = os.fstat(current_fd)
            if not stat.S_ISDIR(info.st_mode) or info.st_nlink < 2:
                raise RegenerationError(f"repository source parent is invalid: {relative}")
            fds.append(current_fd)
        raw = _read_regular_at(current_fd, parts[-1])
        assert raw is not None
        return raw
    finally:
        for fd in reversed(fds):
            os.close(fd)


def _source_binding(relative: str, raw: bytes) -> dict[str, Any]:
    return {"path": relative, "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def _observe_source_freeze(module: Any) -> dict[str, Any]:
    if _metrics_module_name(module) != METRICS_MODULE:
        raise RegenerationError("production reduction requires the exact metrics module")
    module_path = getattr(module, "__file__", None)
    if not isinstance(module_path, str) or Path(module_path).absolute() != ROOT / METRICS_SOURCE_PATH:
        raise RegenerationError("metrics module source path drift")
    head = _git_text("rev-parse", "HEAD")
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RegenerationError("source-freeze HEAD is not canonical 40-hex")
    if _git_bytes("status", "--porcelain=v1", "--untracked-files=all"):
        raise RegenerationError("independent reduction requires a clean source-freeze worktree")
    bindings: dict[str, dict[str, Any]] = {}
    for key, relative in (
        ("independent_reducer", REDUCER_SOURCE_PATH),
        ("metrics_module", METRICS_SOURCE_PATH),
    ):
        raw = _read_repository_source(relative)
        if _git_bytes("show", f"{head}:{relative}") != raw:
            raise RegenerationError(f"source bytes differ from source-freeze HEAD: {relative}")
        bindings[key] = _source_binding(relative, raw)
    return {
        "head_commit": head,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": bindings,
    }


def _validate_source_freeze(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "head_commit", "worktree_clean", "sources_exactly_equal_head", "sources"
    }:
        raise RegenerationError("source-freeze observation field set drift")
    head = value["head_commit"]
    if not isinstance(head, str) or len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise RegenerationError("source-freeze HEAD drift")
    if value["worktree_clean"] is not True or value["sources_exactly_equal_head"] is not True:
        raise RegenerationError("source-freeze source custody failed")
    sources = value["sources"]
    expected = {
        "independent_reducer": REDUCER_SOURCE_PATH,
        "metrics_module": METRICS_SOURCE_PATH,
    }
    if not isinstance(sources, Mapping) or set(sources) != set(expected):
        raise RegenerationError("source binding identity set drift")
    normalized: dict[str, dict[str, Any]] = {}
    for key, relative in expected.items():
        binding = sources[key]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "bytes", "sha256"}:
            raise RegenerationError(f"source binding field set drift: {key}")
        digest = binding["sha256"]
        count = binding["bytes"]
        if (
            binding["path"] != relative
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise RegenerationError(f"source binding value drift: {key}")
        normalized[key] = dict(binding)
    return {
        "head_commit": head,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": normalized,
    }


def _call(module: Any, name: str, *arguments: Any) -> Any:
    function = getattr(module, name, None)
    if not callable(function):
        raise RegenerationError(f"metrics module lacks required pure API: {name}")
    try:
        return function(*arguments)
    except RegenerationError:
        raise
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as exc:
        raise RegenerationError(f"pure metrics API rejected evidence: {name}") from exc


def build_regeneration_receipt(
    output_root: Path | str,
    *,
    metrics_module: Any | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    if not root.is_absolute() or ".." in root.parts or root.is_symlink():
        raise RegenerationError("--output-root must be an absolute non-symlink lexical path")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise RegenerationError("official output root is absent or unresolved") from exc
    if resolved_root != root:
        raise RegenerationError("official output root must not traverse symbolic links")
    module = metrics_module if metrics_module is not None else _load_metrics_module()
    authority = validate_reducer_authority(_call(module, "reducer_authority"), module)
    source_freeze = _validate_source_freeze(_observe_source_freeze(module))

    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RegenerationError("official output root cannot be opened safely") from exc
    try:
        graph_raw = _read_regular_at(root_fd, GRAPH_FILE)
        query_raw = _read_regular_at(root_fd, QUERY_FILE)
        calibration_raw = _read_regular_at(root_fd, CALIBRATION_FILE)
        stage_a_beliefs_raw = _read_regular_at(root_fd, STAGE_A_BELIEFS_FILE)
        stage_a_metrics_raw = _read_regular_at(root_fd, STAGE_A_METRICS_FILE)
        stage_b_trace_raw = _read_regular_at(root_fd, STAGE_B_TRACE_FILE, optional=True)
        stage_b_metrics_raw = _read_regular_at(root_fd, STAGE_B_METRICS_FILE, optional=True)
    finally:
        os.close(root_fd)
    assert all(
        raw is not None
        for raw in (
            graph_raw,
            query_raw,
            calibration_raw,
            stage_a_beliefs_raw,
            stage_a_metrics_raw,
        )
    )
    if (stage_b_trace_raw is None) != (stage_b_metrics_raw is None):
        raise RegenerationError("Stage-B trace and metrics must be both present or both absent")

    graph_manifest = parse_canonical_json(graph_raw, label=GRAPH_FILE)
    calibration = parse_canonical_json(calibration_raw, label=CALIBRATION_FILE)
    supplied_stage_a = parse_canonical_json(stage_a_metrics_raw, label=STAGE_A_METRICS_FILE)
    if not all(isinstance(item, dict) for item in (graph_manifest, calibration, supplied_stage_a)):
        raise RegenerationError("graph, calibration, and Stage-A metrics must be JSON objects")
    query_rows = _parse_jsonl(query_raw, label=QUERY_FILE)
    belief_rows = _parse_jsonl(stage_a_beliefs_raw, label=STAGE_A_BELIEFS_FILE)
    # Invoke every pure metrics validator independently of aggregate reduction.
    _call(module, "validate_graph_manifest", graph_manifest)
    _call(module, "validate_query_rows", tuple(query_rows), graph_manifest)
    _call(
        module,
        "validate_belief_rows",
        tuple(belief_rows),
        tuple(query_rows),
        graph_manifest,
    )
    _call(
        module,
        "validate_calibration",
        calibration,
        graph_manifest,
        tuple(query_rows),
    )
    graphs = _graph_nodes(graph_manifest)
    queries, candidates, role_counts = _validate_queries(query_rows, graphs)
    calibration_validation = _validate_calibration_evidence(
        calibration,
        query_rows=query_rows,
        authority=authority["calibration"],
    )
    stage_a_validation = _validate_stage_a_beliefs(
        belief_rows,
        queries=queries,
        candidates=candidates,
    )

    recomputed_stage_a = _call(
        module,
        "recompute_stage_a_metrics",
        graph_manifest,
        tuple(query_rows),
        tuple(belief_rows),
        calibration,
    )
    if not isinstance(recomputed_stage_a, Mapping):
        raise RegenerationError("Stage-A reducer must return a mapping")
    recomputed_stage_a = dict(recomputed_stage_a)
    _reject_nonfinite(recomputed_stage_a, label="recomputed_stage_a_metrics")
    rebuilt_stage_a_raw = canonical_document_bytes(recomputed_stage_a)
    if stage_a_metrics_raw != rebuilt_stage_a_raw:
        raise RegenerationError("stage_a_metrics.json differs from exact recomputation")
    authorized = _call(module, "stage_a_authorizes_stage_b", recomputed_stage_a)
    if not isinstance(authorized, bool):
        raise RegenerationError("stage_a_authorizes_stage_b must return bool")
    stage_b_present = stage_b_trace_raw is not None
    if stage_b_present != authorized:
        raise RegenerationError("conditional Stage-B evidence does not match Stage-A authorization")

    trace_rows: list[dict[str, Any]] | None = None
    stage_b_validation: dict[str, Any] | None = None
    rebuilt_stage_b_raw: bytes | None = None
    if stage_b_present:
        assert stage_b_trace_raw is not None and stage_b_metrics_raw is not None
        trace_rows = _parse_jsonl(stage_b_trace_raw, label=STAGE_B_TRACE_FILE)
        supplied_stage_b = parse_canonical_json(stage_b_metrics_raw, label=STAGE_B_METRICS_FILE)
        if not isinstance(supplied_stage_b, dict):
            raise RegenerationError("stage_b_metrics.json must contain an object")
        stage_b_validation = _validate_stage_b_trace(
            trace_rows,
            module=module,
            graph_manifest=graph_manifest,
            query_rows=query_rows,
            stage_a_metrics=recomputed_stage_a,
            authority=authority["stage_b_trace"],
        )
        recomputed_stage_b = _call(
            module,
            "recompute_stage_b_metrics",
            graph_manifest,
            tuple(query_rows),
            tuple(trace_rows),
            recomputed_stage_a,
        )
        if not isinstance(recomputed_stage_b, Mapping):
            raise RegenerationError("Stage-B reducer must return a mapping")
        recomputed_stage_b = dict(recomputed_stage_b)
        _reject_nonfinite(recomputed_stage_b, label="recomputed_stage_b_metrics")
        rebuilt_stage_b_raw = canonical_document_bytes(recomputed_stage_b)
        if stage_b_metrics_raw != rebuilt_stage_b_raw:
            raise RegenerationError("stage_b_metrics.json differs from exact recomputation")

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "pass": True,
        "mode": "INDEPENDENT_PERSISTED_EVIDENCE_REDUCTION_ONLY",
        "metrics_module": _metrics_module_name(module),
        "source_freeze": source_freeze,
        "reducer_authority_digest": canonical_digest(authority),
        "inputs": {
            "graph_manifest": _binding(GRAPH_FILE, graph_raw),
            "query_ledger": _binding(QUERY_FILE, query_raw, rows=len(query_rows)),
            "calibration": _binding(CALIBRATION_FILE, calibration_raw),
            "stage_a_beliefs": _binding(
                STAGE_A_BELIEFS_FILE, stage_a_beliefs_raw, rows=len(belief_rows)
            ),
            "stage_a_metrics": _binding(STAGE_A_METRICS_FILE, stage_a_metrics_raw),
            "conditional_stage_b_trace": (
                None
                if stage_b_trace_raw is None
                else _binding(STAGE_B_TRACE_FILE, stage_b_trace_raw, rows=len(trace_rows or ()))
            ),
            "conditional_stage_b_metrics": (
                None
                if stage_b_metrics_raw is None
                else _binding(STAGE_B_METRICS_FILE, stage_b_metrics_raw)
            ),
        },
        "heldout_query_validation": {
            "queries": len(queries),
            "all_query_rows": len(query_rows),
            "role_counts": role_counts,
            "graphs_referenced": len(
                {str(row["graph_id"]) for row in queries.values()}
            ),
            "candidate_identity_coverage_validated": True,
        },
        "calibration_validation": calibration_validation,
        "stage_a_validation": stage_a_validation,
        "stage_a_metrics_exact_byte_equal": True,
        "recomputed_stage_a_metrics_sha256": hashlib.sha256(rebuilt_stage_a_raw).hexdigest(),
        "stage_a_authorizes_stage_b": authorized,
        "conditional_stage_b_present": stage_b_present,
        "conditional_stage_b_validation": stage_b_validation,
        "conditional_stage_b_metrics_exact_byte_equal": None if not stage_b_present else True,
        "recomputed_stage_b_metrics_sha256": (
            None if rebuilt_stage_b_raw is None else hashlib.sha256(rebuilt_stage_b_raw).hexdigest()
        ),
        "scientific_execution_counters": {
            "model_initializations": 0,
            "encoder_initializations": 0,
            "training_steps": 0,
            "inference_calls": 0,
            "simulator_steps": 0,
        },
    }
    return receipt


def _external_receipt_location(output_root: Path | str, output: Path | str) -> tuple[int, str]:
    root = Path(output_root)
    destination = Path(output)
    if not destination.is_absolute() or ".." in destination.parts or destination.name in {"", ".", ".."}:
        raise RegenerationError("--output must be an absolute lexical file path")
    root_resolved = root.resolve(strict=True)
    parent_resolved = destination.parent.resolve(strict=True)
    try:
        parent_resolved.relative_to(root_resolved)
    except ValueError:
        pass
    else:
        raise RegenerationError("regeneration receipt must never be inside official output root")
    try:
        parent_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise RegenerationError("receipt parent cannot be opened safely") from exc
    return parent_fd, destination.name


def emit_regeneration_receipt(
    output_root: Path | str,
    output: Path | str,
    receipt: Mapping[str, Any],
) -> bytes:
    raw = canonical_document_bytes(dict(receipt))
    parent_fd, leaf = _external_receipt_location(output_root, output)
    try:
        try:
            fd = os.open(
                leaf,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            existing = _read_regular_at(parent_fd, leaf)
            if existing != raw:
                raise RegenerationError("existing external receipt bytes drift")
        else:
            try:
                offset = 0
                while offset < len(raw):
                    written = os.write(fd, raw[offset:])
                    if written <= 0:
                        raise OSError("short receipt write")
                    offset += written
                os.fsync(fd)
            finally:
                os.close(fd)
            os.fsync(parent_fd)
    finally:
        os.close(parent_fd)
    return raw


def verify_and_emit(
    output_root: Path | str,
    output: Path | str,
    *,
    metrics_module: Any | None = None,
) -> dict[str, Any]:
    receipt = build_regeneration_receipt(output_root, metrics_module=metrics_module)
    emit_regeneration_receipt(output_root, output, receipt)
    return receipt


def validate_existing_regeneration_receipt(
    output_root: Path | str,
    output: Path | str,
    *,
    metrics_module: Any | None = None,
) -> dict[str, Any]:
    parent_fd, leaf = _external_receipt_location(output_root, output)
    try:
        raw = _read_regular_at(parent_fd, leaf)
    finally:
        os.close(parent_fd)
    assert raw is not None
    supplied = parse_canonical_json(raw, label=str(output))
    if not isinstance(supplied, dict):
        raise RegenerationError("external regeneration receipt must contain an object")
    rebuilt = build_regeneration_receipt(output_root, metrics_module=metrics_module)
    if raw != canonical_document_bytes(rebuilt):
        raise RegenerationError("external receipt differs from exact source-bound rebuild")
    return rebuilt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-existing", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_existing:
        receipt = validate_existing_regeneration_receipt(args.output_root, args.output)
    else:
        receipt = verify_and_emit(args.output_root, args.output)
    print(
        json.dumps(
            {
                "status": "PASS",
                "receipt": str(args.output),
                "document_sha256": hashlib.sha256(
                    canonical_document_bytes(receipt)
                ).hexdigest(),
                "stage_a_authorizes_stage_b": receipt["stage_a_authorizes_stage_b"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
