"""Pure reducers for OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1.

Only JSON-compatible graph, query, belief, calibration, and trace evidence is
accepted.  This module imports no model framework and performs no rendering,
encoding, inference, simulation, training, or file I/O.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Any

from lewm.safety import occluded_goal_topological_belief_v1_contract as C


class OccludedGoalMetricsError(ValueError):
    """Raised when reducer evidence is incomplete, ambiguous, or inconsistent."""


GRAPH_ROOT_FIELDS = {
    "schema", "experiment_id", "identity_disjointness", "graphs", "content_digest"
}
GRAPH_FIELDS = {
    "graph_id", "episode_id", "scene_id", "procedural_seed", "episode_path_id",
    "identity_domain",
    "family", "role", "goal_node_id", "nodes", "edges", "phase_a_traversal",
    "family_adequacy_witness", "constructed_set_caveat", "stage_b_start_node_id",
    "stage_b_decision_start_node_id", "stage_b_prelude_edge_ids",
}
IDENTITY_DISJOINTNESS_FIELDS = {
    "schema", "experiment_id", "identity_domain", "authority",
    "current_projection", "comparisons",
}
CURRENT_IDENTITY_PROJECTION_FIELDS = {
    "scene_identity_count", "scene_identity_canonical_json_sha256",
    "episode_or_state_identity_count",
    "episode_or_state_identity_canonical_json_sha256",
    "textual_path_identity_count", "textual_path_identity_canonical_json_sha256",
    "numeric_seed_count", "numeric_seed_canonical_json_sha256",
    "procedural_seed_minimum", "procedural_seed_maximum",
}
IDENTITY_COMPARISON_FIELDS = {
    "scene_id_overlap_count", "scene_id_sha256_overlap_count",
    "episode_or_state_id_overlap_count", "procedural_seed_overlap_count",
    "textual_path_identity_overlap_count", "structured_path_overlap_count",
}
NODE_FIELDS = {
    "node_id", "alias_group_id", "node_kind", "module_index", "side_label",
    "keyframe_observation_id", "observation_descriptor", "teacher_visit_index",
    "goal_keyframe", "pixel_template_id", "pixel_sha256", "keyframe_timestamp_s",
}
EDGE_FIELDS = {
    "edge_id", "source_node_id", "target_node_id", "port_label",
    "executed_action_label", "relative_waypoint", "edge_cost", "teacher_edge",
    "physically_executable", "oracle_admissible",
}
PHASE_A_VISIT_FIELDS = {
    "visit_index", "node_id", "observation_id", "arrival_edge_id",
    "executed_action_label", "timestamp_s",
}
FAMILY_WITNESS_FIELDS = {
    "family", "alias_group_id", "alias_node_ids", "different_next_port_labels",
    "witness_node_ids", "witness_edge_ids", "cycle_node_ids", "cycle_edge_ids",
}
QUERY_FIELDS = {
    "query_id", "episode_id", "family", "role", "graph_id", "query_index",
    "depth_since_last_unambiguous_observation",
    "true_node_id", "true_next_edge_id", "true_next_port_label",
    "prior_visited_node_ids", "history_observation_ids",
    "history_executed_action_labels", "unresolved_condition_ids",
    "candidate_node_ids", "alias_node_ids", "alias_distractor_node_ids",
    "oracle_admissible_edge_ids", "query_observation_id", "query_pixel_template_id",
    "query_pixel_sha256", "query_timestamp_s", "goal_visible",
    "stage_b_choice_macros", "stage_b_local_candidate_outcomes",
}
STAGE_B_MACRO_FIELDS = {
    "outcome", "destination_node_id", "constituent_edge_ids",
}
STAGE_B_CANDIDATE_OUTCOME_FIELDS = {
    "candidate_index", "candidate_name", "actual_edge_id", "actual_port_label",
    "endpoint_node_id", "prefix_distance_m", "immediate_contact",
    "successor_viable", "stuck", "oracle_admissible",
}
CALIBRATION_FIELDS = {
    "schema", "experiment_id", "selection_role", "calibration_episode_ids",
    "selected_parameters", "selected_grid_index", "grid_results",
    "grid_query_results", "content_digest",
}
CALIBRATION_PARAMETER_FIELDS = set(C.CALIBRATION_GRID_ORDER)
CALIBRATION_RESULT_FIELDS = {
    "grid_index", "parameters", "correct_next_edge_accuracy", "localisation_top3",
    "normalized_graph_distance_regret", "false_confident_localisation_rate",
    "abstention_rate",
}
CALIBRATION_QUERY_RESULT_FIELDS = {
    "grid_index", "query_id", "edge_correct", "localisation_top3",
    "normalized_regret", "false_confident", "abstained",
}
BELIEF_FIELDS = {
    "query_id", "condition_id", "node_ids", "observation_similarities",
    "observation_likelihoods", "transition_prior_probabilities",
    "preprojection_probabilities", "posterior_probabilities", "selected_node_id",
    "selected_edge_id", "selected_port_label", "normalized_entropy", "abstained",
    "action_history_position_mapping",
}
STAGE_B_TRACE_FIELDS = {
    "step_id", "execution_id", "episode_id", "family", "condition_id",
    "source_condition_id", "decision_index", "query_id", "query_index",
    "selected_port_label", "true_next_port_label", "abstained",
    "observation_id", "observation_pixel_sha256", "actual_node_id_before",
    "actual_node_id_after", "actual_alias_group_id", "selected_node_id",
    "selected_alias_group_id", "prior_visited_node_ids", "belief_node_ids",
    "belief_probabilities", "normalized_entropy", "entropy_threshold",
    "belief_proposed_port_label", "belief_selected_edge_id",
    "filter_update_observation_ids", "filter_update_pixel_sha256s",
    "filter_update_node_ids", "filter_update_incoming_action_labels",
    "filter_update_observation_similarities",
    "filter_update_observation_likelihoods",
    "filter_update_transition_prior_probabilities",
    "filter_update_preprojection_probabilities",
    "filter_update_posterior_probabilities",
    "stored_relative_waypoint", "action_disposition", "ranker_called",
    "local_candidate_scores", "local_oracle_admissible_candidate_indices",
    "local_candidate_index", "local_candidate_name", "local_candidate_score",
    "local_execution_success", "ranker_latency_ms", "planning_latency_ms",
    "ranker_previous_applied_command", "ranker_control_history",
    "selected_candidate_applied_commands", "post_decision_previous_applied_command",
    "post_decision_control_history",
    "executed_port_label", "executed_choice_edge_id", "constituent_node_ids",
    "constituent_edge_ids", "constituent_action_labels",
    "constituent_edge_costs_m", "constituent_observation_ids",
    "constituent_observation_pixel_sha256s", "reobservation_id",
    "reobservation_pixel_sha256", "oracle_admissible_execution",
    "geodesic_distance_before_m", "geodesic_distance_after_m", "decision_budget",
    "prelude_distance_m",
    "consecutive_nonmovement_decisions", "recovery_pending_before",
    "recovery_pending_after", "terminal_reason",
    "false_confident_wrong_turn", "false_place_merge", "false_loop_closure",
    "geodesic_progress_m", "replan", "recovery", "immediate_contact", "stuck",
    "executed_distance_m", "oracle_path_distance_m", "terminal",
    "goal_reached",
}


def stage_a_condition_ids() -> tuple[str, ...]:
    return tuple(C.CONDITION_IDS)


def deterministic_action_history_position_mapping(
    query_id: str, action_count: int
) -> tuple[int, ...]:
    """Return the frozen query-bound complete rotation derangement."""
    _string(query_id, "query_id")
    if isinstance(action_count, bool) or not isinstance(action_count, int) or action_count < 2:
        raise OccludedGoalMetricsError("shuffled action histories require at least two positions")
    shift = 1 + (int(hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:8], 16) % (action_count - 1))
    return tuple((index + shift) % action_count for index in range(action_count))


def graph_manifest_authority() -> dict[str, Any]:
    return {
        "root_fields": sorted(GRAPH_ROOT_FIELDS),
        "graph_fields": sorted(GRAPH_FIELDS),
        "node_fields": sorted(NODE_FIELDS),
        "edge_fields": sorted(EDGE_FIELDS),
        "phase_a_visit_fields": sorted(PHASE_A_VISIT_FIELDS),
        "family_adequacy_witness_fields": sorted(FAMILY_WITNESS_FIELDS),
        "identity_disjointness_fields": sorted(IDENTITY_DISJOINTNESS_FIELDS),
        "current_identity_projection_fields": sorted(CURRENT_IDENTITY_PROJECTION_FIELDS),
        "identity_comparison_fields": sorted(IDENTITY_COMPARISON_FIELDS),
        "prior_exclusion_authority": copy.deepcopy(C.PRIOR_PANEL_EXCLUSION_AUTHORITY),
        "graphs": C.EPISODE_COUNT,
        "families": list(C.FAMILY_IDS),
        "roles": dict(C.SPLIT_EPISODE_COUNTS),
        "port_labels": list(C.PORT_LABEL_ORDER),
    }


def query_row_authority() -> dict[str, Any]:
    return {
        "fields": sorted(QUERY_FIELDS),
        "stage_b_macro_fields": sorted(STAGE_B_MACRO_FIELDS),
        "stage_b_candidate_outcome_fields": sorted(STAGE_B_CANDIDATE_OUTCOME_FIELDS),
        "stage_b_local_candidate_ids": list(C.STAGE_B_LOCAL_CANDIDATE_IDS),
        "roles": dict(C.SPLIT_EPISODE_COUNTS),
        "rows": C.EPISODE_COUNT * C.QUERIES_PER_EPISODE,
        "queries_per_episode": C.QUERIES_PER_EPISODE,
        "query_indices": list(range(C.QUERIES_PER_EPISODE)),
        "depths_since_last_unambiguous_observation": list(range(4, 12)),
        "history_lengths": "complete registered prefix; not inferred from query depth",
        "condition_specific_unresolved": True,
    }


def belief_row_authority() -> dict[str, Any]:
    return {
        "fields": sorted(BELIEF_FIELDS),
        "condition_ids": list(C.CONDITION_IDS),
        "rows": C.HELDOUT_QUERY_COUNT * len(C.CONDITION_IDS),
        "probability_floor_for_metrics": C.PROBABILITY_FLOOR,
        "node_order": "exact candidate_node_ids order from the query row",
        "raw_evidence": [
            "observation_similarities", "observation_likelihoods",
            "transition_prior_probabilities", "preprojection_probabilities",
            "posterior_probabilities", "selected_node_id", "selected_port_label",
            "selected_edge_id", "normalized_entropy", "abstained",
            "action_history_position_mapping",
        ],
    }


def calibration_authority() -> dict[str, Any]:
    return {
        "fields": sorted(CALIBRATION_FIELDS),
        "parameter_fields": list(C.CALIBRATION_GRID_ORDER),
        "grid": {key: list(values) for key, values in C.CALIBRATION_GRID.items()},
        "selection_role": "CALIBRATION",
        "episodes": C.SPLIT_EPISODE_COUNTS["CALIBRATION"],
        "selection": copy.deepcopy(C.CALIBRATION_SELECTION),
        "grid_result_fields": sorted(CALIBRATION_RESULT_FIELDS),
        "grid_query_result_fields": sorted(CALIBRATION_QUERY_RESULT_FIELDS),
        "grid_results": 144,
        "grid_query_results": 144 * C.HELDOUT_QUERY_COUNT,
    }


def stage_b_trace_authority() -> dict[str, Any]:
    return {
        "fields": sorted(STAGE_B_TRACE_FIELDS),
        "condition_ids": list(C.STAGE_B_CONDITION_IDS),
        "episodes_per_condition": C.STAGE_B_GATE["episodes_per_condition"],
        "identity": ["condition_id", "episode_id", "decision_index"],
        "terminal_rows_per_execution": 1,
        "execution_policy": copy.deepcopy(C.STAGE_B_EXECUTION_POLICY),
        "action_dispositions": [
            "EXECUTED_CANDIDATE", "ENTROPY_ABSTENTION", "NO_ROUTE_WITNESS",
            "NO_ORACLE_ADMISSIBLE_LOCAL_CANDIDATE",
        ],
        "terminal_reasons": [
            "GOAL_REACHED", "NONMOVEMENT_LIMIT", "DECISION_BUDGET_EXHAUSTED",
        ],
    }


def reducer_authority() -> dict[str, Any]:
    return C.attach_content_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.reducer_authority.v1",
            "experiment_id": C.EXPERIMENT_ID,
            "graph_manifest": graph_manifest_authority(),
            "query_rows": query_row_authority(),
            "belief_rows": belief_row_authority(),
            "calibration": calibration_authority(),
            "stage_b_trace": stage_b_trace_authority(),
            "stage_a_authorization_path": "decision.stage_b_authorized",
        }
    )


def _require_mapping(value: Any, *, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OccludedGoalMetricsError(f"{label} field set drift")
    return dict(value)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise OccludedGoalMetricsError(f"{label} must be a nonempty string")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OccludedGoalMetricsError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise OccludedGoalMetricsError(f"{label} must be finite")
    return result


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise OccludedGoalMetricsError(f"{label} must be Boolean")
    return value


def _strings(value: Any, label: str, *, unique: bool = True) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise OccludedGoalMetricsError(f"{label} must be a sequence")
    result = tuple(_string(item, f"{label}[]") for item in value)
    if unique and len(set(result)) != len(result):
        raise OccludedGoalMetricsError(f"{label} contains duplicates")
    return result


def _validate_digest(value: Mapping[str, Any], label: str) -> None:
    try:
        C.validate_content_digest(value)
    except C.OccludedGoalContractError as exc:
        raise OccludedGoalMetricsError(f"{label} content digest drift") from exc


def _canonical_identity_sha256(values: Sequence[Any]) -> str:
    normalized = sorted(set(values))
    return hashlib.sha256(C.canonical_json_bytes(normalized)[:-1]).hexdigest()


def _current_identity_projection(graphs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scenes = [graph["scene_id"] for graph in graphs]
    episodes = [graph["episode_id"] for graph in graphs]
    paths = [graph["episode_path_id"] for graph in graphs]
    seeds = [graph["procedural_seed"] for graph in graphs]
    graph_ids = [graph["graph_id"] for graph in graphs]
    episode_or_state = episodes + graph_ids
    return {
        "scene_identity_count": len(set(scenes)),
        "scene_identity_canonical_json_sha256": _canonical_identity_sha256(scenes),
        "episode_or_state_identity_count": len(set(episode_or_state)),
        "episode_or_state_identity_canonical_json_sha256": _canonical_identity_sha256(episode_or_state),
        "textual_path_identity_count": len(set(paths)),
        "textual_path_identity_canonical_json_sha256": _canonical_identity_sha256(paths),
        "numeric_seed_count": len(set(seeds)),
        "numeric_seed_canonical_json_sha256": _canonical_identity_sha256(seeds),
        "procedural_seed_minimum": min(seeds),
        "procedural_seed_maximum": max(seeds),
    }


def build_identity_disjointness_evidence(
    graphs: Sequence[Mapping[str, Any]], comparisons: Mapping[str, Any]
) -> dict[str, Any]:
    """Build nested evidence after the runner performs bound-authority set comparisons."""
    comparison = _require_mapping(
        comparisons, fields=IDENTITY_COMPARISON_FIELDS, label="identity comparisons"
    )
    return {
        "schema": "occluded_goal_topological_belief_v1.identity_disjointness.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "identity_domain": C.IDENTITY_DOMAIN,
        "authority": copy.deepcopy(C.PRIOR_PANEL_EXCLUSION_AUTHORITY),
        "current_projection": _current_identity_projection(graphs),
        "comparisons": comparison,
    }


def validate_identity_disjointness(
    value: Mapping[str, Any], graphs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    row = _require_mapping(
        value, fields=IDENTITY_DISJOINTNESS_FIELDS, label="identity_disjointness"
    )
    if (
        row["schema"] != "occluded_goal_topological_belief_v1.identity_disjointness.v1"
        or row["experiment_id"] != C.EXPERIMENT_ID
        or row["identity_domain"] != C.IDENTITY_DOMAIN
        or row["authority"] != C.PRIOR_PANEL_EXCLUSION_AUTHORITY
    ):
        raise OccludedGoalMetricsError("identity disjointness authority drift")
    projection = _require_mapping(
        row["current_projection"],
        fields=CURRENT_IDENTITY_PROJECTION_FIELDS,
        label="current identity projection",
    )
    if projection != _current_identity_projection(graphs):
        raise OccludedGoalMetricsError("current identity projection differs from graph rows")
    comparisons = _require_mapping(
        row["comparisons"], fields=IDENTITY_COMPARISON_FIELDS, label="identity comparisons"
    )
    for field in sorted(IDENTITY_COMPARISON_FIELDS):
        value = comparisons[field]
        if isinstance(value, bool) or not isinstance(value, int) or value != 0:
            raise OccludedGoalMetricsError(f"prior identity overlap is nonzero: {field}")
    prior = C.PRIOR_PANEL_EXCLUSION_AUTHORITY["prior_projection"]
    if prior["identity_domain_occurrences"] != 0 or prior["reserved_seed_overlap_count"] != 0:
        raise OccludedGoalMetricsError("frozen prior projection does not reserve the new domain")
    return copy.deepcopy(row)


def _reverse_distances(node_ids: tuple[str, ...], edges: Sequence[Mapping[str, Any]], goal: str) -> dict[str, float]:
    reverse: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in edges:
        reverse[str(edge["target_node_id"])].append((str(edge["source_node_id"]), float(edge.get("edge_cost", 1.0))))
    distance = {goal: 0.0}; queue: list[tuple[float, str]] = [(0.0, goal)]
    while queue:
        cost, target = heapq.heappop(queue)
        if cost != distance.get(target):
            continue
        for source, edge_cost in sorted(reverse.get(target, ())):
            candidate = cost + edge_cost
            if candidate < distance.get(source, math.inf):
                distance[source] = candidate
                heapq.heappush(queue, (candidate, source))
    if set(distance) != set(node_ids):
        raise OccludedGoalMetricsError("every graph node must reach the goal")
    return distance


def _shortest_edge_path(
    edges: Sequence[Mapping[str, Any]], start: str, destination: str
) -> tuple[str, ...]:
    """Weighted shortest path with complete edge-id-sequence tie-breaking."""
    if start == destination:
        return ()
    outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        if edge["physically_executable"] and edge["oracle_admissible"]:
            outgoing[str(edge["source_node_id"])].append(edge)
    queue: list[tuple[float, tuple[str, ...], str]] = [(0.0, (), start)]
    best: dict[str, tuple[float, tuple[str, ...]]] = {start: (0.0, ())}
    while queue:
        cost, path, node = heapq.heappop(queue)
        if (cost, path) != best.get(node):
            continue
        if node == destination:
            return path
        for edge in sorted(outgoing.get(node, ()), key=lambda item: item["edge_id"]):
            candidate = (
                cost + float(edge["edge_cost"]),
                path + (str(edge["edge_id"]),),
            )
            target = str(edge["target_node_id"])
            if target not in best or candidate < best[target]:
                best[target] = candidate
                heapq.heappush(queue, (*candidate, target))
    raise OccludedGoalMetricsError(
        f"no executable oracle-admissible path from {start!r} to {destination!r}"
    )


def _macro_path_after_choice(
    graph: Mapping[str, Any], choice_edge_id: str, destination: str
) -> tuple[str, ...]:
    edge_by_id = {str(edge["edge_id"]): edge for edge in graph["edges"]}
    edge = edge_by_id.get(choice_edge_id)
    if edge is None or not edge["physically_executable"] or not edge["oracle_admissible"]:
        raise OccludedGoalMetricsError("Stage-B choice edge is absent or inadmissible")
    remainder = _shortest_edge_path(
        graph["edges"], str(edge["target_node_id"]), destination
    )
    return (choice_edge_id, *remainder)


def validate_graph_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    root = _require_mapping(value, fields=GRAPH_ROOT_FIELDS, label="graph_manifest")
    _validate_digest(root, "graph_manifest")
    if root["schema"] != "occluded_goal_topological_belief_v1.graph_manifest.v1":
        raise OccludedGoalMetricsError("graph manifest schema drift")
    if root["experiment_id"] != C.EXPERIMENT_ID:
        raise OccludedGoalMetricsError("graph manifest experiment drift")
    if not isinstance(root["graphs"], list) or len(root["graphs"]) != C.EPISODE_COUNT:
        raise OccludedGoalMetricsError("graph manifest must contain exactly 96 graphs")
    normalized: list[dict[str, Any]] = []
    graph_ids: set[str] = set(); episode_ids: set[str] = set(); scene_ids: set[str] = set()
    procedural_seeds: set[int] = set(); episode_path_ids: set[str] = set()
    role_counts: dict[str, int] = defaultdict(int); family_role: dict[tuple[str, str], int] = defaultdict(int)
    for index, original in enumerate(root["graphs"]):
        graph = _require_mapping(original, fields=GRAPH_FIELDS, label=f"graph[{index}]")
        graph_id = _string(graph["graph_id"], f"graph[{index}].graph_id")
        episode_id = _string(graph["episode_id"], f"graph[{index}].episode_id")
        scene_id = _string(graph["scene_id"], f"graph[{index}].scene_id")
        episode_path_id = _string(graph["episode_path_id"], f"graph[{index}].episode_path_id")
        if graph["identity_domain"] != C.IDENTITY_DOMAIN:
            raise OccludedGoalMetricsError("graph identity domain drift")
        if (
            not graph_id.startswith(f"{C.IDENTITY_DOMAIN}-")
            or not scene_id.startswith(C.SCENE_ID_PREFIX)
            or not episode_id.startswith(C.EPISODE_ID_PREFIX)
            or not episode_path_id.startswith(C.EPISODE_PATH_ID_PREFIX)
        ):
            raise OccludedGoalMetricsError("graph identities escaped the reserved ogtb-v1 namespace")
        procedural_seed = graph["procedural_seed"]
        if isinstance(procedural_seed, bool) or not isinstance(procedural_seed, int) or procedural_seed < 0:
            raise OccludedGoalMetricsError("procedural_seed must be a nonnegative integer")
        family = _string(graph["family"], f"graph[{index}].family")
        role = _string(graph["role"], f"graph[{index}].role")
        if (graph_id in graph_ids or episode_id in episode_ids or scene_id in scene_ids or
                procedural_seed in procedural_seeds or episode_path_id in episode_path_ids):
            raise OccludedGoalMetricsError("graph, episode, scene, seed, and path identities must be globally unique")
        graph_ids.add(graph_id); episode_ids.add(episode_id); scene_ids.add(scene_id)
        procedural_seeds.add(procedural_seed); episode_path_ids.add(episode_path_id)
        if graph["constructed_set_caveat"] != C.CONSTRUCTED_SET_CAVEAT:
            raise OccludedGoalMetricsError("constructed challenge-set caveat drift")
        if family not in C.FAMILY_IDS or role not in C.SPLIT_ROLE_IDS:
            raise OccludedGoalMetricsError("graph family/role drift")
        role_counts[role] += 1; family_role[(family, role)] += 1
        if not isinstance(graph["nodes"], list) or len(graph["nodes"]) < 4:
            raise OccludedGoalMetricsError("each graph requires at least four nodes")
        nodes: list[dict[str, Any]] = []; node_ids: list[str] = []; alias_members: dict[str, list[str]] = defaultdict(list)
        keyframes: set[str] = set(); visit_orders: set[int] = set(); keyframe_times: set[float] = set(); goal_flags: list[str] = []
        for node_index, original_node in enumerate(graph["nodes"]):
            node = _require_mapping(original_node, fields=NODE_FIELDS, label=f"graph[{index}].node[{node_index}]")
            node_id = _string(node["node_id"], "node_id")
            alias = _string(node["alias_group_id"], "alias_group_id")
            _string(node["node_kind"], "node_kind")
            module_index = node["module_index"]
            if isinstance(module_index, bool) or not isinstance(module_index, int) or module_index < 0:
                raise OccludedGoalMetricsError("module_index must be a nonnegative integer")
            _string(node["side_label"], "side_label")
            keyframe = _string(node["keyframe_observation_id"], "keyframe_observation_id")
            _string(node["observation_descriptor"], "observation_descriptor")
            _string(node["pixel_template_id"], "pixel_template_id")
            pixel_sha = _string(node["pixel_sha256"], "pixel_sha256")
            if len(pixel_sha) != 64 or any(char not in "0123456789abcdef" for char in pixel_sha):
                raise OccludedGoalMetricsError("pixel_sha256 must be lowercase SHA-256 hex")
            keyframe_time = _finite(node["keyframe_timestamp_s"], "keyframe_timestamp_s")
            visit_order = node["teacher_visit_index"]
            if isinstance(visit_order, bool) or not isinstance(visit_order, int) or visit_order < 0:
                raise OccludedGoalMetricsError("teacher_visit_index must be a nonnegative integer")
            is_goal = _bool(node["goal_keyframe"], "goal_keyframe")
            if node_id in node_ids:
                raise OccludedGoalMetricsError("duplicate node identity")
            if keyframe in keyframes or visit_order in visit_orders or keyframe_time in keyframe_times:
                raise OccludedGoalMetricsError("node keyframe/teacher visit order must be unique")
            node_ids.append(node_id); alias_members[alias].append(node_id)
            keyframes.add(keyframe); visit_orders.add(visit_order); keyframe_times.add(keyframe_time)
            if is_goal:
                goal_flags.append(node_id)
            nodes.append({**node, "node_id": node_id, "alias_group_id": alias})
        goal = _string(graph["goal_node_id"], "goal_node_id")
        if goal not in node_ids:
            raise OccludedGoalMetricsError("goal node is absent")
        if goal_flags != [goal]:
            raise OccludedGoalMetricsError("exactly the goal node must carry goal_keyframe=true")
        stage_b_start = _string(graph["stage_b_start_node_id"], "stage_b_start_node_id")
        stage_b_decision_start = _string(
            graph["stage_b_decision_start_node_id"], "stage_b_decision_start_node_id"
        )
        if stage_b_start != goal:
            raise OccludedGoalMetricsError("Stage-B must start at the observed goal node")
        if stage_b_decision_start not in node_ids or stage_b_decision_start == goal:
            raise OccludedGoalMetricsError("Stage-B decision start must be a registered non-goal node")
        if not isinstance(graph["edges"], list) or not graph["edges"]:
            raise OccludedGoalMetricsError("graph edges must be nonempty")
        edges: list[dict[str, Any]] = []; edge_ids: set[str] = set(); source_ports: set[tuple[str, str]] = set()
        for edge_index, original_edge in enumerate(graph["edges"]):
            edge = _require_mapping(original_edge, fields=EDGE_FIELDS, label=f"graph[{index}].edge[{edge_index}]")
            edge_id = _string(edge["edge_id"], "edge_id")
            source = _string(edge["source_node_id"], "source_node_id")
            target = _string(edge["target_node_id"], "target_node_id")
            port = _string(edge["port_label"], "port_label")
            if edge_id in edge_ids or source not in node_ids or target not in node_ids or source == target:
                raise OccludedGoalMetricsError("invalid or duplicate directed edge")
            if port not in C.PORT_LABEL_ORDER or (source, port) in source_ports:
                raise OccludedGoalMetricsError("port labels must be unique per source")
            waypoint = edge["relative_waypoint"]
            if not isinstance(waypoint, (list, tuple)) or len(waypoint) != 3:
                raise OccludedGoalMetricsError("relative_waypoint must contain x,y,yaw")
            waypoint = [_finite(item, "relative_waypoint") for item in waypoint]
            action = _string(edge["executed_action_label"], "executed_action_label")
            edge_cost = _finite(edge["edge_cost"], "edge_cost")
            if edge_cost <= 0.0:
                raise OccludedGoalMetricsError("edge_cost must be positive")
            _bool(edge["teacher_edge"], "teacher_edge")
            _bool(edge["physically_executable"], "physically_executable")
            _bool(edge["oracle_admissible"], "oracle_admissible")
            if edge["oracle_admissible"] and not edge["physically_executable"]:
                raise OccludedGoalMetricsError("oracle-admissible edges must be physically executable")
            edge_ids.add(edge_id); source_ports.add((source, port))
            edges.append({**edge, "executed_action_label": action, "relative_waypoint": waypoint, "edge_cost": edge_cost})
        distances = _reverse_distances(tuple(node_ids), edges, goal)
        outgoing = defaultdict(list)
        for edge in edges:
            outgoing[edge["source_node_id"]].append(edge)
        for node_id in node_ids:
            if node_id != goal and not outgoing[node_id]:
                raise OccludedGoalMetricsError("non-goal graph node has no outgoing edge")
        prelude = _strings(graph["stage_b_prelude_edge_ids"], "stage_b_prelude_edge_ids")
        expected_prelude = _shortest_edge_path(edges, stage_b_start, stage_b_decision_start)
        if prelude != expected_prelude or not prelude:
            raise OccludedGoalMetricsError(
                "Stage-B prelude must be the deterministic executable path from goal to query zero"
            )
        if not any(
            edge["physically_executable"] and edge["oracle_admissible"]
            for edge in outgoing[stage_b_decision_start]
        ):
            raise OccludedGoalMetricsError("Stage-B decision start has no executable admissible edge")
        # Every multi-node registered alias must differ in a route port.  Singleton
        # groups are ordinary, non-aliased graph nodes and are permitted.
        route_port = {}
        for node_id in node_ids:
            if node_id == goal:
                continue
            candidates = sorted(outgoing[node_id], key=lambda row: (row["edge_cost"] + distances[row["target_node_id"]], row["edge_id"]))
            route_port[node_id] = candidates[0]["port_label"]
        for members in alias_members.values():
            active = [node for node in members if node != goal]
            if len(active) >= 2 and len({route_port[node] for node in active}) < 2:
                raise OccludedGoalMetricsError("each multi-node alias group must contain route-conflicting nodes")
            if len(active) >= 2 and (
                len({node["observation_descriptor"] for node in nodes if node["node_id"] in active}) != 1 or
                len({node["pixel_template_id"] for node in nodes if node["node_id"] in active}) != 1 or
                len({node["pixel_sha256"] for node in nodes if node["node_id"] in active}) != 1
            ):
                raise OccludedGoalMetricsError("registered alias nodes must share exact observation-template evidence")

        traversal_raw = graph["phase_a_traversal"]
        if not isinstance(traversal_raw, list) or not traversal_raw:
            raise OccludedGoalMetricsError("Phase-A traversal evidence must be nonempty")
        traversal: list[dict[str, Any]] = []; encountered: set[str] = set(); phase_a_observations: set[str] = set(); previous_time = -math.inf
        edge_by_id = {edge["edge_id"]: edge for edge in edges}
        node_by_id = {node["node_id"]: node for node in nodes}
        goal_visits = 0
        for visit_index, original_visit in enumerate(traversal_raw):
            visit = _require_mapping(original_visit, fields=PHASE_A_VISIT_FIELDS, label=f"phase_a_traversal[{visit_index}]")
            if visit["visit_index"] != visit_index or isinstance(visit["visit_index"], bool):
                raise OccludedGoalMetricsError("Phase-A visit indices must be contiguous from zero")
            node_id = _string(visit["node_id"], "Phase-A node_id")
            if node_id not in node_by_id:
                raise OccludedGoalMetricsError("Phase-A traversal escaped the graph")
            observation_id = _string(visit["observation_id"], "Phase-A observation_id")
            if observation_id in phase_a_observations:
                raise OccludedGoalMetricsError("Phase-A observation occurrence identities must be unique")
            timestamp = _finite(visit["timestamp_s"], "Phase-A timestamp_s")
            if timestamp <= previous_time:
                raise OccludedGoalMetricsError("Phase-A timestamps must strictly increase")
            previous_time = timestamp
            if visit_index == 0:
                if visit["arrival_edge_id"] is not None or visit["executed_action_label"] is not None:
                    raise OccludedGoalMetricsError("first Phase-A visit cannot have an arrival edge/action")
            else:
                arrival_edge = _string(visit["arrival_edge_id"], "Phase-A arrival_edge_id")
                action = _string(visit["executed_action_label"], "Phase-A executed_action_label")
                edge = edge_by_id.get(arrival_edge)
                if edge is None or edge["source_node_id"] != traversal[-1]["node_id"] or edge["target_node_id"] != node_id or edge["executed_action_label"] != action:
                    raise OccludedGoalMetricsError("Phase-A traversal edge/action is not consecutive")
                if not edge["teacher_edge"] or not edge["physically_executable"]:
                    raise OccludedGoalMetricsError("Phase-A traversal must use executable teacher edges")
            if node_id not in encountered:
                if observation_id != node_by_id[node_id]["keyframe_observation_id"] or not math.isclose(timestamp, float(node_by_id[node_id]["keyframe_timestamp_s"]), rel_tol=0.0, abs_tol=1e-12):
                    raise OccludedGoalMetricsError("first Phase-A encounter must bind the node keyframe occurrence")
            phase_a_observations.add(observation_id); encountered.add(node_id); goal_visits += int(node_id == goal)
            traversal.append(dict(visit))
        if encountered != set(node_ids):
            raise OccludedGoalMetricsError("every posterior-bank node requires a Phase-A teacher encounter")
        if goal_visits != 1:
            raise OccludedGoalMetricsError("Phase-A traversal must encounter the goal exactly once")
        for node in nodes:
            first_visit = next(visit["visit_index"] for visit in traversal if visit["node_id"] == node["node_id"])
            if node["teacher_visit_index"] != first_visit:
                raise OccludedGoalMetricsError("node teacher_visit_index differs from Phase-A provenance")

        witness = _require_mapping(graph["family_adequacy_witness"], fields=FAMILY_WITNESS_FIELDS, label="family_adequacy_witness")
        if witness["family"] != family:
            raise OccludedGoalMetricsError("family witness family drift")
        witness_alias = _string(witness["alias_group_id"], "witness alias_group_id")
        witness_alias_nodes = _strings(witness["alias_node_ids"], "witness alias_node_ids")
        witness_ports = _strings(witness["different_next_port_labels"], "witness ports")
        witness_nodes = _strings(witness["witness_node_ids"], "witness nodes")
        witness_edges = _strings(witness["witness_edge_ids"], "witness edges")
        cycle_nodes = _strings(witness["cycle_node_ids"], "cycle nodes")
        cycle_edges = _strings(witness["cycle_edge_ids"], "cycle edges")
        if len(witness_alias_nodes) < 2 or any(node not in node_by_id or node_by_id[node]["alias_group_id"] != witness_alias for node in witness_alias_nodes):
            raise OccludedGoalMetricsError("family witness alias membership drift")
        derived_ports = {route_port[node] for node in witness_alias_nodes if node != goal}
        if set(witness_ports) != derived_ports or len(derived_ports) < 2:
            raise OccludedGoalMetricsError("family witness must bind distinct oracle route ports")
        if any(node not in node_by_id for node in witness_nodes) or any(edge not in edge_by_id for edge in witness_edges):
            raise OccludedGoalMetricsError("family witness escaped graph identities")
        if not witness_nodes or not witness_edges or not set(witness_alias_nodes).issubset(witness_nodes):
            raise OccludedGoalMetricsError("family witness must contain its alias nodes and structural edges")
        expected_kind = C.ALIASING_AUTHORITY["family_witness_node_kinds"][family]
        if any(node_by_id[node]["node_kind"] != expected_kind for node in witness_alias_nodes):
            raise OccludedGoalMetricsError("family witness node_kind drift")
        if family in {"REPEATED_CORRIDOR", "REPEATED_ROOM"} and len({node_by_id[node]["module_index"] for node in witness_alias_nodes}) < 2:
            raise OccludedGoalMetricsError("repeated structures must occupy distinct graph modules")
        if family == "MIRRORED_JUNCTION" and len({node_by_id[node]["side_label"] for node in witness_alias_nodes}) < 2:
            raise OccludedGoalMetricsError("mirrored junction witness must bind differing sides")
        if family == "LOOP_ALIAS":
            if len(cycle_nodes) < 2 or len(cycle_edges) != len(cycle_nodes):
                raise OccludedGoalMetricsError("LOOP_ALIAS requires an explicit nonempty cycle")
            for cycle_index, edge_id in enumerate(cycle_edges):
                edge = edge_by_id.get(edge_id)
                source = cycle_nodes[cycle_index]; target = cycle_nodes[(cycle_index + 1) % len(cycle_nodes)]
                if edge is None or edge["source_node_id"] != source or edge["target_node_id"] != target:
                    raise OccludedGoalMetricsError("LOOP_ALIAS cycle evidence is not a closed graph cycle")
            if not set(witness_alias_nodes).issubset(cycle_nodes):
                raise OccludedGoalMetricsError("LOOP_ALIAS aliases must occur on the registered cycle")
            if len(traversal) == len(encountered):
                raise OccludedGoalMetricsError("LOOP_ALIAS Phase-A traversal must include a genuine revisit")
        elif cycle_nodes or cycle_edges:
            raise OccludedGoalMetricsError("cycle witness fields are reserved for LOOP_ALIAS")
        normalized.append({**graph, "nodes": nodes, "edges": edges, "phase_a_traversal": traversal, "family_adequacy_witness": dict(witness)})
    if dict(role_counts) != C.SPLIT_EPISODE_COUNTS:
        raise OccludedGoalMetricsError("graph split totals drift")
    expected_family_role = {(family, role): C.SPLIT_FAMILY_EPISODE_COUNTS[role] for family in C.FAMILY_IDS for role in C.SPLIT_ROLE_IDS}
    if dict(family_role) != expected_family_role:
        raise OccludedGoalMetricsError("graph family/role counts drift")
    if procedural_seeds != set(C.PROCEDURAL_SEED_VALUES):
        raise OccludedGoalMetricsError("procedural seeds differ from the reserved 96-value range")
    disjointness = validate_identity_disjointness(root["identity_disjointness"], normalized)
    return {**root, "identity_disjointness": disjointness, "graphs": normalized}


def validate_query_rows(rows: Sequence[Mapping[str, Any]], graph_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = validate_graph_manifest(graph_manifest)
    graphs = {graph["graph_id"]: graph for graph in manifest["graphs"]}
    episodes = {graph["episode_id"]: graph for graph in manifest["graphs"]}
    if len(rows) != C.EPISODE_COUNT * C.QUERIES_PER_EPISODE:
        raise OccludedGoalMetricsError("query ledger must contain exactly 768 rows")
    normalized: list[dict[str, Any]] = []; identities: set[str] = set(); query_observation_ids: set[str] = set()
    episode_indices: dict[str, set[int]] = defaultdict(set)
    episode_depths: dict[str, set[int]] = defaultdict(set)
    for index, original in enumerate(rows):
        row = _require_mapping(original, fields=QUERY_FIELDS, label=f"query[{index}]")
        query_id = _string(row["query_id"], "query_id")
        episode_id = _string(row["episode_id"], "episode_id")
        graph_id = _string(row["graph_id"], "graph_id")
        if query_id in identities or episode_id not in episodes or graph_id not in graphs:
            raise OccludedGoalMetricsError("query identity or binding drift")
        identities.add(query_id)
        graph = graphs[graph_id]
        if graph["episode_id"] != episode_id or row["family"] != graph["family"] or row["role"] != graph["role"]:
            raise OccludedGoalMetricsError("query graph/family/role binding drift")
        query_index = row["query_index"]
        if isinstance(query_index, bool) or not isinstance(query_index, int) or not 0 <= query_index < C.QUERIES_PER_EPISODE:
            raise OccludedGoalMetricsError("query index drift")
        if query_index in episode_indices[episode_id]:
            raise OccludedGoalMetricsError("duplicate episode query index")
        episode_indices[episode_id].add(query_index)
        depth = row["depth_since_last_unambiguous_observation"]
        if isinstance(depth, bool) or not isinstance(depth, int) or not 4 <= depth <= 11:
            raise OccludedGoalMetricsError("depth since last unambiguous observation drift")
        if depth in episode_depths[episode_id]:
            raise OccludedGoalMetricsError("duplicate episode query depth")
        episode_depths[episode_id].add(depth)
        node_ids = tuple(node["node_id"] for node in graph["nodes"])
        node_by_id = {node["node_id"]: node for node in graph["nodes"]}
        keyframe_ids = {node["keyframe_observation_id"] for node in graph["nodes"]}
        phase_a_observation_ids = {visit["observation_id"] for visit in graph["phase_a_traversal"]}
        phase_a_final_timestamp = max(float(visit["timestamp_s"]) for visit in graph["phase_a_traversal"])
        action_labels = {edge["executed_action_label"] for edge in graph["edges"]}
        tables = _graph_tables(graph)
        candidates = _strings(row["candidate_node_ids"], "candidate_node_ids")
        if candidates != node_ids:
            raise OccludedGoalMetricsError("candidate node order differs from graph order")
        true_node = _string(row["true_node_id"], "true_node_id")
        if true_node not in node_ids or true_node == graph["goal_node_id"]:
            raise OccludedGoalMetricsError("invalid true query node")
        prior = _strings(row["prior_visited_node_ids"], "prior_visited_node_ids")
        if any(node not in node_ids for node in prior):
            raise OccludedGoalMetricsError("prior visited node escaped graph")
        unresolved = _strings(row["unresolved_condition_ids"], "unresolved_condition_ids")
        if any(condition not in C.CONDITION_IDS for condition in unresolved):
            raise OccludedGoalMetricsError("unknown unresolved condition")
        if not {"CURRENT_FRAME_NEAREST_NODE", "FIXED_WINDOW_SEQUENCE"}.issubset(unresolved):
            raise OccludedGoalMetricsError("current-frame and fixed-window conditions must remain alias-unresolved")
        if "FULL_BELIEF" in unresolved:
            raise OccludedGoalMetricsError("the complete registered history must resolve FULL_BELIEF at every query")
        if row["goal_visible"] is not False:
            raise OccludedGoalMetricsError("the goal must be occluded in every query")
        alias_nodes = _strings(row["alias_node_ids"], "alias_node_ids")
        distractors = _strings(row["alias_distractor_node_ids"], "alias_distractor_node_ids")
        if len(alias_nodes) != 2 or len(distractors) != 1 or set(alias_nodes) != {true_node, distractors[0]}:
            raise OccludedGoalMetricsError("registered alias evidence must be the true node plus one distractor")
        distractor = distractors[0]
        if distractor == true_node or distractor not in node_by_id or node_by_id[distractor]["alias_group_id"] != node_by_id[true_node]["alias_group_id"]:
            raise OccludedGoalMetricsError("alias distractor identity/group drift")
        if tables["best_port"][distractor] == tables["best_port"][true_node]:
            raise OccludedGoalMetricsError("alias distractor must require a different route port")
        query_observation_id = _string(row["query_observation_id"], "query_observation_id")
        if query_observation_id in query_observation_ids or query_observation_id in phase_a_observation_ids or query_observation_id in keyframe_ids:
            raise OccludedGoalMetricsError("query observation occurrence must be unique and Phase-A-disjoint")
        query_observation_ids.add(query_observation_id)
        query_time = _finite(row["query_timestamp_s"], "query_timestamp_s")
        if query_time <= phase_a_final_timestamp:
            raise OccludedGoalMetricsError("query capture must occur after Phase-A teacher traversal")
        query_template = _string(row["query_pixel_template_id"], "query_pixel_template_id")
        query_sha = _string(row["query_pixel_sha256"], "query_pixel_sha256")
        if len(query_sha) != 64 or any(char not in "0123456789abcdef" for char in query_sha):
            raise OccludedGoalMetricsError("query_pixel_sha256 must be lowercase SHA-256 hex")
        alias_evidence = [node_by_id[node] for node in alias_nodes]
        if any(node["pixel_template_id"] != query_template or node["pixel_sha256"] != query_sha for node in alias_evidence):
            raise OccludedGoalMetricsError("query capture must match the registered exact-alias pixel evidence")
        admissible = _strings(row["oracle_admissible_edge_ids"], "oracle_admissible_edge_ids")
        expected_admissible = tuple(
            edge["edge_id"] for edge in graph["edges"]
            if edge["source_node_id"] == true_node and edge["physically_executable"] and edge["oracle_admissible"]
        )
        if admissible != expected_admissible or len(admissible) < 2 or row["true_next_edge_id"] not in admissible:
            raise OccludedGoalMetricsError("query must bind all and at least two executable oracle-admissible outgoing edges")
        observation_history = _strings(row["history_observation_ids"], "history_observation_ids", unique=False)
        action_history = _strings(row["history_executed_action_labels"], "history_executed_action_labels", unique=False)
        if len(observation_history) != len(action_history) + 1 or len(observation_history) < depth + 1:
            raise OccludedGoalMetricsError("query history must be a complete prefix covering the alias depth")
        if any(item not in action_labels for item in action_history):
            raise OccludedGoalMetricsError("query action history escaped registered edge actions")
        if any(item in phase_a_observation_ids or item in keyframe_ids for item in observation_history):
            raise OccludedGoalMetricsError("Phase-C history occurrences must be disjoint from Phase-A references")
        if observation_history[-1] != query_observation_id:
            raise OccludedGoalMetricsError("query observation must terminate the complete history prefix")
        true_port = _string(row["true_next_port_label"], "true_next_port_label")
        true_edge = _string(row["true_next_edge_id"], "true_next_edge_id")
        if true_port != tables["best_port"][true_node] or true_edge != tables["best_edge"][true_node]:
            raise OccludedGoalMetricsError("query next edge/port differs from shortest-path authority")
        raw_macros = row["stage_b_choice_macros"]
        if not isinstance(raw_macros, Mapping) or set(raw_macros) != set(admissible):
            raise OccludedGoalMetricsError(
                "Stage-B choice macros must be keyed by every admissible query edge"
            )
        macros: dict[str, dict[str, Any]] = {}
        for choice_edge_id in admissible:
            macro = _require_mapping(
                raw_macros[choice_edge_id],
                fields=STAGE_B_MACRO_FIELDS,
                label=f"stage_b_choice_macros[{choice_edge_id}]",
            )
            outcome = _string(macro["outcome"], "Stage-B macro outcome")
            expected_outcome = (
                "CORRECT_ADVANCE" if choice_edge_id == true_edge else "WRONG_RETURN"
            )
            if outcome != expected_outcome:
                raise OccludedGoalMetricsError("Stage-B macro outcome disagrees with true edge")
            destination = _string(macro["destination_node_id"], "macro destination_node_id")
            macro_edges = _strings(macro["constituent_edge_ids"], "macro constituent_edge_ids")
            if not macro_edges or macro_edges[0] != choice_edge_id:
                raise OccludedGoalMetricsError("Stage-B macro must begin with its keyed choice edge")
            macros[choice_edge_id] = {
                "outcome": outcome,
                "destination_node_id": destination,
                "constituent_edge_ids": list(macro_edges),
            }
        raw_outcomes = row["stage_b_local_candidate_outcomes"]
        if not isinstance(raw_outcomes, list) or len(raw_outcomes) != len(C.STAGE_B_LOCAL_CANDIDATE_IDS):
            raise OccludedGoalMetricsError("Stage-B requires all twelve preregistered candidate outcomes")
        edge_by_id = {edge["edge_id"]: edge for edge in graph["edges"]}
        candidate_outcomes: list[dict[str, Any]] = []
        for candidate_index, original_outcome in enumerate(raw_outcomes):
            outcome = _require_mapping(
                original_outcome,
                fields=STAGE_B_CANDIDATE_OUTCOME_FIELDS,
                label=f"stage_b_local_candidate_outcomes[{candidate_index}]",
            )
            if outcome["candidate_index"] != candidate_index or isinstance(outcome["candidate_index"], bool):
                raise OccludedGoalMetricsError("Stage-B candidate indices must be canonical 0..11")
            if outcome["candidate_name"] != C.STAGE_B_LOCAL_CANDIDATE_IDS[candidate_index]:
                raise OccludedGoalMetricsError("Stage-B candidate name/index binding drift")
            contact = _bool(outcome["immediate_contact"], "candidate immediate_contact")
            viable = _bool(outcome["successor_viable"], "candidate successor_viable")
            stuck_outcome = _bool(outcome["stuck"], "candidate stuck")
            oracle = _bool(outcome["oracle_admissible"], "candidate oracle_admissible")
            distance = _finite(outcome["prefix_distance_m"], "candidate prefix_distance_m")
            if distance < 0.0:
                raise OccludedGoalMetricsError("candidate prefix distance must be nonnegative")
            actual_edge_id = outcome["actual_edge_id"]
            actual_port = outcome["actual_port_label"]
            endpoint = _string(outcome["endpoint_node_id"], "candidate endpoint_node_id")
            edge = None
            if actual_edge_id is not None:
                actual_edge_id = _string(actual_edge_id, "candidate actual_edge_id")
                edge = edge_by_id.get(actual_edge_id)
                if edge is None or edge["source_node_id"] != true_node:
                    raise OccludedGoalMetricsError("candidate actual edge is not outgoing from query")
                if actual_port != edge["port_label"] or endpoint != edge["target_node_id"]:
                    raise OccludedGoalMetricsError("candidate edge/port/endpoint binding drift")
                if not math.isclose(distance, float(edge["edge_cost"]), rel_tol=0.0, abs_tol=1e-12):
                    raise OccludedGoalMetricsError("candidate prefix distance differs from edge cost")
            elif actual_port is not None or endpoint != true_node or distance != 0.0:
                raise OccludedGoalMetricsError("candidate without an edge must remain at the query")
            expected_oracle = bool(
                edge is not None
                and edge["physically_executable"]
                and edge["oracle_admissible"]
                and not contact
                and viable
                and not stuck_outcome
            )
            if oracle != expected_oracle:
                raise OccludedGoalMetricsError("candidate oracle admissibility is not geometry-derived")
            if oracle and actual_edge_id not in macros:
                raise OccludedGoalMetricsError("admissible candidate lacks a preregistered choice macro")
            candidate_outcomes.append({
                **outcome,
                "actual_edge_id": actual_edge_id,
                "prefix_distance_m": distance,
            })
        if not any(outcome["oracle_admissible"] for outcome in candidate_outcomes):
            raise OccludedGoalMetricsError("query has no oracle-admissible local candidate")
        normalized.append({
            **row,
            "prior_visited_node_ids": list(prior),
            "unresolved_condition_ids": list(unresolved),
            "candidate_node_ids": list(candidates),
            "alias_node_ids": list(alias_nodes),
            "alias_distractor_node_ids": list(distractors),
            "oracle_admissible_edge_ids": list(admissible),
            "history_observation_ids": list(observation_history),
            "history_executed_action_labels": list(action_history),
            "stage_b_choice_macros": macros,
            "stage_b_local_candidate_outcomes": candidate_outcomes,
        })
    required_indices = set(range(C.QUERIES_PER_EPISODE))
    required_depths = set(range(4, 12))
    if set(episode_indices) != set(episodes) or any(indices != required_indices for indices in episode_indices.values()):
        raise OccludedGoalMetricsError("every episode requires query indices 0..7")
    if set(episode_depths) != set(episodes) or any(depths != required_depths for depths in episode_depths.values()):
        raise OccludedGoalMetricsError("every episode requires alias depths 4..11")
    for episode_id in episodes:
        ordered = sorted((row for row in normalized if row["episode_id"] == episode_id), key=lambda row: row["query_index"])
        graph = episodes[episode_id]
        if graph["stage_b_decision_start_node_id"] != ordered[0]["true_node_id"]:
            raise OccludedGoalMetricsError("Stage-B prelude must terminate at query zero")
        if any(float(later["query_timestamp_s"]) <= float(earlier["query_timestamp_s"]) for earlier, later in zip(ordered, ordered[1:])):
            raise OccludedGoalMetricsError("query capture timestamps must increase with query_index")
        for earlier, later in zip(ordered, ordered[1:]):
            earlier_obs = earlier["history_observation_ids"]
            earlier_actions = earlier["history_executed_action_labels"]
            if later["history_observation_ids"][:len(earlier_obs)] != earlier_obs or later["history_executed_action_labels"][:len(earlier_actions)] != earlier_actions:
                raise OccludedGoalMetricsError("query histories must be monotonically extending full prefixes")
        for query_index, query in enumerate(ordered):
            correct_destination = (
                graph["goal_node_id"]
                if query_index == C.QUERIES_PER_EPISODE - 1
                else ordered[query_index + 1]["true_node_id"]
            )
            for choice_edge_id, macro in query["stage_b_choice_macros"].items():
                expected_destination = (
                    correct_destination
                    if choice_edge_id == query["true_next_edge_id"]
                    else query["true_node_id"]
                )
                expected_path = _macro_path_after_choice(
                    graph, choice_edge_id, expected_destination
                )
                if (
                    macro["destination_node_id"] != expected_destination
                    or tuple(macro["constituent_edge_ids"]) != expected_path
                ):
                    raise OccludedGoalMetricsError(
                        "Stage-B macro differs from its deterministic registered graph path"
                    )
    return normalized


def validate_belief_rows(rows: Sequence[Mapping[str, Any]], query_rows: Sequence[Mapping[str, Any]], graph_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifest = validate_graph_manifest(graph_manifest)
    queries = validate_query_rows(query_rows, manifest)
    graphs = {graph["graph_id"]: graph for graph in manifest["graphs"]}
    query_by_id = {row["query_id"]: row for row in queries if row["role"] == "DEVELOPMENT_HELDOUT"}
    expected = {(query_id, condition) for query_id in query_by_id for condition in C.CONDITION_IDS}
    if len(rows) != len(expected):
        raise OccludedGoalMetricsError("belief row count drift")
    observed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, original in enumerate(rows):
        row = _require_mapping(original, fields=BELIEF_FIELDS, label=f"belief[{index}]")
        query_id = _string(row["query_id"], "query_id")
        condition = _string(row["condition_id"], "condition_id")
        identity = (query_id, condition)
        if identity not in expected or identity in observed:
            raise OccludedGoalMetricsError("belief identity coverage drift")
        nodes = _strings(row["node_ids"], "node_ids")
        if nodes != tuple(query_by_id[query_id]["candidate_node_ids"]):
            raise OccludedGoalMetricsError("belief node order drift")
        vectors: dict[str, list[float]] = {}
        for field in (
            "observation_similarities", "observation_likelihoods",
            "transition_prior_probabilities", "preprojection_probabilities",
            "posterior_probabilities",
        ):
            raw = row[field]
            if not isinstance(raw, (list, tuple)) or len(raw) != len(nodes):
                raise OccludedGoalMetricsError(f"{field} vector shape drift")
            vector = [_finite(item, field) for item in raw]
            if field == "observation_similarities" and any(item < -1.0 or item > 1.0 for item in vector):
                raise OccludedGoalMetricsError("cosine observation similarities must be in [-1,1]")
            if field != "observation_similarities":
                if any(item < 0.0 or item > 1.0 for item in vector) or not math.isclose(sum(vector), 1.0, rel_tol=0.0, abs_tol=1e-8):
                    raise OccludedGoalMetricsError(f"{field} must be a normalized probability vector")
            vectors[field] = vector
        prior = vectors["transition_prior_probabilities"]
        likelihood = vectors["observation_likelihoods"]
        if condition == "CURRENT_FRAME_NEAREST_NODE" and any(
            not math.isclose(value, 1.0 / len(nodes), rel_tol=0.0, abs_tol=1e-12)
            for value in prior
        ):
            raise OccludedGoalMetricsError("CURRENT_FRAME_NEAREST_NODE must use a fresh uniform prior")
        product = [a * b for a, b in zip(prior, likelihood)]
        normalizer = sum(product)
        if normalizer <= 0.0:
            raise OccludedGoalMetricsError("transition/observation product has zero mass")
        expected_pre = [value / normalizer for value in product]
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8) for a, b in zip(vectors["preprojection_probabilities"], expected_pre)):
            raise OccludedGoalMetricsError("preprojection posterior differs from prior-times-likelihood")
        pre = vectors["preprojection_probabilities"]
        posterior = vectors["posterior_probabilities"]
        pre_order = _ranking(nodes, pre)
        if condition in {"CURRENT_FRAME_NEAREST_NODE", "FIXED_WINDOW_SEQUENCE", "MAP_FILTER"}:
            expected_posterior = [1.0 if index == pre_order[0] else 0.0 for index in range(len(nodes))]
        elif condition == "TOP_K_BELIEF":
            keep = set(pre_order[: min(C.TOP_K, len(nodes))]); total = sum(pre[index] for index in keep)
            expected_posterior = [pre[index] / total if index in keep else 0.0 for index in range(len(nodes))]
        elif condition == "ORACLE_PLACE_IDENTITY":
            true_index = nodes.index(query_by_id[query_id]["true_node_id"])
            expected_posterior = [1.0 if index == true_index else 0.0 for index in range(len(nodes))]
        else:
            expected_posterior = pre
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8) for a, b in zip(posterior, expected_posterior)):
            raise OccludedGoalMetricsError(f"{condition} posterior projection drift")
        selected = _string(row["selected_node_id"], "selected_node_id")
        posterior_order = _ranking(nodes, posterior)
        if selected != nodes[posterior_order[0]]:
            raise OccludedGoalMetricsError("selected node differs from deterministic posterior MAP")
        graph = graphs[query_by_id[query_id]["graph_id"]]
        selected_port, selected_edge = _route_choice(nodes, posterior, _graph_tables(graph))
        if row["selected_port_label"] != selected_port or row["selected_edge_id"] != selected_edge:
            raise OccludedGoalMetricsError("selected edge/port differs from posterior route-mass authority")
        entropy = _finite(row["normalized_entropy"], "normalized_entropy")
        if not math.isclose(entropy, _entropy(posterior), rel_tol=0.0, abs_tol=1e-10):
            raise OccludedGoalMetricsError("persisted entropy differs from posterior")
        _bool(row["abstained"], "abstained")
        mapping = row["action_history_position_mapping"]
        action_count = len(query_by_id[query_id]["history_executed_action_labels"])
        if not isinstance(mapping, (list, tuple)) or len(mapping) != action_count or any(isinstance(item, bool) or not isinstance(item, int) for item in mapping):
            raise OccludedGoalMetricsError("action history position mapping shape/type drift")
        expected_mapping = list(range(action_count))
        if condition == "SHUFFLED_ACTION_HISTORY":
            expected_mapping = list(deterministic_action_history_position_mapping(query_id, action_count))
        if list(mapping) != expected_mapping:
            raise OccludedGoalMetricsError("action history position mapping differs from frozen deterministic authority")
        if condition == "NO_OBSERVATION_LIKELIHOOD" and any(not math.isclose(value, 1.0 / len(nodes), rel_tol=0.0, abs_tol=1e-10) for value in likelihood):
            raise OccludedGoalMetricsError("NO_OBSERVATION_LIKELIHOOD must persist a uniform likelihood")
        observed[identity] = {**row, "node_ids": list(nodes), "action_history_position_mapping": list(mapping), **vectors}
    if set(observed) != expected:
        raise OccludedGoalMetricsError("belief identity coverage is incomplete")
    for query_id in query_by_id:
        ordinary = observed[(query_id, "FULL_BELIEF")]
        for condition in C.CONDITION_IDS:
            candidate = observed[(query_id, condition)]
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(candidate["observation_similarities"], ordinary["observation_similarities"])):
                raise OccludedGoalMetricsError("observation similarities must be condition-invariant")
            if condition != "NO_OBSERVATION_LIKELIHOOD" and any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(candidate["observation_likelihoods"], ordinary["observation_likelihoods"])):
                raise OccludedGoalMetricsError("observation likelihoods must be condition-invariant outside the ablation")
    return [
        observed[(row["query_id"], condition)]
        for row in queries if row["role"] == "DEVELOPMENT_HELDOUT"
        for condition in C.CONDITION_IDS
    ]


def validate_calibration(
    value: Mapping[str, Any],
    graph_manifest: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    row = _require_mapping(value, fields=CALIBRATION_FIELDS, label="calibration")
    _validate_digest(row, "calibration")
    if row["schema"] != "occluded_goal_topological_belief_v1.calibration.v1" or row["experiment_id"] != C.EXPERIMENT_ID or row["selection_role"] != "CALIBRATION":
        raise OccludedGoalMetricsError("calibration identity drift")
    manifest = validate_graph_manifest(graph_manifest)
    queries = validate_query_rows(query_rows, manifest)
    expected = tuple(sorted(graph["episode_id"] for graph in manifest["graphs"] if graph["role"] == "CALIBRATION"))
    if tuple(sorted(_strings(row["calibration_episode_ids"], "calibration_episode_ids"))) != expected:
        raise OccludedGoalMetricsError("calibration episode binding drift")
    calibration_query_ids = tuple(sorted(query["query_id"] for query in queries if query["role"] == "CALIBRATION"))
    if len(calibration_query_ids) != C.HELDOUT_QUERY_COUNT:
        raise OccludedGoalMetricsError("calibration requires exactly 128 registered queries")
    parameters = _require_mapping(row["selected_parameters"], fields=CALIBRATION_PARAMETER_FIELDS, label="selected_parameters")
    expected_parameters: list[dict[str, float]] = []
    for temperature in C.CALIBRATION_GRID["observation_softmax_temperature"]:
        for compatible in C.CALIBRATION_GRID["action_compatible_edge_probability"]:
            for noise in C.CALIBRATION_GRID["transition_noise_probability"]:
                for entropy in C.CALIBRATION_GRID["normalized_entropy_abstention_threshold"]:
                    expected_parameters.append({
                        "observation_softmax_temperature": temperature,
                        "action_compatible_edge_probability": compatible,
                        "transition_noise_probability": noise,
                        "normalized_entropy_abstention_threshold": entropy,
                    })
    grid_results = row["grid_results"]
    if not isinstance(grid_results, list) or len(grid_results) != len(expected_parameters):
        raise OccludedGoalMetricsError("calibration must persist all 144 grid results")
    query_results = row["grid_query_results"]
    expected_query_result_count = len(expected_parameters) * len(calibration_query_ids)
    if not isinstance(query_results, list) or len(query_results) != expected_query_result_count:
        raise OccludedGoalMetricsError("calibration must persist the complete 144x128 raw outcome cube")
    normalized_query_results: list[dict[str, Any]] = []
    outcome_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for offset, original in enumerate(query_results):
        outcome = _require_mapping(original, fields=CALIBRATION_QUERY_RESULT_FIELDS, label=f"grid_query_results[{offset}]")
        expected_grid_index = offset // len(calibration_query_ids)
        expected_query_id = calibration_query_ids[offset % len(calibration_query_ids)]
        grid_index = outcome["grid_index"]
        if isinstance(grid_index, bool) or not isinstance(grid_index, int) or grid_index != expected_grid_index or outcome["query_id"] != expected_query_id:
            raise OccludedGoalMetricsError("calibration raw cube must use grid-major/query-lexicographic canonical order")
        normalized_outcome = {
            "grid_index": grid_index,
            "query_id": expected_query_id,
            "edge_correct": _bool(outcome["edge_correct"], "edge_correct"),
            "localisation_top3": _bool(outcome["localisation_top3"], "localisation_top3"),
            "normalized_regret": _finite(outcome["normalized_regret"], "normalized_regret"),
            "false_confident": _bool(outcome["false_confident"], "false_confident"),
            "abstained": _bool(outcome["abstained"], "abstained"),
        }
        if not 0.0 <= normalized_outcome["normalized_regret"] <= 1.0:
            raise OccludedGoalMetricsError("calibration per-query normalized_regret must be in [0,1]")
        normalized_query_results.append(normalized_outcome)
        outcome_groups[grid_index].append(normalized_outcome)
    normalized_grid: list[dict[str, Any]] = []
    for index, original in enumerate(grid_results):
        result = _require_mapping(original, fields=CALIBRATION_RESULT_FIELDS, label=f"grid_results[{index}]")
        if result["grid_index"] != index or isinstance(result["grid_index"], bool):
            raise OccludedGoalMetricsError("calibration grid order/index drift")
        candidate = _require_mapping(result["parameters"], fields=CALIBRATION_PARAMETER_FIELDS, label="grid parameters")
        candidate = {key: _finite(candidate[key], key) for key in C.CALIBRATION_GRID_ORDER}
        if candidate != expected_parameters[index]:
            raise OccludedGoalMetricsError("calibration grid tuple differs from exact nested order")
        metrics = {}
        for field in CALIBRATION_RESULT_FIELDS - {"grid_index", "parameters"}:
            metric = _finite(result[field], field)
            if not 0.0 <= metric <= 1.0:
                raise OccludedGoalMetricsError(f"calibration metric must be in [0,1]: {field}")
            metrics[field] = metric
        outcomes = outcome_groups[index]
        regenerated = {
            "correct_next_edge_accuracy": mean(float(item["edge_correct"]) for item in outcomes),
            "localisation_top3": mean(float(item["localisation_top3"]) for item in outcomes),
            "normalized_graph_distance_regret": mean(item["normalized_regret"] for item in outcomes),
            "false_confident_localisation_rate": mean(float(item["false_confident"]) for item in outcomes),
            "abstention_rate": mean(float(item["abstained"]) for item in outcomes),
        }
        if any(not math.isclose(metrics[field], regenerated[field], rel_tol=0.0, abs_tol=1e-12) for field in regenerated):
            raise OccludedGoalMetricsError("calibration aggregate differs from raw per-grid query outcomes")
        normalized_grid.append({"grid_index": index, "parameters": candidate, **metrics})
    winning = min(
        normalized_grid,
        key=lambda result: (
            -result["correct_next_edge_accuracy"],
            -result["localisation_top3"],
            result["normalized_graph_distance_regret"],
            result["false_confident_localisation_rate"],
            result["abstention_rate"],
            result["grid_index"],
        ),
    )
    parameters = {key: _finite(parameters[key], key) for key in C.CALIBRATION_GRID_ORDER}
    grid_index = row["selected_grid_index"]
    if isinstance(grid_index, bool) or not isinstance(grid_index, int) or grid_index != winning["grid_index"]:
        raise OccludedGoalMetricsError("selected_grid_index differs from exact lexicographic winner")
    if parameters != winning["parameters"]:
        raise OccludedGoalMetricsError("selected parameters differ from winning grid tuple")
    return {
        **row,
        "selected_parameters": parameters,
        "grid_results": normalized_grid,
        "grid_query_results": normalized_query_results,
    }


def _graph_tables(graph: Mapping[str, Any]) -> dict[str, Any]:
    node_ids = tuple(node["node_id"] for node in graph["nodes"])
    alias = {node["node_id"]: node["alias_group_id"] for node in graph["nodes"]}
    distances = _reverse_distances(node_ids, graph["edges"], graph["goal_node_id"])
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in graph["edges"]:
        outgoing[edge["source_node_id"]].append(dict(edge))
    best_port: dict[str, str] = {}; best_edge: dict[str, str | None] = {}; port_distance: dict[str, dict[str, float]] = {}
    for node_id in node_ids:
        if node_id == graph["goal_node_id"]:
            best_port[node_id] = C.PORT_LABEL_ORDER[0]; best_edge[node_id] = None; port_distance[node_id] = {}
            continue
        rows = sorted(outgoing[node_id], key=lambda edge: (edge["edge_cost"] + distances[edge["target_node_id"]], edge["edge_id"]))
        best_port[node_id] = rows[0]["port_label"]
        best_edge[node_id] = rows[0]["edge_id"]
        port_distance[node_id] = {edge["port_label"]: edge["edge_cost"] + distances[edge["target_node_id"]] for edge in rows}
    return {"node_ids": node_ids, "alias": alias, "distance": distances, "best_port": best_port, "best_edge": best_edge, "port_distance": port_distance}


def _route_choice(
    nodes: Sequence[str], probabilities: Sequence[float], tables: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    """Return the posterior route-mass winner and its deterministic edge witness.

    Goal-node mass has no outgoing route vote.  The edge witness is the
    highest-posterior hypothesis voting for the winning port, with node-id
    tie-breaking.  If all posterior mass is on terminal nodes, both are null.
    """
    route_mass = {port: 0.0 for port in C.PORT_LABEL_ORDER}
    for node, probability in zip(nodes, probabilities):
        if tables["best_edge"][node] is not None:
            route_mass[tables["best_port"][node]] += float(probability)
    if max(route_mass.values(), default=0.0) <= 0.0:
        return None, None
    selected_port = min(
        C.PORT_LABEL_ORDER,
        key=lambda port: (-route_mass[port], C.PORT_LABEL_ORDER.index(port)),
    )
    witnesses = [
        (float(probabilities[index]), nodes[index])
        for index in range(len(nodes))
        if tables["best_edge"][nodes[index]] is not None
        and tables["best_port"][nodes[index]] == selected_port
    ]
    witness_node = min(witnesses, key=lambda item: (-item[0], item[1]))[1]
    return selected_port, tables["best_edge"][witness_node]


def _ranking(nodes: Sequence[str], probabilities: Sequence[float]) -> list[int]:
    return sorted(range(len(nodes)), key=lambda index: (-probabilities[index], nodes[index]))


def _entropy(probabilities: Sequence[float]) -> float:
    if len(probabilities) <= 1:
        return 0.0
    raw = -sum(value * math.log(value) for value in probabilities if value > 0.0)
    return raw / math.log(len(probabilities))


def _softmax(values: Sequence[float], temperature: float) -> list[float]:
    scaled = [float(value) / temperature for value in values]
    maximum = max(scaled)
    weights = [math.exp(value - maximum) for value in scaled]
    total = sum(weights)
    return [value / total for value in weights]


def _query_detail(query: Mapping[str, Any], belief: Mapping[str, Any], graph: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    tables = _graph_tables(graph)
    nodes = tuple(belief["node_ids"]); probabilities = tuple(float(item) for item in belief["posterior_probabilities"])
    order = _ranking(nodes, probabilities); true = query["true_node_id"]; true_index = nodes.index(true)
    rank = order.index(true_index) + 1; selected_node = nodes[order[0]]; entropy = _entropy(probabilities)
    abstained = entropy > threshold
    if belief["abstained"] is not abstained:
        raise OccludedGoalMetricsError("persisted abstention differs from calibrated entropy threshold")
    confidence = probabilities[order[0]]
    route_mass = {port: 0.0 for port in C.PORT_LABEL_ORDER}
    for node, probability in zip(nodes, probabilities):
        if tables["best_edge"][node] is not None:
            route_mass[tables["best_port"][node]] += probability
    route_order = sorted(C.PORT_LABEL_ORDER, key=lambda port: (-route_mass[port], C.PORT_LABEL_ORDER.index(port)))
    selected_port, selected_edge = _route_choice(nodes, probabilities, tables)
    if belief["selected_node_id"] != selected_node or belief["selected_port_label"] != selected_port or belief["selected_edge_id"] != selected_edge:
        raise OccludedGoalMetricsError("persisted belief selections differ from reducer authority")
    true_port = query["true_next_port_label"]
    if tables["best_port"][true] != true_port:
        raise OccludedGoalMetricsError("query true next port differs from graph shortest-path authority")
    available = tables["port_distance"][true]
    if abstained or selected_port not in available:
        regret = 1.0
    else:
        best = min(available.values()); worst = max(available.values())
        regret = 0.0 if worst == best else (available[selected_port] - best) / (worst - best)
        regret = min(1.0, max(0.0, regret))
    correct_edge = (not abstained and selected_port == true_port)
    wrong_turn = (not abstained and selected_port != true_port)
    unresolved = belief["condition_id"] in query["unresolved_condition_ids"]
    wrong_selected = selected_node != true
    return {
        "query_id": query["query_id"], "episode_id": query["episode_id"], "family": query["family"],
        "query_index": query["query_index"],
        "depth_since_last_unambiguous_observation": query["depth_since_last_unambiguous_observation"],
        "top1": float(not wrong_selected),
        "top3": float(true_index in order[: min(3, len(order))]), "reciprocal_rank": 1.0 / rank,
        "nll": -math.log(max(probabilities[true_index], C.PROBABILITY_FLOOR)),
        "brier": sum((probability - (1.0 if index == true_index else 0.0)) ** 2 for index, probability in enumerate(probabilities)),
        "confidence": confidence, "confidence_correct": float(not wrong_selected), "entropy": entropy,
        "false_confident": float(wrong_selected and not abstained),
        "false_place_merge": float(wrong_selected and not abstained and tables["alias"][selected_node] == tables["alias"][true]),
        "false_loop_closure": float(wrong_selected and not abstained and selected_node in query["prior_visited_node_ids"]),
        "correct_edge": float(correct_edge), "regret": regret,
        "route_top3": float(true_port in route_order[: min(3, len(route_order))]),
        "wrong_turn": float(wrong_turn), "abstained": float(abstained),
        "unresolved": float(unresolved), "correct_unresolved_abstention": float(unresolved and abstained),
        "selected_node_id": None if abstained else selected_node,
        "selected_port_label": None if abstained else selected_port,
    }


def _ece(details: Sequence[Mapping[str, Any]]) -> float:
    total = len(details); result = 0.0
    for bin_index in range(C.ECE_BIN_COUNT):
        lower = bin_index / C.ECE_BIN_COUNT; upper = (bin_index + 1) / C.ECE_BIN_COUNT
        members = [row for row in details if row["confidence"] >= lower and (row["confidence"] < upper or (bin_index == C.ECE_BIN_COUNT - 1 and row["confidence"] <= upper))]
        if members:
            result += len(members) / total * abs(mean(row["confidence"] for row in members) - mean(row["confidence_correct"] for row in members))
    return result


def _aggregate(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not details:
        raise OccludedGoalMetricsError("cannot aggregate an empty query group")
    unresolved = sum(row["unresolved"] for row in details)
    return {
        "query_count": len(details),
        "localisation_top1": mean(row["top1"] for row in details),
        "localisation_top3": mean(row["top3"] for row in details),
        "mean_reciprocal_rank": mean(row["reciprocal_rank"] for row in details),
        "localisation_nll": mean(row["nll"] for row in details),
        "multiclass_brier": mean(row["brier"] for row in details),
        "ece_10_bin": _ece(details),
        "normalized_belief_entropy": mean(row["entropy"] for row in details),
        "false_confident_localisation_rate": mean(row["false_confident"] for row in details),
        "false_place_merge_rate": mean(row["false_place_merge"] for row in details),
        "false_loop_closure_rate": mean(row["false_loop_closure"] for row in details),
        "correct_next_edge_accuracy": mean(row["correct_edge"] for row in details),
        "normalized_graph_distance_regret": mean(row["regret"] for row in details),
        "route_top3": mean(row["route_top3"] for row in details),
        "wrong_turn_rate": mean(row["wrong_turn"] for row in details),
        "abstention_rate": mean(row["abstained"] for row in details),
        "unresolved_query_count": int(unresolved),
        "correct_abstention_when_unresolved": None if unresolved == 0 else sum(row["correct_unresolved_abstention"] for row in details) / unresolved,
    }


def _condition_metrics(details: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families = {family: _aggregate([row for row in details if row["family"] == family]) for family in C.FAMILY_IDS}
    episodes = {episode: _aggregate([row for row in details if row["episode_id"] == episode]) for episode in sorted({row["episode_id"] for row in details})}
    depths = {
        str(depth): _aggregate([
            row for row in details
            if row["depth_since_last_unambiguous_observation"] == depth
        ])
        for depth in range(4, 12)
    }
    return {
        "aggregate": _aggregate(details),
        "families": families,
        "episodes": episodes,
        "depth_since_last_unambiguous_observation": depths,
    }


def _absolute(metrics: Mapping[str, Any]) -> dict[str, Any]:
    aggregate = metrics["aggregate"]
    checks = {
        "localisation_top3": aggregate["localisation_top3"] >= C.ABSOLUTE_GATE["localisation_top3_minimum"],
        "correct_next_edge_accuracy": aggregate["correct_next_edge_accuracy"] >= C.ABSOLUTE_GATE["correct_next_edge_accuracy_minimum"],
        "normalized_graph_distance_regret": aggregate["normalized_graph_distance_regret"] <= C.ABSOLUTE_GATE["normalized_graph_distance_regret_maximum"],
        "false_confident_localisation_rate": aggregate["false_confident_localisation_rate"] <= C.ABSOLUTE_GATE["false_confident_localisation_rate_maximum"],
        "family_correct_next_edge_accuracy": all(row["correct_next_edge_accuracy"] >= C.ABSOLUTE_GATE["per_family_correct_next_edge_accuracy_minimum"] for row in metrics["families"].values()),
    }
    return {"checks": checks, "pass": all(checks.values())}


def _current_margin(candidate: Mapping[str, Any], current: Mapping[str, Any]) -> dict[str, Any]:
    ca, cu = candidate["aggregate"], current["aggregate"]
    values = {
        "correct_next_edge_accuracy_gain": ca["correct_next_edge_accuracy"] - cu["correct_next_edge_accuracy"],
        "localisation_top3_gain": ca["localisation_top3"] - cu["localisation_top3"],
        "normalized_graph_distance_regret_reduction": cu["normalized_graph_distance_regret"] - ca["normalized_graph_distance_regret"],
    }
    checks = {
        "edge": values["correct_next_edge_accuracy_gain"] >= 0.20,
        "top3_or_regret": values["localisation_top3_gain"] >= 0.20 or values["normalized_graph_distance_regret_reduction"] >= 0.15,
    }
    return {"values": values, "checks": checks, "pass": all(checks.values())}


def _full_over_map(full: Mapping[str, Any], map_metrics: Mapping[str, Any]) -> dict[str, Any]:
    fa, ma = full["aggregate"], map_metrics["aggregate"]
    values = {
        "correct_next_edge_accuracy_gain": fa["correct_next_edge_accuracy"] - ma["correct_next_edge_accuracy"],
        "localisation_top3_gain": fa["localisation_top3"] - ma["localisation_top3"],
        "false_confident_localisation_rate_reduction": ma["false_confident_localisation_rate"] - fa["false_confident_localisation_rate"],
        "normalized_graph_distance_regret_reduction": ma["normalized_graph_distance_regret"] - fa["normalized_graph_distance_regret"],
    }
    checks = {"edge": values["correct_next_edge_accuracy_gain"] >= 0.08, "top3": values["localisation_top3_gain"] >= 0.10, "false_confident": values["false_confident_localisation_rate_reduction"] >= 0.05, "regret": values["normalized_graph_distance_regret_reduction"] >= 0.05}
    return {"values": values, "checks": checks, "pass": any(checks.values())}


def _short_history_matches(fixed: Mapping[str, Any], reference: Mapping[str, Any]) -> bool:
    gate = C.SHORT_HISTORY_MATCH_GATE
    return (
        fixed["correct_next_edge_accuracy"]
        >= reference["correct_next_edge_accuracy"]
        - gate["correct_next_edge_accuracy_maximum_deficit"]
        and fixed["localisation_top3"]
        >= reference["localisation_top3"]
        - gate["localisation_top3_maximum_deficit"]
        and fixed["normalized_graph_distance_regret"]
        <= reference["normalized_graph_distance_regret"]
        + gate["normalized_graph_distance_regret_maximum_excess"]
    )


def recompute_stage_a_metrics(graph_manifest: Mapping[str, Any], query_rows: Sequence[Mapping[str, Any]], belief_rows: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]) -> dict[str, Any]:
    manifest = validate_graph_manifest(graph_manifest)
    queries = validate_query_rows(query_rows, manifest)
    beliefs = validate_belief_rows(belief_rows, queries, manifest)
    calibrated = validate_calibration(calibration, manifest, queries)
    graphs = {graph["graph_id"]: graph for graph in manifest["graphs"]}
    query_by_id = {row["query_id"]: row for row in queries}
    threshold = calibrated["selected_parameters"]["normalized_entropy_abstention_threshold"]
    temperature = calibrated["selected_parameters"]["observation_softmax_temperature"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for belief in beliefs:
        if belief["condition_id"] == "NO_OBSERVATION_LIKELIHOOD":
            expected_likelihood = [1.0 / len(belief["node_ids"])] * len(belief["node_ids"])
        else:
            expected_likelihood = _softmax(belief["observation_similarities"], temperature)
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-10) for a, b in zip(belief["observation_likelihoods"], expected_likelihood)):
            raise OccludedGoalMetricsError("observation likelihood differs from frozen similarity-softmax formula")
        query = query_by_id[belief["query_id"]]
        grouped[belief["condition_id"]].append(_query_detail(query, belief, graphs[query["graph_id"]], threshold))
    conditions = {condition: _condition_metrics(grouped[condition]) for condition in C.CONDITION_IDS}
    absolute = {condition: _absolute(conditions[condition]) for condition in C.PRIMARY_CONDITION_IDS}
    current = conditions["CURRENT_FRAME_NEAREST_NODE"]
    current_margins = {condition: _current_margin(conditions[condition], current) for condition in ("FIXED_WINDOW_SEQUENCE", "MAP_FILTER", "TOP_K_BELIEF", "FULL_BELIEF")}
    full_map = _full_over_map(conditions["FULL_BELIEF"], conditions["MAP_FILTER"])
    persistent_candidates = [
        condition for condition in ("MAP_FILTER", "TOP_K_BELIEF", "FULL_BELIEF")
        if absolute[condition]["pass"]
    ]
    strongest = None
    if persistent_candidates:
        strongest = max(persistent_candidates, key=lambda condition: (conditions[condition]["aggregate"]["correct_next_edge_accuracy"], conditions[condition]["aggregate"]["localisation_top3"], -conditions[condition]["aggregate"]["normalized_graph_distance_regret"], -C.CONDITION_IDS.index(condition)))
    fixed = conditions["FIXED_WINDOW_SEQUENCE"]["aggregate"]
    short_matches = False
    if strongest is not None:
        reference = conditions[strongest]["aggregate"]
        short_matches = _short_history_matches(fixed, reference)
    short_signal = strongest is not None and short_matches
    map_signal = absolute["MAP_FILTER"]["pass"] and not full_map["pass"]
    full_signal = absolute["FULL_BELIEF"]["pass"] and current_margins["FULL_BELIEF"]["pass"] and full_map["pass"]
    persistent_gate_unresolved = bool(persistent_candidates) and not short_signal and not map_signal and not full_signal
    if short_signal:
        classification = "SHORT_HISTORY_SUFFICIENT"; strongest_memory = "FIXED_WINDOW_SEQUENCE"
    elif map_signal:
        classification = "TOPOLOGICAL_MAP_SUFFICIENT"; strongest_memory = "MAP_FILTER"
    elif full_signal:
        classification = "TOPOLOGICAL_FULL_BELIEF_SIGNAL"; strongest_memory = "FULL_BELIEF"
    elif persistent_gate_unresolved:
        classification = "TOPOLOGICAL_PERSISTENT_GATE_UNRESOLVED"; strongest_memory = strongest
    else:
        classification = "VJEPA_PLACE_BELIEF_NO_SIGNAL"; strongest_memory = strongest
    stage_b_authorized = classification in {"TOPOLOGICAL_MAP_SUFFICIENT", "TOPOLOGICAL_FULL_BELIEF_SIGNAL"}
    return C.attach_content_digest({
        "schema": "occluded_goal_topological_belief_v1.stage_a_metrics.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "conditions": conditions,
        "gate": {"absolute": absolute, "incremental_over_current": current_margins, "full_incremental_over_map": full_map, "fixed_window_matches_strongest_persistent": short_matches, "strongest_gate_passing_persistent_condition": strongest},
        "calibration_binding": {"content_digest": calibrated["content_digest"], "selected_grid_index": calibrated["selected_grid_index"], "selected_parameters": copy.deepcopy(calibrated["selected_parameters"])},
        "decision": {"primary_classification": classification, "strongest_memory_condition": strongest_memory, "stage_b_authorized": stage_b_authorized, "next_experiment": C.NEXT_DECISION_BY_CLASSIFICATION[classification]},
    })


def stage_a_authorizes_stage_b(stage_a_metrics: Mapping[str, Any]) -> bool:
    _validate_digest(stage_a_metrics, "stage_a_metrics")
    if stage_a_metrics.get("schema") != "occluded_goal_topological_belief_v1.stage_a_metrics.v1":
        raise OccludedGoalMetricsError("Stage-A metrics schema drift")
    decision = stage_a_metrics.get("decision")
    if not isinstance(decision, Mapping) or type(decision.get("stage_b_authorized")) is not bool:
        raise OccludedGoalMetricsError("Stage-A decision authority drift")
    return bool(decision["stage_b_authorized"])


def _stage_b_transition(
    probabilities: Sequence[float],
    action: str,
    node_ids: Sequence[str],
    graph: Mapping[str, Any],
    compatibility: float,
    noise: float,
) -> list[float]:
    if action == "NO_OP":
        return [float(value) for value in probabilities]
    node_index = {node: index for index, node in enumerate(node_ids)}
    outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in graph["edges"]:
        outgoing[str(edge["source_node_id"])].append(edge)
    predicted = [0.0] * len(node_ids)
    for source_index, source in enumerate(node_ids):
        mass = float(probabilities[source_index]); edges = outgoing.get(source, [])
        if not edges:
            predicted[source_index] += mass
            continue
        matching = [edge for edge in edges if edge["executed_action_label"] == action]
        incompatible = [edge for edge in edges if edge not in matching]
        if not matching:
            matching_mass, incompatible_mass = 0.0, 1.0
        elif not incompatible:
            matching_mass, incompatible_mass = 1.0, 0.0
        else:
            matching_mass, incompatible_mass = compatibility, 1.0 - compatibility
        for edge in matching:
            predicted[node_index[edge["target_node_id"]]] += mass * matching_mass / len(matching)
        for edge in incompatible:
            predicted[node_index[edge["target_node_id"]]] += mass * incompatible_mass / len(incompatible)
    predicted = [(1.0 - noise) * value + noise / len(node_ids) for value in predicted]
    total = sum(predicted)
    return [value / total for value in predicted]


def _stage_b_probability_vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise OccludedGoalMetricsError(f"{label} vector shape drift")
    vector = [_finite(item, label) for item in value]
    if any(item < 0.0 or item > 1.0 for item in vector) or not math.isclose(
        sum(vector), 1.0, rel_tol=0.0, abs_tol=1e-8
    ):
        raise OccludedGoalMetricsError(f"{label} must be a normalized probability vector")
    return vector


def _stage_b_oracle_cycle_distance(
    graph: Mapping[str, Any], queries: Sequence[Mapping[str, Any]]
) -> float:
    edge_by_id = {edge["edge_id"]: edge for edge in graph["edges"]}
    edge_ids = list(graph["stage_b_prelude_edge_ids"])
    for query in sorted(queries, key=lambda item: item["query_index"]):
        edge_ids.extend(
            query["stage_b_choice_macros"][query["true_next_edge_id"]]["constituent_edge_ids"]
        )
    return sum(float(edge_by_id[edge_id]["edge_cost"]) for edge_id in edge_ids)


def validate_stage_b_trace_rows(
    graph_manifest: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    stage_a_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not stage_a_authorizes_stage_b(stage_a_metrics):
        raise OccludedGoalMetricsError("Stage B is not authorized by Stage A")
    manifest = validate_graph_manifest(graph_manifest)
    queries = validate_query_rows(query_rows, manifest)
    if not rows:
        raise OccludedGoalMetricsError("Stage-B trace cannot be empty")
    strongest = stage_a_metrics["decision"].get("strongest_memory_condition")
    if strongest not in {"MAP_FILTER", "FULL_BELIEF"}:
        raise OccludedGoalMetricsError("authorized Stage B requires MAP_FILTER or FULL_BELIEF")
    parameters = stage_a_metrics.get("calibration_binding", {}).get("selected_parameters")
    parameters = _require_mapping(
        parameters, fields=CALIBRATION_PARAMETER_FIELDS, label="Stage-B calibration parameters"
    )
    temperature = _finite(parameters["observation_softmax_temperature"], "temperature")
    compatibility = _finite(parameters["action_compatible_edge_probability"], "compatibility")
    noise = _finite(parameters["transition_noise_probability"], "transition noise")
    threshold = _finite(parameters["normalized_entropy_abstention_threshold"], "entropy threshold")
    graphs = {graph["episode_id"]: graph for graph in manifest["graphs"]}
    heldout_queries = [query for query in queries if query["role"] == "DEVELOPMENT_HELDOUT"]
    query_by_id = {query["query_id"]: query for query in heldout_queries}
    episode_queries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in heldout_queries:
        episode_queries[query["episode_id"]].append(query)
    expected_sources = {
        "CURRENT_FRAME_NEAREST_NODE": "CURRENT_FRAME_NEAREST_NODE",
        "STRONGEST_STAGE_A_MEMORY": strongest,
        "ORACLE_PLACE_IDENTITY": "ORACLE_PLACE_IDENTITY",
    }
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[str, str, int]] = set(); step_ids: set[str] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_episode_conditions: dict[str, set[str]] = defaultdict(set)
    all_fresh_occurrences: set[str] = set()
    for index, original in enumerate(rows):
        row = _require_mapping(original, fields=STAGE_B_TRACE_FIELDS, label=f"stage_b[{index}]")
        condition = _string(row["condition_id"], "condition_id")
        episode = _string(row["episode_id"], "episode_id")
        if condition not in C.STAGE_B_CONDITION_IDS or episode not in episode_queries:
            raise OccludedGoalMetricsError("Stage-B condition/heldout episode drift")
        graph = graphs[episode]; query_id = _string(row["query_id"], "query_id")
        query = query_by_id.get(query_id)
        if query is None or query["episode_id"] != episode:
            raise OccludedGoalMetricsError("Stage-B query binding drift")
        if row["family"] != graph["family"] or row["family"] != query["family"]:
            raise OccludedGoalMetricsError("Stage-B family binding drift")
        if row["source_condition_id"] != expected_sources[condition]:
            raise OccludedGoalMetricsError("Stage-B source condition drift")
        decision = row["decision_index"]
        if isinstance(decision, bool) or not isinstance(decision, int) or not 0 <= decision < C.STAGE_B_EXECUTION_POLICY["decision_budget"]:
            raise OccludedGoalMetricsError("Stage-B decision index exceeds the frozen choice budget")
        if row["query_index"] != query["query_index"] or isinstance(row["query_index"], bool):
            raise OccludedGoalMetricsError("Stage-B query index drift")
        identity = (condition, episode, decision)
        if identity in identities:
            raise OccludedGoalMetricsError("duplicate Stage-B step identity")
        identities.add(identity)
        step_id = _string(row["step_id"], "step_id")
        if step_id in step_ids:
            raise OccludedGoalMetricsError("duplicate Stage-B step_id")
        step_ids.add(step_id); _string(row["execution_id"], "execution_id")
        for field in (
            "abstained", "ranker_called", "local_execution_success",
            "oracle_admissible_execution", "recovery_pending_before",
            "recovery_pending_after", "false_confident_wrong_turn",
            "false_place_merge", "false_loop_closure", "replan", "recovery",
            "immediate_contact", "stuck", "terminal", "goal_reached",
        ):
            _bool(row[field], field)
        node_ids = _strings(row["belief_node_ids"], "belief_node_ids")
        graph_node_ids = tuple(node["node_id"] for node in graph["nodes"])
        if node_ids != graph_node_ids:
            raise OccludedGoalMetricsError("Stage-B belief node order differs from graph")
        node_by_id = {node["node_id"]: node for node in graph["nodes"]}
        edge_by_id = {edge["edge_id"]: edge for edge in graph["edges"]}
        update_ids = _strings(row["filter_update_observation_ids"], "filter update observation ids")
        update_shas = _strings(row["filter_update_pixel_sha256s"], "filter update pixel SHAs", unique=False)
        update_nodes = _strings(row["filter_update_node_ids"], "filter update nodes", unique=False)
        update_actions = row["filter_update_incoming_action_labels"]
        update_count = len(update_ids)
        matrices = {}
        for field in (
            "filter_update_observation_similarities",
            "filter_update_observation_likelihoods",
            "filter_update_transition_prior_probabilities",
            "filter_update_preprojection_probabilities",
            "filter_update_posterior_probabilities",
        ):
            value = row[field]
            if not isinstance(value, list) or len(value) != update_count:
                raise OccludedGoalMetricsError(f"{field} update count drift")
            matrices[field] = value
        if not isinstance(update_actions, list) or not (
            len(update_shas) == len(update_nodes) == len(update_actions) == update_count
        ):
            raise OccludedGoalMetricsError("Stage-B filter-update evidence length drift")
        if row["observation_id"] != update_ids[-1] or row["observation_pixel_sha256"] != update_shas[-1]:
            raise OccludedGoalMetricsError("decision observation must be the final carried filter update")
        if row["actual_node_id_before"] != query["true_node_id"] or update_nodes[-1] != query["true_node_id"]:
            raise OccludedGoalMetricsError("Stage-B actual query state drift")
        normalized_updates: dict[str, list[list[float]]] = {field: [] for field in matrices}
        for update_index in range(update_count):
            node = update_nodes[update_index]
            if node not in node_by_id or update_shas[update_index] != node_by_id[node]["pixel_sha256"]:
                raise OccludedGoalMetricsError("Stage-B filter observation/node pixel binding drift")
            sha = update_shas[update_index]
            if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
                raise OccludedGoalMetricsError("Stage-B observation SHA drift")
            action = update_actions[update_index]
            if action is not None:
                _string(action, "filter incoming action")
            similarities = matrices["filter_update_observation_similarities"][update_index]
            if not isinstance(similarities, list) or len(similarities) != len(node_ids):
                raise OccludedGoalMetricsError("Stage-B observation similarity shape drift")
            similarities = [_finite(value, "Stage-B observation similarity") for value in similarities]
            if any(value < -1.0 or value > 1.0 for value in similarities):
                raise OccludedGoalMetricsError("Stage-B observation similarities must be cosine values")
            likelihood = _stage_b_probability_vector(
                matrices["filter_update_observation_likelihoods"][update_index], len(node_ids), "Stage-B observation likelihood"
            )
            expected_likelihood = _softmax(similarities, temperature)
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-10) for a, b in zip(likelihood, expected_likelihood)):
                raise OccludedGoalMetricsError("Stage-B likelihood differs from frozen similarity softmax")
            prior = _stage_b_probability_vector(
                matrices["filter_update_transition_prior_probabilities"][update_index], len(node_ids), "Stage-B transition prior"
            )
            pre = _stage_b_probability_vector(
                matrices["filter_update_preprojection_probabilities"][update_index], len(node_ids), "Stage-B preprojection"
            )
            product = [a * b for a, b in zip(prior, likelihood)]; total = sum(product)
            expected_pre = [value / total for value in product]
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8) for a, b in zip(pre, expected_pre)):
                raise OccludedGoalMetricsError("Stage-B preprojection differs from prior-times-likelihood")
            posterior = _stage_b_probability_vector(
                matrices["filter_update_posterior_probabilities"][update_index], len(node_ids), "Stage-B posterior"
            )
            if condition == "CURRENT_FRAME_NEAREST_NODE" or (
                condition == "STRONGEST_STAGE_A_MEMORY"
                and expected_sources[condition] == "MAP_FILTER"
            ):
                order = _ranking(node_ids, pre); expected_posterior = [float(i == order[0]) for i in range(len(node_ids))]
            elif condition == "ORACLE_PLACE_IDENTITY":
                expected_posterior = [float(node_id == node) for node_id in node_ids]
            else:
                expected_posterior = pre
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8) for a, b in zip(posterior, expected_posterior)):
                raise OccludedGoalMetricsError("Stage-B condition posterior projection drift")
            normalized_updates["filter_update_observation_similarities"].append(similarities)
            normalized_updates["filter_update_observation_likelihoods"].append(likelihood)
            normalized_updates["filter_update_transition_prior_probabilities"].append(prior)
            normalized_updates["filter_update_preprojection_probabilities"].append(pre)
            normalized_updates["filter_update_posterior_probabilities"].append(posterior)
        posterior = normalized_updates["filter_update_posterior_probabilities"][-1]
        belief = _stage_b_probability_vector(row["belief_probabilities"], len(node_ids), "Stage-B belief")
        if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-10) for a, b in zip(belief, posterior)):
            raise OccludedGoalMetricsError("Stage-B decision belief differs from final filter update")
        selected_index = _ranking(node_ids, belief)[0]; selected_node = node_ids[selected_index]
        if row["selected_node_id"] != selected_node:
            raise OccludedGoalMetricsError("Stage-B selected node differs from posterior MAP")
        if row["actual_alias_group_id"] != node_by_id[query["true_node_id"]]["alias_group_id"] or row["selected_alias_group_id"] != node_by_id[selected_node]["alias_group_id"]:
            raise OccludedGoalMetricsError("Stage-B alias-group evidence drift")
        proposed_port, witness_edge_id = _route_choice(node_ids, belief, _graph_tables(graph))
        if row["belief_proposed_port_label"] != proposed_port or row["belief_selected_edge_id"] != witness_edge_id:
            raise OccludedGoalMetricsError("Stage-B route choice differs from posterior route mass")
        entropy = _entropy(belief)
        if not math.isclose(_finite(row["normalized_entropy"], "entropy"), entropy, rel_tol=0.0, abs_tol=1e-10) or not math.isclose(_finite(row["entropy_threshold"], "entropy threshold"), threshold, rel_tol=0.0, abs_tol=1e-12):
            raise OccludedGoalMetricsError("Stage-B entropy/threshold drift")
        abstained = entropy > threshold
        if row["abstained"] != abstained:
            raise OccludedGoalMetricsError("Stage-B abstention differs from calibrated threshold")
        selected_port = None if abstained else proposed_port
        if row["selected_port_label"] != selected_port or row["true_next_port_label"] != query["true_next_port_label"]:
            raise OccludedGoalMetricsError("Stage-B selected/true port evidence drift")
        waypoint = None if witness_edge_id is None else edge_by_id[witness_edge_id]["relative_waypoint"]
        if row["stored_relative_waypoint"] != waypoint:
            raise OccludedGoalMetricsError("Stage-B stored waypoint differs from belief edge witness")
        candidate_outcomes = query["stage_b_local_candidate_outcomes"]
        mask = [outcome["candidate_index"] for outcome in candidate_outcomes if outcome["oracle_admissible"]]
        if row["local_oracle_admissible_candidate_indices"] != mask:
            raise OccludedGoalMetricsError("Stage-B local candidate mask differs from preregistered geometry")
        if abstained:
            disposition = "ENTROPY_ABSTENTION"
        elif witness_edge_id is None:
            disposition = "NO_ROUTE_WITNESS"
        elif not mask:
            disposition = "NO_ORACLE_ADMISSIBLE_LOCAL_CANDIDATE"
        else:
            disposition = "EXECUTED_CANDIDATE"
        if row["action_disposition"] != disposition:
            raise OccludedGoalMetricsError("Stage-B action disposition drift")
        acting = disposition == "EXECUTED_CANDIDATE"
        scores = row["local_candidate_scores"]
        if acting:
            if row["ranker_called"] is not True or not isinstance(scores, list) or len(scores) != len(C.STAGE_B_LOCAL_CANDIDATE_IDS):
                raise OccludedGoalMetricsError("acting Stage-B row must persist all twelve ranker scores")
            scores = [_finite(value, "local candidate score") for value in scores]
            chosen = min(mask, key=lambda candidate_index: (-scores[candidate_index], candidate_index))
            candidate = candidate_outcomes[chosen]
            if row["local_candidate_index"] != chosen or row["local_candidate_name"] != candidate["candidate_name"] or not math.isclose(_finite(row["local_candidate_score"], "chosen candidate score"), scores[chosen], rel_tol=0.0, abs_tol=1e-12):
                raise OccludedGoalMetricsError("Stage-B chosen candidate is not masked score argmax")
            choice_edge_id = candidate["actual_edge_id"]
            if row["executed_choice_edge_id"] != choice_edge_id or row["executed_port_label"] != candidate["actual_port_label"]:
                raise OccludedGoalMetricsError("Stage-B executed edge/port differs from candidate outcome")
            if row["local_execution_success"] != (candidate["actual_port_label"] == proposed_port):
                raise OccludedGoalMetricsError("Stage-B local-execution success drift")
            if row["oracle_admissible_execution"] is not True or row["immediate_contact"] != candidate["immediate_contact"] or row["stuck"] != candidate["stuck"]:
                raise OccludedGoalMetricsError("Stage-B executed candidate outcome drift")
            macro = query["stage_b_choice_macros"][choice_edge_id]
            expected_edges = list(macro["constituent_edge_ids"])
            expected_nodes = [query["true_node_id"]]
            expected_actions = []; expected_costs = []; expected_shas = []
            for edge_id in expected_edges:
                edge = edge_by_id[edge_id]
                if edge["source_node_id"] != expected_nodes[-1] or not edge["physically_executable"] or not edge["oracle_admissible"]:
                    raise OccludedGoalMetricsError("Stage-B macro constituent path is not admissible/contiguous")
                expected_nodes.append(edge["target_node_id"]); expected_actions.append(edge["executed_action_label"])
                expected_costs.append(float(edge["edge_cost"])); expected_shas.append(node_by_id[edge["target_node_id"]]["pixel_sha256"])
            if row["constituent_edge_ids"] != expected_edges or row["constituent_node_ids"] != expected_nodes or row["constituent_action_labels"] != expected_actions or row["constituent_edge_costs_m"] != expected_costs:
                raise OccludedGoalMetricsError("Stage-B constituent path evidence differs from registered macro")
            constituent_observations = _strings(row["constituent_observation_ids"], "constituent observation ids")
            constituent_shas = _strings(row["constituent_observation_pixel_sha256s"], "constituent observation SHAs", unique=False)
            if len(constituent_observations) != len(expected_edges) or list(constituent_shas) != expected_shas:
                raise OccludedGoalMetricsError("Stage-B constituent reobservation evidence drift")
            if row["reobservation_id"] != constituent_observations[-1] or row["reobservation_pixel_sha256"] != constituent_shas[-1]:
                raise OccludedGoalMetricsError("Stage-B terminal macro reobservation drift")
            actual_after = macro["destination_node_id"]
        else:
            if row["ranker_called"] is not False or scores != [] or any(row[field] is not None for field in ("local_candidate_index", "local_candidate_name", "local_candidate_score", "executed_port_label", "executed_choice_edge_id")):
                raise OccludedGoalMetricsError("nonacting Stage-B row contains ranker/execution evidence")
            if row["local_execution_success"] or row["oracle_admissible_execution"] or row["immediate_contact"]:
                raise OccludedGoalMetricsError("nonacting Stage-B row reports an execution outcome")
            if row["constituent_edge_ids"] != [] or row["constituent_action_labels"] != [] or row["constituent_edge_costs_m"] != [] or row["constituent_observation_ids"] != [] or row["constituent_observation_pixel_sha256s"] != [] or row["constituent_node_ids"] != [query["true_node_id"]]:
                raise OccludedGoalMetricsError("nonacting Stage-B row must persist an empty macro")
            _string(row["reobservation_id"], "NO_OP reobservation_id")
            if row["reobservation_pixel_sha256"] != node_by_id[query["true_node_id"]]["pixel_sha256"]:
                raise OccludedGoalMetricsError("NO_OP reobservation pixel drift")
            actual_after = query["true_node_id"]
        if row["actual_node_id_after"] != actual_after:
            raise OccludedGoalMetricsError("Stage-B actual successor differs from executed macro")
        ranker_latency = _finite(row["ranker_latency_ms"], "ranker_latency_ms")
        planning_latency = _finite(row["planning_latency_ms"], "planning_latency_ms")
        if ranker_latency < 0.0 or planning_latency < ranker_latency or (not acting and ranker_latency != 0.0):
            raise OccludedGoalMetricsError("Stage-B latency evidence drift")
        command_vectors: dict[str, list[Any]] = {}
        for field, rows_expected, columns in (
            ("ranker_previous_applied_command", 1, 3),
            ("ranker_control_history", 15, 2),
            ("post_decision_previous_applied_command", 1, 3),
            ("post_decision_control_history", 15, 2),
        ):
            raw = row[field]
            raw_rows = [raw] if rows_expected == 1 else raw
            if not isinstance(raw_rows, list) or len(raw_rows) != rows_expected or any(
                not isinstance(vector, list) or len(vector) != columns for vector in raw_rows
            ):
                raise OccludedGoalMetricsError(f"Stage-B {field} shape drift")
            normalized_vectors = [[_finite(value, field) for value in vector] for vector in raw_rows]
            command_vectors[field] = normalized_vectors[0] if rows_expected == 1 else normalized_vectors
        applied = row["selected_candidate_applied_commands"]
        if acting:
            if not isinstance(applied, list) or len(applied) != 5 or any(
                not isinstance(vector, list) or len(vector) != 3 for vector in applied
            ):
                raise OccludedGoalMetricsError("Stage-B selected candidate applied-command shape drift")
            applied = [[_finite(value, "selected candidate applied command") for value in vector] for vector in applied]
        elif applied != []:
            raise OccludedGoalMetricsError("nonacting Stage-B row persisted applied candidate commands")
        if not acting and (
            command_vectors["post_decision_previous_applied_command"]
            != command_vectors["ranker_previous_applied_command"]
            or command_vectors["post_decision_control_history"]
            != command_vectors["ranker_control_history"]
        ):
            raise OccludedGoalMetricsError("NO_OP changed the applied-command/control history")
        distances = _graph_tables(graph)["distance"]
        before = _finite(row["geodesic_distance_before_m"], "distance before")
        after = _finite(row["geodesic_distance_after_m"], "distance after")
        progress = _finite(row["geodesic_progress_m"], "geodesic progress")
        executed_distance = _finite(row["executed_distance_m"], "executed distance")
        oracle_distance = _finite(row["oracle_path_distance_m"], "oracle path distance")
        expected_oracle_distance = _stage_b_oracle_cycle_distance(graph, episode_queries[episode])
        expected_prelude_distance = sum(
            float(edge_by_id[edge_id]["edge_cost"])
            for edge_id in graph["stage_b_prelude_edge_ids"]
        )
        if not math.isclose(before, distances[query["true_node_id"]], rel_tol=0.0, abs_tol=1e-10) or not math.isclose(after, distances[actual_after], rel_tol=0.0, abs_tol=1e-10) or not math.isclose(progress, before - after, rel_tol=0.0, abs_tol=1e-10) or not math.isclose(executed_distance, sum(float(value) for value in row["constituent_edge_costs_m"]), rel_tol=0.0, abs_tol=1e-10) or not math.isclose(oracle_distance, expected_oracle_distance, rel_tol=0.0, abs_tol=1e-10):
            raise OccludedGoalMetricsError("Stage-B distance/progress evidence drift")
        if not math.isclose(_finite(row["prelude_distance_m"], "prelude distance"), expected_prelude_distance, rel_tol=0.0, abs_tol=1e-10):
            raise OccludedGoalMetricsError("Stage-B prelude distance drift")
        if row["decision_budget"] != C.STAGE_B_EXECUTION_POLICY["decision_budget"] or isinstance(row["decision_budget"], bool):
            raise OccludedGoalMetricsError("Stage-B decision budget drift")
        wrong_confident = bool(not abstained and selected_port != query["true_next_port_label"])
        false_merge = bool(wrong_confident and selected_node != query["true_node_id"] and node_by_id[selected_node]["alias_group_id"] == node_by_id[query["true_node_id"]]["alias_group_id"])
        false_loop = bool(wrong_confident and selected_node != query["true_node_id"] and selected_node in row["prior_visited_node_ids"])
        if row["false_confident_wrong_turn"] != wrong_confident or row["false_place_merge"] != false_merge or row["false_loop_closure"] != false_loop:
            raise OccludedGoalMetricsError("Stage-B false-confidence/merge/loop evidence drift")
        normalized_row = {
            **row,
            "belief_node_ids": list(node_ids), "belief_probabilities": belief,
            "filter_update_observation_ids": list(update_ids),
            "filter_update_pixel_sha256s": list(update_shas),
            "filter_update_node_ids": list(update_nodes),
            **normalized_updates,
            "local_candidate_scores": scores,
            **command_vectors,
            "selected_candidate_applied_commands": applied,
        }
        normalized.append(normalized_row); grouped[(condition, episode)].append(normalized_row)
        seen_episode_conditions[condition].add(episode)
    expected_episodes = set(episode_queries)
    for condition in C.STAGE_B_CONDITION_IDS:
        if seen_episode_conditions[condition] != expected_episodes:
            raise OccludedGoalMetricsError("Stage-B requires the same 16 heldout episodes per condition")
    for (condition, episode), execution in grouped.items():
        execution.sort(key=lambda item: item["decision_index"])
        if [row["decision_index"] for row in execution] != list(range(len(execution))):
            raise OccludedGoalMetricsError("Stage-B decision indices must be contiguous from zero")
        if len(execution) > C.STAGE_B_EXECUTION_POLICY["decision_budget"]:
            raise OccludedGoalMetricsError("Stage-B execution exceeded its choice budget")
        if len({row["execution_id"] for row in execution}) != 1:
            raise OccludedGoalMetricsError("Stage-B execution_id changed within execution")
        graph = graphs[episode]; node_by_id = {node["node_id"]: node for node in graph["nodes"]}
        edge_by_id = {edge["edge_id"]: edge for edge in graph["edges"]}
        ordered_queries = sorted(episode_queries[episode], key=lambda item: item["query_index"])
        prelude_edges = [edge_by_id[edge_id] for edge_id in graph["stage_b_prelude_edge_ids"]]
        expected_update_nodes = [graph["stage_b_start_node_id"], *[edge["target_node_id"] for edge in prelude_edges]]
        expected_update_actions: list[str | None] = [None, *[edge["executed_action_label"] for edge in prelude_edges]]
        expected_visited: list[str] = []
        for node in expected_update_nodes[:-1]:
            if node not in expected_visited:
                expected_visited.append(node)
        carried = [1.0 / len(node_by_id)] * len(node_by_id)
        nonmovement = 0; recovery_pending = False
        forbidden_occurrences = {
            node["keyframe_observation_id"] for node in graph["nodes"]
        } | {visit["observation_id"] for visit in graph["phase_a_traversal"]}
        for registered_query in ordered_queries:
            forbidden_occurrences.update(registered_query["history_observation_ids"])
        for row_index, row in enumerate(execution):
            query = query_by_id[row["query_id"]]
            if row_index == 0:
                if query["query_index"] != 0 or row["actual_node_id_before"] != graph["stage_b_decision_start_node_id"]:
                    raise OccludedGoalMetricsError("Stage-B execution did not begin at query zero after prelude")
            else:
                previous = execution[row_index - 1]
                if row["actual_node_id_before"] != previous["actual_node_id_after"]:
                    raise OccludedGoalMetricsError("Stage-B actual state is not continuous")
                expected_update_nodes = (
                    previous["constituent_node_ids"][1:]
                    if previous["constituent_edge_ids"]
                    else [previous["actual_node_id_after"]]
                )
                expected_update_actions = (
                    previous["constituent_action_labels"]
                    if previous["constituent_edge_ids"]
                    else ["NO_OP"]
                )
                expected_update_ids = (
                    previous["constituent_observation_ids"]
                    if previous["constituent_edge_ids"]
                    else [previous["reobservation_id"]]
                )
                if row["filter_update_observation_ids"] != expected_update_ids:
                    raise OccludedGoalMetricsError("Stage-B carried reobservation identity drift")
                if (
                    row["ranker_previous_applied_command"]
                    != previous["post_decision_previous_applied_command"]
                    or row["ranker_control_history"]
                    != previous["post_decision_control_history"]
                ):
                    raise OccludedGoalMetricsError("Stage-B command/control history is not continuous")
            if row["filter_update_node_ids"] != expected_update_nodes or row["filter_update_incoming_action_labels"] != expected_update_actions:
                raise OccludedGoalMetricsError("Stage-B carried filter path/action sequence drift")
            generated_occurrences = (
                list(row["filter_update_observation_ids"]) if row_index == 0 else []
            )
            generated_occurrences.extend(
                row["constituent_observation_ids"]
                if row["constituent_edge_ids"]
                else [row["reobservation_id"]]
            )
            if len(generated_occurrences) != len(set(generated_occurrences)) or any(
                occurrence in forbidden_occurrences or occurrence in all_fresh_occurrences
                for occurrence in generated_occurrences
            ):
                raise OccludedGoalMetricsError(
                    "Stage-B fresh observation occurrence identity was reused"
                )
            all_fresh_occurrences.update(generated_occurrences)
            if row["prior_visited_node_ids"] != expected_visited:
                raise OccludedGoalMetricsError("Stage-B visited-node history drift")
            for update_index, action in enumerate(expected_update_actions):
                prior = row["filter_update_transition_prior_probabilities"][update_index]
                if condition == "CURRENT_FRAME_NEAREST_NODE":
                    expected_prior = [1.0 / len(node_by_id)] * len(node_by_id)
                elif condition == "ORACLE_PLACE_IDENTITY":
                    expected_prior = [float(node == expected_update_nodes[update_index]) for node in row["belief_node_ids"]]
                elif action is None:
                    expected_prior = carried
                else:
                    expected_prior = _stage_b_transition(
                        carried, action, row["belief_node_ids"], graph, compatibility, noise
                    )
                if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-8) for a, b in zip(prior, expected_prior)):
                    raise OccludedGoalMetricsError("Stage-B transition prior is not carried through executed actions")
                carried = row["filter_update_posterior_probabilities"][update_index]
            if row["recovery_pending_before"] != recovery_pending:
                raise OccludedGoalMetricsError("Stage-B recovery-pending entry state drift")
            acting = row["action_disposition"] == "EXECUTED_CANDIDATE"
            executed_correct = bool(
                acting and row["executed_choice_edge_id"] == query["true_next_edge_id"]
                and row["geodesic_progress_m"] > 0.0
            )
            expected_recovery = bool(recovery_pending and executed_correct)
            if executed_correct:
                recovery_pending = False
            else:
                recovery_pending = True
            if row["recovery"] != expected_recovery or row["recovery_pending_after"] != recovery_pending:
                raise OccludedGoalMetricsError("Stage-B recovery transition drift")
            nonmovement = 0 if acting else nonmovement + 1
            if row["consecutive_nonmovement_decisions"] != nonmovement:
                raise OccludedGoalMetricsError("Stage-B nonmovement counter drift")
            visited_now = (
                row["constituent_node_ids"][:-1]
                if row["constituent_edge_ids"]
                else [row["actual_node_id_before"]]
            )
            for node in visited_now:
                if node not in expected_visited:
                    expected_visited.append(node)
            goal_reached = row["actual_node_id_after"] == graph["goal_node_id"]
            if row["goal_reached"] != goal_reached:
                raise OccludedGoalMetricsError("Stage-B goal outcome differs from actual node")
            if goal_reached:
                terminal_reason = "GOAL_REACHED"
            elif nonmovement == C.STAGE_B_EXECUTION_POLICY["maximum_consecutive_nonmovement_decisions"]:
                terminal_reason = "NONMOVEMENT_LIMIT"
            elif row["decision_index"] == C.STAGE_B_EXECUTION_POLICY["decision_budget"] - 1:
                terminal_reason = "DECISION_BUDGET_EXHAUSTED"
            else:
                terminal_reason = None
            terminal = terminal_reason is not None
            if row["terminal_reason"] != terminal_reason or row["terminal"] != terminal or row["replan"] != (not terminal):
                raise OccludedGoalMetricsError("Stage-B terminal/replan precedence drift")
            if row["stuck"] != (terminal_reason == "NONMOVEMENT_LIMIT"):
                raise OccludedGoalMetricsError("Stage-B stuck flag differs from nonmovement termination")
            if terminal != (row_index == len(execution) - 1):
                raise OccludedGoalMetricsError("Stage-B terminal row must be exactly the final row")
        if not execution[-1]["terminal"]:
            raise OccludedGoalMetricsError("Stage-B execution lacks a terminal outcome")
    return normalized


def _stage_b_condition(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    executions: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        executions[row["episode_id"]].append(row)
    outcomes = []
    for episode, steps in sorted(executions.items()):
        steps = sorted(steps, key=lambda row: row["decision_index"]); terminal = [row for row in steps if row["terminal"]][0]
        decisions = len(steps); correct = sum(not row["abstained"] and row["selected_port_label"] == row["true_next_port_label"] for row in steps)
        wrong = sum(row["executed_port_label"] is not None and row["executed_port_label"] != row["true_next_port_label"] for row in steps)
        prelude = float(terminal["prelude_distance_m"])
        executed = prelude + sum(float(row["executed_distance_m"]) for row in steps); oracle = float(terminal["oracle_path_distance_m"])
        reached = bool(terminal["goal_reached"]); efficiency = min(1.0, oracle / executed) if reached and executed > 0 else 0.0
        outcomes.append({"episode_id": episode, "family": terminal["family"], "goal_reached": reached, "terminal_reason": terminal["terminal_reason"], "decisions": decisions, "correct": correct, "wrong": wrong, "efficiency": efficiency, "executed_distance_m": executed, "false_confident": sum(row["false_confident_wrong_turn"] for row in steps), "false_merge": sum(row["false_place_merge"] for row in steps), "false_loop": sum(row["false_loop_closure"] for row in steps), "abstentions": sum(row["abstained"] for row in steps), "replans": max(0, decisions - 1), "recoveries": sum(row["recovery"] for row in steps), "progress": sum(float(row["geodesic_progress_m"]) for row in steps), "contacts": sum(row["immediate_contact"] for row in steps), "stuck": sum(row["stuck"] for row in steps), "ranker_calls": sum(row["ranker_called"] for row in steps), "local_execution_successes": sum(row["local_execution_success"] for row in steps), "planning_latencies": [float(row["planning_latency_ms"]) for row in steps], "ranker_latencies": [float(row["ranker_latency_ms"]) for row in steps if row["ranker_called"]]})
    decisions = sum(row["decisions"] for row in outcomes); planning_latencies = sorted(value for row in outcomes for value in row["planning_latencies"]); ranker_latencies = sorted(value for row in outcomes for value in row["ranker_latencies"])
    family_reach = {family: sum(row["goal_reached"] for row in outcomes if row["family"] == family) for family in C.FAMILY_IDS}
    ranker_call_count = sum(row["ranker_calls"] for row in outcomes)
    return {"episodes": len(outcomes), "goals_reached": sum(row["goal_reached"] for row in outcomes), "reach_rate": mean(row["goal_reached"] for row in outcomes), "correct_next_edge_accuracy": sum(row["correct"] for row in outcomes) / decisions, "wrong_turn_rate": sum(row["wrong"] for row in outcomes) / decisions, "path_efficiency": mean(row["efficiency"] for row in outcomes), "false_confident_wrong_turn_rate": sum(row["false_confident"] for row in outcomes) / decisions, "false_place_merge_count": sum(row["false_merge"] for row in outcomes), "false_loop_closure_count": sum(row["false_loop"] for row in outcomes), "abstention_rate": sum(row["abstentions"] for row in outcomes) / decisions, "mean_replans": mean(row["replans"] for row in outcomes), "mean_recoveries": mean(row["recoveries"] for row in outcomes), "mean_geodesic_progress_m": mean(row["progress"] for row in outcomes), "immediate_contact_count": sum(row["contacts"] for row in outcomes), "stuck_count": sum(row["stuck"] for row in outcomes), "ranker_call_count": ranker_call_count, "local_execution_success_rate": (sum(row["local_execution_successes"] for row in outcomes) / ranker_call_count if ranker_call_count else 0.0), "mean_planning_latency_ms": mean(planning_latencies), "p95_planning_latency_ms": planning_latencies[min(len(planning_latencies) - 1, math.ceil(0.95 * len(planning_latencies)) - 1)], "mean_ranker_latency_ms": (mean(ranker_latencies) if ranker_latencies else 0.0), "p95_ranker_latency_ms": (ranker_latencies[min(len(ranker_latencies) - 1, math.ceil(0.95 * len(ranker_latencies)) - 1)] if ranker_latencies else 0.0), "family_goals_reached": family_reach, "per_episode": outcomes}


def _stage_b_classification(gate_passed: bool, oracle_goals_reached: int) -> str:
    if gate_passed:
        return "OCCLUDED_GOAL_TOPOLOGICAL_NAVIGATION_SIGNAL"
    if oracle_goals_reached >= C.STAGE_B_GATE["goals_reached_minimum"]:
        return "PLACE_BELIEF_INTERFACE_NO_GO"
    return "LOCAL_EXECUTION_INTERFACE_NO_GO"


def recompute_stage_b_metrics(
    graph_manifest: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
    trace_rows: Sequence[Mapping[str, Any]],
    stage_a_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    traces = validate_stage_b_trace_rows(
        graph_manifest, query_rows, trace_rows, stage_a_metrics
    )
    expected_episodes = set(stage_a_metrics["conditions"]["FULL_BELIEF"]["episodes"])
    if any({row["episode_id"] for row in traces if row["condition_id"] == condition} != expected_episodes for condition in C.STAGE_B_CONDITION_IDS):
        raise OccludedGoalMetricsError("Stage-B episodes differ from Stage-A development-heldout identities")
    grouped = {condition: _stage_b_condition([row for row in traces if row["condition_id"] == condition]) for condition in C.STAGE_B_CONDITION_IDS}
    current = grouped["CURRENT_FRAME_NEAREST_NODE"]; strongest = grouped["STRONGEST_STAGE_A_MEMORY"]; oracle = grouped["ORACLE_PLACE_IDENTITY"]
    oracle_fraction = 0.0 if oracle["goals_reached"] == 0 else strongest["goals_reached"] / oracle["goals_reached"]
    gate = C.STAGE_B_GATE
    checks = {"goals_reached": strongest["goals_reached"] >= gate["goals_reached_minimum"], "reach_gain_over_current": strongest["goals_reached"] - current["goals_reached"] >= gate["reach_gain_over_current_minimum"], "path_efficiency": strongest["path_efficiency"] >= gate["path_efficiency_minimum"], "false_confident_wrong_turn_rate": strongest["false_confident_wrong_turn_rate"] <= gate["false_confident_wrong_turn_rate_maximum"], "oracle_reach_fraction": oracle_fraction >= gate["oracle_reach_fraction_minimum"], "no_family_zero": all(value >= gate["minimum_goals_reached_per_family"] for value in strongest["family_goals_reached"].values())}
    passed = all(checks.values())
    classification = _stage_b_classification(passed, oracle["goals_reached"])
    return C.attach_content_digest({"schema": "occluded_goal_topological_belief_v1.stage_b_metrics.v1", "experiment_id": C.EXPERIMENT_ID, "stage_a_binding": stage_a_metrics["content_digest"], "conditions": grouped, "gate": {"checks": checks, "oracle_reach_fraction": oracle_fraction, "pass": passed}, "decision": {"primary_classification": classification, "next_experiment": C.NEXT_DECISION_BY_CLASSIFICATION[classification]}})


__all__ = [
    "BELIEF_FIELDS", "CALIBRATION_FIELDS", "CURRENT_IDENTITY_PROJECTION_FIELDS",
    "EDGE_FIELDS", "GRAPH_FIELDS", "GRAPH_ROOT_FIELDS", "IDENTITY_COMPARISON_FIELDS",
    "IDENTITY_DISJOINTNESS_FIELDS", "NODE_FIELDS", "QUERY_FIELDS", "STAGE_B_TRACE_FIELDS",
    "OccludedGoalMetricsError", "belief_row_authority", "calibration_authority",
    "build_identity_disjointness_evidence",
    "deterministic_action_history_position_mapping",
    "graph_manifest_authority", "query_row_authority", "recompute_stage_a_metrics",
    "recompute_stage_b_metrics", "reducer_authority", "stage_a_authorizes_stage_b",
    "stage_a_condition_ids", "stage_b_trace_authority", "validate_belief_rows",
    "validate_calibration", "validate_graph_manifest", "validate_query_rows",
    "validate_identity_disjointness", "validate_stage_b_trace_rows",
]
