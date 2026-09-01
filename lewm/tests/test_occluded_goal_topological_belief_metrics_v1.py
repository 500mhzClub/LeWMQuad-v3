from __future__ import annotations

import copy
import hashlib
import math
import unittest
from collections import defaultdict
from statistics import mean

from lewm.safety import occluded_goal_topological_belief_metrics_v1 as M
from lewm.safety import occluded_goal_topological_belief_v1_contract as C


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _node(
    graph_id: str,
    node_id: str,
    alias: str,
    kind: str,
    module: int,
    side: str,
    visit: int,
    timestamp: float,
    template: str,
    pixel_sha: str,
    goal: bool = False,
) -> dict:
    return {
        "node_id": node_id,
        "alias_group_id": alias,
        "node_kind": kind,
        "module_index": module,
        "side_label": side,
        "keyframe_observation_id": f"{graph_id}-phase-a-keyframe-{node_id}",
        "observation_descriptor": template,
        "teacher_visit_index": visit,
        "goal_keyframe": goal,
        "pixel_template_id": template,
        "pixel_sha256": pixel_sha,
        "keyframe_timestamp_s": timestamp,
    }


def _edge(graph_id: str, source: str, target: str, port: str, cost: float) -> dict:
    return {
        "edge_id": f"{graph_id}-{source}-{target}",
        "source_node_id": source,
        "target_node_id": target,
        "port_label": port,
        "executed_action_label": "GO",
        "relative_waypoint": [cost, 0.0, 0.0],
        "edge_cost": cost,
        "teacher_edge": True,
        "physically_executable": True,
        "oracle_admissible": True,
    }


def _graph(family: str, local_index: int, global_index: int) -> dict:
    graph_id = f"ogtb-v1-graph-{family.lower()}-{local_index:02d}"
    episode_id = f"ogtb-v1-episode-{family.lower()}-{local_index:02d}"
    if local_index < 16:
        role = "FIT"
    elif local_index < 20:
        role = "CALIBRATION"
    else:
        role = "DEVELOPMENT_HELDOUT"
    a, b, c, d, goal = (f"{graph_id}-{suffix}" for suffix in ("a", "b", "c", "d", "goal"))
    alias_template = f"template-{family.lower()}-alias"
    alias_sha = _sha(alias_template)
    expected_kind = C.ALIASING_AUTHORITY["family_witness_node_kinds"][family]
    loop = family == "LOOP_ALIAS"
    first_visits = {a: 0, b: 1, c: 3 if loop else 2, d: 4 if loop else 3, goal: 5 if loop else 4}
    nodes = [
        _node(graph_id, a, f"{graph_id}-alias", expected_kind, 0, "LEFT", first_visits[a], float(first_visits[a]), alias_template, alias_sha),
        _node(graph_id, b, f"{graph_id}-alias", expected_kind, 1, "RIGHT", first_visits[b], float(first_visits[b]), alias_template, alias_sha),
        _node(graph_id, c, f"{graph_id}-c-only", "TRANSIT", 2, "CENTER", first_visits[c], float(first_visits[c]), f"{graph_id}-c-template", _sha(f"{graph_id}-c")),
        _node(graph_id, d, f"{graph_id}-d-only", "TRANSIT", 3, "CENTER", first_visits[d], float(first_visits[d]), f"{graph_id}-d-template", _sha(f"{graph_id}-d")),
        _node(graph_id, goal, f"{graph_id}-goal-only", "GOAL", 4, "CENTER", first_visits[goal], float(first_visits[goal]), f"{graph_id}-goal-template", _sha(f"{graph_id}-goal"), True),
    ]
    edges = [
        _edge(graph_id, goal, a, "STRAIGHT", 1.0),
        _edge(graph_id, a, goal, "LEFT", 1.0),
        _edge(graph_id, a, c, "RIGHT", 4.0),
        _edge(graph_id, a, b, "STRAIGHT", 5.0),
        _edge(graph_id, b, goal, "RIGHT", 1.0),
        _edge(graph_id, b, c, "LEFT", 4.0),
    ]
    if loop:
        edges.append(_edge(graph_id, b, a, "REVERSE", 5.0))
    edges.extend([
        _edge(graph_id, c, d, "STRAIGHT", 2.0),
        _edge(graph_id, c, goal, "LEFT", 1.0),
        _edge(graph_id, d, goal, "STRAIGHT", 1.0),
    ])
    by_pair = {(edge["source_node_id"], edge["target_node_id"]): edge for edge in edges}
    sequence = [a, b, a, c, d, goal] if loop else [a, b, c, d, goal]
    traversal = []
    for visit_index, node_id in enumerate(sequence):
        if visit_index == 0:
            arrival = None
            action = None
        else:
            arrival = by_pair[(sequence[visit_index - 1], node_id)]["edge_id"]
            action = "GO"
        first = first_visits[node_id] == visit_index
        traversal.append({
            "visit_index": visit_index,
            "node_id": node_id,
            "observation_id": (
                f"{graph_id}-phase-a-keyframe-{node_id}"
                if first else f"{graph_id}-phase-a-revisit-{visit_index}"
            ),
            "arrival_edge_id": arrival,
            "executed_action_label": action,
            "timestamp_s": float(visit_index),
        })
    witness_edges = [by_pair[(a, goal)]["edge_id"], by_pair[(b, goal)]["edge_id"]]
    cycle_nodes: list[str] = []
    cycle_edges: list[str] = []
    if loop:
        cycle_nodes = [a, b]
        cycle_edges = [by_pair[(a, b)]["edge_id"], by_pair[(b, a)]["edge_id"]]
        witness_edges += cycle_edges
    return {
        "graph_id": graph_id,
        "episode_id": episode_id,
        "scene_id": f"ogtb-v1-scene-{global_index:03d}",
        "procedural_seed": C.PROCEDURAL_SEED_BASE + global_index,
        "episode_path_id": f"ogtb-v1-path-{global_index:03d}",
        "identity_domain": C.IDENTITY_DOMAIN,
        "family": family,
        "role": role,
        "goal_node_id": goal,
        "stage_b_start_node_id": goal,
        "stage_b_decision_start_node_id": b,
        "stage_b_prelude_edge_ids": [
            by_pair[(goal, a)]["edge_id"], by_pair[(a, b)]["edge_id"],
        ],
        "nodes": nodes,
        "edges": edges,
        "phase_a_traversal": traversal,
        "family_adequacy_witness": {
            "family": family,
            "alias_group_id": f"{graph_id}-alias",
            "alias_node_ids": [a, b],
            "different_next_port_labels": ["LEFT", "RIGHT"],
            "witness_node_ids": [a, b],
            "witness_edge_ids": witness_edges,
            "cycle_node_ids": cycle_nodes,
            "cycle_edge_ids": cycle_edges,
        },
        "constructed_set_caveat": C.CONSTRUCTED_SET_CAVEAT,
    }


def _fixture() -> tuple[dict, list[dict]]:
    graphs = []
    for family in C.FAMILY_IDS:
        for local_index in range(24):
            graphs.append(_graph(family, local_index, len(graphs)))
    disjointness = M.build_identity_disjointness_evidence(graphs, {
        "scene_id_overlap_count": 0,
        "scene_id_sha256_overlap_count": 0,
        "episode_or_state_id_overlap_count": 0,
        "procedural_seed_overlap_count": 0,
        "textual_path_identity_overlap_count": 0,
        "structured_path_overlap_count": 0,
    })
    manifest = C.attach_content_digest({
        "schema": "occluded_goal_topological_belief_v1.graph_manifest.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "identity_disjointness": disjointness,
        "graphs": graphs,
    })
    queries = []
    for graph in graphs:
        nodes = [node["node_id"] for node in graph["nodes"]]
        a, b = nodes[:2]
        edges = [edge for edge in graph["edges"] if edge["source_node_id"] == b]
        true_edge = next(edge for edge in edges if edge["target_node_id"] == graph["goal_node_id"])
        base_history = [f"{graph['episode_id']}-phase-c-initial-{index}" for index in range(4)]
        query_occurrences = [f"{graph['episode_id']}-query-observation-{index}" for index in range(8)]
        for query_index in range(8):
            history = base_history + query_occurrences[:query_index + 1]
            admissible_ids = [edge["edge_id"] for edge in edges]
            correct_destination = b if query_index < 7 else graph["goal_node_id"]
            macros = {}
            for edge in edges:
                destination = correct_destination if edge["edge_id"] == true_edge["edge_id"] else b
                macros[edge["edge_id"]] = {
                    "outcome": (
                        "CORRECT_ADVANCE"
                        if edge["edge_id"] == true_edge["edge_id"]
                        else "WRONG_RETURN"
                    ),
                    "destination_node_id": destination,
                    "constituent_edge_ids": list(
                        M._macro_path_after_choice(graph, edge["edge_id"], destination)
                    ),
                }
            candidate_outcomes = []
            for candidate_index, candidate_name in enumerate(C.STAGE_B_LOCAL_CANDIDATE_IDS):
                edge = None if candidate_name == "hold" else edges[candidate_index % len(edges)]
                candidate_outcomes.append({
                    "candidate_index": candidate_index,
                    "candidate_name": candidate_name,
                    "actual_edge_id": None if edge is None else edge["edge_id"],
                    "actual_port_label": None if edge is None else edge["port_label"],
                    "endpoint_node_id": b if edge is None else edge["target_node_id"],
                    "prefix_distance_m": 0.0 if edge is None else edge["edge_cost"],
                    "immediate_contact": False,
                    "successor_viable": True,
                    "stuck": edge is None,
                    "oracle_admissible": edge is not None,
                })
            queries.append({
                "query_id": f"{graph['episode_id']}-query-{query_index}",
                "episode_id": graph["episode_id"],
                "family": graph["family"],
                "role": graph["role"],
                "graph_id": graph["graph_id"],
                "query_index": query_index,
                "depth_since_last_unambiguous_observation": query_index + 4,
                "true_node_id": b,
                "true_next_edge_id": true_edge["edge_id"],
                "true_next_port_label": true_edge["port_label"],
                "prior_visited_node_ids": [a],
                "history_observation_ids": history,
                "history_executed_action_labels": ["GO"] * (len(history) - 1),
                "unresolved_condition_ids": [
                    "CURRENT_FRAME_NEAREST_NODE", "FIXED_WINDOW_SEQUENCE",
                ],
                "candidate_node_ids": nodes,
                "alias_node_ids": [b, a],
                "alias_distractor_node_ids": [a],
                "oracle_admissible_edge_ids": admissible_ids,
                "query_observation_id": query_occurrences[query_index],
                "query_pixel_template_id": graph["nodes"][1]["pixel_template_id"],
                "query_pixel_sha256": graph["nodes"][1]["pixel_sha256"],
                "query_timestamp_s": 20.0 + query_index,
                "goal_visible": False,
                "stage_b_choice_macros": macros,
                "stage_b_local_candidate_outcomes": candidate_outcomes,
            })
    return manifest, queries


def _grid_parameters() -> list[dict]:
    rows = []
    for temperature in C.CALIBRATION_GRID["observation_softmax_temperature"]:
        for compatible in C.CALIBRATION_GRID["action_compatible_edge_probability"]:
            for noise in C.CALIBRATION_GRID["transition_noise_probability"]:
                for entropy in C.CALIBRATION_GRID["normalized_entropy_abstention_threshold"]:
                    rows.append({
                        "observation_softmax_temperature": temperature,
                        "action_compatible_edge_probability": compatible,
                        "transition_noise_probability": noise,
                        "normalized_entropy_abstention_threshold": entropy,
                    })
    return rows


def _calibration(manifest: dict, queries: list[dict], selected_index: int = 0) -> dict:
    episode_ids = sorted(graph["episode_id"] for graph in manifest["graphs"] if graph["role"] == "CALIBRATION")
    query_ids = sorted(row["query_id"] for row in queries if row["role"] == "CALIBRATION")
    grid_results = []
    raw = []
    for grid_index, parameters in enumerate(_grid_parameters()):
        outcomes = []
        for query_offset, query_id in enumerate(query_ids):
            correct = grid_index == selected_index or query_offset < len(query_ids) // 2
            outcome = {
                "grid_index": grid_index,
                "query_id": query_id,
                "edge_correct": correct,
                "localisation_top3": correct,
                "normalized_regret": 0.0 if correct else 1.0,
                "false_confident": not correct,
                "abstained": False,
            }
            raw.append(outcome)
            outcomes.append(outcome)
        grid_results.append({
            "grid_index": grid_index,
            "parameters": parameters,
            "correct_next_edge_accuracy": mean(float(row["edge_correct"]) for row in outcomes),
            "localisation_top3": mean(float(row["localisation_top3"]) for row in outcomes),
            "normalized_graph_distance_regret": mean(row["normalized_regret"] for row in outcomes),
            "false_confident_localisation_rate": mean(float(row["false_confident"]) for row in outcomes),
            "abstention_rate": 0.0,
        })
    return C.attach_content_digest({
        "schema": "occluded_goal_topological_belief_v1.calibration.v1",
        "experiment_id": C.EXPERIMENT_ID,
        "selection_role": "CALIBRATION",
        "calibration_episode_ids": episode_ids,
        "selected_parameters": _grid_parameters()[selected_index],
        "selected_grid_index": selected_index,
        "grid_results": grid_results,
        "grid_query_results": raw,
    })


def _beliefs(manifest: dict, queries: list[dict], scenario: str = "map") -> list[dict]:
    graphs = {graph["graph_id"]: graph for graph in manifest["graphs"]}
    selected_threshold = 0.70 if scenario == "topk" else 0.25
    rows = []
    for query in queries:
        if query["role"] != "DEVELOPMENT_HELDOUT":
            continue
        graph = graphs[query["graph_id"]]
        nodes = query["candidate_node_ids"]
        similarities = [0.0] * len(nodes)
        likelihood = [1.0 / len(nodes)] * len(nodes)
        for condition in C.CONDITION_IDS:
            if condition in {"CURRENT_FRAME_NEAREST_NODE"}:
                prior = [1.0 / len(nodes)] * len(nodes)
            elif condition in {"FIXED_WINDOW_SEQUENCE", "SHUFFLED_ACTION_HISTORY"}:
                prior = [0.96, 0.01, 0.01, 0.01, 0.01]
            elif condition == "MAP_FILTER" and scenario != "map":
                prior = [0.96, 0.01, 0.01, 0.01, 0.01]
            elif condition in {"TOP_K_BELIEF", "FULL_BELIEF"} and scenario == "topk":
                prior = [0.1125, 0.55, 0.1125, 0.1125, 0.1125]
            else:
                prior = [0.01, 0.96, 0.01, 0.01, 0.01]
            condition_likelihood = (
                [1.0 / len(nodes)] * len(nodes)
                if condition == "NO_OBSERVATION_LIKELIHOOD" else likelihood
            )
            products = [left * right for left, right in zip(prior, condition_likelihood)]
            total = sum(products)
            pre = [value / total for value in products]
            order = M._ranking(nodes, pre)
            if condition in {"CURRENT_FRAME_NEAREST_NODE", "FIXED_WINDOW_SEQUENCE", "MAP_FILTER"}:
                posterior = [1.0 if index == order[0] else 0.0 for index in range(len(nodes))]
            elif condition == "TOP_K_BELIEF":
                keep = set(order[:3])
                keep_total = sum(pre[index] for index in keep)
                posterior = [pre[index] / keep_total if index in keep else 0.0 for index in range(len(nodes))]
            elif condition == "ORACLE_PLACE_IDENTITY":
                true_index = nodes.index(query["true_node_id"])
                posterior = [1.0 if index == true_index else 0.0 for index in range(len(nodes))]
            else:
                posterior = pre
            selected_index = M._ranking(nodes, posterior)[0]
            selected_port, selected_edge = M._route_choice(nodes, posterior, M._graph_tables(graph))
            entropy = M._entropy(posterior)
            action_count = len(query["history_executed_action_labels"])
            mapping = list(range(action_count))
            if condition == "SHUFFLED_ACTION_HISTORY":
                mapping = list(M.deterministic_action_history_position_mapping(query["query_id"], action_count))
            rows.append({
                "query_id": query["query_id"],
                "condition_id": condition,
                "node_ids": nodes,
                "observation_similarities": similarities,
                "observation_likelihoods": condition_likelihood,
                "transition_prior_probabilities": prior,
                "preprojection_probabilities": pre,
                "posterior_probabilities": posterior,
                "selected_node_id": nodes[selected_index],
                "selected_edge_id": selected_edge,
                "selected_port_label": selected_port,
                "normalized_entropy": entropy,
                "abstained": entropy > selected_threshold,
                "action_history_position_mapping": mapping,
            })
    return rows


def _stage_b_trace(manifest: dict, queries: list[dict], stage_a: dict) -> list[dict]:
    heldout = [graph for graph in manifest["graphs"] if graph["role"] == "DEVELOPMENT_HELDOUT"]
    query_by_episode: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        if query["role"] == "DEVELOPMENT_HELDOUT":
            query_by_episode[query["episode_id"]].append(query)
    rows: list[dict] = []
    family_offsets: dict[str, int] = defaultdict(int)
    heldout_offsets = {}
    for graph in heldout:
        heldout_offsets[graph["episode_id"]] = family_offsets[graph["family"]]
        family_offsets[graph["family"]] += 1
    for condition in C.STAGE_B_CONDITION_IDS:
        for graph in heldout:
            family_offset = heldout_offsets[graph["episode_id"]]
            fail_current = condition == "CURRENT_FRAME_NEAREST_NODE" and family_offset == 0
            source = {
                "CURRENT_FRAME_NEAREST_NODE": "CURRENT_FRAME_NEAREST_NODE",
                "STRONGEST_STAGE_A_MEMORY": stage_a["decision"]["strongest_memory_condition"],
                "ORACLE_PLACE_IDENTITY": "ORACLE_PLACE_IDENTITY",
            }[condition]
            episode_queries = sorted(query_by_episode[graph["episode_id"]], key=lambda row: row["query_index"])
            node_ids = [node["node_id"] for node in graph["nodes"]]
            node_by_id = {node["node_id"]: node for node in graph["nodes"]}
            edge_by_id = {edge["edge_id"]: edge for edge in graph["edges"]}
            params = stage_a["calibration_binding"]["selected_parameters"]
            carried = [1.0 / len(node_ids)] * len(node_ids)
            visited: list[str] = []
            recovery_pending = False
            nonmovement = 0
            previous_post_ids: list[str] | None = None
            previous_post_nodes: list[str] | None = None
            previous_actions: list[str | None] | None = None
            previous_actual = graph["stage_b_decision_start_node_id"]
            execution_id = f"{condition}-{graph['episode_id']}"
            maximum_decisions = 2 if fail_current else 8
            for decision_index in range(maximum_decisions):
                query_index = 0 if fail_current else decision_index
                query = episode_queries[query_index]
                if decision_index == 0:
                    prelude = [edge_by_id[value] for value in graph["stage_b_prelude_edge_ids"]]
                    update_nodes = [graph["stage_b_start_node_id"], *[edge["target_node_id"] for edge in prelude]]
                    update_actions: list[str | None] = [None, *[edge["executed_action_label"] for edge in prelude]]
                    update_ids = [f"{execution_id}-prelude-{index}" for index in range(len(update_nodes))]
                    for node in update_nodes[:-1]:
                        if node not in visited:
                            visited.append(node)
                else:
                    assert previous_post_ids is not None and previous_post_nodes is not None and previous_actions is not None
                    update_ids = previous_post_ids
                    update_nodes = previous_post_nodes
                    update_actions = previous_actions
                update_shas = [node_by_id[node]["pixel_sha256"] for node in update_nodes]
                similarities_rows = []; likelihood_rows = []; prior_rows = []; pre_rows = []; posterior_rows = []
                for update_offset, (actual_node, action) in enumerate(zip(update_nodes, update_actions)):
                    forced = graph["goal_node_id"] if fail_current and update_offset == len(update_nodes) - 1 else actual_node
                    similarities = [1.0 if node == forced else -1.0 for node in node_ids]
                    likelihood = M._softmax(similarities, params["observation_softmax_temperature"])
                    if condition == "CURRENT_FRAME_NEAREST_NODE":
                        prior = [1.0 / len(node_ids)] * len(node_ids)
                    elif condition == "ORACLE_PLACE_IDENTITY":
                        prior = [float(node == actual_node) for node in node_ids]
                    elif action is None:
                        prior = list(carried)
                    else:
                        prior = M._stage_b_transition(
                            carried, action, node_ids, graph,
                            params["action_compatible_edge_probability"],
                            params["transition_noise_probability"],
                        )
                    product = [a * b for a, b in zip(prior, likelihood)]
                    total = sum(product); pre = [value / total for value in product]
                    if condition == "ORACLE_PLACE_IDENTITY":
                        posterior = [float(node == actual_node) for node in node_ids]
                    elif condition == "CURRENT_FRAME_NEAREST_NODE" or source == "MAP_FILTER":
                        order = M._ranking(node_ids, pre)
                        posterior = [float(index == order[0]) for index in range(len(node_ids))]
                    else:
                        posterior = pre
                    carried = posterior
                    similarities_rows.append(similarities); likelihood_rows.append(likelihood)
                    prior_rows.append(prior); pre_rows.append(pre); posterior_rows.append(posterior)
                belief = posterior_rows[-1]
                selected_node = node_ids[M._ranking(node_ids, belief)[0]]
                proposed_port, witness_edge = M._route_choice(node_ids, belief, M._graph_tables(graph))
                entropy = M._entropy(belief); abstained = entropy > params["normalized_entropy_abstention_threshold"]
                selected_port = None if abstained else proposed_port
                mask = [outcome["candidate_index"] for outcome in query["stage_b_local_candidate_outcomes"] if outcome["oracle_admissible"]]
                if abstained:
                    disposition = "ENTROPY_ABSTENTION"
                elif witness_edge is None:
                    disposition = "NO_ROUTE_WITNESS"
                elif not mask:
                    disposition = "NO_ORACLE_ADMISSIBLE_LOCAL_CANDIDATE"
                else:
                    disposition = "EXECUTED_CANDIDATE"
                acting = disposition == "EXECUTED_CANDIDATE"
                scores: list[float] = []
                chosen = None; chosen_outcome = None
                if acting:
                    true_candidate = min(
                        outcome["candidate_index"] for outcome in query["stage_b_local_candidate_outcomes"]
                        if outcome["actual_edge_id"] == query["true_next_edge_id"] and outcome["oracle_admissible"]
                    )
                    scores = [float(-index) for index in range(len(C.STAGE_B_LOCAL_CANDIDATE_IDS))]
                    scores[true_candidate] = 10.0
                    chosen = min(mask, key=lambda index: (-scores[index], index))
                    chosen_outcome = query["stage_b_local_candidate_outcomes"][chosen]
                    macro = query["stage_b_choice_macros"][chosen_outcome["actual_edge_id"]]
                    constituent_edges = [edge_by_id[value] for value in macro["constituent_edge_ids"]]
                    constituent_edge_ids = [edge["edge_id"] for edge in constituent_edges]
                    constituent_nodes = [query["true_node_id"], *[edge["target_node_id"] for edge in constituent_edges]]
                    constituent_actions = [edge["executed_action_label"] for edge in constituent_edges]
                    constituent_costs = [edge["edge_cost"] for edge in constituent_edges]
                    constituent_ids = [f"{execution_id}-decision-{decision_index}-post-{index}" for index in range(len(constituent_edges))]
                    constituent_shas = [node_by_id[node]["pixel_sha256"] for node in constituent_nodes[1:]]
                    actual_after = macro["destination_node_id"]
                    reobs_id = constituent_ids[-1]; reobs_sha = constituent_shas[-1]
                else:
                    constituent_edge_ids = []; constituent_nodes = [query["true_node_id"]]
                    constituent_actions = []; constituent_costs = []; constituent_ids = []; constituent_shas = []
                    actual_after = query["true_node_id"]
                    reobs_id = f"{execution_id}-decision-{decision_index}-noop"
                    reobs_sha = node_by_id[actual_after]["pixel_sha256"]
                distances = M._graph_tables(graph)["distance"]
                before = distances[query["true_node_id"]]; after = distances[actual_after]
                executed_correct = bool(acting and chosen_outcome["actual_edge_id"] == query["true_next_edge_id"] and before - after > 0.0)
                recovery = recovery_pending and executed_correct
                recovery_before = recovery_pending
                recovery_pending = False if executed_correct else True
                nonmovement = 0 if acting else nonmovement + 1
                goal_reached = actual_after == graph["goal_node_id"]
                if goal_reached:
                    terminal_reason = "GOAL_REACHED"
                elif nonmovement == 2:
                    terminal_reason = "NONMOVEMENT_LIMIT"
                elif decision_index == 15:
                    terminal_reason = "DECISION_BUDGET_EXHAUSTED"
                else:
                    terminal_reason = None
                terminal = terminal_reason is not None
                wrong_confident = not abstained and selected_port != query["true_next_port_label"]
                false_merge = wrong_confident and selected_node != query["true_node_id"] and node_by_id[selected_node]["alias_group_id"] == node_by_id[query["true_node_id"]]["alias_group_id"]
                false_loop = wrong_confident and selected_node != query["true_node_id"] and selected_node in visited
                rows.append({
                    "step_id": f"{execution_id}-{decision_index}", "execution_id": execution_id,
                    "episode_id": graph["episode_id"], "family": graph["family"], "condition_id": condition,
                    "source_condition_id": source, "decision_index": decision_index, "query_id": query["query_id"], "query_index": query_index,
                    "selected_port_label": selected_port, "true_next_port_label": query["true_next_port_label"], "abstained": abstained,
                    "observation_id": update_ids[-1], "observation_pixel_sha256": update_shas[-1],
                    "actual_node_id_before": query["true_node_id"], "actual_node_id_after": actual_after,
                    "actual_alias_group_id": node_by_id[query["true_node_id"]]["alias_group_id"], "selected_node_id": selected_node,
                    "selected_alias_group_id": node_by_id[selected_node]["alias_group_id"], "prior_visited_node_ids": list(visited),
                    "belief_node_ids": node_ids, "belief_probabilities": belief, "normalized_entropy": entropy,
                    "entropy_threshold": params["normalized_entropy_abstention_threshold"], "belief_proposed_port_label": proposed_port,
                    "belief_selected_edge_id": witness_edge, "filter_update_observation_ids": update_ids,
                    "filter_update_pixel_sha256s": update_shas, "filter_update_node_ids": update_nodes,
                    "filter_update_incoming_action_labels": update_actions,
                    "filter_update_observation_similarities": similarities_rows,
                    "filter_update_observation_likelihoods": likelihood_rows,
                    "filter_update_transition_prior_probabilities": prior_rows,
                    "filter_update_preprojection_probabilities": pre_rows,
                    "filter_update_posterior_probabilities": posterior_rows,
                    "stored_relative_waypoint": None if witness_edge is None else edge_by_id[witness_edge]["relative_waypoint"],
                    "action_disposition": disposition, "ranker_called": acting, "local_candidate_scores": scores,
                    "local_oracle_admissible_candidate_indices": mask, "local_candidate_index": chosen,
                    "local_candidate_name": None if chosen_outcome is None else chosen_outcome["candidate_name"],
                    "local_candidate_score": None if chosen is None else scores[chosen],
                    "local_execution_success": bool(acting and chosen_outcome["actual_port_label"] == proposed_port),
                    "ranker_latency_ms": 0.5 if acting else 0.0, "planning_latency_ms": 1.0 if acting else 0.2,
                    "ranker_previous_applied_command": [0.0, 0.0, 0.0],
                    "ranker_control_history": [[0.0, 0.0] for _ in range(15)],
                    "selected_candidate_applied_commands": ([[0.0, 0.0, 0.0] for _ in range(5)] if acting else []),
                    "post_decision_previous_applied_command": [0.0, 0.0, 0.0],
                    "post_decision_control_history": [[0.0, 0.0] for _ in range(15)],
                    "executed_port_label": None if chosen_outcome is None else chosen_outcome["actual_port_label"],
                    "executed_choice_edge_id": None if chosen_outcome is None else chosen_outcome["actual_edge_id"],
                    "constituent_node_ids": constituent_nodes, "constituent_edge_ids": constituent_edge_ids,
                    "constituent_action_labels": constituent_actions, "constituent_edge_costs_m": constituent_costs,
                    "constituent_observation_ids": constituent_ids, "constituent_observation_pixel_sha256s": constituent_shas,
                    "reobservation_id": reobs_id, "reobservation_pixel_sha256": reobs_sha,
                    "oracle_admissible_execution": acting, "geodesic_distance_before_m": before,
                    "geodesic_distance_after_m": after, "decision_budget": 16,
                    "prelude_distance_m": sum(edge_by_id[value]["edge_cost"] for value in graph["stage_b_prelude_edge_ids"]),
                    "consecutive_nonmovement_decisions": nonmovement, "recovery_pending_before": recovery_before,
                    "recovery_pending_after": recovery_pending, "terminal_reason": terminal_reason,
                    "false_confident_wrong_turn": wrong_confident, "false_place_merge": false_merge,
                    "false_loop_closure": false_loop, "geodesic_progress_m": before - after,
                    "replan": not terminal, "recovery": recovery, "immediate_contact": False,
                    "stuck": terminal_reason == "NONMOVEMENT_LIMIT", "executed_distance_m": sum(constituent_costs),
                    "oracle_path_distance_m": M._stage_b_oracle_cycle_distance(graph, episode_queries),
                    "terminal": terminal, "goal_reached": goal_reached,
                })
                for node in constituent_nodes[:-1] if acting else [query["true_node_id"]]:
                    if node not in visited:
                        visited.append(node)
                previous_post_ids = constituent_ids if acting else [reobs_id]
                previous_post_nodes = constituent_nodes[1:] if acting else [actual_after]
                previous_actions = constituent_actions if acting else ["NO_OP"]
                previous_actual = actual_after
    return rows


class OccludedGoalTopologicalBeliefMetricsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.queries = _fixture()

    def test_exact_authority_and_map_decision_regenerate(self) -> None:
        calibration = _calibration(self.manifest, self.queries)
        beliefs = _beliefs(self.manifest, self.queries, "map")
        metrics = M.recompute_stage_a_metrics(self.manifest, self.queries, beliefs, calibration)
        self.assertEqual(len(self.queries), 768)
        self.assertEqual(len(beliefs), 1152)
        self.assertEqual(len(calibration["grid_query_results"]), 18432)
        self.assertEqual(metrics["decision"]["primary_classification"], "TOPOLOGICAL_MAP_SUFFICIENT")
        self.assertTrue(M.stage_a_authorizes_stage_b(metrics))
        self.assertEqual(metrics["conditions"]["MAP_FILTER"]["aggregate"]["correct_next_edge_accuracy"], 1.0)
        self.assertEqual(metrics["conditions"]["CURRENT_FRAME_NEAREST_NODE"]["aggregate"]["correct_next_edge_accuracy"], 0.0)

    def test_full_signal_and_top_k_only_fail_closed(self) -> None:
        calibration = _calibration(self.manifest, self.queries)
        full = M.recompute_stage_a_metrics(
            self.manifest, self.queries, _beliefs(self.manifest, self.queries, "full"), calibration,
        )
        self.assertEqual(full["decision"]["primary_classification"], "TOPOLOGICAL_FULL_BELIEF_SIGNAL")
        self.assertTrue(M.stage_a_authorizes_stage_b(full))
        high_threshold_calibration = _calibration(self.manifest, self.queries, selected_index=3)
        top_k = M.recompute_stage_a_metrics(
            self.manifest, self.queries, _beliefs(self.manifest, self.queries, "topk"), high_threshold_calibration,
        )
        self.assertEqual(top_k["decision"]["primary_classification"], "TOPOLOGICAL_PERSISTENT_GATE_UNRESOLVED")
        self.assertFalse(M.stage_a_authorizes_stage_b(top_k))

    def test_short_history_exact_boundary(self) -> None:
        reference = {
            "correct_next_edge_accuracy": 0.90,
            "localisation_top3": 0.90,
            "normalized_graph_distance_regret": 0.10,
        }
        boundary = {
            "correct_next_edge_accuracy": 0.87,
            "localisation_top3": 0.85,
            "normalized_graph_distance_regret": 0.13,
        }
        self.assertTrue(M._short_history_matches(boundary, reference))
        for field, value in (
            ("correct_next_edge_accuracy", 0.869999),
            ("localisation_top3", 0.849999),
            ("normalized_graph_distance_regret", 0.130001),
        ):
            failed = dict(boundary)
            failed[field] = value
            self.assertFalse(M._short_history_matches(failed, reference), field)

    def test_raw_evidence_tampering_is_rejected(self) -> None:
        calibration = _calibration(self.manifest, self.queries)
        tampered = copy.deepcopy(calibration)
        tampered["grid_query_results"][0]["edge_correct"] = False
        tampered = C.attach_content_digest(tampered)
        with self.assertRaisesRegex(M.OccludedGoalMetricsError, "aggregate differs"):
            M.validate_calibration(tampered, self.manifest, self.queries)
        beliefs = _beliefs(self.manifest, self.queries, "map")
        beliefs[0]["selected_edge_id"] = "forged-edge"
        with self.assertRaisesRegex(M.OccludedGoalMetricsError, "selected edge/port"):
            M.validate_belief_rows(beliefs, self.queries, self.manifest)

    def test_phase_query_and_loop_tampering_is_rejected(self) -> None:
        overlap_tamper = copy.deepcopy(self.manifest)
        overlap_tamper["identity_disjointness"]["comparisons"]["scene_id_overlap_count"] = 1
        overlap_tamper = C.attach_content_digest(overlap_tamper)
        with self.assertRaisesRegex(M.OccludedGoalMetricsError, "overlap is nonzero"):
            M.validate_graph_manifest(overlap_tamper)
        graph_tamper = copy.deepcopy(self.manifest)
        loop = next(graph for graph in graph_tamper["graphs"] if graph["family"] == "LOOP_ALIAS")
        loop["family_adequacy_witness"]["cycle_edge_ids"].reverse()
        graph_tamper = C.attach_content_digest(graph_tamper)
        with self.assertRaisesRegex(M.OccludedGoalMetricsError, "cycle evidence"):
            M.validate_graph_manifest(graph_tamper)
        query_tamper = copy.deepcopy(self.queries)
        query_tamper[0]["query_observation_id"] = self.manifest["graphs"][0]["nodes"][0]["keyframe_observation_id"]
        query_tamper[0]["history_observation_ids"][-1] = query_tamper[0]["query_observation_id"]
        with self.assertRaisesRegex(M.OccludedGoalMetricsError, "Phase-A-disjoint"):
            M.validate_query_rows(query_tamper, self.manifest)

    def test_stage_b_reduction(self) -> None:
        stage_a = M.recompute_stage_a_metrics(
            self.manifest,
            self.queries,
            _beliefs(self.manifest, self.queries, "map"),
            _calibration(self.manifest, self.queries),
        )
        trace = _stage_b_trace(self.manifest, self.queries, stage_a)
        stage_b = M.recompute_stage_b_metrics(
            self.manifest,
            self.queries,
            trace,
            stage_a,
        )
        self.assertTrue(stage_b["gate"]["pass"])
        self.assertEqual(stage_b["conditions"]["STRONGEST_STAGE_A_MEMORY"]["goals_reached"], 16)
        self.assertEqual(stage_b["decision"]["primary_classification"], "OCCLUDED_GOAL_TOPOLOGICAL_NAVIGATION_SIGNAL")
        tampered = copy.deepcopy(trace)
        acting = next(row for row in tampered if row["ranker_called"])
        acting["constituent_edge_costs_m"][0] += 0.25
        with self.assertRaisesRegex(M.OccludedGoalMetricsError, "constituent path evidence"):
            M.recompute_stage_b_metrics(self.manifest, self.queries, tampered, stage_a)
        tampered = copy.deepcopy(trace)
        acting = next(row for row in tampered if row["ranker_called"])
        alternate = next(index for index in acting["local_oracle_admissible_candidate_indices"] if index != acting["local_candidate_index"])
        acting["local_candidate_scores"][alternate] = acting["local_candidate_score"] + 1.0
        with self.assertRaisesRegex(M.OccludedGoalMetricsError, "masked score argmax"):
            M.recompute_stage_b_metrics(self.manifest, self.queries, tampered, stage_a)

    def test_stage_b_classification_precedence_boundaries(self) -> None:
        self.assertEqual(
            M._stage_b_classification(True, 16),
            "OCCLUDED_GOAL_TOPOLOGICAL_NAVIGATION_SIGNAL",
        )
        self.assertEqual(
            M._stage_b_classification(False, 12),
            "PLACE_BELIEF_INTERFACE_NO_GO",
        )
        self.assertEqual(
            M._stage_b_classification(False, 11),
            "LOCAL_EXECUTION_INTERFACE_NO_GO",
        )


if __name__ == "__main__":
    unittest.main()
