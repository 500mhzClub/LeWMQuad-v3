"""Frozen pure contract for OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1.

This module contains declarations and validation only.  Importing it cannot
open a scientific input, construct a model, render a scene, or execute a
filter.  The experiment is development-only and evaluates JEPA place belief
under an oracle, immutable topology; it is not an online-memory or safety
claim.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class OccludedGoalContractError(ValueError):
    """Raised when a prospective contract value drifts from frozen authority."""


EXPERIMENT_ID = "OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1"
STATUS = "SOURCE_ONLY_NOT_EXECUTED"
DEVELOPMENT_ONLY = True
SOURCE_PARENT_COMMIT = "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
CONTRACT_FREEZE_COMMIT_SUBJECT = "Freeze occluded-goal topological belief experiment"
RESULT_COMMIT_SUBJECT = "Evaluate occluded-goal topological belief experiment"

EPISODE_COUNT = 96
QUERIES_PER_EPISODE = 8
FAMILY_IDS = (
    "REPEATED_CORRIDOR",
    "MIRRORED_JUNCTION",
    "LOOP_ALIAS",
    "REPEATED_ROOM",
)
EPISODES_PER_FAMILY = 24
SPLIT_ROLE_IDS = ("FIT", "CALIBRATION", "DEVELOPMENT_HELDOUT")
SPLIT_EPISODE_COUNTS = {"FIT": 64, "CALIBRATION": 16, "DEVELOPMENT_HELDOUT": 16}
SPLIT_FAMILY_EPISODE_COUNTS = {"FIT": 16, "CALIBRATION": 4, "DEVELOPMENT_HELDOUT": 4}
HELDOUT_QUERY_COUNT = 16 * QUERIES_PER_EPISODE
CALIBRATION_QUERY_COUNT = 16 * QUERIES_PER_EPISODE
TOTAL_QUERY_COUNT = EPISODE_COUNT * QUERIES_PER_EPISODE
QUERY_INDEX_VALUES = tuple(range(8))
DEPTH_SINCE_LAST_UNAMBIGUOUS_VALUES = tuple(range(4, 12))
CONSTRUCTED_SET_CAVEAT = (
    "This is a constructed perceptual-aliasing challenge set, not an estimate "
    "of natural alias prevalence."
)
IDENTITY_DOMAIN = "ogtb-v1"
SCENE_ID_PREFIX = "ogtb-v1-scene-"
EPISODE_ID_PREFIX = "ogtb-v1-episode-"
EPISODE_PATH_ID_PREFIX = "ogtb-v1-path-"
PROCEDURAL_SEED_BASE = 5712627296233390080
PROCEDURAL_SEED_VALUES = tuple(range(PROCEDURAL_SEED_BASE, PROCEDURAL_SEED_BASE + EPISODE_COUNT))

PRIOR_PANEL_EXCLUSION_AUTHORITY = {
    "schema": "occluded_goal_topological_belief_v1.prior_panel_exclusion_authority.v1",
    "identity_domain": IDENTITY_DOMAIN,
    "scene_id_prefix": SCENE_ID_PREFIX,
    "episode_id_prefix": EPISODE_ID_PREFIX,
    "episode_path_id_prefix": EPISODE_PATH_ID_PREFIX,
    "procedural_seed_minimum": PROCEDURAL_SEED_VALUES[0],
    "procedural_seed_maximum": PROCEDURAL_SEED_VALUES[-1],
    "authorities": [
        {
            "name": "safe_local_waypoint_panel",
            "path": ".generated/safe_local_waypoint_purpose_built_v1/state_manifest.json",
            "bytes": 51066,
            "sha256": "da67309c073f60d74e4b85427237b19691552a542136e6ddb95939f14b4c5c37",
        },
        {
            "name": "oracle_branch_pilot_v1_2",
            "path": ".generated/go2_oracle_branch_pilot_v1_2/state_manifest.json",
            "bytes": 20147,
            "sha256": "1f76afa94a66eaec0049559f9a47d48a4b50543c0ad4c6cec5060ff5b5ab0d9e",
        },
        {
            "name": "oracle_branch_pilot_v1",
            "path": ".generated/go2_oracle_branch_pilot_v1/identity_manifest.json",
            "bytes": 13073,
            "sha256": "7ebc6aac4eed73d38dec1a6f2be8272f442522d3c43f7a1d2f491b3eafe996c6",
        },
        {
            "name": "counterfactual_predictor_panel",
            "path": "docs/lewm_go2_world_model_counterfactual_calibration_scene_panel_v1_2026-08-02.json",
            "bytes": 8466,
            "sha256": "ad4467e54427c661834755e8062ccea1276602eeab5d1abafa4db9ac79d78581",
        },
        {
            "name": "non_greedy_local_subgoal_panel",
            "path": "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/non_greedy_local_subgoal_jepa_planning_v1/panel_manifest.json",
            "bytes": 249155,
            "sha256": "e464f8df0e651bde541bece687d1b61b8baf4daa246188e9e16dca622d0bfde3",
        },
        {
            "name": "memory_role_place_manifest",
            "path": ".generated/go2_memory_role_place_triplet_index_v1/manifest.json",
            "bytes": 42308,
            "sha256": "a5997d93838419cabaaf8e262db70ed51f6f928195f1a312cadc4768f74ca6ca",
        },
        {
            "name": "memory_role_place_train",
            "path": ".generated/go2_memory_role_place_triplet_index_v1/train.jsonl",
            "bytes": 4687348,
            "sha256": "72044c597286631be6133b45663ef975e222cd10d3f0cee1d0a9c038f0d422b6",
        },
        {
            "name": "memory_role_place_checkpoint_selection",
            "path": ".generated/go2_memory_role_place_triplet_index_v1/checkpoint_selection.jsonl",
            "bytes": 473508,
            "sha256": "a628a1047b6f15223a4fd7d30c5c87fa1914efef0955d70d9bd2f5330c77dcb0",
        },
        {
            "name": "recurrent_predictor_train",
            "path": ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/train.jsonl",
            "bytes": 10328000,
            "sha256": "aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77",
        },
        {
            "name": "recurrent_predictor_validation",
            "path": ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_integrity/val.jsonl",
            "bytes": 1317888,
            "sha256": "83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6",
        },
    ],
    "prior_projection": {
        "scene_identity_count": 1295,
        "scene_identity_canonical_json_sha256": "9106d294415106ae97ff59ae15937c562b497b64f6eff2e57e194120ffd3b35f",
        "scene_identity_sha256_count": 1487,
        "scene_identity_sha256_canonical_json_sha256": "297c988960e418a15e6887b208ea3c474e8e7e3d543c63f931ee2dd63cc2be56",
        "episode_or_state_identity_count": 300,
        "episode_or_state_identity_canonical_json_sha256": "f974c42c8a923bfee776486920198f89d89aab4556f86bd65878dd23bbe1177f",
        "numeric_seed_count": 188,
        "numeric_seed_canonical_json_sha256": "ab1dbb1b641489979a0721f4ba0dab99d147f4d903c72ac1414872955039ccf5",
        "textual_path_geometry_or_source_identity_count": 131764,
        "textual_path_geometry_or_source_identity_canonical_json_sha256": "b988e15e8ed0c5a0a96462546373729c616147b91f014eea47471db0a8f433be",
        "structured_waypoint_or_sequence_path_count": 40,
        "structured_waypoint_or_sequence_path_canonical_json_sha256": "a9d63c5c7a8aae8c93c27c5d08c5e306779dccd790f8c2173df3a72729f77cf3",
        "identity_domain_occurrences": 0,
        "reserved_seed_overlap_count": 0,
    },
    "projection_algorithm": "SHA-256 of the canonical compact-JSON sorted-unique array with no trailing LF",
    "comparison": "exact set intersection for text/numeric identities plus SHA256(current scene_id) against registered predecessor scene hashes",
}

PORT_LABEL_ORDER = ("LEFT", "STRAIGHT", "RIGHT", "REVERSE")
STAGE_B_LOCAL_CANDIDATE_IDS = (
    "straight_fast", "straight_medium", "straight_slow", "arc_left",
    "arc_right", "turn_left", "turn_right", "turn_left_then_go",
    "turn_right_then_go", "go_then_turn_left", "reverse_then_turn", "hold",
)
CONDITION_IDS = (
    "CURRENT_FRAME_NEAREST_NODE",
    "FIXED_WINDOW_SEQUENCE",
    "MAP_FILTER",
    "TOP_K_BELIEF",
    "FULL_BELIEF",
    "ORACLE_PLACE_IDENTITY",
    "NO_ACTION_CONSISTENCY",
    "SHUFFLED_ACTION_HISTORY",
    "NO_OBSERVATION_LIKELIHOOD",
)
PRIMARY_CONDITION_IDS = CONDITION_IDS[:6]
ABLATION_CONDITION_IDS = CONDITION_IDS[6:]
TOP_K = 3
FIXED_WINDOW_OBSERVATIONS = 4
FIXED_WINDOW_TRANSITIONS = 3
PROBABILITY_FLOOR = 1.0e-12
ECE_BIN_COUNT = 10

CALIBRATION_GRID = {
    "observation_softmax_temperature": (0.01, 0.03, 0.10, 0.30),
    "action_compatible_edge_probability": (0.70, 0.85, 0.95),
    "transition_noise_probability": (0.01, 0.05, 0.15),
    "normalized_entropy_abstention_threshold": (0.25, 0.40, 0.55, 0.70),
}
CALIBRATION_GRID_ORDER = (
    "observation_softmax_temperature",
    "action_compatible_edge_probability",
    "transition_noise_probability",
    "normalized_entropy_abstention_threshold",
)
CALIBRATION_SELECTION = {
    "scope": "FULL_BELIEF on the 16 CALIBRATION episodes only",
    "grid_iteration": "nested in CALIBRATION_GRID_ORDER using each listed value order",
    "selection": (
        "lexicographically maximize correct_next_edge_accuracy, then localisation_top3; "
        "minimize normalized_graph_distance_regret, then false_confident_localisation_rate, "
        "then abstention_rate; exact ties select the first grid tuple"
    ),
    "fit_role": (
        "construction and alias-template integrity plus pre-calibration descriptor sanity "
        "reporting only; FIT never selects a parameter"
    ),
    "grid_results_persisted": 144,
    "grid_query_results_persisted": 144 * CALIBRATION_QUERY_COUNT,
    "raw_grid_query_identity": "grid-major then query_id lexicographic; exact (grid_index, query_id) cross-product",
    "heldout_open_before_selection": False,
}

OBSERVATION_LIKELIHOOD = {
    "encoder": "frozen V-JEPA 2.1 target encoder",
    "checkpoint_sha256": "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6",
    "token_grid": [24, 32, 1024],
    "token_layer_norm": {"axis": 1024, "epsilon": 1.0e-5, "affine": False},
    "token_l2_normalization": {"axis": 1024, "norm_floor": 1.0e-12},
    "score": (
        "after token-wise LayerNorm and L2 normalization, aligned-token cosine for "
        "the matched viewpoint, then arithmetic mean over all 768 spatial tokens"
    ),
    "persisted_arrays": {
        "observations.npz": {"dtype": "uint8", "layout": "N,H,W,RGB"},
        "latents.npz/raw_tokens": {"dtype": "float16", "shape_suffix": [768, 1024], "spatial_grid": [24, 32]},
        "latents.npz/spatial_descriptors": {"dtype": "float32", "shape_suffix": [768, 1024], "spatial_grid": [24, 32]},
        "stage_a_beliefs.jsonl/observation_similarities": {"dtype": "JSON finite number", "shape": "one value per full graph node"},
    },
    "likelihood": "softmax(node_similarity / observation_softmax_temperature)",
    "node_reference_count": 1,
    "query_reference_disjoint": True,
    "goal_or_family_input": False,
}

FILTER_AUTHORITY = {
    "prior": "uniform over every episode-local oracle node before query_index 0",
    "prediction": "predicted[j] = sum_i belief[i] * transition(action, j | i)",
    "transition": (
        "(1-transition_noise_probability)*action_conditioned_graph_kernel + "
        "transition_noise_probability/number_of_nodes"
    ),
    "action_conditioned_graph_kernel": (
        "allocate action_compatible_edge_probability uniformly over compatible outgoing "
        "edges and the remainder uniformly over incompatible outgoing edges; if either "
        "set is empty, redistribute its mass over the nonempty set; a node with no outgoing "
        "edge stays at itself"
    ),
    "update": "normalize(predicted * observation_likelihood) in log space",
    "current_frame": "observation likelihood only; no carried prior",
    "fixed_window": (
        "for the latest four observations and three executed transitions (or the available "
        "prefix), score candidate graph paths of length four, sum path scores by final node "
        "from a fresh uniform prior, "
        "renormalize, and commit the highest-probability final node; carry no belief"
    ),
    "map_filter": "after every update retain only the deterministic MAP node with mass one",
    "top_k_belief": "after every update retain the three highest nodes and renormalize",
    "full_belief": "retain and renormalize every episode-local node after every update",
    "oracle": "unit mass on the registered true node",
    "no_action_consistency": "replace action-conditioned graph kernel by uniform outgoing-edge mass",
    "shuffled_action_history": (
        "for each length-L history (L>=2), set shift=1+(int(SHA256(query_id)[:8],16) mod (L-1)) "
        "and persist mapping[j]=(j+shift) mod L, a deterministic complete rotation derangement; "
        "observations, graph, query order, and true labels remain fixed"
    ),
    "no_observation_likelihood": "replace observation likelihood by the all-ones likelihood",
    "ranking_tie_break": "descending probability, then lexicographically smaller node_id",
    "abstention": (
        "abstain iff -sum(p*log(p))/log(number_of_nodes) is strictly greater than the "
        "calibrated normalized-entropy threshold; equality does not abstain"
    ),
}

ALIASING_AUTHORITY = {
    "eligibility_inputs": (
        "exact geometry, immutable oracle graph, exact renderer, registered alias tokens, "
        "action-compatible paths, and reachability only; never V-JEPA/model outcomes"
    ),
    "families": {
        "REPEATED_CORRIDOR": "at least two locally identical corridor segments at different graph locations require different next ports",
        "MIRRORED_JUNCTION": "matched junction views have different goal connectivity and required next ports",
        "LOOP_ALIAS": "revisited locally indistinguishable loop locations require sequence evidence",
        "REPEATED_ROOM": "identical local room layout and landmarks have different neighborhoods and routes",
    },
    "required_per_query": (
        "the true node belongs to a registered alias group containing at least two distinct "
        "nodes, including a node whose oracle next port differs"
    ),
    "phase_a_memory_provenance": (
        "every posterior-bank node binds its first Phase-A teacher encounter and keyframe; "
        "the ordered traversal persists observation occurrence, arrival edge, executed action, "
        "and timestamp, with consecutive executable teacher edges"
    ),
    "query_capture_disjointness": (
        "each Phase-C query capture has a distinct occurrence identity and later timestamp than "
        "all Phase-A captures while binding the same registered pixel template/hash as its exact aliases"
    ),
    "condition_specific_unresolved": (
        "for one condition and query, at least two graph paths remain exactly consistent with "
        "that condition's permitted registered alias tokens, observation window/carry state, "
        "and action history, end at different nodes, and require different next ports"
    ),
    "identifiability": (
        "the current frame is ambiguous, while the complete registered observation/action "
        "history distinguishes the true node at every registered query; CURRENT_FRAME and "
        "FIXED_WINDOW remain condition-specifically unresolved while FULL_BELIEF is resolved"
    ),
    "leakage_forbidden": (
        "family-ID texture/palette, waypoint, action schedule, camera, node index, goal pose, "
        "and query-depth leakage; paired aliases match camera yaw and local pose"
    ),
    "selection": "complete eligible population first, deterministic hash order, then roles; no post-feature replacement",
    "caveat": CONSTRUCTED_SET_CAVEAT,
    "family_witness_node_kinds": {
        "REPEATED_CORRIDOR": "CORRIDOR_SEGMENT",
        "MIRRORED_JUNCTION": "MIRRORED_JUNCTION",
        "LOOP_ALIAS": "LOOP_LOCATION",
        "REPEATED_ROOM": "REPEATED_ROOM",
    },
    "oracle_admissibility": (
        "every query binds all physically executable and oracle-admissible outgoing edges at "
        "its true node, requires at least two, and includes the shortest-path next edge; these "
        "labels are evidence only and never localisation inputs"
    ),
}

METRIC_IDS = (
    "localisation_top1",
    "localisation_top3",
    "mean_reciprocal_rank",
    "localisation_nll",
    "multiclass_brier",
    "ece_10_bin",
    "normalized_belief_entropy",
    "false_confident_localisation_rate",
    "false_place_merge_rate",
    "false_loop_closure_rate",
    "correct_next_edge_accuracy",
    "normalized_graph_distance_regret",
    "route_top3",
    "wrong_turn_rate",
    "abstention_rate",
    "correct_abstention_when_unresolved",
)
METRIC_FORMULAS = {
    "probability_floor": PROBABILITY_FLOOR,
    "nll": "mean -log(max(probability_of_true_node, 1e-12))",
    "brier": "mean sum_node (probability - one_hot_true)^2; no class-count division",
    "ece": "10 equal-width confidence bins on max node probability; left-closed/right-open except final includes 1.0; empty bins contribute zero",
    "entropy": "mean -sum(p*log(p))/log(number_of_nodes); one-node entropy is zero",
    "false_confident": "wrong MAP and not abstained, divided by all queries",
    "false_place_merge": "wrong selected node in the true node's registered alias group and not abstained, divided by all queries",
    "false_loop_closure": (
        "wrong selected node in the query's oracle prior-visited-node set while the true node "
        "differs and not abstained, divided by all queries; descriptive frozen-graph association "
        "proxy only, never an actual merge or loop-closure mutation"
    ),
    "route_distribution": "sum nonterminal node posterior mass by that node's unique shortest-path next port to the oracle goal",
    "route_selection": (
        "choose the largest posterior route mass, ties by PORT_LABEL_ORDER; persist as edge witness "
        "the oracle-best edge of the highest-posterior node voting for that port, ties by node_id; "
        "if all mass is terminal, selected port and edge are null"
    ),
    "correct_next_edge": "nonabstained selected route port equals the true node's registered oracle next port",
    "route_top3": "true oracle port is among the three highest route-distribution ports; ties use PORT_LABEL_ORDER",
    "regret": (
        "for the selected port at the true node, (selected successor distance - best successor "
        "distance)/(worst successor distance - best successor distance); denominator-zero is 0; "
        "invalid ports and abstentions are 1; clamp to [0,1]"
    ),
    "wrong_turn": "nonabstained selected port differs from the true oracle port, divided by all queries",
    "correct_abstention": "abstained condition-specific-unresolved queries divided by condition-specific-unresolved queries; null if none",
    "aggregation": "query macro-average (equivalently episode-balanced because every episode has eight queries), plus per-family, per-episode, and per-depth-since-last-unambiguous-observation",
}

ABSOLUTE_GATE = {
    "localisation_top3_minimum": 0.85,
    "correct_next_edge_accuracy_minimum": 0.80,
    "normalized_graph_distance_regret_maximum": 0.20,
    "false_confident_localisation_rate_maximum": 0.10,
    "per_family_correct_next_edge_accuracy_minimum": 0.65,
}
INCREMENTAL_OVER_CURRENT_GATE = {
    "correct_next_edge_accuracy_gain_minimum": 0.20,
    "and_any_of": {
        "localisation_top3_gain_minimum": 0.20,
        "normalized_graph_distance_regret_reduction_minimum": 0.15,
    },
}
INCREMENTAL_OVER_MAP_GATE = {
    "any_of": {
        "correct_next_edge_accuracy_gain_minimum": 0.08,
        "localisation_top3_gain_minimum": 0.10,
        "false_confident_localisation_rate_reduction_minimum": 0.05,
        "normalized_graph_distance_regret_reduction_minimum": 0.05,
    }
}
SHORT_HISTORY_MATCH_GATE = {
    "correct_next_edge_accuracy_maximum_deficit": 0.03,
    "localisation_top3_maximum_deficit": 0.05,
    "normalized_graph_distance_regret_maximum_excess": 0.03,
}
STAGE_A_CLASSIFICATIONS = (
    "TOPOLOGICAL_MAP_SUFFICIENT",
    "TOPOLOGICAL_FULL_BELIEF_SIGNAL",
    "SHORT_HISTORY_SUFFICIENT",
    "VJEPA_PLACE_BELIEF_NO_SIGNAL",
    "TOPOLOGICAL_PERSISTENT_GATE_UNRESOLVED",
)
STAGE_A_PRECEDENCE = (
    "SHORT_HISTORY_SUFFICIENT when FIXED_WINDOW is within 0.03 edge accuracy, 0.05 top3 recall, and 0.03 regret of the strongest absolute-gate-passing persistent condition",
    "otherwise TOPOLOGICAL_MAP_SUFFICIENT when MAP passes the absolute gate and FULL has no material value over MAP",
    "otherwise TOPOLOGICAL_FULL_BELIEF_SIGNAL when FULL passes absolute, current-margin, and MAP-increment gates",
    "otherwise TOPOLOGICAL_PERSISTENT_GATE_UNRESOLVED when any persistent condition passes absolute but none of the preceding registered decisions applies (including TOP_K-only); fail closed and do not authorize Stage B",
    "otherwise VJEPA_PLACE_BELIEF_NO_SIGNAL",
)

STAGE_B_CONDITION_IDS = (
    "CURRENT_FRAME_NEAREST_NODE",
    "STRONGEST_STAGE_A_MEMORY",
    "ORACLE_PLACE_IDENTITY",
)
STAGE_B_EXECUTION_POLICY = {
    "initial_state": (
        "each heldout graph binds stage_b_start_node_id to its observed goal node; all three "
        "conditions independently start at that exact occurrence with an empty visited set and "
        "a fresh uniform filter prior"
    ),
    "prelude": (
        "before decision zero, execute the prospectively registered unbudgeted deterministic "
        "stage_b_prelude_edge_ids from the observed goal through prefix, anchor, branch marker, "
        "and alias tail to stage_b_decision_start_node_id/query zero; persist and process every "
        "fresh observation and action. CURRENT resets at query zero, MAP/FULL carries the complete "
        "prelude posterior, and ORACLE is one-hot on the actual node"
    ),
    "decision_budget": 16,
    "registered_choice_points": 8,
    "maximum_consecutive_nonmovement_decisions": 2,
    "decision_unit": (
        "one decision is one registered query choice, not one constituent graph edge; the oracle "
        "therefore reaches the goal in exactly eight correct choice decisions even though the "
        "registered path contains more than eight constituent edges"
    ),
    "fresh_observation": (
        "at every decision render a fresh RGB occurrence at the actual node, bind its pixel SHA, "
        "and evaluate the frozen V-JEPA target encoder; exact-byte token-cache reuse is permitted "
        "only when keyed by that SHA and the frozen encoder binding"
    ),
    "belief_update": (
        "never reuse Stage-A belief rows: CURRENT_FRAME uses only the fresh likelihood; strongest "
        "MAP/FULL carries its own Stage-B posterior through every constituent action actually "
        "executed by the preceding macro transition; a registered NO_OP is an exact identity "
        "transition with no transition-noise mixing; ORACLE is unit mass on the actual node"
    ),
    "edge_selection": (
        "derive posterior route mass and its edge witness exactly as Stage A; entropy abstention "
        "executes nothing. Otherwise retrieve the belief-witness edge's stored relative waypoint; "
        "the witness may belong to a wrong place hypothesis and is never replaced by the true edge"
    ),
    "local_ranker": (
        "for a nonabstained decision with a route witness and a nonempty preregistered geometry "
        "mask, call the frozen current-visual ranker exactly once, score all twelve candidates, "
        "then select the highest score among the geometry-derived ORACLE-admissible candidate "
        "indices with candidate-index tie-breaking. Pass only fresh current tokens, stored "
        "waypoint, actual prior command/control history, and the frozen bank; never pass future "
        "tokens or route/contact/oracle labels into the score computation"
    ),
    "control_history_evidence": {
        "previous_applied_command_shape": [3],
        "control_history_shape": [15, 2],
        "selected_short_prefix_applied_command_shape": [5, 3],
        "continuity": "post-decision command/history equals the next decision's ranker input",
    },
    "transition": (
        "an acting decision executes the actual edge preregistered for the selected local candidate, "
        "then automatically executes the deterministic shortest registered constituent-edge path "
        "to the next query (or goal) after a correct port, or through the complete wrong branch "
        "and back to the same query after a wrong port. Both paths and all twelve candidate outcomes "
        "are registered in the query ledger before scoring; shortest-path ties use the "
        "lexicographically smallest complete edge-id sequence. Every constituent edge, action, "
        "target node, cost, observation, and filter update is persisted"
    ),
    "nonmovement": (
        "entropy abstention, absence of a posterior route witness, or an empty preregistered local "
        "oracle-admissibility mask keeps the actual node fixed, executes NO_OP, calls no local "
        "ranker, records zero distance/progress, and reobserves unless the two-decision "
        "nonmovement limit is reached"
    ),
    "recovery": (
        "after any wrong-port return or nonmovement decision, recovery remains pending until a "
        "later correct-port macro advances to the next registered query or goal with positive "
        "geodesic progress; only that later row records recovery=true"
    ),
    "terminal_precedence": (
        "GOAL_REACHED after transition; otherwise NONMOVEMENT_LIMIT at two consecutive nonmovement "
        "decisions; otherwise DECISION_BUDGET_EXHAUSTED after decision 15; otherwise continue"
    ),
    "distance": (
        "remaining distance is exact frozen-graph shortest distance to goal; row progress is "
        "distance_before-distance_after; executed distance is the sum of every persisted "
        "constituent edge cost. Oracle path distance is the exact full registered leave-and-return "
        "cycle: prelude plus all eight correct query macros, never the zero graph distance from "
        "the initial goal node to itself"
    ),
    "fresh_execution_count": 16,
    "conditions_share_start_and_graph": True,
    "oracle_admissibility_is_not_a_safety_claim": True,
}
STAGE_B_GATE = {
    "episodes_per_condition": 16,
    "goals_reached_minimum": 12,
    "reach_gain_over_current_minimum": 4,
    "path_efficiency_minimum": 0.70,
    "false_confident_wrong_turn_rate_maximum": ABSOLUTE_GATE["false_confident_localisation_rate_maximum"],
    "oracle_reach_fraction_minimum": 0.80,
    "minimum_goals_reached_per_family": 1,
}
STAGE_B_METRIC_FORMULAS = {
    "goal_reach": "terminal goal_reached, summed over the 16 executions",
    "edge_accuracy": "nonabstained selected port equals registered true next port, divided by all decisions",
    "wrong_turn": "executed candidate port differs from registered true next port, divided by all decisions",
    "path_efficiency": "per execution goal_reached * min(1, oracle_path_distance/(prelude_distance+sum macro executed_distance)), then mean over all 16; zero when not reached",
    "false_confident_wrong_turn": "persisted nonabstained confident wrong-turn count divided by all decisions",
    "oracle_reach_fraction": "strongest-memory goals reached divided by oracle-place goals reached; zero if oracle reaches none",
    "latency_p95": "nearest-rank order statistic ceil(0.95*N), one-indexed",
}
STAGE_B_CLASSIFICATIONS = (
    "OCCLUDED_GOAL_TOPOLOGICAL_NAVIGATION_SIGNAL",
    "PLACE_BELIEF_INTERFACE_NO_GO",
    "LOCAL_EXECUTION_INTERFACE_NO_GO",
)
NEXT_DECISION_BY_CLASSIFICATION = {
    "TOPOLOGICAL_MAP_SUFFICIENT": "run conditional Stage B with MAP_FILTER as the strongest memory condition",
    "TOPOLOGICAL_FULL_BELIEF_SIGNAL": "run conditional Stage B with FULL_BELIEF as the strongest memory condition",
    "SHORT_HISTORY_SUFFICIENT": "construct longer episodes with greater alias-separation depth before persistent-memory work",
    "VJEPA_PLACE_BELIEF_NO_SIGNAL": "SEQUENCE_CONDITIONED_PLACE_LIKELIHOOD_V1",
    "TOPOLOGICAL_PERSISTENT_GATE_UNRESOLVED": "TOPOLOGICAL_BELIEF_DECISION_V1",
    "OCCLUDED_GOAL_TOPOLOGICAL_NAVIGATION_SIGNAL": "ONLINE_TOPOLOGICAL_MEMORY_CONSTRUCTION_V1",
    "PLACE_BELIEF_INTERFACE_NO_GO": "PLACE_BELIEF_INTERFACE_DIAGNOSTIC_V1",
    "LOCAL_EXECUTION_INTERFACE_NO_GO": "LOCAL_EXECUTION_INTERFACE_DIAGNOSTIC_V1",
}

STAGE_B_AUTHORITY = {
    "authorization": "only TOPOLOGICAL_FULL_BELIEF_SIGNAL or TOPOLOGICAL_MAP_SUFFICIENT",
    "episodes": "16 fresh executions derived one-for-one from development-heldout scenes",
    "conditions": list(STAGE_B_CONDITION_IDS),
    "execution_policy": copy.deepcopy(STAGE_B_EXECUTION_POLICY),
    "pipeline": (
        "localize, choose next oracle-graph edge, retrieve its stored relative waypoint, apply "
        "oracle admissibility, rank the frozen local candidate bank with the frozen current-visual "
        "ranker, execute a short prefix, reobserve, and repeat"
    ),
    "frozen_current_visual_checkpoint_sha256": "e1a2a58ff527b4d2bc210f6b1c8fd1d00d87873691db64e8c3ccb2257fa6c127",
    "outcomes": (
        "goal reach, edge accuracy, wrong turns, path efficiency, replans, recovery, false merge/loop "
        "proxies, abstentions, geodesic progress, descriptive contact/stuck, and latency"
    ),
    "positive_wording": "JEPA topological navigation under oracle topology and oracle admissibility",
    "not_safety": True,
}

CLAIMS = {
    "development_only": True,
    "positive_wording": "JEPA place belief under oracle topology",
    "does_not_establish": (
        "deployment safety",
        "learned contact avoidance",
        "physical Go2 safety",
        "online map construction",
        "hidden beacon discovery",
        "novelty",
        "complete maze navigation",
    ),
}
PROHIBITIONS = {
    "predictor_training": 0,
    "safety_model_training": 0,
    "route_ranker_training": 0,
    "custom_python_audit_hooks": 0,
    "custom_startup_or_forensic_framework": 0,
    "online_node_or_edge_mutation_in_stage_a": 0,
    "topological_memory_construction": 0,
    "novelty_or_beacon_discovery": 0,
}

OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "occluded_goal_topological_belief_v1"
)
RUNTIME_OUTPUT_PATHS = {
    "contract": "contract.json",
    "panel_manifest": "panel_manifest.json",
    "split_manifest": "split_manifest.json",
    "graph_manifest": "graph_manifest.json",
    "keyframe_index": "keyframe_index.json",
    "query_ledger": "query_ledger.jsonl",
    "observations": "observations.npz",
    "latent_index": "latent_index.json",
    "latents": "latents.npz",
    "calibration": "calibration.json",
    "stage_a_beliefs": "stage_a_beliefs.jsonl",
    "stage_a_metrics": "stage_a_metrics.json",
    "conditional_stage_b_trace": "stage_b_trace.jsonl",
    "conditional_stage_b_metrics": "stage_b_metrics.json",
    "result": "result.json",
    "report": "result.md",
    "file_hashes": "file_hashes.json",
}
TRACKED_SOURCE_PATHS = (
    "docs/lewm_go2_occluded_goal_topological_belief_v1_contract_2026-09-01.json",
    "docs/lewm_go2_occluded_goal_topological_belief_v1_fixture_2026-09-01.json",
    "docs/lewm_go2_occluded_goal_topological_belief_v1_output_schema_2026-09-01.json",
    "docs/lewm_go2_occluded_goal_topological_belief_v1_preregistration_2026-09-01.md",
    "docs/lewm_go2_occluded_goal_topological_belief_v1_source_closure_2026-09-01.json",
    "lewm/safety/occluded_goal_topological_belief_metrics_v1.py",
    "lewm/safety/occluded_goal_topological_belief_v1_contract.py",
    "lewm/tests/test_evaluate_occluded_goal_topological_belief_v1.py",
    "lewm/tests/test_occluded_goal_topological_belief_metrics_v1.py",
    "lewm/tests/test_occluded_goal_topological_belief_v1_contract.py",
    "lewm/tests/test_run_occluded_goal_topological_belief_v1.py",
    "scripts/evaluate_occluded_goal_topological_belief_v1.py",
    "scripts/run_occluded_goal_topological_belief_v1.py",
)

SOURCE_DEPENDENCY_PATHS = (
    "AGENTS.md",
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "lewm/safety/non_greedy_local_subgoal_jepa_planning_v1_contract.py",
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_result_2026-08-31.json",
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_result_2026-08-31.md",
)
PREDECESSOR_RESULT_BINDING = {
    "result_commit": SOURCE_PARENT_COMMIT,
    "source_freeze_commit": "bbd1556c928c8b917195442ec3f2155d586de1fe",
    "result_json": {
        "path": "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_result_2026-08-31.json",
        "sha256": "ae65c1ba2ea3e42dd6fe342a4524c8ac16dbdc24d80a40f46fca15a4bdb25d8a",
    },
    "result_markdown": {
        "path": "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_result_2026-08-31.md",
        "sha256": "bdcffbfd5a94b312943585d767d426f7cd2c647bb05fd60a3e4ee4b9f504f541",
    },
    "preserved_status": "DEVELOPMENT_EXPLORATORY_COMPLETE",
    "primary_classification": "NON_GREEDY_REACTIVE_OR_KINEMATIC_BASELINE_DOMINANT",
    "secondary_classifications": (
        "CANDIDATE_FUTURE_DERANGEMENT_MATERIAL",
        "FUTURE_TIME_ORDER_DERANGEMENT_MATERIAL",
    ),
    "predictor_substitution_executed": False,
    "preserved_development_classifications": (
        "DEVELOPMENT_SCIENTIFIC_PAYLOAD_RECOVERED",
        "KINEMATIC_BASELINE_DOMINANT",
        "TRUE_FUTURE_PLAN_AWARE_COST_SIGNAL",
        "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE_NOT_SUPPORTED",
        "TWO_STEP_PLAN_AWARE_JEPA_COST_NO_SIGNAL",
        "PROPRIOCEPTIVE_ROUTE_CONTRIBUTION_NOT_SUPPORTED",
        "RAW_LATENT_GOAL_COST_NO_GO",
        "TRUE_FUTURE_LATENT_GOAL_COST_NO_GO",
    ),
    "safety_workstream": "REQUIREMENTS_ACQUISITION_REQUIRED",
    "scientific_interpretation": (
        "current visual geometry solved the local challenge; true futures used candidate-specific "
        "information but added no registered value over the reactive visual ranker"
    ),
    "current_visual_metrics": {
        "pairwise_accuracy": 0.947917,
        "spearman": 0.914773,
        "top1": 1.0,
        "top3": 1.0,
        "normalized_regret": 0.0,
        "oracle_progress_fraction": 1.0,
    },
    "true_future_derangements": {
        "candidate_trajectory_derangement_material": True,
        "future_time_order_derangement_material": True,
    },
    "true_future_jepa_incremental_route_value_supported": False,
    "stage_b_ran": False,
    "exact_conclusion": (
        "Static geometry visible in the current V-JEPA representation is sufficient for the "
        "tested non-greedy local route decisions. Future rollout is not justified for this task."
    ),
}
VJEPA_ENCODER_BINDING = {
    "helper_path": "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "helper_sha256": "c5bb12ddc4711071dbdbac8c2ad6cc4b7528dd8ceb263b752fd539bd954aa9e2",
    "external_repository_commit": "204698b45b3712590f06245fbfba32d3be539812",
    "constructor": "vjepa2_1_vit_large_384",
    "checkpoint_path": "/home/andrewknowles/.cache/vjepa2_1_vitl_dist_vitG_384.pt",
    "checkpoint_size_bytes": 5151198524,
    "checkpoint_sha256": OBSERVATION_LIKELIHOOD["checkpoint_sha256"],
    "token_grid": [24, 32, 1024],
}
STAGE_B_CURRENT_VISUAL_BINDING = {
    "model_source_path": "lewm/safety/non_greedy_local_subgoal_jepa_planning_v1_contract.py",
    "model_source_sha256": "fd698f7ba09cf9f76a8f6f978f133a5369466e4455cf9ad8bdf7ce5fa5351eed",
    "checkpoint_path": (
        "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
        "non_greedy_local_subgoal_jepa_planning_v1/checkpoints/"
        "current_visual_reactive_ranker_final_epoch_060.pt"
    ),
    "checkpoint_size_bytes": 2470669,
    "checkpoint_sha256": STAGE_B_AUTHORITY["frozen_current_visual_checkpoint_sha256"],
    "parameter_count": 204289,
    "predecessor_result_commit": SOURCE_PARENT_COMMIT,
    "predecessor_source_freeze_commit": PREDECESSOR_RESULT_BINDING["source_freeze_commit"],
}
FROZEN_SOURCE_BINDINGS = {
    "vjepa_encoder_helper": copy.deepcopy(VJEPA_ENCODER_BINDING),
    "stage_b_current_visual_ranker": copy.deepcopy(STAGE_B_CURRENT_VISUAL_BINDING),
}
DIRECT_RUNTIME_POLICY = {
    "runner_interpreter": "/home/andrewknowles/TinyQuadJEPA/bin/python",
    "one_foreground_process_per_major_stage": True,
    "ordinary_stdout_stderr": True,
    "ordinary_json_jsonl_npz": True,
    "sha256_final_files": True,
    "independent_pure_metric_reducer": True,
    "custom_open_auditing": False,
    "launcher_child_roles": False,
    "preexecution_accounting": False,
    "terminal_custody_bundles": False,
    "custom_finalizers": False,
    "startup_retry_quotas": False,
    "thread_environment": {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON with a single trailing LF."""

    def ready(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): ready(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [ready(child) for child in item]
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, float):
            if not math.isfinite(item):
                raise OccludedGoalContractError("canonical JSON forbids non-finite floats")
            return item
        if item is None or isinstance(item, (str, int, bool)):
            return item
        raise OccludedGoalContractError(f"unsupported canonical JSON type: {type(item).__name__}")

    return (
        json.dumps(ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def attach_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    output.pop("content_digest", None)
    output["content_digest"] = hashlib.sha256(canonical_json_bytes(output)[:-1]).hexdigest()
    return output


def validate_content_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(value))
    declared = output.pop("content_digest", None)
    expected = hashlib.sha256(canonical_json_bytes(output)[:-1]).hexdigest()
    if declared != expected:
        raise OccludedGoalContractError("content digest drift")
    output["content_digest"] = declared
    return output


def build_contract() -> dict[str, Any]:
    return attach_content_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.contract.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": STATUS,
            "source_parent_commit": SOURCE_PARENT_COMMIT,
            "predecessor_result_binding": copy.deepcopy(PREDECESSOR_RESULT_BINDING),
            "prior_panel_exclusion_authority": copy.deepcopy(PRIOR_PANEL_EXCLUSION_AUTHORITY),
            "frozen_source_bindings": copy.deepcopy(FROZEN_SOURCE_BINDINGS),
            "commit_subjects": {
                "freeze": CONTRACT_FREEZE_COMMIT_SUBJECT,
                "result": RESULT_COMMIT_SUBJECT,
            },
            "panel": {
                "episode_count": EPISODE_COUNT,
                "queries_per_episode": QUERIES_PER_EPISODE,
                "families": list(FAMILY_IDS),
                "episodes_per_family": EPISODES_PER_FAMILY,
                "split_episode_counts": copy.deepcopy(SPLIT_EPISODE_COUNTS),
                "split_family_episode_counts": copy.deepcopy(SPLIT_FAMILY_EPISODE_COUNTS),
                "aliasing": copy.deepcopy(ALIASING_AUTHORITY),
            },
            "conditions": {
                "ordered": list(CONDITION_IDS),
                "primary": list(PRIMARY_CONDITION_IDS),
                "ablations": list(ABLATION_CONDITION_IDS),
                "top_k": TOP_K,
                "fixed_window_observations": FIXED_WINDOW_OBSERVATIONS,
                "fixed_window_transitions": FIXED_WINDOW_TRANSITIONS,
            },
            "observation_likelihood": copy.deepcopy(OBSERVATION_LIKELIHOOD),
            "filter": copy.deepcopy(FILTER_AUTHORITY),
            "calibration": {
                "grid": {key: list(value) for key, value in CALIBRATION_GRID.items()},
                "grid_order": list(CALIBRATION_GRID_ORDER),
                "selection": copy.deepcopy(CALIBRATION_SELECTION),
            },
            "metrics": {
                "ids": list(METRIC_IDS),
                "formulas": copy.deepcopy(METRIC_FORMULAS),
                "ece_bin_count": ECE_BIN_COUNT,
                "port_label_order": list(PORT_LABEL_ORDER),
            },
            "stage_a": {
                "heldout_queries": HELDOUT_QUERY_COUNT,
                "belief_rows": HELDOUT_QUERY_COUNT * len(CONDITION_IDS),
                "absolute_gate": copy.deepcopy(ABSOLUTE_GATE),
                "incremental_over_current_gate": copy.deepcopy(INCREMENTAL_OVER_CURRENT_GATE),
                "incremental_over_map_gate": copy.deepcopy(INCREMENTAL_OVER_MAP_GATE),
                "short_history_match_gate": copy.deepcopy(SHORT_HISTORY_MATCH_GATE),
                "classifications": list(STAGE_A_CLASSIFICATIONS),
                "precedence": list(STAGE_A_PRECEDENCE),
            },
            "stage_b": copy.deepcopy(STAGE_B_AUTHORITY),
            "stage_b_gate": copy.deepcopy(STAGE_B_GATE),
            "stage_b_metric_formulas": copy.deepcopy(STAGE_B_METRIC_FORMULAS),
            "stage_b_classifications": list(STAGE_B_CLASSIFICATIONS),
            "next_decisions": copy.deepcopy(NEXT_DECISION_BY_CLASSIFICATION),
            "claims": copy.deepcopy(CLAIMS),
            "prohibitions": copy.deepcopy(PROHIBITIONS),
            "runtime_policy": copy.deepcopy(DIRECT_RUNTIME_POLICY),
            "output": {
                "root": str(OUTPUT_ROOT),
                "runtime_paths": copy.deepcopy(RUNTIME_OUTPUT_PATHS),
                "report_publication_is_scientific_identity": False,
                "independent_reducer_reconstructs": ["stage_a_metrics.json", "stage_b_metrics.json when present"],
            },
            "tracked_source_paths": list(TRACKED_SOURCE_PATHS),
            "source_dependency_paths": list(SOURCE_DEPENDENCY_PATHS),
        }
    )


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_content_digest(value)
    expected = build_contract()
    if dict(value) != expected:
        raise OccludedGoalContractError("contract value drift")
    return copy.deepcopy(expected)


__all__ = [name for name in tuple(globals()) if name.isupper()] + [
    "OccludedGoalContractError",
    "attach_content_digest",
    "build_contract",
    "canonical_json_bytes",
    "validate_content_digest",
    "validate_contract",
]
