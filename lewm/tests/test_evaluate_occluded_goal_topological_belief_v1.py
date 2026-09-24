from __future__ import annotations

import ast
import copy
import inspect
import json
import os
from pathlib import Path

import pytest

from scripts import evaluate_occluded_goal_topological_belief_v1 as E


CONDITIONS = E.EXPECTED_REGISTERED_CONDITION_IDS
ABLATIONS = E.EXPECTED_ABLATION_IDS
STAGE_B_CONDITIONS = (
    "CURRENT_FRAME_NEAREST_NODE",
    "STRONGEST_STAGE_A_MEMORY",
    "ORACLE_PLACE_IDENTITY",
)


def _source_freeze() -> dict[str, object]:
    return {
        "head_commit": "a" * 40,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": {
            "independent_reducer": {
                "path": E.REDUCER_SOURCE_PATH,
                "bytes": 101,
                "sha256": "b" * 64,
            },
            "metrics_module": {
                "path": E.METRICS_SOURCE_PATH,
                "bytes": 202,
                "sha256": "c" * 64,
            },
        },
    }


@pytest.fixture(autouse=True)
def _freeze_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(E, "_observe_source_freeze", lambda _module: _source_freeze())


class SyntheticMetrics:
    __name__ = "synthetic_occluded_goal_metrics"

    @staticmethod
    def graph_manifest_authority() -> dict[str, object]:
        return {"graphs": 1, "node_order": "stored"}

    @staticmethod
    def query_row_authority() -> dict[str, object]:
        return {"rows": 768, "heldout_rows": 128}

    @staticmethod
    def belief_row_authority() -> dict[str, object]:
        return {"rows": 1152, "condition_ids": list(CONDITIONS + ABLATIONS)}

    @staticmethod
    def calibration_authority() -> dict[str, object]:
        return {
            "selection_role": "CALIBRATION",
            "grid_results": 144,
            "grid_query_results": 144 * 128,
        }

    @staticmethod
    def stage_b_trace_authority() -> dict[str, object]:
        return {
            "condition_ids": list(STAGE_B_CONDITIONS),
            "episodes_per_condition": 16,
            "identity": ["condition_id", "episode_id", "decision_index"],
            "fields": sorted(_traces()[0]),
            "execution_policy": {"decision_budget": 1},
            "terminal_reasons": [
                "GOAL_REACHED",
                "NONMOVEMENT_LIMIT",
                "DECISION_BUDGET_EXHAUSTED",
            ],
            "terminal_rows_per_execution": 1,
        }

    @staticmethod
    def stage_a_condition_ids() -> tuple[str, ...]:
        return CONDITIONS + ABLATIONS

    @classmethod
    def reducer_authority(cls) -> dict[str, object]:
        value = {
            "schema": E.AUTHORITY_SCHEMA,
            "experiment_id": E.EXPERIMENT_ID,
            "graph_manifest": cls.graph_manifest_authority(),
            "query_rows": cls.query_row_authority(),
            "belief_rows": cls.belief_row_authority(),
            "calibration": cls.calibration_authority(),
            "stage_b_trace": cls.stage_b_trace_authority(),
            "stage_a_authorization_path": "decision.stage_b_authorized",
        }
        return {**value, "content_digest": E.canonical_digest(value)}

    @staticmethod
    def validate_graph_manifest(value: dict[str, object]) -> dict[str, object]:
        return value

    @staticmethod
    def validate_query_rows(
        rows: tuple[dict[str, object], ...], graph_manifest: dict[str, object]
    ) -> list[dict[str, object]]:
        del graph_manifest
        return list(rows)

    @staticmethod
    def validate_belief_rows(
        rows: tuple[dict[str, object], ...],
        query_rows: tuple[dict[str, object], ...],
        graph_manifest: dict[str, object],
    ) -> list[dict[str, object]]:
        del query_rows, graph_manifest
        return list(rows)

    @staticmethod
    def validate_calibration(
        value: dict[str, object],
        graph_manifest: dict[str, object],
        query_rows: tuple[dict[str, object], ...],
    ) -> dict[str, object]:
        del graph_manifest, query_rows
        return value

    @staticmethod
    def validate_stage_b_trace_rows(
        graph_manifest: dict[str, object],
        query_rows: tuple[dict[str, object], ...],
        rows: tuple[dict[str, object], ...],
        stage_a_metrics: dict[str, object],
    ) -> list[dict[str, object]]:
        del graph_manifest, query_rows, stage_a_metrics
        return list(rows)

    @staticmethod
    def recompute_stage_a_metrics(
        graph_manifest: dict[str, object],
        query_rows: tuple[dict[str, object], ...],
        belief_rows: tuple[dict[str, object], ...],
        calibration: dict[str, object],
    ) -> dict[str, object]:
        return {
            "schema": "synthetic.stage_a_metrics.v1",
            "graph_count": len(graph_manifest["graphs"]),
            "query_count": len(query_rows),
            "belief_count": len(belief_rows),
            "decision": {"stage_b_authorized": calibration["authorize_stage_b"]},
        }

    @staticmethod
    def stage_a_authorizes_stage_b(metrics: dict[str, object]) -> bool:
        return bool(metrics["decision"]["stage_b_authorized"])

    @staticmethod
    def recompute_stage_b_metrics(
        graph_manifest: dict[str, object],
        query_rows: tuple[dict[str, object], ...],
        trace_rows: tuple[dict[str, object], ...],
        stage_a_metrics: dict[str, object],
    ) -> dict[str, object]:
        del graph_manifest, query_rows
        return {
            "schema": "synthetic.stage_b_metrics.v1",
            "trace_count": len(trace_rows),
            "stage_a_query_count": stage_a_metrics["query_count"],
        }


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(E.canonical_document_bytes(value))


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(E.canonical_document_bytes(row) for row in rows))


def _queries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(768):
        episode_index = index % 16
        role = (
            "FIT"
            if index < 512
            else "CALIBRATION"
            if index < 640
            else "DEVELOPMENT_HELDOUT"
        )
        query_index = (index - 640) // 16 if role == "DEVELOPMENT_HELDOUT" else index // 16
        rows.append(
            {
                "query_id": f"query-{index:03d}",
                "episode_id": f"episode-{episode_index:02d}",
                "query_index": query_index,
                "role": role,
                "graph_id": f"graph-{episode_index:02d}",
                "candidate_node_ids": ["node-0", "node-1", "node-2"],
                "true_node_id": "node-0",
                "true_next_port_label": "LEFT",
                "stage_b_local_candidate_outcomes": [
                    {
                        "candidate_index": 0,
                        "candidate_name": "candidate-left",
                        "oracle_admissible": True,
                        "actual_edge_id": f"edge-{episode_index:02d}-left",
                        "actual_port_label": "LEFT",
                    },
                    {
                        "candidate_index": 1,
                        "candidate_name": "candidate-right",
                        "oracle_admissible": False,
                        "actual_edge_id": f"edge-{episode_index:02d}-right",
                        "actual_port_label": "RIGHT",
                    },
                ],
            }
        )
    return rows


def _beliefs() -> list[dict[str, object]]:
    return [
        {
            "query_id": f"query-{query_index:03d}",
            "condition_id": condition,
            "node_ids": ["node-0", "node-1", "node-2"],
            "observation_similarities": [0.8, 0.4, -0.1],
            "observation_likelihoods": [0.6, 0.3, 0.1],
            "transition_prior_probabilities": [0.5, 0.3, 0.2],
            "preprojection_probabilities": [0.6, 0.3, 0.1],
            "posterior_probabilities": [0.6, 0.3, 0.1],
            "selected_node_id": "node-0",
            "selected_edge_id": "edge-0",
            "selected_port_label": "LEFT",
            "normalized_entropy": 0.5,
            "abstained": False,
        }
        for query_index in range(640, 768)
        for condition in CONDITIONS + ABLATIONS
    ]


def _traces() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in STAGE_B_CONDITIONS:
        for episode_index in range(16):
            episode_id = f"episode-{episode_index:02d}"
            execution_id = f"synthetic-stage-b:{condition}:{episode_id}"
            edge_id = f"edge-{episode_index:02d}-left"
            observation_id = f"{execution_id}:observation-00"
            reobservation_id = f"{execution_id}:reobservation-00"
            rows.append(
                {
                    "step_id": f"{execution_id}:decision-00",
                    "execution_id": execution_id,
                    "episode_id": episode_id,
                    "family": "REPEATED_CORRIDOR",
                    "condition_id": condition,
                    "source_condition_id": condition,
                    "decision_index": 0,
                    "decision_budget": 1,
                    "query_id": f"query-{640 + episode_index:03d}",
                    "query_index": 0,
                    "actual_node_id_before": "node-0",
                    "actual_node_id_after": "node-1",
                    "actual_alias_group_id": "alias-0",
                    "selected_alias_group_id": "alias-0",
                    "prior_visited_node_ids": [],
                    "observation_id": observation_id,
                    "observation_pixel_sha256": "a" * 64,
                    "reobservation_id": reobservation_id,
                    "reobservation_pixel_sha256": "b" * 64,
                    "filter_update_observation_ids": [observation_id],
                    "filter_update_pixel_sha256s": ["a" * 64],
                    "filter_update_node_ids": ["node-0"],
                    "filter_update_incoming_action_labels": [None],
                    "filter_update_observation_similarities": [[0.8, 0.4, -0.1]],
                    "filter_update_observation_likelihoods": [[0.6, 0.3, 0.1]],
                    "filter_update_transition_prior_probabilities": [[0.5, 0.3, 0.2]],
                    "filter_update_preprojection_probabilities": [[0.6, 0.3, 0.1]],
                    "filter_update_posterior_probabilities": [[0.6, 0.3, 0.1]],
                    "belief_node_ids": ["node-0", "node-1", "node-2"],
                    "belief_probabilities": [0.6, 0.3, 0.1],
                    "selected_node_id": "node-0",
                    "belief_proposed_port_label": "LEFT",
                    "belief_selected_edge_id": edge_id,
                    "selected_port_label": "LEFT",
                    "true_next_port_label": "LEFT",
                    "normalized_entropy": 0.5,
                    "entropy_threshold": 0.9,
                    "abstained": False,
                    "action_disposition": "EXECUTED_CANDIDATE",
                    "ranker_called": True,
                    "local_oracle_admissible_candidate_indices": [0],
                    "local_candidate_scores": [1.0, 0.0],
                    "local_candidate_index": 0,
                    "local_candidate_name": "candidate-left",
                    "local_candidate_score": 1.0,
                    "stored_relative_waypoint": [1.0, 0.0, 0.0],
                    "executed_choice_edge_id": edge_id,
                    "executed_port_label": "LEFT",
                    "local_execution_success": True,
                    "oracle_admissible_execution": True,
                    "constituent_edge_ids": [edge_id],
                    "constituent_node_ids": ["node-0", "node-1"],
                    "constituent_action_labels": ["MOVE_LEFT"],
                    "constituent_edge_costs_m": [1.0],
                    "constituent_observation_ids": [reobservation_id],
                    "constituent_observation_pixel_sha256s": ["b" * 64],
                    "geodesic_distance_before_m": 1.0,
                    "geodesic_distance_after_m": 0.0,
                    "geodesic_progress_m": 1.0,
                    "executed_distance_m": 1.0,
                    "oracle_path_distance_m": 1.0,
                    "ranker_latency_ms": 0.25,
                    "planning_latency_ms": 0.5,
                    "ranker_previous_applied_command": [0.0, 0.0, 0.0],
                    "ranker_control_history": [[0.0, 0.0] for _ in range(15)],
                    "selected_candidate_applied_commands": [
                        [0.1, 0.0, 0.0] for _ in range(5)
                    ],
                    "post_decision_previous_applied_command": [0.1, 0.0, 0.0],
                    "post_decision_control_history": [
                        [0.0, 0.0] for _ in range(14)
                    ] + [[0.1, 0.0]],
                    "prelude_distance_m": 1.0,
                    "recovery_pending_before": False,
                    "recovery_pending_after": False,
                    "recovery": False,
                    "consecutive_nonmovement_decisions": 0,
                    "false_confident_wrong_turn": False,
                    "false_place_merge": False,
                    "false_loop_closure": False,
                    "replan": False,
                    "immediate_contact": False,
                    "stuck": False,
                    "terminal": True,
                    "terminal_reason": "GOAL_REACHED",
                    "goal_reached": True,
                }
            )
    return rows


def _prepare(root: Path, *, stage_b: bool) -> None:
    root.mkdir()
    graph = {
        "schema": "synthetic.graph.v1",
        "graphs": [
            {
                "graph_id": f"graph-{episode_index:02d}",
                "episode_id": f"episode-{episode_index:02d}",
                "nodes": [
                    {"node_id": "node-0", "pixel_sha256": "a" * 64},
                    {"node_id": "node-1", "pixel_sha256": "b" * 64},
                    {"node_id": "node-2", "pixel_sha256": "c" * 64},
                ],
                "edges": [
                    {
                        "edge_id": f"edge-{episode_index:02d}-left",
                        "source_node_id": "node-0",
                        "target_node_id": "node-1",
                        "executed_action_label": "MOVE_LEFT",
                        "edge_cost": 1.0,
                        "relative_waypoint": [1.0, 0.0, 0.0],
                    },
                    {
                        "edge_id": f"edge-{episode_index:02d}-right",
                        "source_node_id": "node-0",
                        "target_node_id": "node-2",
                        "executed_action_label": "MOVE_RIGHT",
                        "edge_cost": 2.0,
                        "relative_waypoint": [1.0, 0.0, 0.0],
                    },
                ],
            }
            for episode_index in range(16)
        ],
    }
    queries = _queries()
    beliefs = _beliefs()
    calibration = {
        "schema": "synthetic.calibration.v1",
        "authorize_stage_b": stage_b,
        "grid_query_results": [
            {
                "grid_index": grid_index,
                "query_id": f"query-{query_index:03d}",
                "edge_correct": True,
                "localisation_top3": True,
                "normalized_regret": 0.0,
                "false_confident": False,
                "abstained": False,
            }
            for grid_index in range(144)
            for query_index in range(512, 640)
        ],
    }
    stage_a_metrics = SyntheticMetrics.recompute_stage_a_metrics(
        graph, tuple(queries), tuple(beliefs), calibration
    )
    _write_json(root / E.GRAPH_FILE, graph)
    _write_jsonl(root / E.QUERY_FILE, queries)
    _write_json(root / E.CALIBRATION_FILE, calibration)
    _write_jsonl(root / E.STAGE_A_BELIEFS_FILE, beliefs)
    _write_json(root / E.STAGE_A_METRICS_FILE, stage_a_metrics)
    if stage_b:
        traces = _traces()
        _write_jsonl(root / E.STAGE_B_TRACE_FILE, traces)
        _write_json(
            root / E.STAGE_B_METRICS_FILE,
            SyntheticMetrics.recompute_stage_b_metrics(
                graph, tuple(queries), tuple(traces), stage_a_metrics
            ),
        )


def test_no_stage_b_exactly_reduces_and_emits_only_external_receipt(tmp_path: Path) -> None:
    root = tmp_path / "official"
    receipt_path = tmp_path / "receipt.json"
    _prepare(root, stage_b=False)
    receipt = E.verify_and_emit(root, receipt_path, metrics_module=SyntheticMetrics)
    assert receipt["pass"] is True
    assert receipt["stage_a_validation"]["rows"] == 128 * 9
    assert receipt["stage_a_validation"]["registered_conditions"] == 6
    assert receipt["stage_a_validation"]["ablations"] == 3
    assert receipt["calibration_validation"]["rows"] == 144 * 128
    assert receipt["calibration_validation"]["raw_outcome_cube_exactly_covered"] is True
    assert receipt["stage_a_authorizes_stage_b"] is False
    assert receipt["conditional_stage_b_present"] is False
    assert "content_digest" not in receipt
    assert receipt_path.read_bytes() == E.canonical_document_bytes(receipt)
    assert not (root / receipt_path.name).exists()
    assert set(receipt["scientific_execution_counters"].values()) == {0}
    assert E.validate_existing_regeneration_receipt(
        root, receipt_path, metrics_module=SyntheticMetrics
    ) == receipt


def test_authorized_stage_b_requires_complete_trace_and_exact_metrics(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=True)
    receipt = E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)
    assert receipt["stage_a_authorizes_stage_b"] is True
    assert receipt["conditional_stage_b_present"] is True
    assert receipt["conditional_stage_b_validation"]["rows"] == 16 * 3
    assert receipt["conditional_stage_b_validation"]["executions"] == 16 * 3
    assert receipt["conditional_stage_b_validation"]["exact_macro_evidence_validated"] is True
    assert receipt["conditional_stage_b_validation"]["exact_state_evidence_validated"] is True
    assert receipt["conditional_stage_b_validation"]["exact_belief_evidence_validated"] is True
    assert receipt["conditional_stage_b_validation"]["exact_ranker_evidence_validated"] is True
    assert receipt["conditional_stage_b_metrics_exact_byte_equal"] is True


def test_stage_b_pair_is_all_or_absent(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=True)
    (root / E.STAGE_B_METRICS_FILE).unlink()
    with pytest.raises(E.RegenerationError, match="both present or both absent"):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("macro", "constituent cost differs from graph"),
        ("state", "macro does not bind actual before/after state"),
        ("belief", "belief differs from the final carried posterior"),
        ("ranker", "ranker selection/macro binding drift"),
    ],
)
def test_stage_b_exact_evidence_categories_are_independently_required(
    tmp_path: Path, tamper: str, message: str
) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=True)
    rows = _traces()
    if tamper == "macro":
        rows[0]["constituent_edge_costs_m"] = [1.5]
        rows[0]["executed_distance_m"] = 1.5
    elif tamper == "state":
        rows[0]["actual_node_id_after"] = "node-2"
    elif tamper == "belief":
        rows[0]["belief_probabilities"] = [0.7, 0.2, 0.1]
    else:
        rows[0]["local_candidate_score"] = 0.5
    _write_jsonl(root / E.STAGE_B_TRACE_FILE, rows)
    with pytest.raises(E.RegenerationError, match=message):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


@pytest.mark.parametrize("authorized", [False, True])
def test_stage_b_presence_exactly_matches_stage_a_authorization(
    tmp_path: Path, authorized: bool
) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=authorized)
    if authorized:
        (root / E.STAGE_B_TRACE_FILE).unlink()
        (root / E.STAGE_B_METRICS_FILE).unlink()
    else:
        traces = _traces()
        metrics_a = json.loads((root / E.STAGE_A_METRICS_FILE).read_text())
        graph = json.loads((root / E.GRAPH_FILE).read_text())
        queries = [json.loads(line) for line in (root / E.QUERY_FILE).read_text().splitlines()]
        _write_jsonl(root / E.STAGE_B_TRACE_FILE, traces)
        _write_json(
            root / E.STAGE_B_METRICS_FILE,
            SyntheticMetrics.recompute_stage_b_metrics(
                graph, tuple(queries), tuple(traces), metrics_a
            ),
        )
    with pytest.raises(E.RegenerationError, match="does not match Stage-A authorization"):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


def test_stage_a_requires_every_query_condition_identity(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=False)
    rows = (root / E.STAGE_A_BELIEFS_FILE).read_bytes().splitlines(keepends=True)
    (root / E.STAGE_A_BELIEFS_FILE).write_bytes(b"".join(rows[:-1]))
    with pytest.raises(E.RegenerationError, match="coverage incomplete"):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


def test_calibration_requires_exact_grid_query_cross_product(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=False)
    calibration = json.loads((root / E.CALIBRATION_FILE).read_text())
    calibration["grid_query_results"].pop()
    _write_json(root / E.CALIBRATION_FILE, calibration)
    with pytest.raises(E.RegenerationError, match="exact grid-by-query"):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "posterior_probabilities",
            [0.6, 0.3, 0.2],
            "posterior_probabilities values do not sum to one",
        ),
        ("node_ids", ["node-1", "node-0", "node-2"], "ordered graph node vector"),
    ],
)
def test_full_vector_and_candidate_coverage_is_exact(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=False)
    rows = [json.loads(line) for line in (root / E.STAGE_A_BELIEFS_FILE).read_text().splitlines()]
    rows[0][field] = value
    _write_jsonl(root / E.STAGE_A_BELIEFS_FILE, rows)
    with pytest.raises(E.RegenerationError, match=message):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


def test_query_candidate_identity_coverage_is_exact(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=False)
    rows = [json.loads(line) for line in (root / E.QUERY_FILE).read_text().splitlines()]
    rows[0]["candidate_node_ids"] = ["node-0", "unknown-node"]
    _write_jsonl(root / E.QUERY_FILE, rows)
    with pytest.raises(E.RegenerationError, match="outside its graph"):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


def test_supplied_metric_bytes_must_equal_recomputation(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=False)
    metrics = json.loads((root / E.STAGE_A_METRICS_FILE).read_text())
    metrics["query_count"] = 127
    _write_json(root / E.STAGE_A_METRICS_FILE, metrics)
    with pytest.raises(E.RegenerationError, match="differs from exact recomputation"):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


def test_supplied_stage_b_metric_bytes_must_equal_recomputation(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=True)
    metrics = json.loads((root / E.STAGE_B_METRICS_FILE).read_text())
    metrics["trace_count"] = 1
    _write_json(root / E.STAGE_B_METRICS_FILE, metrics)
    with pytest.raises(E.RegenerationError, match="differs from exact recomputation"):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


def test_nofollow_rejects_symlinked_scientific_input(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=False)
    target = tmp_path / "graph.json"
    os.replace(root / E.GRAPH_FILE, target)
    (root / E.GRAPH_FILE).symlink_to(target)
    with pytest.raises(E.RegenerationError, match="cannot be opened safely"):
        E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)


def test_receipt_path_inside_official_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "official"
    _prepare(root, stage_b=False)
    receipt = E.build_regeneration_receipt(root, metrics_module=SyntheticMetrics)
    with pytest.raises(E.RegenerationError, match="never be inside"):
        E.emit_regeneration_receipt(root, root / "receipt.json", receipt)


def test_existing_external_receipt_cannot_be_replaced_with_self_consistent_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "official"
    receipt_path = tmp_path / "receipt.json"
    _prepare(root, stage_b=False)
    receipt = E.verify_and_emit(root, receipt_path, metrics_module=SyntheticMetrics)
    tampered = copy.deepcopy(receipt)
    tampered["stage_a_validation"]["queries"] = 127
    receipt_path.write_bytes(E.canonical_document_bytes(tampered))
    with pytest.raises(E.RegenerationError, match="differs from exact source-bound rebuild"):
        E.validate_existing_regeneration_receipt(
            root, receipt_path, metrics_module=SyntheticMetrics
        )


def test_reducer_is_torch_free_and_does_not_import_runner_or_contract() -> None:
    source = (E.ROOT / E.REDUCER_SOURCE_PATH).read_text()
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert not any(name == "torch" or name.startswith("torch.") for name in imports)
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imports)
    assert "scripts.run_occluded_goal_topological_belief_v1" not in imports
    assert "lewm.safety.occluded_goal_topological_belief_v1_contract" not in imports


def test_reducer_accepts_exact_public_metrics_authority() -> None:
    module = E._load_metrics_module()
    authority = E.validate_reducer_authority(module.reducer_authority(), module)
    assert authority["query_rows"]["rows"] == 768
    assert authority["belief_rows"]["rows"] == 128 * 9
    assert authority["calibration"]["grid_results"] == 144
    assert authority["calibration"]["grid_query_results"] == 144 * 128
    assert "phase_a_visit_fields" in authority["graph_manifest"]
    assert "query_pixel_sha256" in authority["query_rows"]["fields"]
    assert "posterior_probabilities" in authority["belief_rows"]["fields"]
    assert "action_history_position_mapping" in authority["belief_rows"]["fields"]
    assert tuple(inspect.signature(module.validate_calibration).parameters) == (
        "value",
        "graph_manifest",
        "query_rows",
    )
    assert authority["stage_b_trace"]["identity"] == [
        "condition_id",
        "episode_id",
        "decision_index",
    ]
    assert tuple(inspect.signature(module.validate_stage_b_trace_rows).parameters) == (
        "graph_manifest",
        "query_rows",
        "rows",
        "stage_a_metrics",
    )
    assert tuple(inspect.signature(module.recompute_stage_b_metrics).parameters) == (
        "graph_manifest",
        "query_rows",
        "trace_rows",
        "stage_a_metrics",
    )
