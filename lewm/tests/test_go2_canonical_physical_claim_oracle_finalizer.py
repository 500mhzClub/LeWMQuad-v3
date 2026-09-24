from __future__ import annotations

from copy import deepcopy
import ast
import hashlib
import inspect
import json
import math

import pytest

from lewm.benchmarks.go2_canonical_physical_claim_oracle_finalizer import (
    CANDIDATE_SCHEMA,
    FINALIZED_SCHEMA,
    ZERO_EVALUATOR_ACCESS_LEDGER,
    ZERO_FORBIDDEN_INPUT_ACCESS_LEDGER,
    finalize_canonical_physical_claim_oracle_regression,
)
from lewm.benchmarks.go2_physical_claim_evaluator import (
    evaluate_physical_claim_trace,
)
from lewm.benchmarks.go2_physical_claim_trace import (
    build_claim_attempt,
    build_claim_trace,
    object_id_reference,
)
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)


_BINDING_SHA = "1" * 64
_IMPLEMENTATION_SHA = "2" * 64
_GEOMETRY_SHA = "3" * 64
_POLICY_SHA = "4" * 64
_ORACLE_CONFIG = {"max_ticks": 2400, "synthetic": True}
_ELIGIBILITY_CONFIG = {"yaw_bins": 16, "synthetic": True}
_SOURCE_MAP = {
    "evaluator": {"path": "synthetic/evaluator.py", "sha256": "5" * 64},
    "finalizer": {"path": "synthetic/finalizer.py", "sha256": "6" * 64},
}
_INPUT_BINDINGS = {
    "development_manifest_sha256": "7" * 64,
    "materialization_sha256": "8" * 64,
    "geometry_contract_sha256": _GEOMETRY_SHA,
    "primitive_registry_sha256": "9" * 64,
    "directional_policy_content_sha256": _POLICY_SHA,
    "oracle_config": _ORACLE_CONFIG,
    "physical_eligibility_config": _ELIGIBILITY_CONFIG,
}
_COMMAND = ["python", "synthetic_runner.py", "--development-only"]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _oracle_event_id(
    *, trace_id: str, episode_id: str, scene_id: str, task_object_id: str
) -> str:
    return _canonical_sha256(
        {
            "domain": "lewm-go2-oracle-claim-attempt-v1",
            "episode_id": episode_id,
            "scene_id": scene_id,
            "task_object_id": task_object_id,
            "trace_id": trace_id,
        }
    )


def _manifest(index: int) -> SceneManifest:
    scene_id = f"synthetic_scene_{index:02d}"
    landmarks = tuple(
        BoxObject(
            object_id=f"{scene_id}_beacon_{color}",
            kind="landmark",
            center_xyz_m=(x, y, 0.5),
            size_xyz_m=(0.20, 0.20, 1.0),
            yaw_rad=0.0,
            material_id=f"landmark_{color}",
        )
        for color, x, y in (
            ("red", 2.0, 0.0),
            ("green", 0.0, 2.0),
            ("blue", -2.0, 0.0),
            ("yellow", 0.0, -2.0),
        )
    )
    return SceneManifest(
        scene_id=scene_id,
        family="synthetic_development",
        difficulty_tier="synthetic",
        topology_seed=index,
        visual_seed=100 + index,
        physics_seed=200 + index,
        world_bounds_xy_m=((-4.0, -4.0), (4.0, 4.0)),
        spawn=SpawnSpec((0.0, 0.0, 0.375), (1.0, 0.0, 0.0, 0.0)),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=landmarks,
        camera_constraints=CameraValidityConstraints(0.08, 0.05, 20.0, 0.1),
        split="development",
    )


def _claim_pose(landmark: BoxObject) -> tuple[float, float, float]:
    tx, ty = float(landmark.center_xyz_m[0]), float(landmark.center_xyz_m[1])
    radius = math.hypot(tx, ty)
    x = tx * (radius - 0.8) / radius
    y = ty * (radius - 0.8) / radius
    return x, y, math.atan2(ty - y, tx - x)


def _evaluated_trace(
    manifest: SceneManifest, *, kind: str
) -> tuple[dict, list[tuple[BoxObject, tuple[float, float, float]]]]:
    trace_id = f"{kind}:{manifest.scene_id}"
    episode_id = f"{kind}:{manifest.scene_id}:episode"
    landmarks = sorted(manifest.landmarks, key=lambda item: item.object_id.encode())
    witnesses = [(landmark, _claim_pose(landmark)) for landmark in landmarks]
    attempts = []
    for index, (landmark, pose) in enumerate(witnesses):
        reference = object_id_reference(landmark.object_id)
        event_id = (
            _oracle_event_id(
                trace_id=trace_id,
                episode_id=episode_id,
                scene_id=manifest.scene_id,
                task_object_id=landmark.object_id,
            )
            if kind == "oracle"
            else f"eligibility:{manifest.scene_id}:{landmark.object_id}"
        )
        attempts.append(
            build_claim_attempt(
                manifest=manifest,
                trace_id=trace_id,
                episode_id=episode_id,
                event_id=event_id,
                tick=index,
                event_index=index,
                requested_target=reference,
                claimed_target=reference,
                robot_pose_world_xy_yaw=pose,
                pose_provenance=(
                    "oracle_full_precision"
                    if kind == "oracle"
                    else "eligibility_candidate_full_precision"
                ),
            )
        )
    raw, task_ids, task_hash = build_claim_trace(
        manifest=manifest,
        trace_id=trace_id,
        episode_id=episode_id,
        controller_claim_attempts=attempts,
    )
    return (
        evaluate_physical_claim_trace(raw, manifest, task_ids, task_hash),
        witnesses,
    )


def _scene_evidence(manifest: SceneManifest) -> tuple[dict, dict]:
    oracle_trace, oracle_witnesses = _evaluated_trace(manifest, kind="oracle")
    eligibility_trace, eligibility_witnesses = _evaluated_trace(
        manifest, kind="eligibility"
    )
    task_ids = list(oracle_trace["physical_claim_summary"]["credited_object_ids"])
    evaluations = oracle_trace["physical_claim_evaluations"]
    colors = sorted(
        landmark.material_id.removeprefix("landmark_")
        for landmark in manifest.landmarks
    )
    oracle = {
        "scene_id": manifest.scene_id,
        "geometry_contract_sha256": _GEOMETRY_SHA,
        "success": True,
        "all_beacons_claimed": True,
        "claimed_count": 4,
        "beacon_count": 4,
        "claimed_beacon_ids": task_ids,
        "claimed_colors": colors,
        "claim_ticks": {
            event["claimed_target_object_id"]: event["tick"] for event in evaluations
        },
        "claim_poses": {
            event["claimed_target_object_id"]: event["robot_pose_world_xy_yaw"]
            for event in evaluations
        },
        "canonical_physical_claim_trace": oracle_trace,
        "failure_class": "success",
        "geometry_failures": [],
        "planner_failures": [],
        "follower_failures": [],
        "collisions": 0,
        "stalls": 0,
        "strict_directional_safe": True,
        "directional_polygon_collision_segments": 0,
        "directional_polygon_initial_pose_feasible": True,
        "directional_polygon_collision_object_ids": [],
        "directional_policy": {
            "content_sha256": _POLICY_SHA,
            "profile": "observed_max_plus_margin",
        },
        "route_planner": {"source": "OnlineBeliefMap.shortest_path", "queries": 4},
    }
    anchors = []
    for index, (landmark, pose) in enumerate(eligibility_witnesses):
        anchors.append(
            {
                "object_id": landmark.object_id,
                "target_xy_m": [
                    float(landmark.center_xyz_m[0]),
                    float(landmark.center_xyz_m[1]),
                ],
                "reachable": True,
                "reachable_claim_state_count": 1,
                "anchor_pose": list(pose),
                "anchor_lattice_state": [index, index + 1, index + 2],
                "anchor_target_distance_m": 0.8,
                "anchor_has_line_of_sight": True,
                "shortest_staged_action_count": index + 1,
                "shortest_staged_action_counts": {"forward": index + 1},
                "physical_claim_decision": "accepted",
                "physical_claim_credited": True,
                "physical_claim_unverifiable_reasons": [],
                "physical_claim_rejection_reasons": [],
            }
        )
    eligibility = {
        "schema": "lewm_go2_physical_scene_eligibility_v1",
        "scene_id": manifest.scene_id,
        "family": manifest.family,
        "policy_content_sha256": _POLICY_SHA,
        "policy_profile": "observed_max_plus_margin",
        "config": _ELIGIBILITY_CONFIG,
        "spawn_clear_at_actual_yaw": True,
        "spawn_snaps_to_lattice": True,
        "claim_anchors": anchors,
        "canonical_physical_claim_trace": eligibility_trace,
        "eligible": True,
        "failure_reason": "",
    }
    return oracle, eligibility


def _suite() -> tuple[dict, dict[str, SceneManifest], dict]:
    manifests = {
        manifest.scene_id: manifest for manifest in (_manifest(index) for index in range(24))
    }
    expected_scene_ids = list(manifests)[::2] + list(manifests)[1::2]
    assert expected_scene_ids != sorted(expected_scene_ids)
    oracle_scenes = []
    eligibility_reports = []
    for scene_id in expected_scene_ids:
        oracle, eligibility = _scene_evidence(manifests[scene_id])
        oracle_scenes.append(oracle)
        eligibility_reports.append(eligibility)
    oracle_aggregate = {
        "scene_count": 24,
        "all_beacons_claimed_scenes": 24,
        "full_4_of_4_claim_scenes": 24,
        "positive_control_success_scenes": 24,
        "claimed_beacons": 96,
        "expected_beacons": 96,
        "collisions": 0,
        "stalls": 0,
        "directional_polygon_collision_segments": 0,
        "strict_directional_safe_scenes": 24,
        "shared_map_routed_scenes": 24,
        "all_claims_zero_collision_zero_stall_gate_passed": True,
        "development_24x4_strict_gate_passed": True,
        "failure_classes": {"success": 24},
    }
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "binding_sha256": _BINDING_SHA,
        "implementation_manifest_sha256": _IMPLEMENTATION_SHA,
        "source_map": deepcopy(_SOURCE_MAP),
        "input_bindings": deepcopy(_INPUT_BINDINGS),
        "command": list(_COMMAND),
        "evaluator_access_ledger": deepcopy(ZERO_EVALUATOR_ACCESS_LEDGER),
        "input_access_ledger": deepcopy(ZERO_FORBIDDEN_INPUT_ACCESS_LEDGER),
        "oracle_report": {
            "schema": "go2_oracle_coverage_positive_control_v1",
            "development_only": True,
            "scene_ids": [report["scene_id"] for report in oracle_scenes],
            "scene_execution": {
                "kind": "spawn_process",
                "worker_count": 6,
                "threads_per_worker": 1,
                "merge_order": "development_manifest_index",
                "worker_runtime_input_file_access": False,
            },
            "geometry_contract": {"sha256": _GEOMETRY_SHA},
            "config": _ORACLE_CONFIG,
            "aggregate": oracle_aggregate,
            "scenes": oracle_scenes,
        },
        "physical_eligibility_reports": eligibility_reports,
    }
    kwargs = {
        "scene_manifests": manifests,
        "expected_scene_ids": expected_scene_ids,
        "expected_binding_sha256": _BINDING_SHA,
        "expected_implementation_manifest_sha256": _IMPLEMENTATION_SHA,
        "expected_source_map": _SOURCE_MAP,
        "expected_input_bindings": _INPUT_BINDINGS,
        "expected_command": _COMMAND,
        "expected_directional_policy_content_sha256": _POLICY_SHA,
    }
    return candidate, manifests, kwargs


def _reevaluate_oracle_scene_trace(
    candidate: dict,
    manifests: dict[str, SceneManifest],
    scene_index: int,
    mutator,
) -> None:
    scene = candidate["oracle_report"]["scenes"][scene_index]
    stored = scene["canonical_physical_claim_trace"]
    raw = {
        "schema": "lewm_go2_claim_trace_v1",
        "trace_id": stored["trace_id"],
        "episode_id": stored["episode_id"],
        "scene_id": stored["scene_id"],
        "physical_manifest_sha256": stored["physical_manifest_sha256"],
        "task_object_ids": stored["task_object_ids"],
        "task_object_set_sha256": stored["task_object_set_sha256"],
        "controller_claim_attempts": stored["controller_claim_attempts"],
        "evaluator_feedback_to_controller": stored["evaluator_feedback_to_controller"],
    }
    mutator(raw)
    scene["canonical_physical_claim_trace"] = evaluate_physical_claim_trace(
        raw,
        manifests[scene["scene_id"]],
        raw["task_object_ids"],
        raw["task_object_set_sha256"],
    )


@pytest.fixture(scope="module")
def synthetic_suite():
    return _suite()


def test_synthetic_24x4_suite_passes_and_content_hash_is_canonical(
    synthetic_suite,
) -> None:
    candidate, _manifests, kwargs = synthetic_suite
    result = finalize_canonical_physical_claim_oracle_regression(
        candidate, **kwargs
    )

    assert result.passed, result.errors
    assert result.errors == ()
    assert result.finalized_payload is not None
    payload = result.finalized_payload
    assert payload["schema"] == FINALIZED_SCHEMA
    assert payload["finalization_passed"] is True
    assert payload["scene_ids"] == kwargs["expected_scene_ids"]
    assert payload["totals"]["oracle_raw_attempt_count"] == 96
    assert payload["totals"]["oracle_evaluation_count"] == 96
    assert payload["totals"]["oracle_accepted_count"] == 96
    assert payload["totals"]["oracle_credited_count"] == 96
    assert payload["totals"]["eligibility_raw_attempt_count"] == 96
    assert payload["totals"]["eligibility_credited_count"] == 96
    assert payload["aggregate"]["oracle_eligibility_task_pairs_equal"] is True
    core = dict(payload)
    content_sha256 = core.pop("content_sha256")
    assert content_sha256 == _canonical_sha256(core)


def test_mutation_matrix_fails_closed(synthetic_suite) -> None:
    base, manifests, kwargs = synthetic_suite
    mutations: list[tuple[str, dict, dict]] = []

    def add(name: str, mutator, *, kwargs_mutator=None) -> None:
        candidate = deepcopy(base)
        local_kwargs = deepcopy(kwargs)
        mutator(candidate)
        if kwargs_mutator is not None:
            kwargs_mutator(local_kwargs)
        mutations.append((name, candidate, local_kwargs))

    add("binding", lambda value: value.__setitem__("binding_sha256", "0" * 64))
    add("source", lambda value: value["source_map"].clear())
    add("input", lambda value: value["input_bindings"].clear())
    add("command", lambda value: value["command"].append("--changed"))
    add(
        "scene execution",
        lambda value: value["oracle_report"]["scene_execution"].__setitem__(
            "worker_count", 5
        ),
    )
    add(
        "evaluator ledger",
        lambda value: value["evaluator_access_ledger"].__setitem__(
            "evaluator_output_reads_by_controller", 1
        ),
    )
    add(
        "evaluator ledger bool zero",
        lambda value: value["evaluator_access_ledger"].__setitem__(
            "evaluator_output_reads_by_controller", False
        ),
    )
    add(
        "input ledger",
        lambda value: value["input_access_ledger"].__setitem__(
            "heldout_payload_opens", 1
        ),
    )
    add(
        "input ledger float one",
        lambda value: value["input_access_ledger"][
            "preflight_hash_reads"
        ].__setitem__("geometry_hash_reads", 1.0),
    )
    add(
        "worker input ledger bool zero",
        lambda value: value["input_access_ledger"].__setitem__(
            "worker_runtime_input_file_opens", False
        ),
    )
    add(
        "scene execution float workers",
        lambda value: value["oracle_report"]["scene_execution"].__setitem__(
            "worker_count", 6.0
        ),
    )
    add(
        "oracle config int as float",
        lambda value: value["oracle_report"]["config"].__setitem__(
            "max_ticks", 2400.0
        ),
    )
    add("scene omitted", lambda value: value["oracle_report"]["scenes"].pop())
    add(
        "event id domain",
        lambda value: _reevaluate_oracle_scene_trace(
            value,
            manifests,
            0,
            lambda raw: raw["controller_claim_attempts"][0].__setitem__(
                "event_id", "wrong"
            ),
        ),
    )
    add(
        "stored evaluation",
        lambda value: value["oracle_report"]["scenes"][0][
            "canonical_physical_claim_trace"
        ]["physical_claim_evaluations"][0].__setitem__("credited", False),
    )
    add(
        "stored factor bool as int",
        lambda value: value["oracle_report"]["scenes"][0][
            "canonical_physical_claim_trace"
        ]["physical_claim_evaluations"][0]["factors"].__setitem__(
            "identity_passes", 1
        ),
    )
    add(
        "route source",
        lambda value: value["oracle_report"]["scenes"][0][
            "route_planner"
        ].__setitem__("source", "InflatedOccupancyGrid.astar"),
    )
    add(
        "collision",
        lambda value: value["oracle_report"]["scenes"][0].__setitem__(
            "collisions", 1
        ),
    )
    add(
        "stall",
        lambda value: value["oracle_report"]["scenes"][0].__setitem__("stalls", 1),
    )
    add(
        "polygon collision",
        lambda value: value["oracle_report"]["scenes"][0].__setitem__(
            "directional_polygon_collision_segments", 1
        ),
    )
    add(
        "eligibility policy",
        lambda value: value["physical_eligibility_reports"][0].__setitem__(
            "policy_content_sha256", "0" * 64
        ),
    )
    add(
        "eligibility config int as float",
        lambda value: value["physical_eligibility_reports"][0][
            "config"
        ].__setitem__("yaw_bins", 16.0),
    )
    add(
        "eligibility false",
        lambda value: value["physical_eligibility_reports"][0].__setitem__(
            "eligible", False
        ),
    )
    add(
        "anchor unreachable",
        lambda value: value["physical_eligibility_reports"][0][
            "claim_anchors"
        ][0].__setitem__("reachable", False),
    )
    add(
        "anchor uncredited",
        lambda value: value["physical_eligibility_reports"][0][
            "claim_anchors"
        ][0].__setitem__("physical_claim_credited", False),
    )
    add(
        "aggregate",
        lambda value: value["oracle_report"]["aggregate"].__setitem__(
            "claimed_beacons", 95
        ),
    )
    add(
        "aggregate int as float",
        lambda value: value["oracle_report"]["aggregate"].__setitem__(
            "claimed_beacons", 96.0
        ),
    )
    add(
        "top level claims",
        lambda value: value["oracle_report"]["scenes"][0].__setitem__(
            "claimed_count", 3
        ),
    )
    add(
        "top level int as float",
        lambda value: value["oracle_report"]["scenes"][0].__setitem__(
            "claimed_count", 4.0
        ),
    )

    def duplicate_trace_id(value: dict) -> None:
        first = value["oracle_report"]["scenes"][0]["canonical_physical_claim_trace"]

        def mutate(raw: dict) -> None:
            raw["trace_id"] = first["trace_id"]
            for attempt in raw["controller_claim_attempts"]:
                attempt["trace_id"] = raw["trace_id"]
                task_id = attempt["requested_target"]["value"]
                attempt["event_id"] = _oracle_event_id(
                    trace_id=raw["trace_id"],
                    episode_id=raw["episode_id"],
                    scene_id=raw["scene_id"],
                    task_object_id=task_id,
                )

        _reevaluate_oracle_scene_trace(value, manifests, 1, mutate)

    add("duplicate trace id", duplicate_trace_id)

    def duplicate_episode_id(value: dict) -> None:
        first = value["oracle_report"]["scenes"][0]["canonical_physical_claim_trace"]

        def mutate(raw: dict) -> None:
            raw["episode_id"] = first["episode_id"]
            for attempt in raw["controller_claim_attempts"]:
                attempt["episode_id"] = raw["episode_id"]
                task_id = attempt["requested_target"]["value"]
                attempt["event_id"] = _oracle_event_id(
                    trace_id=raw["trace_id"],
                    episode_id=raw["episode_id"],
                    scene_id=raw["scene_id"],
                    task_object_id=task_id,
                )

        _reevaluate_oracle_scene_trace(value, manifests, 1, mutate)

    add("duplicate episode id", duplicate_episode_id)

    for name, candidate, local_kwargs in mutations:
        result = finalize_canonical_physical_claim_oracle_regression(
            candidate, **local_kwargs
        )
        assert not result.passed, name
        assert result.errors, name
        assert result.finalized_payload is None, name


def test_malformed_candidate_returns_failure_instead_of_publishing() -> None:
    result = finalize_canonical_physical_claim_oracle_regression(
        {"schema": float("nan")},
        scene_manifests={},
        expected_scene_ids=[],
        expected_binding_sha256=_BINDING_SHA,
        expected_implementation_manifest_sha256=_IMPLEMENTATION_SHA,
        expected_source_map=_SOURCE_MAP,
        expected_input_bindings=_INPUT_BINDINGS,
        expected_command=_COMMAND,
        expected_directional_policy_content_sha256=_POLICY_SHA,
    )
    assert not result.passed
    assert result.finalized_payload is None


@pytest.mark.parametrize("mutation", ("missing", "extra", "substituted"))
def test_expected_scene_order_set_mutations_fail_without_exception(
    synthetic_suite,
    mutation: str,
) -> None:
    candidate, _manifests, kwargs = synthetic_suite
    local_kwargs = deepcopy(kwargs)
    if mutation == "missing":
        local_kwargs["expected_scene_ids"].pop()
    elif mutation == "extra":
        local_kwargs["expected_scene_ids"].append("scene_extra")
    else:
        local_kwargs["expected_scene_ids"][0] = "scene_absent"
    result = finalize_canonical_physical_claim_oracle_regression(
        deepcopy(candidate), **local_kwargs
    )
    assert not result.passed
    assert result.errors
    assert result.finalized_payload is None


def test_finalizer_imports_no_runner_or_controller_modules() -> None:
    from lewm.benchmarks import go2_canonical_physical_claim_oracle_finalizer as module

    tree = ast.parse(inspect.getsource(module))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "lewm.benchmarks.go2_oracle_positive_control" not in imports
    assert "lewm.benchmarks.go2_physical_eligibility" not in imports
    assert "lewm.benchmarks.go2_physical_claim_trace" not in imports
    assert "pathlib" not in imports
