from __future__ import annotations

from copy import deepcopy

from lewm.benchmarks.go2_physical_claim_evaluator import evaluate_physical_claim_trace
from lewm.benchmarks.go2_physical_claim_finalizer import (
    finalize_physical_claim_result,
)
from lewm.benchmarks.go2_physical_claim_result import (
    canonical_physical_claim_status,
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


def _manifest() -> SceneManifest:
    return SceneManifest(
        scene_id="finalizer_toy",
        family="test",
        difficulty_tier="test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-2.0, -2.0), (2.0, 2.0)),
        spawn=SpawnSpec((0.0, 0.0, 0.375), (1.0, 0.0, 0.0, 0.0)),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=(
            BoxObject(
                "beacon_red",
                "landmark",
                (1.0, 0.0, 0.5),
                (0.2, 0.2, 1.0),
                0.0,
                "landmark_red",
            ),
        ),
        camera_constraints=CameraValidityConstraints(0.08, 0.05, 20.0, 0.1),
    )


def _result() -> tuple[SceneManifest, dict]:
    manifest = _manifest()
    reference = object_id_reference("beacon_red")
    attempt = build_claim_attempt(
        manifest=manifest,
        trace_id="trace",
        episode_id="episode",
        event_id="event",
        tick=1,
        event_index=0,
        requested_target=reference,
        claimed_target=reference,
        robot_pose_world_xy_yaw=(0.0, 0.0, 0.0),
        pose_provenance="runtime_full_precision",
    )
    raw, ids, commitment = build_claim_trace(
        manifest=manifest,
        trace_id="trace",
        episode_id="episode",
        controller_claim_attempts=[attempt],
    )
    evaluated = evaluate_physical_claim_trace(raw, manifest, ids, commitment)
    return manifest, {
        "claimed": True,
        "claimed_colors": ["red"],
        "success": True,
        "canonical_physical_claim_trace": evaluated,
        "runtime_evaluator_access_ledger": {
            "evaluator_output_reads_by_controller": 0,
            "evaluator_callbacks_into_controller": 0,
            "evaluator_derived_termination_signals": 0,
        },
    }


def test_finalizer_recomputes_valid_result_exactly() -> None:
    manifest, result = _result()
    finalized = finalize_physical_claim_result(result, scene_manifest=manifest)
    assert finalized.passed
    assert finalized.recomputed_trace == result["canonical_physical_claim_trace"]
    status = canonical_physical_claim_status(
        result,
        scene_manifest=manifest,
        required_task_count=1,
    )
    assert status.valid
    assert status.all_targets_claimed


def test_finalizer_rejects_hash_summary_feedback_ledger_and_proxy_mutations() -> None:
    manifest, base = _result()
    mutations = []
    for path in (
        ("canonical_physical_claim_trace", "trace_content_sha256"),
        (
            "canonical_physical_claim_trace",
            "physical_claim_summary",
            "content_sha256",
        ),
        (
            "canonical_physical_claim_trace",
            "physical_claim_evaluations",
            0,
            "content_sha256",
        ),
        (
            "canonical_physical_claim_trace",
            "physical_claim_evaluations",
            0,
            "physical_contract",
            "claim_distance_m",
        ),
        (
            "canonical_physical_claim_trace",
            "physical_claim_evaluations",
            0,
            "unverifiable_reasons",
        ),
    ):
        value = deepcopy(base)
        cursor = value
        for key in path[:-1]:
            cursor = cursor[key]
        cursor[path[-1]] = (
            1.21 if path[-1] == "claim_distance_m" else ["unknown_reason"]
            if path[-1] == "unverifiable_reasons"
            else "0" * 64
        )
        mutations.append(value)
    omitted = deepcopy(base)
    omitted["canonical_physical_claim_trace"]["physical_claim_evaluations"].pop()
    mutations.append(omitted)
    duplicated_credit = deepcopy(base)
    duplicated_credit["canonical_physical_claim_trace"]["physical_claim_summary"][
        "credited_object_ids"
    ].append("beacon_red")
    mutations.append(duplicated_credit)
    feedback = deepcopy(base)
    feedback["canonical_physical_claim_trace"]["evaluator_feedback_to_controller"] = [
        "forbidden"
    ]
    mutations.append(feedback)
    ledger = deepcopy(base)
    ledger["runtime_evaluator_access_ledger"][
        "evaluator_derived_termination_signals"
    ] = 1
    mutations.append(ledger)
    bool_ledger = deepcopy(base)
    bool_ledger["runtime_evaluator_access_ledger"][
        "evaluator_derived_termination_signals"
    ] = False
    mutations.append(bool_ledger)
    bool_factor = deepcopy(base)
    bool_factor["canonical_physical_claim_trace"]["physical_claim_evaluations"][0][
        "factors"
    ]["identity_passes"] = 1
    mutations.append(bool_factor)
    float_count = deepcopy(base)
    float_count["canonical_physical_claim_trace"]["physical_claim_summary"][
        "credited_count"
    ] = 1.0
    mutations.append(float_count)
    proxy = deepcopy(base)
    proxy["claimed"] = False
    mutations.append(proxy)
    malformed_success = deepcopy(base)
    malformed_success["success"] = "yes"
    mutations.append(malformed_success)
    colors = deepcopy(base)
    colors["claimed_colors"] = ["blue"]
    mutations.append(colors)

    for result in mutations:
        assert not finalize_physical_claim_result(
            result, scene_manifest=manifest
        ).passed
