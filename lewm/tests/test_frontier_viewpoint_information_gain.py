from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from lewm.planning.frontier_viewpoint_information_gain import (
    FrontierViewpointConfig,
    PhysicalViewStateIssuer,
    assert_candidate_executable,
    conservative_visible_cells_for_ray,
    generate_frontier_viewpoint_candidates,
    ordered_closed_cell_supercover_groups,
    predict_candidate_observation,
    score_information_gain_candidates,
    select_information_gain_candidate,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    ConfigurationMorphology,
    ConfigurationPlanner,
    EvidenceAuthority,
    ExecutionBlock,
    ExecutionBlockKind,
    MapFrameIdentity,
    ObservationIdentity,
    PhysicalCellEvidence,
    PhysicalEvidenceTransaction,
    PhysicalLabel,
    PhysicalMemoryConfig,
    PoseProvenance,
    PoseSource,
    RevisionedPhysicalMemory,
    SnapshotBindingError,
    StaleSnapshotError,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _pose(frame: MapFrameIdentity, revision: int) -> PoseProvenance:
    return PoseProvenance(
        source=PoseSource.DEPLOYMENT_ODOMETRY,
        frame_id=frame.frame_id,
        mean_xy_yaw=(0.0, 0.0, 0.0),
        covariance_xy_yaw=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        timestamp_ns=revision,
        synchronization_id=f"sync-{revision}",
        camera_transform_sha256=_hash("camera"),
    )


def _apply(
    memory: RevisionedPhysicalMemory,
    name: str,
    *,
    labels: dict[tuple[int, int], PhysicalLabel] | None = None,
    blocks: tuple[ExecutionBlock, ...] = (),
) -> None:
    frame = memory.map_frame
    evidence = tuple(
        PhysicalCellEvidence(cell=cell, label=label)
        for cell, label in sorted((labels or {}).items())
    )
    transaction = PhysicalEvidenceTransaction(
        observation=ObservationIdentity(
            observation_id=name,
            payload_sha256=_hash(f"payload:{name}"),
            producer_sha256=_hash("learned-producer"),
            authority=EvidenceAuthority.LEARNED_PHYSICAL,
        ),
        map_frame=frame,
        pose=_pose(frame, memory.revision + 1),
        physical_evidence=evidence,
        execution_blocks=blocks,
    )
    memory.apply_transaction(transaction)


def _new_memory(name: str) -> RevisionedPhysicalMemory:
    frame = MapFrameIdentity(session_id=name, origin_xy_m=(0.0, 0.0))
    return RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=frame,
            expected_camera_transform_sha256=_hash("camera"),
        )
    )


def _partial_fixture() -> tuple[
    RevisionedPhysicalMemory,
    ConfigurationPlanner,
    object,
    PhysicalViewStateIssuer,
    object,
]:
    memory = _new_memory("g4-partial")
    _apply(
        memory,
        "initial",
        labels={
            (x, y): PhysicalLabel.FREE
            for x in range(-8, 9)
            for y in range(-8, 9)
        },
    )
    morphology = ConfigurationMorphology()
    domain = {(x, y) for x in range(-15, 16) for y in range(-15, 16)}
    snapshot = memory.create_configuration_snapshot(
        morphology,
        candidate_cells=domain,
    )
    planner = ConfigurationPlanner(memory, morphology)
    issuer = PhysicalViewStateIssuer(memory)
    state = issuer.issue(snapshot)
    return memory, planner, snapshot, issuer, state


def _known_enclosed_fixture() -> tuple[
    RevisionedPhysicalMemory,
    ConfigurationPlanner,
    object,
    PhysicalViewStateIssuer,
    object,
]:
    memory = _new_memory("g4-known-enclosed")
    frame = memory.map_frame
    labels = {
        (x, y): PhysicalLabel.FREE
        for x in range(-20, 21)
        for y in range(-20, 21)
    }
    ring = tuple(
        ExecutionBlock(
            block_id=f"ring-{x}-{y}",
            body_center_xy_m=frame.cell_center((x, y)),
            kind=ExecutionBlockKind.EXECUTION_VETO,
            outcome_sha256=_hash(f"block:{x}:{y}"),
        )
        for x in range(-15, 16)
        for y in range(-15, 16)
        if abs(x) == 15 or abs(y) == 15
    )
    _apply(memory, "known", labels=labels, blocks=ring)
    morphology = ConfigurationMorphology()
    domain = {(x, y) for x in range(-15, 16) for y in range(-15, 16)}
    snapshot = memory.create_configuration_snapshot(
        morphology,
        candidate_cells=domain,
    )
    planner = ConfigurationPlanner(memory, morphology)
    issuer = PhysicalViewStateIssuer(memory)
    state = issuer.issue(
        snapshot,
        pose_xy_variance_m2=0.0025,
        pose_yaw_variance_rad2=0.0004,
    )
    return memory, planner, snapshot, issuer, state


def _candidates(planner, snapshot, issuer, state):
    return generate_frontier_viewpoint_candidates(
        planner,
        snapshot,
        state,
        issuer=issuer,
        start_cell=(0, 0),
        current_yaw_rad=0.0,
    )


def test_candidate_generation_is_deterministic_revision_bound_and_free_only() -> None:
    memory, planner, snapshot, issuer, state = _partial_fixture()
    first = _candidates(planner, snapshot, issuer, state)
    second = _candidates(planner, snapshot, issuer, state)
    assert first.content_sha256 == second.content_sha256
    assert 0 < len(first.candidates) <= 512
    assert len({row.content_sha256 for row in first.candidates}) == len(
        first.candidates
    )
    assert all(row.safe_path[0] == (0, 0) for row in first.candidates)
    assert all(
        cell in snapshot.free_cells
        for candidate in first.candidates
        for cell in candidate.safe_path
    )
    for candidate in first.candidates[:8]:
        assert_candidate_executable(
            planner,
            snapshot,
            first,
            candidate,
            state,
            issuer=issuer,
        )

    _apply(memory, "later", labels={(30, 30): PhysicalLabel.FREE})
    with pytest.raises(StaleSnapshotError):
        generate_frontier_viewpoint_candidates(
            planner,
            snapshot,
            state,
            issuer=issuer,
            start_cell=(0, 0),
            current_yaw_rad=0.0,
        )


def test_issued_state_rejects_forgery_mutation_and_wrong_issuer() -> None:
    _memory, _planner, snapshot, issuer, state = _partial_fixture()
    unknown = next(iter(state.unknown_cells))
    forged = replace(
        state,
        free_cells=state.free_cells | {unknown},
        unknown_cells=state.unknown_cells - {unknown},
        physical_entropy_cells=state.physical_entropy_cells - {unknown},
    )
    with pytest.raises(SnapshotBindingError, match="not issued"):
        issuer.validate_state(snapshot, forged)

    mutated = issuer.issue(snapshot)
    object.__setattr__(mutated, "pose_xy_variance_m2", 0.25)
    with pytest.raises(SnapshotBindingError, match="mutated"):
        issuer.validate_state(snapshot, mutated)

    other_memory = _new_memory("g4-other")
    other_issuer = PhysicalViewStateIssuer(other_memory)
    with pytest.raises(SnapshotBindingError):
        other_issuer.validate_state(snapshot, state)


def test_promoted_visual_history_rejects_caller_authored_cells() -> None:
    memory = RevisionedPhysicalMemory(
        PhysicalMemoryConfig(
            map_frame=MapFrameIdentity(
                session_id="g4-promoted",
                origin_xy_m=(0.0, 0.0),
            ),
            expected_camera_transform_sha256=_hash("camera"),
            promoted_runtime=True,
        )
    )
    snapshot = memory.create_configuration_snapshot(
        ConfigurationMorphology(),
        candidate_cells={(0, 0)},
    )
    issuer = PhysicalViewStateIssuer(memory)
    with pytest.raises(PermissionError, match="qualified camera-view adapter"):
        issuer.record_view(
            snapshot,
            observation_id="caller-authored",
            observation_sha256=_hash("caller-authored"),
            observed_cells={(0, 0)},
            viewpoint_cell=(0, 0),
            yaw_index=0,
        )


def test_fully_known_enclosed_map_generates_movement_viewpoints() -> None:
    _memory, planner, snapshot, issuer, state = _known_enclosed_fixture()
    component = planner.connected_component(snapshot, (0, 0))
    assert len(component.cells) > 32
    assert planner.frontier_cells(snapshot, component).cells == ()
    assert not state.unknown_cells
    candidates = _candidates(planner, snapshot, issuer, state)
    movement_cells = {
        row.reachable_cell
        for row in candidates.candidates
        if row.reachable_cell != (0, 0)
    }
    assert len(movement_cells) == 31
    assert all(
        row.kind == "coverage_frontier"
        for row in candidates.candidates
        if row.reachable_cell != (0, 0)
    )


def test_path_start_and_canonical_path_substitution_fail_closed() -> None:
    _memory, planner, snapshot, issuer, state = _known_enclosed_fixture()
    candidate_set = _candidates(planner, snapshot, issuer, state)
    original = next(row for row in candidate_set.candidates if len(row.safe_path) >= 3)
    shifted = replace(
        original,
        start_cell=original.safe_path[1],
        safe_path=original.safe_path[1:],
        path_cost_m=original.path_cost_m - 0.1,
    )
    with pytest.raises(SnapshotBindingError):
        assert_candidate_executable(
            planner,
            snapshot,
            candidate_set,
            shifted,
            state,
            issuer=issuer,
        )

    skipped = replace(
        original,
        safe_path=(original.safe_path[0], *original.safe_path[2:]),
        path_cost_m=max(0.0, original.path_cost_m - 0.1),
    )
    with pytest.raises(SnapshotBindingError):
        assert_candidate_executable(
            planner,
            snapshot,
            candidate_set,
            skipped,
            state,
            issuer=issuer,
        )


def test_closed_cell_diagonal_corner_obstacle_occludes_downstream() -> None:
    memory = _new_memory("g4-diagonal")
    _apply(
        memory,
        "labels",
        labels={(0, 0): PhysicalLabel.FREE, (1, 0): PhysicalLabel.OCCUPIED},
    )
    morphology = ConfigurationMorphology()
    domain = {(0, 0), (1, 0), (0, 1), (1, 1), (2, 2), (3, 3)}
    snapshot = memory.create_configuration_snapshot(morphology, candidate_cells=domain)
    issuer = PhysicalViewStateIssuer(memory)
    state = issuer.issue(snapshot)
    groups = ordered_closed_cell_supercover_groups((0.5, 0.5), (3.5, 3.5))
    assert {(0, 1), (1, 0), (1, 1)} <= set(groups[1])
    visible = conservative_visible_cells_for_ray(groups, state)
    assert (2, 2) not in visible
    assert (3, 3) not in visible

    boundary_groups = ordered_closed_cell_supercover_groups(
        (0.5, 1.0),
        (3.5, 1.0),
    )
    boundary_cells = {cell for group in boundary_groups for cell in group}
    assert {(2, 0), (2, 1)} <= boundary_cells


def test_missing_domain_and_first_unknown_occlude_downstream() -> None:
    memory = _new_memory("g4-domain-gap")
    _apply(memory, "start-free", labels={(0, 0): PhysicalLabel.FREE})
    morphology = ConfigurationMorphology()
    snapshot = memory.create_configuration_snapshot(
        morphology,
        candidate_cells={(0, 0), (2, 0), (3, 0)},
    )
    issuer = PhysicalViewStateIssuer(memory)
    state = issuer.issue(snapshot)
    groups = ordered_closed_cell_supercover_groups((0.5, 0.5), (3.5, 0.5))
    visible = conservative_visible_cells_for_ray(groups, state)
    assert (2, 0) not in visible
    assert (3, 0) not in visible

    unknown_snapshot = memory.create_configuration_snapshot(
        morphology,
        candidate_cells={(0, 0), (1, 0), (2, 0)},
    )
    unknown_state = issuer.issue(unknown_snapshot)
    visible_unknown = conservative_visible_cells_for_ray(groups, unknown_state)
    assert (1, 0) in visible_unknown
    assert (2, 0) not in visible_unknown


def test_coverage_entropy_and_discovery_are_separate_score_terms() -> None:
    _memory, planner, snapshot, issuer, state = _known_enclosed_fixture()
    candidates = _candidates(planner, snapshot, issuer, state)
    scores = score_information_gain_candidates(
        planner,
        snapshot,
        candidates,
        state,
        issuer=issuer,
    )
    assert all(row.entropy_reduction_cells == 0 for row in scores)
    assert any(row.newly_swept_cells > 0 for row in scores)
    assert any(row.discovery_opportunity_cells > 0 for row in scores)
    assert any(
        row.normalized_coverage_gain > row.normalized_entropy_reduction
        for row in scores
    )

    selected = select_information_gain_candidate(
        planner,
        snapshot,
        candidates,
        state,
        issuer=issuer,
    )
    assert selected is not None
    observation = predict_candidate_observation(selected, state)
    assert observation.newly_swept_cells
    assert not observation.entropy_reduction_cells


def test_recorded_visual_sweep_changes_coverage_without_changing_physical_map() -> None:
    _memory, planner, snapshot, issuer, initial = _known_enclosed_fixture()
    candidates = _candidates(planner, snapshot, issuer, initial)
    selected = select_information_gain_candidate(
        planner,
        snapshot,
        candidates,
        initial,
        issuer=issuer,
    )
    assert selected is not None
    observation = predict_candidate_observation(selected, initial)
    issuer.record_view(
        snapshot,
        observation_id="view-1",
        observation_sha256=_hash("view-1"),
        observed_cells=observation.visible_cells,
        viewpoint_cell=selected.reachable_cell,
        yaw_index=selected.yaw_index,
    )
    updated = issuer.issue(snapshot)
    assert updated.physical_content_sha256 == initial.physical_content_sha256
    assert updated.visually_swept_cells > initial.visually_swept_cells
    assert updated.physical_entropy_cells == initial.physical_entropy_cells
    assert updated.uniform_discovery_cells < initial.uniform_discovery_cells
    with pytest.raises(SnapshotBindingError):
        issuer.validate_state(snapshot, initial)


def test_pose_uncertainty_and_view_diversity_are_candidate_relative() -> None:
    _memory, planner, snapshot, issuer, initial = _known_enclosed_fixture()
    candidates = _candidates(planner, snapshot, issuer, initial)
    scores = score_information_gain_candidates(
        planner,
        snapshot,
        candidates,
        initial,
        issuer=issuer,
    )
    by_candidate = {
        row.candidate_sha256: row
        for row in scores
    }
    start = next(
        candidate
        for candidate in candidates.candidates
        if candidate.reachable_cell == (0, 0) and candidate.yaw_index == 8
    )
    distant = max(candidates.candidates, key=lambda row: row.path_cost_m)
    assert (
        by_candidate[distant.content_sha256].normalized_pose_uncertainty
        > by_candidate[start.content_sha256].normalized_pose_uncertainty
    )

    issuer.record_view(
        snapshot,
        observation_id="view-diversity",
        observation_sha256=_hash("view-diversity"),
        observed_cells=(),
        viewpoint_cell=start.reachable_cell,
        yaw_index=start.yaw_index,
    )
    updated = issuer.issue(snapshot)
    updated_candidates = _candidates(planner, snapshot, issuer, updated)
    updated_scores = score_information_gain_candidates(
        planner,
        snapshot,
        updated_candidates,
        updated,
        issuer=issuer,
    )
    updated_by_candidate = {
        (candidate.reachable_cell, candidate.yaw_index): score
        for candidate in updated_candidates.candidates
        for score in updated_scores
        if score.candidate_sha256 == candidate.content_sha256
    }
    exact = updated_by_candidate[((0, 0), 8)]
    opposite = updated_by_candidate[((0, 0), 0)]
    assert exact.normalized_view_diversity < opposite.normalized_view_diversity
    assert exact.normalized_staleness < opposite.normalized_staleness


def test_config_and_candidate_mutation_fail_before_scoring() -> None:
    _memory, planner, snapshot, issuer, state = _partial_fixture()
    candidates = _candidates(planner, snapshot, issuer, state)
    mutated = candidates.candidates[0]
    object.__setattr__(mutated, "turn_cost_rad", mutated.turn_cost_rad + 0.01)
    with pytest.raises(SnapshotBindingError, match="mutated"):
        score_information_gain_candidates(
            planner,
            snapshot,
            candidates,
            state,
            issuer=issuer,
        )

    with pytest.raises(ValueError, match="frozen"):
        FrontierViewpointConfig(candidate_cap=7)
