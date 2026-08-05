from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import math

import numpy as np
import pytest

from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
    _build_projected_snapshot,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    MapFrameIdentity,
    PhysicalLabel,
    SnapshotBindingError,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    CONFIGURATION_CELL_SIZE_M,
    PHYSICAL_CELL_SIZE_M,
    TwoResolutionConfigurationProjectionV2,
)
from lewm.planning.two_resolution_frontier_viewpoint_v2 import (
    TwoResolutionFrontierViewpointPlannerV2,
    TwoResolutionPhysicalViewStateIssuerV2,
    _visible_physical_cells,
    configuration_cell_center_world_v2,
    configuration_center_in_physical_grid_v2,
    physical_cell_center_world_v2,
)
from lewm_worlds.manifest import (
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)


ORIGIN = (13.7, -9.3)
PHYSICAL_SHAPE = (84, 96)
CONFIGURATION_SHAPE = (42, 48)
START_CONFIGURATION_CELL = (30, 35)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _manifest(scene_id: str) -> SceneManifest:
    return SceneManifest(
        scene_id=scene_id,
        family="g4-v2-synthetic",
        difficulty_tier="unit",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((10.0, -12.0), (20.0, 0.0)),
        spawn=SpawnSpec(
            xyz_m=(17.25, -5.05, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )


def _fixture(
    scene_id: str,
    *,
    origin: tuple[float, float] = ORIGIN,
    occupied: tuple[tuple[int, int], ...] = (),
):
    physical = np.full(
        PHYSICAL_SHAPE,
        int(PhysicalLabel.UNKNOWN),
        dtype=np.uint8,
    )
    physical[28:76, 36:88] = int(PhysicalLabel.FREE)
    for cell in occupied:
        physical[cell] = int(PhysicalLabel.OCCUPIED)
    memory, snapshot, planner = _build_projected_snapshot(
        _manifest(scene_id),
        physical,
        origin_xy_m=origin,
        configuration_shape=CONFIGURATION_SHAPE,
    )
    projection = planner._projection
    assert type(projection) is TwoResolutionConfigurationProjectionV2
    issuer = TwoResolutionPhysicalViewStateIssuerV2(memory, projection)
    authority = TwoResolutionFrontierViewpointPlannerV2(planner, issuer)
    state = issuer.issue(
        snapshot,
        pose_xy_variance_m2=0.0025,
        pose_yaw_variance_rad2=0.0004,
    )
    assert START_CONFIGURATION_CELL in snapshot.free_cells
    return memory, projection, snapshot, planner, issuer, authority, state


def _candidates(authority, snapshot, state):
    return authority.generate(
        snapshot,
        state,
        start_configuration_cell=START_CONFIGURATION_CELL,
        current_yaw_rad=0.17,
    )


@pytest.fixture(scope="module")
def read_only_fixture():
    return _fixture("g4-v2-read-only")


def test_high_nonzero_index_uses_world_conversion_and_distinct_cell_steps(
    read_only_fixture,
) -> None:
    _memory, _projection, snapshot, _planner, _issuer, authority, state = (
        read_only_fixture
    )
    physical_coordinates = configuration_center_in_physical_grid_v2(
        snapshot,
        START_CONFIGURATION_CELL,
    )
    assert physical_coordinates == pytest.approx((61.0, 71.0))
    assert physical_coordinates != START_CONFIGURATION_CELL

    candidate_set = _candidates(authority, snapshot, state)
    moving = next(
        candidate
        for candidate in candidate_set.candidates
        if len(candidate.safe_configuration_path) > 1
    )
    first, second = moving.safe_configuration_path[:2]
    first_world = configuration_cell_center_world_v2(snapshot, first)
    second_world = configuration_cell_center_world_v2(snapshot, second)
    assert math.dist(first_world, second_world) == pytest.approx(
        CONFIGURATION_CELL_SIZE_M
    )
    assert moving.path_cost_m == pytest.approx(
        (len(moving.safe_configuration_path) - 1) * CONFIGURATION_CELL_SIZE_M
    )

    physical_first = physical_cell_center_world_v2(snapshot, (61, 71))
    physical_second = physical_cell_center_world_v2(snapshot, (62, 71))
    assert math.dist(physical_first, physical_second) == pytest.approx(
        PHYSICAL_CELL_SIZE_M
    )
    assert CONFIGURATION_CELL_SIZE_M == pytest.approx(
        2.0 * PHYSICAL_CELL_SIZE_M
    )


def test_frontier_candidates_visibility_scoring_and_selection_are_deterministic(
    read_only_fixture,
) -> None:
    _memory, _projection, snapshot, _planner, _issuer, authority, state = (
        read_only_fixture
    )
    first = _candidates(authority, snapshot, state)
    second = _candidates(authority, snapshot, state)
    assert first.content_sha256 == second.content_sha256
    assert [row.content_sha256 for row in first.candidates] == [
        row.content_sha256 for row in second.candidates
    ]
    assert len(first.candidates) == 512
    assert any(row.kind == "frontier" for row in first.candidates)
    assert all(
        cell in snapshot.free_cells
        for candidate in first.candidates
        for cell in candidate.safe_configuration_path
    )

    frontier = next(
        row
        for row in first.candidates
        if row.kind == "frontier" and row.yaw_index == 8
    )
    observation = authority.predict_observation(
        snapshot, state, first, frontier
    )
    assert observation.visible_physical_cells
    assert observation.entropy_reduction_physical_cells <= (
        state.physical_unknown_cells
    )
    assert all(
        0 <= cell[0] < PHYSICAL_SHAPE[0]
        and 0 <= cell[1] < PHYSICAL_SHAPE[1]
        for cell in observation.visible_physical_cells
    )
    assert any(
        cell[0] >= CONFIGURATION_SHAPE[0]
        or cell[1] >= CONFIGURATION_SHAPE[1]
        for cell in observation.visible_physical_cells
    )

    scores = authority.score(snapshot, state, first)
    assert len(scores) == len(first.candidates)
    assert tuple(score.utility for score in scores) == tuple(
        sorted((score.utility for score in scores), reverse=True)
    )
    selected = authority.select(snapshot, state, first)
    assert selected is not None
    assert selected is next(
        row
        for row in first.candidates
        if row.content_sha256 == scores[0].candidate_sha256
    )
    authority.validate_candidate(snapshot, state, first, selected)
    object.__setattr__(scores[0], "utility", scores[0].utility + 1.0)
    with pytest.raises(SnapshotBindingError, match="score was mutated"):
        authority.select(snapshot, state, first)


def test_view_update_keeps_physical_sweep_and_configuration_history_separate() -> None:
    _memory, _projection, snapshot, _planner, issuer, authority, state = (
        _fixture("g4-v2-view-update")
    )
    candidate_set = _candidates(authority, snapshot, state)
    candidate = next(
        row
        for row in candidate_set.candidates
        if row.kind == "frontier" and row.yaw_index == 8
    )
    observation = authority.predict_observation(
        snapshot, state, candidate_set, candidate
    )
    observed = set(observation.visible_physical_cells)
    observed.add((70, 80))
    issuer.record_development_view(
        snapshot,
        observation_id="physical-view-1",
        observation_sha256=_hash("physical-view-1"),
        observed_physical_cells=observed,
        viewpoint_configuration_cell=candidate.reachable_configuration_cell,
        yaw_index=candidate.yaw_index,
    )
    with pytest.raises(SnapshotBindingError, match="exact live"):
        issuer.validate_state(snapshot, state)

    current = issuer.issue(snapshot)
    assert (70, 80) in current.visually_swept_physical_cells
    assert START_CONFIGURATION_CELL not in current.visually_swept_physical_cells
    assert any(
        row.configuration_cell == candidate.reachable_configuration_cell
        and row.yaw_index == candidate.yaw_index
        for row in current.configuration_view_history
    )
    assert all(
        row.configuration_cell[0] < CONFIGURATION_SHAPE[0]
        and row.configuration_cell[1] < CONFIGURATION_SHAPE[1]
        for row in current.configuration_view_history
    )
    with pytest.raises(SnapshotBindingError):
        authority.validate_candidate(
            snapshot, state, candidate_set, candidate
        )


def test_state_frame_origin_revision_support_copy_and_replay_are_rejected(
    read_only_fixture,
) -> None:
    _memory, _projection, snapshot, _planner, issuer, _authority, state = (
        read_only_fixture
    )
    for replay in (copy.copy(state), copy.deepcopy(state), replace(state)):
        with pytest.raises(SnapshotBindingError, match="exact live"):
            issuer.validate_state(snapshot, replay)
    for replay in (
        replace(state, physical_revision=state.physical_revision + 1),
        replace(
            state,
            configuration_revision=state.configuration_revision + 1,
        ),
    ):
        with pytest.raises(SnapshotBindingError, match="exact live"):
            issuer.validate_state(snapshot, replay)

    foreign_frame = MapFrameIdentity(
        session_id="foreign-configuration",
        origin_xy_m=ORIGIN,
        cell_size_m=CONFIGURATION_CELL_SIZE_M,
        frame_id="foreign_configuration",
    )
    foreign = replace(state, configuration_map_frame=foreign_frame)
    with pytest.raises(SnapshotBindingError, match="exact live"):
        issuer.validate_state(snapshot, foreign)

    wrong_origin = MapFrameIdentity(
        session_id="wrong-origin-configuration",
        origin_xy_m=(ORIGIN[0] + 0.1, ORIGIN[1]),
        cell_size_m=CONFIGURATION_CELL_SIZE_M,
        frame_id="wrong_origin_configuration",
    )
    with pytest.raises(ValueError, match="frame/profile"):
        replace(state, configuration_map_frame=wrong_origin)
    with pytest.raises(ValueError, match="frame/profile"):
        replace(state, free_support_sha256="0" * 64)


def test_candidate_and_set_copy_deepcopy_revision_and_replay_are_rejected(
    read_only_fixture,
) -> None:
    _memory, _projection, snapshot, _planner, _issuer, authority, state = (
        read_only_fixture
    )
    candidate_set = _candidates(authority, snapshot, state)
    candidate = candidate_set.candidates[0]
    authority.validate_candidate(snapshot, state, candidate_set, candidate)

    for replay_set in (
        copy.copy(candidate_set),
        copy.deepcopy(candidate_set),
        replace(candidate_set),
        replace(
            candidate_set,
            configuration_revision=candidate_set.configuration_revision + 1,
        ),
    ):
        with pytest.raises(SnapshotBindingError, match="exact live"):
            authority.validate_candidate(
                snapshot, state, replay_set, candidate
            )
    for replay_candidate in (
        copy.copy(candidate),
        copy.deepcopy(candidate),
        replace(candidate),
        replace(
            candidate,
            physical_revision=candidate.physical_revision + 1,
        ),
    ):
        with pytest.raises(SnapshotBindingError, match="exact live"):
            authority.validate_candidate(
                snapshot, state, candidate_set, replay_candidate
            )


def test_configuration_reprojection_rejects_stale_state_set_frontier_and_path() -> None:
    _memory, projection, snapshot, _planner, issuer, authority, state = (
        _fixture("g4-v2-stale-reprojection")
    )
    candidate_set = _candidates(authority, snapshot, state)
    candidate = candidate_set.candidates[0]
    current_snapshot = projection.project()
    assert current_snapshot.configuration_revision == (
        snapshot.configuration_revision + 1
    )
    with pytest.raises(SnapshotBindingError):
        issuer.validate_state(snapshot, state)
    with pytest.raises(SnapshotBindingError):
        issuer.validate_state(current_snapshot, state)

    current_state = issuer.issue(current_snapshot)
    with pytest.raises(SnapshotBindingError):
        authority.validate_candidate(
            current_snapshot,
            current_state,
            candidate_set,
            candidate,
        )
    current_set = _candidates(authority, current_snapshot, current_state)
    current_candidate = current_set.candidates[0]
    authority.validate_candidate(
        current_snapshot,
        current_state,
        current_set,
        current_candidate,
    )


def test_foreign_frame_projection_and_changed_origin_are_rejected() -> None:
    memory, _projection, _snapshot, _planner, _issuer, _authority, _state = (
        _fixture("g4-v2-owner")
    )
    (
        _foreign_memory,
        foreign_projection,
        foreign_snapshot,
        _foreign_planner,
        _foreign_issuer,
        _foreign_authority,
        _foreign_state,
    ) = _fixture("g4-v2-foreign-same-origin")
    mixed = TwoResolutionPhysicalViewStateIssuerV2(
        memory,
        foreign_projection,
    )
    with pytest.raises(SnapshotBindingError, match="stale or foreign"):
        mixed.issue(foreign_snapshot)

    (
        _other_memory,
        other_projection,
        _other_snapshot,
        _other_planner,
        _other_issuer,
        _other_authority,
        _other_state,
    ) = _fixture(
        "g4-v2-foreign-origin",
        origin=(ORIGIN[0] + 0.5, ORIGIN[1]),
    )
    with pytest.raises(ValueError, match="not aligned"):
        TwoResolutionPhysicalViewStateIssuerV2(memory, other_projection)


def test_physical_ray_stops_after_first_unknown_and_before_occupied(
    read_only_fixture,
) -> None:
    _memory, _projection, _snapshot, _planner, _issuer, _authority, state = (
        read_only_fixture
    )
    groups = tuple(((x, 70),) for x in range(74, 79))
    eligible = frozenset((x, 70) for x in range(74, 79))
    assert _visible_physical_cells(
        groups,
        state,
        eligible_cells=eligible,
    ) == ((74, 70), (75, 70), (76, 70))

    (
        _blocked_memory,
        _blocked_projection,
        _blocked_snapshot,
        _blocked_planner,
        _blocked_issuer,
        _blocked_authority,
        blocked_state,
    ) = _fixture("g4-v2-ray-occupied", occupied=((75, 70),))
    assert _visible_physical_cells(
        groups,
        blocked_state,
        eligible_cells=eligible,
    ) == ((74, 70),)
