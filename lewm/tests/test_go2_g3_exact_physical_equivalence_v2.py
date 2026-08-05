from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from lewm.benchmarks.go2_g3_exact_physical_equivalence_v2 import (
    GOVERNING_DESIGN_PATH,
    GOVERNING_DESIGN_SHA256,
    G3ExactSceneResultV2,
    assert_v2_profile_integrity,
    evaluate_exact_scene_v2,
    independent_cross_grid_configuration_labels,
    independent_exact_rational_supports,
    registered_two_resolution_lattices,
    summarize_exact_scenes_v2,
)
from lewm.planning.geometry_contract import load_geometry_contract
from lewm.planning.revisioned_physical_configuration_memory import (
    MapFrameIdentity,
    PhysicalLabel,
)
from lewm.planning.two_resolution_configuration_projection_v2 import (
    FIXED_PROFILE_V2,
    FREE_SUPPORT_SHA256,
    OCCUPIED_SUPPORT_SHA256,
    PROFILE_SHA256,
)
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)


ROOT = Path(__file__).resolve().parents[2]


def _manifest() -> SceneManifest:
    obstacle = BoxObject(
        object_id="thin-rotated",
        kind="obstacle",
        center_xyz_m=(1.0, 0.5, 0.5),
        size_xyz_m=(0.08, 1.0, 1.0),
        yaw_rad=0.31,
        material_id="test",
    )
    return SceneManifest(
        scene_id="g3-v2-audit-unit",
        family="unit",
        difficulty_tier="unit",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-2.0, -2.0), (2.0, 2.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=(obstacle,),
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )


def _geometry():
    return load_geometry_contract(
        ROOT / "config/go2_generalization_geometry_v2.json",
        repository_root=ROOT,
        verify_sources=False,
    )


@pytest.fixture(scope="module")
def scene_result() -> G3ExactSceneResultV2:
    return evaluate_exact_scene_v2(_manifest(), _geometry())


def test_v2_lattices_share_boundary_origin_and_exact_two_to_one_shape() -> None:
    assert_v2_profile_integrity()
    origin, physical_shape, configuration_shape = registered_two_resolution_lattices(
        _manifest(),
        _geometry(),
    )
    assert origin == pytest.approx((-2.57, -2.57))
    assert physical_shape == (
        2 * configuration_shape[0],
        2 * configuration_shape[1],
    )


def test_independent_oracle_uses_exact_rational_supports_not_profile_offsets() -> None:
    free, occupied = independent_exact_rational_supports()
    assert len(free) == 316 and len(occupied) == 276
    assert free == FIXED_PROFILE_V2.free_support_offsets
    assert occupied == FIXED_PROFILE_V2.occupied_support_offsets
    physical = np.full((52, 54), int(PhysicalLabel.FREE), dtype=np.uint8)
    physical[26, 27] = int(PhysicalLabel.OCCUPIED)
    baseline = independent_cross_grid_configuration_labels(
        physical,
        configuration_shape=(26, 27),
    )
    original = FIXED_PROFILE_V2.occupied_support_offsets
    object.__setattr__(FIXED_PROFILE_V2, "occupied_support_offsets", original[:-1])
    try:
        np.testing.assert_array_equal(
            independent_cross_grid_configuration_labels(
                physical,
                configuration_shape=(26, 27),
            ),
            baseline,
        )
        with pytest.raises(Exception, match="mutated"):
            assert_v2_profile_integrity()
    finally:
        object.__setattr__(FIXED_PROFILE_V2, "occupied_support_offsets", original)
    assert_v2_profile_integrity()


def test_synthetic_scene_carries_exact_projection_safety_route_and_identity_receipts(
    scene_result: G3ExactSceneResultV2,
) -> None:
    result = scene_result
    assert result.profile_sha256 == PROFILE_SHA256
    assert result.free_support_sha256 == FREE_SUPPORT_SHA256
    assert result.occupied_support_sha256 == OCCUPIED_SUPPORT_SHA256
    assert result.physical_lattice_shape == (
        2 * result.lattice_shape[0],
        2 * result.lattice_shape[1],
    )
    assert result.physical_map_frame_sha256 != result.configuration_map_frame_sha256
    assert result.physical_revision == 1
    assert result.configuration_revision == 1
    assert result.independent_label_mismatch_cells == 0
    assert result.component_mismatch_cells == 0
    assert result.astar_probe_count > 0
    assert result.astar_mismatch_count == 0
    assert result.unsafe_free_cells == 0
    assert result.complete_physical_raster is True
    assert result.physical_evidence_authority == "exact_physical"
    assert result.exact_observation_count == 1
    assert result.learned_observation_count == 0
    assert result.physical_execution_block_cells == ()
    assert result.configuration_execution_block_cells == ()
    assert len(result.execution_block_receipt_sha256) == 64
    assert result.production_promotion_authorized is False
    serialized = result.to_dict()
    for key in (
        "physical_map_frame_sha256",
        "configuration_map_frame_sha256",
        "physical_revision",
        "configuration_revision",
        "physical_content_sha256",
        "snapshot_content_sha256",
        "projection_source_sha256",
        "execution_block_receipt_sha256",
        "physical_execution_block_cells",
        "configuration_execution_block_cells",
        "profile_sha256",
        "free_support_sha256",
        "occupied_support_sha256",
        "lattice_identities",
    ):
        assert key in serialized


def _unique_scene_row(
    row: G3ExactSceneResultV2,
    index: int,
) -> G3ExactSceneResultV2:
    scene_id = f"scene-{index:02d}"
    physical_session = f"{scene_id}:g3-v2:physical"
    configuration_session = f"{scene_id}:g3-v2:configuration"
    physical_frame = MapFrameIdentity(
        session_id=physical_session,
        origin_xy_m=row.physical_lattice_origin_xy_m,
        cell_size_m=0.05,
        frame_id=row.physical_frame_id,
    )
    configuration_frame = MapFrameIdentity(
        session_id=configuration_session,
        origin_xy_m=row.configuration_lattice_origin_xy_m,
        cell_size_m=0.10,
        frame_id=row.configuration_frame_id,
    )
    unique_hash = lambda name: hashlib.sha256(f"{scene_id}:{name}".encode()).hexdigest()
    return replace(
        row,
        scene_id=scene_id,
        physical_session_id=physical_session,
        configuration_session_id=configuration_session,
        physical_map_frame_sha256=physical_frame.content_sha256,
        configuration_map_frame_sha256=configuration_frame.content_sha256,
        memory_config_sha256=unique_hash("memory-config"),
        physical_content_sha256=unique_hash("physical-content"),
        snapshot_content_sha256=unique_hash("snapshot"),
        projection_source_sha256=unique_hash("projection-source"),
        claim_endpoints_retained=4,
        beacon_count=4,
    )


def _source_bindings() -> dict[str, str]:
    return {
        GOVERNING_DESIGN_PATH: GOVERNING_DESIGN_SHA256,
        "synthetic": "0" * 64,
    }


def test_summary_requires_24_scene_96_endpoint_identity_complete_gate(
    scene_result: G3ExactSceneResultV2,
) -> None:
    rows = [_unique_scene_row(scene_result, index) for index in range(24)]
    summary = summarize_exact_scenes_v2(rows, source_bindings=_source_bindings())
    assert summary["scene_count"] == 24
    assert summary["beacon_count"] == 96
    assert summary["claim_endpoints_retained"] == 96
    assert summary["identity_receipt_scene_count"] == 24
    assert summary["complete_physical_raster_scene_count"] == 24
    assert summary["candidate_v2_exact_equivalence_pass"] is True
    assert summary["production_promotion_authorized"] is False
    assert summary["learned_projection_implemented"] is False
    assert summary["profile"]["projection_contract_sha256"] == PROFILE_SHA256
    assert summary["governing_design_binding"] == {
        "path": GOVERNING_DESIGN_PATH,
        "file_sha256": GOVERNING_DESIGN_SHA256,
    }

    unsafe = replace(rows[0], unsafe_free_cells=1)
    failed = summarize_exact_scenes_v2(
        [unsafe, *rows[1:]],
        source_bindings=_source_bindings(),
    )
    assert failed["candidate_v2_exact_equivalence_pass"] is False

    with pytest.raises(ValueError, match="governing design"):
        summarize_exact_scenes_v2(rows, source_bindings={"synthetic": "0" * 64})
    with pytest.raises(ValueError, match="identities"):
        summarize_exact_scenes_v2(
            [rows[0], replace(rows[0], scene_id=rows[1].scene_id), *rows[2:]],
            source_bindings=_source_bindings(),
        )


def test_scene_result_rejects_forged_frame_hash_revision_raster_or_authority(
    scene_result: G3ExactSceneResultV2,
) -> None:
    for changes, match in (
        ({"physical_map_frame_sha256": "0" * 64}, "map-frame hashes"),
        ({"configuration_revision": 0}, "identity/authority"),
        ({"physical_unknown_cells": 1}, "complete rasters"),
        ({"physical_evidence_authority": "learned_physical"}, "identity/authority"),
        ({"learned_observation_count": 1}, "identity/authority"),
        ({"exact_sim_tainted": 1}, "identity/authority"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(scene_result, **changes)
