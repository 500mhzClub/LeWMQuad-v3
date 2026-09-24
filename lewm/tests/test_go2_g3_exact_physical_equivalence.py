from __future__ import annotations

from dataclasses import replace

import numpy as np

from lewm.benchmarks.go2_g3_exact_physical_equivalence import (
    G3ExactSceneResult,
    _closed_segment_intersects_rotated_box,
    _closed_square_intersects_box_mask,
    independent_configuration_labels,
    independently_derived_morphology_supports,
    summarize_exact_scenes,
)
from lewm.planning.revisioned_physical_configuration_memory import (
    ConfigurationMorphology,
    PhysicalLabel,
)
from lewm_worlds.manifest import BoxObject


def _box(*, center=(0.0, 0.0), size=(0.04, 1.0), yaw=0.0) -> BoxObject:
    return BoxObject(
        object_id="box",
        kind="obstacle",
        center_xyz_m=(float(center[0]), float(center[1]), 0.5),
        size_xyz_m=(float(size[0]), float(size[1]), 1.0),
        yaw_rad=float(yaw),
        material_id="test",
    )


def test_closed_square_sat_catches_thin_and_rotated_intersections() -> None:
    grid_x = np.asarray([[-0.05], [0.05], [0.15]], dtype=np.float64)
    grid_y = np.zeros_like(grid_x)
    thin = _closed_square_intersects_box_mask(
        grid_x,
        grid_y,
        half_cell_m=0.05,
        box=_box(center=(0.0, 0.0), size=(0.02, 0.5)),
    )
    assert thin[:, 0].tolist() == [True, True, False]

    rotated = _closed_square_intersects_box_mask(
        grid_x,
        grid_y,
        half_cell_m=0.05,
        box=_box(center=(0.15, 0.10), size=(0.04, 0.5), yaw=np.pi / 4.0),
    )
    assert bool(np.any(rotated))


def test_independent_morphology_has_occupied_precedence_and_unknown_band() -> None:
    morphology = ConfigurationMorphology()
    labels = np.full((41, 41), int(PhysicalLabel.FREE), dtype=np.uint8)
    labels[20, 20] = int(PhysicalLabel.OCCUPIED)
    configuration = independent_configuration_labels(labels, morphology)
    assert configuration[20, 20] == int(PhysicalLabel.OCCUPIED)
    interior = configuration[6:-6, 6:-6]
    occupied_count = int(np.count_nonzero(interior == int(PhysicalLabel.OCCUPIED)))
    unknown_count = int(np.count_nonzero(interior == int(PhysicalLabel.UNKNOWN)))
    assert occupied_count == 69
    assert unknown_count > 0
    assert configuration[0, 0] == int(PhysicalLabel.OCCUPIED)


def test_independent_morphology_derives_and_checks_registered_89_69_kernels() -> None:
    free, occupied = independently_derived_morphology_supports(
        physical_cell_size_m=0.10,
        footprint_radius_m=0.47,
    )
    assert len(free) == 89
    assert len(occupied) == 69
    morphology = ConfigurationMorphology()
    assert free == morphology.free_support_offsets
    assert occupied == morphology.occupied_support_offsets
    object.__setattr__(morphology, "free_support_offsets", free[:-1])
    labels = np.full((15, 15), int(PhysicalLabel.FREE), dtype=np.uint8)
    try:
        independent_configuration_labels(labels, morphology)
    except AssertionError as exc:
        assert "independent derivation" in str(exc)
    else:
        raise AssertionError("production support mutation escaped independent derivation")


def test_exact_segment_box_los_catches_thin_rotated_and_tangent_blockers() -> None:
    horizontal = ((-1.0, 0.0), (1.0, 0.0))
    assert _closed_segment_intersects_rotated_box(
        *horizontal,
        _box(size=(0.001, 0.20)),
    )
    assert _closed_segment_intersects_rotated_box(
        *horizontal,
        _box(size=(0.001, 0.20), yaw=np.pi / 4.0),
    )
    assert _closed_segment_intersects_rotated_box(
        (-1.0, 0.10),
        (1.0, 0.10),
        _box(size=(0.20, 0.20)),
    )
    assert not _closed_segment_intersects_rotated_box(
        (-1.0, 0.11),
        (1.0, 0.11),
        _box(size=(0.20, 0.20)),
    )


def _scene_result(**overrides) -> G3ExactSceneResult:
    values = dict(
        scene_id="scene-a",
        family="family",
        lattice_origin_xy_m=(0.0, 0.0),
        lattice_shape=(10, 10),
        physical_free_cells=80,
        physical_occupied_cells=20,
        snapshot_free_cells=50,
        snapshot_occupied_cells=30,
        snapshot_unknown_cells=20,
        independent_label_mismatch_cells=0,
        analytic_free_cells=60,
        unsafe_free_cells=0,
        conservative_false_reject_cells=10,
        strict_binary_label_mismatch_cells=20,
        snapshot_component_cells=50,
        independent_component_cells=50,
        component_mismatch_cells=0,
        astar_probe_count=8,
        astar_mismatch_count=0,
        canonical_component_cells=60,
        canonical_component_false_reject_cells=10,
        claim_endpoints_retained=4,
        beacon_count=4,
        exact_sim_tainted=True,
    )
    values.update(overrides)
    return G3ExactSceneResult(**values)


def test_summary_never_conflates_conservative_candidate_with_legacy_gate() -> None:
    row = _scene_result()
    truncated = summarize_exact_scenes(
        [row], source_bindings={"source": "0" * 64}
    )
    assert truncated["candidate_conservative_equivalence_pass"] is False
    rows = [replace(row, scene_id=f"scene-{index:02d}") for index in range(24)]
    summary = summarize_exact_scenes(rows, source_bindings={"source": "0" * 64})
    assert summary["candidate_conservative_equivalence_pass"] is True
    assert summary["legacy_strict_binary_equivalence_pass"] is False

    unsafe = replace(rows[0], unsafe_free_cells=1)
    summary = summarize_exact_scenes(
        [unsafe, *rows[1:]], source_bindings={"source": "0" * 64}
    )
    assert summary["candidate_conservative_equivalence_pass"] is False

    bad_route = replace(rows[0], astar_mismatch_count=1)
    summary = summarize_exact_scenes(
        [bad_route, *rows[1:]], source_bindings={"source": "0" * 64}
    )
    assert summary["candidate_conservative_equivalence_pass"] is False


def test_summary_rejects_duplicate_scene_identity() -> None:
    row = _scene_result()
    try:
        summarize_exact_scenes([row, row], source_bindings={"source": "0" * 64})
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate exact-equivalence scenes were accepted")
