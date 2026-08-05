from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from lewm.benchmarks import go2_perfect_camera_ray_field_audit_v2 as audit_v2
from lewm.datasets import go2_paired_navigation as labels_v3
from lewm_worlds.manifest import (
    BoxObject,
    CameraValidityConstraints,
    SceneManifest,
    SpawnSpec,
)
from lewm_worlds.planning_grid import InflatedOccupancyGrid


ROOT = Path(__file__).resolve().parents[2]


def _box() -> BoxObject:
    return BoxObject(
        object_id="obstacle",
        kind="obstacle",
        center_xyz_m=(1.6, 0.0, 0.5),
        size_xyz_m=(0.5, 0.8, 1.0),
        yaw_rad=0.2,
        material_id="wall",
    )


def _manifest(*, obstacles: tuple[BoxObject, ...] = ()) -> SceneManifest:
    return SceneManifest(
        scene_id="ray_audit_v2_unit",
        family="open_obstacle_field",
        difficulty_tier="unit_test",
        topology_seed=1,
        visual_seed=2,
        physics_seed=3,
        world_bounds_xy_m=((-5.0, -5.0), (7.0, 5.0)),
        spawn=SpawnSpec(
            xyz_m=(0.0, 0.0, 0.35),
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
        graph_nodes=(),
        graph_edges=(),
        obstacles=obstacles,
        landmarks=(),
        camera_constraints=CameraValidityConstraints(
            min_wall_thickness_m=0.08,
            near_m=0.05,
            far_m=200.0,
            min_camera_clearance_m=0.10,
        ),
        split="train",
    )


def _camera() -> labels_v3.CameraObservation:
    return labels_v3.CameraObservation(
        position_xyz_m=(0.326, 0.0, 0.393),
        lookat_xyz_m=(1.326, 0.0, 0.393),
        up_xyz=(0.0, 0.0, 1.0),
        horizontal_fov_deg=78.323,
        vertical_fov_deg=62.8370386364,
        near_m=0.05,
        image_width_px=224,
        image_height_px=168,
        obstacle_ray_stride_px=2,
    )


def _grid(manifest: SceneManifest) -> InflatedOccupancyGrid:
    return InflatedOccupancyGrid(
        manifest,
        cell_size_m=0.05,
        inflation_m=0.0,
        treat_landmarks_as_obstacles=True,
        treat_distractors_as_obstacles=True,
    )


def _reconstruct(
    grid: InflatedOccupancyGrid,
    *,
    free_mask: np.ndarray,
    rendered: tuple[BoxObject, ...],
    collision: tuple[BoxObject, ...],
) -> audit_v2.RayFieldRasterizationV2:
    return audit_v2.reconstruct_frame_from_perfect_rays(
        camera=audit_v2.CameraRaySpec.from_camera_observation(_camera()),
        rendered_obstacle_boxes=rendered,
        collision_obstacle_boxes=collision,
        base_xy_yaw=(0.0, 0.0, 0.0),
        physical_free_mask=free_mask,
        physical_origin_xy_m=grid.origin_xy,
        physical_cell_size_m=grid.cell_size_m,
    )


def test_v2_preserves_contract_assisted_v1_parity() -> None:
    obstacle = _box()
    manifest = _manifest(obstacles=(obstacle,))
    grid = _grid(manifest)
    expected, supervision, _observed = (
        labels_v3._observable_physical_raster_and_output_labels(
            grid,
            rendered_obstacle_boxes=(obstacle,),
            collision_obstacle_boxes=(obstacle,),
            base_xy_yaw=(0.0, 0.0, 0.0),
            camera=_camera(),
            local_grid=labels_v3.DEFAULT_LOCAL_GRID,
        )
    )
    result = _reconstruct(
        grid,
        free_mask=grid.free_mask,
        rendered=(obstacle,),
        collision=(obstacle,),
    )

    assert np.array_equal(result.contract_labels, expected)
    assert np.all(supervision)
    assert np.array_equal(
        result.observable_ray_only_labels,
        result.v1_rasterization.ray_only_pre_veto_labels,
    )


def test_observable_ray_only_ignores_physical_prior_and_collision_veto() -> None:
    grid = _grid(_manifest())
    collision = _box()
    all_false = np.zeros_like(grid.free_mask, dtype=bool)
    all_true = np.ones_like(grid.free_mask, dtype=bool)

    privileged_changes = _reconstruct(
        grid,
        free_mask=all_false,
        rendered=(),
        collision=(collision,),
    )
    no_privileged_changes = _reconstruct(
        grid,
        free_mask=all_true,
        rendered=(),
        collision=(),
    )

    assert np.array_equal(
        privileged_changes.observable_ray_only_labels,
        no_privileged_changes.observable_ray_only_labels,
    )
    assert np.any(
        privileged_changes.observable_ray_only_labels
        != privileged_changes.collision_vetoed_ray_only_labels
    )
    assert np.any(
        privileged_changes.contract_labels != no_privileged_changes.contract_labels
    )


def test_observable_arm_never_uses_collision_to_create_occupied() -> None:
    grid = _grid(_manifest())
    result = _reconstruct(
        grid,
        free_mask=grid.free_mask,
        rendered=(),
        collision=(_box(),),
    )

    changed = (
        result.observable_ray_only_labels
        != result.collision_vetoed_ray_only_labels
    )
    assert np.any(changed)
    assert np.all(result.observable_ray_only_labels[changed] == audit_v2.FREE_CLASS)
    assert np.all(
        result.collision_vetoed_ray_only_labels[changed] == audit_v2.UNKNOWN_CLASS
    )
    assert not np.any(result.observable_ray_only_labels == audit_v2.OCCUPIED_CLASS)


def _fake_report(index: int) -> dict:
    family = audit_v2.EXPECTED_FAMILIES[index // 64]
    identity_confusion = [[4096, 0, 0], [0, 0, 0], [0, 0, 0]]
    observable_confusion = copy.deepcopy(identity_confusion)
    observable_mismatches = 0
    collision_effect = 0
    if index == 0:
        observable_confusion = [[4093, 3, 0], [0, 0, 0], [0, 0, 0]]
        observable_mismatches = 3
        collision_effect = 3
    return {
        "schema": audit_v2.FRAME_AUDIT_SCHEMA,
        "frame_key": {"family": family, "frame": index},
        "authoritative_labels_sha256": f"{index:064x}",
        "contract_labels_sha256": f"{index:064x}",
        "ray_only_labels_sha256": f"{index:064x}",
        "observable_ray_only_labels_sha256": f"{index + 1:064x}",
        "contract_confusion_reference_rows": copy.deepcopy(identity_confusion),
        "ray_only_confusion_reference_rows": copy.deepcopy(identity_confusion),
        "observable_ray_only_confusion_reference_rows": observable_confusion,
        "contract_mismatch_cell_count": 0,
        "ray_only_mismatch_cell_count": 0,
        "observable_ray_only_mismatch_cell_count": observable_mismatches,
        "collision_veto_effect_on_ray_only_cell_count": collision_effect,
    }


def test_v2_summary_quantifies_observable_only_errors_by_family() -> None:
    reports = [_fake_report(index) for index in range(320)]

    summary = audit_v2.summarize_exact_fit(reports)

    assert summary["contract_assisted"]["exact"] is True
    assert summary["collision_vetoed_ray_only"]["exact"] is True
    assert summary["observable_ray_only"]["exact"] is False
    assert summary["observable_ray_only"]["mismatch_cell_count"] == 3
    assert summary["observable_ray_only"]["mismatch_frame_count"] == 1
    assert summary["observable_ray_only"]["mismatch_class_transitions"] == {
        "unknown->free": 3
    }
    assert summary["families"]["open_obstacle_field"]["observable_ray_only"][
        "mismatch_cell_count"
    ] == 3
    assert summary["families"]["rough_local_dynamics"]["observable_ray_only"][
        "mismatch_cell_count"
    ] == 0
    assert summary["collision_veto_effect_on_ray_only_cell_count"] == 3


def test_v2_summary_fails_closed_on_count_or_family() -> None:
    reports = [_fake_report(index) for index in range(320)]
    with pytest.raises(ValueError, match="exactly 320"):
        audit_v2.summarize_exact_fit(reports[:-1])
    reports[-1]["frame_key"]["family"] = "g2_evaluation"
    with pytest.raises(ValueError, match="unregistered family"):
        audit_v2.summarize_exact_fit(reports)


def test_neutral_v2_import_does_not_load_protected_package_graph() -> None:
    code = f"""
import importlib.util, pathlib, sys
path = pathlib.Path({str((ROOT / 'lewm/benchmarks/go2_perfect_camera_ray_field_audit_v2.py').resolve())!r})
spec = importlib.util.spec_from_file_location('neutral_v2_test', path)
module = importlib.util.module_from_spec(spec)
sys.modules['neutral_v2_test'] = module
spec.loader.exec_module(module)
assert not [name for name in sys.modules if name == 'lewm' or name.startswith('lewm.')]
assert not [name for name in sys.modules if name == 'lewm_worlds' or name.startswith('lewm_worlds.')]
"""
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


def test_v2_dry_run_is_deterministic_and_exercises_third_arm() -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/audit_go2_perfect_camera_ray_field_fit_v2.py"),
        "--dry-run",
    ]
    first = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    second = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(first.stdout)

    assert first.stdout == second.stdout
    assert payload["dry_run"] is True
    assert payload["contract_parity"] is True
    assert payload["deterministic"] is True
    assert payload["observable_ray_only_omits_collision_veto"] is True
    assert payload["g2_or_holdout_payload_opened"] is False
