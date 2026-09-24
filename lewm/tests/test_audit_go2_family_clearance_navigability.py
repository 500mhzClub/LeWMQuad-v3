from __future__ import annotations

import json
from pathlib import Path

import pytest

from lewm.datasets.go2_paired_navigation import OBSERVABLE_FOOTPRINT_RADIUS_M
from scripts.audit_go2_family_clearance_navigability import (
    DEV_OUTPUT_ROOT,
    DEFAULT_FOOTPRINT_RADIUS_M,
    SCENE_CORPUS_ROOT,
    assert_binding_unchanged,
    audit_scene,
    read_bound_file,
    require_development_output,
    require_scene_corpus,
    summarize_records,
)


def _write_manifest(
    path: Path,
    *,
    spawn_xy: tuple[float, float] = (0.0, 0.0),
    walls: list[dict] | None = None,
    obstacles: list[dict] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "world_bounds_xy_m": [[-1.0, -1.0], [1.0, 1.0]],
                "spawn": {"xyz_m": [spawn_xy[0], spawn_xy[1], 0.0]},
                "walls": walls or [],
                "obstacles": obstacles or [],
            }
        )
    )


def _body(
    *, center_xy: tuple[float, float], size_xy: tuple[float, float]
) -> dict:
    return {
        "center_xyz_m": [center_xy[0], center_xy[1], 0.5],
        "size_xyz_m": [size_xy[0], size_xy[1], 1.0],
        "yaw_rad": 0.0,
    }


def _audit(path: Path, radius: float) -> dict:
    return audit_scene(("scene", "family", "train", str(path), radius))


def test_default_radius_matches_canonical_planning_contract() -> None:
    assert DEFAULT_FOOTPRINT_RADIUS_M == pytest.approx(
        OBSERVABLE_FOOTPRINT_RADIUS_M
    )
    assert DEFAULT_FOOTPRINT_RADIUS_M == pytest.approx(0.47)


def test_infeasible_spawn_does_not_fall_back_to_largest_component(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        obstacles=[_body(center_xy=(0.0, 0.0), size_xy=(0.4, 0.4))],
    )

    record = _audit(manifest, 0.24)

    assert record["spawn_in_bounds"] is True
    assert record["spawn_traversable"] is False
    assert record["frac_reachable"] == 0.0
    assert record["frac_reachable_of_traversable"] == 0.0
    assert record["frac_largest_component"] > 0.0
    assert record["manifest_bytes"] == manifest.stat().st_size
    assert len(record["manifest_sha256"]) == 64


def test_canonical_disc_rejects_sub_diameter_corridor(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        walls=[
            _body(center_xy=(0.0, -0.7), size_xy=(2.0, 0.6)),
            _body(center_xy=(0.0, 0.7), size_xy=(2.0, 0.6)),
        ],
    )

    canonical = _audit(manifest, DEFAULT_FOOTPRINT_RADIUS_M)
    directional_proxy = _audit(manifest, 0.24)

    assert canonical["spawn_traversable"] is False
    assert canonical["frac_reachable"] == 0.0
    assert directional_proxy["spawn_traversable"] is True
    assert directional_proxy["frac_reachable"] > 0.0


@pytest.mark.parametrize("spawn_xy", [(2.0, 0.0), (-1.001, 0.0)])
def test_out_of_bounds_spawn_fails_closed(
    tmp_path: Path, spawn_xy: tuple[float, float]
) -> None:
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, spawn_xy=spawn_xy)

    record = _audit(manifest, 0.24)

    assert record["spawn_in_bounds"] is False
    assert record["spawn_traversable"] is False
    assert record["frac_reachable"] == 0.0


def test_summary_keeps_spawn_and_largest_component_distinct(
    tmp_path: Path,
) -> None:
    blocked_manifest = tmp_path / "blocked.json"
    open_manifest = tmp_path / "open.json"
    _write_manifest(
        blocked_manifest,
        obstacles=[_body(center_xy=(0.0, 0.0), size_xy=(0.4, 0.4))],
    )
    _write_manifest(open_manifest)
    blocked = _audit(blocked_manifest, 0.24)
    opened = _audit(open_manifest, 0.24)

    summary = summarize_records([blocked, opened])

    assert summary["all"]["scene_count"] == 2
    assert summary["families"]["family"]["spawn_traversable_count"] == 1
    assert summary["families"]["family"]["fully_spawn_connected_count"] == 1
    assert summary["families"]["family"]["frac_largest_component_mean"] > 0.0


def test_geometry_output_cannot_escape_development_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="development output must remain"):
        require_development_output(tmp_path / "audit.json")
    inside = DEV_OUTPUT_ROOT / "family_clearance_audit" / "audit.json"
    assert require_development_output(inside) == inside


def test_geometry_input_cannot_escape_scene_corpus_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="corpus input must remain"):
        require_scene_corpus(tmp_path / "corpus")
    inside = SCENE_CORPUS_ROOT / "synthetic"
    assert require_scene_corpus(inside) == inside


def test_geometry_input_binding_detects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "manifest.json"
    source.write_text("{}\n")
    _raw, binding = read_bound_file(source)
    assert_binding_unchanged(binding)

    source.write_text('{"changed": true}\n')
    with pytest.raises(RuntimeError, match="input changed after audit"):
        assert_binding_unchanged(binding)
