from __future__ import annotations

from lewm_worlds.families import (
    DEPLOYMENT_CORRIDOR_NARROW_BAND_M,
    DEPLOYMENT_CORRIDOR_WIDE_BAND_M,
    build_family_manifest,
    registered_families,
)


def test_deployment_maze_is_registered_and_uses_platform_clear_corridors() -> None:
    assert "go2_deployment_medium_maze" in registered_families()
    minimum = DEPLOYMENT_CORRIDOR_NARROW_BAND_M[0]
    maximum = DEPLOYMENT_CORRIDOR_WIDE_BAND_M[1]
    for seed in range(40):
        manifest = build_family_manifest(
            scene_seed=seed,
            family="go2_deployment_medium_maze",
            split="candidate",
            difficulty_tier=None,
        )
        widths = [edge.width_m for edge in manifest.graph_edges]
        assert widths
        assert min(widths) >= minimum
        assert max(widths) <= maximum
        colors = [
            landmark.material_id.removeprefix("landmark_")
            for landmark in manifest.landmarks
        ]
        assert sorted(colors) == ["blue", "green", "red", "yellow"]
        assert len(set(colors)) == 4
