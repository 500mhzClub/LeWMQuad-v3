from __future__ import annotations

import numpy as np

from lewm_genesis.vision_quality import assess_rendered_frame


def _checkerboard(size: int = 32) -> np.ndarray:
    board = (np.indices((size, size)).sum(axis=0) % 2).astype(np.uint8) * 255
    return np.repeat(board[..., None], 3, axis=-1)


def test_uniform_grey_rgb_is_low_information_without_depth() -> None:
    rgb = np.full((32, 32, 3), 128, dtype=np.uint8)

    quality = assess_rendered_frame(rgb, None, require_depth=False)

    assert quality["valid"] is False
    assert "low_rgb_texture" in quality["invalid_reasons"]
    assert quality["rgb_stats"]["low_texture"] is True


def test_textured_rgb_without_depth_remains_valid() -> None:
    quality = assess_rendered_frame(_checkerboard(), None, require_depth=False)

    assert quality["valid"] is True
    assert quality["invalid_reasons"] == []


def test_near_wall_depth_rejects_forward_filling_wall_view() -> None:
    rgb = _checkerboard()
    depth = np.full((32, 32), 0.16, dtype=np.float32)

    quality = assess_rendered_frame(rgb, depth, require_depth=True)

    assert quality["valid"] is False
    assert "near_wall_depth" in quality["invalid_reasons"]
    assert quality["depth_stats"]["near_wall"] is True
    assert quality["depth_stats"]["near_wall_fraction"] == 1.0


def test_camera_safety_forward_hit_rejects_geometry_too_close_ahead() -> None:
    quality = assess_rendered_frame(
        _checkerboard(),
        None,
        require_depth=False,
        camera_safety={
            "unsafe": False,
            "forward_hit_m": 0.12,
            "frustum_min_hit_m": 0.12,
        },
    )

    assert quality["valid"] is False
    assert "near_forward_geometry" in quality["invalid_reasons"]
