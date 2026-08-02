from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as contract
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as collector


@dataclass(frozen=True)
class _CameraSafetyConfig:
    safe_clearance_m: float
    near_plane_m: float
    fov_deg: float
    fov_axis: str = "horizontal"
    aspect_ratio: float = 4.0 / 3.0
    max_retract_m: float = 0.08


class _Camera:
    def __init__(
        self,
        *,
        rgb: np.ndarray | None = None,
        depth: np.ndarray | None = None,
    ) -> None:
        self._is_batched = False
        self.rgb = (
            np.zeros((224, 224, 3), dtype=np.uint8) if rgb is None else rgb
        )
        self.depth = (
            np.ones((224, 224), dtype=np.float32) if depth is None else depth
        )
        self.calls: list[tuple[str, object]] = []

    def set_pose(self, *, pos: object, lookat: object, up: object) -> None:
        self.calls.append((
            "set_pose",
            tuple(tuple(float(x) for x in value) for value in (pos, lookat, up)),
        ))

    def render(self, **kwargs: object) -> tuple[object, ...]:
        self.calls.append(("render", dict(kwargs)))
        if kwargs.get("rgb") is True:
            return (self.rgb,)
        if kwargs.get("depth") is True:
            return (None, self.depth)
        raise AssertionError(f"unexpected render call: {kwargs}")


def _render_build(camera: _Camera) -> SimpleNamespace:
    camera_config = SimpleNamespace(
        xyz_body_m=(0.31, -0.02, 0.14),
        rpy_body_rad=(0.01, -0.08, 0.03),
    )
    return SimpleNamespace(
        camera=camera,
        n_envs=1,
        native_resolution=(224, 224),
        stored_resolution=(224, 224),
        visual_mode=contract.TEXTURED_V03_VISUAL_MODE,
        pack=SimpleNamespace(camera=camera_config),
        visible_objects=(),
        to_hwc_uint8=lambda value: np.asarray(value, dtype=np.uint8),
    )


def _historical_pose_recorder(calls: list[dict[str, object]]):
    def historical_pose(
        base_state: dict[str, object], mount: dict[str, object]
    ) -> dict[str, list[float]]:
        calls.append({"base_state": base_state, "mount": mount})
        position = base_state["pose_world"]["position"]  # type: ignore[index]
        return {
            "position": [position["x"], position["y"], position["z"]],  # type: ignore[index]
            "lookat": [float(position["x"]) + 1.0, position["y"], position["z"]],  # type: ignore[index]
            "up": [0.0, 0.0, 1.0],
        }

    return historical_pose


def test_textured_v03_rgb_helper_preserves_historical_pose_inputs() -> None:
    camera = _Camera()
    build = _render_build(camera)
    pose_calls: list[dict[str, object]] = []

    result = collector._render_textured_v03_rgb_from_base_pose(  # noqa: SLF001
        build,
        base_position_xyz_m=[1.0, 2.0, 0.3],
        base_quaternion_wxyz=[2.0, 0.1, 0.2, 0.3],
        historical_camera_pose_from_payload=_historical_pose_recorder(pose_calls),
    )

    assert pose_calls[0]["base_state"]["quat_world_xyzw"] == [  # type: ignore[index]
        pytest.approx(0.1),
        pytest.approx(0.2),
        pytest.approx(0.3),
        pytest.approx(2.0),
    ]
    assert pose_calls[0]["mount"] == {
        "xyz_body_m": [0.31, -0.02, 0.14],
        "rpy_body_rad": [0.01, -0.08, 0.03],
    }
    assert camera.calls[0][0] == "set_pose"
    assert camera.calls[1] == ("render", {"rgb": True, "depth": False})
    assert result["rgb"].shape == (224, 224, 3)


def test_textured_v03_capture_renders_exact_rgb_before_auxiliary_depth() -> None:
    camera = _Camera()
    build = _render_build(camera)
    pose_calls: list[dict[str, object]] = []
    safety_calls: list[dict[str, object]] = []
    quality_calls: list[dict[str, object]] = []

    def safety_metrics(*args: object) -> dict[str, object]:
        config = args[2]
        safety_calls.append({"config": config})
        return {"unsafe": True, "minimum_forward_clearance_m": 0.0}

    def assess(
        rgb: np.ndarray,
        depth: np.ndarray,
        *,
        require_depth: bool,
        camera_safety: dict[str, object],
    ) -> dict[str, object]:
        quality_calls.append({
            "rgb_shape": rgb.shape,
            "depth_shape": depth.shape,
            "require_depth": require_depth,
            "camera_safety": camera_safety,
        })
        return {"valid": True, "reasons": []}

    stage = {
        "native_render_wall_seconds": 0.0,
        "camera_quality_resize_wall_seconds": 0.0,
    }
    result = collector._capture_replayed_frame_textured_v03(  # noqa: SLF001
        build,
        components={
            "base_pos_world": np.asarray([[1.0, 2.0, 0.3]], dtype=np.float32),
            "base_quat_wxyz": np.asarray(
                [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
            ),
        },
        env_index=0,
        historical_camera_pose_from_payload=_historical_pose_recorder(pose_calls),
        camera_pose_from_dict=lambda value: value,
        camera_safety_metrics=safety_metrics,
        camera_safety_config_from_pack=lambda _pack: _CameraSafetyConfig(
            safe_clearance_m=0.1,
            near_plane_m=0.05,
            fov_deg=70.0,
            max_retract_m=0.5,
        ),
        assess_rendered_frame=assess,
        stage_wall_times=stage,
    )

    render_calls = [call for call in camera.calls if call[0] == "render"]
    assert render_calls == [
        ("render", {"rgb": True, "depth": False}),
        ("render", {"rgb": False, "depth": True, "force_render": True}),
    ]
    config = safety_calls[0]["config"]
    assert config.fov_deg == contract.TEXTURED_V03_RENDER_CONTRACT[
        "genesis_yfov_deg"
    ]
    assert config.aspect_ratio == 1.0
    assert config.max_retract_m == 0.0
    assert quality_calls[0]["camera_safety"]["retracted_m"] == 0.0
    assert result["stored_rgb"].shape == (224, 224, 3)
    assert result["depth_persisted"] is False


@pytest.mark.parametrize(
    ("rgb_shape", "depth_shape", "message"),
    [
        ((223, 224, 3), (224, 224), "historical RGB shape changed"),
        ((224, 224, 3), (223, 224), "native RGB/depth shape changed"),
    ],
)
def test_textured_v03_capture_rejects_malformed_render_arrays(
    rgb_shape: tuple[int, ...],
    depth_shape: tuple[int, ...],
    message: str,
) -> None:
    camera = _Camera(
        rgb=np.zeros(rgb_shape, dtype=np.uint8),
        depth=np.ones(depth_shape, dtype=np.float32),
    )
    build = _render_build(camera)
    with pytest.raises(contract.PilotContractError, match=message):
        collector._capture_replayed_frame_textured_v03(  # noqa: SLF001
            build,
            components={
                "base_pos_world": np.asarray(
                    [[1.0, 2.0, 0.3]], dtype=np.float32
                ),
                "base_quat_wxyz": np.asarray(
                    [[1.0, 0.0, 0.0, 0.0]], dtype=np.float32
                ),
            },
            env_index=0,
            historical_camera_pose_from_payload=_historical_pose_recorder([]),
            camera_pose_from_dict=lambda value: value,
            camera_safety_metrics=lambda *_args: {"unsafe": False},
            camera_safety_config_from_pack=lambda _pack: _CameraSafetyConfig(
                safe_clearance_m=0.1,
                near_plane_m=0.05,
                fov_deg=70.0,
            ),
            assess_rendered_frame=lambda *_args, **_kwargs: {
                "valid": True,
                "reasons": [],
            },
            stage_wall_times={
                "native_render_wall_seconds": 0.0,
                "camera_quality_resize_wall_seconds": 0.0,
            },
        )


def test_textured_v03_mesh_cache_binds_expected_obj_bytes(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.obj"
    expected = "v 0 0 0\n"

    def cached_box_obj(
        _size: tuple[float, ...], *, tiles_per_m: float
    ) -> str:
        assert tiles_per_m == 0.7
        mesh_path.write_text(expected, encoding="utf-8")
        return str(mesh_path)

    bindings = collector._prepare_textured_v03_mesh_cache(  # noqa: SLF001
        {
            "walls": [
                {"kind": "wall", "size_xyz_m": [1.0, 0.2, 0.5]}
            ],
            "obstacles": [],
            "landmarks": [],
        },
        runtime={
            "category_for_kind": lambda _kind: "wall",
            "cached_box_obj": cached_box_obj,
            "box_obj_text": lambda _size, *, tiles_per_m: expected,
        },
        selected_textures={"wall": "/synthetic/wall.png"},
    )

    assert len(bindings) == 1
    assert bindings[0]["path"] == str(mesh_path.resolve())
    assert bindings[0]["byte_count"] == len(expected.encode())


def test_textured_v03_mesh_cache_rejects_wrong_derived_bytes(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.obj"
    mesh_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(contract.PilotContractError, match="bytes or identity changed"):
        collector._prepare_textured_v03_mesh_cache(  # noqa: SLF001
            {
                "walls": [
                    {"kind": "wall", "size_xyz_m": [1.0, 0.2, 0.5]}
                ],
                "obstacles": [],
                "landmarks": [],
            },
            runtime={
                "category_for_kind": lambda _kind: "wall",
                "cached_box_obj": lambda _size, *, tiles_per_m: str(mesh_path),
                "box_obj_text": lambda _size, *, tiles_per_m: "expected\n",
            },
            selected_textures={"wall": "/synthetic/wall.png"},
        )
