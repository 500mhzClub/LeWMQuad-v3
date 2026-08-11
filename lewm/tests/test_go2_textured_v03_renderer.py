"""Non-Genesis tests for the exact historical textured_v03 wrapper."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from lewm.oracle.go2_textured_v03_renderer import (
    RESOLUTION_WH,
    BasePose,
    TexturedV03Renderer,
    TexturedV03RendererError,
    _HistoricalApi,
    _historical_api,
    capture_base_pose,
    renderer_contract,
    renderer_contract_digest,
    validate_camera_pack,
)
from scripts.render_replay_v03 import _to_hwc_uint8


class _Robot:
    def __init__(self, position=(1.0, 2.0, 0.5), quaternion=(1.0, 0.0, 0.0, 0.0)):
        self.position = np.asarray(position, dtype=np.float32)
        self.quaternion = np.asarray(quaternion, dtype=np.float32)

    def get_pos(self):
        return self.position

    def get_quat(self):
        return self.quaternion


class _Camera:
    def __init__(self, output):
        self.output = output
        self.pose_calls = []
        self.render_calls = []

    def set_pose(self, **kwargs):
        self.pose_calls.append(kwargs)

    def render(self, **kwargs):
        self.render_calls.append(kwargs)
        return self.output


def _context(*, camera_updates=None, robot=None):
    camera_values = {
        "parent_link": "camera_link",
        "xyz_body_m": (0.3, 0.0, 0.2),
        "rpy_body_rad": (0.0, 0.0, 0.0),
        "native_resolution": (640, 480),
        "training_resolution": (224, 224),
        "fov_axis": "horizontal",
        "fov_deg": 78.323,
        "near_m": 0.05,
        "far_m": 200.0,
    }
    camera_values.update(camera_updates or {})
    return SimpleNamespace(
        runner=SimpleNamespace(_as_np=lambda value: np.asarray(value)),
        build=SimpleNamespace(robot=robot or _Robot()),
        pack=SimpleNamespace(
            scene_id="scene_001",
            camera=SimpleNamespace(**camera_values),
            # Deliberately non-null: the renderer must ignore this and use the
            # nominal camera values above rather than effective jitter.
            camera_extrinsic_jitter=object(),
        ),
    )


def _api(calls, camera):
    def build_scene(gs, manifest, **kwargs):
        calls.append((gs, manifest, kwargs))
        return object(), camera

    return _HistoricalApi(
        build_scene=build_scene,
        camera_pose_from_payload=_camera_pose_from_payload,
        to_hwc_uint8=_to_hwc_uint8,
    )


def _camera_pose_from_payload(payload, mount):
    """Identity-base-quaternion stub matching the historical helper's schema."""

    position = payload["pose_world"]["position"]
    assert payload["quat_world_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert mount["xyz_body_m"] == [0.3, 0.0, 0.2]
    assert mount["rpy_body_rad"] == [0.0, 0.0, 0.0]
    camera_position = [
        float(position["x"]) + mount["xyz_body_m"][0],
        float(position["y"]) + mount["xyz_body_m"][1],
        float(position["z"]) + mount["xyz_body_m"][2],
    ]
    return {
        "position": camera_position,
        "lookat": [camera_position[0] + 1.0, camera_position[1], camera_position[2]],
        "up": [0.0, 0.0, 1.0],
    }


def test_contract_is_exact_versioned_and_digest_bound():
    contract = renderer_contract()
    assert contract["static_scene"]["resolution_wh"] == [224, 224]
    assert contract["static_scene"]["fov_argument_deg"] == 78.323
    assert contract["static_scene"]["textures"] is True
    assert contract["pose"]["camera_extrinsic_jitter"] is False
    assert contract["render"]["downsample_or_resize"] is False
    assert renderer_contract_digest() == (
        "df70a0c16ad421ae93a93c4d9dda0fd4d6f154f42d9710c7fc2f0242c3e8cb1b"
    )


def test_default_api_resolves_exact_historical_functions_without_genesis_init():
    api = _historical_api()
    assert api.build_scene.__module__ == "scripts.render_replay_v03"
    assert api.to_hwc_uint8.__module__ == "scripts.render_replay_v03"
    assert api.camera_pose_from_payload.__module__ == "lewm_genesis.render_replay"


def test_base_pose_capture_is_singleton_only_and_converts_wxyz_to_xyzw():
    pose = capture_base_pose(_context())
    assert pose == BasePose((1.0, 2.0, 0.5), (1.0, 0.0, 0.0, 0.0))
    assert pose.replay_payload()["quat_world_xyzw"] == [0.0, 0.0, 0.0, 1.0]

    batched = _context(robot=_Robot(
        position=((0.0, 0.0, 0.5), (1.0, 0.0, 0.5)),
        quaternion=((1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)),
    ))
    with pytest.raises(TexturedV03RendererError, match="non-batched"):
        capture_base_pose(batched)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"training_resolution": (112, 112)}, "training resolution"),
        ({"fov_axis": "vertical"}, "fov_axis"),
        ({"fov_deg": 78.322}, "fov_deg"),
        ({"near_m": 0.051}, "near_m"),
        ({"far_m": 199.0}, "far_m"),
    ],
)
def test_camera_pack_validation_is_exact(updates, message):
    with pytest.raises(TexturedV03RendererError, match=message):
        validate_camera_pack(_context(camera_updates=updates).pack)


def test_build_and_render_use_only_the_historical_static_path_and_nominal_mount():
    calls = []
    # Historical converter removes the singleton batch and maps float RGB to uint8.
    rgb = np.full((1, 224, 224, 3), 0.5, dtype=np.float32)
    camera = _Camera((rgb,))
    times = iter((10.0, 10.25, 20.0, 20.5))
    gs = object()
    raw_manifest = {
        "scene_id": "scene_001",
        "visual_seed": 7,
        "walls": [], "obstacles": [], "landmarks": [],
        "visual_randomization": {
            "distractor_objects": [{"object_id": "ignored_distractor"}]
        },
    }
    renderer = TexturedV03Renderer(
        _context(), gs=gs, raw_manifest=raw_manifest,
        _api=_api(calls, camera), _clock=lambda: next(times),
    )
    assert renderer.scene_build_runtime_s == pytest.approx(0.25)
    assert len(calls) == 1
    assert calls[0][0] is gs
    assert calls[0][1] == raw_manifest
    assert calls[0][2] == {
        "fov": 78.323, "near": 0.05, "far": 200.0,
        "res": RESOLUTION_WH, "textures": True,
    }

    result = renderer.render(_context())
    assert result.runtime_s == pytest.approx(0.5)
    assert result.image.shape == (224, 224, 3)
    assert result.image.dtype == np.uint8
    assert np.all(result.image == 127)
    assert camera.render_calls == [{"rgb": True, "depth": False}]
    assert result.camera_pose_world["position"] == pytest.approx([1.3, 2.0, 0.7])
    assert result.camera_pose_world["lookat"] == pytest.approx([2.3, 2.0, 0.7])
    assert result.camera_pose_world["up"] == pytest.approx([0.0, 0.0, 1.0])
    assert camera.pose_calls == [{
        "pos": pytest.approx((1.3, 2.0, 0.7)),
        "lookat": pytest.approx((2.3, 2.0, 0.7)),
        "up": pytest.approx((0.0, 0.0, 1.0)),
    }]


def test_stored_pose_renders_independently_after_context_has_moved():
    context = _context()
    stored_pose = capture_base_pose(context)
    # The live CPU context moves after the snapshot pose was captured.
    context.build.robot.position[:] = (99.0, 88.0, 77.0)
    camera = _Camera(np.zeros((224, 224, 3), dtype=np.uint8))
    times = iter((1.0, 1.1, 2.0, 2.4))
    renderer = TexturedV03Renderer(
        context, gs=object(), raw_manifest={"scene_id": "scene_001"},
        _api=_api([], camera), _clock=lambda: next(times),
    )
    result = renderer.render_pose(stored_pose)
    assert result.runtime_s == pytest.approx(0.4)
    assert result.camera_pose_world["position"] == pytest.approx([1.3, 2.0, 0.7])
    assert result.camera_pose_world["lookat"] == pytest.approx([2.3, 2.0, 0.7])


def test_output_shape_and_dtype_are_strictly_rejected():
    context = _context()
    raw_manifest = {"scene_id": "scene_001"}
    for output, message in (
        (np.zeros((224, 224, 4), dtype=np.uint8), "shape"),
        (np.zeros((224, 224, 3), dtype=np.float32), "dtype"),
    ):
        camera = _Camera(output)
        renderer = TexturedV03Renderer(
            context, gs=object(), raw_manifest=raw_manifest,
            _api=_HistoricalApi(
                build_scene=lambda *args, **kwargs: (object(), camera),
                camera_pose_from_payload=_camera_pose_from_payload,
                to_hwc_uint8=lambda value: value,
            ),
        )
        with pytest.raises(TexturedV03RendererError, match=message):
            renderer.render(context)
