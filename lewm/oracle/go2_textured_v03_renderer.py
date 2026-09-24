"""Exact historical ``textured_v03`` renderer for planning branch frames.

The scientific branch simulator remains CPU-only.  This wrapper creates a
separate, single-environment static Genesis scene solely for RGB rendering and
places its camera from the CPU branch context's current base pose.  It binds
the historical 224-square renderer exactly: the nominal (unjittered) platform
camera mount, horizontal FOV value passed through as 78.323 degrees, 0.05/200 m
clip planes, deterministic textures, and no robot, distractors, safety
retraction, jitter, or resizing.

Genesis is imported by the caller and passed as ``gs``.  Merely importing this
module never initializes Genesis, a renderer, or an accelerator.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "go2_textured_v03_renderer_contract_v1"
VERSION = "historical_textured_v03_224_nominal_mount_v1"
RESOLUTION_WH = (224, 224)
FOV_AXIS = "horizontal"
FOV_DEG = 78.323
NEAR_M = 0.05
FAR_M = 200.0


class TexturedV03RendererError(ValueError):
    """A context, camera pack, pose, manifest, or rendered frame is invalid."""


@dataclass(frozen=True)
class BasePose:
    """One non-batched base pose captured from a V1 ``BranchContext``."""

    position_world_xyz: tuple[float, float, float]
    quaternion_world_wxyz: tuple[float, float, float, float]

    def replay_payload(self) -> dict[str, Any]:
        """Return the exact payload order expected by historical replay code."""

        x, y, z = self.position_world_xyz
        qw, qx, qy, qz = self.quaternion_world_wxyz
        return {
            "pose_world": {"position": {"x": x, "y": y, "z": z}},
            "quat_world_xyzw": [qx, qy, qz, qw],
        }


@dataclass(frozen=True)
class TexturedV03RenderResult:
    """One durable historical RGB frame and its exact placement evidence."""

    image: np.ndarray
    camera_pose_world: dict[str, list[float]]
    runtime_s: float


@dataclass(frozen=True)
class _HistoricalApi:
    build_scene: Callable[..., tuple[Any, Any]]
    camera_pose_from_payload: Callable[[dict[str, Any], dict[str, Any]], Any]
    to_hwc_uint8: Callable[[Any], np.ndarray]


def renderer_contract() -> dict[str, Any]:
    """Return the complete versioned historical-render contract."""

    return {
        "schema": SCHEMA,
        "version": VERSION,
        "camera_pack_validation": {
            "training_resolution_wh": list(RESOLUTION_WH),
            "fov_axis": FOV_AXIS,
            "fov_deg": FOV_DEG,
            "near_m": NEAR_M,
            "far_m": FAR_M,
            "comparison": "exact equality after numeric conversion",
        },
        "static_scene": {
            "builder": "scripts.render_replay_v03.build_scene",
            "raw_scene_manifest": True,
            "n_envs": 1,
            "batched": False,
            "resolution_wh": list(RESOLUTION_WH),
            "fov_argument_deg": FOV_DEG,
            "near_argument_m": NEAR_M,
            "far_argument_m": FAR_M,
            "textures": True,
            "included": ["floor", "walls", "obstacles", "landmarks"],
            "excluded": ["robot", "visual_randomization.distractor_objects"],
        },
        "pose": {
            "base_source": (
                "V1 BranchContext runner._as_np(build.robot.get_pos/get_quat)"
            ),
            "base_quaternion_input_order": "wxyz",
            "replay_quaternion_payload_order": "xyzw",
            "camera_mount": "pack.camera nominal xyz_body_m/rpy_body_rad",
            "placement": "lewm_genesis.render_replay._camera_pose_from_payload",
            "camera_extrinsic_jitter": False,
            "camera_safety_retraction": False,
        },
        "render": {
            "primary_api": "render_pose(captured BasePose)",
            "context_convenience_api": "render(context) captures then delegates",
            "call": "camera.render(rgb=True, depth=False)",
            "conversion": "scripts.render_replay_v03._to_hwc_uint8",
            "output_shape_hwc": [224, 224, 3],
            "output_dtype": "uint8",
            "downsample_or_resize": False,
            "runtime_s_scope": (
                "base capture, historical camera placement, RGB render, pixel "
                "conversion, validation, and durable copy"
            ),
        },
    }


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def renderer_contract_digest() -> str:
    return hashlib.sha256(_canonical_json(renderer_contract())).hexdigest()


def raw_manifest_digest(raw_manifest: Mapping[str, Any]) -> str:
    """Bind the exact raw mapping supplied to historical ``build_scene``."""

    if not isinstance(raw_manifest, Mapping):
        raise TexturedV03RendererError("raw_manifest must be a mapping")
    try:
        return hashlib.sha256(_canonical_json(raw_manifest)).hexdigest()
    except (TypeError, ValueError) as exc:
        raise TexturedV03RendererError(
            "raw_manifest must be finite and canonically JSON serialisable"
        ) from exc


def _exact_tuple(value: Any, length: int, field: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise TexturedV03RendererError(f"{field} must contain {length} numbers") from exc
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise TexturedV03RendererError(f"{field} must contain {length} finite numbers")
    return result


def validate_camera_pack(pack: Any) -> dict[str, Any]:
    """Validate the exact historical camera and return its nominal mount."""

    camera = getattr(pack, "camera", None)
    if camera is None:
        raise TexturedV03RendererError("BranchContext pack has no camera")
    try:
        resolution = tuple(int(value) for value in camera.training_resolution)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TexturedV03RendererError(
            "pack.camera.training_resolution must be exactly (224, 224)"
        ) from exc
    if resolution != RESOLUTION_WH:
        raise TexturedV03RendererError(
            f"training resolution must be exactly {RESOLUTION_WH}, got {resolution}"
        )
    if getattr(camera, "fov_axis", None) != FOV_AXIS:
        raise TexturedV03RendererError("camera fov_axis must be exactly 'horizontal'")
    for field, expected in (("fov_deg", FOV_DEG), ("near_m", NEAR_M),
                            ("far_m", FAR_M)):
        try:
            actual = float(getattr(camera, field))
        except (AttributeError, TypeError, ValueError) as exc:
            raise TexturedV03RendererError(f"camera {field} is missing or invalid") from exc
        if not math.isfinite(actual) or actual != expected:
            raise TexturedV03RendererError(
                f"camera {field} must be exactly {expected}, got {actual}"
            )
    xyz = _exact_tuple(getattr(camera, "xyz_body_m", None), 3, "camera.xyz_body_m")
    rpy = _exact_tuple(getattr(camera, "rpy_body_rad", None), 3, "camera.rpy_body_rad")
    return {
        "parent_link": getattr(camera, "parent_link", None),
        "xyz_body_m": list(xyz),
        "rpy_body_rad": list(rpy),
    }


def _single_row(array: Any, width: int, name: str) -> np.ndarray:
    result = np.asarray(array, dtype=np.float64)
    if result.shape == (width,):
        pass
    elif result.shape == (1, width):
        result = result[0]
    else:
        raise TexturedV03RendererError(
            f"{name} must be non-batched ({width},) or singleton (1, {width}); "
            f"got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise TexturedV03RendererError(f"{name} contains a non-finite value")
    return result


def capture_base_pose(context: Any) -> BasePose:
    """Capture current base position and raw Genesis WXYZ quaternion."""

    try:
        runner = context.runner
        robot = context.build.robot
        position = runner._as_np(robot.get_pos())
        quaternion_wxyz = runner._as_np(robot.get_quat())
    except AttributeError as exc:
        raise TexturedV03RendererError(
            "context must expose V1 BranchContext runner/build.robot pose APIs"
        ) from exc
    position_row = _single_row(position, 3, "base position")
    quaternion_row = _single_row(quaternion_wxyz, 4, "base quaternion WXYZ")
    return BasePose(
        position_world_xyz=tuple(float(value) for value in position_row),
        quaternion_world_wxyz=tuple(float(value) for value in quaternion_row),
    )


def _historical_api() -> _HistoricalApi:
    # Imports stay lazy: importing this module must never initialize Genesis.
    # Repository scripts normally add this package root themselves.  Adding it
    # here too keeps the reusable wrapper functional when imported directly by
    # a test or library caller from a source checkout.
    repository_root = Path(__file__).resolve().parents[2]
    for package_root in (
        repository_root / "lewm_genesis",
        repository_root / "lewm_worlds",
    ):
        if package_root.is_dir() and str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
    from lewm_genesis.render_replay import _camera_pose_from_payload
    from scripts.render_replay_v03 import build_scene, _to_hwc_uint8

    return _HistoricalApi(
        build_scene=build_scene,
        camera_pose_from_payload=_camera_pose_from_payload,
        to_hwc_uint8=_to_hwc_uint8,
    )


def _validated_camera_pose(value: Any) -> dict[str, list[float]]:
    if not isinstance(value, Mapping):
        raise TexturedV03RendererError("historical camera placement returned no pose")
    result: dict[str, list[float]] = {}
    for field in ("position", "lookat", "up"):
        result[field] = list(_exact_tuple(value.get(field), 3, f"camera pose {field}"))
    return result


class TexturedV03Renderer:
    """A separate historical static scene reusable across branch snapshots."""

    def __init__(
        self,
        context: Any,
        *,
        gs: Any,
        raw_manifest: Mapping[str, Any],
        _api: _HistoricalApi | None = None,
        _clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.contract_digest = renderer_contract_digest()
        self.camera_mount_body = validate_camera_pack(context.pack)
        self.scene_id = str(getattr(context.pack, "scene_id", ""))
        if not self.scene_id:
            raise TexturedV03RendererError("context pack must bind a non-empty scene_id")
        if not isinstance(raw_manifest, Mapping):
            raise TexturedV03RendererError("raw_manifest must be a mapping")
        # Canonical round-trip both validates and prevents later caller mutation.
        try:
            raw_copy = json.loads(_canonical_json(raw_manifest))
        except (TypeError, ValueError) as exc:
            raise TexturedV03RendererError(
                "raw_manifest must be finite and canonically JSON serialisable"
            ) from exc
        manifest_scene_id = str(raw_copy.get("scene_id", ""))
        if manifest_scene_id != self.scene_id:
            raise TexturedV03RendererError(
                f"raw manifest scene_id {manifest_scene_id!r} does not match "
                f"pack scene_id {self.scene_id!r}"
            )
        self.raw_manifest_digest = raw_manifest_digest(raw_copy)
        self._api = _api if _api is not None else _historical_api()
        self._clock = _clock

        started = self._clock()
        self.scene, self.camera = self._api.build_scene(
            gs,
            raw_copy,
            fov=FOV_DEG,
            near=NEAR_M,
            far=FAR_M,
            res=RESOLUTION_WH,
            textures=True,
        )
        self.scene_build_runtime_s = float(self._clock() - started)
        if self.scene is None or self.camera is None:
            raise TexturedV03RendererError("historical build_scene returned no scene/camera")
        if (not math.isfinite(self.scene_build_runtime_s)
                or self.scene_build_runtime_s < 0.0):
            raise TexturedV03RendererError("scene build runtime is invalid")

    def _render_pose(
        self, base_pose: BasePose, *, started: float,
    ) -> TexturedV03RenderResult:
        if not isinstance(base_pose, BasePose):
            raise TexturedV03RendererError("base_pose must be a captured BasePose")
        # A caller can instantiate the frozen dataclass directly, so validate
        # it again rather than trusting only capture_base_pose().
        position = _single_row(base_pose.position_world_xyz, 3, "base position")
        quaternion = _single_row(
            base_pose.quaternion_world_wxyz, 4, "base quaternion WXYZ"
        )
        validated_base_pose = BasePose(
            tuple(float(value) for value in position),
            tuple(float(value) for value in quaternion),
        )
        camera_pose = _validated_camera_pose(self._api.camera_pose_from_payload(
            validated_base_pose.replay_payload(), self.camera_mount_body
        ))
        self.camera.set_pose(
            pos=tuple(camera_pose["position"]),
            lookat=tuple(camera_pose["lookat"]),
            up=tuple(camera_pose["up"]),
        )

        rendered = self.camera.render(rgb=True, depth=False)
        rgb = rendered[0] if isinstance(rendered, (tuple, list)) else rendered
        image = np.asarray(self._api.to_hwc_uint8(rgb))
        if image.shape != (RESOLUTION_WH[1], RESOLUTION_WH[0], 3):
            raise TexturedV03RendererError(
                f"historical renderer returned shape {image.shape}, expected (224, 224, 3)"
            )
        if image.dtype != np.uint8:
            raise TexturedV03RendererError(
                f"historical renderer returned dtype {image.dtype}, expected uint8"
            )
        durable_image = np.ascontiguousarray(image).copy()
        runtime_s = float(self._clock() - started)
        if not math.isfinite(runtime_s) or runtime_s < 0.0:
            raise TexturedV03RendererError("frame render runtime is invalid")
        return TexturedV03RenderResult(
            image=durable_image,
            camera_pose_world=camera_pose,
            runtime_s=runtime_s,
        )

    def render_pose(self, base_pose: BasePose) -> TexturedV03RenderResult:
        """Render an immutable pose captured before the branch context moved."""

        return self._render_pose(base_pose, started=self._clock())

    def render(self, context: Any) -> TexturedV03RenderResult:
        """Capture the context's current base pose and render it immediately."""

        started = self._clock()
        mount = validate_camera_pack(context.pack)
        if mount != self.camera_mount_body:
            raise TexturedV03RendererError(
                "render context nominal camera mount differs from the built renderer"
            )
        if str(getattr(context.pack, "scene_id", "")) != self.scene_id:
            raise TexturedV03RendererError("render context belongs to a different scene")
        base_pose = capture_base_pose(context)
        return self._render_pose(base_pose, started=started)
