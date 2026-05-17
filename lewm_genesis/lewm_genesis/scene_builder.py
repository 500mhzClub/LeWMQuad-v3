"""Build Genesis scenes from typed ``ScenePack`` objects.

This is the only module in ``lewm_genesis`` that imports ``genesis`` at
runtime. Importing this module without Genesis installed is fine; calling
``build_scene`` raises a clear error.

The higher-level entry point ``build_scene_from_pack`` consumes a
``ScenePack`` produced by :mod:`lewm_genesis.scene_loader` and returns the
Genesis scene, robot, camera, and per-leg foot link handles in LeWM order.

Genesis API surface used:

- ``gs.init(backend=...)`` (process-global, called once via ``initialize_genesis``)
- ``gs.Scene`` with ``SimOptions(dt=...)``
- ``gs.morphs.Plane``, ``gs.morphs.Box``, ``gs.morphs.URDF``
- ``scene.add_camera`` and ``scene.add_entity``
- ``scene.build(n_envs=...)``

The exact Genesis API for attaching a camera to a moving body or for
resolving foot-link indices may require the rollout loop to do manual
per-tick updates. Those concerns live in ``rollout.py``, not here.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

from lewm_genesis.scene_loader import ScenePack, effective_camera_mount_xyz_rpy


_GENESIS_INITIALIZED = False


def _import_genesis():
    try:
        import genesis as gs  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised only when Genesis missing
        raise RuntimeError(
            "Genesis is not installed. Install the project-approved Genesis "
            "runtime before calling lewm_genesis.scene_builder functions."
        ) from exc
    return gs


def initialize_genesis(backend: str = "auto", seed: int | None = None, logging_level: int | None = None) -> None:
    """Idempotently initialize Genesis with the requested backend.

    Subsequent calls are no-ops. ``backend="auto"`` can be overridden with
    ``GS_BACKEND`` and otherwise follows the v2 preference order. Explicit
    backend requests fail loudly if the installed Genesis package does not
    expose that backend.
    """

    global _GENESIS_INITIALIZED
    if _GENESIS_INITIALIZED:
        return

    gs = _import_genesis()
    backend_obj = _resolve_backend(gs, backend)
    init_kwargs: dict[str, Any] = {"backend": backend_obj}
    if seed is not None:
        # Quadrants' compile config accepts signed 32-bit seeds.
        init_kwargs["seed"] = int(seed) & 0x7FFF_FFFF
    if logging_level is not None:
        init_kwargs["logging_level"] = logging_level
    gs.init(**init_kwargs)
    _GENESIS_INITIALIZED = True


def _resolve_backend(gs, backend_name: str):
    name = backend_name.lower().strip()
    explicit = {
        "cpu": "cpu",
        "gpu": "gpu",
        "cuda": "cuda",
        "vulkan": "vulkan",
        "metal": "metal",
        "amdgpu": "amdgpu",
        "amd": "amdgpu",
        "hip": "amdgpu",
        "rocm": "amdgpu",
    }
    env_backend = os.getenv("GS_BACKEND", "").lower().strip()

    if name == "auto" and env_backend and env_backend != "auto":
        name = env_backend

    if name == "auto":
        # Mirror v2's preference order for AMD hardware running ROCm.
        try:
            import torch

            hip = getattr(torch.version, "hip", None)
        except ImportError:
            hip = None
        if hip:
            for attr in ("amdgpu", "vulkan", "gpu", "cuda", "cpu"):
                backend = getattr(gs, attr, None)
                if backend is not None:
                    return backend
        for attr in ("gpu", "cuda", "vulkan", "metal", "cpu"):
            backend = getattr(gs, attr, None)
            if backend is not None:
                return backend
        return gs.cpu
    if name in explicit:
        backend = getattr(gs, explicit[name], None)
        if backend is not None:
            return backend
        available = ", ".join(
            attr
            for attr in dict.fromkeys(explicit.values())
            if getattr(gs, attr, None) is not None
        )
        version = getattr(gs, "__version__", "unknown")
        raise RuntimeError(
            f"Genesis backend '{name}' requested but unavailable in genesis-world {version}. "
            f"Available backend symbols: {available or 'none'}."
        )
    expected = ", ".join(sorted(explicit))
    raise ValueError(f"Unknown Genesis backend '{backend_name}'. Expected one of: auto, {expected}.")


@dataclass
class SceneBuild:
    """The Genesis entities produced from a ``ScenePack``."""

    scene: Any
    robot: Any
    camera: Any
    pack: ScenePack
    n_envs: int


def build_scene_from_pack(
    pack: ScenePack,
    *,
    n_envs: int,
    backend: str = "auto",
    show_viewer: bool = False,
) -> SceneBuild:
    """Build a Genesis scene with ``n_envs`` parallel envs from a ``ScenePack``."""

    gs = _import_genesis()
    initialize_genesis(backend=backend, seed=pack.physics_seed)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=float(pack.timing.physics_dt_s),
            gravity=(0.0, 0.0, -9.81),
        ),
        show_viewer=bool(show_viewer),
        renderer=gs.renderers.Rasterizer(),
    )

    floor_material = _floor_material(gs, pack)
    obstacle_material = _obstacle_material(gs, pack)
    material_lookup = _material_lookup(pack)
    floor_surface = _surface_for(gs, "floor", material_lookup)

    plane_kwargs: dict[str, Any] = {}
    if floor_material is not None:
        plane_kwargs["material"] = floor_material
    if floor_surface is not None:
        plane_kwargs["surface"] = floor_surface
    scene.add_entity(gs.morphs.Plane(), **plane_kwargs)

    for obj in pack.static_objects:
        # Box euler is in degrees per Genesis morph contract; convert from
        # radians stored on the manifest.
        euler_deg = (
            math.degrees(float(obj.roll_rad)),
            math.degrees(float(obj.pitch_rad)),
            math.degrees(float(obj.yaw_rad)),
        )
        morph = gs.morphs.Box(
            pos=obj.center_xyz_m,
            size=obj.size_xyz_m,
            euler=euler_deg,
            fixed=True,
        )
        entity_kwargs: dict[str, Any] = {"name": obj.object_id}
        if obstacle_material is not None:
            entity_kwargs["material"] = obstacle_material
        surface = _surface_for(gs, obj.material_id, material_lookup)
        if surface is not None:
            entity_kwargs["surface"] = surface
        scene.add_entity(morph, **entity_kwargs)

    # Genesis quaternion convention: wxyz (matches the scene manifest).
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(pack.robot.urdf_path),
            pos=pack.robot.spawn_xyz_m,
            quat=pack.robot.spawn_quat_wxyz,
            fixed=False,
        ),
        name="go2",
    )

    cam_pos_world, cam_lookat_world = _initial_camera_pose_world(pack)
    camera = scene.add_camera(
        res=pack.camera.native_resolution,
        pos=cam_pos_world,
        lookat=cam_lookat_world,
        fov=float(pack.camera.fov_deg),
        near=float(pack.camera.near_m),
        far=float(pack.camera.far_m),
        GUI=False,
    )

    scene.build(n_envs=int(n_envs))
    return SceneBuild(scene=scene, robot=robot, camera=camera, pack=pack, n_envs=int(n_envs))


# ---------------------------------------------------------------------------
# Material / surface helpers (data-spec §14 plumbing)
# ---------------------------------------------------------------------------


def _floor_material(gs, pack: ScenePack):
    physics = pack.physics_randomization
    if physics is None:
        return None
    return _build_rigid_material(gs, friction=physics.floor_friction_mu)


def _obstacle_material(gs, pack: ScenePack):
    physics = pack.physics_randomization
    if physics is None:
        return None
    return _build_rigid_material(gs, friction=physics.obstacle_friction_mu)


def _build_rigid_material(gs, *, friction: float):
    # Genesis Rigid friction must live in [1e-2, 5.0]; clamp defensively so
    # an out-of-range manifest value falls back to a sane edge instead of
    # crashing scene build.
    clamped = max(1e-2, min(5.0, float(friction)))
    try:
        return gs.materials.Rigid(friction=clamped)
    except Exception:  # pragma: no cover - exercised only when Genesis API drifts
        return None


def _material_lookup(pack: ScenePack) -> dict[str, tuple[float, float, float, float]]:
    """Resolve material_id -> RGBA from the manifest + landmark palette."""

    from lewm_worlds.randomization import BASE_PALETTE, LANDMARK_PALETTE

    lookup: dict[str, tuple[float, float, float, float]] = {}
    if pack.visual_randomization is not None:
        for override in pack.visual_randomization.material_overrides:
            lookup[override.material_id] = override.rgba
    for key, rgba in LANDMARK_PALETTE.items():
        lookup.setdefault(key, rgba)
    for key, rgba in BASE_PALETTE.items():
        lookup.setdefault(key, rgba)
    return lookup


def _surface_for(
    gs,
    material_id: str,
    lookup: dict[str, tuple[float, float, float, float]],
):
    if not material_id:
        return None
    rgba = lookup.get(material_id)
    if rgba is None:
        return None
    try:
        return gs.surfaces.Default(color=tuple(rgba))
    except Exception:  # pragma: no cover - surface API drift
        return None


def _initial_camera_pose_world(pack: ScenePack) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Compute an initial world-frame camera pose from the spawn + body mount.

    The rollout loop tracks the body each tick (see ``rollout.py``); this
    function only sets a sensible starting pose so the first rendered frame
    is meaningful for previews and unit checks. Per-scene camera-extrinsic
    jitter (data-spec §14) is added to the platform mount before the
    transform so each scene gets a slightly different mount.
    """

    spawn_x, spawn_y, spawn_z = pack.robot.spawn_xyz_m
    spawn_qw, spawn_qx, spawn_qy, spawn_qz = pack.robot.spawn_quat_wxyz
    yaw = _yaw_from_wxyz(spawn_qw, spawn_qx, spawn_qy, spawn_qz)

    (mx, my, mz), (_, jpitch, jyaw) = effective_camera_mount_xyz_rpy(pack)

    effective_yaw = yaw + jyaw
    cos_y, sin_y = math.cos(effective_yaw), math.sin(effective_yaw)
    cam_x = spawn_x + cos_y * mx - sin_y * my
    cam_y = spawn_y + sin_y * mx + cos_y * my
    cam_z = spawn_z + mz

    # Aim 1m forward from the camera, in the body x direction; nominal pitch
    # is -0.1m below horizon at 1m forward (~5.7°), perturbed by jpitch.
    look_x = cam_x + cos_y * 1.0
    look_y = cam_y + sin_y * 1.0
    look_z = cam_z - 0.1 - math.tan(jpitch)
    return (cam_x, cam_y, cam_z), (look_x, look_y, look_z)


def _yaw_from_wxyz(qw: float, qx: float, qy: float, qz: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


# ---------------------------------------------------------------------------
# Backwards-compatible legacy entry point
# ---------------------------------------------------------------------------


def build_genesis_scene(scene_spec: dict[str, Any], platform: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Legacy entry point retained for callers that still pass raw dicts.

    Prefer :func:`build_scene_from_pack`. This shim exists so the
    ``lewm_genesis`` package keeps a stable surface during the bulk-loop port.
    """

    gs = _import_genesis()
    initialize_genesis(backend=str(platform.get("backend", "auto")), seed=int(scene_spec["physics_seed"]))

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=float(platform.get("physics_dt_s", 0.002)),
            gravity=(0.0, 0.0, -9.81),
        ),
        show_viewer=bool(platform.get("show_viewer", False)),
        renderer=gs.renderers.Rasterizer(),
    )
    scene.add_entity(gs.morphs.Plane())

    for obj in scene_spec.get("objects", []):
        scene.add_entity(
            gs.morphs.Box(
                pos=tuple(obj["center_xyz_m"]),
                size=tuple(obj["size_xyz_m"]),
                euler=(0.0, 0.0, float(obj["yaw_rad"])),
            ),
            name=obj["object_id"],
        )

    spawn = scene_spec["spawn"]
    robot = scene.add_entity(
        gs.morphs.URDF(
            file=platform["go2_urdf"],
            pos=tuple(spawn["xyz_m"]),
            quat=tuple(spawn["quat_wxyz"]),
            fixed=False,
        ),
        name="go2",
    )

    camera = scene.add_camera(
        res=tuple(platform.get("camera_resolution", (640, 480))),
        pos=tuple(platform.get("camera_initial_world_pos", (1.0, 0.0, 0.7))),
        lookat=tuple(platform.get("camera_initial_world_lookat", (2.0, 0.0, 0.4))),
        fov=float(platform.get("camera_fov_deg", 78.323)),
        near=float(platform.get("camera_near_m", 0.05)),
        far=float(platform.get("camera_far_m", 200.0)),
        GUI=False,
    )
    scene.build(n_envs=int(platform.get("n_envs", 1)))
    return scene, robot, camera
