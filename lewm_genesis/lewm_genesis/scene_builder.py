"""Construct Genesis scenes from canonical LeWM Genesis specs."""

from __future__ import annotations

from typing import Any


def build_genesis_scene(scene_spec: dict[str, Any], platform: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Build a Genesis scene, robot, and camera from a canonical scene spec.

    Genesis is intentionally an optional runtime dependency. Importing this
    module is safe without Genesis installed; calling this function requires it.
    """

    try:
        import genesis as gs  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on local renderer install.
        raise RuntimeError(
            "Genesis is not installed. Install the project-approved Genesis "
            "runtime before calling build_genesis_scene()."
        ) from exc

    gs.init(backend=platform.get("backend", "cpu"), seed=int(scene_spec["physics_seed"]))
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
