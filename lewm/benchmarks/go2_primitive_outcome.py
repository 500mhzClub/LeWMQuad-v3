from __future__ import annotations

import math
from typing import Any


def _wrap_angle_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _velocity_block(registry: Any, primitive_name: str) -> list[tuple[float, float, float]]:
    primitives = getattr(registry, "primitives")
    if primitive_name not in primitives:
        raise KeyError(f"unknown primitive '{primitive_name}'")
    primitive = primitives[primitive_name]
    if primitive.get("type") != "velocity_block":
        raise ValueError(
            f"primitive '{primitive_name}' is not a velocity_block "
            f"(type={primitive.get('type')!r})"
        )
    command = primitive.get("command", {})
    item = (
        float(command.get("vx_body_mps", 0.0)),
        float(command.get("vy_body_mps", 0.0)),
        float(command.get("yaw_rate_radps", 0.0)),
    )
    return [item for _ in range(int(getattr(registry, "block_size")))]


def body_probe_clearance(
    grid: Any,
    xy: tuple[float, float],
    yaw: float,
    *,
    body_forward_m: float,
    body_half_width_m: float,
    clearance_source: str = "configuration",
) -> float:
    x = float(xy[0])
    y = float(xy[1])
    fx, fy = math.cos(float(yaw)), math.sin(float(yaw))
    lx, ly = -fy, fx
    probes = (
        (0.0, 0.0),
        (body_forward_m, 0.0),
        (body_forward_m, body_half_width_m),
        (body_forward_m, -body_half_width_m),
        (0.0, body_half_width_m),
        (0.0, -body_half_width_m),
    )
    if str(clearance_source) == "obstacle":
        clearance_fn = grid.obstacle_clearance_m
    else:
        clearance_fn = grid.configuration_clearance_m
    return float(min(
        clearance_fn((x + forward * fx + lateral * lx,
                      y + forward * fy + lateral * ly))
        for forward, lateral in probes
    ))


def primitive_body_clearance_and_progress(
    *,
    registry: Any,
    primitive: str,
    grid: Any,
    x_m: float,
    y_m: float,
    yaw_rad: float,
    command_dt_s: float,
    body_forward_m: float,
    body_half_width_m: float,
    clearance_source: str,
    progress_collision_stop_m: float | None = None,
) -> tuple[float, float, float, float]:
    x = float(x_m)
    y = float(y_m)
    yaw = float(yaw_rad)
    min_clearance = body_probe_clearance(
        grid,
        (x, y),
        yaw,
        body_forward_m=body_forward_m,
        body_half_width_m=body_half_width_m,
        clearance_source=clearance_source,
    )
    min_after_start = float("inf")
    final_clearance = float(min_clearance)
    progress_m = 0.0
    collided = bool(
        progress_collision_stop_m is not None
        and float(min_clearance) <= float(progress_collision_stop_m)
    )
    for vx_body, vy_body, yaw_rate in _velocity_block(registry, primitive):
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        next_x = x + (
            float(vx_body) * cos_yaw - float(vy_body) * sin_yaw
        ) * float(command_dt_s)
        next_y = y + (
            float(vx_body) * sin_yaw + float(vy_body) * cos_yaw
        ) * float(command_dt_s)
        next_yaw = _wrap_angle_pi(yaw + float(yaw_rate) * float(command_dt_s))
        step_progress = math.hypot(next_x - x, next_y - y)
        x, y, yaw = next_x, next_y, next_yaw
        step_clearance = body_probe_clearance(
            grid,
            (x, y),
            yaw,
            body_forward_m=body_forward_m,
            body_half_width_m=body_half_width_m,
            clearance_source=clearance_source,
        )
        if not collided:
            if (
                progress_collision_stop_m is not None
                and float(step_clearance) <= float(progress_collision_stop_m)
            ):
                collided = True
            else:
                progress_m += step_progress
        final_clearance = float(step_clearance)
        min_after_start = min(min_after_start, float(step_clearance))
        min_clearance = min(min_clearance, float(step_clearance))
    if not math.isfinite(min_after_start):
        min_after_start = final_clearance
    return (
        float(min_clearance),
        float(min_after_start),
        float(final_clearance),
        float(progress_m),
    )
