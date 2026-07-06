from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "lewm_genesis" / "lewm_genesis"))

from lewm.benchmarks.go2_primitive_outcome import (  # noqa: E402
    primitive_body_clearance_and_progress,
)
from lewm_contract import PrimitiveRegistry


class StepCollisionGrid:
    """Clear until the body crosses x=0.055m, then exactly at contact."""

    def configuration_clearance_m(self, xy: tuple[float, float]) -> float:
        return 0.10 if float(xy[0]) < 0.055 else 0.0

    def obstacle_clearance_m(self, xy: tuple[float, float]) -> float:
        return self.configuration_clearance_m(xy)


class InitiallyContactingGrid:
    def configuration_clearance_m(self, xy: tuple[float, float]) -> float:
        return 0.0

    def obstacle_clearance_m(self, xy: tuple[float, float]) -> float:
        return self.configuration_clearance_m(xy)


def _registry() -> PrimitiveRegistry:
    return PrimitiveRegistry(
        block_size=5,
        command_dt_s=0.1,
        command_order=("vx_body_mps", "vy_body_mps", "yaw_rate_radps"),
        primitives={
            "forward": {
                "type": "velocity_block",
                "train": True,
                "command": {
                    "vx_body_mps": 0.2,
                    "vy_body_mps": 0.0,
                    "yaw_rate_radps": 0.0,
                },
            }
        },
        defaults={},
    )


def test_counterfactual_progress_stops_before_first_contact_step() -> None:
    registry = _registry()

    *_clearance, collision_aware_progress = primitive_body_clearance_and_progress(
        registry=registry,
        primitive="forward",
        grid=StepCollisionGrid(),
        x_m=0.0,
        y_m=0.0,
        yaw_rad=0.0,
        command_dt_s=registry.command_dt_s,
        body_forward_m=0.0,
        body_half_width_m=0.0,
        clearance_source="configuration",
        progress_collision_stop_m=0.0,
    )
    *_clearance, collision_blind_progress = primitive_body_clearance_and_progress(
        registry=registry,
        primitive="forward",
        grid=StepCollisionGrid(),
        x_m=0.0,
        y_m=0.0,
        yaw_rad=0.0,
        command_dt_s=registry.command_dt_s,
        body_forward_m=0.0,
        body_half_width_m=0.0,
        clearance_source="configuration",
        progress_collision_stop_m=None,
    )

    assert collision_aware_progress == pytest.approx(0.04)
    assert collision_blind_progress == pytest.approx(0.10)


def test_counterfactual_progress_is_zero_when_starting_in_contact() -> None:
    registry = _registry()

    *_clearance, progress = primitive_body_clearance_and_progress(
        registry=registry,
        primitive="forward",
        grid=InitiallyContactingGrid(),
        x_m=0.0,
        y_m=0.0,
        yaw_rad=0.0,
        command_dt_s=registry.command_dt_s,
        body_forward_m=0.0,
        body_half_width_m=0.0,
        clearance_source="configuration",
        progress_collision_stop_m=0.0,
    )

    assert progress == pytest.approx(0.0)
