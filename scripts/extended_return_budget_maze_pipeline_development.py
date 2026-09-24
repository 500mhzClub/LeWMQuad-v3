"""Prospective 8,000-step collection and full audit; no native launcher.

Private bindings retain original physics, acquisition and verification code.
Runtime resource admission and actual controller-prefix evidence are separate.
"""
from contextlib import contextmanager

from lewm.extended_return_budget_mission_development import (
    NAVIGATION_TICKS, MAX_COMMAND_TICKS, MAX_OBSERVATIONS)
from lewm.extended_return_budget_controller_development import ExtendedReturnBudgetChainedController
from scripts import extended_budget_anchored_maze_development as original

bind = original._bind
COLLECTION_ALLOWANCE_BYTES = 28*1024**3
MAX_PHYSICS_SAMPLES = 750+50*MAX_COMMAND_TICKS

validate_frame_population = bind(original.replay.validate_frame_population, MAX_FRAMES=MAX_OBSERVATIONS)


class ExtendedReturnBudgetRGBDReplay(original.replay.IntentReturnRGBDReplay):
    __init__ = bind(original.replay.IntentReturnRGBDReplay.__init__,
        validate_frame_population=validate_frame_population)


writer = contextmanager(bind(original.stream.writer.__wrapped__, MAX_OBSERVATIONS=MAX_OBSERVATIONS))
read_rows = bind(original.stream.read_rows, MAX_OBSERVATIONS=MAX_OBSERVATIONS)
depth_packet = bind(original.auxiliary_depth.packet, MAX_OBSERVATIONS=MAX_OBSERVATIONS)
rgb_packet = bind(original.auxiliary_rgb.packet, MAX_OBSERVATIONS=MAX_OBSERVATIONS,
    depth_packet=depth_packet)
capture_witness = bind(original.witness.capture_witness, MAX_OBSERVATIONS=MAX_OBSERVATIONS)
audit_witnesses = bind(original.witness.audit_witnesses, MAX_OBSERVATIONS=MAX_OBSERVATIONS)
renderer_audit = bind(original.renderer.renderer_audit, audit_witnesses=audit_witnesses)


class ExtendedReturnBudgetPairedSession(original.NovelMazeRoundTripSession):
    sensor_packets = bind(original.NovelMazeRoundTripSession.sensor_packets,
        MAX_OBSERVATIONS=MAX_OBSERVATIONS, packet=depth_packet)


class ExtendedReturnBudgetDualSession(original.DualCameraNovelMazeSession, ExtendedReturnBudgetPairedSession):
    sensor_packets = bind(original.DualCameraNovelMazeSession.sensor_packets, packet=rgb_packet)


class ExtendedReturnBudgetRendererSession(
        original.RendererWitnessDualCameraMazeSession, ExtendedReturnBudgetDualSession):
    capture_fixed_rgb = bind(original.RendererWitnessDualCameraMazeSession.capture_fixed_rgb,
        capture_witness=capture_witness)
    sensor_packets = bind(original.RendererWitnessDualCameraMazeSession.sensor_packets,
        capture_witness=capture_witness)


audit_sensors = bind(original.sensors.audit_sensors, IntentReturnRGBDReplay=ExtendedReturnBudgetRGBDReplay)
audit_commands = bind(original.commands.audit_commands, NAVIGATION_TICKS=NAVIGATION_TICKS)
# Evaluator-only native state remains confined to the audit pipeline.
evaluate = bind(original.original_audit.evaluate, MAX_NAVIGATION_TICKS=NAVIGATION_TICKS)
collect = bind(original.episode.collect, NAVIGATION_TICKS=NAVIGATION_TICKS,
    MAX_OBSERVATIONS=MAX_OBSERVATIONS, COLLECTION_ALLOWANCE_BYTES=COLLECTION_ALLOWANCE_BYTES,
    writer=writer, RendererWitnessDualCameraMazeSession=ExtendedReturnBudgetRendererSession,
    ResidualAnchoredContinuationController=ExtendedReturnBudgetChainedController)
artifacts = original.artifacts
audit = bind(original.original_audit.audit, NAVIGATION_TICKS=NAVIGATION_TICKS,
    MAX_COMMAND_TICKS=MAX_COMMAND_TICKS, IntentReturnRGBDReplay=ExtendedReturnBudgetRGBDReplay,
    audit_sensors=audit_sensors, audit_commands=audit_commands, read_rows=read_rows,
    packet=rgb_packet, renderer_audit=renderer_audit, evaluate=evaluate,
    ResidualAnchoredContinuationController=ExtendedReturnBudgetChainedController)


def definition():
    return dict(implementation_class='ExtendedReturnBudgetChainedController',
        navigation_ticks=NAVIGATION_TICKS, max_command_ticks=MAX_COMMAND_TICKS,
        max_observations=MAX_OBSERVATIONS, max_physics_samples=MAX_PHYSICS_SAMPLES,
        collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        extended_return_budget_enabled=True, chained_anchor_reacquisition_enabled=True,
        single_pass_controller_composed=True, single_read_auxiliary_acquisition_adopted=False,
        measured_plane_constrained_estimator=True, original_floor_gate_unchanged=True,
        original_temporal_gate_values_unchanged=True, original_bridge_allowance_unchanged=True,
        fresh_controller_and_memory=True, original_physics_loop_retained=True,
        complete_raw_controller_audit_retained=True, renderer_capture_witnesses_enabled=True,
        physics_paused_during_compute=True, native_pose_input=False,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False,
        goal_achieved=False)
