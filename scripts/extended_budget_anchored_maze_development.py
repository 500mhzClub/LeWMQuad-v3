"""Source-only 4,000-tick development composition of the original maze pipeline.

There is no launcher or queue mutation here. Original function code, defaults,
closures, sensor checks and physics semantics are retained. Only the declared
budget, storage allowance and corresponding bounded dependencies differ.
"""
from contextlib import contextmanager
from types import FunctionType

from lewm import intent_return_rgbd_replay_development as replay
from lewm.novel_maze_round_trip_contract_development import WARMUP_TICKS, DRAIN_TICKS
from lewm.observed_round_trip_mission_development import MAX_NAVIGATION_TICKS
from lewm.joint_visual_surface_memory_development import MAX_FRAMES as MEMORY_FRAMES
from scripts import maze_decision_stream_development as stream
from scripts import novel_maze_auxiliary_packet_development as auxiliary_depth
from scripts import novel_maze_auxiliary_rgb_packet_development as auxiliary_rgb
from scripts import maze_renderer_witness_development as witness
from scripts import renderer_witness_maze_probe_comparison_development as renderer
from scripts import near_field_sensor_audit_development as sensors
from scripts import novel_maze_round_trip_command_audit_development as commands
from scripts import residual_anchored_continuation_maze_episode_development as episode
from scripts import residual_anchored_continuation_maze_audit_development as original_audit
from scripts.novel_maze_round_trip_session_development import NovelMazeRoundTripSession
from scripts.dual_camera_novel_maze_session_development import DualCameraNovelMazeSession
from scripts.renderer_witness_dual_camera_maze_session_development import RendererWitnessDualCameraMazeSession

NAVIGATION_TICKS = 4000
MAX_COMMAND_TICKS = WARMUP_TICKS+NAVIGATION_TICKS+DRAIN_TICKS
MAX_OBSERVATIONS = MAX_COMMAND_TICKS+1
COLLECTION_ALLOWANCE_BYTES = 14*1024**3
assert NAVIGATION_TICKS <= MAX_NAVIGATION_TICKS
assert MAX_OBSERVATIONS <= MEMORY_FRAMES


def _bind(function, **changes):
    """Copy globals without modifying the imported function or its class cell."""
    if type(function) is not FunctionType or not changes.keys() <= function.__globals__.keys():
        raise ValueError('existing explicit function-global substitutions required')
    bound = FunctionType(function.__code__, function.__globals__ | changes,
        function.__name__, function.__defaults__, function.__closure__)
    bound.__kwdefaults__ = None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    return bound


validate_frame_population = _bind(replay.validate_frame_population, MAX_FRAMES=MAX_OBSERVATIONS)


class ExtendedBudgetRGBDReplay(replay.IntentReturnRGBDReplay):
    __init__ = _bind(replay.IntentReturnRGBDReplay.__init__,
        validate_frame_population=validate_frame_population)


writer = contextmanager(_bind(stream.writer.__wrapped__, MAX_OBSERVATIONS=MAX_OBSERVATIONS))
read_rows = _bind(stream.read_rows, MAX_OBSERVATIONS=MAX_OBSERVATIONS)
depth_packet = _bind(auxiliary_depth.packet, MAX_OBSERVATIONS=MAX_OBSERVATIONS)
rgb_packet = _bind(auxiliary_rgb.packet, MAX_OBSERVATIONS=MAX_OBSERVATIONS, depth_packet=depth_packet)
capture_witness = _bind(witness.capture_witness, MAX_OBSERVATIONS=MAX_OBSERVATIONS)
audit_witnesses = _bind(witness.audit_witnesses, MAX_OBSERVATIONS=MAX_OBSERVATIONS)
renderer_audit = _bind(renderer.renderer_audit, audit_witnesses=audit_witnesses)


class ExtendedBudgetPairedSession(NovelMazeRoundTripSession):
    sensor_packets = _bind(NovelMazeRoundTripSession.sensor_packets,
        MAX_OBSERVATIONS=MAX_OBSERVATIONS, packet=depth_packet)


class ExtendedBudgetDualSession(DualCameraNovelMazeSession, ExtendedBudgetPairedSession):
    # The preserved __class__ cell makes the original super() enter the
    # extended paired implementation in this cooperative MRO.
    sensor_packets = _bind(DualCameraNovelMazeSession.sensor_packets, packet=rgb_packet)


class ExtendedBudgetRendererSession(RendererWitnessDualCameraMazeSession, ExtendedBudgetDualSession):
    capture_fixed_rgb = _bind(RendererWitnessDualCameraMazeSession.capture_fixed_rgb,
        capture_witness=capture_witness)
    sensor_packets = _bind(RendererWitnessDualCameraMazeSession.sensor_packets,
        capture_witness=capture_witness)


audit_sensors = _bind(sensors.audit_sensors, IntentReturnRGBDReplay=ExtendedBudgetRGBDReplay)
audit_commands = _bind(commands.audit_commands, NAVIGATION_TICKS=NAVIGATION_TICKS)
collect = _bind(episode.collect, NAVIGATION_TICKS=NAVIGATION_TICKS,
    MAX_OBSERVATIONS=MAX_OBSERVATIONS, COLLECTION_ALLOWANCE_BYTES=COLLECTION_ALLOWANCE_BYTES,
    writer=writer, RendererWitnessDualCameraMazeSession=ExtendedBudgetRendererSession)
artifacts = episode.artifacts
audit = _bind(original_audit.audit, NAVIGATION_TICKS=NAVIGATION_TICKS,
    MAX_COMMAND_TICKS=MAX_COMMAND_TICKS, IntentReturnRGBDReplay=ExtendedBudgetRGBDReplay,
    audit_sensors=audit_sensors, audit_commands=audit_commands, read_rows=read_rows,
    packet=rgb_packet, renderer_audit=renderer_audit)
