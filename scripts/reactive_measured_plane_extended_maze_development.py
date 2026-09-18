"""Four-thousand-tick reactive collection and complete raw controller audit.

Only private function globals are rebound. The existing reactive physics loop,
complete decision replay and command audit keep their original function code.
There is no high-level model argument or loading operation in this pipeline.
"""
from lewm.measured_plane_comparator_controllers_development import MeasuredPlaneReactiveController
from scripts import extended_budget_anchored_maze_development as extended
from scripts import measured_plane_extended_maze_development as learned
from scripts import reactive_floor_transport_maze_episode_development as episode
from scripts import reactive_floor_transport_maze_audit_development as raw_audit
from scripts import reactive_nominal_maze_command_audit_development as commands

NAVIGATION_TICKS = extended.NAVIGATION_TICKS
MAX_COMMAND_TICKS = extended.MAX_COMMAND_TICKS
MAX_OBSERVATIONS = extended.MAX_OBSERVATIONS
COLLECTION_ALLOWANCE_BYTES = extended.COLLECTION_ALLOWANCE_BYTES

collect = extended._bind(episode.collect,
    ReactiveFloorTransportController=MeasuredPlaneReactiveController,
    NAVIGATION_TICKS=NAVIGATION_TICKS, MAX_OBSERVATIONS=MAX_OBSERVATIONS,
    COLLECTION_ALLOWANCE_BYTES=COLLECTION_ALLOWANCE_BYTES,
    writer=extended.writer, RendererWitnessDualCameraMazeSession=extended.ExtendedBudgetRendererSession)
audit_commands = extended._bind(commands.audit_commands, NAVIGATION_TICKS=NAVIGATION_TICKS)
audit = extended._bind(raw_audit.audit,
    ReactiveFloorTransportController=MeasuredPlaneReactiveController,
    NAVIGATION_TICKS=NAVIGATION_TICKS, MAX_COMMAND_TICKS=MAX_COMMAND_TICKS,
    IntentReturnRGBDReplay=extended.ExtendedBudgetRGBDReplay,
    audit_sensors=extended.audit_sensors, audit_commands=audit_commands,
    read_rows=extended.read_rows, packet=extended.rgb_packet, renderer_audit=extended.renderer_audit)
artifacts = episode.artifacts
read_rows = extended.read_rows
rgb_packet = extended.rgb_packet
ExtendedBudgetRGBDReplay = extended.ExtendedBudgetRGBDReplay


def definition():
    return learned.definition() | dict(implementation_class='MeasuredPlaneReactiveController',
        high_level_world_model_loaded=False, candidate_future_outcomes_evaluated=False,
        predictive_surface_or_path_gates_applied=False, learned_residual_used=False,
        fully_nonpredictive_controller=True, reactive_is_whole_method_comparison=True,
        isolated_prediction_ranking_ablation=False, future_constraint_gates_matched=False,
        original_reactive_physics_loop_retained=True, original_reactive_command_audit_retained=True)
