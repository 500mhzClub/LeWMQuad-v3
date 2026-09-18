"""Fresh four-thousand-tick simulation with the measured-plane controller.

This composition creates private function-global dictionaries. It does not
modify the original collector, audit, controller or any live native attempt.
The entire physics loop and full raw audit retain their original function code.
"""
from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
from scripts import extended_budget_anchored_maze_development as original

NAVIGATION_TICKS = original.NAVIGATION_TICKS
MAX_COMMAND_TICKS = original.MAX_COMMAND_TICKS
MAX_OBSERVATIONS = original.MAX_OBSERVATIONS
COLLECTION_ALLOWANCE_BYTES = original.COLLECTION_ALLOWANCE_BYTES

# The substituted global identifies the constructor used by the original
# function body. Only the private copies receive this substitution.
collect = original._bind(original.collect,
    ResidualAnchoredContinuationController=MeasuredPlaneResidualController)
audit = original._bind(original.audit,
    ResidualAnchoredContinuationController=MeasuredPlaneResidualController)
artifacts = original.artifacts
read_rows = original.read_rows
rgb_packet = original.rgb_packet
ExtendedBudgetRGBDReplay = original.ExtendedBudgetRGBDReplay


def definition():
    return dict(implementation_class='MeasuredPlaneResidualController',
        navigation_ticks=NAVIGATION_TICKS,max_command_ticks=MAX_COMMAND_TICKS,
        max_observations=MAX_OBSERVATIONS,collection_allowance_bytes=COLLECTION_ALLOWANCE_BYTES,
        measured_plane_constrained_estimator=True,original_floor_gate_unchanged=True,
        original_temporal_gate_values_unchanged=True,fresh_controller_and_memory=True,
        original_physics_loop_retained=True,complete_raw_controller_audit_retained=True,
        renderer_capture_witnesses_enabled=True,physics_paused_during_compute=True,
        native_pose_input=False,navigation_qualified=False,real_time_qualified=False,
        hardware_qualified=False,goal_achieved=False)
