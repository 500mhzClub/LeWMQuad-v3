"""Source-only native composition; no launcher or queued scene execution."""
from lewm.measured_plane_chained_anchor_development import MeasuredPlaneChainedAnchorController
from scripts import measured_plane_extended_maze_development as original
from scripts.extended_budget_anchored_maze_development import _bind

collect = _bind(original.collect, ResidualAnchoredContinuationController=MeasuredPlaneChainedAnchorController)
audit = _bind(original.audit, ResidualAnchoredContinuationController=MeasuredPlaneChainedAnchorController)
artifacts = original.artifacts
read_rows = original.read_rows
rgb_packet = original.rgb_packet
ExtendedBudgetRGBDReplay = original.ExtendedBudgetRGBDReplay


def definition():
    return original.definition() | dict(implementation_class='MeasuredPlaneChainedAnchorController',
        chained_anchor_reacquisition_enabled=True,direct_corner_flow_missingness_fallback_enabled=True,
        measured_plane_refinement_wraps_all_image_fits=True,
        descriptor_and_chained_association_rules_distinguished=True,
        original_qualified_measurement_conflicts_retained=True,
        original_bridge_allowance_unchanged=True,pose_increments_composed=False,
        completed_controller_boundary_required_before_native_launch=True,
        native_execution_protocol_frozen=False)
