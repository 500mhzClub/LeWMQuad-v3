"""Full-history timing accounting for the prepared chained controller pair.

This module performs no input admission or replay. A future caller must bind
the completed native history and compare every actual decision and packet.
"""
from lewm.measured_plane_chained_anchor_development import (
    MeasuredPlaneChainedAnchorController, MeasuredPlaneChainedAnchorVisualMotion)
from lewm.measured_plane_chained_single_pass_controller_development import (
    MeasuredPlaneChainedSinglePassController)
from scripts import measured_plane_full_history_timing_development as original
from scripts.measured_plane_chained_single_pass_comparison_development import normalize

state_frames = original.state_frames
reference_endpoint = original.reference_endpoint


def observed_state(controller):
    if (type(controller) not in (MeasuredPlaneChainedAnchorController,
                                MeasuredPlaneChainedSinglePassController)
            or type(controller.motion) is not MeasuredPlaneChainedAnchorVisualMotion):
        raise ValueError('exact chained baseline or single-pass controller and motion required')
    # Preserve every motion and mission field and type, including both cameras'
    # chained image caches and reacquisition witnesses. Only the established
    # map/history and registration implementation type normalizations apply.
    return original.observed_state(controller)


def summarize(rows, states, *, frames, model_sha, input_result_sha):
    report = original.summarize(rows, states, frames=frames, model_sha=model_sha,
        input_result_sha=input_result_sha)
    return report | dict(baseline='MeasuredPlaneChainedAnchorController',
        candidate='MeasuredPlaneChainedSinglePassController',
        chained_tracking_in_both_controllers=True,
        chained_tracking_state_normalized=False,
        completed_native_input_admission_required=True,
        native_adoption_performed=False)
