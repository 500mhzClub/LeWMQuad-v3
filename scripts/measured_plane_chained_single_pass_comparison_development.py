"""Reuse only the established performance normalization; retain tracking data."""
from lewm.measured_plane_chained_single_pass_controller_development import CONTROLLER
from lewm.measured_plane_single_pass_controller_development import CONTROLLER as SINGLE_PASS
from scripts.measured_plane_single_pass_comparison_development import normalize as normalize_single_pass

BASELINE = 'measured_plane_chained_anchor_controller_v1'


def normalize(decision):
    if (decision.get('controller') != CONTROLLER
            or decision.get('chained_anchor_reacquisition_enabled') is not True
            or decision.get('direct_corner_flow_missingness_fallback_enabled') is not True):
        raise ValueError('explicit chained single-pass controller and unchanged tracking flags required')
    result = normalize_single_pass(decision | {'controller': SINGLE_PASS})
    if result['controller'] != 'measured_plane_residual_continuation_controller_v1':
        raise ValueError('existing normalization must end at the measured-plane baseline')
    return result | {'controller': BASELINE}
