"""Narrow existing performance normalization with the new observer preserved."""
from lewm.measured_plane_single_pass_controller_development import CONTROLLER
from lewm.single_pass_body_projected_controller_development import CONTROLLER as SINGLE_PASS
from scripts.single_pass_body_projected_replay_development import normalize_candidate
from scripts.single_pass_body_projected_state_development import normalized_state_tree, STATE_TYPE_PATHS


def normalize(decision):
    if (decision.get('controller') != CONTROLLER
            or decision.get('measured_plane_constrained_estimator') is not True):
        raise ValueError('explicit combined measured-plane single-pass controller required')
    result = normalize_candidate(decision | {'controller': SINGLE_PASS})
    if result['controller'] != 'residual_anchored_continuation_controller_v1':
        raise ValueError('existing complete normalization must end at the original controller')
    return result | {'controller': 'measured_plane_residual_continuation_controller_v1'}


def state(controller):
    return normalized_state_tree(dict(memory=controller.memory, floor=controller.mapper.floor,
        occupied=controller.mapper.occupied, residual=controller.residual, history=controller.history))
