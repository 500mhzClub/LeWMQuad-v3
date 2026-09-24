"""Narrow preintervention comparison; no following recorded action outcomes."""
from copy import deepcopy


def normalize_labels(decision):
    result = deepcopy(decision)
    if (result['controller'] != 'measured_floor_transport_round_trip_controller_v1'
            or result.pop('floor_transport_during_missingness_enabled') is not True):
        raise ValueError('explicit measured floor transport successor labels required')
    result['controller'] = 'dual_camera_settled_round_trip_controller_v1'
    mission = result['mission_receipt']
    if mission is not None and mission.get('observed_settling') is not None:
        receipt = mission['observed_settling']
        if receipt['motion_source'] != 'consecutive_admitted_visual_positions_in_floor_reference':
            raise ValueError('explicit mixed floor-reference pose source required')
        receipt['motion_source'] = 'consecutive_admitted_floor_registered_visual_positions'
    return result


def compare_prior(saved, candidate, actual_command, *, frame):
    if not 0 <= frame < 1904 or saved['tick'] != frame:
        raise ValueError('only the fixed complete preintervention prefix may be compared')
    normalized = normalize_labels(candidate)
    if normalized != saved['decision']:
        raise ValueError('complete preintervention decision differs at frame '+str(frame))
    if candidate['terminal'] is not None or candidate['requested_command'] != actual_command:
        raise ValueError('unchanged active preintervention command required')


def admit_native(result, report, *, case):
    if (result['status'] != 'DUAL_CAMERA_SETTLED_MAZE_PILOT_V1_COMPLETE' or len(result['conditions']) != 1
            or result['conditions'][0]['case'] != case
            or result['conditions'][0]['status'] != 'DUAL_CAMERA_SETTLED_MAZE_COLLECTED_AND_RAW_AUDITED'
            or result['conditions'][0]['prefix_comparison']['physical_and_public_prefix_exact'] is not True
            or result['conditions'][0]['prefix_comparison']['complete_candidate_decisions_match_prospective_prefix'] is not True):
        raise ValueError('completed eleventh collection, full raw audit and prospective prefix required')
    for key in ('raw_sensor_reconstruction_pass', 'additional_auxiliary_rgb_reconstructed',
            'raw_model_command_replay_pass', 'raw_command_audit_pass', 'model_state_unchanged'):
        if report[key] is not True: raise ValueError('completed native raw invariant required: '+key)
