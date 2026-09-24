"""Sensor-row-only reconstruction of available bridge/rejoin provenance.

No observer execution, native truth, filesystem access or artifact authority.
Checks saved measurements and history, not hidden candidate fits or failure
causes. Gyro-conditioned rotation agreement is not independent heading evidence.
"""
from copy import deepcopy
import math

from lewm.independent_tracking_numerical_verification_development import require, same


def point(value):
    require(isinstance(value, list) and len(value) == 3 and all(type(v) in (int, float)
        and math.isfinite(v) for v in value), 'finite measured position required')
    return tuple(value)


def verify_continuity(rows, saved_summary):
    """Fixed challenge clocks and at most443 rows; retain every terminal row.

    The saved position disagreement can be reconstructed when both estimates
    are available. Incremental rotation is not in the continuity evidence, so
    its reported disagreement is range-checked, not independently recomputed.
    """
    modes = dict(INITIAL_REFERENCE=0, ANCHOR_MEASUREMENT=0, MEASURED_INCREMENT_BRIDGE=0)
    spans = []; active = None; previous = None; retained = []; nodes = 0
    first_failure = None; failure_reason = None; frames = 0; bridged = 0
    paired_positions = 0; reported_rotation_checks = 0
    def close(outcome, following, evidence=None):
        nonlocal active
        if active is not None:
            span = active | dict(outcome=outcome, following_frame=following)
            if evidence is not None:
                span.update(rejoin_increment_available=evidence['incremental_available'],
                    rejoin_disagreement_m=evidence['disagreement_m'])
            spans.append(span); active = None
    for row in rows:
        frame = frames; now = 1_500_000_000 + frame * 100_000_000
        require(frame < 443 and type(row['frame']) is int and row['frame'] == frame
            and type(row['measured_ns']) is int and row['measured_ns'] == now,
            'complete sequential fixed challenge clocks required')
        arm = row['arms']['temporal_anchor']; pose = arm['pose']; e = arm['continuity']
        if pose is None:
            require(arm['failure'] is not None, 'unavailable pose needs a retained failure reason')
            if first_failure is None:
                first_failure = dict(frame=frame, status=e.get('status') if isinstance(e, dict) else None)
                failure_reason = deepcopy(arm['failure'])
                if isinstance(e, dict) and e.get('status') == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED':
                    require(active is not None and active['frames'] == 10 and e['anchor_available'] is False
                        and e['incremental_available'] is True, 'budget exhaustion needs ten preceding measured bridges')
            else:
                require(arm['failure'] == failure_reason, 'terminal failure reason cannot be rewritten')
            close('TERMINAL_FAILURE', frame); previous = None; frames += 1; continue
        require(first_failure is None and arm['failure'] is None, 'no pose after terminal failure or alongside failure')
        require(isinstance(e, dict) and e['status'] in modes and pose['frame'] == frame
            and pose['measured_ns'] == now and pose['global_history_reset'] is False
            and pose['position_error_bound'] is None and pose['orientation_error_bound'] is None
            and pose['uncertainty_model_validated'] is False and pose['native_pose_input'] is False
            and pose['navigation_qualified'] is False and pose['mode'] == 'gyro'
            and pose['gyro_role'] == 'rotation_estimator', 'same-frame unpromoted gyro-conditioned pose required')
        require(e['uncertainty_calibrated'] is False, 'agreement is not calibrated uncertainty')
        p = point(pose['position_initial_body_m']); mode = e['status']; modes[mode] += 1
        require(type(pose['promoted_keyframe']) is bool and type(pose['reference_frame']) is int,
            'explicit reference and keyframe promotion required')
        if frame == 0:
            require(mode == 'INITIAL_REFERENCE' and p == (0., 0., 0.)
                and pose['reference_frame'] == 0 and pose['promoted_keyframe'] is False,
                'single initial-body reference required')
            same(e['bridge_frames'], 0); same(e['bridge_path_m'], 0.)
            retained = [0]; nodes = 1
            same(arm['selection']['selected_reference'], 0)
            same(arm['selection']['retained_references'], 1)
        else:
            require(mode != 'INITIAL_REFERENCE' and previous is not None
                and e['previous_frame'] == frame - 1 and e['previous_measured_ns'] == now - 100_000_000,
                'immediately preceding accepted visual measurement required')
            require(type(e['anchor_available']) is bool and type(e['incremental_available']) is bool
                and e['measurements_independent'] is False and e['error_bound_m'] is None,
                'explicit correlated unbounded measurements required')
            same(arm['selection']['retained_references'], len(retained))
            same(arm['selection']['selected_reference'], pose['reference_frame'])
            if mode == 'MEASURED_INCREMENT_BRIDGE':
                require(e['incremental_available'] and not e['anchor_available']
                    and pose['promoted_keyframe'] is False and pose['reference_frame'] == frame - 1
                    and arm['selection']['status'] == mode
                    and arm['selection']['selected_reference_retained_anchor'] is False
                    and e['anchored_error_accumulates'] is True,
                    'bridge requires current measured increment without retained-anchor promotion')
                same(list(p), list(point(e['incremental_position_initial_body_m'])), tolerance=0.)
                if active is None: active = dict(start_frame=frame, end_frame=frame, frames=0, measured_path_m=0.)
                active['end_frame'] = frame; active['frames'] += 1; bridged += 1
                active['measured_path_m'] += math.dist(p, previous)
                require(active['frames'] <= 10, 'eleventh unanchored bridge is unavailable, never accepted')
                same(e['bridge_frames'], active['frames']); same(e['total_bridge_frames'], bridged)
                same(e['bridge_path_m'], active['measured_path_m'], tolerance=1e-12)
                require(e['disagreement_m'] is None and e['disagreement_rad'] is None,
                    'absent anchor cannot supply an agreement measurement')
            else:
                require(e['anchor_available'] and pose['reference_frame'] in retained,
                    'anchor measurement must use a currently retained reference')
                same(e['bridge_frames'], 0); same(e['bridge_path_m'], 0.)
                if e['incremental_available']:
                    distance = math.dist(p, point(e['incremental_position_initial_body_m']))
                    same(e['disagreement_m'], distance, tolerance=1e-12)
                    reported_angle = e['disagreement_rad']
                    require(distance <= .02 and type(reported_angle) in (int, float)
                        and math.isfinite(reported_angle) and 0 <= reported_angle <= .10,
                        'accepted rejoin cannot contradict measured position or reported angle envelope')
                    paired_positions += 1; reported_rotation_checks += 1
                else:
                    require(e['disagreement_m'] is None and e['disagreement_rad'] is None,
                        'missing incremental measurement is not observed agreement')
                same(e['preceding_bridge_frames'], active['frames'] if active is not None else 0)
                same(e['preceding_bridge_path_m'], active['measured_path_m'] if active is not None else 0., tolerance=1e-12)
                close('ANCHOR_REJOINED', frame, e)
                if pose['promoted_keyframe']:
                    retained = (retained + [frame])[-8:]; nodes += 1
        same(pose['keyframe_count'], nodes)
        previous = p; frames += 1
    close('END_OF_RECORDING', None)
    summary = dict(frames=frames, available_modes=modes, first_failure=first_failure,
        bridge_spans=spans, total_bridged_frames=bridged,
        unavailable_frames=frames - sum(modes.values()),
        operational_budget_is_error_bound=False, independent_observations=False)
    same(saved_summary, summary, tolerance=1e-12)
    return dict(available_bridge_and_rejoin_history_verified=True, summary=summary,
        recomputed_position_disagreements=paired_positions,
        range_checked_reported_rotation_disagreements=reported_rotation_checks,
        incremental_rotation_witness_saved=False, rotation_disagreements_independently_recomputed=False,
        failure_causes_independently_reconstructed=False, shared_gyro_bias_correction_established=False,
        observer_inference_recomputed=False, raw_artifacts_authenticated_by_interface=False,
        full_challenge_pass=False, navigation_qualified=False, goal_achieved=False)
