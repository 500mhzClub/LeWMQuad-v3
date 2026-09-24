from copy import deepcopy
import numpy as np
import pytest
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.measured_floor_partition_development import FOOT_IDS
from lewm.later_floor_resolution_execution_readout_development import later_floor_resolution_execution


@pytest.fixture
def executed():
    tick = 2; action = 'forward'; feet = sorted(FOOT_IDS)
    original = dict(possible_intersection=True)
    check = dict(possible_intersection=False, non_foot_contacts_exempted=False,
        unresolved_contacts_exempted=False, later_floor_contact_resolution=[])
    for camera, field in (('primary', 'shapes'), ('auxiliary', 'auxiliary_shapes')):
        original[field] = [dict(shape_id='trunk', intersecting_voxels=0)]
        check[field] = [dict(shape_id='trunk', intersecting_voxels=0)]
        for foot in feet:
            count = int(camera == 'primary' and foot == feet[0])
            original[field].append(dict(shape_id=foot, intersecting_voxels=count))
            check[field].append(dict(shape_id=foot, intersecting_voxels=0))
            check['later_floor_contact_resolution'].append(dict(source_camera=camera, shape_id=foot,
                original_intersections=count, resolved_intersections=count, remaining_intersections=0,
                all_original_returns_and_partitions_retained=True,
                enclosures=[dict(resolved=True, latest_ambiguous_sample_frame=0,
                    later_single_view=dict(frame=1, measured_ns=1_600_000_000))] if count else []))
    check['original_contact_check_before_later_floor_resolution'] = original
    prediction = [[[.01, 0.]] for _ in ACTIONS]
    command = candidate_commands(action)[0]
    d = dict(terminal=None, selected_action=action, requested_command=command,
        later_measured_floor_contact_resolution_enabled=True,
        new_selection=dict(action=action, prediction=prediction,
            surface_checks=[deepcopy(check) for _ in ACTIONS]))
    poses = np.zeros((900, 7)); poses[:, 6] = 1.; poses[899, 0] = .01
    tape = [dict(tick=i, pre_sample_index=749+50*i, post_sample_index=799+50*i,
        requested_command=command, completed=True) for i in range(3)]
    return poses, tape, [dict(tick=tick, decision=d)]


def test_complete_actual_interval_has_exact_endpoint_and_no_counterfactual(executed):
    report = later_floor_resolution_execution(*executed)
    assert report['completed_intervals'] == 1 and report['censored_intervals'] == 0
    assert report['records'][0]['forecast_xy_error_m'] == 0.
    assert not report['original_controller_counterfactual_trajectory_inferred']


def test_censored_interval_does_not_infer_endpoint(executed):
    poses, tape, rows = executed; tape[2]['completed'] = False
    report = later_floor_resolution_execution(poses[:875], tape, rows)
    assert report['censored_intervals'] == 1 and report['completed_intervals'] == 0
    assert report['records'][0]['native_body_xy_m'] is None


@pytest.mark.parametrize('fault', ['same_frame', 'future', 'clock', 'missing_hit', 'nonfoot', 'unresolved', 'request'])
def test_invalid_contact_or_execution_attribution_is_rejected(executed, fault):
    poses, tape, rows = executed
    d = rows[0]['decision']; check = d['new_selection']['surface_checks'][ACTIONS.index('forward')]
    q = next(q for q in check['later_floor_contact_resolution'] if q['enclosures'])
    witness = q['enclosures'][0]['later_single_view']
    if fault == 'same_frame': witness.update(frame=0, measured_ns=1_500_000_000)
    elif fault == 'future': witness.update(frame=3, measured_ns=1_800_000_000)
    elif fault == 'clock': witness['measured_ns'] += 1
    elif fault == 'missing_hit': q['enclosures'].clear()
    elif fault == 'nonfoot': check['shapes'][0]['intersecting_voxels'] = 1
    elif fault == 'unresolved': q['remaining_intersections'] = 1
    else: tape[2]['requested_command'] = [0., 0., 0.]
    with pytest.raises(ValueError): later_floor_resolution_execution(poses, tape, rows)
