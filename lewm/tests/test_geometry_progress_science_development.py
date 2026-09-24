"""Readout denominator, stopped-prefix and missing-modality regression checks."""
from copy import deepcopy
import pytest
from lewm.geometry_progress_pilot_development import assignments, progress_outcome
from scripts.read_go2_geometry_progress_science_v1 import summary


def reports():
    rows=[]
    for trial,c in assignments().items():
        arc=c['action'].endswith('_arc')
        matching=c['action']==('left_arc' if c['geometry']=='left_open' else 'right_arc')
        contact=c['action']=='forward' or arc and not matching
        xy=[.25,.05] if arc or contact else [0.,0.]
        o=progress_outcome(xy,complete=not contact,disallowed_contact=contact,
            physical_stop='DISALLOWED_CONTACT' if contact else None,acquisition_stop=None)
        target=dict(motion_valid=not contact,future_image_valid=not contact,
                    contact_valid=True,contact=float(contact))
        rows.append(dict(trial=trial,**c,outcome=o,terminal_displacement_departure_body_xy_m=xy,
            raw_sensor_reconstruction_pass=True,command_stop_replay_pass=True,setup_admitted=True,
            targets=dict(targets=[dict(target) for _ in range(8)]),
            observation_and_control_wall_ms=[10.,120.],
            depth_checks=[dict(within1mm=True,maximum_error_m=.0001)],frames=2,decisions=2,physics_samples=1000))
    return rows


def test_full_accounting_keeps_contact_with_missing_images_and_stopped_progress_separate():
    result=summary(reports())
    assert result['design_and_measurement_gate_pass'] and result['successful_progress']==4
    assert result['contact_episodes']==8 and result['full_horizons']==16
    assert result['target_accounting']['recorded_slots']==192
    assert result['target_accounting']['contact_positive']==64
    assert result['target_accounting']['positive_with_missing_image']==64
    assert result['target_accounting']['positive_with_missing_motion']==64
    assert result['by_action']['forward']['complete_horizon_progress_m']==[]
    assert len(result['by_action']['forward']['stopped_prefix_progress_m'])==4
    assert not result['rgb_benefit_established'] and not result['goal_achieved']


@pytest.mark.parametrize('fault',['subset','assignment','outcome','replay'])
def test_denominator_assignment_score_and_audit_mismatch_rejected(fault):
    rows=reports()
    if fault=='subset':rows.pop()
    if fault=='assignment':rows[0]['action']='hold'
    if fault=='outcome':rows[0]['outcome']['progress_m']+=.01
    if fault=='replay':rows[0]['raw_sensor_reconstruction_pass']=False
    with pytest.raises(ValueError):summary(rows)


def test_native_design_success_does_not_override_depth_measurement_failure():
    rows=reports();rows[0]['depth_checks'][0]['within1mm']=False
    r=summary(rows)
    assert r['panel']['informative_for_next_dataset'] and not r['design_and_measurement_gate_pass']
