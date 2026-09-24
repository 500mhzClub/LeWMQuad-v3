"""Complete roles include native failures; no winning-layout exclusions."""
import pytest
from lewm.geometry_progress_layout_family_development import assignments,progress_outcome
from lewm.geometry_progress_family_accounting_development import summarize


def reports():
    rows=[]
    for t,c in assignments().items():
        a=c['action'];match=a==('left_arc' if c['opening']=='left_open' else 'right_arc')
        contact=a=='forward' or a.endswith('_arc') and not match
        outcome=progress_outcome([.3,0.] if match or contact else [0.,0.],complete=not contact,
            disallowed_contact=contact,physical_stop='DISALLOWED_CONTACT' if contact else None,acquisition_stop=None)
        rows.append(dict(trial=t,**c,outcome=outcome,setup_admitted=True,frames=44,
            raw_sensor_reconstruction_pass=True,command_stop_replay_pass=True,
            targets=dict(targets=[dict(motion_valid=not contact,future_image_valid=not contact,contact_valid=True,
                contact=float(contact)) for _ in range(8)]),
            hard_measurement_failed_frames=[],strict_physical_visibility_pass=True))
    return rows


def test_complete_roles_and_all_contact_targets_retained():
    r=summarize(reports());assert r['episodes']==96 and r['all_measurement_gates_pass'] and r['training_design_informative']
    for role in r['roles'].values():
        assert role['episodes']==48 and role['parameter_clusters']==2 and role['layouts']==4
        assert role['successful_progress']==8 and role['contact_episodes']==16
        assert role['target_accounting']['recorded_slots']==384 and role['target_accounting']['contact_positive']==128
    assert r['episode_exclusions']==[] and not r['navigation_qualified']


def test_a_bad_transfer_measurement_is_retained_and_cannot_pass_whole_gate():
    rows=reports();row=next(r for r in rows if r['data_role']=='geometry_transfer');row['hard_measurement_failed_frames']=[17]
    r=summarize(rows);assert not r['all_measurement_gates_pass'] and r['training_design_informative']
    assert r['roles']['geometry_transfer']['episodes']==48


@pytest.mark.parametrize('fault',['missing','duplicate','role','raw'])
def test_incomplete_or_misassigned_cohort_rejected(fault):
    rows=reports()
    if fault=='missing':rows.pop()
    if fault=='duplicate':rows[-1]=rows[0]
    if fault=='role':rows[0]['data_role']='new_role'
    if fault=='raw':rows[0]['raw_sensor_reconstruction_pass']=False
    with pytest.raises(ValueError):summarize(rows)
