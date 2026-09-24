import copy
import pytest
from lewm.moving_action_switch_family_development import assignments
from lewm.moving_action_switch_accounting_development import summarize


def population():
    return [dict(trial=t,**c,raw_sensor_reconstruction_pass=True,command_stop_replay_pass=True,
        prefix=dict(complete=True,sha256={'packet':c['cluster']+c['prefix_action']}),
        outcome=dict(branch_available=True,complete_schedule=True),targets=None,physical_stop=None,
        hard_measurement_failed_frames=[],strict_physical_visibility_pass=True) for t,c in assignments().items()]


def test_exact_prefix_tampering_fails_without_filtering():
    rows=population();r=summarize(rows)
    assert r['all_measurement_and_prefix_gates_pass'] and len(r['prefix_comparisons'])==24
    assert r['roles']['train']['cells']==r['roles']['geometry_transfer']['cells']==72
    assert r['roles']['train']['repeat_cells']==12
    rows[0]['prefix']['sha256']['packet']='different'
    r=summarize(rows)
    assert not r['all_measurement_and_prefix_gates_pass'] and r['episode_exclusions']==[] and r['cells']==144


def test_missing_cell_or_role_swap_rejected():
    rows=population()
    with pytest.raises(ValueError):summarize(rows[:-1])
    rows[0]['data_role']='wrong'
    with pytest.raises(ValueError):summarize(rows)


def test_uniform_unavailable_prefix_is_counted_mixed_availability_fails():
    rows=population();first=rows[0]
    group=[r for r in rows if (r['cluster'],r['prefix_action'])==(first['cluster'],first['prefix_action'])]
    for r in group:
        r['prefix']=dict(complete=False,sha256={});r['outcome']=dict(branch_available=False,complete_schedule=False)
        r['physical_stop']='DISALLOWED_CONTACT'
    result=summarize(rows)
    assert result['all_measurement_and_prefix_gates_pass']
    assert sum(r['all_six_unavailable'] for r in result['prefix_comparisons'])==1
    group[0]['prefix']=dict(complete=True,sha256={'packet':'unexpected'})
    group[0]['outcome']['branch_available']=True
    assert not summarize(rows)['all_measurement_and_prefix_gates_pass']
