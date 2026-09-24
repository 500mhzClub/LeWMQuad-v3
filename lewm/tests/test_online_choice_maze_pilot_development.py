import copy

import pytest

from lewm.online_choice_maze_pilot_development import corpus,layout_spec,realized_cost,trials
from lewm.counterfactual_maze_development import corpus as previous_corpus


def test_fixed_layouts_are_disjoint_and_trial_population_is_complete():
    layouts=corpus(); rows=trials()
    assert len(layouts)==8 and len(rows)==72 and len({r['scene_id'] for r in rows})==72
    assert {r['procedural_seed'] for r in layouts}==set(range(2026091400,2026091408))
    assert not {r['topology_sha256_dihedral'] for r in layouts}&{r['topology_sha256_dihedral'] for r in previous_corpus()}
    for layout in layouts:
        members=[r for r in rows if r['layout_id']==layout['layout_id']]
        assert len(members)==9 and all(r['geometry']==layout['geometry'] for r in members)
    assert rows==trials()
    rows[0]['geometry']['wall_boxes'].clear()
    assert rows[1]['geometry']['wall_boxes']


@pytest.mark.parametrize('index',[-1,8,True,1.5])
def test_layout_bounds(index):
    with pytest.raises(ValueError): layout_spec(index)


def labels():
    return [{'horizon_ns':h*500_000_000,'motion_valid':True,'delta_xy_yaw_start_body':[.6,0,0],
        'contact_valid':True,'contact_by_horizon':False} for h in range(1,9)]


def test_progress_and_release_failures_are_distinct():
    actual=realized_cost(labels(),prefix_available=True,stop_reason=None,intent_xy=[.8,0])
    assert actual['cost']==pytest.approx(.2) and actual['progress_m']==pytest.approx(.6)
    stopped=realized_cost(labels(),prefix_available=True,stop_reason='DISALLOWED_CONTACT',intent_xy=[.8,0])
    assert stopped['cost']==10 and stopped['kind']=='post_horizon_failure'


@pytest.mark.parametrize('fault',['prefix','contact','unobserved'])
def test_missing_and_contact_motion_are_not_imputed(fault):
    rows=labels()
    rows[-1]['motion_valid']=False; rows[-1]['delta_xy_yaw_start_body']=None
    if fault=='contact': rows[-1]['contact_by_horizon']=True
    if fault=='unobserved': rows[-1]['contact_valid']=False
    result=realized_cost(rows,prefix_available=fault!='prefix',stop_reason='STOP',intent_xy=[.8,0])
    assert result['cost']==10 and result['progress_m'] is None


def test_paired_reduction_keeps_intents_inside_layout():
    from scripts.run_go2_online_choice_maze_pilot_development_v1 import paired_reduction
    rows=[]
    for spec in trials():
        rows.append(spec | {'utility':{'cost':.8 if spec['method']=='always_stop' else .4},
            'stop_reason':None,'branchable':True,'selected_stop':spec['method']=='always_stop'})
    result=paired_reduction(rows)
    assert len(result['layouts'])==8
    contrast=result['paired_cost_comparisons']['supervised_rollout_minus_always_stop']
    assert contrast['mean_cost_delta']==pytest.approx(-.4)
    assert contrast['descriptive_layout_bootstrap_95_percentile']==pytest.approx([-.4,-.4])
    with pytest.raises(ValueError,match='incomplete paired'): paired_reduction(rows[:-1])
