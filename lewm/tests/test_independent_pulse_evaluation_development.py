"""Synthetic-only scientific accounting: layout units, role leakage and pairing."""
from copy import deepcopy
import numpy as np
import pytest
from lewm.independent_layout_inventory_development import build_inventory
from lewm.independent_layout_collection_development import CollectionInventory
from lewm.pulse_timed_dataset_development import PulseTimedDataset
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan,validate_timed_plan
from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation


def fixture():
    inv=CollectionInventory(build_inventory());windows=[];targets=[];roles={}
    for layout in inv.data['layouts']:
        count=6 if layout['role']=='train' else 1+layout['layout_index']%3
        episodes=[e for e in inv.episodes.values() if e['layout_id']==layout['layout_id']][:count]
        for e in episodes:
            b,m=pulse_brake_plan(tuple(e['command']),e['pulse_ticks']);active,offsets=validate_timed_plan(b[None],m[None],1)
            w=dict(condition=e['episode_id'],departure_tick=8,decision_ns=2_300_000_000,action_index=e['action_index'],
                command=e['command'],pulse_ticks=e['pulse_ticks'],history_ready=True,history_observation_indices=[5,6,7,8],
                targets=[dict(offset_ns=int(o),future_valid=bool(v)) for o,v in zip(offsets[0],active[0])])
            t={k:w[k] for k in ('condition','departure_tick','decision_ns','action_index')}
            t.update(target_only=True,targets=[dict(offset_ns=r['offset_ns'],image_target_valid=r['future_valid'],
                motion_valid=r['future_valid'],contact_valid=r['future_valid'],
                motion=[.01*(layout['layout_index']+1),0.,0.] if r['future_valid'] else None,
                contact=0. if r['future_valid'] else None) for r in w['targets']])
            windows.append(w);targets.append(t);roles[w['condition']]=dict(layout_id=e['layout_id'],role=e['role'])
    return inv,windows,targets,roles


def view():
    i,w,t,r=fixture();return IndependentPulseEvaluation(i,PulseTimedDataset(w,t,r))


def prediction(data,offset=0.):
    p=np.zeros((*data['active'].shape,5));p[...,3]=1.;p[...,4]=-30.
    p[...,:2]=np.where(data['targets']['motion_valid'][...,None],data['targets']['motion'][...,:2],0.)
    p[...,0]+=offset
    return dict(indices=data['indices'].copy(),prediction=p)


def test_population_reports_all_planned_layouts_and_missing_episodes():
    v=view();p=v.population('development_eval')
    assert p['planned_episodes']==360 and len(p['layouts'])==3
    assert p['present_episodes']+len(p['absent_from_dataset'])==360
    assert not p['all_planned_episodes_eligible']
    assert not p['visibility_verified_by_interface'] and not p['source_artifacts_verified_by_interface']


def test_macro_is_layout_weighted_not_frame_count_weighted():
    v=view();data=v.arrays('development_eval');entry=prediction(data);entry['prediction'][...,:2]=0.
    result=v.compare(dict(zero=entry),role='development_eval');all_=result['metrics']['zero']['all']
    values=[r['position_error_m'] for r in all_['layouts']]
    assert all_['layout_macro']['position_error_m']==pytest.approx(np.mean(values))
    assert result['independent_layout_units']==3 and result['resubstitution'] is False
    assert not result['population']['all_planned_episodes_eligible']
    pooled=np.mean(data['targets']['motion'][data['targets']['motion_valid'],0])
    if len(set(r['windows'] for r in all_['layouts']))>1:
        assert not np.isclose(np.mean(values),pooled,rtol=0,atol=1e-6)


def test_paired_differences_use_identical_layouts_and_do_not_invent_frame_level_ci():
    v=view();data=v.arrays('development_eval')
    report=v.compare(dict(exact=prediction(data),shifted=prediction(data,.03)),role='development_eval')
    paired=report['paired_comparisons']['exact minus shifted']
    assert paired['paired_layouts']==3 and paired['confidence_interval'] is None
    assert paired['macro_difference']['position_error_m']==pytest.approx(-.03)
    assert all(r['position_error_m']==pytest.approx(-.03) for r in paired['layout_differences'])
    assert report['all_requested_heads_comparable']


def test_same_role_baseline_uses_only_exact_training_draws_with_multiplicity():
    v=view();data=v.arrays('train');ids=[int(data['indices'][0]),int(data['indices'][0]),int(data['indices'][6])]
    model,record=v.fit_action_time(ids)
    assert record['draws']==3 and record['distinct_windows']==2
    expected=(2*data['targets']['motion'][0,0,0]+data['targets']['motion'][6,0,0])/3
    assert model.cells[(0,500_000_000)]['motion_mean'][0]==pytest.approx(expected)
    assert record['training_draw_indices']==ids and not record['evaluation_targets_used']


@pytest.mark.parametrize('role',['selection','development_eval'])
def test_baseline_rejects_any_nontrain_exposure(role):
    v=view();train=v.arrays('train');other=v.arrays(role)
    with pytest.raises(ValueError,match='train-role'):v.fit_action_time([int(train['indices'][0]),int(other['indices'][0])])


def test_unseen_action_baseline_is_unavailable_not_scored_on_favorable_subset():
    v=view();train=v.arrays('train');data=v.arrays('development_eval')
    model,_=v.fit_action_time(train['indices'][train['actions']==0])
    p,missing=model.predict(data['actions'],data['offsets_ns'],data['active']);assert missing
    result=v.compare(dict(empirical=dict(indices=data['indices'],prediction=p),exact=prediction(data)),role='development_eval')
    assert not result['all_requested_heads_comparable'] and 'empirical' in result['unavailable_heads']
    assert 'empirical' not in result['metrics'] and not result['paired_comparisons']
    assert result['metrics']['exact']['all']['windows']==len(data['indices'])


@pytest.mark.parametrize('fault',['order','duplicate','float_index','shape','head_name'])
def test_pair_identity_and_shape_errors_are_hard_failures(fault):
    v=view();data=v.arrays('development_eval');p=prediction(data);name='model'
    if fault=='order':p['indices']=p['indices'][::-1]
    elif fault=='duplicate':p['indices'][1]=p['indices'][0]
    elif fault=='float_index':p['indices']=p['indices'].astype(float)
    elif fault=='shape':p['prediction']=p['prediction'][:1]
    else:name='ambiguous minus name'
    with pytest.raises(ValueError):v.compare({name:p},role='development_eval')


def test_missing_history_counted_separately_from_missing_episode():
    i,w,t,r=fixture();c=w[0]['condition'];w[0]['history_ready']=False;w[0]['history_observation_indices'][0]=None
    v=IndependentPulseEvaluation(i,PulseTimedDataset(w,t,r));p=v.population(r[c]['role'])
    assert p['missing_history']==[c] and c not in p['absent_from_dataset']
    assert p['eligible_episodes']==p['present_episodes']-1


@pytest.mark.parametrize('fault',['layout','role','time','condition'])
def test_frozen_inventory_identity_cannot_be_reassigned(fault):
    i,w,t,r=fixture();c=w[0]['condition']
    if fault=='layout':r[c]['layout_id']='invented-layout'
    elif fault=='role':
        layout=r[c]['layout_id']
        for row in r.values():
            if row['layout_id']==layout:row['role']='selection' if row['role']=='train' else 'train'
    elif fault=='time':w[0]['departure_tick']=t[0]['departure_tick']=9;w[0]['decision_ns']=t[0]['decision_ns']=2_400_000_000
    else:w[0]['condition']=t[0]['condition']='unplanned';r['unplanned']=r.pop(c)
    with pytest.raises(ValueError):IndependentPulseEvaluation(i,PulseTimedDataset(w,t,r))


def test_mutating_caller_dataset_or_returned_arrays_cannot_change_evaluation_view():
    i,w,t,r=fixture();d=PulseTimedDataset(w,t,r);v=IndependentPulseEvaluation(i,d)
    before=v.arrays('train');d._targets[0]['motion'].fill_(999.)
    d.episode_roles[w[0]['condition']]['role']='selection'
    returned=v.arrays('train');returned['targets']['motion'].fill(888.)
    after=v.arrays('train');np.testing.assert_array_equal(before['targets']['motion'],after['targets']['motion'])


def test_partial_endpoints_and_target_masks_are_not_rounded_or_pooled():
    v=view();d=v.arrays('train');result=v.compare(dict(exact=prediction(d)),role='train')
    assert result['resubstitution']
    times=result['metrics']['exact']['by_actual_offset_ns']
    assert '2200000000' in times and '2500000000' in times
    assert all(r['known_horizons']>0 for r in times.values())
