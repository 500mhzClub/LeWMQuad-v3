from copy import deepcopy
import numpy as np
import pytest
from lewm.all_phase_corrected_readout_development import verify_corrected_arrays,difference,summarize_primary
from scripts.all_phase_fit_execution_development import ROSTER
from lewm.all_phase_training_schedule_development import SEEDS


def arrays():
    valid=np.ones((2,8),bool);valid[1,3:]=False
    p=np.zeros((2,8,5),np.float32);p[valid]=[.1,.2,0.,1.,.3]
    return dict(indices=np.array([0,1],np.int64),prediction_valid=valid,
        target_offsets_ns=np.where(valid,np.arange(1,9,dtype=np.int64)*100_000_000,0),direct_outcomes=p)


def test_valid_xy_change_keeps_other_components_and_padding():
    before=arrays();after=deepcopy(before);after['direct_outcomes'][before['prediction_valid'],0]-=.05
    verify_corrected_arrays(before,after,{'direct_outcomes':{}})


@pytest.mark.parametrize('fault',['yaw','contact','clock','indices','padding','nan','extra_head','missing_head'])
def test_prediction_population_or_nonxy_changes_reject(fault):
    before=arrays();after=deepcopy(before)
    if fault=='yaw':after['direct_outcomes'][0,0,2]=.1
    if fault=='contact':after['direct_outcomes'][0,0,4]=.1
    if fault=='clock':after['target_offsets_ns'][0,0]+=1
    if fault=='indices':after['indices'][0]=9
    if fault=='padding':after['direct_outcomes'][1,4,0]=.1
    if fault=='nan':after['direct_outcomes'][0,0,0]=float('nan')
    if fault=='extra_head':after['rollout_outcomes']=after['direct_outcomes'].copy()
    if fault=='missing_head':after.pop('direct_outcomes')
    with pytest.raises(ValueError):verify_corrected_arrays(before,after,{'direct_outcomes':{}})


def records():
    rows=[]
    for assignment in ROSTER:
        offset=SEEDS.index(assignment['seed'])+1
        error=offset+dict(direct=.1,supervised_rollout=.2,jepa=0)[assignment['condition']]
        error+=.3 if assignment['variant']=='no_rgb' else 0
        after=dict(position_error_m=error,yaw_error_rad=.1,contact_brier=.02,motion_targets=4,
            contact_targets=5,contact_positives=1,undefined_yaw=0)
        for role in ('train','geometry_transfer'):
            for source in ('family','switch'):
                for scope in ('all','first_observation'):
                    before=after|dict(position_error_m=error+.1)
                    rows.append(dict(model=assignment['name'],**{k:assignment[k] for k in ('seed','variant','condition')},
                        role=role,source=source,scope=scope,head='direct_outcomes' if assignment['condition']=='direct' else 'rollout_outcomes',
                        primary_head=True,before=deepcopy(before),after=deepcopy(after)))
    return rows


def test_all_seed_and_matched_comparisons_keep_fixed_reference():
    result=summarize_primary(records(),ROSTER,SEEDS)
    grouped=next(r for r in result['three_seed_primary_descriptive'] if
        (r['variant'],r['condition'],r['role'],r['source'],r['scope'])==('full','jepa','train','family','all'))
    assert grouped['metrics']['after']['position_error_m']==dict(mean=2.,optimization_seed_sd=1.)
    assert grouped['independent_maze_confidence_interval'] is False
    for pair in result['corrected_primary_matched_pairs']:
        delta=pair['candidate_minus_reference']['position_error_m']
        if pair['comparison']=='no_rgb_minus_full_same_condition':assert delta==pytest.approx(.3)
        else:
            assert pair['reference_model'].endswith('_full_jepa')
            assert delta==pytest.approx(.1 if pair['candidate_model'].endswith('_direct') else .2)
    assert result['native_assignments_changed'] is False


@pytest.mark.parametrize('fault',['missing_model','missing_role','missing_scope','duplicate','wrong_seed','wrong_head','denominator'])
def test_partial_or_misassigned_summaries_reject(fault):
    rows=records()
    if fault=='missing_model':rows=[r for r in rows if r['model']!=ROSTER[0]['name']]
    if fault=='missing_role':rows=[r for r in rows if r['role']!='geometry_transfer']
    if fault=='missing_scope':rows.pop()
    if fault=='duplicate':rows.append(deepcopy(rows[0]))
    if fault=='wrong_seed':rows[0]['seed']=0
    if fault=='wrong_head':rows[0]['head']='rollout_outcomes'
    if fault=='denominator':rows[0]['after']['motion_targets']-=1
    with pytest.raises(ValueError):summarize_primary(rows,ROSTER,SEEDS)


def test_legitimate_undefined_yaw_difference_between_models_is_retained():
    candidate=records()[0]['after'];reference=deepcopy(candidate)
    candidate=candidate|dict(undefined_yaw=1,yaw_error_rad=None)
    delta=difference(candidate,reference)
    assert delta['yaw_error_rad'] is None and delta['position_error_m']==0
    with pytest.raises(ValueError):difference(candidate|dict(contact_targets=6),reference)
