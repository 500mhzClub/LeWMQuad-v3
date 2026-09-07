import copy
import json

import numpy as np
import pytest

from lewm.temporal_prediction_metrics_development import reduce_predictions,prediction_report,shuffle_population,simple_predictions,initial_decisions


def fixture(n=2):
    metadata=[{'layout_id':f'layout-{i}','action_index':0,'offset_ns':0,'data_role':'train'} for i in range(n)]
    mask=np.zeros((n,8,5),bool); mask[:,0]=True
    mv=mask.all(-1); motion=np.full((n,8,3),np.nan); motion[mv]=0.
    contact=np.full((n,8),np.nan); contact[mv]=0.
    batch={'metadata':metadata,'known_action_blocks':np.zeros((n,8,5,3)),'known_action_valid':mask,
        'targets':{'motion':motion,'motion_valid':mv.copy(),'contact':contact,'contact_valid':mv.copy()}}
    prediction=np.full((n,8,5),np.nan); prediction[mv]=[0,0,0,1,0]
    return prediction,batch


def test_unknown_horizons_and_empty_later_subsets_do_not_create_zero_scores_or_nan_json():
    p,b=fixture(); result=prediction_report(p,b)
    assert result['all']['motion_valid']==2
    assert result['all']['layout_macro']['contact_brier']==.25
    assert result['later']['layout_macro']['contact_brier'] is None
    assert result['by_horizon_seconds']['4.0']['layout_macro']['position_error_m'] is None
    assert result['all']['layout_macro']['contact_monotonicity_violation_fraction'] is None
    json.dumps(result,allow_nan=False)


def test_layout_macro_not_window_micro_and_strict_contact_censoring():
    p,b=fixture(3); b['metadata'][1]['layout_id']='layout-0'
    p[2,0,0]=3.
    result=prediction_report(p,b)
    assert result['all']['layout_macro']['position_error_m']==1.5
    b['targets']['motion_valid'][2,0]=False; b['targets']['motion'][2,0]=np.nan
    b['targets']['contact'][2,0]=1.
    result=prediction_report(p,b)
    assert result['all']['layout_macro']['position_error_m']==0.
    assert result['all']['contributing_layouts']['position_error_m']==1
    assert result['all']['contact_positives']==1


def test_monotonicity_uses_only_adjacent_known_selected_horizons():
    p,b=fixture(1); b['known_action_valid'][0,:2]=True; p[0,1]=[0,0,0,1,-2]
    report=prediction_report(p,b)
    assert report['all']['layout_macro']['contact_monotonicity_violation_fraction']==1.
    assert report['last_known']['layout_macro']['contact_monotonicity_violation_fraction'] is None
    assert report['all']['layouts'][0]['monotonicity_pair_count']==1


def test_targets_outside_plan_rejected():
    p,b=fixture(); b['targets']['contact_valid'][0,7]=True
    with pytest.raises(ValueError,match='contract'): prediction_report(p,b)


def test_shuffle_cells_require_all_layouts_and_preserve_action_offset():
    rows=[{'layout_id':layout,'action_index':a,'offset_ns':offset} for layout in ('a','b','c') for a in (0,1) for offset in (0,500_000_000)]
    rows=[r for r in rows if r!=dict(layout_id='c',action_index=1,offset_ns=500_000_000)]
    donors,eligible=shuffle_population(rows)
    assert eligible.sum()==9
    for i,j in enumerate(donors):
        if eligible[i]:
            assert rows[i]['layout_id']!=rows[j]['layout_id']
            assert (rows[i]['action_index'],rows[i]['offset_ns'])==(rows[j]['action_index'],rows[j]['offset_ns'])
        else: assert i==j


def test_empirical_controls_ignore_eval_labels_and_report_training_fallback():
    p,train=fixture(); _,evaluation=fixture()
    evaluation['metadata'][0]['offset_ns']=500_000_000
    baseline,coverage=simple_predictions(train,evaluation)
    assert coverage['motion_cells']==coverage['contact_cells']==1
    altered=copy.deepcopy(evaluation); altered['targets']['motion'][:]=999.; altered['targets']['contact'][:]=1.
    other,_=simple_predictions(train,altered)
    for name in baseline: assert np.array_equal(baseline[name],other[name])
    assert baseline['training_action_remaining_mean'][0,0,:2]==pytest.approx([0,0])
    train['metadata'][0]['data_role']='validation'
    with pytest.raises(ValueError,match='training role'): simple_predictions(train,evaluation)


def test_action_fallback_missing_stays_failure_not_validation_imputation():
    _,train=fixture(); _,evaluation=fixture(); evaluation['metadata'][0]['action_index']=1
    with pytest.raises(ValueError,match='fallback unavailable'): simple_predictions(train,evaluation)


def test_kinematics_respects_plan_stop_and_unknown_tail():
    _,train=fixture(); _,evaluation=fixture(); evaluation['known_action_blocks'][:,0,:,0]=1.
    prediction,_=simple_predictions(train,evaluation)
    k=prediction['command_kinematics_no_contact']
    assert k[:,0,0]==pytest.approx([.15,.15])
    assert np.count_nonzero(k[:,1:])==0


def test_initial_decisions_exclude_later_states_and_use_recorded_contacts():
    _,b=fixture(6)
    for i in range(6): b['metadata'][i].update(layout_id='one',action_index=i%5,offset_ns=0 if i<5 else 500_000_000)
    b['known_action_valid'][:]=True
    b['targets']['motion_valid'][:]=True; b['targets']['contact_valid'][:]=True
    b['targets']['motion'][:]=0; b['targets']['contact'][:]=0
    p=np.zeros((6,8,5)); p[...,3]=1.; p[...,4]=-30.
    b['targets']['contact'][1,-1]=1; b['targets']['motion_valid'][1,-1]=False; b['targets']['motion'][1,-1]=np.nan
    p[1,-1,:2]=[.8,0]
    report=initial_decisions(p,b)
    assert len(report['rows'])==3
    assert report['rows'][0]['chosen_action_index']==1
    assert report['rows'][0]['contact'] and report['rows'][0]['realized_cost']==10.
