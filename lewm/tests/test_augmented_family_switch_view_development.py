from collections import Counter
import numpy as np
import pytest
import torch
from lewm.augmented_family_switch_view_development import AugmentedFamilySwitchView
from lewm.augmented_family_switch_fit_development import verified_plan,score
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from lewm.moving_action_switch_learning_view_development import MovingActionSwitchView
from lewm.geometry_progress_family_causal_windows_development import remaining_candidate
from lewm.geometry_progress_pilot_development import timed_candidate
from lewm.tests.test_geometry_progress_family_learning_view_development import windows
from lewm.tests.test_moving_action_switch_policy_stream_development import population


def view():
    old=windows()
    for w in old:
        if w['available']:
            w['targets']=[dict(in_plan=5*(i+1)<=40-w['offset_ticks'],offset_ns=(i+1)*500_000_000 if 5*(i+1)<=40-w['offset_ticks'] else 0,
                motion_valid=5*(i+1)<=40-w['offset_ticks'],motion=[0.,0.,0.] if 5*(i+1)<=40-w['offset_ticks'] else None,
                contact_valid=5*(i+1)<=40-w['offset_ticks'],contact=0. if 5*(i+1)<=40-w['offset_ticks'] else None) for i in range(8)]
    return AugmentedFamilySwitchView(FamilyWindowView(old),MovingActionSwitchView(population()))


def test_equal_source_mixture_is_complete_balanced_and_target_invariant():
    v=view();s=v.schedule(updates=1200,batch_size=6,seed=2026091401)
    sources=Counter(v.rows[b[0]]['source'] for b in s['batches'])
    assert sources=={'family':600,'switch':600}
    counts=Counter((v.rows[i]['source'],v.rows[i]['trial']) for b in s['batches'] for i in b)
    assert {n for (source,_),n in counts.items() if source=='family'}=={75}
    assert {n for (source,_),n in counts.items() if source=='switch'}=={50}
    assert len(counts)==120 and all(v.rows[i]['data_role']=='train' for b in s['batches'] for i in b)
    for b in s['batches']:assert len({v.rows[i]['source'] for i in b})==1
    for r in v.rows:
        if r['available']:r['targets']=[dict(motion=[999.,999.,999.],contact=1.)]*8
    assert v.schedule(updates=1200,batch_size=6,seed=2026091401)==s


def test_plan_binding_rejects_cross_source_action_changes():
    v=view();ids=[v.indices('train',source=s)[0] for s in ('family','switch')]
    plans=[remaining_candidate(v.rows[i]['action'],v.rows[i]['offset_ticks']) if v.rows[i]['source']=='family'
        else timed_candidate(v.rows[i]['action']) for i in ids]
    inputs=dict(known_action_blocks=torch.stack([p[0] for p in plans]),known_action_valid=torch.stack([p[1] for p in plans]))
    active,offsets=verified_plan(v,ids,inputs);assert active.shape==offsets.shape==(2,8)
    inputs['known_action_blocks'][1,0,0,0]+=.1
    with pytest.raises(ValueError,match='exact'):verified_plan(v,ids,inputs)


def test_scoring_preserves_sources_horizons_and_complete_prediction_order():
    v=view();ids=v.indices('geometry_transfer');values=np.zeros((len(ids),8,5),np.float32);values[:,:,3]=1.
    arrays=dict(indices=np.array(ids),prediction_valid=np.array([[t['in_plan'] for t in v.rows[i]['targets']] for i in ids]),
        target_offsets_ns=np.array([[t['offset_ns'] for t in v.rows[i]['targets']] for i in ids]),direct_outcomes=values)
    result=score(v,arrays,role='geometry_transfer',head='direct_outcomes')
    assert result['planned_family_windows']==384 and result['planned_switch_cells']==72
    assert {r['source'] for r in result['clusters']}=={'family','switch'}
    new=[r for r in result['clusters'] if r['source']=='switch' and r['scope']=='first_half_second']
    assert sum(r['motion_targets'] for r in new)==sum(r['contact_targets'] for r in new)==72
    assert all(r['contact_brier']==.25 for r in new)
    arrays['indices']=arrays['indices'][::-1]
    with pytest.raises(ValueError,match='ordered'):score(v,arrays,role='geometry_transfer',head='direct_outcomes')
