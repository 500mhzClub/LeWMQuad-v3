import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from lewm.tests.test_temporal_rgb_body_jepa_development import batch
from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA
from scripts.run_go2_temporal_rgb_body_learning_comparison_development_v1 import predictions,take,state_identity,source_closure,paired_summary,SEEDS,ROOT


@pytest.mark.parametrize('condition',['direct','supervised_rollout','jepa'])
def test_chunked_intact_prediction_matches_actual_model_and_shuffle_is_nonmutating(condition):
    data=batch(); data['metadata']=[{'layout_id':'a','action_index':0,'offset_ns':0},
        {'layout_id':'b','action_index':0,'offset_ns':0}]
    saved=copy.deepcopy(data); model=TemporalRGBBodyJEPA(16).eval()
    prediction,z,eligible=predictions(model,data,condition)
    with torch.no_grad(): full=model(data['observation_history'],data['known_action_blocks'],data['known_action_valid'])
    assert np.allclose(prediction['direct'],full['direct_outcomes'].numpy(),rtol=0,atol=1e-6)
    if condition!='direct': assert np.allclose(prediction['rollout'],full['rollout_outcomes'].numpy(),rtol=0,atol=1e-6)
    assert np.allclose(z,full['latent'].numpy(),rtol=0,atol=1e-6)
    shuffled,_,_=predictions(model,data,condition,'rgb_shuffle')
    assert eligible.all() and not np.array_equal(shuffled['direct'],prediction['direct'])
    for k in data['observation_history']: assert torch.equal(data['observation_history'][k],saved['observation_history'][k])


def test_seed_identity_and_nested_indexing():
    torch.manual_seed(2026091700); a=TemporalRGBBodyJEPA(16)
    torch.manual_seed(2026091700); b=TemporalRGBBodyJEPA(16)
    assert state_identity(a)==state_identity(b)
    with torch.no_grad(): next(b.parameters()).add_(.1)
    assert state_identity(a)!=state_identity(b)
    data=batch(); data['metadata']=[{'layout_id':'a'},{'layout_id':'b'}]
    result=take(data,[1])
    assert result['observation_history']['rgb'].shape[0]==1 and result['metadata']==[{'layout_id':'b'}]


def test_ignore_aware_source_closure_includes_imports_without_protected_paths():
    closure=source_closure()
    assert 'lewm/rgb_body_jepa_reference_development.py' in closure
    assert 'lewm/temporal_prediction_metrics_development.py' in closure
    assert 'lewm/causal_sensor_state.py' in closure
    for name,sha in closure.items():
        path=Path(name)
        assert (ROOT/path).is_file() and len(sha)==64
        assert not any(p in ('sealed','sealed_test.json') or p.startswith('sealed_') for p in path.parts)


def test_paired_summary_preserves_seed_and_layout_pairing():
    runs=[]
    for seed in SEEDS:
        for condition,delta in (('direct',0.),('supervised_rollout',1.),('jepa',2.)):
            rows=[{'layout_id':f'v{i}','position_error_m':i+delta,'contact_brier':delta,
                'regret':delta,'contact':delta} for i in range(8)]
            runs.append({'seed':seed,'condition':condition,
                'validation':{'intact':{head:{'later':{'layouts':rows}} for head in ('direct','rollout')}},
                'initial_decisions':{head:{'layouts':rows} for head in ('direct','rollout')}})
    result=paired_summary(runs)
    assert result['jepa_direct_minus_supervised_rollout_direct']['later_position_error_m']['mean_delta']==1.
    assert result['jepa_direct_minus_supervised_rollout_direct']['later_contact_brier']['per_seed_mean_delta']==[1.,1.,1.]
    assert result['jepa_rollout_minus_jepa_direct']['initial_regret']['layout_bootstrap_95_percentile']==[0.,0.]
