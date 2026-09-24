import numpy as np
import pytest

from lewm.counterfactual_decision_diagnostic_development import decision_rows


def inputs():
    prediction=np.zeros((5,8,5)); prediction[...,3]=1; prediction[...,4]=-30
    motion=np.zeros((5,8,3)); motion[1,:,:2]=[.8,0]; motion[2,:,:2]=[0,.8]; motion[3,:,:2]=[0,-.8]
    prediction[...,:2]=motion[...,:2]
    targets={'motion':motion,'contact':np.zeros((5,8)),'motion_valid':np.ones((5,8),dtype=bool),'contact_valid':np.ones((5,8),dtype=bool)}
    return prediction,targets,[{'layout_id':'a','action_index':i} for i in range(5)]


def test_perfect_predictions_choose_each_intent():
    prediction,targets,metadata=inputs(); result=decision_rows(prediction,targets,metadata)
    assert [r['chosen_action_index'] for r in result['rows']]==[1,2,3]
    assert result['layout_macro']['regret']==0


def test_contact_censoring_is_not_imputed_and_unknown_rejects_layout():
    prediction,targets,metadata=inputs()
    targets['contact'][1,-1]=1; targets['motion_valid'][1,-1]=False; targets['motion'][1,-1]=np.nan
    result=decision_rows(prediction,targets,metadata)
    assert result['rows'][0]['realized_cost']==10
    assert result['rows'][0]['regret']==pytest.approx(9.2)
    targets['contact_valid'][1,-1]=False
    result=decision_rows(prediction,targets,metadata)
    assert result['excluded_incomplete_layouts']==['a'] and result['layout_macro'] is None


def test_all_stop_tie_and_missing_sibling_rejection():
    prediction,targets,metadata=inputs(); prediction[...,:2]=0
    assert decision_rows(prediction,targets,metadata)['layout_macro']['stop']==1
    with pytest.raises(ValueError,match='five sibling'): decision_rows(prediction[:-1],{k:v[:-1] for k,v in targets.items()},metadata[:-1])
