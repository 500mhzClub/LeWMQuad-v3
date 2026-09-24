import ast
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from lewm.training_translation_bias_development import fit_translation_bias,TrainingTranslationBiasModel
from lewm.training_bias_predictive_selection_development import select
from lewm.observation_horizon_predictive_selection_development import select as original_select
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA
from lewm.training_bias_goal_probe_development import TrainingBiasGoalProbe
from lewm.eight_step_planning_goal_probe_development import EightStepPlanningGoalProbe
from lewm.tests.test_training_translation_bias_development import fixture
from lewm.tests.test_observation_horizon_goal_selection_development import history


def test_corrected_selector_uses_exact_causal_forecast_transformation():
    rows,arrays,schedule=fixture();receipt=fit_translation_bias(rows,arrays,schedule,head='direct_outcomes')
    base=ObservationHorizonRGBBodyJEPA(8).eval();model=TrainingTranslationBiasModel(base,{'direct_outcomes':receipt})
    kwargs=dict(head='direct_outcomes',input_variant='full',goal_body_xy_m=[1.,0.],contact_penalty_m=1.2)
    raw=original_select(base,history(),**kwargs);result=select(model,history(),**kwargs)
    expected=np.asarray(raw['prediction'],np.float32);expected[...,:2]-=np.asarray(receipt['applied_bias_xy_m'],np.float32)
    np.testing.assert_array_equal(result['prediction'],expected)
    assert result['model_prediction_corrected'] and result['translation_bias_training_only']
    assert result['target_offsets_ns']==raw['target_offsets_ns']
    with pytest.raises(ValueError):select(base,history(),**kwargs)


def test_route_view_and_scan_state_machine_are_unchanged():
    def methods(path,classname):
        tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
        return [ast.dump(n) for n in node.body]
    assert methods('lewm/training_bias_waypoint_selection_development.py','TrainingBiasWaypointSelector')==methods(
        'lewm/observation_horizon_waypoint_selection_development.py','ObservationHorizonWaypointSelector')


def test_one_command_and_original_failure_arrival_contract_remain_inherited():
    assert TrainingBiasGoalProbe.advance is EightStepPlanningGoalProbe.advance
    controller=TrainingBiasGoalProbe(object(),object(),condition='direct',variant='full',persistent=True)
    result=controller.observe({}, {}, {},now_ns=1)
    assert result['terminal']=='SENSOR_OR_MODEL_FAILURE' and result['requested_command']==[0.,0.,0.]
    assert result['maximum_open_loop_command_ticks']==1 and result['goal_initial_body_xy_m']==[1.2,0.]
