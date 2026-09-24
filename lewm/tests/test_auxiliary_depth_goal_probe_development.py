from lewm.auxiliary_depth_goal_probe_development import AuxiliaryDepthGoalProbe
from lewm.training_bias_goal_probe_development import TrainingBiasGoalProbe
from lewm.training_bias_goal_probe_development import TrainingBiasEightStepSelector


def test_controller_inherits_original_execution_and_selector():
    c=AuxiliaryDepthGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    assert AuxiliaryDepthGoalProbe.advance is TrainingBiasGoalProbe.advance
    assert type(c.selector) is TrainingBiasEightStepSelector
    assert c.memory is c.mapper.surface


def test_invalid_or_missing_sensing_stops_and_latches_without_model_call():
    c=AuxiliaryDepthGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    first=c.observe({}, {}, {},now_ns=1)
    assert first['terminal']=='SENSOR_OR_MODEL_FAILURE' and first['requested_command']==[0.,0.,0.]
    assert first['auxiliary_floor_partition_receipt'] is None
    later=c.observe({}, {}, {},now_ns=2)
    assert later['terminal']==first['terminal'] and later['requested_command']==[0.,0.,0.]
