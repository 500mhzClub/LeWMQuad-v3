import pytest

from lewm.stop_conditioned_comparators_development import CONTROLLERS
from lewm.stop_conditioned_settling_development import StopConditionedSettlingMission
from lewm.extended_return_budget_current_planning_development import ExtendedReturnBudgetCurrentPlanningMap
from scripts import stop_conditioned_comparator_pipeline_development as pipeline


@pytest.mark.parametrize('mode', tuple(CONTROLLERS))
def test_all_comparators_share_stopping_rule_and_keep_their_treatments(mode):
    mission = dict(goal_initial_body_xy_m=[.2, 0.],
        return_initial_body_xy_m=[0., 0.], require_return_after_goal=True)
    kwargs = dict(public_mission=mission, navigation_ticks=8000)
    if mode == 'reactive':
        c = CONTROLLERS[mode](None, **kwargs)
        assert not hasattr(c, 'model') and not hasattr(c, 'residual')
    else:
        c = CONTROLLERS[mode](None, None, condition='jepa', variant='full', persistent=True, **kwargs)
        assert c.residual is c.selector.residual
    assert isinstance(c.mission, StopConditionedSettlingMission)
    assert c.memory is c.mapper.surface
    if mode in ('frozen_reference', 'nominal'):
        assert c.selector.forecast_source == ('frozen_world_model' if mode == 'frozen_reference' else 'nominal_requested_twist')
    if mode == 'current_planning':
        assert isinstance(c.mapper, ExtendedReturnBudgetCurrentPlanningMap)
    for i in range(5):
        result = c.mission.advance([.2, 0., 0.], frame=i,
            now_ns=1_500_000_000+i*100_000_000,
            previous_requested_command=[.16, 0., .45] if i < 4 else [0., 0., 0.])
    assert result['quiet_intervals'] == 0 and not result['arrivals']
    # Collection and replay both call the executor carrying this same table.
    assert pipeline.collect.__globals__['_execute'] is pipeline.execute
    assert pipeline.audit.__globals__['_execute'] is pipeline.execute
    assert pipeline.execute.__globals__['CONTROLLERS'][mode] is CONTROLLERS[mode]
