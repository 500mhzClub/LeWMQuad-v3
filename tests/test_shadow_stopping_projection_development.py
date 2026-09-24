from copy import deepcopy
from types import SimpleNamespace
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.planned_stopping_projection_development import PlannedStoppingProjectionMixin
from lewm.shadow_stopping_projection_development import ShadowStoppingProjectionMixin


class ExistingSelection:
    planning_translation_pulse = False

    def _select_clear_prediction(self, selected, prediction, snapshot, position, rotation):
        self.selection_calls += 1
        return deepcopy(selected)


class Enforced(PlannedStoppingProjectionMixin, ExistingSelection):
    pass


class Shadow(ShadowStoppingProjectionMixin, Enforced):
    pass


def run(cls, action='forward', wall_x=65):
    runtime = cls()
    runtime.selection_calls = 0
    runtime.clearance_turn = 'existing_recovery'
    p = np.zeros((6,8,5)); p[:,:,3] = 1.
    selection = dict(action=action, action_index=ACTIONS.index(action),
        requested_command=[.2,0.,0.] if action=='forward' else [0.,0.,0.],
        candidates=[dict(action=a,utility_m=i) for i,a in enumerate(ACTIONS)],
        memory_forecast_candidates=[dict(action=a,nominal_predicted_path_clear=a!='right_turn',
            reserve_recovery_path_clear=False) for a in ACTIONS])
    saved = deepcopy(selection)
    result = runtime._select_clear_prediction(selection,p,
        SimpleNamespace(fine_occupied={(wall_x,y) for y in range(-100,101)}),np.zeros(3),np.eye(3))
    assert selection == saved
    assert runtime.selection_calls == 1
    return runtime,result


def test_shadow_preserves_selected_translation_and_recovery_latch():
    on,selected_on = run(Enforced)
    off,selected_off = run(Shadow)
    assert selected_on['action']=='left_turn' and on.clearance_turn is None
    assert selected_off['action']=='forward' and off.clearance_turn=='existing_recovery'
    check = selected_off['planned_stopping_projection']
    assert check['would_change_action'] and check['shadow_after_action']=='left_turn'
    assert not check['changed'] and not check['enforced']
    assert check['candidates']==selected_on['planned_stopping_projection']['candidates']
    assert selected_off['memory_forecast_candidates']==selected_on['memory_forecast_candidates']


def test_clear_translation_hold_and_turn_do_not_trigger_shadow_intervention():
    for action,wall_x in [('forward',100),('hold',65),('left_turn',65),('right_turn',65)]:
        runtime,result = run(Shadow,action,wall_x)
        assert result['action']==action
        assert not result['planned_stopping_projection']['would_change_action']
        assert runtime.clearance_turn=='existing_recovery'
