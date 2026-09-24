from types import SimpleNamespace
import numpy as np
from lewm.rollout_selection_off_development import select_current_clearance,RolloutSelectionOffMixin


def test_blocked_current_clearance_holds_including_view_requests():
    for scan in (None,.5):
        result=select_current_clearance([1.,.2],scan_error=scan,clearance_m=.449)
        assert result['action']=='hold'
        assert not any(r['eligible'] for r in result['candidates'])
        assert not result['unknown_space_inferred_free']


def test_clear_view_excludes_translation_and_current_goal_feedback_moves():
    assert select_current_clearance([1.,0.],clearance_m=.6)['action']=='forward'
    result=select_current_clearance([0.,0.],scan_error=-.6,clearance_m=.6)
    assert result['action']=='right_turn'
    assert all(not r['eligible'] for r in result['candidates'] if not r['eligible_for_view'])


def test_forecast_values_and_forecast_selection_chain_cannot_affect_action():
    class ForbiddenChain:
        def _select_clear_prediction(self,*a,**kw):
            raise AssertionError('predictive selection chain must not run')
    class Runtime(RolloutSelectionOffMixin,ForbiddenChain):
        planning_translation_pulse=False
    runtime=Runtime();snapshot=SimpleNamespace(fine_occupied={(100,y) for y in range(-100,101)})
    outputs=[]
    for action,forecast in [('left_turn',object()),('hold',np.full((6,8,5),np.nan))]:
        selected=dict(waypoint_body_xy_m=[1.,0.],action=action,candidates=object())
        outputs.append(runtime._select_clear_prediction(selected,forecast,snapshot,np.zeros(3),np.eye(3)))
    assert outputs[0]==outputs[1]
    assert outputs[0]['action']=='forward'
    assert not outputs[0]['learned_candidate_rollouts_used_for_selection']
