import math
import pytest
from lewm.continuous_reactive_selection_development import select_reactive
from lewm.geometry_progress_pilot_development import candidate_commands


@pytest.mark.parametrize('heading,action',[(0.,'forward'),(.4,'left_arc'),(-.4,'right_arc'),(math.pi,'left_turn')])
def test_current_direction_uses_existing_six_command_feedback(heading,action):
    row=select_reactive([math.cos(heading),math.sin(heading)],current_clearance_m=.6)
    assert row['action']==action
    assert row['requested_command']==candidate_commands(action)[0]
    assert not row['candidate_future_outcomes_evaluated']


def test_viewing_and_current_collision_never_request_translation():
    row=select_reactive([0.,0.],scan_error=.4,current_clearance_m=.6)
    assert row['action']=='left_turn'
    assert all(not r['eligible'] for r in row['candidates'] if r['requested_command'][0]>0)
    assert select_reactive([1.,0.],current_clearance_m=.45)['action']=='hold'
