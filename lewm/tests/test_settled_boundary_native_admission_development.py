from copy import deepcopy
import pytest
from scripts.run_go2_settled_boundary_maze_pilot_v1 import admit_prefix


def prefix():
    return dict(status='SETTLED_BOUNDARY_CONTROLLER_PREFIX_COMPLETE',frames=1867,
        first_mission_behavior_difference=1866,final_requested_command=[0.,0.,0.],
        complete_decisions_exact_outside_declared_mission_fields=True,actual_requested_commands_exact=True,
        stopped_before_later_decisions=True,model_state_unchanged=True,new_native_execution=False,
        navigation_qualified=False,final_mission_receipt=dict(frame=1866,phase='OUTBOUND',hold_required=True,
            terminal=None,failure=None,arrivals=[],quiet_intervals=8,measured_settling_required=True,
            observed_settling=dict(first_quiet_observation_starts_dwell=True)))


def test_exact_nonterminal_delayed_arrival_prefix_is_admitted_without_mutation():
    p=prefix();old=deepcopy(p);admit_prefix(p);assert p==old


@pytest.mark.parametrize('field,value',[('status','RUNNING'),('frames',1866),
    ('first_mission_behavior_difference',100),('final_requested_command',[0.,0.,.45]),
    ('complete_decisions_exact_outside_declared_mission_fields',False),('actual_requested_commands_exact',False),
    ('stopped_before_later_decisions',False),('model_state_unchanged',False),
    ('new_native_execution',True),('navigation_qualified',True)])
def test_incomplete_or_changed_scientific_prefix_rejected(field,value):
    p=prefix();p[field]=value
    with pytest.raises(ValueError):admit_prefix(p)


@pytest.mark.parametrize('field,value',[('phase','RETURN'),('terminal','failed'),
    ('arrivals',[{}]),('quiet_intervals',10),('measured_settling_required',False)])
def test_arrival_must_still_be_delayed_at_bound_intervention(field,value):
    p=prefix();p['final_mission_receipt'][field]=value
    with pytest.raises(ValueError):admit_prefix(p)
