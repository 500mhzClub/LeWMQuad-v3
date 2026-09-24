import ast
from pathlib import Path
import pytest
from lewm.matched_model_waypoint_selection_development import MatchedModelWaypointSelector
from lewm.matched_model_goal_probe_development import MatchedModelGoalProbe


@pytest.mark.parametrize('condition',['direct','supervised_rollout','jepa'])
@pytest.mark.parametrize('variant',['full','no_rgb'])
def test_assigned_head_and_input_treatment_survive_sensor_failure(condition,variant):
    controller=MatchedModelGoalProbe(object(),object(),condition=condition,variant=variant,persistent=True)
    assert controller.selector.head==('direct_outcomes' if condition=='direct' else 'rollout_outcomes')
    row=controller.observe({}, {}, {}, now_ns=1)
    assert (row['model_condition'],row['input_variant'])==(condition,variant)
    assert row['terminal']=='SENSOR_OR_MODEL_FAILURE' and row['requested_command']==[0.,0.,0.]
    assert row['goal_initial_body_xy_m']==[1.2,0.]


def test_no_implicit_condition_or_variant_fallback():
    with pytest.raises(ValueError):MatchedModelWaypointSelector(condition='best',variant='full')
    with pytest.raises(ValueError):MatchedModelWaypointSelector(condition='jepa',variant='latest_packet_only')


def function(module,name):
    tree=ast.parse(Path(module.__file__).read_text())
    return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)


def test_only_trained_head_and_variant_change_selection_logic():
    from lewm import matched_model_waypoint_selection_development as new
    from lewm import active_view_waypoint_selection_development as old
    for name in ('scan_rank','restrict'):
        assert ast.dump(function(new,name))==ast.dump(function(old,name))
    class Normalize(ast.NodeTransformer):
        def visit_Attribute(self,node):
            if isinstance(node.value,ast.Name) and node.value.id=='self' and node.attr in ('head','variant'):
                return ast.Constant(value='rollout_outcomes' if node.attr=='head' else 'full')
            return self.generic_visit(node)
    assert ast.dump(Normalize().visit(function(new,'choose')))==ast.dump(function(old,'choose'))


def test_native_goal_and_actuator_audits_remain_identical():
    from scripts import matched_model_goal_audit_development as new
    from scripts import active_view_goal_audit_development as old
    for name in ('audit_commands','native_goal'):
        assert ast.dump(function(new,name))==ast.dump(function(old,name))
