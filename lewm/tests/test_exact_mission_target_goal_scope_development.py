import ast
from pathlib import Path
from lewm import exact_mission_target_goal_probe_development as current
from lewm import training_bias_goal_probe_development as prior
from lewm.observed_floor_contact_development import ObservedFloorContactGoalProbe


def test_all_learned_scoring_constraints_and_execution_are_unchanged():
    def body(module,cls):
        tree=ast.parse(Path(module.__file__).read_text())
        node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
        return ast.dump(next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='choose'))
    assert body(current,'ExactMissionTargetEightStepSelector')==body(prior,'TrainingBiasEightStepSelector')
    assert current.ExactMissionTargetGoalProbe.observe is ObservedFloorContactGoalProbe.observe
    assert current.ExactMissionTargetGoalProbe.advance is ObservedFloorContactGoalProbe.advance
    a=current.ExactMissionTargetGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    b=ObservedFloorContactGoalProbe(object(),object(),condition='jepa',variant='full',persistent=True)
    assert type(a.mapper) is type(b.mapper) and type(a.motion) is type(b.motion)
