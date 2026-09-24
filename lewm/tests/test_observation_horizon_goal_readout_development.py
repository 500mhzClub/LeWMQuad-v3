import ast
from pathlib import Path
from scripts import read_go2_observation_horizon_goal_probe_v1 as current
from scripts import read_go2_observation_replan_goal_probe_v1 as previous
from lewm.tests.test_augmented_family_switch_goal_readout_development import rows,tape


def test_predecessor_and_comparison_boundary_preserve_actual_prefix():
    assert current.PRIOR==previous.INPUT and current.CASES==previous.CASES
    a,b=rows(8),rows(8);ta,tb=tape(7),tape(7);tb[3]['requested_command']=[.2,0,0]
    assert current.common_prefix_length(a,b,ta,tb)==(4,3,None)
    def body(module,name):
        return ast.dump(next(n for n in ast.parse(Path(module.__file__).read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('summarize','common_prefix_length','compare'):assert body(current,name)==body(previous,name)
