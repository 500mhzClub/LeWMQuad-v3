"""Original independent reactive collector/audit with the six-action selector."""
from types import SimpleNamespace

from lewm.six_action_reactive_controller_development import SixActionReactiveController
from scripts import stop_conditioned_independent_maze_pipeline_development as original

bind = original.bind
previous = original.comparators.previous
CONTROLLERS = {'reactive': SixActionReactiveController}


def functions(mode):
    if mode != 'reactive': raise ValueError('model-free reactive pipeline required')
    collect, audit = previous.functions(mode)
    scene = dict(specification=original.layouts.specification, public_mission=original.layouts.public_mission)
    return bind(collect, **scene), bind(audit, **scene, evaluate=original.evaluate)


execute = bind(previous._execute, CONTROLLERS=CONTROLLERS, functions=functions,
    guarded=SimpleNamespace(CheckedController=original.guarded.CheckedController,
        session_type=original.session_type))
collect = bind(previous.collect, _execute=execute)
audit = bind(original.audit, _audit=bind(previous.audit, _execute=execute))
artifacts = previous.artifacts
resource_artifacts = original.resource_artifacts
