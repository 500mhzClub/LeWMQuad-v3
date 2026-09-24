"""Existing independent collector and audit with one contact-score intervention."""
from types import SimpleNamespace

from lewm.commitment_contact_controller_development import CommitmentContactController
from scripts import stop_conditioned_independent_maze_pipeline_development as original

MODE = 'commitment_contact'
bind = original.bind
layouts = original.layouts
previous = original.comparators.previous
CONTROLLERS = {MODE: CommitmentContactController}


def require_mode(mode):
    if mode != MODE:
        raise ValueError('explicit commitment_contact experiment mode required')
    return mode


providers = bind(previous.functions, require_mode=require_mode, CONTROLLERS=CONTROLLERS)


def functions(mode):
    collection, audit_function = providers(mode)
    scene = dict(specification=layouts.specification, public_mission=layouts.public_mission)
    return bind(collection, **scene), bind(audit_function, **scene, evaluate=original.evaluate)


execute = bind(previous._execute, require_mode=require_mode, CONTROLLERS=CONTROLLERS,
    functions=functions, guarded=SimpleNamespace(
        CheckedController=original.guarded.CheckedController, session_type=original.session_type))
collect = bind(previous.collect, _execute=execute)
audit = bind(original.audit, _audit=bind(previous.audit, _execute=execute))
artifacts = bind(previous.artifacts, require_mode=require_mode)
resource_artifacts = original.resource_artifacts
