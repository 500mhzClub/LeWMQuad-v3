"""Updated controllers on the eight fixed independent development mazes.

Reuse the original scene construction, acquisition and raw evaluator. Only
the scene selection, longer budget and shared stopping rule are composed here.
The controller receives the coordinate-only public mission, not the layout.
"""
from types import SimpleNamespace

from lewm import independent_round_trip_layouts_development as layouts
from lewm import independent_round_trip_evaluation_development as evaluation
from scripts.independent_round_trip_session_development import IndependentRoundTripPhysicalInit
from scripts import stop_conditioned_comparator_pipeline_development as comparators

base = comparators.previous.extended
guarded = comparators.previous.guarded
bind = base.bind
evaluate = bind(evaluation.evaluate, MAX_NAVIGATION_TICKS=8000)


class IndependentExtendedSession(base.ExtendedReturnBudgetRendererSession, IndependentRoundTripPhysicalInit):
    """Preserve extended RGB-D acquisition around independent scene initialization."""


def session_type(guard):
    class CheckedSession(IndependentExtendedSession):
        def sensor_packets(self):
            frame = (len(self.samples)-750)//50
            guard.check('before_packet', frame)
            result = super().sensor_packets()
            guard.check('after_packet', frame)
            return result
    return CheckedSession


def functions(mode):
    collection, audit_function = comparators.functions(mode)
    scene = dict(specification=layouts.specification, public_mission=layouts.public_mission)
    return bind(collection, **scene), bind(audit_function, **scene, evaluate=evaluate)


execute = bind(comparators.execute, functions=functions,
    guarded=SimpleNamespace(CheckedController=guarded.CheckedController, session_type=session_type))
collect = bind(comparators.collect, _execute=execute)
_audit = bind(comparators.audit, _execute=execute)


def audit(*args, **kwargs):
    result = _audit(*args, **kwargs)
    return result | dict(independent_layout_development_execution=True,
        reused_development_layout=False, independence_scope='fixed eight-layout development inventory',
        zero_request_boundary_required_before_dwell=True)


artifacts = comparators.artifacts
resource_artifacts = comparators.resource_artifacts
