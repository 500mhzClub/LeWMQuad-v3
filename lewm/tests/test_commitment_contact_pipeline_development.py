from types import SimpleNamespace

import pytest

from scripts import commitment_contact_independent_pipeline_development as pipeline
from lewm.extended_return_budget_comparator_controllers_development import ExtendedReturnBudgetForecastSelector
from lewm.tests.test_commitment_contact_controller_development import scored

# Match the injectable globals present in the real collection/audit modules.
ResidualAnchoredContinuationController = None
RendererWitnessDualCameraMazeSession = None
audit_sensors = None


def exercise_injected_controller(*args, **kwargs):
    # The real dispatcher injects this constructor into collection and replay.
    controller = ResidualAnchoredContinuationController(kwargs['model'], None,
        public_mission=dict(goal_initial_body_xy_m=[.2, 0.], return_initial_body_xy_m=[0., 0.],
            require_return_after_goal=True), navigation_ticks=8000,
        condition='jepa', variant='full', persistent=True)
    choice = controller.selector.choose(None, None, None, None, now_ns=0)
    return dict(action=choice['action'], raw_model_command_replay_pass=True)


@pytest.mark.parametrize('phase', ['collection', 'audit'])
def test_dispatcher_uses_intervention_for_collection_and_replay(monkeypatch, phase):
    pipeline.functions(pipeline.MODE)
    original = scored()
    assert original['action'] != 'forward'
    monkeypatch.setattr(ExtendedReturnBudgetForecastSelector, 'choose', lambda *a, **k: original)
    finished = []

    class Guard:
        def __init__(self, *args): pass
        def check(self, *args): return {}
        def finish(self, error): finished.append(error)

    execute = pipeline.bind(pipeline.execute,
        resources=SimpleNamespace(ResourceGuard=Guard, admission=lambda value: None),
        guarded=SimpleNamespace(CheckedController=lambda c, guard: c, session_type=lambda guard: object),
        functions=lambda mode: (exercise_injected_controller, exercise_injected_controller))
    result = execute(pipeline.MODE, phase, (), dict(model=object(),
        output='unused', input_root='unused', episode_name='unused'))
    assert result['action'] == 'forward' and finished == [None]
