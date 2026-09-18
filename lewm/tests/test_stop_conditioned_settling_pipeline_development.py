import pytest

from scripts import run_go2_stop_conditioned_settling_maze02_v1 as trial
from lewm.stop_conditioned_settling_development import StopConditionedSettlingController

# Globals replaced by the existing collector/auditor binder in this fixture.
ResidualAnchoredContinuationController = None
RendererWitnessDualCameraMazeSession = None
audit_sensors = None


@pytest.mark.parametrize('phase', ['collection', 'audit'])
def test_both_execution_paths_install_same_new_controller(monkeypatch, tmp_path, phase):
    stages = []
    class Guard:
        def __init__(self, *args):
            pass
        def check(self, stage, *args):
            stages.append(stage)
            return {}
        def finish(self, error):
            assert error is None
    monkeypatch.setattr(trial.guarded.resources, 'ResourceGuard', Guard)
    monkeypatch.setattr(trial.guarded.resources, 'admission', lambda sample: None)
    # The real function binder replaces the same collector/auditor global used
    # by physical execution, without starting Genesis in a component test.
    def body(*args, **kwargs):
        return ResidualAnchoredContinuationController(None, None,
            public_mission=dict(goal_initial_body_xy_m=[.2, 0.],
                return_initial_body_xy_m=[0., 0.], require_return_after_goal=True),
            navigation_ticks=8000, condition='direct', variant='no_rgb', persistent=True)
    monkeypatch.setattr(trial.base, 'collect' if phase == 'collection' else 'audit', body)
    kwargs = dict(episode_name='synthetic')
    kwargs['output' if phase == 'collection' else 'input_root'] = tmp_path
    wrapped = getattr(trial, 'collect' if phase == 'collection' else 'audit')(**kwargs)
    assert isinstance(wrapped.controller, StopConditionedSettlingController)
    assert stages == ['begin', 'completed']
    assert wrapped.controller.mission.navigation_ticks == 8000
