import numpy as np
import pytest

from scripts import stop_conditioned_independent_maze_pipeline_development as trial
from scripts.novel_maze_round_trip_physical_session_development import NovelMazeBaseSession


@pytest.mark.parametrize('index', [0, 7])
def test_extended_constructor_builds_independent_scene(monkeypatch, tmp_path, index):
    from lewm_genesis import visible_robot_union_rgbd_scene_development as scene
    class SceneBoundary(Exception):
        pass
    calls = []
    def stop_at_scene(pack, **kwargs):
        calls.append((pack, kwargs))
        raise SceneBoundary()
    monkeypatch.setattr(scene, 'build_scene_from_pack', stop_at_scene)
    spec = trial.layouts.specification(index)
    with pytest.raises(SceneBoundary):
        trial.IndependentExtendedSession(spec, tmp_path)
    assert calls[0][0] == trial.layouts.pack(spec)
    assert calls[0][1]['appearance_seed'] == spec['appearance_seed']
    assert calls[0][1]['backend'] == 'cpu'


def test_independent_session_accepts_frame_beyond_old_budget(monkeypatch):
    class AcquisitionBoundary(Exception):
        pass
    def stop_at_acquisition(self):
        raise AcquisitionBoundary()
    monkeypatch.setattr(NovelMazeBaseSession, 'sensor_packets', stop_at_acquisition)
    obj = object.__new__(trial.IndependentExtendedSession)
    obj.samples = range(750+50*5000)
    obj.renderer_witnesses = dict(primary=[], paired=[], failures=[])
    with pytest.raises(AcquisitionBoundary):
        obj.sensor_packets()


@pytest.mark.parametrize('mode', ['frozen_reference', 'nominal', 'reactive', 'current_planning'])
def test_both_paths_use_independent_scene_and_same_evaluator(mode):
    collection, audit = trial.functions(mode)
    for function in (collection, audit):
        assert function.__globals__['specification'] is trial.layouts.specification
        assert function.__globals__['public_mission'] is trial.layouts.public_mission
        assert function.__globals__['NAVIGATION_TICKS'] == 8000
    assert audit.__globals__['evaluate'] is trial.evaluate
    assert trial.execute.__globals__['guarded'].session_type is trial.session_type


def test_long_trace_without_arrivals_remains_a_failure():
    n = 750+50*4004
    pose = np.zeros((n, 7)); pose[:, 0] = -1.3; pose[:, 2] = .3; pose[:, 6] = 1.
    raw = dict(base_pose_world=pose, base_twist_world=np.zeros((n, 6)),
        requested_command=np.zeros((n, 3)), timestamp_s=np.arange(1, n+1)*.002,
        physics_contact=np.zeros(n, bool))
    result = trial.evaluate(raw, dict(arrivals=[], terminal='MISSION_TICK_BUDGET_EXHAUSTED'),
        dict(schedule_terminal='MISSION_TICK_BUDGET_EXHAUSTED', terminal_zero_ticks=10,
            physical_stop=None, acquisition_stop=None), layout_index=0)
    assert not result['native_round_trip_candidate_pass']
    assert not result['verified_round_trip']
