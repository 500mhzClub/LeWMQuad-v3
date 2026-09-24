import ast
from copy import deepcopy
import inspect
import textwrap

import numpy as np

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.depth_motion_evaluation_development import moving_depth_check, reduce_motion
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth
from scripts import run_go2_moving_rgbd_whole_task_development_v1 as runner
from scripts import run_go2_whole_task_navigation_sampling_correction_development_v1 as original
from scripts.moving_rgbd_session_development import MovingRGBDSession
from scripts.single_sample_rgbd_session_development import SingleSampleRGBDSession


def test_distinct_prospective_launch_schema_is_used():
    assert runner.SCHEMA == 'moving_rgbd_whole_task_development.v1'
    tree = ast.parse(textwrap.dedent(inspect.getsource(runner.main)))
    schemas = [v for n in ast.walk(tree) if isinstance(n, ast.Dict)
               for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and k.value == 'schema']
    assert len(schemas) == 1 and isinstance(schemas[0], ast.Name) and schemas[0].id == 'SCHEMA'


def test_original_collect_and_controller_unchanged_only_acquisition_subclass_changes():
    def syntax(f): return ast.dump(ast.parse(textwrap.dedent(inspect.getsource(f))))
    assert syntax(runner.collect) == syntax(original.collect)
    assert runner.WholeTaskNavigation is original.WholeTaskNavigation
    assert runner.FastGyroSession is MovingRGBDSession
    for name in ('capture_fixed_rgb', 'command_tick', '_sample', 'settle_recorded'):
        assert getattr(MovingRGBDSession, name) is getattr(SingleSampleRGBDSession, name)
    expected = [s for s in original.trial_specs() if s['memory_arm'] == 'episodic']
    assert len(runner.trial_specs()) == 2
    for a, b in zip(runner.trial_specs(), expected, strict=True):
        assert a['scene_id'] != b['scene_id']
        assert {k:v for k,v in a.items() if k != 'scene_id'} == {k:v for k,v in b.items() if k != 'scene_id'}


def test_native_geometric_check_applies_to_all_wall_scenes_without_marker_labels():
    transform = np.array(BODY_FROM_OPTICAL); transform[2, 3] += .32
    for spec in runner.trial_specs():
        ref = expected_optical_depth(spec['geometry']['wall_boxes'], transform)
        native = np.full((480, 640), 200., np.float32)
        native[np.ix_(ref['rows'], ref['columns'])] = np.where(np.isfinite(ref['expected_depth_m']), ref['expected_depth_m'], 200.)
        assert moving_depth_check(native, spec['geometry']['wall_boxes'], transform)['passes']
        assert not moving_depth_check(native+.02, spec['geometry']['wall_boxes'], transform)['passes']


def example():
    poses = np.array([[0., 0., .32, 0., 0., 0., 1.], [.03, .01, .32, 0., 0., 0., 1.]])
    cameras = [{'physical_sample_index': i} for i in range(2)]
    observations = [{'observation_index': 0, 'observer': {'measured_ns': 0, 'motion': None,
        'position_initial_body_m': [0., 0., 0.]}},
        {'observation_index': 1, 'observer': {'measured_ns': 100_000_000,
        'position_initial_body_m': [.03, .01, 0.],
        'motion': {'translation_previous_body_m': [.03, .01, 0.],
                   'observable_projection_previous_body_m': [.03, .01, 0.],
                   'weak_directions_previous_body': [], 'status': 'OBSERVED_TRANSLATION', 'rank': 3}}}]
    return {'base_pose_world': poses}, cameras, observations


def test_moving_state_metric_requires_actual_motion_accuracy_and_unbroken_position():
    raw, cameras, observations = example()
    assert reduce_motion(raw, cameras, observations)['passes_declared_moving_state_check']
    bad = deepcopy(observations)
    bad[1]['observer']['motion']['translation_previous_body_m'][0] += .011
    assert not reduce_motion(raw, cameras, bad)['passes_declared_moving_state_check']
    bad = deepcopy(observations); bad[1]['observer']['position_initial_body_m'] = None
    assert not reduce_motion(raw, cameras, bad)['passes_declared_moving_state_check']


def test_unobserved_forward_motion_is_scored_missing_not_zero_or_success():
    raw, cameras, observations = example()
    motion = observations[1]['observer']['motion']
    motion.update(translation_previous_body_m=None, observable_projection_previous_body_m=[0., .01, 0.],
                  weak_directions_previous_body=[[1., 0., 0.]], status='PARTIALLY_OBSERVED_TRANSLATION', rank=2)
    observations[1]['observer']['position_initial_body_m'] = None
    result = reduce_motion(raw, cameras, observations)
    assert result['fully_observed_intervals'] == 0 and result['maximum_step_error_m'] is None
    assert result['rows'][1]['observable_projection_error_m'] == 0.
    assert not result['passes_declared_moving_state_check']
