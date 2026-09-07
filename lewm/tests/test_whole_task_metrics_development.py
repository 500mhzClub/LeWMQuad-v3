"""Fixed population and evaluator rejects invented home/task completion."""
from copy import deepcopy

import numpy as np
import pytest

from lewm.whole_task_metrics_development import marker_centres_occluded, ray_box_entry, reduce_whole_task
from lewm.whole_task_scene_development import trial_specs


def fixture():
    count = 2000
    pose = np.zeros((count, 7)); pose[:, 2] = .32; pose[:, 6] = 1.
    pose[800:1400, 0] = 1.
    raw = {'timestamp_s': .002*np.arange(1, count+1), 'base_pose_world': pose,
           'base_twist_world': np.zeros((count, 6)), 'requested_command': np.zeros((count, 3)),
           'physics_contact': np.zeros(count, dtype=bool)}
    decisions = []
    for i, sample in enumerate((749, 799, 849, 899, 1099, 1749)):
        decisions.append({'decision_ns': int(round(raw['timestamp_s'][sample]*1e9)), 'pre_sample_index': sample,
                          'controller': {'marker': {'newly_discovered': ['marker'] if i == 4 else [],
                                                    'detections': ['marker'] if i == 4 else []}}})
    return raw, decisions


def reduce(raw, decisions, **kwargs):
    options = dict(terminal='HOME_CANDIDATE_ROUTE_HYPOTHESIS', stop_reason=None,
                   sensor_fault=None, initial_marker_occluded=True)
    options.update(kwargs)
    return reduce_whole_task(raw, 749, decisions, **options)


def test_actual_detected_departure_return_not_just_controller_flag():
    raw, decisions = fixture(); result = reduce(raw, decisions)
    assert result['physical_task_success'] and not result['false_home_claim']
    raw['base_pose_world'][1400:, 0] = 1.
    result = reduce(raw, decisions)
    assert not result['physical_task_success'] and result['false_home_claim']


@pytest.mark.parametrize('fault', ['no_detection', 'initial_visible', 'never_departed', 'no_release', 'contact', 'moving', 'unstable'])
def test_each_false_task_success_is_rejected(fault):
    raw, decisions = fixture()
    if fault == 'no_detection': decisions[4]['controller']['marker']['newly_discovered'] = []
    if fault == 'initial_visible': decisions[0]['controller']['marker']['detections'] = ['marker']
    if fault == 'never_departed': raw['base_pose_world'][:, 0] = 0.
    if fault == 'no_release': raw['requested_command'][-1, 0] = .3
    if fault == 'contact': raw['physics_contact'][1000] = True
    if fault == 'moving': raw['base_twist_world'][-1, 0] = .2
    if fault == 'unstable': raw['base_pose_world'][-1, 2] = .1
    assert not reduce(raw, decisions)['physical_task_success']


def test_physical_return_can_be_reported_without_an_unsupported_controller_claim():
    raw, decisions = fixture()
    result = reduce(raw, decisions, terminal='FAILED_LOCAL_TIMEOUT')
    assert result['physical_task_success'] and result['physical_return_without_home_claim']
    assert not result['controller_home_claim']
    assert not reduce(raw, decisions, initial_marker_occluded=False)['physical_task_success']


def test_connected_fresh_paired_layouts_with_physically_occluded_marker_centres():
    specs = trial_specs(); assert len(specs) == 4
    for first, second in zip(specs[::2], specs[1::2]):
        assert first['geometry'] == second['geometry'] and first['procedural_seed'] == second['procedural_seed']
        assert first['memory_arm'] != second['memory_arm']
        layout = first['evaluation_layout']
        reached = {(0, 0)}
        for _ in layout['cells']:
            for a, b in layout['edges']:
                if tuple(a) in reached: reached.add(tuple(b))
                if tuple(b) in reached: reached.add(tuple(a))
        assert reached == {tuple(c) for c in layout['cells']}
        assert marker_centres_occluded(first, [.326, 0., .363])
        assert any(sum(list(cell) in edge for edge in layout['edges']) >= 3 for cell in reached)
        assert set(first['geometry']) == {'spawn_se2_world', 'wall_boxes'}


def test_ray_box_occlusion_is_segment_bounded_and_rotation_aware():
    box = {'centre_xyz': [1., 0., 0.], 'size_xyz': [.2, 1., 1.], 'yaw_rad': 0.}
    assert ray_box_entry([0., 0., 0.], [2., 0., 0.], box) == pytest.approx(.45)
    assert ray_box_entry([0., 0., 0.], [.5, 0., 0.], box) is None
    assert ray_box_entry([0., 2., 0.], [2., 2., 0.], box) is None
    box['yaw_rad'] = np.pi/2
    assert ray_box_entry([0., 0., 0.], [2., 0., 0.], box) == pytest.approx(.25)
