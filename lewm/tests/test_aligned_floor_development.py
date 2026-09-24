from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from lewm_genesis.aligned_floor_development import (
    add_aligned_floor, aligned_floor_identity, check_aligned_floor_identity,
)
from lewm.tests.test_single_sample_rgbd_evidence_development import floor_record
from lewm.visual_surface_depth_evaluation_development import check_floor_identity


def identity():
    record = floor_record()
    record.pop('scope')
    record.update(schema='aligned_native_floor_development_v1',
                  collision_visualization=False, appearance_collision=False,
                  appearance_visualization=True,
                  scope='evaluation-only scene identity; never policy input')
    record['visual_position_world_m'][2] = float(np.float32(.005))
    return record


def test_actual_world_alignment_not_nominal_zero_local_vertices():
    result = check_aligned_floor_identity(identity())
    assert result['scene_surface_alignment_verified']
    assert result['maximum_visual_collision_offset_m'] < 1e-9
    assert not any(result[k] for k in ('contact_model_validated', 'physical_clearance_qualified',
                                      'hardware_calibrated', 'navigation_qualified'))
    row = identity()
    row['visual_local_vertices_m'] = np.asarray(row['visual_local_vertices_m'])[row['visual_faces']].reshape(-1, 3).tolist()
    row['visual_faces'] = [[0, 1, 2], [3, 4, 5]]
    assert check_aligned_floor_identity(row) == result


@pytest.mark.parametrize('fault', ['legacy_offset', 'overcorrect', 'tilt', 'physical_shift',
    'collision_normal', 'extra_collision', 'extra_visual', 'no_collision', 'no_visual',
    'nan', 'hole', 'winding', 'extent', 'shape', 'scope', 'extra_field'])
def test_invalid_alignment_rejected(fault):
    row = identity()
    if fault == 'legacy_offset': row['visual_position_world_m'][2] = 0.
    if fault == 'overcorrect': row['visual_position_world_m'][2] = .01
    if fault == 'tilt': row['visual_quaternion_wxyz'][1] = .01
    if fault == 'physical_shift': row['collision_position_world_m'][2] = .005
    if fault == 'collision_normal': row['collision_plane_data'][2] = -1.
    if fault == 'extra_collision': row['appearance_collision'] = True
    if fault == 'extra_visual': row['collision_visualization'] = True
    if fault == 'no_collision': row['collision_enabled'] = False
    if fault == 'no_visual': row['appearance_visualization'] = False
    if fault == 'nan': row['visual_local_vertices_m'][0][0] = np.nan
    if fault == 'hole': row['visual_faces'][1] = row['visual_faces'][0]
    if fault == 'winding': row['visual_faces'][1] = [0, 3, 2]
    if fault == 'extent': row['visual_local_vertices_m'][0][0] = -499.
    if fault == 'shape': row['visual_position_world_m'] = [0., .005]
    if fault == 'scope': row['scope'] = 'policy input'
    if fault == 'extra_field': row['oracle_pose'] = [0., 0., 0.]
    with pytest.raises(ValueError): check_aligned_floor_identity(row)


def test_old_identity_and_checker_remain_incompatible():
    assert check_floor_identity(floor_record())['rendered_floor_below_collision_m'] == .005
    with pytest.raises(ValueError): check_aligned_floor_identity(floor_record())
    with pytest.raises(ValueError): check_floor_identity(identity())


def test_builder_passes_distinct_physical_and_visual_roles():
    calls = []
    def add(morph, **kwargs):
        calls.append((morph, kwargs)); return len(calls)
    scene = SimpleNamespace(add_entity=add)
    gs = SimpleNamespace(morphs=SimpleNamespace(Plane=lambda **kw: kw))
    physical, visual = object(), object()
    assert add_aligned_floor(scene, gs, material=physical, surface=visual) == (1, 2)
    assert calls == [({'visualization': False}, {'material': physical}),
                     ({'pos': (0., 0., .005), 'collision': False}, {'surface': visual})]


def test_readback_checks_geometry_counts_before_reading():
    Plane = type('Plane', (), {})
    obj = SimpleNamespace(morph=Plane(), geoms=[object()], vgeoms=[object()])
    with pytest.raises(ValueError): aligned_floor_identity(obj, deepcopy(obj))
