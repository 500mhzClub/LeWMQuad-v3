import math

import numpy as np
import pytest

from lewm.physical_execution_development import KINDS, WIDTHS, build_case, evaluate_execution, rotation_xyzw


def success_arrays():
    return {'base_pose_world': np.tile([.7,0,.3,0,0,0,1.], (251,1)),
            'base_twist_world': np.zeros((251,6)),
            'physics_contact': np.zeros(251, dtype=np.uint8),
            'phase': np.array([1]+[2]*250, dtype=np.uint8)}


CROSSING = {'is_selected_edge': True, 'sustained_beyond_samples': 100,
            'normal_dot_displacement_m': .001}


def test_fixed_panel_has_eight_unique_fresh_identities_and_seeds():
    cases = [build_case(kind,width) for kind in KINDS for width in WIDTHS]
    assert len({case['scene_id'] for case in cases}) == 8
    assert [case['procedural_seed'] for case in cases] == list(range(2026090500,2026090508))
    assert all(case['scene_id'].startswith('go2-contact-execution-dev-v1-') for case in cases)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('width', WIDTHS)
def test_case_port_width_direction_route_and_wall_geometry(kind,width):
    case = build_case(kind,width)
    geom = case['geometry']
    edge = geom['selected_directed_edge']
    segment = np.asarray(edge['opening_segment_world'])
    normal = np.asarray(edge['opening_normal_world'])
    assert np.linalg.norm(segment[1]-segment[0]) == pytest.approx(width)
    assert np.linalg.norm(normal) == pytest.approx(1)
    assert np.dot(segment[1]-segment[0],normal) == pytest.approx(0, abs=1e-12)
    assert np.allclose(geom['teacher_route_polyline_world'][1],segment.mean(axis=0))
    assert np.dot(np.asarray(geom['spawn_se2_world'])[:2]-segment.mean(axis=0),normal) < 0
    assert len({row['wall_id'] for row in geom['wall_boxes']}) == 7
    assert all(min(row['size_xyz']) > 0 for row in geom['wall_boxes'])


def test_turns_are_mirrored_but_spawn_heading_does_not_rotate_with_room():
    left, right = (build_case(kind,.75)['geometry'] for kind in ('left90','right90'))
    assert np.asarray(left['selected_directed_edge']['opening_segment_world']).mean(axis=0) == pytest.approx([0,.6])
    assert np.asarray(right['selected_directed_edge']['opening_segment_world']).mean(axis=0) == pytest.approx([0,-.6])
    assert left['spawn_se2_world'][2] == right['spawn_se2_world'][2] == 0


def test_rigid_body_rotation_is_proper_and_xyzw_ordered():
    rot = rotation_xyzw([0,0,math.sin(math.pi/4),math.cos(math.pi/4)])
    assert rot @ [1,0,0] == pytest.approx([0,1,0], abs=1e-12)
    assert np.linalg.det(rot) == pytest.approx(1)
    assert rot.T @ rot == pytest.approx(np.eye(3), abs=1e-12)
    with pytest.raises(ValueError):
        rotation_xyzw([0,0,0,2])


def test_valid_execution_endpoint_passes():
    result = evaluate_execution(build_case('straight',.75), success_arrays(), stop_reason=None, crossing=CROSSING)
    assert result['status'] == 'SUCCESS' and all(result['checks'].values())


@pytest.mark.parametrize('failure', ['contact','heading','speed','angular_speed','lateral','height','beyond','braking','attitude'])
def test_arrival_and_entire_trace_negative_controls(failure):
    arrays = success_arrays()
    if failure == 'contact':
        arrays['physics_contact'][0] = 1  # earlier contact cannot be hidden by good endpoint
    elif failure == 'heading':
        arrays['base_pose_world'][-1,3:] = [0,0,math.sin(.3),math.cos(.3)]
    elif failure == 'speed':
        arrays['base_twist_world'][-1,0] = .2
    elif failure == 'angular_speed':
        arrays['base_twist_world'][-1,5] = .3
    elif failure == 'lateral':
        arrays['base_pose_world'][-1,1] = .35
    elif failure == 'height':
        arrays['base_pose_world'][-1,2] = .19
    elif failure == 'beyond':
        arrays['base_pose_world'][-1,0] = .61
    elif failure == 'braking':
        arrays['phase'][-1] = 1
    elif failure == 'attitude':
        arrays['base_pose_world'][-1,3:] = [math.sin(.3),0,0,math.cos(.3)]
    assert evaluate_execution(build_case('straight',.75),arrays,stop_reason=None,crossing=CROSSING)['status'] == 'PHYSICAL_FAILURE'


@pytest.mark.parametrize('crossing', [None, {}, {'is_selected_edge': False},
                                    {**CROSSING,'sustained_beyond_samples':99},
                                    {**CROSSING,'normal_dot_displacement_m':0}])
def test_unverified_or_insufficient_crossing_cannot_pass(crossing):
    assert evaluate_execution(build_case('straight',.75),success_arrays(),stop_reason=None,crossing=crossing)['status'] == 'PHYSICAL_FAILURE'


def test_early_stop_cannot_be_reported_as_success():
    assert evaluate_execution(build_case('straight',.75),success_arrays(),stop_reason='BODY_STABILITY_LIMIT',crossing=CROSSING)['status'] == 'PHYSICAL_FAILURE'


def test_nonfinite_measurement_is_not_a_scientific_negative():
    arrays = success_arrays()
    arrays['base_pose_world'][0,0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        evaluate_execution(build_case('straight',.75),arrays,stop_reason=None,crossing=CROSSING)
