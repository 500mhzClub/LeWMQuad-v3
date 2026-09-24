"""Fresh development geometry and physical execution endpoints, not a benchmark."""
from __future__ import annotations

import math
from collections.abc import Mapping
import numpy as np

KINDS = ('straight', 'offset', 'left90', 'right90')
WIDTHS = (0.75, 1.0)
SEED_BASE = 2026090500


def rotation_xyzw(quaternion):
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1, rtol=0, atol=1e-5):
        raise ValueError('finite unit xyzw quaternion required')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def build_case(kind, width):
    if kind not in KINDS or width not in WIDTHS:
        raise ValueError('case outside the fixed development panel')
    index = KINDS.index(kind) * len(WIDTHS) + WIDTHS.index(width)
    theta = {'straight': 0., 'offset': 0., 'left90': math.pi/2, 'right90': -math.pi/2}[kind]
    lateral = 0.25 if kind == 'offset' else 0.
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    def point(x, y):
        return (rot @ [x, y]).tolist()
    def rectangle(x0, x1, y0, y1):
        return [point(x0,y0), point(x1,y0), point(x1,y1), point(x0,y1)]
    walls = []
    def wall(name, x, y, length, angle):
        walls.append({'wall_id': name, 'centre_xyz': [*point(x,y), 0.3],
                      'size_xyz': [length, 0.08, 0.6], 'yaw_rad': theta + angle,
                      'material_id': 'NEUTRAL_WALL'})
    low, high = lateral-width/2, lateral+width/2
    wall('back', -1.04, 0, 2.4, math.pi/2)
    wall('side_low', -0.2, -1.24, 1.6, 0)
    wall('side_high', -0.2, 1.24, 1.6, 0)
    wall('front_low', 0.64, (-1.2+low)/2, low+1.2, math.pi/2)
    wall('front_high', 0.64, (high+1.2)/2, 1.2-high, math.pi/2)
    wall('corridor_low', 1.2, low-0.04, 1.2, 0)
    wall('corridor_high', 1.2, high+0.04, 1.2, 0)
    scene_id = f'go2-contact-execution-dev-v1-{kind}-width-{int(width*100):03d}'
    geometry = {
        'spawn_se2_world': [0.,0.,0.], 'wall_boxes': walls,
        'source_node': {'node_id': 'source', 'centre_world': point(-0.2,0),
                        'boundary_polygon_world': rectangle(-1,.6,-1.2,1.2)},
        'target_node': {'node_id': 'target', 'centre_world': point(.95,lateral),
                        'boundary_polygon_world': rectangle(.8,1.3,low,high)},
        'selected_directed_edge': {'edge_id': 'selected-edge',
            'source_node_id': 'source', 'target_node_id': 'target',
            'opening_segment_world': [point(.6,low),point(.6,high)],
            'opening_normal_world': point(1,0),
            'edge_region_polygon_world': rectangle(.6,1.8,low,high)},
        'competing_directed_edges': [],
        'teacher_route_polyline_world': [point(0,0),point(.6,lateral),point(1.3,lateral)],
    }
    return {'scene_id': scene_id, 'family': f'CONTACT_EXECUTION_{kind.upper()}',
            'case_index': index, 'kind': kind, 'width_m': width,
            'procedural_seed': SEED_BASE+index, 'geometry': geometry}


def evaluate_execution(spec, arrays, *, stop_reason, crossing):
    """Usable local arrival, not mere visual/position similarity.

    Crossing is supplied by the existing directed physical crossing function.
    Arrays include all recorded settle/teacher/brake samples, with phase 0/1/2.
    No success-rate threshold is inferred from the eight-case development panel.
    """
    poses = np.asarray(arrays['base_pose_world'], dtype=float)
    twists = np.asarray(arrays['base_twist_world'], dtype=float)
    contact = np.asarray(arrays['physics_contact'])
    phase = np.asarray(arrays['phase'])
    if (poses.ndim != 2 or poses.shape[1] != 7 or len(poses) == 0
            or twists.shape != (len(poses),6) or contact.shape != (len(poses),)
            or phase.shape != (len(poses),) or not np.isfinite(poses).all()
            or not np.isfinite(twists).all()):
        raise ValueError('complete finite measured execution arrays required')
    rot = rotation_xyzw(poses[-1,3:7])
    edge = spec['geometry']['selected_directed_edge']
    opening = np.asarray(edge['opening_segment_world'])
    normal = np.asarray(edge['opening_normal_world'])
    tangent = (opening[1]-opening[0]) / np.linalg.norm(opening[1]-opening[0])
    delta = poses[-1,:2] - opening.mean(axis=0)
    yaw = math.atan2(rot[1,0],rot[0,0])
    desired = math.atan2(normal[1],normal[0])
    yaw_error = abs(math.atan2(math.sin(yaw-desired), math.cos(yaw-desired)))
    roll = math.atan2(rot[2,1],rot[2,2])
    pitch = math.asin(float(np.clip(-rot[2,0],-1,1)))
    speed = float(np.linalg.norm(twists[-1,:2]))
    checks = {
        'no_early_physical_stop': stop_reason is None,
        'no_disallowed_contact': not bool(contact.any()),
        'sustained_correct_crossing': bool(isinstance(crossing, Mapping)
            and crossing.get('is_selected_edge') is True
            and crossing.get('sustained_beyond_samples', 0) >= 100
            and crossing.get('normal_dot_displacement_m', 0) > 0),
        'braking_phase_completed': bool(np.count_nonzero(phase == 2) == 250),
        'arrival_beyond_port': float(delta @ normal) >= .02,
        'arrival_lateral_margin': abs(float(delta @ tangent)) <= spec['width_m']/2 - .10,
        'arrival_heading': yaw_error <= .35,
        'arrival_speed': speed <= .10,
        'arrival_angular_speed': abs(float(twists[-1,5])) <= .25,
        'arrival_body_height': float(poses[-1,2]) >= .20,
        'arrival_body_attitude': max(abs(roll),abs(pitch)) <= .50,
    }
    return {'status': 'SUCCESS' if all(checks.values()) else 'PHYSICAL_FAILURE',
            'checks': checks, 'stop_reason': stop_reason, 'crossing': crossing,
            'arrival': {'base_xyz_world_m': poses[-1,:3].tolist(),
                        'heading_error_rad': yaw_error, 'speed_xy_mps': speed,
                        'angular_velocity_world_z_rad_s': float(twists[-1,5]),
                        'distance_beyond_port_m': float(delta @ normal),
                        'lateral_offset_m': float(delta @ tangent)}}
