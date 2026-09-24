"""Measured-posture collision support, not a future swept-volume certificate."""
import xml.etree.ElementTree as ET

import numpy as np

from lewm.causal_ground_plane_development import verify_robot_geometry, URDF_SHA256
from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.simulated_body_observation_development import JOINT_NAMES, validate_policy_packet


def origin_transform(element):
    result = np.eye(4)
    if element is None:
        return result
    xyz = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
        raise ValueError('finite three-component URDF origin required')
    result[:3, :3] = (rotation_increment([0, 0, rpy[2]]) @ rotation_increment([0, rpy[1], 0])
                      @ rotation_increment([rpy[0], 0, 0]))
    result[:3, 3] = xyz
    return result


def primitive_support_radius(kind, dimensions, directions):
    """Exact support about primitive center along arbitrary finite directions."""
    size, normals = np.asarray(dimensions, dtype=float), np.asarray(directions, dtype=float)
    expected = {'box': 3, 'cylinder': 2, 'sphere': 1}
    if (kind not in expected or size.shape != (expected[kind],) or np.any(size <= 0)
            or not np.isfinite(size).all() or normals.ndim != 2 or normals.shape[1:] != (3,)
            or not len(normals) or not np.isfinite(normals).all()
            or np.any(np.linalg.norm(normals, axis=1) <= 1e-12)):
        raise ValueError('positive primitive dimensions and nonzero directions required')
    if kind == 'box':
        return np.abs(normals) @ (size / 2)
    if kind == 'sphere':
        return size[0] * np.linalg.norm(normals, axis=1)
    radius, length = size
    return radius * np.linalg.norm(normals[:, :2], axis=1) + length / 2 * np.abs(normals[:, 2])


class ArticulatedCollisionGeometry:
    def __init__(self, urdf_path):
        verify_robot_geometry(urdf_path)
        root = ET.parse(urdf_path).getroot()
        self._links = {e.get('name') for e in root.findall('link')}
        self._joints, self._shapes = [], []
        for node in root.findall('joint'):
            kind = node.get('type')
            if kind not in ('fixed', 'revolute'):
                raise ValueError('undeclared robot joint type')
            axis = np.fromstring(node.find('axis').get('xyz'), sep=' ') if kind == 'revolute' else np.zeros(3)
            if kind == 'revolute' and (axis.shape != (3,) or not np.isclose(np.linalg.norm(axis), 1.)):
                raise ValueError('unit actuated joint axis required')
            self._joints.append({'name': node.get('name'), 'kind': kind, 'parent': node.find('parent').get('link'),
                                 'child': node.find('child').get('link'), 'axis': axis,
                                 'origin': origin_transform(node.find('origin')),
                                 'separate_rigid_group': kind == 'revolute' or node.get('dont_collapse') == 'true'})
        if {j['name'] for j in self._joints if j['kind'] == 'revolute'} != set(JOINT_NAMES):
            raise ValueError('exact ordered robot joint contract required')
        children = {j['child'] for j in self._joints}
        if self._links - children != {'base'} or len(children) != len(self._joints):
            raise ValueError('one base-rooted kinematic tree required')
        for link in root.findall('link'):
            for index, node in enumerate(link.findall('collision')):
                elements = list(node.find('geometry'))
                if len(elements) != 1:
                    raise ValueError('one collision primitive required')
                primitive = elements[0]
                kind = primitive.tag
                if kind == 'box': size = np.fromstring(primitive.get('size'), sep=' ')
                elif kind == 'sphere': size = np.array([float(primitive.get('radius'))])
                elif kind == 'cylinder': size = np.array([float(primitive.get('radius')), float(primitive.get('length'))])
                else: raise ValueError('unsupported collision geometry; do not substitute a mesh bounding guess')
                primitive_support_radius(kind, size, np.eye(3))
                self._shapes.append({'shape_id': f"{link.get('name')}:{index}", 'link': link.get('name'),
                                     'kind': kind, 'dimensions': size, 'origin': origin_transform(node.find('origin'))})
        if len(self._shapes) != 27:
            raise ValueError('reviewed27-primitive collision population required')

    def transforms(self, joint_position):
        q = np.asarray(joint_position, dtype=float)
        if q.shape != (12,) or not np.isfinite(q).all():
            raise SensorContractError('twelve finite ordered measured joint positions required')
        angles = dict(zip(JOINT_NAMES, q, strict=True))
        transforms, groups = {'base': np.eye(4)}, {'base': 'base'}
        pending = list(self._joints)
        while pending:
            ready = [j for j in pending if j['parent'] in transforms]
            if not ready:
                raise ValueError('disconnected or cyclic robot chain')
            for joint in ready:
                rotation = np.eye(4)
                if joint['kind'] == 'revolute':
                    rotation[:3, :3] = rotation_increment(joint['axis'] * angles[joint['name']])
                transforms[joint['child']] = transforms[joint['parent']] @ joint['origin'] @ rotation
                groups[joint['child']] = joint['child'] if joint['separate_rigid_group'] else groups[joint['parent']]
                pending.remove(joint)
        return transforms, groups

    def supports(self, joint_position, directions_body):
        normals = np.asarray(directions_body, dtype=float)
        transforms, groups = self.transforms(joint_position)
        rows = []
        for shape in self._shapes:
            transform = transforms[shape['link']] @ shape['origin']
            local_normals = normals @ transform[:3, :3]
            radius = primitive_support_radius(shape['kind'], shape['dimensions'], local_normals)
            center = normals @ transform[:3, 3]
            rows.append({'shape_id': shape['shape_id'], 'link': shape['link'], 'urdf_rigid_group': groups[shape['link']],
                         'kind': shape['kind'], 'center_body_m': transform[:3, 3].tolist(),
                         'lower': (center - radius).tolist(), 'upper': (center + radius).tolist()})
        return {'robot_geometry_sha256': URDF_SHA256, 'shapes': rows,
                'directions_body': normals.tolist(), 'lower': np.min([r['lower'] for r in rows], axis=0).tolist(),
                'upper': np.max([r['upper'] for r in rows], axis=0).tolist(),
                'future_swept_volume_qualified': False, 'environment_clearance_qualified': False,
                'scope': 'instantaneous nominal URDF support; no joint uncertainty, future gait or environment observation'}

    def observe(self, packet, *, now_ns):
        validate_policy_packet(packet)
        now = _ns(now_ns, 'body geometry clock')
        joints = packet['sensor_state']['sensed']['joints']
        if (packet['sensor_state']['decision_ns'] != now or packet['image']['measured_ns'] != now
                or joints['measured_ns'][-1] != now or not joints['valid'][-1, :12].all()):
            raise SensorContractError('current RGB/body packet and ordered joint positions required')
        return {'decision_ns': now, **self.supports(joints['values'][-1, :12], np.eye(3))}
