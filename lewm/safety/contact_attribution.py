"""Prospective per-environment contact attribution for Genesis contact packets.

Caller supplies CPU arrays and explicit scene topology. No simulator is opened,
no links are guessed from names, and no frozen experiment imports this module.
The existing contact ontology supplies the label, with missing force reported
explicitly rather than presented as measured collision evidence.
"""
from __future__ import annotations

from numbers import Integral
from typing import Mapping

import numpy as np

from lewm.safety.contact_hazard_ontology_v1 import is_disallowed_contact


def _link_set(values, name):
    values = tuple(values)
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 0 for v in values):
        raise ValueError(f'{name} needs nonnegative integer link identities')
    return set(map(int, values))


def attribute_contacts(contact_data: Mapping, *, environment_index=None,
                       robot_link_ids, support_link_ids, ground_link_ids,
                       link_names=None, environment_object_ids=None):
    """Return valid external robot contacts with per-pair force and identity.

    Batched arrays [environment, contact, ...] require an explicit environment
    index and validity mask, including the one-environment case. Unbatched
    arrays [contact, ...] require environment_index=None. Invalid rows and self
    contacts never enter the external-contact result. Unknown link/object names
    remain None. A missing force is classified conservatively by the historical
    ontology but marked unavailable, not a measured nonzero force.
    """
    robot = _link_set(robot_link_ids, 'robot')
    support = _link_set(support_link_ids, 'support')
    ground = _link_set(ground_link_ids, 'ground')
    if not robot or not support <= robot or robot & ground:
        raise ValueError('robot/support/ground topology is inconsistent')
    if not isinstance(contact_data, Mapping):
        raise ValueError('contact packet must be a mapping')
    if not contact_data:
        return []
    if not {'link_a', 'link_b'} <= contact_data.keys():
        raise ValueError('both contact-link arrays are required')
    links_a, links_b = (np.asarray(contact_data[name]) for name in ('link_a', 'link_b'))
    if (links_a.shape != links_b.shape or links_a.ndim not in (1, 2)
            or any(value.dtype.kind not in 'iu' for value in (links_a, links_b))):
        raise ValueError('matching integer contact-link arrays required')
    batched = links_a.ndim == 2
    if batched:
        if (isinstance(environment_index, (bool, np.bool_))
                or not isinstance(environment_index, Integral)
                or not 0 <= environment_index < links_a.shape[0]):
            raise ValueError('batched contacts require an explicit valid environment index')
        if 'valid_mask' not in contact_data:
            raise ValueError('batched contacts require an explicit validity mask')
        select = int(environment_index)
    else:
        if environment_index is not None:
            raise ValueError('unbatched contacts have no environment axis')
        select = slice(None)
    valid = np.asarray(contact_data.get('valid_mask', np.ones(links_a.shape, dtype=bool)))
    if valid.shape != links_a.shape or valid.dtype != np.bool_:
        raise ValueError('contact validity must be a matching boolean array')
    vectors = {}
    for name in ('force_a', 'force_b', 'position'):
        if name in contact_data:
            value = np.asarray(contact_data[name], dtype=float)
            if value.shape != links_a.shape + (3,):
                raise ValueError(f'{name} must preserve environment/contact/vector axes')
            vectors[name] = value[select]
    names, objects = dict(link_names or {}), dict(environment_object_ids or {})
    for mapping in (names, objects):
        _link_set(mapping, 'name mapping')
        if any(not isinstance(value, str) or not value for value in mapping.values()):
            raise ValueError('link/object names must be nonempty strings')
    rows = []
    for index, (a_raw, b_raw, active) in enumerate(zip(links_a[select], links_b[select], valid[select], strict=True)):
        if not active:
            continue
        a, b = int(a_raw), int(b_raw)
        if a < 0 or b < 0:
            raise ValueError('valid contact has a negative link identity')
        a_robot, b_robot = a in robot, b in robot
        if a_robot == b_robot:
            continue
        robot_id, other_id = (a, b) if a_robot else (b, a)
        force_key = 'force_a' if a_robot else 'force_b'
        force = vectors[force_key][index] if force_key in vectors else None
        position = vectors['position'][index] if 'position' in vectors else None
        for value in (force, position):
            if value is not None and not np.isfinite(value).all():
                raise ValueError('valid contact contains a nonfinite vector')
        magnitude = float(np.linalg.norm(force)) if force is not None else None
        if magnitude is not None and not np.isfinite(magnitude):
            raise ValueError('contact force norm is nonfinite')
        rows.append({
            'environment_index': int(environment_index) if batched else None,
            'contact_index': index,
            'robot_link_id': robot_id,
            'robot_link_name': names.get(robot_id),
            'environment_link_id': other_id,
            'environment_link_name': names.get(other_id),
            'environment_object_id': objects.get(other_id),
            'force_on_robot_world_n': force.tolist() if force is not None else None,
            'force_magnitude_n': magnitude,
            'force_status': 'measured' if force is not None else 'unavailable',
            'position_world_m': position.tolist() if position is not None else None,
            'disallowed': is_disallowed_contact(
                robot_link_id=robot_id, environment_link_id=other_id,
                foot_link_ids=support, ground_link_ids=ground,
                self_contact=False, force_magnitude_n=magnitude),
        })
    return rows
