import numpy as np
import pytest

from lewm.safety.contact_attribution import attribute_contacts
from lewm.safety.contact_hazard_ontology_v1 import DISALLOWED_CONTACT_FORCE_FLOOR_N


TOPOLOGY = dict(robot_link_ids={1, 2}, support_link_ids={2}, ground_link_ids={0})


def packet():
    return {
        'link_a': np.array([[1, 2]]), 'link_b': np.array([[10, 0]]),
        'force_a': np.array([[[0., 0., 0.], [0., 0., 100.]]]),
        'force_b': np.array([[[0., 0., 0.], [0., 0., -100.]]]),
        'position': np.array([[[0.2, 0.1, 0.3], [0., 0., 0.]]]),
        'valid_mask': np.array([[True, True]]),
    }


@pytest.mark.parametrize('reverse', [False, True])
def test_per_contact_force_floor_is_independent_of_other_contact_forces(reverse):
    data = packet()
    if reverse:
        data = {key: value[:, ::-1] for key, value in data.items()}
    rows = attribute_contacts(data, environment_index=0, **TOPOLOGY)
    assert len(rows) == 2
    assert not any(row['disallowed'] for row in rows)
    wall = next(row for row in rows if row['environment_link_id'] == 10)
    assert wall['force_magnitude_n'] == 0
    assert wall['force_status'] == 'measured'


def test_unbatched_and_explicit_one_environment_agree():
    data = packet()
    batched = attribute_contacts(data, environment_index=0, **TOPOLOGY)
    unbatched = attribute_contacts({key: value[0] for key, value in data.items()}, **TOPOLOGY)
    for a, b in zip(batched, unbatched, strict=True):
        assert a.pop('environment_index') == 0
        assert b.pop('environment_index') is None
        assert a == b


def test_explicit_environment_selection_does_not_mix_contacts():
    data = {key: np.repeat(value, 2, axis=0) for key, value in packet().items()}
    data['force_a'][1, 0] = [0, 2, 0]
    assert not any(row['disallowed'] for row in attribute_contacts(data, environment_index=0, **TOPOLOGY))
    rows = attribute_contacts(data, environment_index=1, **TOPOLOGY)
    assert rows[0]['disallowed'] and rows[0]['force_magnitude_n'] == 2


def test_robot_on_side_b_uses_force_b_and_retains_attribution():
    data = packet()
    data['link_a'][0, 0], data['link_b'][0, 0] = 10, 1
    data['force_a'][0, 0], data['force_b'][0, 0] = [99, 0, 0], [0, -2, 0]
    row = attribute_contacts(data, environment_index=0, link_names={1: 'body', 10: 'wall_link'},
                             environment_object_ids={10: 'left_wall'}, **TOPOLOGY)[0]
    assert row['force_on_robot_world_n'] == [0, -2, 0]
    assert row['force_magnitude_n'] == 2 and row['disallowed']
    assert row['robot_link_name'] == 'body'
    assert row['environment_object_id'] == 'left_wall'
    assert row['position_world_m'] == [0.2, 0.1, 0.3]


def test_invalid_padding_is_not_a_contact_or_nonfinite_error():
    data = packet()
    data['valid_mask'][0, 0] = False
    data['link_a'][0, 0] = -1
    data['force_a'][0, 0] = np.nan
    rows = attribute_contacts(data, environment_index=0, **TOPOLOGY)
    assert len(rows) == 1 and rows[0]['contact_index'] == 1


def test_missing_force_is_explicit_not_fabricated_measurement():
    data = packet()
    del data['force_a']
    row = attribute_contacts(data, environment_index=0, **TOPOLOGY)[0]
    assert row['force_status'] == 'unavailable'
    assert row['force_magnitude_n'] is None and row['disallowed']
    assert row['environment_object_id'] is None


def test_self_contact_does_not_become_external_contact():
    data = packet()
    data['link_b'][0, 0] = 2
    rows = attribute_contacts(data, environment_index=0, **TOPOLOGY)
    assert len(rows) == 1 and rows[0]['environment_link_id'] == 0


@pytest.mark.parametrize('force, expected', [(DISALLOWED_CONTACT_FORCE_FLOOR_N, False),
                                           (DISALLOWED_CONTACT_FORCE_FLOOR_N + 1e-6, True)])
def test_exact_force_floor_contract(force, expected):
    data = packet()
    data['force_a'][0, 0] = [force, 0, 0]
    assert attribute_contacts(data, environment_index=0, **TOPOLOGY)[0]['disallowed'] is expected


@pytest.mark.parametrize('index', [None, True, -1, 1, 0.0])
def test_batched_environment_must_be_explicit_and_valid(index):
    with pytest.raises(ValueError, match='environment index'):
        attribute_contacts(packet(), environment_index=index, **TOPOLOGY)


@pytest.mark.parametrize('mutation', ['missing_mask', 'integer_mask', 'wrong_force_axis',
                                    'nonfinite_force', 'floating_link_id'])
def test_malformed_or_unresolved_valid_evidence_rejected(mutation):
    data = packet()
    if mutation == 'missing_mask':
        del data['valid_mask']
    elif mutation == 'integer_mask':
        data['valid_mask'] = data['valid_mask'].astype(int)
    elif mutation == 'wrong_force_axis':
        data['force_a'] = data['force_a'][0]
    elif mutation == 'nonfinite_force':
        data['force_a'][0, 0] = np.nan
    elif mutation == 'floating_link_id':
        data['link_a'] = data['link_a'].astype(float)
    with pytest.raises(ValueError):
        attribute_contacts(data, environment_index=0, **TOPOLOGY)


def test_unbatched_environment_axis_cannot_be_invented():
    with pytest.raises(ValueError, match='no environment axis'):
        attribute_contacts({key: value[0] for key, value in packet().items()}, environment_index=0, **TOPOLOGY)


def test_empty_packet_and_contradictory_topology():
    assert attribute_contacts({}, **TOPOLOGY) == []
    with pytest.raises(ValueError, match='topology'):
        attribute_contacts(packet(), environment_index=0, robot_link_ids={0, 1},
                           support_link_ids={1}, ground_link_ids={0})
