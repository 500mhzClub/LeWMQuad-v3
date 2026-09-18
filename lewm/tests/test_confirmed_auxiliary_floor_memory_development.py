from copy import deepcopy
import numpy as np
import pytest
from lewm.causal_sensor_state import SensorContractError
from lewm.confirmed_auxiliary_floor_memory_development import ConfirmedAuxiliaryFloorMemory, ConfirmedAuxiliaryFloorMap
from lewm.confirmed_floor_round_trip_controller_development import ConfirmedFloorRoundTripController
from lewm.observed_floor_contact_development import ObservedFloorContactMemory, ObservedFloorContactMap
from lewm.view_reentry_round_trip_controller_development import ViewReentryRoundTripController
from lewm.tests.test_observed_floor_contact_development import Geometry, memory, insert


def setup():
    m, now, witness = memory(ConfirmedAuxiliaryFloorMemory)
    m.confirmation_ns = now; m.confirmation_receipt = {'synthetic': True}
    return m, now, witness


def add(m, witness, *, original_floor, confirmed_floor, point=(.8, 0., -.32)):
    insert(m, 'auxiliary', original_floor, witness, point)
    m.confirmed_auxiliary_partition.insert([point], np.array([confirmed_floor]), witness)


def test_confirmed_foot_floor_preserves_complete_original_failed_check_and_returns():
    m, now, w = setup(); add(m, w, original_floor=False, confirmed_floor=True)
    original = ObservedFloorContactMemory.footprint(m, Geometry(), [0., 0.], 0., now_ns=now)
    counts = deepcopy(m.auxiliary_index.sample_counts); old_partition = m.auxiliary_partition.other_returns
    result = m.footprint(Geometry(), [0., 0.], 0., now_ns=now)
    assert original['possible_intersection'] and not result['possible_intersection']
    assert result['original_auxiliary_floor_contact_check'] == original
    assert result['shapes'] == original['shapes']
    assert result['observed_ground_contacts'][0]['support_status'] == 'UNKNOWN'
    assert not result['ground_support_approved'] and not result['unobserved_space_certified']
    assert m.auxiliary_index.sample_counts == counts and m.auxiliary_partition.other_returns == old_partition
    assert not m.floor_cells


@pytest.mark.parametrize('source', ['primary', 'auxiliary'])
def test_any_remaining_unknown_blocks_even_in_confirmed_floor_voxel(source):
    m, now, w = setup(); add(m, w, original_floor=False, confirmed_floor=True)
    if source == 'auxiliary': add(m, w, original_floor=False, confirmed_floor=False)
    else: insert(m, 'primary', False, w)
    result = m.footprint(Geometry(), [0., 0.], 0., now_ns=now)
    assert result['possible_intersection'] and result[source+'_possible_intersection']
    assert not result['non_floor_or_unknown_contacts_exempted']


@pytest.mark.parametrize('source', ['primary', 'auxiliary'])
def test_nonfoot_contact_is_never_exempted_by_confirmed_plane(source):
    m, now, w = setup()
    if source == 'auxiliary': add(m, w, original_floor=False, confirmed_floor=True)
    else: insert(m, 'primary', True, w)
    result = m.footprint(Geometry('base:0'), [0., 0.], 0., now_ns=now)
    assert result['possible_intersection'] and result[source+'_possible_intersection']
    assert not result['non_foot_contacts_exempted']
    old = result['original_auxiliary_floor_contact_check']
    assert [x for x in result['auxiliary_shapes'] if x['shape_id']=='base:0'] == [x for x in old['auxiliary_shapes'] if x['shape_id']=='base:0']


def test_stale_and_missing_return_evidence_rejected_and_mission_inherited():
    m, now, w = setup()
    with pytest.raises(SensorContractError): m.footprint(Geometry(), [0., 0.], 0., now_ns=now+100_000_000)
    insert(m, 'auxiliary', False, w)
    with pytest.raises(SensorContractError): m.footprint(Geometry(), [0., 0.], 0., now_ns=now)
    assert ConfirmedAuxiliaryFloorMap.observe is ObservedFloorContactMap.observe
    assert ConfirmedFloorRoundTripController.observe is ViewReentryRoundTripController.observe
    assert ConfirmedFloorRoundTripController.advance is ViewReentryRoundTripController.advance
