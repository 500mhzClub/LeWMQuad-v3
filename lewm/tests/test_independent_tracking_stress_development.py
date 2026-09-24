"""Actual frozen observers on synthetic packets; no native challenge evidence."""
from copy import deepcopy

import numpy as np
import pytest

from lewm.causal_depth_observation_development import validate_depth, rgb_digest
from lewm.fast_gyro_development import FastRelativeOrientation, validate_fast_packet
from lewm.independent_tracking_stress_development import (
    ARMS, SCENARIOS, ONSET_FRAME, INTERVAL_NS, PairedStressObserver,
    definition, identity, packet_digest, transform)
from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion, REFERENCE_RULES
from lewm.temporal_anchor_continuity_development import TemporalAnchorVisualLedMotion, CONTINUITY_RULES
from lewm.tests.test_rgbd_correspondence_motion_development import packets, texture


@pytest.fixture(scope='module')
def single():
    return next(packets([texture()]))


def test_fixed_definition_returns_private_copy_and_preserves_production_rules():
    assert ONSET_FRAME == 84 and len(SCENARIOS) == 11
    assert CONTINUITY_RULES['maximum_bridge_frames'] == 10
    assert REFERENCE_RULES['maximum_candidate_disagreement_m'] == .02
    before = identity(); d = definition(); d['onset_frame'] = 0; d['arms'].clear()
    assert identity() == before and definition()['arms'] == list(ARMS)


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_preonset_packets_are_identical_and_never_alias(single, scenario):
    before = packet_digest(single)
    modified, evidence = transform(single,scenario,83,single[3]-83*INTERVAL_NS)
    assert packet_digest(modified) == before and not evidence['onset_reached']
    modified[0]['image']['rgb'][:] = 0
    modified[1]['depth_m'][:] = 0
    modified[2]['values'][:] = 1.
    assert packet_digest(single) == before


@pytest.mark.parametrize('scenario', ['rgb_unavailable_1','depth_unavailable_1','gyro_unavailable_1',
                                    'repeated_rgb','depth_drift','shared_gyro_bias'])
def test_packet_fault_has_explicit_changed_binding_without_touching_source(single, scenario):
    before = packet_digest(single)
    modified, evidence = transform(single,scenario,84,single[3]-84*INTERVAL_NS)
    assert packet_digest(modified) != before and packet_digest(single) == before
    assert evidence['packet_changes'] and evidence['onset_reached']
    assert not evidence['noise_distribution_calibrated']
    if scenario == 'rgb_unavailable_1': assert modified[0]['image']['available_ns'] == single[3]+1
    if scenario == 'depth_unavailable_1':
        assert not modified[1]['valid'].any() and not modified[1]['depth_m'].any()
    if scenario == 'repeated_rgb':
        assert modified[1]['rgb_sha256'] == rgb_digest(modified[0])
        np.testing.assert_array_equal(modified[1]['depth_m'], single[1]['depth_m'])


def test_depth_drift_caps_and_preserves_unknowns_and_range(single):
    p,d,f,now = deepcopy(single)
    d['depth_m'][0,0] = 0.; d['valid'][0,0] = False
    d['depth_m'][0,1] = 4.99
    modified,_ = transform((p,d,f,now),'depth_drift',120,now-120*INTERVAL_NS)
    pp,dd,ff,nn = modified
    validate_depth(dd,pp,now_ns=nn)
    assert not dd['valid'][0,:2].any() and not dd['depth_m'][0,:2].any()
    assert dd['depth_m'][1,1] == pytest.approx(2.04)
    assert d['depth_m'][0,1] == np.float32(4.99)


def test_gyro_bias_is_same_measurement_across_both_histories_and_packet_boundary():
    seq = list(packets([texture()]*3)); first = seq[0][3]-83*INTERVAL_NS
    orientation = FastRelativeOrientation(); previous = None
    for frame,item in enumerate(seq,83):
        (p,d,f,now),_ = transform(item,'shared_gyro_bias',frame,first)
        validate_fast_packet(f,p,now_ns=now)
        if previous is None: orientation.begin(p,f,now_ns=now)
        else:
            np.testing.assert_array_equal(previous['values'][-1],f['values'][0])
            orientation.step(p,f,now_ns=now)
        previous = f
    assert f['values'][:,2].min() == .02
    assert orientation.samples_integrated == 100


def test_gyro_outage_persists_only_in_histories_containing_lost_samples():
    seq = list(packets([texture()]*4)); first = seq[0][3]-83*INTERVAL_NS
    seen = {}
    for frame,item in enumerate(seq,83):
        (p,d,f,now),_ = transform(item,'gyro_unavailable_1',frame,first)
        for channel in (p['sensor_state']['sensed']['gyro'],f):
            for ns,valid,values in zip(channel['measured_ns'],channel['valid'],channel['values'],strict=True):
                key = int(ns)
                value = valid.tolist(),values.tolist()
                if key in seen: assert seen[key] == value
                seen[key] = value
        if frame == 84: assert not f['valid'][-1].any()
        if frame == 85: assert not f['valid'][0].any() and f['valid'][-1].all()
        if frame == 86: assert f['valid'].all()  # Slow history can still contain outage.


def test_nominal_hook_matches_original_classes_exactly():
    paired = PairedStressObserver('nominal')
    originals = dict(original=MultiReferenceVisualLedMotion(),temporal_anchor=TemporalAnchorVisualLedMotion())
    for item in packets([texture()]*4):
        result = paired.observe(item)
        assert result['source_packet_sha256'] == result['intervened_packet_sha256']
        for arm,model in originals.items():
            p,d,f,now = deepcopy(item); expected = model.observe(p,d,f,now_ns=now)
            assert result['arms'][arm]['pose'] == expected['current_pose']
            assert result['arms'][arm]['selection'] == expected['reference_selection']
            assert result['arms'][arm]['reference_injections'] == []


@pytest.fixture
def warmed():
    # All scenarios are identical before84. Each test gets a fresh actual
    # warmup; OpenCV keypoints are deliberately not copied/pickled as state.
    paired = PairedStressObserver('nominal')
    remaining = []
    for frame,item in enumerate(packets([texture()]*(ONSET_FRAME+13))):
        if frame < ONSET_FRAME:
            result = paired.observe(item)
            assert all(result['arms'][a]['pose'] is not None for a in ARMS)
        else: remaining.append(item)
    return paired,remaining


def branch(warmed, scenario):
    model = warmed[0]; model.scenario = scenario
    for arm in ARMS: model.models[arm].model.stress_scenario = scenario
    return model


@pytest.mark.parametrize('length', [1,10,11])
def test_actual_measured_bridge_rejoins_or_exhausts_fixed_budget(warmed,length):
    model = branch(warmed,f'anchor_absence_{length}')
    original_refs = tuple(r.frame for r in model.models['temporal_anchor'].model.references)
    for offset,item in enumerate(warmed[1]):
        row = model.observe(item); a,b = (row['arms'][k] for k in ARMS)
        assert a['pose'] is None and a['failure'] is not None
        if offset == 0: assert a['reference_injections']
        if length == 11 and offset >= 10:
            assert b['pose'] is None and b['failure'] is not None
            if offset == 10: assert b['continuity']['status'] == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
        else:
            assert b['pose'] is not None and b['pose']['position_error_bound'] is None
            if offset < length:
                assert b['continuity']['status'] == 'MEASURED_INCREMENT_BRIDGE'
                assert b['continuity']['bridge_frames'] == offset+1
                assert not b['pose']['promoted_keyframe']
                assert tuple(r.frame for r in model.models['temporal_anchor'].model.references) == original_refs
            elif offset == length:
                assert b['continuity']['status'] == 'ANCHOR_MEASUREMENT'
                assert b['continuity']['preceding_bridge_frames'] == length
    assert not row['navigation_qualified']


@pytest.mark.parametrize('scenario', ['rgb_unavailable_1','depth_unavailable_1','gyro_unavailable_1'])
def test_current_measurement_loss_is_terminal_even_after_clean_packets_return(warmed,scenario):
    model = branch(warmed,scenario)
    for offset,item in enumerate(warmed[1][:3]):
        row = model.observe(item)
        for arm in ARMS:
            assert row['arms'][arm]['pose'] is None
            assert row['arms'][arm]['failure'] is not None
            assert row['arms'][arm]['reference_injections'] == []
            assert row['arms'][arm]['observer_update_attempted'] == (offset == 0)


def test_qualified_contradiction_is_explicitly_asymmetric_and_terminal(warmed):
    model = branch(warmed,'anchor_increment_conflict')
    for offset,item in enumerate(warmed[1][:3]):
        row = model.observe(item)
        assert row['arms']['original']['pose'] is not None
        assert row['arms']['original']['reference_injections'] == []
        candidate = row['arms']['temporal_anchor']
        assert candidate['pose'] is None and candidate['failure'] is not None
        if offset == 0:
            assert candidate['reference_injections'][0]['kind'] == 'qualified_increment_position_corrupted'
            assert candidate['continuity']['status'] == 'ANCHOR_INCREMENT_CONFLICT'


def test_depth_bias_can_produce_false_static_motion_without_an_error_bound(warmed):
    model = branch(warmed,'depth_drift')
    for item in warmed[1][:3]: row = model.observe(item)
    pose = row['arms']['temporal_anchor']['pose']
    assert pose is not None and np.linalg.norm(pose['position_initial_body_m']) > .003
    assert pose['position_error_bound'] is None and not row['uncertainty_calibrated']


@pytest.mark.parametrize('scenario', ['repeated_rgb','shared_gyro_bias'])
def test_corrupted_sensor_stream_keeps_complete_rows_without_claiming_success(warmed,scenario):
    model = branch(warmed,scenario)
    failures = {arm:False for arm in ARMS}
    for offset,item in enumerate(warmed[1][:3]):
        row = model.observe(item)
        assert row['frame'] == ONSET_FRAME+offset
        assert row['intervention']['packet_changes']
        for arm in ARMS:
            result = row['arms'][arm]
            assert result['observer_update_attempted'] == (not failures[arm])
            if result['pose'] is None:
                assert result['failure'] is not None
                failures[arm] = True
            else:
                assert not failures[arm]
                assert result['pose']['position_error_bound'] is None
        assert not row['navigation_qualified'] and not row['uncertainty_calibrated']


def test_outage_is_not_reported_applied_after_all_history_samples_expire(single):
    modified,evidence = transform(single,'gyro_unavailable_1',120,single[3]-120*INTERVAL_NS)
    assert packet_digest(modified) == packet_digest(single)
    assert evidence['onset_reached'] and evidence['packet_changes'] == []
    assert evidence['affected_gyro_history_entries'] == 0


def test_returned_rows_do_not_alias_state_or_other_observer_and_bad_clock_latches(single):
    model = PairedStressObserver('nominal'); row = model.observe(single)
    row['arms']['original']['pose']['position_initial_body_m'][0] = 999.
    assert model.models['original'].last_visual['position_initial_body_m'][0] == 0.
    assert row['arms']['temporal_anchor']['pose']['position_initial_body_m'][0] == 0.
    with pytest.raises(ValueError,match='clock'): model.observe(single)
    assert model.failed
    with pytest.raises(ValueError,match='cannot restart'): model.observe(single)


@pytest.mark.parametrize('scenario,frame', [('unknown',0), ('nominal',True), ('nominal',-1)])
def test_invalid_intervention_selection_rejected(single,scenario,frame):
    with pytest.raises(ValueError): transform(single,scenario,frame,single[3])


def test_digest_binds_dtype_shape_metadata_and_values(single):
    before = packet_digest(single)
    for field in ('dtype','shape','metadata','values'):
        item = deepcopy(single)
        if field == 'dtype': item[1]['depth_m'] = item[1]['depth_m'].astype(np.float64)
        if field == 'shape': item[1]['depth_m'] = item[1]['depth_m'].reshape(-1)
        if field == 'metadata': item[1]['calibration_id'] = 'different'
        if field == 'values': item[1]['depth_m'][0,0] += .1
        assert packet_digest(item) != before
