"""Synthetic route-memory contracts; these tests do not establish physical return."""
from copy import deepcopy
from dataclasses import asdict
import json
import hashlib
import math

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.memory.episodic_route_hypotheses_development import EpisodicRouteHypotheses, current_view
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.rgb_exit_candidates_development import ExitCandidate
from lewm.tests.test_observed_traversal_controller_development import Stream


def frames(count=8):
    stream = Stream()
    result = []
    for tick in range(count):
        packet, _, now = stream.frame(tick)
        attitude = {'decision_ns': now, 'start_ns': 1_600_000_000,
                    'rotation_initial_body_from_current_body': np.eye(3).tolist(),
                    'samples_integrated': tick*50, 'gyro_rate_hz': 500,
                    'integration': 'causal_midpoint', 'hardware_calibrated': False}
        result.append((packet, attitude, now))
    return result


def proposal(now, bearing=0., packet=None):
    rgb = packet['image']['rgb'] if packet is not None else frames(1)[0][0]['image']['rgb']
    image_id = hashlib.sha256(rgb.tobytes()).hexdigest()
    return asdict(ExitCandidate(image_id + ':proposal-0', now, bearing, max(-math.pi, bearing-.1),
                               min(math.pi, bearing+.1), 4, 12, 3))


def start(memory, frame):
    p, a, t = frame
    return memory.start(p, a, now_ns=t)


def begin(memory, frame, *, bearing=0., mode='OUTWARD'):
    p, a, t = frame
    return memory.begin(p, a, proposal(t, bearing, p), now_ns=t, mode=mode)


def finish(memory, frame, status='ARRIVAL_CANDIDATE'):
    p, a, t = frame
    return memory.finish(p, a, now_ns=t, status=status)


def test_same_image_is_an_alternative_not_a_place_merge():
    f = frames(); m = EpisodicRouteHypotheses()
    start(m, f[0]); begin(m, f[0]); finish(m, f[1]); begin(m, f[1]); finish(m, f[2])
    s = m.snapshot()
    assert len(s['visits']) == 3 and s['hypothesized_route_depth'] == 2
    assert len({v['visit_event_id'] for v in s['visits']}) == 3
    association = s['visits'][-1]['association']
    assert len(association['ranked_appearance_alternatives']) == 2
    assert all(a['rgb_block_mean_l1'] == 0 for a in association['ranked_appearance_alternatives'])
    assert association['unknown_retained'] and association['selected_place_identity'] is None
    assert not association['scores_are_probabilities']
    assert all(v['place_identity'] is None and not v['qualified_arrival'] for v in s['visits'])
    assert s['trusted_graph_edges'] == 0


def test_two_outward_two_return_attempts_end_only_in_home_hypothesis():
    f = frames(); m = EpisodicRouteHypotheses()
    start(m, f[0]); begin(m, f[0]); finish(m, f[1])
    begin(m, f[1], bearing=math.pi/2); finish(m, f[2])
    intent = m.return_intent()
    assert intent['target_visit_event_id'] == 'visit-0001'
    assert intent['direction_initial_body'] == pytest.approx([0., -1., 0.], abs=1e-12)
    begin(m, f[2], bearing=-math.pi/2, mode='RETURN'); finish(m, f[3])
    assert m.return_intent()['target_visit_event_id'] == 'visit-0000'
    begin(m, f[3], bearing=math.pi, mode='RETURN'); s = finish(m, f[4])
    assert s['return_intent']['kind'] == 'HOME_CANDIDATE'
    assert not s['mission_complete'] and not s['home_verified'] and s['trusted_graph_edges'] == 0
    assert len(s['visits']) == 5 and len(s['attempts']) == 4
    assert [a['intended_predecessor_event_id'] for a in s['attempts']] == [None, None, 'visit-0001', 'visit-0000']
    assert all(a['observed_target_place_identity'] is None for a in s['attempts'])
    with pytest.raises(SensorContractError): begin(m, f[5], mode='RETURN')


def test_reverse_intent_cannot_execute_without_a_matching_fresh_exit():
    f = frames(); m = EpisodicRouteHypotheses()
    start(m, f[0]); begin(m, f[0]); finish(m, f[1])
    p, a, t = f[2]
    assert m.choose_return(p, a, [], now_ns=t)['kind'] == 'OBSERVE_RETURN_DIRECTION'
    before = m.snapshot()
    with pytest.raises(SensorContractError): begin(m, f[3], mode='RETURN')
    assert m.snapshot() == before
    p, a, t = f[3]
    with pytest.raises(SensorContractError):
        m.choose_return(p, a, [proposal(t-100_000_000, math.pi)], now_ns=t)
    assert m.snapshot() == before
    result = m.choose_return(p, a, [proposal(t, math.pi)], now_ns=t)
    assert result['kind'] == 'RETURN_CANDIDATE' and not result['qualified_traversal']
    assert not result['clearance_qualified']


def test_departure_and_return_bearings_share_relative_gyro_frame():
    f = frames(); m = EpisodicRouteHypotheses()
    f[0][1]['rotation_initial_body_from_current_body'] = rotation_increment([0, 0, math.pi/2]).tolist()
    start(m, f[0]); begin(m, f[0]); finish(m, f[1])
    assert m.return_intent()['direction_initial_body'] == pytest.approx([0, -1, 0], abs=1e-12)
    f[2][1]['rotation_initial_body_from_current_body'] = rotation_increment([0, 0, -math.pi/2]).tolist()
    begin(m, f[2], mode='RETURN')  # Current forward is the observed relative south ray.
    assert m.snapshot()['pending']['intended_predecessor_event_id'] == 'visit-0000'


@pytest.mark.parametrize('status', ['FAILED_EXECUTION', 'PHYSICAL_STOP'])
@pytest.mark.parametrize('mode', ['OUTWARD', 'RETURN'])
def test_failure_preserves_history_and_does_not_pop_or_certify_return(status, mode):
    f = frames(); m = EpisodicRouteHypotheses()
    start(m, f[0]); begin(m, f[0]); finish(m, f[1])
    begin(m, f[2], mode=mode, bearing=math.pi if mode == 'RETURN' else 0.)
    s = finish(m, f[3], status)
    assert s['hypothesized_route_depth'] == 1 and s['phase'] == 'UNCERTAIN_AFTER_FAILURE'
    assert s['attempts'][-1]['status'] == status and len(s['visits']) == 3
    assert s['return_intent']['kind'] == 'UNCERTAIN_AFTER_FAILURE' and s['trusted_graph_edges'] == 0
    with pytest.raises(SensorContractError): begin(m, f[4])


@pytest.mark.parametrize('fault', ['rgb', 'body', 'reference', 'episode', 'privileged', 'attitude_privileged',
                                 'rotation', 'reflection', 'future_image', 'stale_clock'])
def test_bad_or_rewritten_observations_reject_without_mutating_memory(fault):
    f = frames(); m = EpisodicRouteHypotheses(); start(m, f[0])
    p, a, t = deepcopy(f[0])
    if fault == 'rgb': p['image']['rgb'][0, 0, 0] += 1
    if fault == 'body': p['sensor_state']['sensed']['joints']['values'][-1, 0] = .1
    if fault == 'reference': a['start_ns'] -= 1
    if fault == 'episode': p['sensor_state']['identity'] = (0, 0, 1)
    if fault == 'privileged': p['world_pose'] = [0, 0, 0]
    if fault == 'attitude_privileged': a['cell_id'] = 'cell-0'
    if fault == 'rotation': a['rotation_initial_body_from_current_body'][0][0] = float('nan')
    if fault == 'reflection': a['rotation_initial_body_from_current_body'][0][0] = -1.
    if fault == 'future_image': p['image']['available_ns'] += 1
    if fault == 'stale_clock': t -= 1
    before = m.snapshot()
    with pytest.raises(SensorContractError): m.begin(p, a, proposal(t), now_ns=t)
    assert m.snapshot() == before


def test_observation_storage_is_copied_and_snapshot_is_serializable_and_detached():
    f = frames(); m = EpisodicRouteHypotheses(); start(m, f[0]); begin(m, f[0]); finish(m, f[1])
    before = m.snapshot()
    f[0][0]['image']['rgb'][:] = 0
    s = m.snapshot(); s['visits'][0]['place_identity'] = 'invented'
    s['attempts'][0]['departure']['direction_initial_body'][0] = 99
    assert m.snapshot() == before
    json.dumps(before, allow_nan=False)


def test_current_pixels_determine_descriptor_and_hash_not_external_tags():
    p, a, t = frames()[0]
    first = current_view(p, a, now_ns=t)
    assert len(first.descriptor) == 48
    p['image']['rgb'][:120, :160] = [0, 0, 0]
    second = current_view(p, a, now_ns=t)
    assert first.rgb_sha256 != second.rgb_sha256 and first.body_sha256 == second.body_sha256
    assert second.descriptor[:3] == (0., 0., 0.)
    assert second.descriptor[3:] == first.descriptor[3:]


def test_invalid_attempt_lifecycle_is_atomic():
    f = frames(); m = EpisodicRouteHypotheses()
    with pytest.raises(SensorContractError): begin(m, f[0])
    start(m, f[0])
    with pytest.raises(SensorContractError): start(m, f[1])
    with pytest.raises(SensorContractError): finish(m, f[1])
    begin(m, f[0]); before = m.snapshot()
    with pytest.raises(SensorContractError): begin(m, f[1])
    with pytest.raises(SensorContractError): finish(m, f[0])
    with pytest.raises(SensorContractError): finish(m, f[1], 'VERIFIED_ARRIVAL')
    assert m.snapshot() == before


@pytest.mark.parametrize('pending', [False, True])
@pytest.mark.parametrize('status', ['PHYSICAL_STOP', 'FAILED_SENSOR', 'FAILED_EXECUTION'])
def test_abort_latches_fault_without_fabricating_terminal_observation(pending, status):
    f = frames(); m = EpisodicRouteHypotheses(); start(m, f[0])
    if pending: begin(m, f[0])
    s = m.abort(now_ns=f[1][2], status=status)
    assert len(s['visits']) == 1 and len(s['attempts']) == int(pending)
    assert s['phase'] == 'UNCERTAIN_AFTER_FAILURE' and s['pending'] is None
    assert s['faults'][0]['status'] == status
    if pending: assert s['attempts'][0]['terminal_visit_event_id'] is None
    with pytest.raises(SensorContractError): begin(m, f[2])
    with pytest.raises(SensorContractError): m.abort(now_ns=f[2][2], status=status)


def test_unbound_proposal_and_stale_observation_are_rejected():
    f = frames(); m = EpisodicRouteHypotheses(); start(m, f[0])
    p, a, t = f[1]; candidate = proposal(t); candidate['observation_id'] = 'unbound:proposal-0'
    before = m.snapshot()
    with pytest.raises(SensorContractError): m.begin(p, a, candidate, now_ns=t)
    assert m.snapshot() == before
    begin(m, f[1]); finish(m, f[2])
    with pytest.raises(SensorContractError): begin(m, f[1])


def test_acquired_multiview_context_preserves_view_identity_without_place_claims():
    f = frames(); m = EpisodicRouteHypotheses(); start(m, f[0])
    f[1][0]['image']['rgb'][:] = [200, 0, 100]
    p, a, t = f[1]
    m.remember_view(p, a, now_ns=t); before = m.snapshot()
    m.remember_view(p, a, now_ns=t)
    assert m.snapshot() == before
    begin(m, f[1])
    assert len(m.snapshot()['visits'][0]['context_views']) == 1
    f[2][0]['image']['rgb'][:] = [200, 0, 100]
    s = finish(m, f[2])
    alternative = s['visits'][1]['association']['ranked_appearance_alternatives'][0]
    assert alternative['matched_observation_id'] == f'observation-{t}'
    assert alternative['rgb_block_mean_l1'] == 0 and not alternative['same_place_verified']
    assert not s['visits'][0]['context_physical_anchor_verified']


def test_context_and_return_selection_require_idle_active_memory():
    f = frames(); m = EpisodicRouteHypotheses(); p, a, t = f[0]
    with pytest.raises(SensorContractError): m.remember_view(p, a, now_ns=t)
    with pytest.raises(SensorContractError): m.choose_return(p, a, [], now_ns=t)
    start(m, f[0]); begin(m, f[0])
    p, a, t = f[1]
    with pytest.raises(SensorContractError): m.remember_view(p, a, now_ns=t)
    with pytest.raises(SensorContractError): m.choose_return(p, a, [], now_ns=t)
