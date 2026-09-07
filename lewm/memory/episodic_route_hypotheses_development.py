"""RGB/body visit history and tentative backtracking, not a recognized-place graph.

The caller supplies policy-only packets and an uninterrupted causal gyro attitude.
Visit IDs label observation events, never cells. A provisional arrival advances a
route hypothesis, not a verified edge. Return completion remains a HOME_CANDIDATE
for independent physical evaluation. This component neither certifies clearance
nor detects beacons. It must not authorize motion after an executor/sensor fault.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from lewm.causal_sensor_state import SensorContractError, _identity, _ns
from lewm.continuation_branch_development import observed_branch, wrap
from lewm.simulated_body_observation_development import validate_policy_packet


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False, default=_json_value).encode()).hexdigest()


@dataclass(frozen=True)
class VisitView:
    observation_id: str
    timestamp_ns: int
    episode: tuple
    orientation_reference_ns: int
    rgb_sha256: str
    body_sha256: str
    descriptor: tuple
    rotation: tuple


def current_view(packet, attitude, *, now_ns):
    """Derive an immutable view; no supplied identity, descriptor or oracle pose.

    The 4x4 RGB block-mean descriptor is an appearance baseline with no threshold
    or probability calibration. Heading/viewpoint invariance is not claimed.
    The body hash binds the supplied causal histories, including missingness.
    Stream overlap/gyro integration must also be checked by the upstream tracker.
    """
    validate_policy_packet(packet)
    now_ns = _ns(now_ns, 'visit clock')
    if set(attitude) != {'decision_ns', 'start_ns', 'rotation_initial_body_from_current_body',
                         'samples_integrated', 'gyro_rate_hz', 'integration', 'hardware_calibrated'}:
        raise SensorContractError('only causal relative-attitude fields are accepted')
    state = packet['sensor_state']
    if (packet['image']['measured_ns'] != now_ns or state['decision_ns'] != now_ns
            or attitude['decision_ns'] != now_ns):
        raise SensorContractError('current image, body and attitude required')
    reference = _ns(attitude['start_ns'], 'orientation reference')
    if reference > now_ns:
        raise SensorContractError('future orientation reference')
    rotation = np.asarray(attitude['rotation_initial_body_from_current_body'], dtype=float)
    if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
            or not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-8)
            or not np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-8)):
        raise SensorContractError('proper relative orientation required')
    rgb = np.asarray(packet['image']['rgb'])
    descriptor = rgb.reshape(4, 120, 4, 160, 3).mean(axis=(1, 3)).ravel()/255.
    return VisitView(f'observation-{now_ns}', now_ns, _identity(state['identity']), reference,
                     hashlib.sha256(rgb.tobytes()).hexdigest(), _digest(state),
                     tuple(descriptor.tolist()), tuple(rotation.ravel().tolist()))


class EpisodicRouteHypotheses:
    """An outward stack with explicit, unverified predecessor-return intents.

    All prior visit appearances are retained as association alternatives alongside
    UNKNOWN; none is merged. Reverse bearings require fresh exit evidence before
    an attempt. Each return arrival pops only a *hypothesized* route step. Failure
    preserves the attempted movement and terminal view and suspends routing.
    """
    def __init__(self):
        self._views = {}
        self._visits = []
        self._attempts = []
        self._route = []
        self._pending = None
        self._last_ns = -1
        self._reference = None
        self._phase = 'UNSTARTED'
        self._faults = []

    def _check_view(self, packet, attitude, now_ns):
        view = current_view(packet, attitude, now_ns=now_ns)
        reference = (view.episode, view.orientation_reference_ns)
        if self._reference is not None and reference != self._reference:
            raise SensorContractError('episode/reset or orientation reference changed')
        if view.timestamp_ns < self._last_ns:
            raise SensorContractError('stale visit observation')
        prior = self._views.get(view.timestamp_ns)
        if prior is not None and prior != view:
            raise SensorContractError('rewritten observation at an existing timestamp')
        return view

    def _accept(self, view):
        self._views[view.timestamp_ns] = view
        self._last_ns = view.timestamp_ns
        self._reference = (view.episode, view.orientation_reference_ns)

    def _branch(self, view, candidate, now_ns):
        if not candidate['observation_id'].startswith(view.rgb_sha256 + ':proposal-'):
            raise SensorContractError('exit proposal must be bound to the current RGB bytes')
        return observed_branch(candidate, np.array(view.rotation).reshape(3, 3), decision_ns=now_ns)

    def _alternatives(self, view):
        scored = []
        for visit in self._visits:
            # Retain the observed viewpoint responsible for the score. Taking a
            # minimum across views is retrieval, not a likelihood or a merge.
            distance, matched = min((float(np.mean(np.abs(np.array(view.descriptor)-v.descriptor))),
                                     v.observation_id) for v in [visit['view'], *visit['context_views']])
            scored.append({'visit_event_id': visit['visit_event_id'],
                           'rgb_block_mean_l1': distance, 'matched_observation_id': matched,
                           'same_place_verified': False})
        scored.sort(key=lambda row: (row['rgb_block_mean_l1'], row['visit_event_id']))
        return {'ranked_appearance_alternatives': scored, 'unknown_retained': True,
                'selected_place_identity': None, 'scores_are_probabilities': False}

    def _visit(self, view, status):
        visit = {'visit_event_id': f'visit-{len(self._visits):04d}', 'view': view,
                 'context_views': [], 'context_physical_anchor_verified': False,
                 'association': self._alternatives(view), 'arrival_status': status,
                 'place_identity': None, 'qualified_arrival': False}
        self._visits.append(visit)
        return visit['visit_event_id']

    def start(self, packet, attitude, *, now_ns):
        if self._phase != 'UNSTARTED':
            raise SensorContractError('memory is single-episode; create a new instance for reset')
        view = self._check_view(packet, attitude, now_ns)
        self._accept(view)
        self._visit(view, 'INITIAL_OBSERVATION')
        self._phase = 'EXPLORING'
        return self.snapshot()

    def remember_view(self, packet, attitude, *, now_ns):
        """Retain an acquired scan/alignment view under the current visit event.

        This association is temporal context, not proof that scanning preserved
        position. The caller supplies actual selected-view events, not a future
        panorama or views from a predicted destination. Duplicate views at the
        same timestamp are idempotent; rewritten data are rejected.
        """
        if self._pending is not None or self._phase not in ('EXPLORING', 'RETURNING'):
            raise SensorContractError('idle active visit required for acquired context')
        view = self._check_view(packet, attitude, now_ns)
        self._accept(view)
        self._retain_context(view)

    def _retain_context(self, view):
        visit = self._visits[-1]
        known = [visit['view'], *visit['context_views']]
        if all(v.timestamp_ns != view.timestamp_ns for v in known):
            visit['context_views'].append(view)

    def return_intent(self):
        """An observation/alignment target, never authority to translate."""
        if self._pending is not None:
            return {'kind': 'EXECUTING', 'qualified_traversal': False}
        if self._phase in ('UNSTARTED', 'UNCERTAIN_AFTER_FAILURE'):
            return {'kind': self._phase, 'qualified_traversal': False}
        if not self._route:
            return {'kind': 'HOME_CANDIDATE' if self._phase == 'RETURNING' else 'AT_INITIAL_ROUTE_ORIGIN',
                    'target_visit_event_id': self._visits[0]['visit_event_id'],
                    'home_verified': False, 'qualified_traversal': False}
        step = self._route[-1]
        return {'kind': 'OBSERVE_RETURN_DIRECTION', 'outward_attempt_id': step['attempt_id'],
                'target_visit_event_id': step['source_visit_event_id'],
                'direction_initial_body': [-v for v in step['departure']['direction_initial_body']],
                'requires_fresh_exit': True, 'translation_compensated': False,
                'qualified_traversal': False, 'clearance_qualified': False}

    def choose_return(self, packet, attitude, candidates, *, now_ns):
        """Match a current floor-extension bearing to the return hypothesis.

        A fixed 0.35-rad angular gate is an integration design parameter, not
        calibrated association/clearance. No fresh match requests observation.
        """
        if self._pending is not None or self._phase not in ('EXPLORING', 'RETURNING'):
            raise SensorContractError('idle active visit required for return observation')
        view = self._check_view(packet, attitude, now_ns)
        choice = self._return_choice(view, candidates, now_ns)
        self._accept(view)
        self._retain_context(view)
        return choice

    def _return_choice(self, view, candidates, now_ns):
        intent = self.return_intent()
        if intent['kind'] != 'OBSERVE_RETURN_DIRECTION':
            return intent
        branches = [self._branch(view, c, now_ns) for c in candidates]
        target = intent['direction_initial_body']
        heading = math.atan2(target[1], target[0])
        options = []
        for branch in branches:
            d = branch['direction_initial_body']
            error = abs(wrap(math.atan2(d[1], d[0])-heading))
            if error <= .35:
                options.append((error, -branch['candidate']['support_points'],
                                branch['candidate']['observation_id'], branch))
        if not options:
            return deepcopy(intent)
        error, _, _, branch = min(options, key=lambda x: x[:3])
        return {**deepcopy(intent), 'kind': 'RETURN_CANDIDATE', 'candidate': branch['candidate'],
                'angular_error_rad': error, 'observation_id': view.observation_id,
                'decision_ns': now_ns}

    def begin(self, packet, attitude, candidate, *, now_ns, mode='OUTWARD'):
        if (self._pending is not None or self._phase not in ('EXPLORING', 'RETURNING')
                or mode not in ('OUTWARD', 'RETURN') or (mode == 'OUTWARD' and self._phase == 'RETURNING')):
            raise SensorContractError('active idle route and explicit supported attempt mode required')
        view = self._check_view(packet, attitude, now_ns)
        departure = self._branch(view, candidate, now_ns)
        target = None
        if mode == 'RETURN':
            choice = self._return_choice(view, [candidate], now_ns)
            if choice['kind'] != 'RETURN_CANDIDATE':
                raise SensorContractError('fresh observed return bearing must match the current intent')
            target = choice['target_visit_event_id']
        self._accept(view)
        self._retain_context(view)
        self._pending = {'attempt_id': f'attempt-{len(self._attempts):04d}', 'mode': mode,
                         'source_visit_event_id': self._visits[-1]['visit_event_id'],
                         'source_observation': view, 'started_ns': now_ns, 'departure': departure,
                         'intended_predecessor_event_id': target, 'qualified_traversal': False}
        if mode == 'RETURN':
            self._phase = 'RETURNING'
        return self._pending['attempt_id']

    def finish(self, packet, attitude, *, now_ns, status):
        if self._pending is None:
            raise SensorContractError('no active route attempt')
        if status not in ('ARRIVAL_CANDIDATE', 'FAILED_EXECUTION', 'PHYSICAL_STOP'):
            raise SensorContractError('explicit provisional executor terminal status required')
        view = self._check_view(packet, attitude, now_ns)
        if view.timestamp_ns <= self._pending['started_ns']:
            raise SensorContractError('positive-duration terminal observation required')
        self._accept(view)
        terminal_id = self._visit(view, status)
        attempt = {**self._pending, 'finished_ns': now_ns, 'status': status,
                   'terminal_visit_event_id': terminal_id, 'observed_target_place_identity': None}
        self._attempts.append(attempt)
        if status == 'ARRIVAL_CANDIDATE':
            if attempt['mode'] == 'OUTWARD':
                self._route.append(attempt)
            else:
                self._route.pop()
        else:
            self._phase = 'UNCERTAIN_AFTER_FAILURE'
        self._pending = None
        return self.snapshot()

    def abort(self, *, now_ns, status):
        """Seal faults even when no valid terminal packet/attitude can be obtained.

        A scan may fail between traversals: it still suspends the entire route.
        No terminal RGB, new visit or arrival is fabricated on this path.
        """
        now_ns = _ns(now_ns, 'fault clock')
        if (self._phase in ('UNSTARTED', 'UNCERTAIN_AFTER_FAILURE') or now_ns < self._last_ns
                or status not in ('FAILED_SENSOR', 'PHYSICAL_STOP', 'FAILED_EXECUTION')):
            raise SensorContractError('active route and chronological explicit fault required')
        self._faults.append({'decision_ns': now_ns, 'status': status,
                             'attempt_id': self._pending['attempt_id'] if self._pending else None})
        if self._pending is not None:
            self._attempts.append({**self._pending, 'finished_ns': now_ns, 'status': status,
                                   'terminal_visit_event_id': None, 'observed_target_place_identity': None})
        self._pending = None
        self._last_ns = now_ns
        self._phase = 'UNCERTAIN_AFTER_FAILURE'
        return self.snapshot()

    def snapshot(self):
        def serialize(value):
            if isinstance(value, VisitView):
                return asdict(value)
            if isinstance(value, dict):
                return {k: serialize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [serialize(v) for v in value]
            return deepcopy(value)
        return serialize({'phase': self._phase, 'visits': self._visits,
                          'attempts': self._attempts, 'pending': self._pending,
                          'hypothesized_route_depth': len(self._route),
                          'return_intent': self.return_intent(), 'trusted_graph_edges': 0,
                          'faults': self._faults, 'home_verified': False, 'mission_complete': False})
