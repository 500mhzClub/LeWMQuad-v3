"""Online observation/attempt records that cannot assert a trusted map edge."""
import copy
import math

from lewm.rgb_exit_candidates_development import ExitCandidate


class ProvisionalTraversalLedger:
    def __init__(self):
        self.record = None

    def begin(self, *, observation_id, decision_ns, candidate):
        if self.record is not None or not isinstance(candidate, dict):
            raise ValueError('one unqualified sensor proposal required')
        proposal = ExitCandidate(**candidate)
        if (not isinstance(observation_id, str) or not observation_id
                or type(decision_ns) is not int or decision_ns < 0
                or proposal.timestamp_ns != decision_ns
                or not proposal.observation_id.startswith(observation_id + ':proposal-')):
            raise ValueError('proposal must belong to the current source observation')
        self.record = {'source_observation_id': observation_id, 'started_ns': decision_ns,
                       'exit_proposal': copy.deepcopy(candidate), 'status': 'PENDING',
                       'arrival': None, 'trusted_graph_edges': 0, 'qualified_traversal': False}

    def finish(self, *, status, arrival=None):
        if self.record is None or self.record['status'] != 'PENDING': raise ValueError('pending observation attempt required')
        if status not in ('ARRIVAL_CANDIDATE','FAILED_TIMEOUT','FAILED_NO_VISUAL_CHANGE','FAILED_SETTLING','FAILED_PROGRESS','FAILED_SENSOR','PHYSICAL_STOP'):
            raise ValueError('explicit provisional terminal required')
        if status == 'ARRIVAL_CANDIDATE':
            fields = {'observation_id', 'decision_ns', 'command_progress_proxy_m',
                      'required_body_extent_plus_margin_m', 'floor_mask_change_fraction',
                      'body_quiet_proxy', 'place_identity', 'qualified_arrival'}
            if (not isinstance(arrival, dict) or set(arrival) != fields
                    or arrival['qualified_arrival'] is not False or arrival['place_identity'] is not None
                    or arrival['body_quiet_proxy'] is not True
                    or not isinstance(arrival['observation_id'], str) or not arrival['observation_id']
                    or type(arrival['decision_ns']) is not int
                    or arrival['decision_ns'] <= self.record['started_ns']):
                raise ValueError('fresh, explicitly unqualified sensory arrival required')
            values = [arrival[k] for k in ('command_progress_proxy_m',
                      'required_body_extent_plus_margin_m', 'floor_mask_change_fraction')]
            if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values)
                    or values[0] < values[1] or values[1] < .8 or not .10 <= values[2] <= 1.):
                raise ValueError('finite provisional arrival evidence required')
        elif arrival is not None:
            raise ValueError('failed attempts cannot carry an arrival')
        self.record.update(status=status, arrival=copy.deepcopy(arrival))

    def snapshot(self):
        return copy.deepcopy(self.record)
