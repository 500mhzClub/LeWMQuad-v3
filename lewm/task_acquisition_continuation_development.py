"""Factor stopped arrival observation and evidence-triggered partial scanning."""
import copy

from lewm.continuation_branch_development import choose_side_branch
from lewm.persistent_alignment_continuation_development import PersistentAlignedContinuation
from lewm.stop_observe_traversal_development import StopObserveTraversal


POLICIES = ('baseline', 'arrival_only', 'scan_only', 'both')


class TaskAcquisitionContinuation(PersistentAlignedContinuation):
    def __init__(self, method, geometry, template=None, *, acquisition_policy):
        if acquisition_policy not in POLICIES:
            raise ValueError('explicit fixed evidence-acquisition policy required')
        self.acquisition_policy = acquisition_policy
        super().__init__(method, geometry, template)

    def _traversal(self):
        if self.acquisition_policy in ('arrival_only', 'both'):
            return StopObserveTraversal(self.method, self.geometry, self.template)
        return super()._traversal()

    def _observe(self, packet, fast_packet, *, now_ns):
        result = super()._observe(packet, fast_packet, now_ns=now_ns)
        result['acquisition_policy'] = self.acquisition_policy
        result['scan_stop_evidence'] = None
        scan = result['scan']
        if (self.acquisition_policy in ('scan_only', 'both') and result['stage'] == 'SCAN'
                and self.status == 'RUNNING' and self.stage == 'SCAN'
                and scan['new_completed_view'] is not None):
            current = [row for row in self.observations if row['observed_ns'] == now_ns]
            selected = choose_side_branch(current, self.incoming, now_ns=now_ns)
            if selected is not None:
                self.selected = selected
                self.stage, self.hold_since = 'HOLD_ALIGN', now_ns
                result['next_stage'] = self.stage
                result['requested_command'] = [0., 0., 0.]
                result['selected_side_branch'] = copy.deepcopy(selected)
                result['scan_stop_evidence'] = {
                    'reason': 'FRESH_SIDE_BRANCH_AT_ACQUIRED_VIEW',
                    'decision_ns': now_ns,
                    'observed_ns': selected['observed_ns'],
                    'completed_target_views': scan['completed_target_views'],
                    'full_circle_complete': False}
                # The nested scan remains SCANNING and keeps its proposed
                # command for audit. The outer command cancels further turning.
                # Never relabel this interrupted acquisition as a full scan.
        return result
