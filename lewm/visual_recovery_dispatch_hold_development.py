"""Cancel pre-recovery commitments when a measured weak view is published."""
from lewm.persistent_local_visual_recovery_development import PersistentLocalVisualRuntime

REASON = 'WAITING_FOR_POST_VISUAL_RECOVERY_PLAN'


class PublishingSupportRegistration:
    def __init__(self, original, publish):
        self.original = original
        self.publish = publish

    def observe(self, *args, **kwargs):
        evidence = self.original.observe(*args, **kwargs)
        receipt = evidence.get('visual_support')
        if evidence.get('current_pose') is not None and receipt is not None:
            self.publish(receipt)
        return evidence


class VisualRecoveryDispatchHoldRuntime(PersistentLocalVisualRuntime):
    def __init__(self, *args, **kwargs):
        self.visual_plan_minimum_ns = -1
        self.visual_dispatch_events = []
        super().__init__(*args, **kwargs)
        self.registration = PublishingSupportRegistration(
            self.registration, self._publish_visual_recovery)

    def _publish_visual_recovery(self, receipt):
        active = receipt.get('recovery_state_at_observation')
        if active is None:
            return
        trigger = active['trigger_ns']
        if not receipt.get('camera_cadence_recovery') or trigger > receipt['measured_ns']:
            raise ValueError('camera-time recovery evidence required')
        with self.lock:
            if trigger <= self.visual_plan_minimum_ns:
                return
            published = self.clock_ns()
            if published < receipt['measured_ns']:
                raise ValueError('future recovery evidence forbidden')
            self.visual_plan_minimum_ns = trigger
            cancelled = []
            for plan in self.plans:
                if plan.observed_ns < trigger and plan.expires_ns > published:
                    self.rejected_windows[plan.observed_ns] = REASON
                    self.commitment_ledger.veto(plan, published)
                    cancelled.append(plan.observed_ns)
            self.visual_dispatch_events.append(dict(
                frame=receipt['frame'], measured_ns=receipt['measured_ns'],
                trigger_ns=trigger, published_ns=published,
                selected_features=receipt['selected_features'],
                cancelled_plan_observations_ns=cancelled))

    def _store_plan(self, plan, completed, prefix):
        # The controller already holds its lock. Catch a pre-trigger plan that
        # was still computing when the registration thread published recovery.
        if plan.observed_ns < self.visual_plan_minimum_ns:
            self.planning[-1].update(committed=False, discard_reason=REASON)
            return
        super()._store_plan(plan, completed, prefix)

    def _command_gate(self, result, now_ns):
        # Called inside the commitment ledger's request lock, before it records
        # the actual request. Existing prefix and obstacle checks still apply.
        result = super()._command_gate(result, now_ns)
        if (result['reason'] == 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED'
                and result.get('command_observation_ns', -1) < self.visual_plan_minimum_ns):
            result = result | dict(requested_command=[0., 0., 0.], reason=REASON)
        return result | dict(visual_recovery_plan_minimum_ns=self.visual_plan_minimum_ns)
