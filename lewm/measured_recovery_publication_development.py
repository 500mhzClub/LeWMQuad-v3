"""Charge recovery-publication service before taking the simulator request lock."""
from lewm.measured_latency_simulation_development import MeasuredLatencyClock
from lewm.visual_recovery_dispatch_hold_development import REASON


class MeasuredRecoveryPublicationMixin:
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

        # MeasuredLatencyClock can wait until main-thread physics advances.
        # The main thread also needs self.lock to request its next command.
        # Never wait for that clock while owning the request lock.
        ready = self.clock_ns()
        base = getattr(self.clock_ns, 'base', self.clock_ns)
        if not isinstance(base, MeasuredLatencyClock):
            raise TypeError('this development publisher requires the measured simulation clock')
        with self.lock:
            if trigger <= self.visual_plan_minimum_ns:
                return
            # This is a non-waiting snapshot, including any clock advancement
            # between becoming ready and acquiring the publication lock.
            with base.condition:
                published = base.ns
            if published < max(ready, receipt['measured_ns']):
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
