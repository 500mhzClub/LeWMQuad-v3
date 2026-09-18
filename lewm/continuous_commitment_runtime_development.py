"""400-ms commands with forecast-prefix execution checks and current-depth veto."""
from lewm.independent_depth_process_development import IndependentDepthProcessRuntime
from lewm.continuous_commitment_ledger_development import ContinuousCommitmentLedger


class ContinuousCommitmentRuntime(IndependentDepthProcessRuntime):
    def __init__(self,*args,**kwargs):
        self.commitment_ledger=ContinuousCommitmentLedger()
        super().__init__(*args,commit_ticks=4,**kwargs)
        if self.planning_delay_ticks!=3:raise ValueError('three-tick prefix required')

    def _prefix_commands(self,observed_ns):
        return self.commitment_ledger.prefix_at(observed_ns)

    def _store_plan(self,plan,completed,prefix):
        self.commitment_ledger.commit(plan,completed,prefix)
        self.plans.append(plan)

    def request(self,*,now_ns):
        result=super().request(now_ns=now_ns)
        with self.lock:
            result=self._command_gate(result,now_ns)
            active=[p for p in self.plans if p.dispatch_ns<=now_ns<p.expires_ns]
            plan=active[-1] if active else None
            if plan is not None:
                if not self.commitment_ledger.prefix_was_requested(plan):
                    result=result|dict(requested_command=[0.,0.,0.],
                        reason='COMMITTED_PREFIX_NOT_EXECUTED',command_observation_ns=plan.observed_ns)
                if result['reason']!='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED':
                    self.rejected_windows[plan.observed_ns]=result['reason']
                    self.commitment_ledger.veto(plan,now_ns)
            self.commitment_ledger.record_request(now_ns,result['requested_command'])
        return result

    def _command_gate(self,result,now_ns):
        return result
