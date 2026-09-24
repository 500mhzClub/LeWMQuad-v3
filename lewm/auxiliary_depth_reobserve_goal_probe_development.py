"""Bounded zero-command reobservation after a valid but infeasible forecast."""
from lewm.auxiliary_depth_goal_probe_development import AuxiliaryDepthGoalProbe

NO_FEASIBLE='NO_PHASE_CANDIDATE_SATISFYING_SURFACE_AND_NOMINAL_CONSTRAINTS'
MAX_CONSECUTIVE_WAIT_COMMANDS=10


class AuxiliaryDepthReobserveGoalProbe(AuxiliaryDepthGoalProbe):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.infeasible_wait_count=0
        self.infeasible_wait_active=False
        self.feasible_action_recoveries=0

    def advance(self,policy,evidence,*,now_ns):
        was_terminal=self.terminal is not None
        self.infeasible_wait_active=False
        result=super().advance(policy,evidence,now_ns=now_ns)
        selection=result['new_selection']
        if (not was_terminal and result['terminal']==NO_FEASIBLE
                and selection is not None and 'prediction' in selection
                and selection['action'] is None and not selection['view_budget_exhausted']):
            if result['requested_command']!=[0.,0.,0.] or self.failure is not None:
                raise ValueError('only valid zero-command infeasibility can enter reobservation')
            self.infeasible_wait_count+=1
            if self.infeasible_wait_count<=MAX_CONSECUTIVE_WAIT_COMMANDS:
                self.terminal=None
                self.infeasible_wait_active=True
        elif not was_terminal and selection is not None and selection.get('action') is not None:
            if self.infeasible_wait_count:self.feasible_action_recoveries+=1
            self.infeasible_wait_count=0
        return self._result(result['requested_command'],selection,result['observed_goal_distance_m'])

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(
            controller='auxiliary_depth_reobserve_goal_probe_v1',
            infeasible_wait_active=self.infeasible_wait_active,
            consecutive_infeasible_observations=self.infeasible_wait_count,
            maximum_consecutive_wait_commands=MAX_CONSECUTIVE_WAIT_COMMANDS,
            feasible_action_recoveries=self.feasible_action_recoveries,
            infeasible_wait_is_clearance_certificate=False)
