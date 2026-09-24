"""Receding execution: reassess the unchanged 500-ms forecast every 100 ms."""
from lewm.overlap_retention_goal_probe_development import OverlapRetentionGoalProbe
from lewm.matched_model_goal_probe_development import WARMUP_TICKS


class ObservationReplanGoalProbe(OverlapRetentionGoalProbe):
    def advance(self,policy,evidence,*,now_ns):
        # The inherited advance still admits the entire actual observation,
        # performs arrival/time checks and handles every failure. A live plan
        # is consumed for only its first command, then discarded at the next
        # observation. Forecast timing and the prospective action bank do not
        # change, and no unexecuted endpoint is treated as a measured state.
        if self.terminal is None and self.tick>=WARMUP_TICKS:
            self.action=None
            self.plan_offset=0
        return super().advance(policy,evidence,now_ns=now_ns)

    def _result(self,command,selection,distance):
        return super()._result(command,selection,distance)|dict(
            controller='observation_replan_goal_probe_v1',
            maximum_open_loop_command_ticks=1,model_forecast_horizon_ns=500_000_000,
            receding_execution_endpoint_is_not_full_forecast_endpoint=True)
