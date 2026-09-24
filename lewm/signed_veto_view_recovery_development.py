"""Keep the vetoed arc's turn direction when acquiring the next measured view."""
import math


class SignedVetoViewMixin:
    def _translation_veto_recovery(self, now_ns, plan):
        recovery = super()._translation_veto_recovery(now_ns, plan)
        # Straight translations retain the original leftward fallback. The
        # parent calls this hook under its lock, only for a newly vetoed plan.
        direction = -1 if plan.command[2] < 0 else 1
        return recovery | dict(view_angle_rad=direction * math.pi / 4,
            vetoed_command=list(plan.command),
            vetoed_command_observation_ns=plan.observed_ns)
