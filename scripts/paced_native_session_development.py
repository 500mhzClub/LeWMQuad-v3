"""Original independent-maze acquisition with commands serviced every 20 ms."""
import numpy as np

from lewm_genesis.lewm_contract import apply_safety_limits_batch
from scripts.independent_round_trip_session_development import IndependentRoundTripSession


def command_policy_step(session, requested):
    """One gait update and ten physics samples; no camera or high-level wait.

    Preserve the original 100-ms delta limit: every request inside that window
    is bounded relative to its initial command, not a newly replenished budget.
    """
    requested=np.asarray(requested,dtype=np.float32)
    if requested.shape!=(3,) or not np.isfinite(requested).all():
        raise ValueError('finite three-component requested command required')
    runner=session.ctx.runner
    if runner._policy_dt_ns!=20_000_000 or runner._physics_steps_per_policy!=10:
        raise ValueError('original 20-ms gait / 2-ms physics timing required')
    window=runner._sim_time_ns//100_000_000
    if getattr(session,'_dispatch_window',None)!=window:
        session._dispatch_window=window
        session._dispatch_previous=runner._last_executed.copy()
    block,_=apply_safety_limits_batch(requested[None,None,:],
        session._dispatch_previous,runner.safety)
    command=block[0,0]
    observation=runner._build_observation(command[None,:])
    targets=session.ctx.policy.act(observation)
    runner._apply_joint_targets(targets)
    base_time=float(runner._sim_time_ns)/1e9
    base_ns=int(runner._sim_time_ns)
    for step in range(10):
        session.ctx.build.scene.step()
        # A guard may stop inside _sample. Retain the time and command of
        # the physics step that actually happened, including partial ticks.
        runner._sim_time_ns=base_ns+(step+1)*2_000_000
        runner._last_executed=command[None,:].copy()
        session._sample(requested,command,base_time+(step+1)*.002)
        if getattr(session,'physics_clock_callback',None) is not None:
            session.physics_clock_callback(runner._sim_time_ns)
    return command.tolist()


class PacedNativeSession(IndependentRoundTripSession):
    command_policy_step=command_policy_step
