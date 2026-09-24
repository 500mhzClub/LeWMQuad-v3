"""Audit-only physical snapshots at the existing paced command boundary.

This reuses the native mutable solver-field inventory, not Scene.get_state().
It does not claim fidelity until recorded 800-ms traces have been reproduced.
Decision packets and pending high-level commands are separate frozen inputs.
"""
import copy
import random

import numpy as np

from scripts import run_go2_oracle_branch_pilot_v1 as native


SESSION_FIELDS = (
    '_dispatch_window', '_dispatch_previous', '_command_history', '_control_history',
    '_last_controller_observation', '_previous_policy_action', '_low_level_policy_state',
    'sensor', 'observations', 'fast_sensor', 'fast_buffer', 'phase', 'edge_index',
    'sample_time',
)
CONTEXT_FIELDS = (
    'ticks_executed', 'policy_steps', 'episode_ticks', 'reset_in_last_block',
    'episode_start_reset_count', 'last_block_executed',
)


def boundary(session):
    runner, policy = session.ctx.runner, session.ctx.policy
    if runner.n_envs != 1 or runner._sim_time_ns % 100_000_000:
        raise ValueError('single-environment decision-time camera boundary required')
    if runner._policy_dt_ns != 20_000_000 or runner._physics_steps_per_policy != 10:
        raise ValueError('unchanged paced policy/physics contract required')
    if policy._last_actions is None or getattr(policy._policy, 'is_recurrent', False):
        raise ValueError('initialized current feed-forward locomotion policy required')
    return int(runner._sim_time_ns)


def capture(session):
    import torch

    stamp = boundary(session)
    ctx = session.ctx
    runner = ctx.runner
    fields = ctx.solver_fields
    if not fields or len({name for name, _ in fields}) != len(fields):
        raise ValueError('nonempty unambiguous native solver inventory required')
    objects = {name:copy.deepcopy(getattr(runner, name)) for name in native._HARNESS_OBJECT_FIELDS}
    objects.update(episode_states=copy.deepcopy(runner.episode_states),
        _sim_time_ns=stamp, _sequence_id_counter=runner._sequence_id_counter)
    return dict(schema='decision_headroom_paced_physical_snapshot.v1',
        measured_ns=stamp, scene_step=int(ctx.build.scene.t),
        solver=native.dump_solver_state(fields),
        runner_arrays={name:np.array(getattr(runner, name), copy=True) for name in native._HARNESS_ARRAY_FIELDS},
        runner_objects=objects,
        context={name:copy.deepcopy(getattr(ctx, name)) for name in CONTEXT_FIELDS},
        session={name:copy.deepcopy(getattr(session, name)) for name in SESSION_FIELDS if hasattr(session, name)},
        policy_last_actions=np.array(ctx.policy._last_actions, copy=True),
        rng=dict(python=random.getstate(), numpy=np.random.get_state(),
            runner=copy.deepcopy(runner._rng.bit_generator.state), spawn=runner._spawn_rng.getstate(),
            torch_cpu=torch.get_rng_state().clone(),
            torch_devices=[s.clone().cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []),
        scope='Full inventoried mutable physics fields and low-level execution state; high-level decision packet stored separately.',
        fidelity_qualified=False)


def restore(session, snapshot):
    import torch

    if snapshot['schema'] != 'decision_headroom_paced_physical_snapshot.v1':
        raise ValueError('unexpected audit snapshot schema')
    if getattr(session, 'physics_clock_callback', None) is not None:
        raise ValueError('cannot rewind a live source/controller clock; finish source collection first')
    ctx = session.ctx
    if set(snapshot['solver']) != {name for name, _ in ctx.solver_fields}:
        raise ValueError('solver inventory differs from captured physical state')
    native.load_solver_state(ctx.solver_fields, snapshot['solver'])
    ctx.build.scene._t = snapshot['scene_step']
    for name, value in snapshot['runner_arrays'].items():
        setattr(ctx.runner, name, np.array(value, copy=True))
    for name, value in snapshot['runner_objects'].items():
        setattr(ctx.runner, name, copy.deepcopy(value))
    for name, value in snapshot['context'].items():
        setattr(ctx, name, copy.deepcopy(value))
    for name in SESSION_FIELDS:
        if name in snapshot['session']:
            setattr(session, name, copy.deepcopy(snapshot['session'][name]))
        elif hasattr(session, name):
            delattr(session, name)
    ctx.policy._last_actions = np.array(snapshot['policy_last_actions'], copy=True)
    # Physics time was rewound; invalidate visual caches before RNG restore.
    ctx.build.scene._visualizer.reset()
    rng = snapshot['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    ctx.runner._rng.bit_generator.state = copy.deepcopy(rng['runner'])
    ctx.runner._spawn_rng.setstate(rng['spawn'])
    torch.set_rng_state(rng['torch_cpu'])
    if len(rng['torch_devices']) != torch.cuda.device_count():
        raise ValueError('device RNG inventory differs from capture')
    if rng['torch_devices']:
        torch.cuda.set_rng_state_all(rng['torch_devices'])
    if boundary(session) != snapshot['measured_ns']:
        raise ValueError('restored timestamp differs from source decision time')
