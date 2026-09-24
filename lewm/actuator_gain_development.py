"""Explicit actuator identity intervention for a bounded development study."""
import numpy as np


GAIN_ARMS = ('default', 'checkpoint')


def as_array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def read_gains(robot, indices):
    return {key: as_array(getter(indices)).reshape(-1).astype(float).tolist()
            for key, getter in (('kp', robot.get_dofs_kp), ('kv', robot.get_dofs_kv))}


def configure_gains(robot, indices, env_cfg, arm):
    if arm not in GAIN_ARMS:
        raise ValueError('unknown gain arm')
    if len(indices) != 12 or len(set(indices)) != 12 or any(
            not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in indices):
        raise ValueError('twelve distinct resolved leg DOFs required')
    expected = {'kp': float(env_cfg['kp']), 'kv': float(env_cfg['kd'])}
    if expected != {'kp':20., 'kv':.5}:
        raise ValueError('checkpoint gains differ from fixed study identity')
    before = read_gains(robot, indices)
    if before != {'kp':[100.]*12, 'kv':[10.]*12}:
        raise ValueError('native initial gains differ from measured default identity')
    if arm == 'checkpoint':
        robot.set_dofs_kp([expected['kp']]*12, indices)
        robot.set_dofs_kv([expected['kv']]*12, indices)
    after = read_gains(robot, indices)
    intended = {key:[value]*12 for key,value in expected.items()} if arm == 'checkpoint' else before
    if after != intended:
        raise ValueError('actuator gain readback mismatch')
    return {'arm':arm, 'dof_indices_rollout_order':indices, 'before':before,
            'effective':after, 'expected_checkpoint':expected}
