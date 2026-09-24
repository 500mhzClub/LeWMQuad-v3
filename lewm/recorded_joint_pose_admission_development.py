"""Restore only JSON's episode-identity tuple encoding, then use live admission."""
from lewm.causal_sensor_state import SensorContractError
from lewm.joint_sensor_anchored_goal_development import current_joint_pose


def recorded_joint_pose(evidence, *, identity, now_ns):
    recorded = evidence.get('identity') if isinstance(evidence, dict) else None
    if (type(recorded) is not list or len(recorded) != 3
            or any(type(v) is not int or v < 0 for v in recorded)):
        raise SensorContractError('three nonnegative JSON integer identity fields required')
    # No timestamp, identity value, pose, witness or scientific gate changes.
    return current_joint_pose(evidence | dict(identity=tuple(recorded)), identity=identity, now_ns=now_ns)
