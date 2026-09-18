"""Restore explicit JSON identity representation before unchanged pose checks."""
from lewm.floor_registered_pose_readout_development import json_identity
from lewm.joint_floor_registered_evidence_development import current_joint_floor_registered_pose


def registered_json_pose(evidence, *, now_ns):
    restored = json_identity(evidence)
    restored['original_visual_evidence'] = json_identity(restored['original_visual_evidence'])
    return current_joint_floor_registered_pose(restored, identity=(0, 0, 0), now_ns=now_ns)
