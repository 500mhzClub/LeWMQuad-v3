"""Restore only declared JSON episode identities before unchanged comparison."""
from copy import deepcopy
from lewm.floor_registered_pose_readout_development import json_identity
from lewm.dual_camera_controller_prefix_comparison_development import compare_primary_decision


def restore(decision):
    result = deepcopy(decision)
    result['original_visual_evidence'] = json_identity(result['original_visual_evidence'])
    result['evidence'] = json_identity(result['evidence'])
    result['evidence']['original_visual_evidence'] = json_identity(result['evidence']['original_visual_evidence'])
    return result


def compare_json_primary_decision(original, candidate, policy, image, auxiliary, *, now_ns):
    return compare_primary_decision(restore(original), restore(candidate), policy, image, auxiliary, now_ns=now_ns)
