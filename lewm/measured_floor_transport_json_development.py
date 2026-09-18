"""Restore only explicit JSON identity fields before unchanged typed pose checks."""
from lewm.floor_registered_pose_readout_development import json_identity
from lewm.measured_floor_transport_development import SCHEMA, current_measured_floor_pose


def restore_evidence(evidence):
    e = json_identity(evidence)
    e['original_visual_evidence'] = json_identity(e['original_visual_evidence'])
    if e['schema'] == SCHEMA:
        witness = dict(e['floor_transport']); anchor = json_identity(witness['anchor'])
        anchor['original_visual_evidence'] = json_identity(anchor['original_visual_evidence'])
        witness['anchor'] = anchor; e['floor_transport'] = witness
    return e


def current_json_pose(evidence, *, now_ns):
    return current_measured_floor_pose(restore_evidence(evidence), identity=(0, 0, 0), now_ns=now_ns)
