"""Persist actual acquired rows, including empty/partial trials, without overwrite.

Same existing sensor schemas/arrays, new exclusive bounded writer. This does not
reconstruct, fill, or align missing data; mismatched partial lengths remain visible
and disqualify a completed observation tape in the later raw audit.
"""
import numpy as np
from lewm.simulated_body_observation_development import SCHEMAS, CAMERA_CALIBRATION
from lewm.causal_depth_observation_development import SCHEMA, calibration_metadata
from scripts.run_go2_contact_attributed_execution_development_v1 import CONTACT_FIELDS


def stack(rows):
    if not rows: return {}
    keys = set(rows[0])
    if any(set(r) != keys for r in rows):
        raise ValueError('recorded row member drift')
    return {k: np.stack([r[k] for r in rows]) for k in sorted(keys)}


def persist_snapshot(session, store):
    store.npz('physics_trace.npz', stack(session.samples))
    if session.packets:
        sizes = [p['link_a'].shape[1] for p in session.packets]
        contacts = {k: np.concatenate([p[k][0] for p in session.packets], axis=0) for k in CONTACT_FIELDS}
        contacts['frame_offsets'] = np.cumsum([0, *sizes], dtype=np.int64)
        contacts['frame_timestamp_s'] = np.asarray(session.packet_times)
    else:
        contacts = {}  # No fabricated measurements/contacts for a zero-sample failure.
    store.npz('native_contacts.npz', contacts)
    store.json('contact_events.json', session.contact_events)
    store.json('contact_topology.json', dict(
        link_names=session.link_names, environment_object_ids=session.object_ids,
        robot_link_ids=sorted(session._contact_topology['robot']),
        support_link_ids=sorted(session._contact_topology['support']),
        ground_link_ids=sorted(session._contact_topology['ground']),
        native_environment_count=1, selected_environment_index=0,
        packing='public native fields selected at environment 0, concatenated over frames with offsets'))
    for name, rows in (('ideal_sensor_samples.npz', session.sensor_rows),
                       ('policy_histories.npz', session.packet_rows),
                       ('fast_gyro_samples.npz', session.fast_rows),
                       ('fast_gyro_histories.npz', session.fast_packets)):
        store.npz(name, stack(rows))
    store.json('policy_observations.json', dict(
        schema='causal_rgb_body_routes_development.v1', camera_calibration_id=CAMERA_CALIBRATION,
        sensor_assumption='ideal_simulated_body_origin_50hz_zero_latency',
        history_file='policy_histories.npz', frames=session.model_manifest,
        sensor_schemas=[dict(name=s.name, channels=s.channels, units=s.units, role=s.role,
            calibration_id=s.calibration_id, history_length=s.history_length, max_age_ns=s.max_age_ns) for s in SCHEMAS]))
    store.json('camera_audit.json', session.image_audit)
    store.json('depth_observations.json', dict(schema=SCHEMA, calibration=calibration_metadata(), frames=session.depth_manifest))
    store.json('depth_camera_audit.json', session.depth_audit)
