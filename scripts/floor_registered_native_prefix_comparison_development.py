"""Bind actual physics/public prefix and candidate decisions to prospective replay."""
from itertools import islice
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def compare(prior, current, prefix_root, prefix_report):
    frames = prefix_report['frames']; changed = prefix_report['first_requested_command_difference']
    if type(frames) is not int or type(changed) is not int or changed < 0 or frames != changed+1:
        raise ValueError('bound first-intervention observation prefix required')
    count = 750+50*changed; hashes = []
    for directory in (prior, current):
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as z:
            if any(len(z[k]) < count for k in z.files): raise ValueError('complete physical prefix required')
            hashes.append(fingerprint({k: z[k][:count] for k in z.files}))
    if hashes[0] != hashes[1]: raise ValueError('physics changed before the registered-pose command')
    a, b = [read_json(p, 'command_tape.json') for p in (prior, current)]
    if (len(a) <= changed or len(b) <= changed
            or any(a[i]['requested_command'] != b[i]['requested_command'] for i in range(changed))
            or a[changed]['requested_command'] != prefix_report['prior_requested_command']
            or b[changed]['requested_command'] != prefix_report['final_requested_command']):
        raise ValueError('actual intervention differs from prospective bound command')
    readers = [IntentReturnRGBDReplay(p) for p in (prior, current)]
    acquisitions = [read_json(p, 'auxiliary_camera_audit.json') for p in (prior, current)]
    checked = 0
    for i, (old, new, bound) in enumerate(zip(islice(read_rows(prior), frames),
            islice(read_rows(current), frames), islice(read_rows(prefix_root), frames), strict=True)):
        public = []
        for directory, reader, rows in zip((prior, current), readers, acquisitions, strict=True):
            p, d, f, now = reader.packet(i)
            auxiliary = packet(directory, i, p, public_acquisition(rows[i]), now_ns=now)
            public.append(fingerprint((p, d, f, auxiliary, now)))
        if public[0] != public[1]: raise ValueError('public observations differ before intervention')
        if new['decision'] != bound['decision']:
            raise ValueError('actual candidate decision differs from complete prospective replay')
        if new['decision']['original_visual_evidence'] != old['decision']['evidence']:
            raise ValueError('raw visual estimator changed before intervention')
        if not old['tick'] == new['tick'] == bound['tick'] == i:
            raise ValueError('complete ordered prefix clocks required')
        checked += 1
    if checked != frames: raise ValueError('complete prospective observation prefix required')
    return dict(common_prefix_frames=frames, first_changed_command=changed, raw_physics_prefix_sha256=hashes[0],
        physical_and_public_prefix_exact=True, raw_visual_evidence_prefix_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,
        corrected_pose_map_contact_residual_mission_is_the_declared_change=True,
        prefix_includes_observation_before_intervention=True, unexecuted_outcomes_inferred=False)
