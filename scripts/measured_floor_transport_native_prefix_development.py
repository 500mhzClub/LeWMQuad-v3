"""Compare actual physics, public packets and every bound prospective decision."""
from itertools import islice
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.measured_floor_transport_prefix_development import compare_prior
from scripts.measured_floor_transport_intervention_development import prefix_shape
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def compare(prior, current, prefix_root, prefix_report):
    frames, changed = prefix_shape(prefix_report)
    count = 750+50*changed; hashes = []
    for directory in (prior, current):
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as z:
            if any(len(z[k]) < count for k in z.files): raise ValueError('complete physical prefix required')
            hashes.append(fingerprint({k:z[k][:count] for k in z.files}))
    if hashes[0] != hashes[1]: raise ValueError('physics differs before floor-transport intervention')
    tapes = [read_json(p, 'command_tape.json') for p in (prior, current)]
    if (any(len(t) < frames for t in tapes)
            or any(tapes[0][i]['requested_command'] != tapes[1][i]['requested_command'] for i in range(changed))
            or tapes[1][changed]['requested_command'] != prefix_report['final_requested_command']):
        raise ValueError('exact prior commands and bound intervention command required')
    readers = [IntentReturnRGBDReplay(p) for p in (prior, current)]
    acquisitions = [read_json(p, 'auxiliary_camera_audit.json') for p in (prior, current)]
    checked = 0
    for i, (old, new, bound) in enumerate(zip(islice(read_rows(prior), frames),
            islice(read_rows(current), frames), islice(read_rows(prefix_root), frames), strict=True)):
        if not old['tick'] == new['tick'] == bound['tick'] == i: raise ValueError('ordered complete prefix required')
        public = []
        for directory, reader, rows in zip((prior, current), readers, acquisitions, strict=True):
            p, d, f, now = reader.packet(i)
            image, auxiliary = packet(directory, i, p, public_acquisition(rows[i]), now_ns=now)
            public.append(fingerprint((p, d, f, auxiliary, image, now)))
        if public[0] != public[1]: raise ValueError('paired public input differs before intervention')
        if new['decision'] != bound['decision']: raise ValueError('complete native decision differs from prospective replay')
        if i < changed: compare_prior(old, new['decision'], tapes[0][i]['requested_command'], frame=i)
        elif (bound['preintervention_complete_decision_exact'] is not False
                or new['decision']['original_visual_evidence'] != old['decision']['original_visual_evidence']):
            raise ValueError('first transport must retain the same current raw visual evidence')
        if any(row['decision']['requested_command'] != tape[i]['requested_command']
                for row, tape in zip((old, new), tapes, strict=True)):
            raise ValueError('saved decisions differ from actual tape')
        checked += 1
    if checked != frames: raise ValueError('complete prospective prefix required')
    return dict(common_prefix_frames=frames, first_intervention_frame=changed,
        raw_physics_prefix_sha256=hashes[0], physical_prefix_samples=count,
        physical_and_public_prefix_exact=True, all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,
        observation_before_intervention_included=True, unexecuted_outcomes_inferred=False,
        original_intervention_command=tapes[0][changed]['requested_command'],
        candidate_intervention_command=tapes[1][changed]['requested_command'])
