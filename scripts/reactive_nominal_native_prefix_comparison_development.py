"""Exact physical/public evidence up to the first prospectively changed command."""
from itertools import islice
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def compare(prior, current, prefix_report):
    frames = prefix_report['frames']; changed = prefix_report['first_requested_command_difference']
    if type(frames) is not int or type(changed) is not int or changed < 0 or frames != changed+1:
        raise ValueError('bound first-intervention observation prefix required')
    count = 750+50*changed; raw_hashes = []
    for directory in (prior, current):
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as z:
            if any(len(z[k]) < count for k in z.files): raise ValueError('complete physical prefix required')
            raw_hashes.append(fingerprint({k: z[k][:count] for k in z.files}))
    if raw_hashes[0] != raw_hashes[1]: raise ValueError('physics differed before reactive intervention')
    old_tape, new_tape = [read_json(p, 'command_tape.json') for p in (prior, current)]
    if (len(old_tape) <= changed or len(new_tape) <= changed
            or any(old_tape[i]['requested_command'] != new_tape[i]['requested_command'] for i in range(changed))
            or old_tape[changed]['requested_command'] != prefix_report['prior_requested_command']
            or new_tape[changed]['requested_command'] != prefix_report['final_requested_command']):
        raise ValueError('native commands must match the prospectively bound intervention')
    readers = [IntentReturnRGBDReplay(p) for p in (prior, current)]
    acquisitions = [read_json(p, 'auxiliary_camera_audit.json') for p in (prior, current)]
    checked = 0
    for i, (a, b) in enumerate(zip(islice(read_rows(prior), frames), islice(read_rows(current), frames), strict=True)):
        values = []
        for directory, reader, rows in zip((prior, current), readers, acquisitions, strict=True):
            policy, depth, fast, now = reader.packet(i)
            auxiliary = packet(directory, i, policy, public_acquisition(rows[i]), now_ns=now)
            values.append(fingerprint((policy, depth, fast, auxiliary, now)))
        if values[0] != values[1]: raise ValueError('public sensor prefix differs before reactive intervention')
        a, b = a['decision'], b['decision']
        for key in ('evidence', 'memory_receipt', 'mission_receipt', 'auxiliary_floor_partition_receipt'):
            if a[key] != b[key]: raise ValueError('observer/map/mission prefix changed')
        sb = b['new_selection'] or {}
        if b['learned_model_used'] or b['candidate_future_outcomes_evaluated']:
            raise ValueError('reactive baseline may not consume predicted outcomes')
        if 'prediction' in sb or 'nominal_path_checks' in sb:
            raise ValueError('learned forecast gates cannot be relabeled reactive')
        checked += 1
    if checked != frames: raise ValueError('complete prospective observation prefix required')
    return dict(common_prefix_frames=frames, first_changed_command=changed, raw_physics_prefix_sha256=raw_hashes[0],
        physical_and_public_prefix_exact=True, observer_map_and_mission_prefix_exact=True, learned_forecasts_used_by_baseline=False, future_constraint_gates_matched=False,
        prefix_includes_observation_before_intervention=True, unexecuted_outcomes_inferred=False)
