"""Exact physical/public evidence up to the first prospectively changed command."""
from itertools import islice
from copy import deepcopy
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
    if raw_hashes[0] != raw_hashes[1]: raise ValueError('physics differed before recovery command')
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
        if values[0] != values[1]: raise ValueError('public sensor prefix differs before recovery')
        a, b = a['decision'], b['decision']
        compare_decisions(a, b)
        checked += 1
    if checked != frames: raise ValueError('complete prospective observation prefix required')
    return dict(common_prefix_frames=frames, first_changed_command=changed, raw_physics_prefix_sha256=raw_hashes[0],
        physical_and_public_prefix_exact=True, observer_raw_map_mission_forecast_and_original_constraint_prefix_exact=True,
        declared_auxiliary_floor_confirmation_only=True,
        prefix_includes_observation_before_intervention=True, unexecuted_outcomes_inferred=False)


def compare_decisions(original, revised):
    if set(revised) != set(original) | {'current_primary_floor_confirmation_enabled'}:
        raise ValueError('only declared controller receipt extension allowed')
    if revised['current_primary_floor_confirmation_enabled'] is not True:
        raise ValueError('confirmed floor controller identity required')
    memory = deepcopy(revised['memory_receipt'])
    if memory is not None:
        if 'current_primary_floor_confirmation' not in memory['auxiliary_receipt']:
            raise ValueError('explicit auxiliary classification receipt required')
        memory['auxiliary_receipt'].pop('current_primary_floor_confirmation')
    if original['memory_receipt'] != memory:
        raise ValueError('original raw map receipt changed')
    for key in ('evidence', 'mission_receipt', 'auxiliary_floor_partition_receipt',
                'causal_residual_receipt', 'observed_goal_distance_m'):
        if original[key] != revised[key]:
            raise ValueError('observer/map/mission/residual prefix changed')
    a, b = original['new_selection'], revised['new_selection']
    if (a is None) != (b is None):
        raise ValueError('original selection timing changed')
    if a is None: return
    for key in ('prediction', 'nominal_action_checks', 'nominal_path_checks', 'phase_allowed_actions',
                'proposal', 'waypoint_map_xy_m', 'goal_body_xy_m', 'causal_score_residual_receipt'):
        if a.get(key) != b.get(key):
            raise ValueError('forecast/nominal constraint/selection input changed')
    for old, new in zip(a.get('surface_checks', []), b.get('surface_checks', []), strict=True):
        if new.get('original_auxiliary_floor_contact_check') != old:
            raise ValueError('complete original surface witness changed')
        if new['shapes'] != old['shapes'] or new['primary_possible_intersection'] != old['primary_possible_intersection']:
            raise ValueError('primary footprint changed')
        if new['non_foot_contacts_exempted'] or new['non_floor_or_unknown_contacts_exempted']:
            raise ValueError('non-foot or unknown contact exemption forbidden')
