"""Require the full prospective height intervention and its identical native past."""
from itertools import islice
import numpy as np
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.partial_floor_height_prefix_development import compare_step, MAX_FRAMES, BOUNDARY_FRAME
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.replay_go2_partial_floor_height_prefix_v1 import INPUT, CASE
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

FRAMES = MAX_FRAMES
CHANGED = BOUNDARY_FRAME
FORECASTS = CHANGED-3


def admit_prefix(root, result):
    if (result['status'] != 'PARTIAL_FLOOR_HEIGHT_PREFIX_V1_COMPLETE'
            or result['model_loaded'] is not True or result['model_training'] is not False
            or result['native_execution'] is not False or result['shadow_replay_only'] is not True):
        raise ValueError('completed fresh partial-height replay required')
    report = result['report']
    expected = dict(case=CASE[0], layout_index=1, frames=FRAMES, maximum_frames=FRAMES, boundary_frame=CHANGED,
        exact_original_decisions=CHANGED, raw_model_forecast_comparisons=FORECASTS,
        original_actual_commands_before_intervention_exact=True, prior_commands_compared=CHANGED,
        final_terminal=None, final_failure=None, full_controller_recovered_at_boundary=True,
        stopped_at_original_failed_observation=True, following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True, raw_visual_tracker_unchanged=True,
        model_state_sha256=MODEL_STATE, model_state_unchanged=True, unexecuted_outcomes_inferred=False,
        native_execution=False, navigation_verified=False)
    for key, value in expected.items():
        if report[key] != value or type(report[key]) is not type(value):
            raise ValueError('complete prospective height intervention required: '+key)
    count = 0; last = None
    originals = islice(read_rows(INPUT/CASE[0]), FRAMES)
    for i, (old, row) in enumerate(zip(originals, read_rows(root), strict=True)):
        if i >= FRAMES or row['tick'] != i or old['tick'] != i:
            raise ValueError('exact505 ordered prospective decisions required')
        if row['public_input_arrays_unchanged'] is not True:
            raise ValueError('public inputs must remain unchanged')
        check = compare_step(old['decision'], row['decision'], row['original_requested_command'], frame=i)
        if (check != row['comparison'] or row['decision']['terminal'] is not None
                or row['decision']['failure'] is not None):
            raise ValueError('saved replay must reconstruct against original complete decisions')
        count += 1; last = row
    if count != FRAMES: raise ValueError('all505 decisions required')
    if (last['comparison'] != report['boundary_comparison']
            or last['comparison']['controller_recovered'] is not True
            or last['comparison']['partial_height_admitted'] is not True
            or last['comparison']['requested_command_changed'] is not True
            or last['decision']['requested_command'] != report['final_requested_command']
            or last['original_requested_command'] != [0., 0., 0.]
            or last['decision']['evidence']['partial_floor_height'] != report['partial_height_receipt']):
        raise ValueError('changed current command and exact measured-height receipt required')
    return report|dict(prior_requested_command=[0., 0., 0.])


def compare(prior, current, prefix_root, report):
    if report['frames'] != FRAMES or report['boundary_frame'] != CHANGED:
        raise ValueError('fixed505-observation intervention required')
    count = 750+50*CHANGED; hashes = []
    for directory in (prior, current):
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
            if any(len(archive[k]) < count for k in archive.files): raise ValueError('complete physical prefix required')
            hashes.append(fingerprint({k:archive[k][:count] for k in archive.files}))
    if hashes[0] != hashes[1]: raise ValueError('physics differs before the height intervention')
    tapes = [read_json(p, 'command_tape.json') for p in (prior, current)]
    if (any(len(t) < FRAMES for t in tapes)
            or any(not tapes[j][i]['completed'] for j in (0, 1) for i in range(CHANGED))
            or any(tapes[0][i]['requested_command'] != tapes[1][i]['requested_command'] for i in range(CHANGED))
            or tapes[0][CHANGED]['requested_command'] != report['prior_requested_command']
            or tapes[1][CHANGED]['requested_command'] != report['final_requested_command']):
        raise ValueError('actual commands must match the prospective height intervention')
    readers = [IntentReturnRGBDReplay(p) for p in (prior, current)]
    acquisitions = [read_json(p, 'auxiliary_camera_audit.json') for p in (prior, current)]
    checked = forecasts = 0
    for i, (old, new, bound) in enumerate(zip(islice(read_rows(prior), FRAMES),
            islice(read_rows(current), FRAMES), islice(read_rows(prefix_root), FRAMES), strict=True)):
        if not old['tick'] == new['tick'] == bound['tick'] == i: raise ValueError('ordered complete decisions required')
        for row in (old, new):
            if row['observation_index'] != i or row['pre_sample_index'] != 749+50*i:
                raise ValueError('actual observation endpoint required')
        public = []
        for directory, reader, rows in zip((prior, current), readers, acquisitions, strict=True):
            p, d, f, now = reader.packet(i)
            image, auxiliary = packet(directory, i, p, public_acquisition(rows[i]), now_ns=now)
            public.append(fingerprint((p, d, f, auxiliary, image, now)))
        if public[0] != public[1]: raise ValueError('paired native public prefix differs')
        if new['decision'] != bound['decision']:
            raise ValueError('complete new native decision differs from prospective replay')
        check = compare_step(old['decision'], new['decision'], tapes[0][i]['requested_command'], frame=i)
        if check != bound['comparison']: raise ValueError('native comparison differs from prospective replay')
        forecasts += int(check['raw_model_forecasts_compared'])
        for row, tape in zip((old, new), tapes, strict=True):
            if row['decision']['requested_command'] != tape[i]['requested_command']:
                raise ValueError('native requested command differs from recorded decision')
        checked += 1
    if checked != FRAMES or forecasts != FORECASTS: raise ValueError('complete505-observation/501-forecast prefix required')
    return dict(common_prefix_frames=FRAMES, first_intervention_frame=CHANGED,
        physical_prefix_samples=count, raw_physics_prefix_sha256=hashes[0],
        physical_and_public_prefix_exact=True, all_preintervention_observed_state_exact=True,
        all_preintervention_requested_commands_exact=True, complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=forecasts, all_compared_raw_model_forecasts_exact=True,
        original_intervention_command=tapes[0][CHANGED]['requested_command'],
        candidate_intervention_command=tapes[1][CHANGED]['requested_command'],
        candidate_intervention_command_completed=tapes[1][CHANGED]['completed'],
        following_physical_outcomes_compared=False, unexecuted_outcomes_inferred=False)
