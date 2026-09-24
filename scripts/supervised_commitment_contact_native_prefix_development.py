"""Fixed four-observation intervention; physical comparison ends before it acts."""
from itertools import islice
import numpy as np
from lewm.commitment_contact_prefix_development import PrefixComparison, MAX_FRAMES
from lewm.supervised_rollout_maze_study_development import SUPERVISED_STATE
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.replay_go2_supervised_commitment_contact_prefix_v1 import INPUT, CASE, DIAGNOSTIC_SHA
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

FRAMES = 4
CHANGED = 3


def fixed_report(report):
    expected = dict(case=CASE, layout_index=1, frames=FRAMES, maximum_frames=MAX_FRAMES,
        first_requested_command_difference=CHANGED, raw_model_forecast_comparisons=1,
        final_requested_command=[.16, 0., -.45], prior_requested_command=[0., 0., -.45],
        final_terminal=None, prior_terminal=None, final_failure=None,
        complete_selection_transform_verified_every_frame=True, original_actual_commands_before_intervention_exact=True,
        all_shared_observed_state_exact=True, stopped_at_first_changed_command_or_either_terminal=True,
        following_recorded_observations_consumed=False, public_input_arrays_unchanged=True,
        model_state_sha256=SUPERVISED_STATE, model_state_unchanged=True,
        scored_pose_horizon_ns=100_000_000, scored_contact_horizon_ns=100_000_000,
        path_constraint_horizon_ns=800_000_000, contact_scores_calibrated=False,
        unexecuted_outcomes_inferred=False, native_execution=False, navigation_verified=False)
    for key, value in expected.items():
        if report[key] != value or type(report[key]) is not type(value):
            raise ValueError('fixed completed contact-horizon intervention required: '+key)
    return report


def admit_prefix(root, result):
    if (result['status'] != 'SUPERVISED_COMMITMENT_CONTACT_PREFIX_V1_COMPLETE'
            or result['diagnostic_result_sha256'] != DIAGNOSTIC_SHA
            or result['model_loaded'] is not True or result['model_training'] is not False
            or result['native_execution'] is not False or result['shadow_replay_only'] is not True):
        raise ValueError('completed original nonphysical contact-horizon prefix required')
    report = fixed_report(result['report']); comparator = PrefixComparison(); count = forecasts = 0
    for i, (old, row) in enumerate(zip(islice(read_rows(INPUT/CASE), FRAMES), read_rows(root), strict=True)):
        if (i >= FRAMES or old['tick'] != i or row['tick'] != i
                or old['observation_index'] != i or old['pre_sample_index'] != 749+50*i
                or row['public_input_arrays_unchanged'] is not True):
            raise ValueError('complete ordered original public prefix required')
        check = comparator.compare(old['decision'], row['decision'], row['original_requested_command'], frame=i)
        if (check != row['comparison'] or check['stop'] is not (i == CHANGED)
                or check['requested_command_changed'] is not (i == CHANGED)
                or row['decision']['terminal'] is not None or row['decision']['failure'] is not None):
            raise ValueError('all original saved comparisons must reconstruct exactly')
        count += 1; forecasts += int(check['raw_model_forecasts_compared'])
    if (count != FRAMES or forecasts != 1 or check != report['boundary_comparison']
            or row['decision']['requested_command'] != report['final_requested_command']
            or row['original_requested_command'] != report['prior_requested_command']):
        raise ValueError('complete four-observation/one-forecast boundary and summary required')
    return report


def compare(prior, current, prefix_root, report):
    fixed_report(report); samples = 750+50*CHANGED; hashes = []
    for directory in (prior, current):
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
            if not archive.files or any(len(archive[k]) < samples for k in archive.files):
                raise ValueError('complete preintervention physical prefix required')
            hashes.append(fingerprint({k: archive[k][:samples] for k in archive.files}))
    if hashes[0] != hashes[1]: raise ValueError('physics differs before the contact-horizon command')
    tapes = [read_json(p, 'command_tape.json') for p in (prior, current)]
    if (any(len(t) < FRAMES for t in tapes)
            or any(tapes[j][i]['completed'] is not True for j in (0, 1) for i in range(FRAMES))
            or any(tapes[0][i]['requested_command'] != tapes[1][i]['requested_command'] for i in range(CHANGED))
            or tapes[0][CHANGED]['requested_command'] != report['prior_requested_command']
            or tapes[1][CHANGED]['requested_command'] != report['final_requested_command']):
        raise ValueError('completed actual commands must match the fixed causal intervention')
    readers = [IntentReturnRGBDReplay(p) for p in (prior, current)]
    acquisitions = [read_json(p, 'auxiliary_camera_audit.json') for p in (prior, current)]
    comparator = PrefixComparison(); checked = forecasts = 0
    for i, (old, new, bound) in enumerate(zip(islice(read_rows(prior), FRAMES),
            islice(read_rows(current), FRAMES), islice(read_rows(prefix_root), FRAMES), strict=True)):
        if not old['tick'] == new['tick'] == bound['tick'] == i:
            raise ValueError('ordered original, native and prospective decisions required')
        for row in (old, new):
            if row['observation_index'] != i or row['pre_sample_index'] != 749+50*i:
                raise ValueError('exact actual observation endpoint required')
        public = []
        for directory, reader, rows in zip((prior, current), readers, acquisitions, strict=True):
            p, d, f, now = reader.packet(i)
            image, aux = packet(directory, i, p, public_acquisition(rows[i]), now_ns=now)
            public.append(fingerprint((p, d, f, image, aux, now)))
        if public[0] != public[1]: raise ValueError('paired public sensor prefix differs')
        if new['decision'] != bound['decision']: raise ValueError('complete native decision differs from prospective replay')
        check = comparator.compare(old['decision'], new['decision'], tapes[0][i]['requested_command'], frame=i)
        if check != bound['comparison'] or check['stop'] is not (i == CHANGED):
            raise ValueError('native causal comparison differs from the stopped prospective replay')
        for row, tape in zip((old, new), tapes, strict=True):
            if row['decision']['requested_command'] != tape[i]['requested_command']:
                raise ValueError('actual command must equal saved controller request')
        checked += 1; forecasts += int(check['raw_model_forecasts_compared'])
    if checked != FRAMES or forecasts != 1: raise ValueError('complete fixed native prefix required')
    return dict(common_prefix_frames=FRAMES, first_intervention_frame=CHANGED,
        physical_prefix_samples=samples, raw_physics_prefix_sha256=hashes[0],
        physical_and_public_prefix_exact=True, shared_observed_state_receipts_exact=True,
        all_preintervention_requested_commands_exact=True, complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=forecasts, all_compared_raw_model_forecasts_exact=True,
        original_intervention_command=tapes[0][CHANGED]['requested_command'],
        candidate_intervention_command=tapes[1][CHANGED]['requested_command'],
        candidate_intervention_command_completed=True, following_physical_outcomes_compared=False,
        unexecuted_outcomes_inferred=False)
