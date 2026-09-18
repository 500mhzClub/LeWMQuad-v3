"""Admit an ordinary commitment-contact raw prefix and pair its prospective execution."""
from itertools import islice
import numpy as np
from lewm.commitment_contact_anchored_prefix_development import PrefixComparison
from scripts import replay_go2_commitment_contact_anchored_prefix_v1 as replay
from scripts.reached_frontier_native_prefix_development import public_packets
from scripts.maze_decision_stream_development import read_rows
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.navigation_artifact_root_development import artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import verify
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def boundary(report):
    frames = report['frames']; changed = report['first_changed_command_frame']
    if (type(frames) is not int or frames != 4 or type(changed) is not int or changed != 3
            or report['original_requested_command'] != [0., 0., .45]
            or report['candidate_requested_command'] != [.2, 0., 0.]
            or type(report['raw_model_forecast_comparisons']) is not int
            or report['raw_model_forecast_comparisons'] != 1):
        raise ValueError('exact original left-turn-to-forward raw prefix required')
    return frames, changed


def reconstruct(report, originals, prospective, tape, expected_boundary_selection):
    frames, changed = boundary(report)
    if len(originals) != frames or len(prospective) != frames or len(tape) < frames:
        raise ValueError('complete ordered original and prospective prefix required')
    forecasts = 0; comparator = PrefixComparison()
    for i, (old, saved) in enumerate(zip(originals, prospective, strict=True)):
        command = tape[i]
        if (old['tick'] != i or saved['tick'] != i or command['tick'] != i
                or command['completed'] is not True or command['pre_sample_index'] != 749+50*i
                or command['post_sample_index'] != 799+50*i
                or saved['original_requested_command'] != command['requested_command']):
            raise ValueError('actual complete original command and exact endpoints required')
        for flag in ('original_complete_decision_reconstructed', 'public_input_arrays_unchanged',
                'complete_retained_observed_state_equal', 'selected_pending_forecasts_checked'):
            if saved[flag] is not True: raise ValueError('complete raw replay receipt required: '+flag)
        expected = expected_boundary_selection if i == changed else old['decision']['new_selection']
        if saved['decision']['new_selection'] != expected:
            raise ValueError('exact saved boundary selection required')
        check = comparator.compare(old['decision'], saved['decision'], command['requested_command'], frame=i)
        if check != saved['comparison'] or check['requested_command_changed'] is not (i == changed):
            raise ValueError('first and only changed request must reconstruct exactly')
        forecasts += int(check['raw_model_forecasts_compared'])
    if (forecasts != report['raw_model_forecast_comparisons']
            or originals[-1]['decision']['requested_command'] != report['original_requested_command']
            or prospective[-1]['decision']['requested_command'] != report['candidate_requested_command']):
        raise ValueError('complete summary must reconstruct from actual saved streams')
    return dict(frames=frames, first_intervention_frame=changed, raw_model_forecast_comparisons=forecasts,
        complete_saved_comparisons_reconstructed=True)


def admit_prefix(root, result):
    if result['status'] != 'COMMITMENT_CONTACT_ANCHORED_RAW_PREFIX_V1_COMPLETE' or result['native_execution'] is not False:
        raise ValueError('completed raw controller prefix required')
    verify(result['source_sha256']); verify_artifacts(root, result['artifact_sha256'])
    launch = read_json(root, 'launch.json')
    if (launch['source_sha256'] != result['source_sha256']
            or launch['saved_selection_result_sha256'] != replay.SAVED_SHA
            or launch['original_launch_sha256'] != replay.LAUNCH_SHA
            or launch['input_admission']['original_worker_terminal_sha256'] != result['original_worker_terminal_sha256']):
        raise ValueError('exact original worker, saved boundary and raw source identity required')
    report = result['report']; frames, _ = boundary(report)
    expected = dict(original_complete_decisions_reconstructed=True, candidate_matches_saved_selection_boundary=True,
        complete_retained_observed_state_exact=True, selected_pending_forecasts_checked=True, model_state_sha256=replay.MODEL_SHA,
        model_state_unchanged=True, public_input_arrays_unchanged=True,
        no_observation_after_changed_request_consumed=True, native_execution=False,
        unexecuted_outcomes_inferred=False, navigation_verified=False)
    if any(report[k] != v or type(report[k]) is not type(v) for k,v in expected.items()):
        raise ValueError('exact model, observation and causal-scope report required')
    _, saved_boundary, _ = replay.saved_inputs()
    prior = replay.original.OUTPUT/replay.CASE[0]
    originals = list(islice(read_rows(prior), frames)); prospective = list(read_rows(root))
    reconstruct(report, originals, prospective, read_json(prior, 'command_tape.json'), saved_boundary['expected_candidate_selection'])
    for actual, saved in zip(public_packets(prior, frames), prospective, strict=True):
        if actual != saved['public_input_sha256']: raise ValueError('original raw public packet differs from prefix')
    return report


def executed_boundary(tapes, report):
    frames, changed = boundary(report)
    if len(tapes) != 2 or any(len(t) < frames for t in tapes):
        raise ValueError('two complete physical command prefixes required')
    for tape in tapes:
        for i in range(frames):
            t = tape[i]
            if (t['tick'] != i or t['completed'] is not True
                    or t['pre_sample_index'] != 749+50*i or t['post_sample_index'] != 799+50*i):
                raise ValueError('every prefix command including the intervention must complete')
    if (any(tapes[0][i]['requested_command'] != tapes[1][i]['requested_command'] for i in range(changed))
            or tapes[0][changed]['requested_command'] != report['original_requested_command']
            or tapes[1][changed]['requested_command'] != report['candidate_requested_command']):
        raise ValueError('identical prior commands and exact changed command required')


def compare(prior, current, prefix_root, report):
    frames, changed = boundary(report); samples = 750+50*changed; hashes = []
    for directory in (prior, current):
        with np.load(artifact_path(directory.parent, directory.name+'/physics_trace.npz'), allow_pickle=False) as raw:
            if not raw.files or any(len(raw[k]) < samples for k in raw.files):
                raise ValueError('all shared preintervention physics samples required')
            hashes.append(fingerprint({k: raw[k][:samples] for k in raw.files}))
    if hashes[0] != hashes[1]: raise ValueError('physical trajectory differs before intervention')
    tapes = [read_json(p, 'command_tape.json') for p in (prior, current)]; executed_boundary(tapes, report)
    originals = list(islice(read_rows(prior), frames)); actual = list(islice(read_rows(current), frames))
    prospective = list(read_rows(prefix_root)); _, saved_boundary, _ = replay.saved_inputs()
    reconstructed = reconstruct(report, originals, prospective, tapes[0], saved_boundary['expected_candidate_selection'])
    if len(actual) != frames: raise ValueError('complete actual native prefix required')
    for i, (old, new, saved, p, q) in enumerate(zip(originals, actual, prospective,
            public_packets(prior, frames), public_packets(current, frames), strict=True)):
        if p != q or p != saved['public_input_sha256']: raise ValueError('all paired public packets must match')
        for row, tape in ((old, tapes[0]), (new, tapes[1])):
            if (row['tick'] != i or row['observation_index'] != i or row['pre_sample_index'] != 749+50*i
                    or row['decision']['requested_command'] != tape[i]['requested_command']):
                raise ValueError('actual ordered observation and command endpoints required')
        if new['decision'] != saved['decision']:
            raise ValueError('complete actual controller decision differs from raw prospective replay')
    return dict(common_prefix_frames=frames, first_intervention_frame=changed, physical_prefix_samples=samples,
        raw_physics_prefix_sha256=hashes[0], physical_and_public_prefix_exact=True,
        all_preintervention_requested_commands_exact=True, complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=reconstructed['raw_model_forecast_comparisons'],
        all_compared_raw_model_forecasts_exact=True,
        original_intervention_command=report['original_requested_command'],
        candidate_intervention_command=report['candidate_requested_command'],
        candidate_intervention_command_completed=True, following_physical_outcomes_compared=False,
        unexecuted_outcomes_inferred=False)
