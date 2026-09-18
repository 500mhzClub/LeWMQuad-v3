"""Reconstruct frontier replay evidence and pair a fresh physical intervention."""
from itertools import islice
import numpy as np
from lewm.reached_frontier_prefix_comparison_development import compare_step
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts import replay_go2_reached_frontier_maze03_prefix_v1 as replay
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.navigation_artifact_root_development import artifact_path
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def boundary(report):
    changed = report['first_request_or_terminal_difference']; frames = report['frames']
    reached = report['first_reached_frontier_frame']; different = report['first_normalized_decision_difference']
    if (type(changed) is not int or not 3 <= changed < 3014
            or type(frames) is not int or frames != changed+1
            or type(reached) is not int or type(different) is not int
            or not 3 <= reached <= different <= changed
            or report['final_terminal'] is not None
            or report['final_requested_command'] == report['prior_requested_command']
            or report['stop_reason'] != 'first_changed_request_or_terminal'):
        raise ValueError('complete reached-frontier replay with first changed nonterminal command required')
    return frames, changed


def reconstruct(report, originals, prospective, tape):
    frames, changed = boundary(report)
    if len(originals) != frames or len(prospective) != frames or len(tape) < frames:
        raise ValueError('complete original and prospective first-divergence prefix required')
    seen = False; first_reached = different = None; forecasts = 0
    for i, (old, saved) in enumerate(zip(originals, prospective, strict=True)):
        if (old['tick'] != i or saved['tick'] != i or tape[i]['tick'] != i
                or tape[i]['completed'] is not True
                or old['decision']['requested_command'] != tape[i]['requested_command']
                or saved['original_requested_command'] != tape[i]['requested_command']):
            raise ValueError('ordered actual original completed commands required')
        for flag in ('original_complete_decision_reconstructed', 'public_input_arrays_unchanged',
                'complete_retained_contact_state_equal'):
            if saved[flag] is not True: raise ValueError('complete raw replay evidence required: '+flag)
        check = compare_step(old['decision'], saved['decision'], tape[i]['requested_command'],
            frame=i, frontier_previously_reached=seen)
        if (check != saved['comparison']
                or (check['requested_command_changed'] or check['terminal_changed']) is not (i == changed)
                or old['decision']['terminal'] is not None or saved['decision']['terminal'] is not None):
            raise ValueError('first and only changed nonterminal boundary must reconstruct')
        if check['frontier_reached_this_frame']:
            seen = True
            if first_reached is None: first_reached = i
        if not check['normalized_complete_decision_exact'] and different is None: different = i
        forecasts += int(check['raw_model_forecasts_compared'])
    totals = dict(first_reached_frontier_frame=first_reached, first_normalized_decision_difference=different,
        raw_model_forecast_comparisons=forecasts,
        final_requested_command=prospective[-1]['decision']['requested_command'],
        prior_requested_command=originals[-1]['decision']['requested_command'],
        final_terminal=prospective[-1]['decision']['terminal'],
        final_frontier_transition_receipt=prospective[-1]['decision']['last_frontier_transition_receipt'],
        changed_selection=prospective[-1]['decision']['new_selection'])
    if any(report[k] != v for k, v in totals.items()):
        raise ValueError('complete replay summary must reconstruct from original and prospective streams')
    return dict(frames=frames, first_intervention_frame=changed,
        raw_model_forecast_comparisons=forecasts, first_reached_frontier_frame=first_reached,
        first_normalized_decision_difference=different, complete_saved_comparisons_reconstructed=True)


def public_packets(directory, frames):
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    for i in range(frames):
        p, d, f, now = reader.packet(i)
        image, auxiliary = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
        yield fingerprint((p, d, f, auxiliary, image, now))


def admit_prefix(root, result):
    if (result['status'] != 'REACHED_FRONTIER_MAZE03_PREFIX_V1_COMPLETE'
            or result['model_loaded'] is not True or result['model_training'] is not False
            or result['native_execution'] is not False):
        raise ValueError('completed recorded-data frontier prefix required')
    report = result['report']; frames, _ = boundary(report)
    expected = dict(case=replay.original.CASE[0], model_state_sha256=replay.MODEL_SHA,
        stopped_at_first_changed_request_or_terminal=True, following_recorded_observations_consumed=False,
        original_complete_decisions_reconstructed=True, unchanged_observed_and_executed_residual_state_exact=True,
        complete_retained_contact_state_equal=True, accumulated_observation_cells_unchanged=True,
        public_input_arrays_unchanged=True, model_state_unchanged=True, native_execution=False,
        unexecuted_outcomes_inferred=False, memory_advantage_established=False, navigation_verified=False)
    if any(report[k] != v or type(report[k]) is not type(v) for k, v in expected.items()):
        raise ValueError('exact model, observation and causal-scope claims required')
    prior = replay.original.OUTPUT/replay.original.CASE[0]
    originals = list(islice(read_rows(prior), frames)); prospective = list(read_rows(root))
    reconstruct(report, originals, prospective, read_json(prior, 'command_tape.json'))
    for actual, saved in zip(public_packets(prior, frames), prospective, strict=True):
        if actual != saved['public_input_sha256']: raise ValueError('raw original public packets differ from prefix')
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
                raise ValueError('all prefix commands including intervention must actually complete')
    if (any(tapes[0][i]['requested_command'] != tapes[1][i]['requested_command'] for i in range(changed))
            or tapes[0][changed]['requested_command'] != report['prior_requested_command']
            or tapes[1][changed]['requested_command'] != report['final_requested_command']):
        raise ValueError('identical preintervention requests and exact new boundary command required')


def compare(prior, current, prefix_root, report):
    frames, changed = boundary(report); samples = 750+50*changed; hashes = []
    for directory in (prior, current):
        with np.load(artifact_path(directory.parent, directory.name+'/physics_trace.npz'), allow_pickle=False) as saved:
            if not saved.files or any(len(saved[k]) < samples for k in saved.files):
                raise ValueError('all shared preintervention physics samples required')
            hashes.append(fingerprint({k: saved[k][:samples] for k in saved.files}))
    if hashes[0] != hashes[1]: raise ValueError('physical trajectory differs before intervention')
    tapes = [read_json(p, 'command_tape.json') for p in (prior, current)]
    executed_boundary(tapes, report)
    originals = list(islice(read_rows(prior), frames)); actual = list(islice(read_rows(current), frames))
    prospective = list(read_rows(prefix_root))
    reconstructed = reconstruct(report, originals, prospective, tapes[0])
    if len(actual) != frames: raise ValueError('complete actual native prefix required')
    for i, (old, new, saved, p, q) in enumerate(zip(originals, actual, prospective,
            public_packets(prior, frames), public_packets(current, frames), strict=True)):
        if p != q or p != saved['public_input_sha256']: raise ValueError('all paired raw public packets must match')
        for row, tape in ((old, tapes[0]), (new, tapes[1])):
            if (row['tick'] != i or row['observation_index'] != i or row['pre_sample_index'] != 749+50*i
                    or row['decision']['requested_command'] != tape[i]['requested_command']):
                raise ValueError('complete ordered actual decision and command endpoints required')
        if new['decision'] != saved['decision']:
            raise ValueError('complete native controller decision differs from prospective replay')
    return dict(common_prefix_frames=frames, first_intervention_frame=changed, physical_prefix_samples=samples,
        raw_physics_prefix_sha256=hashes[0], physical_and_public_prefix_exact=True,
        all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=reconstructed['raw_model_forecast_comparisons'],
        all_compared_raw_model_forecasts_exact=True,
        first_reached_frontier_frame=reconstructed['first_reached_frontier_frame'],
        first_observed_decision_difference=reconstructed['first_normalized_decision_difference'],
        original_intervention_command=report['prior_requested_command'],
        candidate_intervention_command=report['final_requested_command'],
        candidate_intervention_command_completed=True,
        following_physical_outcomes_compared=False, unexecuted_outcomes_inferred=False)
