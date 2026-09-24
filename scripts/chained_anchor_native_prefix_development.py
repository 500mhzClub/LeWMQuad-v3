"""Compare a fresh simulation to the authenticated anchor intervention prefix.

Native geometry is used only here, after execution, to evaluate prefix identity.
This helper launches no simulation and supplies no controller input.
"""
from contextlib import closing
from itertools import islice
import numpy as np

from scripts import verify_go2_chained_anchor_controller_completion_v1 as completed
from scripts.navigation_artifact_root_development import artifact_path
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.maze_decision_stream_development import read_rows

replay = completed.run
FRAMES = 854
INTERVENTION = 853
PHYSICS_SAMPLES = 750+50*INTERVENTION


def boundary(report):
    expected = dict(frames=FRAMES, exact_original_decisions=INTERVENTION,
        original_forecasts_compared=850, boundary_terminal=None, boundary_failure=None,
        model_state_sha256=replay.MODEL_SHA, model_state_unchanged=True,
        following_recorded_observations_consumed=False,
        actual_original_commands_before_intervention_exact=True, new_command_executed=False,
        original_native_failure_preserved=True, native_execution=False, navigation_qualified=False, goal_achieved=False)
    if any(not completed.equal(report.get(k), v) for k, v in expected.items()):
        raise ValueError('complete fixed controller replay and admitted boundary required')
    check = report['boundary_comparison']
    if not completed.equal(check, dict(stop=True, complete_original_decision_exact=False,
            original_forecast_compared=False, controller_admitted_reacquired_pose=True,
            requested_command_changed=False, original_controller_failed=False, navigation_recovered=False)):
        raise ValueError('exact anchor admission without changed boundary command required')
    if report['boundary_selected_action'] != 'right_turn' or report['boundary_requested_command'] != [0., 0., -.45]:
        raise ValueError('actual fixed right-turn boundary required')


def command_row(row, frame):
    if (type(row.get('tick')) is not int or row['tick'] != frame or row['completed'] is not True
            or row['pre_sample_index'] != 749+50*frame or row['post_sample_index'] != 799+50*frame):
        raise ValueError('complete actual command and exact physical endpoints required')


def executed_boundary(tapes, report):
    boundary(report)
    if len(tapes) != 2 or any(len(t) < FRAMES for t in tapes):
        raise ValueError('two complete command prefixes required')
    for tape in tapes:
        for frame in range(FRAMES): command_row(tape[frame], frame)
    if (any(not completed.equal(tapes[0][i]['requested_command'], tapes[1][i]['requested_command'])
            for i in range(FRAMES))
            or not completed.equal(tapes[1][INTERVENTION]['requested_command'], report['boundary_requested_command'])):
        raise ValueError('all original commands and completed candidate boundary command must match')


def reconstruct(report, originals, prospective, original_tape, observed):
    boundary(report)
    if len(original_tape) < FRAMES: raise ValueError('complete original command prefix required')
    count = exact = forecasts = 0
    saved = None
    for frame, (old, saved, visual) in enumerate(zip(originals, prospective, observed, strict=True)):
        if frame >= FRAMES: raise ValueError('following observation is outside the intervention prefix')
        check = completed.check_row(old, saved, visual, original_tape[frame],
            frame=frame, public_sha=visual['public_packet_sha256'])
        count += 1
        exact += int(check['complete_original_decision_exact'])
        forecasts += int(check['original_forecast_compared'])
    reconstructed = completed.reconstruct_report(count, exact, forecasts, saved)
    if not completed.equal(reconstructed, report): raise ValueError('whole replay report must reconstruct')
    return forecasts


def public_packets(directory):
    reader = replay.observer.IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    for frame in range(FRAMES):
        p, d, fast, now = reader.packet(frame)
        image, aux = replay.observer.packet(directory, frame, p,
            replay.observer.public_acquisition(acquisitions[frame]), now_ns=now)
        yield replay.observer.fingerprint((p, d, fast, image, aux))


def compare(prior, current, prefix_root, report):
    boundary(report)
    hashes = []
    for directory in (prior, current):
        with np.load(artifact_path(directory.parent, directory.name+'/physics_trace.npz'), allow_pickle=False) as raw:
            if not raw.files or any(len(raw[k]) < PHYSICS_SAMPLES+50 for k in raw.files):
                raise ValueError('all prefix physics and completed boundary command samples required')
            hashes.append(replay.observer.fingerprint({k:raw[k][:PHYSICS_SAMPLES] for k in raw.files}))
    if hashes[0] != hashes[1]: raise ValueError('physical trajectory differs before intervention')
    tapes = [read_json(p, 'command_tape.json') for p in (prior, current)]
    executed_boundary(tapes, report)
    with closing(read_rows(prior)) as old, closing(read_rows(prefix_root)) as saved, \
            closing(read_rows(replay.observer.OUTPUT)) as visual:
        forecasts = reconstruct(report, islice(old, FRAMES), saved, tapes[0], visual)
    count = 0
    with closing(read_rows(current)) as actual, closing(read_rows(prefix_root)) as prospective:
        for frame, (new, saved, p, q) in enumerate(zip(islice(actual, FRAMES), prospective,
                public_packets(prior), public_packets(current), strict=True)):
            if (frame >= FRAMES or new['tick'] != frame or new['observation_index'] != frame
                    or new['pre_sample_index'] != 749+50*frame or p != q or p != saved['public_input_sha256']
                    or not completed.equal(new['decision'], saved['decision'])
                    or not completed.equal(new['decision']['requested_command'], tapes[1][frame]['requested_command'])):
                raise ValueError('complete actual physical/public/controller prefix must match replay')
            count += 1
    if count != FRAMES: raise ValueError('complete actual intervention observation required')
    return dict(common_prefix_frames=count, first_intervention_frame=INTERVENTION,
        physical_prefix_samples=PHYSICS_SAMPLES, raw_physics_prefix_sha256=hashes[0],
        physical_and_public_prefix_exact=True, all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True, original_forecasts_compared=forecasts,
        candidate_intervention_command=report['boundary_requested_command'], candidate_intervention_command_completed=True,
        intervention_command_changed=False, boundary_command_samples_present=50,
        anchor_evidence_intervention=True, original_controller_failed_at_intervention=False,
        following_physical_outcomes_compared=False, navigation_verified=False, unexecuted_outcomes_inferred=False)
