"""Verify actual execution of the first forecast-source command intervention."""
from contextlib import closing
from itertools import islice

import numpy as np

from scripts import replay_go2_measured_plane_forecast_source_prefix_v1 as replay
from scripts.measured_plane_native_prefix_development import command
from scripts.navigation_artifact_root_development import artifact_path

run = replay.run
FRAMES = 4
INTERVENTION = 3
PHYSICS_SAMPLES = 900
LAUNCH_SHA = 'f06e93dcb7dd4bc81242a7274aec7751edde77a6c0d66e05356175d061060b76'
RESULT_SHA = '0d080057b6fb4802e103623496e92e0bbb7be77474cfdbf9a9e6d08c500b8478'


def boundary(report):
    if (report['frames'] != FRAMES or report['learned_forecasts'] != 1
            or report['actual_model_forward_calls'] != [1, 0]
            or report['model_states_unchanged'] is not True
            or report['model_state_sha256'] != replay.job.MODEL_SHA
            or report['fully_nonpredictive_arm'] is not False
            or report['native_execution'] is not False
            or report['retrospective_navigation_outcomes_inferred'] is not False):
        raise ValueError('exact completed four-frame forecast-source comparison required')
    old, new = report['boundary_baseline'], report['boundary_candidate']
    expected = replay.compare(old, new, replay.learned_reference(old), frame=INTERVENTION)
    if (report['boundary_comparison'] != expected or expected['requested_command_changed'] is not True
            or expected['terminal_boundary'] is not False):
        raise ValueError('exact first nonterminal command intervention required')
    for decision, action, requested in ((old, 'left_arc', [.16, 0., .45]), (new, 'forward', [.2, 0., 0.])):
        if (decision['terminal'] is not None or decision['failure'] is not None
                or decision['new_selection']['action'] != action
                or decision['requested_command'] != requested
                or decision['new_selection']['phase_admissible_candidates'] != 6):
            raise ValueError('original six-feasible-action left-arc versus forward boundary required')


def executed_boundary(old, new, report):
    boundary(report)
    if len(old) < FRAMES or len(new) < FRAMES: raise ValueError('complete executed command prefixes required')
    for frame in range(FRAMES):
        command(old[frame], frame); command(new[frame], frame)
        if frame < INTERVENTION and old[frame]['requested_command'] != new[frame]['requested_command']:
            raise ValueError('preintervention physical commands must match')
    if (old[INTERVENTION]['requested_command'] != report['boundary_baseline']['requested_command']
            or new[INTERVENTION]['requested_command'] != report['boundary_candidate']['requested_command']):
        raise ValueError('actual learned and nominal boundary commands must complete')


def packets(directory):
    reader = run.pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions = run.read_json(directory, 'auxiliary_camera_audit.json')
    for frame in range(FRAMES):
        p, d, f, now = reader.packet(frame)
        image, aux = run.pipeline.rgb_packet(directory, frame, p,
            run.public_acquisition(acquisitions[frame]), now_ns=now)
        yield run.fingerprint((p, d, f, image, aux))


def compare(prior, current, report):
    boundary(report)
    physics = []
    for directory in (prior, current):
        with np.load(artifact_path(directory.parent, directory.name+'/physics_trace.npz'), allow_pickle=False) as raw:
            if not raw.files or any(len(raw[key]) < PHYSICS_SAMPLES+50 for key in raw.files):
                raise ValueError('complete common physics and actual new-command interval required')
            physics.append(run.fingerprint({key: raw[key][:PHYSICS_SAMPLES] for key in raw.files}))
    if physics[0] != physics[1]: raise ValueError('entire preintervention physical trajectory must match')
    old_tape, new_tape = [run.read_json(directory, 'command_tape.json') for directory in (prior, current)]
    executed_boundary(old_tape, new_tape, report)
    count = 0
    with closing(run.pipeline.read_rows(prior)) as baseline, closing(run.pipeline.read_rows(current)) as nominal, \
            closing(run.pipeline.read_rows(replay.OUTPUT)) as saved:
        for frame, (old, new, expected, old_public, new_public) in enumerate(zip(
                islice(baseline, FRAMES), islice(nominal, FRAMES), saved, packets(prior), packets(current), strict=True)):
            if (old['tick'] != frame or new['tick'] != frame or expected['tick'] != frame
                    or old['observation_index'] != frame or new['observation_index'] != frame
                    or old['pre_sample_index'] != 749+50*frame or new['pre_sample_index'] != 749+50*frame
                    or old_public != new_public or old_public != expected['public_packet_sha256']
                    or run.canonical(old['decision']) != run.canonical(replay.learned_reference(expected['baseline']))
                    or run.canonical(new['decision']) != run.canonical(expected['decision'])
                    or old['decision']['requested_command'] != old_tape[frame]['requested_command']
                    or new['decision']['requested_command'] != new_tape[frame]['requested_command']):
                raise ValueError('complete actual physics, public packets and both controller decisions must match')
            count += 1
    if count != FRAMES: raise ValueError('complete prospective first-action intervention required')
    return dict(common_prefix_frames=FRAMES, first_changed_command_frame=INTERVENTION,
        physical_prefix_samples=PHYSICS_SAMPLES, raw_physics_prefix_sha256=physics[0],
        physical_and_public_prefix_exact=True, all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,
        complete_baseline_decisions_match_prospective_prefix=True,
        baseline_intervention_command=[.16, 0., .45], candidate_intervention_command=[.2, 0., 0.],
        candidate_intervention_command_completed=True, boundary_command_samples_present=50,
        following_physical_outcomes_compared=False, navigation_verified=False,
        unexecuted_outcomes_inferred=False, both_arms_predictive=True)
