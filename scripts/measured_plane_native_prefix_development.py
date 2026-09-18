"""Post-execution comparison at the measured-plane controller's first new command.

Physics is evaluator-only. This module neither supplies native state to a
controller nor infers the outcome of a command from the predecessor trajectory.
"""
from contextlib import closing
from itertools import islice
import numpy as np
from scripts import verify_go2_measured_plane_controller_prefix_v1 as completed
from scripts.navigation_artifact_root_development import artifact_path

job,run = completed.job,completed.run
FRAMES = 123
INTERVENTION = 122
PHYSICS_SAMPLES = 750+50*INTERVENTION
COMPLETION_SHA = 'ca03a74ce91eba199ae485acc6ee5729872557e428b42e8ef805a99135dc4bbf'
RESULT_SHA = '8385e643b776865a44d9271404e8c05a8acc46b37e7ff9bc4b8bf48396e93047'


def boundary(report):
    expected = dict(frames=FRAMES,raw_forecast_comparisons=120,model_state_sha256=job.MODEL_SHA,
        model_states_unchanged=True,following_changed_command_outcome_consumed=False,
        changed_command_executed=False,native_execution=False,navigation_recovered=False,goal_achieved=False)
    if any(run.canonical(report.get(k)) != run.canonical(v) for k,v in expected.items()):
        raise ValueError('exact verified measured-plane controller prefix required')
    check = report['boundary_comparison']
    expected_check = dict(frame=INTERVENTION,complete_original_decision_exact=True,
        complete_candidate_observer_and_floor_exact=True,original_forecast_compared=True,
        requested_command_changed=True,terminal_changed=False,stop=True,
        stop_reason='FIRST_CHANGED_REQUEST_OR_TERMINAL',changed_command_executed=False,
        following_unexecuted_outcome_consumed=False,navigation_recovered=False,
        model_or_memory_advantage_established=False)
    if run.canonical(check) != run.canonical(expected_check):
        raise ValueError('exact first changed command boundary required')
    for key,command,action in (('boundary_original',[0.,0.,0.],'hold'),
            ('boundary_candidate',[0.,0.,-.45],'right_turn')):
        d = report[key]
        if (d['terminal'] is not None or d['failure'] is not None
                or d['requested_command'] != command or d['new_selection']['action'] != action):
            raise ValueError('original hold and prospective right turn required')


def command(row,frame):
    if (type(row['tick']) is not int or row['tick'] != frame or row['completed'] is not True
            or row['pre_sample_index'] != 749+50*frame or row['post_sample_index'] != 799+50*frame):
        raise ValueError('actual completed command endpoints required')


def executed_boundary(prior,current,report):
    boundary(report)
    if len(prior) < FRAMES or len(current) < FRAMES:
        raise ValueError('both full command prefixes required')
    for frame in range(FRAMES):
        command(prior[frame],frame); command(current[frame],frame)
        if frame < INTERVENTION and prior[frame]['requested_command'] != current[frame]['requested_command']:
            raise ValueError('preintervention commands must match')
    if (prior[INTERVENTION]['requested_command'] != report['boundary_original']['requested_command']
            or current[INTERVENTION]['requested_command'] != report['boundary_candidate']['requested_command']):
        raise ValueError('new native execution must actually complete the new boundary command')


def public_packets(directory):
    reader = run.pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions = run.read_json(directory,'auxiliary_camera_audit.json')
    for frame in range(FRAMES):
        p,d,f,now = reader.packet(frame)
        image,aux = run.pipeline.rgb_packet(directory,frame,p,
            run.public_acquisition(acquisitions[frame]),now_ns=now)
        yield run.fingerprint((p,d,f,image,aux))


def compare(prior,current,report):
    """Inputs must be source/artifact authenticated by the prospective launcher."""
    boundary(report)
    physics = []
    for directory in (prior,current):
        with np.load(artifact_path(directory.parent,directory.name+'/physics_trace.npz'),allow_pickle=False) as raw:
            if not raw.files or any(len(raw[k]) < PHYSICS_SAMPLES+50 for k in raw.files):
                raise ValueError('complete shared physics prefix and new command interval required')
            physics.append(run.fingerprint({k:raw[k][:PHYSICS_SAMPLES] for k in raw.files}))
    if physics[0] != physics[1]: raise ValueError('physical trajectory before new command must match')
    old_tape,new_tape = [run.read_json(p,'command_tape.json') for p in (prior,current)]
    executed_boundary(old_tape,new_tape,report)
    count = 0
    with closing(run.pipeline.read_rows(current)) as actual, closing(run.pipeline.read_rows(job.OUTPUT)) as saved:
        for frame,(new,expected,old_public,new_public) in enumerate(zip(islice(actual,FRAMES),saved,
                public_packets(prior),public_packets(current),strict=True)):
            if (frame >= FRAMES or new['tick'] != frame or new['observation_index'] != frame
                    or new['pre_sample_index'] != 749+50*frame or expected['tick'] != frame
                    or old_public != new_public or old_public != expected['public_packet_sha256']
                    or run.canonical(new['decision']) != run.canonical(expected['decision'])
                    or new['decision']['requested_command'] != new_tape[frame]['requested_command']):
                raise ValueError('complete fresh public/controller prefix must reproduce the candidate replay')
            count += 1
    if count != FRAMES: raise ValueError('full candidate boundary observation required')
    return dict(common_prefix_frames=count,first_changed_command_frame=INTERVENTION,
        physical_prefix_samples=PHYSICS_SAMPLES,raw_physics_prefix_sha256=physics[0],
        physical_and_public_prefix_exact=True,all_preintervention_requested_commands_exact=True,
        complete_candidate_decisions_match_prospective_prefix=True,original_forecasts_compared=120,
        original_intervention_command=[0.,0.,0.],candidate_intervention_command=[0.,0.,-.45],
        candidate_intervention_command_completed=True,intervention_command_changed=True,
        boundary_command_samples_present=50,following_physical_outcomes_compared=False,
        navigation_verified=False,unexecuted_outcomes_inferred=False)
