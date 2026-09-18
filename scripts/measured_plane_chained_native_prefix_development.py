"""Evaluator-only physical prefix at the completed replay's actual boundary.

This is source preparation, not a native launcher. The eventual launcher must
authenticate the completed controller replay and both complete native artifact
rosters before calling compare(). No boundary index or new action is selected
here, and no following physical outcome is inferred from the old trajectory.
"""
from contextlib import closing
from itertools import islice

import numpy as np

from scripts import replay_go2_measured_plane_chained_controller_prefix_v1 as replay
from scripts.measured_plane_native_prefix_development import command
from scripts.navigation_artifact_root_development import artifact_path

run = replay.run


def boundary(report):
    count = report['frames']
    if type(count) is not int or not 4 <= count <= 3124:
        raise ValueError('bounded actual completed controller prefix required')
    for key in ('two_fresh_independent_models','models_unchanged_without_gradients',
            'complete_original_decisions_reproduced','public_packets_unchanged'):
        if report.get(key) is not True:raise ValueError('completed original controller replay required: '+key)
    for key in ('following_changed_command_outcome_consumed','changed_command_executed',
            'native_execution','navigation_recovered','real_time_qualified','hardware_qualified','goal_achieved'):
        if report.get(key) is not False:raise ValueError('unexecuted diagnostic boundary required: '+key)
    if report['model_state_sha256'] != replay.inputs.job.MODEL_SHA:
        raise ValueError('same exact corrected model identity required')
    old,new = report['boundary_original'],report['boundary_candidate']
    check = replay.comparison.compare(old,new,old,frame=count-1,maximum_frames=3124,
        model_calls=report['boundary_comparison']['actual_model_forward_calls'])
    if (check != report['boundary_comparison'] or not check['stop']
            or check['stop_reason'] != 'FIRST_CHANGED_REQUEST_OR_TERMINAL'
            or not (check['requested_command_changed'] or check['terminal_changed'])
            or new['terminal'] is not None or new['failure'] is not None):
        raise ValueError('first actual intervention with an admitted candidate required')
    return dict(frames=count,intervention=count-1,physics_samples=750+50*(count-1),
        command_changed=check['requested_command_changed'],terminal_changed=check['terminal_changed'])


def executed_boundary(old,new,report):
    bound=boundary(report);count=bound['frames'];intervention=bound['intervention']
    if len(old)<count or len(new)<count:raise ValueError('complete executed command prefixes required')
    for frame in range(count):
        command(old[frame],frame);command(new[frame],frame)
        if frame<intervention and old[frame]['requested_command'] != new[frame]['requested_command']:
            raise ValueError('all preintervention requested commands must match')
    if (old[intervention]['requested_command'] != report['boundary_original']['requested_command']
            or new[intervention]['requested_command'] != report['boundary_candidate']['requested_command']):
        raise ValueError('actual boundary commands must match completed replay')
    return bound


def prefix_availability(collection,report):
    """Account for early negatives; this metadata check reconstructs no prefix."""
    bound=boundary(report);count=bound['frames']
    keys=('decisions','command_ticks','completed_ticks','physics_samples')
    if any(type(collection.get(key)) is not int or collection[key]<0 for key in keys):
        raise ValueError('nonnegative integer original collection counts required')
    if not collection['completed_ticks']<=collection['command_ticks']<=collection['decisions']<=collection['command_ticks']+1:
        raise ValueError('consistent original command and observation counts required')
    reason=('INTERVENTION_OBSERVATION_NOT_REACHED' if collection['decisions']<count else
        'INTERVENTION_COMMAND_NOT_ISSUED' if collection['command_ticks']<count else
        'INTERVENTION_COMMAND_INCOMPLETE' if collection['completed_ticks']<count else
        'INSUFFICIENT_PHYSICAL_SAMPLES' if collection['physics_samples']<bound['physics_samples']+50 else None)
    return dict(required_common_prefix_frames=count,required_physical_prefix_samples=bound['physics_samples'],
        boundary_interval_available_by_collection_counts=reason is None,unavailable_reason=reason,
        actual_physical_prefix_reconstructed=False,complete_candidate_decisions_reconstructed=False,
        full_raw_prefix_compare_still_required=reason is None,navigation_verified=False)


def public_packets(directory,count):
    reader=run.pipeline.ExtendedBudgetRGBDReplay(directory)
    acquisitions=run.read_json(directory,'auxiliary_camera_audit.json')
    if len(reader.frames)<count or len(acquisitions)<count:
        raise ValueError('complete boundary observation population required')
    for frame in range(count):
        p,d,f,now=reader.packet(frame)
        image,aux=run.pipeline.rgb_packet(directory,frame,p,
            run.public_acquisition(acquisitions[frame]),now_ns=now)
        yield run.fingerprint((p,d,f,image,aux))


def compare(prior,current,report):
    bound=boundary(report);count=bound['frames'];samples=bound['physics_samples']
    tapes=[run.read_json(directory,'command_tape.json') for directory in (prior,current)]
    executed_boundary(*tapes,report)
    physics=[]
    for directory,tape in zip((prior,current),tapes,strict=True):
        path=artifact_path(directory.parent,directory.name+'/physics_trace.npz')
        with np.load(path,allow_pickle=False) as raw:
            if not raw.files or any(len(raw[key])<samples+50 for key in raw.files):
                raise ValueError('complete shared physics and actual boundary command interval required')
            physics.append(run.fingerprint({key:raw[key][:samples] for key in raw.files}))
            np.testing.assert_array_equal(raw['requested_command'][samples:samples+50],
                np.tile(np.asarray(tape[bound['intervention']]['requested_command']),(50,1)))
    if physics[0] != physics[1]:raise ValueError('entire preintervention physical trajectory must match')
    frames=0
    with closing(run.pipeline.read_rows(prior)) as baseline, closing(run.pipeline.read_rows(current)) as candidate, \
            closing(run.pipeline.read_rows(replay.OUTPUT)) as saved, \
            closing(public_packets(prior,count)) as old_packets, closing(public_packets(current,count)) as new_packets:
        for frame,(old,new,expected,old_public,new_public) in enumerate(zip(
                islice(baseline,count),islice(candidate,count),saved,old_packets,new_packets,strict=True)):
            if (frame>=count or old['tick'] != frame or new['tick'] != frame or expected['tick'] != frame
                    or old['observation_index'] != frame or new['observation_index'] != frame
                    or old['pre_sample_index'] != 749+50*frame or new['pre_sample_index'] != 749+50*frame
                    or old_public != new_public or old_public != expected['public_packet_sha256']
                    or run.canonical(old['decision']) != run.canonical(expected['original'])
                    or run.canonical(new['decision']) != run.canonical(expected['decision'])
                    or old['decision']['requested_command'] != tapes[0][frame]['requested_command']
                    or new['decision']['requested_command'] != tapes[1][frame]['requested_command']):
                raise ValueError('complete actual public packets and both controller decisions must reproduce')
            check=replay.comparison.compare(old['decision'],new['decision'],expected['original'],
                frame=frame,maximum_frames=3124,model_calls=expected['actual_model_forward_calls'])
            if (check != expected['comparison'] or check['stop'] != (frame==bound['intervention'])):
                raise ValueError('actual completed replay first boundary must reproduce without earlier change')
            frames+=1
    if frames != count:raise ValueError('complete actual boundary observation required')
    return dict(common_prefix_frames=count,first_changed_decision_frame=bound['intervention'],
        physical_prefix_samples=samples,raw_physics_prefix_sha256=physics[0],
        physical_and_public_prefix_exact=True,all_preintervention_requested_commands_exact=True,
        complete_original_decisions_match_prospective_prefix=True,
        complete_candidate_decisions_match_prospective_prefix=True,
        original_intervention_command=report['boundary_original']['requested_command'],
        candidate_intervention_command=report['boundary_candidate']['requested_command'],
        intervention_command_changed=bound['command_changed'],intervention_terminal_changed=bound['terminal_changed'],
        candidate_intervention_command_completed=True,boundary_command_samples_present=50,
        following_physical_outcomes_compared=False,navigation_verified=False,unexecuted_outcomes_inferred=False)
