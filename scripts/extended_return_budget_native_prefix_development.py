"""Actual native prefix must match the admitted prospective budget intervention."""
from contextlib import closing
from itertools import islice

import numpy as np

from scripts import replay_go2_extended_return_budget_controller_prefix_v1 as replay
from scripts import extended_return_budget_maze_pipeline_development as extended
from scripts.measured_plane_native_prefix_development import command
from scripts.navigation_artifact_root_development import artifact_path

run = replay.run
comparison = replay.pair.comparison
original_rows = replay.pair.pipeline.read_rows
extended_rows = extended.read_rows
RAW_FIELDS = frozenset(('timestamp_s','base_pose_world','base_twist_world','joint_position','joint_velocity',
    'requested_command','applied_command','post_slew_applied_command','physics_contact','phase','edge_index'))


def boundary(report):
    required = dict(frames=comparison.MAX_PREFIX_OBSERVATIONS,
        budget_only_preboundary_decisions_supported=True,complete_recorded_decisions_reproduced=True,
        reconstructed_public_packets_unchanged=True,initial_fixed_and_stopping_states_checked=True,
        baseline_navigation_ticks=4000,candidate_navigation_ticks=8000,
        full_native_history_replayed=False,following_intervention_observations_consumed=False,
        changed_command_executed=False,physical_prefix_verified=False,native_execution=False,
        verified_round_trip=False,navigation_qualified=False,real_time_qualified=False,
        hardware_qualified=False,goal_achieved=False)
    if any(type(report.get(k)) is not type(v) or report[k]!=v for k,v in required.items()):
        raise ValueError('complete positive prospective budget-only prefix required')
    ids=report['identities']
    if (ids['initial_model_sha256'] != [replay.pair.MODEL_SHA]*2
            or ids['final_model_sha256'] != [replay.pair.MODEL_SHA]*2
            or any(ids.get(k) is not True for k in ('independent_model_storage',
                'both_models_in_evaluation_mode','models_without_gradients'))):
        raise ValueError('same independent unchanged assigned models required')
    row=report['boundary']
    if (type(row['frame']) is not int or row['frame']!=comparison.BOUNDARY
            or row['stop'] is not True or row['stop_reason']!='FIRST_NORMALIZED_DECISION_DIFFERENCE'
            or row['complete_normalized_decision_exact'] is not False
            or row['original_terminal']!='MISSION_TICK_BUDGET_EXHAUSTED'
            or row['candidate_terminal'] is not None or row['terminal_changed'] is not True
            or row['original_requested_command']!=[0.,0.,0.]
            or type(row['requested_command_changed']) is not bool
            or row['requested_command_changed']!=(row['candidate_requested_command']!=row['original_requested_command'])):
        raise ValueError('actual old deadline and nonterminal candidate intervention required')
    return dict(frames=report['frames'],intervention=row['frame'],physics_samples=750+50*row['frame'],
        command_changed=row['requested_command_changed'],terminal_changed=True)


def prefix_availability(collection,report):
    bound=boundary(report);count=bound['frames']
    for key in ('decisions','command_ticks','completed_ticks','physics_samples'):
        if type(collection.get(key)) is not int or collection[key]<0:
            raise ValueError('actual nonnegative integer collection counts required')
    if (not collection['completed_ticks']<=collection['command_ticks']<=collection['decisions']<=collection['command_ticks']+1
            or collection['decisions']>8014 or collection['command_ticks']>8013
            or collection['physics_samples']>401400):
        raise ValueError('consistent bounded longer-trial population required')
    reason=('INTERVENTION_OBSERVATION_NOT_REACHED' if collection['decisions']<count else
        'INTERVENTION_COMMAND_NOT_ISSUED' if collection['command_ticks']<count else
        'INTERVENTION_COMMAND_INCOMPLETE' if collection['completed_ticks']<count else
        'INSUFFICIENT_PHYSICAL_SAMPLES' if collection['physics_samples']<bound['physics_samples']+50 else None)
    return dict(required_common_prefix_frames=count,required_physical_prefix_samples=bound['physics_samples'],
        boundary_interval_available_by_collection_counts=reason is None,unavailable_reason=reason,
        actual_physical_prefix_reconstructed=False,full_raw_prefix_compare_still_required=reason is None,
        navigation_verified=False)


def public_packets(directory,count,*,longer):
    pipeline=extended if longer else replay.pair.pipeline
    cls=extended.ExtendedReturnBudgetRGBDReplay if longer else pipeline.ExtendedBudgetRGBDReplay
    reader=cls(directory);acquisitions=run.read_json(directory,'auxiliary_camera_audit.json')
    if len(reader.frames)<count or len(acquisitions)<count:
        raise ValueError('complete actual intervention packet population required')
    for frame in range(count):
        p,d,f,now=reader.packet(frame)
        image,aux=pipeline.rgb_packet(directory,frame,p,run.public_acquisition(acquisitions[frame]),now_ns=now)
        yield run.fingerprint((p,d,f,image,aux)),now


def executed_boundary(tapes,report):
    bound=boundary(report);count=bound['frames'];intervention=bound['intervention']
    if len(tapes)!=2 or any(len(t)<count for t in tapes):
        raise ValueError('both actually completed intervention command intervals required')
    for frame in range(count):
        for tape in tapes:command(tape[frame],frame)
        if frame<intervention and run.fingerprint(tapes[0][frame])!=run.fingerprint(tapes[1][frame]):
            raise ValueError('every complete preintervention command record must match')
    expected=report['boundary']
    if (tapes[0][intervention]['requested_command']!=expected['original_requested_command']
            or tapes[1][intervention]['requested_command']!=expected['candidate_requested_command']):
        raise ValueError('both actually executed boundary requests must match prospective evidence')
    return bound


def compare(prior,current,report,*,prior_bindings,current_bindings,replay_bindings):
    """Caller authenticates launches/models/full audits; rehash all supplied inputs.

No native controller, simulator or observation after the intervention is run.
The actual 50-sample boundary request is checked without comparing its outcome.
"""
    bound=boundary(report);count=bound['frames'];samples=bound['physics_samples']
    required={'physics_trace.npz','command_tape.json','context_decisions.jsonl.gz','policy_observations.json',
        'policy_histories.npz','depth_observations.json','fast_gyro_histories.npz','auxiliary_camera_audit.json'}
    required|={f'{kind}_{frame:04d}.{suffix}' for frame in range(count)
        for kind,suffix in (('rgb','png'),('depth','npz'),('auxiliary_rgb','png'),('auxiliary_depth','npz'))}
    for directory,ids in ((prior,prior_bindings),(current,current_bindings)):
        if not {directory.name+'/'+name for name in required}<=ids.keys():
            raise ValueError('every consumed original and candidate raw prefix artifact must be bound')
        run.verify_artifacts(directory.parent,ids)
    if not {'report.json',replay.pair.pipeline.stream.NAME}<=replay_bindings.keys():
        raise ValueError('complete prospective report and decision stream bindings required')
    run.verify_artifacts(replay.OUTPUT,replay_bindings)
    if run.canonical(run.read_json(replay.OUTPUT,'report.json'))!=run.canonical(report):
        raise ValueError('same authenticated complete prospective report required')
    tapes=[run.read_json(p,'command_tape.json') for p in (prior,current)]
    executed_boundary(tapes,report);physics=[]
    for directory,tape,maximum in zip((prior,current),tapes,(201400,401400),strict=True):
        with np.load(artifact_path(directory.parent,directory.name+'/physics_trace.npz'),allow_pickle=False) as raw:
            if (set(raw.files)!=RAW_FIELDS or any(not samples+50<=len(raw[k])<=maximum for k in raw.files)
                    or len({len(raw[k]) for k in raw.files})!=1):
                raise ValueError('complete bounded actual physics schema and boundary interval required')
            physics.append(run.fingerprint({key:raw[key][:samples] for key in raw.files}))
            np.testing.assert_array_equal(raw['requested_command'][samples:samples+50],
                np.tile(np.asarray(tape[bound['intervention']]['requested_command']),(50,1)))
    if physics[0]!=physics[1]:raise ValueError('every preintervention physics field must match')
    tracker=comparison.PrefixComparison();frames=0
    with closing(original_rows(prior)) as old_rows,closing(extended_rows(current)) as new_rows, \
            closing(replay.pair.pipeline.read_rows(replay.OUTPUT)) as expected_rows, \
            closing(public_packets(prior,count,longer=False)) as old_public, \
            closing(public_packets(current,count,longer=True)) as new_public:
        for frame,(old,new,expected,a,b) in enumerate(zip(islice(old_rows,count),islice(new_rows,count),
                expected_rows,old_public,new_public,strict=True)):
            for row in (old,new):
                if (type(row['tick']) is not int or row['tick']!=frame
                        or type(row['observation_index']) is not int or row['observation_index']!=frame
                        or type(row['pre_sample_index']) is not int or row['pre_sample_index']!=749+50*frame):
                    raise ValueError('actual consecutive physical observation identities required')
            if (frame>=count or expected['tick']!=frame or a!=b
                    or a!=(expected['public_packet_sha256'],expected['observation_now_ns'])
                    or expected['public_inputs_unchanged'] is not True
                    or run.fingerprint(old['decision'])!=expected['recorded_decision_sha256']
                    or run.canonical(new['decision'])!=run.canonical(expected['decision'])
                    or old['decision']['requested_command']!=tapes[0][frame]['requested_command']
                    or new['decision']['requested_command']!=tapes[1][frame]['requested_command']):
                raise ValueError('actual packets and both complete decisions must match the prospective prefix')
            checked=tracker.observe(expected['baseline'],new['decision'],old['decision'],
                model_calls=expected['actual_model_forward_calls'])
            if (run.canonical(checked)!=run.canonical(expected['comparison'])
                    or checked['stop']!=(frame==bound['intervention'])):
                raise ValueError('same complete prospective first intervention required')
            frames+=1
    if frames!=count or not tracker.report()['budget_only_preboundary_decisions_supported']:
        raise ValueError('complete positive actual budget intervention required')
    for root,ids in ((prior.parent,prior_bindings),(current.parent,current_bindings),(replay.OUTPUT,replay_bindings)):
        run.verify_artifacts(root,ids)
    return dict(common_prefix_frames=count,first_changed_decision_frame=bound['intervention'],
        physical_prefix_samples=samples,raw_physics_prefix_sha256=physics[0],
        physical_and_public_prefix_exact=True,all_preintervention_complete_commands_exact=True,
        complete_original_decisions_match_prospective_prefix=True,complete_candidate_decisions_match_prospective_prefix=True,
        original_intervention_command=report['boundary']['original_requested_command'],
        candidate_intervention_command=report['boundary']['candidate_requested_command'],
        intervention_command_changed=bound['command_changed'],intervention_terminal_changed=True,
        candidate_intervention_command_completed=True,boundary_command_samples_present=50,
        all_supplied_artifacts_rehashed_before_and_after=True,full_native_audits_and_launch_admission_performed=False,
        following_physical_outcomes_compared=False,navigation_verified=False,unexecuted_outcomes_inferred=False)
