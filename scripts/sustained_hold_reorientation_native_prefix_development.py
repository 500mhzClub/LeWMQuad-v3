"""Authenticate the raw sustained-turn prefix and compare prospective native physics."""
from itertools import islice
import numpy as np
from lewm.sustained_hold_reorientation_prefix_development import compare_step
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts import replay_go2_sustained_hold_reorientation_maze02_prefix_v1 as replay
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet,public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.navigation_artifact_root_development import artifact_path,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import verify
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

LAUNCH_SHA='6f68017be0f68198ab08af97fb4d90f33ebbdc2b1b56f8da123e8b34b4dd89df'


def boundary(report):
    if (type(report['frames']) is not int or report['frames']!=407
            or type(report['first_changed_command_frame']) is not int or report['first_changed_command_frame']!=406
            or report['raw_model_forecast_comparisons']!=404
            or report['original_requested_command']!=[0.,0.,0.]
            or report['candidate_requested_command']!=[0.,0.,.45]):
        raise ValueError('exact 407-observation sustained-turn boundary required')
    return 407,406


def reconstruct(report,originals,prospective,tape,expected):
    frames,changed=boundary(report)
    if len(originals)!=frames or len(prospective)!=frames or len(tape)<frames or len(expected)!=frames:
        raise ValueError('complete ordered original and prospective prefix required')
    forecasts=0
    for i,(old,saved,wanted) in enumerate(zip(originals,prospective,expected,strict=True)):
        command=tape[i]
        if (old['tick']!=i or saved['tick']!=i or command['tick']!=i or command['completed'] is not True
                or command['pre_sample_index']!=749+50*i or command['post_sample_index']!=799+50*i
                or saved['original_requested_command']!=command['requested_command']
                or replay.saved.identity(old)!=wanted['original_row_sha256']):
            raise ValueError('exact original row and completed physical command required')
        for flag in ('original_complete_decision_reconstructed','public_input_arrays_unchanged','complete_retained_contact_state_equal'):
            if saved[flag] is not True:raise ValueError('complete raw comparison evidence required: '+flag)
        check=compare_step(old['decision'],saved['decision'],command['requested_command'],frame=i,
            expected_selection_sha256=wanted['candidate_selection_sha256'])
        if check!=saved['comparison'] or check['requested_command_changed'] is not (i==changed):
            raise ValueError('first and only changed request must reconstruct')
        forecasts+=int(check['raw_model_forecasts_compared'])
    if (forecasts!=report['raw_model_forecast_comparisons']
            or originals[-1]['decision']['requested_command']!=report['original_requested_command']
            or prospective[-1]['decision']['requested_command']!=report['candidate_requested_command']):
        raise ValueError('complete prefix summary must reconstruct from actual rows')
    return dict(frames=frames,first_intervention_frame=changed,raw_model_forecast_comparisons=forecasts,
        complete_saved_comparisons_reconstructed=True)


def public_packets(directory,frames):
    reader=IntentReturnRGBDReplay(directory);acquisitions=read_json(directory,'auxiliary_camera_audit.json')
    for i in range(frames):
        p,d,f,now=reader.packet(i)
        image,auxiliary=packet(directory,i,p,public_acquisition(acquisitions[i]),now_ns=now)
        # This exact order is frozen in the sustained raw replay.
        yield fingerprint((p,d,f,image,auxiliary,now))


def admit_prefix(root,result):
    if result['status']!='SUSTAINED_HOLD_REORIENTATION_RAW_PREFIX_V1_COMPLETE' or result['native_execution'] is not False:
        raise ValueError('completed original raw sustained prefix required')
    if read_json(root,'result.json')!=result or (root/'failure.json').exists():
        raise ValueError('actual completed result without replacement of a failure required')
    verify(result['source_sha256']);verify_artifacts(root,result['artifact_sha256']|{'launch.json':LAUNCH_SHA})
    launch=read_json(root,'launch.json')
    if (launch['source_sha256']!=result['source_sha256'] or launch['saved_prefix_sha256']!=replay.SAVED_SHA
            or launch['model_state_sha256']!=replay.MODEL_SHA or launch['frames']!=407
            or launch['input_admission']['native_result_sha256']!=replay.saved.prior.NATIVE_SHA):
        raise ValueError('exact original native source, model and saved boundary required')
    report=result['report'];frames,_=boundary(report)
    flags=dict(original_complete_decisions_reconstructed=True,candidate_matches_every_saved_selection=True,
        observed_map_and_contact_state_exact=True,prior_pending_model_forecasts_exact=True,
        model_state_sha256=replay.MODEL_SHA,model_state_unchanged=True,
        no_observation_after_changed_request_consumed=True,changed_command_executed=False,
        native_execution=False,unexecuted_outcomes_inferred=False,navigation_verified=False)
    if any(report[k]!=v or type(report[k]) is not type(v) for k,v in flags.items()):
        raise ValueError('exact complete model, observation and causal-scope report required')
    expected=replay.saved_inputs()
    prior=replay.original.OUTPUT/replay.original.CASE[0]
    originals=list(islice(read_rows(prior),frames));prospective=list(read_rows(root))
    reconstruct(report,originals,prospective,read_json(prior,'command_tape.json'),expected['comparisons'])
    for actual,saved in zip(public_packets(prior,frames),prospective,strict=True):
        if actual!=saved['public_input_sha256']:raise ValueError('actual raw public packet differs from replay')
    verify(result['source_sha256']);verify_artifacts(root,result['artifact_sha256'])
    return report


def executed_boundary(tapes,report):
    frames,changed=boundary(report)
    if len(tapes)!=2 or any(len(t)<frames for t in tapes):
        raise ValueError('two complete physical command prefixes required')
    for tape in tapes:
        for i in range(frames):
            t=tape[i]
            if (t['tick']!=i or t['completed'] is not True
                    or t['pre_sample_index']!=749+50*i or t['post_sample_index']!=799+50*i):
                raise ValueError('every prefix command including intervention must complete')
    if (any(tapes[0][i]['requested_command']!=tapes[1][i]['requested_command'] for i in range(changed))
            or tapes[0][changed]['requested_command']!=report['original_requested_command']
            or tapes[1][changed]['requested_command']!=report['candidate_requested_command']):
        raise ValueError('identical earlier commands and exact changed command required')


def compare(prior,current,prefix_root,report):
    frames,changed=boundary(report);samples=750+50*changed;hashes=[]
    for directory in (prior,current):
        with np.load(artifact_path(directory.parent,directory.name+'/physics_trace.npz'),allow_pickle=False) as raw:
            if not raw.files or any(len(raw[k])<samples+50 for k in raw.files):
                raise ValueError('preintervention physics and all 50 intervention samples required')
            hashes.append(fingerprint({k:raw[k][:samples] for k in raw.files}))
    if hashes[0]!=hashes[1]:raise ValueError('physical trajectory differs before intervention')
    tapes=[read_json(p,'command_tape.json') for p in (prior,current)];executed_boundary(tapes,report)
    originals=list(islice(read_rows(prior),frames));actual=list(islice(read_rows(current),frames))
    prospective=list(read_rows(prefix_root));expected=replay.saved_inputs()
    reconstructed=reconstruct(report,originals,prospective,tapes[0],expected['comparisons'])
    if len(actual)!=frames:raise ValueError('complete actual native prefix required')
    for i,(old,new,saved,p,q) in enumerate(zip(originals,actual,prospective,
            public_packets(prior,frames),public_packets(current,frames),strict=True)):
        if p!=q or p!=saved['public_input_sha256']:raise ValueError('all paired public packets must match')
        for row,tape in ((old,tapes[0]),(new,tapes[1])):
            if (row['tick']!=i or row['observation_index']!=i or row['pre_sample_index']!=749+50*i
                    or row['decision']['requested_command']!=tape[i]['requested_command']):
                raise ValueError('actual ordered observations and commands required')
        if new['decision']!=saved['decision']:
            raise ValueError('complete actual decision differs from prospective raw replay')
    return dict(common_prefix_frames=frames,first_intervention_frame=changed,physical_prefix_samples=samples,
        raw_physics_prefix_sha256=hashes[0],physical_and_public_prefix_exact=True,
        all_preintervention_requested_commands_exact=True,complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=reconstructed['raw_model_forecast_comparisons'],
        all_compared_raw_model_forecasts_exact=True,original_intervention_command=report['original_requested_command'],
        candidate_intervention_command=report['candidate_requested_command'],candidate_intervention_command_completed=True,
        following_physical_outcomes_compared=False,unexecuted_outcomes_inferred=False)
