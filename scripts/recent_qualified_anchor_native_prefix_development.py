"""Admit a positive retention replay and compare its exact fresh native prefix."""
from itertools import islice
import numpy as np
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.recent_qualified_anchor_prefix_development import PrefixComparison,MAX_FRAMES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet,public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def boundary(report):
    changed=report['first_requested_command_difference'];frames=report['frames']
    if (type(changed) is not int or not 0<=changed<MAX_FRAMES or type(frames) is not int
            or frames!=changed+1 or report['maximum_frames']!=MAX_FRAMES
            or report['final_terminal'] is not None or report['final_failure'] is not None
            or report['final_requested_command']==[0.,0.,0.]
            or report['final_requested_command']==report['prior_requested_command']):
        raise ValueError('nonterminal changed movement at the complete replay boundary required')
    return frames,changed


def admit_prefix(root,result):
    from scripts.replay_go2_recent_qualified_anchor_prefix_v1 import INPUT,CASE,INPUT_SHA
    if (result['status']!='RECENT_QUALIFIED_ANCHOR_PREFIX_V1_COMPLETE'
            or result['native_result_sha256']!=INPUT_SHA or result['model_loaded'] is not True
            or result['model_training'] is not False or result['native_execution'] is not False
            or result['shadow_replay_only'] is not True):
        raise ValueError('completed fixed-input nonphysical retention replay required')
    report=result['report'];frames,changed=boundary(report)
    expected=dict(case=CASE[0],layout_index=1,model_state_sha256=MODEL_STATE,model_state_unchanged=True,
        original_actual_commands_before_intervention_exact=True,
        stopped_at_first_changed_command_or_either_terminal=True,following_recorded_observations_consumed=False,
        public_input_arrays_unchanged=True,unexecuted_outcomes_inferred=False,native_execution=False,navigation_verified=False)
    for key,value in expected.items():
        if report[key]!=value or type(report[key]) is not type(value):raise ValueError('exact replay admission required: '+key)
    comparison=PrefixComparison();count=forecasts=exact=attempts=qualified=0;last=old=None
    for i,(old,last) in enumerate(zip(islice(read_rows(INPUT/CASE[0]),frames),read_rows(root),strict=True)):
        if i>=frames or old['tick']!=i or last['tick']!=i or last['public_input_arrays_unchanged'] is not True:
            raise ValueError('complete ordered prospective prefix required')
        check=comparison.compare(old['decision'],last['decision'],last['original_requested_command'],frame=i)
        if check!=last['comparison'] or check['stop'] is not (i==changed) or check['requested_command_changed'] is not (i==changed):
            raise ValueError('all saved causal comparisons must reconstruct, stopping only at the boundary')
        count+=1;forecasts+=int(check['raw_model_forecasts_compared']);exact+=int(check['complete_original_decision_exact'])
        attempts+=check['extra_reference_attempts'];qualified+=check['extra_qualified_references']
    if count!=frames:raise ValueError('untruncated replay decisions required')
    totals=dict(raw_model_forecast_comparisons=forecasts,exact_original_decisions=exact,
        extra_reference_attempts=attempts,extra_qualified_references=qualified,
        first_reference_attempt=comparison.first_reference_attempt,first_qualified_reference=comparison.first_qualified_reference,
        first_decision_difference=comparison.first_decision_difference,
        first_requested_command_difference=comparison.first_command_difference,
        final_requested_command=last['decision']['requested_command'],prior_requested_command=old['decision']['requested_command'],
        final_terminal=last['decision']['terminal'],prior_terminal=old['decision']['terminal'],
        final_failure=last['decision']['failure'],boundary_comparison=last['comparison'])
    for key,value in totals.items():
        if report[key]!=value:raise ValueError('reconstructed prefix summary differs: '+key)
    if not qualified:raise ValueError('at least one qualified retained-view intervention required')
    return report


def compare(prior,current,prefix_root,report):
    frames,changed=boundary(report);samples=750+50*changed;hashes=[]
    for directory in (prior,current):
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
            if any(len(archive[k])<samples for k in archive.files):raise ValueError('complete shared physical prefix required')
            hashes.append(fingerprint({k:archive[k][:samples] for k in archive.files}))
    if hashes[0]!=hashes[1]:raise ValueError('physics differs before the changed command')
    tapes=[read_json(p,'command_tape.json') for p in (prior,current)]
    if (any(len(t)<frames for t in tapes)
            or any(tapes[j][i]['completed'] is not True for j in (0,1) for i in range(changed))
            or any(tapes[0][i]['requested_command']!=tapes[1][i]['requested_command'] for i in range(changed))
            or tapes[0][changed]['requested_command']!=report['prior_requested_command']
            or tapes[1][changed]['requested_command']!=report['final_requested_command']):
        raise ValueError('exact prior commands and actual prospective intervention required')
    readers=[IntentReturnRGBDReplay(p) for p in (prior,current)]
    acquisitions=[read_json(p,'auxiliary_camera_audit.json') for p in (prior,current)]
    comparator=PrefixComparison();checked=forecasts=exact=0
    for i,(old,new,saved) in enumerate(zip(islice(read_rows(prior),frames),islice(read_rows(current),frames),
            islice(read_rows(prefix_root),frames),strict=True)):
        if not old['tick']==new['tick']==saved['tick']==i:raise ValueError('ordered complete native decisions required')
        for row in (old,new):
            if row['observation_index']!=i or row['pre_sample_index']!=749+50*i:raise ValueError('actual observation endpoint required')
        public=[]
        for directory,reader,acq in zip((prior,current),readers,acquisitions,strict=True):
            p,d,f,now=reader.packet(i);image,aux=packet(directory,i,p,public_acquisition(acq[i]),now_ns=now)
            public.append(fingerprint((p,d,f,image,aux,now)))
        if public[0]!=public[1]:raise ValueError('paired physical public observations differ')
        if new['decision']!=saved['decision']:raise ValueError('complete native decision differs from prospective replay')
        check=comparator.compare(old['decision'],new['decision'],tapes[0][i]['requested_command'],frame=i)
        if check!=saved['comparison']:raise ValueError('native comparison differs from prospective replay')
        for row,tape in zip((old,new),tapes,strict=True):
            if row['decision']['requested_command']!=tape[i]['requested_command']:raise ValueError('actual native request differs from decision')
        checked+=1;forecasts+=int(check['raw_model_forecasts_compared']);exact+=int(check['complete_original_decision_exact'])
    if (checked!=frames or forecasts!=report['raw_model_forecast_comparisons']
            or exact!=report['exact_original_decisions'] or not comparator.stopped
            or comparator.first_command_difference!=changed):
        raise ValueError('complete physical and prospective boundary required')
    return dict(common_prefix_frames=frames,first_intervention_frame=changed,physical_prefix_samples=samples,
        raw_physics_prefix_sha256=hashes[0],physical_and_public_prefix_exact=True,
        all_preintervention_requested_commands_exact=True,complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=forecasts,all_compared_raw_model_forecasts_exact=True,
        complete_original_decisions_exact=exact,first_reference_attempt=comparator.first_reference_attempt,
        first_qualified_reference=comparator.first_qualified_reference,first_observed_decision_difference=comparator.first_decision_difference,
        changed_visual_and_downstream_state_compared_to_prospective_replay=True,
        original_intervention_command=tapes[0][changed]['requested_command'],candidate_intervention_command=tapes[1][changed]['requested_command'],
        candidate_intervention_command_completed=tapes[1][changed]['completed'],
        following_physical_outcomes_compared=False,unexecuted_outcomes_inferred=False)
