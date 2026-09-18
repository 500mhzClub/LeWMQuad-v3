"""Admit all matched objective prefixes and compare their fresh physical starts."""
from itertools import islice
import numpy as np
from lewm.independent_floor_transport_study_development import MODEL_STATE, LAYOUTS
from lewm.supervised_rollout_maze_study_development import SUPERVISED_STATE
from lewm.matched_rollout_objective_admission_development import NAMES, CONDITIONS, BASE_STATE
from lewm.matched_objective_prefix_development import compare_step, MAX_FRAMES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

FRAMES=4
CHANGED=3


def corrections(admission):
    return {c:admission['coefficients'][n]['heads']['rollout_outcomes']['applied_bias_xy_m']
        for n,c in zip(NAMES,CONDITIONS,strict=True)}


def admit_prefixes(root,result,admission):
    if (result['status']!='MATCHED_OBJECTIVE_PREFIXES_V1_COMPLETE'
            or result['model_loaded'] is not True or result['model_training'] is not False
            or result['native_execution'] is not False or result['all_fixed_cases_executed'] is not True
            or len(result['conditions'])!=len(LAYOUTS)):
        raise ValueError('complete fixed actual-model objective comparison required')
    reports=[]
    for index,report in zip(LAYOUTS,result['conditions'],strict=True):
        expected=dict(case=f'full_jepa_novel_maze_{index:02d}',layout_index=index,frames=FRAMES,
            maximum_frames=MAX_FRAMES[index],first_prediction_difference=CHANGED,paired_forecast_banks=1,
            first_requested_command_difference=CHANGED,first_terminal_difference=None,
            jepa_final_requested_command=[.16,0.,.45],supervised_final_requested_command=[0.,0.,-.45],
            jepa_terminal=None,supervised_terminal=None,complete_original_jepa_decisions_exact=True,
            shared_observed_state_exact=True,prior_actual_commands_exact=True,prior_commands_compared=3,
            stopped_before_following_a_changed_command=True,following_recorded_observations_consumed=False,
            public_input_arrays_unchanged=True,model_states_unchanged=True,model_training=False,
            native_execution=False,unexecuted_outcomes_inferred=False,jepa_advantage_established=False)
        for key,value in expected.items():
            if report[key]!=value or type(report[key]) is not type(value):
                raise ValueError('exact completed objective prefix required: '+key)
        states=(MODEL_STATE,SUPERVISED_STATE)
        models=[dict(name=n,condition=c,base_state_sha256=b,corrected_state_sha256=s,
            head='rollout_outcomes',model_training=False) for n,c,b,s in zip(NAMES,CONDITIONS,BASE_STATE,states,strict=True)]
        if report['models']!=models: raise ValueError('exact paired corrected model identities required')
        rows=list(read_rows(root/report['case']))
        if len(rows)!=FRAMES: raise ValueError('all four and only four saved decisions required')
        for i,row in enumerate(rows):
            if row['tick']!=i or row['public_input_arrays_unchanged'] is not True:
                raise ValueError('ordered unmodified public inputs required')
            check=compare_step(row['jepa_decision'],row['jepa_decision'],row['decision'],
                row['original_requested_command'],frame=i,layout=index,corrections=corrections(admission))
            if (check!=row['comparison'] or check['requested_command_changed'] is not (i==CHANGED)
                    or check['terminal_changed'] is not False
                    or check['both_full_forecast_banks_present'] is not (i==CHANGED)
                    or check['raw_prediction_changed'] is not (i==CHANGED)):
                raise ValueError('complete saved causal comparison required')
        if (rows[-1]['jepa_decision']['requested_command']!=report['jepa_final_requested_command']
                or rows[-1]['decision']['requested_command']!=report['supervised_final_requested_command']):
            raise ValueError('exact final report and decisions must agree')
        reports.append(report)
    return reports


def compare(prior,current,prefix_root,report,admission):
    index=report['layout_index']
    if index not in LAYOUTS or report['frames']!=FRAMES or report['first_requested_command_difference']!=CHANGED:
        raise ValueError('fixed four-observation paired intervention required')
    count=750+50*CHANGED; hashes=[]
    for directory in (prior,current):
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
            if any(len(archive[k])<count for k in archive.files): raise ValueError('complete physical prefix required')
            hashes.append(fingerprint({k:archive[k][:count] for k in archive.files}))
    if hashes[0]!=hashes[1]: raise ValueError('physics differs before the objective command intervention')
    tapes=[read_json(p,'command_tape.json') for p in (prior,current)]
    if (any(len(t)<FRAMES for t in tapes)
            or any(not tapes[j][i]['completed'] for j in (0,1) for i in range(CHANGED))
            or any(tapes[0][i]['requested_command']!=tapes[1][i]['requested_command'] for i in range(CHANGED))
            or tapes[0][CHANGED]['requested_command']!=report['jepa_final_requested_command']
            or tapes[1][CHANGED]['requested_command']!=report['supervised_final_requested_command']):
        raise ValueError('actual commands must match the paired objective intervention')
    readers=[IntentReturnRGBDReplay(p) for p in (prior,current)]
    acquisitions=[read_json(p,'auxiliary_camera_audit.json') for p in (prior,current)]
    checked=banks=0
    for i,(old,new,bound) in enumerate(zip(islice(read_rows(prior),FRAMES),
            islice(read_rows(current),FRAMES),islice(read_rows(prefix_root/report['case']),FRAMES),strict=True)):
        if not old['tick']==new['tick']==bound['tick']==i: raise ValueError('ordered complete decisions required')
        for row in (old,new):
            if row['observation_index']!=i or row['pre_sample_index']!=749+50*i:
                raise ValueError('actual observation endpoint required')
        public=[]
        for directory,reader,rows in zip((prior,current),readers,acquisitions,strict=True):
            p,d,f,now=reader.packet(i)
            image,auxiliary=packet(directory,i,p,public_acquisition(rows[i]),now_ns=now)
            public.append(fingerprint((p,d,f,auxiliary,image,now)))
        if public[0]!=public[1]: raise ValueError('paired native public prefix differs')
        if new['decision']!=bound['decision']:
            raise ValueError('complete supervised native decision differs from prospective replay')
        check=compare_step(old['decision'],bound['jepa_decision'],new['decision'],tapes[0][i]['requested_command'],
            frame=i,layout=index,corrections=corrections(admission))
        if check!=bound['comparison']: raise ValueError('native observed state/objective comparison differs')
        banks+=int(check['both_full_forecast_banks_present'])
        for row,tape in zip((old,new),tapes,strict=True):
            if row['decision']['requested_command']!=tape[i]['requested_command']:
                raise ValueError('native request differs from saved decision')
        checked+=1
    if checked!=FRAMES or banks!=1: raise ValueError('complete four-observation and paired forecast bank required')
    return dict(common_prefix_frames=FRAMES,first_intervention_frame=CHANGED,physical_prefix_samples=count,
        raw_physics_prefix_sha256=hashes[0],physical_and_public_prefix_exact=True,shared_observed_state_exact=True,
        all_preintervention_requested_commands_exact=True,complete_candidate_decisions_match_prospective_prefix=True,
        complete_original_jepa_decisions_exact=True,paired_forecast_banks=banks,
        original_intervention_command=tapes[0][CHANGED]['requested_command'],
        candidate_intervention_command=tapes[1][CHANGED]['requested_command'],
        candidate_intervention_command_completed=tapes[1][CHANGED]['completed'],
        following_physical_outcomes_compared=False,unexecuted_outcomes_inferred=False)
