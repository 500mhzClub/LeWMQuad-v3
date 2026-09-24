"""Both fixed native outcomes and matched causal-prefix comparisons."""
import argparse
from collections import Counter
import numpy as np
from scripts.run_go2_augmented_family_switch_goal_probe_v1 import OUTPUT as INPUT,PRIOR,PRIOR_SHA,CASES,TRIAL
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_augmented_family_switch_goal_readout_v1_attempt_001'
PROTOCOL='docs/go2_augmented_family_switch_goal_readout_v1_2026-09-08.md'


def summarize(record):
    name=record['case'];rows=read_json(INPUT/name,'context_decisions.json')
    audit=read_json(INPUT,name+'_audit.json')
    selections=[dict(tick=r['tick'],**r['decision']['new_selection']) for r in rows
        if r['decision']['new_selection'] is not None]
    first_failure=next((dict(tick=r['tick'],failure=r['decision']['failure'],
        evidence=r['decision']['evidence']) for r in rows if r['decision']['failure']),None)
    forecast=[s for s in selections if 'prediction' in s]
    pose_errors=audit['observed_pose_xy_errors_m']
    return dict(case=name,model_name=record['model_name'],condition=record['condition'],
        collection=record['collection'],goal=audit['goal'],first_failure=first_failure,
        raw_sensor_reconstruction_pass=audit['raw_sensor_reconstruction_pass'],
        raw_model_command_replay_pass=audit['raw_model_command_replay_pass'],
        model_state_unchanged=audit['model_state_unchanged'],
        strict_physical_visibility_pass=audit['strict_physical_visibility_pass'],
        hard_measurement_failed_frames=audit['hard_measurement_failed_frames'],
        maximum_observed_pose_xy_error_m=max(pose_errors,default=None),
        selections=len(selections),model_forecasts=len(forecast),
        selected_actions=dict(Counter(str(s['action']) for s in selections)),
        selection_trace=[{k:s[k] for k in ('tick','mode','action','waypoint_map_xy_m',
            'phase_admissible_candidates') if k in s} for s in selections],
        no_jointly_feasible_action_selections=sum(s['action'] is None for s in forecast),
        nominal_constraint_changed_selections=sum(s['action']!=s['before_nominal_constraint_action'] for s in forecast),
        observer_retention_trace=[dict(tick=r['tick'],overlap=r['decision']['evidence'].get('overlap_retention'),
            continuity=(r['decision']['evidence'].get('continuity_evidence') or {}).get('status'),
            promotion_reason=(r['decision']['evidence'].get('current_pose') or {}).get('promotion_reason'))
            for r in rows if r['decision']['evidence'] is not None],
        timing={k:timing([r[k] for r in rows if k in r]) for k in ('acquisition_wall_ms',
            'controller_wall_ms','observation_and_control_wall_ms','iteration_with_command_wall_ms')},
        measured_under_two_case_concurrency=True,real_time_qualified=False,
        independent_maze_evaluation=False,navigation_qualified=False)


def common_prefix_length(rows_a,rows_b,tape_a,tape_b):
    """Include the observation before the first different command or terminal."""
    command=next((i for i,(a,b) in enumerate(zip(tape_a,tape_b))
        if a['requested_command']!=b['requested_command']),None)
    if command is None and len(tape_a)!=len(tape_b):command=min(len(tape_a),len(tape_b))
    terminal=next((i for i,(a,b) in enumerate(zip(rows_a,rows_b))
        if a['decision']['terminal']!=b['decision']['terminal']),None)
    differences=[v for v in (command,terminal) if v is not None]
    frames=min(len(rows_a),len(rows_b),min(differences)+1 if differences else max(len(rows_a),len(rows_b)))
    return frames,command,terminal


def compare(a,b):
    rows=[read_json(p,'context_decisions.json') for p in (a,b)]
    tapes=[read_json(p,'command_tape.json') for p in (a,b)]
    frames,command,terminal=common_prefix_length(*rows,*tapes)
    arrays={}
    for filename,count in (('physics_trace.npz',750+50*(frames-1)),('policy_histories.npz',frames),
            ('fast_gyro_histories.npz',frames)):
        hashes=[]
        for p in (a,b):
            with np.load(p/filename,allow_pickle=False) as z:
                values={k:z[k][:count] for k in z.files}
                if any(len(v)!=count for v in values.values()):raise ValueError('incomplete comparison prefix')
                hashes.append(fingerprint(values))
        arrays[filename]=dict(first_sha256=hashes[0],second_sha256=hashes[1],exact=hashes[0]==hashes[1])
    rgb=[[r['rgb_sha256'] for r in read_json(p,'camera_audit.json')[:frames]] for p in (a,b)]
    observed=[[{k:r['decision'][k] for k in ('evidence','memory_receipt','observed_goal_distance_m')}
        for r in rr[:frames]] for rr in rows]
    return dict(first=str(a),second=str(b),first_requested_command_difference=command,
        first_terminal_difference=terminal,common_prefix_frames=frames,
        comparison_includes_observation_before_differing_command=True,
        arrays=arrays,common_prefix_rgb_exact=rgb[0]==rgb[1],
        common_prefix_observer_memory_exact=observed[0]==observed[1],
        model_predictions_expected_identical=False,unexecuted_outcomes_inferred=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe-result-sha256',required=True);args=parser.parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive two-case native readout')
    verify_artifacts(INPUT,{'result.json':args.probe_result_sha256});result=read_json(INPUT,'result.json')
    assert result['status']=='AUGMENTED_FAMILY_SWITCH_GOAL_PROBE_COMPLETE' and result['cases']==[list(c) for c in CASES]
    ids={'result.json':args.probe_result_sha256,**result['artifact_sha256']};verify_artifacts(INPUT,ids)
    verify_artifacts(PRIOR,{'result.json':PRIOR_SHA});prior=read_json(PRIOR,'result.json')
    prior_ids={'result.json':PRIOR_SHA,**prior['artifact_sha256']};verify_artifacts(PRIOR,prior_ids)
    old=read_json(INPUT,'launch.json');verify(old)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_augmented_family_switch_goal_probe_v1.py',
        'lewm/tests/test_augmented_family_switch_goal_readout_development.py'),old['source_sha256'])
    launch=old|dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),
        input_artifact_sha256=ids,prior_probe_artifact_sha256=prior_ids,native_execution=False)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        reports=[summarize(r) for r in result['conditions']]
        assert len(reports)==2 and {r['case'] for r in reports}=={c[0] for c in CASES}
        comparisons=[compare(PRIOR/('full_direct_'+TRIAL),INPUT/c[0]) for c in CASES]
        comparisons.append(compare(INPUT/CASES[0][0],INPUT/CASES[1][0]))
        verify(launch);verify_artifacts(INPUT,ids);verify_artifacts(PRIOR,prior_ids)
        write_json(OUTPUT/'result.json',dict(status='AUGMENTED_FAMILY_SWITCH_GOAL_READOUT_COMPLETE',
            conditions=reports,causal_prefix_comparisons=comparisons,
            measured_goal_successes=sum(r['goal']['verified_goal_reached'] and r['strict_physical_visibility_pass']
                and not r['hard_measurement_failed_frames'] for r in reports),
            probe_result_sha256=args.probe_result_sha256,source_sha256=sources,launch_sha256=digest(OUTPUT/'launch.json'),
            prior_outcomes_changed=False,checkpoint_selection_performed=False,native_execution=False,
            independent_maze_evaluation=False,navigation_qualified=False,goal_achieved=False))
        print('AUGMENTED_FAMILY_SWITCH_GOAL_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AUGMENTED_FAMILY_SWITCH_GOAL_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
