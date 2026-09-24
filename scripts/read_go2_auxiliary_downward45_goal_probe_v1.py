"""Both fixed native outcomes and matched causal-prefix comparisons."""
import argparse
from collections import Counter
import numpy as np
from scripts.run_go2_auxiliary_downward45_goal_probe_v1 import OUTPUT as INPUT,PREVIOUS as PRIOR,PREVIOUS_SHA as PRIOR_SHA,CASES,TRIAL
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT=BASE/'go2_auxiliary_downward45_goal_readout_v1_attempt_001'
PROTOCOL='docs/go2_auxiliary_downward45_goal_readout_v1_2026-09-08.md'


def constraint_summary(selection):
    """Recorded veto evidence only; no reranking or unexecuted outcome labels."""
    records=[]
    for surface,nominal,path in zip(selection['surface_checks'],selection['nominal_action_checks'],
            selection['nominal_path_checks'],strict=True):
        action=nominal['action'];assert path['action']==action
        records.append(dict(action=action,phase_allowed=action in selection['phase_allowed_actions'],
            possible_surface_intersection=surface['possible_intersection'],
            nominal_first_step_clear=nominal['nominal_disk_connector_clear'],
            nominal_eight_steps_clear=path['all_predicted_segments_nominally_clear'],
            first_step_minimum_observed_cell_distance_m=nominal['minimum_observed_cell_distance_m'],
            nearest_observed_cell=nominal['nearest_observed_cell'],
            primary_hits=[dict(shape_id=s['shape_id'],voxels=s['intersecting_voxels'])
                for s in surface['shapes'] if s['intersecting_voxels']],
            auxiliary_hits=[dict(shape_id=s['shape_id'],voxels=s['intersecting_voxels'])
                for s in surface['auxiliary_shapes'] if s['intersecting_voxels']],
            uncovered_feet=[dict(shape_id=f['shape_id'],
                auxiliary_floor_voxels=f['auxiliary_floor']['intersecting_voxels'],
                auxiliary_other_voxels=f['auxiliary_other_or_unknown']['intersecting_voxels'])
                for f in surface['auxiliary_foot_floor_contacts'] if not f['measured_floor_contact_rule_eligible']]))
    return records


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
        active_wait_ticks=[r['tick'] for r in rows if r['decision']['infeasible_wait_active']],
        feasible_recoveries=[dict(tick=r['tick'],action=r['decision']['selected_action'],
            command=r['decision']['requested_command'])
            for i,r in enumerate(rows) if r['decision']['feasible_action_recoveries']>
                (rows[i-1]['decision']['feasible_action_recoveries'] if i else 0)],
        maximum_consecutive_infeasible_observations=max(r['decision']['consecutive_infeasible_observations'] for r in rows),
        auxiliary_frames=len(audit['auxiliary_sensor_audit']),
        auxiliary_visibility_pass_frames=sum(r['auxiliary_visibility_pass'] for r in audit['auxiliary_sensor_audit']),
        auxiliary_robot_occluded_frames=[r['frame'] for r in audit['auxiliary_sensor_audit'] if r['robot_pixels']],
        auxiliary_partition_trace=[dict(tick=r['tick'],receipt=r['decision']['auxiliary_floor_partition_receipt'])
            for r in rows if r['decision']['auxiliary_floor_partition_receipt'] is not None],
        selections=len(selections),model_forecasts=len(forecast),
        selected_actions=dict(Counter(str(s['action']) for s in selections)),
        selection_trace=[{k:s[k] for k in ('tick','mode','action','waypoint_map_xy_m',
            'phase_admissible_candidates') if k in s} for s in selections],
        constraint_trace=[dict(tick=s['tick'],selected_action=s['action'],candidates=constraint_summary(s)) for s in forecast],
        no_jointly_feasible_action_selections=sum(s['action'] is None for s in forecast),
        first_step_nominal_constraint_changed_selections=sum(s['first_step_action']!=s['before_nominal_constraint_action'] for s in forecast),
        eight_step_planning_changed_selections=sum(s['action']!=s['first_step_action'] for s in forecast),
        later_path_vetoed_candidate_forecasts=sum(c['nominal_disk_connector_clear'] and not p['all_predicted_segments_nominally_clear']
            for s in forecast for c,p in zip(s['nominal_action_checks'],s['nominal_path_checks'],strict=True)),
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


def compare(a,b,*,auxiliary_pair=False):
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
    predictions=[[(r['decision']['new_selection'] or {}).get('prediction') for r in rr[:frames]] for rr in rows]
    auxiliary=None
    if auxiliary_pair:
        auxiliary=[]
        for i in range(frames):
            hashes=[]
            for root in (a,b):
                with np.load(root/f'auxiliary_depth_{i:04d}.npz',allow_pickle=False) as z:
                    hashes.append(fingerprint({k:z[k] for k in ('depth_m','valid')}))
            auxiliary.append(dict(frame=i,first_sha256=hashes[0],second_sha256=hashes[1],exact=hashes[0]==hashes[1]))
    return dict(first=str(a),second=str(b),first_requested_command_difference=command,
        first_terminal_difference=terminal,common_prefix_frames=frames,
        comparison_includes_observation_before_differing_command=True,
        arrays=arrays,common_prefix_rgb_exact=rgb[0]==rgb[1],
        common_prefix_observer_memory_exact=observed[0]==observed[1],
        model_predictions_expected_identical=False,common_prefix_model_forecasts_exact=predictions[0]==predictions[1],
        auxiliary_comparison=auxiliary,
        common_prefix_auxiliary_exact=all(r['exact'] for r in auxiliary) if auxiliary is not None else None,
        unexecuted_outcomes_inferred=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe-result-sha256',required=True);args=parser.parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive two-case native readout')
    verify_artifacts(INPUT,{'result.json':args.probe_result_sha256});result=read_json(INPUT,'result.json')
    assert result['status']=='AUXILIARY_DOWNWARD45_GOAL_PROBE_COMPLETE' and result['cases']==[list(c) for c in CASES]
    ids={'result.json':args.probe_result_sha256,**result['artifact_sha256']};verify_artifacts(INPUT,ids)
    verify_artifacts(PRIOR,{'result.json':PRIOR_SHA});prior=read_json(PRIOR,'result.json')
    prior_ids={'result.json':PRIOR_SHA,**prior['artifact_sha256']};verify_artifacts(PRIOR,prior_ids)
    old=read_json(INPUT,'launch.json');verify(old)
    sources=discover_sources((PROTOCOL,'scripts/read_go2_auxiliary_downward45_goal_probe_v1.py',
        'lewm/tests/test_auxiliary_downward45_goal_readout_development.py'),old['source_sha256'])
    launch=old|dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),
        input_artifact_sha256=ids,prior_probe_artifact_sha256=prior_ids,native_execution=False)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        reports=[summarize(r) for r in result['conditions']]
        assert len(reports)==2 and {r['case'] for r in reports}=={c[0] for c in CASES}
        comparisons=[compare(PRIOR/c[0],INPUT/c[0]) for c in CASES]
        comparisons.append(compare(INPUT/CASES[0][0],INPUT/CASES[1][0],auxiliary_pair=True))
        verify(launch);verify_artifacts(INPUT,ids);verify_artifacts(PRIOR,prior_ids)
        write_json(OUTPUT/'result.json',dict(status='AUXILIARY_DOWNWARD45_GOAL_READOUT_COMPLETE',
            conditions=reports,causal_prefix_comparisons=comparisons,
            measured_goal_successes=sum(r['goal']['verified_goal_reached'] and r['strict_physical_visibility_pass']
                and not r['hard_measurement_failed_frames'] for r in reports),
            probe_result_sha256=args.probe_result_sha256,source_sha256=sources,launch_sha256=digest(OUTPUT/'launch.json'),
            prior_outcomes_changed=False,checkpoint_selection_performed=False,native_execution=False,
            independent_maze_evaluation=False,navigation_qualified=False,goal_achieved=False))
        print('AUXILIARY_DOWNWARD45_GOAL_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_AUXILIARY_DOWNWARD45_GOAL_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
