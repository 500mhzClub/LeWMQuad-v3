"""Receipt-bound actual view/waypoint phase and native goal outcome readout."""
import argparse
from collections import Counter
import hashlib
import math
import numpy as np
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from scripts.run_go2_nominal_action_goal_probe_v1 import OUTPUT as INPUT, CASES
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing
from scripts.run_go2_continuous_connector_goal_probe_v1 import OUTPUT as ORIGINAL

OUTPUT=BASE/'go2_nominal_action_goal_readout_v1_attempt_001'
PROTOCOL='docs/go2_nominal_action_goal_readout_v1_2026-09-08.md'
ORIGINAL_SHA='ee58cdda5586b0558665000d24fbc14c3858adad6b0efd052fbce466739c9af9'


def summarize(record):
    name=record['case'];rows=read_json(INPUT/name,'context_decisions.json')
    tape=read_json(INPUT/name,'command_tape.json');audit=read_json(INPUT,name+'_audit.json')
    selections=[dict(tick=r['tick'],**r['decision']['new_selection']) for r in rows if r['decision']['new_selection'] is not None]
    forecasts=[s for s in selections if 'prediction' in s]
    transitions=[];last=None;headings=[]
    for row in rows:
        decision=row['decision'];mode=decision['planner_mode']
        if mode!=last:transitions.append(dict(tick=row['tick'],mode=mode));last=mode
        receipt=decision['memory_receipt'];evidence=decision['evidence']
        if receipt is not None and evidence is not None and evidence['current_pose'] is not None:
            R=np.asarray(receipt['map_from_initial'])@np.asarray(evidence['current_pose']['rotation_initial_body_from_current_body'])
            headings.append(math.atan2(R[1,0],R[0,0]))
    commands=Counter();nonzero=Counter()
    for item in tape:
        d=rows[item['tick']]['decision']
        if item['completed'] and d['terminal'] is None:
            commands[d['planner_mode']]+=1
            nonzero[d['planner_mode']]+=int(any(item['requested_command']))
    receipts=[r['decision']['memory_receipt'] for r in rows if r['decision']['memory_receipt'] is not None]
    first_failure=None
    for row in rows:
        d=row['decision']
        if d['failure']:
            continuity=(d['evidence'] or {}).get('continuity_evidence') or {}
            first_failure=dict(tick=row['tick'],failure=d['failure'],
                continuity={k:continuity.get(k) for k in ('status','anchor_available','incremental_available',
                    'anchor_failure','incremental_failure','bridge_frames')})
            break
    return dict(case=name,model_name=record['model_name'],condition=record['condition'],variant=record['variant'],
        trial=record['trial'],collection=record['collection'],goal=record['goal'],first_failure=first_failure,
        selections=len(selections),model_forecasts=len(forecasts),
        phase_transitions=transitions,completed_active_command_intervals=dict(commands),
        nonzero_active_command_intervals=dict(nonzero),
        selected_actions=dict(Counter(str(s['action']) for s in selections)),
        forecast_phase_counts=dict(Counter(s['mode'] for s in forecasts)),
        floor_proposal_status_counts=dict(Counter(s['proposal']['status'] for s in selections)),
        selection_trace=[{k:s[k] for k in ('tick','mode','action','scan_target_map_yaw_rad',
            'scan_heading_delta_rad','scan_index','scan_sign','waypoint_map_xy_m','phase_admissible_candidates') if k in s}
            for s in selections],
        first_map_receipt=receipts[0] if receipts else None,last_map_receipt=receipts[-1] if receipts else None,
        measured_map_heading_range_rad=[min(headings),max(headings)] if headings else None,
        raw_sensor_reconstruction_pass=audit['raw_sensor_reconstruction_pass'],
        raw_model_command_replay_pass=audit['raw_model_command_replay_pass'],
        model_state_unchanged=audit['model_state_unchanged'],
        strict_physical_visibility_pass=audit['strict_physical_visibility_pass'],
        hard_measurement_failed_frames=audit['hard_measurement_failed_frames'],
        maximum_observed_pose_xy_error_m=max(audit['observed_pose_xy_errors_m'],default=None),
        timing={k:timing([r[k] for r in rows if k in r]) for k in ('acquisition_wall_ms',
            'controller_wall_ms','observation_and_control_wall_ms','iteration_with_command_wall_ms')},
        nominal_constraint_changed_selections=sum(s['action']!=s['before_nominal_constraint_action'] for s in forecasts),
        no_jointly_feasible_action_selections=sum(s['action'] is None for s in forecasts),
        nominal_feasibility_trace=[dict(tick=s['tick'],action=s['action'],before=s['before_nominal_constraint_action'],
            feasible=s['phase_admissible_candidates'],
            measured_start_clearance_m=s['proposal']['start_clearance']['minimum_observed_cell_distance_m'],
            selected_predicted_clearance_m=None if s['action_index'] is None else s['nominal_action_checks'][s['action_index']]['minimum_observed_cell_distance_m'],
            nominal_rejected=[c['action'] for c in s['nominal_action_checks'] if not c['nominal_disk_connector_clear']],
            surface_rejected=[ACTIONS[i] for i,c in enumerate(s['surface_checks']) if c['possible_intersection']]) for s in forecasts],
        every_selected_forecast_passes_nominal_and_surface=all(s['action_index'] is None or (
            s['nominal_action_checks'][s['action_index']]['nominal_disk_connector_clear'] and
            not s['surface_checks'][s['action_index']]['possible_intersection']) for s in forecasts),
        navigation_qualified=False,physical_backtracking_demonstrated=False)


def compare_prior(case):
    old,new=ORIGINAL/case,INPUT/case
    old_rows,new_rows=[read_json(root,'context_decisions.json') for root in (old,new)]
    old_tape,new_tape=[read_json(root,'command_tape.json') for root in (old,new)]
    first=next((i for i,(a,b) in enumerate(zip(old_tape,new_tape))
        if a['requested_command']!=b['requested_command']),None)
    length_only=first is None and len(old_tape)!=len(new_tape)
    if length_only:first=min(len(old_tape),len(new_tape))
    frames=min(len(old_rows),len(new_rows),first+1 if first is not None else max(len(old_rows),len(new_rows)))
    end=749+50*(frames-1)
    def observations(rows):
        return [{k:r['decision'][k] for k in ('evidence','memory_receipt','observed_goal_distance_m')}
            for r in rows[:frames]]
    def predictions(rows):
        return [None if r['decision']['new_selection'] is None else r['decision']['new_selection'].get('prediction')
            for r in rows[:frames]]
    witnesses={}
    for filename,count in (('physics_trace.npz',end+1),('policy_histories.npz',frames),('fast_gyro_histories.npz',frames)):
        pair=[]
        for root in (old,new):
            with np.load(root/filename,allow_pickle=False) as z:
                arrays={k:z[k][:count] for k in z.files}
                assert all(len(v)==count for v in arrays.values())
                pair.append({k:dict(dtype=v.dtype.str,shape=list(v.shape),
                    sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()})
        witnesses[filename]=dict(original=pair[0],current=pair[1],prefix_array_identity=pair[0]==pair[1])
    rgb=[[r['rgb_sha256'] for r in read_json(root,'camera_audit.json')[:frames]] for root in (old,new)]
    return dict(case=case,first_requested_command_difference=first,difference_is_tape_length_only=length_only,
        common_command_prefix_observation_frames=frames,last_common_observation_native_sample=end,
        common_prefix_rgb_identical=rgb[0]==rgb[1],
        common_prefix_observer_map_evidence_identical=observations(old_rows)==observations(new_rows),
        common_prefix_all_model_predictions_identical=predictions(old_rows)==predictions(new_rows),
        arrays=witnesses,command_tapes_identical=old_tape==new_tape,
        common_prefix_compared_before_differing_command_execution=True,
        unexecuted_outcomes_inferred=False,uncontended_timing_comparison=False)


def selected_nominal_witnesses(case):
    """A conflict with one already observed square proves nominal path conflict.

    A miss cannot establish clearance from the other occupied squares. Future
    measured endpoints are descriptive executed outcomes, never planner inputs.
    """
    rows=read_json(INPUT/case,'context_decisions.json');tape=read_json(INPUT/case,'command_tape.json')
    records=[]
    for row in rows:
        d=row['decision'];s=d['new_selection'];tick=row['tick']
        if s is None or 'prediction' not in s or s['action'] is None:continue
        cell=s['proposal']['start_clearance']['nearest_observed_cell']
        if cell is None:continue
        pose=d['evidence']['current_pose'];B=np.asarray(d['memory_receipt']['map_from_initial'])
        R=np.asarray(pose['rotation_initial_body_from_current_body']);p=np.asarray(pose['position_initial_body_m'])
        start=(B@p)[:2];prediction=s['prediction'][ACTIONS.index(s['action'])][0]
        end=(B@(p+R@np.r_[prediction[:2],0.]))[:2]
        forecast=nominal_connector(start,end,[cell]);actual=None
        if tick+5<len(rows) and tick+5<=len(tape):
            interval=tape[tick:tick+5];expected=candidate_commands(s['action'])[:5]
            completed=all(t['completed'] and t['requested_command']==list(c) for t,c in zip(interval,expected,strict=True))
            later=rows[tick+5]['decision']['evidence']
            if completed and later is not None and later['current_pose'] is not None and later['terminal_failure'] is None:
                endpoint=(B@np.asarray(later['current_pose']['position_initial_body_m']))[:2]
                actual=dict(endpoint_map_xy_m=endpoint.tolist(),
                    endpoint_error_m=float(np.linalg.norm(endpoint-end)),
                    straight_chord_to_measured_endpoint=nominal_connector(start,endpoint,[cell]),
                    intermediate_motion_clearance_certified=False)
        records.append(dict(tick=tick,mode=s['mode'],action=s['action'],observed_square=cell,
            predicted_endpoint_map_xy_m=end.tolist(),predicted_chord=forecast,executed_endpoint=actual,
            all_occupied_squares_checked=False,articulated_collision_proved=False))
    return dict(case=case,records=records,
        predicted_chord_nominal_conflicts=sum(not r['predicted_chord']['nominal_disk_connector_clear'] for r in records),
        waypoint_predicted_chord_nominal_conflicts=sum(r['mode']=='WAYPOINT' and not r['predicted_chord']['nominal_disk_connector_clear'] for r in records),
        measured_endpoint_chord_nominal_conflicts=sum(r['executed_endpoint'] is not None and not r['executed_endpoint']['straight_chord_to_measured_endpoint']['nominal_disk_connector_clear'] for r in records),
        commands_changed=False,native_pose_input=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe-result-sha256',required=True);args=parser.parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive active-view readout')
    verify_artifacts(INPUT,{'result.json':args.probe_result_sha256})
    result=read_json(INPUT,'result.json')
    assert result['status']=='NOMINAL_ACTION_GOAL_PROBE_COMPLETE' and result['cases']==[list(c) for c in CASES]
    bindings={'result.json':args.probe_result_sha256}|result['artifact_sha256'];verify_artifacts(INPUT,bindings)
    verify_artifacts(ORIGINAL,{'result.json':ORIGINAL_SHA});old=read_json(ORIGINAL,'result.json')
    assert old['status']=='CONTINUOUS_CONNECTOR_GOAL_PROBE_COMPLETE'
    old_ids={'result.json':ORIGINAL_SHA}|old['artifact_sha256'];verify_artifacts(ORIGINAL,old_ids)
    original=read_json(INPUT,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/read_go2_nominal_action_goal_probe_v1.py'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        input_artifact_sha256=bindings,original_probe_artifact_sha256=old_ids,native_execution=False)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        reports=[summarize(r) for r in result['conditions']]
        grouped=[]
        for model in sorted(set(r['model_name'] for r in reports)):
            cases=[r for r in reports if r['model_name']==model]
            assert len(cases)==2
            grouped.append(dict(model=model,cases=[r['case'] for r in cases],
                native_verified_goals=sum(r['goal']['verified_goal_reached'] for r in cases),
                measurement_clean_verified_goals=sum(r['goal']['verified_goal_reached'] and not r['hard_measurement_failed_frames'] for r in cases),
                physical_stops=sum(r['collection']['physical_stop'] is not None for r in cases)))
        repeat=[compare_prior(c[0]) for c in CASES]
        nominal=[selected_nominal_witnesses(c[0]) for c in CASES]
        verify(launch);verify_artifacts(INPUT,bindings);verify_artifacts(ORIGINAL,old_ids)
        write_json(OUTPUT/'result.json',dict(status='NOMINAL_ACTION_GOAL_READOUT_COMPLETE',conditions=reports,models=grouped,
            prior_controller_comparisons=repeat,
            selected_action_nominal_witnesses=nominal,
            source_sha256=sources,launch_sha256=digest(OUTPUT/'launch.json'),probe_result_sha256=args.probe_result_sha256,
            original_outcomes_changed=False,native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('NOMINAL_ACTION_GOAL_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_NOMINAL_ACTION_GOAL_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
