"""Read closed controller receipts while the separate native raw audit continues."""
from collections import Counter
import numpy as np
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from scripts.run_go2_residual_first_interval_maze_pilot_v1 import OUTPUT as INPUT, CASE
from scripts.maze_decision_stream_development import read_rows, NAME as DECISIONS
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT=BASE/'go2_residual_maze02_stagnation_v1_attempt_001'
PROTOCOL='docs/go2_residual_maze02_stagnation_v1_2026-09-09.md'
COLLECTION_SHA='f7a946687f108c354a7c6f4e5155530875ca0da7bcd7e57f9564ff89c23fa5b4'
SNAPSHOTS=(463,503,1000,2000,2990)


def describe_selection(s):
    if not s or 'prediction' not in s: return s
    p=np.asarray(s['prediction'],float)
    if p.shape!=(6,8,5) or not np.isfinite(p).all(): raise ValueError('complete finite saved predictions required')
    fallback=s.get('residual_first_interval_feasibility'); candidates=[]
    for i,action in enumerate(ACTIONS):
        candidate=s['candidates'][i]; path=s['nominal_path_checks'][i]
        if candidate['action']!=action: raise ValueError('original candidate order required')
        distances=[segment['minimum_observed_cell_distance_m'] for segment in path['segments']
            if segment['minimum_observed_cell_distance_m'] is not None]
        candidates.append(dict(action=action,utility_m=candidate['utility_m'],
            raw_first_xy_m=p[i,0,:2].tolist(),raw_last_xy_m=p[i,-1,:2].tolist(),
            first_contact_score=float(np.exp(-np.logaddexp(0.,-p[i,0,4]))),
            last_contact_score=float(np.exp(-np.logaddexp(0.,-p[i,-1,4]))),
            first_predicted_yaw_rad=float(np.arctan2(p[i,0,2],p[i,0,3])),
            causal_scoring_body_xy_m=candidate.get('causal_scoring_body_xy_m'),
            executed_waypoint_distance_progress_m=candidate.get('executed_waypoint_distance_progress_m'),
            executed_waypoint_alignment_progress_m=candidate.get('executed_waypoint_alignment_progress_m'),
            phase_allowed=action in s['phase_allowed_actions'],
            original_surface_clear=not s['surface_checks'][i]['possible_intersection'],
            original_nominal_path_clear=path['all_predicted_segments_nominally_clear'],
            original_minimum_observed_clearance_m=min(distances) if distances else None,
            corrected_fallback_eligible=None if fallback is None else action in fallback['eligible_actions']))
    return dict(action=s['action'],mode=s['mode'],score_contract=s['score_contract'],
        waypoint_map_xy_m=s.get('waypoint_map_xy_m'),goal_body_xy_m=s['goal_body_xy_m'],
        proposal_status=s['proposal']['status'],route_cells=s['proposal']['route_cells'],
        scored_pose_horizon_ns=s.get('scored_pose_horizon_ns'),scored_contact_horizon_ns=s.get('scored_contact_horizon_ns'),
        path_constraint_horizon_ns=s.get('path_constraint_horizon_ns'),
        residual_correction_xy_m=s.get('causal_score_residual_receipt',{}).get('correction_xy_m'),
        residual_source_ticks=s.get('causal_score_residual_receipt',{}).get('residual_source_ticks'),
        phase_admissible_candidates=s['phase_admissible_candidates'],fallback_present=fallback is not None,
        candidates=candidates)


def main():
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive closed-stream diagnosis required')
    name=CASE[0]; ids={name+'/result.json':COLLECTION_SHA}
    verify_artifacts(INPUT,ids); collection=read_json(INPUT/name,'result.json')
    if (collection['status']!='RESIDUAL_FIRST_INTERVAL_MAZE_TERMINAL_AUDIT_REQUIRED'
            or collection['decisions']!=3014 or collection['command_ticks']!=3013
            or collection['schedule_terminal']!='MISSION_TICK_BUDGET_EXHAUSTED'):
        raise ValueError('exact closed original collection required; raw audit remains separate')
    ids[name+'/'+DECISIONS]=digest(INPUT/name/DECISIONS); ids['launch.json']=digest(INPUT/'launch.json')
    verify_artifacts(INPUT,ids); old=read_json(INPUT,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/diagnose_go2_residual_maze02_stagnation_v1.py'),old['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<RESERVE_BYTES+64*1024**2:
        raise ValueError('bounded closed-stream diagnosis resources unavailable')
    launch=old|dict(protocol=PROTOCOL,source_sha256=sources,output_root=str(OUTPUT),closed_collection_bindings=ids,
        snapshots=list(SNAPSHOTS),hardware=resources,native_execution=False,native_scene_workers=0,
        model_loaded=False,model_training=False,minimum_available_ram_bytes=8*1024**3,
        raw_audit_complete=False,scientific_native_outcomes_verified=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json',launch)
    print('RESIDUAL_STAGNATION_DIAGNOSIS_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    try:
        selected=Counter(); requests=Counter(); snapshots=[]; runs=[]; start=None; frames=0; first_terminal=None; fallback_attempts=0
        for row in read_rows(INPUT/name):
            i=row['tick']; d=row['decision']; s=d['new_selection'] or {}; frames+=1
            if d['terminal'] is not None and first_terminal is None:
                first_terminal=dict(frame=i,terminal=d['terminal'],failure=d['failure'])
            if d['terminal'] is None:
                requests[tuple(d['requested_command'])]+=1
                if s.get('action') is not None: selected[s['action']]+=1
            if s.get('residual_first_interval_feasibility') is not None: fallback_attempts+=1
            holding=d['terminal'] is None and s.get('action')=='hold'
            if holding and start is None: start=i
            if not holding and start is not None:
                runs.append(dict(first_frame=start,last_frame=i-1,observations=i-start)); start=None
            if i in SNAPSHOTS:
                evidence=d.get('evidence') or {}; pose=evidence.get('current_pose') or {}
                snapshots.append(dict(frame=i,requested_command=d['requested_command'],terminal=d['terminal'],
                    observed_goal_distance_m=d['observed_goal_distance_m'],
                    observed_position_initial_body_m=pose.get('position_initial_body_m'),
                    selection=describe_selection(s)))
        if start is not None: runs.append(dict(first_frame=start,last_frame=frames-1,observations=frames-start))
        if frames!=collection['decisions'] or [r['frame'] for r in snapshots]!=list(SNAPSHOTS):
            raise ValueError('complete closed decision population and fixed snapshots required')
        verify(launch); verify_artifacts(INPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='RESIDUAL_MAZE02_STAGNATION_DIAGNOSIS_V1_COMPLETE',
            source_sha256=sources,launch_sha256=digest(OUTPUT/'launch.json'),closed_collection_bindings=ids,
            frames=frames,first_terminal=first_terminal,selected_actions=dict(selected),
            active_requested_commands=[dict(command=list(k),observations=v) for k,v in sorted(requests.items())],
            fallback_attempts=fallback_attempts,hold_runs=runs,snapshots=snapshots,
            model_loaded=False,native_execution=False,raw_audit_complete=False,
            scientific_native_outcomes_verified=False,policy_rescoring_performed=False,
            unexecuted_outcomes_inferred=False,navigation_qualified=False,goal_achieved=False))
        print('RESIDUAL_STAGNATION_DIAGNOSIS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_RESIDUAL_STAGNATION_DIAGNOSIS_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()
