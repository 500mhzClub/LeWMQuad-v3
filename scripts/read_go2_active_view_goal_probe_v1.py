"""Receipt-bound actual view/waypoint phase and native goal outcome readout."""
import argparse
from collections import Counter
import math
import numpy as np
from scripts.run_go2_active_view_goal_probe_v1 import OUTPUT as INPUT, CASES
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_learned_goal_bootstrap_probe_v1 import timing

OUTPUT=BASE/'go2_active_view_goal_readout_v1_attempt_001'
PROTOCOL='docs/go2_active_view_goal_readout_v1_2026-09-08.md'


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
    return dict(case=name,collection=record['collection'],goal=record['goal'],
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
        navigation_qualified=False,physical_backtracking_demonstrated=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--probe-result-sha256',required=True);args=parser.parse_args()
    if not __debug__:raise ValueError('audit assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive active-view readout')
    verify_artifacts(INPUT,{'result.json':args.probe_result_sha256})
    result=read_json(INPUT,'result.json')
    assert result['status']=='ACTIVE_VIEW_GOAL_PROBE_COMPLETE' and result['cases']==[list(c) for c in CASES]
    bindings={'result.json':args.probe_result_sha256}|result['artifact_sha256'];verify_artifacts(INPUT,bindings)
    original=read_json(INPUT,'launch.json')
    sources=discover_sources((PROTOCOL,'scripts/read_go2_active_view_goal_probe_v1.py'),original['source_sha256'])
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),
        input_artifact_sha256=bindings,native_execution=False)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    try:
        reports=[summarize(r) for r in result['conditions']]
        verify(launch);verify_artifacts(INPUT,bindings)
        write_json(OUTPUT/'result.json',dict(status='ACTIVE_VIEW_GOAL_READOUT_COMPLETE',conditions=reports,
            source_sha256=sources,launch_sha256=digest(OUTPUT/'launch.json'),probe_result_sha256=args.probe_result_sha256,
            original_outcomes_changed=False,native_execution=False,navigation_qualified=False,goal_achieved=False))
        print('ACTIVE_VIEW_READOUT_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_ACTIVE_VIEW_READOUT_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()
