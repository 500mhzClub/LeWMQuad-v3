"""Check recorded learned forecasts against the map available at their observation."""
from pathlib import Path
import json
import argparse
import cv2
import numpy as np
import torch
from scripts.diagnose_alignment_route_switches_development import RecordedPoseMap,saved_pose
from scripts.in_memory_public_replay_development import PublicReplay
from scripts.run_go2_paced_native_prefix_development import load_assigned,BASE
from lewm.fine_stored_obstacle_routing_development import fine_distances
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.delayed_action_planning_development import delayed_candidate_inputs
from lewm.observation_horizon_input_ablation_development import transform_inputs
from lewm.geometry_progress_pilot_development import ACTIONS,candidate_commands
from lewm.waypoint_alignment_planning_development import score_waypoint_alignment


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-name',default='go2_frontier_visit_native_layout00_v1_attempt_001')
    parser.add_argument('--frames',default='648,652,656,660,664,668,672')
    args=parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary artifact basename required')
    root=BASE/args.root_name
    output=root/'memory_forecast_clearance_diagnostic.json'
    if output.exists():raise ValueError('preserve diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    reader=PublicReplay(root/'native');mapper=RecordedPoseMap();mapper.retain_fine_obstacles=True
    admission=json.loads((BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'/'launch.json').read_text())['input_admission']['correction_admission']
    model,condition,variant=load_assigned(admission,'seed_2026091001_full_jepa')
    poses={r['frame']:r['registered_pose'] for r in json.loads((root/'poses.json').read_text())}
    requests={r['now_ns']:r['requested_command'] for r in json.loads((root/'requests.json').read_text())}
    targets={int(f) for f in args.frames.split(',')}
    if not targets or any(f<4 or f%4 for f in targets):raise ValueError('positive four-tick planning frames required')
    plans={r['frame']:r for r in json.loads((root/'planning.json').read_text()) if r['frame'] in targets}
    snapshots={};rows=[]
    for frame in range(0,max(targets)+1,4):
        policy,depth,_,_,aux,now=reader.packet(frame)
        snapshots[frame]=mapper.update(policy,depth,poses[frame],auxiliary_depth=aux,measured_ns=now)
        if frame not in targets:continue
        plan=plans[frame];snapshot=snapshots[plan['map_frame']]
        history=[reader.packet(f)[0] for f in range(frame-3,frame+1)]
        inputs=transform_inputs(delayed_candidate_inputs(causal_history_tensors(history,now),
            plan['committed_prefix'],delay_ticks=3,commit_ticks=4),input_variant=variant)
        out=model(**inputs)
        assert out['prediction_valid'].all()
        prediction=out['rollout_outcomes'].cpu().numpy()
        if 'motion_correction' in plan:
            from lewm.closed_loop_motion_residual_development import FrozenMotionResidual
            receipt=plan['motion_correction']
            assert np.array_equal(prediction[:,:,:2],np.asarray(receipt['raw_forecast_xy_m'],np.float32))
            prediction=FrozenMotionResidual().correct(prediction,poses,frame,plan['committed_prefix'])
            assert np.array_equal(prediction[:,:,:2],np.asarray(receipt['corrected_forecast_xy_m'],np.float32))
        selection=score_waypoint_alignment(prediction,plan['selection']['waypoint_body_xy_m'],delay_ticks=3,commit_ticks=4)
        errors=[abs(a['utility_m']-b['utility_m']) for a,b in zip(selection['candidates'],plan['selection']['candidates'])]
        assert max(errors)<1e-7
        p,R,_=saved_pose(poses[frame]);B=np.asarray(snapshot.map_from_initial);Q,q=B@R,B@p
        cells=np.asarray(sorted(snapshot.fine_occupied),int).reshape(-1,2)
        candidates=[]
        for i,action in enumerate(ACTIONS):
            points=q[:2]+np.vstack((np.zeros(2),prediction[i,:,:2]))@Q[:2,:2].T
            minima=[]
            for begin,end in zip(points[:-1],points[1:]):
                distances=fine_distances(begin,end,cells)
                minima.append(None if not len(distances) else float(distances.min()))
            candidates.append(dict(action=action,minimum_distances_m=minima,
                predicted_commit_collision=any(d is not None and d<=.45 for d in minima[3:7])))
        distances=fine_distances(q[:2],q[:2],cells)
        start_clearance=None if not len(distances) else float(distances.min())
        expected=plan['committed_prefix']+[candidate_commands(plan['action'])[0]]*4
        matches=[all(ns in requests and np.allclose(requests[ns],command,atol=1e-8,rtol=0)
            for ns in range(now+i*100_000_000,now+(i+1)*100_000_000,20_000_000)) for i,command in enumerate(expected)]
        actual=np.array([(B@np.asarray(poses[f]['position_initial_body_m']))[:2] for f in range(frame,frame+8)])
        actual_clearances=[]
        for begin,end in zip(actual[:-1],actual[1:]):
            distances=fine_distances(begin,end,cells)
            actual_clearances.append(None if not len(distances) else float(distances.min()))
        selected_index=ACTIONS.index(plan['action'])
        selected_predicted=q[:2]+prediction[selected_index,:7,:2]@Q[:2,:2].T
        rows.append(dict(frame=frame,map_frame=snapshot.frame,recorded_action=plan['action'],
            route_status=plan['route_status'],start_clearance_m=start_clearance,
            requested_forecast_intervals_executed=matches,
            subsequent_observed_path_clearances_against_same_map_m=actual_clearances,
            selected_prediction_position_errors_m=np.linalg.norm(actual[1:]-selected_predicted,axis=1).tolist(),
            maximum_replayed_utility_difference=max(errors),candidates=candidates,prediction=prediction.tolist()))
    report=dict(rows=rows,public_sensors_and_recorded_estimator_map_only=True,native_state_read=False,
        future_native_trajectory_used=False,subsequent_estimator_positions_used_only_after_forecasting=True,
        requested_forecast_intervals_checked=True,counterfactual_navigation_success_claimed=False)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('MEMORY_CLEARANCE_DIAGNOSTIC',[(r['frame'],r['recorded_action'],r['start_clearance_m'],
        [c['action'] for c in r['candidates'] if not c['predicted_commit_collision']]) for r in rows],flush=True)


if __name__=='__main__':main()
