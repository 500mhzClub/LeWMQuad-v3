"""Reconstruct map proposals from saved public sensors and recorded estimator poses."""
from pathlib import Path
import json
import argparse
import time
import cv2
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.multirate_routing_map_development import MultirateRoutingMap
from lewm.vectorized_connector_routing_development import propose
from lewm.observed_floor_waypoint_development import inflated_cells
from scripts.in_memory_public_replay_development import PublicReplay


def saved_pose(evidence, **kwargs):
    return (np.asarray(evidence['position_initial_body_m']),
        np.asarray(evidence['rotation_initial_body_from_current_body']), evidence)


class RecordedPoseMap(MultirateRoutingMap):
    update=bind(MultirateRoutingMap.update,current_measured_floor_pose=saved_pose)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--continuous-connector',action='store_true')
    parser.add_argument('--fine-stored',action='store_true')
    parser.add_argument('--root-name',default='go2_waypoint_alignment_round_trip_native_layout00_v1_attempt_001')
    args=parser.parse_args()
    proposer=propose
    if args.continuous_connector:
        from lewm.continuous_start_connector_development import propose as proposer
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary artifact basename required')
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')/args.root_name
    output=root/('fine_stored_route_switch_diagnostic.json' if args.fine_stored else
        'continuous_route_switch_diagnostic.json' if args.continuous_connector else 'route_switch_diagnostic.json')
    if output.exists():raise ValueError('preserve diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    reader=PublicReplay(root/'native');mapper=RecordedPoseMap()
    mapper.retain_fine_obstacles=args.fine_stored
    poses={r['frame']:r['registered_pose'] for r in json.loads((root/'poses.json').read_text())}
    plans={r['frame']:r for r in json.loads((root/'planning.json').read_text()) if 'selection' in r}
    snapshots={};rows=[]
    for frame in range(0,max(poses)+1,4):
        policy,depth,_,_,aux,now=reader.packet(frame)
        snapshots[frame]=mapper.update(policy,depth,poses[frame],auxiliary_depth=aux,measured_ns=now)
        if frame not in plans:continue
        plan=plans[frame];snapshot=snapshots[plan['map_frame']]
        B=np.asarray(snapshot.map_from_initial);p,R,_=saved_pose(poses[frame]);q=B@p
        began=time.perf_counter()
        active_proposer=proposer
        if args.fine_stored:
            from lewm.fine_stored_obstacle_routing_development import proposer as fine_proposer
            active_proposer=fine_proposer(snapshot)
        route=active_proposer(snapshot.floor,snapshot.occupied,q[:2],(B@np.array([0.,2.6,0.]))[:2])
        route_s=time.perf_counter()-began
        keys=np.asarray(list(snapshot.occupied),float)
        delta=np.maximum(np.maximum(keys*.05-q[:2],q[:2]-(keys+1)*.05),0.)
        distance=float(np.linalg.norm(delta,axis=1).min()) if len(keys) else None
        blocked=inflated_cells(snapshot.occupied)
        rows.append(dict(frame=frame,planned_status=plan['route_status'],replayed_status=route['status'],route_s=route_s,
            pose_map_xy_m=q[:2].tolist(),pose_cell_blocked=tuple(np.floor(q[:2]/.05).astype(int)) in blocked,
            nearest_occupied_cell_distance_m=distance,route=route))
    report=dict(public_sensor_and_recorded_estimator_output_only=True,native_state_read=False,
        rows=rows,missing_routes=sum(r['replayed_status']=='ADDITIONAL_VIEW_REQUIRED' for r in rows),
        missing_routes_pose_cell_blocked=sum(r['replayed_status']=='ADDITIONAL_VIEW_REQUIRED' and r['pose_cell_blocked'] for r in rows))
    with output.open('x') as f:json.dump(report,f,indent=2)
    print('ROUTE_SWITCH_DIAGNOSTIC',json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)


if __name__=='__main__':main()
