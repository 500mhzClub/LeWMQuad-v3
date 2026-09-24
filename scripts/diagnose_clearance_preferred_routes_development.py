"""Compare route geometry on recorded public observations before a native trial."""
import argparse
import json
import time
from pathlib import Path
import cv2
import numpy as np
from scripts.in_memory_public_replay_development import PublicReplay
from scripts.diagnose_alignment_route_switches_development import RecordedPoseMap,saved_pose
from lewm.fine_stored_obstacle_routing_development import proposer,cached_clearance
from lewm.clearance_preferred_route_development import preferred_path
from lewm.clearance_lookahead_development import clear_route_target
from lewm.observed_floor_waypoint_development import centre


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root-name',required=True)
    args=parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary development basename required')
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')/args.root_name
    output=root/'clearance_preferred_routes_diagnostic.json'
    if output.exists():raise ValueError('preserve prior diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    frames=(400,500,600,624)
    poses={r['frame']:r['registered_pose'] for r in json.loads((root/'poses.json').read_text())}
    plans={r['frame']:r for r in json.loads((root/'planning.json').read_text()) if r.get('frame') in frames}
    required={p['map_frame'] for p in plans.values()}
    goal=json.loads((root/'launch.json').read_text())['public_mission']['goal_initial_body_xy_m']
    reader=PublicReplay(root/'native');mapper=RecordedPoseMap();mapper.retain_fine_obstacles=True
    snapshots={};rows=[]
    for frame in range(0,max(required)+1,4):
        policy,depth,_,_,aux,now=reader.packet(frame)
        snapshot=mapper.update(policy,depth,poses[frame],auxiliary_depth=aux,measured_ns=now)
        if frame in required:snapshots[frame]=snapshot
    for frame,plan in sorted(plans.items()):
        snapshot=snapshots[plan['map_frame']];B=np.asarray(snapshot.map_from_initial)
        p,_,_=saved_pose(poses[frame]);position=(B@p)[:2]
        route=proposer(snapshot)(snapshot.floor,snapshot.occupied,position,(B@np.r_[goal,0.])[:2])
        original=route['route_cells']
        if not original:
            rows.append(dict(frame=frame,status=route['status']));continue
        began=time.perf_counter()
        alternative,receipt=preferred_path(snapshot.floor,snapshot.occupied,original[0],original[-1])
        elapsed=time.perf_counter()-began
        geometry=cached_clearance(snapshot.fine_occupied)
        def measure(path):
            points=[centre(c) for c in path]
            distances=[geometry.minimum(a,b) for a,b in zip(points,points[1:])]
            target,lookahead=clear_route_target(points,position,snapshot.fine_occupied)
            return dict(route_cells=path,length_m=.05*(len(path)-1),
                minimum_clearance_m=min(distances) if distances else geometry.minimum(points[0],points[0]),
                median_clearance_m=float(np.median(distances)) if distances else None,
                lookahead=lookahead)
        rows.append(dict(frame=frame,map_frame=snapshot.frame,status=route['status'],
            position_map_xy_m=position.tolist(),original=measure(original),preferred=measure(alternative),
            added_routing_s=elapsed,receipt=receipt))
    report=dict(public_sensor_and_recorded_estimator_output_only=True,native_state_read=False,
        frontier_exclusions_replayed=False,comparison_uses_same_proposed_entry_and_target=True,
        prospective_navigation_benefit_unproven=True,rows=rows)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps([{k:v for k,v in row.items() if k not in ('original','preferred','receipt')}|
        {label:{k:v for k,v in row[label].items() if k not in ('route_cells','lookahead')} for label in ('original','preferred') if label in row}
        for row in rows]),flush=True)


if __name__=='__main__':main()
