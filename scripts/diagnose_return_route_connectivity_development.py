"""Compare coarse routing and exact fine-cell connectivity on saved public maps."""
import argparse
import heapq
import json
from pathlib import Path
import cv2
import numpy as np
from scripts.in_memory_public_replay_development import PublicReplay
from scripts.diagnose_alignment_route_switches_development import RecordedPoseMap
from lewm.fine_stored_obstacle_routing_development import proposer,cached_clearance
from lewm.observed_floor_waypoint_development import centre,NEIGHBOURS


def fine_path(snapshot,position,goal):
    geometry=cached_clearance(snapshot.fine_occupied)
    def clear(a,b):
        d=geometry.minimum(a,b)
        return d is None or d>.45+1e-12
    candidates=sorted((c for c in snapshot.floor if np.linalg.norm(centre(c)-position)<=1.25),
        key=lambda c:(float(np.linalg.norm(centre(c)-position)),c))
    seed=next((c for c in candidates if clear(position,centre(c))),None)
    target=tuple(map(int,np.floor(goal/.05)))
    if seed is None:return dict(status='NO_FINE_CLEAR_CONNECTOR')
    def heuristic(c):return abs(c[0]-target[0])+abs(c[1]-target[1])
    queue=[(heuristic(seed),0,seed)];distance={seed:0};parent={seed:None};edges={}
    while queue:
        _,cost,cell=heapq.heappop(queue)
        if cost!=distance[cell]:continue
        if cell==target and clear(centre(cell),goal):
            path=[];here=cell
            while here is not None:path.append(here);here=parent[here]
            path=path[::-1];points=[position]+[centre(c) for c in path]+[goal]
            return dict(status='EXACT_FINE_ROUTE_FOUND',route_cells=path,
                minimum_continuous_clearance_m=min(geometry.minimum(a,b) for a,b in zip(points,points[1:])),
                route_length_m=sum(float(np.linalg.norm(b-a)) for a,b in zip(points,points[1:])),
                reached_cells=len(parent))
        for dx,dy in NEIGHBOURS:
            nxt=(cell[0]+dx,cell[1]+dy)
            if nxt not in snapshot.floor or cost+1>=distance.get(nxt,float('inf')):continue
            edge=tuple(sorted((cell,nxt)))
            if edge not in edges:edges[edge]=clear(centre(cell),centre(nxt))
            if not edges[edge]:continue
            distance[nxt]=cost+1;parent[nxt]=cell
            heapq.heappush(queue,(cost+1+heuristic(nxt),cost+1,nxt))
    return dict(status='NO_EXACT_FINE_ROUTE',reached_cells=len(parent),goal_cell_observed=target in snapshot.floor)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    parser.add_argument('--frames',type=int,nargs='+',required=True)
    parser.add_argument('--mission-phase',choices=['RETURN','OUTBOUND'],default='RETURN')
    args=parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):raise ValueError('ordinary development root required')
    root=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')/args.root_name
    prefix=args.mission_phase.lower()
    output=root/f'{prefix}_route_connectivity_diagnostic.json'
    if output.exists():raise ValueError('preserve diagnostic')
    plans={r['frame']:r for r in json.loads((root/'planning.json').read_text()) if r['frame'] in args.frames}
    if set(plans)!=set(args.frames):raise ValueError('recorded planning frames required')
    poses={r['frame']:r['registered_pose'] for r in json.loads((root/'poses.json').read_text())}
    events=json.loads((root/'stage_events.json').read_text())
    required={p['map_frame'] for p in plans.values()};snapshots={}
    update_frames=sorted(r['frame'] for r in events if r['stage']=='mapping' and r['frame']<=max(required))
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False)
    reader=PublicReplay(root/'native');mapper=RecordedPoseMap();mapper.retain_fine_obstacles=True
    for frame in update_frames:
        policy,depth,_,_,aux,now=reader.packet(frame)
        snapshot=mapper.update(policy,depth,poses[frame],auxiliary_depth=aux,measured_ns=now)
        if frame in required:snapshots[frame]=snapshot
        if frame%400==0:print('MAP_REPLAY',frame,flush=True)
    rows=[]
    goal_key='return_initial_body_xy_m' if args.mission_phase=='RETURN' else 'goal_initial_body_xy_m'
    goal_initial=json.loads((root/'launch.json').read_text())['public_mission'][goal_key]
    for frame,plan in sorted(plans.items()):
        snapshot=snapshots[plan['map_frame']];B=np.asarray(snapshot.map_from_initial)
        position=(B@np.asarray(poses[frame]['position_initial_body_m']))[:2]
        goal=(B@np.r_[goal_initial,0.])[:2]
        coarse=proposer(snapshot)(snapshot.floor,snapshot.occupied,position,goal)
        exact=fine_path(snapshot,position,goal)
        rows.append(dict(frame=frame,map_frame=snapshot.frame,position_map_xy_m=position.tolist(),
            recorded_route_status=plan['route_status'],coarse_replay_status=coarse['status'],
            fine_connectivity=exact,floor_cells=len(snapshot.floor),occupied_cells=len(snapshot.occupied)))
        np.savez_compressed(root/f'{prefix}_map_snapshot_{frame}.npz',floor=np.asarray(sorted(snapshot.floor)),
            occupied=np.asarray(sorted(snapshot.occupied)),fine_occupied=np.asarray(sorted(snapshot.fine_occupied)),
            position=position,goal=goal)
        print('CONNECTIVITY',json.dumps(rows[-1]),flush=True)
    report=dict(public_sensor_and_recorded_pose_only=True,native_state_used=False,mission_phase=args.mission_phase,
        recorded_mapping_update_frames_replayed=True,nominal_radius_m=.45,
        fine_routes_are_diagnostic_only=True,rows=rows)
    with output.open('x') as f:json.dump(report,f,indent=2)


if __name__=='__main__':main()
