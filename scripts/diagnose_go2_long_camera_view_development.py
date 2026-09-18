"""Replay mapping and camera geometry during the longest recorded view task."""
import argparse
from collections import Counter, defaultdict
import json
import math
import time

import cv2
import numpy as np
import torch

from lewm.camera_frontier_viewpoint_development import directed_rotation, floor_cell_projection
from lewm.frontier_visit_runtime_development import wrap
from lewm.two_cm_floor_extent_development import configure
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.reconstruct_go2_frontier_stall_development import RecordedPoseMap


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root-name', required=True)
    root=path(parser.parse_args().root_name)
    if (root/'depth_retention.json').exists(): raise ValueError('retained raw depth required')
    events=[e for e in read(root,'frontier_visits.json')['events'] if 'camera_viewpoint' in e]
    event=max(events,key=lambda e:e['completed_ns']-e['started_ns'])
    output=root/'long_camera_view_geometry_replay_v1'; output.mkdir()
    cell=tuple(event['unknown_neighbour'])
    waypoint=np.asarray(event['camera_viewpoint']['viewpoint_map_xy_m'])
    poses={r['frame']:r['registered_pose'] for r in read(root,'poses.json')}
    plans=defaultdict(list)
    for p in read(root,'planning.json'):
        if 'selection' in p and event['started_ns']<=p['measured_ns']<event['completed_ns']:
            plans[p['map_frame']].append(p)
    updates=sorted((r for r in read(root,'stage_events.json')
        if r['stage']=='mapping' and r['frame']<=max(plans)),key=lambda r:r['completed_ns'])
    configure(); cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    reader=NoisyPublicReplay(root/'native'); mapper=RecordedPoseMap()
    rows=[]; started=time.monotonic()
    for index,update in enumerate(updates):
        frame=update['frame']; p,d,_,_,aux,now=reader.packet(frame)
        snapshot=mapper.update(p,d,poses[frame],auxiliary_depth=aux,measured_ns=now)
        B=np.asarray(snapshot.map_from_initial)
        for plan in plans[frame]:
            scope=plan['selection']['routing_memory_scope']
            if (scope['routing_floor_cells']!=len(snapshot.floor)
                    or scope['routing_fine_obstacle_cells']!=len(snapshot.fine_occupied)):
                raise ValueError('saved planning map counts differ')
            pose=poses[plan['frame']]
            p=B@np.asarray(pose['position_initial_body_m'])
            R=B@np.asarray(pose['rotation_initial_body_from_current_body'])
            planned,heading=directed_rotation(R,p[:2],cell)
            projected=floor_cell_projection(cell,p,planned,snapshot.floor_height)
            actual=floor_cell_projection(cell,p,R,snapshot.floor_height)
            distance=float(np.linalg.norm(p[:2]-waypoint))
            close=distance<=.10; feasible=any(r['fully_projected'] for r in projected)
            gate='view' if close and feasible else 'outside_arrival_radius' if not close else 'projection_infeasible'
            rows.append(dict(frame=plan['frame'],map_frame=frame,measured_ns=plan['measured_ns'],
                route_status=plan['route_status'],action=plan['action'],on_time=plan['on_time'],
                position_map_xy_m=p[:2].tolist(),waypoint_distance_m=distance,view_gate=gate,
                heading_error_rad=wrap(heading-math.atan2(R[1,0],R[0,0])),
                actual_patch_fully_projected=any(r['fully_projected'] for r in actual),
                patch_observed=cell in snapshot.floor|snapshot.occupied,
                planned_projection=projected,
                clearance_turn=plan['selection'].get('clearance_turn'),
                scan_heading_error_rad=plan['selection'].get('scan_heading_error_rad')))
        if index%100==0:
            print('VIEW_GEOMETRY_REPLAY',frame,'elapsed_s',round(time.monotonic()-started,1),flush=True)
    if len(rows)!=sum(map(len,plans.values())): raise ValueError('all episode plans required')
    gates=Counter(r['view_gate'] for r in rows)
    combinations=Counter((r['route_status'],r['view_gate']) for r in rows)
    report=dict(root_name=root.name,event=event,mapping_updates=len(updates),plans=len(rows),
        view_gate_counts=dict(gates),route_gate_counts=[dict(route=k[0],gate=k[1],plans=v) for k,v in combinations.items()],
        view_gate_switches=sum(a['view_gate']!=b['view_gate'] for a,b in zip(rows,rows[1:])),
        minimum_waypoint_distance_m=min(r['waypoint_distance_m'] for r in rows),
        maximum_waypoint_distance_m=max(r['waypoint_distance_m'] for r in rows),
        actual_projected_plans=sum(r['actual_patch_fully_projected'] for r in rows),
        aligned_view_plans=sum(r['view_gate']=='view' and abs(r['heading_error_rad'])<=.1 for r in rows),
        elapsed_s=time.monotonic()-started,original_registered_poses_used=True,
        raw_tracking_revalidated=False,delivered_noise_hashes_checked=True,native_state_used=False,
        alternative_trajectory_evaluated=False)
    for name,value in [('rows.json',rows),('result.json',report)]:
        with (output/name).open('x') as f: json.dump(value,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='event'}),flush=True)


if __name__=='__main__': main()
