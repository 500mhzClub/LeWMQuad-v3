"""Replay early recorded maps and test current-position viewing of a blocked patch."""
from collections import Counter
import json

import cv2
import numpy as np
import torch

from lewm.camera_frontier_viewpoint_development import directed_rotation, floor_cell_projection
from lewm.observed_floor_waypoint_development import centre, segment_cells
from scripts.replay_go2_no_early_release_map_entry_development import RecordedCurrentPlaneMap
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.run_go2_coverage_translation_view_development import ROOT, collection


def main():
    root = collection.study.BASE/ROOT
    output = root/'coverage_view_stall_diagnosis_v1.json'
    if output.exists(): raise ValueError('preserve completed diagnosis')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    configure()
    read = lambda name: json.loads((root/name).read_text())
    plans = [r for r in read('planning.json') if 'selection' in r]
    poses = {r['frame']:r['registered_pose'] for r in read('poses.json')}
    frames = sorted({r['frame'] for r in read('stage_events.json')
        if r['stage']=='mapping' and r['frame']<=400})
    mapper = RecordedCurrentPlaneMap(); reader = NoisyPublicReplay(root/'native')
    snapshots = {}
    for frame in frames:
        policy, depth, _, _, auxiliary, now = reader.packet(frame)
        snapshots[frame] = mapper.update(policy,depth,poses[frame],
            auxiliary_depth=auxiliary,measured_ns=now)
    rows = []
    for plan in plans:
        request = plan['selection'].get('coverage_view_request')
        if not request or plan['frame']>400: continue
        snap = snapshots[plan['map_frame']]
        scope = plan['selection']['routing_memory_scope']
        assert scope['retained_floor_cells']==len(snap.floor)
        assert scope['retained_fine_obstacle_cells']==len(snap.fine_occupied)
        pose=poses[plan['frame']]; B=np.asarray(snap.map_from_initial)
        p=B@np.asarray(pose['position_initial_body_m'])
        R=B@np.asarray(pose['rotation_initial_body_from_current_body'])
        target=tuple(request['target_cell'])
        directed,heading=directed_rotation(R,p[:2],target)
        projection=floor_cell_projection(target,p,directed,snap.floor_height)
        for camera in projection:
            camera['stored_obstacle_blocks_ray']=bool(segment_cells(
                camera['camera_origin_map_xy_m'],centre(target)) & snap.occupied)
        visit=request.get('visit')
        rows.append(dict(frame=plan['frame'],map_frame=snap.frame,action=plan['action'],
            target_cell=list(target),target_still_unknown=target not in snap.floor|snap.occupied,
            position_map_m=p.tolist(),rotation_map_from_body=R.tolist(),floor_height=snap.floor_height,
            requested_viewpoint_distance_m=None if visit is None else float(np.linalg.norm(
                p[:2]-visit['camera_viewpoint']['viewpoint_map_xy_m'])),
            current_position_directed_heading_rad=heading,current_position_projection=projection,
            current_position_view_hypothesis=any(c['fully_projected'] and not
                c['stored_obstacle_blocks_ray'] for c in projection),
            coverage_rejected=plan['selection'].get('translation_footprint_coverage',{}).get('rejected',False)))
    events = {}
    for plan in plans:
        for event in plan['selection'].get('coverage_view_request',{}).get('recent_events',[]):
            events[(event['started_ns'],event['completed_ns'])]=event
    result=dict(schema='coverage_view_stall_diagnosis.v1',
        mission=read('continuous_native_arrival_evaluation.json'),
        planning_records=len(plans),on_time=sum(r['on_time'] for r in plans),
        actions=dict(Counter(r['action'] for r in plans)),
        coverage_rejections=sum(r['selection'].get('translation_footprint_coverage',{}).get('rejected',False) for r in plans),
        view_status_counts=dict(Counter(str(r['selection'].get('coverage_view_request',{}).get('status')) for r in plans)),
        requested_targets=dict(Counter(str(r['selection'].get('coverage_view_request',{}).get('target_cell')) for r in plans)),
        recorded_coverage_events=list(events.values()),
        early_maps_replayed=len(frames),early_request_states_compared=len(rows),
        early_states_with_current_position_view_hypothesis=sum(r['current_position_view_hypothesis'] for r in rows),
        recorded_map_counts_matched=True,delivered_depth_digests_verified=True,
        native_pose_or_geometry_used=False,alternative_navigation_executed=False,
        projection_is_not_observation=True,rows=rows)
    with output.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','recorded_coverage_events')},indent=2))


if __name__=='__main__':main()
