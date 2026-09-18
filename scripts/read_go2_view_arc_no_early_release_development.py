"""Trace the exposed ablation's clearance loss using retained physical evidence."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL, body_points
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.auxiliary_downward45_depth_observation_development import body_points as auxiliary_points
from scripts.live_depth_noise_session_development import NoisyPublicReplay

ROOT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_view_arc_no_early_release_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001')


def main():
    read = lambda name: json.loads((ROOT/name).read_text())
    plans = [r for r in read('planning.json') if 'selection' in r]
    requests = read('requests.json')
    dispatched = defaultdict(list)
    observed = {}
    for r in requests:
        if 'command_observation_ns' in r:
            dispatched[r['command_observation_ns']].append(r)
        if r.get('nominal_connector'):
            observed.setdefault(r['observation_measured_ns'], r)
    camera = {r['frame']:r for r in read('native/in_memory_camera_observations.json')['frames']}
    walls = read('launch.json')['fresh_layout_inventory']['layouts'][1]['geometry']['wall_boxes']
    assert all(w['yaw_rad'] == 0 for w in walls)
    centres = np.array([w['centre_xyz'][:2] for w in walls])
    half = np.array([w['size_xyz'][:2] for w in walls])*.5
    with np.load(ROOT/'native/physics_trace.npz', allow_pickle=False) as data:
        native = data['base_pose_world'][[camera[r['frame']]['physical_sample_index'] for r in plans]].copy()
    distances = np.linalg.norm(np.maximum(np.abs(native[:,None,:2]-centres[None,:,:])-half[None,:,:], 0), axis=2)
    nearest = distances.argmin(axis=1)
    minimum = distances[np.arange(len(plans)), nearest]
    rows = []
    for r, pose, wall, distance in zip(plans, native, nearest, minimum):
        s = r['selection']; chosen = next(c for c in s['memory_forecast_candidates'] if c['action']==r['action'])
        fresh = observed.get(r['measured_ns'])
        rows.append(dict(frame=r['frame'], action=r['action'], on_time=r['on_time'],
            native_base_pose_world=pose.tolist(), nearest_wall_id=walls[int(wall)]['wall_id'],
            physical_wall_clearance_m=float(distance),
            stored_current_clearance_m=r.get('lookahead',{}).get('start_clearance_m'),
            chosen_predicted_minimum_clearance_m=chosen['minimum_predicted_path_clearance_m'],
            chosen_clearance_mode=chosen['clearance_check_mode'],
            all_forecast_paths_blocked=all(not c['nominal_predicted_path_clear'] for c in s['memory_forecast_candidates']),
            suppressed_early_release=s.get('suppressed_early_heading_release'),
            clearance_turn=s.get('clearance_turn'),
            matching_applied_intervals=sum(np.allclose(v['applied_command'],s['requested_command'],atol=1e-7)
                for v in dispatched[r['measured_ns']]),
            current_dispatch_clearance_m=None if fresh is None else fresh['nominal_connector']['minimum_observed_cell_distance_m']))
    reader = NoisyPublicReplay(ROOT/'native')
    depth_rows = []
    for frame in (560, 568, 576, 584, 600, 620, 628, 630, 632, 636):
        # Also inspect frame 630 even though it is between planning observations.
        with np.load(ROOT/'native/physics_trace.npz', allow_pickle=False) as data:
            pose = data['base_pose_world'][camera[frame]['physical_sample_index']].copy()
        dists = np.linalg.norm(np.maximum(np.abs(pose[None,:2]-centres)-half,0),axis=1)
        wall = int(dists.argmin()); closest_xy = np.clip(pose[:2],centres[wall]-half[wall],centres[wall]+half[wall])
        closest_world = np.r_[closest_xy,pose[2]]
        rotation = rotation_xyzw(pose[3:])
        policy, primary, _, _, auxiliary, now = reader.packet(frame)
        for name, packet, points_fn, extrinsic in (
                ('primary',primary,body_points,np.asarray(BODY_FROM_OPTICAL)),
                ('auxiliary',auxiliary,auxiliary_points,body_from_optical())):
            cloud = points_fn(packet,policy,now_ns=now,stride=4)
            world = cloud['points_body_m'][cloud['valid']]@rotation.T+pose[:3]
            above = world[(world[:,2]>.03)&(world[:,2]<.65)]
            surface = np.linalg.norm(np.maximum(np.abs(above[:,:2]-centres[wall])-half[wall],0),axis=1)<=.015
            points = above[surface]
            optical = ((closest_world-pose[:3])@rotation-extrinsic[:3,3])@extrinsic[:3,:3]
            uv = optical[:2]/optical[2]*FOCAL+[319.5,239.5] if optical[2]>0 else None
            depth_rows.append(dict(frame=frame,camera=name,nearest_wall_id=walls[wall]['wall_id'],
                physical_wall_clearance_m=float(dists[wall]),valid_pixels=int(packet['valid'].sum()),
                sampled_above_floor_points=len(above),sampled_points_near_nearest_wall_box=len(points),
                minimum_sampled_nearest_wall_point_xy_distance_m=None if not len(points) else
                    float(np.linalg.norm(points[:,:2]-pose[:2],axis=1).min()),
                closest_wall_point_at_base_height_optical_m=optical.tolist(),
                closest_wall_point_pixel_xy=None if uv is None else uv.tolist(),
                closest_wall_point_within_depth_range=bool(.2<=optical[2]<=5.),
                closest_wall_point_in_image=bool(uv is not None and (uv>=0).all() and (uv<[640,480]).all())))
    late = [r for r in rows if r['frame']>=700]
    suppressed = [r for r in rows if r['suppressed_early_release']]
    result = dict(schema='no_early_release_clearance_entry_diagnosis.v1',
        evaluation=read('short_pulse_navigation_evaluation_v1.json'),
        pipeline_faults=read('pipeline_faults.json'),
        suppressed_release_frames=[r['frame'] for r in suppressed],
        suppressed_release_frames_with_applied_commands=[r['frame'] for r in suppressed if r['matching_applied_intervals']],
        arc_fallback_frames=[r['frame'] for r in plans if 'measured_view_arc_recovery' in r['selection']],
        actions=dict(Counter(r['action'] for r in plans)),
        first_planning_frame_inside_physical_nominal_margin=next(r['frame'] for r in rows if r['physical_wall_clearance_m']<=.45),
        first_all_forecast_paths_blocked_frame=next(r['frame'] for r in rows if r['all_forecast_paths_blocked']),
        first_current_obstacle_veto=next(r for r in requests if r['reason']=='CURRENT_OBSERVED_OBSTACLE_VETO'),
        late_frames=[700,4800], late_samples=len(late),
        late_physical_clearance_min_m=min(r['physical_wall_clearance_m'] for r in late),
        late_physical_clearance_median_m=float(np.median([r['physical_wall_clearance_m'] for r in late])),
        late_physical_clearance_max_m=max(r['physical_wall_clearance_m'] for r in late),
        late_physical_clearance_at_or_below_nominal_count=sum(r['physical_wall_clearance_m']<=.45 for r in late),
        raw_delivered_noisy_depth_digests_verified=True, depth_samples=depth_rows, rows=rows,
        physical_geometry_and_native_pose_evaluator_only=True,
        circular_footprint_not_articulated_contact_geometry=True,
        depth_probe_height_band_world_m=[.03,.65], depth_probe_stride=4,
        nearest_wall_association_tolerance_m=.015,
        nearest_point_projection_not_full_surface_visibility_certificate=True,
        causal_navigation_benefit_of_ablation_proven=False, full_failure_depth_preserved=True)
    with (ROOT/'clearance_entry_diagnosis_v1.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','depth_samples','first_current_obstacle_veto')},indent=2))
    for r in depth_rows:
        print(json.dumps(r))


if __name__=='__main__':
    main()
