"""Read only the ten completed C3 development logs; no simulator or model calls."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from lewm import decision_headroom_json_v42_development as output
from lewm.decision_headroom_reference_development import ReferenceGeometry
from lewm.physical_execution_development import rotation_xyzw
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL

REPO = Path(__file__).resolve().parents[1]
BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_navigation_capability_v1_attempt_001')
ROOT = BASE / 'grid_c3_failure_log_diagnosis_attempt001'


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wall_ray(pose, walls):
    """Evaluator-only optical centre ray to axis-aligned wall boxes, in metres."""
    transform = np.asarray(BODY_FROM_OPTICAL)
    rotation = rotation_xyzw(pose[3:])
    origin = pose[:3] + rotation @ transform[:3, 3]
    direction = rotation @ transform[:3, 2]
    hits = []
    for wall in walls:
        assert wall['yaw_rad'] == 0
        low = np.asarray(wall['centre_xyz']) - np.asarray(wall['size_xyz']) / 2
        high = np.asarray(wall['centre_xyz']) + np.asarray(wall['size_xyz']) / 2
        enter, leave = 0., float('inf')
        for axis in range(3):
            if abs(direction[axis]) < 1e-12:
                if not low[axis] <= origin[axis] <= high[axis]:
                    enter, leave = 1., 0.
                    break
            else:
                a, b = sorted(((low[axis]-origin[axis])/direction[axis],
                               (high[axis]-origin[axis])/direction[axis]))
                enter, leave = max(enter, a), min(leave, b)
        if enter <= leave:
            hits.append((enter, wall['wall_id']))
    return dict(distance_m=min(hits)[0], wall_id=min(hits)[1]) if hits else None


def analyse(index, protocol):
    root = BASE / f'runs/v0_grid_c3_screen_C1_dev{index:02d}_ep0_attempt001'
    names = ('episode.json','episode_evaluation.json','planning.json','requests.json',
             'poses.json','startup_recovery.json','specification.json','acquisitions.json',
             'native/physics_trace.npz','native_clearance_summary_arrays.npz')
    episode, evaluation, plans, requests, poses, startup, spec, acquisitions = [load(root/n) for n in names[:8]]
    with np.load(root/names[8], allow_pickle=False) as f:
        times, native = f['timestamp_s'].copy(), f['base_pose_world'].copy()
    with np.load(root/names[9], allow_pickle=False) as f:
        clearance_times, clearance = f['timestamp_s'].copy(), f['separation_lower_m'].copy()
    start = acquisitions[0]['measured_ns']/1e9
    mission_times = times-start
    selected = [r for r in plans if 'selection' in r]
    holds = evaluation['hold_details']
    walls = [dict(center=np.asarray(w['centre_xyz'][:2]), size=np.asarray(w['size_xyz'][:2]),
                  yaw=w['yaw_rad']) for w in spec['geometry']['wall_boxes']]
    cfg = protocol['definitions']['spl']
    target = episode['home_se2_world'][:2] if evaluation['beacon_success'] else episode['beacon_xy_world']
    geo = ReferenceGeometry(walls, protocol['generator']['world_bounds_xy_m'], target,
        radius_m=cfg['inflation_radius_m'], clearance_m=cfg['additional_clearance_m'], resolution_m=cfg['grid_resolution_m'])
    def physical(t):
        return native[min(int(np.searchsorted(mission_times, t)), len(native)-1)]
    def window(lo, hi):
        pp = [r for r in plans if lo <= r['frame']*.1 < hi]
        ss = [r for r in pp if 'selection' in r]
        hh = [r for r in holds if lo <= r['frame']*.1 < hi]
        qq = [r for r in requests if lo <= r['simulator_ns']/1e9-start < hi]
        commands = np.asarray([r['applied_command'] for r in qq])
        tt = np.arange(lo, min(hi, mission_times[-1]), .1)
        points = np.asarray([physical(t)[:2] for t in tt] + [physical(min(hi, mission_times[-1]))[:2]])
        categories = Counter(r['category'] for r in hh)
        rules = Counter()
        for r in hh:
            for k in ('observation_action_space_exclusions','motion_clearance_exclusions','stopping_projection_exclusions'):
                if r.get(k): rules[k] += 1
            if r.get('override_reason'): rules[r['override_reason']] += 1
        return dict(start_s=lo,end_s=min(hi,mission_times[-1]),planning_records=len(pp),selected=len(ss),
            holds=len(hh),hold_fraction_scored=len(hh)/len(ss) if ss else None,
            unscored_reasons=dict(Counter(r.get('reason') for r in pp if 'selection' not in r)),
            hold_categories=dict(categories),overlapping_hold_exclusion_rules=dict(rules),
            route_status=dict(Counter(r.get('route_status') for r in ss)),
            actions=dict(Counter(r['action'] for r in ss)),
            executed_zero_fraction=float(np.mean(np.all(commands==0,axis=1))) if len(commands) else None,
            executed_translation_fraction=float(np.mean(np.linalg.norm(commands[:,:2],axis=1)>0)) if len(commands) else None,
            executed_turn_fraction=float(np.mean(commands[:,2]!=0)) if len(commands) else None,
            dispatch_reasons=dict(Counter(r['reason'] for r in qq)),
            path_100ms_m=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
            net_displacement_m=float(np.linalg.norm(points[-1]-points[0])),
            xy_extent_m=np.ptp(points,axis=0),
            remaining_active_target_start=geo.distance_and_heading(points[0]),
            remaining_active_target_end=geo.distance_and_heading(points[-1]))
    first = selected[0]
    first_pose = physical(first['frame']*.1)
    recovery_tail = window(max(0.,mission_times[-1]-10), mission_times[-1]+.000001)
    end_mask = clearance_times >= times[-1]-10
    yaw = np.unwrap(np.asarray([math.atan2(rotation_xyzw(p[3:])[1,0],rotation_xyzw(p[3:])[0,0])
        for p in native[np.flatnonzero(mission_times>=max(0.,mission_times[-1]-10))[::50]]]))
    return dict(episode_id=episode['episode_id'],startup_fixed=index in (1,2,6,8,9),
        source=str(root),input_sha256={n:sha(root/n) for n in names},
        beacon_reached=evaluation['beacon_success'],beacon_elapsed_s=evaluation['outbound']['elapsed_s'] if evaluation['beacon_success'] else None,
        round_trip=evaluation['round_trip_success'],terminal_s=mission_times[-1],
        active_target='HOME' if evaluation['beacon_success'] else 'BEACON',
        remaining_shortest_path=geo.distance_and_heading(native[-1,:2]),
        remaining_reference_specification=cfg,
        endpoint_reference_footprint_clearance_m=float(geo.footprint_clearance(native[-1,:2])),
        total_native_path_m=evaluation['outbound']['actual_path_m']+evaluation['return_leg']['actual_path_m'],
        startup=startup,first_planning_s=first['frame']*.1,
        first_planning_map=first['selection']['routing_memory_scope'],
        first_planning_optical_centre_wall_ray=wall_ray(first_pose,spec['geometry']['wall_boxes']),
        last_planning_map=selected[-1]['selection']['routing_memory_scope'],
        first_view_budget_exhaustion=next((r['frame']*.1 for r in plans if r.get('reason')=='VIEW_BUDGET_EXHAUSTED'),None),
        holds_by_phase=evaluation['stall_by_phase'],hold_categories=evaluation['hold_categories'],
        whole_episode=window(0.,mission_times[-1]+.000001),
        windows=[window(float(t),float(t+60)) for t in range(0,int(math.ceil(mission_times[-1])),60)],
        terminal_10s=recovery_tail,terminal_10s_minimum_articulated_clearance_m=float(clearance[end_mask].min()),
        terminal_10s_absolute_yaw_travel_rad=float(np.abs(np.diff(yaw)).sum()),
        terminal_optical_centre_wall_ray=wall_ray(native[-1],spec['geometry']['wall_boxes']),
        final_raw_pose=poses[-1]['raw_pose'])


def main():
    started=time.monotonic()
    ROOT.mkdir(exist_ok=False)
    output.install(ROOT)
    protocol=load(REPO/'docs/go2_navigation_capability_preregistration_v1_2026-09-25.json')
    rows=[]
    for index in range(10):
        row=analyse(index,protocol)
        with (ROOT/f'dev{index:02d}.json').open('x') as stream: json.dump(row,stream,indent=2)
        rows.append(dict(episode_id=row['episode_id'],result=str(ROOT/f'dev{index:02d}.json')))
        print(index,row['remaining_shortest_path'],row['whole_episode']['hold_fraction_scored'],
              row['whole_episode']['executed_zero_fraction'],flush=True)
    result=dict(schema='navigation_capability_failure_log_diagnosis.v1',rows=rows,
        wall_s=time.monotonic()-started,script_sha256=sha(Path(__file__)),
        exploratory=True,no_physics=True,no_models=True,no_sealed_inputs=True,
        hold_denominator='Scored selections only, as preregistered; unscored planning and executed zeros reported separately.',
        distance='Original SPL inflated true geometry at final native base XY; unavailable endpoints remain unresolved.',
        binding_counts_overlap=True)
    with (ROOT/'result.json').open('x') as stream: json.dump(result,stream,indent=2)


if __name__=='__main__': main()
