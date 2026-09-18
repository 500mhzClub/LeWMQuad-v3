"""Replay observed maps to compare forecasts at the measured-view conflict."""
import json
import time
import cv2
import numpy as np
import torch
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.memory_forecast_clearance_development import reserve_recovery_clear
from lewm.clearance_turn_recovery_development import stepwise_recovery_clear
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.verify_go2_polygon_floor_coverage_development import RecordedPolygonMap
from lewm.projected_polygon_floor_coverage_development import warmup
from scripts import run_go2_polygon_floor_repeatability_development as run


def main():
    root=run.BASE/run.root_name(4);output=root/'polygon_return_forecast_clearance_probe_v1.json'
    if output.exists():raise ValueError('preserve completed diagnostic')
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1);configure();warmup()
    read=lambda name:json.loads((root/name).read_text())
    poses={r['frame']:r['registered_pose'] for r in read('poses.json')}
    plans={r['frame']:r for r in read('planning.json') if 'selection' in r}
    wanted=(2640,2668,2672,2876,2880,4204,4208)
    needed={plans[f]['map_frame'] for f in wanted}
    frames=sorted({r['frame'] for r in read('stage_events.json')
        if r['stage']=='mapping' and r['frame']<=max(needed)})
    mapper=RecordedPolygonMap();reader=NoisyPublicReplay(root/'native');snapshots={};started=time.monotonic()
    for frame in frames:
        policy,depth,_,_,auxiliary,now=reader.packet(frame)
        snap=mapper.update(policy,depth,poses[frame],auxiliary_depth=auxiliary,measured_ns=now)
        if frame in needed:snapshots[frame]=snap
        if frame%400==0:print('CLEARANCE_MAP_REPLAY',frame,flush=True)
    rows=[]
    for frame in wanted:
        plan=plans[frame];snap=snapshots[plan['map_frame']];scope=plan['selection']['routing_memory_scope']
        assert scope['retained_floor_cells']==len(snap.floor)
        assert scope['retained_fine_obstacle_cells']==len(snap.fine_occupied)
        B=np.asarray(snap.map_from_initial);p=B@np.asarray(poses[frame]['position_initial_body_m'])
        R=B@np.asarray(poses[frame]['rotation_initial_body_from_current_body'])
        clearance=cached_clearance(snap.fine_occupied);correction=plan['motion_correction']
        forecasts=dict(neural=correction['corrected_forecast_xy_m'],
            pose_command=correction['pose_command_forecast_xy_m'],
            command_history=correction['command_history_forecast_xy_yaw'])
        comparison={}
        for name,forecast in forecasts.items():
            candidates=[]
            for index,action in enumerate(ACTIONS):
                xy=np.asarray(forecast)[index,:,:2]
                points=p[:2]+np.vstack((np.zeros(2),xy))@R[:2,:2].T
                distances=[clearance.minimum(a,b) for a,b in zip(points[:-1],points[1:])]
                if name=='neural':
                    saved=plan['selection']['memory_forecast_candidates'][index]
                    np.testing.assert_allclose(distances,saved['segment_clearances_m'],rtol=0,atol=1e-10)
                minimum=min(distances);required=.45 if action=='hold' else .48
                eligible=(minimum>required+1e-12 or (action!='hold' and
                    (reserve_recovery_clear(distances,required) or
                    (action in ('left_turn','right_turn') and stepwise_recovery_clear(distances,required)))))
                candidates.append(dict(action=action,minimum_m=minimum,segment_clearances_m=distances,
                    nominal_footprint_clear=minimum>.45+1e-12,reserve_eligible=bool(eligible)))
            comparison[name]=candidates
        rows.append(dict(frame=frame,map_frame=snap.frame,recorded_action=plan['action'],
            current_observed_clearance_m=clearance.minimum(p[:2],p[:2]),
            preferred_action=plan['selection']['before_memory_filter_action'],forecasts=comparison))
    result=dict(schema='polygon_return_forecast_clearance_probe.v1',rows=rows,
        recorded_map_updates_replayed=len(frames),recorded_map_counts_matched=True,
        all_recorded_neural_segment_clearances_matched=True,delivered_noisy_depth_digests_verified=True,
        native_state_or_geometry_used=False,alternative_actions_executed=False,
        full_selector_or_dispatch_reexecuted=False,alternative_navigation_outcome_proven=False,
        wall_seconds=time.monotonic()-started,
        snapshots={str(f):dict(floor=sorted(s.floor),fine_occupied=sorted(s.fine_occupied),
            map_from_initial=s.map_from_initial) for f,s in snapshots.items()})
    run.previous.previous.save(output,result)
    for row in rows:
        print(json.dumps(dict(frame=row['frame'],current=row['current_observed_clearance_m'],
            turns={name:[{k:v for k,v in c.items() if k!='segment_clearances_m'} for c in candidates
                if c['action'] in ('left_turn','right_turn')] for name,candidates in row['forecasts'].items()})))


if __name__=='__main__':main()
