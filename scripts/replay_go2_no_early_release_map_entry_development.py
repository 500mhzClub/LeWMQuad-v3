"""Reconstruct only recorded routing-map updates before an exposed clearance loss."""
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from lewm.current_plane_floor_coverage_development import CurrentPlaneFloorRoutingMap
from lewm.fine_stored_obstacle_routing_development import cached_clearance
from lewm.two_cm_floor_extent_development import configure
from scripts.diagnose_alignment_route_switches_development import saved_pose
from scripts.live_depth_noise_session_development import NoisyPublicReplay

ROOT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_view_arc_no_early_release_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001')


class RecordedCurrentPlaneMap(CurrentPlaneFloorRoutingMap):
    _read_pose = staticmethod(saved_pose)


def main():
    output = ROOT/'map_entry_replay_v1.json'
    if output.exists():
        raise ValueError('preserve completed reconstruction')
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False); torch.set_num_threads(1)
    configure()
    read = lambda name: json.loads((ROOT/name).read_text())
    poses = {r['frame']:r['registered_pose'] for r in read('poses.json')}
    plans = [r for r in read('planning.json') if 'selection' in r and r['frame']<=640]
    frames = sorted({r['frame'] for r in read('stage_events.json') if r['stage']=='mapping' and r['frame']<=636})
    reader = NoisyPublicReplay(ROOT/'native'); mapper = RecordedCurrentPlaneMap()
    snapshots = {}; cell_frames = {}; update_rows = []; began = time.monotonic()
    for frame in frames:
        policy, depth, _, _, auxiliary, now = reader.packet(frame)
        previous = set(mapper.fine_occupied)
        snapshot = mapper.update(policy,depth,poses[frame],auxiliary_depth=auxiliary,measured_ns=now)
        snapshots[frame] = snapshot
        if not previous <= snapshot.fine_occupied:
            raise ValueError('historical obstacle cells unexpectedly removed')
        for cell in snapshot.current_fine_occupied:
            cell_frames.setdefault(cell,[]).append(frame)
        update_rows.append(dict(frame=frame,floor_cells=len(snapshot.floor),
            fine_obstacle_cells=len(snapshot.fine_occupied),new_fine_obstacle_cells=len(snapshot.fine_occupied-previous)))
        if frame%100==0:
            print('MAP_REPLAY',frame,'seconds',round(time.monotonic()-began,1),flush=True)
    comparisons = []
    for plan in plans:
        snapshot = snapshots[plan['map_frame']]
        scope = plan['selection']['routing_memory_scope']
        counts_match = (scope['retained_floor_cells']==len(snapshot.floor)
            and scope['retained_fine_obstacle_cells']==len(snapshot.fine_occupied))
        if not counts_match:
            raise ValueError(f'reconstructed map counts differ at planning frame {plan["frame"]}')
        B = np.asarray(snapshot.map_from_initial)
        position = B@np.asarray(poses[plan['frame']]['position_initial_body_m'])
        clearance = cached_clearance(snapshot.fine_occupied)
        distance = clearance.minimum(position[:2],position[:2])
        recorded = plan.get('lookahead',{}).get('start_clearance_m')
        if recorded is not None and abs(recorded-distance)>1e-10:
            raise ValueError(f'current clearance differs at planning frame {plan["frame"]}')
        gap = np.maximum(np.maximum(clearance.low-position[:2],position[:2]-clearance.high),0.)
        index = int(np.linalg.norm(gap,axis=1).argmin())
        cell = tuple(map(int,clearance.cells[index]))
        comparisons.append(dict(frame=plan['frame'],map_frame=snapshot.frame,
            recorded_current_clearance_m=recorded,replayed_current_clearance_m=distance,
            counts_match=counts_match,position_map_m=position.tolist(),
            nearest_fine_cell=list(cell),nearest_cell_observed_frames=cell_frames[cell]))
    selected = {}
    for frame in (0, 80, 400, 520, 556, 564, 572, 596, 624, 628, 632, 636):
        snapshot = snapshots[frame]
        selected[str(frame)] = dict(frame=frame,map_from_initial=snapshot.map_from_initial,
            floor_height=snapshot.floor_height,floor=sorted(snapshot.floor),
            fine_occupied=sorted(snapshot.fine_occupied),current_fine_occupied=sorted(snapshot.current_fine_occupied))
    result = dict(schema='recorded_clearance_map_entry_replay.v1',
        recorded_map_updates_replayed=len(frames),recorded_plans_compared=len(comparisons),
        recorded_map_cell_counts_matched=True,recorded_current_clearances_matched=True,
        obstacle_cells_removed=0,delivered_noisy_depth_digests_verified=True,
        poses_are_recorded_estimator_outputs=True,tracking_reexecuted=False,
        native_pose_and_wall_geometry_used=False,navigation_reexecuted=False,
        alternative_navigation_success_proven=False,wall_seconds=time.monotonic()-began,
        updates=update_rows,plan_comparisons=comparisons,snapshots=selected,
        fine_cell_observation_frames=[dict(cell=list(k),frames=v) for k,v in sorted(cell_frames.items())])
    with output.open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('updates','plan_comparisons','snapshots','fine_cell_observation_frames')},indent=2),flush=True)


if __name__=='__main__':
    main()
