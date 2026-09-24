"""Profile exact routing proposals on reconstructed recorded development maps."""
import cProfile
import io
import json
import pstats
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from lewm.fine_stored_obstacle_routing_development import proposer
from lewm.fine_goal_route_development import fine_goal_route
from lewm.cached_fine_goal_route_development import cached_fine_goal_route
from lewm.clearance_preferred_route_development import refine_proposal
from lewm.two_cm_floor_extent_development import configure
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.replay_go2_no_early_release_map_entry_development import RecordedCurrentPlaneMap
from scripts.run_go2_current_position_coverage_view_development import BASE,ROOT


def main():
    root=BASE/ROOT; output=root/'routing_profile_v1'
    output.mkdir(exist_ok=False)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1);configure()
    read=lambda name:json.loads((root/name).read_text())
    plans={r['frame']:r for r in read('planning.json') if 'selection'in r}
    poses={r['frame']:r['registered_pose'] for r in read('poses.json')}
    wanted=(200,600,1000,1400,1600)
    frames=sorted({r['frame'] for r in read('stage_events.json') if
        r['stage']=='mapping' and r['frame']<=max(plans[f]['map_frame'] for f in wanted)})
    required={plans[f]['map_frame'] for f in wanted}; snapshots={}
    reader=NoisyPublicReplay(root/'native');mapper=RecordedCurrentPlaneMap()
    for frame in frames:
        policy,depth,_,_,auxiliary,now=reader.packet(frame)
        snap=mapper.update(policy,depth,poses[frame],auxiliary_depth=auxiliary,measured_ns=now)
        if frame in required:snapshots[frame]=snap
        if frame%400==0:print('RECONSTRUCTED_MAP',frame,flush=True)
    goal=read('launch.json')['public_mission']['goal_initial_body_xy_m']
    rows=[]
    for frame in wanted:
        plan=plans[frame];snapshot=snapshots[plan['map_frame']]
        scope=plan['selection']['routing_memory_scope']
        assert scope['retained_floor_cells']==len(snapshot.floor)
        assert scope['retained_fine_obstacle_cells']==len(snapshot.fine_occupied)
        B=np.asarray(snapshot.map_from_initial)
        position=B@np.asarray(poses[frame]['position_initial_body_m'])
        mapped_goal=(B@np.r_[goal,0.])[:2]
        propose=proposer(snapshot)
        def call():return propose(snapshot.floor,snapshot.occupied,position[:2],mapped_goal)
        route=call();elapsed=[]
        for _ in range(5):
            started=time.perf_counter_ns();assert call()==route
            elapsed.append((time.perf_counter_ns()-started)/1e6)
        profile=cProfile.Profile();profile.runcall(call)
        stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(20)
        (output/f'frame{frame}_profile.txt').write_text(stream.getvalue())
        row=dict(frame=frame,map_frame=snapshot.frame,floor=sorted(snapshot.floor),
            occupied=sorted(snapshot.occupied),fine_occupied=sorted(snapshot.fine_occupied),
            position_map_xy_m=position[:2].tolist(),goal_map_xy_m=mapped_goal.tolist(),
            proposal=route,unprofiled_elapsed_ms=elapsed,median_ms=float(np.median(elapsed)))
        rows.append(row)
        print('PROFILE',frame,row['median_ms'],flush=True)
        if frame==1600:print(stream.getvalue(),flush=True)
    result=dict(recorded_map_updates_replayed=len(frames),recorded_map_counts_matched=True,
        delivered_noisy_depth_digests_verified=True,native_pose_or_wall_geometry_used=False,
        scope='routing proposer only, excludes frontier visit state and other runtime wrappers',
        full_controller_reexecuted=False,alternative_navigation_outcome_proven=False,rows=rows)
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')


def profile_refinements(cached=False):
    output=BASE/ROOT/'routing_profile_v1'
    result_path=output/('cached_refinement_result.json' if cached else 'refinement_result.json')
    if result_path.exists():raise ValueError('preserve completed refinement profile')
    rows=[]
    for saved in json.loads((output/'result.json').read_text())['rows']:
        snapshot=SimpleNamespace(**{k:frozenset(map(tuple,saved[k]))
            for k in ('floor','occupied','fine_occupied')})
        position=np.asarray(saved['position_map_xy_m']);goal=np.asarray(saved['goal_map_xy_m'])
        route=refine_proposal(saved['proposal'],snapshot.floor,snapshot.occupied)
        def call():return (cached_fine_goal_route if cached else fine_goal_route)(snapshot,position,goal,route)
        elapsed=[]
        for _ in range(3):
            started=time.perf_counter_ns();result=call();elapsed.append((time.perf_counter_ns()-started)/1e6)
        profile=cProfile.Profile();profile.runcall(call)
        stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats('cumtime').print_stats(20)
        label='cached_refinement' if cached else 'refinement'
        (output/f'frame{saved["frame"]}_{label}_profile.txt').write_text(stream.getvalue())
        row=dict(frame=saved['frame'],elapsed_ms=elapsed,median_ms=float(np.median(elapsed)),
            status=result['status'],fine_goal_route_found='fine_goal_route' in result)
        rows.append(row);print(json.dumps(row),flush=True)
        if saved['frame']==1600:print(stream.getvalue(),flush=True)
    result_path.write_text(json.dumps(dict(rows=rows,native_state_used=False,
        scope='fine-goal refinement after the same preferred coarse route',navigation_outcome_tested=False),indent=2)+'\n')


if __name__=='__main__':
    if sys.argv[1:]==['--refinements']:profile_refinements()
    elif sys.argv[1:]==['--cached-refinements']:profile_refinements(cached=True)
    elif not sys.argv[1:]:main()
    else:raise ValueError('use no arguments or --refinements')
