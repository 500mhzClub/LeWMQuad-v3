"""After the fixed batch, profile fine-goal search on recorded observed maps."""
import cProfile
import io
import json
import pstats
import time

import cv2
import numpy as np
import torch

from lewm.cached_fine_connectivity_development import fine_goal_route,search_graph,floor_index
from lewm.cached_fine_goal_route_development import cached_graph_geometry
from lewm.fine_stored_obstacle_routing_development import proposer
from lewm.clearance_preferred_route_development import refine_proposal
from lewm.two_cm_floor_extent_development import configure
from lewm.projected_polygon_floor_coverage_development import warmup
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.verify_go2_polygon_floor_coverage_development import RecordedPolygonMap
from scripts import run_go2_polygon_floor_repeatability_development as run


def main():
    if not (run.BASE/run.root_name(4)/'frozen_readout_navigation_readout_v1.json').exists():
        raise ValueError('complete the fixed native batch before heavy replay')
    root=run.BASE/run.root_name(1);output=root/'fine_goal_routing_profile_v1'
    output.mkdir(exist_ok=False)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(1);configure();warmup()
    read=lambda name:json.loads((root/name).read_text())
    plans={r['frame']:r for r in read('planning.json') if 'selection' in r}
    poses={r['frame']:r['registered_pose'] for r in read('poses.json')}
    wanted=(2372,2644,2920,2924)
    required={plans[f]['map_frame'] for f in wanted}
    frames=sorted({r['frame'] for r in read('stage_events.json')
        if r['stage']=='mapping' and r['frame']<=max(required)})
    reader=NoisyPublicReplay(root/'native');mapper=RecordedPolygonMap();snapshots={}
    started=time.monotonic()
    for frame in frames:
        policy,depth,_,_,auxiliary,now=reader.packet(frame)
        snapshot=mapper.update(policy,depth,poses[frame],auxiliary_depth=auxiliary,measured_ns=now)
        if frame in required:snapshots[frame]=snapshot
        if frame%400==0:print('POLYGON_ROUTING_MAP_REPLAY',frame,flush=True)
    goal=np.asarray(read('launch.json')['public_mission']['goal_initial_body_xy_m'])
    rows=[]
    for frame in wanted:
        plan=plans[frame];snap=snapshots[plan['map_frame']]
        scope=plan['selection']['routing_memory_scope']
        assert len(snap.floor)==scope['retained_floor_cells']
        assert len(snap.fine_occupied)==scope['retained_fine_obstacle_cells']
        B=np.asarray(snap.map_from_initial)
        p=(B@np.asarray(poses[frame]['position_initial_body_m']))[:2]
        g=(B@np.r_[goal,0.])[:2]
        base=proposer(snap)
        def coarse_call():return base(snap.floor,snap.occupied,p,g)
        coarse_start=time.perf_counter();coarse=coarse_call()
        coarse_ms=(time.perf_counter()-coarse_start)*1000
        refinement_start=time.perf_counter();preferred=refine_proposal(coarse,snap.floor,snap.occupied)
        refinement_ms=(time.perf_counter()-refinement_start)*1000
        coarse_profile=cProfile.Profile();coarse_profile.runcall(coarse_call)
        coarse_stream=io.StringIO();pstats.Stats(coarse_profile,stream=coarse_stream).sort_stats('cumtime').print_stats(25)
        (output/f'frame_{frame}_coarse_profile.txt').write_text(coarse_stream.getvalue())
        # Only the fine-goal subproblem is reconstructed. The sentinel asks
        # whether this exact observed graph connects to the current public goal;
        # it does not reconstruct frontier-visit state or a full controller plan.
        initial=dict(status='DIAGNOSTIC_FINE_GOAL_QUERY',nominal_radius_m=.45)
        def call():return fine_goal_route(snap,p,g,initial)
        search_graph.cache_clear();floor_index.cache_clear();cached_graph_geometry.cache_clear()
        cold=time.perf_counter();result=call();cold_ms=(time.perf_counter()-cold)*1000
        warm=time.perf_counter();again=call();warm_ms=(time.perf_counter()-warm)*1000
        def scientific(value):
            return {k:({kk:vv for kk,vv in v.items() if kk!='routing_s'} if k=='fine_goal_route' else v)
                for k,v in value.items()}
        assert scientific(result)==scientific(again)
        search_graph.cache_clear();floor_index.cache_clear();cached_graph_geometry.cache_clear()
        profiler=cProfile.Profile();profiler.runcall(call)
        stream=io.StringIO();pstats.Stats(profiler,stream=stream).sort_stats('cumtime').print_stats(25)
        (output/f'frame_{frame}_cold_profile.txt').write_text(stream.getvalue())
        rows.append(dict(frame=frame,map_frame=snap.frame,recorded_route_status=plan['route_status'],
            floor=sorted(snap.floor),occupied=sorted(snap.occupied),fine_occupied=sorted(snap.fine_occupied),
            position_map_xy_m=p.tolist(),goal_map_xy_m=g.tolist(),
            coarse_ms=coarse_ms,preferred_refinement_ms=refinement_ms,coarse_proposal=coarse,
            preferred_proposal=preferred,
            cold_ms=cold_ms,warm_ms=warm_ms,result=result))
        print('FINE_GOAL_PROFILE',frame,'cold_ms',round(cold_ms,2),'warm_ms',round(warm_ms,2),
            'coarse_ms',round(coarse_ms,2),'status',result['status'],flush=True)
    run.previous.previous.save(output/'result.json',dict(rows=rows,
        recorded_map_updates_replayed=len(frames),recorded_map_cell_counts_matched=True,
        delivered_noisy_depth_digests_verified=True,native_state_or_maze_graph_used=False,
        full_controller_or_frontier_state_replayed=False,
        scope='coarse proposal and fine-goal subproblems with recorded observed map/pose and public goal',
        cold_label_resets_only=['search_graph','floor_index','cached_graph_geometry'],
        alternative_navigation_outcome_proven=False,wall_seconds=time.monotonic()-started))


if __name__=='__main__':main()
