"""Test raw/local floor-coverage union on the recorded gyro-floor layout-2 stall.

Replays mapping inputs with fixed admitted poses. No new poses, obstacle
geometry, controller state, commands or physics are used in the comparison.
"""
import hashlib
import json
import time
import numpy as np
from lewm.eligible_floor_registration_development import bind
from lewm.multirate_routing_map_development import Geometry
from lewm.local_inverse_depth_floor_development import local_depth
from lewm.fine_stored_obstacle_routing_development import proposer
from scripts.probe_go2_noisy_routing_floor_development import RecordedMap, LocalCoverageMap
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE


class UnionCoverageGeometry(Geometry):
    def floor_coverage(self,depth,valid,*args,**kwargs):
        raw=super().floor_coverage(depth,valid,*args,**kwargs)
        estimated,supported=local_depth(depth,valid)
        local=super().floor_coverage(estimated,supported,*args,**kwargs)
        return raw|dict(covered=raw['covered']|local['covered'],
            raw_floor_candidate_pixels=raw['floor_candidate_pixels'],
            local_floor_candidate_pixels=local['floor_candidate_pixels'],
            floor_coverage_union=True)


class UnionCoverageMap(RecordedMap):
    update=bind(RecordedMap.update,Geometry=UnionCoverageGeometry)


def main():
    root=BASE/'go2_live_gyro_height_floor_noise_2mm_native_layout02_4800_v1_attempt_001'
    output=root/'union_floor_coverage_prefix_probe_v1.json'
    if output.exists():raise ValueError('preserve prior diagnostic')
    plans=json.loads((root/'planning.json').read_text())
    plan=next(p for p in plans if p.get('route_status')=='OBSERVED_COMPONENT_HAS_NO_FRONTIER')
    end=plan['map_frame'];visits=json.loads((root/'frontier_visits.json').read_text())
    assert len(visits['events'])==1 and visits['events'][0]['map_frame']==end
    excluded={tuple(c) for c in visits['excluded_cells']}
    events=json.loads((root/'stage_events.json').read_text())
    frames=sorted({e['frame'] for e in events if e['stage']=='mapping' and e['frame']<=end})
    assert frames[0]==0 and frames[-1]==end
    witnesses={p['map_frame']:p['selection']['routing_memory_scope']
        for p in plans if 'selection' in p and p['map_frame']<=end}
    poses={p['frame']:p['registered_pose'] for p in json.loads((root/'poses.json').read_text())}
    launch=json.loads((root/'launch.json').read_text());reader=NoisyPublicReplay(root/'native')
    maps=dict(local=LocalCoverageMap(),raw=RecordedMap(),union=UnionCoverageMap())
    checked=[];rows=[];started=time.monotonic()
    for frame in frames:
        p,d,fast,rgb,aux,now=reader.packet(frame)
        snapshots={n:m.update(p,d,poses[frame],auxiliary_depth=aux,measured_ns=now) for n,m in maps.items()}
        local=snapshots['local'];union=snapshots['union'];raw=snapshots['raw']
        assert union.floor==local.floor|raw.floor
        assert all(s.floor_height==local.floor_height and s.occupied==local.occupied and
            s.fine_occupied==local.fine_occupied for s in snapshots.values())
        if frame in witnesses:
            w=witnesses[frame]
            assert len(local.floor)==w['retained_floor_cells'] and len(local.fine_occupied)==w['retained_fine_obstacle_cells'],frame
            checked.append(frame)
        rows.append(dict(frame=frame,floor_counts={n:len(s.floor) for n,s in snapshots.items()}))
    goal=np.asarray(local.map_from_initial)@np.r_[launch['public_mission']['goal_initial_body_xy_m'],0.]
    variants={n:dict(floor_cells=sorted(s.floor),routes={label:proposer(s)(s.floor,s.occupied,
        s.position_map[:2],goal[:2],excluded_frontiers=cells)
        for label,cells in [('saved_exclusions',excluded),('no_exclusions',set())]}) for n,s in snapshots.items()}
    result=dict(layout_index=2,planning_frame=plan['frame'],map_frame=end,
        exact_saved_map_count_witnesses=checked,frames=rows,variants=variants,
        position_map=local.position_map,excluded_cells=sorted(excluded),
        union_exactly_preserves_both_component_floor_sets=True,
        floor_height_and_obstacle_geometry_unchanged=True,fixed_recorded_public_poses=True,
        actual_delivered_noisy_packet_digests_verified=True,full_pose_acceptance_replayed=False,
        full_controller_state_replayed=False,counterfactual_navigation_executed=False,
        native_physics_used=False,wall_seconds=time.monotonic()-started,
        source_sha256={__file__:hashlib.sha256(open(__file__,'rb').read()).hexdigest()})
    with output.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(output=str(output),witnesses=len(checked),floor_counts=rows[-1],
        routes={n:v['routes'] for n,v in variants.items()})),flush=True)


if __name__=='__main__':main()
