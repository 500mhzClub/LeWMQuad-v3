"""Explicitly authorised, geometry-only startup checks of the 180 new episodes.

Only exact hash-bound registry paths are opened. Sealed results are aggregate
only. No rendering, simulator, learned model, navigation, or legacy inventory.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from lewm import decision_headroom_json_v42_development as output
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.tiled_density_routed_floor_cell_index_development import observed_floor_cell_index
from lewm.navigation_capability_map_domain_development import (
    GENERATOR_BOUNDS_XY_M, GENERATOR_DIAGONAL_M, MAP_HALF_WIDTH_M, POINT_BOUND_M)

REPO=Path(__file__).resolve().parents[1]


def floor_estimate(spec, episode):
    """Analytic ground-ray/wall-box intersections; no RGB or simulated sensors.

    Ground-only valid quads conservatively exclude mixed floor/wall pixels.
    Apply the deployed floor-index geometry to those ideal geometric depths.
    Body roll/pitch zero, stand height 0.32 m; no noise or robot self-occlusion.
    """
    rr,cc=np.indices((480,640))
    optical=np.stack(((cc+.5-320)/FOCAL,(rr+.5-240)/FOCAL,np.ones_like(cc)),axis=-1)
    T=np.asarray(BODY_FROM_OPTICAL)
    x,y,yaw=episode['home_se2_world'];c,s=np.cos(yaw),np.sin(yaw)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    origin=np.array([x,y,.32])+R@T[:3,3]
    directions=optical@T[:3,:3].T@R.T
    ground=np.divide(-origin[2],directions[:,:,2],out=np.full((480,640),np.inf),where=directions[:,:,2]<-1e-12)
    valid=(ground>=.2)&(ground<=5.)
    indices=np.flatnonzero(valid);d=directions.reshape(-1,3)[indices];g=ground.ravel()[indices]
    clear=np.ones(len(indices),bool)
    for wall in spec['geometry']['wall_boxes']:
        assert wall['yaw_rad']==0., 'Only registered axis-aligned generator family'
        centre=np.asarray(wall['centre_xyz']);size=np.asarray(wall['size_xyz'])
        lo,hi=centre-size/2,centre+size/2
        enter=np.zeros(len(d));leave=g.copy()
        for axis in range(3):
            parallel=np.abs(d[:,axis])<1e-12
            a=np.divide(lo[axis]-origin[axis],d[:,axis],out=np.full(len(d),-np.inf),where=~parallel)
            b=np.divide(hi[axis]-origin[axis],d[:,axis],out=np.full(len(d),np.inf),where=~parallel)
            enter=np.maximum(enter,np.minimum(a,b));leave=np.minimum(leave,np.maximum(a,b))
            if not lo[axis]<=origin[axis]<=hi[axis]:leave[parallel]=-np.inf
        clear &= ~((enter<=leave)&(enter<g-1e-9))
    valid.ravel()[indices]=clear
    depth=np.where(valid,ground,0.).astype(np.float32)
    index=observed_floor_cell_index(depth,valid,np.array([0.,0.,1.]))
    return int(np.count_nonzero(index['ground_cells']))


def main():
    protocol=json.loads((REPO/'docs/go2_navigation_capability_preregistration_v1_2026-09-25.json').read_text())
    assert protocol['generator']['world_bounds_xy_m']==[list(p) for p in GENERATOR_BOUNDS_XY_M]
    base=Path(protocol['output_root']);output.install(base)
    registry=json.loads((base/'registry.json').read_text())
    root=base/'startup_geometry_correctness_c2_2026-09-26';root.mkdir(exist_ok=False)
    def read(binding,role):
        p=Path(binding['path']);expected=base/'sets'/('sealed_test_v1' if role=='sealed_test' else role)
        assert p.parent==expected and p.suffix=='.json'
        data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==binding['sha256']
        return json.loads(data)
    counts={r:dict(episodes=0,old_target_coverage_pass=0,old_whole_extent_coverage_pass=0,
        new_target_coverage_pass=0,new_whole_extent_coverage_pass=0,
        initial_floor_estimate_sufficient=0,initial_floor_estimate_insufficient=0) for r in ('dev_tune','validation','sealed_test')}
    dev=[];started=time.monotonic()
    bounds=np.asarray(GENERATOR_BOUNDS_XY_M)
    corners=np.array([[x,y] for x in bounds[:,0] for y in bounds[:,1]])
    for entry in registry['entries']:
        role=entry['role'];spec=read(entry['maze'],role)
        for binding in entry['episodes']:
            e=read(binding,role);assert e['role']==spec['data_role']==role
            home=np.array(e['home_se2_world'][:2]);yaw=e['home_se2_world'][2]
            c,s=np.cos(yaw),np.sin(yaw);world_to_start=np.array([[c,s],[-s,c]])
            extent=(corners-home)@world_to_start.T
            targets=(np.array([home,e['beacon_xy_world']])-home)@world_to_start.T
            assert (home>=bounds[0]).all() and (home<=bounds[1]).all()
            old_t=bool(np.max(np.abs(targets))<=4.9);old_e=bool(np.max(np.abs(extent))<5.)
            new_t=bool(np.max(np.abs(targets))<=POINT_BOUND_M);new_e=bool(np.max(np.abs(extent))<MAP_HALF_WIDTH_M)
            n=floor_estimate(spec,e);enough=n>=100
            row=counts[role];row['episodes']+=1
            for key,value in [('old_target_coverage_pass',old_t),('old_whole_extent_coverage_pass',old_e),
                ('new_target_coverage_pass',new_t),('new_whole_extent_coverage_pass',new_e),
                ('initial_floor_estimate_sufficient',enough),('initial_floor_estimate_insufficient',not enough)]:row[key]+=int(value)
            if role=='dev_tune':dev.append(dict(episode=e['episode_id'],old_targets_pass=old_t,
                old_whole_extent_pass=old_e,new_coverage_pass=new_t and new_e,estimated_floor_quads=n,
                estimated_floor_requirement_pass=enough))
    assert sum(r['episodes'] for r in counts.values())==180
    assert all(r['new_target_coverage_pass']==r['new_whole_extent_coverage_pass']==r['episodes'] for r in counts.values())
    result=dict(schema='navigation_capability_startup_structural_checks.v1',counts=counts,
        physics_steps=0,rendered_frames=0,models_loaded=0,wall_s=time.monotonic()-started,
        registry_sha256=hashlib.sha256((base/'registry.json').read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        domain=dict(generator_bounds_xy_m=GENERATOR_BOUNDS_XY_M,maximum_displacement_m=GENERATOR_DIAGONAL_M,
            map_half_width_m=MAP_HALF_WIDTH_M,point_guard_bound_m=POINT_BOUND_M,
            derivation='ceil(generator envelope diagonal + unchanged 0.1-m guard); independent of sampled episodes'),
        floor_estimate=dict(required_ground_quads=100,nominal_body_height_m=.32,
            method='Analytic pinhole rays to ground with wall-box occlusion, then unchanged deployed floor-index geometry.',
            limitations='Nominal level body; no settling, depth noise, self-occlusion or measured gravity. Geometric estimate only, not measured startup qualification.'),
        sealed_access='Only registered new-set bound paths; aggregate counts retained; no per-episode sealed outputs.',
        dev_tune=dev)
    with (root/'result.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(output.dumps(dict(counts=counts,domain=result['domain'],wall_s=result['wall_s'])))

if __name__=='__main__':main()
