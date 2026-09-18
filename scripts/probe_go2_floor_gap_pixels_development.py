"""Diagnose original all-pixel coverage rejection in a recorded routing gap."""
import json
import hashlib
import time
import numpy as np
from lewm.multirate_routing_map_development import Geometry
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL, FOCAL
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.local_inverse_depth_floor_development import local_depth
from scripts.probe_go2_noisy_routing_floor_development import LocalCoverageMap
from scripts.live_depth_noise_session_development import NoisyPublicReplay
from scripts.compare_continuous_navigation_arms_development import BASE


def quads(mask):return mask[:-1,:-1]&mask[1:,:-1]&mask[:-1,1:]&mask[1:,1:]


def main():
    root=BASE/'go2_live_gyro_height_floor_noise_2mm_native_layout02_4800_v1_attempt_001'
    output=root/'floor_gap_pixel_rejection_probe_v1.json'
    if output.exists():raise ValueError('preserve completed probe')
    prefix=json.loads((root/'union_floor_coverage_prefix_probe_v1.json').read_text())
    route=prefix['variants']['local']['routes']['no_exclusions']
    cells=np.array(sorted({tuple(c) for c in route['unknown_connector_cells']+route['route_cells']}),int)
    poses={p['frame']:p['registered_pose'] for p in json.loads((root/'poses.json').read_text())}
    reader=NoisyPublicReplay(root/'native');initial=None;best={};visible_counts={};started=time.monotonic()
    T=np.asarray(BODY_FROM_OPTICAL)
    for item in prefix['frames']:
        frame=item['frame'];p,d,fast,rgb,aux,now=reader.packet(frame);pose=poses[frame]
        if initial is None:initial=LocalCoverageMap().update(p,d,pose,auxiliary_depth=aux,measured_ns=now)
        B=np.asarray(initial.map_from_initial);height=initial.floor_height
        Q=B@np.asarray(pose['rotation_initial_body_from_current_body']);q=B@np.asarray(pose['position_initial_body_m'])
        for camera,packet in [('primary',d),('auxiliary',aux)]:
            R,t=reference_pose(Q,q) if camera=='auxiliary' else (Q,q)
            xy=(cells[:,None,:]+np.array([[0,0],[1,0],[1,1],[0,1]]))*.05
            world=np.concatenate((xy,np.full((*xy.shape[:-1],1),height)),axis=-1)
            optical=((world-t)@R-T[:3,3])@T[:3,:3];z=optical[...,2]
            uv=optical[...,:2]/np.maximum(z[...,None],1e-12)*FOCAL+[319.5,239.5]
            lo,hi=uv.min(1)-1e-9,uv.max(1)+1e-9
            visible=((z>=.2)&(z<=5)).all(1)&(lo>=0).all(1)&(hi<[639,479]).all(1)
            if not visible.any():continue
            for variant in ('raw','local'):
                depth,valid=packet['depth_m'],packet['valid']
                if variant=='local':depth,valid=local_depth(depth,valid)
                geometry=Geometry()
                coverage=geometry.floor_coverage(depth,valid,R,t,height,cells=cells)
                mesh=geometry.index(depth,valid,R[2])['ground_cells']
                near=quads(np.abs(geometry.body_projection(depth)@R[2]+t[2]-height)<=.01)
                valid_quads=quads(valid);good=mesh&near
                for i in np.flatnonzero(visible):
                    a=coverage['projected_lower_xy'][i];b=coverage['projected_upper_xy'][i]+1
                    sl=np.s_[a[1]:b[1],a[0]:b[0]];count=good[sl].size
                    good_count=int(good[sl].sum())
                    assert (good_count==count)==bool(coverage['covered'][i])
                    key=f'{cells[i,0]},{cells[i,1]}:{variant}'
                    visible_counts[key]=visible_counts.get(key,0)+1
                    row=dict(cell=cells[i].tolist(),variant=variant,frame=frame,camera=camera,
                        projected_quads=count,passing_quads=good_count,passing_fraction=good_count/count,
                        invalid_quads=int((~valid_quads[sl]).sum()),
                        height_rejected_quads=int((~near[sl]).sum()),
                        mesh_rejected_quads=int((~mesh[sl]).sum()),
                        fully_covered=bool(coverage['covered'][i]))
                    if key not in best or row['passing_fraction']>best[key]['passing_fraction']:best[key]=row
                geometry.close()
    report=dict(mapping_frames=len(prefix['frames']),cells=cells.tolist(),best_observed_rectangle=list(best.values()),
        visible_rectangle_counts=visible_counts,selection='highest original passing-quad fraction over recorded prefix',
        rejection_counts_overlap=True,thresholds_or_classifications_changed=False,
        fixed_recorded_public_poses=True,actual_delivered_noisy_packets_verified=True,
        native_physics_used=False,new_navigation_executed=False,wall_seconds=time.monotonic()-started,
        source_sha256={__file__:hashlib.sha256(open(__file__,'rb').read()).hexdigest()})
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
