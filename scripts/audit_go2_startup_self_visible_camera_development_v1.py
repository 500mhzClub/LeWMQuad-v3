"""Saved startup renders: sparse independent mesh rays and exact artifact audit."""
import json

import numpy as np
import trimesh

from lewm.causal_depth_observation_development import FOCAL
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_startup_self_visible_camera_development_v1 import OUTPUT
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

IDENTITIES={'launch.json':'6a3c2fbdddb7ffd2db550e7e659db18ef9643c53fdaee09af467106e3ec46ec9',
 'result.json':'5df4a458504ef413b92e3a232c663ddccbda318636cd76b69b58fcd8f74ddc15'}


def main():
    if (OUTPUT/'sparse_ray_audit_launch.json').exists(): raise ValueError('one-shot saved-render audit only')
    inputs={str((OUTPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()}; verify_bindings(inputs)
    launch=read_json(OUTPUT,'launch.json'); verify(launch); result=read_json(OUTPUT,'result.json')
    inputs|={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}; verify_bindings(inputs)
    sources=discover_sources(('scripts/audit_go2_startup_self_visible_camera_development_v1.py',),launch['source_sha256'])
    audit=dict(source_sha256=sources,input_sha256=inputs,grid='u=16+32i i0..19;v=16+32j j0..14',
        scope='independent sparse triangle ray intersection against saved visual mesh; not all-pixel or physical validation',
        depth_tolerance_m=.001,renderer_near_m=.05)
    write_json(OUTPUT/'sparse_ray_audit_launch.json',audit)
    try:
        mesh=trimesh.load_mesh(OUTPUT/'visible/robot_visual.ply',process=False,file_type='ply')
        cameras=read_json(OUTPUT/'visible','cameras.json'); controls=read_json(OUTPUT/'background','cameras.json'); rows={}
        vv,uu=np.meshgrid(np.arange(16,480,32),np.arange(16,640,32),indexing='ij'); u,v=uu.ravel(),vv.ravel()
        for name,camera in cameras.items():
            if camera!=controls[name] or camera['physics_steps']!=0: raise ValueError('matched zero-step camera witness required')
            T=np.asarray(camera['world_from_optical']); origin=T[:3,3]
            directions=np.stack(((u+.5-320)/FOCAL,(v+.5-240)/FOCAL,np.ones(len(u))),axis=1)@T[:3,:3].T
            with np.load(OUTPUT/'visible'/(name+'.npz'),allow_pickle=False) as raw: visible=raw['optical_depth_m'][v,u]
            with np.load(OUTPUT/'background'/(name+'.npz'),allow_pickle=False) as raw: background=raw['optical_depth_m'][v,u]
            locations,rays,triangles=mesh.ray.intersects_location(np.tile(origin,(len(u),1)),directions,multiple_hits=True)
            hit_z=((locations-origin)@T[:3,:3])[:,2]
            closest=np.full(len(u),np.inf); raw_closest=np.full(len(u),np.inf)
            np.minimum.at(raw_closest,rays[hit_z>0],hit_z[hit_z>0])
            good=hit_z>=.05; np.minimum.at(closest,rays[good],hit_z[good])
            z=np.full(len(u),np.inf); down=directions[:,2]<-1e-12; z[down]=-origin[2]/directions[down,2]
            floor=down&(z>=.2)&(z<=5)&(abs(background-z)<=.001)
            expected=np.minimum(closest,z); errors=abs(visible[floor]-expected[floor])
            rows[name]=dict(grid_rays=len(u),background_floor_rays=int(floor.sum()),triangle_hits=len(hit_z),
                maximum_floor_domain_depth_error_m=float(errors.max()) if len(errors) else None,
                floor_domain_depth_mismatches=int((errors>.001).sum()),
                robot_intersections_before_renderer_near=int((raw_closest<.05).sum()),
                raw_floor_domain_visible_depth_m=visible[floor].tolist(),expected_depth_m=expected[floor].tolist())
            print('SPARSE_ROBOT_RAY_AUDIT',name,rows[name]['floor_domain_depth_mismatches'],flush=True)
        verify_bindings(sources|inputs)
        write_json(OUTPUT/'sparse_ray_audit.json',dict(status='SPARSE_RAY_AUDIT_COMPLETE',cameras=rows,
            total_mismatches=sum(r['floor_domain_depth_mismatches'] for r in rows.values()),
            all_pixels_verified=False,robot_mesh_shared=True,physical_uncertainty_calibrated=False))
    except Exception as error:
        write_json(OUTPUT/'sparse_ray_audit_failure.json',dict(status='TERMINAL_SPARSE_RAY_AUDIT_FAILURE',reason=repr(error))); raise


if __name__=='__main__': main()
