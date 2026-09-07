"""Post-audit explanation only: both-sided versus front-facing saved mesh rays."""
import numpy as np
import trimesh

from lewm.causal_depth_observation_development import FOCAL
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.run_go2_startup_self_visible_camera_development_v1 import OUTPUT
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources


def main():
    if (OUTPUT/'ray_disagreement_launch.json').exists(): raise ValueError('exclusive post-audit diagnosis')
    old=read_json(OUTPUT,'sparse_ray_audit_launch.json'); verify_bindings(old['source_sha256']|old['input_sha256'])
    inputs=old['input_sha256']|{str((OUTPUT/n).relative_to(ROOT)):digest(OUTPUT/n) for n in ('sparse_ray_audit.json','sparse_ray_audit_launch.json')}
    sources=discover_sources(('scripts/diagnose_go2_startup_camera_ray_disagreement_v1.py',),old['source_sha256'])
    write_json(OUTPUT/'ray_disagreement_launch.json',dict(source_sha256=sources,input_sha256=inputs,
        scope='POST-HOC explanation, not revised success criterion or renderer change; retain both-sided audit failure'))
    mesh=trimesh.load_mesh(OUTPUT/'visible/robot_visual.ply',process=False,file_type='ply'); rows={}
    vv,uu=np.meshgrid(np.arange(16,480,32),np.arange(16,640,32),indexing='ij'); u,v=uu.ravel(),vv.ravel()
    for name,camera in read_json(OUTPUT/'visible','cameras.json').items():
        T=np.asarray(camera['world_from_optical']); origin=T[:3,3]
        directions=np.stack(((u+.5-320)/FOCAL,(v+.5-240)/FOCAL,np.ones(len(u))),axis=1)@T[:3,:3].T
        locations,rays,triangles=mesh.ray.intersects_location(np.tile(origin,(len(u),1)),directions,multiple_hits=True)
        zhit=((locations-origin)@T[:3,:3])[:,2]; front=np.einsum('ij,ij->i',mesh.face_normals[triangles],directions[rays])<0
        zfloor=np.full(len(u),np.inf); down=directions[:,2]<-1e-12; zfloor[down]=-origin[2]/directions[down,2]
        with np.load(OUTPUT/'visible'/(name+'.npz'),allow_pickle=False) as f: visible=f['optical_depth_m'][v,u]
        with np.load(OUTPUT/'background'/(name+'.npz'),allow_pickle=False) as f: background=f['optical_depth_m'][v,u]
        floor=down&(zfloor>=.2)&(zfloor<=5)&(abs(background-zfloor)<=.001)
        expected={}
        for label,selection in [('both_sided',zhit>=.05),('front_facing',(zhit>=.05)&front)]:
            nearest=np.full(len(u),np.inf); np.minimum.at(nearest,rays[selection],zhit[selection]); expected[label]=np.minimum(nearest,zfloor)
        bad={k:floor&(abs(visible-e)>.001) for k,e in expected.items()}
        rows[name]=dict(both_sided_mismatches=int(bad['both_sided'].sum()),front_facing_mismatches=int(bad['front_facing'].sum()),
            mismatches_removed_by_backface_exclusion=int((bad['both_sided']&~bad['front_facing']).sum()),
            residual_pixels=[dict(u=int(u[i]),v=int(v[i]),visible_depth=float(visible[i]),front_facing_expected_depth=float(expected['front_facing'][i]))
                for i in np.flatnonzero(bad['front_facing'])])
        print('POSTHOC_RAY_DIAGNOSIS',name,rows[name],flush=True)
    verify_bindings(sources|inputs)
    write_json(OUTPUT/'ray_disagreement.json',dict(status='POSTHOC_RAY_DIAGNOSIS_COMPLETE',cameras=rows,
        original_audit_failure_retained=True,sensor_visibility_qualified=False))


if __name__=='__main__': main()
