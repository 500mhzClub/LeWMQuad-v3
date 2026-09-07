"""Saved native identity/mesh witness audit; no new native scene or rendering."""
import json

import numpy as np
import trimesh

from lewm_genesis.appearance_surface_development import triangle_identity
from lewm_genesis.rgbd_motion_scene_development import independently_seeded_surfaces
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.probe_go2_rgbd_motion_scene_development_v1 import OUTPUT,pack,ARMS,APPEARANCE_SEED
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json

IDENTITIES={'launch.json':'76fd6be7675427fd4bbb275a5b0de5dba116f613948aff09eb98c36888c86744',
    'result.json':'d2a088a415a59f3cf3b7963d1bc62aa7941a547f9be30017d5711af50011ed30'}


def audit():
    bindings={str((OUTPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(bindings)
    launch=read_json(OUTPUT,'launch.json'); result=read_json(OUTPUT,'result.json'); verify(launch)
    bindings|={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    own='scripts/audit_go2_rgbd_motion_scene_preflight_v1.py';bindings[own]=digest(ROOT/own);verify_bindings(bindings)
    definition=pack(); boxes=[dict(wall_id=o.object_id,centre_xyz=o.center_xyz_m,size_xyz=o.size_xyz_m,yaw_rad=o.yaw_rad) for o in definition.static_objects]
    baseline=read_json(OUTPUT,'reference_identity.json'); visuals=[];robots=[];names={'reference_identity.json'}
    assert baseline['physics_steps']==0 and len(baseline['robot'])==27 and len(baseline['environment'])==6
    for arm in ARMS:
        directory=OUTPUT/arm; actual=read_json(directory,'physical_identity.json')
        assert actual==baseline
        native=read_json(directory,'visual_identity.json');robot=read_json(directory,'robot_native_identity.json')
        visuals.append(native);robots.append(robot)
        assert native==visuals[0] and robot==robots[0]
        assert native['robot_present'] is True and native['physics_stepped'] is False
        assert [{k:v for k,v in row.items() if k not in ('geom_id','link_id')} for row in robot]==baseline['robot']
        assert [{k:v for k,v in row.items() if k not in ('geom_id','link_id')} for row in native['physical_geometries']]==baseline['environment']
        for name,expected in independently_seeded_surfaces(boxes,arm,APPEARANCE_SEED):
            mesh=trimesh.load_mesh(directory/(name+'.ply'),file_type='ply',process=False)
            np.testing.assert_array_equal(mesh.vertices,expected.vertices.astype(np.float32))
            np.testing.assert_array_equal(mesh.faces,expected.faces)
            np.testing.assert_array_equal(mesh.visual.vertex_colors,expected.visual.vertex_colors)
            wanted=next(row['geometry'] for row in native['visual_surfaces'] if row['name']==name)
            assert triangle_identity(mesh.vertices,mesh.faces)==wanted
            names.add(f'{arm}/{name}.ply')
        names|={f'{arm}/{n}.json' for n in ('physical_identity','visual_identity','robot_native_identity')}
    assert set(result['artifact_sha256'])==names and len(names)==28
    verify(launch);verify_bindings(bindings)
    return dict(status='SAVED_ROBOT_SCENE_NATIVE_AND_VISUAL_WITNESSES_AUDITED',artifact_count=28,
        robot_collision_shapes=27,environment_collision_shapes=6,identities=IDENTITIES,
        auditor_source_sha256={own:bindings[own]},physics_steps=0,new_rendered_frames=0,
        physical_execution_validated=False,navigation_qualified=False)


if __name__=='__main__':
    target=OUTPUT/'raw_artifact_audit.json'
    if target.exists():raise ValueError('fresh saved-witness audit only')
    result=audit();write_json(target,result);print(json.dumps(result),flush=True)
