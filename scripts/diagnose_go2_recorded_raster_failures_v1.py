"""Read-only localization of two frozen raster failures; never rescore or repair."""
import itertools
import json
from pathlib import Path
import numpy as np
from PIL import Image
from lewm.causal_depth_observation_development import FOCAL
from lewm.physical_first_surface_depth_development import expected_optical_depth, evaluate_visibility
from scripts.navigation_artifact_root_development import BASE, verify_artifacts
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json, read_npz

OUTPUT = Path('docs/go2_recorded_raster_failure_diagnostic_2026-09-06.json')
SOURCE = 'scripts/diagnose_go2_recorded_raster_failures_v1.py'
UNION = BASE/'go2_union_wall_rgb_repeatability_probe_v1_attempt_001'
BATCH = BASE/'go2_independent_layout_collection_v1_l00_attempt_001'
UNION_IDS = {'launch.json':'631d327beaa45807f42aa0f72234c124651141e569f354304ba249afaf97630d',
             'result.json':'6a64c9e07d87ae91f0a07c46189dc74b07b1f9625c87c4a7a54f2799c617189d'}
BATCH_IDS = {'launch.json':'1d46515179b91d7134c83b6247f0cc22db1de786e3a99a1db9bde82ca85fb218',
             'failure.json':'b97a6d5d3dad66fdd0fbffaad811dfb3b6feee5e9f4c7363e8aec3f5a04a556b'}


def projected_edges(boxes, transform):
    """All physical box edges, clipped to positive 5mm optical depth for diagnosis."""
    T = np.asarray(transform, float); result = []
    signs = list(itertools.product((-1, 1), repeat=3))
    for box in boxes:
        if box['yaw_rad'] != 0: raise ValueError('fixed axis-aligned diagnostic population')
        world = np.asarray(box['centre_xyz'])+np.asarray(signs)*np.asarray(box['size_xyz'])/2
        optical = (world-T[:3,3])@T[:3,:3]
        for i,j in itertools.combinations(range(8), 2):
            if sum(a != b for a,b in zip(signs[i], signs[j])) != 1: continue
            a,b = optical[i].copy(),optical[j].copy()
            if max(a[2], b[2]) <= .005: continue
            if a[2] < .005: a += (.005-a[2])/(b[2]-a[2])*(b-a)
            if b[2] < .005: b += (.005-b[2])/(a[2]-b[2])*(a-b)
            uv = np.stack((a,b)); uv = uv[:,:2]/uv[:,2,None]*FOCAL+[320,240]
            if np.linalg.norm(uv[1]-uv[0]) < 1e-12: continue
            result.append((box['wall_id'], uv, world[[i,j]]))
    return result


def nearest_edge(edges, row, column):
    point = np.array([column+.5,row+.5]); best = None
    for name,uv,xyz in edges:
        d = uv[1]-uv[0]; t = np.clip((point-uv[0])@d/(d@d),0,1)
        distance = float(np.linalg.norm(point-(uv[0]+t*d)))
        if best is None or distance < best['distance_pixels']:
            best = dict(wall_id=name,distance_pixels=distance,world_endpoints=xyz.tolist(),
                        image_endpoints=uv.tolist())
    return best


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive diagnostic output')
    source_sha = digest(Path(SOURCE))
    verify_artifacts(UNION, UNION_IDS); verify_artifacts(BATCH, BATCH_IDS)
    ul,bl = read_json(UNION,'launch.json'),read_json(BATCH,'launch.json')
    verify(ul); verify(bl)
    ur,bf = read_json(UNION,'result.json'),read_json(BATCH,'failure.json')
    ub = ur['artifact_sha256']; verify_artifacts(UNION,ub)
    bb = BATCH_IDS | bf['commits'] | bf['prechecks']; verify_artifacts(BATCH,bb)
    commit = read_json(BATCH,'episode_063_commit.json'); bb |= commit['artifact_sha256']
    verify_artifacts(BATCH,bb)
    rgb_rows=[]
    for i in (0,1):
        a = np.array(Image.open(UNION/f'repeat_0/rgb_{i:04d}.png'))
        b = np.array(Image.open(UNION/f'repeat_1/rgb_{i:04d}.png'))
        edges=projected_edges(ul['scene_specification']['geometry']['wall_boxes'],ul['world_from_optical_poses'][i])
        changed=np.argwhere(np.any(a != b,axis=2))
        assert len(changed)==ur['comparisons'][i]['changed_rgb_pixels']
        rgb_rows.append(dict(pose_index=i,changed_pixels=[dict(row=int(y),column=int(x),
            rgb_a=a[y,x].tolist(),rgb_b=b[y,x].tolist(),nearest_physical_edge=nearest_edge(edges,y,x)) for y,x in changed]))
    directory=BATCH/commit['trial']; spec=read_json(directory,'specification.json')
    T=np.asarray(read_json(directory,'camera_audit.json')[11]['world_from_optical'])
    native=read_npz(directory,'native_depth_0011.npz')['optical_depth_m']
    boxes=spec['geometry']['wall_boxes']; ref=expected_optical_depth(boxes,T)
    expected=ref['expected_depth_m']; measured=native[np.ix_(ref['rows'],ref['columns'])]
    report=evaluate_visibility(native,boxes,T,render_near_m=.005)
    assert report==read_json(BATCH,'episode_063_raw_precheck.json')['report']['depth_checks'][11]['physical_visibility']
    bad=ref['surface_interior'] & np.isfinite(expected) & (expected<4.98) & (np.abs(measured-expected)>.001)
    edges=projected_edges(boxes,T); depth_rows=[]
    for y,x in np.argwhere(bad):
        v,u=int(ref['rows'][y]),int(ref['columns'][x]); ray=np.array([(u+.5-320)/FOCAL,(v+.5-240)/FOCAL,1.])
        direction=T[:3,:3]@ray
        depth_rows.append(dict(row=v,column=u,expected_m=float(expected[y,x]),native_m=float(measured[y,x]),
            expected_object=ref['object_names'][ref['object_index'][y,x]],
            expected_hit=(T[:3,3]+expected[y,x]*direction).tolist(),
            native_ray_endpoint=(T[:3,3]+measured[y,x]*direction).tolist(),
            native_3x3=native[v-1:v+2,u-1:u+2].tolist(),nearest_physical_edge=nearest_edge(edges,v,u)))
    verify(ul); verify(bl); verify_artifacts(UNION,UNION_IDS|ub); verify_artifacts(BATCH,bb)
    assert digest(Path(SOURCE))==source_sha
    write_json(OUTPUT,dict(status='RECORDED_RASTER_FAILURES_LOCALIZED_NOT_RESCORED',source_sha256={SOURCE:source_sha},
        union_input_sha256=UNION_IDS|ub,batch_input_sha256=bb,rgb_differences=rgb_rows,
        depth_trial=commit['trial'],depth_frame=11,original_visibility_result=report,failed_sampled_rays=depth_rows,
        limitations=['Projected box edges include hidden/internal edges; proximity is diagnostic, not causal proof.',
        'No changed threshold, pixel correction, rerender, training eligibility or native precision qualification.'],
        original_failures_preserved=True,goal_achieved=False))
    print(json.dumps(dict(output=str(OUTPUT),sha256=digest(OUTPUT),rgb=rgb_rows,depth=depth_rows)))


if __name__=='__main__': main()
