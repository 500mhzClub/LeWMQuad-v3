"""Independent saved-RGB/depth reconstruction and cross-arm geometry audit."""
from copy import deepcopy
import json

import cv2
import numpy as np
from PIL import Image
import trimesh

from lewm.causal_depth_observation_development import from_native_depth
from lewm.physical_execution_development import rotation_xyzw
from lewm.rgbd_correspondence_motion_development import RGBDCorrespondenceMotion
from lewm.rgbd_dataset_development import load_rgbd_observation
from lewm.visual_surface_depth_evaluation_development import expected_optical_depth
from lewm_genesis.appearance_surface_development import surfaces, triangle_identity
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify
from scripts.run_go2_appearance_information_meshset_development_v2 import OUTPUT, ARMS, B, FRAME_INDICES, boxes_from_recording
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json

IDENTITIES = {'launch.json':'4e7b4bf77c3449d77bae09d7426a40e3b320409bc937afc2eace475f5ad5efb7',
              'result.json':'b0689df36aaf202058cd5d85ba88c0cee34391def6496b3259d2e72072e43b4f'}


def depth_array(directory, index):
    with np.load(directory/f'depth_{index:04d}.npz', allow_pickle=False) as data:
        assert data.files == ['optical_depth_m']
        result = data['optical_depth_m'].copy()
    assert result.shape == (480,640) and result.dtype == np.float32
    return result


def audit():
    cv2.setNumThreads(1)
    assert set(IDENTITIES) == {'launch.json','result.json'}
    bindings = {str((OUTPUT/n).relative_to(ROOT)): h for n,h in IDENTITIES.items()}
    verify_bindings(bindings)
    launch = read_json(OUTPUT,'launch.json'); result = read_json(OUTPUT,'result.json')
    assert result['status'] == 'MATCHED_APPEARANCE_RENDER_ASSAY_COMPLETE'
    verify(launch)
    bindings |= {str((OUTPUT/n).relative_to(ROOT)): h for n,h in result['artifact_sha256'].items()}
    own = 'scripts/audit_go2_appearance_information_meshset_development_v2.py'
    bindings[own] = digest(ROOT/own); verify_bindings(bindings)
    boxes = boxes_from_recording(); cameras = read_json(B,'camera_audit.json')
    expected_names = {'scored_pairs.json'}
    for arm in ARMS:
        expected_names |= {arm+'/'+n for n in ('native_identity.json','terminal_native_identity.json','sensor_predictions.json','camera_evaluation.json')}
        expected_names |= {arm+'/'+n+'.ply' for n in ['ground_visual',*[b['wall_id']+'_visual' for b in boxes]]}
        expected_names |= {f'{arm}/{kind}_{i:04d}.{suffix}' for i in range(14) for kind,suffix in (('rgb','png'),('depth','npz'))}
    assert set(result['artifact_sha256']) == expected_names
    reference_identity = read_json(OUTPUT/'neutral','native_identity.json')
    all_predictions = {}; statistics = {}; maximum_ray_error = 0.
    for arm_index, arm in enumerate(ARMS):
        directory = OUTPUT/arm
        assert read_json(directory,'native_identity.json') == reference_identity
        assert read_json(directory,'terminal_native_identity.json') == reference_identity
        for name, wanted in surfaces(boxes, arm):
            loaded = trimesh.load_mesh(directory/(name+'.ply'),file_type='ply',process=False)
            np.testing.assert_array_equal(loaded.vertices, wanted.vertices.astype(np.float32))
            np.testing.assert_array_equal(loaded.faces, wanted.faces)
            np.testing.assert_array_equal(loaded.visual.vertex_colors, wanted.visual.vertex_colors)
            identity = next(r['geometry'] for r in reference_identity['visual_surfaces'] if r['name']==name)
            assert triangle_identity(loaded.vertices,loaded.faces) == identity
        saved = read_json(directory,'sensor_predictions.json')
        captures = read_json(directory,'camera_evaluation.json')
        assert len(saved)==len(captures)==len(FRAME_INDICES)==14
        observer = RGBDCorrespondenceMotion()
        for index, frame in enumerate(FRAME_INDICES):
            native = depth_array(directory,index)
            np.testing.assert_array_equal(native,depth_array(OUTPUT/'neutral',index))
            transform = np.asarray(cameras[frame]['world_from_optical'])
            assert captures[index]['source_B_frame']==frame and captures[index]['scene_steps']==0
            np.testing.assert_array_equal(captures[index]['world_from_optical'],transform)
            reference = expected_optical_depth(boxes,transform,stride=8,floor_z_m=0.)
            true = reference['expected_depth_m']; good = reference['surface_interior']&(true>.22)&(true<4.98)
            errors = np.abs(native[np.ix_(reference['rows'],reference['columns'])][good]-true[good])
            assert len(errors)>1000 and np.isfinite(errors).all() and errors.max()<=.001
            assert len(errors)==captures[index]['interior_depth_rays']
            assert float(errors.max())==captures[index]['maximum_depth_error_m']
            maximum_ray_error=max(maximum_ray_error,float(errors.max()))
            with Image.open(directory/f'rgb_{index:04d}.png') as image:
                rgb=np.asarray(image).copy()
            policy,_=load_rgbd_observation(B,frame); fast=load_fast_packet(B,frame)
            policy=deepcopy(policy); fast=deepcopy(fast)
            policy['image']['rgb']=rgb
            policy['sensor_state']['identity']=(2,arm_index,0); fast['identity']=(2,arm_index,0)
            now=policy['sensor_state']['decision_ns']
            depth=from_native_depth(native,policy,measured_ns=now,available_ns=now,now_ns=now)
            prediction=observer.observe(policy,depth,fast,now_ns=now)
            assert saved[index]['source_B_frame']==frame
            assert prediction=={k:v for k,v in saved[index].items() if k not in ('source_B_frame','observer_wall_ms')}
        all_predictions[arm]=saved
        motion=[r['motion'] for r in saved[1:]]
        statistics[arm]=dict(reconstructed_frames=14,pairs=13,
            accepted=sum(r['translation_previous_body_m'] is not None for r in motion),
            minimum_keypoints=min(r['current_keypoints'] for r in motion),
            maximum_keypoints=max(r['current_keypoints'] for r in motion),
            minimum_inliers=min(r['inliers'] for r in motion),
            maximum_inliers=max(r['inliers'] for r in motion))
    # Scoring only after all 42 predictions have been reconstructed sensor-only.
    scores=read_json(OUTPUT,'scored_pairs.json')
    with np.load(B/'physics_trace.npz',allow_pickle=False) as raw:
        for arm, tape in all_predictions.items():
            assert len(scores[arm])==13
            errors=[]
            for previous,current,scored in zip(tape[:-1],tape[1:],scores[arm],strict=True):
                i,j=[cameras[r['source_B_frame']]['physical_sample_index'] for r in (previous,current)]
                a,b=raw['base_pose_world'][i],raw['base_pose_world'][j]
                true=rotation_xyzw(a[3:]).T@(b[:3]-a[:3])
                motion=current['motion']; prediction=motion['translation_previous_body_m']
                error=None if prediction is None else float(np.linalg.norm(np.asarray(prediction)-true))
                assert scored==dict(source_B_frame=current['source_B_frame'],status=motion['status'],
                    translation_error_m=error,keypoints=motion['current_keypoints'],matches=motion['mutual_ratio_matches'],
                    lifted=motion['lifted_matches'],inliers=motion['inliers'],observer_wall_ms=current['observer_wall_ms'])
                if error is not None:errors.append(error)
            statistics[arm]|=dict(maximum_error_m=max(errors,default=None),mean_error_m=float(np.mean(errors)) if errors else None)
            summary=result['summaries'][arm]
            assert summary['pairs']==13 and summary['accepted']==len(errors)
            assert summary['maximum_error_m']==max(errors,default=None)
    verify(launch); verify_bindings(bindings)
    return dict(status='APPEARANCE_RGBD_RECONSTRUCTION_AND_GEOMETRY_AUDIT_PASS',statistics=statistics,
        artifact_count=len(expected_names),all_cross_arm_depth_arrays_bit_identical=True,
        maximum_interior_ray_error_m=maximum_ray_error,identities=IDENTITIES,
        auditor_source_sha256={own:bindings[own]},native_pose_used_for_reconstruction=False,
        threshold_changed=False,original_B_result_changed=False,navigation_qualified=False)


if __name__=='__main__':
    target=OUTPUT/'raw_artifact_audit.json'
    if target.exists():raise ValueError('fresh independent audit only')
    result=audit(); write_json(target,result); print(json.dumps(result),flush=True)
