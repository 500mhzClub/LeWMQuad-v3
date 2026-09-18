"""Evaluator-only primary/auxiliary raster, metric and physical visibility audit."""
import hashlib
import numpy as np
from PIL import Image
from lewm.auxiliary_tilted_depth_geometry_development import body_from_optical,CALIBRATION_ID
from lewm.causal_depth_observation_development import INTRINSICS
from lewm.native_link_segmentation_map_development import decode
from lewm.physical_execution_development import rotation_xyzw
from lewm.raster_footprint_visibility_development import evaluate_footprint
from scripts.near_field_sensor_audit_development import read_json,read_npz
from scripts.geometry_progress_family_audit_development import validate_rasters as validate_primary_precision


def audit_rasters_and_footprints(directory,spec,cameras,sensors):
    records=[read_json(directory,f'raster_{i:04d}.json') for i in range(len(cameras))];normalized=[]
    for record in records:
        order=record['order']
        if (set(order)!={'order','surfaces','robot_visual_geometries','total_nodes'}
                or order['order']!='floor_walls_robot_visual_geometry_order' or order['robot_visual_geometries']!=33
                or order['total_nodes']!=35 or order!=records[0]['order']):
            raise ValueError('fixed complete floor/wall/33-robot-node raster required')
        normalized.append(record|dict(order=dict(order='floor_first',roles=['floor','walls'],surfaces=order['surfaces'])))
    # The original precision and static-surface identity checks are applied only
    # after separately validating the additional robot population above.
    validate_primary_precision(normalized,cameras);scores=[]
    for i,camera in enumerate(cameras):
        native=read_npz(directory,f'native_depth_{i:04d}.npz')['optical_depth_m']
        score=evaluate_footprint(native,spec['geometry']['wall_boxes'],camera['world_from_optical'],render_near_m=.005)
        assert score['original_strict_score']==sensors['depth_checks'][i]['physical_visibility']
        scores.append(score)
    return scores


def audit_auxiliary(directory,spec,raw,cameras,topology,result):
    rows=read_json(directory,'auxiliary_camera_audit.json')
    assert len(rows)==len(cameras)==result['auxiliary_frames']
    E=body_from_optical();reports=[];robot_entity=len(spec['geometry']['wall_boxes'])+1
    sampling=dict(draw_framebuffer_is_single_sample_target=True,draw_framebuffer_is_multisample_target=False,
        samples=0,sample_buffers=0,multisample_enabled=False,pixel_scale=1)
    for i,(row,primary) in enumerate(zip(rows,cameras,strict=True)):
        sample=749+50*i;stamp=int(round(float(raw['timestamp_s'][sample])*1e9))
        assert row['frame']==i and row['physical_sample_index']==primary['physical_sample_index']==sample
        assert row['measured_ns']==stamp and row['calibration_id']==CALIBRATION_ID
        assert row['same_physical_sample_as_primary'] and row['robot_visualization_enabled']
        assert row['segmentation_is_evaluator_only'] and row['sampling_readback']==sampling
        assert row['native_near_m']==.005 and row['native_far_m']==200.
        np.testing.assert_allclose(row['native_intrinsics'],INTRINSICS,rtol=0,atol=1e-7)
        np.testing.assert_array_equal(row['body_from_optical'],E)
        pose=raw['base_pose_world'][sample];R=rotation_xyzw(pose[3:]);H=np.eye(4)
        H[:3,:3]=R@E[:3,:3];H[:3,3]=pose[:3]+R@E[:3,3]
        np.testing.assert_allclose(row['world_from_optical'],H,rtol=0,atol=1e-12)
        with np.load(directory/f'auxiliary_depth_{i:04d}.npz',allow_pickle=False) as z:
            native=z['native_optical_depth_m'];depth=z['depth_m'];valid=z['valid'];seg=z['diagnostic_segmentation']
        assert native.shape==depth.shape==valid.shape==seg.shape==(480,640)
        assert native.dtype==depth.dtype==np.float32 and valid.dtype==bool and seg.dtype.kind in 'iu'
        assert hashlib.sha256(native.tobytes()).hexdigest()==row['native_depth_sha256']
        assert hashlib.sha256(seg.tobytes()).hexdigest()==row['diagnostic_segmentation_sha256']
        with Image.open(directory/f'auxiliary_rgb_{i:04d}.png') as im:rgb=np.array(im)
        assert hashlib.sha256(rgb.tobytes()).hexdigest()==row['rgb_sha256']
        mask=np.isfinite(native)&(native>=.2)&(native<=5.)
        assert np.array_equal(mask,valid) and np.array_equal(depth,np.where(mask,native,np.float32(0.)))
        mapping={int(k):tuple(v) if isinstance(v,list) else v for k,v in row['diagnostic_segmentation_map'].items()}
        clean,ids=decode(mapping,robot_entity)
        assert clean==row['diagnostic_segmentation_map'] and ids==row['robot_segmentation_ids']
        assert all(mapping[k][1] in topology['robot_link_ids'] for k in ids)
        robot_pixels=int(np.isin(seg,ids).sum());assert robot_pixels==row['robot_pixels']
        assert int(valid.sum())==row['valid_depth_pixels']
        score=evaluate_footprint(native,spec['geometry']['wall_boxes'],H,render_near_m=.005)
        passed=bool(robot_pixels==0 and score['stable_interior_metric_pass'] and not score['near_occlusion_failure']
            and score['original_strict_score']['passes_sampled_physical_visibility'])
        reports.append(dict(frame=i,raw_auxiliary_reconstruction_exact=True,robot_pixels=robot_pixels,
            footprint_checks=score,auxiliary_visibility_pass=passed,
            self_occluded_samples_qualified=False,native_geometry_is_evaluator_only=True))
    return reports
