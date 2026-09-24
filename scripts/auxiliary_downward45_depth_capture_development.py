"""Actual secondary optical view at an unchanged native physical sample."""
import hashlib
import time
import numpy as np
from PIL import Image
from lewm.native_link_segmentation_map_development import decode
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical,CALIBRATION_ID
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL,INTRINSICS
from lewm.physical_execution_development import rotation_xyzw
from lewm.physical_semantics import world_from_optical
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from lewm_genesis.visible_robot_raster_order_development import verify_order
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.run_go2_contact_attributed_execution_development_v1 import array


def capture(session,directory,frame):
    build=session.ctx.build;robot,camera=build.robot,build.camera;started=time.perf_counter()
    if not robot.morph.visualization or not robot.morph.collision:raise ValueError('visible physical robot required')
    p=array(robot.get_pos()).reshape(-1,3)[0];q=array(robot.get_quat()).reshape(-1,4)[0]
    R=rotation_xyzw(q[[1,2,3,0]]);E=body_from_optical();T=np.asarray(BODY_FROM_OPTICAL)
    position=p+R@E[:3,3];optical_R=R@E[:3,:3];H=np.eye(4);H[:3,:3]=optical_R;H[:3,3]=position
    before=(int(session.ctx.runner._sim_time_ns),len(session.samples));primary=np.array(camera.transform,copy=True)
    try:
        camera.set_pose(pos=position,lookat=position+optical_R[:,2],up=-optical_R[:,1])
        check_optical_pose(camera.transform,H)
        if (tuple(camera.res)!=(640,480) or not np.allclose(camera.intrinsics,INTRINSICS,atol=1e-7,rtol=0)
                or camera.near!=.005 or camera.far!=200. or camera._raytracer is not None or camera._batch_renderer is not None):
            raise ValueError('exact reviewed auxiliary raster geometry required')
        rgb_render=camera.render(rgb=True,depth=False,segmentation=False,normal=False)
        depth_render=camera.render(rgb=False,depth=True,segmentation=False,normal=False)
        sampling=sampling_readback(camera)
        segmentation_render=camera.render(rgb=False,depth=False,segmentation=True,colorize_seg=False,normal=False)
        rgb=np.asarray(session.ctx.runner._extract_rgb(rgb_render)).reshape(480,640,3)
        native=np.asarray(depth_render[1]).reshape(480,640);seg=np.asarray(segmentation_render[2]).reshape(480,640)
        if rgb.dtype!=np.uint8 or native.dtype!=np.float32 or seg.dtype.kind not in 'iu':raise ValueError('native auxiliary pixel encoding changed')
        context=camera._rasterizer._context
        if context.segmentation_level!='link':raise ValueError('explicit diagnostic link segmentation required')
        mapping,robot_ids=decode(context.seg_idxc_map,int(robot.idx))
        if not robot_ids:raise ValueError('robot visual links absent from native segmentation map')
        valid=np.isfinite(native)&(native>=.2)&(native<=5.);depth=np.where(valid,native,np.float32(0.))
        np.savez_compressed(directory/f'auxiliary_depth_{frame:04d}.npz',native_optical_depth_m=native,
            depth_m=depth,valid=valid,diagnostic_segmentation=seg)
        Image.fromarray(rgb).save(directory/f'auxiliary_rgb_{frame:04d}.png')
        check_optical_pose(camera.transform,H);verify_order(build,session.raster_order)
        if before!=(int(session.ctx.runner._sim_time_ns),len(session.samples)):raise ValueError('physics advanced during auxiliary acquisition')
        return dict(frame=frame,physical_sample_index=before[1]-1,measured_ns=before[0],
            calibration_id=CALIBRATION_ID,body_from_optical=E.tolist(),world_from_optical=H.tolist(),
            native_intrinsics=np.asarray(camera.intrinsics).tolist(),native_near_m=camera.near,native_far_m=camera.far,
            native_depth_sha256=hashlib.sha256(native.tobytes()).hexdigest(),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            diagnostic_segmentation_sha256=hashlib.sha256(seg.tobytes()).hexdigest(),
            diagnostic_segmentation_map=mapping,robot_segmentation_ids=robot_ids,
            robot_pixels=int(np.isin(seg,robot_ids).sum()),valid_depth_pixels=int(valid.sum()),
            same_physical_sample_as_primary=True,robot_visualization_enabled=True,
            sampling_readback=sampling,capture_wall_s=time.perf_counter()-started,
            segmentation_is_evaluator_only=True,zero_acquisition_latency_assumed=True,hardware_mount_validated=False)
    finally:
        cp=p+R@T[:3,3];camera.set_pose(pos=cp,lookat=cp+R[:,0],up=R[:,2])
        check_optical_pose(camera.transform,world_from_optical(cp,R[:,0],R[:,2]))
        if not np.array_equal(primary,camera.transform):raise ValueError('primary camera transform not restored exactly')


