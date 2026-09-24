"""Direct paired RGB-D packets; persist captured arrays after the timed run."""
import hashlib
import io
import json
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image

from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL,INTRINSICS,from_native_depth
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth as auxiliary_depth
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.simulated_body_observation_development import SCHEMAS
from lewm.physical_execution_development import rotation_xyzw
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from lewm_genesis.visible_robot_raster_order_development import verify_order
from scripts.paced_native_session_development import PacedNativeSession
from scripts.rgbd_shadow_motion_session_development import appearance_environment_identity
from scripts.single_sample_rgbd_session_development import sampling_readback
from scripts.run_go2_contact_attributed_execution_development_v1 import array
from scripts.run_go2_causal_rgb_body_capture_development_v1 import ObservationSession
from scripts.fast_gyro_scan_session_development import FastGyroSession
from scripts.raw_depth_archive_development import SCHEMA as RAW_DEPTH_SCHEMA, packet_digest

ARCHIVE_WORKERS=4


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def save_depth_archive(path,*,compression=zipfile.ZIP_DEFLATED,compresslevel=1,**arrays):
    """Standard lossless NPZ with lower compression cost after timed execution."""
    with zipfile.ZipFile(path,'x',compression=compression,compresslevel=compresslevel) as archive:
        for name,array in arrays.items():
            payload=io.BytesIO()
            np.lib.format.write_array(payload,np.asarray(array),allow_pickle=False)
            archive.writestr(name+'.npy',payload.getvalue())


def persist_camera_pair(row, directory, *, native_depth_only=False,
        compression=zipfile.ZIP_DEFLATED, compresslevel=1):
    frame=row['frame'];hashes={}
    for label,(rgb,native),depth in zip(('primary','auxiliary'),row['images'],
            (row['depth'],row['auxiliary_depth']),strict=True):
        rgb_name=f'rgb_{frame:04d}.png' if label=='primary' else f'auxiliary_rgb_{frame:04d}.png'
        Image.fromarray(rgb).save(directory/rgb_name,compress_level=1)
        arrays = dict(native_optical_depth_m=native)
        if not native_depth_only:
            arrays.update(depth_m=depth['depth_m'],valid=depth['valid'])
        save_depth_archive(directory/f'{label}_depth_{frame:04d}.npz',
            compression=compression, compresslevel=compresslevel, **arrays)
        hashes[label]=dict(rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            native_depth_sha256=hashlib.sha256(native.tobytes()).hexdigest())
        if native_depth_only:
            hashes[label]['derived_packet_sha256'] = packet_digest(depth)
    return {k:row[k] for k in ('frame','measured_ns','physical_sample_index','transforms','acquisition_wall_ms')}|dict(pixel_sha256=hashes)


class InMemoryPairedCameraSession(PacedNativeSession):
    native_depth_only = False
    archive_compression = zipfile.ZIP_DEFLATED
    archive_compression_level = 1
    archive_workers = ARCHIVE_WORKERS
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.captured_pairs=[]

    def _render_pair(self):
        robot,camera=self.ctx.build.robot,self.ctx.build.camera
        if tuple(camera.res)!=(640,480) or camera.near!=.005 or camera.far!=200.:
            raise ValueError('original calibrated raster geometry required')
        if not np.allclose(camera.intrinsics,INTRINSICS,atol=1e-7,rtol=0):
            raise ValueError('original intrinsics required')
        p=array(robot.get_pos()).reshape(-1,3)[0]
        q=array(robot.get_quat()).reshape(-1,4)[0]
        R=rotation_xyzw(q[[1,2,3,0]])
        before=(int(self.ctx.runner._sim_time_ns),len(self.samples))
        images=[];transforms=[]
        for E in (np.asarray(BODY_FROM_OPTICAL),body_from_optical()):
            H=np.eye(4);H[:3,:3]=R@E[:3,:3];H[:3,3]=p+R@E[:3,3]
            camera.set_pose(pos=H[:3,3],lookat=H[:3,3]+H[:3,2],up=-H[:3,1])
            check_optical_pose(camera.transform,H)
            rgb_render=camera.render(rgb=True,depth=False,segmentation=False,normal=False)
            rgb=np.asarray(self.ctx.runner._extract_rgb(rgb_render)).reshape(480,640,3).copy()
            depth_render=camera.render(rgb=False,depth=True,segmentation=False,normal=False)
            depth=np.asarray(depth_render[1]).reshape(480,640).copy()
            if rgb.dtype!=np.uint8 or depth.dtype!=np.float32:
                raise ValueError('original native pixel encodings required')
            images.append((rgb,depth));transforms.append(H.tolist())
        E=np.asarray(BODY_FROM_OPTICAL);position=p+R@E[:3,3]
        camera.set_pose(pos=position,lookat=position+R[:,0],up=R[:,2])
        if before!=(int(self.ctx.runner._sim_time_ns),len(self.samples)):
            raise ValueError('paired camera acquisition cannot advance physics')
        return images,transforms

    def _static_identity(self):
        return dict(order=verify_order(self.ctx.build,self.raster_order),
            environment=appearance_environment_identity(self),sampling=sampling_readback(self.ctx.build.camera))

    def settle_recorded(self):
        super().settle_recorded()
        # Initialize the raster resources before starting the experiment clock.
        # These discarded warmup renders are never submitted as observations.
        self._render_pair()
        self.initial_camera_identity=self._static_identity()
        write(self.output/'camera_setup_identity.json',self.initial_camera_identity)

    def sensor_packets(self):
        now=int(self.ctx.runner._sim_time_ns);frame=len(self.captured_pairs)
        if now!=1_500_000_000+frame*100_000_000 or len(self.samples)!=750+50*frame:
            raise ValueError('actual consecutive paired-camera boundary required')
        started=time.perf_counter_ns();images,transforms=self._render_pair()
        policy=self.observations.packet(images[0][0],now)
        depth=from_native_depth(images[0][1],policy,measured_ns=now,available_ns=now,now_ns=now)
        auxiliary=auxiliary_depth(images[1][1],policy,measured_ns=now,available_ns=now,now_ns=now)
        rgb=from_captured_rgb(images[1][0],auxiliary,policy,measured_ns=now,available_ns=now,now_ns=now)
        fast=self.fast_buffer.packet(now_ns=now)
        row={'image_ns':np.int64(now),'decision_ns':np.int64(now)}
        for schema in SCHEMAS:
            for field in ('values','valid','measured_ns','available_ns'):
                row[f'{schema.name}_{field}']=policy['sensor_state'][schema.role][schema.name][field]
        self.packet_rows.append(row)
        self.model_manifest.append(dict(rgb_file=f'rgb_{frame:04d}.png',image_ns=now,decision_ns=now))
        self.fast_packets.append({k:np.asarray(fast[k]).copy() for k in ('values','valid','measured_ns','available_ns')})
        self.captured_pairs.append(dict(frame=frame,measured_ns=now,physical_sample_index=len(self.samples)-1,
            images=images,transforms=transforms,depth=depth,auxiliary_depth=auxiliary,
            acquisition_wall_ms=(time.perf_counter_ns()-started)/1e6))
        return policy,depth,fast,auxiliary,rgb,now

    def persist_observations(self,directory):
        # Preserve the original body and high-rate sample formats, but explicitly
        # identify the new image evidence format; no synthetic legacy witnesses.
        ObservationSession.persist_observations(self,directory)
        for name,rows in (('fast_gyro_samples.npz',self.fast_rows),('fast_gyro_histories.npz',self.fast_packets)):
            np.savez_compressed(directory/name,**({k:np.stack([r[k] for r in rows]) for k in rows[0]} if rows else {}))
        def persist_pair(row):
            return persist_camera_pair(row, directory, native_depth_only=self.native_depth_only,
                compression=self.archive_compression, compresslevel=self.archive_compression_level)
        # Each task owns one frame's distinct paths. map preserves acquisition
        # order in metadata; all native/render work remains on the main thread.
        with ThreadPoolExecutor(max_workers=self.archive_workers) as executor:
            metadata=list(executor.map(persist_pair,self.captured_pairs))
        terminal=self._static_identity()
        write(directory/'camera_terminal_identity.json',terminal)
        write(directory/'in_memory_camera_observations.json',dict(
            schema=RAW_DEPTH_SCHEMA if self.native_depth_only else 'in_memory_paired_camera_development.v1',frames=metadata,
            static_identity_unchanged=terminal==self.initial_camera_identity,
            diagnostic_segmentation_captured=False,per_frame_static_witnesses_captured=False,
            lossless_archive_compression_level=self.archive_compression_level,
            lossless_archive_compression_method=self.archive_compression,
            post_run_archive_workers=self.archive_workers,
            pixels_saved_after_timed_execution=True,ideal_simulation_timestamps=True))
        if terminal!=self.initial_camera_identity:raise ValueError('static camera/environment identity changed')


class RawDepthPairedCameraSession(InMemoryPairedCameraSession):
    """Identical live capture; retain native depth and packet digests on disk."""
    native_depth_only = True


class LzmaRawDepthPairedCameraSession(RawDepthPairedCameraSession):
    """Same NPZ arrays and live packets; stronger lossless post-run compression."""
    archive_compression = zipfile.ZIP_LZMA
    archive_compression_level = None
    archive_workers = 12
