"""Snapshot sensors on the physics owner; finish RGB-D in the renderer worker."""
from collections import deque
from copy import copy
import time
import numpy as np

from lewm.causal_depth_observation_development import from_native_depth
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth as auxiliary_depth
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from lewm.simulated_body_observation_development import SCHEMAS
from scripts import snapshot_camera_renderer_development as renderer
from scripts.raw_depth_archive_development import packet_digest
from scripts.replay_go2_depth_noise_tracking_development import SEED, perturbed_packet
from scripts.run_go2_contact_attributed_execution_development_v1 import array
from scripts.run_go2_stopping_projection_transfer_development import FreshCameraSession


def freeze_sensor_owner(owner, now):
    """Own the current history; run the unchanged packet builder off-thread.

    The native buffers append chronological samples and current acquisitions
    have no future availability. Older than the latest requested history cannot
    enter this decision. Copies prevent later appends or edits changing it.
    """
    frozen=copy(owner); frozen.buffer=copy(owner.buffer); frozen.buffer._samples={}
    for name,schema in owner.buffer.schemas.items():
        samples=tuple(owner.buffer._samples[name])[-schema.history_length:]
        if any(mt>now or at>now for mt,at,_,_ in samples):
            raise ValueError('current native sensor histories cannot contain future samples')
        frozen.buffer._samples[name]=tuple((mt,at,data.copy(),valid.copy()) for mt,at,data,valid in samples)
    return frozen


def render_packet(snapshot):
    result=renderer.render(snapshot['qpos'],snapshot['measured_ns'])
    images=result['images']; now=snapshot['measured_ns']; frame=snapshot['frame']
    policy=snapshot['body_buffer'].packet(images[0][0],now)
    fast=snapshot['fast_buffer'].packet(now_ns=now)
    depth=from_native_depth(images[0][1],policy,measured_ns=now,available_ns=now,now_ns=now)
    auxiliary=auxiliary_depth(images[1][1],policy,measured_ns=now,available_ns=now,now_ns=now)
    rgb=from_captured_rgb(images[1][0],auxiliary,policy,measured_ns=now,available_ns=now,now_ns=now)
    row=dict(frame=frame,measured_ns=now,physical_sample_index=snapshot['physical_sample_index'],
        images=images,transforms=result['transforms'],depth=packet_digest(depth),
        auxiliary_depth=packet_digest(auxiliary))
    policy,depth,fast,rgb,auxiliary,now=perturbed_packet((policy,depth,fast,rgb,auxiliary,now),
        layout=snapshot['layout_index'],frame=frame,sigma_m=2/1000)
    row['live_depth_noise']=dict(seed=SEED,layout_index=snapshot['layout_index'],sigma_mm=2,
        primary_sha256=packet_digest(depth),auxiliary_sha256=packet_digest(auxiliary))
    ready=time.perf_counter_ns()
    row['acquisition_wall_ms']=(ready-snapshot['started_wall_ns'])/1e6
    return dict(row=row,packets=(policy,depth,fast,auxiliary,rgb,now),
        renderer_completed_wall_ns=ready,rendering_wall_ns=result['rendering_wall_ns'])


class AsynchronousCameraSession(FreshCameraSession):
    def __init__(self,*args,renderer_executor,**kwargs):
        super().__init__(*args,**kwargs)
        self.renderer_executor=renderer_executor
        self.camera_pending=deque(); self.camera_started=0; self.async_receipts=[]

    def begin_sensor_packets(self):
        if len(self.camera_pending)>=2:
            raise RuntimeError('renderer exceeded two in-flight camera acquisitions')
        started=time.perf_counter_ns(); now=int(self.ctx.runner._sim_time_ns); frame=self.camera_started
        if now!=1_500_000_000+frame*100_000_000 or len(self.samples)!=750+50*frame:
            raise ValueError('actual consecutive paired-camera boundary required')
        qpos=array(self.ctx.build.robot.get_qpos()).copy()
        body=freeze_sensor_owner(self.observations,now)
        fast=freeze_sensor_owner(self.fast_buffer,now)
        snapshot=dict(qpos=qpos,body_buffer=body,fast_buffer=fast,measured_ns=now,frame=frame,
            physical_sample_index=len(self.samples)-1,started_wall_ns=started,
            layout_index=self.noise_layout_index)
        future=self.renderer_executor.submit(render_packet,snapshot)
        receipt=dict(frame=frame,measured_ns=now,started_wall_ns=started,
            snapshot_submitted_wall_ns=time.perf_counter_ns(),
            physical_sample_index=len(self.samples)-1)
        self.camera_pending.append((future,receipt)); self.camera_started+=1
        return receipt

    def poll_sensor_packets(self):
        ready=[]
        while self.camera_pending and self.camera_pending[0][0].done():
            future,receipt=self.camera_pending.popleft(); result=future.result()
            p,d,fast,a,rgb,now=result['packets']
            if result['row']['frame']!=len(self.captured_pairs):
                raise ValueError('ordered completed camera acquisitions required')
            row={'image_ns':np.int64(now),'decision_ns':np.int64(now)}
            for schema in SCHEMAS:
                for field in ('values','valid','measured_ns','available_ns'):
                    row[f'{schema.name}_{field}']=p['sensor_state'][schema.role][schema.name][field]
            self.packet_rows.append(row)
            self.model_manifest.append(dict(rgb_file=f'rgb_{receipt["frame"]:04d}.png',image_ns=now,decision_ns=now))
            self.fast_packets.append({k:np.asarray(fast[k]).copy() for k in ('values','valid','measured_ns','available_ns')})
            self.captured_pairs.append(result['row'])
            receipt=receipt|dict(renderer_completed_wall_ns=result['renderer_completed_wall_ns'],
                rendering_wall_ns=result['rendering_wall_ns'],received_wall_ns=time.perf_counter_ns())
            self.async_receipts.append(receipt); ready.append((result['packets'],receipt))
        return ready
