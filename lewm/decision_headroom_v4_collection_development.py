"""Mission-wide audit sampling, using unchanged live acquisition and controllers."""
import copy
import io
import json
import hashlib
from pathlib import Path
from PIL import Image
from lewm.decision_headroom_packet_development import DecisionPacketCaptureMixin
from lewm.decision_headroom_snapshot_development import capture
from lewm.decision_headroom_v4_development import PhaseReservoir, active_objective, encode_snapshot


class MissionSampling:
    def __init__(self,run_id,output,budget):
        self.reservoir=PhaseReservoir(2026092309,run_id);self.output=output;self.budget=budget
        self.packets={};self.images={};self.camera_metadata=[];self.failures=[]

    def camera(self,session):
        row=session.captured_pairs[-1];frame=row['frame'];encoded={};hashes={}
        for name,(rgb,depth) in zip(('primary','auxiliary'),row['images'],strict=True):
            stream=io.BytesIO();Image.fromarray(rgb).save(stream,format='PNG',compress_level=1)
            encoded[name]=stream.getvalue();hashes[name]=dict(rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        self.images[frame]=encoded
        self.camera_metadata.append({k:copy.deepcopy(row[k]) for k in ('frame','measured_ns','physical_sample_index','transforms','acquisition_wall_ms','live_depth_noise')}|dict(pixel_sha256=hashes,primary_depth_packet_sha256=row['depth'],auxiliary_depth_packet_sha256=row['auxiliary_depth']))
        # No file retirement: raw archival buffers are never written; live packets
        # and frame-count metadata remain unchanged. RGB retained losslessly in RAM.
        row['images']=None;row['depth']=None;row['auxiliary_depth']=None
        keep=set(range(max(0,frame-10),frame+1))
        for phase_rows in self.reservoir.rows.values():
            for candidate in phase_rows:
                f=candidate['frame'];keep.update(range(f-10,f+9))
        for f in list(self.images):
            if f not in keep:del self.images[f]

    def decision(self,controller,session,frame):
        packet=controller.audit_packets.pop(frame,None)
        if packet is None:return
        objective=active_objective(packet);admitted,replaced=self.reservoir.consider(frame,objective['phase'])
        for old in replaced:self.packets.pop(old,None)
        if not admitted:return
        try:
            payload,binding=encode_snapshot(dict(physical=capture(session),decision=packet))
            self.packets[frame]=(payload,binding,objective)
        except Exception as exc:
            # Keep the sampled identity as unresolved; never replace a failed member.
            self.packets[frame]=(None,None,objective);self.failures.append(dict(frame=frame,reason=repr(exc)))

    def finish(self,arm,layout):
        from scripts.run_go2_decision_headroom_branches_development import save
        members=self.reservoir.final();metadata=[];keep=set()
        for member in members:
            frame=member['frame'];payload,binding,objective=self.packets[frame]
            root=self.output/f'state_{frame:04d}';root.mkdir()
            state=member|dict(measured_ns=1_500_000_000+frame*100_000_000,source_controller=arm,layout=layout,active_objective=objective)
            if payload is None:state.update(status='unresolved',reason='SNAPSHOT_SERIALIZATION_LIMIT')
            else:
                self.budget.admit_write(len(payload));(root/'snapshot_packet.pkl.zlib').write_bytes(payload)
                state.update(status='captured',bundle=binding)
            save(root/'snapshot.json',state);metadata.append(state);keep.update(range(frame-10,frame+9))
        native=self.output/'native'
        for frame in sorted(keep):
            if frame not in self.images:continue
            for name,payload in self.images[frame].items():
                self.budget.admit_write(len(payload));prefix='rgb' if name=='primary' else 'auxiliary_rgb'
                (native/f'{prefix}_{frame:04d}.png').write_bytes(payload)
        save(native/'in_memory_camera_observations.json',dict(frames=[r for r in self.camera_metadata if r['frame'] in keep],retention='Final representative source windows only; lossless native capture; all camera timestamps recorded separately'))
        save(self.output/'sampling.json',dict(counts=self.reservoir.counts,decisions=self.reservoir.log,members=members,failures=self.failures,seed=2026092309))
        self.packets.clear();self.images.clear()
        return metadata
