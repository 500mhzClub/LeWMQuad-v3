"""Replay each pre-fix pilot command tape, without source snapshots or new decisions.

Compare all retained sensor pixel/packet hashes, native traces and PNG bytes.
Missing source evidence never establishes replay equivalence.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch

from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_development as owner


def main(arm):
    protocol=json.loads(owner.PROTOCOL.read_text());base=Path(protocol['output_root'])
    output.install(base)
    root=base/'sensor_regeneration_2026-09-26'/arm
    root.mkdir(parents=True,exist_ok=False)
    source=base/f'runs/v0_pilot_{arm}_dev00_ep0_attempt{2 if arm=="C0" else 1:03d}'
    budget=owner.Budget(base,protocol);budget.admit_persist(32*1024**2)
    spec=json.loads((source/'specification.json').read_text())
    requests=json.loads((source/'requests.json').read_text())
    path=source/'native/in_memory_camera_observations.json'
    frames=json.loads(path.read_text())['frames'] if path.exists() else None
    with np.load(source/'native/physics_trace.npz',allow_pickle=False) as a:
        trace={k:a[k].copy() for k in a.files}
    owner.save(root/'config.json',dict(controller=arm,source=str(source),
        source_config_sha256=owner.sha(source/'config.json'),script_sha256=owner.sha(__file__),
        commands_only=True,source_snapshot_used=False,controller_recomputed=False,
        source_frames_available=frames is not None,implementation_check=True,
        frame_arrays_retained=False,cap_hours=120,original_pilot_unchanged=True))
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    started=time.monotonic();session=None;hashes=[];matched=0;native_exact=True
    # C4 is last in the fixed replay queue. Measure the existing lossless
    # native-depth archive format in memory for the unqualified C0 fallback.
    # These bytes are discarded; verification criteria and physics are unchanged.
    compressed_depth_sizes=[]
    try:
        owner.source.previous.warmup();owner.source.previous.study.cohort.stable.floor.configure()
        owner.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        native=root/'native';native.mkdir()
        session=owner.make_session(spec,native)
        session.install_contact_identity()
        owner.source.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        session.settle_recorded()
        owner.source.admit_context_setup(session,owner.sha(owner.__file__))
        for key,values in trace.items():
            np.testing.assert_array_equal(np.stack([r[key] for r in session.samples]),values[:750])
        for tick,request in enumerate(requests):
            budget.check()
            assert int(session.ctx.runner._sim_time_ns)==request['simulator_ns']
            if tick%5==0:
                packets=session.sensor_packets();camera=session.captured_pairs[-1]
                row=dict(frame=tick//5,measured_ns=camera['measured_ns'],pixel_sha256={},
                         live_depth_noise=camera['live_depth_noise'])
                for label,(rgb,depth),key in zip(('primary','auxiliary'),camera['images'],
                                               ('depth','auxiliary_depth'),strict=True):
                    current=dict(rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                        native_depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest(),
                        derived_packet_sha256=camera[key])
                    if arm=='C4':
                        stream=io.BytesIO();payload=io.BytesIO()
                        np.lib.format.write_array(payload,np.asarray(depth),allow_pickle=False)
                        with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_LZMA) as archive:
                            archive.writestr('native_optical_depth_m.npy',payload.getvalue())
                        compressed_depth_sizes.append(len(stream.getvalue()))
                    row['pixel_sha256'][label]=current
                    if frames is not None:
                        expected=frames[tick//5]
                        assert current==expected['pixel_sha256'][label], (tick//5,label,'pixels')
                        filename=f'rgb_{tick//5:04d}.png' if label=='primary' else f'auxiliary_rgb_{tick//5:04d}.png'
                        retained=np.asarray(Image.open(source/'native'/filename).convert('RGB'))
                        assert hashlib.sha256(retained.tobytes()).hexdigest()==current['rgb_sha256']
                        matched+=1
                if frames is not None:
                    assert row['live_depth_noise']==frames[tick//5]['live_depth_noise'], (tick//5,'consumed depth')
                hashes.append(row);camera['images']=[]
            session.phase=2;applied=session.command_policy_step(request['requested_command'])
            np.testing.assert_array_equal(applied,request['applied_command'])
            end=request['post_sample_index']
            for key,values in trace.items():
                np.testing.assert_array_equal(np.stack([r[key] for r in session.samples[-10:]]),values[end-9:end+1])
        owner.save(root/'regenerated_sensor_hashes.json',hashes)
        owner.save(root/'result.json',dict(controller=arm,status='PASS' if frames is not None else 'UNQUALIFIED_MISSING_SOURCE_FRAMES',
            frame_pairs=len(hashes),bitwise_rgb_matches=matched,bitwise_depth_packet_matches=matched,
            bitwise_native_depth_matches=matched,native_trace_exact=native_exact,
            source_snapshot_used=False,retention='hashes_only' if frames is not None else 'full_frames',
            reason=None if frames is not None else 'C0 pilot did not retain source images or sensor hashes; exact physics cannot establish image fidelity.',
            wall_s=time.monotonic()-started,simulated_s=len(requests)*.02,
            source_frames_unchanged=True,verification_from_commands_and_seed=True))
        if compressed_depth_sizes:
            owner.save(root/'native_depth_storage_measurement.json',dict(
                controller=arm,arrays=len(compressed_depth_sizes),format='ZIP_LZMA native_optical_depth_m.npy',
                total_bytes=sum(compressed_depth_sizes),maximum_bytes_per_camera_frame=max(compressed_depth_sizes),
                mean_bytes_per_camera_frame=float(np.mean(compressed_depth_sizes)),
                inference='Sizing evidence for full-depth fallback; not a guarantee about unseen views',
                arrays_retained=False,physics_or_acceptance_changed=False))
    except BaseException as exc:
        owner.save(root/'failure.json',dict(controller=arm,error=repr(exc),traceback=traceback.format_exc(),
            retention='full_frames',hash_only_qualified=False,wall_s=time.monotonic()-started,
            verified_frame_pairs=len(hashes),automatic_retry=False))
        raise
    finally:
        if session is not None:session.ctx.build.scene.destroy()
        owner.source.shutdown_genesis()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--controller',choices=['C0','C1','C2','C3','C4'],required=True)
    main(parser.parse_args().controller)
