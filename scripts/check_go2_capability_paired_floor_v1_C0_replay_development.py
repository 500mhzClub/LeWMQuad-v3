"""One approved full-RGB-D corrected C0 gate replay. Any mismatch stops the gate."""
import hashlib
import json
import time
import traceback
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_paired_floor_v1_development as owner


def main():
    protocol=json.loads(owner.PROTOCOL.read_text());base=Path(protocol['output_root'])
    output.install(base)
    owner.verify_environment(owner.REPO/'docs/go2_navigation_capability_environment_pin_2026-09-26.json')
    root=base/'paired_floor_v1_C0_sensor_replay';root.mkdir(exist_ok=False)
    source=base/'runs'/protocol['first_corrected_C0_assignment']
    config=json.loads((source/'config.json').read_text())
    assert config['controller']=='C0' and config['retention']=='full_frames'
    assert config['harness_sha256']==owner.sha(owner.FREEZE)
    assert json.loads((source/'oracle_prefix_check.json').read_text())['passed']
    budget=owner.Budget(base,protocol);budget.admit_persist(32*1024**2)
    spec=json.loads((source/'specification.json').read_text())
    requests=json.loads((source/'requests.json').read_text())
    frames=json.loads((source/'native/in_memory_camera_observations.json').read_text())['frames']
    with np.load(source/'native/physics_trace.npz',allow_pickle=False) as a:trace={k:a[k].copy() for k in a.files}
    owner.save(root/'config.json',dict(source=str(source),source_config_sha256=owner.sha(source/'config.json'),
        harness_sha256=owner.sha(owner.FREEZE),script_sha256=owner.sha(__file__),commands_only=True,
        source_snapshot_used=False,controller_recomputed=False,implementation_check=True,
        strict_failure='Stop for possible C0 validity problem; no retry or retention fallback'))
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    started=time.monotonic();session=None;verified=0
    try:
        owner.source.previous.warmup();owner.source.previous.study.cohort.stable.floor.configure()
        owner.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        native=root/'native';native.mkdir()
        session=owner.make_session(spec,native,full_frames=True)
        session.install_contact_identity()
        owner.source.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        session.settle_recorded();owner.source.admit_context_setup(session,owner.sha(owner.__file__))
        for key,values in trace.items():
            np.testing.assert_array_equal(np.stack([r[key] for r in session.samples]),values[:len(session.samples)])
        for tick,request in enumerate(requests):
            budget.check()
            assert int(session.ctx.runner._sim_time_ns)==request['simulator_ns']
            if tick%5==0:
                session.sensor_packets();camera=session.captured_pairs[-1]
                actual=camera['consumed_hash_record'];expected=frames[tick//5]
                for key in ('pixel_sha256','live_depth_noise','consumed_packet_sha256','arrays','measured_ns','physical_sample_index'):
                    assert actual[key]==expected[key],(tick//5,key)
                for label,(rgb,depth) in zip(('primary','auxiliary'),camera['images'],strict=True):
                    filename=f'rgb_{tick//5:04d}.png' if label=='primary' else f'auxiliary_rgb_{tick//5:04d}.png'
                    retained=np.asarray(Image.open(source/'native'/filename).convert('RGB'))
                    np.testing.assert_array_equal(retained,rgb)
                camera['images']=[];verified+=1
            session.phase=2;applied=session.command_policy_step(request['requested_command'])
            np.testing.assert_array_equal(applied,request['applied_command'])
            end=request['post_sample_index']
            for key,values in trace.items():
                np.testing.assert_array_equal(np.stack([r[key] for r in session.samples[-10:]]),values[end-9:end+1])
        assert verified==len(frames)
        owner.save(root/'result.json',dict(status='PASS',harness_sha256=owner.sha(owner.FREEZE),
            source_config_sha256=owner.sha(source/'config.json'),source=str(source),frame_pairs=verified,
            bitwise_rgb_matches=2*verified,bitwise_depth_packet_matches=2*verified,
            full_typed_packet_matches=4*verified,native_trace_exact=True,
            verification_from_commands_and_seed=True,wall_s=time.monotonic()-started))
    except BaseException as exc:
        owner.save(root/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            verified_frame_pairs=verified,possible_C0_validity_problem=True,stop_required=True,
            automatic_retry=False,full_frame_fallback=False,wall_s=time.monotonic()-started))
        raise
    finally:
        if session is not None:session.ctx.build.scene.destroy()
        owner.source.shutdown_genesis()


if __name__=='__main__':main()
