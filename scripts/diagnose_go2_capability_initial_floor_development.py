"""Authorised first-frame regeneration of failed dev 01/0; no navigation retry."""
import json
import time
from pathlib import Path
import numpy as np
import cv2
import torch
from lewm import decision_headroom_json_v42_development as output
from lewm.multirate_routing_map_development import Geometry
from lewm.auxiliary_downward45_depth_geometry_development import reference_pose
from lewm.current_plane_floor_coverage_development import current_paired_plane
from scripts import run_go2_navigation_capability_correctness_c1_development as owner


def main():
    protocol=json.loads(owner.PROTOCOL.read_text());base=Path(protocol['output_root'])
    output.install(base)
    screen=base/'cohorts/v0_task_c1_C1_screen/result.json'
    assert screen.exists(), 'Do not compete with or alter the running screen'
    owner.verify_environment(owner.REPO/'docs/go2_navigation_capability_environment_pin_2026-09-26.json')
    root=base/'initial_floor_01_first_frame_diagnosis';root.mkdir(exist_ok=False)
    source=base/'runs/v0_task_c1_screen_C1_dev01_ep0_attempt001'
    spec=json.loads((source/'specification.json').read_text())
    expected=json.loads((source/'native/in_memory_camera_observations.json').read_text())['frames'][0]
    owner.save(root/'config.json',dict(source=str(source),source_config_sha256=owner.sha(source/'config.json'),
        script_sha256=owner.sha(__file__),physics='original 1.5-s settling only; zero mission commands',
        purpose='Confirm initial-floor failure on regenerated first consumed frame; not a retried mission',
        new_frame_retention=False,comparison='all retained consumed hash fields bitwise'))
    budget=owner.Budget(base,protocol);budget.admit_persist(32*1024**2)
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    session=None;started=time.monotonic()
    try:
        owner.source.previous.warmup();owner.source.previous.study.cohort.stable.floor.configure()
        owner.source.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        native=root/'native';native.mkdir()
        session=owner.make_session(spec,native,full_frames=True);session.install_contact_identity()
        owner.source.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        session.settle_recorded();owner.source.admit_context_setup(session,owner.sha(owner.__file__))
        policy,depth,fast,auxiliary,auxrgb,measured=session.sensor_packets()
        actual=session.captured_pairs[-1]['consumed_hash_record']
        for key in ('pixel_sha256','live_depth_noise','consumed_packet_sha256','arrays','measured_ns','physical_sample_index'):
            assert actual[key]==expected[key], key
        up=policy['sensor_state']['sensed']['specific_force']['values'].mean(0);up=up/np.linalg.norm(up)
        forward=np.array([1.,0.,0.])-up*up[0];forward/=np.linalg.norm(forward)
        B=np.stack((forward,np.cross(up,forward),up));auxR,_=reference_pose(B,np.zeros(3))
        geometry=Geometry()
        try:
            counts=[int(np.count_nonzero(geometry.index(p['depth_m'],p['valid'],R[2])['ground_cells']))
                for p,R in ((depth,B),(auxiliary,auxR))]
        finally:geometry.close()
        plane=current_paired_plane(depth,auxiliary,up)
        owner.save(root/'result.json',dict(status='PASS',all_consumed_frame_hash_fields_match=True,
            primary_measured_floor_quads=counts[0],auxiliary_measured_floor_quads=counts[1],required_quads=100,
            current_paired_plane_available=plane['available'],source_first_planning_decision_exists=False,
            cause='Primary-only map initialization cannot measure 100 floor quads; paired sensors evaluated separately.',
            primary_depth_valid_pixels=int(depth['valid'].sum()),auxiliary_depth_valid_pixels=int(auxiliary['valid'].sum()),
            frame_hash_record=actual,wall_s=time.monotonic()-started,mission_commands_executed=0))
        print(output.dumps(dict(floor_quads=counts,paired_plane=plane['available'])))
    except BaseException as exc:
        owner.save(root/'failure.json',dict(error=repr(exc),stop_required=True));raise
    finally:
        if session is not None:session.ctx.build.scene.destroy()
        owner.source.shutdown_genesis()

if __name__=='__main__':main()
