"""Exact JEPA replay and read-only complete-foot subdivision witnesses."""
import json
import time
import cv2
import numpy as np
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.auxiliary_depth_reobserve_goal_probe_development import AuxiliaryDepthReobserveGoalProbe
from lewm.paired_retained_floor_coverage_development import PairedRetainedFloorPatches
from lewm.retained_floor_visibility_diagnosis_development import explain
from scripts.auxiliary_depth_packet_replay_development import packet as auxiliary_packet,public_acquisition
from lewm.partitioned_floor_patch_coverage_development import coverage
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.run_go2_auxiliary_depth_reobserve_goal_probe_v1 import OUTPUT as INPUT,CORRECTION,FITS
from scripts.read_go2_auxiliary_depth_reobserve_goal_probe_v1 import OUTPUT as READOUT
from scripts.navigation_artifact_root_development import BASE,create_output,validate_root,verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

INPUT_SHA='3675f09b90770ed9345154b8fe926c66fd6029e9c7bc7a6f2d53cba0c6861193'
READOUT_SHA='2fe50e68c57ebec738daebfc70f589752fda1dc7c43de2dfdc6f93a10efd9e8e'
OUTPUT=BASE/'go2_paired_retained_floor_coverage_v1_attempt_001'
PROTOCOL='docs/go2_paired_retained_floor_coverage_v1_2026-09-08.md'
CASE='full_jepa_family_episode_039'
MODEL='seed_2026091001_full_jepa'


def diagnose(memory,geometry,selection):
    candidates=[];joint=PairedRetainedFloorPatches(memory.patches,memory.auxiliary_patches)
    for candidate,prediction,check in zip(selection['candidates'],selection['prediction'],selection['surface_checks'],strict=True):
        dx,dy,sy,cy,_=prediction[0];yaw=float(np.arctan2(sy,cy));c,s=np.cos(yaw),np.sin(yaw)
        R=memory.rotation@np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        p=memory.position+memory.rotation@np.array([dx,dy,0.])
        shapes={v['shape_id']:v for v in geometry.supports(memory.joints,R)['shapes']}
        pending=[f for f in check['auxiliary_foot_floor_contacts'] if not f['measured_floor_contact_rule_eligible']
            and f['auxiliary_floor']['intersecting_voxels'] and not f['auxiliary_other_or_unknown']['intersecting_voxels']]
        feet=[]
        for foot in pending:
            centre=(memory.map_from_initial@(p+R@np.asarray(shapes[foot['shape_id']]['center_body_m'])))[:2]
            assert memory.patches.coverage([centre.tolist()])[0]==foot['primary_patch']
            assert memory.auxiliary_patches.coverage([centre.tolist()])[0]==foot['auxiliary_patch']
            partitions=[dict(divisions=n,primary=coverage(memory.patches,centre,divisions=n),
                auxiliary=coverage(memory.auxiliary_patches,centre,divisions=n),
                joint=coverage(joint,centre,divisions=n)) for n in (1,2,4,8)]
            feet.append(dict(shape_id=foot['shape_id'],centre_map_xy_m=centre.tolist(),
                original_primary_witness=foot['primary_patch'],original_auxiliary_witness=foot['auxiliary_patch'],
                partitions=partitions,joint_visibility_explanation=explain(joint,centre)
                    if candidate['action']=='hold' and not partitions[-1]['joint']['entire_original_square_covered'] else None))
        candidates.append(dict(action=candidate['action'],original_possible_intersection=check['possible_intersection'],feet=feet))
    return dict(candidates=candidates,retained_primary_frames=len(memory.patches.frames),
        retained_auxiliary_frames=len(memory.auxiliary_patches.frames),
        original_foot_radius_m=.022,controller_unchanged=True,unexecuted_outcomes_inferred=False)


def main():
    if not __debug__:raise ValueError('audit assertions required')
    cv2.setNumThreads(1);torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive diagnostic attempt')
    verify_artifacts(INPUT,{'result.json':INPUT_SHA});result=read_json(INPUT,'result.json')
    assert result['status']=='AUXILIARY_DEPTH_REOBSERVE_GOAL_PROBE_COMPLETE'
    bindings={'result.json':INPUT_SHA,**result['artifact_sha256']};verify_artifacts(INPUT,bindings)
    verify_artifacts(READOUT,{'result.json':READOUT_SHA});readout=read_json(READOUT,'result.json')
    assert readout['status']=='AUXILIARY_DEPTH_REOBSERVE_GOAL_READOUT_COMPLETE' and readout['probe_result_sha256']==INPUT_SHA
    readout_ids={'result.json':READOUT_SHA,'launch.json':readout['launch_sha256']};verify_artifacts(READOUT,readout_ids)
    original=read_json(INPUT,'launch.json');verify(original)
    sources=discover_sources((PROTOCOL,'scripts/diagnose_go2_paired_retained_floor_coverage_v1.py',
        'lewm/tests/test_partitioned_floor_patch_coverage_development.py',
        'lewm/tests/test_paired_retained_floor_coverage_development.py',
        'docs/go2_auxiliary_depth_reobserve_goal_probe_result_2026-09-08.md'),readout['source_sha256'])
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+256*1024**2:
        raise ValueError('diagnostic resource allowance unavailable')
    admission=original['correction_admission'];model,condition,variant=load_assigned(admission,MODEL)
    assert (condition,variant)==('jepa','full') and digest(URDF)==original['robot_urdf_sha256']
    launch=original|dict(source_sha256=sources,protocol=PROTOCOL,output_root=str(OUTPUT),hardware=resources,
        diagnostic_input_artifact_sha256=bindings,diagnostic_readout_artifact_sha256=readout_ids,
        diagnostic_case=CASE,diagnostic_model=MODEL,native_execution=False,model_training=False,
        controller_changed=False,division_levels=[1,2,4,8],diagnostic_ticks=[62,99],maximum_diagnostic_frames=256)
    verify(launch);create_output(OUTPUT);write_json(OUTPUT/'launch.json',launch)
    print('PAIRED_RETAINED_FLOOR_COVERAGE_LAUNCHED',digest(OUTPUT/'launch.json'),flush=True)
    started=time.perf_counter()
    try:
        reader=IntentReturnRGBDReplay(INPUT/CASE);rows=read_json(INPUT/CASE,'context_decisions.json')
        acquisitions=read_json(INPUT/CASE,'auxiliary_camera_audit.json')
        assert len(rows)==len(reader.frames) and 0<len(rows)<=256
        geometry=ArticulatedCollisionGeometry(URDF)
        controller=AuxiliaryDepthReobserveGoalProbe(model,geometry,condition=condition,variant=variant,persistent=True)
        before=state_digest(model.state_dict());diagnostics=[]
        for tick,row in enumerate(rows):
            assert row['tick']==row['observation_index']==tick
            policy,depth,fast,now=reader.packet(tick)
            aux=auxiliary_packet(INPUT/CASE,tick,policy,public_acquisition(acquisitions[tick]),now_ns=now)
            replayed=controller.observe(policy,depth,fast,now_ns=now,auxiliary_depth=aux)
            assert json.loads(json.dumps(replayed))==row['decision'],('exact recorded decision',tick)
            selection=replayed['new_selection']
            if tick in (62,99):
                assert selection and 'prediction' in selection and selection['action'] is None
                diagnostics.append(dict(tick=tick,**diagnose(controller.memory,geometry,selection)))
            if tick%25==0:print('EXACT_REPLAY',tick,flush=True)
        assert len(diagnostics)==2 and state_digest(model.state_dict())==before
        assert all(p.grad is None for p in model.parameters())
        write_json(OUTPUT/'coverage.json',dict(terminal_diagnostics=diagnostics,all_replayed_decisions_exact=True))
        verify(launch);verify_artifacts(INPUT,bindings);verify_artifacts(READOUT,readout_ids)
        verify_artifacts(FITS,admission['base_admission']['fit_artifact_sha256'])
        verify_artifacts(CORRECTION,admission['correction_artifact_sha256'])
        assert digest(URDF)==original['robot_urdf_sha256']
        write_json(OUTPUT/'result.json',dict(status='PAIRED_RETAINED_FLOOR_COVERAGE_COMPLETE',
            source_sha256=sources,input_result_sha256=INPUT_SHA,readout_result_sha256=READOUT_SHA,
            case=CASE,model=MODEL,exact_replayed_decisions=len(rows),model_state_unchanged=True,
            artifact_sha256={n:digest(OUTPUT/n) for n in ('launch.json','coverage.json')},
            hardware_after=hardware(),wall_s=time.perf_counter()-started,
            native_execution=False,model_training=False,controller_changed=False,
            diagnostic_only=True,navigation_qualified=False,goal_achieved=False))
        print('PAIRED_RETAINED_FLOOR_COVERAGE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_FLOOR_COVERAGE_DIAGNOSTIC_FAILURE',reason=repr(error)));raise


if __name__=='__main__':main()

