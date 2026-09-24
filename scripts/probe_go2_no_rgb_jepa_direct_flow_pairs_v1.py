"""Evaluate an existing association fallback on the fixed failed image pair."""
import cv2
import numpy as np
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.direct_corner_flow_association_development import tracked_points
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts import diagnose_go2_no_rgb_jepa_maze02_matches_v1 as diagnosis
from scripts.navigation_artifact_root_development import BASE,validate_root,create_output,verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest,verify,write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.novel_maze_auxiliary_rgb_packet_development import packet,public_acquisition
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

SOURCE = 'scripts/probe_go2_no_rgb_jepa_direct_flow_pairs_v1.py'
OUTPUT = BASE/'go2_no_rgb_jepa_direct_flow_pairs_v1_attempt_001'
DIAGNOSIS_SHA = '95b0315758c4065086bb11076a3e8f4767f1e660d15c51cb3ea2d72aa6a689a7'


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive direct-flow pair probe')
    verify_artifacts(diagnosis.OUTPUT,{'result.json':DIAGNOSIS_SHA})
    predecessor = read_json(diagnosis.OUTPUT,'result.json')
    assert predecessor['status']=='NO_RGB_JEPA_MAZE02_MATCH_DIAGNOSIS_V1_COMPLETE'
    verify_artifacts(diagnosis.OUTPUT,predecessor['artifact_sha256']); verify(predecessor['source_sha256'])
    admitted = read_json(diagnosis.OUTPUT,'launch.json'); old = read_json(diagnosis.OUTPUT,'diagnosis.json')
    inputs = admitted['input_artifact_sha256']; verify_artifacts(diagnosis.INPUT,inputs)
    sources = discover_sources((SOURCE,),predecessor['source_sha256']); verify(sources)
    resources = hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+64*1024**2:
        raise ValueError('bounded CPU probe needs 8GiB RAM and 40GiB+64MiB disk')
    cv2.setNumThreads(1); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_artifact_sha256=inputs,
        predecessor_result_sha256=DIAGNOSIS_SHA,hardware=resources,reference_frame=858,current_frame=859,
        existing_direct_flow_implementation_unchanged=True,native_execution=False,model_loaded=False))
    try:
        directory = diagnosis.INPUT/diagnosis.CASE; reader = IntentReturnRGBDReplay(directory)
        acquisitions = read_json(directory,'auxiliary_camera_audit.json'); features = {}
        for frame in (858,859):
            policy,depth,_,now = reader.packet(frame)
            assert now==1_500_000_000+frame*100_000_000
            image,auxiliary = packet(directory,frame,policy,public_acquisition(acquisitions[frame]),now_ns=now)
            features[frame] = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'],depth),
                                  auxiliary=CornerSupportFeatureFrame(image['rgb'],auxiliary))
            assert {c:f.witness() for c,f in features[frame].items()}==old['feature_witnesses'][str(frame)]
        rows=[]; arrays={}
        for camera in ('primary','auxiliary'):
            a,b = features[858][camera],features[859][camera]
            original = matched_points(a,b)
            previous = next(r for r in old['rows'] if r['reference_frame']==858 and r['camera']==camera)
            assert len(original[0])==previous['counts']['valid_depth_pair']
            values,receipt = tracked_points(a,b); repeated,check = tracked_points(a,b)
            assert receipt==check
            for name,x,y in zip(('reference_points','current_points','reference_pixels','current_pixels'),values,repeated,strict=True):
                assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
                arrays[camera+'_'+name]=x
            assert len(values[0])==receipt['counts']['valid_depth_pair']
            rows.append(dict(camera=camera,original_valid_depth_pairs=len(original[0]),direct_flow=receipt,
                repeated_association_arrays_byte_exact=True))
        np.savez_compressed(OUTPUT/'associations.npz',**arrays)
        verify(sources); verify_artifacts(diagnosis.INPUT,inputs)
        verify_artifacts(diagnosis.OUTPUT,{'result.json':DIAGNOSIS_SHA}|predecessor['artifact_sha256'])
        write_json(OUTPUT/'probe.json',dict(rows=rows,original_failures_preserved=True,
            existing_fallback_source_unchanged=True,full_observer_history_replayed=False,
            rigid_registration_evaluated=False,gyro_gate_evaluated=False,pose_admitted=False,
            command_selected=False,new_model_inference=False,native_execution=False,goal_achieved=False))
        ids={n:digest(OUTPUT/n) for n in ('launch.json','associations.npz','probe.json')}; verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='NO_RGB_JEPA_DIRECT_FLOW_PAIR_PROBE_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,native_execution=False,goal_achieved=False))
        print('DIRECT_FLOW_PAIR_PROBE_COMPLETE',digest(OUTPUT/'result.json'),rows,flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_DIRECT_FLOW_PAIR_PROBE_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__':main()
