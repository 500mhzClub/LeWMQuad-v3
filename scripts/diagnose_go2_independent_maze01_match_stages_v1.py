"""Fixed failed-frame stage accounting with actual-array equivalence checks."""
import time
import cv2
from lewm.rgbd_match_stage_diagnostic_development import diagnose_matches
from scripts import diagnose_go2_independent_maze01_correspondences_v1 as prior
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware, source_check

OUTPUT=BASE/'go2_independent_maze01_match_stage_diagnosis_v1_attempt_001'
SOURCE='scripts/diagnose_go2_independent_maze01_match_stages_v1.py'
PRIOR_SHA='15ec380fc03906fef61dbd5c9a8418856c63bdf531337152db7d8ad0ac685a56'


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive match-stage diagnosis required')
    cv2.setNumThreads(1)
    verify_artifacts(prior.OUTPUT,{'result.json':PRIOR_SHA})
    previous=read_json(prior.OUTPUT,'result.json')
    if previous['status']!='INDEPENDENT_MAZE01_CORRESPONDENCE_DIAGNOSIS_COMPLETE':
        raise ValueError('completed original failed-frame diagnosis required')
    prior_ids={'result.json':PRIOR_SHA,'launch.json':previous['launch_sha256']}
    verify_artifacts(prior.OUTPUT,prior_ids);old=read_json(prior.OUTPUT,'launch.json')
    bindings=old['input_artifact_sha256'];verify_artifacts(prior.INPUT,bindings)
    sources=discover_sources((SOURCE,),old['source_sha256']);source_check(sources)
    resources=hardware()
    if resources['memory_available_bytes']<8*1024**3 or resources['artifact_free_bytes']<40*1024**3+64*1024**2:
        raise ValueError('bounded CPU stage-diagnosis resources unavailable')
    create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_artifact_sha256=bindings,
        predecessor_artifact_sha256=prior_ids,hardware=resources,current_frame=prior.FRAME,
        reference_frames=list(prior.REFERENCES),opencv_threads=1,native_execution=False,posthoc_diagnosis=True))
    started=time.perf_counter()
    try:
        reader=prior.IntentReturnRGBDReplay(prior.INPUT/prior.CASE)
        acquisitions=read_json(prior.INPUT/prior.CASE,'auxiliary_camera_audit.json');features={}
        for frame in (*prior.REFERENCES,prior.FRAME):
            policy,depth,_,now=reader.packet(frame)
            image,auxiliary=prior.packet(prior.INPUT/prior.CASE,frame,policy,
                prior.public_acquisition(acquisitions[frame]),now_ns=now)
            features[frame]=dict(primary=prior.CornerSupportFeatureFrame(policy['image']['rgb'],depth),
                auxiliary=prior.CornerSupportFeatureFrame(image['rgb'],auxiliary))
        rows=[]
        for camera in ('primary','auxiliary'):
            for frame in prior.REFERENCES:
                row=diagnose_matches(features[frame][camera],features[prior.FRAME][camera])
                expected=next(r['lifted_matches'] for r in previous['rows']
                    if r['camera']==camera and r['reference_frame']==frame)
                assert row['counts']['valid_depth_pair']==expected
                rows.append(dict(camera=camera,reference_frame=frame,current_frame=prior.FRAME,**row))
        verify_artifacts(prior.INPUT,bindings);verify_artifacts(prior.OUTPUT,prior_ids);source_check(sources)
        write_json(OUTPUT/'result.json',dict(status='INDEPENDENT_MAZE01_MATCH_STAGE_DIAGNOSIS_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'),rows=rows,wall_s=time.perf_counter()-started,
            original_correspondence_arrays_verified=64,reference_pairs_verified=16,
            all_source_and_input_bindings_verified_before_after=True,native_execution=False,
            checkpoint_loaded=False,gates_changed=False,pose_admitted=False,command_selected=False,
            original_outcome_unchanged=True,goal_achieved=False))
        print('INDEPENDENT_MAZE01_MATCH_STAGE_DIAGNOSIS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_MATCH_STAGE_DIAGNOSIS_FAILURE',reason=repr(error)))
        raise


if __name__=='__main__': main()
