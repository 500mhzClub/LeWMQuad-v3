"""Reconstruct every frozen sensor prediction, independently of native scoring."""
import json

from lewm.rgbd_correspondence_motion_development import RGBDCorrespondenceMotion
from lewm.rgbd_dataset_development import load_rgbd_observation
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.fast_gyro_scan_session_development import load_fast_packet
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import OUTPUT,A,B,verify
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,write_json
from scripts.startup_raw_sensor_audit_development import read_json

IDENTITIES={'launch.json':'8a725bfe6fbdbd7220962d5195282660c479ed16350ea09c32c4be74064111a4',
            'result.json':'b5aae296ed45dd565b27318d1d42dca06340684de25557dfe3dc1dd7776ad32b'}


def audit():
    import cv2
    cv2.setNumThreads(1)
    bindings={str((OUTPUT/n).relative_to(ROOT)):h for n,h in IDENTITIES.items()};verify_bindings(bindings)
    launch=read_json(OUTPUT,'launch.json');result=read_json(OUTPUT,'result.json');verify(launch)
    bindings|={str((OUTPUT/n).relative_to(ROOT)):h for n,h in result['artifact_sha256'].items()}
    own='scripts/audit_go2_rgbd_correspondence_motion_diagnostic_v1.py';bindings[own]=digest(ROOT/own)
    verify_bindings(bindings);stats={}
    for label,path in (('A',A),('B',B)):
        saved=read_json(OUTPUT,label+'_sensor_predictions.json');decisions=read_json(path,'motion_decisions.json')
        assert len(saved)==len(decisions)
        model=RGBDCorrespondenceMotion()
        for frame,row in enumerate(saved):
            p,d=load_rgbd_observation(path,frame);f=load_fast_packet(path,frame)
            actual=model.observe(p,d,f,now_ns=p['sensor_state']['decision_ns'])
            assert actual=={k:v for k,v in row.items() if k not in ('observation_index','observer_wall_ms')}
            assert row['observation_index']==decisions[frame]['observation_index']==frame
        motion=[r['motion'] for r in saved[1:]]
        stats[label]=dict(reconstructed_frames=len(saved),transitions=len(motion),
            maximum_current_keypoints=max(r['current_keypoints'] for r in motion),
            maximum_mutual_matches=max(r['mutual_ratio_matches'] for r in motion),
            maximum_lifted_matches=max(r['lifted_matches'] for r in motion),
            accepted=sum(r['translation_previous_body_m'] is not None for r in motion))
        if label=='B':
            weak=[r['motion'] for r in saved if 4_100_000_000<=r['measured_ns']<=5_300_000_000]
            stats[label]['weak_interval_pairs']=len(weak)
            stats[label]['weak_interval_maximum_current_keypoints']=max(r['current_keypoints'] for r in weak)
    verify(launch);verify_bindings(bindings)
    return dict(status='ALL_FROZEN_RGBD_SENSOR_PREDICTIONS_RECONSTRUCTED',statistics=stats,
        identities=IDENTITIES,auditor_source_sha256={own:bindings[own]},native_pose_used_for_reconstruction=False,
        threshold_changed=False,original_results_changed=False,navigation_qualified=False)


if __name__=='__main__':
    target=OUTPUT/'raw_artifact_audit.json'
    if target.exists():raise ValueError('fresh reconstruction audit output only')
    result=audit();write_json(target,result);print(json.dumps(result),flush=True)
