"""Reproduce recorded pre-gyro match/consensus failures without admitting a pose."""
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.joint_rgbd_rigid_pose_development import register
from lewm.rgbd_match_stage_diagnostic_development import diagnose_matches
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, verify_artifacts, validate_root, create_output
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify, digest, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

SOURCE = 'scripts/diagnose_go2_no_rgb_jepa_maze02_matches_v1.py'
TEST = 'lewm/tests/test_no_rgb_jepa_match_failure_trace_development.py'
PROOF = 'docs/go2_all_phase_adapter_no_rgb_jepa_maze02_verification_2026-09-10.json'
PROOF_SHA = 'f2e2cdb317d913eecaee7b49712e3133ed78306ab10a64e99b14f1721d148a18'
CONTEXT = 'docs/go2_all_phase_adapter_no_rgb_jepa_maze02_visual_failure_2026-09-10.json'
CONTEXT_SHA = '75b3410956b6cf036adfc9052837881254b5a5aba750c51ddd9fddb7a8ddb316'
INPUT = BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'
OUTPUT = BASE/'go2_no_rgb_jepa_maze02_match_diagnosis_v1_attempt_001'
CASE = 'all_phase_no_rgb_jepa_residual_maze_02'
FRAME = 859
ANCHORS = (850, 849, 848, 847, 846, 845, 844, 840)


def early_failure_trace(points, expected):
    """Trace only original register locals; identity gyro is unused before these failures."""
    allowed = ('insufficient rigid-pose matches', 'insufficient rigid consensus after pruning')
    if expected not in allowed:
        raise ValueError('only recorded pre-gyro failure stages are supported')
    if sys.gettrace() is not None:
        raise ValueError('untraced single-thread diagnostic required')
    captured = {}; trace_error = []
    def trace(frame,event,arg):
        if frame.f_code is not register.__code__: return None
        frame.f_trace_lines = False
        if event == 'return':
            try:
                values = frame.f_locals
                captured.update(lifted_matches=len(values['a']),
                    valid_proposals=values.get('valid_candidates',0),
                    initial_consensus_points=values.get('initial_count'),
                    pruning_rounds=values.get('rounds',0),
                    final_consensus_points=int(values['mask'].sum()) if 'mask' in values else None,
                    initial_consensus_indices=np.flatnonzero(values['best']).tolist()
                        if values.get('best') is not None else None,
                    final_consensus_indices=np.flatnonzero(values['mask']).tolist()
                        if 'mask' in values else None)
            except Exception as error: trace_error.append(repr(error))
        return trace
    actual = None
    try:
        sys.settrace(trace)
        try: register(*points,gyro_rotation=np.eye(3),mode='joint',frame=FRAME)
        except SensorContractError as error: actual = str(error)
    finally:
        sys.settrace(None)
    if trace_error or not captured or actual != expected:
        raise ValueError('original pre-gyro failure did not reproduce: '+str((actual,expected,trace_error)))
    return captured | dict(original_failure=actual, original_failure_reproduced=True,
        identity_gyro_argument_unused_before_failure=True, actual_gyro_gate_reexecuted=False,
        converged_pose_available=False, pose_admitted=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive recorded-match diagnostic')
    verify({PROOF:PROOF_SHA,CONTEXT:CONTEXT_SHA})
    proof = json.loads((ROOT/PROOF).read_text()); context = json.loads((ROOT/CONTEXT).read_text())
    assert proof['case'] == CASE and proof['first_terminal']['frame'] == FRAME
    assert context['completed_verification_sha256'] == PROOF_SHA
    sources = discover_sources((SOURCE,TEST,PROOF,CONTEXT),proof['source_sha256']); verify(sources)
    bindings = proof['artifact_sha256']; verify_artifacts(INPUT,bindings)
    raw = context['frames'][-1]['original_visual_evidence']
    assert context['frames'][-1]['frame'] == FRAME and raw['status'] == 'VISUAL_TERMINAL_FAILURE'
    expected = {}
    for camera,selection,continuity in (
        ('primary',raw['camera_selection']['primary_reference_selection'],raw['camera_selection']['primary_continuity']),
        ('auxiliary',raw['reference_selection'],raw['continuity_evidence'])):
        assert tuple(r['reference_frame'] for r in selection['attempts']) == ANCHORS
        assert all(r['status']=='REJECTED' for r in selection['attempts'])
        expected[camera] = {r['reference_frame']:r['reason'] for r in selection['attempts']}
        expected[camera][FRAME-1] = continuity['incremental_failure']
    resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+64*1024**2:
        raise ValueError('8GiB available RAM and 40GiB+64MiB disk required')
    cv2.setNumThreads(1); create_output(OUTPUT)
    write_json(OUTPUT/'launch.json',dict(source_sha256=sources,input_artifact_sha256=bindings,
        completed_verification_sha256=PROOF_SHA,context_sha256=CONTEXT_SHA,hardware=resources,
        frame=FRAME,anchors=list(ANCHORS),incremental_reference=FRAME-1,native_execution=False,
        checkpoint_loaded=False,thresholds_changed=False,opencv_threads=1,
        scope='Bounded original feature/match and pre-gyro registration diagnosis; no full observer-history replay.'))
    start = time.perf_counter()
    try:
        directory = INPUT/CASE; reader = IntentReturnRGBDReplay(directory)
        acquisitions = read_json(directory,'auxiliary_camera_audit.json'); features = {}
        for frame in sorted(set(ANCHORS) | {FRAME-1,FRAME}):
            policy,depth,_,now = reader.packet(frame)
            image,auxiliary = packet(directory,frame,policy,public_acquisition(acquisitions[frame]),now_ns=now)
            features[frame] = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'],depth),
                auxiliary=CornerSupportFeatureFrame(image['rgb'],auxiliary))
        assert features[FRAME-1]['primary'].witness() == raw['last_accepted_feature_witness']
        assert features[FRAME-1]['auxiliary'].witness() == raw['auxiliary_feature_witness']
        rows = []
        for camera in ('primary','auxiliary'):
            for ref in (*ANCHORS,FRAME-1):
                a,b = features[ref][camera],features[FRAME][camera]
                stages = diagnose_matches(a,b)
                points = matched_points(a,b)
                before = [x.tobytes() for x in points]
                failure = early_failure_trace(points,expected[camera][ref])
                assert before == [x.tobytes() for x in points]
                assert failure['lifted_matches'] == stages['counts']['valid_depth_pair']
                rows.append(dict(camera=camera,reference_frame=ref,current_frame=FRAME,
                    reference_role='increment' if ref==FRAME-1 else 'retained_anchor',
                    **stages,registration=failure))
        verify(sources); verify_artifacts(INPUT,bindings)
        report = dict(rows=rows,feature_witnesses={str(f):{c:x.witness() for c,x in v.items()}
            for f,v in features.items()},all_eighteen_original_pre_gyro_failures_reproduced=True,
            original_correspondence_arrays_byte_exact=True,previous_feature_witnesses_reproduced=True,
            native_execution=False,model_inference=False,thresholds_changed=False,pose_admitted=False,
            full_observer_history_replayed=False,original_outcome_unchanged=True,
            actual_gyro_gate_reexecuted=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT/'diagnosis.json',report)
        ids = {n:digest(OUTPUT/n) for n in ('launch.json','diagnosis.json')}; verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='NO_RGB_JEPA_MAZE02_MATCH_DIAGNOSIS_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=ids,wall_s=time.perf_counter()-start,
            native_execution=False,goal_achieved=False))
        print('NO_RGB_JEPA_MATCH_DIAGNOSIS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
        for row in rows: print(row['camera'],row['reference_frame'],row['counts'],row['registration'],flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_NO_RGB_JEPA_MATCH_DIAGNOSIS_FAILURE',reason=repr(error)))
        raise


if __name__ == '__main__': main()
