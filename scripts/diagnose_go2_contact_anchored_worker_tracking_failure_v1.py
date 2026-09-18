"""Bounded raw-match diagnosis of the finished contact worker, pending parent audit."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import FunctionType

import cv2
import numpy as np
import psutil

from scripts import run_go2_commitment_contact_anchored_maze02_pilot_v1 as native
from scripts import diagnose_go2_no_rgb_jepa_maze02_matches_v1 as original_trace
from scripts.maze_decision_stream_development import read_rows, NAME
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.await_go2_all_phase_translation_bias_v1 import BOOT
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.rgbd_match_stage_diagnostic_development import diagnose_matches

SOURCE = 'scripts/diagnose_go2_contact_anchored_worker_tracking_failure_v1.py'
OUTPUT = ROOT/'docs/go2_contact_anchored_worker_tracking_failure_diagnosis_2026-09-11.json'
LAUNCH_SHA = '8d6a736c2f1f6de1f461fa6952341c1a7f3d321ccdd37881ac27a6da35191b8c'
WORKER_SHA = '5b2791c83053ea944db281bb4e6990e5eddee0a00ac89e4e03ccef8cd185bac3'
WORKER_PID = 2813548
FRAME = 561
ANCHORS = (560, 559, 558, 557, 556, 555, 553, 551)
FAILURE = 'same-episode current visual evidence required'

# Preserve the tested trace exactly; only its fixed registration seed frame
# changes from the earlier case to this actual observation frame.
failure_trace = FunctionType(original_trace.early_failure_trace.__code__,
    original_trace.early_failure_trace.__globals__ | dict(FRAME=FRAME), 'failure_trace')


def worker_ended():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT or psutil.pid_exists(WORKER_PID):
        raise ValueError('original worker must have ended on the original boot')


def main():
    worker_ended(); root = native.OUTPUT; name = native.CASE[0]; directory = root/name
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive worker diagnosis output required')
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('original run failure must be retained')
    bindings = {'launch.json':LAUNCH_SHA, name+'_worker_terminal.json':WORKER_SHA}
    verify_artifacts(root, bindings)
    launch = read_json(root, 'launch.json'); record = read_json(root, name+'_worker_terminal.json')
    sources = discover_sources((SOURCE,), launch['source_sha256']); verify(sources)
    for path, sha in record['artifact_sha256'].items():
        if path in bindings and bindings[path] != sha: raise ValueError('conflicting worker artifact identity')
        bindings[path] = sha
    bindings[name+'_worker.log'] = record['worker_log_sha256']
    verify_artifacts(root, bindings)
    required = [name+'/'+p for p in native.artifacts(2, record['collection'])]
    required += [name+s for s in ('_audit.json', '_prefix_comparison.json', '_readout.json')]
    if any(p not in bindings for p in required): raise ValueError('complete raw worker artifact roster required')
    collection = read_json(directory, 'result.json'); audit = read_json(root, name+'_audit.json')
    if (record['collection'] != collection or record['readout'] != read_json(root, name+'_readout.json')
            or record['prefix_comparison'] != read_json(root, name+'_prefix_comparison.json')):
        raise ValueError('actual stored worker receipts required')
    native.require_worker(record, audit, launch['input_admission']['prefix_report'])
    with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
        if native.case_readout(audit, collection, archive['physics_contact']) != record['readout']:
            raise ValueError('actual physics/contact/timing readout must reconstruct')
    tape = read_json(directory, 'command_tape.json'); selected = {}; stream = hashlib.sha256()
    actions = Counter(); count = 0; compact = []
    for row in read_rows(directory):
        frame = row['tick']; d = row['decision']
        if frame != count: raise ValueError('complete sequential observation population required')
        count += 1
        stream.update(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n')
        if frame < len(tape):
            t = tape[frame]
            if (t['tick'] != frame or t['requested_command'] != d['requested_command']):
                raise ValueError('every actual requested command must match saved decisions')
        if frame < FRAME:
            if d['terminal'] is not None or d['tick'] != frame: raise ValueError('same original preterminal history required')
        elif (d['tick'] != FRAME-1 or d['terminal'] != 'SENSOR_OR_MODEL_FAILURE'
                or d['failure'] != FAILURE or d['requested_command'] != [0., 0., 0.]):
            raise ValueError('original terminal rejection and zero-command drain required')
        if frame in set(ANCHORS) | {FRAME}: selected[frame] = row
        actions[str(d['selected_action'])] += 1
        if frame >= 550 and frame <= FRAME:
            v = d['original_visual_evidence']
            compact.append(dict(frame=frame, selected_action=d['selected_action'], requested_command=d['requested_command'],
                failure=d['failure'], observed_goal_distance_m=d['observed_goal_distance_m'],
                status=v['status'], terminal_failure=v['terminal_failure'], camera_selection=v['camera_selection'],
                reference_selection=v['reference_selection'], continuity_evidence=v['continuity_evidence']))
    if (count != 572 or count != collection['decisions'] or len(tape) != count-1
            or collection['terminal_zero_ticks'] != 10 or count-FRAME-1 != 10):
        raise ValueError('original complete 572-observation worker and terminal drain required')
    raw = selected[FRAME]['decision']['original_visual_evidence']; expected = {}
    for camera, selection, continuity in (
        ('primary', raw['camera_selection']['primary_reference_selection'], raw['camera_selection']['primary_continuity']),
        ('auxiliary', raw['reference_selection'], raw['continuity_evidence'])):
        if (tuple(r['reference_frame'] for r in selection['attempts']) != ANCHORS
                or any(r['status'] != 'REJECTED' for r in selection['attempts'])
                or continuity['previous_frame'] != FRAME-1):
            raise ValueError('exact original retained references and incremental failure required')
        expected[camera] = {r['reference_frame']:r['reason'] for r in selection['attempts']}
        if expected[camera][FRAME-1] != continuity['incremental_failure']:
            raise ValueError('same previous-frame anchor and incremental failure required')
    cv2.setNumThreads(1)
    if cv2.ocl.useOpenCL(): raise ValueError('fixed CPU OpenCV required')
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); features = {}
    for frame in sorted(selected):
        policy, depth, _, now = reader.packet(frame)
        image, auxiliary = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
        features[frame] = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'], depth),
            auxiliary=CornerSupportFeatureFrame(image['rgb'], auxiliary))
    if (features[FRAME-1]['primary'].witness() != raw['last_accepted_feature_witness']
            or features[FRAME-1]['auxiliary'].witness() != raw['auxiliary_feature_witness']):
        raise ValueError('actual prior accepted raw feature witnesses must reconstruct')
    pairs = []
    for camera in ('primary', 'auxiliary'):
        for role, ref in [('retained_anchor', f) for f in ANCHORS] + [('increment', FRAME-1)]:
            a, b = features[ref][camera], features[FRAME][camera]
            stages = diagnose_matches(a, b); points = matched_points(a, b)
            before = [v.tobytes() for v in points]
            failure = failure_trace(points, expected[camera][ref])
            if before != [v.tobytes() for v in points] or failure['lifted_matches'] != stages['counts']['valid_depth_pair']:
                raise ValueError('original correspondence arrays must remain byte-exact')
            pairs.append(dict(camera=camera, reference_frame=ref, current_frame=FRAME, reference_role=role,
                **stages, registration=failure))
    worker_ended(); verify(sources); verify_artifacts(root, bindings)
    write_json(OUTPUT, dict(status='CONTACT_ANCHORED_WORKER_RAW_TRACKING_FAILURE_RECONSTRUCTED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, source_count=len(sources),
        native_launch_sha256=LAUNCH_SHA, worker_terminal_sha256=WORKER_SHA, artifact_sha256=bindings,
        original_worker_ended=True, final_parent_result_required_for_run_completion=True,
        final_parent_completion_authenticated=False, model_state_sha256=record['model_state_sha256'],
        complete_decisions=count, canonical_decision_stream_sha256=stream.hexdigest(),
        terminal_observation_frame=FRAME, last_admitted_decision_frame=FRAME-1,
        readout=record['readout'], original_prefix_comparison=record['prefix_comparison'],
        actions=dict(actions), terminal_context=compact, raw_match_pairs=pairs,
        feature_witnesses={str(f):{c:x.witness() for c,x in v.items()} for f,v in features.items()},
        all_eighteen_original_pre_gyro_failures_reproduced=True,
        full_training_ancestry_reexecuted=False, raw_model_audit_reexecuted=False,
        full_observer_history_replayed=False, actual_gyro_gate_reexecuted=False,
        native_execution=False, pose_admitted=False, model_inference=False,
        original_outcome_unchanged=True, policy_selected=False, goal_achieved=False))
    print('CONTACT_WORKER_TRACKING_DIAGNOSIS', digest(OUTPUT), len(sources), count, len(pairs), flush=True)
    for p in pairs:
        if p['reference_role'] == 'increment': print(p['camera'], p['counts'], p['registration'], flush=True)


if __name__ == '__main__': main()
