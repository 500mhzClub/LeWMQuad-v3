"""Fixed recorded-return pair diagnosis; no observer, model or native execution.

Compare existing descriptor association and existing chained image association
for all eight retained references, both cameras, at the first/final frame of the
last measured bridge. Endpoint registration retains the original rigid/gyro
gates. It does not admit a global pose or apply measured-plane refinement.
"""
import gzip
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import psutil

from lewm.causal_rgb_dataset_development import _leaf
from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.chained_corner_flow_association_development import chained_points
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.fast_gyro_development import FastRelativeOrientation
from lewm.joint_rgbd_rigid_pose_development import register
from lewm.auxiliary_reference_pose_adapter_development import gyro_in_reference, pose_in_body
from scripts.extended_budget_anchored_maze_development import ExtendedBudgetRGBDReplay, rgb_packet
from scripts.novel_maze_auxiliary_rgb_packet_development import public_acquisition
from scripts.navigation_artifact_root_development import BASE, validate_root
from scripts.startup_source_inventory_development import discover_sources, allowed_relative
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest

SOURCE = 'scripts/probe_go2_measured_plane_return_anchor_pairs_v1.py'
TEST = 'lewm/tests/test_measured_plane_return_anchor_pair_probe_development.py'
INPUT = BASE/'go2_measured_plane_dispatch_recovery_v1_attempt_001'
CASE = 'no_rgb_direct_measured_plane_maze_02'
LAUNCH_SHA = '93d0af54af1d07eb1981338e17313f5e08bca00f257a57c3a4bee055ef44aacb'
COLLECTION_SHA = '922cf4a1134eb5458010e12f89ba65fe427b839476837ab035d8ceace4181680'
OUTPUT = BASE/'go2_measured_plane_return_anchor_pair_probe_v1_attempt_001'
REFERENCES = (3100, 3099, 3098, 3096, 3095, 3094, 3093, 3092)
TARGETS = (3103, 3113)
FRAMES = tuple(range(min(REFERENCES), max(TARGETS)+1))
CAMERAS = ('primary', 'auxiliary')


def require(condition, reason):
    if not condition: raise ValueError(reason)


def write_json(path, value):
    with path.open('x') as target:
        json.dump(value, target, indent=2, allow_nan=False)
        target.write('\n')


def verify_sources(sources):
    for name, expected in sources.items():
        require(digest(ROOT/allowed_relative(name)) == expected, 'source changed: '+name)


def input_names():
    names = ['result.json', 'policy_observations.json', 'policy_histories.npz',
        'fast_gyro_histories.npz', 'depth_observations.json', 'auxiliary_camera_audit.json',
        'context_decisions.jsonl.gz']
    names += [f'{prefix}_{frame:04d}.{suffix}' for frame in FRAMES
        for prefix, suffix in (('rgb', 'png'), ('depth', 'npz'),
            ('auxiliary_rgb', 'png'), ('auxiliary_depth', 'npz'))]
    return tuple(names)


def summarize_row(row):
    decision = row['decision']; visual = decision.get('original_visual_evidence') or {}
    continuity = visual.get('continuity_evidence') or {}
    camera = visual.get('camera_selection') or {}
    selection = visual.get('reference_selection') or {}
    def compact(value):
        return {key: value.get(key) for key in ('status', 'bridge_frames', 'incremental_available',
            'anchor_available', 'anchor_failure', 'incremental_failure')}
    return dict(tick=row['tick'], terminal=decision['terminal'], failure=decision['failure'],
        requested_command=decision['requested_command'], visual_status=visual.get('status'),
        terminal_failure=visual.get('terminal_failure'), continuity=compact(continuity),
        reference_attempts=selection.get('attempts'), selected_camera=camera.get('selected_camera'),
        auxiliary_attempted=camera.get('auxiliary_attempted'),
        primary_continuity=compact(camera.get('primary_continuity') or {}))


def recorded_boundary(directory):
    rows = {}; count = 0
    with gzip.open(_leaf(directory, 'context_decisions.jsonl.gz'), 'rt') as source:
        for tick, line in enumerate(source):
            # Only decode the fixed boundary rows; verify the full line count.
            if tick in (3102, 3103, 3112, 3113):
                row = json.loads(line)
                require(row['tick'] == row['observation_index'] == tick, 'recorded boundary clock changed')
                rows[tick] = summarize_row(row)
            count += 1
    require(count == 3124 and len(rows) == 4, 'exact collected observation population required')
    require(rows[3102]['continuity']['status'] == 'ANCHOR_MEASUREMENT',
        'last bridge must follow an admitted anchor')
    require(rows[3103]['continuity']['status'] == 'MEASURED_INCREMENT_BRIDGE'
        and rows[3103]['continuity']['bridge_frames'] == 1, 'first frame of final bridge required')
    require(rows[3112]['continuity']['bridge_frames'] == 10
        and rows[3112]['terminal'] is None, 'unchanged final admitted bridge required')
    require(rows[3113]['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
        and rows[3113]['continuity']['status'] == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
        and rows[3113]['primary_continuity']['status'] == 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
        and rows[3113]['auxiliary_attempted'] is True, 'same recorded two-camera stop required')
    for tick in TARGETS:
        require(tuple(item['reference_frame'] for item in rows[tick]['reference_attempts']) == REFERENCES,
            'complete original retained reference roster required')
    return list(rows.values())


def fit_pair(features, gyros, reference, current, camera, method):
    require(reference in REFERENCES and current in TARGETS and camera in CAMERAS,
        'fixed original reference/camera/target required')
    require(method in ('descriptor', 'chained'), 'existing fixed association required')
    result = dict(reference_frame=reference, current_frame=current, camera=camera,
        method=method, qualified=False, failure=None, failure_stage=None)
    stage = 'association'
    try:
        if method == 'descriptor':
            values = matched_points(features[reference][camera], features[current][camera])
            association = dict(association='original_descriptor', endpoint_depth_pairs=len(values[0]))
        else:
            sequence = [(frame, 1_500_000_000+100_000_000*frame, features[frame][camera])
                for frame in range(reference, current+1)]
            values, association = chained_points(sequence)
            repeated, check = chained_points(sequence)
            require(association == check and all(a.dtype == b.dtype and a.shape == b.shape
                and a.tobytes() == b.tobytes() for a, b in zip(values, repeated, strict=True)),
                'chained association must repeat byte-exactly')
        result['association'] = association
        result['endpoint_arrays'] = {name:dict(shape=list(array.shape), dtype=str(array.dtype),
            sha256=hashlib.sha256(array.tobytes()).hexdigest()) for name, array in zip(
                ('reference_points', 'current_points', 'reference_pixels', 'current_pixels'), values, strict=True)}
        relative = gyros[reference].T@gyros[current]
        stage = 'endpoint_registration'
        R, t, mask, evidence = register(*values,
            gyro_rotation=relative if camera == 'primary' else gyro_in_reference(relative),
            mode='joint', frame=current)
        if camera == 'auxiliary': R, t = pose_in_body(R, t)
        result.update(qualified=True, evidence=evidence,
            reference_body_from_current_body=R.tolist(), translation_reference_body_m=t.tolist(),
            inlier_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    except SensorContractError as error:
        result.update(failure=str(error), failure_stage=stage)
    return result


def main():
    started = time.monotonic(); directory = INPUT/CASE
    validate_root(INPUT); validate_root(OUTPUT, must_exist=False)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive diagnostic output required')
    require(digest(INPUT/'launch.json') == LAUNCH_SHA, 'fixed native launch required')
    require(digest(_leaf(directory, 'result.json')) == COLLECTION_SHA, 'fixed completed collection required')
    collection = json.loads(_leaf(directory, 'result.json').read_text())
    require(collection['decisions'] == collection['rgbd_frames'] == collection['auxiliary_frames'] == 3124
        and collection['schedule_terminal'] == 'SENSOR_OR_MODEL_FAILURE'
        and collection['terminal_zero_ticks'] == 10, 'closed negative collection with complete drain required')
    inherited = json.loads((INPUT/'launch.json').read_text())['source_sha256']
    sources = discover_sources((SOURCE, TEST), inherited); verify_sources(sources)
    resources = dict(memory_available_bytes=psutil.virtual_memory().available,
        artifact_free_bytes=shutil.disk_usage(BASE).free)
    require(resources['memory_available_bytes'] >= 8*1024**3
        and resources['artifact_free_bytes'] >= 41*1024**3, 'bounded CPU diagnostic resources required')
    artifacts = {name:digest(_leaf(directory, name)) for name in input_names()}
    OUTPUT.mkdir()
    write_json(OUTPUT/'launch.json', dict(source_sha256=sources, input_sha256=artifacts,
        native_launch_sha256=LAUNCH_SHA, collection_sha256=COLLECTION_SHA, input_root=str(directory),
        references=list(REFERENCES), targets=list(TARGETS), cameras=list(CAMERAS),
        resources=resources, native_audit_required_for_navigation_claim=True,
        population=64, maximum_workers=1, opencv_threads=1))
    try:
        cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
        boundary = recorded_boundary(directory)
        reader = ExtendedBudgetRGBDReplay(directory)
        acquisitions = json.loads(_leaf(directory, 'auxiliary_camera_audit.json').read_text())
        require(len(reader.frames) == len(acquisitions) == 3124, 'complete paired manifest required')
        features = {}; gyros = {}; orientation = FastRelativeOrientation()
        for frame in FRAMES:
            policy, depth, fast, now = reader.packet(frame)
            require(now == 1_500_000_000+frame*100_000_000, 'fixed measured frame clock required')
            state = (orientation.begin(policy, fast, now_ns=now) if frame == FRAMES[0]
                else orientation.step(policy, fast, now_ns=now))
            gyros[frame] = np.asarray(state['rotation_initial_body_from_current_body'])
            image, auxiliary = rgb_packet(directory, frame, policy,
                public_acquisition(acquisitions[frame]), now_ns=now)
            features[frame] = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'], depth),
                auxiliary=CornerSupportFeatureFrame(image['rgb'], auxiliary))
        pairs = [fit_pair(features, gyros, reference, current, camera, method)
            for current in TARGETS for camera in CAMERAS for reference in REFERENCES
            for method in ('descriptor', 'chained')]
        require(len(pairs) == 64, 'complete fixed pair population required')
        verify_sources(sources)
        for name, expected in artifacts.items():
            require(digest(_leaf(directory, name)) == expected, 'raw input changed: '+name)
        require(digest(INPUT/'launch.json') == LAUNCH_SHA, 'native launch changed')
        report = dict(status='MEASURED_PLANE_RETURN_ANCHOR_PAIR_PROBE_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'), input_sha256=artifacts,
            recorded_boundary=boundary, pairs=pairs,
            feature_witnesses={str(frame):{camera:f.witness() for camera,f in views.items()}
                for frame,views in features.items()},
            gyro=dict(start_frame=FRAMES[0], end_frame=FRAMES[-1],
                public_packets_validated=len(FRAMES), samples_integrated=orientation.samples_integrated),
            elapsed_seconds=time.monotonic()-started,
            scope=dict(posthoc_pair_probe=True, native_final_audit_verified=False,
                rigid_and_gyro_thresholds_unchanged=True, all_pair_failures_preserved=True,
                measured_plane_refinement_applied=False, global_pose_envelopes_checked=False,
                anchor_increment_conflicts_checked=False, full_observer_replayed=False,
                pose_increments_composed=False, pose_admitted=False, model_loaded=False,
                controller_changed=False, native_execution=False, bridge_allowance_changed=False,
                navigation_recovered=False, goal_achieved=False))
        write_json(OUTPUT/'result.json', report)
        print(json.dumps(dict(status=report['status'], result_sha256=digest(OUTPUT/'result.json'),
            elapsed_seconds=report['elapsed_seconds'], outcomes=[dict(target=current, camera=camera, method=method,
                qualified=sum(p['qualified'] for p in pairs if
                    (p['current_frame'],p['camera'],p['method'])==(current,camera,method)))
                for current in TARGETS for camera in CAMERAS for method in ('descriptor','chained')])), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='MEASURED_PLANE_RETURN_ANCHOR_PAIR_PROBE_FAILED',
            failure=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__': main()
