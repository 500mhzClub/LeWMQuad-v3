"""Refine the three frozen qualifying image pairs against recorded floor planes.

This bounded diagnostic conditions on the audited baseline visual pose state.
It reconstructs raw floor measurements but does not replay an observer, promote
an anchor, compare anchor/increment estimates, load a model, or execute physics.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
import time
import traceback
from types import SimpleNamespace

import numpy as np

from scripts import probe_go2_measured_plane_return_anchor_pairs_v1 as previous
from lewm import measured_plane_dual_camera_pose_development as plane
from lewm.auxiliary_reference_pose_adapter_development import body_from_reference

SOURCE = 'scripts/probe_go2_measured_plane_return_anchor_refinement_v1.py'
TEST = 'lewm/tests/test_measured_plane_return_anchor_refinement_development.py'
OUTPUT = previous.BASE/'go2_measured_plane_return_anchor_refinement_v1_attempt_001'
PAIR_LAUNCH_SHA = '1abd1409776931187bdc2ce4959d57e3d47146c4350222f71f06f2377bcfc267'
PAIR_RESULT_SHA = 'd3a01cc5b667b7038964adf04ee135179af8a35c7d8bf0ed875b5baae5831c6e'
NATIVE_RESULT_SHA = '4ecd63cd96aff9277103609b3034ed87e736e1fff1bb3352113941ff2c26aa18'
PAIRS = ((3100, 3103, 'primary'), (3099, 3103, 'primary'), (3100, 3103, 'auxiliary'))
STATE_FRAMES = (3099, 3100, 3102, 3103)
require = previous.require


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def selected_pairs(report):
    require(report['status'] == 'MEASURED_PLANE_RETURN_ANCHOR_PAIR_PROBE_COMPLETE'
        and len(report['pairs']) == 64, 'same complete original pair population required')
    rows = [row for row in report['pairs'] if row['qualified']]
    require(tuple((row['reference_frame'], row['current_frame'], row['camera']) for row in rows) == PAIRS
        and all(row['method'] == 'chained' for row in rows), 'exact three previously qualifying pairs required')
    return rows


def inputs():
    for root in (previous.INPUT, previous.OUTPUT): previous.validate_root(root)
    for path, sha in ((previous.OUTPUT/'launch.json', PAIR_LAUNCH_SHA),
            (previous.OUTPUT/'result.json', PAIR_RESULT_SHA),
            (previous.INPUT/'launch.json', previous.LAUNCH_SHA),
            (previous.INPUT/'result.json', NATIVE_RESULT_SHA)):
        require(previous.digest(path) == sha, 'fixed completed input identity changed: '+str(path))
    native = json.loads((previous.INPUT/'result.json').read_text())
    prior_launch = json.loads((previous.OUTPUT/'launch.json').read_text())
    prior = json.loads((previous.OUTPUT/'result.json').read_text())
    selected_pairs(prior)
    require(native['status'] == 'MEASURED_PLANE_DISPATCH_RECOVERY_V1_COMPLETE'
        and prior['input_sha256'] == prior_launch['input_sha256']
        and prior['launch_sha256'] == PAIR_LAUNCH_SHA
        and set(prior_launch['input_sha256']) == set(previous.input_names()),
        'complete audited native input and fixed original pair receipt required')
    for root in (previous.INPUT, previous.OUTPUT):
        require(not (root/'failure.json').exists() and not (root/'failure.json').is_symlink(),
            'preserve failed predecessor without diagnostic bypass')
    for name, sha in prior_launch['input_sha256'].items():
        require(native['artifact_sha256'].get(previous.CASE+'/'+name) == sha,
            'pair input must be bound by completed native result')
        require(previous.digest(previous._leaf(previous.INPUT/previous.CASE, name)) == sha,
            'recorded pair input changed: '+name)
    previous.verify_sources(prior_launch['source_sha256'])
    sources = previous.discover_sources((SOURCE, TEST), prior_launch['source_sha256'])
    previous.verify_sources(sources)
    return prior_launch, prior, sources


def recorded_states(directory):
    states = {}; count = 0
    with gzip.open(previous._leaf(directory, 'context_decisions.jsonl.gz'), 'rt') as source:
        for frame, line in enumerate(source):
            count += 1
            if frame not in STATE_FRAMES: continue
            row = json.loads(line); decision = row['decision']; visual = decision['original_visual_evidence']
            pose = visual['current_pose']; receipt = visual['measured_plane_evidence']
            now = 1_500_000_000+frame*100_000_000
            require(row['tick'] == row['observation_index'] == pose['frame'] == receipt['frame'] == frame
                and pose['measured_ns'] == receipt['measured_ns'] == now
                and decision['terminal'] is None and decision['failure'] is None
                and pose['native_pose_input'] is False and receipt['raw_native_pose_used'] is False,
                'same admitted baseline public pose and measured plane frame required')
            states[frame] = dict(pose=pose, plane=receipt,
                original_row_sha256=hashlib.sha256(line.encode()).hexdigest())
    require(count == 3124 and set(states) == set(STATE_FRAMES), 'complete fixed baseline state population required')
    return states


def recorded_features(directory, prior):
    reader = previous.ExtendedBudgetRGBDReplay(directory)
    acquisitions = json.loads(previous._leaf(directory, 'auxiliary_camera_audit.json').read_text())
    require(len(reader.frames) == len(acquisitions) == 3124, 'complete paired manifest required')
    features = {}; gyros = {}; orientation = previous.FastRelativeOrientation()
    for frame in previous.FRAMES:
        policy, depth, fast, now = reader.packet(frame)
        require(now == 1_500_000_000+frame*100_000_000, 'same measured frame clock required')
        state = (orientation.begin(policy, fast, now_ns=now) if frame == previous.FRAMES[0]
            else orientation.step(policy, fast, now_ns=now))
        gyros[frame] = np.asarray(state['rotation_initial_body_from_current_body'])
        image, auxiliary = previous.rgb_packet(directory, frame, policy,
            previous.public_acquisition(acquisitions[frame]), now_ns=now)
        features[frame] = dict(primary=previous.CornerSupportFeatureFrame(policy['image']['rgb'], depth),
            auxiliary=previous.CornerSupportFeatureFrame(image['rgb'], auxiliary))
        require({camera: value.witness() for camera, value in features[frame].items()}
            == prior['feature_witnesses'][str(frame)], 'same original feature witnesses required')
    require(orientation.samples_integrated == prior['gyro']['samples_integrated'], 'same complete gyro interval required')
    return features, gyros


def reconstruct_plane(features, receipt, frame):
    now = 1_500_000_000+frame*100_000_000
    require(receipt['frame'] == frame and receipt['measured_ns'] == now
        and receipt['up_reference_visual_frame'] == frame-1
        and receipt['current_up_uses_public_gyro'] is False,
        'original previous-visual-pose floor extraction required')
    up = np.asarray(receipt['up_reference_rotation']).T@np.asarray(receipt['initial_up_body'])
    clouds = []
    for camera, transform in (('primary', np.asarray(plane.BODY_FROM_OPTICAL)),
            ('auxiliary', plane.body_from_optical())):
        depth = features[frame][camera].depth
        require(depth['measured_ns'] == now and depth['available_ns'] <= now
            and plane.depth_hash(depth) == receipt['depth_sha256'][camera], 'same available raw paired floor depths required')
        clouds.append(plane.measured_candidates(depth['depth_m'], depth['valid'], transform, up)[0])
    actual = plane.fit_joint_plane(*clouds, up)
    require(canonical(actual) == canonical(receipt['joint_plane']), 'raw floor-plane reconstruction changed')
    require(actual['available'], 'these three fixed pairs require available measured planes')
    plane.validate_joint_plane(actual, up)
    return actual


def refine_pair(features, gyros, states, planes, expected):
    reference, current, camera = (expected[k] for k in ('reference_frame', 'current_frame', 'camera'))
    require((reference, current, camera) in PAIRS, 'only the three frozen qualifying pairs permitted')
    repeated = previous.fit_pair(features, gyros, reference, current, camera, 'chained')
    require(repeated == expected, 'original qualified endpoint pair must reproduce exactly')
    sequence = [(frame, 1_500_000_000+frame*100_000_000, features[frame][camera])
        for frame in range(reference, current+1)]
    (a, b, ua, ub), _ = previous.chained_points(sequence)
    relative = gyros[reference].T@gyros[current]
    R, t, mask, registration = previous.register(a, b, ua, ub,
        gyro_rotation=relative if camera == 'primary' else previous.gyro_in_reference(relative),
        mode='joint', frame=current)
    require(registration == expected['evidence']
        and hashlib.sha256(mask.tobytes()).hexdigest() == expected['inlier_mask_sha256'],
        'same original inlier population required')
    if camera == 'auxiliary':
        R, t = previous.pose_in_body(R, t); A, offset = body_from_reference()
        a, b = a@A.T+offset, b@A.T+offset
    require(R.tolist() == expected['reference_body_from_current_body']
        and t.tolist() == expected['translation_reference_body_m'], 'same body endpoint fit required')
    pose = states[reference]['pose']; previous_pose = states[current-1]['pose']
    ref = SimpleNamespace(rotation=np.asarray(pose['rotation_initial_body_from_current_body']),
        position=np.asarray(pose['position_initial_body_m']))
    last_R = np.asarray(previous_pose['rotation_initial_body_from_current_body'])
    last_p = np.asarray(previous_pose['position_initial_body_m'])
    global_R, global_p = ref.rotation@R, ref.position+ref.rotation@t
    registration = deepcopy(registration) | dict(reference_inlier_points_body_m=a[mask].tolist(),
        current_inlier_points_body_m=b[mask].tolist(), reference_inlier_pixels=ua[mask].tolist(),
        current_inlier_pixels=ub[mask].tolist())
    candidate = dict(reference=ref, R=global_R, p=global_p, local_R=R, t=t, registration=registration)
    result = dict(reference_frame=reference, current_frame=current, camera=camera,
        original_endpoint_reproduced=True, qualified=False, failure=None, failure_stage=None)
    stage = 'original_global_motion_envelopes'
    try:
        if (np.linalg.norm(t) > plane.RIGID_RULES['maximum_reference_translation_m']
                or np.linalg.norm(global_p-last_p) > plane.RIGID_RULES['maximum_increment_translation_m']
                or plane.angle(last_R.T@global_R) > plane.RIGID_RULES['maximum_increment_rotation_rad']):
            raise plane.SensorContractError('original body-frame displacement envelope rejected')
        stage = 'measured_plane_refinement'
        refined = plane.refine(candidate, planes[reference], planes[current], camera=camera,
            gyro=relative, last_p=last_p, last_R=last_R)
        result.update(qualified=True, registration=refined['registration'],
            position_initial_body_m=refined['p'].tolist(),
            rotation_initial_body_from_current_body=refined['R'].tolist())
    except plane.SensorContractError as error:
        result.update(failure=str(error), failure_stage=stage, failure_type=type(error).__name__)
    return result


def main(preflight=False):
    previous.validate_root(OUTPUT, must_exist=False)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive diagnostic output required')
    prior_launch, prior, sources = inputs()
    resources = dict(memory_available_bytes=previous.psutil.virtual_memory().available,
        artifact_free_bytes=previous.shutil.disk_usage(previous.BASE).free)
    require(resources['memory_available_bytes'] >= 8*1024**3 and resources['artifact_free_bytes'] >= 41*1024**3,
        'bounded diagnostic resources required')
    if preflight:
        print('RETURN_ANCHOR_REFINEMENT_PREFLIGHT', len(sources), len(PAIRS), flush=True); return
    started = time.monotonic(); process = previous.psutil.Process(); OUTPUT.mkdir()
    previous.write_json(OUTPUT/'launch.json', dict(source_sha256=sources,
        input_sha256=prior_launch['input_sha256'], native_result_sha256=NATIVE_RESULT_SHA,
        pair_launch_sha256=PAIR_LAUNCH_SHA, pair_result_sha256=PAIR_RESULT_SHA,
        pairs=PAIRS, resources=resources, maximum_workers=1, opencv_threads=1, automatic_retry=False,
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        boot_id=previous.Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    try:
        previous.cv2.setNumThreads(1); previous.cv2.ocl.setUseOpenCL(False)
        directory = previous.INPUT/previous.CASE
        states = recorded_states(directory); features, gyros = recorded_features(directory, prior)
        planes = {frame: reconstruct_plane(features, states[frame]['plane'], frame) for frame in (3099, 3100, 3103)}
        results = [refine_pair(features, gyros, states, planes, row) for row in selected_pairs(prior)]
        require(inputs() == (prior_launch, prior, sources), 'all recorded inputs and source bindings must remain exact')
        previous.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_RETURN_ANCHOR_REFINEMENT_V1_COMPLETE',
            launch_sha256=previous.digest(OUTPUT/'launch.json'), source_sha256=sources,
            native_result_sha256=NATIVE_RESULT_SHA, pair_result_sha256=PAIR_RESULT_SHA,
            state_witnesses=states, reconstructed_planes=planes, pairs=results,
            elapsed_seconds=time.monotonic()-started,
            scope=dict(posthoc_pair_probe=True, conditioned_on_audited_baseline_visual_state=True,
                raw_paired_floor_planes_reconstructed=True, original_qualified_pairs_reproduced=True,
                original_inliers_and_gate_values_retained=True, all_three_outcomes_preserved=True,
                anchor_increment_conflicts_checked=False, full_observer_replayed=False,
                pose_admitted=False, model_loaded=False, controller_changed=False,
                native_execution=False, bridge_allowance_changed=False, navigation_recovered=False, goal_achieved=False)))
        print(json.dumps(dict(status='MEASURED_PLANE_RETURN_ANCHOR_REFINEMENT_V1_COMPLETE',
            result_sha256=previous.digest(OUTPUT/'result.json'), outcomes=[{k:row.get(k) for k in
                ('reference_frame', 'current_frame', 'camera', 'qualified', 'failure', 'failure_stage')} for row in results])), flush=True)
    except Exception as error:
        previous.write_json(OUTPUT/'failure.json', dict(status='MEASURED_PLANE_RETURN_ANCHOR_REFINEMENT_V1_FAILED',
            failure=repr(error), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--preflight', action='store_true')
    main(parser.parse_args().preflight)
