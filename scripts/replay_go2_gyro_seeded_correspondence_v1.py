"""Paired complete development replay, followed by separate native evaluation."""
import json
import shutil
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lewm.gyro_seeded_rgbd_correspondence_development import GyroSeededVisualLedMotion
from lewm.joint_rgbd_rigid_pose_development import angle
from scripts.replay_go2_registration_conditioning_v1 import (
    OUTPUT as PREDECESSOR, COHORTS, TRIALS, Store, BUDGET, RESERVE, serial,
    IntentReturnRGBDReplay, MultiReferenceVisualLedMotion, verify_launch,
    verify_bindings, discover_sources, ROOT, BASE, read_json, read_npz,
    verify_artifacts, artifact_path, validate_root, create_output, digest, write_json, require)

OUTPUT = BASE / 'go2_gyro_seeded_correspondence_replay_v1_attempt_001'
SOURCE = 'scripts/replay_go2_gyro_seeded_correspondence_v1.py'
TEST = 'lewm/tests/test_gyro_seeded_correspondence_replay_development.py'
PROTOCOL = 'docs/go2_gyro_seeded_correspondence_replay_v1_2026-09-07.md'
RECEIPTS = {'launch.json': '23dbfff75d651323a443a62b8ea319d343f6c6a4f536126749b825ee86fe1b71',
    'result.json': 'dc392027aa0b9e65566b436257485369f2cdf468daa80acc43908d436415d28e'}
MODELS = dict(original=MultiReferenceVisualLedMotion, gyro_seeded=GyroSeededVisualLedMotion)


def preflight():
    validate_root(OUTPUT, must_exist=False)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive development replay; no retry/resume')
    verify_artifacts(PREDECESSOR, RECEIPTS)
    old = read_json(PREDECESSOR, 'launch.json'); verify_launch(old)
    result = read_json(PREDECESSOR, 'result.json')
    require(result['status'] == 'WHOLE_STREAM_REGISTRATION_WITNESS_COMPLETE'
        and not (PREDECESSOR / 'failure.json').exists(), 'completed original witness required')
    witness = RECEIPTS | result['output_sha256']; verify_artifacts(PREDECESSOR, witness)
    sources = discover_sources((SOURCE, TEST, PROTOCOL,
        'lewm/tests/test_gyro_seeded_rgbd_correspondence_development.py'), old['source_sha256'])
    verify_bindings(sources)
    require(shutil.disk_usage(BASE).free >= BUDGET + RESERVE, 'full replay allowance and reserve required')
    return old | dict(source_sha256=sources, predecessor_replay_sha256=witness,
        streams={c: list(MODELS) for c in COHORTS}, output_root=str(OUTPUT),
        correspondence_method='gyro_seeded_bidirectional_flow_v1', correspondence_checks_changed=True,
        estimator_modified=True, gates_modified=True, rigid_registration_gates_changed=False,
        paired_common_frame_errors=True, orientation_errors_reported=True,
        independent_validation=False, model_training=False, physical_execution=False,
        calibrated_uncertainty=False, navigation_qualified=False, goal_achieved=False)


def verify(launch):
    verify_launch(launch)
    verify_artifacts(PREDECESSOR, launch['predecessor_replay_sha256'])


def replay_trial(cohort, trial, store):
    reader = IntentReturnRGBDReplay(COHORTS[cohort]['root'] / trial)
    stem = cohort + '_' + trial; name = stem + '_estimates.jsonl'
    models = {arm: factory() for arm, factory in MODELS.items()}
    counts = {arm: dict(frames=0, available=0, first_failure=None) for arm in models}
    old_name = stem + '_original_estimates.jsonl'
    with artifact_path(PREDECESSOR, old_name).open() as witness, store.path(name).open('xb') as target:
        for frame in range(len(reader.frames)):
            p, d, f, now = reader.packet(frame); arms = {}
            old_line = witness.readline(); require(bool(old_line), 'complete original witness required')
            old = json.loads(old_line)
            require(old['frame'] == frame and old['measured_ns'] == now, 'exact original frame/clock required')
            for arm, model in models.items():
                start = time.perf_counter_ns(); result = model.observe(p, d, f, now_ns=now)
                ms = (time.perf_counter_ns() - start) / 1e6
                row = dict(pose=serial(result['current_pose']), selection=serial(result['reference_selection']),
                    failure=serial(result['terminal_failure']), observer_wall_ms=ms,
                    correspondence_attempts=serial(result.get('correspondence_attempts', [])))
                if arm == 'original':
                    require(row['pose'] == old['pose'] and row['selection'] == old['selection']
                        and row['failure'] == old['failure'], 'exact original pose/selection/failure required')
                c = counts[arm]; c['frames'] += 1; c['available'] += int(row['pose'] is not None)
                if row['pose'] is None and c['first_failure'] is None: c['first_failure'] = frame
                arms[arm] = row
            store.record(target, dict(frame=frame, measured_ns=now, arms=arms,
                native_pose_input=False, navigation_qualified=False))
            if frame % 250 == 0:
                print('GYRO_SEEDED_REPLAY', stem, frame, {a: c['available'] for a, c in counts.items()}, flush=True)
        require(not witness.readline(), 'no omitted original witness frames')
    store.hashes[name] = digest(store.output / name)
    return dict(arms=counts, frames=len(reader.frames), estimates_file=name,
        original_witness_frames_checked=len(reader.frames), post_failure_frames_retained=True)


def distribution(values):
    return dict(count=len(values), mean=float(np.mean(values)) if values else None,
        median=float(np.median(values)) if values else None,
        p95=float(np.percentile(values, 95)) if values else None,
        maximum=max(values) if values else None)


def evaluate_trial(cohort, trial, report, store):
    marker = 'all_sensor_estimates_complete.json'
    require(marker in store.hashes, 'complete sensor phase required before native evaluation')
    verify_artifacts(store.output, {marker: store.hashes[marker]})
    phase = read_json(store.output, marker)
    require(phase['native_coordinates_parsed'] is False
        and set(phase['reports']) == {c + '_' + t for c in COHORTS for t in TRIALS}
        and all(set(r['arms']) == set(MODELS) for r in phase['reports'].values()),
        'complete fixed paired sensor population required')
    require(report == phase['reports'][cohort + '_' + trial], 'exact bound sensor report required')
    verify_artifacts(store.output, {report['estimates_file']: phase['output_sha256'][report['estimates_file']]})
    raw = read_npz(COHORTS[cohort]['root'] / trial, 'physics_trace.npz')
    native = raw['base_pose_world']; R0 = Rotation.from_quat(native[749, 3:]).as_matrix()
    stats = {a: {k: [] for k in ('position_m', 'orientation_rad', 'paired_position_m',
        'paired_orientation_rad', 'all_frame_observer_ms', 'available_frame_observer_ms')} for a in MODELS}
    availability = {k: 0 for k in ('both', 'original_only', 'gyro_seeded_only', 'neither')}
    name = cohort + '_' + trial + '_evaluation.jsonl'; count = 0
    with artifact_path(store.output, report['estimates_file']).open() as source, store.path(name).open('xb') as target:
        for line in source:
            row = json.loads(line); frame = row['frame']; sample = 749 + 50 * frame
            require(frame == count and set(row['arms']) == set(MODELS), 'complete ordered paired observations required')
            require(sample < len(native) and abs(raw['timestamp_s'][sample] * 1e9 - row['measured_ns']) < 1.,
                'exact native evaluator clock required')
            actual_p = (native[sample, :3] - native[749, :3]) @ R0
            actual_R = R0.T @ Rotation.from_quat(native[sample, 3:]).as_matrix()
            have = {a: r['pose'] is not None for a, r in row['arms'].items()}
            both = all(have.values()); category = ('both' if both else 'original_only' if have['original']
                else 'gyro_seeded_only' if have['gyro_seeded'] else 'neither')
            availability[category] += 1; errors = {}
            for arm, r in row['arms'].items():
                s = stats[arm]; s['all_frame_observer_ms'].append(r['observer_wall_ms'])
                if r['pose'] is None: errors[arm] = None; continue
                p = np.asarray(r['pose']['position_initial_body_m']); R = np.asarray(r['pose']['rotation_initial_body_from_current_body'])
                pe = float(np.linalg.norm(p - actual_p)); re = angle(R.T @ actual_R)
                s['position_m'].append(pe); s['orientation_rad'].append(re)
                s['available_frame_observer_ms'].append(r['observer_wall_ms'])
                if both: s['paired_position_m'].append(pe); s['paired_orientation_rad'].append(re)
                errors[arm] = dict(position_m=pe, orientation_rad=re)
            store.record(target, dict(frame=frame, measured_ns=row['measured_ns'], native_sample_index=sample,
                availability=category, errors=errors, evaluator_only=True, navigation_qualified=False))
            count += 1
    store.hashes[name] = digest(store.output / name)
    require(count == report['frames'] and all(len(stats[a]['position_m']) == report['arms'][a]['available'] for a in MODELS),
        'whole recorded population denominators required')
    return dict(frames=count, availability=availability,
        arms={a: {k: distribution(v) for k, v in s.items()} for a, s in stats.items()},
        correlated_development_frames_not_independent_trials=True,
        timings_exclude_packet_loading_and_control=True, concurrent_cpu_work_possible=True,
        independent_validation=False, realtime_qualified=False, navigation_qualified=False)


def run():
    cv2.setNumThreads(1); launch = preflight(); create_output(OUTPUT); store = Store(OUTPUT)
    reports = {}; evaluations = {}; active = None
    try:
        store.save('launch.json', launch)
        for cohort in COHORTS:
            for trial in TRIALS:
                active = cohort + '_' + trial
                reports[active] = replay_trial(cohort, trial, store)
                store.save(active + '_replay.json', reports[active])
        verify(launch); verify_artifacts(OUTPUT, store.hashes)
        store.save('all_sensor_estimates_complete.json', dict(reports=reports,
            output_sha256=dict(store.hashes), native_coordinates_parsed=False))
        for cohort in COHORTS:
            for trial in TRIALS:
                active = cohort + '_' + trial
                evaluations[active] = evaluate_trial(cohort, trial, reports[active], store)
                store.save(active + '_summary.json', evaluations[active])
                print('GYRO_SEEDED_EVALUATION', active, evaluations[active], flush=True)
        verify(launch); verify_artifacts(OUTPUT, store.hashes)
        store.save('result.json', dict(status='GYRO_SEEDED_PAIRED_DEVELOPMENT_REPLAY_COMPLETE', reports=reports,
            evaluations=evaluations, output_sha256=dict(store.hashes), correspondence_checks_changed=True,
            rigid_registration_gates_changed=False, physical_recovery=False, model_training=False,
            independent_validation=False, navigation_qualified=False, goal_achieved=False))
    except BaseException as error:
        write_json(OUTPUT / 'failure.json', dict(status='TERMINAL_GYRO_SEEDED_REPLAY_FAILURE',
            active=active, reason=repr(error), completed_replays=list(reports), completed_evaluations=list(evaluations),
            output_sha256=dict(store.hashes), retry_performed=False))
        raise


if __name__ == '__main__': run()
