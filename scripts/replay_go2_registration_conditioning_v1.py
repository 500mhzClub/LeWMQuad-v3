"""Whole-stream frozen-observer witnesses, then evaluator-only native errors."""
from contextlib import ExitStack
import json
import os
import shutil
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.intent_room_return_scene_development import TRIALS
from lewm.multi_reference_rgbd_pose_development import MultiReferenceVisualLedMotion
from lewm.balanced_multi_reference_rgbd_development import BalancedVisualLedMotion
from lewm.rgbd_registration_trace_development import TracePairs
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json, read_npz
from scripts.analyze_go2_ground_plane_development_v1 import verify_bindings
from scripts.probe_go2_rgbd_correspondence_motion_development_v1 import verify

OUTPUT = BASE / 'go2_registration_conditioning_replay_v1_attempt_001'
BALANCED = ROOT / '.generated/go2_balanced_feature_replay_v1_attempt_001'
BALANCED_RESULT_SHA = '4710a28bab24babe71b6f48a288c9e2f9795937e11638996bbd38e50a3791ed3'
SOURCE = 'scripts/replay_go2_registration_conditioning_v1.py'
TEST = 'lewm/tests/test_registration_conditioning_replay_development.py'
PROTOCOL = 'docs/go2_registration_conditioning_replay_v1_2026-09-07.md'
COHORTS = dict(
    inner=dict(root=BASE / 'go2_inner_arrival_room_return_v1_attempt_001', arms=('original',),
        receipts={'launch.json': '53ec5a2a04831ddbc8ae3932038ce90a5f12c797d38087bb6fb95a3e8778667a',
            'result.json': '938090fce09f77b9c55c919d6ee3b29d71dde5b5939374a542aa0edf5a7c0456',
            'raw_return_audit.json': '07ae6d3a3ff3183e015617159cbd499faf751cbf13af646ac1ddcfa2bc0b8810'}),
    intent=dict(root=BASE / 'go2_intent_room_return_v1_attempt_001', arms=('original', 'balanced'),
        receipts={'launch.json': '7a5c427ca521de367a301376fafd262aeda5f6e16b7ed876f2b40e87ce0b1a91',
            'result.json': '27e2f91eaece8667e48fb98d75ce3ee8cfcc3a1c9d0052a97033cad5acc321d3',
            'raw_return_audit.json': 'a350c74d4f5851a7486bf01a8276be7f9eb420a83cd8198ad684159b8385923d'}))
MODELS = dict(original=MultiReferenceVisualLedMotion, balanced=BalancedVisualLedMotion)
BUDGET = 2 * 1024**3
RESERVE = 40 * 1024**3


def require(ok, message):
    if not ok: raise ValueError(message)


def serial(value):
    return json.loads(json.dumps(value, allow_nan=False))


def preflight():
    validate_root(OUTPUT, must_exist=False)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive diagnostic attempt; no retry/resume')
    definitions = {}; inputs = {}; inherited = {}
    for cohort, row in COHORTS.items():
        root = row['root']; verify_artifacts(root, row['receipts'])
        old = read_json(root, 'launch.json'); verify(old)
        terminal = read_json(root, 'result.json')
        require(not terminal['absent_expected_artifacts'], 'complete audited original cohort required')
        bindings = row['receipts'] | terminal['artifact_sha256']; verify_artifacts(root, bindings)
        for name, sha in old['source_sha256'].items():
            require(name not in inherited or inherited[name] == sha, 'predecessor source identities must agree')
            inherited[name] = sha
        definitions[cohort] = old; inputs[cohort] = bindings
    witness = {str((BALANCED / 'result.json').relative_to(ROOT)): BALANCED_RESULT_SHA}
    verify_bindings(witness); result = read_json(BALANCED, 'result.json')
    require(result['status'] == 'BALANCED_FEATURE_REPLAY_COMPLETE' and set(result['conditions']) == set(TRIALS),
        'complete original balanced-replay witness required')
    for trial in TRIALS:
        for arm in ('original', 'balanced'):
            name = trial + '_' + arm + '_estimates.json'
            witness[str((BALANCED / name).relative_to(ROOT))] = result['conditions'][trial]['arms'][arm]['estimates_sha256']
    verify_bindings(witness)
    sources = discover_sources((SOURCE, TEST, PROTOCOL, 'lewm/tests/test_rgbd_registration_trace_development.py'), inherited)
    verify_bindings(sources)
    require(shutil.disk_usage(BASE).free >= RESERVE + BUDGET, 'whole diagnostic allowance and reserve required')
    return dict(source_sha256=sources, predecessor_definitions=definitions, cohort_input_sha256=inputs,
        balanced_witness_sha256=witness, streams={c: list(r['arms']) for c, r in COHORTS.items()},
        maximum_artifact_bytes=BUDGET, minimum_free_bytes=RESERVE,
        estimator_modified=False, gates_modified=False, physical_execution=False, model_training=False,
        candidate_diagnostics_fed_to_observer=False, native_parse_after_all_estimates=True,
        calibrated_uncertainty=False, realtime_qualified=False, navigation_qualified=False, goal_achieved=False)


def verify_launch(launch):
    verify_bindings(launch['source_sha256'] | launch['balanced_witness_sha256'])
    for cohort, definition in launch['predecessor_definitions'].items():
        verify(definition)
        verify_artifacts(COHORTS[cohort]['root'], launch['cohort_input_sha256'][cohort])


class Store:
    def __init__(self, output):
        self.output = validate_root(output); self.used = 0; self.hashes = {}

    def capacity(self, n):
        require(self.used + n <= BUDGET and shutil.disk_usage(BASE).free >= RESERVE + n, 'diagnostic storage reserve/budget')

    def path(self, name):
        import re
        require(type(name) is str and re.fullmatch('[a-z][a-z0-9_]*[.]jsonl?', name)
            and not name.startswith('sealed_'), 'ordinary explicit diagnostic name required')
        p = self.output / name
        require(not p.exists() and not p.is_symlink(), 'exclusive diagnostic artifact required')
        return p

    def record(self, target, value):
        raw = (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode()
        require(len(raw) <= 1024**2, 'bounded single diagnostic row required')
        self.capacity(len(raw)); target.write(raw); target.flush(); os.fsync(target.fileno()); self.used += len(raw)

    def save(self, name, value):
        raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
        require(len(raw) <= 32 * 1024**2, 'bounded diagnostic metadata required')
        self.capacity(len(raw)); p = self.path(name)
        with p.open('xb') as out: out.write(raw); out.flush(); os.fsync(out.fileno())
        self.used += len(raw); self.hashes[name] = digest(p)


def replay_trial(cohort, trial, store):
    source = COHORTS[cohort]; root = source['root']; reader = IntentReturnRGBDReplay(root / trial)
    original = read_json(root / trial, 'servo_decisions.json')
    expected = ({arm: read_json(BALANCED, trial + '_' + arm + '_estimates.json') for arm in source['arms']}
        if cohort == 'intent' else {})  # Inner uses its original physical decision stream.
    models = {a: MODELS[a]() for a in source['arms']}; reports = {}
    names = {a: cohort + '_' + trial + '_' + a + '_estimates.jsonl' for a in models}
    require(all(len(rows) == len(reader.frames) for rows in expected.values()), 'entire original replay population required')
    with ExitStack() as stack:
        files = {a: stack.enter_context(store.path(name).open('xb')) for a, name in names.items()}
        counts = {a: dict(frames=0, available=0, exact_recorded_decisions=0, exact_replay_frames=0,
            candidate_calls=0, first_failure=None) for a in models}
        for frame in range(len(reader.frames)):
            p, d, f, now = reader.packet(frame)
            for arm, model in models.items():
                start = time.perf_counter_ns()
                with TracePairs() as trace: result = model.observe(p, d, f, now_ns=now)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                pose = serial(result['current_pose']); count = counts[arm]
                if arm == 'original' and frame < len(original):
                    require(pose == original[frame]['evidence']['current_pose'], 'exact original physical-decision pose required')
                    count['exact_recorded_decisions'] += 1
                if arm in expected:
                    old = expected[arm][frame]
                    require(pose == old['pose'] and serial(result['reference_selection']) == old['selection']
                        and serial(result['terminal_failure']) == old['terminal_failure'], 'exact original replay outcome required')
                    count['exact_replay_frames'] += 1
                pairs = trace.record(); count['frames'] += 1; count['available'] += int(pose is not None)
                count['candidate_calls'] += len(pairs)
                if pose is None and count['first_failure'] is None: count['first_failure'] = frame
                store.record(files[arm], dict(cohort=cohort, trial=trial, arm=arm, frame=frame, measured_ns=now,
                    pose=pose, selection=result['reference_selection'], failure=result['terminal_failure'],
                    instrumented_observer_wall_ms=elapsed, pair_witnesses=pairs,
                    native_pose_input=False, gates_modified=False))
            if frame % 250 == 0:
                print('REGISTRATION_WITNESS_REPLAY', cohort, trial, frame,
                    {a: counts[a]['available'] for a in models}, flush=True)
        for arm in models:
            require(counts[arm]['exact_recorded_decisions'] == len(original) if arm == 'original' else True,
                'complete original decision population required')
            reports[arm] = counts[arm] | dict(estimates_file=names[arm],
                post_failure_frames_retained=True, observer_instrumented=True)
    for name in names.values(): store.hashes[name] = digest(store.output / name)
    return reports


def evaluate_trial(cohort, trial, arm, report, store):
    """Parse native coordinates only after ALL nine estimate streams are saved."""
    marker = 'all_sensor_estimates_complete.json'
    require(marker in store.hashes, 'bound complete sensor phase required before native evaluation')
    verify_artifacts(store.output, {marker: store.hashes[marker]})
    phase = read_json(store.output, marker)
    require(phase['native_coordinates_parsed'] is False
        and set(phase['reports']) == {c + '_' + t for c in COHORTS for t in TRIALS},
        'complete fixed sensor-only replay population required')
    verify_artifacts(store.output, {report['estimates_file']: phase['estimates_sha256'][report['estimates_file']]})
    raw = read_npz(COHORTS[cohort]['root'] / trial, 'physics_trace.npz')
    native = raw['base_pose_world']; R0 = Rotation.from_quat(native[749, 3:]).as_matrix()
    name = cohort + '_' + trial + '_' + arm + '_evaluation.jsonl'
    stats = {}; frames = 0; available = 0; maximum = None; timings = []; available_timings = []
    component_times = {'matching': [], 'registration': []}
    with artifact_path(store.output, report['estimates_file']).open() as source, store.path(name).open('xb') as target:
        for line in source:
            row = json.loads(line); frame = row['frame']; require(frame == frames, 'complete ordered estimator rows required')
            sample = 749 + 50 * frame
            require(sample < len(native) and abs(raw['timestamp_s'][sample] * 1e9 - row['measured_ns']) < 1.,
                'exact native evaluator clock required')
            actual = (native[sample, :3] - native[749, :3]) @ R0
            error = None if row['pose'] is None else float(np.linalg.norm(np.asarray(row['pose']['position_initial_body_m']) - actual))
            if error is not None:
                available += 1; maximum = error if maximum is None else max(maximum, error)
                available_timings.append(row['instrumented_observer_wall_ms'])
            timings.append(row['instrumented_observer_wall_ms'])
            pairs = []
            for witness in row['pair_witnesses']:
                for component in component_times:
                    key = 'instrumented_' + component + '_wall_ms'
                    if key in witness: component_times[component].append(witness[key])
                p = witness['diagnostic_position_initial_body_m']; registration = witness['registration']
                pair_error = None if p is None else float(np.linalg.norm(np.asarray(p) - actual))
                gates = registration.get('original_gate_failures', {}) if registration else {}
                group = ('candidate_qualified' if witness['original_candidate_accepted'] else
                    'grid_only_rejected' if any(gates.get(k) for k in ('reference_grid', 'current_grid'))
                        and not gates.get('inlier_fraction') and not gates.get('reference_translation') else 'other_rejected')
                s = stats.setdefault(group, dict(calls=0, diagnostic_candidates=0, max_position_error_m=None))
                s['calls'] += 1
                if pair_error is not None:
                    s['diagnostic_candidates'] += 1
                    s['max_position_error_m'] = pair_error if s['max_position_error_m'] is None else max(s['max_position_error_m'], pair_error)
                pairs.append(dict(reference_frame=witness['reference_frame'], diagnostic_group=group,
                    native_position_error_m=pair_error, original_candidate_accepted=witness['original_candidate_accepted']))
            store.record(target, dict(frame=frame, measured_ns=row['measured_ns'], native_sample_index=sample,
                available_pose_error_m=error, candidates=pairs, evaluator_only=True,
                rejection_reclassified=False, navigation_qualified=False))
            frames += 1
    store.hashes[name] = digest(store.output / name)
    require(frames == report['frames'] and available == report['available'], 'full evaluator denominator required')
    return dict(frames=frames, available=available, first_failure=report['first_failure'],
        maximum_available_position_error_m=maximum, candidate_groups=stats,
        instrumented_all_frame_observer_ms_median=float(np.median(timings)),
        instrumented_all_frame_observer_ms_p95=float(np.percentile(timings, 95)),
        instrumented_available_observer_ms_median=float(np.median(available_timings)) if available_timings else None,
        instrumented_available_observer_ms_p95=float(np.percentile(available_timings, 95)) if available_timings else None,
        instrumented_pair_components={k: dict(calls=len(v), median_ms=float(np.median(v)) if v else None,
            p95_ms=float(np.percentile(v, 95)) if v else None) for k, v in component_times.items()},
        timing_includes_trace_overhead_and_terminal_noops=True, realtime_qualified=False,
        correlated_development_frames_not_independent_trials=True, uncertainty_calibrated=False,
        thresholds_selected=False, gates_modified=False, navigation_qualified=False)


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
        verify_launch(launch); verify_artifacts(OUTPUT, store.hashes)
        store.save('all_sensor_estimates_complete.json', dict(reports=reports,
            estimates_sha256=dict(store.hashes), native_coordinates_parsed=False))
        for cohort, row in COHORTS.items():
            for trial in TRIALS:
                for arm in row['arms']:
                    active = cohort + '_' + trial + '_' + arm
                    evaluations[active] = evaluate_trial(cohort, trial, arm, reports[cohort + '_' + trial][arm], store)
                    store.save(active + '_summary.json', evaluations[active])
                    print('REGISTRATION_WITNESS_EVALUATION', active, evaluations[active], flush=True)
        verify_launch(launch); verify_artifacts(OUTPUT, store.hashes)
        store.save('result.json', dict(status='WHOLE_STREAM_REGISTRATION_WITNESS_COMPLETE',
            reports=reports, evaluations=evaluations, output_sha256=dict(store.hashes),
            estimator_modified=False, gates_modified=False, physical_recovery=False,
            model_training=False, realtime_qualified=False, navigation_qualified=False, goal_achieved=False))
    except BaseException as error:
        write_json(OUTPUT / 'failure.json', dict(status='TERMINAL_REGISTRATION_WITNESS_FAILURE',
            active=active, reason=repr(error), completed_replays=list(reports),
            completed_evaluations=list(evaluations), output_sha256=dict(store.hashes), retry_performed=False))
        raise


if __name__ == '__main__': run()
