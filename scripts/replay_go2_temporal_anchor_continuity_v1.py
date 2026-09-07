"""Paired complete development replay, followed by separate native evaluation."""
import json
import shutil
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from lewm.temporal_anchor_continuity_development import TemporalAnchorVisualLedMotion, CONTINUITY_RULES
from lewm.joint_rgbd_rigid_pose_development import angle
from scripts.replay_go2_registration_conditioning_v1 import (
    OUTPUT as PREDECESSOR, COHORTS, TRIALS, Store, BUDGET, RESERVE, serial,
    IntentReturnRGBDReplay, MultiReferenceVisualLedMotion, verify_launch,
    verify_bindings, discover_sources, ROOT, BASE, read_json, read_npz,
    verify_artifacts, artifact_path, validate_root, create_output, digest, write_json, require)

OUTPUT = BASE / 'go2_temporal_anchor_continuity_v1_attempt_001'
SOURCE = 'scripts/replay_go2_temporal_anchor_continuity_v1.py'
TEST = 'lewm/tests/test_temporal_anchor_continuity_replay_development.py'
PROTOCOL = 'docs/go2_temporal_anchor_continuity_v1_2026-09-07.md'
RECEIPTS = {'launch.json': '23dbfff75d651323a443a62b8ea319d343f6c6a4f536126749b825ee86fe1b71',
    'result.json': 'dc392027aa0b9e65566b436257485369f2cdf468daa80acc43908d436415d28e'}
FROZEN_BASELINE = BASE / 'go2_measured_pose_correspondence_replay_v1_attempt_001'
BASELINE_RECEIPTS = {'launch.json': 'a82c4e75a459896c70ec3f3486c26c00fba321dab55f078bd27ba7f2e9d4a8b6',
    'result.json': '1d44e2e72497f479536924d155358de4a806f50573c8f686c0ca02c326b0c7ea'}
MODELS = dict(original=MultiReferenceVisualLedMotion, temporal_anchor=TemporalAnchorVisualLedMotion)


class ContinuityAudit:
    """Sensor-only whole-stream accounting; no estimator feedback or truth."""
    def __init__(self):
        self.frames = 0; self.previous = None; self.first_failure = None
        self.available_modes = dict(INITIAL_REFERENCE=0, ANCHOR_MEASUREMENT=0, MEASURED_INCREMENT_BRIDGE=0)
        self.spans = []; self.active = None; self.total_bridged = 0

    def close_span(self, outcome, frame, evidence=None):
        if self.active is None: return
        span = self.active | dict(outcome=outcome, following_frame=frame)
        if evidence is not None:
            span.update(rejoin_increment_available=evidence['incremental_available'],
                rejoin_disagreement_m=evidence['disagreement_m'])
        self.spans.append(span); self.active = None

    def observe(self, row):
        frame = row['frame']; now = row['measured_ns']; arm = row['arms']['temporal_anchor']
        require(frame == self.frames, 'ordered complete continuity frames required')
        pose, e = arm['pose'], arm['continuity']
        if pose is None:
            if self.first_failure is None:
                self.first_failure = dict(frame=frame, status=e.get('status') if isinstance(e, dict) else None)
            self.close_span('TERMINAL_FAILURE', frame)
            self.previous = None; self.frames += 1
            return 'UNAVAILABLE'
        require(self.first_failure is None, 'no pose after terminal continuity failure')
        require(isinstance(e, dict) and e['status'] in self.available_modes
            and pose['frame'] == frame and pose['measured_ns'] == now
            and pose['position_error_bound'] is None and not pose['global_history_reset'],
            'current measured continuity provenance required')
        mode = e['status']; self.available_modes[mode] += 1
        if frame == 0:
            require(mode == 'INITIAL_REFERENCE' and self.previous is None, 'single initial visual reference required')
        else:
            require(mode != 'INITIAL_REFERENCE' and self.previous is not None
                and e['previous_frame'] == frame - 1
                and e['previous_measured_ns'] == self.previous['measured_ns'] == now - 100_000_000,
                'uninterrupted previous measured frame required')
            if mode == 'MEASURED_INCREMENT_BRIDGE':
                require(e['incremental_available'] and not e['anchor_available']
                    and not pose['promoted_keyframe'] and pose['reference_frame'] == frame - 1,
                    'bridge must use a current increment without anchor promotion')
                require(arm['selection']['status'] == mode
                    and not arm['selection']['selected_reference_retained_anchor'], 'explicit bridge reference selection required')
                np.testing.assert_array_equal(pose['position_initial_body_m'], e['incremental_position_initial_body_m'])
                if self.active is None:
                    self.active = dict(start_frame=frame, end_frame=frame, frames=0, measured_path_m=0.)
                self.active['frames'] += 1; self.active['end_frame'] = frame
                self.active['measured_path_m'] += float(np.linalg.norm(
                    np.asarray(pose['position_initial_body_m']) - self.previous['position_initial_body_m']))
                self.total_bridged += 1
                require(self.active['frames'] == e['bridge_frames'] <= CONTINUITY_RULES['maximum_bridge_frames']
                    and e['total_bridge_frames'] == self.total_bridged
                    and abs(e['bridge_path_m'] - self.active['measured_path_m']) < 1e-12,
                    'exact bounded bridge population and measured path required')
            else:
                require(e['anchor_available'] and e['bridge_frames'] == 0, 'anchor observation required')
                if self.active is not None:
                    require(e['preceding_bridge_frames'] == self.active['frames']
                        and abs(e['preceding_bridge_path_m'] - self.active['measured_path_m']) < 1e-12,
                        'rejoining must retain preceding bridge history')
                    self.close_span('ANCHOR_REJOINED', frame, e)
                else:
                    require(e['preceding_bridge_frames'] == 0, 'no fabricated bridge history')
        self.previous = pose; self.frames += 1
        return mode

    def summary(self):
        spans = list(self.spans)
        if self.active is not None: spans.append(self.active | dict(outcome='END_OF_RECORDING', following_frame=None))
        return dict(frames=self.frames, available_modes=self.available_modes.copy(),
            first_failure=self.first_failure, bridge_spans=spans, total_bridged_frames=self.total_bridged,
            unavailable_frames=self.frames - sum(self.available_modes.values()),
            operational_budget_is_error_bound=False, independent_observations=False)


def preflight():
    validate_root(OUTPUT, must_exist=False)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive development replay; no retry/resume')
    verify_artifacts(PREDECESSOR, RECEIPTS)
    old = read_json(PREDECESSOR, 'launch.json'); verify_launch(old)
    result = read_json(PREDECESSOR, 'result.json')
    require(result['status'] == 'WHOLE_STREAM_REGISTRATION_WITNESS_COMPLETE'
        and not (PREDECESSOR / 'failure.json').exists(), 'completed original witness required')
    witness = RECEIPTS | result['output_sha256']; verify_artifacts(PREDECESSOR, witness)
    verify_artifacts(FROZEN_BASELINE, BASELINE_RECEIPTS)
    baseline_launch = read_json(FROZEN_BASELINE, 'launch.json'); verify_launch(baseline_launch)
    require(baseline_launch['predecessor_replay_sha256'] == witness, 'same original replay witness required')
    baseline_result = read_json(FROZEN_BASELINE, 'result.json')
    require(baseline_result['status'] == 'MEASURED_POSE_PAIRED_DEVELOPMENT_REPLAY_COMPLETE'
        and not (FROZEN_BASELINE / 'failure.json').exists(), 'completed frozen measured-pose baseline required')
    baseline_witness = BASELINE_RECEIPTS | baseline_result['output_sha256']
    verify_artifacts(FROZEN_BASELINE, baseline_witness)
    sources = discover_sources((SOURCE, TEST, PROTOCOL,
        'lewm/tests/test_temporal_anchor_continuity_development.py'), baseline_launch['source_sha256'])
    verify_bindings(sources)
    require(shutil.disk_usage(BASE).free >= BUDGET + RESERVE, 'full replay allowance and reserve required')
    return old | dict(source_sha256=sources, predecessor_replay_sha256=witness,
        measured_pose_baseline_replay_sha256=baseline_witness, measured_pose_baseline_rerun=False,
        streams={c: list(MODELS) for c in COHORTS}, output_root=str(OUTPUT),
        correspondence_method='original_descriptor_with_temporal_anchor_continuity_v1', continuity_rules=dict(CONTINUITY_RULES), correspondence_checks_changed=False,
        estimator_modified=True, gates_modified=True, rigid_registration_gates_changed=False,
        paired_common_frame_errors=True, orientation_errors_reported=True,
        independent_validation=False, model_training=False, physical_execution=False,
        calibrated_uncertainty=False, navigation_qualified=False, goal_achieved=False)


def verify(launch):
    verify_launch(launch)
    verify_artifacts(PREDECESSOR, launch['predecessor_replay_sha256'])
    verify_artifacts(FROZEN_BASELINE, launch['measured_pose_baseline_replay_sha256'])


def replay_trial(cohort, trial, store):
    reader = IntentReturnRGBDReplay(COHORTS[cohort]['root'] / trial)
    stem = cohort + '_' + trial; name = stem + '_estimates.jsonl'
    models = {arm: factory() for arm, factory in MODELS.items()}
    counts = {arm: dict(frames=0, available=0, first_failure=None) for arm in models}
    continuity = ContinuityAudit()
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
                    continuity=serial(result.get('continuity_evidence')))
                if arm == 'original':
                    require(row['pose'] == old['pose'] and row['selection'] == old['selection']
                        and row['failure'] == old['failure'], 'exact original pose/selection/failure required')
                c = counts[arm]; c['frames'] += 1; c['available'] += int(row['pose'] is not None)
                if row['pose'] is None and c['first_failure'] is None: c['first_failure'] = frame
                arms[arm] = row
            row = dict(frame=frame, measured_ns=now, arms=arms,
                native_pose_input=False, navigation_qualified=False)
            continuity.observe(row)
            store.record(target, row)
            if frame % 250 == 0:
                print('TEMPORAL_ANCHOR_REPLAY', stem, frame, {a: c['available'] for a, c in counts.items()}, flush=True)
        require(not witness.readline(), 'no omitted original witness frames')
    store.hashes[name] = digest(store.output / name)
    return dict(arms=counts, frames=len(reader.frames), estimates_file=name,
        original_witness_frames_checked=len(reader.frames), post_failure_frames_retained=True,
        continuity=continuity.summary())


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
        'paired_orientation_rad', 'all_frame_observer_ms', 'available_frame_observer_ms',
        'incremental_position_m', 'incremental_orientation_rad')} for a in MODELS}
    bridge_stats = {k: [] for k in ('position_m', 'orientation_rad', 'incremental_position_m', 'incremental_orientation_rad')}
    previous = {a: None for a in MODELS}; continuity = ContinuityAudit()
    availability = {k: 0 for k in ('both', 'original_only', 'temporal_anchor_only', 'neither')}
    name = cohort + '_' + trial + '_evaluation.jsonl'; count = 0
    with artifact_path(store.output, report['estimates_file']).open() as source, store.path(name).open('xb') as target:
        for line in source:
            row = json.loads(line); frame = row['frame']; sample = 749 + 50 * frame
            mode = continuity.observe(row)
            require(frame == count and set(row['arms']) == set(MODELS), 'complete ordered paired observations required')
            require(sample < len(native) and abs(raw['timestamp_s'][sample] * 1e9 - row['measured_ns']) < 1.,
                'exact native evaluator clock required')
            actual_p = (native[sample, :3] - native[749, :3]) @ R0
            actual_R = R0.T @ Rotation.from_quat(native[sample, 3:]).as_matrix()
            have = {a: r['pose'] is not None for a, r in row['arms'].items()}
            both = all(have.values()); category = ('both' if both else 'original_only' if have['original']
                else 'temporal_anchor_only' if have['temporal_anchor'] else 'neither')
            availability[category] += 1; errors = {}
            for arm, r in row['arms'].items():
                s = stats[arm]; s['all_frame_observer_ms'].append(r['observer_wall_ms'])
                if r['pose'] is None:
                    errors[arm] = None; previous[arm] = None
                    continue
                p = np.asarray(r['pose']['position_initial_body_m']); R = np.asarray(r['pose']['rotation_initial_body_from_current_body'])
                pe = float(np.linalg.norm(p - actual_p)); re = angle(R.T @ actual_R)
                s['position_m'].append(pe); s['orientation_rad'].append(re)
                s['available_frame_observer_ms'].append(r['observer_wall_ms'])
                if both: s['paired_position_m'].append(pe); s['paired_orientation_rad'].append(re)
                incremental_position = incremental_rotation = None
                before = previous[arm]
                if before is not None:
                    incremental_position = float(np.linalg.norm((p - before[0]) - (actual_p - before[2])))
                    incremental_rotation = angle((before[1].T @ R).T @ (before[3].T @ actual_R))
                    s['incremental_position_m'].append(incremental_position)
                    s['incremental_orientation_rad'].append(incremental_rotation)
                previous[arm] = p, R, actual_p, actual_R
                errors[arm] = dict(position_m=pe, orientation_rad=re,
                    incremental_position_m=incremental_position, incremental_orientation_rad=incremental_rotation)
                if arm == 'temporal_anchor' and mode == 'MEASURED_INCREMENT_BRIDGE':
                    require(incremental_position is not None, 'bridge requires previous current pose for incremental evaluation')
                    for k in bridge_stats: bridge_stats[k].append(errors[arm][k])
            store.record(target, dict(frame=frame, measured_ns=row['measured_ns'], native_sample_index=sample,
                availability=category, continuity_mode=mode, errors=errors, evaluator_only=True, navigation_qualified=False))
            count += 1
    store.hashes[name] = digest(store.output / name)
    require(count == report['frames'] and all(len(stats[a]['position_m']) == report['arms'][a]['available'] for a in MODELS),
        'whole recorded population denominators required')
    require(continuity.summary() == report['continuity']
        and len(bridge_stats['position_m']) == report['continuity']['total_bridged_frames'],
        'same complete sensor-only bridge population required')
    return dict(frames=count, availability=availability,
        arms={a: {k: distribution(v) for k, v in s.items()} for a, s in stats.items()},
        continuity=continuity.summary(), bridged_frame_errors={k: distribution(v) for k, v in bridge_stats.items()},
        incremental_error_is_consecutive_estimate_difference=True,
        correlated_development_frames_not_independent_trials=True,
        timings_exclude_packet_loading_and_control=True, concurrent_cpu_work_possible=True,
        independent_validation=False, realtime_qualified=False, navigation_qualified=False)


def compare_frozen(reports, previous):
    """Compare complete populations without rerunning or selecting old arms."""
    require(previous['status'] == 'MEASURED_POSE_PAIRED_DEVELOPMENT_REPLAY_COMPLETE'
        and set(reports) == set(previous['reports']) == set(previous['evaluations']),
        'complete frozen measured-pose population required')
    for key, r in reports.items():
        old = previous['reports'][key]
        require(r['frames'] == old['frames']
            and r['arms']['original'] == old['arms']['original'],
            'identical original trajectory availability/failure population required')
    return dict(result_sha256=BASELINE_RECEIPTS['result.json'],
        reports=previous['reports'], evaluations=previous['evaluations'],
        original_streams_independently_matched_to_same_witness=True,
        measured_pose_baseline_rerun=False, cross_run_timing_comparison_qualified=False,
        independent_validation=False, navigation_qualified=False)


def frozen_comparison(reports):
    binding = {'result.json': BASELINE_RECEIPTS['result.json']}
    verify_artifacts(FROZEN_BASELINE, binding)
    result = compare_frozen(reports, read_json(FROZEN_BASELINE, 'result.json'))
    verify_artifacts(FROZEN_BASELINE, binding)
    return result


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
                print('TEMPORAL_ANCHOR_EVALUATION', active, evaluations[active], flush=True)
        verify(launch); verify_artifacts(OUTPUT, store.hashes)
        store.save('frozen_measured_pose_baseline_comparison.json', frozen_comparison(reports))
        verify_artifacts(OUTPUT, store.hashes)
        store.save('result.json', dict(status='TEMPORAL_ANCHOR_PAIRED_DEVELOPMENT_REPLAY_COMPLETE', reports=reports,
            evaluations=evaluations, output_sha256=dict(store.hashes), correspondence_checks_changed=False,
            rigid_registration_gates_changed=False, physical_recovery=False, model_training=False,
            independent_validation=False, navigation_qualified=False, goal_achieved=False))
    except BaseException as error:
        write_json(OUTPUT / 'failure.json', dict(status='TERMINAL_TEMPORAL_ANCHOR_REPLAY_FAILURE',
            active=active, reason=repr(error), completed_replays=list(reports), completed_evaluations=list(evaluations),
            output_sha256=dict(store.hashes), retry_performed=False))
        raise


if __name__ == '__main__': run()
