"""Read-only independent row accounting for the fixed completed replay.

No estimator execution, artifact writes, source export, recursive discovery,
native reads before a bound complete sensor phase, or physical qualification.
"""
import argparse
import itertools
import json

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.replay_go2_temporal_anchor_continuity_v1 import (
    OUTPUT, PREDECESSOR, COHORTS, TRIALS, verify, verify_artifacts,
    read_json, read_npz, artifact_path)

LAUNCH_SHA = 'd774dddf1b012dc0f69116d7e11e0f3307c0c5e549e807ef65e227b0d240b143'
ARMS = ('original', 'temporal_anchor')


def same(actual, expected, *, tolerance=1e-12):
    """Exact structure/denominators; small numerical-only float tolerance."""
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys()
        for key, value in expected.items():same(actual[key], value, tolerance=tolerance)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):same(a, b, tolerance=tolerance)
    elif type(expected) is float:
        assert type(actual) in (float, int) and np.isfinite(actual) and np.isfinite(expected)
        assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)
    else:
        assert type(actual) is type(expected) and actual == expected, (actual, expected)


def statistics(values):
    if not values:return dict(count=0, mean=None, median=None, p95=None, maximum=None)
    v = np.asarray(values, dtype=np.float64)
    assert np.isfinite(v).all()
    return dict(count=len(v), mean=float(v.sum()/len(v)), median=float(np.quantile(v, .5)),
        p95=float(np.quantile(v, .95)), maximum=float(np.max(v)))


def pose_errors(p, R, truth_p, truth_R, previous):
    # Quaternion projection/composition is independent of the runner's angle().
    result = dict(position_m=float(np.linalg.norm(p-truth_p)),
        orientation_rad=float((Rotation.from_matrix(R).inv()*Rotation.from_matrix(truth_R)).magnitude()),
        incremental_position_m=None, incremental_orientation_rad=None)
    if previous is not None:
        op, oR, tp, tR = previous
        result['incremental_position_m'] = float(np.linalg.norm((p-op)-(truth_p-tp)))
        observed = Rotation.from_matrix(oR).inv()*Rotation.from_matrix(R)
        actual = Rotation.from_matrix(tR).inv()*Rotation.from_matrix(truth_R)
        result['incremental_orientation_rad'] = float((observed.inv()*actual).magnitude())
    return result


def verify_stream(cohort, trial, result):
    assert result['status'] == 'TEMPORAL_ANCHOR_PAIRED_DEVELOPMENT_REPLAY_COMPLETE'
    marker = 'all_sensor_estimates_complete.json'
    verify_artifacts(OUTPUT, {marker: result['output_sha256'][marker]})
    phase = read_json(OUTPUT, marker)
    assert phase['native_coordinates_parsed'] is False and phase['reports'] == result['reports']
    assert set(phase['reports']) == {c+'_'+t for c in COHORTS for t in TRIALS}
    key = cohort+'_'+trial; report = result['reports'][key]; evaluation = result['evaluations'][key]
    native = read_npz(COHORTS[cohort]['root']/trial, 'physics_trace.npz')
    poses = native['base_pose_world']; origin = Rotation.from_quat(poses[749, 3:])
    metric_names = ('position_m', 'orientation_rad', 'incremental_position_m', 'incremental_orientation_rad')
    stats = {arm: {k: [] for k in (*metric_names, 'paired_position_m', 'paired_orientation_rad',
        'all_frame_observer_ms', 'available_frame_observer_ms')} for arm in ARMS}
    bridged = {k: [] for k in metric_names}; before = {a: None for a in ARMS}
    first = {a: None for a in ARMS}; categories = dict(both=0, original_only=0, temporal_anchor_only=0, neither=0)
    modes = dict(INITIAL_REFERENCE=0, ANCHOR_MEASUREMENT=0, MEASURED_INCREMENT_BRIDGE=0)
    previous_pose = None; anchors = []; spans = []; active = None; bridge_count = 0; failure = None; count = 0
    def close(outcome, frame, evidence=None):
        nonlocal active
        if active is None:return
        active.update(outcome=outcome, following_frame=frame)
        if evidence is not None:active.update(rejoin_increment_available=evidence['incremental_available'],
            rejoin_disagreement_m=evidence['disagreement_m'])
        spans.append(active);active = None
    with artifact_path(OUTPUT, report['estimates_file']).open() as a, \
            artifact_path(OUTPUT, key+'_evaluation.jsonl').open() as b, \
            artifact_path(PREDECESSOR, key+'_original_estimates.jsonl').open() as c:
        for lines in itertools.zip_longest(a, b, c):
            assert all(lines), 'different complete populations'
            row, scored, original = map(json.loads, lines)
            n = count; now = row['measured_ns']; sample = 749+50*n
            assert row['frame'] == scored['frame'] == original['frame'] == n
            assert now == scored['measured_ns'] == original['measured_ns']
            assert scored['native_sample_index'] == sample and abs(native['timestamp_s'][sample]*1e9-now) < 1
            assert set(row['arms']) == set(ARMS)
            for field in ('pose', 'selection', 'failure'):same(row['arms']['original'][field], original[field], tolerance=0.)
            present = {a: row['arms'][a]['pose'] is not None for a in ARMS}
            category = ('both' if all(present.values()) else 'original_only' if present['original']
                else 'temporal_anchor_only' if present['temporal_anchor'] else 'neither')
            assert scored['availability'] == category;categories[category] += 1
            candidate = row['arms']['temporal_anchor'];pose = candidate['pose'];e = candidate['continuity']
            mode = 'UNAVAILABLE' if pose is None else e['status'];assert scored['continuity_mode'] == mode
            if pose is None:
                if failure is None:failure = dict(frame=n, status=e.get('status') if isinstance(e, dict) else None)
                close('TERMINAL_FAILURE', n)
            else:
                assert failure is None and pose['frame'] == n and pose['measured_ns'] == now
                assert not pose['global_history_reset'] and pose['position_error_bound'] is None
                modes[mode] += 1
                if n == 0:
                    assert mode == 'INITIAL_REFERENCE';anchors = [0]
                else:
                    assert previous_pose is not None and e['previous_frame'] == n-1
                    assert e['previous_measured_ns'] == previous_pose['measured_ns'] == now-100_000_000
                    if mode == 'MEASURED_INCREMENT_BRIDGE':
                        assert not e['anchor_available'] and e['incremental_available']
                        assert not pose['promoted_keyframe'] and pose['reference_frame'] == n-1
                        same(pose['position_initial_body_m'], e['incremental_position_initial_body_m'], tolerance=0.)
                        assert candidate['selection']['status'] == mode and not candidate['selection']['selected_reference_retained_anchor']
                        if active is None:active = dict(start_frame=n, end_frame=n, frames=0, measured_path_m=0.)
                        active['end_frame'] = n;active['frames'] += 1;bridge_count += 1
                        active['measured_path_m'] += float(np.linalg.norm(np.subtract(
                            pose['position_initial_body_m'], previous_pose['position_initial_body_m'])))
                        assert e['bridge_frames'] == active['frames'] <= 10 and e['total_bridge_frames'] == bridge_count
                        same(e['bridge_path_m'], active['measured_path_m'])
                    else:
                        assert mode == 'ANCHOR_MEASUREMENT' and e['anchor_available'] and e['bridge_frames'] == 0
                        assert pose['reference_frame'] in anchors
                        if e['incremental_available']:
                            distance = float(np.linalg.norm(np.subtract(pose['position_initial_body_m'], e['incremental_position_initial_body_m'])))
                            same(e['disagreement_m'], distance);assert distance <= .02
                        if active is not None:
                            assert e['preceding_bridge_frames'] == active['frames']
                            same(e['preceding_bridge_path_m'], active['measured_path_m']);close('ANCHOR_REJOINED', n, e)
                        else:assert e['preceding_bridge_frames'] == 0
                        if pose['promoted_keyframe']:anchors = (anchors+[n])[-8:]
            truth_p = origin.inv().apply(poses[sample, :3]-poses[749, :3])
            truth_R = (origin.inv()*Rotation.from_quat(poses[sample, 3:])).as_matrix()
            for arm in ARMS:
                saved = row['arms'][arm];s = stats[arm];ms = saved['observer_wall_ms']
                assert np.isfinite(ms) and ms >= 0;s['all_frame_observer_ms'].append(ms)
                if not present[arm]:
                    assert scored['errors'][arm] is None;before[arm] = None
                    if first[arm] is None:first[arm] = n
                    continue
                assert first[arm] is None, 'terminal observer resumed'
                p = np.asarray(saved['pose']['position_initial_body_m']);R = np.asarray(saved['pose']['rotation_initial_body_from_current_body'])
                errors = pose_errors(p, R, truth_p, truth_R, before[arm]);before[arm] = p, R, truth_p, truth_R
                assert scored['errors'][arm].keys() == errors.keys()
                for name, value in errors.items():
                    same(scored['errors'][arm][name], value, tolerance=1e-10 if 'orientation' in name else 1e-12)
                s['available_frame_observer_ms'].append(ms)
                for name, value in errors.items():
                    if value is not None:s[name].append(value)
                    if arm == 'temporal_anchor' and mode == 'MEASURED_INCREMENT_BRIDGE':
                        assert value is not None;bridged[name].append(value)
                if category == 'both':
                    s['paired_position_m'].append(errors['position_m']);s['paired_orientation_rad'].append(errors['orientation_rad'])
            previous_pose = pose;count += 1
    if active is not None:close('END_OF_RECORDING', None)
    continuity = dict(frames=count, available_modes=modes, first_failure=failure, bridge_spans=spans,
        total_bridged_frames=bridge_count, unavailable_frames=count-sum(modes.values()),
        operational_budget_is_error_bound=False, independent_observations=False)
    same(report['continuity'], continuity);same(evaluation['continuity'], continuity);same(evaluation['availability'], categories)
    assert count == report['frames'] == evaluation['frames']
    for arm in ARMS:
        assert report['arms'][arm] == dict(frames=count, available=len(stats[arm]['position_m']), first_failure=first[arm])
        for metric, values in stats[arm].items():
            same(evaluation['arms'][arm][metric], statistics(values), tolerance=1e-10 if 'orientation' in metric else 1e-12)
    for metric, values in bridged.items():
        assert len(values) == bridge_count
        same(evaluation['bridged_frame_errors'][metric], statistics(values), tolerance=1e-10 if 'orientation' in metric else 1e-12)
    return dict(stream=key, frames=count, availability=categories, bridge_spans=spans,
        bridged_frame_errors=evaluation['bridged_frame_errors'])


def run(result_sha256):
    receipts = {'launch.json': LAUNCH_SHA, 'result.json': result_sha256}
    verify_artifacts(OUTPUT, receipts)
    launch = read_json(OUTPUT, 'launch.json');verify(launch)
    result = read_json(OUTPUT, 'result.json');verify_artifacts(OUTPUT, result['output_sha256'])
    assert result['status'] == 'TEMPORAL_ANCHOR_PAIRED_DEVELOPMENT_REPLAY_COMPLETE'
    assert not (OUTPUT/'failure.json').exists()
    phase = read_json(OUTPUT, 'all_sensor_estimates_complete.json')
    assert phase['native_coordinates_parsed'] is False and phase['reports'] == result['reports']
    verify_artifacts(OUTPUT, phase['output_sha256'])
    assert set(result['reports']) == set(result['evaluations']) == {c+'_'+t for c in COHORTS for t in TRIALS}
    verified = []
    for cohort in COHORTS:
        for trial in TRIALS:
            row = verify_stream(cohort, trial, result);verified.append(row)
            print('STREAM_VERIFIED', json.dumps(row, allow_nan=False), flush=True)
    assert sum(row['frames'] for row in verified) == 7433
    verify_artifacts(OUTPUT, receipts | result['output_sha256'])
    print('ALL_7433_ROWS_BRIDGES_ERRORS_AND_OUTPUT_BINDINGS_VERIFIED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-sha256', required=True)
    run(parser.parse_args().result_sha256)
