"""Compact complete-population readout of a bound joint-continuity experiment.

Does not rerun observers, raw audits, native acquisition or training. It retains
all 32 streams and reads scored rows to localize accepted allocation violations.
"""
import argparse
import hashlib
from pathlib import Path

from scripts import run_go2_joint_continuity_comparison_v1 as experiment
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_stress_cohort_development as stress
from scripts import navigation_artifact_root_development as custody

DOCUMENT = 'docs/go2_joint_continuity_comparison_scientific_readout_2026-09-07.json'


def violations(rows):
    import math
    counts = {a: dict(available=0, position_violations=0, orientation_violations=0,
        first_position_violation=None, first_orientation_violation=None) for a in base.ARMS}
    seen = 0
    for row in rows:
        base.require(row['frame'] == seen and seen < 443, 'complete ordered score rows')
        for a in base.ARMS:
            error = row['errors'][a]
            if error is None: continue
            c = counts[a]; c['available'] += 1
            for metric, threshold in (('position', .02), ('orientation', math.radians(2))):
                key = metric + ('_m' if metric == 'position' else '_rad')
                if error[key] > threshold:
                    c[metric + '_violations'] += 1
                    if c['first_' + metric + '_violation'] is None:
                        c['first_' + metric + '_violation'] = dict(frame=seen, error=error[key])
        seen += 1
    base.require(seen == 443, 'all failed/unavailable frames retained')
    return counts


def readout(expected_sha256):
    output = experiment.OUTPUT
    custody.verify_artifacts(output, {'result.json': expected_sha256})
    result = base.read(output, 'result.json')
    base.require(result['status'] == 'JOINT_CONTINUITY_DEVELOPMENT_COMPARISON_COMPLETE'
        and result['streams'] == 32 and result['frames_per_stream'] == 443
        and result['arm_meaning'] == experiment.ARM_MEANING, 'exact complete comparison')
    custody.verify_artifacts(output, result['artifact_sha256'])
    launch = base.read(output, 'launch.json')
    experiment.prior.source_check(launch['source_sha256'])
    phase = base.read(output, 'sensor_phase_complete.json')
    expected = {experiment.stem(t,s) for t in base.TRIALS for s in experiment.SCENARIOS}
    base.require(set(result['scores']) == set(phase['reports']) == expected, 'complete fixed readout population')
    streams = {}; grouped = {}
    for scenario in experiment.SCENARIOS:
        subset = []
        for trial in base.TRIALS:
            key = experiment.stem(trial, scenario)
            report = phase['reports'][key]['report']; scored = result['scores'][key]
            missing = violations(stress.rows(output, key + '_evaluation.jsonl'))
            for arm in base.ARMS:
                base.require(missing[arm]['available'] == report['arms'][arm]['available'], 'same available error population')
            streams[key] = dict(trial=trial, scenario=scenario, score=scored['score'],
                availability=report['arms'], first_failure=report['first_failure'],
                continuity=report['continuity'], rotation_checks=phase['reports'][key]['rotations'],
                exposure=report['exposure'], allocation_violations=missing,
                active_observer_timing=report['active_observer_timing'],
                baseline_inference_reused=scenario in experiment.REUSE,
                baseline_timing_historical=scenario in experiment.REUSE,
                numerical_score_independently_reconstructed=scored['independently_reconstructed'],
                representation=scored['representation'])
            subset.append(streams[key])
        grouped[scenario] = {arm: dict(
            complete_tapes=sum(s['availability'][arm]['available'] == 443 for s in subset),
            available_frames=sum(s['availability'][arm]['available'] for s in subset),
            total_frames=8*443,
            complete_pose_allocation_met_tapes=sum(s['score']['empirical_local_pose_allocation_met'][arm] for s in subset),
            maximum_available_position_m=max(s['score']['arms'][arm]['position_m']['maximum'] for s in subset),
            maximum_available_orientation_rad=max(s['score']['arms'][arm]['orientation_rad']['maximum'] for s in subset),
            accepted_position_violations=sum(s['allocation_violations'][arm]['position_violations'] for s in subset),
            accepted_orientation_violations=sum(s['allocation_violations'][arm]['orientation_violations'] for s in subset),
            onset_update_attempted_tapes=sum(s['exposure'][arm]['onset_update_attempted'] for s in subset),
            active_updates_above_100ms=sum(s['active_observer_timing'][arm]['over_100ms'] for s in subset))
            for arm in base.ARMS}
    criteria = experiment.continuation_criteria(output, phase, result['scores'])
    base.require(criteria == result['criteria'], 'declared continuation arithmetic changed')
    custody.verify_artifacts(output, {'result.json': expected_sha256} | result['artifact_sha256'])
    experiment.prior.source_check(launch['source_sha256'])
    return dict(status='JOINT_CONTINUITY_COMPARISON_SCIENTIFIC_READOUT_COMPLETE',
        experiment_result_sha256=expected_sha256, output_root=str(output),
        summary_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        artifact_sha256=result['artifact_sha256'], source_sha256=launch['source_sha256'],
        prior_posthoc_result_sha256=result['original_result_sha256'],
        criteria=criteria, scenarios=grouped, streams=streams, arm_meaning=experiment.ARM_MEANING,
        elapsed_seconds=result['elapsed_seconds'], artifact_bytes_before_result=result['artifact_bytes_before_result'],
        workload=base.read(output,'workload.json'), hardware=launch['hardware'],
        limitations=dict(development_exposed_tapes=True, scene_clusters=2,
            independent_layout_replication_established=False, original_attempt_remains_failed=True,
            strict_depth_failures_retained=3, intended_motion_coverage_tapes=0,
            gyro_bias_estimated=False, correlated_depth_error_corrected=False,
            observer_timing_excludes_acquisition_control_and_copying=True,
            physics_continued_during_observer_compute=False,
            inherited_raw_audit_reexecuted=False, native_data_independently_acquired=False,
            candidate_adopted=False, learned_model_selected_commands=False,
            navigation_qualified=False, goal_achieved=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-sha256',required=True)
    result=readout(parser.parse_args().result_sha256)
    target=experiment.prior.ROOT / DOCUMENT
    base.require(target.resolve()==target,'exact nonsymlink result document')
    data=base.encode(result)
    base.require(len(data)<=base.MAX_METADATA,'bounded summary document')
    with target.open('xb') as stream:stream.write(data)
    print(result['status'],hashlib.sha256(data).hexdigest(),DOCUMENT)
