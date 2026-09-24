"""Summarize the completed post-hoc analysis without repeating raw audits.

Authenticates the exact derived result and all its artifacts, source bindings,
and original paired estimate streams. Reads no native arrays. Denominators,
strict visibility failures, partial availability and actual fault exposure stay
visible. Observer timing is conditioned on real update attempts separately.
"""
import hashlib
import itertools
from pathlib import Path

from scripts import read_go2_tracking_posthoc_raw_accuracy_v1 as analysis
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_stress_cohort_development as stress
from scripts import navigation_artifact_root_development as custody
from lewm.independent_tracking_numerical_verification_development import same
from scripts.verify_go2_temporal_anchor_continuity_v1 import statistics

RESULT_SHA256 = '99b0ed6c47dc0c974579bc145b9888e6e3edd64a69db633f929fd327ab526531'
DOCUMENT = 'docs/go2_tracking_posthoc_raw_accuracy_scientific_readout_2026-09-07.json'


def paired_rows(estimates, evaluated, stream):
    """Read the whole population; no cheap terminal no-op timing as active work."""
    timing = {a: [] for a in base.ARMS}
    changes = {k: [] for k in ('position_m', 'orientation_rad',
                               'incremental_position_m', 'incremental_orientation_rad')}
    frames = 0
    for row, scored in itertools.zip_longest(estimates, evaluated):
        base.require(row is not None and scored is not None and row['frame'] == scored['frame'] == frames,
                     'complete aligned summary streams required')
        for arm in base.ARMS:
            first = stream['arm_availability'][arm]['first_failure']
            attempted = first is None or frames <= first
            if 'observer_update_attempted' in row['arms'][arm]:
                base.require(row['arms'][arm]['observer_update_attempted'] is attempted,
                             'actual stress update exposure disagrees')
            if attempted:
                timing[arm].append(row['arms'][arm]['observer_wall_ms'])
        if scored['availability'] == 'both':
            for metric in changes:
                a = scored['errors']['original'][metric]
                b = scored['errors']['temporal_anchor'][metric]
                if a is not None and b is not None:
                    changes[metric].append(b - a)
        frames += 1
    base.require(frames == stream['score']['frames'], 'full summary population required')
    return dict(shared_support_candidate_minus_original={k: statistics(v) for k, v in changes.items()},
        actual_update_timing={a: dict(statistics_ms=statistics(v),
            above_100ms=sum(x > 100 for x in v), includes_first_failed_update=True,
            excludes_post_terminal_noops=True, excludes_acquisition_and_control=True) for a, v in timing.items()})


def summarize():
    output = analysis.OUTPUT
    custody.verify_artifacts(output, {'result.json': RESULT_SHA256})
    result = base.read(output, 'result.json')
    base.require(result['status'] == 'POSTHOC_RAW_ACCURACY_AND_NONIDENTITY_COMPLETE'
        and result['completed_raw_audits'] == 8 and result['completed_pose_streams'] == 96
        and result['predecessor_comparisons'] == 48, 'exact completed analysis required')
    custody.verify_artifacts(output, result['artifact_sha256'])
    launch = base.read(output, 'launch.json')
    analysis.source_check(launch['source_sha256'])
    custody.verify_artifacts(analysis.INPUT, launch['stream_sha256'] | launch['original_inner_bindings'])
    reports = {}; raw_failures = {}; timing = {}; contrasts = {}
    for trial in base.TRIALS:
        report = base.read(output, trial + '_result.json')
        raw = base.read(output, trial + '_raw_audit.json')
        base.require(report['status'] == 'POSTHOC_TAPE_RAW_AND_ACCURACY_COMPLETE'
            and set(report['streams']) == set(analysis.SCENARIOS), 'complete tape and stream population required')
        reports[trial] = report
        bad = []
        for depth, footprint in zip(raw['sensors']['depth_checks'], raw['footprints'], strict=True):
            same(depth['physical_visibility'], footprint['score']['original_strict_score'], tolerance=0.)
            if not (depth['within1mm'] and depth['physical_visibility']['passes_sampled_physical_visibility']):
                bad.append(dict(depth=depth, footprint=footprint))
        raw_failures[trial] = dict(frames=len(raw['sensors']['depth_checks']), strict_failure_frames=bad,
            all_stable_footprints_pass=report['stable_footprint_metric_pass'],
            first_physical_stop=report['first_physical_stop'],
            sensor_contact_reconstruction_exact=raw['sensors']['sensor_contact_reconstruction_exact'])
        timing[trial] = {}; contrasts[trial] = {}
        for scenario, stream in report['streams'].items():
            name = trial + '_estimates.jsonl' if scenario == 'base' else stress.stream_name(trial, scenario)
            evaluated = trial + '__' + scenario + '_evaluation.jsonl'
            extra = paired_rows(stress.rows(analysis.INPUT, name), stress.rows(output, evaluated), stream)
            timing[trial][scenario] = extra['actual_update_timing']
            contrasts[trial][scenario] = extra['shared_support_candidate_minus_original']
    scenarios = {}
    for scenario in analysis.SCENARIOS:
        population = [reports[t]['streams'][scenario] for t in base.TRIALS]
        scenarios[scenario] = {a: dict(
            complete_tapes=sum(s['arm_availability'][a]['available'] == s['score']['frames'] for s in population),
            available_frames=sum(s['arm_availability'][a]['available'] for s in population),
            total_frames=sum(s['score']['frames'] for s in population),
            empirical_pose_allocation_met_tapes=sum(s['score']['empirical_local_pose_allocation_met'][a] for s in population),
            conditional_max_position_m=max(s['score']['arms'][a]['position_m']['maximum'] for s in population),
            conditional_max_orientation_rad=max(s['score']['arms'][a]['orientation_rad']['maximum'] for s in population),
            onset_update_attempted_tapes=sum(s['stress_exposure']['exposure'][a]['onset_update_attempted']
                for s in population) if scenario != 'base' else None,
            updates_with_changed_packet=sum(s['stress_exposure']['exposure'][a]['updates_with_changed_packet']
                for s in population) if scenario != 'base' else None,
            total_bridge_frames=sum(s['continuity']['summary']['total_bridged_frames'] for s in population)
                if a == 'temporal_anchor' else None) for a in base.ARMS}
    interpretation = dict(
        original_attempt_remains_failed=True, complete_command_tapes_are_not_complete_motion_coverage=True,
        strict_depth_failures_retained=True, boundary_footprints_do_not_certify_failed_pixels=True,
        conditional_pose_errors_are_not_full_tape_success=True,
        scene_clusters=2, support_conditions=2, turn_directions=2,
        independent_layout_replication_established=False, matched_gyro_rotation_is_not_independent_heading=True,
        observer_timing_excludes_acquisition_and_control=True, online_physics_continued_during_compute=False,
        stress_mechanisms_are_not_hardware_calibration=True,
        original_sensor_admission_reused=True, raw_audit_algorithms_independently_reimplemented=False,
        independent_pose_arithmetic_used_private_normalized_view=True,
        native_data_independently_acquired=False, full_challenge_pass=False,
        learned_model_selected_commands=False, navigation_qualified=False, goal_achieved=False)
    custody.verify_artifacts(output, {'result.json': RESULT_SHA256} | result['artifact_sha256'])
    custody.verify_artifacts(analysis.INPUT, launch['stream_sha256'] | launch['original_inner_bindings'])
    analysis.source_check(launch['source_sha256'])
    own = Path(__file__).resolve()
    return dict(status='POSTHOC_TRACKING_SCIENTIFIC_READOUT_COMPLETE', result_sha256=RESULT_SHA256,
        result=result, summary_source_sha256=hashlib.sha256(own.read_bytes()).hexdigest(),
        original_definition_sha256=launch['original_definition_sha256'],
        source_sha256=launch['source_sha256'], original_stream_sha256=launch['stream_sha256'],
        diagnostic_sha256=launch['diagnostic_sha256'], original_inner_bindings=launch['original_inner_bindings'],
        original_outside_bindings=launch['original_outside_bindings'],
        workload=base.read(output, 'workload.json'), scenarios=scenarios,
        coverage={t: r['coverage'] for t, r in reports.items()},
        streams={t: r['streams'] for t, r in reports.items()},
        raw_sensor_visibility=raw_failures, active_update_timing=timing,
        shared_support_contrasts=contrasts, interpretation=interpretation)


if __name__ == '__main__':
    result = summarize()
    target = analysis.ROOT / DOCUMENT
    base.require(target.resolve() == target, 'exact nonsymlink summary path')
    data = base.encode(result)
    base.require(len(data) <= base.MAX_METADATA, 'bounded scientific document')
    with target.open('xb') as stream:
        stream.write(data)
    print(result['status'], DOCUMENT, hashlib.sha256(data).hexdigest())
