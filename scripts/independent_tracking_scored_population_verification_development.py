"""Read-only verification of all eight base and 88 stress scoring streams.

This is a component of the challenge reader, not launch/keeper/predecessor
authentication. The caller supplies the authenticated result and protocol hashes.
Sensor transforms and raw physics audits reuse production algorithms; coverage,
pose errors and continuity are checked by separately implemented arithmetic.
No observers, native simulator, training, writer, or directory discovery runs.
"""
from lewm import independent_tracking_numerical_verification_development as numerical
from lewm.independent_tracking_continuity_verification_development import verify_continuity
from lewm.independent_tracking_challenge_development import TRIALS, specification
from lewm.independent_tracking_stress_development import SCENARIOS
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_stress_cohort_development as stress
from scripts.navigation_artifact_root_development import validate_root, verify_artifacts


def expected_names():
    """Exact production roster before result.json; no failed/optional population."""
    names = {'launch.json', 'collection_complete.json', 'sensor_phase_complete.json', stress.PHASE}
    names |= {t + suffix for t in TRIALS for suffix in (
        '_receipt.json', '_estimates.jsonl', '_evaluation.jsonl', '_audit.json',
        '_worker_request.json', '_worker_exit.json', '_worker_receipt.json')}
    names |= {stress.stream_name(t, s, e) for t in TRIALS for s in SCENARIOS for e in (False, True)}
    return names


def verify_scored_population(output, result_sha256, protocol_sha256):
    output = validate_root(output)
    verify_artifacts(output, {'result.json': result_sha256})
    saved = base.read(output, 'result.json')
    base.require(saved['status'] == 'EIGHT_TRIAL_BASE_AND_FIXED_STRESS_RAW_AUDIT_AND_SCORING_COMPLETE'
        and saved['protocol_sha256'] == protocol_sha256
        and saved['trials'] == list(TRIALS) and saved['scenarios'] == list(SCENARIOS)
        and set(saved['scores']) == set(TRIALS) and set(saved['stress_scores']) == set(TRIALS)
        and saved['resource_contract'] == stress.resource_contract()
        and saved['stress_arms_evaluated'] is True, 'complete fixed scoring population required')
    for key in ('stress_mechanism_validation_is_hardware_qualification',
                'predecessor_prefix_comparison_performed', 'independent_observations_verified',
                'full_challenge_pass', 'navigation_qualified', 'real_time_qualified', 'goal_achieved'):
        base.require(saved[key] is False, 'scoring result cannot grant qualification: ' + key)
    bindings = saved['output_sha256']
    base.require(type(bindings) is dict and set(bindings) == expected_names(),
        'exact complete pre-scoring-terminal artifact roster required')
    verify_artifacts(output, bindings)
    phase_bindings = {'collection_complete.json': saved['collection_sha256'],
        'sensor_phase_complete.json': saved['base_phase_sha256'], stress.PHASE: saved['stress_phase_sha256']}
    base.require(all(bindings[name] == sha for name, sha in phase_bindings.items()),
        'result and artifact phase bindings differ')

    # This gate reconstructs every sensor intervention/exposure before the first
    # native-coordinate auditor is imported or called. It does not rerun models.
    episodes, phase, treated = stress.admit_complete_sensor_phase(output,
        saved['collection_sha256'], saved['base_phase_sha256'], saved['stress_phase_sha256'])
    continuity = {}
    for trial in TRIALS:
        reports = {'base': phase['reports'][trial]} | treated['reports'][trial]
        continuity[trial] = {arm: verify_continuity(stress.rows(output, report['estimates_file']),
            report['continuity']) for arm, report in reports.items()}
        base.require(set(saved['stress_scores'][trial]) == set(SCENARIOS),
            'all eleven stress scores required')

    from scripts import independent_tracking_evaluation_development as scoring
    verified = {}; audits = {}
    for trial in TRIALS:
        episode = episodes[trial]['result']
        stored = base.read(output, trial + '_audit.json')
        # One expensive native/sensor audit per trial, not one per stress stream.
        raw, actual = scoring._raw_audit(output, trial, episode, protocol_sha256)
        base.require(base.encode(stored['raw_audit']) == base.encode(actual),
            'saved raw audit differs from reexecuted audit')
        coverage = numerical.verify_coverage(raw, episode, stored['raw_audit']['coverage'],
            direction=specification(trial)['direction'])
        base.require(stored['base_pose_score'] == saved['scores'][trial]
            and stored['stress_pose_scores'] == saved['stress_scores'][trial],
            'per-trial and complete-result score copies differ')
        poses = {}
        reports = {'base': phase['reports'][trial]} | treated['reports'][trial]
        for scenario, report in reports.items():
            name = trial + '_evaluation.jsonl' if scenario == 'base' else stress.stream_name(trial, scenario, True)
            score = saved['scores'][trial] if scenario == 'base' else saved['stress_scores'][trial][scenario]
            poses[scenario] = numerical.verify_pose_stream(raw,
                stress.rows(output, report['estimates_file']), stress.rows(output, name), report, score)
        verified[trial] = dict(coverage=coverage, pose_streams=poses, continuity=continuity[trial],
            # Preserve actual exposure: recording an onset is not proof a failed
            # arm received it, nor is an unexercised stress a successful test.
            stress_exposure={s: {k: treated['reports'][trial][s][k] for k in (
                'onset_recorded', 'complete_requested_tape', 'changed_packet_frames', 'exposure')}
                for s in SCENARIOS})
        audits[trial] = actual
        del raw
    claims = dict(
        all_intended_motion_covered=all(a['coverage']['intended_motion_covered'] for a in audits.values()),
        all_candidate_base_pose_allocations_met=all(saved['scores'][t][
            'empirical_local_pose_allocation_met']['temporal_anchor'] for t in TRIALS),
        strict_depth_visibility_pass=all(bool(a['sensors']['depth_checks']) and all(
            x['within1mm'] and x['physical_visibility']['passes_sampled_physical_visibility']
            for x in a['sensors']['depth_checks']) for a in audits.values()))
    for key, value in claims.items():
        base.require(saved[key] is value, 'aggregate claim differs from complete population: ' + key)
    # Reauthenticate raw bytes and all scored outputs after inspection without
    # decoding every RGB packet a second time. No directory walk is needed.
    base.verify_collection(output, saved['collection_sha256'])
    verify_artifacts(output, bindings | {'result.json': result_sha256})
    return dict(status='COMPLETE_SCORED_POPULATION_ACCOUNTING_VERIFIED',
        result_sha256=result_sha256, protocol_sha256=protocol_sha256,
        trial_count=len(TRIALS), pose_stream_count=len(TRIALS) * (1 + len(SCENARIOS)),
        reports=verified, verified_aggregate_claims=claims,
        complete_sensor_phase_admitted_before_native_arrays=True,
        independent_numerical_coverage_pose_and_continuity_checks=True,
        sensor_transform_and_raw_audit_algorithms_independent=False,
        observer_inference_recomputed=False, source_and_launch_authority_verified=False,
        outside_keeper_terminal_verified=False, predecessor_comparison_verified=False,
        incremental_rotation_witness_saved=False, shared_gyro_bias_correction_established=False,
        independent_scene_count_established=False, full_challenge_pass=False,
        navigation_qualified=False, real_time_qualified=False, goal_achieved=False)
