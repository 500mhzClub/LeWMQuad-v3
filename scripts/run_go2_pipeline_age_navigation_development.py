"""One exposed-maze mission testing an explicit 250-ms observation age bound."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts.run_go2_obstacle_grouping_navigation_development import study
from lewm.pipeline_age_dispatch_development import PipelineAgeRuntime
from scripts import evaluate_go2_short_pulse_navigation_development as evaluation

ROOT = 'go2_pipeline_age_250ms_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001'
REFERENCE = 'go2_obstacle_grouping_navigation_reference_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    output = study.BASE/ROOT
    if args.evaluate:
        selected = SimpleNamespace(**(vars(study) | {
            'ROOT': ROOT, 'ASSIGNMENTS': ((1, 'supervised_rollout'),)}))
        result = bind(evaluation.evaluate, study=selected)(1)
        requests = json.loads((output/'requests.json').read_text())
        passed = [r for r in requests if r['reason'] == 'CURRENT_NOMINAL_OBSTACLE_TEST_PASSED']
        if not passed or any(r.get('observation_age_limit_ns') != 250_000_000 for r in passed):
            raise ValueError('actual accepted dispatch must use the prospective age bound')
        for row in passed:
            age = row['now_ns']-row['observation_measured_ns']
            if not 0 <= age <= 250_000_000:
                raise ValueError('accepted observation exceeds the age bound')
            if any(row['requested_command'][:2]) and row['observation_age_allowance_s'] != age/1e9:
                raise ValueError('translation must charge actual age to stopping distance')
        treatment = dict(actual_dispatch_age_treatment_verified=True,
            accepted_requests=len(passed),
            accepted_requests_older_than_200ms=sum(r['observation_age_ns'] > 200_000_000 for r in passed),
            maximum_accepted_age_ns=max(r['observation_age_ns'] for r in passed),
            actual_translation_age_charged=True, stopping_bound_calibrated=False)
        (output/'actual_pipeline_age_treatment_v1.json').write_text(json.dumps(treatment, indent=2)+'\n')
        print(json.dumps(result | treatment, indent=2))
        return
    if not (study.BASE/REFERENCE/'short_pulse_navigation_evaluation_v1.json').is_file():
        raise ValueError('completed reference required')
    if output.exists():
        raise ValueError('preserve this prospective attempt')
    if sorted(os.sched_getaffinity(0)) != study.cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('original maze-1 CPU allocation required')
    if shutil.disk_usage(study.BASE).free < 4*1024**3:
        raise ValueError('four GiB recording headroom required')
    before = sys.getswitchinterval()
    if before != .005:
        raise ValueError('reference parent switch interval required')
    plan = json.loads(study.PLAN.read_text())
    model = plan['models']['supervised_rollout']

    def write(name, value):
        if name == 'launch.json':
            value = value | dict(experiment='pipeline_age_navigation_v1',
                study_arm='supervised_rollout', comparison_condition='observation_age_250ms',
                reference_root_name=REFERENCE, exposed_development_layout=True,
                parent_python_switch_interval_before_s=before,
                parent_python_switch_interval_s=sys.getswitchinterval(),
                spawned_worker_switch_interval_changed=False,
                independent_obstacle_observer='GyroConditionedAuxiliaryObstacles',
                neural_snapshot=dict(filename=model['filename'], sha256=model['sha256'],
                    model_state_sha256=model['model_sha256']), training_seed=plan['training_seed'],
                command_history_fit_sha256=plan['command_history_fit_sha256'],
                external_neural_motion_correction=False, contact_score_mode='disabled',
                sensor_noise_sigma_mm=2, gyro_noise_model='ideal',
                navigation_tick_budget=4800, observed_arrival_radius_m=.02,
                physical_arrival_requirement_m=.04, nominal_footprint_radius_m=.45,
                terminal_translation_pulses=True,
                terminal_translation_command_duration_ns=100_000_000,
                observation_age_limit_changed=True, maximum_observation_age_ns=250_000_000,
                reference_maximum_observation_age_ns=200_000_000, world_model_changed=False,
                age_charged_to_stopping_projection=True, other_dispatch_guards_changed=False,
                parallel_analysis_during_mission=False, final_evaluation=False,
                hardware_validated=False, extra_sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                    for p in (__file__, 'scripts/run_go2_short_pulse_navigation_development.py',
                        'lewm/short_pulse_navigation_runtime_development.py',
                        'lewm/pipeline_age_dispatch_development.py')})
        bind(study.source.write, OUTPUT=output)(name, value)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(model, **kwargs):
            return PipelineAgeRuntime(model, prediction_source='neural',
                registration_executor=executor, navigation_ticks=4800,
                arrival_radius_m=.02, **kwargs)

        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=1,
            specification=study.layouts.specification, public_mission=study.layouts.public_mission,
            MODEL_ASSIGNMENT='supervised_rollout', MODEL_LOADER=study.load_model,
            PacedNativeSession=partial(study.FreshCameraSession, noise_layout_index=1, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=study.previous.reference.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()
