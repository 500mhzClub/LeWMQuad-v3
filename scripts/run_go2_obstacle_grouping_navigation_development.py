"""Two exposed-maze missions isolating occupied-cell grouping implementation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.packed_cell_obstacles_development import PackedCellObstacles
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_short_pulse_navigation_development as study
from scripts import evaluate_go2_short_pulse_navigation_development as evaluation

CONDITIONS = ('reference', 'packed')
ROOT = 'go2_obstacle_grouping_navigation_{condition}_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001'
SOURCE = Path('lewm/packed_cell_obstacles_development.py')
EQUIVALENCE = study.BASE/'go2_packed_cell_obstacles_full_direct_equivalence_v1_attempt_001/result.json'


def initialize_packed_obstacles():
    from lewm import independent_depth_process_development as process
    study.previous.reference.previous.initialize_obstacles()
    process._observer = PackedCellObstacles()


def evaluate(condition):
    root = study.BASE/ROOT.format(condition=condition)
    launch = json.loads((root/'launch.json').read_text())
    if launch['observer_grouping_condition'] != condition:
        raise ValueError('recorded observer treatment must match requested condition')
    selected = SimpleNamespace(**(vars(study) | {
        'ROOT': root.name, 'ASSIGNMENTS': ((1, 'supervised_rollout'),)}))
    result = bind(evaluation.evaluate, study=selected)(1)
    return dict(condition=condition, source_short_pulse_assignment=12, **result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    if args.evaluate:
        print(json.dumps(evaluate(args.condition), indent=2))
        return
    last = study.BASE/study.ROOT.format(index=1, arm='jepa')
    if not (last/'short_pulse_navigation_evaluation_v1.json').exists():
        raise ValueError('finish the fourteen fixed missions before this timing experiment')
    if args.condition == 'packed':
        previous = study.BASE/ROOT.format(condition='reference')
        if not (previous/'short_pulse_navigation_evaluation_v1.json').exists():
            raise ValueError('finish and evaluate the reference mission first')
    evidence = json.loads(EQUIVALENCE.read_text())
    if (evidence['status'] != 'COMPLETE' or evidence['frames'] != 2532
            or hashlib.sha256(SOURCE.read_bytes()).hexdigest() != evidence['source_sha256']):
        raise ValueError('use the recorded equivalent observer implementation')
    if sorted(os.sched_getaffinity(0)) != study.cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('use the original maze-1 CPU allocation')
    if shutil.disk_usage(study.BASE).free < 4*1024**3:
        raise ValueError('four GiB recording headroom required')
    output = study.BASE/ROOT.format(condition=args.condition)
    plan = json.loads(study.PLAN.read_text())
    model = plan['models']['supervised_rollout']
    initializer = (study.previous.reference.previous.initialize_obstacles
        if args.condition == 'reference' else initialize_packed_obstacles)

    def write(name, value):
        if name == 'launch.json':
            value = value | dict(experiment='obstacle_grouping_navigation_v1',
                study_arm='supervised_rollout', observer_grouping_condition=args.condition,
                independent_obstacle_observer=('GyroConditionedAuxiliaryObstacles'
                    if args.condition == 'reference' else 'PackedCellObstacles'),
                comparison_condition=args.condition, planned_conditions=list(CONDITIONS),
                source_short_pulse_assignment=12, planned_layout_count=1,
                planned_native_assignments=2, new_independent_development_layout=False,
                exposed_development_layout=True, fixed_condition_order=list(CONDITIONS),
                neural_snapshot=dict(filename=model['filename'], sha256=model['sha256'],
                    model_state_sha256=model['model_sha256']), training_seed=plan['training_seed'],
                command_history_fit_sha256=plan['command_history_fit_sha256'],
                external_neural_motion_correction=False, contact_score_mode='disabled',
                sensor_noise_sigma_mm=2, gyro_noise_model='ideal',
                navigation_tick_budget=4800, observed_arrival_radius_m=.02,
                physical_arrival_requirement_m=.04, nominal_footprint_radius_m=.45,
                terminal_translation_pulses=True, terminal_translation_command_duration_ns=100_000_000,
                grouping_changes_occupied_cells=False, observation_age_limit_changed=False,
                model_and_controller_changed=False, parallel_analysis_during_mission=False,
                final_evaluation=False, hardware_validated=False,
                extra_sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                    __file__, str(SOURCE), 'scripts/run_go2_short_pulse_navigation_development.py',
                    'lewm/short_pulse_navigation_runtime_development.py')})
        bind(study.source.write, OUTPUT=output)(name, value)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(model, **kwargs):
            return study.PulsePredictiveRuntime(model, prediction_source='neural',
                registration_executor=executor, navigation_ticks=4800,
                arrival_radius_m=.02, **kwargs)

        bind(study.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=1,
            specification=study.layouts.specification, public_mission=study.layouts.public_mission,
            MODEL_ASSIGNMENT='supervised_rollout', MODEL_LOADER=study.load_model,
            PacedNativeSession=partial(study.FreshCameraSession, noise_layout_index=1, noise_sigma_mm=2),
            write=write, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=study.previous.reference.initialize_pose, MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=initializer, OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__ == '__main__':
    main()
