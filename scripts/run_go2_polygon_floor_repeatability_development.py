"""Four fixed J/S repetitions using the existing readout runner and polygon map."""
import argparse
import hashlib
import json
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.projected_polygon_floor_coverage_development import initialize_mapping
from scripts import run_go2_polygon_floor_navigation_development as pilot

previous=pilot.previous
BASE=pilot.BASE
PLAN=Path('docs/go2_polygon_floor_repeatability_plan_2026-09-17.json')
ARMS=('jepa','supervised_rollout','supervised_rollout','jepa')
RAW_WRITE=previous.study.source.write
OUTPUT=None  # Bound by the existing runner for each exact mission output.


def root_name(number):
    return f'go2_polygon_floor_repeatability_{number:02d}_{ARMS[number-1]}_noise_2mm_native_layout01_4800_v1_attempt_001'


def sources():
    path='scripts/run_go2_polygon_floor_repeatability_development.py'
    return pilot.sources()|{path:hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def write(name,value):
    if name=='launch.json':
        value=value|dict(experiment='polygon_floor_repeatability_v1',
            actual_mapping_class='ProjectedPolygonFloorRoutingMap',projected_polygon_floor_coverage=True,
            controller_unchanged_from_repeatability_batch=False,
            controller_unchanged_from_polygon_floor_pilot=True,
            reference_root_path=str(BASE/pilot.ROOT),
            reference_root_name=pilot.ROOT,world_model_changed=False,
            models_are_same_frozen_readouts=True)
    bind(RAW_WRITE,OUTPUT=OUTPUT)(name,value)


def prepare():
    frozen=json.loads(pilot.PLAN.read_text())
    assert pilot.sources()==frozen['source_sha256']
    result=json.loads((BASE/pilot.ROOT/'polygon_floor_navigation_readout_v1.json').read_text())
    assert result['navigation']['round_trip'] and not result['pipeline_faults']
    assert not any((BASE/root_name(n)).exists() for n in range(1,5))
    models=json.loads(previous.PLAN.read_text())['models']
    plan=frozen|dict(schema='polygon_floor_repeatability_plan.v1',
        assignments=[[1,a] for a in ARMS],root_names=[root_name(n) for n in range(1,5)],
        models={a:models[a] for a in set(ARMS)},source_sha256=sources(),
        planned_native_assignments=4,reference_root=str(BASE/pilot.ROOT),
        intervention='fixed repetitions of unchanged polygon-floor controller with frozen JEPA/supervised readouts',
        fixed_before_first_execution=True,controller_changes_during_batch=False,
        model_changes_during_batch=False,native_jobs_run_sequentially=True,
        parallel_heavy_work_during_mission=False,pilot_excluded_from_batch_totals=True,
        retain_full_batch_depth_until_analysis=True,preserve_every_failure=True,
        stop_on_scientific_failure=False,no_extra_repetitions_to_obtain_success=True,
        unchanged=['model snapshots and fitted heads','six candidates','tracking','view planning',
            'polygon floor map','footprint filter','obstacle map','clearance and stopping checks',
            '2mm depth noise and ideal gyro','4800-tick budget','CPU group and planning deadlines'],
        limitations=['one exposed maze','two repetitions per model','one training seed',
            'no fresh-layout reliability or causal JEPA advantage established','ideal gyro and no hardware validation'])
    previous.previous.save(PLAN,plan)
    print('PREPARED four polygon-floor repetitions: JEPA, supervised, supervised, JEPA',flush=True)


def run(number):
    # Keep the established runner, model loading, deadlines, logging and evaluator.
    # Substitute only the explicit mapper initializer and launch annotations.
    original_mapper=previous.study.cohort.learned.initialize_mapping
    original_writer=previous.study.source.write
    previous.study.cohort.learned.initialize_mapping=initialize_mapping
    previous.study.source.write=write
    try:
        bind(previous.run,PLAN=PLAN,ARMS=ARMS,root_name=root_name,source_hashes=sources)(number)
    finally:
        previous.study.cohort.learned.initialize_mapping=original_mapper
        previous.study.source.write=original_writer


def evaluate(number):
    bind(previous.evaluate,PLAN=PLAN,ARMS=ARMS,root_name=root_name)(number)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--evaluate',action='store_true');parser.add_argument('--assignment',type=int,choices=range(1,5))
    args=parser.parse_args()
    if args.prepare and args.assignment is None and not args.evaluate:prepare()
    elif not args.prepare and args.assignment is not None:
        evaluate(args.assignment) if args.evaluate else run(args.assignment)
    else:raise ValueError('prepare or run/evaluate one fixed assignment')
