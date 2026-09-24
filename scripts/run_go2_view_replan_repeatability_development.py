"""Four fixed repetitions of the current controller, with frozen J/S models."""
import argparse
from collections import Counter
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
from scripts import run_go2_interrupted_view_replan_development as previous
from scripts import run_go2_publication_repaired_navigation_replication_development as native
from scripts import evaluate_go2_short_pulse_navigation_development as evaluation
from scripts.read_go2_interrupted_view_replan_development import physical_return_edges
from scripts.navigation_artifact_root_development import validate_root

BASE=previous.BASE
PLAN=Path('docs/go2_view_replan_repeatability_plan_2026-09-17.json')
ARMS=('jepa','supervised_rollout','supervised_rollout','jepa')
REFERENCE=BASE/previous.ROOT
study=native.study


def root_name(number):
    return f'go2_view_replan_repeatability_{number:02d}_{ARMS[number-1]}_noise_2mm_native_layout01_4800_v1_attempt_001'


def source_hashes():
    return previous.source_hashes() | {__file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def save(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2);stream.write('\n')


def prepare():
    frozen=json.loads(previous.PLAN.read_text())
    if previous.source_hashes()!=frozen['source_sha256']:
        raise ValueError('preserve completed controller sources')
    if any((BASE/root_name(n)).exists() for n in range(1,5)):
        raise ValueError('preserve every assigned attempt')
    plan=dict(schema='view_replan_repeatability_plan.v1',
        assignments=[[1,arm] for arm in ARMS],root_names=[root_name(n) for n in range(1,5)],
        models=frozen['models'],training_seed=frozen['training_seed'],
        command_history_fit_sha256=frozen['command_history_fit_sha256'],
        source_sha256=source_hashes(),reference_root=str(REFERENCE),output_base=str(BASE),
        inventory_sha256=hashlib.sha256(native.INVENTORY.read_bytes()).hexdigest(),
        planned_native_assignments=4,fixed_before_first_execution=True,
        controller_changes_during_batch=False,model_selection_from_outcomes=False,
        native_jobs_run_sequentially=True,parallel_analysis_during_mission=False,
        primary_outcome='physically verified goal-and-home round trip with no disallowed contact',
        secondary_outcomes=['goal arrival','physical backtracking','tracking loss',
            'planning deadlines','view interruption events','executed-window forecast error'],
        navigation_ticks=4800,candidate_count=6,planning_extra_ns=20_000_000,
        exposed_development_layout=True,new_independent_development_layout=False,
        preserve_every_failure=True,retain_full_batch_depth_until_analysis=True,
        stop_on_scientific_failure=False,no_extra_repetitions_to_obtain_success=True,
        sensor_noise_sigma_mm=2,gyro_noise_model='ideal',
        measured_simulation_not_real_time=True,final_evaluation=False,hardware_validated=False,
        limitations=['one exposed maze and one training seed',
            'two executions per model cannot establish reliable generalization',
            'asynchronous trajectories differ; intervention benefit is not isolated'])
    save(PLAN,plan)
    print('PREPARED four fixed repetitions: JEPA, supervised, supervised, JEPA',flush=True)


def evaluate(number):
    arm=ARMS[number-1];root=BASE/root_name(number)
    selected=SimpleNamespace(BASE=BASE,ROOT=root.name,PLAN=PLAN,ASSIGNMENTS=((1,arm),))
    result=bind(evaluation.evaluate,study=selected,
        xy=bind(evaluation.xy,validate_root=bind(validate_root,BASE=BASE)))(1)
    plans=[p for p in json.loads((root/'planning.json').read_text()) if 'selection' in p]
    frontier=json.loads((root/'frontier_visits.json').read_text())
    coverage={}
    for p in plans:
        for event in p['selection'].get('coverage_view_request',{}).get('recent_events',[]):
            coverage[(event['started_ns'],event['completed_ns'])]=event
    events=frontier['events']+list(coverage.values())
    output=dict(schema='view_replan_repeatability_readout.v1',batch_assignment=number,arm=arm,
        navigation=result,physical_backtracking=physical_return_edges(root),
        pipeline_faults=json.loads((root/'pipeline_faults.json').read_text()),
        actions=dict(Counter(p['action'] for p in plans)),
        interrupted_view_events=[e for e in events
            if e['completion_reason']=='WEAK_VISUAL_SUPPORT_INTERRUPTED_VIEW'],
        coverage_rejections=sum(p['selection'].get('translation_footprint_coverage',{}).get('rejected',False) for p in plans),
        observed_coverage_patches=sum(e['unknown_cell_observed'] for e in coverage.values()),
        frontier_event_reasons=dict(Counter(e['completion_reason'] for e in frontier['events'])),
        exposed_layout=True,causal_intervention_benefit_established=False)
    save(root/'view_replan_repeatability_readout_v1.json',output)
    print('BATCH_ASSIGNMENT_EVALUATED',number,arm,'round_trip',result['round_trip'],
        'interrupted_views',len(output['interrupted_view_events']),flush=True)


def run(number):
    arm=ARMS[number-1];output=BASE/root_name(number);plan=json.loads(PLAN.read_text())
    if source_hashes()!=plan['source_sha256'] or output.exists():
        raise ValueError('fixed sources and unused output required')
    if plan['assignments']!=[[1,a] for a in ARMS]:
        raise ValueError('retain fixed assignment order')
    if number>1 and not (BASE/root_name(number-1)/'view_replan_repeatability_readout_v1.json').exists():
        raise ValueError('evaluate preceding assignment first')
    if hashlib.sha256(native.INVENTORY.read_bytes()).hexdigest()!=plan['inventory_sha256']:
        raise ValueError('same layout required')
    if sorted(os.sched_getaffinity(0))!=study.cohort.transfer.CPU_GROUPS[1]:
        raise ValueError('same CPU group required')
    if shutil.disk_usage(BASE).free<4*1024**3:
        raise ValueError('four GiB recording headroom required')
    reference=json.loads((REFERENCE/'launch.json').read_text())
    model=plan['models'][arm];resources=native.baseline.hardware();runtimes=[]

    def write(name,value):
        if name=='launch.json':
            value=reference | value | dict(experiment='view_replan_repeatability_v1',
                study_arm=arm,comparison_condition=arm,model_assignment=arm,training_condition=arm,
                neural_snapshot=dict(filename=model['filename'],sha256=model['sha256'],model_state_sha256=model['model_sha256']),
                hardware=resources,reference_root_name=previous.ROOT,reference_root_path=str(REFERENCE),
                batch_assignment=number,fixed_sequential_assignments=plan['assignments'],
                planned_native_assignments=4,planned_layout_count=1,
                prospective_plan=str(PLAN),prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                extra_sources=plan['source_sha256'],actual_runtime_class=previous.InterruptedViewRuntime.__name__,
                output_base=str(BASE),model_input_base=str(study.BASE),
                neural_reference_is_unused_for_control=False,world_model_changed=False,
                exposed_development_layout=True,new_independent_development_layout=False,
                final_evaluation=False,hardware_validated=False)
        bind(study.source.write,OUTPUT=output)(name,value)
        if name=='requests.json' and runtimes:
            for filename,record in (
                    ('visual_dispatch_events.json',runtimes[0].visual_dispatch_events),
                    ('planning_latency_stress.json',runtimes[0].clock_ns.rows),
                    ('live_planning_profile.json',runtimes[0].plan_profile_rows)):
                bind(study.source.write,OUTPUT=output)(filename,record)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(native.baseline.registration_ready).result()

        def runtime(model,**kwargs):
            kwargs['clock_ns']=native.PlanningLatencyClock(kwargs['clock_ns'],20_000_000)
            result=previous.InterruptedViewRuntime(model,prediction_source='neural',
                registration_executor=executor,navigation_ticks=4800,arrival_radius_m=.02,**kwargs)
            runtimes.append(result)
            return result

        bind(study.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=1,
            validate_root=bind(validate_root,BASE=BASE),
            specification=native.layouts.specification,public_mission=native.layouts.public_mission,
            MODEL_ASSIGNMENT=arm,MODEL_LOADER=bind(study.load_model,PLAN=PLAN),
            PacedNativeSession=partial(native.FreshCameraSession,noise_layout_index=1,noise_sigma_mm=2),
            write=write,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(native.baseline.initialize_pose,str(output)),MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,
            initialize_mapping=study.cohort.learned.initialize_mapping)()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--evaluate',action='store_true');parser.add_argument('--assignment',type=int,choices=range(1,5))
    args=parser.parse_args()
    if args.prepare:
        if args.assignment is not None or args.evaluate:raise ValueError('prepare separately')
        return prepare()
    if args.assignment is None:raise ValueError('fixed assignment required')
    return evaluate(args.assignment) if args.evaluate else run(args.assignment)


if __name__=='__main__':main()
