"""Four prospective exposed-maze missions with fixed readout treatments."""
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
from lewm.pulse_timed_training_runner_development import state_digest
from scripts import fit_go2_frozen_motion_readout_development as readout
from scripts import run_go2_view_replan_repeatability_development as previous
from scripts import evaluate_go2_short_pulse_navigation_development as evaluation
from scripts.navigation_artifact_root_development import validate_root
from scripts.read_go2_interrupted_view_replan_development import physical_return_edges

BASE=previous.BASE
PLAN=Path('docs/go2_frozen_readout_navigation_plan_2026-09-17.json')
ARMS=('command_history','jepa','supervised_rollout','untrained')
native=previous.native
study=previous.study
Runtime=previous.previous.InterruptedViewRuntime


def root_name(number):
    return f'go2_frozen_readout_navigation_{number:02d}_{ARMS[number-1]}_noise_2mm_native_layout01_4800_v1_attempt_001'


def source_hashes():
    return previous.source_hashes()|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
        'scripts/run_go2_frozen_readout_navigation_development.py',
        'scripts/fit_go2_frozen_motion_readout_development.py','lewm/frozen_motion_readout_development.py',
        'lewm/command_history_residual_learning_development.py','scripts/command_history_residual_snapshot_development.py')}


def prepare():
    frozen=json.loads(previous.PLAN.read_text())
    assert previous.source_hashes()==frozen['source_sha256']
    assert not any((BASE/root_name(n)).exists() for n in range(1,5))
    fits=json.loads((readout.OUTPUT/'result.json').read_text());assert fits['status']=='COMPLETE'
    scored=json.loads((readout.OUTPUT/'prediction_evaluation/result.json').read_text())
    assert scored['total']['windows']==3726
    model_records={}
    for arm in ARMS:
        feature_arm='jepa' if arm=='command_history' else arm
        model=readout.load_readout(feature_arm);record=fits['records'][feature_arm]
        assert state_digest(model.state_dict())==record['model_sha256']
        model_records[arm]=record|dict(feature_arm=feature_arm)
    plan=dict(schema='frozen_readout_navigation_plan.v1',assignments=[[1,a] for a in ARMS],
        root_names=[root_name(n) for n in range(1,5)],models=model_records,
        source_sha256=source_hashes(),inventory_sha256=hashlib.sha256(native.INVENTORY.read_bytes()).hexdigest(),
        command_history_fit_sha256=frozen['command_history_fit_sha256'],
        readout_plan_sha256=hashlib.sha256(readout.PLAN.read_bytes()).hexdigest(),
        reference_controller_plan=str(previous.PLAN),actual_runtime_class=Runtime.__name__,
        planned_native_assignments=4,fixed_before_first_execution=True,
        controller_changes_during_batch=False,model_changes_during_batch=False,
        native_jobs_run_sequentially=True,parallel_heavy_work_during_mission=False,
        primary_outcome='physically verified goal-and-home round trip with no disallowed contact',
        secondary_outcomes=['physical backtracking','tracking loss','deadline performance','forecast error','recovery stalls'],
        navigation_ticks=4800,candidate_count=6,planning_extra_ns=20_000_000,
        command_reference_computes_unused_JEPA_readout=True,
        exposed_development_layout=True,new_independent_development_layout=False,
        sensor_noise_sigma_mm=2,gyro_noise_model='ideal',
        measured_simulation_not_real_time=True,final_evaluation=False,hardware_validated=False,
        preserve_every_failure=True,retain_full_batch_depth_until_analysis=True,
        stop_on_scientific_failure=False,no_extra_repetitions_to_obtain_success=True,
        limitations=['single exposed maze and one execution per arm','one training seed',
            'prediction improvement is not navigation improvement','no established JEPA advantage from prior scores',
            'asynchronous trajectories differ; results cannot isolate repeatable causal effects'])
    previous.save(PLAN,plan);print('PREPARED four fixed native readout comparisons',flush=True)


def evaluate(number):
    arm=ARMS[number-1];root=BASE/root_name(number)
    selected=SimpleNamespace(BASE=BASE,ROOT=root.name,PLAN=PLAN,ASSIGNMENTS=((1,arm),))
    result=bind(evaluation.evaluate,study=selected,
        xy=bind(evaluation.xy,validate_root=bind(validate_root,BASE=BASE)))(1)
    output=dict(schema='frozen_readout_navigation_readout.v1',batch_assignment=number,arm=arm,
        navigation=result,physical_backtracking=physical_return_edges(root),
        pipeline_faults=json.loads((root/'pipeline_faults.json').read_text()),
        exposed_layout=True,repeatability_or_causal_JEPA_advantage_established=False)
    previous.save(root/'frozen_readout_navigation_readout_v1.json',output)
    print('READOUT_MISSION_EVALUATED',number,arm,'round_trip',result['round_trip'],flush=True)


def run(number):
    arm=ARMS[number-1];output=BASE/root_name(number);plan=json.loads(PLAN.read_text())
    if output.exists() or source_hashes()!=plan['source_sha256']:raise ValueError('unused output and fixed source required')
    assert plan['assignments']==[[1,a] for a in ARMS]
    if number>1 and not (BASE/root_name(number-1)/'frozen_readout_navigation_readout_v1.json').exists():
        raise ValueError('evaluate preceding assignment before next mission')
    assert hashlib.sha256(native.INVENTORY.read_bytes()).hexdigest()==plan['inventory_sha256']
    assert hashlib.sha256(readout.PLAN.read_bytes()).hexdigest()==plan['readout_plan_sha256']
    assert sorted(os.sched_getaffinity(0))==study.cohort.transfer.CPU_GROUPS[1]
    if shutil.disk_usage(BASE).free<4*1024**3:raise ValueError('four GiB recording headroom required')
    model_record=plan['models'][arm];feature_arm=model_record['feature_arm']
    reference=json.loads((previous.REFERENCE/'launch.json').read_text());runtimes=[];hardware=native.baseline.hardware()

    def load_model(assignment):
        assert assignment==arm
        model=readout.load_readout(feature_arm)
        assert state_digest(model.state_dict())==model_record['model_sha256']
        # The runtime condition selects the rollout head. The untrained control
        # has the same architecture, but its representation received no updates.
        head_condition='supervised_rollout' if feature_arm=='supervised_rollout' else 'jepa'
        return model,head_condition,'full'

    def write(name,value):
        if name=='launch.json':
            value=reference|value|dict(experiment='frozen_readout_navigation_v1',
                study_arm=arm,comparison_condition=arm,model_assignment=arm,training_condition=feature_arm,
                neural_snapshot=dict(format='frozen_parent_plus_ridge_motion_readout',
                    filename=str(readout.OUTPUT/feature_arm/'readout.npz'),sha256=model_record['readout_sha256'],
                    parent_model_state_sha256=model_record['parent_model_sha256'],model_state_sha256=model_record['model_sha256']),
                model_feature_condition=feature_arm,untrained_features=feature_arm=='untrained',
                hardware=hardware,batch_assignment=number,fixed_sequential_assignments=plan['assignments'],
                prospective_plan=str(PLAN),prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                extra_sources=plan['source_sha256'],actual_runtime_class=Runtime.__name__,
                planned_native_assignments=4,planned_layout_count=1,
                output_base=str(BASE),model_input_base=str(readout.OUTPUT),
                neural_reference_is_unused_for_control=arm=='command_history',world_model_changed=True,
                frozen_feature_motion_readout=True,controller_unchanged_from_repeatability_batch=True,
                reference_root_path=str(previous.REFERENCE),exposed_development_layout=True,
                new_independent_development_layout=False,final_evaluation=False,hardware_validated=False)
        bind(study.source.write,OUTPUT=output)(name,value)
        if name=='requests.json' and runtimes:
            for filename,record in (('visual_dispatch_events.json',runtimes[0].visual_dispatch_events),
                    ('planning_latency_stress.json',runtimes[0].clock_ns.rows),('live_planning_profile.json',runtimes[0].plan_profile_rows)):
                bind(study.source.write,OUTPUT=output)(filename,record)

    study.cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=study.previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(native.baseline.registration_ready).result()
        def runtime(model,**kwargs):
            kwargs['clock_ns']=native.PlanningLatencyClock(kwargs['clock_ns'],20_000_000)
            result=Runtime(model,prediction_source='command_history' if arm=='command_history' else 'neural',
                registration_executor=executor,navigation_ticks=4800,arrival_radius_m=.02,**kwargs)
            runtimes.append(result);return result
        bind(study.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=1,
            validate_root=bind(validate_root,BASE=BASE),specification=native.layouts.specification,
            public_mission=native.layouts.public_mission,MODEL_ASSIGNMENT=arm,MODEL_LOADER=load_model,
            PacedNativeSession=partial(native.FreshCameraSession,noise_layout_index=1,noise_sigma_mm=2),
            write=write,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=partial(native.baseline.initialize_pose,str(output)),MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=study.previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=study.cohort.stable.obstacles_ready,initialize_mapping=study.cohort.learned.initialize_mapping)()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--evaluate',action='store_true');parser.add_argument('--assignment',type=int,choices=range(1,5));args=parser.parse_args()
    if args.prepare and args.assignment is None and not args.evaluate:prepare()
    elif not args.prepare and args.assignment is not None:
        evaluate(args.assignment) if args.evaluate else run(args.assignment)
    else:raise ValueError('prepare or run/evaluate one fixed assignment')
