"""Fourteen fixed missions comparing pulse-trained prediction and controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import shutil
from lewm.eligible_floor_registration_development import bind
from lewm import short_pulse_navigation_layouts_development as layouts
from lewm.short_pulse_navigation_runtime_development import PulsePredictiveRuntime,PulseInstantaneousRuntime,COMMAND_FIT
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_neural_rgb_transfer_development as previous
from scripts.nominal_motion_residual_snapshot_development import load_snapshot
from scripts.train_go2_short_pulse_residual_development import OUTPUT as FITS
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

cohort=previous.study.cohort
source=cohort.stable.source
BASE=source.BASE
ARMS=('jepa','direct','supervised_rollout','pose_command','command_history','instantaneous','reactive')
ASSIGNMENTS=tuple((0,a) for a in ARMS)+tuple((1,a) for a in reversed(ARMS))
ROOT='go2_short_pulse_navigation_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
INVENTORY=Path('docs/go2_short_pulse_navigation_layout_inventory_2026-09-16.json')
INVENTORY_SHA256='0a7f65cf540c7715adb887ee0c3c9e3ccc41733cd9dfb0a20a3f4dc11fe3766b'
PLAN=Path('docs/go2_short_pulse_navigation_plan_2026-09-16.json')


class FreshPhysicalInit(cohort.IndependentRoundTripPhysicalInit):
    __init__=bind(cohort.IndependentRoundTripPhysicalInit.__init__,specification=layouts.specification,pack=layouts.pack)


class FreshCameraSession(cohort.LiveDepthNoiseMixin,cohort.CompactDepthRetentionMixin,
        cohort.LzmaRawDepthPairedCameraSession,FreshPhysicalInit):
    pass


def model_condition(arm):
    return arm if arm in ('jepa','direct','supervised_rollout','reactive') else 'supervised_rollout'


def prepare_plan():
    if PLAN.exists():raise ValueError('preserve fixed prospective plan')
    models={c:json.loads((FITS/(c+'_fit.json')).read_text()) for c in ARMS[:3]}
    record=dict(assignments=[list(a) for a in ASSIGNMENTS],models=models,
        command_history_fit_sha256=hashlib.sha256((COMMAND_FIT/'command_only.npz').read_bytes()).hexdigest(),
        inventory_sha256=INVENTORY_SHA256,navigation_ticks=4800,training_seed=2026091001,
        observed_arrival_radius_m=.02,physical_arrival_radius_m=.04,
        main_comparison='raw nominal-composed neural forecasts, fitted pose-command motion, matched-data command-history motion, instantaneous ranking, reactive',
        final_evaluation=False,model_selection_from_new_maze_outcomes=False,
        fixed_before_first_navigation=True,external_neural_motion_correction=False)
    with PLAN.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    print('PROSPECTIVE_PLAN',len(ASSIGNMENTS),'assignments',flush=True)


def load_model(arm):
    condition=model_condition(arm);record=json.loads(PLAN.read_text())['models'][condition]
    trainer=load_snapshot(FITS,record['filename'],sha256=record['sha256'],
        expected_binding=record['binding'],expected_config=record['configuration'])
    trainer.model.eval()
    return trainer.model,condition,'full'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--assignment',type=int,choices=range(1,15));args=parser.parse_args()
    if args.prepare:
        if args.assignment is not None:raise ValueError('prepare or execute, separately')
        return prepare_plan()
    if args.assignment is None:raise ValueError('fixed assignment required')
    index,arm=ASSIGNMENTS[args.assignment-1];plan=json.loads(PLAN.read_text())
    if plan['assignments']!=[list(a) for a in ASSIGNMENTS]:raise ValueError('fixed assignment order required')
    if hashlib.sha256(INVENTORY.read_bytes()).hexdigest()!=INVENTORY_SHA256:
        raise ValueError('fixed new development geometries required')
    if hashlib.sha256((COMMAND_FIT/'command_only.npz').read_bytes()).hexdigest()!=plan['command_history_fit_sha256']:
        raise ValueError('fixed command-history predictor required')
    if args.assignment>1:
        i,a=ASSIGNMENTS[args.assignment-2]
        if not (BASE/ROOT.format(index=i,arm=a)/'continuous_native_arrival_evaluation.json').exists():
            raise ValueError('evaluate preceding completed assignment first')
    if sorted(os.sched_getaffinity(0))!=cohort.transfer.CPU_GROUPS[index]:
        raise ValueError('assigned CPU group required')
    if shutil.disk_usage(BASE).free<4*1024**3:raise ValueError('four GiB native recording headroom required')
    resources=hardware();condition=model_condition(arm);predictive=arm!='reactive'
    output=BASE/ROOT.format(index=index,arm=arm)
    record=None if not predictive else plan['models'][condition]

    def write(name,value):
        if name=='launch.json':
            value=value|dict(experiment='short_pulse_navigation_v1',study_arm=arm,
                comparison_condition=arm,model_assignment=arm,
                planned_conditions=list(ARMS),fixed_sequential_assignments=plan['assignments'],
                planned_layout_count=2,planned_native_assignments=14,
                navigation_tick_budget=4800,full_mission_implemented=True,
                fresh_layout_inventory=layouts.build_inventory(),frozen_layout_inventory_sha256=INVENTORY_SHA256,
                prospective_plan_sha256=hashlib.sha256(PLAN.read_bytes()).hexdigest(),
                new_independent_development_layout=True,layout_novelty_scope='distinct_from_explicit_82_layout_registry',
                training_seed=None if record is None else plan['training_seed'],training_condition=condition,
                neural_snapshot=None if record is None else dict(filename=record['filename'],sha256=record['sha256'],model_state_sha256=record['model_sha256']),
                neural_reference_is_unused_for_control=arm in ('pose_command','command_history'),
                external_neural_motion_correction=False,command_history_fit_sha256=plan['command_history_fit_sha256'],
                command_duration_ns=400_000_000,maximum_command_duration_ns=400_000_000,
                terminal_translation_pulses=predictive,terminal_translation_command_duration_ns=100_000_000,
                observed_arrival_radius_m=.02,physical_arrival_requirement_m=.04,
                nominal_footprint_radius_m=.45,stopping_allowance_s=.5,
                sensor_noise_sigma_mm=2,gyro_noise_model='ideal',camera_hardware_calibrated=False,
                persistent_routing_memory=True,shared_perception_and_view_recovery=True,
                predictive_clearance_and_stopping_guards=predictive,
                instantaneous_ranking_retains_predictive_guards=arm=='instantaneous',
                reactive_comparison_is_not_isolated_ranking_ablation=arm=='reactive',
                forecast_yaw_source=('command' if arm=='pose_command' else 'command_history' if arm=='command_history' else 'learned') if predictive else 'none',
                contact_score_mode='disabled' if predictive else 'not_used',hardware=resources,
                native_jobs_run_sequentially=True,final_evaluation=False,hardware_validated=False,
                extra_sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (__file__,
                    'lewm/short_pulse_navigation_runtime_development.py','lewm/short_pulse_navigation_layouts_development.py',
                    'lewm/nominal_motion_residual_learning_development.py','scripts/nominal_motion_residual_snapshot_development.py',
                    'lewm/instantaneous_waypoint_score_development.py','lewm/signed_veto_view_recovery_development.py',
                    'lewm/floor_reacquisition_development.py','lewm/planned_stopping_projection_development.py')})
        bind(source.write,OUTPUT=output)(name,value)

    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=previous.reference.previous.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(model,**kwargs):
            if kwargs['condition']!=condition:raise ValueError('assigned prediction head required')
            extra=dict(registration_executor=executor,navigation_ticks=4800,arrival_radius_m=.02)
            if arm=='reactive':return previous.reference.SignedReactiveRuntime(model,**extra,**kwargs)
            cls=PulseInstantaneousRuntime if arm=='instantaneous' else PulsePredictiveRuntime
            prediction_source=arm if arm in ('pose_command','command_history') else 'neural'
            return cls(model,prediction_source=prediction_source,**extra,**kwargs)

        bind(source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=index,specification=layouts.specification,
            public_mission=layouts.public_mission,MODEL_ASSIGNMENT=arm,MODEL_LOADER=load_model,
            PacedNativeSession=partial(FreshCameraSession,noise_layout_index=index,noise_sigma_mm=2),
            write=write,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=previous.reference.initialize_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=previous.reference.previous.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,initialize_mapping=cohort.learned.initialize_mapping)()


if __name__=='__main__':main()
