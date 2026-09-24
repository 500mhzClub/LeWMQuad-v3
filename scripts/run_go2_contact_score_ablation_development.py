"""Four fixed contact-score pilot assignments on development layouts 0 and 1."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.contact_score_ablation_development import ContactScoreAblationMixin, MODES
from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_pose_command_xy_ablation_development as xy

cohort = xy.cohort


class ContactScoreRuntime(ContactScoreAblationMixin, xy.XYSourceRuntime):
    pass


def annotate_contact(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='contact_score_ablation_pose_command_xy_pilot_v1',
            comparison='learned_contact_score_vs_disabled_with_pose_command_xy_and_learned_yaw',
            comparison_condition=MODE, contact_score_mode=MODE,
            planned_conditions=list(MODES), planned_layout_indices=[0,1],
            planned_layout_count=2, planned_native_assignments=4,
            learned_yaw_and_contact_retained=MODE=='learned', learned_yaw_retained=True,
            contact_predictions_used_for_scoring=MODE=='learned',
            both_contact_alternatives_computed_in_both_arms=True,
            zero_contact_score_is_not_a_contact_free_prediction=True,
            xy_yaw_and_physical_guards_unchanged=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/contact_score_ablation_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    emit=bind(annotate_contact, MODE=MODE, RAW_WRITE=RAW_WRITE)
    bind(xy.annotate, XY_SOURCE='pose_command', RAW_WRITE=emit)(name,value)


def make_writer(output, index, mode):
    return bind(cohort.make_writer, annotate=bind(annotate, MODE=mode))(
        output,index,'supervised_rollout')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layout-index',type=int,choices=(0,1),required=True)
    parser.add_argument('--contact-score',choices=MODES,required=True)
    args=parser.parse_args();i=args.layout_index
    if sorted(os.sched_getaffinity(0))!=cohort.transfer.CPU_GROUPS[i%2]:
        raise ValueError('assigned physical CPU group required')
    if hashlib.sha256(cohort.INVENTORY.read_bytes()).hexdigest()!=cohort.INVENTORY_SHA256:
        raise ValueError('fixed development inventory required')
    output=cohort.stable.source.BASE/f'go2_contact_score_ablation_pose_command_xy_{args.contact_score}_noise_2mm_native_layout{i:02d}_4800_v1_attempt_001'
    cohort.stable.source.validate_root(output,must_exist=False)
    if output.exists():raise ValueError('preserve every fixed pilot outcome')
    writer=make_writer(output,i,args.contact_score)
    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()
        def runtime(*runtime_args,**kwargs):
            if kwargs.get('condition')!='supervised_rollout':raise ValueError('fixed supervised reference required')
            return ContactScoreRuntime(*runtime_args,contact_score_mode=args.contact_score,
                forecast_xy_source='pose_command',registration_executor=executor,
                navigation_ticks=4800,arrival_radius_m=.02,**kwargs)
        bind(cohort.stable.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=i,
            specification=cohort.layouts.specification,public_mission=cohort.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(cohort.PostTrainingCameraSession,noise_layout_index=i,noise_sigma_mm=2),
            write=writer,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=cohort.training.initialize_gyro_coherent_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__=='__main__':main()
