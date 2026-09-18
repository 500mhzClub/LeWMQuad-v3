"""Two exposed learned-motion follow-ups with the same committed viewing change."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from scripts.run_go2_committed_camera_view_development import CommittedViewRuntime
from scripts import run_go2_combined_perception_motion_development as combined

cohort=combined.cohort
ROOT='go2_committed_camera_view_learned_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'
REFERENCE='go2_combined_perception_motion_learned_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


def annotate_followup(name,value):
    if name=='launch.json':
        value=value|dict(experiment='committed_camera_view_learned_followups_v1',
            comparison='complete_started_view_turn_before_reassessing_camera_evidence',
            comparison_condition='committed_view',reference_root_name=REFERENCE.format(index=INDEX),
            planned_conditions=['committed_view'],planned_layout_indices=[2,3],
            planned_layout_count=2,planned_native_assignments=2,
            fixed_first_source_by_layout=None,new_independent_development_layout=False,
            exposed_development_layout=True,
            layout_novelty_scope='previously_exposed_combined_comparison_layouts_2_3',
            committed_camera_view_turn=True,
            approach_radius_rechecked_during_committed_view=False,
            aligned_actual_projection_and_fresh_map_required=True,
            original_failed_assignment_retained=True,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'lewm/committed_camera_frontier_view_development.py',
                    'scripts/run_go2_committed_camera_view_development.py')})
    RAW_WRITE(name,value)


def annotate(name,value):
    emit=bind(annotate_followup,INDEX=INDEX,RAW_WRITE=RAW_WRITE)
    bind(combined.annotate,SOURCE='learned',RAW_WRITE=emit)(name,value)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layout-index',type=int,choices=(2,3),required=True)
    index=parser.parse_args().layout_index
    if sorted(os.sched_getaffinity(0))!=cohort.transfer.CPU_GROUPS[index%2]:
        raise ValueError('original layout CPU group required')
    finished=cohort.stable.source.BASE/'go2_combined_perception_motion_four_layout_summary_v1_attempt_001/result.json'
    if not finished.is_file(): raise ValueError('finish the fixed eight assignments first')
    output=cohort.stable.source.BASE/ROOT.format(index=index)
    cohort.stable.source.validate_root(output,must_exist=False)
    if output.exists(): raise ValueError('preserve the single follow-up outcome')
    if hashlib.sha256(combined.INVENTORY.read_bytes()).hexdigest()!=combined.INVENTORY_SHA256:
        raise ValueError('same layout specification required')
    writer=bind(cohort.make_writer,annotate=bind(annotate,INDEX=index))(output,index,'supervised_rollout')
    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args,**kwargs):
            if kwargs.get('condition')!='supervised_rollout': raise ValueError('same frozen model required')
            return CommittedViewRuntime(*args,motion_prediction_source='learned',
                registration_executor=executor,navigation_ticks=4800,arrival_radius_m=.02,**kwargs)

        bind(cohort.stable.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=index,
            specification=combined.layouts.specification,public_mission=combined.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(combined.FreshCameraSession,noise_layout_index=index,noise_sigma_mm=2),
            write=writer,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=combined.initialize_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__=='__main__': main()
