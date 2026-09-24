"""One camera-viewpoint follow-up on the diagnosed layout-0 exploration stall."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_registered_round_trip_development import registration_ready
from lewm.camera_frontier_visits_development import CameraFrontierRuntimeMixin
from scripts import run_go2_yaw_source_ablation_development as yaw

cohort = yaw.cohort
ROOT_NAME = 'go2_camera_frontier_viewpoint_learned_yaw_noise_2mm_native_layout00_4800_v1_attempt_001'
REFERENCE = 'go2_yaw_source_ablation_learned_noise_2mm_native_layout00_4800_v1_attempt_001'


class CameraRuntime(CameraFrontierRuntimeMixin, yaw.YawRuntime):
    pass


def annotate_camera(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='camera_frontier_viewpoint_learned_yaw_v1',
            comparison='camera_viewpoint_exploration_followup',
            comparison_condition='camera_frontier_viewpoint', reference_root_name=REFERENCE,
            planned_conditions=['camera_frontier_viewpoint'], planned_layout_indices=[0],
            planned_layout_count=1, planned_native_assignments=1,
            camera_projection_selects_viewpoint=True,
            camera_projection_does_not_admit_unknown_floor=True,
            fresh_mapped_patch_required_for_observation_completion=True,
            unresolved_viewpoints_remembered=True,
            unreachable_viewpoint_search_exclusions_reset_by_map_growth=True,
            frontier_view_standoff_route_m=None, frontier_panorama=False,
            frontier_panorama_heading_stages=1, frontier_panorama_step_radians=None,
            nearby_panorama_directed_view=False, nearby_panorama_radius_m=None,
            frontier_exclusion_requires_observed_pose_arrival=False,
            frontier_exclusion_arrival_radius_m=None,
            arrival_panorama_view_start_position_recorded=False,
            recent_reference_refresh_from_accepted_anchor=False,
            tracker_identical_to_original_yaw_reference=True,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/camera_frontier_viewpoint_development.py',
                    'lewm/camera_frontier_visits_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    bind(yaw.annotate, YAW_SOURCE='learned',
        RAW_WRITE=bind(annotate_camera, RAW_WRITE=RAW_WRITE))(name, value)


def make_writer(output):
    return bind(cohort.make_writer, annotate=annotate)(output, 0, 'supervised_rollout')


def main():
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[0]:
        raise ValueError('layout-0 physical CPU group required')
    output = cohort.stable.source.BASE / ROOT_NAME
    cohort.stable.source.validate_root(output, must_exist=False)
    if output.exists(): raise ValueError('preserve this fixed follow-up and all predecessors')
    if hashlib.sha256(cohort.INVENTORY.read_bytes()).hexdigest() != cohort.INVENTORY_SHA256:
        raise ValueError('same development inventory required')
    writer = make_writer(output); cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*args, **kwargs):
            if kwargs.get('condition') != 'supervised_rollout':
                raise ValueError('same frozen supervised model required')
            return CameraRuntime(*args, forecast_yaw_source='learned',
                contact_score_mode='disabled', forecast_xy_source='pose_command',
                registration_executor=executor, navigation_ticks=4800, arrival_radius_m=.02, **kwargs)

        bind(cohort.stable.source.main, OUTPUT=output, COUNT=4814, LAYOUT_INDEX=0,
            specification=cohort.layouts.specification, public_mission=cohort.layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout',
            PacedNativeSession=partial(cohort.PostTrainingCameraSession, noise_layout_index=0, noise_sigma_mm=2),
            write=writer, CLOCK_MODE='measured_simulation', PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=cohort.training.initialize_gyro_coherent_pose,
            MEASURED_RUNTIME_CLASS=runtime, OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,
            OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__ == '__main__': main()
