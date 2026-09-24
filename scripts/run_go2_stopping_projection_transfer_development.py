"""Twelve fixed fresh-maze learned, fitted-motion and reactive assignments."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from multiprocessing import get_context
import os
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm import stopping_projection_transfer_layouts_development as layouts
from lewm.committed_camera_frontier_view_development import CommittedCameraFrontierRuntimeMixin
from lewm.process_registered_round_trip_development import registration_ready
from scripts import run_go2_planned_stopping_projection_development as stopping

cohort = stopping.cohort
combined = stopping.combined
CONDITIONS = ('learned', 'pose_command', 'reactive')
INVENTORY = Path('docs/go2_stopping_projection_transfer_layout_inventory_2026-09-15.json')
INVENTORY_SHA256 = 'ef34cccd5ce4a62bed742ee4a5eff02e54618c0ad1b81b6d0be787ebb089f06f'
ROOT = 'go2_stopping_projection_transfer_{condition}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


class FreshPhysicalInit(cohort.IndependentRoundTripPhysicalInit):
    __init__ = bind(cohort.IndependentRoundTripPhysicalInit.__init__,
        specification=layouts.specification, pack=layouts.pack)


class FreshCameraSession(cohort.LiveDepthNoiseMixin, cohort.CompactDepthRetentionMixin,
        cohort.LzmaRawDepthPairedCameraSession, FreshPhysicalInit):
    pass


class CurrentReactiveRuntime(CommittedCameraFrontierRuntimeMixin, cohort.reactive.MatchedHeadingReactiveRuntime):
    pass


def annotate_transfer(name, value):
    if name == 'launch.json':
        predictive = TREATMENT != 'reactive'
        value = value | dict(experiment='stopping_projection_three_controller_transfer_v1',
            comparison='learned_and_fitted_predictive_motion_vs_instantaneous_reactive',
            comparison_condition=TREATMENT, reference_root_name=None,
            planned_conditions=list(CONDITIONS), planned_layout_indices=[0,1,2,3],
            planned_layout_count=4, planned_native_assignments=12,
            fixed_dispatch_pairs=[[[0,'learned'],[1,'pose_command']],
                [[0,'pose_command'],[1,'reactive']],[[0,'reactive'],[1,'learned']],
                [[2,'reactive'],[3,'learned']],[[2,'learned'],[3,'reactive']],
                [[2,'pose_command'],[3,'pose_command']]],
            fixed_first_source_by_layout=None, fresh_layout_inventory=layouts.build_inventory(),
            frozen_layout_inventory_sha256=INVENTORY_SHA256,
            new_independent_development_layout=True, exposed_development_layout=False,
            layout_novelty_scope='distinct_from_explicit_72_layout_source_registry',
            tracker='LocalViewRevisitMotion', local_view_reference_bank=True,
            maximum_extra_local_view_references=8,
            recent_reference_refresh_from_accepted_anchor=True,
            maximum_recent_reference_age_ns=400_000_000, bridge_measurements_promoted=False,
            committed_camera_view_turn=True, approach_radius_rechecked_during_committed_view=False,
            aligned_actual_projection_and_fresh_map_required=True,
            planned_stopping_projection=predictive, actual_dispatch_guards_unchanged=True,
            floor_reacquisition_enabled=False,
            motion_prediction_source=TREATMENT if predictive else 'none',
            forecast_xy_source=TREATMENT if predictive else 'none',
            forecast_yaw_source=('learned' if TREATMENT=='learned' else 'command') if predictive else 'none',
            contact_score_mode='disabled' if predictive else 'not_used',
            reactive_recovery_rules_differ_from_predictive=True,
            predictive_clearance_and_stopping_projection_absent_in_reactive=True,
            shared_perception_and_frontier_views=True,
            isolated_predictive_scoring_effect_established=False,
            fully_model_free_controller=not predictive, final_evaluation=False,
            extra_sources=value['extra_sources'] | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__, 'lewm/stopping_projection_transfer_layouts_development.py',
                    'lewm/local_view_revisit_tracking_development.py',
                    'lewm/coherent_reference_refresh_development.py',
                    'lewm/recent_anchored_reference_refresh_development.py',
                    'lewm/committed_camera_frontier_view_development.py',
                    'lewm/planned_stopping_projection_development.py',
                    'scripts/run_go2_local_view_revisit_native_development.py',
                    'scripts/run_go2_planned_stopping_projection_development.py')})
    RAW_WRITE(name, value)


def annotate(name, value):
    emit = bind(annotate_transfer, TREATMENT=TREATMENT, RAW_WRITE=RAW_WRITE)
    if TREATMENT == 'reactive':
        bind(combined.camera.annotate_camera, RAW_WRITE=emit)(name, value)
    else:
        bind(combined.annotate, SOURCE=TREATMENT, RAW_WRITE=emit)(name, value)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--layout-index', type=int, choices=range(4), required=True)
    parser.add_argument('--condition', choices=CONDITIONS, required=True)
    args=parser.parse_args();i=args.layout_index;predictive=args.condition!='reactive'
    if sorted(os.sched_getaffinity(0)) != cohort.transfer.CPU_GROUPS[i%2]:
        raise ValueError('assigned layout CPU group required')
    if hashlib.sha256(INVENTORY.read_bytes()).hexdigest()!=INVENTORY_SHA256:
        raise ValueError('fixed fresh inventory required')
    output=cohort.stable.source.BASE/ROOT.format(index=i,condition=args.condition)
    cohort.stable.source.validate_root(output,must_exist=False)
    if output.exists():raise ValueError('preserve every fixed assignment')
    condition='supervised_rollout' if predictive else 'reactive'
    writer=bind(cohort.make_writer,annotate=bind(annotate,TREATMENT=args.condition))(output,i,condition)
    cohort.stable.floor.configure()
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),
            initializer=cohort.gyro.initialize_registration) as executor:
        assert executor.submit(registration_ready).result()

        def runtime(*runtime_args,**kwargs):
            if kwargs.get('condition')!=condition:raise ValueError('fixed model assignment required')
            extra=dict(registration_executor=executor,navigation_ticks=4800,arrival_radius_m=.02)
            if predictive:
                return stopping.StoppingAwareRuntime(*runtime_args,motion_prediction_source=args.condition,**extra,**kwargs)
            return CurrentReactiveRuntime(*runtime_args,**extra,**kwargs)

        bind(cohort.stable.source.main,OUTPUT=output,COUNT=4814,LAYOUT_INDEX=i,
            specification=layouts.specification,public_mission=layouts.public_mission,
            MODEL_ASSIGNMENT='seed_2026091001_full_supervised_rollout' if predictive else 'reactive',
            PacedNativeSession=partial(FreshCameraSession,noise_layout_index=i,noise_sigma_mm=2),
            write=writer,CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,
            POSE_INITIALIZER=stopping.previous.initialize_pose,MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=cohort.gyro.initialize_obstacles,OBSTACLE_READY=cohort.stable.obstacles_ready,
            initialize_mapping=cohort.learned.initialize_mapping)()


if __name__=='__main__':main()
