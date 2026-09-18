"""Fitted-motion control matched to the completed learned wall-clock mission."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_async_camera_wall_mission_development as learned

probe=learned.probe
ROOT='go2_async_camera_wall_mission_pose_command_layout01_4800_v1_attempt_001'


def fitted_runtime(*args,**kwargs):
    if kwargs.get('motion_prediction_source')!='learned':
        raise ValueError('expected original launcher motion-source binding')
    kwargs['motion_prediction_source']='pose_command'
    return probe.previous.WallDeadlineRuntime(*args,**kwargs)


def finish(name,value):
    if name=='launch.json':
        value=value|dict(experiment='asynchronous_wall_learned_fitted_comparison_v1',
            comparison='learned_corrected_xy_and_yaw_vs_fitted_pose_command_xy_and_integrated_yaw',
            comparison_condition='pose_command',motion_prediction_source='pose_command',
            forecast_xy_source='pose_command',forecast_yaw_source='command',
            learned_yaw_retained=False,neural_xy_used_for_scoring=False,neural_outcomes_used_for_scoring=False,
            reference_root_name=learned.ROOT,planned_conditions=['learned','pose_command'],
            planned_native_assignments=2,fixed_dispatch_pairs=None,
            predictive_planning_in_both_arms=True,fully_model_free_controller=False,
            both_native_missions_run_alone=True,
            extra_sources=value['extra_sources']|{__file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    RAW_WRITE(name,value)


def annotate(name,value):
    bind(learned.annotate,RAW_WRITE=bind(finish,RAW_WRITE=RAW_WRITE))(name,value)


def main():
    root=probe.cohort.stable.source.BASE/learned.ROOT
    result=json.loads((root/'continuous_native_arrival_evaluation.json').read_text())
    if not result['round_trip_arrival_checks_passed']:
        raise ValueError('complete learned reference evaluation before the fitted control')
    bind(probe.main,ROOT=ROOT,NAVIGATION_TICKS=4800,annotate=annotate,
        previous=SimpleNamespace(WallDeadlineRuntime=fitted_runtime))()


if __name__=='__main__':main()
