"""Full wall-clock recovery test after the recorded reactive floor rejection."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from lewm.floor_reacquisition_development import FloorReacquisitionRuntimeMixin,initialize_registration
from scripts import run_go2_async_wall_reactive_control_development as original

ROOT='go2_wall_reactive_floor_reacquisition_layout01_4800_v1_attempt_001'


class ReacquiringWallReactiveRuntime(FloorReacquisitionRuntimeMixin,original.WallReactiveRuntime):
    pass


def finish(name,value):
    if name=='launch.json':
        value=value|dict(experiment='wall_reactive_floor_reacquisition_followup_v1',
            reference_root_name=original.ROOT,prior_attempt_preserved=original.ROOT,
            diagnostic_followup_not_replacement=True,
            comparison='hold_and_reacquire_after_partial_floor_conflict',
            comparison_condition='reactive',planned_conditions=['reactive'],planned_native_assignments=1,
            actual_runtime_class='ReacquiringWallReactiveRuntime',
            registration='ReacquiringFloorRegistration',floor_reacquisition_enabled=True,
            temporary_floor_conflict_is_missing_observation=True,
            consecutive_accepted_poses_before_resuming_planning=4,
            rejected_floor_pose_used_by_map_or_mission=False,
            floor_rejection_resets_arrival_dwell=True,floor_rejection_cancels_pending_commands=True,
            global_budget_continues_during_missingness=True,
            floor_candidate_and_acceptance_thresholds_unchanged=True,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'lewm/floor_reacquisition_development.py')})
    RAW_WRITE(name,value)


def annotate(name,value):
    bind(original.annotate,RAW_WRITE=bind(finish,RAW_WRITE=RAW_WRITE))(name,value)


def main():
    base=original.cohort.stable.source.BASE
    diagnostic=base/original.ROOT/'gyro_coherent_floor_local_view_revisit_replay_v1'
    replay=json.loads((diagnostic/'result.json').read_text())
    check=json.loads((diagnostic/'reacquisition_terminal_frame_check_v1.json').read_text())
    if (replay['matching_recorded_raw_poses']!=replay['recorded_raw_poses']
            or replay['failure']['frame']!=check['frame'] or not check['anchor_and_reference_unchanged']):
        raise ValueError('reproduce the original conflict before the recovery follow-up')
    gyro=SimpleNamespace(**(vars(original.cohort.gyro)|dict(initialize_registration=initialize_registration)))
    cohort=SimpleNamespace(**(vars(original.cohort)|dict(gyro=gyro)))
    bind(original.main,ROOT=ROOT,annotate=annotate,cohort=cohort,
        WallReactiveRuntime=ReacquiringWallReactiveRuntime)()


if __name__=='__main__':main()
