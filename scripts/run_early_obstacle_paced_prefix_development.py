"""Paced early current-plane obstacle check with vetoed windows latched to zero."""
import hashlib
from pathlib import Path

from lewm.eligible_floor_registration_development import bind
from lewm.process_mapped_runtime_development import EarlyObstacleRuntime
from scripts import run_feature_budget300_paced_prefix_development as source

OUTPUT=source.source.source.BASE/'go2_early_obstacle_paced_recorded_prefix_v1_attempt_001'
_write=bind(source.source.write,OUTPUT=OUTPUT)


def write(name,value):
    if name=='launch.json':
        value=value|dict(mapping_and_pose_separate_processes=True,corner_budget_per_camera=300,
            processes_ready_before_stream=True,current_obstacles_before_registration=True,
            veto_latched_for_command_window=True,
            extra_sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                'lewm/process_mapped_runtime_development.py','lewm/feature_budget_300_tracker_development.py',
                'scripts/run_feature_budget300_paced_prefix_development.py',__file__)})
    if name=='result.json':
        value=value|dict(status='EARLY_OBSTACLE_PACED_PREFIX_COMPLETE',
            current_obstacles_before_registration=True,obstacle_coordinate_frame='current_body',
            veto_latched_for_command_window=True,mapping_and_pose_separate_processes=True,
            packet_transfer_and_obstacle_extraction_included_in_stage_timing=True)
    _write(name,value)


def main():
    bind(source.main,OUTPUT=OUTPUT,write=write,
        verify_evidence=bind(source.verify_evidence,_write=_write),
        ProcessSeparatedRuntime=EarlyObstacleRuntime)()


if __name__=='__main__':main()
