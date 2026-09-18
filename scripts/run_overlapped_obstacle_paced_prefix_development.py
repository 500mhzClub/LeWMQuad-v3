"""Overlap current point extraction with the next image-tracking update."""
from lewm.eligible_floor_registration_development import bind
from lewm.process_mapped_runtime_development import OverlappedObstacleRuntime
from scripts import run_early_obstacle_paced_prefix_development as source

OUTPUT=source.OUTPUT.parent/'go2_overlapped_obstacle_paced_recorded_prefix_v1_attempt_001'
_write=bind(source.source.source.write,OUTPUT=OUTPUT)
base_write=bind(source.write,_write=_write)


def write(name,value):
    if name in ('launch.json','result.json'):
        value=value|dict(obstacle_extraction_worker='registration_before_floor_registration',
            obstacle_extraction_overlaps_next_tracking=True)
    base_write(name,value)


if __name__=='__main__':
    # Bind the verification writer as well, so completed predecessors stay intact.
    verify=bind(source.source.verify_evidence,_write=_write)
    candidate_main=bind(source.source.main,verify_evidence=verify)
    bind(candidate_main,OUTPUT=OUTPUT,write=write,
        ProcessSeparatedRuntime=OverlappedObstacleRuntime)()
