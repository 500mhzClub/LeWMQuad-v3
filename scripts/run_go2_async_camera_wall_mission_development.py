"""First full exposed-maze mission using actual host deadlines and async cameras."""
import hashlib
import json
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_async_camera_wall_probe_development as probe

ROOT='go2_async_camera_wall_mission_learned_layout01_4800_v1_attempt_001'


def finish(name,value):
    if name=='launch.json':
        value=value|dict(experiment='asynchronous_camera_full_wall_mission_v1',
            navigation_tick_budget=4800,intended_camera_frames=4801,
            timing_probe_not_full_navigation_trial=False,full_mission_implemented=True,
            prior_attempt_preserved=probe.ROOT,
            owner_cyclic_gc_deferred_during_bounded_probe=False,
            owner_cyclic_gc_deferred_during_bounded_mission=True,
            reference_root_name=probe.ROOT,exposed_development_layout=True,
            extra_sources=value['extra_sources']|{__file__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    RAW_WRITE(name,value)


def annotate(name,value):
    bind(probe.annotate,RAW_WRITE=bind(finish,RAW_WRITE=RAW_WRITE))(name,value)


def main():
    result=json.loads((probe.cohort.stable.source.BASE/probe.ROOT/'async_host_deadline_diagnostic_v1.json').read_text())
    if result['failure'] is not None or result['nonzero_applied_intervals']==0:
        raise ValueError('complete moving short probe before a full wall mission')
    bind(probe.main,ROOT=ROOT,NAVIGATION_TICKS=4800,annotate=annotate)()


if __name__=='__main__':main()
