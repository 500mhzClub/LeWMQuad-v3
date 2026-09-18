"""Prospective 300-ms planning with measured CPU latency charged to simulation."""
import hashlib
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from scripts.in_memory_paired_camera_session_development import InMemoryPairedCameraSession
from scripts import run_go2_paced_native_prefix_development as source

OUTPUT=source.BASE/'go2_measured_latency_native_prefix_layout00_v1_attempt_001'
_write=bind(source.write,OUTPUT=OUTPUT)


def write(name,value):
    if name=='launch.json':
        value=value|dict(host_real_time_execution_claimed=False,in_memory_camera_packets=True,
            simulator_lag_forces_zero=False,minimum_worker_latency='actual_measured_service_duration',
            extra_sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                __file__,'scripts/in_memory_paired_camera_session_development.py',
                'lewm/measured_latency_simulation_development.py')})
    _write(name,value)


if __name__=='__main__':
    bind(source.main,OUTPUT=OUTPUT,PacedNativeSession=InMemoryPairedCameraSession,write=write,
        CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3)()
