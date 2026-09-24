"""Prospective timed prefix using direct in-memory camera packets."""
import hashlib
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from scripts.in_memory_paired_camera_session_development import InMemoryPairedCameraSession
from scripts import run_go2_paced_native_prefix_development as source

OUTPUT=source.BASE/'go2_in_memory_camera_native_prefix_layout00_v1_attempt_001'
_write=bind(source.write,OUTPUT=OUTPUT)


def write(name,value):
    if name=='launch.json':
        value=value|dict(in_memory_camera_packets=True,static_checks_at_setup_and_termination=True,
            camera_resources_warmed_before_clock=True,diagnostic_segmentation_captured=False,
            extra_sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                __file__,'scripts/in_memory_paired_camera_session_development.py')})
    _write(name,value)


if __name__=='__main__':
    bind(source.main,OUTPUT=OUTPUT,PacedNativeSession=InMemoryPairedCameraSession,write=write)()
