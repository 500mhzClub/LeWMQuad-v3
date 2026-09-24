"""Bounded prospective continuous outbound, measured arrival dwell, and return."""
import hashlib
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.feature_budget_150_tracker_development import initialize_pose_150
from lewm.feature_budget_100_tracker_development import initialize_pose_100
from lewm.independent_depth_process_development import initialize_obstacles,obstacles_ready
from lewm.continuous_round_trip_runtime_development import ContinuousRoundTripRuntime
from lewm.process_registered_round_trip_development import (
    ProcessRegisteredRoundTripRuntime,initialize_registration,registration_ready)
from lewm.local_tracked_round_trip_development import LocalTrackedRoundTripRuntime
from scripts.in_memory_paired_camera_session_development import InMemoryPairedCameraSession
from scripts import run_go2_paced_native_prefix_development as source

OUTPUT=source.BASE/'go2_continuous_round_trip_native_layout00_v1_attempt_007'
_write=bind(source.write,OUTPUT=OUTPUT)


def write(name,value):
    if name=='launch.json':
        value=value|dict(host_real_time_execution_claimed=False,in_memory_camera_packets=True,
            independent_current_depth_process=True,simulator_lag_forces_zero=False,
            independent_registration_process=True,
            vectorized_same_connector_geometry=True,
            visual_tracking_in_main_process=False,
            simulation_release_resolution_ns=2_000_000,corner_budget_per_camera=100,
            command_duration_ns=400_000_000,actual_prefix_checked_before_dispatch=True,
            full_mission_implemented=True,measured_round_trip_enabled=True,navigation_tick_budget=1800,
            extra_sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                __file__,'scripts/in_memory_paired_camera_session_development.py',
                'lewm/feature_budget_150_tracker_development.py',
                'lewm/independent_depth_obstacle_development.py','lewm/independent_depth_runtime_development.py',
                'lewm/independent_depth_process_development.py','lewm/measured_latency_simulation_development.py',
                'lewm/continuous_commitment_runtime_development.py','lewm/continuous_commitment_ledger_development.py',
                'lewm/continuous_round_trip_runtime_development.py','lewm/process_registered_round_trip_development.py',
                'lewm/vectorized_connector_routing_development.py','lewm/local_tracked_round_trip_development.py',
                'lewm/feature_budget_100_tracker_development.py')})
    _write(name,value)


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=1,mp_context=get_context('spawn'),initializer=initialize_registration) as executor:
        assert executor.submit(registration_ready).result()
        def runtime(*args,**kwargs):
            return ProcessRegisteredRoundTripRuntime(*args,registration_executor=executor,**kwargs)
        bind(source.main,OUTPUT=OUTPUT,COUNT=601,PacedNativeSession=InMemoryPairedCameraSession,write=write,
            CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,POSE_INITIALIZER=initialize_pose_100,
            MEASURED_RUNTIME_CLASS=runtime,
            OBSTACLE_INITIALIZER=initialize_obstacles,OBSTACLE_READY=obstacles_ready)()
