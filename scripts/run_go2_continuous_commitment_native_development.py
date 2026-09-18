"""Twenty-second prospective test of continuous 400-ms model-selected windows."""
import hashlib
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.feature_budget_150_tracker_development import initialize_pose_150
from lewm.independent_depth_process_development import initialize_obstacles,obstacles_ready
from lewm.continuous_commitment_runtime_development import ContinuousCommitmentRuntime
from scripts.in_memory_paired_camera_session_development import InMemoryPairedCameraSession
from scripts import run_go2_paced_native_prefix_development as source

OUTPUT=source.BASE/'go2_continuous_commitment_20s_native_layout00_v1_attempt_001'
_write=bind(source.write,OUTPUT=OUTPUT)


def write(name,value):
    if name=='launch.json':
        value=value|dict(host_real_time_execution_claimed=False,in_memory_camera_packets=True,
            independent_current_depth_process=True,simulator_lag_forces_zero=False,
            simulation_release_resolution_ns=2_000_000,corner_budget_per_camera=150,
            command_duration_ns=400_000_000,actual_prefix_checked_before_dispatch=True,
            extra_sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (
                __file__,'scripts/in_memory_paired_camera_session_development.py',
                'lewm/feature_budget_150_tracker_development.py',
                'lewm/independent_depth_obstacle_development.py','lewm/independent_depth_runtime_development.py',
                'lewm/independent_depth_process_development.py','lewm/measured_latency_simulation_development.py',
                'lewm/continuous_commitment_runtime_development.py','lewm/continuous_commitment_ledger_development.py')})
    _write(name,value)


if __name__=='__main__':
    bind(source.main,OUTPUT=OUTPUT,COUNT=201,PacedNativeSession=InMemoryPairedCameraSession,write=write,
        CLOCK_MODE='measured_simulation',PLANNING_DELAY_TICKS=3,POSE_INITIALIZER=initialize_pose_150,
        MEASURED_RUNTIME_CLASS=ContinuousCommitmentRuntime,
        OBSTACLE_INITIALIZER=initialize_obstacles,OBSTACLE_READY=obstacles_ready)()
