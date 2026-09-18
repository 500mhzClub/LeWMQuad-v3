"""Read-only call-input equality diagnosis on unchanged controller replay."""
import hashlib
import sys
import json
import time

import cv2
import numpy as np
from lewm.floor_footprint_bounds_development import observed_floor_cell_index
import torch
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.confirmed_floor_round_trip_controller_development import ConfirmedFloorRoundTripController
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.training_translation_bias_model_admission_development import load_assigned

INPUT = BASE/'go2_confirmed_floor_maze_pilot_v1_attempt_001'
OUTPUT = BASE/'go2_floor_index_reuse_diagnosis_v1_attempt_001'
RESULT = '5ee4ef051e1a506f205aae51610deece18f755fb5440c40b1113e22e5ba317ee'
PROTOCOL = 'docs/go2_floor_index_reuse_diagnosis_v1_2026-09-09.md'
PROFILE_FRAMES = (20, 60, 100)


class CallInputs:
    def __init__(self):
        self.groups = {}; self.calls = []

    def callback(self, frame, event, arg):
        if event != 'call' or frame.f_code is not observed_floor_cell_index.__code__:
            return
        d = np.asarray(frame.f_locals['depth'])
        v = np.asarray(frame.f_locals['valid'])
        u = np.asarray(frame.f_locals['up'], dtype=float)
        key = (d.dtype.str, d.shape, d.tobytes(), v.dtype.str, v.shape, v.tobytes(), u.shape, u.tobytes())
        if key not in self.groups:
            self.groups[key] = len(self.groups)
        caller = frame.f_back
        self.calls.append(dict(group=self.groups[key], depth_sha256=hashlib.sha256(key[2]).hexdigest(),
            valid_sha256=hashlib.sha256(key[5]).hexdigest(), up_sha256=hashlib.sha256(key[7]).hexdigest(),
            up=u.tolist(), depth_dtype=d.dtype.str, caller_file=caller.f_code.co_filename,
            caller_function=caller.f_code.co_name, caller_line=caller.f_lineno,
            byte_equality_used_for_grouping=True))

    def runcall(self, function, *args, **kwargs):
        if sys.getprofile() is not None:
            raise ValueError('unprofiled diagnostic process required')
        sys.setprofile(self.callback)
        try:
            return function(*args, **kwargs)
        finally:
            sys.setprofile(None)


def main():
    if not __debug__:
        raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive diagnostic output required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    verify_artifacts(INPUT, {'result.json': RESULT})
    result = read_json(INPUT, 'result.json')
    assert result['status'] == 'CONFIRMED_FLOOR_MAZE_PILOT_COMPLETE'
    bindings = {'result.json': RESULT, **result['artifact_sha256']}
    verify_artifacts(INPUT, bindings)
    old = read_json(INPUT, 'launch.json'); verify(old)
    sources = discover_sources((PROTOCOL, 'scripts/diagnose_go2_floor_index_reuse_v1.py'), old['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 4*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+128*1024**2:
        raise ValueError('diagnostic resource envelope unavailable')
    launch = old | dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT),
        input_artifact_sha256=bindings, hardware=resources, maximum_frames=101,
        profile_frames=list(PROFILE_FRAMES), native_execution=False, model_training=False,
        native_scene_workers=0, cpu_processes=1, numerical_threads=1,
        concurrency_reason='immutable predecessor replay independent of current single native scene',
        minimum_available_ram_bytes=4*1024**3, output_allowance_bytes=128*1024**2,
        os_resource_limits_enforced=False, real_time_qualified=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('FLOOR_INDEX_REUSE_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        name, index, variant, condition, model_name = old['planned_case']
        model, c, v = load_assigned(old['correction_admission'], model_name)
        assert (c, v) == (condition, variant)
        before = state_digest(model.state_dict())
        controller = ConfirmedFloorRoundTripController(model, ArticulatedCollisionGeometry(URDF),
            public_mission=public_mission(index), navigation_ticks=NAVIGATION_TICKS,
            persistent=True, condition=condition, variant=variant)
        directory = INPUT/name; reader = IntentReturnRGBDReplay(directory)
        acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
        timings = []; profiles = []
        for i, recorded in enumerate(read_rows(directory)):
            if i > 100:
                break
            assert recorded['tick'] == i
            p, d, f, now = reader.packet(i)
            auxiliary = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
            profiler = CallInputs() if i in PROFILE_FRAMES else None
            started = time.perf_counter()
            if profiler is None:
                decision = controller.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary)
            else:
                decision = profiler.runcall(controller.observe, p, d, f, now_ns=now, auxiliary_depth=auxiliary)
            elapsed = 1000*(time.perf_counter()-started)
            if json.loads(json.dumps(decision)) != recorded['decision']:
                raise ValueError(f'complete controller decision differs at frame {i}')
            timings.append(dict(frame=i, controller_wall_ms=elapsed, profiled=profiler is not None,
                original_controller_wall_ms=recorded['controller_wall_ms'],
                original_acquisition_wall_ms=recorded['acquisition_wall_ms']))
            if profiler is not None:
                profiles.append(dict(frame=i, calls=profiler.calls, exact_input_groups=len(profiler.groups)))
                print('FLOOR_INDEX_REUSE_FRAME', i, elapsed, flush=True)
        assert len(timings) == 101 and len(profiles) == 3
        assert state_digest(model.state_dict()) == before
        verify(launch); verify_artifacts(INPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='FLOOR_INDEX_REUSE_DIAGNOSIS_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'), input_result_sha256=RESULT,
            source_sha256=sources, timings=timings, profiles=profiles,
            complete_decisions_exact=True, model_state_unchanged=True,
            model_state_sha256=before, hardware_after=hardware(),
            native_execution=False, model_training=False, real_time_qualified=False,
            navigation_qualified=False, goal_achieved=False))
        print('FLOOR_INDEX_REUSE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='FLOOR_INDEX_REUSE_DIAGNOSIS_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__':
    main()
