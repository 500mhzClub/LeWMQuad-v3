"""Fixed evaluator-only physical trace diagnosis; final full audit still required."""
import json

import numpy as np

from lewm.novel_maze_round_trip_evaluation_development import evaluate
from lewm.physical_execution_development import rotation_xyzw
from scripts import probe_go2_measured_plane_return_anchor_pairs_v1 as probe
from scripts.startup_source_inventory_development import discover_sources

SOURCE = 'scripts/diagnose_go2_measured_plane_return_turn_v1.py'
OUTPUT = probe.ROOT/'docs/go2_measured_plane_return_turn_trace_diagnosis_2026-09-12.json'
START_FRAME = 3062
FAILURE_FRAME = 3113


def main():
    require, digest = probe.require, probe.digest
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive fixed diagnosis output required')
    require(digest(probe.INPUT/'launch.json') == probe.LAUNCH_SHA, 'exact original native launch required')
    launch = json.loads((probe.INPUT/'launch.json').read_text())
    sources = discover_sources((SOURCE,), launch['source_sha256']); probe.verify_sources(sources)
    directory = probe.INPUT/probe.CASE
    names = ('result.json','physics_trace.npz','command_tape.json')
    inputs = {name:digest(probe._leaf(directory,name)) for name in names}
    require(inputs['result.json'] == probe.COLLECTION_SHA, 'same closed negative collection required')
    collection = json.loads(probe._leaf(directory,'result.json').read_text())
    mission = collection['mission_receipt']
    require(mission['frame'] == FAILURE_FRAME-1 and len(mission['arrivals']) == 1
        and mission['arrivals'][0]['frame'] == START_FRAME
        and collection['decisions'] == 3124 and collection['terminal_zero_ticks'] == 10,
        'exact observed arrival and return failure boundary required')
    with np.load(probe._leaf(directory,'physics_trace.npz'),allow_pickle=False) as saved:
        raw = {key:saved[key] for key in ('timestamp_s','base_pose_world','base_twist_world',
            'requested_command','physics_contact')}
    evaluation = evaluate(raw,mission,collection,layout_index=2)
    pose = raw['base_pose_world']; require(len(pose) == 156900, 'complete original physical samples required')
    R0 = rotation_xyzw(pose[749,3:]); local = (pose[:,:3]-pose[749,:3])@R0
    start,end = 749+50*START_FRAME,749+50*FAILURE_FRAME
    rotations = [rotation_xyzw(row[3:]) for row in pose[start:end+1]]
    yaw = np.unwrap([np.arctan2(R[1,0],R[0,0]) for R in rotations])
    tape = json.loads(probe._leaf(directory,'command_tape.json').read_text())
    require(len(tape) == collection['command_ticks'] == 3123, 'complete original command tape required')
    commands = []
    for frame in range(START_FRAME,FAILURE_FRAME):
        item = tape[frame]; a,b = 749+50*frame,799+50*frame
        require(item['tick'] == frame and item['completed'] is True
            and (item['pre_sample_index'],item['post_sample_index']) == (a,b),
            'actual completed return command endpoints required')
        np.testing.assert_array_equal(raw['requested_command'][a+1:b+1],
            np.tile(np.asarray(item['requested_command']), (50,1)))
        commands.append(dict(frame=frame,requested_command=item['requested_command']))
    last = mission['frame']; observed = np.asarray(mission['observed_settling']['current_position_initial_body_m'])
    require(mission['observed_settling']['current_frame'] == last, 'last accepted measured position required')
    report = dict(status='MEASURED_PLANE_RETURN_TURN_TRACE_DIAGNOSIS_COMPLETE',
        source_sha256=sources,input_root=str(directory),input_sha256=inputs,
        native_launch_sha256=probe.LAUNCH_SHA,physical_evaluation=evaluation,
        contact_sample_count=int(np.count_nonzero(raw['physics_contact'])),
        return_until_failure=dict(start_frame=START_FRAME,failure_frame=FAILURE_FRAME,
            duration_s=float(raw['timestamp_s'][end]-raw['timestamp_s'][start]),
            xy_displacement_m=float(np.linalg.norm(local[end,:2]-local[start,:2])),
            xy_path_length_m=float(np.linalg.norm(np.diff(local[start:end+1,:2],axis=0),axis=1).sum()),
            unwrapped_world_yaw_change_rad=float(yaw[-1]-yaw[0]),
            maximum_xy_speed_m_s=float(np.linalg.norm(raw['base_twist_world'][start:end+1,:2],axis=1).max()),
            completed_requested_commands=commands),
        last_accepted_position=dict(frame=last,observed_xy_m=observed[:2].tolist(),
            physical_xy_m=local[749+50*last,:2].tolist(),
            xy_error_m=float(np.linalg.norm(observed[:2]-local[749+50*last,:2]))),
        physical_positions=[dict(frame=frame,xy_initial_body_m=local[749+50*frame,:2].tolist())
            for frame in (3062,3102,3103,3112,3113,3123)],
        scope=dict(evaluator_only=True,closed_recorded_trace=True,controller_native_pose_access=False,
            new_native_execution=False,full_sensor_reconstruction_performed=False,
            full_command_slew_audit_performed=False,final_native_audit_verified=False,
            candidate_tracking_accuracy_measured=False,navigation_recovered=False,goal_achieved=False))
    probe.verify_sources(sources)
    for name,expected in inputs.items(): require(digest(probe._leaf(directory,name)) == expected, 'input changed: '+name)
    require(digest(probe.INPUT/'launch.json') == probe.LAUNCH_SHA, 'original native launch changed')
    probe.write_json(OUTPUT,report)
    print(json.dumps(dict(output=str(OUTPUT),sha256=digest(OUTPUT),
        arrival_windows=evaluation['arrival_windows'],contact_samples=report['contact_sample_count'],
        return_xy_displacement_m=report['return_until_failure']['xy_displacement_m'],
        return_yaw_change_rad=report['return_until_failure']['unwrapped_world_yaw_change_rad'],
        last_accepted_xy_error_m=report['last_accepted_position']['xy_error_m'])),flush=True)


if __name__ == '__main__':main()
