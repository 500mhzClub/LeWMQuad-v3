"""Evaluate reported continuous-mission arrivals against saved native physics."""
import argparse
import json
from pathlib import Path
import numpy as np
from lewm.physical_execution_development import rotation_xyzw

BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')


def evaluate(root):
    metadata=json.loads((root/'native/in_memory_camera_observations.json').read_text())
    frames={r['frame']:r for r in metadata['frames']}
    mission=json.loads((root/'mission.json').read_text())
    poses=json.loads((root/'poses.json').read_text())
    requests={r['simulator_ns']:r['requested_command'] for r in json.loads((root/'requests.json').read_text())}
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as arrays:
        physics=arrays['base_pose_world'];contacts=int(np.count_nonzero(arrays['physics_contact']))
    origin=physics[frames[0]['physical_sample_index']]
    R0=rotation_xyzw(origin[3:])
    def local(indices):return (physics[indices,:3]-origin[:3])@R0
    rows=[]
    for arrival in mission[-1]['arrivals']:
        frame=arrival['frame'];start=frame-10
        if start<0 or any(f not in frames for f in range(start,frame+1)):
            raise ValueError('complete one-second recorded arrival population required')
        first,last=frames[start],frames[frame]
        if last['measured_ns']-first['measured_ns']!=1_000_000_000:
            raise ValueError('actual one-second arrival interval required')
        indices=np.arange(first['physical_sample_index'],last['physical_sample_index']+1)
        positions=local(indices);target=np.asarray(arrival['target_initial_body_xy_m'])
        distances=np.linalg.norm(positions[:,:2]-target,axis=1)
        boundaries=local([frames[f]['physical_sample_index'] for f in range(start,frame+1)])
        speeds=np.linalg.norm(np.diff(boundaries,axis=0),axis=1)/.1
        times=range(first['measured_ns'],last['measured_ns'],20_000_000)
        zero=all(t in requests and np.array_equal(requests[t],[0.,0.,0.]) for t in times)
        geometry=bool(np.all(distances<=.04));quiet=bool(np.all(speeds<=.05))
        rows.append(dict(phase=arrival['phase'],frame=frame,observed_distance_m=arrival['observed_distance_m'],
            native_minimum_distance_m=float(distances.min()),native_maximum_distance_m=float(distances.max()),
            native_final_distance_m=float(distances[-1]),physical_radius_m=.04,
            dwell_seconds=1.,native_maximum_100ms_speed_m_s=float(speeds.max()),
            all_requested_intervals_zero=zero,physical_distance_passed=geometry,
            measured_motion_quiet=quiet,arrival_checks_passed=geometry and quiet and zero))
    actual=local([frames[p['frame']]['physical_sample_index'] for p in poses])
    estimated=np.array([p['registered_pose']['position_initial_body_m'] for p in poses])
    errors=np.linalg.norm(actual-estimated,axis=1)
    clean=(root/'result.json').exists() and not (root/'failure.json').exists()
    paired=[r['phase'] for r in rows]==['OUTBOUND','RETURN'] and all(r['arrival_checks_passed'] for r in rows)
    return dict(camera_pairs=len(frames),registered_poses=len(poses),arrivals=rows,
        mission_terminal=mission[-1]['terminal'],disallowed_contact_samples=contacts,
        static_camera_identity_unchanged=metadata['static_identity_unchanged'],
        clean_result_available=clean,owner_exit_status_checked_separately=True,
        round_trip_arrival_checks_passed=paired and contacts==0,
        median_position_error_m=float(np.median(errors)),maximum_position_error_m=float(errors.max()),
        native_state_evaluator_only=True,raw_sensor_audit_complete=False,
        host_real_time_qualified=False,real_sensor_uncertainty_calibrated=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root-name',required=True)
    args=parser.parse_args()
    if Path(args.root_name).name!=args.root_name or args.root_name.startswith('sealed'):
        raise ValueError('ordinary development artifact basename required')
    root=BASE/args.root_name;output=root/'continuous_native_arrival_evaluation.json'
    if output.exists():raise ValueError('preserve evaluation')
    report=evaluate(root)
    with output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
