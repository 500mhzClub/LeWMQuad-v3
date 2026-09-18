"""Bounded provisional diagnosis of persisted raw packets, not native admission.

The original native worker may still be verifying its physical prefix. This
script neither admits that execution nor changes or reruns its controller.
"""
from datetime import datetime, timezone
import gzip
import json
import numpy as np

from scripts.extended_budget_anchored_maze_development import ExtendedBudgetRGBDReplay, depth_packet
from scripts.novel_maze_auxiliary_packet_development import public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.navigation_artifact_root_development import BASE, artifact_path, verify_artifacts
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.floor_pose_registration_development import measured_candidates, register_pose
from lewm.joint_measured_floor_plane_development import fit_joint_plane, validate_joint_plane
from lewm.joint_floor_registered_evidence_development import depth_hash
from lewm.joint_sensor_anchored_goal_development import current_joint_pose

SOURCE = 'scripts/diagnose_go2_extended_budget_floor_boundary_v1.py'
NATIVE = BASE/'go2_no_rgb_direct_extended_budget_maze02_pilot_v1_attempt_001'
CASE = 'no_rgb_direct_extended_budget_anchored_maze_02'
OUTPUT = ROOT/'docs/go2_extended_budget_floor_boundary_provisional_2026-09-11.json'
FRAMES = (0, 3836, 3837)
FAILURE = 'floor registration exceeds fixed development correction gates'
FIXED = {
    'launch.json': '7380fbd8c83306dfe544fd9760e13a9eb8a0b70cb93ade06e1972096044c8113',
    CASE+'_audit.json': '1a043f14fa90825dabb425821cdba663aef49c7dd26637b893ae6f2c7739806e',
    CASE+'_readout.json': '6fd958a93f18337adbb5b7af4d1232ce14ef2b52ba95455d197e86370a08cf7d',
    CASE+'/context_decisions.jsonl.gz': 'fd90c366497a6c51f1b7ac92524a0d2ff9a440488ef0d1d2bdb645d6df0d47b8',
}


def select_rows(directory):
    """Skip bounded lines; decode only initial, last accepted and rejected rows."""
    selected = {}
    with gzip.open(directory/'context_decisions.jsonl.gz', 'rb') as stream:
        for frame in range(FRAMES[-1]+1):
            line = stream.readline(32*1024**2+1)
            if not line.endswith(b'\n') or len(line) > 32*1024**2:
                raise ValueError('bounded complete decision line required')
            if frame in FRAMES:
                row = json.loads(line)
                if row['tick'] != frame:
                    raise ValueError('exact selected original observation frame required')
                decision = row['decision']
                if decision['original_visual_evidence']['terminal_failure'] is not None:
                    raise ValueError('original visual observer must be nonterminal')
                if frame == FRAMES[-1]:
                    if (decision['terminal'] != 'SENSOR_OR_MODEL_FAILURE'
                            or decision['failure'] != FAILURE or decision['tick'] != FRAMES[-2]
                            or decision['requested_command'] != [0., 0., 0.]):
                        raise ValueError('exact saved pre-planning floor rejection required')
                elif decision['terminal'] is not None or decision['tick'] != frame:
                    raise ValueError('initial and preceding frame must be accepted')
                selected[frame] = row
    return selected


def reconstruct(directory, selected):
    reader = ExtendedBudgetRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    reference = selected[0]['decision']['evidence']['floor_registration']['reference']
    up = np.asarray(reference['initial_up_body'])
    fits = []
    for frame in FRAMES:
        decision = selected[frame]['decision']
        raw = decision['original_visual_evidence']
        if raw['identity'] != [0, 0, 0]:
            raise ValueError('exact saved episode identity required')
        raw = raw | dict(identity=tuple(raw['identity']))
        policy, primary, _, now = reader.packet(frame)
        auxiliary = depth_packet(directory, frame, policy,
            public_acquisition(acquisitions[frame]), now_ns=now)
        p, R, pose = current_joint_pose(raw, identity=(0, 0, 0), now_ns=now)
        if (pose['frame'] != frame or pose['depth_sha256'] != depth_hash(primary)
                or pose['auxiliary_depth_sha256'] != depth_hash(auxiliary)):
            raise ValueError('same-frame original pose and both raw depth hashes required')
        if frame == 0:
            force = policy['sensor_state']['sensed']['specific_force']
            command = policy['sensor_state']['control']['applied_command']
            measured_up = force['values'].mean(0)
            measured_up /= np.linalg.norm(measured_up)
            if (not force['valid'].all() or not command['valid'].all()
                    or np.any(np.abs(command['values']) > 1e-8)
                    or not np.array_equal(up, measured_up)):
                raise ValueError('original quiet public initial gravity reference required')
        clouds = [measured_candidates(d['depth_m'], d['valid'], E, R.T@up)[0]
            for d, E in ((primary, np.asarray(BODY_FROM_OPTICAL)), (auxiliary, body_from_optical()))]
        joint = fit_joint_plane(*clouds, R.T@up)
        validate_joint_plane(joint, R.T@up)
        n = np.asarray(reference['joint_plane']['normal_body'])
        a = R@np.asarray(joint['normal_body'])
        height = joint['offset_body_m']-reference['joint_plane']['offset_body_m']-float(n@p)
        tilt = float(np.arctan2(np.linalg.norm(np.cross(a, n)), a@n))
        try:
            correction = register_pose(p, R, reference['joint_plane'], joint)
            error = None
        except ValueError as exc:
            error = str(exc)
            correction = None
        if frame < FRAMES[-1]:
            saved = decision['evidence']['floor_registration']
            if (error is not None or saved['reference'] != reference
                    or joint != saved['joint_plane'] or correction != saved['correction']):
                raise ValueError('accepted raw planes and corrections must reconstruct exactly')
        elif error != FAILURE:
            raise ValueError('original terminal rejection must reproduce')
        fits.append(dict(frame=frame, joint_plane=joint, raw_position_initial_body_m=p.tolist(),
            normal_translation_correction_m=height, normal_alignment_rad=tilt,
            height_gate_exceeded=bool(abs(height) > .05), tilt_gate_exceeded=bool(tilt > .10),
            maximum_correction_m=.05, maximum_tilt_correction_rad=.10,
            correction=correction, reproduced_failure=error, both_raw_depth_packets_used=True,
            visual_pose_inference_reexecuted=False, native_pose_used=False))
    return fits


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive diagnostic receipt required')
    verify_artifacts(NATIVE, FIXED)
    launch = read_json(NATIVE, 'launch.json')
    sources = discover_sources((SOURCE,), launch['source_sha256'])
    verify(sources)
    names = ['result.json', 'policy_observations.json', 'policy_histories.npz',
        'depth_observations.json', 'fast_gyro_histories.npz', 'auxiliary_camera_audit.json']
    names += [f'{kind}_{frame:04d}.{suffix}' for frame in FRAMES
        for kind, suffix in [('rgb', 'png'), ('depth', 'npz'), ('auxiliary_depth', 'npz')]]
    bindings = FIXED | {CASE+'/'+name: digest(artifact_path(NATIVE, CASE+'/'+name)) for name in names}
    verify_artifacts(NATIVE, bindings)
    directory = NATIVE/CASE
    collection = read_json(directory, 'result.json')
    if collection['decisions'] != 3848 or collection['rgbd_frames'] != 3848:
        raise ValueError('exact persisted collection population required')
    selected = select_rows(directory)
    fits = reconstruct(directory, selected)
    verify(sources)
    verify_artifacts(NATIVE, bindings)
    write_json(OUTPUT, dict(status='PROVISIONAL_PERSISTED_EXTENDED_BUDGET_FLOOR_REJECTION_RECONSTRUCTED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources,
        artifact_sha256=bindings, diagnostic_frames=list(FRAMES),
        original_launch_source_count=len(launch['source_sha256']), raw_depth_floor_reconstructions=fits,
        selected_decisions_decoded=3, complete_decision_stream_decoded=False,
        native_completion_admitted=False, native_worker_may_still_be_verifying=True,
        full_raw_model_audit_reexecuted=False, original_visual_inference_reexecuted=False,
        raw_packet_files_unchanged_before_and_after=True, fixed_floor_gates_unchanged=True,
        native_execution=False, policy_selected=False, navigation_qualified=False,
        real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
    print('PROVISIONAL_FLOOR_RECONSTRUCTION', digest(OUTPUT), len(sources), flush=True)
    print(json.dumps(fits, indent=2), flush=True)


if __name__ == '__main__':
    main()
