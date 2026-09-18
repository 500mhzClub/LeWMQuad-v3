"""Authenticate the completed hold intervention and reconstruct its floor rejection."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from scripts import await_go2_hold_reorientation_maze02_native_v1 as wait
from scripts.maze_decision_stream_development import read_rows
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify, digest, write_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.novel_maze_auxiliary_packet_development import packet, public_acquisition
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.floor_pose_registration_development import measured_candidates, register_pose
from lewm.joint_measured_floor_plane_development import fit_joint_plane, validate_joint_plane
from lewm.joint_floor_registered_evidence_development import depth_hash
from lewm.joint_sensor_anchored_goal_development import current_joint_pose

SOURCE = 'scripts/diagnose_go2_completed_hold_reorientation_maze02_v1.py'
WAIT_SHA = '7b81a8e5b39d39d8d13cbaea015ba64c9e08eb55c685072479b951bf3c334e50'
WAIT_LAUNCH = 'e27675df102b072b62f4351483363ab9ee80e9e0244e1193c398392f716a47f6'
NATIVE_SHA = '39e4e616361fde21d4cb5b6480794d3e062ace7bebfed04d70031e02a87937e9'
NATIVE_LAUNCH = '2d3e8c2ad103e3b955a9163453743e372cb068dcf6b00c1999f020425e729484'
OWNER = dict(pid=2671835, created=1789036731.41, command=[
    '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', wait.SOURCE])
OUTPUT = ROOT/'docs/go2_completed_hold_reorientation_maze02_diagnosis_2026-09-11.json'
FAILURE = 'floor registration exceeds fixed development correction gates'


def floor_boundary(directory, selected):
    """Reconstruct three actual measured planes; no visual or model inference."""
    reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json')
    reference = selected[0]['decision']['evidence']['floor_registration']['reference']
    up = np.asarray(reference['initial_up_body'])
    fits = []
    for frame in (0, 1993, 1994):
        decision = selected[frame]['decision']
        raw = decision['original_visual_evidence']
        # Restore only the live tuple type lost by the saved JSON encoding.
        if raw['identity'] != [0, 0, 0]:
            raise ValueError('exact saved episode identity required')
        raw = raw | dict(identity=tuple(raw['identity']))
        policy, primary, _, now = reader.packet(frame)
        auxiliary = packet(directory, frame, policy, public_acquisition(acquisitions[frame]), now_ns=now)
        p, R, pose = current_joint_pose(raw, identity=(0, 0, 0), now_ns=now)
        if (pose['frame'] != frame or pose['depth_sha256'] != depth_hash(primary)
                or pose['auxiliary_depth_sha256'] != depth_hash(auxiliary)):
            raise ValueError('same-frame original visual witness and both raw depth packets required')
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
            error = str(exc); correction = None
        if frame < 1994:
            saved = decision['evidence']['floor_registration']
            if (error is not None or saved['reference'] != reference
                    or joint != saved['joint_plane'] or correction != saved['correction']):
                raise ValueError('initial and last accepted planes and corrections must reconstruct exactly')
        elif error != FAILURE:
            raise ValueError('original terminal floor correction rejection must reproduce')
        fits.append(dict(frame=frame, joint_plane=joint, raw_position_initial_body_m=p.tolist(),
            normal_translation_correction_m=height, normal_alignment_rad=tilt,
            height_gate_exceeded=abs(height) > .05, tilt_gate_exceeded=tilt > .10,
            maximum_correction_m=.05, maximum_tilt_correction_rad=.10,
            correction=correction, reproduced_failure=error, both_raw_depth_packets_used=True,
            visual_pose_inference_reexecuted=False, native_pose_used=False))
    return fits


def main():
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != wait.BOOT or wait.owner_live(OWNER):
        raise ValueError('original hold waiter must have ended on recorded boot')
    for root in (wait.OUTPUT, wait.native.OUTPUT):
        if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
            raise ValueError('original execution failure must be retained')
    verify_artifacts(wait.OUTPUT, {'result.json': WAIT_SHA, 'launch.json': WAIT_LAUNCH})
    saved = read_json(wait.OUTPUT, 'result.json'); registration = read_json(wait.OUTPUT, 'launch.json')
    sources = discover_sources((SOURCE,), registration['source_sha256']); verify(sources)
    verify_artifacts(wait.OUTPUT, saved['artifact_sha256'])
    if (saved['status'] != 'HOLD_REORIENTATION_MAZE02_NATIVE_WAIT_V1_COMPLETE'
            or saved['source_sha256'] != registration['source_sha256']
            or registration['boot_id'] != wait.BOOT or registration['waiter_pid'] != OWNER['pid']):
        raise ValueError('exact completed original waiter required')
    completion = wait.authenticate_completed(sources, read_json(wait.OUTPUT, 'input_completion.json'))
    if completion != saved['report'] or completion != read_json(wait.OUTPUT, 'native_completion.json'):
        raise ValueError('original native completion must reconstruct')
    root = wait.native.OUTPUT
    verify_artifacts(root, {'result.json': NATIVE_SHA, 'launch.json': NATIVE_LAUNCH})
    result = read_json(root, 'result.json'); launch = read_json(root, 'launch.json')
    record = result['conditions'][0]; name = record['case']; directory = root/name
    collection = read_json(directory, 'result.json'); audit = read_json(root, name+'_audit.json')
    with np.load(directory/'physics_trace.npz', allow_pickle=False) as archive:
        contact = archive['physics_contact']
        if len(contact) != collection['physics_samples']: raise ValueError('complete physics contacts required')
        readout = wait.native.case_readout(audit, collection, contact)
    if readout != record['readout'] or collection != record['collection']:
        raise ValueError('original native readout must reconstruct')
    count = 0; terminal = None; selected = {}; events = []; compact = []; modes = Counter(); actions = Counter()
    stream_hash = hashlib.sha256()
    for row in read_rows(directory):
        frame = row['tick']; d = row['decision']
        stream_hash.update(json.dumps(row, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n')
        if frame != count: raise ValueError('complete ordered decision history required')
        count += 1
        if frame in (0, 405, 1993, 1994): selected[frame] = row
        if terminal is not None:
            if d['tick'] != 1993 or d['terminal'] != 'SENSOR_OR_MODEL_FAILURE' or d['requested_command'] != [0., 0., 0.]:
                raise ValueError('original terminal zero-command drain required')
            continue
        if d['terminal'] is not None:
            terminal = frame
            if (frame != 1994 or d['tick'] != 1993 or d['failure'] != FAILURE
                    or d['requested_command'] != [0., 0., 0.] or d['new_selection'] is not None):
                raise ValueError('exact original pre-planning registration rejection required')
        elif d['tick'] != frame: raise ValueError('current admitted decision identity required')
        selection = d.get('new_selection') or {}; event = selection.get('hold_reorientation')
        if event:
            if event['frame'] != frame or event['selected_action'] != d['selected_action']:
                raise ValueError('current intervention receipt must match requested action')
            events.append(event)
        modes[d['planner_mode']] += 1; actions[str(d['selected_action'])] += 1
        evidence = d.get('evidence') or {}; visual = d.get('original_visual_evidence') or {}
        correction = (evidence.get('floor_registration') or {}).get('correction')
        compact.append(dict(frame=frame, decision_tick=d['tick'], action=d['selected_action'],
            requested_command=d['requested_command'], terminal=d['terminal'], failure=d['failure'],
            observed_goal_distance_m=d['observed_goal_distance_m'],
            floor_status=evidence.get('status'), visual_status=visual.get('status'),
            correction=correction, hold_reorientation=event is not None))
    if (count != 2005 or count != collection['decisions'] or count != collection['rgbd_frames']
            or count-terminal-1 != 10 or events[0]['frame'] != record['prefix_comparison']['first_intervention_frame']):
        raise ValueError('complete 2005 observations, original intervention and ten drain observations required')
    fits = floor_boundary(directory, selected)
    verify(sources); verify_artifacts(root, result['artifact_sha256'] | {'result.json': NATIVE_SHA})
    verify_artifacts(wait.OUTPUT, saved['artifact_sha256'] | {'result.json': WAIT_SHA})
    if wait.owner_live(OWNER): raise ValueError('original owner unexpectedly live')
    write_json(OUTPUT, dict(status='COMPLETED_HOLD_REORIENTATION_FLOOR_REJECTION_RECONSTRUCTED',
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=sources, source_count=len(sources),
        native_source_count=len(result['source_sha256']), waiter_result_sha256=WAIT_SHA,
        native_result_sha256=NATIVE_SHA, native_launch_sha256=NATIVE_LAUNCH,
        complete_native_artifact_count=len(result['artifact_sha256']), original_owner_ended=True,
        original_completion_verifier_reexecuted=True, model_state_sha256=launch['model_state_sha256'],
        complete_decisions=count, terminal_observation_frame=terminal, last_admitted_decision_frame=1993,
        terminal_drain_observations=10, canonical_decision_stream_sha256=stream_hash.hexdigest(),
        readout=readout, original_prefix_comparison=record['prefix_comparison'],
        hold_reorientation_events=events, modes=dict(modes), actions=dict(actions),
        decision_summary=compact, raw_depth_floor_reconstructions=fits,
        raw_sensor_model_audit_reexecuted=False, full_training_ancestry_reexecuted=False,
        original_physical_prefix_comparison_reexecuted=False, unexecuted_actions_evaluated=False,
        policy_selected=False, native_execution=False, new_sensor_data_collected=False,
        navigation_qualified=False, real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
    print('HOLD_DIAGNOSIS_VERIFIED', digest(OUTPUT), len(sources), count, len(events), flush=True)
    print('TERMINAL_CORRECTION', fits[-1]['normal_translation_correction_m'], fits[-1]['normal_alignment_rad'], flush=True)


if __name__ == '__main__': main()
