"""Actual raw startup identity and observed renderer endpoints; no navigation claim."""
from itertools import islice
import numpy as np
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from scripts.maze_renderer_witness_development import audit_witnesses
from scripts.renderer_witness_dual_camera_maze_session_development import ARTIFACT
from scripts.maze_decision_stream_development import read_rows
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


def renderer_audit(directory):
    cameras = read_json(directory, 'camera_audit.json')
    depths = read_json(directory, 'depth_camera_audit.json')
    auxiliary = read_json(directory, 'auxiliary_camera_audit.json')
    captures = []
    for i, (c, d, a) in enumerate(zip(cameras, depths, auxiliary, strict=True)):
        clock = d['physics_clock_before_after_ns']
        if (clock != [a['measured_ns'], a['measured_ns']]
                or c['physical_sample_index'] != d['physical_sample_index'] or c['physical_sample_index'] != a['physical_sample_index']):
            raise ValueError('paired actual raw capture boundary required')
        captures.append(dict(frame=i, physical_sample_index=c['physical_sample_index'], measured_ns=a['measured_ns'],
            primary_world_from_optical=c['world_from_optical'], primary_rgb_sha256=c['rgb_sha256'],
            primary_native_depth_sha256=d['native_depth_sha256'], auxiliary_rgb_sha256=a['rgb_sha256'],
            auxiliary_native_depth_sha256=a['native_depth_sha256']))
    return audit_witnesses(read_json(directory, ARTIFACT), captures)


def compare(prior, current):
    hashes = []
    for directory in (prior, current):
        with np.load(directory/'physics_trace.npz', allow_pickle=False) as z:
            if any(len(z[k]) < 900 for k in z.files): raise ValueError('three complete warmup commands required')
            if directory == current and any(len(z[k]) != 900 for k in z.files):
                raise ValueError('no physics beyond the declared startup probe')
            hashes.append(fingerprint({k:z[k][:900] for k in z.files}))
    if hashes[0] != hashes[1]: raise ValueError('renderer queries changed the physical startup prefix')
    tapes = [read_json(p, 'command_tape.json') for p in (prior, current)]
    if len(tapes[1]) != 3 or any(t['requested_command'] != [0., 0., 0.] or not t['completed'] for tape in tapes for t in tape[:3]):
        raise ValueError('exactly three complete zero-command warmup intervals required')
    readers = [IntentReturnRGBDReplay(p) for p in (prior, current)]
    acquisitions = [read_json(p, 'auxiliary_camera_audit.json') for p in (prior, current)]
    checked = 0
    for i, (old, new) in enumerate(zip(islice(read_rows(prior), 3), read_rows(current), strict=True)):
        if old['tick'] != i or new['tick'] != i or old['decision'] != new['decision']:
            raise ValueError('complete startup controller decision changed')
        values = []
        for directory, reader, rows in zip((prior, current), readers, acquisitions, strict=True):
            p, d, f, now = reader.packet(i)
            image, auxiliary = packet(directory, i, p, public_acquisition(rows[i]), now_ns=now)
            native = {}
            for filename, keys in ((f'native_depth_{i:04d}.npz', ('optical_depth_m',)),
                    (f'auxiliary_depth_{i:04d}.npz', ('native_optical_depth_m', 'diagnostic_segmentation'))):
                with np.load(directory/filename, allow_pickle=False) as z:
                    native[filename] = {k:z[k] for k in keys}
            values.append(fingerprint((p, d, f, auxiliary, image, now, native)))
        if values[0] != values[1]: raise ValueError('raw capture or public packets changed under renderer queries')
        checked += 1
    if checked != 3: raise ValueError('complete three-observation startup probe required')
    return dict(frames=3, physics_samples=900, physical_prefix_sha256=hashes[0],
        complete_controller_decisions_exact=True, raw_captures_and_public_packets_exact=True,
        actual_zero_commands_exact=True, unexecuted_navigation_outcomes_inferred=False,
        renderer_witnesses=renderer_audit(current), navigation_qualified=False)
