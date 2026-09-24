"""Blinded physics-only pilot branches from fresh native source snapshots."""
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
from PIL import Image

from lewm.decision_headroom_snapshot_development import restore
from lewm.decision_headroom_reference_development import ReferenceGeometry
from lewm.physical_execution_development import rotation_xyzw
from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
from scripts.whole_task_physics_session_development import GeometryFreePhysicalSample


ACTIONS = ('hold', 'forward', 'left_arc', 'right_arc', 'left_turn', 'right_turn')
TRACE_FIELDS = ('timestamp_s', 'base_pose_world', 'base_twist_world', 'joint_position',
                'joint_velocity', 'physics_contact', 'requested_command', 'applied_command')
RECORD_LISTS = ('samples', 'packets', 'packet_times', 'contact_events', 'sensor_rows',
                'fast_rows', 'guard_rows')


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def load_bound(path, binding):
    raw = path.read_bytes()
    if len(raw) != binding['bytes'] or hashlib.sha256(raw).hexdigest() != binding['sha256']:
        raise ValueError('fresh pilot artifact differs from its recorded binding')
    # Only locally created, hash-bound pilot snapshots are deserialized.
    return pickle.loads(raw)


def swept_clearance(arrays, geometry, origin_ns):
    """Geometry-only physical evidence, with no cost or method ranking."""
    xy = arrays['base_pose_world'][:, :2]
    margin = geometry.footprint_clearance(xy)
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    lower = np.minimum(margin[:-1], margin[1:]) - step / 2
    return dict(schema='decision_headroom_swept_disk_clearance.v1',
        offset_ns=(np.rint(arrays['timestamp_s']*1e9).astype(np.int64)-origin_ns).tolist(),
        sampled_clearance_m=margin.tolist(), segment_lower_bound_m=lower.tolist(),
        minimum_sampled_clearance_m=float(margin.min()),
        minimum_interpolated_clearance_lower_bound_m=float(lower.min()) if len(lower) else float(margin[0]),
        footprint_radius_m=geometry.radius, minimum_reference_clearance_m=geometry.clearance,
        interpolation='straight centre segment between native 2-ms samples; 1-Lipschitz bound',
        continuous_between_physics_steps_qualified=False,
        disallowed_contact_observed=bool(arrays['physics_contact'].any()),
        evaluator_only=True, cost_or_method_comparison=False)


def execute(session, snapshot, requested, output, budget, *, identity, geometry, expected_applied=None):
    attempt_started, cpu_started = time.monotonic(), time.process_time()
    if np.asarray(requested).shape != (40, 3):
        raise ValueError('exact 800-ms tape at the unchanged 20-ms service cadence required')
    # Admit the whole recording before native work; avoid rescanning all
    # retained files for each of the sixteen small images.
    budget.admit_write(16 * (2 * 640 * 480 * 3 + 4096) + 1024**2)
    budget.reserve_branch(identity)
    output.mkdir(exist_ok=False)
    restore_started = time.monotonic()
    restore(session, snapshot)
    restore_wall_s = time.monotonic()-restore_started
    for name in RECORD_LISTS:
        if hasattr(session, name):
            setattr(session, name, [])
    origin = int(session.ctx.runner._sim_time_ns)
    command = session.ctx.runner._last_executed[0]
    initial = GeometryFreePhysicalSample._sample(session, command, command, origin / 1e9)
    started, applied, images = time.monotonic(), [], []
    terminal = None
    try:
        for tick, request in enumerate(requested):
            budget.check('branch_step')
            actual = session.command_policy_step(request)
            applied.append(actual)
            if expected_applied is not None:
                np.testing.assert_allclose(actual, expected_applied[tick], rtol=0, atol=1e-7)
            if (tick + 1) % 5 == 0:
                offset_ms = (tick + 1) * 20
                pixels, transforms = session._render_pair()
                bindings = {}
                for label, pair in zip(('primary', 'auxiliary'), pixels, strict=True):
                    rgb = pair[0]
                    if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                        raise ValueError('unchanged native RGB raster required')
                    path = output / f'{label}_{offset_ms:03d}ms.png'
                    Image.fromarray(rgb).save(path, compress_level=1)
                    bindings[label] = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        pixels_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
                images.append(dict(offset_ms=offset_ms, images=bindings, transforms=transforms))
    except PhysicalStop as exc:
        terminal = dict(kind='PHYSICAL_STOP', reason=str(exc))
    except BaseException as exc:
        terminal = dict(kind='TECHNICAL_FAILURE', reason=repr(exc))
        raise
    finally:
        rows = [initial, *session.samples]
        arrays = {key:np.stack([r[key] for r in rows]) for key in TRACE_FIELDS}
        # A latched resource stop must still preserve its small failure trace.
        # Normal writes are admitted first; the owner reserves 1 MiB closeout.
        if not budget.stopped:
            budget.admit_write(2 * sum(a.nbytes for a in arrays.values()) + 128 * 1024)
        with (output / 'physics_trace.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        save(output / 'contact_events.json', session.contact_events)
        save(output / 'commands.json', dict(requested=np.asarray(requested).tolist(), applied=applied,
            partial_final_step_retained_in_physics_trace=True))
        save(output / 'frames.json', images)
        save(output / 'swept_clearance.json', swept_clearance(arrays, geometry, origin))
        save(output / 'result.json', dict(identity=identity, terminal=terminal,
            physics_samples=len(rows), requested_policy_steps=len(requested), completed_policy_steps=len(applied),
            actual_horizon_ms=(session.ctx.runner._sim_time_ns-origin)/1e6,
            completed=terminal is None and len(applied)==40, wall_s=time.monotonic()-started,
            restore_wall_s=restore_wall_s, attempt_wall_s=time.monotonic()-attempt_started,
            process_cpu_s=time.process_time()-cpu_started,
            reference_cost_computed=False, method_selections_or_rankings_computed=False,
            raw_depth_or_dense_features_retained=False))
        budget.check('branch_persisted')
    return arrays, terminal


def compare_trace(actual, expected, origin_ns, tolerances):
    actual_ns = np.rint(actual['timestamp_s'] * 1e9).astype(np.int64)
    expected_ns = np.rint(expected['timestamp_s'] * 1e9).astype(np.int64)
    rows = []
    for offset in range(0, 800_000_001, 100_000_000):
        a = np.flatnonzero(actual_ns == origin_ns + offset)
        e = np.flatnonzero(expected_ns == origin_ns + offset)
        if len(a) != 1 or len(e) != 1:
            rows.append(dict(offset_ms=offset//1_000_000, valid=False, reason='MISSING_OR_DUPLICATE_PHYSICAL_SAMPLE'))
            continue
        ap, ep = actual['base_pose_world'][a[0]], expected['base_pose_world'][e[0]]
        ar, er = rotation_xyzw(ap[3:]), rotation_xyzw(ep[3:])
        difference = np.arctan2(ar[1, 0], ar[0, 0]) - np.arctan2(er[1, 0], er[0, 0])
        yaw = abs(float(np.arctan2(np.sin(difference), np.cos(difference))))
        position = float(np.linalg.norm(ap[:3] - ep[:3]))
        ac = bool(actual['physics_contact'][actual_ns <= origin_ns + offset].any())
        ec = bool(expected['physics_contact'][(expected_ns >= origin_ns) & (expected_ns <= origin_ns + offset)].any())
        passed = position <= tolerances['position_m'] and yaw <= tolerances['yaw_rad'] and ac == ec
        rows.append(dict(offset_ms=offset//1_000_000, valid=True, position_error_m=position,
            yaw_error_rad=yaw, contact_match=ac==ec, passed=passed))
    return dict(passed=all(r.get('passed',False) for r in rows), horizons=rows)


def run(session, source_root, budget, *, tolerances):
    """Called only after the source owner has stopped its controller and persisted it."""
    session.physics_clock_callback = None
    metadata = json.loads((source_root / 'snapshots.json').read_text())
    requests = json.loads((source_root / 'requests.json').read_text())
    with np.load(source_root / 'native/physics_trace.npz', allow_pickle=False) as archive:
        expected = {k:archive[k].copy() for k in TRACE_FIELDS}
    spec = json.loads((source_root / 'specification.json').read_text())
    walls = [dict(center=w['centre_xyz'][:2], size=w['size_xyz'][:2], yaw=w['yaw_rad'])
             for w in spec['geometry']['wall_boxes']]
    # The surrounding walls define the maze. The extra reference boundary is
    # outside their extent and cannot replace a missing physical obstacle.
    extents = np.array([w['center'] for w in walls])
    bounds = np.array([extents.min(axis=0)-2., extents.max(axis=0)+2.])
    parameters = json.loads(Path('docs/go2_decision_headroom_reference_sanity_v1_2026-09-23.json').read_text())['draft_cost_parameters']
    geometry = ReferenceGeometry(walls, bounds, spec['geometry']['spawn_se2_world'][:2],
        radius_m=parameters['footprint_radius_m'], clearance_m=parameters['minimum_clearance_m'],
        resolution_m=parameters['geodesic_grid_resolution_m'])
    source_images = {r['measured_ns']:r for r in json.loads(
        (source_root / 'native/in_memory_camera_observations.json').read_text())['frames']}
    state_results = []
    for state in metadata:
        root = source_root / f"state_{state['frame']:04d}"
        snapshot = load_bound(root / state['physical']['path'], state['physical'])
        decision = load_bound(root / state['decision']['path'], state['decision'])
        if tuple(decision['candidate_order']) != ACTIONS:
            raise ValueError('source candidate ordering differs from branch labels')
        stamp = state['measured_ns']
        source_trace = [r for r in requests if stamp <= r['simulator_ns'] < stamp + 800_000_000]
        if [r['simulator_ns'] for r in source_trace] != list(range(stamp, stamp + 800_000_000, 20_000_000)):
            save(root / 'qualification_failure.json', dict(reason='INCOMPLETE_SOURCE_APPLIED_TRACE'))
            state_results.append(dict(frame=state['frame'], restoration_passed=False))
            continue
        source_applied = np.asarray([r['applied_command'] for r in source_trace])
        replay_results = []
        for repeat in range(3):
            identity = f"{source_root.name}/state_{state['frame']:04d}/source_trace_{repeat}"
            actual, terminal = execute(session, snapshot, source_applied, root / f'source_trace_{repeat}',
                budget, identity=identity, geometry=geometry, expected_applied=source_applied)
            frames = json.loads((root / f'source_trace_{repeat}/frames.json').read_text())
            image_match = [dict(offset_ms=r['offset_ms'], **{
                label:r['images'][label]['pixels_sha256'] == source_images[stamp+r['offset_ms']*1_000_000]['pixel_sha256'][label]['rgb_sha256']
                for label in ('primary', 'auxiliary')}) for r in frames]
            replay_results.append(compare_trace(actual, expected, stamp, tolerances) | dict(
                terminal=terminal, rgb_bitwise_matches=image_match,
                rgb_bitwise_restored=len(image_match)==8 and all(r['primary'] and r['auxiliary'] for r in image_match)))
        passed = all(r['passed'] and r['terminal'] is None for r in replay_results)
        save(root / 'restoration.json', dict(passed=passed, repeats=replay_results,
            tolerances=tolerances, full_recorded_applied_trace_reexecuted=True,
            systematic_mismatch_not_used_as_repeat_noise=True))
        if not passed:
            state_results.append(dict(frame=state['frame'], restoration_passed=False))
            continue
        branch_results = []
        for action_index, action in enumerate(ACTIONS):
            tape = np.repeat(decision['candidate_requested_commands'][action_index], 5, axis=0)
            projected = np.repeat(decision['candidate_applied_commands'][action_index], 5, axis=0)
            first = None
            comparisons, pairwise, previous = [], [], []
            for repeat in range(3):
                identity = f"{source_root.name}/state_{state['frame']:04d}/{action}/{repeat}"
                actual, terminal = execute(session, snapshot, tape, root / f'{action}_{repeat}',
                    budget, identity=identity, geometry=geometry, expected_applied=projected)
                if first is None:
                    first = actual
                comparisons.append(compare_trace(actual, first, stamp, tolerances) | dict(terminal=terminal))
                for index, (prior, prior_terminal) in enumerate(previous):
                    pairwise.append(compare_trace(actual, prior, stamp, tolerances) | dict(
                        repeat_a=index, repeat_b=repeat, terminal_a=prior_terminal, terminal_b=terminal,
                        same_terminal=terminal==prior_terminal,
                        same_final_recorded_time=bool(actual['timestamp_s'][-1]==prior['timestamp_s'][-1])))
                previous.append((actual,terminal))
            branch_results.append(dict(action=action, repeat_comparisons=comparisons,
                pairwise_repeat_comparisons=pairwise))
        save(root / 'repeat_variability.json', dict(branches=branch_results,
            comparator='first repeat of same physical state and candidate',
            source_restoration_comparator_separate=True, cost_or_method_comparison=False))
        state_results.append(dict(frame=state['frame'], restoration_passed=True))
    save(source_root / 'branch_pilot_result.json', dict(states=state_results,
        comparative_rows_computed=False, phase2_authorized=False))
