"""Evaluator-only native dwell and traversed-edge evidence for eight independent development mazes.

This module must never be imported by a controller or sensor packet adapter.
Its numerical checks do not replace raw sensor/command/visibility auditing.
"""
import numpy as np
from lewm.physical_execution_development import rotation_xyzw
from lewm.novel_maze_round_trip_scene_development import (
    CELLS, START, PITCH_M, edge, evaluator_route)
from lewm.independent_round_trip_layouts_development import specification, public_mission
from lewm.observed_round_trip_mission_development import MAX_NAVIGATION_TICKS


def graph(index):
    """Evaluator-only exact graph from the complete fixed new inventory."""
    spec = specification(index)
    return tuple(tuple(tuple(c) for c in e) for e in spec["evaluation_layout"]["edges"])


def loop_erased(cells):
    path = []
    for cell in cells:
        if cell in path: path = path[:path.index(cell)+1]
        else: path.append(cell)
    return path


def traversed_edges(xy_world_m, layout_index, *, first_sample):
    xy = np.asarray(xy_world_m, float)
    if xy.ndim != 2 or xy.shape[1] != 2 or not len(xy) or not np.isfinite(xy).all():
        raise ValueError('finite nonempty native XY trajectory required')
    links = set(graph(layout_index)); cells = np.floor(xy/PITCH_M+.5).astype(int)
    visits = [tuple(cells[0])]; crossings = []; invalid = []
    for i in np.flatnonzero(np.any(cells[1:] != cells[:-1], axis=1))+1:
        a, b = tuple(cells[i-1]), tuple(cells[i])
        valid = a in CELLS and b in CELLS and edge(a, b) in links
        row = dict(sample_index=int(first_sample+i), from_cell=list(map(int, a)),
            to_cell=list(map(int, b)), declared_open_edge=valid)
        crossings.append(row)
        if not valid: invalid.append(row)
        visits.append(b)
    maximum_step = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).max(initial=0.))
    return dict(crossings=crossings, invalid_crossings=invalid,
        visited_cells=[list(map(int, c)) for c in visits],
        loop_erased_cells=[list(map(int, c)) for c in loop_erased(visits)],
        all_positions_in_maze=bool(all(tuple(c) in CELLS for c in cells)),
        maximum_native_step_m=maximum_step,
        native_step_bound_pass=maximum_step <= .3*.002+1e-7,
        evaluator_only=True, collision_clearance_certified=False)


def evaluate(raw, mission_receipt, collection, *, layout_index):
    poses = np.asarray(raw['base_pose_world'], float)
    times = np.asarray(raw['timestamp_s'], float)
    twists = np.asarray(raw['base_twist_world'], float)
    requests = np.asarray(raw['requested_command'], float)
    n = len(poses)
    if (poses.shape != (n, 7) or twists.shape != (n, 6) or requests.shape != (n, 3)
            or times.shape != (n,) or not 750 <= n <= 750+50*(3+MAX_NAVIGATION_TICKS+10)
            or not all(np.isfinite(a).all() for a in (poses, times, twists, requests))
            or not np.allclose(times, np.arange(1, n+1)*.002, atol=5e-9, rtol=0)
            or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('complete bounded native 2ms trajectory and command arrays required')
    contacts = np.asarray(raw['physics_contact'])
    if contacts.shape != (n,): raise ValueError('per-physics-sample contact evidence required')
    instructions = public_mission(layout_index)
    local = (poses[:, :3]-poses[749, :3])@rotation_xyzw(poses[749, 3:])
    arrivals = mission_receipt['arrivals']
    if not isinstance(arrivals, list) or len(arrivals) > 2:
        raise ValueError('at most two ordered observed arrival claims required')
    windows = []; previous_frame = -1
    for i, arrival in enumerate(arrivals):
        phase = ('OUTBOUND', 'RETURN')[i]
        target = instructions['goal_initial_body_xy_m' if i == 0 else 'return_initial_body_xy_m']
        frame = arrival['frame']
        if (type(frame) is not int or not previous_frame < frame <= 3+MAX_NAVIGATION_TICKS
                or arrival['phase'] != phase or arrival['target_initial_body_xy_m'] != target
                or arrival['measured_ns'] != 1_500_000_000+frame*100_000_000
                or arrival['quiet_intervals'] != 10 or arrival['native_verified'] is not False):
            raise ValueError('original ordered observed arrival identity required')
        end = 749+50*frame
        if end >= n or end-500 < 749: raise ValueError('complete observed arrival window required')
        distance = np.linalg.norm(local[end-500:end+1, :2]-target, axis=1)
        speed = np.linalg.norm(twists[end-500:end+1, :3], axis=1)
        quiet = bool((distance <= .06).all() and (speed <= .05).all()
            and (requests[end-499:end+1] == 0.).all())
        windows.append(dict(phase=phase, frame=frame, end_sample=end,
            maximum_goal_distance_m=float(distance.max()), maximum_speed_m_s=float(speed.max()),
            native_one_second_arrival_and_quiet_pass=quiet))
        previous_frame = frame
    outbound = returning = None; backtrack = False
    if windows:
        split = windows[0]['end_sample']
        outbound = traversed_edges(poses[749:split+1, :2], layout_index, first_sample=749)
        end = windows[1]['end_sample'] if len(windows) == 2 else n-1
        returning = traversed_edges(poses[split:end+1, :2], layout_index, first_sample=split)
        expected = [list(c) for c in evaluator_route(graph(layout_index))]
        backtrack = bool(len(windows) == 2
            and outbound['loop_erased_cells'] == expected
            and returning['loop_erased_cells'] == list(reversed(expected))
            and all(r['all_positions_in_maze'] and r['native_step_bound_pass']
                and not r['invalid_crossings'] for r in (outbound, returning)))
    else:
        outbound = traversed_edges(poses[749:, :2], layout_index, first_sample=749)
    terminal_distance = np.linalg.norm(local[-501:, :2], axis=1)
    terminal_quiet = bool(n >= 1250 and (terminal_distance <= .06).all()
        and (np.linalg.norm(twists[-501:, :3], axis=1) <= .05).all()
        and (requests[-500:] == 0.).all())
    candidate = bool(len(windows) == 2 and all(w['native_one_second_arrival_and_quiet_pass'] for w in windows)
        and backtrack and terminal_quiet and not contacts.any()
        and collection['schedule_terminal'] == mission_receipt['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE'
        and collection['terminal_zero_ticks'] == 10
        and collection['physical_stop'] is None and collection['acquisition_stop'] is None)
    return dict(arrival_windows=windows, outbound_traversal=outbound, return_traversal=returning,
        physically_retraced_outbound_route=backtrack, terminal_native_quiet_pass=terminal_quiet,
        native_round_trip_candidate_pass=candidate, verified_round_trip=False,
        requires_raw_sensor_command_and_visibility_audit=True, evaluator_only=True,
        controller_native_state_access=False, independent_navigation_claim=False)
