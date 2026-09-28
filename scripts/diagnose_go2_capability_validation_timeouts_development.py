"""Timeout diagnosis for capability qualification: slow-but-progressing versus stuck.

Read-only over preserved records. For each timed-out mission: remaining true-map
shortest-path distance to the active target at 480 s (pre-registered 0.46-m
inflation + 5-mm clearance, 20-mm grid), time split into translate / turn-only /
hold from applied 20-ms commands, selected-hold fraction, and 30-s windows of
geodesic progress, as in the paired-floor leg diagnosis. The budget is unchanged.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from lewm.decision_headroom_reference_development import ReferenceGeometry

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO/'docs/go2_navigation_capability_completed_support_v4_2026-09-27.json'
# Descriptive rule fixed before reading any timeout: "progressing" if the active-target
# geodesic distance fell by at least 0.25 m over the final 120 s, otherwise "stuck".
FINAL_WINDOW_S, PROGRESS_M = 120., .25


def diagnose(root):
    read = lambda n: json.loads((root/n).read_text())
    ep, ev, plans, requests, spec = (read(n) for n in ('episode.json', 'episode_evaluation.json', 'planning.json', 'requests.json', 'specification.json'))
    with np.load(root/'native/physics_trace.npz', allow_pickle=False) as f:
        t = f['timestamp_s']-1.5
        native = f['base_pose_world'].copy()
    end = float(t[-1])
    walls = [dict(center=w['centre_xyz'][:2], size=w['size_xyz'][:2], yaw=w['yaw_rad']) for w in spec['geometry']['wall_boxes']]
    geometry = lambda xy: ReferenceGeometry(walls, [[-2.1, -2.1], [3.4, 3.4]], xy, radius_m=.46, clearance_m=.005, resolution_m=.02)
    geos = {'OUTBOUND': geometry(ep['beacon_xy_world']), 'RETURN': geometry(ep['home_se2_world'][:2])}
    logged = {r['phase']: r for r in ev['arrivals']}
    beacon = t[np.searchsorted(t, logged['OUTBOUND']['frame']*.1)] if 'OUTBOUND' in logged else None
    qt = np.array([r['simulator_ns']/1e9-1.5 for r in requests])
    q = np.array([r['applied_command'] for r in requests])
    qend = np.minimum(qt+.02, end)
    moving = np.any(q != 0, axis=1)
    translating = np.any(q[:, :2] != 0, axis=1)

    def pose(s):
        return native[min(np.searchsorted(t, s), len(t)-1)]

    snapped = []

    def remaining(leg, s):
        point = pose(s)[:2]
        value = geos[leg].distance_and_heading(point)
        if value.get('valid'):
            return float(value['distance_m'])
        # Robot centre inside the inflated wall margin: snap to the nearest valid
        # point within 0.5 m and add the offset (recorded, conservative upper bound).
        best = None
        for r in np.arange(.02, .52, .02):
            for a in np.linspace(0, 2*np.pi, 32, endpoint=False):
                v = geos[leg].distance_and_heading(point+r*np.array([np.cos(a), np.sin(a)]))
                if v.get('valid'):
                    best = min(best or np.inf, v['distance_m']+r)
            if best is not None:
                snapped.append(dict(leg=leg, time_s=round(float(s), 2), offset_m=round(float(r), 2), reason=value['reason']))
                return float(best)
        raise ValueError(f'no reference geodesic near {point} ({value["reason"]})')

    def window(lo, hi, leg):
        dt = np.maximum(0, np.minimum(qend, hi)-np.maximum(qt, lo))
        selected = [r for r in plans if 'selection' in r and lo <= r['measured_ns']/1e9-1.5 < hi]
        holds = sum(r['action'] == 'hold' for r in selected)
        points = native[(t >= lo) & (t <= hi), :2]
        return dict(leg=leg, start_s=round(lo, 2), end_s=round(hi, 2), translating_s=float(dt[translating].sum()),
            turn_only_s=float(dt[moving & ~translating].sum()), holding_s=float(dt[~moving].sum()),
            hold_time_fraction=float(dt[~moving].sum()/max(hi-lo, 1e-9)), selected_decisions=len(selected), selected_holds=holds,
            selected_hold_fraction=holds/len(selected) if selected else None,
            actions=dict(Counter(r['action'] for r in selected)), routes=dict(Counter(r.get('route_status') for r in selected)),
            path_m=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.,
            net_displacement_m=float(np.linalg.norm(pose(hi)[:2]-pose(lo)[:2])),
            remaining_start_m=remaining(leg, lo), remaining_end_m=remaining(leg, hi))
    legs = {'OUTBOUND': window(0., beacon if beacon is not None else end, 'OUTBOUND')}
    if beacon is not None:
        legs['RETURN'] = window(beacon, end, 'RETURN')
    active = 'RETURN' if beacon is not None else 'OUTBOUND'
    windows = [window(float(lo), min(float(lo)+30, row['end_s']), leg)
               for leg, row in legs.items() for lo in np.arange(row['start_s'], row['end_s']-1e-8, 30)]
    # Longest run of consecutive 30-s windows with < 0.1 m geodesic progress (stuck intervals).
    longest = run = 0.
    for w in windows:
        run = run+(w['end_s']-w['start_s']) if w['remaining_start_m']-w['remaining_end_m'] < .1 else 0.
        longest = max(longest, run)
    final = remaining(active, end)
    earlier = remaining(active, max(legs[active]['start_s'], end-FINAL_WINDOW_S))
    return dict(assignment=root.name, episode_id=ep['episode_id'], controller=read('config.json')['controller'],
        beacon_reached_s=None if beacon is None else float(beacon), terminal_s=end, active_leg=active,
        shortest_outbound_m=ep['shortest_outbound_m'], shortest_return_m=ep['shortest_return_m'],
        remaining_shortest_path_at_480s_m=final, remaining_shortest_path_120s_before_end_m=earlier,
        classification='progressing' if earlier-final >= PROGRESS_M else 'stuck',
        longest_no_progress_s=longest,
        rule=f'progressing if active-target geodesic distance fell >= {PROGRESS_M} m over the final {FINAL_WINDOW_S:.0f} s',
        whole_mission_hold_time_fraction=float(sum(w['holding_s'] for w in legs.values())/end),
        whole_mission_selected_hold_fraction=float(sum(w['selected_holds'] for w in legs.values())/max(1, sum(w['selected_decisions'] for w in legs.values()))),
        legs=legs, windows=windows, snapped_distance_queries=snapped)


def main(out):
    protocol = json.loads(PROTOCOL.read_text())
    base = Path(protocol['output_root'])
    rows = []
    for root in sorted((base/'runs').glob('v4_completed_support_validation_*_attempt001')):
        ev_path, result_path = root/'episode_evaluation.json', root/'result.json'
        if not ev_path.exists():
            continue
        ev, result = json.loads(ev_path.read_text()), json.loads(result_path.read_text())
        if result['policy_steps'] >= 24000 and not ev['round_trip_success']:
            rows.append(diagnose(root))
    record = dict(schema='navigation_capability_timeout_diagnosis.v1', budget_s=480, budget_changed=False, timeouts=rows,
        method='Read-only preserved records; true-map geodesic at pre-registered inflation; applied-command time split; 30-s windows')
    if out:
        with out.open('x') as stream:
            json.dump(record, stream, indent=1)
            stream.write('\n')
    for r in rows:
        print(r['controller'], r['episode_id'], r['active_leg'], 'remaining', round(r['remaining_shortest_path_at_480s_m'], 2),
              '(120 s earlier', round(r['remaining_shortest_path_120s_before_end_m'], 2), ')', r['classification'], 'longest no-progress', r['longest_no_progress_s'],
              'hold time', round(r['whole_mission_hold_time_fraction'], 3), 'selected holds', round(r['whole_mission_selected_hold_fraction'], 3))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path)
    main(p.parse_args().out)
