"""Locate excluded frontiers in the retained failed trajectory's observed map."""
import argparse
from collections import deque
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from lewm.current_pair_routing_memory_development import CurrentPairRoutingSnapshot
from lewm.observed_floor_waypoint_development import centre, inflated_cells, NEIGHBOURS
from scripts.compare_continuous_navigation_arms_development import path, read
from scripts.run_go2_contact_score_ablation_development import ContactScoreRuntime


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root-name', required=True)
    parser.add_argument('--frames', type=int, nargs=2, default=(1204, 3736))
    args = parser.parse_args(); root = path(args.root_name)
    output = root / 'frontier_map_reconstruction_v1'
    probes = read(output, 'probes.json')
    runtime = ContactScoreRuntime.__new__(ContactScoreRuntime)
    runtime.mission_latest = dict(phase='OUTBOUND')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    results = []
    for ax, frame in zip(axes, args.frames, strict=True):
        probe = next(r for r in probes if r['planning_frame'] == frame)
        values = read(output, f'map_{probe["map_frame"]:04d}.json')
        for key in ('floor', 'occupied', 'fine_occupied', 'current_floor',
                'current_occupied', 'current_fine_occupied'):
            values[key] = frozenset(tuple(c) for c in values[key])
        snapshot = CurrentPairRoutingSnapshot(**values)
        available = snapshot.floor - inflated_cells(snapshot.occupied)
        seed = tuple(probe['no_exclusions']['route_cells'][0])
        reached = {seed}; queue = deque([seed])
        while queue:
            c = queue.popleft()
            for dx, dy in NEIGHBOURS:
                n = c[0]+dx, c[1]+dy
                if n in available and n not in reached: reached.add(n); queue.append(n)
        observed = snapshot.floor | snapshot.occupied
        frontier = {c for c in reached if any((c[0]+dx, c[1]+dy) not in observed
            for dx, dy in NEIGHBOURS)}
        excluded = {tuple(c) for c in probe['excluded_cells']}
        upper = {c for c in excluded if 0 < centre(c)[0] < .7 and 2.3 < centre(c)[1] < 2.8}
        runtime.frontier_visits = SimpleNamespace(excluded=excluded-upper)
        route = runtime._routing_proposer(snapshot)(snapshot.floor, snapshot.occupied,
            probe['position_map_xy_m'], probe['no_exclusions']['goal_map_xy_m'])
        # Point inside the already observed upper passage, beyond the small gap.
        passage_cell = (30, 52)
        results.append(dict(planning_frame=frame, map_frame=snapshot.frame,
            reachable_cells=len(reached), reachable_frontier_cells=sorted(frontier),
            excluded_reachable_frontiers=sorted(frontier & excluded),
            remaining_frontiers=sorted(frontier-excluded),
            upper_exclusions_removed=sorted(upper), upper_only_unexcluded_route=route,
            upper_passage_probe_cell=list(passage_cell),
            upper_passage_probe_observed_floor=passage_cell in snapshot.floor,
            upper_passage_probe_coarse_traversable=passage_cell in available,
            upper_passage_probe_in_robot_component=passage_cell in reached,
            retrospective_queries_only=True))
        for cells, color, label in ((snapshot.floor, '#c5dcf1', 'Observed floor'),
                (available-reached, '#bab6d9', 'Other traversable components'),
                (reached, '#429777', 'Robot-connected traversable floor'),
                (snapshot.occupied, '#303030', 'Observed obstacles')):
            if cells:
                a=(np.asarray(sorted(cells))+.5)*.05
                ax.scatter(*a.T, marker='s', s=5, c=color, label=label)
        if excluded:
            a=(np.asarray(sorted(excluded))+.5)*.05
            ax.scatter(*a.T, marker='x', s=18, c='#d97913', label='Excluded frontier targets')
        if route['route_cells']:
            a=(np.asarray(route['route_cells'])+.5)*.05
            ax.plot(*a.T, color='#bb275c', lw=1.7, label='Query with upper exclusions removed')
        ax.scatter(*probe['position_map_xy_m'], c='red', s=45, zorder=5, label='Recorded robot pose')
        ax.scatter(*centre(passage_cell), marker='*', c='#6b41bb', s=85, label='Observed upper passage')
        ax.set(xlim=(-.8,4.7), ylim=(-2.2,3.5), xlabel='Map x (m)', ylabel='Map y (m)',
            title=f'Frame {frame}: {len(frontier-excluded)} unexcluded reachable frontiers')
        ax.set_aspect('equal')
    axes[1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1, .5))
    fig.suptitle('Saved sensor-map reconstruction — route queries, no alternative execution')
    fig.tight_layout()
    for suffix in ('png', 'svg'): fig.savefig(output/f'frontier_exclusion_diagnosis.{suffix}', dpi=160)
    with (output/'exclusion_diagnosis.json').open('x') as stream: json.dump(results, stream, indent=2)
    print(json.dumps([{k:v for k,v in r.items() if k not in
        ('upper_only_unexcluded_route','reachable_frontier_cells','excluded_reachable_frontiers',
        'remaining_frontiers','upper_exclusions_removed')} | dict(
            frontiers=len(r['reachable_frontier_cells']), excluded=len(r['excluded_reachable_frontiers']),
            upper_unexcluded_target=r['upper_only_unexcluded_route'].get('target_map_xy_m')) for r in results]))


if __name__ == '__main__': main()
