"""Consolidate the four fixed outcomes and plot evaluator-only physical paths."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from scripts import run_go2_return_routing_memory_development as study


def main():
    output = Path('docs/go2_return_routing_memory_result_2026-09-17.json')
    figures = [Path(f'docs/go2_return_routing_memory_trajectories_2026-09-17.{suffix}')
        for suffix in ('png', 'svg')]
    assert not any(p.exists() for p in [output, *figures])
    plan = json.loads(study.PLAN.read_text())
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for number, (index, arm) in enumerate(study.ASSIGNMENTS, 1):
        root = study.BASE / study.root_name(number)
        read = lambda name: json.loads((root / name).read_text())
        outcome = read(study.READOUT)
        launch = read('launch.json')
        assert launch['extra_sources'] == plan['source_sha256']
        assert launch['neural_snapshot']['model_state_sha256'] == plan['models'][arm]['model_sha256']
        assert launch['frozen_layout_inventory_sha256'] == plan['inventory_sha256']
        assert launch['assigned_return_routing_scope'] == study.RUNTIMES[arm].return_scope
        actual = read('actual_controller_treatment_v1.json')
        assert actual['actual_treatment_verified']
        mission = read('mission.json')
        goal = next(a for a in outcome['navigation']['arrivals'] if a['phase'] == 'OUTBOUND')
        assert goal['arrival_checks_passed']
        home = next((a for a in outcome['navigation']['arrivals'] if a['phase'] == 'RETURN'), None)
        metadata = read('native/in_memory_camera_observations.json')
        frames = {r['frame']: r for r in metadata['frames']}
        end = home['frame'] if home is not None else mission[-1]['frame']
        elapsed = (mission[end]['measured_ns'] - mission[goal['frame']]['measured_ns']) / 1e9
        plans = read('planning.json')
        row = dict(assignment=number, layout=index, arm=arm, root=str(root),
            goal_verified=True, round_trip=outcome['navigation']['round_trip'],
            simulation_s=outcome['navigation']['simulation_s'],
            return_elapsed_s=elapsed, return_elapsed_is_censored=home is None,
            contacts=outcome['navigation']['contacts'],
            goal_frame=goal['frame'], home_frame=None if home is None else home['frame'],
            plans=outcome['navigation']['plans'], plans_on_time=outcome['navigation']['plans_on_time'],
            pipeline_faults=outcome['pipeline_faults'],
            local_turn_memory_plans=outcome['route_turn_memory_plans'],
            memory_scope=outcome['return_memory_execution'],
            physical_backtracking=outcome['physical_backtracking'],
            view_budget_exhausted_entries=sum(p.get('reason') == 'VIEW_BUDGET_EXHAUSTED' for p in plans),
            matched_xy=read('saved_short_pulse_same_window_xy_v1.json')['rmse_mm'])
        rows.append(row)
        with np.load(root / 'native/physics_trace.npz', allow_pickle=False) as data:
            physics = data['base_pose_world']
        origin = physics[frames[0]['physical_sample_index']]
        target = origin[:3] + rotation_xyzw(origin[3:]) @ np.r_[launch['public_mission']['goal_initial_body_xy_m'], 0.]
        ax = axes[index, 0 if arm == 'persistent_return' else 1]
        for wall in read('native/camera_setup_identity.json')['environment']['physical_geometries']:
            if wall['geom_type'] != 'BOX':
                continue
            half = np.asarray(wall['data'][:2]) / 2
            local = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
            quat = np.asarray(wall['quaternion_world_wxyz'])
            rotation = rotation_xyzw(quat[[1, 2, 3, 0]])
            points = np.c_[local, np.zeros(4)] @ rotation.T + np.asarray(wall['position_world_m'])
            ax.add_patch(Polygon(points[:, :2], facecolor='#586270', edgecolor='none'))
        for first, last, color, label, width in (
                (0, goal['frame'], '#4793c4', 'Outbound', 1.4),
                (goal['frame'], end, '#d46025', 'Return', 2.0)):
            xy = physics[[frames[f]['physical_sample_index'] for f in range(first, last + 1)], :2]
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=width, label=label)
        ax.scatter(*origin[:2], s=65, color='#168058', edgecolor='white', zorder=4, label='Home')
        ax.scatter(*target[:2], s=140, marker='*', color='#f4c63d', edgecolor='#725c14', zorder=4, label='Goal')
        if home is None:
            endpoint = physics[frames[end]['physical_sample_index'], :2]
            ax.scatter(*endpoint, marker='x', s=75, color='#ad2929', zorder=5, label='Stopped')
        condition = 'Persistent return map' if arm == 'persistent_return' else 'Latest-pair return map'
        result = f'Home reached in {elapsed:.1f} s after goal' if home else f'No home arrival after {elapsed:.1f} s'
        ax.set_title(f'Layout {index + 1}: {condition}\n{result}', fontsize=11)
        ax.set_xlabel('World x (m)'); ax.set_ylabel('World y (m)')
        ax.set_aspect('equal'); ax.grid(alpha=.15); ax.margins(.06)
    for pair in axes:
        xmin = min(ax.get_xlim()[0] for ax in pair); xmax = max(ax.get_xlim()[1] for ax in pair)
        ymin = min(ax.get_ylim()[0] for ax in pair); ymax = max(ax.get_ylim()[1] for ax in pair)
        for ax in pair:
            ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(.5, .028), frameon=False)
    fig.suptitle('Remembered routing geometry and the return journey', fontsize=15)
    fig.text(.5, .015, 'Same JEPA model; persistent outbound maps; zero contacts. Physical traces used only for evaluation.',
        ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .955))
    for path in figures:
        fig.savefig(path, dpi=170)
    plt.close(fig)
    result = dict(schema='return_routing_memory_complete_result.v1', status='complete',
        assignments_completed=4, prospective_layouts=2, rows=rows,
        totals={arm: dict(goals=sum(r['goal_verified'] for r in rows if r['arm'] == arm),
            round_trips=sum(r['round_trip'] for r in rows if r['arm'] == arm),
            contacts=sum(r['contacts'] for r in rows if r['arm'] == arm)) for arm in study.RUNTIMES},
        source_and_model_assignments_verified=True,
        supports_return_routing_memory_in_this_controller=True,
        outbound_trajectories_matched=False, model_internal_memory_ablated=False,
        fully_memoryless_comparison=False, jepa_superiority_established=False,
        broad_reliability_established=False, hardware_validated=False,
        limitations=['two same-family layouts and one execution per condition per layout',
            'asynchronous outbound trajectories and deadline fractions differ',
            'routing geometry is ablated; predictive obstacle history and other temporal state remain',
            'ideal gyro and 2mm synthetic depth noise; physics pauses during computation'],
        figures=[str(p) for p in figures])
    with output.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(dict(totals=result['totals'], rows=[{k: r[k] for k in (
        'assignment', 'arm', 'layout', 'round_trip', 'simulation_s', 'return_elapsed_s',
        'local_turn_memory_plans')} for r in rows], figures=result['figures'])), flush=True)


if __name__ == '__main__':
    main()
