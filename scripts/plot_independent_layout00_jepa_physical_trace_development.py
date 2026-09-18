"""Plot the closed native trace and the selected report's audit status."""
import argparse
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def main(report_path):
    report = json.loads(report_path.read_text())
    root = Path(report['root'])
    launch = json.loads((root/'launch.json').read_text())
    physical = report.get('physical_evaluation', report.get('native_evaluation'))
    verified = report['verified_round_trip'] is True
    windows = physical['arrival_windows']
    with np.load(root/report['case']/'physics_trace.npz', allow_pickle=False) as raw:
        xy = np.array(raw['base_pose_world'][:, :2], copy=True)
    assert len(windows) == 2 and physical['native_round_trip_candidate_pass']
    split = windows[0]['end_sample']
    end = windows[1]['end_sample']
    scene = launch['scene_specification']
    goal = np.asarray(scene['evaluation_layout']['goal_cell']) * scene['evaluation_layout']['pitch_m']
    start = xy[749]
    panels = (
        ('Outbound: exploration and goal arrival', 749, split, '#b45309'),
        ('Return: six valid cell crossings', split, end, '#087f8c'),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.9), sharex=True, sharey=True)
    for ax, (title, first, last, colour) in zip(axes, panels):
        for wall in scene['geometry']['wall_boxes']:
            x, y = wall['centre_xyz'][:2]
            w, h = wall['size_xyz'][:2]
            assert wall['yaw_rad'] == 0.
            ax.add_patch(Rectangle((x-w/2, y-h/2), w, h,
                facecolor='#c3c7cd', edgecolor='#777e88', linewidth=.4))
        # Plot at 20 ms resolution, retaining the exact arrival endpoint.
        samples = np.unique(np.r_[np.arange(first, last+1, 10), last])
        ax.plot(xy[samples, 0], xy[samples, 1], color=colour, lw=1.5,
            label='Executed native trajectory', zorder=3)
        ax.scatter(*start, s=50, color='#253248', edgecolor='white',
            linewidth=.8, label='Home', zorder=5)
        ax.scatter(*goal, s=130, marker='*', color='#7c3aed', edgecolor='white',
            linewidth=.6, label='Goal cell centre', zorder=5)
        ax.scatter(*xy[last], s=100, marker='o', facecolors='none',
            edgecolors=colour, linewidth=1.5, label='Arrival position', zorder=6)
        ax.set(title=title, xlabel='World x (m)', xlim=(-2.15, 3.45), ylim=(-2.15, 3.45))
        ax.set_aspect('equal')
        ax.grid(alpha=.14, linewidth=.6)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('World y (m)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .17),
        ncol=4, frameon=False, fontsize=9)
    fig.suptitle('Independent development maze 0 — full-RGB JEPA physical trajectory',
        fontsize=14, y=.98)
    subtitle = ('Verified round trip · full sensor, command and visibility audit passed' if verified else
        'Physical round-trip candidate pass · full sensor, command and visibility audit pending')
    fig.text(.5, .93, subtitle,
        ha='center', fontsize=10, color='#4b5563')
    out, back = windows
    summary = (
        f"Arrival-window maxima: outbound {100*out['maximum_goal_distance_m']:.2f} cm / "
        f"{100*out['maximum_speed_m_s']:.2f} cm/s; return "
        f"{100*back['maximum_goal_distance_m']:.2f} cm / "
        f"{100*back['maximum_speed_m_s']:.2f} cm/s. Limits: 6 cm / 5 cm/s.\n"
        '10 outbound and 6 return crossings; zero invalid crossings; zero recorded contact flags.\n'
        'One independent development layout. Physics paused during computation; current simulation sensing assumptions.'
    )
    fig.text(.5, .10, summary, ha='center', va='center', fontsize=9, linespacing=1.65)
    fig.subplots_adjust(left=.07, right=.98, top=.87, bottom=.29, wspace=.12)
    output = report_path.with_suffix('')
    for extension in ('.png', '.svg'):
        path = output.with_suffix(extension)
        fig.savefig(path, dpi=180, facecolor='white')
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path,
        default=Path('docs/go2_independent_layout00_jepa_physical_readout_2026-09-13.json'))
    main(parser.parse_args().report)
