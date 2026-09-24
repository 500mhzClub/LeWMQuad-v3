"""Plot matched outcomes, including failed missions, using evaluator-only physics."""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from lewm.physical_execution_development import rotation_xyzw
from scripts.navigation_artifact_root_development import BASE, validate_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--comparison-root-name', required=True)
    args = parser.parse_args()
    output = validate_root(BASE/args.comparison_root_name)
    comparison = json.loads((output/'result.json').read_text())
    targets = [output/f'native_navigation_comparison.{suffix}' for suffix in ('png', 'svg')]
    if any(p.exists() for p in targets): raise ValueError('preserve existing comparison figures')
    conditions = comparison.get('conditions')
    summaries = conditions if conditions is not None else {k:comparison[k] for k in ('learned', 'reactive')}
    labels = list(summaries)
    titles = {'jepa':'JEPA', 'direct':'Direct', 'supervised_rollout':'Supervised rollout'}
    if comparison.get('comparison') == 'learned_contact_score_vs_disabled_with_pose_command_xy_and_learned_yaw':
        titles.update(learned='Learned contact score', disabled='Contact score disabled')
    elif comparison.get('comparison') == 'learned_vs_pose_command_XY_with_learned_yaw_contact_retained':
        titles.update(learned='Learned XY', pose_command='Pose/command XY')
    elif comparison.get('comparison') == 'learned_vs_command_yaw_with_pose_command_xy_and_disabled_contact':
        titles.update(learned='Learned yaw', command='Command yaw')
    elif comparison.get('comparison') == 'learned_corrected_xy_and_yaw_vs_fitted_pose_command_xy_and_integrated_yaw':
        titles.update(learned='Learned motion', pose_command='Pose/command motion')
    sensor_labels = set()
    fig, axes = plt.subplots(1, len(labels), figsize=(5.5*len(labels), 6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, label in zip(axes, labels):
        summary = summaries[label]; root = validate_root(BASE/summary['root_name'])
        launch = json.loads((root/'launch.json').read_text())
        noise = launch.get('synthetic_depth_noise')
        if noise is None and 'sensor_noise_sigma_mm' in launch:
            noise = {'sigma_mm': launch['sensor_noise_sigma_mm']}
        sensor_labels.add('Ideal-sensor simulation' if noise is None else
            f'Simulated depth with {noise["sigma_mm"]:g} mm synthetic noise SD')
        evaluation = summary['independent_arrival_evaluation']
        metadata = json.loads((root/'native/in_memory_camera_observations.json').read_text())
        frames = {r['frame']:r for r in metadata['frames']}
        with np.load(root/'native/physics_trace.npz', allow_pickle=False) as arrays:
            physics = arrays['base_pose_world']
        origin = physics[frames[0]['physical_sample_index']]
        R0 = rotation_xyzw(origin[3:])
        goal = origin[:3]+R0@np.r_[launch['public_mission']['goal_initial_body_xy_m'], 0.]
        walls = json.loads((root/'native/camera_setup_identity.json').read_text())['environment']['physical_geometries']
        for wall in walls:
            if wall['geom_type'] != 'BOX': continue
            local = np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*np.asarray(wall['data'][:2])/2
            quat = np.asarray(wall['quaternion_world_wxyz'])
            rotation = rotation_xyzw(quat[[1,2,3,0]])
            points = np.c_[local, np.zeros(4)]@rotation.T+np.asarray(wall['position_world_m'])
            ax.add_patch(Polygon(points[:,:2], facecolor='#49515b', edgecolor='none'))
        outbound = next((r for r in evaluation['arrivals'] if r['phase']=='OUTBOUND'), None)
        split = outbound['frame'] if outbound else max(frames)
        for start, end, color, phase in ((0, split, '#1672b8', 'Outbound'),
                (split, max(frames), '#c75624', 'Return phase')):
            if start == end: continue
            xy = physics[[frames[f]['physical_sample_index'] for f in range(start,end+1)], :2]
            ax.plot(xy[:,0], xy[:,1], color=color, lw=1.7, label=phase)
        final = physics[frames[max(frames)]['physical_sample_index'], :2]
        ax.scatter(*origin[:2], s=65, color='#126144', edgecolor='white', zorder=4, label='Home')
        ax.scatter(*goal[:2], s=130, marker='*', color='#e1a322', edgecolor='#694c0b', zorder=4, label='Goal')
        ax.scatter(*final, s=55, marker='x', color='#922b2b', zorder=5, label='Final position')
        clean = evaluation['disallowed_contact_samples'] == 0
        status = ('Verified round trip' if evaluation['round_trip_arrival_checks_passed'] else
            'Verified goal; return incomplete' if clean and outbound and outbound['arrival_checks_passed'] else
            'No verified arrival')
        ax.set_title(f'{titles.get(label, label.title())}: {status}\n{summary["native_10hz_horizontal_path_length_m"]:.1f} m travelled; '
            f'{max(frames)/10:.1f} s recorded', fontsize=11)
        ax.set_xlabel('World x (m)'); ax.set_aspect('equal'); ax.grid(alpha=.15); ax.margins(.06)
    legend_items = {label:handle for ax in axes
        for handle,label in zip(*ax.get_legend_handles_labels())}
    legend_labels = [label for label in ('Outbound', 'Return phase', 'Home', 'Goal', 'Final position')
        if label in legend_items]
    handles = [legend_items[label] for label in legend_labels]
    fig.legend(handles, legend_labels, loc='lower center', bbox_to_anchor=(.5, .085),
        ncol=len(legend_labels), fontsize=9)
    axes[0].set_ylabel('World y (m)')
    layout = comparison['layout_index'] if conditions is not None else comparison['matched_settings']['layout_index']
    fig.suptitle(comparison.get('figure_title') or
        f'Matched navigation outcomes — development layout {layout:02d}', fontsize=14)
    contacts = ', '.join(f'{titles.get(label, label)}: {summaries[label]["independent_arrival_evaluation"]["disallowed_contact_samples"]}'
        for label in labels)
    fig.text(.5, .025, f'{"; ".join(sorted(sensor_labels))} · disallowed contact samples ({contacts})\n'
        'Native geometry and trajectory used only for evaluation', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.20,1,.93))
    for path in targets: fig.savefig(path, dpi=170)
    plt.close(fig)
    print(json.dumps(dict(figures=[str(p) for p in targets])))


if __name__ == '__main__': main()
