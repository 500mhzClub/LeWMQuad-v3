"""Plot every frozen sensitivity case, including its rejected-frame endpoint."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scripts.replay_go2_current_gyro_bias_development import ROOTS, MEMBERS, RESULT, output


def main():
    result = json.loads(RESULT.read_text())
    lookup = {(r['root'], r['condition']): r for r in result['assignments']}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), sharex='col', constrained_layout=True)
    labels = ('Original gyro', '+0.001 rad/s yaw bias', '−0.001 rad/s yaw bias')
    for column, root in enumerate(ROOTS):
        for member, label in zip(MEMBERS, labels):
            rows = json.loads((output(root, member)/'accuracy.json').read_text())['rows']
            summary = lookup[root, member.name]
            t = [r['elapsed_sensor_s'] for r in rows]
            for row_index, metric in enumerate(('xy_error_mm', 'orientation_error_deg')):
                axis = axes[row_index, column]
                line, = axis.plot(t, [r[metric] for r in rows], label=label, linewidth=1.3)
                if summary['failure'] is not None:
                    axis.plot(t[-1], rows[-1][metric], 'x', color=line.get_color(), markersize=8)
        axes[0, column].set_title(f'Recorded journey {column+1}')
        axes[1, column].set_xlabel('Recorded elapsed time (s)')
    axes[0, 0].set_ylabel('Raw position error in XY (mm)')
    axes[1, 0].set_ylabel('Raw orientation error (degrees)')
    for axis in axes.flat:
        axis.grid(alpha=.25)
    axes[0, 0].legend(fontsize=9)
    figure.suptitle('Current tracker: synthetic gyro-bias sensitivity\n'
        'Fixed recorded journeys; × marks last accepted pose before rejection', fontsize=12)
    for suffix in ('png', 'svg'):
        destination = RESULT.with_name('go2_current_gyro_bias_2026-09-17.' + suffix)
        if destination.exists():
            raise ValueError('preserve existing figure')
        figure.savefig(destination, dpi=160)
        print(destination)
    plt.close(figure)


if __name__ == '__main__':
    main()
