"""Optimistic FIFO timing calculation using the completed tracker time trace.

This does not replay sensors or infer a delayed controller's physical behavior.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_sampled_plane_recorded_tracker_v1_attempt_001')
OUTPUT = Path('docs/go2_sampled_tracker_fifo_timing_2026-09-13')


def main():
    for suffix in ('.json', '.png', '.svg'):
        if OUTPUT.with_suffix(suffix).exists():
            raise ValueError('preserve existing timing analysis')
    raw = (ROOT/'frames.jsonl').read_bytes()
    result_bytes = (ROOT/'result.json').read_bytes()
    result = json.loads(result_bytes)
    assert result['status'] == 'SAMPLED_RECORDED_TRACKER_COMPARISON_COMPLETE'
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) == result['observations'] == 4740
    assert [r['frame'] for r in rows] == list(range(len(rows)))
    assert all(r['full_tracker_result_equal'] for r in rows)
    service = np.asarray([r['tracker_s'] for r in rows])
    assert np.isfinite(service).all() and (service > 0).all()
    period = .1; arrivals = np.arange(len(rows))*period
    finish = 0.; waits = []; ages = []
    for arrival, duration in zip(arrivals, service, strict=True):
        start = max(float(arrival), finish)
        waits.append(start-float(arrival))
        finish = start+float(duration)
        ages.append(finish-float(arrival))
    waits, ages = np.asarray(waits), np.asarray(ages)
    peak = int(np.argmax(ages))
    report = dict(status='RECORDED_TRACKER_FIFO_TIMING_ANALYSIS_COMPLETE',
        observations=len(rows), arrival_period_s=period,
        assumptions=dict(one_serial_worker=True, uninterrupted_periodic_arrivals=True,
            unbounded_fifo=True, all_observations_processed=True, preemption=False,
            recorded_durations_reused_in_original_order=True,
            acquisition_cost_s=0., map_cost_s=0., planning_cost_s=0.,
            recorded_shared_host_wall_times=True),
        mean_service_ms=float(service.mean()*1000),
        isolated_calls_over_100ms=int((service>period).sum()),
        completion_age_median_s=float(np.median(ages)),
        completion_age_p95_s=float(np.percentile(ages, 95)),
        peak_completion_age=dict(frame=peak, age_s=float(ages[peak]),
            queue_wait_s=float(waits[peak]), service_s=float(service[peak])),
        observations_waiting=int((waits>1e-9).sum()),
        completion_age_over_100ms=int((ages>period+1e-9).sum()),
        completion_age_over_800ms=int((ages>.8+1e-9).sum()),
        completion_age_over_1s=int((ages>1.+1e-9).sum()),
        final_completion_age_s=float(ages[-1]),
        source_trace_sha256=hashlib.sha256(raw).hexdigest(),
        source_result_sha256=hashlib.sha256(result_bytes).hexdigest(),
        timing_simulation_only=True, new_tracker_execution=False,
        native_execution=False, dropped_frame_tracker_support_tested=False,
        delayed_navigation_outcome_inferred=False, continuous_execution_qualified=False)
    with OUTPUT.with_suffix('.json').open('x') as out:
        json.dump(report, out, indent=2, allow_nan=False); out.write('\n')

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
        gridspec_kw=dict(height_ratios=[1, 1.4]), constrained_layout=True)
    axes[0].plot(arrivals, service*1000, color='#2878a5', linewidth=.65)
    axes[0].axhline(100, color='#a33b20', linestyle='--', linewidth=1, label='100 ms period')
    axes[0].set_ylabel('Measured tracker call (ms)')
    axes[0].legend(loc='upper right', frameon=False)
    axes[1].plot(arrivals, ages, color='#7c3c88', linewidth=1.1, label='Completion age with FIFO')
    axes[1].axhline(.8, color='#9e6619', linestyle=':', linewidth=1, label='800 ms prediction horizon')
    axes[1].axhline(.1, color='#a33b20', linestyle='--', linewidth=1)
    axes[1].set_ylabel('Calculated output age (s)')
    axes[1].set_xlabel('Observation arrival time at 10 Hz (s)')
    axes[1].legend(loc='upper left', frameon=False)
    for ax in axes:
        ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)
        ax.set_xlim(arrivals[0], arrivals[-1]); ax.set_ylim(bottom=0)
    fig.suptitle('Recorded tracking times accumulate into stale output\n'
        'Optimistic queue calculation: no acquisition, mapping or planning cost', fontsize=12)
    fig.savefig(OUTPUT.with_suffix('.png'), dpi=160)
    fig.savefig(OUTPUT.with_suffix('.svg'))
    plt.close(fig)
    print(json.dumps({k:report[k] for k in ('mean_service_ms', 'completion_age_median_s',
        'peak_completion_age', 'completion_age_over_100ms', 'completion_age_over_800ms',
        'completion_age_over_1s', 'final_completion_age_s')}))


if __name__ == '__main__':
    main()
