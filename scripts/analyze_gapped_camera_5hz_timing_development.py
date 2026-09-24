"""Tracking accuracy and optimistic FIFO age for the terminal 5-Hz recording."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_gapped_camera_5hz_journey_v1_attempt_001')
OUTPUT = Path('docs/go2_gapped_camera_5hz_timing_2026-09-13')


def main():
    for suffix in ('.json', '.png', '.svg'):
        if OUTPUT.with_suffix(suffix).exists(): raise ValueError('preserve existing analysis')
    path = ROOT/('result.json' if (ROOT/'result.json').exists() else 'partial_result.json')
    raw = path.read_bytes(); result = json.loads(raw)
    assert result['status'] in ('GAPPED_CAMERA_RECORDED_PREFIX_COMPLETE', 'GAPPED_CAMERA_RECORDED_PREFIX_FAILED')
    rows = result['records']; assert [r['source_frame'] for r in rows] == list(range(0, 2*len(rows), 2))
    complete = result['status'] == 'GAPPED_CAMERA_RECORDED_PREFIX_COMPLETE'
    if complete: assert len(rows) == 1720
    else: assert result['failure']['next_visual_frame'] == len(rows)*2
    service = np.asarray([r['tracker_ms']/1000 for r in rows])
    assert np.isfinite(service).all() and (service > 0).all()
    arrivals = (np.asarray([r['measured_ns'] for r in rows])-rows[0]['measured_ns'])/1e9
    finish = 0.; waits = []; ages = []
    for arrival, duration in zip(arrivals, service, strict=True):
        start = max(float(arrival), finish); waits.append(start-float(arrival))
        finish = start+float(duration); ages.append(finish-float(arrival))
    waits = np.asarray(waits); ages = np.asarray(ages); peak = int(np.argmax(ages))
    report = dict(status='GAPPED_CAMERA_5HZ_RECORDED_TIMING_ANALYSIS_COMPLETE',
        source_result_sha256=hashlib.sha256(raw).hexdigest(), visual_observations=len(rows),
        source_artifact=path.name, complete_recorded_journey=complete,
        tracking_failure=result['failure'], accepted_prefix_only=not complete,
        failing_call_duration_included=False,
        gyro_packets_ingested=result['gyro_packets_ingested'], camera_period_ms=200,
        camera_counts=dict(Counter(r['camera'] for r in rows)),
        continuity_counts=dict(Counter(r['continuity_status'] for r in rows)),
        native_position_error_m=result['native_position_error_m'],
        native_rotation_error_rad=result['native_rotation_error_rad'],
        tracker_ms=dict(mean=float(service.mean()*1000), median=float(np.median(service)*1000),
            p95=float(np.percentile(service, 95)*1000), maximum=float(service.max()*1000),
            calls_over_200ms=int((service>.2).sum())),
        fifo_completion_age_ms=dict(median=float(np.median(ages)*1000),
            p95=float(np.percentile(ages, 95)*1000), maximum=float(ages.max()*1000),
            final=float(ages[-1]*1000), outputs_over_200ms=int((ages>.2+1e-9).sum())),
        peak=dict(source_frame=rows[peak]['source_frame'],
            queue_wait_ms=float(waits[peak]*1000), service_ms=float(service[peak]*1000)),
        assumptions=dict(one_serial_worker=True, all_camera_frames_processed=True,
            unbounded_fifo=True, shared_host_recorded_durations_reused=True,
            acquisition_cost_s=0., map_cost_s=0., planning_cost_s=0.),
        timing_simulation_only=True, native_execution=False,
        delayed_navigation_outcome_inferred=False, continuous_execution_qualified=False)
    with OUTPUT.with_suffix('.json').open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(arrivals, [r['native_position_error_m']*1000 for r in rows], lw=.9)
    axes[0].set_ylabel('Position error (mm)')
    axes[1].plot(arrivals, service*1000, lw=.7, color='#2878a5')
    axes[1].axhline(200, ls='--', color='#a33b20', lw=1)
    axes[1].set_ylabel('Tracker call (ms)')
    axes[2].plot(arrivals, ages*1000, lw=.8, color='#7c3c88')
    axes[2].axhline(200, ls='--', color='#a33b20', lw=1)
    axes[2].set_ylabel('FIFO completion age (ms)')
    axes[2].set_xlabel('Recorded observation time (s)')
    for ax in axes:
        ax.grid(alpha=.2); ax.set_ylim(bottom=0); ax.set_xlim(arrivals[0], arrivals[-1])
    fig.suptitle(('Complete recorded journey' if complete else 'Accepted prefix before tracking failure')+
        ' at 5 Hz\n'
        'Queue calculation excludes acquisition, mapping and planning', fontsize=12)
    fig.savefig(OUTPUT.with_suffix('.png'), dpi=160); fig.savefig(OUTPUT.with_suffix('.svg'))
    plt.close(fig); print(json.dumps(report))


if __name__ == '__main__': main()
