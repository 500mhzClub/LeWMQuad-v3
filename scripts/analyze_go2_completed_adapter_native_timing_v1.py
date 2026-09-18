"""Fixed post-hoc timing decomposition of the two completed adapter cases.

Authenticate selected completed artifacts, stream every saved decision, and
reconstruct the original timing readouts. No sensors, inference or native scene
are rerun. Different policy histories preclude a causal timing comparison.
"""
import json
import sys
import time
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from scripts.navigation_artifact_root_development import BASE, artifact_path, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.maze_decision_stream_development import read_rows, NAME
from lewm.all_phase_residual_maze02_readout_development import timing

SOURCE = 'scripts/analyze_go2_completed_adapter_native_timing_v1.py'
PRIOR = 'docs/go2_independent_round_trip_queue_completion_verification_2026-09-10.json'
PRIOR_SHA = '2bca5d360aa409cb4a8e552f8934d044495c7eca5c7f4e0d3fd222418e13442e'
INPUT = BASE/'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001'
LAUNCH_SHA = '97c307ce8d2178494e7484f14b3b0cdcd8d7d0ceaab2f94d94d59da44e39b17a'
OUTPUT = ROOT/'docs/go2_completed_adapter_native_timing_2026-09-10.json'
FIGURE = ROOT/'docs/go2_completed_adapter_native_timing_2026-09-10.png'
CASES = (
    ('all_phase_full_jepa_residual_maze_02',
     '617056f19ba4928aa9ff7738616947e6e63a387cc6046353e30617ce50afa57e', 1529, 1515),
    ('all_phase_full_supervised_rollout_residual_maze_02',
     '60fdf8e04413f26cf689490b92d9797ef6a94b3b4161d6d2ed46c8b32bd7b8e2', 3014, 3000),
)
FIELDS = ('acquisition_wall_ms', 'controller_wall_ms', 'observation_and_control_wall_ms',
    'iteration_with_receipt_wall_ms', 'decision_receipt_write_wall_ms',
    'primary_capture_wall_ms', 'auxiliary_capture_wall_ms', 'paired_renderer_query_wall_ms')


def read(name):
    return json.loads(artifact_path(INPUT, name).read_text())


def summaries(rows):
    if not rows: raise ValueError('nonempty declared timing window required')
    result = {field:timing([row[field] for row in rows]) | dict(
        total_ms=float(np.sum([row[field] for row in rows]))) for field in FIELDS}
    result.update(first_tick=rows[0]['tick'], last_tick=rows[-1]['tick'], observations=len(rows),
        controller_fraction_of_observation_and_control=(result['controller_wall_ms']['total_ms']/
            result['observation_and_control_wall_ms']['total_ms']),
        controller_fraction_of_iteration_with_receipt=(result['controller_wall_ms']['total_ms']/
            result['iteration_with_receipt_wall_ms']['total_ms']))
    return result


def analyze(case):
    name, worker_sha, count, navigation_count = case
    anchors = {'launch.json': LAUNCH_SHA, name+'_worker_terminal.json': worker_sha}
    verify_artifacts(INPUT, anchors); record = read(name+'_worker_terminal.json')
    if record['status'] != 'ALL_PHASE_RESIDUAL_MAZE02_COLLECTED_AND_RAW_AUDITED' or record['collection']['decisions'] != count:
        raise ValueError('exact completed original worker population required')
    required = [name+'/'+n for n in (NAME, 'decision_stream_timing.jsonl', 'camera_audit.json',
        'auxiliary_camera_audit.json', 'renderer_capture_witnesses.json')]+[name+'_readout.json']
    bindings = anchors | {n:record['artifact_sha256'][n] for n in required}
    verify_artifacts(INPUT, bindings)
    primary = read(name+'/camera_audit.json'); auxiliary = read(name+'/auxiliary_camera_audit.json')
    witness = read(name+'/renderer_capture_witnesses.json')
    with artifact_path(INPUT, name+'/decision_stream_timing.jsonl').open() as stream:
        write_times = [json.loads(line) for line in stream]
    if any(len(value) != count for value in (primary, auxiliary, witness['paired'], write_times)):
        raise ValueError('complete original acquisition and write timing populations required')
    compact = []
    for index, row in enumerate(read_rows(INPUT/name)):
        if row['tick'] != index or row['observation_index'] != index or write_times[index]['tick'] != index:
            raise ValueError('complete ordered timing rows required')
        if auxiliary[index]['frame'] != index or witness['paired'][index]['frame'] != index:
            raise ValueError('same original acquisition frame required')
        a, c, total = (row[k] for k in FIELDS[:3])
        if abs(total-(a+c)) > 1e-6:
            raise ValueError('original acquisition/controller intervals must sum to their joint interval')
        entry = {k:row[k] for k in FIELDS[:3]}
        entry.update({k:write_times[index][k] for k in FIELDS[3:5]})
        entry.update(tick=index, terminal=row['decision']['terminal'],
            navigation=(index >= 3 and row['decision']['terminal'] is None),
            primary_capture_wall_ms=1000*primary[index]['capture_wall_time_s'],
            auxiliary_capture_wall_ms=1000*auxiliary[index]['capture_wall_s'],
            paired_renderer_query_wall_ms=witness['paired'][index]['query_wall_ms'])
        if any(type(entry[k]) not in (int,float) or not np.isfinite(entry[k]) or entry[k] < 0 for k in FIELDS):
            raise ValueError('finite nonnegative recorded timings required')
        if entry['iteration_with_receipt_wall_ms'] < row.get('iteration_with_command_wall_ms', total):
            raise ValueError('full recorded iteration must contain the earlier intervals')
        compact.append(entry)
        if index and index % 500 == 0: print('TIMING_ROWS_READ', name, index+1, flush=True)
    if len(compact) != count:
        raise ValueError('every completed observation must be retained')
    readout = read(name+'_readout.json')
    for field, key in (('observation_and_control_wall_ms','observation_and_control'),
            ('iteration_with_receipt_wall_ms','iteration_with_receipt')):
        if timing([row[field] for row in compact]) != readout[key]:
            raise ValueError('the original complete timing readout must reproduce exactly')
    navigation = [row for row in compact if row['navigation']]
    if len(navigation) != navigation_count:
        raise ValueError('exact original nonterminal navigation decision population required')
    windows = dict(all_observations=summaries(compact), navigation=summaries(navigation),
        first_402_navigation=summaries(navigation[:402]), after_first_402_navigation=summaries(navigation[402:]),
        first_100_navigation=summaries(navigation[:100]), last_100_navigation=summaries(navigation[-100:]))
    bins = [summaries(navigation[i:i+100]) for i in range(0,len(navigation),100)]
    verify_artifacts(INPUT, bindings)
    return dict(case=name, artifact_sha256=bindings, windows=windows, navigation_bins_of_up_to_100=bins,
        original_complete_timing_readout_reproduced=True, observed_rows=compact,
        raw_sensor_or_controller_audit_reexecuted=False, model_inference=False,
        warmup_observations=3, terminal_observations=count-navigation_count-3,
        verified_round_trip=record['verified_round_trip'], strict_physical_visibility_pass=record['strict_physical_visibility_pass'])


def plot(cases):
    fig = Figure(figsize=(11,7), constrained_layout=True); FigureCanvasAgg(fig)
    for ax, case, label in zip(fig.subplots(2,1), cases, ('JEPA: completed development episode',
            'Supervised: completed development episode'), strict=True):
        bins = case['navigation_bins_of_up_to_100']; x = [(row['first_tick']+row['last_tick'])/2 for row in bins]
        for field, legend, color in [('controller_wall_ms','Controller','#2166ac'),
                ('acquisition_wall_ms','Packet acquisition','#d95f02'),
                ('iteration_with_receipt_wall_ms','Iteration through receipt write','#7570b3')]:
            ax.plot(x, [row[field]['median_ms'] for row in bins], marker='.', label=legend, color=color)
        ax.axhline(100, color='black', linestyle='--', linewidth=1, label='100 ms command interval')
        ax.set_title(label); ax.set_ylabel('Recorded median wall time (ms)')
        ax.set_xlabel('Original observation index (up to 100 navigation observations per bin)')
        ax.grid(alpha=.2); ax.legend(fontsize=8, loc='upper left')
    fig.suptitle('Original simulation timing; physics paused during compute\nDifferent policy histories; no optimized-controller or hardware timing claim', fontsize=11)
    with FIGURE.open('xb') as stream: fig.savefig(stream, format='png', dpi=160)


def main():
    if len(sys.argv) != 1 or any(path.exists() or path.is_symlink() for path in (OUTPUT, FIGURE)):
        raise ValueError('fixed exclusive derived timing analysis required')
    started = time.perf_counter(); verify({PRIOR: PRIOR_SHA})
    prior = json.loads((ROOT/PRIOR).read_text()); verify(prior['source_sha256'])
    sources = discover_sources((SOURCE, PRIOR, 'docs/go2_independent_round_trip_queue_completion_result_2026-09-10.md'),
        prior['source_sha256']); verify(sources)
    cases = [analyze(case) for case in CASES]
    plot(cases); verify(sources)
    write_json(OUTPUT, dict(status='COMPLETED_ADAPTER_NATIVE_TIMING_DECOMPOSITION_COMPLETE',
        source_sha256=sources, source_count=len(sources), cases=cases, numpy_version=np.__version__,
        figure_sha256=digest(FIGURE), wall_s=time.perf_counter()-started,
        original_native_timings_only=True, observations_replayed=False, new_layout_sensor_data_consumed=False,
        native_execution=False, model_inference=False, hardware_timing_measured=False,
        causal_cross_model_speed_comparison=False, timing_optimization_validated=False,
        timing_qualification=False, goal_achieved=False))
    print('NATIVE_TIMING_DECOMPOSITION_COMPLETE', digest(OUTPUT), len(sources), flush=True)
    for case in cases:
        print(json.dumps(dict(case=case['case'], windows=case['windows'])), flush=True)


if __name__ == '__main__': main()
