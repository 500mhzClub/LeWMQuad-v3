"""Fixed 32-stream development comparison on previously exposed tapes.

Gyro continuity is the baseline; joint continuity is the new candidate. Legacy
scorer column names are mapped explicitly below. No native collection, training,
observer reset, gate tuning, original-output overwrite or controller adoption.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np

from lewm.temporal_anchor_continuity_development import TemporalAnchorVisualLedMotion
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorVisualLedMotion
from lewm.independent_tracking_stress_development import transform, packet_digest, ONSET_FRAME
from lewm.joint_continuity_history_verification_development import verify_joint_history
from lewm.joint_rotation_witness_verification_development import verify_rotations
from lewm.posthoc_tracking_numerical_view_development import sensor_convention_view
from lewm import independent_tracking_numerical_verification_development as numerical
from scripts import read_go2_tracking_posthoc_raw_accuracy_v1 as prior
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_stress_cohort_development as stress
from scripts import independent_tracking_evaluation_development as scoring
from scripts import navigation_artifact_root_development as custody
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_science_v1 import RESULT_SHA256 as PRIOR_RESULT_SHA256
from scripts.verify_go2_temporal_anchor_continuity_v1 import statistics

OUTPUT = custody.BASE / 'go2_joint_continuity_comparison_v1_attempt_001'
SOURCE = 'scripts/run_go2_joint_continuity_comparison_v1.py'
TEST = 'lewm/tests/test_joint_continuity_comparison_development.py'
PROTOCOL = 'docs/go2_joint_temporal_anchor_candidate_2026-09-07.md'
SCENARIOS = ('nominal', 'gyro_bias_positive', 'gyro_bias_negative', 'rgb_unavailable')
ARM_MEANING = dict(original='gyro_continuity', temporal_anchor='joint_continuity')
REUSE = dict(nominal='base', gyro_bias_positive='shared_gyro_bias', rgb_unavailable='rgb_unavailable_1')
FIELDS = ('pose', 'selection', 'failure', 'continuity', 'observer_wall_ms')
STREAM_BOUND = 443 * base.MAX_ROW
TOTAL_BOUND = 64 * STREAM_BOUND + 36 * base.MAX_METADATA


def intervention(packet, scenario, frame):
    base.require(scenario in SCENARIOS, 'fixed comparison scenario')
    if scenario != 'gyro_bias_negative':
        mapping = dict(nominal='nominal', gyro_bias_positive='shared_gyro_bias', rgb_unavailable='rgb_unavailable_1')
        return transform(packet, mapping[scenario], frame, 1_500_000_000)[0]
    p, d, f, now = deepcopy(packet)
    base.require(now == 1_500_000_000 + frame * 100_000_000, 'complete intervention clock')
    for channel in (p['sensor_state']['sensed']['gyro'], f):
        use = channel['measured_ns'] >= 1_500_000_000 + ONSET_FRAME * 100_000_000
        channel['values'][use, 2] -= .02
    return p, d, f, now


def input_name(trial, scenario):
    old = REUSE.get(scenario)
    return trial + '_estimates.jsonl' if old == 'base' else stress.stream_name(trial, old) if old else None


def stem(trial, scenario):
    base.require(trial in base.TRIALS and scenario in SCENARIOS, 'fixed tape/scenario output')
    return trial + '__' + scenario


class Store:
    def __init__(self, output, trial, scenario):
        self.output = custody.validate_root(output)
        base.require(self.output not in (prior.INPUT, prior.OUTPUT), 'distinct comparison root')
        self.trial = trial; self.scenario = scenario; self.hashes = {}

    @contextmanager
    def stream(self, name):
        if name == self.trial + '_evaluation.jsonl': name = stem(self.trial, self.scenario) + '_evaluation.jsonl'
        base.require(name in {stem(self.trial, self.scenario) + suffix for suffix in
            ('_estimates.jsonl', '_evaluation.jsonl')}, 'exclusive declared comparison stream')
        path = self.output / name
        base.require(path.resolve() == path, 'nonsymlink stream')
        self.rows = self.used = 0
        with path.open('xb') as stream:
            yield stream
        with path.open('rb') as stream: self.hashes[name] = hashlib.file_digest(stream, 'sha256').hexdigest()

    def append(self, stream, row):
        encoded = base.encode(row)
        base.require(self.rows < 443 and len(encoded) <= base.MAX_ROW
            and self.used + len(encoded) <= STREAM_BOUND, 'finite row/stream envelope')
        base.require(shutil.disk_usage(self.output).free >= prior.RESERVE_BYTES + len(encoded), 'storage reserve')
        stream.write(encoded); self.rows += 1; self.used += len(encoded)


def observe(model, packet):
    started = time.perf_counter_ns()
    p, d, f, now = packet
    result = model.observe(p, d, f, now_ns=now)
    return dict(pose=result['current_pose'], selection=result['reference_selection'],
        failure=result['terminal_failure'], continuity=result['continuity_evidence'],
        observer_wall_ms=(time.perf_counter_ns() - started) / 1e6)


def summaries(rows):
    counts = {a: dict(frames=0, available=0, first_failure=None) for a in base.ARMS}
    failures = dict.fromkeys(base.ARMS)
    continuity = base.ContinuityAudit()
    categories = dict(both=0, original_only=0, temporal_anchor_only=0, neither=0)
    exposure = {a: dict(update_attempts=0, changed_packet_updates=0, onset_update_attempted=False) for a in base.ARMS}
    timing = {a: [] for a in base.ARMS}; changed = 0; total = 0
    for row in rows:
        frame = total; now = 1_500_000_000 + frame * 100_000_000
        base.require(frame < 443 and row['frame'] == frame and row['measured_ns'] == now
            and row['arm_meaning'] == ARM_MEANING and set(row['arms']) == set(base.ARMS)
            and row['native_pose_input'] is False and row['navigation_qualified'] is False, 'full matched chronology')
        base.require(row['availability'] == stress.category(row), 'paired availability')
        categories[row['availability']] += 1
        differs = row['source_packet_sha256'] != row['intervened_packet_sha256']; changed += differs
        for arm, label in (('original', 'gyro'), ('temporal_anchor', 'joint')):
            r = row['arms'][arm]; pose = r['pose']; c = counts[arm]; e = exposure[arm]
            attempted = c['first_failure'] is None
            base.require(set(r) == set(FIELDS), 'exact saved observer fields')
            base.require(type(r['observer_wall_ms']) in (int, float)
                and math.isfinite(r['observer_wall_ms']) and r['observer_wall_ms'] >= 0, 'finite observer time')
            if attempted: timing[arm].append(r['observer_wall_ms'])
            e['update_attempts'] += attempted; e['changed_packet_updates'] += attempted and differs
            if frame == ONSET_FRAME: e['onset_update_attempted'] = attempted
            if pose is not None:
                base.require(attempted and r['failure'] is None and pose['mode'] == label
                    and pose['frame'] == frame and pose['measured_ns'] == now
                    and pose['global_history_reset'] is False and pose['position_error_bound'] is None
                    and pose['orientation_error_bound'] is None and pose['native_pose_input'] is False,
                    'same-frame measured pose without reset or invented bound')
                numerical._rotation(pose['rotation_initial_body_from_current_body'])
                xyz = np.asarray(pose['position_initial_body_m'])
                base.require(xyz.shape == (3,) and np.isfinite(xyz).all(), 'finite observed translation')
                c['available'] += 1
            else:
                base.require(r['failure'] is not None, 'retained terminal reason')
                if attempted: c['first_failure'] = frame; failures[arm] = r['failure']
                else: base.require(r['failure'] == failures[arm], 'terminal failure cannot change')
            c['frames'] += 1
        continuity.observe(row); total += 1
    return dict(frames=total, arms=counts, availability=categories, continuity=continuity.summary(),
        first_failure=failures, exposure=exposure, changed_packet_frames=changed,
        active_observer_timing={a: dict(statistics_ms=statistics(v), over_100ms=sum(x > 100 for x in v))
            for a, v in timing.items()})


def replay(trial, scenario, output=None, sources=None, frames=443):
    """Sensor-only new joint inference; exact saved gyro baseline where possible."""
    if sources is not None: prior.source_check(sources)
    reader = base.IntentReturnRGBDReplay(prior.INPUT / trial)
    base.require(len(reader.frames) == 443 and 1 <= frames <= 443, 'complete original tape')
    joint = JointTemporalAnchorVisualLedMotion()
    baseline = TemporalAnchorVisualLedMotion() if scenario == 'gyro_bias_negative' else None
    old = iter(stress.rows(prior.INPUT, input_name(trial, scenario))) if baseline is None else None
    collected = []; started = time.monotonic()
    for frame in range(frames):
        packet = reader.packet(frame); before = packet_digest(packet)
        modified = intervention(packet, scenario, frame); modified_hash = packet_digest(modified)
        if old is not None:
            saved = next(old)
            base.require(saved['frame'] == frame and saved['measured_ns'] == modified[3], 'exact reused frame')
            if scenario != 'nominal':
                base.require(saved['source_packet_sha256'] == before
                    and saved['intervened_packet_sha256'] == modified_hash, 'exact reused intervention bytes')
            gyro = {k: saved['arms']['temporal_anchor'][k] for k in FIELDS}
        else: gyro = observe(baseline, deepcopy(modified))
        candidate = observe(joint, modified)
        base.require(packet_digest(packet) == before and packet_digest(modified) == modified_hash,
                     'observer must not mutate source or transformed packet')
        row = dict(frame=frame, measured_ns=packet[3], scenario=scenario, arm_meaning=ARM_MEANING,
            source_packet_sha256=before, intervened_packet_sha256=modified_hash,
            arms=dict(original=gyro, temporal_anchor=candidate), native_pose_input=False, navigation_qualified=False)
        row['availability'] = stress.category(row)
        collected.append(row)
    if old is not None and frames == 443: base.require(next(old, None) is None, 'no extra reused frames')
    report = summaries(collected)
    verify_joint_history(collected, report['continuity'])
    rotations = verify_rotations(collected)
    # Bounded to <=443 paired rows; no RGB/depth packets retained in the list.
    hashes = {}
    if output is not None:
        store = Store(output, trial, scenario); name = stem(trial, scenario) + '_estimates.jsonl'
        with store.stream(name) as stream:
            for row in collected: store.append(stream, row)
        hashes = store.hashes
        report |= dict(estimates_file=name, estimates_sha256=hashes[name])
    scientific = [{k: v for k, v in row.items() if k != 'arms'} | dict(
        arms={a: {k: v for k, v in arm.items() if k != 'observer_wall_ms'} for a, arm in row['arms'].items()})
        for row in collected]
    if sources is not None: prior.source_check(sources)
    return dict(report=report, rotations=rotations, artifact_sha256=hashes,
        scientific_sha256=hashlib.sha256(base.encode(scientific)).hexdigest(),
        elapsed_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def benchmark_job(trial):
    return replay(trial, 'nominal', frames=12)


def benchmark():
    measurements = []
    for width in (1, 2, 4):
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=width, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=prior.worker_limits) as pool:
            results = list(pool.map(benchmark_job, base.TRIALS[:4]))
        measurements.append(dict(workers=width, wall_seconds=time.monotonic() - started,
            scientific_sha256=[r['scientific_sha256'] for r in results],
            max_rss_bytes=max(r['peak_rss_bytes'] for r in results)))
        print('JOINT_BENCHMARK', width, measurements[-1]['wall_seconds'], flush=True)
    base.require(all(m['scientific_sha256'] == measurements[0]['scientific_sha256'] for m in measurements),
                 'parallel prefix scientific outputs differ')
    return dict(measurements=measurements, selected_workers=min(measurements, key=lambda x:x['wall_seconds'])['workers'],
        scope='twelve-frame nominal sensor-only prefix on the same four tapes; no truth or full raw audit')


def admit_sensor_population(output, phase, sources):
    prior.source_check(sources)
    expected = {stem(t, s) for t in base.TRIALS for s in SCENARIOS}
    base.require(set(phase['reports']) == expected and phase['native_coordinates_parsed'] is False,
                 'all 32 new paired sensor streams required before truth')
    custody.verify_artifacts(output, phase['artifact_sha256'])
    for trial in base.TRIALS:
        reader = base.IntentReturnRGBDReplay(prior.INPUT / trial)
        rows = {s: list(stress.rows(output, phase['reports'][stem(trial,s)]['report']['estimates_file'])) for s in SCENARIOS}
        base.require(all(len(v) == 443 for v in rows.values()), 'all failed and available rows required')
        for frame in range(443):
            packet = reader.packet(frame); before = packet_digest(packet)
            for scenario in SCENARIOS:
                row = rows[scenario][frame]
                base.require(row['scenario'] == scenario and row['source_packet_sha256'] == before
                    and row['intervened_packet_sha256'] == packet_digest(intervention(packet, scenario, frame)),
                    'new intervention differs from actual source packet')
        for scenario in SCENARIOS:
            report = phase['reports'][stem(trial, scenario)]['report']
            actual = summaries(rows[scenario])
            base.require(actual == {k: report[k] for k in actual}, 'sensor summary reconstruction differs')
            verify_joint_history(rows[scenario], report['continuity']); verify_rotations(rows[scenario])
            if scenario in REUSE:
                for row, saved in zip(rows[scenario], stress.rows(prior.INPUT, input_name(trial,scenario)), strict=True):
                    base.require(row['arms']['original'] == {k:saved['arms']['temporal_anchor'][k] for k in FIELDS},
                                 'reused gyro-continuity output changed')
    prior.source_check(sources)
    custody.verify_artifacts(output, phase['artifact_sha256'])


def evaluate(output, phase, episodes, sources):
    admit_sensor_population(output, phase, sources)
    scores = {}; bindings = dict(phase['artifact_sha256'])
    for trial in base.TRIALS:
        with np.load(custody.artifact_path(prior.INPUT, trial + '/physics_trace.npz'), allow_pickle=False) as z:
            raw = {k:z[k] for k in ('timestamp_s', 'base_pose_world')}
        base.require(len(raw['timestamp_s']) == episodes[trial]['result']['physics_samples'], 'complete original truth')
        view, provenance = sensor_convention_view(raw)
        for scenario in SCENARIOS:
            key = stem(trial, scenario); report = phase['reports'][key]['report']
            store = Store(output, trial, scenario)
            score = scoring._score_trial(store, trial, report, raw)
            checked = numerical.verify_pose_stream(view, stress.rows(output, report['estimates_file']),
                stress.rows(output, key + '_evaluation.jsonl'), report, score)
            bindings.update(store.hashes)
            scores[key] = dict(score=score, representation=provenance,
                independently_reconstructed=checked['numerical_pose_score_reconstruction_verified'])
    return scores, bindings


def continuation_criteria(output, phase, scores):
    nominal = all(scores[stem(t,'nominal')]['score']['arms']['temporal_anchor']['position_m']['count'] >=
                  scores[stem(t,'nominal')]['score']['arms']['original']['position_m']['count'] for t in base.TRIALS)
    allocation = all(scores[stem(t,s)]['score']['arms']['temporal_anchor']['position_m']['maximum'] is not None and
        scores[stem(t,s)]['score']['arms']['temporal_anchor']['position_m']['maximum'] <= .02 and
        scores[stem(t,s)]['score']['arms']['temporal_anchor']['orientation_rad']['maximum'] <= math.radians(2)
        for t in base.TRIALS for s in SCENARIOS[:3])
    missing = all(phase['reports'][stem(t,'rgb_unavailable')]['report']['arms']['temporal_anchor']['first_failure'] == ONSET_FRAME
                  for t in base.TRIALS)
    # Compare bias-induced changes from each arm's nominal error, using the same
    # frames present in all four estimates. Aggregate tape means, not frames as N.
    effects = {}
    for s in ('gyro_bias_positive', 'gyro_bias_negative'):
        effects[s] = {}
        for t in base.TRIALS:
            clean = list(stress.rows(output, stem(t,'nominal') + '_evaluation.jsonl'))
            biased = list(stress.rows(output, stem(t,s) + '_evaluation.jsonl'))
            values = {a:{m:[] for m in ('position_m','orientation_rad')} for a in base.ARMS}
            for n,b in zip(clean,biased,strict=True):
                if n['frame'] < ONSET_FRAME or n['availability'] != 'both' or b['availability'] != 'both': continue
                for a in base.ARMS:
                    for m in values[a]: values[a][m].append(b['errors'][a][m] - n['errors'][a][m])
            effects[s][t] = {a:{m:statistics(v) for m,v in fields.items()} for a,fields in values.items()}
    reduced = {}
    for s, tapes in effects.items():
        reduced[s] = all(all(v[a][m]['count'] > 0 for a in base.ARMS) for v in tapes.values() for m in ('position_m','orientation_rad')) and all(
            sum(v['temporal_anchor'][m]['mean'] for v in tapes.values()) < sum(v['original'][m]['mean'] for v in tapes.values())
            for m in ('position_m','orientation_rad'))
    return dict(nominal_availability_not_reduced_on_any_tape=nominal,
        every_available_nominal_and_biased_candidate_pose_within_empirical_allocation=allocation,
        missing_rgb_terminal_at_onset_on_all_tapes=missing,
        paired_bias_induced_error_reduced=reduced, per_tape_bias_induced_error=effects,
        development_continuation_criterion_met=nominal and allocation and missing and all(reduced.values()),
        candidate_adopted=False, independent_validation=False, navigation_qualified=False)


def run():
    started = time.monotonic()
    episodes, original_streams, original_definition = prior.admit()
    custody.verify_artifacts(prior.OUTPUT, {'result.json': PRIOR_RESULT_SHA256})
    old_result = base.read(prior.OUTPUT, 'result.json')
    custody.verify_artifacts(prior.OUTPUT, old_result['artifact_sha256'])
    old_launch = base.read(prior.OUTPUT, 'launch.json')
    prior.source_check(old_launch['source_sha256'])
    sources = discover_sources([SOURCE, TEST, PROTOCOL,
        'lewm/tests/test_joint_temporal_anchor_continuity_development.py'], old_launch['source_sha256'])
    prior.source_check(sources); hardware = prior.hardware()
    base.require(hardware['memory_available_bytes'] >= 4*prior.WORKER_MEMORY_BYTES + 4*1024**3
        and hardware['artifact_free_bytes'] >= TOTAL_BOUND + prior.RESERVE_BYTES, 'declared finite resources unavailable')
    output = custody.create_output(OUTPUT)
    prior.metadata(output, 'launch.json', dict(source_sha256=sources, hardware=hardware,
        original_result_sha256=PRIOR_RESULT_SHA256, original_stream_sha256=original_streams,
        trials=list(base.TRIALS), scenarios=list(SCENARIOS), arm_meaning=ARM_MEANING,
        reused_gyro_baseline_scenarios=REUSE, new_gyro_baseline_scenarios=['gyro_bias_negative'],
        source_protocol=PROTOCOL, maximum_workers=4, worker_address_space_limit_bytes=prior.WORKER_MEMORY_BYTES,
        total_output_bound_bytes=TOTAL_BOUND, stream_bound_bytes=STREAM_BOUND,
        free_space_reserve_bytes=prior.RESERVE_BYTES,
        adopted=False, native_collection=False, training=False, raw_audits_repeated=False,
        inherited_strict_visibility_pass=old_result['strict_depth_visibility_pass'],
        inherited_motion_coverage=old_result['all_intended_motion_covered'],
        evaluation_scope='exposed development tapes, not independent validation',
        paired_bias_effect_aggregation='equal-weight tape means on four-way common post-onset support; both metrics and signs must improve'))
    try:
        workload = benchmark(); prior.metadata(output,'workload.json',workload)
        prior.source_check(sources)
        reports = {}; bindings = {}
        with ProcessPoolExecutor(max_workers=workload['selected_workers'], mp_context=multiprocessing.get_context('spawn'),
                                 initializer=prior.worker_limits) as pool:
            futures = {pool.submit(replay,t,s,output,sources):stem(t,s) for t in base.TRIALS for s in SCENARIOS}
            for future in as_completed(futures):
                key = futures[future]; value = future.result()
                reports[key] = value; bindings.update(value['artifact_sha256'])
                prior.metadata(output, key + '_sensor_result.json', value)
                print('JOINT_REPLAY_COMPLETE', key, value['report']['arms'], round(value['elapsed_seconds'],2), flush=True)
        phase = dict(reports=reports, artifact_sha256=bindings, native_coordinates_parsed=False)
        prior.metadata(output, 'sensor_phase_complete.json', phase)
        scores, bindings = evaluate(output, phase, episodes, sources)
        criteria = continuation_criteria(output, phase, scores)
        fresh = prior.admit()
        base.require(fresh == (episodes, original_streams, original_definition), 'original inputs changed')
        prior.source_check(sources); custody.verify_artifacts(output, bindings)
        all_names = ['launch.json','workload.json','sensor_phase_complete.json'] + [k+'_sensor_result.json' for k in reports]
        for name in all_names:
            with custody.artifact_path(output,name).open('rb') as f: bindings[name]=hashlib.file_digest(f,'sha256').hexdigest()
        total = sum(custody.artifact_path(output,n).stat().st_size for n in bindings)
        base.require(total <= TOTAL_BOUND - base.MAX_METADATA, 'complete derived output byte envelope')
        result = dict(status='JOINT_CONTINUITY_DEVELOPMENT_COMPARISON_COMPLETE', scores=scores,
            criteria=criteria, artifact_sha256=bindings, artifact_bytes_before_result=total,
            streams=32, frames_per_stream=443, arm_meaning=ARM_MEANING,
            elapsed_seconds=time.monotonic()-started, original_result_sha256=PRIOR_RESULT_SHA256,
            original_attempt_unchanged=True, raw_audits_repeated=False, strict_visibility_pass=False,
            independent_validation=False, gyro_bias_estimated=False, candidate_adopted=False,
            learned_model_used=False, navigation_qualified=False, goal_achieved=False)
        sha,_=prior.metadata(output,'result.json',result)
        print('JOINT_COMPARISON_RESULT',sha,json.dumps(criteria),flush=True)
    except Exception as error:
        prior.metadata(output,'failure.json',dict(status='JOINT_COMPARISON_FAILED',error=repr(error),
            traceback=traceback.format_exc(),candidate_adopted=False,navigation_qualified=False))
        raise


if __name__ == '__main__': run()
