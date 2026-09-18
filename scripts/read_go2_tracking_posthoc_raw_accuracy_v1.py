"""Distinct post-hoc analysis: eight raw audits, 96 scores, 48 comparisons.

Reuses the exact completed sensor-admission evidence; never resumes the failed
native attempt. Full raw audits run once per tape. No monkeypatch, simulator,
observer inference, training, or protected-root discovery. Outputs are exclusive
new derived artifacts; scientific failures remain in the complete population.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import multiprocessing
import os
from pathlib import Path
import resource
import shutil
import time
import traceback

import numpy as np

from lewm import independent_tracking_numerical_verification_development as numerical
from lewm.independent_tracking_continuity_verification_development import verify_continuity
from lewm.posthoc_tracking_numerical_view_development import sensor_convention_view
from scripts import independent_tracking_cohort_development as base
from scripts import independent_tracking_stress_cohort_development as stress
from scripts import independent_tracking_evaluation_development as scoring
from scripts import independent_tracking_predecessor_comparison_development as predecessors
from scripts import read_go2_tracking_sensor_convention_coverage_v1 as admission
from scripts.posthoc_tracking_raw_audit_development import raw_audit_sensor_convention
from scripts import navigation_artifact_root_development as custody
from scripts.startup_source_inventory_development import allowed_relative, discover_sources

ROOT = admission.prior.ROOT
INPUT = admission.prior.OUTPUT
OUTPUT = custody.BASE / 'go2_tracking_posthoc_raw_accuracy_v1_attempt_001'
SOURCE = 'scripts/read_go2_tracking_posthoc_raw_accuracy_v1.py'
TEST = 'lewm/tests/test_posthoc_tracking_raw_accuracy_development.py'
PROTOCOL = 'docs/go2_independent_tracking_challenge_v1_2026-09-07.md'
ADMISSION_SOURCES = {
    'scripts/read_go2_tracking_sensor_convention_coverage_v1.py':
        '0c4681710b48ae6458551459d8f34cca2ac58bc3877f5db97f46c4c433748ae2',
    'scripts/read_go2_independent_tracking_quaternion_diagnostic_v1.py':
        'fb9addc99f0697e84ba37d5bbadf79779c586797af75f81479672e2d2f12bc0c',
    'lewm/representation_aware_tracking_coverage_development.py':
        '2a04935198bbefa992db8e0adcdf049f68a091b7b7a63ab5915dfaaf1d8e7484',
}
SCENARIOS = ('base', *stress.SCENARIOS)
WORKER_MEMORY_BYTES = 16 * 1024**3
TAPE_BYTES = 768 * 1024**2
RESERVE_BYTES = 5 * 1024**3
TOTAL_BYTES = 8 * TAPE_BYTES + 4 * base.MAX_METADATA


def source_check(bindings):
    for name, expected in bindings.items():
        path = ROOT / allowed_relative(name)
        base.require(path.resolve() == path and path.is_file(), 'ordinary source required')
        with path.open('rb') as stream:
            base.require(hashlib.file_digest(stream, 'sha256').hexdigest() == expected,
                         'source identity changed: ' + name)


def admit():
    source_check(ADMISSION_SOURCES)
    episodes, bindings, definition = admission.admitted_inputs()
    source_check(ADMISSION_SOURCES)
    return episodes, bindings, definition


def metadata(output, name, value):
    custody.validate_root(output)
    base.require('/' not in name and allowed_relative(name).name == name, 'flat named metadata required')
    path = output / name
    base.require(path.resolve() == path, 'nonsymlink output required')
    encoded = base.encode(value)
    base.require(len(encoded) <= base.MAX_METADATA, 'bounded metadata required')
    base.require(shutil.disk_usage(output).free >= RESERVE_BYTES + len(encoded), 'output reserve exhausted')
    with path.open('xb') as stream:
        stream.write(encoded)
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


class ScoreDestination:
    """Original scorer reads INPUT; this explicit adapter writes only OUTPUT."""
    def __init__(self, destination, trial):
        self.output = INPUT
        self.destination = custody.validate_root(destination)
        base.require(self.destination != INPUT and trial in base.TRIALS, 'distinct derived output required')
        self.trial = trial
        self.scenario = None
        self.hashes = {}
        self.sizes = {}

    def name(self):
        base.require(self.scenario in SCENARIOS, 'fixed scoring scenario required')
        return self.trial + '__' + self.scenario + '_evaluation.jsonl'

    @contextmanager
    def stream(self, requested):
        base.require(requested == self.trial + '_evaluation.jsonl', 'exact scorer destination interface')
        path = self.destination / self.name()
        base.require(path.resolve() == path, 'nonsymlink derived stream required')
        self.current_bytes = 0
        self.current_rows = 0
        with path.open('xb') as stream:
            yield stream
        self.sizes[path.name] = self.current_bytes
        with path.open('rb') as stream:
            self.hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()

    def append(self, target, row):
        data = base.encode(row)
        base.require(len(data) <= base.MAX_ROW and self.current_rows < 443, 'bounded evaluation row population')
        base.require(sum(self.sizes.values()) + self.current_bytes + len(data) <= TAPE_BYTES,
                     'finite per-tape byte budget')
        base.require(shutil.disk_usage(self.destination).free >= RESERVE_BYTES + len(data), 'output reserve exhausted')
        target.write(data)
        self.current_bytes += len(data)
        self.current_rows += 1


def worker_limits():
    resource.setrlimit(resource.RLIMIT_AS, (WORKER_MEMORY_BYTES, WORKER_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (443 * base.MAX_ROW, 443 * base.MAX_ROW))
    import cv2
    cv2.setNumThreads(1)


def analyze_tape(destination, trial, episode, reports, protocol_sha, sources):
    started = time.monotonic()
    stage = 'source_authentication'
    completed = {}
    store = ScoreDestination(destination, trial)
    artifacts = {}
    try:
        source_check(sources)
        stage = 'raw_sensor_contact_geometry_setup_command_stop_raster_audit'
        raw, audit = raw_audit_sensor_convention(INPUT, trial, episode, protocol_sha)
        # The raw audit consumed original bytes. Freeze its numeric arrays before
        # downstream calculations; the independent checker receives a private view.
        for value in raw.values():
            value.setflags(write=False)
        view, provenance = sensor_convention_view(raw)
        numerical.same({k: audit['coverage'][k] for k in (
            'original_frozen_norm_gate_passes', 'original_max_absolute_norm_deviation')},
            {k: provenance[k] for k in ('original_frozen_norm_gate_passes',
                                       'original_max_absolute_norm_deviation')}, tolerance=0.)
        coverage_check = numerical.verify_coverage(view, episode, audit['coverage']['coverage'],
                                                   direction=base.specification(trial)['direction'])
        name = trial + '_raw_audit.json'
        artifacts[name] = metadata(destination, name, audit)[0]
        for scenario in SCENARIOS:
            stage = 'score_and_verify:' + scenario
            report = reports[scenario]
            store.scenario = scenario
            score = scoring._score_trial(store, trial, report, raw)
            checked = numerical.verify_pose_stream(view,
                stress.rows(INPUT, report['estimates_file']),
                stress.rows(destination, store.name()), report, score)
            continuity = verify_continuity(stress.rows(INPUT, report['estimates_file']), report['continuity'])
            first_failure = dict.fromkeys(base.ARMS)
            for row in stress.rows(INPUT, report['estimates_file']):
                for arm in base.ARMS:
                    if row['arms'][arm]['pose'] is None and first_failure[arm] is None:
                        first_failure[arm] = dict(frame=row['frame'], reason=row['arms'][arm]['failure'])
            completed[scenario] = dict(score=score, arm_availability=report['arms'],
                first_failure=first_failure, continuity=continuity,
                numerical_pose_score_reconstruction_verified=checked['numerical_pose_score_reconstruction_verified'],
                stress_exposure={k: report[k] for k in ('onset_recorded', 'complete_requested_tape',
                    'changed_packet_frames', 'exposure')} if scenario != 'base' else None)
        stage = 'current_predecessor_witness'
        reader = base.IntentReturnRGBDReplay(INPUT / trial)
        witness = predecessors.extract_witness(raw, base.read(INPUT, trial + '/static_objects.json'),
            reader, base.read(INPUT, trial + '/camera_audit.json'))
        stage = 'source_reauthentication'
        source_check(sources)
        artifacts.update(store.hashes)
        custody.verify_artifacts(destination, artifacts)
        return dict(status='POSTHOC_TAPE_RAW_AND_ACCURACY_COMPLETE', trial=trial,
            coverage=audit['coverage'], coverage_check=coverage_check,
            representation=provenance, streams=completed, current_witness=witness,
            raw_audit_sha256=artifacts, strict_depth_visibility_pass=all(x['within1mm'] and
                x['physical_visibility']['passes_sampled_physical_visibility'] for x in audit['sensors']['depth_checks']),
            raw_sensor_frames=audit['sensors']['paired_frames_reconstructed'],
            stable_footprint_metric_pass=all(x['score']['stable_interior_metric_pass'] for x in audit['footprints']),
            first_physical_stop=audit['first_physical_stop'],
            elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except Exception as error:
        return dict(status='POSTHOC_TAPE_ANALYSIS_FAILED', trial=trial, stage=stage,
            error=repr(error), traceback=traceback.format_exc(), streams=completed,
            uncompleted_scenarios=[s for s in SCENARIOS if s not in completed],
            artifact_sha256=artifacts | store.hashes,
            elapsed_seconds=time.monotonic() - started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def benchmark_unit(trial):
    """Bounded actual-tape components; not a second full audit or score."""
    from scripts.near_field_sensor_audit_development import read_npz, classify, contact_packet
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.physical_execution_development import rotation_xyzw
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    from lewm.raster_footprint_visibility_development import evaluate_footprint
    started = time.monotonic()
    directory = INPUT / trial
    raw = read_npz(directory, 'physics_trace.npz')
    contacts = read_npz(directory, 'native_contacts.npz')
    topology = base.read(INPUT, trial + '/contact_topology.json')
    geometry = ArticulatedCollisionGeometry(URDF)
    fingerprint = []
    for i in range(1000):
        R = rotation_xyzw(raw['base_pose_world'][i, 3:])
        shapes = geometry.supports(raw['joint_position'][i], R)['shapes']
        fingerprint.append((len(classify(contact_packet(contacts, i), topology)),
                            [s['lower'][2] for s in shapes]))
    cameras = base.read(INPUT, trial + '/camera_audit.json')
    walls = base.specification(trial)['geometry']['wall_boxes']
    for i in range(8):
        depth = read_npz(directory, f'native_depth_{i:04d}.npz')['optical_depth_m']
        fingerprint.append(evaluate_footprint(depth, walls, cameras[i]['world_from_optical'], render_near_m=.005))
    return dict(trial=trial, sha256=hashlib.sha256(base.encode(fingerprint)).hexdigest(),
        seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def benchmark():
    # Same four independent tapes at each width; includes fresh worker startup
    # and NPZ decoding. Component speed is not a whole-audit throughput proof.
    measurements = []
    for width in (1, 2, 4):
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=width, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker_limits) as pool:
            values = list(pool.map(benchmark_unit, base.TRIALS[:4]))
        measurements.append(dict(workers=width, wall_seconds=time.monotonic() - started, tapes=values))
        print('COMPONENT_BENCHMARK', width, measurements[-1]['wall_seconds'], flush=True)
    reference = [(v['trial'], v['sha256']) for v in measurements[0]['tapes']]
    base.require(all([(v['trial'], v['sha256']) for v in m['tapes']] == reference for m in measurements),
                 'parallel component arithmetic changed')
    fastest = min(measurements, key=lambda m: m['wall_seconds'])
    return dict(measurements=measurements, selected_workers=fastest['workers'],
        identical_component_results=True, full_audit_benchmark=False,
        component_scope='1000 contact/geometry samples and eight raster footprints per tape')


def hardware():
    import psutil
    gpus = {}
    for card in ('card0', 'card1'):
        path = Path('/sys/class/drm') / card / 'device'
        gpus[card] = {key: int((path / key).read_text()) for key in (
            'gpu_busy_percent', 'mem_info_vram_total', 'mem_info_vram_used') if (path / key).is_file()}
    return dict(affinity=sorted(os.sched_getaffinity(0)), physical_cpus=psutil.cpu_count(logical=False),
        logical_cpus=psutil.cpu_count(), cpu_busy_percent=psutil.cpu_percent(interval=1),
        memory_available_bytes=psutil.virtual_memory().available, gpus=gpus,
        artifact_free_bytes=shutil.disk_usage(custody.BASE).free,
        workspace_free_bytes=shutil.disk_usage(ROOT).free,
        competing_python=[dict(pid=p.pid, name=p.info['name'], rss=p.info['memory_info'].rss)
            for p in psutil.process_iter(['name', 'memory_info'])
            if 'python' in p.info['name'].lower() and p.pid != os.getpid()])


def run():
    started = time.monotonic()
    episodes, streams, definition = admit()
    sources = discover_sources([SOURCE, TEST], definition['source_sha256'])
    source_check(sources)
    preflight = hardware()
    base.require(preflight['memory_available_bytes'] >= 4 * WORKER_MEMORY_BYTES + 4 * 1024**3,
                 'four-worker memory allowance unavailable')
    base.require(preflight['artifact_free_bytes'] >= TOTAL_BYTES + RESERVE_BYTES, 'finite output envelope unavailable')
    output = custody.create_output(OUTPUT)
    launch = dict(schema='posthoc_tracking_raw_accuracy.v1', output_root=str(output),
        source_sha256=sources, original_inner_bindings=admission.prior.INNER_BINDINGS,
        original_outside_bindings=admission.prior.OUTSIDE_BINDINGS,
        diagnostic_sha256=admission.DIAGNOSTIC_SHA, original_definition_sha256=admission.prior.DEFINITION,
        stream_sha256=streams, trials=list(base.TRIALS), scenarios=list(SCENARIOS),
        hardware=preflight, maximum_workers=4, worker_address_space_limit_bytes=WORKER_MEMORY_BYTES,
        per_tape_output_bound_bytes=TAPE_BYTES, total_output_bound_bytes=TOTAL_BYTES,
        minimum_free_space_bytes=RESERVE_BYTES, output_os_quota_enforced=False,
        raw_audits_per_tape=1, scoring_streams=96, predecessor_comparisons=48,
        reused_exact_admitted_sensor_evidence=True, sensor_transforms_reexecuted=False,
        new_raw_sensor_reconstruction_required=True, original_attempt_passed=False,
        native_simulator_executed=False, observer_inference_recomputed=False)
    metadata(output, 'launch.json', launch)
    reports = {}
    try:
        measurements = benchmark()
        source_check(sources)
        metadata(output, 'workload.json', measurements)
        phase = base.read(INPUT, 'sensor_phase_complete.json')['reports']
        treated = base.read(INPUT, stress.PHASE)['reports']
        with ProcessPoolExecutor(max_workers=measurements['selected_workers'],
                mp_context=multiprocessing.get_context('spawn'), initializer=worker_limits) as pool:
            futures = {pool.submit(analyze_tape, output, trial, episodes[trial]['result'],
                {'base': phase[trial]} | treated[trial], definition['source_sha256'][PROTOCOL], sources): trial
                for trial in base.TRIALS}
            for future in as_completed(futures):
                trial = futures[future]
                try:
                    report = future.result()
                except Exception as error:
                    report = dict(status='POSTHOC_WORKER_FAILED', trial=trial, error=repr(error),
                                  streams={}, uncompleted_scenarios=list(SCENARIOS))
                reports[trial] = report
                metadata(output, trial + '_result.json', report)
                print('TAPE_COMPLETE', trial, report['status'], report.get('elapsed_seconds'), flush=True)
        complete = all(r['status'] == 'POSTHOC_TAPE_RAW_AND_ACCURACY_COMPLETE' for r in reports.values())
        prior = predecessors.load_predecessors() if complete else None
        comparison = {t: {key + '/' + old: predecessors.compare_witnesses(w, reports[t]['current_witness'])
            for key, cohort in prior['cohorts'].items() for old, w in cohort['witnesses'].items()}
            for t in base.TRIALS} if complete else {}
        if complete:
            metadata(output, 'predecessors.json', dict(predecessors=prior, comparisons=comparison))
            for cohort in prior['cohorts'].values():
                custody.verify_artifacts(Path(cohort['root']),
                    cohort['receipt_sha256'] | cohort['selected_artifact_sha256'])
        # Authenticate all original raw, phase, stream, terminal and source bytes
        # again. This reuses sensor admission and does not repeat raw audits.
        again_episodes, again_streams, again_definition = admit()
        base.require(again_episodes == episodes and again_streams == streams and again_definition == definition,
                     'input identities changed during analysis')
        source_check(sources)
        artifact_names = ['launch.json', 'workload.json'] + [t + '_result.json' for t in base.TRIALS]
        if complete:
            artifact_names.append('predecessors.json')
        artifact_bindings = {}
        total = 0
        for t, report in reports.items():
            artifact_names.extend(report.get('raw_audit_sha256', report.get('artifact_sha256', {})))
        for name in artifact_names:
            path = custody.artifact_path(output, name)
            total += path.stat().st_size
            with path.open('rb') as stream:
                artifact_bindings[name] = hashlib.file_digest(stream, 'sha256').hexdigest()
        base.require(total <= TOTAL_BYTES - base.MAX_METADATA, 'total finite output budget exceeded')
        result = dict(status='POSTHOC_RAW_ACCURACY_AND_NONIDENTITY_COMPLETE' if complete else 'POSTHOC_ANALYSIS_INCOMPLETE',
            output_root=str(output), artifact_sha256=artifact_bindings, artifact_bytes_before_result=total,
            trial_status={t: reports[t]['status'] for t in base.TRIALS},
            completed_raw_audits=sum(r['status'] == 'POSTHOC_TAPE_RAW_AND_ACCURACY_COMPLETE' for r in reports.values()),
            completed_pose_streams=sum(len(r['streams']) for r in reports.values()),
            predecessor_comparisons=sum(len(v) for v in comparison.values()),
            all_predecessor_nonidentity_checks_pass=all(c['nonidentity_checks_pass'] for v in comparison.values()
                for c in v.values()) if comparison else None,
            all_intended_motion_covered=all(r['coverage']['coverage']['intended_motion_covered'] for r in reports.values()) if complete else None,
            strict_depth_visibility_pass=all(r['strict_depth_visibility_pass'] for r in reports.values()) if complete else None,
            elapsed_seconds=time.monotonic() - started, hardware_after=hardware(),
            source_and_input_bytes_reauthenticated=True, original_failure_preserved=True,
            original_arrays_modified=False, original_attempt_passed=False,
            original_attempt_resumed=False, native_simulator_executed=False,
            observer_inference_recomputed=False, stress_sensor_transforms_reexecuted=False,
            independent_numerical_arithmetic_on_private_normalized_view=True,
            independent_native_data=False, independent_scene_count_established=False,
            shared_gyro_bias_correction_established=False, full_challenge_pass=False,
            navigation_qualified=False, real_time_qualified=False, goal_achieved=False)
        metadata(output, 'result.json', result)
        return result
    except Exception as error:
        metadata(output, 'failure.json', dict(status='POSTHOC_ANALYSIS_FAILED', error=repr(error),
            traceback=traceback.format_exc(), completed_trials=list(reports),
            original_attempt_passed=False, original_failure_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(base.encode(run()).decode(), end='', flush=True)
