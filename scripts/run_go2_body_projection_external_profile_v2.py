"""Own one externally sampled full replay; never attach to an existing job."""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import psutil

from scripts.external_body_projection_profile_v2_bindings_development import replay, admission, failed_sources
from scripts.body_projection_external_sample_summary_development import summarize
from scripts.navigation_artifact_root_development import artifact_path, create_output, validate_root, verify_artifacts
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, verify, write_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.await_go2_all_phase_translation_bias_v1 import owner_live, BOOT

SOURCE = 'scripts/run_go2_body_projection_external_profile_v2.py'
TEST = 'lewm/tests/test_body_projection_external_profile_v2_development.py'
CHECKER = 'scripts/verify_go2_body_projection_external_profile_completion_v2.py'
PROTOCOL = 'docs/go2_body_projection_external_profile_v2_2026-09-11.md'
OUTPUT = replay.OUTPUT
TOOL = ROOT/'.generated/tools/go2_py_spy_0_4_2_v1/py-spy'
TOOL_SHA = '9b4d1f39b2a47ae44f4c6a46f615dcc0287d7755beba5065f32391951e07d594'
SMOKE = '.generated/tools/go2_py_spy_0_4_2_v1/marker_nonblocking_owned_child_smoke_v3/result.json'
SMOKE_SHA = '341eb89e317ba7d99ea530c9933df1b8e158ed70a9173f60c6062c64c92c7369'
PROBE = 'docs/go2_body_projection_external_profile_actual_admission_2026-09-11.json'
PROBE_SHA = 'ca94e56b28c99a3b36904b25ebe5d6e76e28ba1e9f3cf90c65117ef9a1a5913b'
SLOTS = {
    'docs/go2_body_projected_tiled_controller_replay_execution_2026-09-11.json':
        '39755d9dbec68b6fd5525dd9389c741a22d4e7ae0e30626a1fbe7fe91092e351',
    'docs/go2_body_projected_tiled_completion_watch_execution_2026-09-11.json':
        '941b495b77588555b74a550b1d822ca49ed31bb94a302377d202d424782514dd',
    'docs/go2_body_projection_external_profile_actual_admission_execution_2026-09-11.json':
        'cd1ee0aea24857295f2a72db24b1ea3df388ad75c8b91cb9033ddece2f7a172b',
}
ENVIRONMENT = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    PYTHONHASHSEED='0', OPENCV_OPENCL_RUNTIME='disabled')


def owner(pid=None):
    process = psutil.Process(pid)
    return dict(pid=process.pid, created=process.create_time(), command=process.cmdline())


def read(name, maximum=32*1024**2):
    path = artifact_path(OUTPUT, name)
    if path.stat().st_size > maximum:
        raise ValueError('bounded explicit profile artifact required')
    return json.loads(path.read_text())


def verify_tool():
    if TOOL.resolve() != TOOL or digest(TOOL) != TOOL_SHA:
        raise ValueError('exact pinned nonsymlink external profiler required')


def sources_and_resources():
    if (not __debug__ or any(os.environ.get(k) != v for k,v in ENVIRONMENT.items())
            or replay.original.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic environment required')
    verify_tool()
    verify({SMOKE:SMOKE_SHA, PROBE:PROBE_SHA} | SLOTS)
    probe = json.loads((ROOT/PROBE).read_text())
    if (probe['status'] != 'BODY_PROJECTION_EXTERNAL_ACTUAL_ADMISSION_VERIFIED'
            or probe['reference_verification_sha256'] != admission.VERIFICATION_SHA
            or probe['original_rows'] != 1428 or probe['original_completion_reconstructed'] is not True):
        raise ValueError('completed actual admission probe required')
    for name in SLOTS:
        record = json.loads((ROOT/name).read_text())
        if record['boot_id'] != BOOT or owner_live(record['owner']):
            raise ValueError('original CPU replay, completion watcher and admission probe must be ended')
    witness = admission.reference_witness()
    seeds = (SOURCE, TEST, CHECKER, PROTOCOL, SMOKE, PROBE, *SLOTS,
        'scripts/body_projection_external_profile_admission_development.py',
        'lewm/tests/test_body_projection_external_profile_admission_development.py',
        'lewm/tests/test_external_body_projection_profile_replay_development.py',
        'lewm/tests/test_body_projection_external_sample_summary_development.py',
        'docs/go2_external_body_projection_replay_source_derivative_2026-09-11.json')
    sources = discover_sources(seeds, failed_sources(witness['source_sha256']))
    verify(sources)
    hardware = replay.original.reference.hardware()
    replay.paired.original.resources_for(hardware)
    return sources, hardware


def profiler_command(pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError('original owned child PID required')
    return [str(TOOL), 'record', '--nonblocking', '--format', 'speedscope', '--output',
        str(OUTPUT/'profile.json'), '--rate', '100', '--idle', '--threads',
        '--full-filenames', '--pid', str(pid)]


def authenticate_launch(sha):
    verify_artifacts(OUTPUT, {'launch.json':sha})
    launch = read('launch.json')
    if (launch['boot_id'] != BOOT or launch['tool_sha256'] != TOOL_SHA
            or launch['reference_verification_sha256'] != admission.VERIFICATION_SHA
            or launch['environment'] != ENVIRONMENT
            or launch['input_admission_pending_at_parent_launch'] is not True
            or launch['automatic_retry'] is not False or launch['sampling_mode'] != 'nonblocking'
            or launch['stack_snapshot_consistency_guaranteed'] is not False):
        raise ValueError('exact prospective launch required')
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('original boot required')
    verify(launch['source_sha256']); verify_tool()
    return launch


def child(launch_sha):
    launch = authenticate_launch(launch_sha)
    if os.getppid() != launch['owner']['pid'] or not owner_live(launch['owner']):
        raise ValueError('live original owning parent required before input admission')
    execution = dict(owner=owner(), boot_id=BOOT, parent=launch['owner'], launch_sha256=launch_sha)
    write_json(OUTPUT/'child_execution.json', execution)
    try:
        before = admission.admit_completed()
        witness, prior, _, rows = before
        verify(launch['source_sha256'])
        libc = ctypes.CDLL(None, use_errno=True)
        libc.prctl.argtypes = [ctypes.c_int]+[ctypes.c_ulong]*4
        libc.prctl.restype = ctypes.c_int
        if libc.prctl(0x59616d61, os.getppid(), 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), 'parent-scoped profiler permission failed')
        ready = dict(status='ORIGINAL_INPUTS_ADMITTED_CHILD_READY',
            child_execution_sha256=digest(OUTPUT/'child_execution.json'), native_thread_id=os.getpid(),
            reference_verification_sha256=admission.VERIFICATION_SHA)
        write_json(OUTPUT/'child_ready.json', ready)
        print('EXTERNAL_PROFILE_CHILD_READY', digest(OUTPUT/'child_ready.json'), flush=True)
        line = sys.stdin.readline()
        if re.fullmatch(r'GO [0-9a-f]{64}\n', line) is None:
            raise ValueError('original parent profiler-start handshake required')
        verify_artifacts(OUTPUT, {'profiler_execution.json':line.split()[1]})
        profiler = read('profiler_execution.json')
        if (profiler['owner']['command'] != profiler_command(os.getpid())
                or profiler['child'] != execution['owner'] or profiler['parent'] != launch['owner']
                or profiler['boot_id'] != BOOT or not owner_live(profiler['owner'])
                or not owner_live(launch['owner'])):
            raise ValueError('live exact original profiler and parent required')
        replay.original.cv2.setNumThreads(1); replay.original.torch.set_num_threads(1)
        replay.original.torch.use_deterministic_algorithms(True)
        report = replay.replay(rows, prior['report'])
        if admission.admit_completed() != before:
            raise ValueError('complete original inputs changed during replay')
        verify(launch['source_sha256']); verify_tool()
        if not owner_live(profiler['owner']) or not owner_live(launch['owner']):
            raise ValueError('original profiler and parent must remain live through final admission')
        write_json(OUTPUT/'child_result.json', dict(status='BODY_PROJECTION_EXTERNALLY_SAMPLED_CHILD_COMPLETE',
            launch_sha256=launch_sha, child_execution_sha256=digest(OUTPUT/'child_execution.json'),
            profiler_execution_sha256=digest(OUTPUT/'profiler_execution.json'),
            comparison_sha256=digest(OUTPUT/'comparison.jsonl'), report=report,
            sensing_scope=witness['sensing_scope'], original_inputs_reauthenticated_before_and_after=True))
        print('EXTERNAL_PROFILE_CHILD_COMPLETE', digest(OUTPUT/'child_result.json'), flush=True)
    except BaseException as error:
        write_json(OUTPUT/'child_failure.json', dict(status='TERMINAL_EXTERNAL_PROFILE_CHILD_FAILURE',
            reason=repr(error), automatic_retry=False))
        raise


def wait_for_line(path, prefix, process):
    """Wait on this owned process; elapsed observation time never restarts it."""
    while True:
        with path.open('rb') as stream:
            size = stream.seek(0, 2); stream.seek(max(0, size-32768))
            complete = stream.read(32768).decode().split('\n')[:-1]
        for line in complete:
            if line.startswith(prefix):
                return line
        if process.poll() is not None:
            raise RuntimeError('original owned process ended before readiness')
        time.sleep(.2)


def validate_report(report, rows, prior_rows, prior_report):
    replay.compare_profile_rows(rows, prior_rows)
    true_flags = ('retained_state_identity_established', 'complete_original_decisions_reconstructed',
        'model_state_unchanged', 'sensor_acquisition_profiled', 'no_observation_1428_consumed',
        'invocation_frozen_footprint_receipts', 'complete_normalized_candidate_decisions_exact',
        'external_profiler_covers_entire_child', 'controller_window_markers_provided',
        'normalization_outside_marked_controller_windows')
    false_flags = ('controller_observe_only_profiled', 'profiler_overhead_removed', 'isolated_benchmark',
        'speedup_established', 'native_execution', 'policy_changed', 'real_time_qualified',
        'navigation_qualified', 'normalization_outside_profiled_region', 'cprofile_hooks_enabled',
        'external_profile_file_verified')
    fixed = dict(frames=1428, raw_model_forecast_comparisons=1425, last_replayed_observation=1427,
        controller='BodyProjectedTiledController', model_state_sha256=replay.original.reference.MODEL_SHA,
        observed_state_checks=prior_report['observed_state_checks'], normalized_state_type_paths=replay.STATE_TYPE_PATHS)
    keys = set(true_flags+false_flags) | set(fixed) | {'windows','state_size_snapshots'}
    if (set(report) != keys or any(report[k] is not True for k in true_flags)
            or any(report[k] is not False for k in false_flags)
            or any(report[k] != v for k,v in fixed.items())):
        raise ValueError('exact original full replay, state identities and negative scope required')
    if set(report['windows']) != set(replay.WINDOWS) or set(report['state_size_snapshots']) != set(replay.WINDOWS):
        raise ValueError('three original windows and descriptive size snapshots required')
    for row in rows:
        frame = row['frame']
        expected = replay.MARKER_WINDOWS.get(frame)
        value = row['controller_wall_s']
        if (row['profiled_window'] != expected or type(value) not in (int,float)
                or not math.isfinite(value) or value <= 0):
            raise ValueError('all original window assignments and finite measured times required')
    for name, (first,last) in replay.WINDOWS.items():
        window = report['windows'][name]
        if (set(window) != {'observations','python_stack_markers'}
                or window['python_stack_markers'] != [replay.MARKERS[i] for i in range(first,last+1)]
                or len(window['observations']) != 10):
            raise ValueError('all thirty exact observation markers required')
        for frame, observation in zip(range(first,last+1), window['observations'], strict=True):
            if (set(observation) != {'frame','action','controller_wall_s_with_profiling'}
                    or observation['frame'] != frame
                    or observation['controller_wall_s_with_profiling'] != rows[frame]['controller_wall_s']
                    or (name == 'repeated_hold' and observation['action'] != 'hold')):
                raise ValueError('window observations must match actual complete timing rows')


def inspect_completed_child(launch_sha, prior):
    launch = authenticate_launch(launch_sha)
    for name in ('failure.json','child_failure.json'):
        if (OUTPUT/name).exists() or (OUTPUT/name).is_symlink():
            raise ValueError('failed original profiling attempt cannot be accepted')
    execution = read('execution.json'); child_execution = read('child_execution.json')
    profiler = read('profiler_execution.json'); result = read('child_result.json')
    if (execution['child_returncode'] != 0 or execution['profiler_returncode'] != 0
            or execution['both_owned_children_reaped'] is not True
            or execution['launch_sha256'] != launch_sha
            or execution['child'] != child_execution['owner'] or execution['profiler'] != profiler['owner']
            or child_execution['parent'] != launch['owner'] or profiler['parent'] != launch['owner']
            or profiler['child'] != child_execution['owner']
            or profiler['owner']['command'] != profiler_command(child_execution['owner']['pid'])
            or any(record['boot_id'] != BOOT for record in (execution, child_execution, profiler))
            or owner_live(child_execution['owner']) or owner_live(profiler['owner'])):
        raise ValueError('original child and profiler must both be ended and reaped successfully')
    bindings = {'launch.json':result['launch_sha256'], 'comparison.jsonl':result['comparison_sha256'],
        'child_execution.json':result['child_execution_sha256'],
        'profiler_execution.json':result['profiler_execution_sha256']}
    verify_artifacts(OUTPUT, bindings)
    if (result['status'] != 'BODY_PROJECTION_EXTERNALLY_SAMPLED_CHILD_COMPLETE'
            or result['launch_sha256'] != launch_sha
            or result['original_inputs_reauthenticated_before_and_after'] is not True
            or result['sensing_scope'] != prior[0]['sensing_scope']):
        raise ValueError('complete original child admission and sensing scope required')
    rows = [json.loads(line) for line in artifact_path(OUTPUT,'comparison.jsonl').read_text().splitlines()]
    validate_report(result['report'], rows, prior[3], prior[1]['report'])
    summary = summarize(read('profile.json', maximum=256*1024**2),
        marker_source_file=str(ROOT/replay.SOURCE), owner_native_thread_id=child_execution['owner']['pid'])
    log = artifact_path(OUTPUT, 'profiler_stdout.txt').read_text()
    counts = re.findall(r'Samples: ([0-9]+) Errors: ([0-9]+)', log)
    if counts != [(str(summary['total_samples']), '0')]:
        raise ValueError('complete profiler sample accounting and zero reported sampling errors required')
    return result, summary


def parent(preflight_only=False):
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive full external profile; no retry or resume')
    sources, hardware = sources_and_resources()
    if preflight_only:
        print('BODY_PROJECTION_EXTERNAL_PROFILE_PREFLIGHT', len(sources), json.dumps(hardware), flush=True)
        return
    create_output(OUTPUT)
    launch = dict(source_sha256=sources, owner=owner(), boot_id=BOOT, hardware=hardware,
        environment=ENVIRONMENT, tool_sha256=TOOL_SHA, reference_verification_sha256=admission.VERIFICATION_SHA,
        input_admission_pending_at_parent_launch=True, automatic_retry=False,
        native_execution=False, model_training=False, sampling_mode='nonblocking',
        stack_snapshot_consistency_guaranteed=False)
    write_json(OUTPUT/'launch.json', launch); launch_sha = digest(OUTPUT/'launch.json')
    print('BODY_PROJECTION_EXTERNAL_PROFILE_LAUNCHED', launch_sha, flush=True)
    children=[]; logs=[]
    execution=dict(launch_sha256=launch_sha, boot_id=BOOT, parent=launch['owner'])
    def log(name):
        stream=(OUTPUT/name).open('x'); logs.append(stream); return stream
    try:
        process = subprocess.Popen([sys.executable,'-B',SOURCE,'--child-launch-sha256',launch_sha],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=log('child_stdout.txt'), stderr=log('child_stderr.txt'), text=True)
        children.append(process); execution['child']=owner(process.pid)
        line=wait_for_line(OUTPUT/'child_stdout.txt', 'EXTERNAL_PROFILE_CHILD_READY ', process)
        verify_artifacts(OUTPUT, {'child_ready.json':line.split()[-1]})
        ready=read('child_ready.json'); child_execution=read('child_execution.json')
        if (ready['native_thread_id'] != process.pid or child_execution['owner'] != execution['child']
                or child_execution['parent'] != launch['owner']):
            raise ValueError('exact newly owned admitted child required')
        verify_artifacts(OUTPUT, {'child_execution.json':ready['child_execution_sha256']})
        profiler=subprocess.Popen(profiler_command(process.pid), cwd=ROOT,
            stdout=log('profiler_stdout.txt'), stderr=log('profiler_stderr.txt'))
        children.append(profiler); execution['profiler']=owner(profiler.pid)
        write_json(OUTPUT/'profiler_execution.json',dict(owner=execution['profiler'],
            child=execution['child'], parent=launch['owner'], boot_id=BOOT))
        wait_for_line(OUTPUT/'profiler_stdout.txt','py-spy> Sampling process',profiler)
        process.stdin.write('GO '+digest(OUTPUT/'profiler_execution.json')+'\n')
        process.stdin.flush(); process.stdin.close()
        while process.poll() is None:
            if profiler.poll() is not None and process.poll() is None:
                raise RuntimeError('original profiler ended before original child')
            time.sleep(.2)
        execution['child_returncode']=process.wait()
        execution['profiler_returncode']=profiler.wait()
        execution['both_owned_children_reaped']=True
        write_json(OUTPUT/'execution.json',execution)
        if execution['child_returncode'] != 0 or execution['profiler_returncode'] != 0:
            raise RuntimeError('both original subprocesses must exit zero')
        prior=admission.admit_completed()
        result, summary=inspect_completed_child(launch_sha, prior)
        names=('launch.json','execution.json','child_execution.json','profiler_execution.json',
            'child_ready.json','child_result.json','comparison.jsonl','profile.json',
            'child_stdout.txt','child_stderr.txt','profiler_stdout.txt','profiler_stderr.txt')
        bindings={name:digest(artifact_path(OUTPUT,name)) for name in names}
        verify_artifacts(OUTPUT,bindings); verify(sources)
        write_json(OUTPUT/'result.json',dict(status='BODY_PROJECTION_EXTERNAL_PROFILE_V2_COMPLETE',
            source_sha256=sources, artifact_sha256=bindings, summary=summary,
            sensing_scope=result['sensing_scope'], actual_reference_reauthenticated_after_child_exit=True,
            controller='BodyProjectedTiledController', native_execution=False,
            sampling_mode='nonblocking', stack_snapshot_consistency_guaranteed=False,
            real_time_qualified=False, navigation_qualified=False, goal_achieved=False))
        print('BODY_PROJECTION_EXTERNAL_PROFILE_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        for process in reversed(children):
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
        if not (OUTPUT/'execution.json').exists():
            write_json(OUTPUT/'execution.json',execution | {'owned_returncodes':[p.returncode for p in children]})
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_EXTERNAL_FULL_PROFILE_FAILURE',
            reason=repr(error), automatic_retry=False, evidence_preserved=True))
        raise
    finally:
        for stream in logs: stream.close()


def main():
    parser=argparse.ArgumentParser(); group=parser.add_mutually_exclusive_group()
    group.add_argument('--source-preflight-only',action='store_true')
    group.add_argument('--child-launch-sha256')
    args=parser.parse_args()
    if args.child_launch_sha256: child(args.child_launch_sha256)
    else: parent(args.source_preflight_only)


if __name__ == '__main__':
    main()
