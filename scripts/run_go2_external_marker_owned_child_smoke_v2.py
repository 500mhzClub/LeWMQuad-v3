"""One synthetic attach-mode probe; preserve failure and reap both owned children.

This does not accept a PID argument or attach to any pre-existing process.
It grants no full replay admission and reports no performance qualification.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / '.generated/tools/go2_py_spy_0_4_2_v1/py-spy'
TOOL_SHA = '9b4d1f39b2a47ae44f4c6a46f615dcc0287d7755beba5065f32391951e07d594'
OUTPUT = TOOL.parent / 'marker_owned_child_smoke_v2'
CHILD = 'scripts/external_body_projection_marker_smoke_child_v2.py'
SOURCES = [CHILD, 'scripts/run_go2_external_marker_owned_child_smoke_v2.py',
    'scripts/external_body_projection_profile_replay_development.py',
    'scripts/body_projection_external_sample_summary_development.py']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, data):
    with (OUTPUT / name).open('x') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity(process):
    p = psutil.Process(process.pid)
    return dict(pid=p.pid, created=p.create_time(), command=p.cmdline())


def wait_for_text(path, predicate, process):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        value = path.read_text()
        if predicate(value):
            return value
        if process.poll() is not None:
            raise RuntimeError('owned synthetic process ended before handshake')
        time.sleep(.02)
    raise TimeoutError('synthetic handshake exceeded thirty seconds')


def main():
    if digest(TOOL) != TOOL_SHA:
        raise ValueError('pinned profiler binary required')
    OUTPUT.mkdir(exist_ok=False)
    children = []
    handles = []
    execution = dict(boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        binary_sha256=TOOL_SHA, parent=identity(psutil.Process()),
        explicit_source_sha256={name:digest(ROOT/name) for name in SOURCES},
        recursive_source_closure_claimed=False, existing_process_attach=False,
        global_ptrace_settings_changed=False)
    write('launch.json', execution)
    try:
        def log(name):
            stream = (OUTPUT/name).open('x')
            handles.append(stream)
            return stream
        child = subprocess.Popen([sys.executable, '-B', CHILD], cwd=ROOT,
            stdin=subprocess.PIPE, stdout=log('child_stdout.txt'), stderr=log('child_stderr.txt'),
            text=True)
        children.append(child)
        execution['child'] = identity(child)
        ready_text = wait_for_text(OUTPUT/'child_stdout.txt', lambda s:'\n' in s, child)
        ready = json.loads(ready_text.splitlines()[0])
        if ready != dict(status='READY', pid=child.pid, parent=os.getpid(), native_thread_id=child.pid):
            raise ValueError('original owned main-thread readiness required')
        command = [str(TOOL), 'record', '--format', 'speedscope', '--output',
            str(OUTPUT/'profile.json'), '--rate', '100', '--idle', '--threads',
            '--full-filenames', '--pid', str(child.pid)]
        profiler = subprocess.Popen(command, cwd=ROOT,
            stdout=log('profiler_stdout.txt'), stderr=log('profiler_stderr.txt'))
        children.append(profiler)
        execution['profiler'] = identity(profiler)
        wait_for_text(OUTPUT/'profiler_stdout.txt', lambda s:'Sampling process' in s, profiler)
        child.stdin.write('GO\n'); child.stdin.flush(); child.stdin.close()
        execution['child_returncode'] = child.wait(timeout=30)
        execution['profiler_returncode'] = profiler.wait(timeout=30)
        if execution['child_returncode'] != 0 or execution['profiler_returncode'] != 0:
            raise RuntimeError('both original child and profiler must exit zero')
        stdout = (OUTPUT/'child_stdout.txt').read_text().splitlines()
        if len(stdout) != 2:
            raise ValueError('exact synthetic child output required')
        completed = json.loads(stdout[1])
        from scripts.external_body_projection_profile_replay_development import MARKERS
        from scripts.body_projection_external_sample_summary_development import summarize
        if completed != dict(status='EXTERNAL_OBSERVATION_MARKER_SMOKE_COMPLETE',
                pid=child.pid, native_thread_id=child.pid, frames=list(MARKERS),
                controller_executed=False, raw_sensor_or_checkpoint_access=False):
            raise ValueError('complete original marker workload required')
        summary = summarize(json.loads((OUTPUT/'profile.json').read_text()),
            marker_source_file=str(ROOT/SOURCES[2]), owner_native_thread_id=child.pid)
        if any(row['function'] != 'synthetic_observation'
                for window in summary['windows'].values() for row in window['sampled_leaf_locations']):
            raise ValueError('only actual synthetic observation descendants expected')
        execution['both_owned_children_reaped'] = True
        write('execution.json', execution)
        artifacts = ['launch.json', 'execution.json', 'profile.json', 'child_stdout.txt',
            'child_stderr.txt', 'profiler_stdout.txt', 'profiler_stderr.txt']
        result = dict(status='EXTERNAL_MARKER_OWNED_CHILD_SMOKE_V2_COMPLETE',
            summary=summary, artifact_sha256={name:digest(OUTPUT/name) for name in artifacts},
            child_returncode=0, profiler_returncode=0, both_owned_children_reaped=True,
            controller_executed=False, profiler_overhead_qualified=False)
        write('result.json', result)
        print(json.dumps(dict(status=result['status'], marked_samples=summary['marked_samples'],
            result_sha256=digest(OUTPUT/'result.json'))), flush=True)
    except BaseException as error:
        # Termination applies only to the two Popen objects created by this probe.
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait(timeout=5)
        execution['owned_returncodes'] = [child.returncode for child in children]
        if not (OUTPUT/'execution.json').exists():
            write('execution.json', execution)
        write('failure.json', dict(status='TERMINAL_SYNTHETIC_SMOKE_FAILURE',
            error_type=type(error).__name__, reason=str(error), automatic_retry=False))
        raise
    finally:
        for stream in handles:
            stream.close()


if __name__ == '__main__':
    main()
