"""Fixed challenge process-tree containment and bounded outside failure evidence.

No generic unit launcher, cgroup writer, experiment retry, or native initializer.
The bound is charged cgroup memory, not a proof of peak RSS or workload fit.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from scripts import tracking_kernel_scope_development as kernel
from scripts.tracking_kernel_scope_development import unified_group

from scripts.navigation_artifact_root_development import (
    BASE, create_output, validate_root, artifact_path, verify_artifacts)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = BASE / 'go2_independent_tracking_supervision_v1_attempt_001'
INNER_OUTPUT = BASE / 'go2_independent_tracking_challenge_v1_attempt_001'
UNIT = kernel.CHALLENGE_UNIT
PROBE_OUTPUTS = {mode: BASE / f'go2_tracking_keeper_memory_{mode}_v1_attempt_001'
    for mode in kernel.PROBE_UNITS}
MEMORY_BYTES = 8 * 1024**3
TASKS = 512
RUNTIME_SECONDS = 48 * 60 * 60
FILE_BYTES = 32 * 1024**2
TOTAL_BYTES = 3 * FILE_BYTES
RESERVE_BYTES = 40 * 1024**3
FILES = ('request.json', 'unit.log', 'terminal.json')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def hash_value(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value), 'exact SHA256 required')
    return value


def contract():
    return dict(schema='independent_tracking_memory_scope.v1', unit=UNIT,
        memory_max_bytes=MEMORY_BYTES, swap_max_bytes=0, oom_group=1,
        maximum_tasks=TASKS, runtime_max_seconds=RUNTIME_SECONDS,
        parent_replay_scoring_and_native_children_in_same_scope=True,
        supervisor_output_root=str(OUTPUT), supervisor_files=list(FILES),
        supervisor_maximum_bytes=TOTAL_BYTES,
        workload_fit_proved=False, strict_instantaneous_rss_bound_proved=False)


def outside_scope():
    group = unified_group(Path('/proc/self/cgroup').read_text())
    require(UNIT not in group.parts, 'supervisor must be outside the workload OOM group')
    return dict(pid=os.getpid(), cgroup=str(group))


def own_scope():
    require(kernel.profile(UNIT) == dict(memory_bytes=MEMORY_BYTES, tasks=TASKS,
        runtime_seconds=RUNTIME_SECONDS), 'unchanged challenge resource profile required')
    return kernel.admit_scope(UNIT)


def require_fresh_unit():
    kernel.require_fresh_unit(UNIT)


def request(definition_sha256, study_result_sha256):
    return dict(schema='independent_tracking_outside_supervision_request.v1',
        definition_sha256=hash_value(definition_sha256),
        study_result_sha256=hash_value(study_result_sha256),
        scope_contract=contract(), output_root=str(INNER_OUTPUT), supervisor=outside_scope())


def admit_request(request_sha256, definition_sha256, study_result_sha256):
    scope = own_scope()
    verify_artifacts(OUTPUT, {'request.json': hash_value(request_sha256)})
    path = artifact_path(OUTPUT, 'request.json')
    require(path.stat().st_size <= FILE_BYTES, 'bounded outside request')
    row = json.loads(path.read_text())
    require(set(row) == {'schema', 'definition_sha256', 'study_result_sha256',
        'scope_contract', 'output_root', 'supervisor'}
        and row['schema'] == 'independent_tracking_outside_supervision_request.v1'
        and row['definition_sha256'] == hash_value(definition_sha256)
        and row['study_result_sha256'] == hash_value(study_result_sha256)
        and row['scope_contract'] == contract() and row['output_root'] == str(INNER_OUTPUT),
        'exact external request, study and definition required')
    supervisor = row['supervisor']
    require(set(supervisor) == {'pid', 'cgroup'} and type(supervisor['pid']) is int
        and supervisor['pid'] > 0, 'outside supervisor identity required')
    process = Path('/proc') / str(supervisor['pid'])
    require(process.stat().st_uid == os.getuid(), 'owned live outside supervisor required')
    group = unified_group((process / 'cgroup').read_text())
    require(str(group) == supervisor['cgroup'] and UNIT not in group.parts,
        'live outside supervisor must remain outside workload OOM group')
    require(not (OUTPUT / 'terminal.json').exists(), 'terminal supervised attempt cannot resume')
    return dict(request_sha256=request_sha256, scope=scope, supervisor=supervisor)


def service_command(python, source, environment, definition_sha256, study_result_sha256, request_sha256):
    require(Path(python) == ROOT / '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python'
        and source == 'scripts/run_go2_independent_tracking_challenge_v1.py', 'fixed challenge command only')
    for value in (definition_sha256, study_result_sha256, request_sha256):
        hash_value(value)
    return kernel.service_prefix(UNIT) + [
        f'--setenv={k}={v}' for k, v in sorted(environment.items())] + [
        str(python), str(ROOT / source), '--scoped-parent',
        '--definition-sha256', definition_sha256, '--study-result-sha256', study_result_sha256,
        '--supervisor-request-sha256', request_sha256]


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class EvidenceStore:
    """Three exclusive bounded files outside the challenge; never scans artifacts."""
    def __init__(self, *, probe_mode=None):
        require(probe_mode is None or probe_mode in PROBE_OUTPUTS, 'exact optional tiny probe mode')
        self.output = create_output(OUTPUT if probe_mode is None else PROBE_OUTPUTS[probe_mode])

    def fresh_path(self, name):
        require(name in FILES, 'exact supervisor evidence roster')
        validate_root(self.output)
        path = self.output / name
        require(path.resolve() == path and not path.exists() and not path.is_symlink(),
            'exclusive supervisor evidence; no overwrite or retry')
        require(shutil.disk_usage(self.output).free >= RESERVE_BYTES + FILE_BYTES,
            'outside evidence storage reserve exhausted')
        return path

    def save(self, name, value):
        require(name in ('request.json', 'terminal.json'), 'JSON evidence path required')
        # Metadata consists only of fixed requests and bounded status fields,
        # never native data, captured stdout, or recursively enumerated artifacts.
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        require(len(raw) <= FILE_BYTES, 'bounded supervisor JSON')
        path = self.fresh_path(name)
        with path.open('xb') as stream:
            require(stream.write(raw) == len(raw), 'complete evidence write')
            stream.flush(); os.fsync(stream.fileno())
        sync_directory(self.output)
        return sha256(path)


def relay_process(command, store):
    """Wait on the exact child handle, retaining a bounded prefix and terminal tail.

The log cap is NOT permission to truncate scientific recordings. A truncated
diagnostic log disqualifies supervised completion. No retry and no fallback.
The caller distinguishes supervisor interruption from known child termination.
"""
    path = store.fresh_path('unit.log')
    half = FILE_BYTES // 2
    total = prefix = 0
    tail = bytearray()
    with path.open('xb', buffering=0) as log:
        sync_directory(store.output)
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        try:
            while True:
                chunk = process.stdout.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                n = min(len(chunk), half - prefix)
                if n:
                    require(log.write(chunk[:n]) == n, 'complete log prefix write')
                    prefix += n
                tail.extend(chunk[n:])
                if len(tail) > half:
                    del tail[:-half]
            returncode = process.wait()
        finally:
            # Popen's context manager would wait on an interrupted supervisor,
            # potentially hiding the interruption until a 48h workload ended.
            # Do not infer unit termination or restart it on an observation error.
            process.stdout.close()
            os.fsync(log.fileno())
        if tail:
            require(log.write(tail) == len(tail), 'complete log tail write')
        os.fsync(log.fileno())
    sync_directory(store.output)
    return dict(systemd_run_returncode=returncode, child_handle_terminal=True,
        workload_termination_inferred_from_returncode=False,
        log_total_bytes=total, log_retained_bytes=prefix + len(tail),
        log_omitted_bytes=total - prefix - len(tail),
        log_sha256=sha256(path), log_complete=total <= FILE_BYTES,
        failure_cause_inferred_from_exit_code=False)
