"""Fixed tiny parent/leaf workloads for NEW outside-keeper fit and group-OOM tests.

No ML/native imports, data/artifact writes, checkpoints or experiments. Each
process checks its actual admitted 64MiB/no-swap/group-OOM/16-task scope before
allocating or starting the leaf. Overflow intentionally kills this probe group.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts import tracking_kernel_scope_development as kernel

SOURCE = 'scripts/probe_go2_tracking_keeper_memory_development.py'
KERNEL_SOURCE = 'scripts/tracking_kernel_scope_development.py'
PYTHON = kernel.ROOT / '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python'
GRANT = b'ALLOCATE_ONCE\n'


def verify_sources(source_sha256, kernel_sha256):
    for name, expected in ((SOURCE, source_sha256), (KERNEL_SOURCE, kernel_sha256)):
        path = kernel.ROOT / name
        kernel.require(path.resolve() == path and path.is_file(), 'exact ordinary probe source')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        kernel.require(actual == expected, 'probe source binding changed')


def emit(stage, **fields):
    print(json.dumps(dict(schema='tracking_keeper_memory_probe_event.v1', stage=stage, **fields)), flush=True)


def run_parent(mode, command):
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    emit('LEAF_STARTED', mode=mode, parent_pid=os.getpid(), leaf_pid=process.pid)
    # Child scheduling must not let OOM kill the parent before this identity
    # event is flushed. Only this one-shot pipe grant releases allocation.
    kernel.require(process.stdin.write(GRANT) == len(GRANT), 'complete one-shot allocation grant')
    process.stdin.close()
    code = process.wait()
    emit('PARENT_RETURNED', mode=mode, returncode=code)
    kernel.require(mode == 'fit' and code == 0, 'overflow must not return through live probe parent')
    return 0


def wait_for_allocation_grant():
    kernel.require(sys.stdin.buffer.readline(64) == GRANT, 'exact parent grant before allocation')


def run(mode, role, source_sha256, kernel_sha256):
    kernel.require(mode in kernel.PROBE_UNITS and role in ('parent', 'leaf'), 'fixed tiny mode/role')
    scope = kernel.admit_scope(kernel.PROBE_UNITS[mode])
    kernel.require(Path.cwd() == kernel.ROOT and str(sys.executable) == str(PYTHON), 'exact tiny probe runtime')
    verify_sources(source_sha256, kernel_sha256)
    kernel.require(not any(name in sys.modules for name in ('torch', 'numpy', 'genesis')),
        'tiny probe must not import ML or native libraries')
    emit('SCOPE_ADMITTED', mode=mode, role=role, scope=scope, native_execution=False)
    if role == 'parent':
        command = [str(PYTHON), str(kernel.ROOT / SOURCE), '--mode', mode, '--role', 'leaf',
            '--source-sha256', source_sha256, '--kernel-sha256', kernel_sha256]
        return run_parent(mode, command)
    wait_for_allocation_grant()
    requested = 8 * 1024**2 if mode == 'fit' else 128 * 1024**2
    emit('ALLOCATION_REQUEST', mode=mode, bytes=requested, pid=os.getpid())
    payload = bytearray(requested)
    for i in range(0, len(payload), 4096):
        payload[i] = 1
    root = Path('/sys/fs/cgroup') / scope['cgroup'].lstrip('/')
    emit('ALLOCATION_RETURNED', mode=mode, bytes=len(payload), pid=os.getpid(),
        current_bytes=int((root / 'memory.current').read_text()),
        peak_bytes=int((root / 'memory.peak').read_text()))
    kernel.require(mode == 'fit', 'overflow allocation unexpectedly returned')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=kernel.PROBE_UNITS, required=True)
    parser.add_argument('--role', choices=('parent', 'leaf'), required=True)
    parser.add_argument('--source-sha256', required=True)
    parser.add_argument('--kernel-sha256', required=True)
    a = parser.parse_args()
    return run(a.mode, a.role, a.source_sha256, a.kernel_sha256)


if __name__ == '__main__':
    raise SystemExit(main())
