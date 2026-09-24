"""One-shot, assignment-boundary handover; never interrupts an active trial.

Watch only root closeout events. SIGSTOP is reversible: if the next source has
already been admitted, immediately continue the original owner and wait for its
next closeout. No debugger injection or changes to the running Python process.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import struct
import time

import psutil


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reconstruct(root, order):
    """Read reservations, not model outputs; reject anything short of a boundary."""
    root = Path(root)
    started, finished, branches, source_ns, measurements = [], [], set(), {}, []
    last_elapsed = 0.
    with (root / 'budget_events.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            last_elapsed = row['elapsed_s']
            kind = row['kind']
            if kind == 'source_started':
                case = row['case']
                if case in started:
                    raise ValueError('duplicate source admission')
                started.append(case)
                source_ns[case] = 0
            elif kind == 'source_finished':
                finished.append(row['case'])
            elif kind == 'source_physics_reserved':
                source_ns[row['case']] += row['ns']
            elif kind == 'branch_reserved':
                if row['identity'] in branches:
                    raise ValueError('duplicate branch reservation')
                branches.add(row['identity'])
            elif kind == 'resource_check':
                if row['violated']:
                    raise ValueError('prior resource stop')
                measurements.append(row)
    closed = {int(p.stem.split('_')[1]): json.loads(p.read_text())
              for p in root.glob('cell_*_closeout.json')}
    if started != order[:len(started)] or finished != started or set(closed) != set(started):
        return None
    if any(closed[case]['case'] != case for case in started):
        raise ValueError('closeout identity mismatch')
    if {int(p.name.split('_')[1]) for p in root.glob('source_[0-9][0-9]')} != set(started):
        raise ValueError('unexpected source directory')
    if not measurements or (root / 'failure.json').exists() or (root / 'collection_result.json').exists():
        raise ValueError('not an unfinished, running audit')
    snapshots = []
    for case in started:
        path = root / f'source_{case:02d}' / 'snapshots.json'
        if path.exists():
            snapshots.extend([case, s['frame']] for s in json.loads(path.read_text()))
    return dict(closed_cases=started, outcomes=[closed[c] for c in started],
                remaining_cases=order[len(started):], source_ns=source_ns,
                branches=sorted(branches), snapshots=snapshots,
                last_elapsed_s=last_elapsed, measurement_count=len(measurements),
                prior_cpu_s=max(r['cpu_s'] for r in measurements),
                peak_rss=max(r['aggregate_rss_bytes'] for r in measurements),
                peak_vram=max(r['gpu_used_bytes'] for r in measurements),
                peak_retained=max(r['retained_bytes'] for r in measurements),
                cache_growth_carry=max(r['external_cache_growth_bytes'] for r in measurements),
                closeout_sha256={f'cell_{c:02d}_closeout.json': digest(root / f'cell_{c:02d}_closeout.json') for c in started},
                journal_sha256=digest(root / 'budget_events.jsonl'),
                journal_bytes=(root / 'budget_events.jsonl').stat().st_size)


def wait_boundary(root, pid, created, config):
    root = Path(root)
    owner = psutil.Process(pid)
    if abs(owner.create_time() - created) > .01:
        raise ValueError('owner PID reused')
    order = [case for layout in config['execution_order'] for case in range(layout * 3, layout * 3 + 3)]
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
    if fd < 0 or libc.inotify_add_watch(fd, os.fsencode(root), 0x8) < 0:
        raise OSError(ctypes.get_errno(), 'cannot watch assignment closeouts')
    print('WAITING_FOR_COMPLETE_ASSIGNMENT', pid, flush=True)
    try:
        while owner.is_running() and owner.status() != psutil.STATUS_ZOMBIE:
            if not select.select([fd], [], [], 30)[0]:
                continue
            data = os.read(fd, 65536)
            offset = 0
            names = []
            while offset < len(data):
                _, mask, _, size = struct.unpack_from('iIII', data, offset)
                name = data[offset + 16:offset + 16 + size].split(b'\0')[0].decode()
                offset += 16 + size
                if mask & 0x8 and name.startswith('cell_') and name.endswith('_closeout.json'):
                    names.append(name)
            if not names:
                continue
            owner.suspend()
            keep_stopped = False
            try:
                deadline = time.monotonic() + 5
                while owner.status() != psutil.STATUS_STOPPED:
                    if time.monotonic() > deadline:
                        raise RuntimeError('owner did not stop')
                    time.sleep(.01)
                state = reconstruct(root, order)
                if state is None:
                    print('MISSED_BOUNDARY_CONTINUING_UNCHANGED', names, flush=True)
                    continue
                children = owner.children(recursive=True)
                if any('multiprocessing.resource_tracker' not in ' '.join(p.cmdline()) for p in children):
                    raise RuntimeError('non-tracker child remains at boundary')
                cpu = owner.cpu_times()
                # Conservatively include process startup CPU and elapsed wall time.
                total_cpu = cpu.user + cpu.system + cpu.children_user + cpu.children_system
                total_cpu += sum(p.cpu_times().user + p.cpu_times().system for p in children)
                state.update(schema='headroom_v42_boundary.v1', old_pid=pid, old_created=created,
                             paused_at_epoch=time.time(), original_wall_origin_epoch=created,
                             prior_cpu_s=max(state['prior_cpu_s'], total_cpu),
                             tracker_children=[dict(pid=p.pid, created=p.create_time()) for p in children],
                             original_admission=json.loads((root / 'pilot_execution_admission.json').read_text()),
                             original_admission_sha256=digest(root / 'pilot_execution_admission.json'),
                             protocol_sha256=digest('docs/go2_decision_headroom_protocol_v42_2026-09-23.json'),
                             no_active_source=True, no_trials_repeated=True)
                from lewm import decision_headroom_json_v42_development as output_json
                output_json.install(root)
                with (root / 'handover_boundary.json').open('x') as stream:
                    json.dump(state, stream, indent=2)
                keep_stopped = True
                print('SAFE_BOUNDARY_PAUSED', state['closed_cases'], 'NEXT', state['remaining_cases'], flush=True)
                return
            finally:
                if not keep_stopped:
                    owner.resume()
        raise RuntimeError('owner exited before a handover boundary')
    finally:
        os.close(fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--created', type=float, required=True)
    args = parser.parse_args()
    config = json.loads(Path('docs/go2_decision_headroom_protocol_v42_2026-09-23.json').read_text())
    wait_boundary(config['execution_caps']['output_root'], args.pid, args.created, config)
