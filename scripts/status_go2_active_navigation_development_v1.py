"""Read-only status of two fixed development jobs; never an outcome verifier.

Only SHA-bound ordinary execution records, process metadata and bounded JSONL
tails are read. No datasets, growing gzip streams, checkpoints or sealed paths
are opened. This command neither launches nor restarts any work.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import psutil

ROOT = Path('/home/andrewknowles/Workspace/LeWMQuad-v3')
BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
REPLAY_ROOT = BASE/'go2_body_projected_tiled_late_history_v1_attempt_001'
NATIVE_ROOT = BASE/'go2_no_rgb_direct_extended_budget_maze02_pilot_v1_attempt_001'
REPLAY_RECORD = 'docs/go2_body_projected_tiled_controller_replay_execution_2026-09-11.json'
WATCH_RECORD = 'docs/go2_body_projected_tiled_completion_watch_execution_2026-09-11.json'
NATIVE_RECORD = 'docs/go2_extended_budget_native_worker_launch_observation_2026-09-11.json'
BINDINGS = {
    REPLAY_RECORD: '39755d9dbec68b6fd5525dd9389c741a22d4e7ae0e30626a1fbe7fe91092e351',
    WATCH_RECORD: '941b495b77588555b74a550b1d822ca49ed31bb94a302377d202d424782514dd',
    NATIVE_RECORD: 'ed8d0c8a6a98f8de2c9335e0847c8adbf9642adbd216d18a95b3954cd80326ed',
}


def ordinary(path):
    if any(part in ('sealed', 'sealed_test.json') or part.startswith('sealed_') for part in path.parts):
        raise ValueError('protected path is inaccessible')
    if path.resolve() != path:
        raise ValueError('status reader requires fixed nonsymlink paths')
    return path


def record(name):
    path = ordinary(ROOT/name)
    if path.stat().st_size > 2_000_000:
        raise ValueError('bounded execution record required')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != BINDINGS[name]:
        raise ValueError('execution record identity changed')
    return json.loads(data)


def owner_status(owner, boot):
    result = dict(pid=owner['pid'], recorded_creation=owner['created'])
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != boot:
        return result | dict(state='unknown_boot_changed')
    try:
        process = psutil.Process(owner['pid'])
        if process.create_time() != owner['created']:
            return result | dict(state='unknown_pid_reused')
        status = process.status()
        if status == psutil.STATUS_ZOMBIE:
            return result | dict(state='ended_zombie')
        if process.cmdline() != owner['command']:
            return result | dict(state='unknown_command_changed')
        return result | dict(state='live', process_status=status,
            resident_bytes=process.memory_info().rss)
    except psutil.NoSuchProcess:
        return result | dict(state='ended')
    except psutil.AccessDenied:
        return result | dict(state='unknown_access_denied')


def last_complete(path, key):
    path = ordinary(path)
    if not path.exists():
        return None
    with path.open('rb') as handle:
        size = handle.seek(0, 2)
        offset = max(0, size-32_768)
        handle.seek(offset)
        data = handle.read(32_768)
    lines = data.split(b'\n')[:-1]
    if offset:
        lines = lines[1:]
    if not lines:
        return None
    value = json.loads(lines[-1])[key]
    if type(value) is not int or value < 0:
        raise ValueError('nonnegative integer complete progress row required')
    return value


def file_presence(root):
    return {name: ordinary(root/name).exists() for name in ('result.json', 'failure.json')}


def snapshot():
    replay, watch, native = (record(name) for name in (REPLAY_RECORD, WATCH_RECORD, NATIVE_RECORD))
    frame = last_complete(REPLAY_ROOT/'comparison.jsonl', 'frame')
    timing = NATIVE_ROOT/'no_rgb_direct_extended_budget_anchored_maze_02'/'decision_stream_timing.jsonl'
    return dict(
        utc=datetime.now(timezone.utc).isoformat(), diagnostic_only=True,
        recorded_process_identities_authenticated=True,
        body_projection_replay=dict(owner=owner_status(replay['owner'], replay['boot_id']),
            watcher=owner_status(watch['owner'], watch['boot_id']),
            last_complete_frame=frame, processed_observations=None if frame is None else frame+1,
            planned_observations=1428, terminal_file_presence_unverified=file_presence(REPLAY_ROOT)),
        extended_budget_native=dict(
            launcher=owner_status(native['processes']['native_launcher'], native['boot_id']),
            worker=owner_status(native['processes']['native_worker'], native['boot_id']),
            last_recorded_tick=last_complete(timing, 'tick'), navigation_tick_budget=4000,
            terminal_file_presence_unverified=file_presence(NATIVE_ROOT)),
        resources=dict(memory_available_bytes=psutil.virtual_memory().available,
            artifact_free_bytes=shutil.disk_usage(BASE).free,
            workspace_free_bytes=shutil.disk_usage(ROOT).free),
        runtime_artifact_authentication_performed=False, raw_or_model_admission_performed=False,
        navigation_outcome_verified_by_this_command=False)


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2, allow_nan=False))
