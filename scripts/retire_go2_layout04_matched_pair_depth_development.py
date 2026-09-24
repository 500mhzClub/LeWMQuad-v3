"""Retire depth NPZ leaves from the completed, diagnosed, superseded layout-4 matched pair.

Policy: docs/go2_development_artifact_retention_2026-09-14.md. Retires only regular
single-link primary/auxiliary depth NPZ leaves. Every non-depth file is preserved and
hash-verified before and after. Writes a receipt and a per-root depth_retention.json.

Dry run by default; pass --apply to retire.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
import sys
import time
from pathlib import Path

BASE = Path('/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
RECEIPT = Path('/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/depth_retirement_layout04_matched_pair_2026-09-18')
ROOTS = [
    'go2_cached_fine_goal_lzma_hold_relative_recovery_pulse_round_trip_native_layout04_4800_v1_attempt_001',
    'go2_jit_floor_cached_fine_goal_lzma_pulse_reactive_round_trip_native_layout04_4800_v1_attempt_001',
    'go2_jit_floor_cached_fine_goal_lzma_hold_relative_recovery_pulse_round_trip_native_layout04_4800_v1_attempt_001',
]


# Headroom must be checked on the filesystem backing each path actually used, not on a
# guessed mount point. /home/andrewknowles/Workspace is its own XFS volume, so querying
# '/' reports an unrelated filesystem and can wrongly clear or wrongly fail the gate.
ADMISSION = {
    'workspace artifacts': str(BASE),
    'steam_drive artifacts': '/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1',
    'RecoveryStorage artifacts': ('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/'
                                  'navigation_development_artifacts_v1'),
    'tmp': '/tmp',
}


def free_bytes(path):
    status = os.statvfs(path)
    return status.f_bavail * status.f_frsize


def is_depth(name):
    return name.startswith(('primary_depth_', 'auxiliary_depth_')) and name.endswith('.npz')


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return path, digest.hexdigest()


def scan(root):
    depth = []
    keep = []
    for directory, _, files in os.walk(root):
        for name in sorted(files):
            path = os.path.join(directory, name)
            status = os.lstat(path)
            if is_depth(name) and os.path.isfile(path) and not os.path.islink(path) and status.st_nlink == 1:
                depth.append((path, status.st_blocks * 512))
            else:
                keep.append(path)
    return depth, keep


def main():
    apply = '--apply' in sys.argv
    started = time.monotonic()
    report = {'roots': {}, 'apply': apply}
    all_keep = []
    all_depth = []
    for root in ROOTS:
        depth, keep = scan(BASE / root)
        all_depth += depth
        all_keep += keep
        report['roots'][root] = {'depth_leaves': len(depth),
                                 'allocated_bytes': sum(b for _, b in depth),
                                 'preserved_files': len(keep)}
    total = sum(b for _, b in all_depth)
    print(f'depth leaves {len(all_depth)}  allocated {total}  preserved {len(all_keep)}', flush=True)

    with ProcessPoolExecutor(max_workers=12) as pool:
        before = dict(pool.map(sha, all_keep, chunksize=64))
    print(f'hashed {len(before)} preserved files in {time.monotonic() - started:.1f}s', flush=True)

    if not apply:
        print('DRY RUN - nothing removed. Re-run with --apply')
        return

    removed = freed = 0
    for path, allocated in all_depth:
        os.unlink(path)
        removed += 1
        freed += allocated
    print(f'retired {removed} leaves, {freed} allocated bytes', flush=True)

    with ProcessPoolExecutor(max_workers=12) as pool:
        after = dict(pool.map(sha, all_keep, chunksize=64))
    assert set(after) == set(before), 'preserved file set changed'
    changed = [p for p in before if before[p] != after[p]]
    assert not changed, f'{len(changed)} preserved files changed: {changed[:5]}'
    print(f'VERIFIED: all {len(after)} preserved file hashes unchanged', flush=True)

    LOST = ('Raw depth was retired, not merely unused. Preserving every failure record is '
            'NOT preserving replayability. The following can no longer be reproduced directly '
            'from these roots and would require regeneration, which is not promised to '
            'reproduce the original closed-loop trajectory: perception and tracker replays '
            'over primary/auxiliary depth; stable-reference and compiled-floor replays; '
            'raw-sensor public replay and its sample verification; any depth-dependent '
            'floor-extraction or occupancy re-derivation. All RGB, physics, commands, poses, '
            'diagnostics, comparison summaries, results and failure records are preserved '
            'and were hash-verified before and after.')

    RECEIPT.mkdir(parents=True, exist_ok=True)
    report.update(removed_leaves=removed, freed_allocated_bytes=freed,
                  preserved_files_verified=len(after), wall_s=time.monotonic() - started,
                  policy='docs/go2_development_artifact_retention_2026-09-14.md',
                  script_sha256=sha(os.path.abspath(__file__))[1],
                  basis='completed, diagnosed, superseded matched pair; no pending raw replay or training input',
                  excluded_active_reference='go2_dense_horizon_untimed_exposed_maze_full_v1_attempt_001',
                  replayability_lost=LOST,
                  free_bytes_after={label: free_bytes(path) for label, path in ADMISSION.items()})
    (RECEIPT / 'result.json').write_text(json.dumps(report, indent=1))
    (RECEIPT / 'deletion_manifest.json').write_text(json.dumps(
        {'leaves': [p for p, _ in all_depth]}, indent=1))
    (RECEIPT / 'preserved_hashes.json').write_text(json.dumps(after, indent=1))
    for root in ROOTS:
        (BASE / root / 'depth_retention.json').write_text(json.dumps(dict(
            depth_retired=True, date='2026-09-18', receipt=str(RECEIPT),
            policy='docs/go2_development_artifact_retention_2026-09-14.md',
            note=LOST), indent=1))
    print(f'receipt: {RECEIPT}')
    print('Admission check on the filesystems actually used (by path, not by mount guess):')
    for label, path in ADMISSION.items():
        free = free_bytes(path)
        gate = 'PASS' if free >= 2 * 2**30 else 'BELOW 2 GiB GATE'
        print(f'  {label:42s} free {free/2**30:7.2f} GiB  {gate}')


if __name__ == '__main__':
    main()
