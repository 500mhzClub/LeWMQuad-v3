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

    RECEIPT.mkdir(parents=True, exist_ok=True)
    report.update(removed_leaves=removed, freed_allocated_bytes=freed,
                  preserved_files_verified=len(after), wall_s=time.monotonic() - started,
                  policy='docs/go2_development_artifact_retention_2026-09-14.md',
                  basis='completed, diagnosed, superseded matched pair; no pending raw replay or training input')
    (RECEIPT / 'result.json').write_text(json.dumps(report, indent=1))
    (RECEIPT / 'preserved_hashes.json').write_text(json.dumps(after, indent=1))
    for root in ROOTS:
        (BASE / root / 'depth_retention.json').write_text(json.dumps(dict(
            depth_retired=True, date='2026-09-18', receipt=str(RECEIPT),
            policy='docs/go2_development_artifact_retention_2026-09-14.md',
            note='Primary/auxiliary depth NPZ arrays intentionally retired. Exact historical '
                 'sensor replay requires regeneration and is not promised to reproduce the '
                 'closed-loop trajectory. All RGB, physics, commands, poses, diagnostics, '
                 'results and failure records are preserved and were hash-verified.'), indent=1))
    print(f'receipt: {RECEIPT}')


if __name__ == '__main__':
    main()
