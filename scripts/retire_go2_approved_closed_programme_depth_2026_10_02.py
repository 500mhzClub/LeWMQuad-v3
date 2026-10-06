"""One-shot depth retirement authorized by Andrew on 2 October 2026 ("approved, reture unndeeded depth data").

Exactly the four closed-programme roots proposed that evening (11-13 September recordings, listed as
not referenced by the current programme in the 29 September C3-v2 storage survey). Only the per-frame
depth leaves (depth_N.npz, native_depth_N.npz, auxiliary_depth_N.npz) go; every other file, including
depth_camera_audit.json and depth_observations.json, is hash-verified before and after.
"""
import hashlib
import json
import os
import re
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated')
DATA = BASE / 'navigation_development_artifacts_v1'
RECEIPT = BASE / 'depth_retirement_closed_programme_roots_2026-10-02'
ROOTS = {
    'go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_direct_v1_attempt_001': 24042,
    'go2_stop_conditioned_independent_00_frozen_reference_seed_2026091001_full_supervised_rollout_v1_attempt_001': 24042,
    'go2_stop_conditioned_settling_maze02_v1_attempt_001': 14250,
    'go2_extended_return_budget_maze02_v1_attempt_001': 14247,
}
LEAF = re.compile(r'(native_|auxiliary_)?depth_\d+\.npz')
NOTE = ('Depth-only retirement explicitly approved by Andrew on 2026-10-02. Historical full-depth '
        'replay is unavailable; regeneration is not guaranteed to reproduce the original '
        'closed-loop recordings. Results, failures and all original non-depth files remain.')


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')
    assert json.loads(path.read_text()) == value


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def free():
    s = os.statvfs(BASE)
    return s.f_bavail * s.f_frsize


def scan():
    depth, keep, links = [], [], {}
    for name, expected in ROOTS.items():
        root = DATA / name
        assert root.is_dir() and not root.is_symlink()
        assert not (root / 'depth_retention.json').exists()
        count = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != 'sealed' and not d.startswith('sealed_'))
            for d in list(dirs):
                p = Path(directory) / d
                if p.is_symlink():
                    links[str(p)] = os.readlink(p)
                    dirs.remove(d)
            for filename in sorted(files):
                if filename == 'sealed_test.json':
                    continue
                p = Path(directory) / filename
                s = p.lstat()
                if LEAF.fullmatch(filename):
                    assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1
                    depth.append(dict(path=str(p), size=s.st_size, allocated=s.st_blocks * 512,
                                      inode=s.st_ino, device=s.st_dev, mtime_ns=s.st_mtime_ns))
                    count += 1
                elif p.is_symlink():
                    links[str(p)] = os.readlink(p)
                else:
                    assert stat.S_ISREG(s.st_mode), str(p)
                    keep.append(str(p))
        assert count == expected, (name, count, expected)
    return depth, keep, links


def main():
    start = time.monotonic()
    before_free = free()
    depth, keep, links = scan()
    RECEIPT.mkdir(exist_ok=False)
    write(RECEIPT / 'deletion_manifest.json', depth)
    print(f'Inventoried {len(depth)} depth leaves; hashing {len(keep)} preserved files', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        before = dict(zip(keep, pool.map(sha, keep)))
    write(RECEIPT / 'preserved_hashes.json', before)
    write(RECEIPT / 'preserved_symlinks.json', links)
    write(RECEIPT / 'authority.json', dict(
        user_text='approved, reture unndeeded depth data', approver='Andrew', date='2026-10-02',
        proposal='four-root depth-only retirement proposed 2 October (C3-v2 storage survey candidates)',
        script_sha256=sha(__file__), roots=ROOTS, leaf_pattern=LEAF.pattern, note=NOTE))
    for name in ROOTS:
        write(DATA / name / 'depth_retention.json', dict(
            status='retirement_in_progress', depth_retired=False, receipt=str(RECEIPT), note=NOTE))
    print('Preserved hashes and exact deletion inventory saved; retiring depth only', flush=True)
    removed = 0
    try:
        for entry in depth:
            p = Path(entry['path'])
            s = p.lstat()
            assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1
            assert (s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns) == (
                entry['inode'], entry['device'], entry['size'], entry['mtime_ns'])
            p.unlink()
            removed += 1
        with ThreadPoolExecutor(max_workers=8) as pool:
            after = dict(zip(keep, pool.map(sha, keep)))
        assert before == after, 'Preserved file mismatch'
        assert all(os.readlink(p) == target for p, target in links.items())
        assert all(not os.path.lexists(e['path']) for e in depth)
        for name in ROOTS:
            write(DATA / name / 'depth_retention.json', dict(
                status='completed', depth_retired=True, date='2026-10-02',
                retired_files=ROOTS[name], receipt=str(RECEIPT), note=NOTE,
                preserved_non_depth_hashes_verified=True))
        result = dict(status='completed', removed_files=removed,
                      retired_bytes=sum(e['size'] for e in depth),
                      retired_allocated_bytes=sum(e['allocated'] for e in depth),
                      preserved_files_verified=len(keep), preserved_symlinks=len(links),
                      free_bytes_before=before_free, free_bytes_after=free(),
                      wall_seconds=time.monotonic()-start, roots=ROOTS, note=NOTE)
        write(RECEIPT / 'result.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write(RECEIPT / 'failure.json', dict(removed_files=removed, error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
