"""Depth-only retirement of the closed navigation roots in the 4 October storage review, Tier 1 (all three drives).

Authority: Andrew, 4 October 2026: "delete anything youve identified that isnt used or relevent", then, after the
final review, "remove any artefacts not required" (docs/go2_storage_review_2026-10-04.md).

Roots are the review's Tier 1 candidates on RecoveryStorage, the workspace drive and /mnt/steam_drive. Protected
roots (capability root, decision-headroom lineage, current manifests' sources, the mission runtime's inputs, the
predictor's ancestor checkpoints and the policy's pinned references) are refused by name.
Only per-frame depth leaves go (depth_N, native_depth_N, auxiliary_depth_N, primary_depth_N; .npz or .npy, exact
name match, single link). Every other file is SHA-256 hashed before and after; symlinks are recorded and checked.
Hard-linked depth leaves are left in place and listed.

  --roots LIST.json --plan OUT.json   inventory only: per-root leaf counts and bytes; nothing is changed
  --execute PLAN.json                 re-inventory, assert it equals the plan, then hash, retire, verify, mark, receipt
"""
import argparse
import hashlib
import json
import os
import re
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NAV_DIRS = {
    'data': Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1'),
    'workspace_nav': Path('/mnt/workspace_drive/LeWMQuad-v3/.generated/navigation_development_artifacts_v1'),
    'workspace_top': Path('/mnt/workspace_drive/LeWMQuad-v3/.generated'),
    'steam': Path('/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1'),
}
RECEIPT = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/depth_retirement_storage_review_tier1_2026-10-04')
PROTECTED_PREFIXES = ('go2_decision_headroom', 'go2_headroom_', 'go2_navigation_capability_v1')
PROTECTED = {
    'go2_geometry_progress_family_v1_attempt_001', 'go2_maze_view_training_v1_attempt_001',
    'go2_maze_view_transfer_v1_attempt_001', 'go2_moving_action_switch_family_v1_attempt_001',
    'go2_short_pulse_learning_v1_attempt_001', 'go2_short_pulse_command_control_v1_attempt_001',
    'go2_maze_view_readout_v1_attempt_003', 'go2_horizon_dense_predictor_v1_attempt_001',
    'go2_balanced_start_horizon_actions_v1_attempt_001', 'go2_balanced_start_actions_v1_attempt_001',
    'go2_balanced_start_predictor_v1_attempt_001', 'go2_frozen_vjepa_native_adaptation_v1_attempt_001',
    'go2_full_heading_training_v1_attempt_001', 'go2_full_heading_readout_v1_attempt_001',
    'go2_all_phase_adapter_maze02_matched_native_v1_attempt_001',
    'go2_clearance_preferred_arc_recovery_learned_round_trip_native_layout03_v1_attempt_001'}
LEAF = re.compile(r'(native_|auxiliary_|primary_)?depth_\d+\.np[zy]')
NOTE = ('Depth-only retirement authorized by Andrew on 2026-10-04 after the storage review. Historical full-depth '
        'replay is unavailable; regeneration is not guaranteed to reproduce the original closed-loop recordings. '
        'Results, failures and all original non-depth files remain.')


def write(path, value):
    path.write_text(json.dumps(value, indent=1) + '\n')
    assert json.loads(path.read_text()) == value


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def free(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize


def guard(root):
    assert 'sealed' not in str(root), root
    assert not root.name.startswith(PROTECTED_PREFIXES) and root.name not in PROTECTED, root
    assert root.is_dir() and not root.is_symlink(), root


def scan_root(root):
    guard(root)
    depth, keep, links, shared = [], [], {}, []
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
            if stat.S_ISLNK(s.st_mode):
                links[str(p)] = os.readlink(p)
            elif LEAF.fullmatch(filename) and stat.S_ISREG(s.st_mode) and s.st_nlink == 1:
                depth.append(dict(path=str(p), size=s.st_size, allocated=s.st_blocks * 512,
                                  inode=s.st_ino, device=s.st_dev, mtime_ns=s.st_mtime_ns))
            else:
                assert stat.S_ISREG(s.st_mode), str(p)
                if LEAF.fullmatch(filename):
                    shared.append(str(p))
                if filename != 'depth_retention.json':
                    keep.append(str(p))
    return depth, keep, links, shared


def candidates(names):
    roots = []
    for drive, name in names:
        root = NAV_DIRS[drive] / name
        guard(root)
        roots.append((drive, root))
    return roots


def inventory(roots, workers=12):
    with ThreadPoolExecutor(workers) as pool:
        return dict(zip([str(r) for _, r in roots], pool.map(lambda dr: scan_root(dr[1]), roots)))


def summary(inv):
    return {root: dict(leaves=len(d), bytes=sum(e['size'] for e in d), allocated=sum(e['allocated'] for e in d),
                       kept_files=len(k), symlinks=len(l), shared_depth_left=len(s))
            for root, (d, k, l, s) in inv.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--roots', type=Path, help='JSON list of [drive, root name] (plan mode)')
    p.add_argument('--plan', type=Path)
    p.add_argument('--execute', type=Path)
    a = p.parse_args()
    if a.plan:
        names = json.loads(a.roots.read_text())
        inv = inventory(candidates(names))
        write(a.plan, dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'), leaf_pattern=LEAF.pattern,
                           roots=names, summary=summary(inv)))
        s = summary(inv).values()
        print(json.dumps(dict(roots=len(names), leaves=sum(r['leaves'] for r in s),
                              allocated_gib=round(sum(r['allocated'] for r in s) / 2**30, 2),
                              kept_files=sum(r['kept_files'] for r in s),
                              shared_depth_left=sum(r['shared_depth_left'] for r in s))))
        return
    plan = json.loads(a.execute.read_text())
    start = time.monotonic()
    roots = candidates(plan['roots'])
    drives = {d: NAV_DIRS[d] for d, _ in roots}
    before_free = {d: free(p) for d, p in drives.items()}
    inv = inventory(roots)
    assert summary(inv) == plan['summary'], 'inventory changed since the plan'
    RECEIPT.mkdir(exist_ok=False)
    depth = [e for d, _, _, _ in inv.values() for e in d]
    keep = [k for _, ks, _, _ in inv.values() for k in ks]
    links = {k: v for _, _, ls, _ in inv.values() for k, v in ls.items()}
    write(RECEIPT / 'deletion_manifest.json', depth)
    write(RECEIPT / 'shared_depth_left.json', {r: s for r, (_, _, _, s) in inv.items() if s})
    print(f'Inventoried {len(depth)} depth leaves in {len(roots)} roots; hashing {len(keep)} preserved files', flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        before = dict(zip(keep, pool.map(sha, keep)))
    write(RECEIPT / 'preserved_hashes.json', before)
    write(RECEIPT / 'preserved_symlinks.json', links)
    write(RECEIPT / 'authority.json', dict(
        user_text=['delete anything youve identified that isnt used or relevent', 'remove any artefacts not required'],
        approver='Andrew', date='2026-10-04', review='docs/go2_storage_review_2026-10-04.md (Tier 1)',
        script_sha256=sha(__file__), plan_sha256=sha(a.execute), leaf_pattern=LEAF.pattern, note=NOTE))
    previous = {}
    for _, root in roots:
        marker = root / 'depth_retention.json'
        previous[str(root)] = json.loads(marker.read_text()) if marker.exists() else None
        write(marker, dict(status='retirement_in_progress', depth_retired=False, receipt=str(RECEIPT), note=NOTE,
                           previous_marker=previous[str(root)]))
    write(RECEIPT / 'previous_markers.json', previous)
    print('Preserved hashes and exact deletion inventory saved; retiring depth only', flush=True)
    removed = 0
    try:
        for entry in depth:
            path = Path(entry['path'])
            s = path.lstat()
            assert stat.S_ISREG(s.st_mode) and s.st_nlink == 1
            assert (s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns) == (
                entry['inode'], entry['device'], entry['size'], entry['mtime_ns'])
            path.unlink()
            removed += 1
        with ThreadPoolExecutor(max_workers=12) as pool:
            after = dict(zip(keep, pool.map(sha, keep)))
        assert before == after, 'Preserved file mismatch'
        assert all(os.readlink(p) == target for p, target in links.items())
        assert all(not os.path.lexists(e['path']) for e in depth)
        for _, root in roots:
            write(root / 'depth_retention.json', dict(
                status='completed', depth_retired=True, date='2026-10-04',
                retired_files=plan['summary'][str(root)]['leaves'], receipt=str(RECEIPT), note=NOTE,
                preserved_non_depth_hashes_verified=True, previous_marker=previous[str(root)]))
        result = dict(status='completed', roots=len(roots), removed_files=removed,
                      retired_bytes=sum(e['size'] for e in depth),
                      retired_allocated_bytes=sum(e['allocated'] for e in depth),
                      preserved_files_verified=len(keep), preserved_symlinks=len(links),
                      free_bytes_before=before_free, free_bytes_after={d: free(p) for d, p in drives.items()},
                      wall_seconds=time.monotonic() - start, note=NOTE)
        write(RECEIPT / 'result.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write(RECEIPT / 'failure.json', dict(removed_files=removed, error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
