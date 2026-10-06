"""Exact script versions behind each training-lineage artefact (consolidation, 6 October 2026).

For the research-artefact migration (docs/go2_research_artifact_migration_plan_2026-10-06.md): retraining every component
from scratch needs the code that actually produced each checkpoint and dataset, which is often an older version of a
script than the one at HEAD.

For each lineage artefact directory (list below), every top-level *.json file under 8 MB is scanned for strings naming a
repository Python file, together with any SHA-256 recorded next to it (a 64-hex string in the same dict, or a
`<path> <sha>` pair). Each (path, sha256) is then matched against the file's git history: the newest commit whose blob
has that SHA-256. Paths with no recorded hash are resolved at the artefact's creation time is not attempted; they are
listed as `unhashed`.

Custody: only lineage artefact directories are read, never a sealed path; git blobs are read only for the named paths.

Usage: discover_go2_lineage_scripts_2026_10_06.py --out JSON
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO = Path(__file__).resolve().parents[1]
R = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
W = Path('/mnt/workspace_drive/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
STEAM = Path('/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1')
LINEAGE = {
    'T2 native adaptation': STEAM/'go2_frozen_vjepa_native_adaptation_v1_attempt_001',
    'T3 balanced-start predictor': W/'go2_balanced_start_predictor_v1_attempt_001',
    'T4 horizon dense predictor': W/'go2_horizon_dense_predictor_v1_attempt_001',
    'T5a full-heading readout': W/'go2_full_heading_readout_v1_attempt_001',
    'T5b maze-view readout': R/'go2_maze_view_readout_v1_attempt_003',
    'data balanced-start actions': W/'go2_balanced_start_actions_v1_attempt_001',
    'data balanced-start horizon actions': W/'go2_balanced_start_horizon_actions_v1_attempt_001',
    'data moving-action switch family': R/'go2_moving_action_switch_family_v1_attempt_001',
    'data geometry-progress family': R/'go2_geometry_progress_family_v1_attempt_001',
    'data short-pulse learning': R/'go2_short_pulse_learning_v1_attempt_001',
    'T6 C1 command model': R/'go2_short_pulse_command_control_v1_attempt_001',
    'data maze-view transfer': R/'go2_maze_view_transfer_v1_attempt_001',
}
SEALED = re.compile(r'(^|/)(sealed_test\.json$|sealed/|sealed_[^/]*/)')
PY = re.compile(r'((?:scripts|lewm|lewm_genesis|lewm_worlds)/[A-Za-z0-9_./-]+\.py)')
HEX = re.compile(r'^[0-9a-f]{64}$')


def scan(value, found, context=None):
    if isinstance(value, dict):
        hexes = [v for v in value.values() if isinstance(v, str) and HEX.match(v)]
        for k, v in value.items():
            m = PY.search(k) if isinstance(k, str) else None
            if m and isinstance(v, str) and HEX.match(v):
                found[m.group(1)].add(v)
            elif m and isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, str) and HEX.match(vv):
                        found[m.group(1)].add(vv)
            scan(v, found, hexes)
    elif isinstance(value, list):
        for v in value:
            scan(v, found, context)
    elif isinstance(value, str):
        for m in PY.finditer(value):
            path = m.group(1)
            rest = value[m.end():].strip().split()
            if rest and HEX.match(rest[0]):
                found[path].add(rest[0])
            elif context and len(context) == 1:
                found[path].add(context[0])
            else:
                found[path].add(None)


def git(*args):
    return subprocess.run(['git', *args], cwd=REPO, capture_output=True, check=False).stdout


def resolve(path, sha):
    """Newest commit whose version of `path` has this SHA-256 (also tries the path's repo-relative variants)."""
    for commit in git('log', '--all', '--format=%H', '--', path).decode().split():
        blob = git('show', f'{commit}:{path}')
        if blob and hashlib.sha256(blob).hexdigest() == sha:
            return commit
    return None


def main(out):
    report, all_paths = {}, defaultdict(set)
    for name, root in LINEAGE.items():
        found = defaultdict(set)
        if not root.exists():
            report[name] = dict(root=str(root), missing=True)
            continue
        for f in sorted(root.glob('*.json')):
            if SEALED.search(str(f)) or f.stat().st_size > 8_000_000:
                continue
            try:
                scan(json.loads(f.read_text()), found)
            except (ValueError, UnicodeDecodeError):
                continue
        entries = []
        for path, shas in sorted(found.items()):
            if SEALED.search(path):
                continue
            hashed = sorted(s for s in shas if s)
            head = REPO/path
            head_sha = hashlib.sha256(head.read_bytes()).hexdigest() if head.exists() else None
            for sha in hashed:
                commit = None if sha == head_sha else resolve(path, sha)
                entries.append(dict(path=path, sha256=sha, at_head=sha == head_sha, commit=commit,
                                    resolved=sha == head_sha or commit is not None))
                all_paths[path].add(sha)
            if not hashed:
                entries.append(dict(path=path, sha256=None, at_head=None, commit=None, resolved=False, unhashed=True))
                all_paths[path].add(None)
        report[name] = dict(root=str(root), scripts=entries)
    summary = dict(paths=len(all_paths), versions=sum(len(v) for v in all_paths.values()),
                   unresolved=sorted({e['path'] for r in report.values() for e in r.get('scripts', []) if e.get('sha256') and not e['resolved']}),
                   unhashed=sorted({e['path'] for r in report.values() for e in r.get('scripts', []) if e.get('unhashed')}))
    Path(out).write_text(json.dumps(dict(summary=summary, lineage=report), indent=1)+'\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    main(p.parse_args().out)
