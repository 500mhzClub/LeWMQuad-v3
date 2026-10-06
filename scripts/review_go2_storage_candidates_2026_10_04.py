"""Storage review, 4 October 2026 (Andrew: "do a full review on what stale artefacts can be removed"). Read-only.

For every navigation development root on RecoveryStorage, plus the other large project directories on the data drive:
- size and last modification, from a filename-only walk that never enters `sealed*` paths;
- remaining bulk by file class: per-frame depth NPZs, RGB frames (rgb_N.png, auxiliary_rgb_N.png), raster frame JSONs,
  and everything else (results, failure records, traces, configurations);
- whether it carries a depth_retention.json marker (depth already retired);
- lineage protection: named by a current-programme data manifest (the decoder-fit feature-cache sources, the C3-v2 /
  C3-v3 / C4 data manifests, and the E1 survey's lineage files), so the stage-2 refit would need it;
- code and document references: the repository's own text files (listed by `rg --files`, which honours `.ignore`;
  sealed paths skipped) that name it, split into code (lewm, scripts, lewm_genesis, lewm_worlds, config) and documents,
  with the newest referencing file's modification date;
- protected by rule: the capability programme root, and the decision-headroom audit lineage (Andrew: do not touch).
Nothing is changed. Output: JSON and a markdown summary.

Usage: rg --files > LIST; review_go2_storage_candidates_2026_10_04.py --out DIR --files-from LIST [--workers N] [--nav-dir DIR]
The navigation folders reviewed were RecoveryStorage's (the default), the workspace drive's and /mnt/steam_drive's.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import time

REPO = Path(__file__).resolve().parents[1]
NAV = Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
CAPABILITY = NAV/'go2_navigation_capability_v1_attempt_001'
WORKSPACE_NAV = Path('/mnt/workspace_drive/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
LINEAGE = [CAPABILITY/'dev_c3_cache_v1/items.json', CAPABILITY/'dev_c3_cache_v1/frame_rows.json',
           CAPABILITY/'c3v2_data_v1/frame_paths.json', CAPABILITY/'c3v2_data_v1/train_samples.json',
           CAPABILITY/'c3v2_data_v1/heldout_samples.json', CAPABILITY/'c3v3_data_v1/frame_paths.json',
           CAPABILITY/'c3v3_data_v1/train_samples.json', CAPABILITY/'c3v3_data_v1/heldout_onpolicy_decisions.json',
           CAPABILITY/'c4_preparation/frame_paths.json', CAPABILITY/'c4_preparation/samples.json',
           NAV/'go2_maze_view_readout_v1_attempt_003/frame_paths.json', NAV/'go2_maze_view_readout_v1_attempt_003/plan.json',
           WORKSPACE_NAV/'go2_horizon_dense_predictor_v1_attempt_001/frame_paths.json',
           WORKSPACE_NAV/'go2_horizon_dense_predictor_v1_attempt_001/plan.json',
           WORKSPACE_NAV/'go2_horizon_dense_predictor_v1_attempt_001/samples.json',
           NAV/'go2_maze_view_transfer_v1_attempt_001/transfer_targets.json']
PROTECTED_PREFIXES = ('go2_decision_headroom', 'go2_headroom_', 'go2_navigation_capability_v1')
DEPTH = re.compile(r'(native_|auxiliary_|primary_)?depth_\d+\.np[zy]$')
RGB = re.compile(r'(auxiliary_)?rgb_\d+\.png$|^\d+\.png$')
RASTER = re.compile(r'raster_\d+\.json$')
CODE_TOP = ('lewm', 'scripts', 'lewm_genesis', 'lewm_worlds', 'config')
TOKEN = re.compile(r'[A-Za-z0-9_.\-]{6,}')


def walk(root):
    out = dict(depth=[0, 0], rgb=[0, 0], raster=[0, 0], other=[0, 0], marker=False, newest=0.)
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if 'sealed' not in d]
        for f in files:
            if 'sealed' in f:
                continue
            p = os.path.join(directory, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            size = st.st_blocks*512
            out['newest'] = max(out['newest'], st.st_mtime)
            if f == 'depth_retention.json':
                out['marker'] = True
            key = 'depth' if DEPTH.search(f) else 'rgb' if RGB.search(f) else 'raster' if RASTER.search(f) else 'other'
            out[key][0] += 1
            out[key][1] += size
    return out


def lineage_names(names):
    found = {}
    for path in LINEAGE:
        if not path.exists():
            continue
        text = path.read_text(errors='ignore')
        for n in names:
            if n in text:  # loose substring match: protects more than an exact path match
                found.setdefault(n, []).append(str(path.relative_to(path.parents[2])))
    return found


def repo_references(names, files_from):
    files = Path(files_from).read_text().split()  # `rg --files` from the repository root (honours .ignore)
    files = [f for f in files if 'sealed' not in f and not f.endswith(('.png', '.npz', '.npy', '.pt', '.ply', '.pdf'))]
    nameset = set(names)
    refs = {}
    for f in files:
        p = REPO/f
        try:
            if p.stat().st_size > 8_000_000:
                continue
            text = p.read_text(errors='ignore')
        except OSError:
            continue
        hits = nameset.intersection(TOKEN.findall(text))
        hits |= {n for n in names if len(n) < 6 and n in text}
        if not hits:
            continue
        kind = 'code' if f.split('/')[0] in CODE_TOP else 'doc'
        mtime = p.stat().st_mtime
        for n in hits:
            r = refs.setdefault(n, dict(code=[], doc=[], newest=0.))
            r[kind].append(f)
            r['newest'] = max(r['newest'], mtime)
    return refs


def main(out, workers, files_from, nav=NAV):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roots = sorted(p for p in Path(nav).iterdir() if p.is_dir() and 'sealed' not in p.name)
    names = [p.name for p in roots]
    started = time.monotonic()
    cache = out/'nav_walk.json'
    if cache.exists():
        walked = json.loads(cache.read_text())
    else:
        with ThreadPoolExecutor(workers) as pool:
            walked = dict(zip(names, pool.map(walk, roots)))
        cache.write_text(json.dumps(walked)+'\n')
    lineage = lineage_names(names)
    refs = repo_references(names, files_from)
    rows = []
    for n in names:
        w, r = walked[n], refs.get(n, dict(code=[], doc=[], newest=0.))
        total = sum(w[k][1] for k in ('depth', 'rgb', 'raster', 'other'))
        protected = n.startswith(PROTECTED_PREFIXES) or n in lineage
        rows.append(dict(root=n, total_bytes=total, depth=w['depth'], rgb=w['rgb'], raster=w['raster'], other=w['other'],
                         depth_retired_marker=w['marker'], newest_file=time.strftime('%Y-%m-%d', time.localtime(w['newest'])) if w['newest'] else None,
                         lineage=lineage.get(n, []), code_refs=sorted(r['code'])[:20], doc_refs=sorted(r['doc'])[:20],
                         n_code_refs=len(r['code']), n_doc_refs=len(r['doc']),
                         newest_ref=time.strftime('%Y-%m-%d', time.localtime(r['newest'])) if r['newest'] else None,
                         protected=protected,
                         protection=('rule' if n.startswith(PROTECTED_PREFIXES) else 'lineage' if n in lineage else None)))
    (out/'nav_roots.json').write_text(json.dumps(dict(created=time.strftime('%Y-%m-%dT%H:%M:%S'), wall_s=time.monotonic()-started,
                                                      lineage_files=[str(p) for p in LINEAGE if p.exists()], rows=rows), indent=1)+'\n')
    gib = lambda b: b/2**30
    cand = [r for r in rows if not r['protected']]
    lines = ['# Storage review: navigation development roots (read-only, 4 October 2026)', '',
             f"Roots: {len(rows)}; protected {len(rows)-len(cand)} "
             f"({sum(r['protection'] == 'rule' for r in rows)} by rule, {sum(r['protection'] == 'lineage' for r in rows)} by current lineage).", '',
             '| Class | Protected roots (GiB) | Unprotected roots (GiB) |', '|---|---:|---:|']
    for k in ('depth', 'rgb', 'raster', 'other'):
        lines.append(f"| {k} | {gib(sum(r[k][1] for r in rows if r['protected'])):.1f} | {gib(sum(r[k][1] for r in cand)):.1f} |")
    lines.append(f"| total | {gib(sum(r['total_bytes'] for r in rows if r['protected'])):.1f} | {gib(sum(r['total_bytes'] for r in cand)):.1f} |")
    (out/'nav_roots.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--files-from', required=True, help='output of `rg --files` run at the repository root')
    p.add_argument('--nav-dir', default=str(NAV), help='navigation artifact folder to review (one per drive)')
    a = p.parse_args()
    main(a.out, a.workers, a.files_from, a.nav_dir)
