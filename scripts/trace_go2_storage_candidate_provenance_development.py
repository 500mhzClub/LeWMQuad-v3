"""Dataset provenance of a storage candidate, and its manifest (Andrew, 30 Sep 2026). Read-only.

Confirms, by tracing dataset provenance rather than code references, whether any current
training set or the transfer set was derived from the candidate directory. The current sets
are C3/C4 v1, v2, the C3-v3 round, and the frozen C3 predictor and readout lineage.

The trace:
1. Roots: the records of every current dataset and fit: plans, results, sample and frame
   lists, collection records, and the pre-registered model bindings.
2. From each record, collect every path into an artifact directory and every sha256 string.
3. Resolve sha256 strings that name checkpoints (every `.pt` in both artifact roots is
   hashed) to their directories.
4. Follow every referenced directory's own records recursively. For directories that supply
   frames, first-level subdirectory records (specifications, launches, tapes) are read too.
The candidate is "derived from" if it enters this closure, or if any file in it has a sha256
that appears in the closure's records. Every candidate file is hashed for the manifest, which
is written into the repository with a summary.
"""
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

REPO = owner.REPO
BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
ROOTS = [BASE.parent, REPO/'.generated/navigation_development_artifacts_v1']
SHA = re.compile(r'^[0-9a-f]{64}$')
MARK = '/navigation_development_artifacts_v1/'
PREREG = REPO/'docs/go2_navigation_capability_preregistration_v1_2026-09-25.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def walk(value, paths, hashes):
    if isinstance(value, dict):
        for k, v in value.items():
            walk(k, paths, hashes)
            walk(v, paths, hashes)
    elif isinstance(value, list):
        for v in value:
            walk(v, paths, hashes)
    elif isinstance(value, str):
        if SHA.match(value):
            hashes.add(value)
        elif MARK in value:
            paths.add(value.split(MARK, 1)[1].split('/', 1)[0])


def read_records(files):
    paths, hashes, read = set(), set(), []
    for f in files:
        try:
            walk(json.loads(Path(f).read_text()), paths, hashes)
            read.append(str(f))
        except (ValueError, UnicodeDecodeError, OSError):
            continue
    return paths, hashes, read


def locate(name):
    return next((r/name for r in ROOTS if (r/name).is_dir()), None)


def records_of(directory, frame_source):
    files = sorted(p for p in directory.glob('*.json'))
    if frame_source:
        for sub in sorted(p for p in directory.iterdir() if p.is_dir()):
            files += [sub/n for n in ('specification.json', 'launch.json', 'command_tape.json', 'branch_specification.json') if (sub/n).exists()]
    return files


def main(candidate_name):
    output.install(BASE)
    candidate = locate(candidate_name)
    # Manifest of the candidate (every file hashed).
    files = sorted(p for p in candidate.rglob('*') if p.is_file())
    with ProcessPoolExecutor(12) as pool:
        digests = list(pool.map(sha, files, chunksize=64))
    manifest = [(str(p.relative_to(candidate)), p.stat().st_size, d) for p, d in zip(files, digests)]
    candidate_hashes = {d for _, _, d in manifest}
    # Checkpoint index for sha256 resolution.
    checkpoints = {}
    for root in ROOTS:
        for p in root.glob('*/*.pt'):
            checkpoints.setdefault(sha(p), p.parent.name)
    # Roots of the trace: current datasets and fits (C3/C4 v1, v2, the round) and the transfer set.
    root_files = [PREREG] + [BASE/n for n in (
        'c3v2_data_v1/train_samples.json', 'c3v2_data_v1/heldout_samples.json', 'c3v2_data_v1/frame_paths.json', 'c3v2_data_v1/result.json',
        'c4_preparation/samples.json', 'c4_preparation/frame_paths.json', 'c4_preparation/result.json',
        'c4_fit_attempt002/plan.json', 'c4_fit_attempt002/result.json', 'c4v2_fit_v1/plan.json', 'c4v2_fit_v1/result.json',
        'c3v2_readout_fit_v1/plan.json', 'c3v2_readout_fit_v1/result.json', 'c3v2_rest_turn_recordings_v1/plan.json',
        'c3v2_rest_turn_recordings_v1/samples.json', 'c3v2_sets_v1/registry.json', 'c3v3_sets_v1/registry.json',
        'cohorts/c3v3_onpolicy_c1/config.json')] + list((BASE/'c3v3_data_v1').glob('*.json'))
    root_files = [f for f in root_files if f.exists()]
    for name in ('go2_maze_view_readout_v1_attempt_003', 'go2_full_heading_readout_v1_attempt_001', 'go2_horizon_dense_predictor_v1_attempt_001',
                 'go2_maze_view_transfer_v1_attempt_001'):
        root_files += records_of(locate(name), frame_source=False)
    paths, hashes, read = read_records(root_files)
    frame_sources = set(paths)
    queue, closure = sorted(paths | {checkpoints[h] for h in hashes if h in checkpoints}), {}
    while queue:
        name = queue.pop()
        if name in closure:
            continue
        directory = locate(name)
        closure[name] = str(directory) if directory else None
        if directory is None or name == candidate_name:
            continue
        p, h, r = read_records(records_of(directory, frame_source=name in frame_sources))
        read += r
        hashes |= h
        for new in (p | {checkpoints[x] for x in h if x in checkpoints}) - set(closure):
            queue.append(new)
    shared = sorted(candidate_hashes & hashes)
    result = dict(schema='storage_candidate_provenance.v1', candidate=candidate_name, candidate_path=str(candidate),
                  files=len(manifest), bytes=sum(s for _, s, _ in manifest),
                  by_extension={e: dict(files=sum(1 for n, _, _ in manifest if n.endswith(e)),
                                        bytes=sum(s for n, s, _ in manifest if n.endswith(e)))
                                for e in sorted({os.path.splitext(n)[1] for n, _, _ in manifest})},
                  traced_record_files=len(read), closure_directories=sorted(closure),
                  candidate_in_closure=candidate_name in closure, candidate_hashes_in_closure_records=len(shared),
                  shared_hash_examples=shared[:10], checkpoints_indexed=len(checkpoints),
                  derived_from_candidate=candidate_name in closure or bool(shared), tracer_sha256=sha(__file__))
    out = REPO/'docs/storage_manifests'
    out.mkdir(exist_ok=True)
    manifest_path = out/f'{candidate_name}.manifest.tsv.gz'
    with gzip.open(manifest_path, 'xt') as stream:
        stream.write('path\tbytes\tsha256\n')
        for n, s, d in manifest:
            stream.write(f'{n}\t{s}\t{d}\n')
    result['manifest'] = dict(path=str(manifest_path.relative_to(REPO)), sha256=sha(manifest_path))
    (BASE/'e1_storage').mkdir(exist_ok=True)
    owner.save(BASE/'e1_storage'/f'provenance_{candidate_name}.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'closure_directories'} | dict(closure_size=len(closure)), indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'go2_supervised_rollout_mazes_v1_attempt_001')
