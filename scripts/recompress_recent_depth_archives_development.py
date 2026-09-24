"""Apply the exact September-14 proposal after explicit user authorization.

Only compression changes. Each replacement is verified before an atomic rename;
the journal records both container hashes before replacing the original. A
partial run remains readable by PublicReplay and is never silently restarted.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time
import zipfile

from lewm.causal_rgb_dataset_development import _protected
from scripts.in_memory_public_replay_development import PublicReplay

PROPOSAL = Path('docs/go2_recent_depth_archive_recompression_proposal_2026-09-14.json')
PROPOSAL_SHA256 = '839d553d3f69a3bba33b1ca876a451dfa087c95ac40b93992c2db4787530b6cc'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(path):
    path = Path(path)
    if _protected(path.absolute()) or _protected(path.resolve()) or path.is_symlink():
        raise ValueError('ordinary non-symlink archive required')
    original = path.read_bytes()
    output = BytesIO()
    with zipfile.ZipFile(BytesIO(original)) as old:
        infos = old.infolist()
        if (len(infos) != 1 or infos[0].filename != 'native_optical_depth_m.npy'
                or infos[0].compress_type != zipfile.ZIP_DEFLATED):
            raise ValueError('expected original raw-depth DEFLATE archive')
        member = old.read(infos[0])
        info = copy(infos[0]); info.compress_type = zipfile.ZIP_LZMA
        with zipfile.ZipFile(output, 'w') as new:
            new.comment = old.comment
            new.writestr(info, member)
    compressed = output.getvalue()
    with zipfile.ZipFile(BytesIO(compressed)) as check:
        if check.namelist() != [info.filename] or check.read(info.filename) != member:
            raise ValueError('recompression changed NPY member bytes')
    return compressed, dict(path=str(path), old_sha256=sha(original),
        new_sha256=sha(compressed), old_bytes=len(original), new_bytes=len(compressed),
        npy_member_sha256=sha(member), npy_member_bytes_equal=True)


def append(journal, value):
    journal.write(json.dumps(value) + '\n')
    journal.flush(); os.fsync(journal.fileno())


def replace(path, compressed, receipt, journal):
    path = Path(path)
    if sha(compressed) != receipt['new_sha256']:
        raise ValueError('prepared replacement changed')
    temporary = path.with_name(path.name + '.lzma-pending')
    # Exclusive creation preserves evidence of any interrupted previous attempt.
    with temporary.open('xb') as stream:
        stream.write(compressed); stream.flush(); os.fsync(stream.fileno())
    try:
        if path.is_symlink() or sha(path.read_bytes()) != receipt['old_sha256']:
            raise ValueError('original changed after preparation')
        if sha(temporary.read_bytes()) != receipt['new_sha256']:
            raise ValueError('temporary archive changed during writing')
        append(journal, receipt | dict(state='prepared'))
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        append(journal, dict(state='replaced', path=str(path), sha256=receipt['new_sha256']))
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-approved-proposal', action='store_true')
    args = parser.parse_args()
    if not args.apply_approved_proposal:
        parser.error('replacement requires prior user authorization and --apply-approved-proposal')
    raw = PROPOSAL.read_bytes()
    if sha(raw) != PROPOSAL_SHA256:
        raise ValueError('proposal differs from the exact reviewed target list')
    proposal = json.loads(raw)
    # Workspace targets first: enough room for the next experiment sooner.
    for target in sorted(proposal['targets'], key=lambda t: t['volume'] != 'workspace'):
        root = Path(target['root'])
        reader = PublicReplay(root / 'native')
        if reader.depth_witnesses is None or len(reader.depth_witnesses) != target['frames']:
            raise ValueError('complete capture witnesses required')
        report = root / 'lossless_depth_recompression_v1'
        report.mkdir()  # Never overwrite or silently resume an earlier attempt.
        started = time.monotonic(); old_bytes = 0; new_bytes = 0
        with (report / 'journal.jsonl').open('x') as journal, ThreadPoolExecutor(max_workers=4) as pool:
            append(journal, dict(proposal_sha256=PROPOSAL_SHA256, target=target,
                source_sha256=sha(Path(__file__).read_bytes())))
            for start in range(0, target['frames'], 4):
                # Bounded batches keep only eight compressed archives in memory.
                jobs = [(frame, [root / pattern.format(frame=frame)
                    for pattern in target['exact_archive_name_patterns']])
                    for frame in range(start, min(start + 4, target['frames']))]
                futures = {(frame, path): pool.submit(prepare, path)
                    for frame, paths in jobs for path in paths}
                for frame, paths in jobs:
                    for path in paths:
                        compressed, receipt = futures[frame, path].result()
                        replace(path, compressed, receipt, journal)
                        old_bytes += receipt['old_bytes']; new_bytes += receipt['new_bytes']
                    reader.packet(frame)  # Original capture pixels and complete packet digests.
                    append(journal, dict(state='capture_packet_verified', frame=frame))
                if start % 100 == 0:
                    print(root.name, 'frames', jobs[-1][0] + 1, 'saved_GiB',
                        round((old_bytes-new_bytes)/2**30, 3), flush=True)
        result = dict(frames=target['frames'], archive_count=target['archive_count'],
            original_depth_bytes=old_bytes, recompressed_depth_bytes=new_bytes,
            reclaimed_bytes=old_bytes-new_bytes, all_npy_member_bytes_equal=True,
            all_captured_public_packets_verified=True, elapsed_s=time.monotonic()-started,
            other_existing_files_changed=False, proposal_sha256=PROPOSAL_SHA256)
        with (report / 'result.json').open('x') as stream:
            json.dump(result, stream, indent=2)
        print('RECOMPRESSION_RESULT', root.name, json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
