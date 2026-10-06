"""Removal of the May training corpus outside the current implementation's training lineage (option a).

Authority: Andrew, 4 October 2026. Asked "do we need the may training corpus", he chose option (a): keep the lineage
only. See docs/go2_storage_review_2026-10-04.md.

Lineage: the frozen predictor descends (by checkpoint hash) from the August temporal model, which was trained on cached
features of 18,690 rendered frames in 80 corpus scenes (temporal_rows.jsonl, two_step_rows.jsonl, proprio_rows.jsonl).
Kept:
- those 18,690 frames;
- every file of those 80 scenes;
- every small per-scene file (summaries, metadata, replay plans, render logs);
- driver logs;
- the scene corpus, which lives outside this folder.
Removed, for the other 1,370 scenes: rendered RGB frames, `.mcap` recordings, `messages.jsonl`, `frames.jsonl` and
`labels.jsonl`. The rendered RGB frames of the 80 lineage scenes that are not lineage frames are removed too.

  --plan DIR      inventory only; nothing is changed
  --execute DIR   re-inventory, assert equal to the plan, record hashes, remove, verify lineage frames, receipt, marker
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HOME = Path('/home/andrewknowles')
WS = Path('/mnt/workspace_drive')
CORPUS = WS / 'LeWMQuad-v3/.generated/datagen_full'
RENDER = CORPUS / 'render_textured_v03'
ROLLOUT = CORPUS / 'rollout'
CACHE = HOME / '.cache/lewm_go2_temporal_v03'
RECEIPT = HOME / 'RecoveryStorage/LeWMQuad-v3/.generated/storage_review_may_corpus_removal_2026-10-04'
BULK = {'messages.jsonl', 'frames.jsonl', 'labels.jsonl'}
SCENES = 1450


def write(path, value):
    path.write_text(json.dumps(value, indent=0) + '\n')
    assert json.loads(path.read_text()) == value


def sha(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sealed(path):
    return any(p == 'sealed' or p.startswith('sealed_') or p == 'sealed_test.json' for p in Path(path).parts)


def norm(p):
    return str(p).replace(str(HOME / 'Workspace') + '/', str(WS) + '/')


def ident(path):
    s = os.lstat(path)
    return dict(path=str(path), size=s.st_size, allocated=s.st_blocks * 512, inode=s.st_ino, device=s.st_dev,
                mtime_ns=s.st_mtime_ns)


def free(path):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize


def lineage():
    pngs, scenes = set(), set()

    def walk(x):
        if isinstance(x, str):
            if x.endswith('.png') and 'render_textured_v03' in x:
                pngs.add(norm(x))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for f in (CACHE / 'temporal_rows.jsonl', CACHE / 'two_step/two_step_rows.jsonl'):
        for line in f.read_text().splitlines():
            walk(json.loads(line))
    for line in (CACHE / 'proprio_v1/proprio_rows.jsonl').read_text().splitlines():
        scenes.add(json.loads(line)['scene'])
    scenes |= {Path(p).parts[-3] for p in pngs}
    assert len(scenes) == 80 and len(pngs) == 18690, (len(scenes), len(pngs))
    assert all(os.path.isfile(p) for p in pngs)
    return pngs, scenes


def scene_of(stage_dir_name):
    m = re.fullmatch(r'\d{6}_(.+)', stage_dir_name)
    return m.group(1) if m else stage_dir_name


def rollout_plan(scenes):
    remove, seen = [], set()
    for split in sorted(ROLLOUT.iterdir()):
        for directory, dirs, files in os.walk(split, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != 'sealed' and not d.startswith('sealed_'))
            parent = Path(directory)
            rel = parent.relative_to(ROLLOUT).parts  # split/family/chunk/stage/scene
            if len(rel) != 5 or rel[3] not in ('raw', 'plan', 'labels', 'rollout'):
                continue
            scene = scene_of(parent.name)
            seen.add(scene)
            if scene in scenes:
                continue
            for f in sorted(files):
                p = parent / f
                if f in BULK or f.endswith('.mcap'):
                    assert stat.S_ISREG(os.lstat(p).st_mode), p
                    remove.append(ident(p))
    assert len(seen) == SCENES and scenes <= seen, (len(seen), len(scenes - seen))
    return remove


def render_scene(scene_dir, keep):
    rgb = scene_dir / 'rgb'
    n, alloc, kept, inodes = 0, 0, 0, set()
    if rgb.is_dir():
        with os.scandir(rgb) as it:
            for e in it:
                if norm(e.path) in keep:
                    kept += 1
                    continue
                s = e.stat(follow_symlinks=False)
                assert stat.S_ISREG(s.st_mode), e.path
                n += 1
                if s.st_ino not in inodes:
                    inodes.add(s.st_ino)
                    alloc += s.st_blocks * 512
    return dict(scene=scene_dir.name, files=n, allocated=alloc, kept=kept)


def render_plan(keep):
    scenes = sorted(p for p in RENDER.iterdir() if p.is_dir() and not sealed(p))
    assert len(scenes) == SCENES, len(scenes)
    with ThreadPoolExecutor(12) as pool:
        return list(pool.map(lambda d: render_scene(d, keep), scenes))


def make_plan():
    pngs, scenes = lineage()
    render = render_plan(pngs)
    assert sum(r['kept'] for r in render) == len(pngs)
    return dict(lineage_scenes=sorted(scenes), lineage_frames=sorted(pngs), render=render, rollout=rollout_plan(scenes))


def summary(plan):
    g = lambda b: round(b / 2**30, 2)
    return dict(lineage_scenes=len(plan['lineage_scenes']), lineage_frames=len(plan['lineage_frames']),
                render_files=sum(r['files'] for r in plan['render']), render_gib=g(sum(r['allocated'] for r in plan['render'])),
                rollout_files=len(plan['rollout']), rollout_gib=g(sum(r['allocated'] for r in plan['rollout'])))


def unlink_checked(entry):
    s = os.lstat(entry['path'])
    assert stat.S_ISREG(s.st_mode)
    assert (s.st_ino, s.st_dev, s.st_size, s.st_mtime_ns) == (
        entry['inode'], entry['device'], entry['size'], entry['mtime_ns']), entry['path']
    os.unlink(entry['path'])


def execute(out):
    start = time.monotonic()
    before_free = free(WS)
    plan_path = out / 'plan.json'
    plan = json.loads(plan_path.read_text())
    now = make_plan()
    assert now == plan, 'corpus changed since the plan'
    keep = set(plan['lineage_frames'])
    RECEIPT.mkdir(exist_ok=False)
    shutil.copy2(plan_path, RECEIPT / 'plan.json')
    write(RECEIPT / 'authority.json', dict(
        user_text=['remove any artefacts not required', 'a'], approver='Andrew', date='2026-10-04',
        option='(a) keep the training lineage only: 80 scenes, 18,690 frames, small per-scene files, scene corpus',
        review='docs/go2_storage_review_2026-10-04.md', script_sha256=sha(__file__), plan_sha256=sha(plan_path)))
    done = {}
    try:
        with ThreadPoolExecutor(8) as pool:
            lineage_before = dict(zip(sorted(keep), pool.map(sha, sorted(keep))))
        write(RECEIPT / 'lineage_frames_sha256.json', lineage_before)
        mcap = [r for r in plan['rollout'] if r['path'].endswith('.mcap')]
        with ThreadPoolExecutor(8) as pool:
            write(RECEIPT / 'removed_recordings_sha256.json',
                  dict(zip([r['path'] for r in mcap], pool.map(lambda r: sha(r['path']), mcap))))
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(unlink_checked, plan['rollout']))
        done['rollout_files'] = len(plan['rollout'])

        def clear(row):
            scene = RENDER / row['scene']
            assert render_scene(scene, keep) == row, row['scene']
            n = 0
            rgb = scene / 'rgb'
            if rgb.is_dir():
                with os.scandir(rgb) as it:
                    for e in it:
                        if norm(e.path) not in keep:
                            os.unlink(e.path)
                            n += 1
            assert n == row['files'], (row['scene'], n)
            return n
        with ThreadPoolExecutor(8) as pool:
            done['render_files'] = sum(pool.map(clear, plan['render']))
        with ThreadPoolExecutor(8) as pool:
            lineage_after = dict(zip(sorted(keep), pool.map(sha, sorted(keep))))
        assert lineage_after == lineage_before, 'lineage frame mismatch'
        marker = dict(
            status='lineage_only', date='2026-10-04', receipt=str(RECEIPT),
            kept=('the 80 scenes and 18,690 rendered frames the current predictor lineage was trained on, every '
                  'small per-scene file (summaries, metadata, replay plans, render logs), driver logs; the scene '
                  'corpus lives in .generated/scene_corpus'),
            removed=('for the other 1,370 scenes: rendered RGB frames, .mcap recordings, messages.jsonl, '
                     'frames.jsonl and labels.jsonl; non-lineage rendered frames of the 80 lineage scenes'),
            regeneration='rollout from the scene corpus, then convert/label/plan and about two days of GPU '
                         'rendering; not guaranteed bit-identical', lineage_scenes=plan['lineage_scenes'])
        write(CORPUS / 'corpus_retention.json', marker)
        result = dict(status='completed', removed=done, summary=summary(plan), free_bytes_before=before_free,
                      free_bytes_after=free(WS), wall_seconds=time.monotonic() - start,
                      lineage_frames_verified=len(keep))
        write(RECEIPT / 'result.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write(RECEIPT / 'failure.json', dict(completed=done, error=repr(exc)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path)
    p.add_argument('--execute', type=Path)
    a = p.parse_args()
    if a.plan:
        a.plan.mkdir(parents=True, exist_ok=True)
        plan = make_plan()
        write(a.plan / 'plan.json', plan)
        print(json.dumps(summary(plan)))
    else:
        execute(a.execute)
