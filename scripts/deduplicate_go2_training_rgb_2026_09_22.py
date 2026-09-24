"""Preserve all image paths/bytes while sharing identical completed train frames.

Bounded to three explicit historical training chunks; never traverses other
splits. Shared RGB inodes must remain immutable. No result or depth is removed.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time
import traceback

BASE = Path('.generated/datagen_full')
TRAIN = BASE/'rollout/train/large_enclosed_maze'
RENDER = BASE/'render_textured_v03'
OUTPUT = Path('.generated/training_rgb_deduplication_2026-09-22')
TARGET_BYTES = 6*1024**3


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def safe(path):
    assert not any(p == 'sealed' or p.startswith('sealed_') or p == 'sealed_test.json'
        for p in path.parts)
    assert not path.is_symlink()
    assert not any(p == 'sealed' or p.startswith('sealed_') or p == 'sealed_test.json'
        for p in path.resolve().parts)
    return path


def save(path, data):
    with path.open('x') as stream:
        json.dump(data, stream, indent=2)
        stream.write('\n')


def scenes():
    selected = []
    for chunk in ('chunk_0000', 'chunk_0040', 'chunk_0080'):
        directory = safe(TRAIN/chunk/'plan')
        for item in sorted(directory.iterdir()):
            if item.name == 'sealed' or item.name.startswith('sealed_') or item.name == 'sealed_test.json':
                continue
            if item.is_symlink() or not item.is_dir():
                continue
            plan_path = safe(item/'render_replay_plan.json')
            plan = json.loads(plan_path.read_text())
            assert plan['split'] == 'train' and plan['scene_family'] == 'large_enclosed_maze'
            scene = plan['scene_id']
            assert scene.startswith('large_enclosed_maze_') and Path(scene).name == scene
            root = safe(RENDER/scene)
            summary = json.loads(safe(root/'summary.json').read_text())
            assert summary['split'] == 'train' and summary['render_status'] == 'complete'
            assert summary['scene_id'] == scene and Path(summary['plan']).resolve() == plan_path.resolve()
            assert safe(root/'.render_done').is_file()
            selected.append(dict(scene=scene, root=str(root), plan=str(plan_path),
                plan_sha256=digest(plan_path), summary_sha256=digest(root/'summary.json'),
                frames=summary['frame_count']))
    assert 0 < len(selected) <= 120
    return selected


def process_scene(record, index):
    root = safe(Path(record['root']))
    grouped = defaultdict(list)
    count = 0
    for path in safe(root/'rgb').iterdir():
        if path.name == 'sealed' or path.name.startswith('sealed_') or path.name == 'sealed_test.json':
            continue
        if path.suffix != '.png':
            continue
        safe(path)
        info = path.lstat()
        assert stat.S_ISREG(info.st_mode)
        grouped[info.st_size].append((path, info))
        count += 1
    assert count == record['frames']
    replacements = []
    for items in grouped.values():
        if len(items) < 2:
            continue
        originals = {}
        for path, info in sorted(items):
            sha = digest(path)
            if sha not in originals:
                originals[sha] = (path, info)
                continue
            source, source_info = originals[sha]
            assert source_info.st_dev == info.st_dev
            if info.st_nlink != 1 or source_info.st_ino == info.st_ino:
                continue
            replacements.append(dict(path=str(path), source=str(source), sha256=sha,
                size=info.st_size, inode_before=info.st_ino, source_inode=source_info.st_ino,
                mtime_ns_before=info.st_mtime_ns, duplicate_allocated_bytes=info.st_blocks*512))
    save(OUTPUT/f'scene_{index:03d}_manifest.json', dict(record=record, replacements=replacements))
    with (OUTPUT/f'scene_{index:03d}_completed.jsonl').open('x') as receipt:
        for row in replacements:
            source, path = safe(Path(row['source'])), safe(Path(row['path']))
            before = path.lstat()
            assert before.st_ino == row['inode_before'] and before.st_nlink == 1
            assert before.st_mtime_ns == row['mtime_ns_before'] and before.st_size == row['size']
            assert source.stat().st_ino == row['source_inode']
            assert digest(source) == digest(path) == row['sha256']
            temporary = path.with_name(path.name+'.dedup_link_tmp')
            assert not temporary.exists()
            os.link(source, temporary, follow_symlinks=False)
            try:
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            assert path.stat().st_ino == source.stat().st_ino
            assert digest(path) == row['sha256']
            receipt.write(json.dumps(row)+'\n')
            receipt.flush()
    summary = dict(scene=record['scene'], frames=count, replaced=len(replacements),
        duplicate_allocated_bytes=sum(r['duplicate_allocated_bytes'] for r in replacements),
        all_replaced_image_hashes_preserved=True, image_paths_removed=0,
        shared_rgb_inodes_must_remain_immutable=True)
    save(root/'rgb_deduplication_2026-09-22.json', summary|dict(receipt_root=str(OUTPUT.resolve())))
    return summary


def main():
    selected = scenes()
    # The active readout fit's image population must be outside these scenes.
    active = Path('.generated/navigation_development_artifacts_v1/go2_multihorizon_motion_readout_v1_attempt_001/frame_paths.json')
    roots = [Path(r['root']).resolve() for r in selected]
    assert all(not any(Path(p).resolve().is_relative_to(r) for r in roots)
        for p in json.loads(active.read_text()))
    OUTPUT.mkdir(exist_ok=False)
    free_before = shutil.disk_usage(OUTPUT).free
    save(OUTPUT/'plan.json', dict(scenes=selected, target_duplicate_allocated_bytes=TARGET_BYTES,
        maximum_scenes=120, source_sha256=digest(Path(__file__)),
        policy='byte-identical hardlink deduplication, not data retirement',
        per_scene_only=True, preserve_all_paths_and_content_hashes=True,
        shared_images_must_remain_immutable=True, free_bytes_before=free_before))
    save(OUTPUT/'process.json', dict(pid=os.getpid()))
    completed = []
    started = time.monotonic()
    try:
        for index, record in enumerate(selected):
            summary = process_scene(record, index)
            completed.append(summary)
            eliminated = sum(r['duplicate_allocated_bytes'] for r in completed)
            print('TRAIN_RGB_DEDUP', index+1, summary['replaced'], eliminated,
                round(time.monotonic()-started, 1), flush=True)
            if eliminated >= TARGET_BYTES:
                break
        save(OUTPUT/'result.json', dict(status='COMPLETE', scenes=completed,
            duplicate_allocated_bytes=sum(r['duplicate_allocated_bytes'] for r in completed),
            free_bytes_before=free_before, free_bytes_after=shutil.disk_usage(OUTPUT).free,
            wall_s=time.monotonic()-started, frames_or_results_removed=0,
            note='free-space difference also includes unrelated concurrent writes'))
    except BaseException as error:
        save(OUTPUT/'failure.json', dict(reason=repr(error), traceback=traceback.format_exc(),
            completed_scenes=completed, partial_scene_receipts_retained=True))
        raise


if __name__ == '__main__':
    main()
