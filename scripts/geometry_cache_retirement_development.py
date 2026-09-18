"""Metadata-only selection and deletion of explicitly approved ordinary cache leaves."""
import os
from pathlib import Path
import stat
from scripts.scene_geometry_cache_identity_development import cache_leaf


def metadata(path):
    s = Path(path).lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != 1:
        raise ValueError('owned singly linked regular cache leaf required')
    return dict(inode=s.st_ino, byte_count=s.st_size, allocated_bytes=s.st_blocks*512,
        mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns, owner_uid=s.st_uid, link_count=s.st_nlink)


def candidates(inventory, current_geometry):
    keep = {r['expected_cache_name'] for r in current_geometry['geometries']}
    if not keep: raise ValueError('nonempty current geometry key retention set required')
    result = {name: entry for name, entry in inventory.items() if name not in keep}
    for name in set(inventory) | keep:
        cache_leaf(Path(current_geometry['cache_root']), Path(current_geometry['cache_root'])/name)
    return result, sorted(keep)


def validate(root, entries, keep):
    if set(entries) & set(keep): raise ValueError('current scene cache key cannot be retired')
    for name, expected in entries.items():
        path = cache_leaf(root, Path(root)/name)
        if metadata(path) != expected: raise ValueError('cache metadata changed since proposal: '+name)


def retire(root, entries, keep, record):
    # Caller owns explicit user authorization and job quiescence. Validate the
    # complete proposed population before unlinking its first member.
    validate(root, entries, keep)
    for name, expected in entries.items():
        path = cache_leaf(root, Path(root)/name)
        if metadata(path) != expected: raise ValueError('cache changed immediately before retirement: '+name)
        path.unlink()
        record(name)
