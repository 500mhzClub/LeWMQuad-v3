"""Bind only cache leaves derived from an explicitly constructed current scene."""
import hashlib
from pathlib import Path
import re
import stat


def cache_leaf(root, path):
    root = Path(root); path = Path(path)
    if (root.resolve() != root or root.is_symlink() or path.parent != root
            or not re.fullmatch(r'[0-9a-f]{64}\.gsd', path.name)
            or path.is_symlink() or path.resolve() != path):
        raise ValueError('ordinary exact current-scene GSD cache leaf required')
    return path


def capture(entities, root, path_for_geometry):
    rows = []; bindings = {}; seen = set()
    for entity in entities:
        for geom in entity.geoms:
            index = int(geom.idx)
            if index in seen: raise ValueError('unique native geometry enumeration required')
            seen.add(index)
            expected = cache_leaf(root, path_for_geometry(geom))
            preprocessed = bool(geom._is_preprocessed)
            if preprocessed and (Path(geom._gsd_path) != expected or not expected.is_file()):
                raise ValueError('loaded native cache must match the current geometry key')
            present = expected.exists()
            if present and expected.name not in bindings:
                before = expected.stat()
                if not stat.S_ISREG(before.st_mode): raise ValueError('regular current-scene cache required')
                with expected.open('rb') as stream: sha = hashlib.file_digest(stream, 'sha256').hexdigest()
                after = expected.stat()
                if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                    raise ValueError('cache changed during identity capture')
                bindings[expected.name] = dict(sha256=sha, byte_count=before.st_size)
            rows.append(dict(entity_name=str(entity.name), geometry_index=index, geometry_type=geom.type.name,
                preprocessed=preprocessed, expected_cache_name=expected.name, cache_present=present))
    if not rows: raise ValueError('nonempty actual current scene required')
    return dict(geometries=rows, cache_bindings=bindings, geometry_count=len(rows),
        unique_existing_cache_count=len(bindings), cache_root=str(root),
        cache_payloads_deserialized_by_inspector=False, inspected_geometry_source='explicit current scene only')
