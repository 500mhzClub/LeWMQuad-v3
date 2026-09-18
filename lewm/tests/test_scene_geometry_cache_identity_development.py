from types import SimpleNamespace
import hashlib
import pytest
from scripts.scene_geometry_cache_identity_development import capture, cache_leaf


def test_only_geometry_derived_cache_leaves_are_read_and_duplicate_keys_are_bound_once(tmp_path):
    path = tmp_path/('a'*64+'.gsd'); path.write_bytes(b'opaque current geometry bytes')
    absent = tmp_path/('b'*64+'.gsd')
    def geom(i, p, loaded):
        return SimpleNamespace(idx=i, type=SimpleNamespace(name='BOX'), _is_preprocessed=loaded, _gsd_path=str(p), path=p)
    gs = [geom(0, path, True), geom(1, path, True), geom(2, absent, False)]
    r = capture([SimpleNamespace(name='known_scene', geoms=gs)], tmp_path, lambda g: g.path)
    assert r['geometry_count'] == 3 and r['unique_existing_cache_count'] == 1
    assert r['cache_bindings'][path.name]['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not r['cache_payloads_deserialized_by_inspector'] and not absent.exists()
    gs[0]._gsd_path = str(absent)
    with pytest.raises(ValueError): capture([SimpleNamespace(name='known_scene', geoms=gs)], tmp_path, lambda g: g.path)


def test_protected_nonnamed_and_linked_leaves_cannot_be_opened(tmp_path):
    for name in ['sealed_test.json', 'sealed/a.gsd', 'sealed_legacy/a.gsd', '../outside.gsd', 'arbitrary.gsd']:
        with pytest.raises(ValueError): cache_leaf(tmp_path, tmp_path/name)
    leaf = tmp_path/('c'*64+'.gsd'); leaf.symlink_to(tmp_path/'missing')
    with pytest.raises(ValueError): cache_leaf(tmp_path, leaf)
