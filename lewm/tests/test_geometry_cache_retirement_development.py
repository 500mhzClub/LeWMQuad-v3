import pytest
from scripts.geometry_cache_retirement_development import candidates, metadata, retire


def test_current_geometry_keys_are_retained_and_only_named_cache_leaves_removed(tmp_path):
    names = ['a'*64+'.gsd', 'b'*64+'.gsd']
    for name in names: (tmp_path/name).write_bytes(name.encode())
    inventory = {n: metadata(tmp_path/n) for n in names}
    entries, keep = candidates(inventory, dict(cache_root=str(tmp_path), geometries=[dict(expected_cache_name=names[0])]))
    removed = []; retire(tmp_path, entries, keep, removed.append)
    assert (tmp_path/names[0]).is_file() and not (tmp_path/names[1]).exists() and removed == [names[1]]


def test_changed_late_member_aborts_before_any_unlink_and_protected_path_is_rejected(tmp_path):
    names = ['a'*64+'.gsd', 'b'*64+'.gsd']
    for name in names: (tmp_path/name).write_bytes(b'old')
    entries = {n: metadata(tmp_path/n) for n in names}
    (tmp_path/names[1]).write_bytes(b'changed')
    with pytest.raises(ValueError): retire(tmp_path, entries, [], lambda n: pytest.fail('partial removal before full validation'))
    assert all((tmp_path/n).exists() for n in names)
    with pytest.raises(ValueError): retire(tmp_path, {'sealed_test.json': {}}, [], lambda n: None)
