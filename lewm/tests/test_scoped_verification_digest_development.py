"""Content integrity and scope isolation under repeated ancestry and mutations."""
import hashlib
import os
from pathlib import Path
from types import FunctionType
import pytest
from scripts import scoped_verification_digest_development as scoped


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_leaf(path, expected):
    if digest(path) != expected: raise ValueError('original binding mismatch')
    return 'checked'


def verify_diamond(path, expected):
    first = verify_leaf(path, expected)
    if digest(path) != expected: raise ValueError('original diamond mismatch')
    second = verify_leaf(path, expected)
    return first, second


def verify_comprehension(path, expected):
    return [verify_leaf(path, expected) for _ in range(3)]


def test_nested_code_globals_are_bound_without_changing_the_original(tmp_path):
    path = tmp_path/'data'; path.write_bytes(b'original')
    result, report = scoped.verify_with_scoped_digests(verify_comprehension, digest, path, digest(path))
    assert result == ['checked']*3 and report['guarded_cache_hits'] == 2
    assert report['digest_requests'] == 3 and report['isolated_verification_functions'] == 2


def test_same_original_conditions_execute_with_isolated_globals_and_exact_hashes(tmp_path):
    path = tmp_path/'data'; path.write_bytes(b'original data')
    original_globals = dict(verify_diamond.__globals__)
    expected = digest(path)
    result, report = scoped.verify_with_scoped_digests(verify_diamond, digest, path, expected)
    assert result == verify_diamond(path, expected) == ('checked', 'checked')
    assert report['digest_requests'] == 3 and report['guarded_cache_hits'] == 2
    assert report['unique_files'] == 1 and report['isolated_verification_functions'] == 2
    assert report['initial_hashed_bytes'] == report['final_hashed_bytes'] == path.stat().st_size
    assert not report['cache_retained_after_call'] and report['every_cached_file_freshly_rehashed']
    assert all(verify_diamond.__globals__[k] is v for k,v in original_globals.items())


def test_cloned_methods_retain_code_closure_defaults_and_other_dependencies():
    cache = scoped.DigestScope()
    copied, clones = scoped.isolated_verifier(verify_diamond, digest, cache)
    assert copied is not verify_diamond and set(clones) == {verify_diamond, verify_leaf}
    for old, new in clones.items():
        assert new.__code__ is old.__code__ and new.__closure__ is old.__closure__
        assert new.__defaults__ is old.__defaults__ and new.__kwdefaults__ is old.__kwdefaults__
        for name, value in old.__globals__.items():
            expected = cache if value is digest else clones.get(value, value) if isinstance(value, FunctionType) else value
            # Only globals actually referenced by the function are rebound.
            if name not in old.__code__.co_names: expected = value
            assert new.__globals__[name] is expected


@pytest.mark.parametrize('when', ['before_repeat', 'before_finish', 'replace_inode', 'restore_mtime', 'mode'])
def test_any_changed_cached_file_is_rejected(tmp_path, when):
    path = tmp_path/'data'; path.write_bytes(b'old')
    scope = scoped.DigestScope(); scope(path); before = path.stat()
    if when == 'replace_inode':
        other = tmp_path/'replacement'; other.write_bytes(b'old'); other.replace(path)
    elif when == 'mode': path.chmod(0o400)
    else:
        path.write_bytes(b'new')
        if when == 'restore_mtime': os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError): scope(path) if when == 'before_repeat' else scope.finish()
    if when != 'before_repeat': assert scope.closed and not scope.entries


def test_final_content_hash_is_required_even_if_metadata_guard_were_insufficient(monkeypatch, tmp_path):
    path = tmp_path/'data'; path.write_bytes(b'old'); scope = scoped.DigestScope(); scope(path)
    original = scoped.fresh_digest
    def changed(path):
        value, identity = original(path)
        return '0'*64, identity
    monkeypatch.setattr(scoped, 'fresh_digest', changed)
    with pytest.raises(ValueError, match='final digest'): scope.finish()
    assert scope.closed and not scope.entries


@pytest.mark.parametrize('kind', ['sealed_dir', 'sealed_prefix', 'sealed_file', 'symlink', 'parent', 'directory'])
def test_protected_or_nonordinary_paths_rejected_before_content_reads(monkeypatch, tmp_path, kind):
    ordinary = tmp_path/'ordinary'; ordinary.write_bytes(b'allowed')
    if kind == 'sealed_dir': path = tmp_path/'sealed'/'data'
    elif kind == 'sealed_prefix': path = tmp_path/'sealed_fixture'/'data'
    elif kind == 'sealed_file': path = tmp_path/'sealed_test.json'
    elif kind == 'symlink': path = tmp_path/'link'; path.symlink_to(ordinary)
    elif kind == 'parent': path = tmp_path/'..'/'data'
    else: path = tmp_path
    # No protected file or directory is ever created or opened by these tests.
    def forbidden(*args, **kwargs): pytest.fail('nonordinary payload opened')
    monkeypatch.setattr(scoped.os, 'open', forbidden)
    with pytest.raises(ValueError): scoped.DigestScope()(path)


def test_failed_original_verification_never_returns_cached_success(tmp_path):
    path = tmp_path/'data'; path.write_bytes(b'old')
    with pytest.raises(ValueError, match='original binding mismatch'):
        scoped.verify_with_scoped_digests(verify_diamond, digest, path, '0'*64)
    path.write_bytes(b'new')
    value, report = scoped.verify_with_scoped_digests(verify_diamond, digest, path, digest(path))
    assert value == ('checked', 'checked') and report['unique_files'] == 1


def test_cache_cannot_be_reused_after_finish_and_population_is_bounded(tmp_path):
    one = tmp_path/'one'; two = tmp_path/'two'; one.write_bytes(b'1'); two.write_bytes(b'2')
    scope = scoped.DigestScope(maximum_files=1); scope(one)
    with pytest.raises(ValueError, match='allowance'): scope(two)
    scope.finish()
    with pytest.raises(ValueError, match='closed'): scope(one)
    with pytest.raises(ValueError, match='closed'): scope.finish()
    assert not scope.entries


def test_changed_file_during_hash_is_rejected(monkeypatch, tmp_path):
    path = tmp_path/'data'; path.write_bytes(b'old')
    original = scoped.os.fstat; calls = []
    def stat_and_mutate(fd):
        calls.append(fd)
        if len(calls) == 2: path.write_bytes(b'new')
        return original(fd)
    monkeypatch.setattr(scoped.os, 'fstat', stat_and_mutate)
    with pytest.raises(ValueError, match='during hashing'): scoped.fresh_digest(path)


def test_first_original_digest_is_executed_and_final_hash_is_independent(tmp_path):
    path = tmp_path/'data'; path.write_bytes(b'original'); calls = []
    def original(path): calls.append(path); return digest(path)
    scope = scoped.DigestScope(digest_function=original)
    assert scope(path) == scope(path) == digest(path) and calls == [path]
    report = scope.finish()
    assert calls == [path] and report['every_cached_file_freshly_rehashed']


def test_change_to_earlier_file_during_final_population_check_fails(monkeypatch, tmp_path):
    one = tmp_path/'one'; two = tmp_path/'two'; one.write_bytes(b'1'); two.write_bytes(b'2')
    scope = scoped.DigestScope(); scope(one); scope(two)
    original = scoped.fresh_digest
    def changing(path):
        value = original(path)
        if path == two: one.write_bytes(b'3')
        return value
    monkeypatch.setattr(scoped, 'fresh_digest', changing)
    with pytest.raises(ValueError, match='population changed'): scope.finish()
    assert scope.closed and not scope.entries
