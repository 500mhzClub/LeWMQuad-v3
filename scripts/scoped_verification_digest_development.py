"""Isolated verification digests with stable-file guards and final fresh hashes.

No cache crosses one verification call. Every cached path is freshly hashed
again before success; original verification conditions are still executed.
This helper is not installed in any frozen or running verifier.
"""
import hashlib
import os
from pathlib import Path
import stat
from types import CodeType, FunctionType

MAX_FILES = 200_000
CHUNK_BYTES = 4*1024**2


def ordinary_path(path):
    path = Path(path)
    if any(p in ('sealed', 'sealed_test.json', '..') or p.startswith('sealed_') for p in path.parts):
        raise ValueError('protected or noncanonical digest path forbidden')
    path = path.absolute()
    if path.resolve(strict=True) != path:
        raise ValueError('nonsymlink digest path required')
    return path


def identity(value):
    if not stat.S_ISREG(value.st_mode): raise ValueError('ordinary digest file required')
    return tuple(getattr(value, key) for key in ('st_dev', 'st_ino', 'st_mode', 'st_uid', 'st_gid',
        'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink'))


def fresh_digest(path):
    path = ordinary_path(path); before = identity(path.stat())
    fd = os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as stream:
        if identity(os.fstat(stream.fileno())) != before:
            raise ValueError('digest file changed while opening')
        value = hashlib.sha256()
        while chunk := stream.read(CHUNK_BYTES): value.update(chunk)
        if identity(os.fstat(stream.fileno())) != before:
            raise ValueError('digest file changed during hashing')
    if ordinary_path(path) != path or identity(path.stat()) != before:
        raise ValueError('digest path changed during hashing')
    return value.hexdigest(), before


class DigestScope:
    def __init__(self, *, maximum_files=MAX_FILES, digest_function=None):
        if type(maximum_files) is not int or maximum_files < 1:
            raise ValueError('positive bounded digest population required')
        self.maximum_files = maximum_files
        self.digest_function = digest_function
        self.entries = {}; self.closed = False
        self.requests = self.hits = self.initial_bytes = self.final_bytes = 0

    def __call__(self, path):
        if self.closed: raise ValueError('digest scope is closed')
        path = ordinary_path(path); current = identity(path.stat()); self.requests += 1
        if path in self.entries:
            value, previous = self.entries[path]
            if current != previous: raise ValueError('cached digest file identity changed')
            self.hits += 1
            return value
        if len(self.entries) >= self.maximum_files: raise ValueError('digest scope file allowance exceeded')
        if self.digest_function is None:
            value, observed = fresh_digest(path)
        else:
            # Execute the original digest on the first request for each path.
            # Final independent streaming SHA-256 must agree with its result.
            value = self.digest_function(path)
            if ordinary_path(path) != path: raise ValueError('digest path changed')
            observed = identity(path.stat())
        if observed != current: raise ValueError('digest file changed before hashing')
        self.entries[path] = (value, observed); self.initial_bytes += current[5]
        return value

    def finish(self):
        if self.closed: raise ValueError('digest scope is closed')
        try:
            for path, (expected, previous) in self.entries.items():
                actual, current = fresh_digest(path); self.final_bytes += current[5]
                if current != previous or actual != expected:
                    raise ValueError('fresh final digest or file identity changed')
            # Detect changes to an earlier path while later paths were hashed.
            for path, (_, previous) in self.entries.items():
                if ordinary_path(path) != path or identity(path.stat()) != previous:
                    raise ValueError('digest population changed during final verification')
            return dict(digest_requests=self.requests, guarded_cache_hits=self.hits,
                unique_files=len(self.entries), initial_hashed_bytes=self.initial_bytes,
                final_hashed_bytes=self.final_bytes, every_cached_file_freshly_rehashed=True,
                cache_retained_after_call=False)
        finally:
            self.close()

    def close(self):
        self.entries.clear(); self.closed = True


def isolated_verifier(verifier, digest_function, scoped_digest):
    """Clone only named verification functions and their explicit digest global."""
    if not isinstance(verifier, FunctionType) or not verifier.__name__.startswith('verify'):
        raise ValueError('explicit verification function required')
    if not isinstance(digest_function, FunctionType): raise ValueError('explicit original digest function required')
    clones = {}
    def referenced_names(code):
        names = set(code.co_names)
        for constant in code.co_consts:
            if isinstance(constant, CodeType): names.update(referenced_names(constant))
        return names
    def clone(function):
        if function in clones: return clones[function]
        namespace = function.__globals__.copy()
        result = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
        result.__kwdefaults__ = function.__kwdefaults__; result.__annotations__ = function.__annotations__
        result.__qualname__ = function.__qualname__; result.__doc__ = function.__doc__
        clones[function] = result
        for name in referenced_names(function.__code__):
            target = namespace.get(name)
            if target is digest_function: namespace[name] = scoped_digest
            elif (isinstance(target, FunctionType) and target.__name__.startswith('verify')
                    and (target.__module__ == verifier.__module__
                        or target.__module__.startswith(('scripts.', 'lewm.')))):
                namespace[name] = clone(target)
        return result
    return clone(verifier), clones


def verify_with_scoped_digests(verifier, digest_function, *args, maximum_files=MAX_FILES, **kwargs):
    scope = DigestScope(maximum_files=maximum_files, digest_function=digest_function)
    try:
        copied, clones = isolated_verifier(verifier, digest_function, scope)
        result = copied(*args, **kwargs)
        report = scope.finish()
        return result, report|dict(isolated_verification_functions=len(clones),
            imported_module_globals_mutated=False, original_verification_conditions_executed=True)
    finally:
        scope.close()
