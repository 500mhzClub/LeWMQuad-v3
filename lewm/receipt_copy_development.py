"""Container copying with inline atomic leaves and standard fallback semantics.

Separately named candidate, not a replacement of copy.deepcopy in any module.
Keep a single memo for aliasing, cycles, custom copying and NumPy fallback.
"""
from copy import deepcopy

_ATOMIC = frozenset((type(None), bool, int, float, complex, str, bytes))
_MISSING = object()


def _clone(value, memo):
    identity = id(value)
    saved = memo.get(identity, _MISSING)
    if saved is not _MISSING:
        return saved
    kind = type(value)
    if kind in _ATOMIC:
        return value
    if kind is dict:
        result = {}
        memo[identity] = result
        for key, child in value.items():
            # Match ordinary assignment evaluation: copy value before key.
            copied_child = memo.get(id(child), child) if type(child) in _ATOMIC else _clone(child, memo)
            copied_key = memo.get(id(key), key) if type(key) in _ATOMIC else _clone(key, memo)
            result[copied_key] = copied_child
    elif kind is list:
        result = []
        memo[identity] = result
        for child in value:
            result.append(memo.get(id(child), child) if type(child) in _ATOMIC else _clone(child, memo))
    elif kind is tuple:
        items = [memo.get(id(child), child) if type(child) in _ATOMIC else _clone(child, memo) for child in value]
        # A list beneath this tuple can recursively finish the tuple first.
        saved = memo.get(identity, _MISSING)
        if saved is not _MISSING:
            return saved
        result = value if all(a is b for a, b in zip(value, items)) else tuple(items)
    else:
        return deepcopy(value, memo)
    if result is not value:
        memo[identity] = result
        # Preserve standard deepcopy's lifetime guarantee for temporary sources.
        try:
            memo[id(memo)].append(value)
        except KeyError:
            memo[id(memo)] = [value]
    return result


def copy_receipt(value):
    return _clone(value, {})
