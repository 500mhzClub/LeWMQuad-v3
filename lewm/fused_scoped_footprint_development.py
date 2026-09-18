"""Single-pass freezing and direct cloning of invocation-local receipts.

The original cache lifetime, exact keys, failure checks and capacity remain.
Independent queries still receive independent containers, including on hits.
These read-only Python containers have the original non-security semantics.
"""
import numpy as np

from lewm.frozen_footprint_receipts_development import _ReceiptDict, _ReceiptList
from lewm.scoped_footprint_reuse_development import ScopedFootprintReuse


class _UnsupportedGraph(Exception):
    pass


def freeze_ordinary_footprint(value):
    """Validate and freeze together; unsupported graphs retain input identity."""
    if type(value) is not dict:
        return value
    memo, active = {}, set()

    def visit(node):
        kind = type(node)
        if kind in (type(None), bool, int, float, str):
            return node
        if kind not in (dict, list):
            raise _UnsupportedGraph
        identity = id(node)
        if identity in active:
            raise _UnsupportedGraph
        if identity in memo:
            return memo[identity]
        active.add(identity)
        if kind is dict:
            out = dict.__new__(_ReceiptDict)
            memo[identity] = out
            for key, child in node.items():
                if type(key) is not str:
                    raise _UnsupportedGraph
                dict.__setitem__(out, key, visit(child))
        else:
            out = list.__new__(_ReceiptList)
            memo[identity] = out
            for child in node:
                list.append(out, visit(child))
        active.remove(identity)
        return out

    try:
        return visit(value)
    except _UnsupportedGraph:
        return value


def _clone_cached_receipt(value):
    """Clone only a graph already produced by our validated freeze operation."""
    memo = {}

    def clone(node):
        kind = type(node)
        if kind not in (_ReceiptDict, _ReceiptList):
            return node
        identity = id(node)
        if identity in memo:
            return memo[identity]
        if kind is _ReceiptDict:
            out = dict.__new__(_ReceiptDict)
            memo[identity] = out
            for key, child in node.items():
                dict.__setitem__(out, key, clone(child))
        else:
            out = list.__new__(_ReceiptList)
            memo[identity] = out
            for child in node:
                list.append(out, clone(child))
        return out

    return clone(value)


class FusedScopedFootprintReuse(ScopedFootprintReuse):
    def footprint(self, geometry, displacement_body_xy, yaw_rad, *, now_ns, persistent=True):
        if not self._active:
            raise ValueError('footprint query may only be used inside its scope')
        self.requests += 1
        current = getattr(self._memory, '_current', None)
        if current is not None:
            current(now_ns)
        key = None
        if (geometry is self._geometry and type(now_ns) is int and type(persistent) is bool
                and type(yaw_rad) is float and type(displacement_body_xy) in (list, tuple, np.ndarray)):
            try:
                xy = np.asarray(displacement_body_xy, dtype=float)
                yaw = np.asarray(yaw_rad, dtype=float)
                if xy.shape == (2,) and yaw.shape == () and np.isfinite(xy).all() and np.isfinite(yaw):
                    key = (xy.tobytes(), yaw.tobytes(), now_ns, persistent)
            except (TypeError, ValueError, OverflowError):
                pass
        if key is not None and key in self._cache:
            self.hits += 1
            return _clone_cached_receipt(self._cache[key])
        self.computations += 1
        result = freeze_ordinary_footprint(self._memory.footprint(
            geometry, displacement_body_xy, yaw_rad, now_ns=now_ns, persistent=persistent))
        if key is not None and type(result) is _ReceiptDict and len(self._cache) < self.MAX_ENTRIES:
            self._cache[key] = result
        return result
