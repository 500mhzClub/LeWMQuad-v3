"""Inline primitive receipt leaves while preserving the original scoped cache."""
from types import FunctionType

from lewm import fused_scoped_footprint_development as original
from lewm.frozen_footprint_receipts_development import _ReceiptDict, _ReceiptList

_PRIMITIVE = (type(None), bool, int, float, str)


def freeze_ordinary_footprint(value):
    """Same acyclic ordinary graph contract; visit only nonprimitive children."""
    if type(value) is not dict:
        return value
    memo, active = {}, set()

    def visit(node):
        kind = type(node)
        if kind not in (dict, list):
            raise original._UnsupportedGraph
        identity = id(node)
        if identity in active:
            raise original._UnsupportedGraph
        if identity in memo:
            return memo[identity]
        active.add(identity)
        if kind is dict:
            out = dict.__new__(_ReceiptDict)
            memo[identity] = out
            for key, child in node.items():
                if type(key) is not str:
                    raise original._UnsupportedGraph
                dict.__setitem__(out, key, child if type(child) in _PRIMITIVE else visit(child))
        else:
            out = list.__new__(_ReceiptList)
            memo[identity] = out
            for child in node:
                list.append(out, child if type(child) in _PRIMITIVE else visit(child))
        active.remove(identity)
        return out

    try:
        return visit(value)
    except original._UnsupportedGraph:
        return value


def _clone_cached_receipt(value):
    """Clone validated frozen containers; return all other leaves unchanged."""
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
                dict.__setitem__(out, key, clone(child) if type(child) in (_ReceiptDict, _ReceiptList) else child)
        else:
            out = list.__new__(_ReceiptList)
            memo[identity] = out
            for child in node:
                list.append(out, clone(child) if type(child) in (_ReceiptDict, _ReceiptList) else child)
        return out

    return clone(value)


def _isolated_footprint():
    function = original.FusedScopedFootprintReuse.footprint
    if function.__closure__ is not None:
        raise ValueError('closure-free original cache query required')
    clone = FunctionType(function.__code__, function.__globals__ | dict(
        freeze_ordinary_footprint=freeze_ordinary_footprint,
        _clone_cached_receipt=_clone_cached_receipt), function.__name__, function.__defaults__)
    clone.__kwdefaults__ = function.__kwdefaults__
    return clone


class AtomicLeafFusedScopedFootprintReuse(original.FusedScopedFootprintReuse):
    footprint = _isolated_footprint()
