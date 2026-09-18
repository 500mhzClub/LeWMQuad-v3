"""Borrow read-only surface receipts inside the unchanged recovery calculation.

Only the six reviewed functions below receive this private copy provider. They
read surface_checks and replace that field, but never mutate its descendants.
Borrowing is limited to ordinary container graphs disjoint from every other
selection field. A changed public result detaches every borrowed receipt.
"""
from copy import deepcopy
from types import FunctionType, MappingProxyType
from lewm.receipt_copied_anchored_selection_development import ORIGINAL_FUNCTIONS
from lewm.residual_anchored_continuation_development import reconsider_anchored_continuation


def plain_graph_ids(value):
    """Return container identities; reject custom values and cyclic graphs."""
    seen = set()
    active = set()

    def visit(node):
        kind = type(node)
        if kind in (type(None), bool, int, float, str):
            return True
        if kind not in (dict, list):
            return False
        identity = id(node)
        if identity in active:
            return False
        if identity in seen:
            return True
        if kind is dict and any(type(key) is not str for key in node):
            return False
        seen.add(identity)
        active.add(identity)
        ok = all(visit(child) for child in (node.values() if kind is dict else node))
        active.remove(identity)
        return ok

    return seen if visit(value) else None


class SurfaceReceiptWorkspace:
    """One invocation only; no borrowed object survives in controller state."""
    def __init__(self):
        self.surfaces = {}
        self.borrowed = {}

    def copy(self, value):
        if type(value) is dict and type(value.get('surface_checks')) is list:
            surface = value['surface_checks']
            identity = id(surface)
            if identity not in self.surfaces:
                # Retain the exact source object as well as its identities, so
                # an object id cannot be reused within this invocation.
                self.surfaces[identity] = (surface, plain_graph_ids(surface))
            surface_ids = self.surfaces[identity][1]
            others = {key: child for key, child in value.items() if key != 'surface_checks'}
            other_ids = plain_graph_ids(others)
            if (surface_ids is not None and other_ids is not None
                    and id(value) not in surface_ids and surface_ids.isdisjoint(other_ids)):
                self.borrowed[identity] = surface
                return deepcopy(value, {identity: surface})
        return deepcopy(value)

    def finish(self, result, original):
        if result is original or not self.borrowed:
            return result
        memo = {}

        def detach(node):
            identity = id(node)
            if identity in memo:
                return memo[identity]
            if identity in self.borrowed:
                return deepcopy(node, memo)
            # Non-borrowed leaves already have the original calculation's
            # ownership. In particular, do not execute custom copying twice.
            if type(node) is dict:
                output = {}
                memo[identity] = output
                output.update((key, detach(child)) for key, child in node.items())
                return output
            if type(node) is list:
                output = []
                memo[identity] = output
                output.extend(detach(child) for child in node)
                return output
            return node

        return detach(result)


def isolated_selection_functions(workspace):
    clones = {}
    copier = workspace.copy
    for original in ORIGINAL_FUNCTIONS:
        if original.__closure__ is not None or original.__globals__.get('deepcopy') is not deepcopy:
            raise ValueError('reviewed closure-free functions and original copy provider required')
        namespace = original.__globals__.copy()
        copied = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
        copied.__kwdefaults__ = original.__kwdefaults__
        copied.__annotations__ = original.__annotations__.copy()
        copied.__qualname__ = original.__qualname__
        copied.__module__ = original.__module__
        clones[original] = copied
    for copied in clones.values():
        namespace = copied.__globals__
        namespace['deepcopy'] = copier
        for name, target in tuple(namespace.items()):
            if isinstance(target, FunctionType) and target in clones:
                namespace[name] = clones[target]
    return MappingProxyType(clones)


def reconsider_with_shared_surface(selection, receipt, mapper, geometry, *, now_ns):
    workspace = SurfaceReceiptWorkspace()
    functions = isolated_selection_functions(workspace)
    result = functions[reconsider_anchored_continuation](
        selection, receipt, mapper, geometry, now_ns=now_ns)
    return workspace.finish(result, selection)
