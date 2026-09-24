"""Invocation-local read-only footprint receipts, detached before public return.

These containers optimize ordinary deepcopy calls in the existing selector.
They are not a security boundary against explicitly invoking base-class C
mutators. No frozen receipt is stored in the mapper or exposed by the selector.
"""
from lewm.shared_surface_anchored_selection_development import plain_graph_ids


def _readonly(*args, **kwargs):
    raise TypeError('footprint receipt is read-only during selection')


class _ReceiptDict(dict):
    __slots__ = ()
    __init__ = __setitem__ = __delitem__ = __ior__ = _readonly
    clear = pop = popitem = setdefault = update = _readonly

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _ReceiptList(list):
    __slots__ = ()
    __init__ = __setitem__ = __delitem__ = __iadd__ = __imul__ = _readonly
    append = clear = extend = insert = pop = remove = reverse = sort = _readonly

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def freeze_footprint(value):
    """Copy only an acyclic ordinary JSON graph; retain unsupported inputs."""
    if type(value) is not dict or plain_graph_ids(value) is None:
        return value
    memo = {}

    def freeze(node):
        kind = type(node)
        if kind not in (dict, list):
            return node
        identity = id(node)
        if identity in memo:
            return memo[identity]
        if kind is dict:
            out = dict.__new__(_ReceiptDict)
            memo[identity] = out
            for key, child in node.items():
                dict.__setitem__(out, key, freeze(child))
        else:
            out = list.__new__(_ReceiptList)
            memo[identity] = out
            for child in node:
                list.append(out, freeze(child))
        return out

    return freeze(value)


def detach_receipts(value):
    """Return ordinary public containers, preserving aliases and owned leaves."""
    memo = {}

    def detach(node):
        kind = type(node)
        if kind not in (dict, list, tuple, _ReceiptDict, _ReceiptList):
            return node
        identity = id(node)
        if identity in memo:
            return memo[identity]
        if kind is tuple:
            children = [detach(child) for child in node]
            # A tuple/list cycle may have constructed this tuple while its
            # children were being visited, as in ordinary deepcopy.
            if identity in memo:
                return memo[identity]
            out = tuple(children)
            memo[identity] = out
        elif kind in (dict, _ReceiptDict):
            out = {}
            memo[identity] = out
            out.update((key, detach(child)) for key, child in node.items())
        else:
            out = []
            memo[identity] = out
            out.extend(detach(child) for child in node)
        return out

    return detach(value)


class FootprintReceiptMemory:
    """Forward every query to the original memory; never cache across calls."""
    __slots__ = ('_memory',)

    def __init__(self, memory):
        self._memory = memory

    def __getattr__(self, name):
        return getattr(self._memory, name)

    def footprint(self, *args, **kwargs):
        return freeze_footprint(self._memory.footprint(*args, **kwargs))


class FootprintReceiptMap:
    """A selector-only view; bound map methods still operate on the real map."""
    __slots__ = ('_mapper', 'surface')

    def __init__(self, mapper):
        self._mapper = mapper
        self.surface = FootprintReceiptMemory(mapper.surface)

    def __getattr__(self, name):
        return getattr(self._mapper, name)
