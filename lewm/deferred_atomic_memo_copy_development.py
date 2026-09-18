"""Skip atomic memo lookups until a fallback copier can have changed the memo.

Ordinary container traversal writes only container identities and keepalive
entries. The private memo cannot contain an atomic override until it is passed
to standard deepcopy. Once that happens, retain all original atomic lookups for
the rest of the operation, including keys copied after a custom value copier.
"""
from copy import deepcopy

_ATOMIC = frozenset((type(None),bool,int,float,complex,str,bytes))
_MISSING = object()


def _clone(value,memo,exposed):
    identity=id(value)
    saved=memo.get(identity,_MISSING)
    if saved is not _MISSING: return saved
    kind=type(value)
    if kind in _ATOMIC: return value
    if kind is dict:
        result={};memo[identity]=result
        for key,child in value.items():
            if type(child) in _ATOMIC:
                copied_child=memo.get(id(child),child) if exposed[0] else child
            else: copied_child=_clone(child,memo,exposed)
            # A custom child may just have inserted an override for this key.
            if type(key) in _ATOMIC:
                copied_key=memo.get(id(key),key) if exposed[0] else key
            else: copied_key=_clone(key,memo,exposed)
            result[copied_key]=copied_child
    elif kind is list:
        result=[];memo[identity]=result
        for child in value:
            if type(child) in _ATOMIC:
                result.append(memo.get(id(child),child) if exposed[0] else child)
            else: result.append(_clone(child,memo,exposed))
    elif kind is tuple:
        items=[]
        for child in value:
            if type(child) in _ATOMIC:
                items.append(memo.get(id(child),child) if exposed[0] else child)
            else: items.append(_clone(child,memo,exposed))
        saved=memo.get(identity,_MISSING)
        if saved is not _MISSING: return saved
        result=value if all(a is b for a,b in zip(value,items)) else tuple(items)
    else:
        exposed[0]=True
        return deepcopy(value,memo)
    if result is not value:
        memo[identity]=result
        try: memo[id(memo)].append(value)
        except KeyError: memo[id(memo)]=[value]
    return result


def copy_receipt(value):
    return _clone(value,{},[False])
