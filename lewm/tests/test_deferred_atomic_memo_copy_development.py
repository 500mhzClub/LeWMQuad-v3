"""Original graph semantics plus transitions from private to exposed memo."""
from copy import deepcopy
from types import FunctionType
import pytest
from lewm.deferred_atomic_memo_copy_development import copy_receipt
from lewm.tests import test_receipt_copy_development as original


for _name,_function in vars(original).items():
    if _name.startswith('test_') and isinstance(_function,FunctionType):
        _clone=FunctionType(_function.__code__,_function.__globals__ | dict(copy_receipt=copy_receipt),
            _name,_function.__defaults__,_function.__closure__)
        _clone.__kwdefaults__=_function.__kwdefaults__
        _clone.__dict__.update(_function.__dict__)
        globals()[_name]=_clone


@pytest.mark.parametrize('replacement',[None,False,0,'replaced'])
def test_custom_atomic_override_takes_effect_only_after_fallback(replacement):
    marker='separately allocated marker '+str(object())
    class Custom:
        def __deepcopy__(self,memo):
            memo[id(marker)]=replacement
            return 'custom'
    source=[marker,{'nested':Custom()},marker,(marker,),{marker:marker}]
    assert copy_receipt(source)==deepcopy(source)
    result=copy_receipt(source)
    assert result[0] is marker and result[2] is replacement
    assert result[4]=={replacement:replacement}


def test_nested_custom_can_replace_an_outer_key_before_its_assignment():
    marker='outer key '+str(object())
    class Custom:
        def __deepcopy__(self,memo):
            memo[id(marker)]='key replacement'
            return {'outer':memo[id(source)]}
    source={marker:[Custom()], 'again':marker}
    result=copy_receipt(source)
    assert result['key replacement'][0]['outer'] is result
    assert result['again']=='key replacement'


def test_ordinary_immutable_leaves_keep_identity_and_mutable_aliases_are_detached():
    leaves=[None,False,True,10**80,float('-0.0'),1j,b'bytes','text']
    shared={'leaves':leaves};source=[shared,shared,tuple(leaves)]
    result=copy_receipt(source)
    assert result[0] is result[1] and result[0] is not shared
    assert all(a is b for a,b in zip(leaves,result[0]['leaves']))
    assert result[2] is source[2]
