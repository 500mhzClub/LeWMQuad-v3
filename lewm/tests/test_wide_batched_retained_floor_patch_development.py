"""Preserve earliest witnesses, input ownership and failures while skipping empty work."""
from copy import deepcopy
from types import FunctionType
import ast
import inspect
import textwrap
import numpy as np
import pytest
from lewm import visibility_batched_retained_floor_patch_development as original
from lewm import wide_batched_retained_floor_patch_development as candidate
from lewm.tests.test_batched_retained_floor_patch_development import frame, GOOD, BAD, prefix


def paired(frames):
    old=original.VisibilityBatchedRetainedFloorPatches();new=candidate.WideBatchedRetainedFloorPatches()
    old.frames=new.frames=frames
    return old,new


def test_only_private_batch_size_binding_changes():
    old=original.VisibilityBatchedRetainedFloorPatches.coverage
    new=candidate.WideBatchedRetainedFloorPatches.coverage
    assert old.__code__ is new.__code__ and old.__closure__ is new.__closure__
    assert old.__globals__['FRAME_BATCH']==32 and new.__globals__['FRAME_BATCH']==128
    for key in old.__globals__:
        if key!='FRAME_BATCH':assert old.__globals__[key] is new.__globals__[key]
    assert candidate.WideBatchedRetainedFloorPatches.append is original.VisibilityBatchedRetainedFloorPatches.append


@pytest.mark.parametrize('count',[0,1,31,32,33,65,127,128,129,256,4096])
@pytest.mark.parametrize('queries',[0,1,4,128])
def test_batch_boundaries_preserve_complete_results(count,queries):
    frames=[frame(i,GOOD if i==count-1 else BAD,yaw=np.pi if i%5 else 0.) for i in range(count)]
    old,new=paired(frames)
    xy=np.column_stack((np.linspace(.1,4.8,queries),np.linspace(-.2,.2,queries)))
    assert new.coverage(xy)==old.coverage(xy)


@pytest.mark.parametrize('radius',[.001,.022,.1])
def test_rotated_translated_and_pixel_depth_boundaries(radius):
    rng=np.random.default_rng(2026091101)
    frames=[frame(i,GOOD if i%3 else BAD,yaw=rng.uniform(-np.pi,np.pi),p=rng.uniform(-2.,2.,3)) for i in range(65)]
    old,new=paired(frames);xy=rng.uniform(-4.9,4.9,(128,2))
    assert new.coverage(xy,radius)==old.coverage(xy,radius)


def test_fewer_projection_calls_preserve_full_negative_receipts():
    old,new=paired([frame(i,yaw=np.pi) for i in range(257)])
    counts=[];values=[]
    for memory in (old,new):
        calls=[];fn=type(memory).coverage
        project=fn.__globals__['projected_rectangles']
        def counted(frames,*args):
            calls.append(len(frames));return project(frames,*args)
        clone=FunctionType(fn.__code__,fn.__globals__|{'projected_rectangles':counted},fn.__name__,fn.__defaults__)
        values.append(clone(memory,[[1.,0.]]));counts.append(calls)
    assert values[0]==values[1]
    assert counts==[[32]*8+[1],[128,128,1]]


def test_invisible_missing_prefix_still_fails_and_later_unused_one_is_not_read():
    invisible=frame(0,yaw=np.pi);invisible.pop('prefix')
    for memory in paired([invisible]):
        with pytest.raises(KeyError,match='prefix'):memory.coverage([[1.,0.]])
    later=frame(1,yaw=np.pi);later.pop('prefix')
    old,new=paired([frame(0),later]);assert new.coverage([[1.,0.]])==old.coverage([[1.,0.]])


def test_mixed_earliest_witnesses_and_returned_containers_remain_independent():
    left=np.zeros((479,639),bool);left[:,320:]=True
    right=np.zeros((479,639),bool);right[:,:320]=True
    frames=[frame(0,yaw=np.pi),frame(1,prefix(left)),frame(2,BAD),frame(3,prefix(right)),frame(4,GOOD)]
    old,new=paired(frames);xy=[[1.,.1],[1.,-.1],[1.,0.],[0.,0.]]
    expected=old.coverage(xy);actual=new.coverage(xy);assert actual==expected
    assert {r['coverage_witness']['witness']['frame'] for r in actual if r['complete_nominal_foot_patch']}=={1,3,4}
    for row in actual:
        if row['coverage_witness']:row['coverage_witness']['witness']['evidence'].append('changed')
    assert new.coverage(xy)==expected
    assert all(f['witness']['evidence']==['original'] for f in frames)


@pytest.mark.parametrize('fault',['overflow','missing_metadata'])
def test_original_error_fallback_and_early_exit_are_preserved(fault):
    later=frame(1,p=(1e308,1e308,1e308)) if fault=='overflow' else frame(1)
    if fault=='missing_metadata':later.pop('R')
    old,new=paired([frame(0),later])
    with np.errstate(all='raise'):assert new.coverage([[1.,0.]])==old.coverage([[1.,0.]])
    old,new=paired([frame(0,BAD),later])
    with np.errstate(all='raise'):
        with pytest.raises((FloatingPointError,KeyError)) as first:old.coverage([[1.,0.]])
        with pytest.raises(type(first.value)):new.coverage([[1.,0.]])


@pytest.mark.parametrize('xy,radius',[([1.,0.],.022),([[5.,0.]],.022),([[np.nan,0.]],.022),
    ([[1.,0.]],0.),([[1.,0.]],.101),([[1.,0.]],np.inf),(np.zeros((129,2)),.022)])
def test_query_rejections_are_unchanged(xy,radius):
    old,new=paired([frame(0)])
    with pytest.raises(ValueError) as a:old.coverage(xy,radius)
    with pytest.raises(ValueError) as b:new.coverage(xy,radius)
    assert str(a.value)==str(b.value)


@pytest.mark.parametrize('fault',['overflow','missing_metadata','missing_prefix'])
def test_unused_fault_beyond_original_batch_does_not_change_early_success(fault):
    frames=[frame(i,GOOD if i==0 else BAD) for i in range(129)]
    if fault=='overflow':frames[70]['p']=np.full(3,1e308)
    if fault=='missing_metadata':frames[70].pop('R')
    if fault=='missing_prefix':frames[70].pop('prefix')
    old,new=paired(frames)
    with np.errstate(all='raise'):assert new.coverage([[1.,0.]])==old.coverage([[1.,0.]])
