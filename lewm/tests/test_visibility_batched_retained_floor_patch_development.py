"""Preserve earliest witnesses, input ownership and failures while skipping empty work."""
from copy import deepcopy
import ast
import inspect
import textwrap
import numpy as np
import pytest
from lewm import batched_retained_floor_patch_development as original
from lewm import visibility_batched_retained_floor_patch_development as candidate
from lewm.tests.test_batched_retained_floor_patch_development import frame, GOOD, BAD, prefix


def paired(frames):
    old=original.BatchedRetainedFloorPatches();new=candidate.VisibilityBatchedRetainedFloorPatches()
    old.frames=new.frames=frames
    return old,new


def test_only_batch_visibility_guard_changes_original_algorithm():
    old=ast.parse(textwrap.dedent(inspect.getsource(original.BatchedRetainedFloorPatches.coverage)))
    new=ast.parse(textwrap.dedent(inspect.getsource(candidate.VisibilityBatchedRetainedFloorPatches.coverage)))
    body=new.body[0].body[3].body
    assert isinstance(body[-2],ast.Assign) and body[-2].targets[0].id=='any_visible'
    body.pop(-2)
    loop=body[-1];guard=loop.body[0]
    assert isinstance(guard,ast.If) and guard.test.value.id=='any_visible'
    assert isinstance(guard.orelse[0],ast.Expr)
    loop.body[0:1]=guard.body
    assert ast.dump(old)==ast.dump(new)
    assert candidate.projected_rectangles is original.projected_rectangles
    assert candidate.record_coverage is original.record_coverage
    assert candidate.VisibilityBatchedRetainedFloorPatches.append is original.BatchedRetainedFloorPatches.append


@pytest.mark.parametrize('count',[0,1,31,32,33,65,130,4096])
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


def test_empty_rows_skip_recorder_but_keep_identical_negative_receipts(monkeypatch):
    old,new=paired([frame(i,yaw=np.pi) for i in range(65)])
    calls=[];record=original.record_coverage
    def counted(*args):calls.append(args[2]['witness']['frame']);return record(*args)
    monkeypatch.setattr(original,'record_coverage',counted)
    monkeypatch.setattr(candidate,'record_coverage',counted)
    expected=old.coverage([[1.,0.]]);assert len(calls)==65
    calls.clear();assert new.coverage([[1.,0.]])==expected and calls==[]


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
