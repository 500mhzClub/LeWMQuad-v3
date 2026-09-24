"""Bounded paired microbenchmark; no model, native scene or navigation claim."""
from datetime import datetime,timezone
from types import FunctionType
import time
import json
import os
import numpy as np
from lewm import visibility_batched_retained_floor_patch_development as original
from lewm import progressive_batched_retained_floor_patch_development as candidate
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,verify,write_json,digest
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

SOURCE='scripts/benchmark_go2_progressive_batched_retained_patch_v1.py'
TEST='lewm/tests/test_progressive_batched_retained_floor_patch_development.py'
OUTPUT=ROOT/'docs/go2_progressive_batched_retained_patch_microbenchmark_2026-09-11.json'
PROFILE_WITNESS='docs/go2_receipt_copied_profile_completion_verification_2026-09-11.json'
PROFILE_WITNESS_SHA='b94eb7781acd5a45a9ea93329e73f513a1d20fd27630f5497f9454c03d5a0119'
PREDECESSOR='docs/go2_wide_batched_retained_patch_microbenchmark_2026-09-11.json'
PREDECESSOR_SHA='5ad2f7f8c9a8deb894203e4763a0ea51dae3aaebeac9b24293d89c6160b40e2e'
REPEATS=30
FRAMES=1428
QUERIES=np.asarray([[1.,-.1],[1.,0.],[1.,.1],[1.2,0.]])


def memories(pattern):
    good=np.zeros((480,640),np.int32)
    bad=np.zeros((480,640),np.int32)
    bad[1:,1:]=np.ones((479,639),np.int32).cumsum(0,dtype=np.int32).cumsum(1,dtype=np.int32)
    good.flags.writeable=False;bad.flags.writeable=False
    frames=[]
    for i in range(FRAMES):
        visible=(pattern in ('visible_uncovered','immediate_witness')
            or pattern=='sparse_visibility' and (i%32==0 or i==FRAMES-1))
        yaw=0. if visible else np.pi;c,s=np.cos(yaw),np.sin(yaw)
        frames.append(dict(R=np.asarray([[c,-s,0.],[s,c,0.],[0.,0.,1.]]),p=np.zeros(3),floor_height=-.32,
            prefix=good if pattern=='immediate_witness' or pattern=='sparse_visibility' and i==FRAMES-1 else bad,
            witness=dict(frame=i,measured_ns=i*100_000_000,evidence=['synthetic'])))
    old=original.VisibilityBatchedRetainedFloorPatches();new=candidate.ProgressiveBatchedRetainedFloorPatches()
    old.frames=new.frames=frames
    return old,new


def counted(memory):
    counts=dict(recorder_calls=0,empty_visibility_rows=0,nonempty_visibility_rows=0,projection_calls=0,projected_frame_rows=0)
    def record(*args):
        counts['recorder_calls']+=1
        counts['nonempty_visibility_rows' if args[5].any() else 'empty_visibility_rows']+=1
        return original.record_coverage(*args)
    fn=type(memory).coverage
    original_projection=fn.__globals__['projected_rectangles']
    def projected(frames,*args):
        counts['projection_calls']+=1;counts['projected_frame_rows']+=len(frames)
        return original_projection(frames,*args)
    clone=FunctionType(fn.__code__,fn.__globals__|{'record_coverage':record,'projected_rectangles':projected},fn.__name__,fn.__defaults__,fn.__closure__)
    clone.__kwdefaults__=fn.__kwdefaults__
    result=clone(memory,QUERIES)
    return counts,result


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink():raise ValueError('exclusive microbenchmark; do not overwrite')
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):
        raise ValueError('fixed single-thread CPU environment required')
    verify({PROFILE_WITNESS:PROFILE_WITNESS_SHA,PREDECESSOR:PREDECESSOR_SHA})
    inherited=json.loads((ROOT/PREDECESSOR).read_text())['source_sha256']
    sources=discover_sources((SOURCE,TEST,PROFILE_WITNESS,PREDECESSOR),inherited);verify(sources)
    resources=hardware();results=[]
    for pattern in ('invisible','sparse_visibility','visible_uncovered','immediate_witness'):
        old,new=memories(pattern)
        a,expected=counted(old);b,actual=counted(new)
        if actual!=expected:raise ValueError('complete synthetic coverage receipts differ')
        old.coverage(QUERIES);new.coverage(QUERIES)
        times=[[],[]]
        for repeat in range(REPEATS):
            for index in ((0,1) if repeat%2==0 else (1,0)):
                started=time.perf_counter();value=(old,new)[index].coverage(QUERIES)
                times[index].append(time.perf_counter()-started)
                if value!=expected:raise ValueError('paired complete receipts changed')
        results.append(dict(pattern=pattern,frames=FRAMES,queries=QUERIES.tolist(),repeats=REPEATS,
            alternating_execution_order=True,baseline_counts=a,candidate_counts=b,complete_results_equal=True,
            baseline_seconds=times[0],candidate_seconds=times[1],
            baseline_median_ms=1000*float(np.median(times[0])),candidate_median_ms=1000*float(np.median(times[1])),
            total_time_reduction_percent=100*(1-sum(times[1])/sum(times[0]))))
    verify(sources)
    write_json(OUTPUT,dict(status='PROGRESSIVE_BATCHED_RETAINED_PATCH_MICROBENCHMARK_COMPLETE',
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=sources,source_count=len(sources),hardware=resources,
        cases=results,baseline_frame_batch=32,candidate_first_frame_batch=32,candidate_remaining_frame_batch=128,predecessor_sha256=PREDECESSOR_SHA,profile_witness_sha256=PROFILE_WITNESS_SHA,synthetic_history=True,shared_read_only_prefix_arrays=True,isolated_benchmark=False,
        instrumentation_outside_timing=True,native_execution=False,model_inference=False,
        complete_controller_replay=False,real_time_qualified=False,navigation_qualified=False,goal_achieved=False))
    print('PROGRESSIVE_PATCH_MICROBENCHMARK_COMPLETE',digest(OUTPUT),len(sources),flush=True)
    for row in results:print(row['pattern'],row['baseline_median_ms'],row['candidate_median_ms'],row['total_time_reduction_percent'],flush=True)


if __name__=='__main__':main()
