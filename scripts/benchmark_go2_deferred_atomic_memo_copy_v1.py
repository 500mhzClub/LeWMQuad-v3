"""Paired component timing on two fixed serialized controller decisions.

Serialization does not retain the original runtime graph's alias structure.
This experiment measures the copier on these exact serialized receipts only;
it is not an end-to-end controller benchmark or a native navigation run.
"""
import gc
import json
import statistics
import time
from lewm.receipt_copy_development import copy_receipt as baseline
from lewm.deferred_atomic_memo_copy_development import copy_receipt as candidate
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import BASE,artifact_path
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,verify,write_json
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

SOURCE='scripts/benchmark_go2_deferred_atomic_memo_copy_v1.py'
TEST='lewm/tests/test_deferred_atomic_memo_copy_development.py'
INPUT_ROOT=BASE/'go2_measured_plane_controller_prefix_v1_attempt_001'
INPUT_SHA='8385e643b776865a44d9271404e8c05a8acc46b37e7ff9bc4b8bf48396e93047'
OUTPUT=ROOT/'docs/go2_deferred_atomic_memo_copy_component_benchmark_2026-09-11.json'
FAILURE=ROOT/'docs/go2_deferred_atomic_memo_copy_component_benchmark_failure_2026-09-11.json'
ROUNDS=7
COPIES=10


def main():
    if OUTPUT.exists() or OUTPUT.is_symlink() or FAILURE.exists() or FAILURE.is_symlink():
        raise ValueError('exclusive component benchmark output required; no retry')
    sources=discover_sources((SOURCE,TEST),{});verify(sources)
    path=artifact_path(INPUT_ROOT,'result.json')
    if digest(path) != INPUT_SHA: raise ValueError('fixed completed controller result required')
    result=json.loads(path.read_text())
    if result['status'] != 'MEASURED_PLANE_CONTROLLER_PREFIX_V1_COMPLETE': raise ValueError('completed input required')
    hw=hardware()
    if hw['memory_available_bytes'] < 4*1024**3: raise ValueError('4 GiB available RAM required')
    payloads={k:result['report'][k] for k in ('boundary_original','boundary_candidate')}
    functions={'baseline':baseline,'candidate':candidate}
    gc_was_enabled=gc.isenabled()
    try:
        outcomes={}
        for name,value in payloads.items():
            before=json.dumps(value,sort_keys=True,allow_nan=False)
            for fn in functions.values():
                copied=fn(value)
                if copied != value or copied is value: raise ValueError('complete detached equal copy required')
                if json.dumps(copied,sort_keys=True,allow_nan=False) != before: raise ValueError('all serialized evidence must agree')
                del copied
                fn(value)
            gc.collect();gc.disable()
            rounds=[]
            for index in range(ROUNDS):
                order=('baseline','candidate') if index%2 == 0 else ('candidate','baseline')
                row=dict(round=index,order=list(order),copies=COPIES)
                for arm in order:
                    fn=functions[arm]
                    start=time.perf_counter_ns()
                    for _ in range(COPIES):
                        copied=fn(value)
                        del copied
                    row[arm+'_ns']=time.perf_counter_ns()-start
                rounds.append(row)
            if gc_was_enabled: gc.enable()
            if json.dumps(value,sort_keys=True,allow_nan=False) != before: raise ValueError('source evidence mutated')
            medians={arm:statistics.median(r[arm+'_ns']/COPIES/1e6 for r in rounds) for arm in functions}
            outcomes[name]=dict(serialized_bytes=len(before.encode()),rounds=rounds,median_copy_ms=medians,
                paired_median_reduction_percent=100*(1-medians['candidate']/medians['baseline']),
                complete_serialized_evidence_equal=True,source_unchanged=True)
        verify(sources)
        if digest(path) != INPUT_SHA: raise ValueError('input identity changed')
        report=dict(status='DEFERRED_ATOMIC_MEMO_COPY_COMPONENT_BENCHMARK_COMPLETE',source_sha256=sources,
            input_result_sha256=INPUT_SHA,hardware=hw,rounds=ROUNDS,copies_per_round=COPIES,
            alternating_arm_order=True,gc_disabled_during_timing=True,outcomes=outcomes,
            serialized_runtime_aliases_not_preserved=True,actual_controller_replay=False,
            native_execution=False,real_time_qualified=False,navigation_qualified=False,goal_achieved=False)
        write_json(OUTPUT,report)
        print('DEFERRED_MEMO_COMPONENT_COMPLETE',digest(OUTPUT),
            {k:v['median_copy_ms'] for k,v in outcomes.items()},flush=True)
    except BaseException as error:
        write_json(FAILURE,dict(status='TERMINAL_DEFERRED_MEMO_COMPONENT_BENCHMARK_FAILURE',reason=repr(error),automatic_retry=False))
        raise
    finally:
        if gc_was_enabled: gc.enable()
        else: gc.disable()


if __name__ == '__main__':main()
