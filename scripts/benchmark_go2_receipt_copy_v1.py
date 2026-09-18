"""Paired component-copy timings on fixed immutable decision receipts."""
from copy import deepcopy
import json
import statistics
import time
from lewm.receipt_copy_development import copy_receipt
from lewm.novel_maze_round_trip_contract_development import RESERVE_BYTES
from scripts.maze_decision_stream_development import read_rows
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch as verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

INPUT = BASE/'go2_joint_floor_registered_maze_prefix_v1_attempt_001'
OUTPUT = BASE/'go2_receipt_copy_benchmark_v1_attempt_001'
RESULT = 'c1a2c347d5aea7a1e0db352a451daf4add20c9c42905886712778cbdd53166ea'
PROTOCOL = 'docs/go2_receipt_copy_benchmark_v1_2026-09-09.md'
FRAMES = (20, 60, 100, 959)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive component benchmark required')
    verify_artifacts(INPUT, {'result.json': RESULT})
    result = read_json(INPUT, 'result.json')
    assert result['status'] == 'JOINT_FLOOR_REGISTERED_MAZE_PREFIX_COMPLETE'
    bindings = {'result.json': RESULT, **result['artifact_sha256']}
    verify_artifacts(INPUT, bindings)
    old = read_json(INPUT, 'launch.json')
    sources = discover_sources((PROTOCOL, 'scripts/benchmark_go2_receipt_copy_v1.py',
        'lewm/tests/test_receipt_copy_development.py'), result['source_sha256'])
    resources = hardware()
    if resources['memory_available_bytes'] < 4*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+128*1024**2:
        raise ValueError('component benchmark resource envelope unavailable')
    launch = old | dict(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        input_artifact_sha256=bindings, frames=list(FRAMES), repeats=8, hardware=resources,
        cpu_processes=1, numerical_threads=1, native_execution=False, model_training=False,
        model_loaded=False, controller_modified=False, controlled_full_loop_comparison=False)
    verify(launch); create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('RECEIPT_COPY_BENCHMARK_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        records=[]
        for row in read_rows(INPUT):
            frame = row['tick']
            if frame not in FRAMES: continue
            value = row['decision']['new_selection']; expected = encode(value)
            assert isinstance(value, dict) and 'prediction' in value
            functions = {'deepcopy': deepcopy, 'copy_receipt': copy_receipt}
            for fn in functions.values(): assert encode(fn(value)) == expected
            timings = {name:[] for name in functions}
            for repeat in range(8):
                order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
                for name in order:
                    begin = time.perf_counter_ns(); copied = functions[name](value)
                    elapsed = (time.perf_counter_ns()-begin)/1e6
                    assert encode(copied) == expected
                    timings[name].append(elapsed)
                    del copied
            medians = {name:statistics.median(values) for name,values in timings.items()}
            records.append(dict(frame=frame, serialized_bytes=len(expected), timings_ms=timings,
                medians_ms=medians, component_speed_ratio=medians['deepcopy']/medians['copy_receipt'],
                copied_values_exact=True))
            print('RECEIPT_COPY_BENCHMARK_FRAME', frame, medians, flush=True)
        assert len(records) == len(FRAMES)
        verify(launch); verify_artifacts(INPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='RECEIPT_COPY_BENCHMARK_COMPLETE',
            launch_sha256=digest(OUTPUT/'launch.json'), input_result_sha256=RESULT,
            source_sha256=sources, records=records, hardware_after=hardware(),
            controller_modified=False, native_execution=False, model_training=False,
            live_alias_distribution_tested=False, controlled_full_loop_comparison=False,
            real_time_qualified=False, navigation_qualified=False, goal_achieved=False))
        print('RECEIPT_COPY_BENCHMARK_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='RECEIPT_COPY_BENCHMARK_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
