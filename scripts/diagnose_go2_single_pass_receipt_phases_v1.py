"""Locate remaining controller costs after the verified combined optimization."""
from itertools import islice
import hashlib
import json
import time
import cv2
import numpy as np
import torch
from lewm.single_pass_receipt_phase_timing_development import (
    PhaseTiming, PhaseTimedSinglePassReceiptController, model_forward_timing)
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.independent_floor_transport_study_development import MODEL_STATE
from lewm.novel_maze_round_trip_scene_development import public_mission
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS, RESERVE_BYTES
from lewm.intent_return_rgbd_replay_development import IntentReturnRGBDReplay
from lewm.pulse_timed_training_runner_development import state_digest
from scripts.benchmark_go2_single_pass_receipt_copied_v1 import (
    OUTPUT as BENCHMARK, INPUT, CASE, FRAMES, verify_inputs, verify_predecessors)
from scripts.analyze_go2_ground_plane_development_v1 import URDF
from scripts.novel_maze_auxiliary_rgb_packet_development import packet, public_acquisition
from scripts.maze_decision_stream_development import read_rows
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json
from scripts.training_translation_bias_model_admission_development import load_assigned
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

OUTPUT = BASE/'go2_single_pass_receipt_phases_v1_attempt_001'
PROTOCOL = 'docs/go2_single_pass_receipt_phases_v1_2026-09-09.md'
BENCHMARK_SHA = 'd688f2ed9d30177d2e55fb98e9c9f25d2b035f2258d86449ac8e86615cd13c72'
ALLOWANCE = 128*1024**2


def admit(result):
    if (result['status'] != 'SINGLE_PASS_RECEIPT_COPIED_BENCHMARK_V1_COMPLETE'
            or result['native_execution'] is not False or result['model_training'] is not False):
        raise ValueError('completed nonexecuting combined benchmark required')
    expected = dict(frames=FRAMES, complete_original_and_candidate_decisions_exact=True,
        public_inputs_unchanged=True, both_model_states_unchanged=True, model_state_sha256=MODEL_STATE,
        native_execution=False, model_training=False)
    for key, value in expected.items():
        actual = result['report'][key]
        if actual != value or type(actual) is not type(value):
            raise ValueError('full verified combined controller episode required: '+key)


def verify_all(launch):
    verify_inputs(launch)
    verify_artifacts(BENCHMARK, launch['combined_benchmark_artifact_sha256'])
    verify_predecessors()


def timed_step(controller, timing, inputs, original, *, frame):
    p, d, f, auxiliary, image, now = inputs
    if (original['tick'] != frame or original['observation_index'] != frame
            or original['pre_sample_index'] != 749+50*frame or now != 1_500_000_000+frame*100_000_000):
        raise ValueError('exact ordered original observation required')
    before = fingerprint(inputs); timing.reset()
    decision = controller.observe(p, d, f, now_ns=now, auxiliary_depth=auxiliary, auxiliary_rgb=image)
    phases = timing.snapshot(); total = phases['controller.observe']['inclusive_ns']
    if sum(row['exclusive_ns'] for row in phases.values()) != total:
        raise ValueError('exclusive phases must partition the complete controller duration')
    if fingerprint(inputs) != before: raise ValueError('phase instrumentation mutated public inputs')
    normalized = json.loads(json.dumps(decision, allow_nan=False))
    if normalized != original['decision']:
        raise ValueError('complete instrumented decision differs at frame '+str(frame))
    payload = json.dumps(normalized, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(frame=frame, phases=phases, controller_wall_ms=total/1e6,
        complete_decision_exact=True, public_inputs_unchanged=True,
        decision_sha256=hashlib.sha256(payload).hexdigest(), warmup=frame < 3,
        terminal=decision['terminal'])


def aggregate(rows):
    active = [r for r in rows if not r['warmup'] and r['terminal'] is None]
    if not active: raise ValueError('active controller observations required')
    labels = sorted({label for row in active for label in row['phases']})
    return dict(active_observations=len(active), controller_median_ms=float(np.median(
        [r['controller_wall_ms'] for r in active])),
        controller_over_100ms=sum(r['controller_wall_ms'] > 100 for r in active),
        phases={label:dict(
            calls=sum(r['phases'].get(label, {}).get('calls', 0) for r in active),
            mean_exclusive_ms=float(np.mean([r['phases'].get(label, {}).get('exclusive_ns', 0)/1e6 for r in active])),
            mean_inclusive_ms=float(np.mean([r['phases'].get(label, {}).get('inclusive_ns', 0)/1e6 for r in active])))
            for label in labels},
        instrumentation_overhead_included=True, controlled_speed_comparison=False,
        acquisition_and_receipt_io_timed=False, real_time_qualified=False)


def replay(launch):
    model, c, v = load_assigned(launch['correction_admission'], CASE[4])
    if (c, v) != (CASE[3], CASE[2]) or state_digest(model.state_dict()) != MODEL_STATE:
        raise ValueError('same assigned original model required')
    timing = PhaseTiming()
    controller = PhaseTimedSinglePassReceiptController(model, ArticulatedCollisionGeometry(URDF), timing=timing,
        public_mission=public_mission(2), navigation_ticks=NAVIGATION_TICKS, persistent=True, condition=c, variant=v)
    directory = INPUT/CASE[0]; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    if len(reader.frames) != FRAMES or len(acquisitions) != FRAMES or len(tape) != FRAMES-1:
        raise ValueError('complete fixed original episode population required')
    rows = []
    with model_forward_timing(model, timing), (OUTPUT/'phase_timings.jsonl').open('x') as stream, (
            OUTPUT/'resource_monitor.jsonl').open('x') as monitor:
        for i, original in enumerate(islice(read_rows(directory), FRAMES)):
            if i < len(tape):
                if tape[i]['requested_command'] != original['decision']['requested_command'] or not tape[i]['completed']:
                    raise ValueError('original completed actual command required')
            elif original['decision']['terminal'] is None:
                raise ValueError('uncommanded final observation must be terminal')
            p, d, f, now = reader.packet(i)
            image, auxiliary = packet(directory, i, p, public_acquisition(acquisitions[i]), now_ns=now)
            row = timed_step(controller, timing, (p, d, f, auxiliary, image, now), original, frame=i)
            rows.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
            if i%64 == 0:
                resources = hardware(); monitor.write(json.dumps(dict(frame=i, **resources))+'\n'); monitor.flush()
                if resources['artifact_free_bytes'] < RESERVE_BYTES+ALLOWANCE:
                    raise ValueError('phase diagnosis storage reserve unavailable')
                print('SINGLE_PASS_RECEIPT_PHASE_FRAME', i, flush=True)
    if len(rows) != FRAMES: raise ValueError('all original observations required; truncated replay rejected')
    if state_digest(model.state_dict()) != MODEL_STATE or any(p.grad is not None for p in model.parameters()):
        raise ValueError('unchanged model and absent gradients required')
    if model._forward_hooks or model._forward_pre_hooks: raise ValueError('timing hooks must be removed')
    return dict(frames=len(rows), complete_decisions_exact=True, public_inputs_unchanged=True,
        model_state_unchanged=True, model_state_sha256=MODEL_STATE, model_hooks_removed=True,
        exclusive_time_partitions_controller_duration=True, phase_summary=aggregate(rows),
        native_execution=False, model_training=False)


def main():
    if not __debug__: raise ValueError('assertions required')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive optimized controller diagnosis required')
    verify_artifacts(BENCHMARK, {'result.json':BENCHMARK_SHA}); result = read_json(BENCHMARK, 'result.json'); admit(result)
    ids = dict(result['artifact_sha256']); ids['result.json'] = BENCHMARK_SHA
    verify_artifacts(BENCHMARK, ids); old = read_json(BENCHMARK, 'launch.json')
    sources = discover_sources((PROTOCOL, 'scripts/diagnose_go2_single_pass_receipt_phases_v1.py',
        'lewm/tests/test_single_pass_receipt_phase_timing_development.py'), result['source_sha256'])
    launch = old|dict(protocol=PROTOCOL, output_root=str(OUTPUT), source_sha256=sources,
        combined_benchmark_artifact_sha256=ids, implementation_class='PhaseTimedSinglePassReceiptController',
        controller_variants=['single_pass_receipt_copied'], native_execution=False, native_scene_workers=0,
        model_loaded=True, model_training=False, replay_workers=1, numerical_threads=1,
        separate_model_and_controller_per_variant=False, order='original observation order',
        minimum_available_ram_bytes=8*1024**3, output_allowance_bytes=ALLOWANCE,
        instrumentation_overhead_included=True, controlled_speed_comparison=False,
        concurrency_reason='one CPU timing replay may overlap one native scene and one hold replay with measured headroom')
    verify_all(launch); resources = hardware(); launch['hardware'] = resources
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < RESERVE_BYTES+ALLOWANCE:
        raise ValueError('bounded optimized phase diagnosis resources unavailable')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch); started = time.perf_counter()
    print('SINGLE_PASS_RECEIPT_PHASE_LAUNCHED', digest(OUTPUT/'launch.json'), flush=True)
    try:
        report = replay(launch); verify_all(launch)
        bindings = {n:digest(OUTPUT/n) for n in ('launch.json', 'phase_timings.jsonl', 'resource_monitor.jsonl')}
        verify_artifacts(OUTPUT, bindings)
        write_json(OUTPUT/'result.json', dict(status='SINGLE_PASS_RECEIPT_PHASE_V1_COMPLETE', report=report,
            source_sha256=sources, artifact_sha256=bindings, combined_benchmark_result_sha256=BENCHMARK_SHA,
            wall_s=time.perf_counter()-started, hardware_after=hardware(), native_execution=False,
            model_training=False, instrumentation_overhead_included=True, controlled_speed_comparison=False,
            real_time_qualified=False, navigation_qualified=False, goal_achieved=False))
        print('SINGLE_PASS_RECEIPT_PHASE_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_SINGLE_PASS_RECEIPT_PHASE_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
