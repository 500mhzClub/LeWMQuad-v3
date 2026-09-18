"""Recorded-prefix execution and closed-output audit; launcher admission separate."""
from contextlib import closing
import json
import math
import time

import psutil

from scripts import extended_return_budget_prefix_comparison_development as comparison
from scripts import measured_plane_chained_full_history_inputs_development as inputs
from scripts import replay_go2_measured_plane_single_pass_full_history_v1 as previous

run, native = inputs.run, inputs.native
pipeline = run.pipeline
packets = previous.packets
assigned_model = native.assigned_model
state_digest = previous.original.state_digest
geometry_factory = previous.original.ArticulatedCollisionGeometry
public_mission = previous.original.public_mission
MODEL_SHA = native.worker_inputs.job.MODEL_SHA
URDF = native.worker_inputs.job.URDF
MAX_OUTPUT_BYTES = 2*1024**3
FIXED_STATE_FRAMES = (0, 3, 61, 122, 255, 511, 1023, 2047, 3071, 4002)


def expected_states(count):
    if type(count) is not int or not 1 <= count <= comparison.MAX_PREFIX_OBSERVATIONS:
        raise ValueError('bounded nonempty prospective prefix required')
    return sorted({-1, count-1, *(f for f in FIXED_STATE_FRAMES if f < count)})


def require_admission(admission):
    # This is a shape/binding guard, not independent authentication. The caller
    # must invoke inputs.admit and authenticate the completed timing result.
    required = dict(frames=4014, native_case=native.CASE[0],
        model_state_sha256=MODEL_SHA, native_owner_ended=True,
        original_worker_ended=True, complete_raw_audit_verified=True,
        original_physical_prefix_reconstructed=True,
        original_schedule_terminal='MISSION_TICK_BUDGET_EXHAUSTED')
    if any(type(admission.get(k)) is not type(v) or admission[k] != v for k,v in required.items()):
        raise ValueError('complete original ended 4014-observation budget result required')
    sha = admission.get('learned_result_sha256')
    if type(sha) is not str or len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
        raise ValueError('actual original result SHA-256 required')


def _geometry_hash(geometry):
    return run.fingerprint(comparison.observer_state_tree(geometry))


def _state(controllers, frame, *, require_equal):
    hashes = [run.fingerprint(comparison.observed_state(c)) for c in controllers]
    return dict(frame=frame, baseline_sha256=hashes[0], candidate_sha256=hashes[1],
        observed_state_equal=hashes[0] == hashes[1], equality_required=require_equal)


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def make_report(tracker, rows, states, identities, admission):
    summary = tracker.report()
    count = summary['frames']
    if len(rows) != count or [r['tick'] for r in rows] != list(range(count)):
        raise ValueError('complete consecutive prefix output required')
    if [s['frame'] for s in states] != expected_states(count):
        raise ValueError('initial, fixed and actual stopping state checkpoints required')
    for state in states:
        frame = state['frame']
        required = frame < 0 or rows[frame]['comparison']['complete_normalized_decision_exact']
        equal = state['baseline_sha256'] == state['candidate_sha256']
        if (not _sha(state['baseline_sha256']) or not _sha(state['candidate_sha256'])
                or state['equality_required'] is not required
                or state['observed_state_equal'] is not equal or required and not equal):
            raise ValueError('complete state equality required before decision intervention')
    geometry_sha = _geometry_hash(geometry_factory(URDF))
    expected_ids = dict(initial_model_sha256=[MODEL_SHA]*2, final_model_sha256=[MODEL_SHA]*2,
        initial_geometry_sha256=[geometry_sha]*2, final_geometry_sha256=[geometry_sha]*2,
        independent_model_storage=True, both_models_in_evaluation_mode=True,
        models_without_gradients=True)
    if run.canonical(identities) != run.canonical(expected_ids):
        raise ValueError('original separate unchanged models and articulated geometry required')
    totals = [sum(row['actual_model_forward_calls'][i] for row in rows) for i in range(2)]
    return summary | dict(input_result_sha256=admission['learned_result_sha256'],
        actual_model_forward_calls=totals, observed_state_checks=states, identities=identities,
        complete_recorded_decisions_reproduced=True, reconstructed_public_packets_unchanged=True,
        initial_fixed_and_stopping_states_checked=True,
        complete_saved_output_reconstruction_required=True, input_admission_performed_by_caller=True,
        full_native_history_replayed=False, original_observation_population=admission['frames'],
        baseline_navigation_ticks=4000, candidate_navigation_ticks=8000,
        native_execution=False, real_time_qualified=False, hardware_qualified=False)


def replay(admission, *, output):
    require_admission(admission)
    for name in (pipeline.stream.NAME, 'resource_monitor.jsonl', 'state_checks.json', 'identities.json'):
        if (output/name).exists() or (output/name).is_symlink():
            raise ValueError('exclusive prefix artifacts; no retry, resume or overwrite')
    models = [assigned_model() for _ in range(2)]
    storage = [{v.untyped_storage().data_ptr() for v in m.state_dict().values() if v.numel()} for m in models]
    if (models[0] is models[1] or not storage[0].isdisjoint(storage[1])
            or any(m.training or state_digest(m.state_dict()) != MODEL_SHA
                or any(p.grad is not None for p in m.parameters()) for m in models)):
        raise ValueError('two fresh independent original evaluation models required')
    options = dict(public_mission=public_mission(2), condition=native.CASE[3],
        variant=native.CASE[2], persistent=True)
    controllers = [cls(model, geometry_factory(URDF), navigation_ticks=budget, **options)
        for cls, model, budget in zip((comparison.candidate.MeasuredPlaneChainedSinglePassController,
            comparison.candidate.ExtendedReturnBudgetChainedController), models, (4000,8000), strict=True)]
    identities = dict(initial_model_sha256=[state_digest(m.state_dict()) for m in models],
        initial_geometry_sha256=[_geometry_hash(c.geometry) for c in controllers], independent_model_storage=True)
    calls = [0,0]
    def record(index):
        def hook(module,args,result): calls[index] += 1
        return hook
    directory = native.OUTPUT/native.CASE[0]
    tape = run.read_json(directory, 'command_tape.json')
    handles = [model.register_forward_hook(record(i)) for i,model in enumerate(models)]
    tracker = comparison.PrefixComparison(); states = []; summaries = []
    try:
        initial = _state(controllers, -1, require_equal=True); states.append(initial)
        if not initial['observed_state_equal']:
            raise ValueError('complete initial state must match after declared budget/type normalization')
        with closing(pipeline.read_rows(directory)) as references, \
                closing(packets(directory, admission['frames'])) as public, \
                pipeline.writer(output) as append, (output/'resource_monitor.jsonl').open('x') as monitor:
            for frame in range(comparison.MAX_PREFIX_OBSERVATIONS):
                reference = next(references); packet, now = next(public)
                previous.comparison.reference_endpoint(reference, tape, frame, admission['frames'])
                before = run.fingerprint(packet); p,d,f,image,aux = packet
                prior_calls = calls.copy(); decisions = []
                for controller in controllers:
                    decision = controller.observe(p,d,f,now_ns=now,auxiliary_rgb=image,auxiliary_depth=aux)
                    decisions.append(json.loads(run.canonical(decision)))
                    if run.fingerprint(packet) != before:
                        raise ValueError('controller changed the reconstructed public packet')
                baseline, extended = decisions
                invoked = [a-b for a,b in zip(calls, prior_calls, strict=True)]
                try:
                    checked = tracker.observe(baseline,extended,reference['decision'],model_calls=invoked)
                except Exception as error:
                    append(dict(tick=frame,baseline=baseline,decision=extended,comparison_failure=repr(error)))
                    raise
                row = dict(tick=frame, observation_index=frame, pre_sample_index=reference['pre_sample_index'],
                    baseline=baseline, decision=extended, comparison=checked,
                    recorded_decision_sha256=run.fingerprint(reference['decision']),
                    public_packet_sha256=before, observation_now_ns=now,
                    public_inputs_unchanged=True, actual_model_forward_calls=invoked)
                append(row)
                # Keep scalar receipts only; complete decisions remain in the stream.
                summaries.append({k:v for k,v in row.items() if k not in ('baseline','decision')})
                if frame in FIXED_STATE_FRAMES or checked['stop']:
                    state = _state(controllers,frame,require_equal=checked['complete_normalized_decision_exact'])
                    states.append(state)
                    if state['equality_required'] and not state['observed_state_equal']:
                        raise ValueError('retained controller state changed before intervention at '+str(frame))
                if frame % 100 == 0 or checked['stop']:
                    memory = psutil.virtual_memory(); disk = psutil.disk_usage(run.BASE)
                    monitor.write(json.dumps(dict(frame=frame, monotonic_s=time.monotonic(),
                        rss_bytes=psutil.Process().memory_info().rss, memory_available_bytes=memory.available,
                        artifact_free_bytes=disk.free))+'\n'); monitor.flush()
                    if memory.available < 8*1024**3 or disk.free < 41*1024**3:
                        raise ValueError('retain original RAM and native artifact reserve')
                    print('EXTENDED_RETURN_PREFIX_FRAME',frame,flush=True)
                if (output/pipeline.stream.NAME).stat().st_size > MAX_OUTPUT_BYTES:
                    raise ValueError('bounded prefix output exceeded 2 GiB')
                if checked['stop']: break
    finally:
        for handle in handles: handle.remove()
        run.write_json(output/'state_checks.json',states)
    identities.update(final_model_sha256=[state_digest(m.state_dict()) for m in models],
        final_geometry_sha256=[_geometry_hash(c.geometry) for c in controllers],
        both_models_in_evaluation_mode=all(not m.training for m in models),
        models_without_gradients=all(p.grad is None for m in models for p in m.parameters()))
    run.write_json(output/'identities.json',identities)
    return make_report(tracker,summaries,states,identities,admission)


def check_output(report, admission, *, output):
    """Reconstruct every saved receipt and packet through the actual stop only."""
    require_admission(admission)
    directory = native.OUTPUT/native.CASE[0]
    tape = run.read_json(directory, 'command_tape.json')
    tracker = comparison.PrefixComparison(); summaries = []
    with closing(pipeline.read_rows(output)) as saved, closing(pipeline.read_rows(directory)) as references, \
            closing(packets(directory,admission['frames'])) as public:
        for row in saved:
            if tracker.stopped or len(summaries) >= comparison.MAX_PREFIX_OBSERVATIONS:
                raise ValueError('no saved row or public packet after the intervention boundary')
            frame = len(summaries); reference = next(references); packet, now = next(public)
            previous.comparison.reference_endpoint(reference,tape,frame,admission['frames'])
            if (type(row['observation_index']) is not int or row['observation_index'] != frame
                    or row['pre_sample_index'] != reference['pre_sample_index']
                    or row['public_packet_sha256'] != run.fingerprint(packet)
                    or row['observation_now_ns'] != now or row['public_inputs_unchanged'] is not True
                    or row['recorded_decision_sha256'] != run.fingerprint(reference['decision'])):
                raise ValueError('each saved row must bind the actual original decision and public packet')
            checked = tracker.observe(row['baseline'],row['decision'],reference['decision'],
                model_calls=row['actual_model_forward_calls'])
            if run.canonical(checked) != run.canonical(row['comparison']):
                raise ValueError('complete saved comparison must reconstruct')
            summaries.append({k:v for k,v in row.items() if k not in ('baseline','decision')})
    expected = make_report(tracker,summaries,run.read_json(output,'state_checks.json'),
        run.read_json(output,'identities.json'),admission)
    if run.canonical(expected) != run.canonical(report):
        raise ValueError('complete prefix report must reconstruct')
    if (output/pipeline.stream.NAME).stat().st_size > MAX_OUTPUT_BYTES:
        raise ValueError('bounded closed prefix output exceeded 2 GiB')
    with (output/'resource_monitor.jsonl').open() as stream:
        resources = [json.loads(line) for line in stream]
    required_frames = sorted({tracker.frames-1, *range(0,tracker.frames,100)})
    if [r['frame'] for r in resources] != required_frames:
        raise ValueError('every fixed and actual stopping resource sample required')
    prior = -1.
    for row in resources:
        if (any(type(row[k]) is not int or row[k] <= 0
                for k in ('rss_bytes','memory_available_bytes','artifact_free_bytes'))
                or row['memory_available_bytes'] < 8*1024**3 or row['artifact_free_bytes'] < 41*1024**3
                or type(row['monotonic_s']) not in (int,float) or not math.isfinite(row['monotonic_s'])
                or row['monotonic_s'] < prior):
            raise ValueError('finite ordered resource samples preserving RAM and disk reserve required')
        prior = row['monotonic_s']
