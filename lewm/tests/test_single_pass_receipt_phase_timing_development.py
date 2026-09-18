"""Instrument real sensor paths, preserve decisions, and reject partial evidence."""
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from lewm.single_pass_receipt_phase_timing_development import PhaseTiming, PhaseTimedSinglePassReceiptController
from lewm.single_pass_receipt_copied_controller_development import SinglePassReceiptCopiedController
from lewm.single_pass_sample_bounds_development import SinglePassMeasuredSampleBoundsIndex
from lewm.tests.test_single_pass_receipt_copied_controller_development import indices
from lewm.tests.test_single_pass_later_floor_controller_development import kwargs
from lewm.tests.test_frame_floor_cache_development import equal
from scripts import diagnose_go2_single_pass_receipt_phases_v1 as runner


def test_actual_dual_camera_mapping_and_fail_closed_stop_are_exact(monkeypatch):
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
    from lewm.fast_gyro_development import FastGyroBuffer
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    timing = PhaseTiming()
    old = SinglePassReceiptCopiedController(None, None, **kwargs())
    new = PhaseTimedSinglePassReceiptController(None, None, timing=timing, **kwargs())
    assert new.memory is new.mapper.surface and new.residual is new.selector.residual
    values = indices(new.memory)
    assert len(values) == len({id(v) for v in values}) == 8
    assert all(type(v) is SinglePassMeasuredSampleBoundsIndex and not v.cells for v in values)
    p, d, a, _, now = packets()
    image = from_captured_rgb(p['image']['rgb'], a, p, measured_ns=now, available_ns=now, now_ns=now)
    gyro = FastGyroBuffer((0, 0, 0))
    for t in range(now-100_000_000, now+1, 2_000_000):
        gyro.append(np.zeros(3), np.ones(3, bool), measured_ns=t, available_ns=t)
    f = gyro.packet(now_ns=now)
    for image_value, stamp in ((image, now), (None, now+100_000_000)):
        rows = []; timing.reset()
        for controller in (old, new):
            pp, dd, ff, aa, ii = deepcopy((p, d, f, a, image_value))
            rows.append(controller.observe(pp, dd, ff, auxiliary_rgb=ii, auxiliary_depth=aa, now_ns=stamp))
        equal(*rows)
        values = timing.snapshot()
        assert sum(v['exclusive_ns'] for v in values.values()) == values['controller.observe']['inclusive_ns']
        if image_value is not None:
            assert rows[0]['terminal'] is None
            assert values['map.observe']['calls'] == values['motion.observe']['calls'] == 1
            assert values['memory.primary_insert']['calls'] == 1
        else:
            assert rows[0]['terminal'] == 'SENSOR_OR_MODEL_FAILURE'
            assert rows[0]['requested_command'] == [0., 0., 0.]


@pytest.mark.parametrize('fault', [None, 'decision', 'mutation', 'endpoint', 'timestamp', 'partition'])
def test_step_checks_inputs_complete_decision_and_exclusive_partition(fault):
    timing = PhaseTiming(); decision = {'terminal':None, 'requested_command':[0., 0., 0.], 'memory':{'cells':7}}
    original = dict(tick=3, observation_index=3, pre_sample_index=899, decision=deepcopy(decision))
    inputs = ({}, {}, {}, {}, {}, 1_800_000_000)
    if fault == 'endpoint': original['pre_sample_index'] += 1
    elif fault == 'timestamp': inputs = (*inputs[:-1], 1_800_000_001)
    class Controller:
        def observe(self, policy, *a, **k):
            with timing.scope('controller.observe'):
                with timing.scope('selector.choose'):
                    if fault == 'decision': decision['memory']['cells'] += 1
                    if fault == 'mutation': policy['changed'] = True
            if fault == 'partition': timing.values['selector.choose']['exclusive_ns'] += 1
            return decision
    if fault:
        with pytest.raises(ValueError): runner.timed_step(Controller(), timing, inputs, original, frame=3)
    else:
        r = runner.timed_step(Controller(), timing, inputs, original, frame=3)
        assert r['complete_decision_exact'] and r['public_inputs_unchanged'] and not r['warmup']


@pytest.mark.parametrize('fault', [None, 'short', 'mismatch', 'command', 'model', 'population', 'nonterminal_last'])
def test_full_replay_bounds_cleanup_and_rejection(monkeypatch, tmp_path, fault):
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path); monkeypatch.setattr(runner, 'FRAMES', 6)
    model = torch.nn.Identity(); state = {'value':runner.MODEL_STATE}
    monkeypatch.setattr(runner, 'load_assigned', lambda *a:(model, runner.CASE[3], runner.CASE[2]))
    monkeypatch.setattr(runner, 'state_digest', lambda *a:state['value'])
    monkeypatch.setattr(runner, 'ArticulatedCollisionGeometry', lambda *a:None)
    monkeypatch.setattr(runner, 'hardware', lambda:dict(artifact_free_bytes=2**40))
    reads = []; packets = []
    def decision(i):
        return dict(terminal='BUDGET' if i == 5 and fault != 'nonterminal_last' else None,
            requested_command=[0., 0., 0.], memory={'cells':i})
    def rows(*a):
        for i in range(6):
            if fault == 'short' and i == 4: return
            if fault in ('mismatch', 'command') and i > 3: pytest.fail('consumed future decision after rejection')
            reads.append(i)
            yield dict(tick=i, observation_index=i, pre_sample_index=749+50*i, decision=decision(i))
        pytest.fail('consumed beyond fixed population')
    monkeypatch.setattr(runner, 'read_rows', rows)
    class Reader:
        frames = list(range(5 if fault == 'population' else 6))
        def packet(self, i):
            packets.append(i); return {'tick':i}, {}, {}, 1_500_000_000+i*100_000_000
    monkeypatch.setattr(runner, 'IntentReturnRGBDReplay', lambda *a:Reader())
    tape = [dict(requested_command=[0., 0., 0.], completed=True) for _ in range(5)]
    if fault == 'command': tape[3]['completed'] = False
    monkeypatch.setattr(runner, 'read_json', lambda p, n:tape if n == 'command_tape.json' else [{}]*6)
    monkeypatch.setattr(runner, 'public_acquisition', lambda row:row)
    monkeypatch.setattr(runner, 'packet', lambda *a, **k:({}, {}))
    class Controller:
        def __init__(self, *a, timing, **k): self.timing = timing
        def observe(self, policy, *a, **k):
            i = policy['tick']; d = decision(i)
            with self.timing.scope('controller.observe'):
                model(torch.ones(1))
                if i == 3:
                    if fault == 'mismatch': d['memory']['cells'] += 1
                    if fault == 'model': state['value'] = 'changed'
            return d
    monkeypatch.setattr(runner, 'PhaseTimedSinglePassReceiptController', Controller)
    if fault:
        with pytest.raises(ValueError): runner.replay(dict(correction_admission={}))
    else:
        report = runner.replay(dict(correction_admission={}))
        assert report['frames'] == 6 and packets == reads == list(range(6))
        assert report['phase_summary']['active_observations'] == 2
        assert report['model_hooks_removed'] and report['exclusive_time_partitions_controller_duration']
    assert not model._forward_hooks and not model._forward_pre_hooks


@pytest.mark.parametrize('fault', [None, 'frames', 'model', 'equality', 'native', 'boolean_count'])
def test_completed_combined_evidence_is_required(fault):
    result = dict(status='SINGLE_PASS_RECEIPT_COPIED_BENCHMARK_V1_COMPLETE', native_execution=False,
        model_training=False, report=dict(frames=514, complete_original_and_candidate_decisions_exact=True,
            public_inputs_unchanged=True, both_model_states_unchanged=True, model_state_sha256=runner.MODEL_STATE,
            native_execution=False, model_training=False))
    if fault == 'frames': result['report']['frames'] = 513
    elif fault == 'model': result['report']['model_state_sha256'] = '0'*64
    elif fault == 'equality': result['report']['complete_original_and_candidate_decisions_exact'] = False
    elif fault == 'native': result['native_execution'] = True
    elif fault == 'boolean_count': result['report']['public_inputs_unchanged'] = 1
    if fault:
        with pytest.raises(ValueError): runner.admit(result)
    else: runner.admit(result)


def test_aggregation_excludes_warmup_and_terminal_without_double_counting():
    def row(frame, value, warmup=False, terminal=None):
        return dict(frame=frame, warmup=warmup, terminal=terminal, controller_wall_ms=value,
            phases={'controller.observe':dict(calls=1, inclusive_ns=value*1e6, exclusive_ns=value*.25*1e6),
                'child':dict(calls=2, inclusive_ns=value*.75*1e6, exclusive_ns=value*.75*1e6)})
    summary = runner.aggregate([row(0, 999, True), row(3, 80), row(4, 120), row(5, 999, terminal='BUDGET')])
    assert summary['active_observations'] == 2 and summary['controller_median_ms'] == 100
    assert summary['controller_over_100ms'] == 1
    assert sum(v['mean_exclusive_ms'] for v in summary['phases'].values()) == 100
    assert summary['phases']['child']['calls'] == 4
    assert summary['instrumentation_overhead_included'] and not summary['controlled_speed_comparison']
