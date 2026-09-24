"""Prospective replay must not consume a changed command's recorded future."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import replay_go2_supervised_commitment_contact_prefix_v1 as runner
from lewm.tests.test_commitment_contact_development import decisions


@pytest.mark.parametrize('fault', [None, 'public_mutation', 'model_changed', 'incomplete_command'])
def test_real_comparator_stops_reader_before_changed_command_future(monkeypatch, tmp_path, fault):
    source = tmp_path/'input'; output = tmp_path/'output'; output.mkdir(); (output/runner.DECISIONS).write_bytes(b'')
    monkeypatch.setattr(runner, 'INPUT', source); monkeypatch.setattr(runner, 'OUTPUT', output)
    packets = []; stored = []; checks = []; records = []; current = []
    for i in range(3):
        old, new = decisions(i)
        for value in (old, new): value['original_visual_evidence'] = {'status':'CURRENT_VISUAL_POSE'}
        if i < 2:
            for value in (old, new): value.update(new_selection=None, selected_action=None, requested_command=[0., 0., 0.])
        records.append(dict(tick=i, observation_index=i, pre_sample_index=749+50*i, decision=old)); current.append(new)
    def read_rows(path):
        assert path == source/runner.CASE
        yield from records
        pytest.fail('consumed next observation after changed request')
    monkeypatch.setattr(runner, 'read_rows', read_rows)
    def observed_packet(i):
        packets.append(i)
        return {'frame':i}, {}, {}, 1_500_000_000+100_000_000*i
    monkeypatch.setattr(runner, 'IntentReturnRGBDReplay', lambda p:SimpleNamespace(frames=[None]*3014, packet=observed_packet))
    monkeypatch.setattr(runner, 'read_json', lambda p,n:[{}]*3014 if n == 'auxiliary_camera_audit.json' else
        [dict(completed=fault != 'incomplete_command', requested_command=[0., 0., 0.])]*3013)
    monkeypatch.setattr(runner, 'packet', lambda *a,**k:({}, {}))
    monkeypatch.setattr(runner, 'public_acquisition', lambda r:r)
    model = SimpleNamespace(state_dict=lambda:{}, parameters=lambda:[])
    monkeypatch.setattr(runner, 'load_assigned', lambda *a:(model, 'supervised_rollout', 'full'))
    calls = []
    def state(value):
        calls.append(True)
        return '0'*64 if fault == 'model_changed' and len(calls) > 1 else runner.SUPERVISED_STATE
    monkeypatch.setattr(runner, 'state_digest', state)
    monkeypatch.setattr(runner, 'ArticulatedCollisionGeometry', lambda *a:object())
    def controller(*a, **kwargs):
        assert a[0] is model and kwargs['condition'] == 'supervised_rollout' and kwargs['persistent']
        def observe(policy, *args, **kwargs):
            i = policy['frame']
            if fault == 'public_mutation': policy['changed'] = True
            return deepcopy(current[i])
        return SimpleNamespace(observe=observe)
    monkeypatch.setattr(runner, 'CommitmentContactController', controller)
    monkeypatch.setattr(runner, 'current_dual_camera_pose', lambda *a,**k:checks.append('raw'))
    monkeypatch.setattr(runner, 'current_measured_floor_pose', lambda *a,**k:checks.append('registered'))
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda *a:SimpleNamespace(free=100*1024**3))
    @contextmanager
    def writer(path): yield stored.append
    monkeypatch.setattr(runner, 'writer', writer)
    if fault is None:
        report = runner.replay({'correction_admission':{}})
        assert packets == [0, 1, 2] and len(stored) == 3 and checks == ['raw', 'registered']*3
        assert report['first_requested_command_difference'] == 2 and report['raw_model_forecast_comparisons'] == 1
        assert report['model_state_unchanged'] and not report['following_recorded_observations_consumed']
        assert not report['native_execution'] and not report['unexecuted_outcomes_inferred']
    else:
        with pytest.raises(ValueError): runner.replay({'correction_admission':{}})
        if fault == 'model_changed': assert packets == [0, 1, 2]
        else: assert packets == [0] and 'comparison_failure' in stored[0]


@pytest.mark.parametrize('fault', [None, 'native', 'diagnostic', 'benchmark', 'context_error', 'context_mutation'])
def test_scoped_verifier_retains_exact_first_worker_and_original_conditions(monkeypatch, fault):
    launch = dict(replay_input_bindings=dict(runner.FIXED), diagnostic_result_sha256=runner.DIAGNOSTIC_SHA,
        verification_benchmark_result_sha256=runner.BENCHMARK_SHA)
    calls = []
    if fault == 'native': launch['replay_input_bindings']['launch.json'] = '0'*64
    elif fault == 'diagnostic': launch['diagnostic_result_sha256'] = '0'*64
    elif fault == 'benchmark': launch['verification_benchmark_result_sha256'] = '0'*64
    monkeypatch.setattr(runner, 'admit_benchmark', lambda:calls.append('benchmark'))
    def scoped(function, digest, value):
        assert function is runner.verify_input_context and digest is runner.digest and value is launch
        calls.append('context')
        if fault == 'context_error': raise ValueError('original check failed')
        if fault == 'context_mutation': value['changed'] = True
        return None, {}
    monkeypatch.setattr(runner, 'verify_with_scoped_digests', scoped)
    if fault is None:
        runner.verify_inputs(launch); assert calls == ['benchmark', 'context']
    else:
        with pytest.raises(ValueError): runner.verify_inputs(launch)
        if fault in ('native', 'diagnostic', 'benchmark'): assert calls == []
