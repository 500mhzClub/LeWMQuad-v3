"""Fresh models, retained failure evidence and no bypass of the current queue."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from scripts import run_go2_supervised_commitment_contact_maze01_pilot_v1 as runner
from scripts import supervised_commitment_contact_queue_gate_development as gate
from lewm.tests.test_supervised_commitment_contact_native_development import evidence


@pytest.mark.parametrize('failed_stage', [None, 'audit', 'prefix', 'verification', 'preflight'])
def test_worker_fresh_models_and_preserved_raw_evidence(monkeypatch, tmp_path, failed_stage):
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    launch = dict(robot_urdf_sha256='a'*64, source_sha256={runner.PROTOCOL:'b'*64}, correction_admission={}, preflight_only=failed_stage == 'preflight', prefix_report={})
    monkeypatch.setattr(runner, 'read_json', lambda *a:launch); monkeypatch.setattr(runner, 'verify_artifacts', lambda *a:None)
    calls = []; models = []
    def verify(*a):
        calls.append('verify')
        if failed_stage == 'verification' and calls.count('verify') == 2: raise ValueError('synthetic verification failure')
    monkeypatch.setattr(runner, 'verify_inputs', verify)
    digest = runner.digest; monkeypatch.setattr(runner, 'digest', lambda p:'a'*64 if p == runner.URDF else digest(p))
    monkeypatch.setattr(runner, 'state_digest', lambda d:runner.SUPERVISED_STATE)
    monkeypatch.setattr(runner, 'ArticulatedCollisionGeometry', lambda *a:object())
    def load(*a):
        models.append(SimpleNamespace(state_dict=lambda:{})); return models[-1], runner.CASE[3], runner.CASE[2]
    monkeypatch.setattr(runner, 'load_assigned', load)
    def collect(index, definition, **kwargs):
        assert index == 1 and kwargs['model'] is models[0] and kwargs['episode_name'] == runner.CASE[0]
        path = tmp_path/runner.CASE[0]; path.mkdir(); (path/'retained_raw.json').write_text('{}'); calls.append('collect')
        return {'fixture':True}
    def audit(*args, **kwargs):
        assert kwargs['model'] is models[1] and models[1] is not models[0]; calls.append('audit')
        if failed_stage == 'audit': raise ValueError('synthetic audit failure')
        return dict(verified_round_trip=False, native_evaluation={}, strict_physical_visibility_pass=False,
            hard_measurement_failed_frames=[], renderer_capture_audit={})
    def compare(*a):
        calls.append('prefix')
        if failed_stage == 'prefix': raise ValueError('synthetic prefix failure')
        return {'physical_and_public_prefix_exact':True}
    monkeypatch.setattr(runner, 'collect', collect); monkeypatch.setattr(runner, 'audit', audit)
    monkeypatch.setattr(runner, 'compare', compare); monkeypatch.setattr(runner, 'artifacts', lambda *a:['retained_raw.json'])
    record = runner.worker('c'*64)
    assert (tmp_path/(runner.CASE[0]+'_worker_terminal.json')).is_file()
    if failed_stage == 'preflight':
        assert not models and not calls and 'preflight cannot execute' in record['failure']; return
    assert len(models) == 2 and runner.CASE[0]+'/retained_raw.json' in record['artifact_sha256']
    if failed_stage is None:
        assert record['status'] == 'SUPERVISED_COMMITMENT_CONTACT_MAZE01_COLLECTED_AND_RAW_AUDITED'
        assert not record['verified_round_trip'] and record['model_state_unchanged'] and record['commitment_contact_policy_enabled']
    else:
        assert record['status'] == 'SUPERVISED_COMMITMENT_CONTACT_MAZE01_WORKER_FAILED'
        assert 'synthetic '+failed_stage+' failure' in record['failure']
    if failed_stage != 'audit': assert runner.CASE[0]+'_audit.json' in record['artifact_sha256']
    if failed_stage in (None, 'verification'): assert runner.CASE[0]+'_prefix_comparison.json' in record['artifact_sha256']


def context():
    result, _, _ = evidence()
    prefix = dict(diagnostic_result_sha256=runner.DIAGNOSTIC_SHA, verification_benchmark_result_sha256=runner.BENCHMARK_SHA,
        replay_input_bindings=dict(runner.FIXED), correction_admission={'model':'fixture'})
    launch = dict(verification_benchmark_result_sha256=runner.BENCHMARK_SHA, prospective_prefix_result_sha256=runner.PREFIX_SHA,
        prefix_artifact_sha256={'result.json':runner.PREFIX_SHA}, prior_artifact_sha256=dict(runner.FIXED),
        correction_admission=deepcopy(prefix['correction_admission']), prefix_report=result['report'], planned_case=list(runner.CASE),
        scene_specification=runner.specification(1), public_mission=runner.public_mission(1), model_state_sha256=runner.SUPERVISED_STATE,
        implementation_class='CommitmentContactController', commitment_contact_policy_enabled=True,
        scored_pose_horizon_ns=100_000_000, scored_contact_horizon_ns=100_000_000, path_constraint_horizon_ns=800_000_000,
        contact_penalty_coefficient_m=1.2, source_sha256={}, preflight_only=True, queue_completion=None, queue_result_sha256=None)
    matched = {k:'fixed_'+k for k in runner.MATCHED_KEYS}
    launch.update(matched); prefix.update(matched)
    return launch, prefix


@pytest.mark.parametrize('fault', [None, 'benchmark', 'prefix', 'prefix_binding', 'native_binding', 'original_error', 'mutation', 'return'])
def test_fresh_scope_checks_fixed_identities_and_preserves_context(monkeypatch, fault):
    launch, _ = context(); calls = []
    if fault == 'benchmark': launch['verification_benchmark_result_sha256'] = '0'*64
    elif fault == 'prefix': launch['prospective_prefix_result_sha256'] = '0'*64
    elif fault == 'prefix_binding': launch['prefix_artifact_sha256']['result.json'] = '0'*64
    elif fault == 'native_binding': launch['prior_artifact_sha256']['launch.json'] = '0'*64
    monkeypatch.setattr(runner, 'admit_benchmark', lambda:calls.append('benchmark'))
    def scoped(function, digest, value):
        assert function is runner.verify_input_context and digest is runner.digest and value is launch
        calls.append('original')
        if fault == 'original_error': raise ValueError('original failed')
        if fault == 'mutation': value['changed'] = True
        return ('bad' if fault == 'return' else None), {}
    monkeypatch.setattr(runner, 'verify_with_scoped_digests', scoped)
    if fault is None:
        runner.verify_inputs(launch); assert calls == ['benchmark', 'original']
    else:
        with pytest.raises(ValueError): runner.verify_inputs(launch)
        if fault in ('benchmark', 'prefix', 'prefix_binding', 'native_binding'): assert not calls


@pytest.mark.parametrize('fault', [None, 'actual', 'queue_changed', 'diagnostic', 'bindings', 'model', 'report', 'case',
    'scene', 'implementation', 'horizon', 'coefficient', 'original_error', 'bad_preflight', 'budget'])
def test_original_replay_context_and_queue_gate_are_required(monkeypatch, fault):
    launch, prefix = context(); report = deepcopy(launch['prefix_report']); calls = []
    if fault in ('actual', 'queue_changed'): launch.update(preflight_only=False, queue_result_sha256='q'*64, queue_completion={'verified':True})
    if fault == 'diagnostic': prefix['diagnostic_result_sha256'] = '0'*64
    elif fault == 'bindings': prefix['replay_input_bindings']['extra'] = '0'*64
    elif fault == 'model': launch['correction_admission'] = {}
    elif fault == 'report': launch['prefix_report']['frames'] = 5
    elif fault == 'case': launch['planned_case'][1] = 2
    elif fault == 'scene': launch['scene_specification'] = runner.specification(2)
    elif fault == 'implementation': launch['implementation_class'] = 'other'
    elif fault == 'horizon': launch['path_constraint_horizon_ns'] = 100_000_000
    elif fault == 'coefficient': launch['contact_penalty_coefficient_m'] = .1
    elif fault == 'bad_preflight': launch['preflight_only'] = None
    elif fault == 'budget': launch['navigation_ticks'] = 'changed'
    monkeypatch.setattr(runner, 'verify', lambda *a:calls.append('environment'))
    monkeypatch.setattr(runner, 'verify_artifacts', lambda *a:None)
    monkeypatch.setattr(runner, 'read_json', lambda p,n:prefix if n == 'launch.json' else {'report':report})
    def original(value):
        assert value is prefix; calls.append('original')
        if fault == 'original_error': raise ValueError('original failed')
    monkeypatch.setattr(runner, 'verify_prefix_context', original)
    monkeypatch.setattr(runner, 'verify_queue_launch', lambda *a:calls.append('queue_launch'))
    def queue(sha, sources):
        calls.append('queue_completion')
        assert sha == 'q'*64
        return {'verified':fault != 'queue_changed'}
    monkeypatch.setattr(runner, 'verify_queue_completion', queue)
    if fault in (None, 'actual'):
        runner.verify_input_context(launch)
        assert calls == ['environment', 'original', 'queue_launch']+(['queue_completion'] if fault == 'actual' else [])
    else:
        with pytest.raises(ValueError): runner.verify_input_context(launch)


@pytest.mark.parametrize('fault', [None, 'failed_status', 'short', 'order', 'raw', 'physical', 'unverified', 'retry', 'source', 'launch'])
def test_queue_completion_requires_every_ordered_receipt_without_selecting_successes(fault):
    launch = {'source_sha256':{'source':'sha'}}
    result = dict(status='PREPARED_NATIVE_QUEUE_V1_COMPLETE', source_sha256=deepcopy(launch['source_sha256']),
        artifact_sha256={'launch.json':gate.QUEUE_LAUNCH_SHA}, automatic_retry=False, completed=[])
    for spec in (gate.SUPERVISED, *gate.JOBS):
        result['completed'].append(dict(output_root=str(gate.QUEUE.parent/spec['output']), cases=spec['cases'],
            all_raw_audits_pass=True, all_physical_prefixes_pass=True, original_verifier_reexecuted=True,
            scientific_success_required=False, measured_round_trip_successes=0))
    if fault == 'failed_status': result['status'] = 'FAILED'
    elif fault == 'short': result['completed'].pop()
    elif fault == 'order': result['completed'].reverse()
    elif fault == 'raw': result['completed'][0]['all_raw_audits_pass'] = False
    elif fault == 'physical': result['completed'][0]['all_physical_prefixes_pass'] = False
    elif fault == 'unverified': result['completed'][0]['original_verifier_reexecuted'] = False
    elif fault == 'retry': result['automatic_retry'] = True
    elif fault == 'source': result['source_sha256'] = {}
    elif fault == 'launch': result['artifact_sha256']['launch.json'] = '0'*64
    if fault is None: assert gate.admit_queue_result(result, launch) == result['completed']
    else:
        with pytest.raises(ValueError): gate.admit_queue_result(result, launch)


def test_live_native_owner_prevents_another_scene(monkeypatch):
    monkeypatch.setattr(gate, 'competitors', lambda:[{'pid':123}])
    with pytest.raises(ValueError): gate.require_native_idle()
    monkeypatch.setattr(gate, 'competitors', lambda:[])
    gate.require_native_idle()
