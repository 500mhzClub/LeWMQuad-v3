"""Adoption requires a complete pair and changes verification execution only."""
import ast
from copy import deepcopy
from pathlib import Path
import pytest
from scripts import partial_floor_height_scoped_verification_admission_development as admission
from scripts import run_go2_partial_floor_height_maze01_scoped_verification_pilot_v1 as runner
from scripts import run_go2_partial_floor_height_maze01_pilot_v1 as original


def fixture():
    scope = dict(every_cached_file_freshly_rehashed=True, cache_retained_after_call=False,
        imported_module_globals_mutated=False, original_verification_conditions_executed=True,
        unique_files=2, digest_requests=5, guarded_cache_hits=3, initial_hashed_bytes=100, final_hashed_bytes=100)
    rows = [dict(mode=mode, verified=True, input_context_unchanged=True,
        digest_scope=scope if mode == 'scoped' else None) for mode in ('scoped', 'original')]
    report = dict(pairs=1, fixed_order=['scoped', 'original'], both_verifiers_completed=True,
        input_context_unchanged=True, digest_results_reused_across_calls=False,
        imported_verifier_globals_changed=False, rows=rows)
    return dict(status='SCOPED_VERIFICATION_DIGEST_BENCHMARK_V1_COMPLETE', model_loaded=False,
        native_execution=False, artifact_sha256={}, source_sha256={}, report=report), rows


@pytest.mark.parametrize('fault', [None, 'status', 'target', 'incomplete', 'order', 'retained', 'fresh_hash',
    'global_mutation', 'hash_bytes', 'hit_count', 'stage_file', 'native'])
def test_only_completed_matching_pair_admits_reuse(monkeypatch, fault):
    result, rows = fixture(); files = {'result.json':result, 'launch.json':{'target_result_sha256':admission.PRIOR_SHA},
        'scoped_verification.json':deepcopy(rows[0]), 'original_verification.json':deepcopy(rows[1])}
    scope = rows[0]['digest_scope']
    if fault == 'status': result['status'] = 'FAILED'
    elif fault == 'target': files['launch.json']['target_result_sha256'] = '0'*64
    elif fault == 'incomplete': result['report']['both_verifiers_completed'] = False
    elif fault == 'order': result['report']['fixed_order'].reverse()
    elif fault == 'retained': scope['cache_retained_after_call'] = True
    elif fault == 'fresh_hash': scope['every_cached_file_freshly_rehashed'] = False
    elif fault == 'global_mutation': scope['imported_module_globals_mutated'] = True
    elif fault == 'hash_bytes': scope['final_hashed_bytes'] -= 1
    elif fault == 'hit_count': scope['guarded_cache_hits'] += 1
    elif fault == 'stage_file': files['original_verification.json']['verified'] = False
    elif fault == 'native': result['native_execution'] = True
    if fault in ('retained', 'fresh_hash', 'global_mutation', 'hash_bytes', 'hit_count'):
        files['scoped_verification.json'] = deepcopy(rows[0])
    monkeypatch.setattr(admission, 'read_json', lambda p,n:files[n])
    checks = []
    monkeypatch.setattr(admission, 'verify_artifacts', lambda *args:checks.append('artifacts'))
    monkeypatch.setattr(admission, 'verify_sources', lambda *args:checks.append('sources'))
    if fault is None:
        assert admission.admit_benchmark() is result and checks == ['artifacts', 'artifacts', 'sources']
    else:
        with pytest.raises(ValueError): admission.admit_benchmark()


@pytest.mark.parametrize('fault', [None, 'benchmark', 'original_error', 'input_mutation', 'wrong_return'])
def test_new_scope_executes_original_pilot_verifier_and_preserves_launch(monkeypatch, fault):
    launch = {'verification_benchmark_result_sha256':admission.BENCHMARK_SHA, 'fixture':True}
    before = deepcopy(launch); calls = []
    monkeypatch.setattr(admission, 'admit_benchmark', lambda:calls.append('admit'))
    if fault == 'benchmark': launch['verification_benchmark_result_sha256'] = '0'*64
    def scoped(function, digest, value):
        assert function is original.verify_inputs and digest is admission.digest and value is launch
        calls.append('scoped')
        if fault == 'original_error': raise ValueError('original validation failed')
        if fault == 'input_mutation': value['changed'] = True
        return ('bad' if fault == 'wrong_return' else None), {}
    monkeypatch.setattr(admission, 'verify_with_scoped_digests', scoped)
    if fault is None:
        assert admission.verify_inputs(launch) is None
        assert launch == before and calls == ['admit', 'scoped']
    else:
        with pytest.raises(ValueError): admission.verify_inputs(launch)
        if fault == 'benchmark': assert not calls


def test_scientific_worker_and_all_runtime_components_remain_original():
    for name in ('collect', 'artifacts', 'audit', 'admit_prefix', 'compare', 'load_assigned',
        'specification', 'public_mission', 'ArticulatedCollisionGeometry'):
        assert getattr(runner, name) is getattr(original, name)
    assert runner.CASE == original.CASE and runner.PRIOR_SHA == original.PRIOR_SHA
    assert runner.OUTPUT != original.OUTPUT and runner.PROTOCOL != original.PROTOCOL
    assert runner.verify_inputs is admission.verify_inputs
    def worker(path):
        return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'worker')
    class Normalize(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str): node.value = node.value.replace('_SCOPED_VERIFICATION', '')
            return node
    assert ast.dump(worker(original.__file__)) == ast.dump(Normalize().visit(worker(runner.__file__)))
