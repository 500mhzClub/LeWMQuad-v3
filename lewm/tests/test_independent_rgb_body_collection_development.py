"""Fresh membership, causal eligibility, durable failure and terminal accounting."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
from lewm.independent_layout_collection_development import CollectionInventory
from lewm.independent_layout_inventory_development import build_inventory
from scripts.independent_rgb_body_batch_development import (BATCHES, BATCH_BUDGET, EPISODE_ALLOWANCE,
    RESERVE, INVENTORY_IDS, output_root, validate_launch, commit_episode, episode_artifacts, eligibility)


@pytest.fixture(scope='module')
def inventory(): return CollectionInventory(build_inventory())


def launch_fixture(inv, batch='l00'):
    ids = list(inv.episode_ids(batch))
    return dict(batch=batch, output_root=str(output_root(batch)), planned_trials=ids,
        conditions={c: inv.specification(c) for c in ids}, role=inv.specification(ids[0])['data_role'],
        inventory_sha256=INVENTORY_IDS['inventory.json'], maximum_batch_bytes=BATCH_BUDGET,
        episode_storage_allowance_bytes=EPISODE_ALLOWANCE, minimum_free_bytes=RESERVE,
        eligibility_contract='RGB_BODY_TERMINAL_COVERAGE_V1', model_training=False)


def evidence(kind='boundary'):
    coverage = dict(candidate_acquisition_complete=kind not in ('setup', 'infra'),
        classification='SETUP_FAILED' if kind == 'setup' else 'INFRASTRUCTURE_TRUNCATED' if kind == 'infra' else 'PHYSICAL_TERMINAL_RECORDED',
        paired_frames=1)
    window = None if kind == 'setup' else dict(history_ready=kind != 'history')
    score = dict(stable_interior_metric_pass=kind != 'interior', near_occlusion_failure=kind == 'near',
        original_strict_score=dict(passes_sampled_physical_visibility=kind != 'boundary'))
    return coverage, window, [dict(frame=0, score=score)]


def test_all12_fresh_roots_and_exact_roles(inventory):
    assert len({str(output_root(b)) for b in BATCHES}) == 12
    for batch in BATCHES: validate_launch(launch_fixture(inventory, batch), inventory, batch)
    assert [launch_fixture(inventory, b)['role'] for b in BATCHES].count('train') == 6
    for value in ('l12', '../sealed', 'l00/../l01', True):
        with pytest.raises(ValueError): output_root(value)


@pytest.mark.parametrize('fault', ['missing', 'reordered', 'spec', 'role', 'budget', 'inventory', 'output', 'contract'])
def test_launch_rejects_changed_population_roles_or_contract(inventory, fault):
    launch = launch_fixture(inventory)
    if fault == 'missing': launch['planned_trials'].pop()
    elif fault == 'reordered': launch['planned_trials'].reverse()
    elif fault == 'spec': launch['conditions'][launch['planned_trials'][0]]['friction_mu'] += .01
    elif fault == 'role': launch['role'] = 'development_eval'
    elif fault == 'budget': launch['maximum_batch_bytes'] += 1
    elif fault == 'inventory': launch['inventory_sha256'] = '0' * 64
    elif fault == 'output': launch['output_root'] = str(output_root('l01'))
    else: launch['eligibility_contract'] = 'relaxed'
    with pytest.raises(ValueError): validate_launch(launch, inventory, 'l00')


@pytest.mark.parametrize('kind', ['boundary', 'contact', 'setup', 'infra', 'history', 'interior', 'near'])
def test_modality_specific_eligibility_never_implies_depth_navigation(kind):
    row = eligibility(*evidence(kind))
    assert row['rgb_body_prediction_eligible'] == (kind in ('boundary', 'contact'))
    assert bool(row['hard_measurement_failed_frames']) == (kind in ('interior', 'near'))
    assert not row['depth_navigation_qualified'] and not row['boundary_depth_repaired']
    if kind == 'boundary': assert row['strict_depth_failed_frames'] == [0]
    coverage, window, frames = evidence(kind)
    with pytest.raises(ValueError): eligibility(coverage, window, frames + frames)


def test_exact_core_native_collector_is_reused_without_changed_motion_or_sensing():
    from scripts.run_go2_independent_rgb_body_collection_v1 import collect
    from scripts.run_go2_core_ordered_union_dynamic_sensor_pilot_v1 import collect as frozen_collect
    assert collect is frozen_collect


def test_eligibility_wrapper_rejects_old_outputs_before_raw_access(inventory):
    from scripts.independent_rgb_body_audit_development import audit_rgb_body_condition
    from scripts.independent_layout_batch_development import output_root as old_output
    spec = inventory.specification(inventory.episode_ids('l00')[0])
    with pytest.raises(ValueError, match='predecessor'):
        audit_rgb_body_condition(old_output('l00') / spec['trial'], spec, {}, 'definition', batch='l00')


def test_union_roster_binds_raster_metadata_without_unrelated_files(tmp_path, inventory):
    spec = inventory.specification(inventory.episode_ids('l00')[0]); result = dict(setup_checked=False, rgbd_frames=1)
    directory = tmp_path / spec['trial']; directory.mkdir()
    (directory / 'result.json').write_text('{}'); (directory / 'unrelated').write_text('no')
    row = commit_episode(tmp_path, spec, result)
    assert row['artifact_bytes'] == 2 and len(row['artifact_sha256']) == 1
    names = episode_artifacts(spec, result)
    assert 'visual_meshes/wall_union_visual.ply' in names and 'raster_0000.json' in names
    assert len(names) == len(set(names)) and not any('unrelated' in p for p in row['artifact_sha256'])


@pytest.mark.parametrize('fault', ['none', 'setup', 'boundary', 'infrastructure', 'storage', 'artifact', 'precheck', 'interior', 'near', 'packet'])
def test_driver_keeps_fixed_cases_and_stops_only_on_declared_failure(monkeypatch, tmp_path, inventory, fault):
    import scripts.run_go2_independent_rgb_body_collection_v1 as runner
    ids = list(inventory.episode_ids('l00')); calls = []; out = tmp_path / 'batch'
    launch = dict(planned_trials=ids, source_sha256={runner.PROTOCOL: '0' * 64}, role='train')
    monkeypatch.setattr(runner, 'preflight', lambda batch: (inventory, launch))
    monkeypatch.setattr(runner, 'output_root', lambda batch: out)
    monkeypatch.setattr(runner, 'create_output', lambda p: p.mkdir())
    monkeypatch.setattr(runner, 'verify_ordered_launch', lambda p: None)
    monkeypatch.setattr(runner, 'validate_launch', lambda *a: None)
    monkeypatch.setattr(runner, 'verify_artifacts', lambda *a: None)
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda p: SimpleNamespace(free=0 if fault == 'storage' and calls else 100 * 1024**3))
    def collect(inv, output, run, trial, definition):
        assert run == trial
        calls.append(trial)
        if fault == 'infrastructure' and len(calls) == 2: raise RuntimeError('synthetic infrastructure failure')
        return dict(physical_stop='DISALLOWED_CONTACT', acquisition_stop='PACKET_CONTRACT_STOP: synthetic' if fault == 'packet' else None)
    monkeypatch.setattr(runner, 'collect', collect)
    monkeypatch.setattr(runner, 'commit_episode', lambda o, s, r: dict(trial=s['trial'], result=r,
        artifact_sha256={}, absent_expected_artifacts=['missing'] if fault == 'artifact' else [], artifact_bytes=0))
    def audit(*a, **k):
        if fault == 'precheck': raise ValueError('synthetic raw failure')
        cov, window, frames = evidence(fault if fault in ('setup', 'boundary', 'interior', 'near') else 'contact')
        return dict(coverage=cov, eligibility=eligibility(cov, window, frames))
    monkeypatch.setattr(runner, 'audit_rgb_body_condition', audit)
    if fault in ('none', 'setup', 'boundary'):
        runner.run_batch('l00'); result = json.loads((out / 'result.json').read_text())
        assert calls == ids and len(result['prechecks']) == len(result['commits']) == 120
        assert not (out / 'failure.json').exists()
    else:
        with pytest.raises((ValueError, RuntimeError)): runner.run_batch('l00')
        result = json.loads((out / 'failure.json').read_text())
        assert len(result['commits']) == 1 and calls == ids[:2 if fault == 'infrastructure' else 1]
        assert not (out / 'result.json').exists()
        assert result['uncommitted_trial'] == (ids[1] if fault == 'infrastructure' else None)
        if fault in ('interior', 'near', 'packet'): assert len(result['prechecks']) == 1


@pytest.mark.parametrize('fault', ['none', 'bytes', 'precheck_index', 'condition_order', 'artifact_bytes', 'changed_artifact'])
def test_terminal_reader_reconstructs_durable_commits(monkeypatch, tmp_path, inventory, fault):
    import scripts.audit_go2_independent_rgb_body_collection_v1 as audit
    from scripts.run_go2_successive_choice_maze_development_v1 import digest
    ids = list(inventory.episode_ids('l00')); trial = ids[0]; spec = inventory.specification(trial)
    result = dict(setup_checked=False, rgbd_frames=0); (tmp_path / 'launch.json').write_text('{}')
    for name in episode_artifacts(spec, result):
        p = tmp_path / trial / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'synthetic-roster-only')
    commit = commit_episode(tmp_path, spec, result)
    if fault == 'artifact_bytes': commit['artifact_bytes'] += 1
    (tmp_path / 'episode_000_commit.json').write_text(json.dumps(commit))
    (tmp_path / 'episode_000_raw_precheck.json').write_text('{}')
    terminal = dict(status='TERMINAL_RGB_BODY_LAYOUT_COLLECTION_FAILURE', batch='l00', planned_trials=ids,
        conditions={trial: result}, commits={'episode_000_commit.json': digest(tmp_path / 'episode_000_commit.json')},
        prechecks={'episode_000_raw_precheck.json': digest(tmp_path / 'episode_000_raw_precheck.json')}, uncommitted_trial=ids[1],
        committed_bytes=2 + commit['artifact_bytes'] + (tmp_path / 'episode_000_commit.json').stat().st_size + 2)
    if fault == 'bytes': terminal['committed_bytes'] += 1
    elif fault == 'precheck_index': terminal['prechecks'] = {'episode_001_raw_precheck.json': '0' * 64}
    elif fault == 'condition_order': terminal['conditions'] = {ids[1]: result}
    elif fault == 'changed_artifact': (tmp_path / trial / 'physics_trace.npz').write_bytes(b'changed')
    (tmp_path / 'failure.json').write_text(json.dumps(terminal))
    monkeypatch.setattr(audit, 'verify_ordered_launch', lambda p: None)
    monkeypatch.setattr(audit, 'validate_launch', lambda *a: None)
    def verify(root, bindings):
        for p, h in bindings.items(): assert digest(root / p) == h
    monkeypatch.setattr(audit, 'verify_artifacts', verify)
    if fault == 'none':
        _, actual, committed, bindings = audit.load_terminal_batch(tmp_path, inventory, 'l00')
        assert actual['uncommitted_trial'] == ids[1] and list(committed) == [trial]
        assert committed[trial]['raw_precheck'] == {} and 'episode_000_raw_precheck.json' in bindings
    else:
        with pytest.raises(AssertionError): audit.load_terminal_batch(tmp_path, inventory, 'l00')


def test_launch_without_terminal_record_is_not_a_stopped_batch(monkeypatch, tmp_path, inventory):
    import scripts.audit_go2_independent_rgb_body_collection_v1 as audit
    (tmp_path / 'launch.json').write_text('{}')
    monkeypatch.setattr(audit, 'verify_ordered_launch', lambda p: None)
    monkeypatch.setattr(audit, 'validate_launch', lambda *a: None)
    with pytest.raises(ValueError, match='terminal'): audit.load_terminal_batch(tmp_path, inventory, 'l00')


@pytest.mark.parametrize('fault', ['none', 'changed_precheck', 'raw_failure'])
def test_terminal_audit_preserves_full_population_and_never_trains(monkeypatch, tmp_path, inventory, fault):
    import scripts.audit_go2_independent_rgb_body_collection_v1 as audit
    ids = list(inventory.episode_ids('l00'))
    (tmp_path / 'failure.json').write_text('{}')
    launch = dict(source_sha256={audit.PROTOCOL: '0' * 64}, role='train')
    terminal = dict(status='TERMINAL_RGB_BODY_LAYOUT_COLLECTION_FAILURE', committed_bytes=2,
        uncommitted_trial=ids[2])
    cov, window, frames = evidence('setup')
    cov.update(contact_event_count=0, command_schedule_complete=False)
    row = dict(report=dict(recorded_sensor_reconstruction_pass=True, setup_admitted=False,
        schedule_complete=False, departure_present=False, physical_visibility_pass=False, target_contact_positive=0),
        prefix=dict(status='MISSING_DEPARTURE_PREFIX', native_samples=750, frames=1, sha256={}),
        coverage=cov, eligibility=eligibility(cov, window, frames), window=None, targets=None)
    committed = {ids[0]: dict(result={}, absent_expected_artifacts=[], raw_precheck={} if fault == 'changed_precheck' else row),
                 ids[1]: dict(result={}, absent_expected_artifacts=['native_contacts.npz'])}
    monkeypatch.setattr(audit, 'output_root', lambda b: tmp_path)
    monkeypatch.setattr(audit, 'load_inventory', lambda: inventory)
    monkeypatch.setattr(audit, 'load_terminal_batch', lambda *a: (launch, terminal, committed, {}))
    monkeypatch.setattr(audit, 'verify_ordered_launch', lambda l: None)
    monkeypatch.setattr(audit, 'verify_artifacts', lambda *a: None)
    monkeypatch.setattr(audit.shutil, 'disk_usage', lambda p: SimpleNamespace(free=100 * 1024**3))
    def raw_audit(*a, **k):
        if fault == 'raw_failure': raise ValueError('synthetic raw corruption')
        return row
    monkeypatch.setattr(audit, 'audit_rgb_body_condition', raw_audit)
    if fault == 'none':
        audit.run_audit('l00')
        result = json.loads((tmp_path / 'rgb_body_layout_audit.json').read_text())
        assert result['expected_trials'] == len(result['conditions']) == result['population']['planned_cases'] == 120
        assert result['audited_trials'] == 1 and result['eligible_departures'] == 0
        assert result['population']['counts']['SETUP_FAILED'] == 1
        assert result['population']['counts']['INVALID_RECORDING'] == 1
        assert result['population']['counts']['INFRASTRUCTURE_TRUNCATED'] == 1
        assert result['population']['counts']['UNATTEMPTED'] == 117
        assert len(result['prefix_comparisons']) == 120 and not result['model_trained']
        assert json.loads((tmp_path / 'rgb_body_eligible_windows.json').read_text()) == []
    else:
        with pytest.raises((AssertionError, ValueError)): audit.run_audit('l00')
        failure = json.loads((tmp_path / 'rgb_body_layout_audit_failure.json').read_text())
        assert failure['planned_trials'] == ids and failure['completed_reports'] == []
        assert not (tmp_path / 'rgb_body_layout_audit.json').exists()


@pytest.mark.parametrize('fault', ['none', 'physical_collection_failure', 'preflight', 'missing_terminal', 'existing'])
def test_stage_audits_its_own_terminal_failure_without_restarting(monkeypatch, tmp_path, fault):
    import scripts.run_go2_independent_rgb_body_stage_v1 as stage
    output = tmp_path / 'stage'; calls = []
    monkeypatch.setattr(stage, 'output_root', lambda b: output)
    if fault == 'existing': output.mkdir()
    def collect(batch):
        calls.append('collect')
        if fault == 'preflight': raise RuntimeError('no output')
        output.mkdir()
        if fault != 'missing_terminal':
            (output / ('failure.json' if fault == 'physical_collection_failure' else 'result.json')).write_text('{}')
        if fault == 'physical_collection_failure': raise RuntimeError('preserved terminal failure')
    monkeypatch.setattr(stage, 'run_batch', collect)
    monkeypatch.setattr(stage, 'run_audit', lambda b: calls.append('audit'))
    if fault == 'none': stage.run_stage('l00')
    else:
        with pytest.raises((RuntimeError, ValueError)): stage.run_stage('l00')
    assert calls == ([] if fault == 'existing' else ['collect', 'audit'] if fault in ('none', 'physical_collection_failure') else ['collect'])
