"""Fixed batch membership, durable commits and history-aware raw command audit."""
from copy import deepcopy
from types import SimpleNamespace
import json
import numpy as np
import pytest
from lewm.independent_layout_collection_development import CollectionInventory,schedule
from lewm.independent_layout_inventory_development import build_inventory
from lewm.tests.test_independent_pulse_context_development import trace_fixture,prefix_fixture
from scripts.independent_layout_batch_development import BATCHES,output_root,commit_episode,episode_artifacts
from scripts.independent_layout_collection_audit_development import audit_commands
from scripts.audit_go2_independent_layout_collection_v1 import prefix_comparisons
from scripts.audit_go2_independent_pulse_context_pilot_v1 import prefix_witness


@pytest.mark.parametrize('history',['quiet','recent_forward'])
@pytest.mark.parametrize('action',range(6))
def test_exact_float64_request_audit_for_both_histories_and_all_actions(history,action):
    raw,tape,rows,result=trace_fixture(action);planned=schedule(action,history)
    raw['requested_command']=np.zeros(raw['requested_command'].shape,np.float64)
    raw['applied_command']=np.zeros(raw['applied_command'].shape,np.float64)
    for i,(item,expected) in enumerate(zip(tape,planned,strict=True)):
        item.update(expected);rows[i]['decision']=expected|dict(terminal=False)
        lo,hi=item['pre_sample_index'],item['post_sample_index'];raw['requested_command'][lo+1:hi+1]=expected['requested_command']
        raw['applied_command'][lo+1:hi+1]=raw['applied_command'][lo]+np.clip(
            np.asarray(expected['requested_command'],np.float32)-raw['applied_command'][lo],[-.25,0,-.35],[.25,0,.35])
    audit_commands(raw,tape,rows,result,action,history)
    bad=deepcopy(raw);bad['requested_command'][1150,0]+=1e-9
    with pytest.raises(AssertionError):audit_commands(bad,tape,rows,result,action,history)
    with pytest.raises(AssertionError):audit_commands(raw,tape,rows,result,action,'quiet' if history=='recent_forward' else 'recent_forward')


def test_output_roots_are_exact_separate_owned_batches():
    assert len({str(output_root(b)) for b in BATCHES})==12
    for name in ('l12','l00/../l01','../sealed','train',True):
        with pytest.raises(ValueError):output_root(name)


def test_prefix_comparisons_never_mix_context_history_or_support_and_keep_absences():
    inv=CollectionInventory(build_inventory());ids=inv.episode_ids('l00');witness=prefix_witness(*prefix_fixture())
    prefixes={c:deepcopy(witness) for c in ids};pairs=prefix_comparisons(inv,'l00',prefixes)
    assert len(pairs)==120 and sum(p['is_reference'] for p in pairs.values())==20
    for c,p in pairs.items():
        assert p['matched']
        for k in ('layout_id','context_kind','history_kind','support'):
            assert inv.episodes[c][k]==inv.episodes[p['reference']][k]
    prefixes.pop(ids[0]);pairs=prefix_comparisons(inv,'l00',prefixes)
    assert all(not pairs[c]['matched'] for c in ids[:6])
    assert all(pairs[c]['matched'] for c in ids[6:])
    changed=deepcopy(witness);changed['sha256']['native/new_field']='bad'
    prefixes[ids[0]]=witness;prefixes[ids[7]]=changed;pairs=prefix_comparisons(inv,'l00',prefixes)
    assert not pairs[ids[7]]['matched'] and pairs[ids[6]]['matched']


def test_commit_binds_only_exact_episode_roster_and_records_absence(tmp_path):
    inv=CollectionInventory(build_inventory());spec=inv.specification(inv.episode_ids('l00')[0])
    result=dict(setup_checked=False,rgbd_frames=0);directory=tmp_path/spec['trial'];directory.mkdir()
    (directory/'result.json').write_text('{}');(directory/'unrelated.json').write_text('not part of evidence')
    row=commit_episode(tmp_path,spec,result)
    assert set(row['artifact_sha256'])=={spec['trial']+'/result.json'}
    assert row['artifact_bytes']==2
    assert spec['trial']+'/physics_trace.npz' in row['absent_expected_artifacts']
    assert not any('unrelated' in p for p in row['artifact_sha256'])
    assert len(episode_artifacts(spec,result))==len(set(episode_artifacts(spec,result)))


@pytest.mark.parametrize('fault',['none','infrastructure','storage','missing_artifact','raw_precheck','visibility'])
def test_batch_driver_preserves_commits_and_never_retries_or_skips_physical_stops(monkeypatch,tmp_path,fault):
    import scripts.run_go2_independent_layout_collection_v1 as runner
    inv=CollectionInventory(build_inventory());ids=list(inv.episode_ids('l00'));output=tmp_path/'batch';calls=[]
    launch=dict(planned_trials=ids,source_sha256={runner.PROTOCOL:'0'*64},role='train')
    monkeypatch.setattr(runner,'preflight',lambda b:(inv,launch));monkeypatch.setattr(runner,'output_root',lambda b:output)
    monkeypatch.setattr(runner,'create_output',lambda p:p.mkdir())
    monkeypatch.setattr(runner,'verify',lambda p:None);monkeypatch.setattr(runner,'validate_launch',lambda *a:None)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda p:SimpleNamespace(free=0 if fault=='storage' and calls else 100*1024**3))
    def collect(inventory,out,c,definition):
        calls.append(c)
        if fault=='infrastructure' and len(calls)==2:raise RuntimeError('synthetic episode failure')
        return dict(physical_stop='DISALLOWED_CONTACT',setup_admitted=False,setup_checked=False,rgbd_frames=0)
    monkeypatch.setattr(runner,'collect',collect)
    def precheck(*args):
        if fault=='raw_precheck':raise AssertionError('synthetic raw reconstruction failure')
        return dict(departure_present=False,target_contact_positive=0,physical_visibility_pass=False if fault=='visibility' else None),{},None,None
    monkeypatch.setattr(runner,'audit_condition',precheck)
    monkeypatch.setattr(runner,'commit_episode',lambda out,spec,r:dict(trial=spec['trial'],result=r,artifact_sha256={},
        absent_expected_artifacts=['missing'] if fault=='missing_artifact' else [],artifact_bytes=0))
    if fault=='none':
        runner.run_batch('l00');r=json.loads((output/'result.json').read_text())
        assert calls==ids and len(r['commits'])==120 and r['uncommitted_trial'] is None
        assert len(r['prechecks'])==120
    else:
        with pytest.raises((RuntimeError,ValueError,AssertionError)):runner.run_batch('l00')
        r=json.loads((output/'failure.json').read_text());assert len(r['commits'])==1
        assert (output/'episode_000_commit.json').is_file() and not (output/'result.json').exists()
        assert calls==ids[:2 if fault=='infrastructure' else 1]
        assert r['uncommitted_trial']==(ids[1] if fault=='infrastructure' else None)
        if fault=='visibility':
            assert len(r['prechecks'])==1
            assert not json.loads((output/'episode_000_raw_precheck.json').read_text())['report']['physical_visibility_pass']


def test_terminal_reader_requires_a_terminal_record_not_just_launch(monkeypatch,tmp_path):
    import scripts.audit_go2_independent_layout_collection_v1 as audit
    inv=CollectionInventory(build_inventory());(tmp_path/'launch.json').write_text('{}')
    monkeypatch.setattr(audit,'verify',lambda p:None);monkeypatch.setattr(audit,'validate_launch',lambda *a:None)
    with pytest.raises(ValueError,match='terminal'):audit.load_terminal_batch(tmp_path,inv,'l00')


def test_empty_failed_batch_is_preserved_as_unattempted_not_fake_negative_data(monkeypatch,tmp_path):
    import scripts.audit_go2_independent_layout_collection_v1 as audit
    inv=CollectionInventory(build_inventory());ids=list(inv.episode_ids('l00'))
    (tmp_path/'launch.json').write_text('{}')
    terminal=dict(status='TERMINAL_LAYOUT_COLLECTION_FAILURE',batch='l00',planned_trials=ids,conditions={},commits={},
        prechecks={},committed_bytes=2,uncommitted_trial=None)
    (tmp_path/'failure.json').write_text(json.dumps(terminal))
    monkeypatch.setattr(audit,'verify',lambda p:None);monkeypatch.setattr(audit,'validate_launch',lambda *a:None)
    monkeypatch.setattr(audit,'verify_artifacts',lambda *a:None)
    _,actual,committed,bindings=audit.load_terminal_batch(tmp_path,inv,'l00')
    assert not committed and actual['conditions']=={} and set(bindings)=={'launch.json','failure.json'}
    assert all(not p['matched'] for p in audit.prefix_comparisons(inv,'l00',{}).values())


@pytest.mark.parametrize('fault',['none','bytes','precheck_index','condition_order','artifact_bytes','changed_artifact'])
def test_terminal_reader_reconstructs_durable_commit_and_precheck_accounting(monkeypatch,tmp_path,fault):
    import scripts.audit_go2_independent_layout_collection_v1 as audit
    from scripts.run_go2_successive_choice_maze_development_v1 import digest
    inv=CollectionInventory(build_inventory());ids=list(inv.episode_ids('l00'));c=ids[0];spec=inv.specification(c)
    result=dict(setup_checked=False,rgbd_frames=0);(tmp_path/'launch.json').write_text('{}')
    for name in episode_artifacts(spec,result):
        p=tmp_path/c/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic-roster-only')
    commit=commit_episode(tmp_path,spec,result)
    if fault=='artifact_bytes':commit['artifact_bytes']+=1
    (tmp_path/'episode_000_commit.json').write_text(json.dumps(commit))
    (tmp_path/'episode_000_raw_precheck.json').write_text('{}')
    terminal=dict(status='TERMINAL_LAYOUT_COLLECTION_FAILURE',batch='l00',planned_trials=ids,
        conditions={c:result},commits={'episode_000_commit.json':digest(tmp_path/'episode_000_commit.json')},
        prechecks={'episode_000_raw_precheck.json':digest(tmp_path/'episode_000_raw_precheck.json')},uncommitted_trial=ids[1],
        committed_bytes=2+commit['artifact_bytes']+(tmp_path/'episode_000_commit.json').stat().st_size+2)
    if fault=='bytes':terminal['committed_bytes']+=1
    elif fault=='precheck_index':terminal['prechecks']={'episode_001_raw_precheck.json':'0'*64}
    elif fault=='condition_order':terminal['conditions']={ids[1]:result}
    elif fault=='changed_artifact':(tmp_path/c/'physics_trace.npz').write_bytes(b'changed')
    (tmp_path/'failure.json').write_text(json.dumps(terminal))
    monkeypatch.setattr(audit,'verify',lambda p:None);monkeypatch.setattr(audit,'validate_launch',lambda *a:None)
    def verify_bound(root,bindings):
        for p,h in bindings.items():assert digest(root/p)==h
    monkeypatch.setattr(audit,'verify_artifacts',verify_bound)
    if fault=='none':
        _,r,committed,b=audit.load_terminal_batch(tmp_path,inv,'l00')
        assert r['uncommitted_trial']==ids[1] and list(committed)==[c]
        assert committed[c]['raw_precheck']=={} and 'episode_000_raw_precheck.json' in b
    else:
        with pytest.raises(AssertionError):audit.load_terminal_batch(tmp_path,inv,'l00')
