"""Native prefix causality, exact source scope and failure-artifact preservation."""
import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pytest
from scripts import reactive_floor_transport_native_prefix_development as prefix
from scripts import run_go2_reactive_floor_transport_maze_pilot_v1 as runner


def evidence():
    selected = {'action':'forward'}
    report = dict(case='full_jepa_novel_maze_00', frames=4, first_requested_command_difference=3,
        first_terminal_policy_difference=None, final_requested_command=[.2,0.,0.],
        prior_requested_command=[.16,0.,.45], final_terminal=None,
        causal_observations_maps_and_settling_mission_exact=True,
        original_actual_commands_before_intervention_exact=True,
        stopped_at_first_command_or_terminal_difference=True, following_recorded_observations_consumed=False,
        learned_model_used=False, candidate_future_outcomes_evaluated=False, learned_residual_used=False,
        changed_selection=selected)
    rows = []
    for i in range(4):
        decision = {key:{'fixture':i} for key in prefix.SHARED}
        decision.update(requested_command=[0.,0.,0.] if i < 3 else [.2,0.,0.], terminal=None, failure=None,
            learned_model_used=False, candidate_future_outcomes_evaluated=False,
            new_selection=None if i < 3 else deepcopy(selected))
        rows.append(dict(tick=i, decision=decision, shared_observed_state_exact=True,
            current_requested_command_matches_original=i < 3,
            original_requested_command=[0.,0.,0.] if i < 3 else [.16,0.,.45]))
    return dict(status='REACTIVE_FLOOR_TRANSPORT_PREFIX_V1_COMPLETE', model_loaded=False,
        native_execution=False, report=report), rows


@pytest.mark.parametrize('fault', [None, 'frames', 'following', 'learned', 'short', 'extra',
    'prior_command', 'current_command', 'state', 'selection'])
def test_admission_requires_saved_four_decisions_and_exact_summary(monkeypatch, fault):
    result, rows = evidence()
    if fault == 'frames': result['report']['frames'] = 5
    elif fault == 'following': result['report']['following_recorded_observations_consumed'] = True
    elif fault == 'learned': rows[3]['decision']['learned_model_used'] = True
    elif fault == 'short': rows.pop()
    elif fault == 'extra': rows.append(deepcopy(rows[-1]))
    elif fault == 'prior_command': rows[1]['original_requested_command'] = [.2,0.,0.]
    elif fault == 'current_command': rows[3]['decision']['requested_command'] = [0.,0.,.45]
    elif fault == 'state': rows[0]['shared_observed_state_exact'] = False
    elif fault == 'selection': result['report']['changed_selection'] = {'action':'left_turn'}
    monkeypatch.setattr(prefix, 'read_rows', lambda *a:iter(rows))
    if fault is None: assert prefix.admit_prefix(None, result) == result['report']
    else:
        with pytest.raises(ValueError): prefix.admit_prefix(None, result)


def native_fixture(monkeypatch, tmp_path):
    result, bound = evidence(); old = tmp_path/'old'; new = tmp_path/'new'; saved = tmp_path/'saved'
    for path in (old,new,saved): path.mkdir()
    data = dict(timestamp_s=np.arange(1000)*.002, base_pose_world=np.zeros((1000,7)))
    for path in (old,new): np.savez(path/'physics_trace.npz', **data)
    decisions = {old:deepcopy(bound), new:deepcopy(bound), saved:deepcopy(bound)}
    tapes = {}
    for path in (old,new):
        for i,row in enumerate(decisions[path]):
            row.update(observation_index=i, pre_sample_index=749+50*i)
            if path == old: row['decision']['requested_command'] = row['original_requested_command']
        tapes[path] = [dict(requested_command=row['decision']['requested_command'], completed=True) for row in decisions[path]]
    reads = []; public_reads = []; packet_changes = {}
    def read_rows(path):
        for i,row in enumerate(decisions[path]):
            reads.append((path,i)); yield row
        pytest.fail('comparison consumed a fifth recorded decision')
    monkeypatch.setattr(prefix, 'read_rows', read_rows)
    monkeypatch.setattr(prefix, 'read_json', lambda path,name:tapes[path] if name == 'command_tape.json' else [{}]*4)
    def reader(path):
        def packet(i):
            public_reads.append((path,i))
            return {'frame':i, 'value':packet_changes.get((path,i),0)}, {}, {}, 1_500_000_000+i*100_000_000
        return SimpleNamespace(packet=packet)
    monkeypatch.setattr(prefix, 'IntentReturnRGBDReplay', reader)
    monkeypatch.setattr(prefix, 'public_acquisition', lambda x:x)
    monkeypatch.setattr(prefix, 'packet', lambda *a, **k:({},{}))
    return result['report'], old, new, saved, decisions, tapes, reads, public_reads, packet_changes, data


@pytest.mark.parametrize('fault', [None, 'physical_prefix', 'public', 'state', 'decision',
    'prior_command', 'current_command', 'short', 'new_future', 'partial_new_command'])
def test_full_native_prefix_excludes_new_outcome_but_catches_prior_mismatch(monkeypatch, tmp_path, fault):
    report, old, new, saved, decisions, tapes, reads, public_reads, changes, data = native_fixture(monkeypatch,tmp_path)
    if fault == 'physical_prefix':
        data['base_pose_world'][899,0] = .1; np.savez(new/'physics_trace.npz', **data)
    elif fault == 'new_future':
        data['base_pose_world'][900:,0] = 100.; np.savez(new/'physics_trace.npz', **data)
    elif fault == 'public': changes[(new,3)] = 1
    elif fault == 'state': decisions[old][3]['decision']['mission_receipt'] = {'altered':True}
    elif fault == 'decision': decisions[new][3]['decision']['new_selection']['action'] = 'different'
    elif fault == 'prior_command': tapes[new][2]['requested_command'] = [.1,0.,0.]
    elif fault == 'current_command': tapes[new][3]['requested_command'] = [0.,0.,.45]
    elif fault == 'short': np.savez(new/'physics_trace.npz', **{k:v[:899] for k,v in data.items()})
    elif fault == 'partial_new_command': tapes[new][3]['completed'] = False
    if fault in (None, 'new_future', 'partial_new_command'):
        report = prefix.compare(old,new,saved,report)
        assert report['physical_prefix_samples'] == 900 and report['common_prefix_frames'] == 4
        assert report['complete_candidate_decisions_match_prospective_prefix']
        assert report['candidate_intervention_command_completed'] == (fault != 'partial_new_command')
        assert len(reads) == 12 and len(public_reads) == 8
        assert not report['following_physical_outcomes_compared']
    else:
        with pytest.raises(ValueError): prefix.compare(old,new,saved,report)


def function(path, name):
    return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_collector_and_raw_audit_preserve_original_physics_and_evaluation_scope():
    class Normalize(ast.NodeTransformer):
        def visit_FunctionDef(self,n):
            keep = [i for i,a in enumerate(n.args.kwonlyargs) if a.arg not in ('model','condition','variant')]
            n.args.kwonlyargs = [n.args.kwonlyargs[i] for i in keep]
            n.args.kw_defaults = [n.args.kw_defaults[i] for i in keep]
            return self.generic_visit(n)
        def visit_Name(self,n):
            if n.id in ('MeasuredFloorTransportController','ReactiveFloorTransportController'): n.id = 'ComparedController'
            return n
        def visit_Constant(self,n):
            if isinstance(n.value,str):
                n.value = n.value.replace('REACTIVE_FLOOR_TRANSPORT_MAZE','MEASURED_FLOOR_TRANSPORT_MAZE')
                n.value = n.value.replace('online_reactive_round_trip_command','online_learned_round_trip_command')
            return n
        def visit_Assign(self,n):
            if ast.unparse(n.targets[0]) == 'before':
                assert ast.unparse(n.value) == 'state_digest(model.state_dict())'; return None
            return self.generic_visit(n)
        def visit_Assert(self,n):
            expression = ast.unparse(n.test)
            if expression in ("not hasattr(controller, 'model') and (not hasattr(controller, 'residual'))",
                    "result['learned_model_used'] is False and result['candidate_future_outcomes_evaluated'] is False",
                    'state_digest(model.state_dict()) == before and all((p.grad is None for p in model.parameters()))'):
                return None
            return self.generic_visit(n)
        def visit_Call(self,n):
            if isinstance(n.func,ast.Name) and n.func.id in ('MeasuredFloorTransportController','ReactiveFloorTransportController'):
                n.args = [a for a in n.args if not (isinstance(a,ast.Name) and a.id == 'model')]
                n.keywords = [k for k in n.keywords if k.arg not in ('persistent','condition','variant')]
            n.keywords = [k for k in n.keywords if k.arg not in
                ('learned_model_used','candidate_future_outcomes_evaluated','model_state_unchanged','high_level_world_model_used')]
            for k in n.keywords:
                if k.arg == 'raw_model_command_replay_pass': k.arg = 'raw_controller_command_replay_pass'
            return self.generic_visit(n)
    for kind,names in (('episode',('collect','artifacts')),('audit',('audit',))):
        for name in names:
            old = function('scripts/measured_floor_transport_maze_'+kind+'_development.py',name)
            new = function('scripts/reactive_floor_transport_maze_'+kind+'_development.py',name)
            assert ast.dump(Normalize().visit(old)) == ast.dump(Normalize().visit(new)), (kind,name)
    old = function('scripts/novel_maze_round_trip_command_audit_development.py','audit_commands')
    new = function('scripts/reactive_nominal_maze_command_audit_development.py','audit_commands')
    assert ast.dump(old) == ast.dump(Normalize().visit(new))


@pytest.mark.parametrize('failed_stage', [None,'collection','audit','prefix'])
def test_worker_preserves_collection_and_audit_when_later_stage_fails(monkeypatch,tmp_path,failed_stage):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path)
    launch = {'robot_urdf_sha256':'a'*64,'source_sha256':{runner.PROTOCOL:'b'*64},'prefix_report':{}}
    monkeypatch.setattr(runner,'read_json',lambda *a:launch)
    monkeypatch.setattr(runner,'verify_inputs',lambda *a:None)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    digest = runner.digest
    monkeypatch.setattr(runner,'digest',lambda path:'a'*64 if path == runner.URDF else digest(path))
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    calls = []
    def collect(index,definition,**kwargs):
        assert 'model' not in kwargs and index == 0 and definition == 'b'*64
        calls.append('collection'); directory = tmp_path/runner.CASE; directory.mkdir()
        (directory/'retained_raw.json').write_text('{}')
        if failed_stage == 'collection': raise ValueError('synthetic collection failure')
        return {'fixture_collection':True}
    def audit(index,result,definition,**kwargs):
        assert 'model' not in kwargs and result == {'fixture_collection':True}
        calls.append('audit')
        if failed_stage == 'audit': raise ValueError('synthetic audit failure')
        return dict(verified_round_trip=False,native_evaluation={},strict_physical_visibility_pass=False,
            hard_measurement_failed_frames=[],renderer_capture_audit={})
    def compare(*args):
        calls.append('prefix')
        if failed_stage == 'prefix': raise ValueError('synthetic prefix failure')
        return {'physical_and_public_prefix_exact':True}
    monkeypatch.setattr(runner,'collect',collect); monkeypatch.setattr(runner,'audit',audit)
    monkeypatch.setattr(runner,'compare',compare); monkeypatch.setattr(runner,'artifacts',lambda *a:['retained_raw.json'])
    terminal = runner.worker('c'*64)
    assert (tmp_path/runner.CASE/'retained_raw.json').is_file()
    assert (tmp_path/(runner.CASE+'_worker_terminal.json')).is_file()
    if failed_stage is None:
        assert terminal['status'] == 'REACTIVE_FLOOR_TRANSPORT_COLLECTED_AND_RAW_AUDITED'
        assert not terminal['verified_round_trip'] and calls == ['collection','audit','prefix']
    else:
        assert terminal['status'] == 'REACTIVE_FLOOR_TRANSPORT_WORKER_FAILED'
        assert 'synthetic '+failed_stage+' failure' in terminal['failure']
        assert calls == ['collection','audit','prefix'][:['collection','audit','prefix'].index(failed_stage)+1]
        if failed_stage != 'collection': assert runner.CASE+'/retained_raw.json' in terminal['artifact_sha256']
        if failed_stage == 'prefix': assert runner.CASE+'_audit.json' in terminal['artifact_sha256']
