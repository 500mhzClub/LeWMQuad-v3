"""Unchanged physical/evaluation code and preserved evidence across worker failures."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from lewm.independent_floor_transport_study_development import MODEL_STATE
from scripts import run_go2_direct_flow_maze01_pilot_v1 as runner


def function(path,name):
    return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)


def test_original_native_and_evaluation_calculations_are_preserved():
    class Normalize(ast.NodeTransformer):
        def visit_Name(self,n):
            if n.id=='DirectFlowFloorTransportController': n.id='MeasuredFloorTransportController'
            return n
        def visit_Constant(self,n):
            if isinstance(n.value,str): n.value=n.value.replace('DIRECT_FLOW_MAZE01','MEASURED_FLOOR_TRANSPORT_MAZE')
            return n
        def visit_Call(self,n):
            n.keywords=[k for k in n.keywords if k.arg!='direct_corner_flow_missingness_fallback_enabled']
            return self.generic_visit(n)
        def visit_Assert(self,n):
            if ast.unparse(n.test)=="result['direct_corner_flow_missingness_fallback_enabled'] is True": return None
            return self.generic_visit(n)
    for kind,names in [('episode',('collect','artifacts')),('audit',('audit',))]:
        for name in names:
            old=function('scripts/measured_floor_transport_maze_'+kind+'_development.py',name)
            new=function('scripts/direct_flow_maze01_'+kind+'_development.py',name)
            assert ast.dump(old)==ast.dump(Normalize().visit(new)),(kind,name)


def test_fixed_original_maze1_and_completed_corrected_tracking_prefix():
    assert runner.CASE==('full_jepa_direct_flow_maze_01',1,'full','jepa','seed_2026091001_full_jepa')
    assert runner.LEARNED_CASE[0]=='full_jepa_novel_maze_01'
    assert runner.PREFIX.name=='go2_direct_flow_maze01_prefix_v2_attempt_001'
    assert runner.PREFIX_SHA=='345b54f1b3c647be516040f2b8bf03b4ab3c709b9737dc0bdf7c78bedcacdbe3'
    assert runner.OUTPUT.name=='go2_direct_flow_maze01_pilot_v1_attempt_001'


@pytest.mark.parametrize('failed_stage',[None,'audit','prefix','verification'])
def test_worker_uses_fresh_models_and_retains_evidence_after_failure(monkeypatch,tmp_path,failed_stage):
    monkeypatch.setattr(runner,'OUTPUT',tmp_path)
    launch=dict(robot_urdf_sha256='a'*64,source_sha256={runner.PROTOCOL:'b'*64},correction_admission={},
        prefix_report={'model_state_sha256':MODEL_STATE})
    monkeypatch.setattr(runner,'read_json',lambda *a:launch)
    monkeypatch.setattr(runner,'verify_artifacts',lambda *a:None)
    calls=[]; models=[]
    def verify(*a):
        calls.append('verify')
        if failed_stage=='verification' and calls.count('verify')==2: raise ValueError('synthetic verification failure')
    monkeypatch.setattr(runner,'verify_inputs',verify)
    digest=runner.digest; monkeypatch.setattr(runner,'digest',lambda p:'a'*64 if p==runner.URDF else digest(p))
    monkeypatch.setattr(runner,'state_digest',lambda d:MODEL_STATE)
    monkeypatch.setattr(runner,'ArticulatedCollisionGeometry',lambda *a:object())
    def load(*a):
        models.append(SimpleNamespace(state_dict=lambda:{})); return models[-1],runner.CASE[3],runner.CASE[2]
    monkeypatch.setattr(runner,'load_assigned',load)
    def collect(index,definition,**kwargs):
        assert index==1 and kwargs['model'] is models[0] and kwargs['episode_name']==runner.CASE[0]
        path=tmp_path/runner.CASE[0]; path.mkdir(); (path/'retained_raw.json').write_text('{}'); calls.append('collect')
        return {'fixture':True}
    def audit(index,result,definition,**kwargs):
        assert kwargs['model'] is models[1] and models[1] is not models[0]; calls.append('audit')
        if failed_stage=='audit': raise ValueError('synthetic audit failure')
        return dict(verified_round_trip=False,native_evaluation={},strict_physical_visibility_pass=False,
            hard_measurement_failed_frames=[],renderer_capture_audit={})
    def compare(*a):
        calls.append('prefix')
        if failed_stage=='prefix': raise ValueError('synthetic prefix failure')
        return {'physical_and_public_prefix_exact':True}
    monkeypatch.setattr(runner,'collect',collect); monkeypatch.setattr(runner,'audit',audit)
    monkeypatch.setattr(runner,'compare',compare); monkeypatch.setattr(runner,'artifacts',lambda *a:['retained_raw.json'])
    record=runner.worker('c'*64)
    assert len(models)==2 and runner.CASE[0]+'/retained_raw.json' in record['artifact_sha256']
    assert (tmp_path/(runner.CASE[0]+'_worker_terminal.json')).is_file()
    if failed_stage is None:
        assert record['status']=='DIRECT_FLOW_MAZE01_COLLECTED_AND_RAW_AUDITED'
        assert not record['verified_round_trip'] and record['model_state_unchanged']
        assert record['direct_corner_flow_missingness_fallback_enabled']
    else:
        assert record['status']=='DIRECT_FLOW_MAZE01_WORKER_FAILED'
        assert 'synthetic '+failed_stage+' failure' in record['failure']
    if failed_stage!='audit': assert runner.CASE[0]+'_audit.json' in record['artifact_sha256']
