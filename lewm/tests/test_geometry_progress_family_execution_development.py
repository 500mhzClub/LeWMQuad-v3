"""Family adaptation preserves reviewed physical/audit mechanics and bounded jobs."""
import ast
from copy import deepcopy
from pathlib import Path
import pytest
from scripts import geometry_progress_family_episode_development as episode
from scripts import geometry_progress_family_audit_development as audit
from scripts import probe_go2_geometry_progress_family_scaling_v1 as bench


def source(name):return Path(name).read_text()
def function(text,name):return next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==name)


def test_exact_constructor_with_only_family_specification_binding_changed():
    a=source('scripts/geometry_progress_family_session_development.py').replace('GeometryProgressFamily','GeometryProgressHeightUnion').replace(
        'geometry_progress_layout_family_development','geometry_progress_near_field_development')
    b=source('scripts/geometry_progress_height_union_session_development.py')
    assert ast.dump(ast.parse(a))==ast.dump(ast.parse(b))


def test_collection_and_audit_bodies_only_add_explicit_root_parameter():
    a=source(episode.__file__).replace('GeometryProgressFamilySession','GeometryProgressHeightUnionSession').replace(
        'GEOMETRY_PROGRESS_FAMILY','GEOMETRY_PROGRESS_HEIGHT_UNION').replace('directory=output/trial','directory=OUTPUT/trial')
    new=function(a,'collect');old=function(source('scripts/run_go2_geometry_progress_height_union_v1.py'),'collect')
    assert ast.unparse(new.body.pop(0))=='validate_root(output)'
    new.args.kwonlyargs=[];new.args.kw_defaults=[]
    assert ast.dump(new)==ast.dump(old)
    assert ast.dump(function(source(episode.__file__),'artifacts'))==ast.dump(function(source('scripts/run_go2_geometry_progress_height_union_v1.py'),'artifacts'))
    a=source(audit.__file__).replace('directory=input_root/trial','directory=INPUT/trial')
    new=function(a,'audit_condition');old=function(source('scripts/audit_go2_geometry_progress_height_union_v1.py'),'audit_condition')
    assert ast.unparse(new.body.pop(0))=='validate_root(input_root)'
    new.args.kwonlyargs=[];new.args.kw_defaults=[]
    assert ast.dump(new)==ast.dump(old)
    for name in ('validate_rasters','audit_rasters_and_footprints'):
        assert ast.dump(function(a,name))==ast.dump(function(source('scripts/audit_go2_geometry_progress_height_union_v1.py'),name))


def phase(seconds):
    return dict(all_workers_completed=True,wall_s=seconds,records={t:dict(hard_measurement_failed_frames=[],
        outcome=dict(complete_horizon=True,physical_stop=None,acquisition_stop=None)) for t in bench.cases()})


def comparisons():return [dict(trial=t,exact_equal=True) for t in bench.cases()]


def test_parallel_selection_requires_both_identity_and_measured_speedup():
    assert bench.decision(phase(100),phase(50),comparisons())['selected_workers']==4
    assert bench.decision(phase(100),phase(90),comparisons())['selected_workers']==1
    rows=comparisons();rows[0]['exact_equal']=False
    assert bench.decision(phase(100),phase(50),rows)['selected_workers']==1
    assert bench.decision(phase(100),phase(50),comparisons()[:-1])['selected_workers']==1


def test_serial_sensor_or_infrastructure_failure_blocks_dataset_admission():
    p=phase(100);p['records'][bench.cases()[0]]['hard_measurement_failed_frames']=[0]
    assert bench.decision(p,phase(50),comparisons())['selected_workers'] is None
    p=phase(100);p['all_workers_completed']=False
    assert bench.decision(p,phase(50),comparisons())['selected_workers'] is None


def test_existing_benchmark_root_refused_before_any_preflight(monkeypatch,tmp_path):
    monkeypatch.setattr(bench,'OUTPUT',tmp_path)
    monkeypatch.setattr(bench,'validate_root',lambda p,**k:p)
    monkeypatch.setattr(bench,'preflight',lambda **k:pytest.fail('cannot preflight existing attempt'))
    with pytest.raises(ValueError,match='exclusive'):bench.main()


def test_unknown_worker_trial_rejected_before_creating_log(monkeypatch,tmp_path):
    from scripts import geometry_progress_family_runtime_development as runtime
    monkeypatch.setattr(runtime,'validate_root',lambda p:p)
    with pytest.raises(ValueError,match='exact family worker request'):
        runtime.run_episode(dict(output=str(tmp_path),trial='../bad',record_signature=False))
    assert list(tmp_path.iterdir())==[]


def test_worker_verification_failure_is_terminal_and_cannot_overwrite_claim(monkeypatch,tmp_path):
    from scripts import geometry_progress_family_runtime_development as runtime
    monkeypatch.setattr(runtime,'validate_root',lambda p:p)
    def fail(*a,**k):raise ValueError('synthetic source mismatch')
    monkeypatch.setattr(runtime,'verify_artifacts',fail)
    monkeypatch.setattr(runtime,'collect',lambda *a,**k:pytest.fail('failed verification cannot construct scene'))
    request=dict(output=str(tmp_path),trial=bench.cases()[0],record_signature=False,launch_sha256='0'*64)
    r=runtime.run_episode(request)
    assert r['status']=='FAMILY_WORKER_FAILED' and 'source mismatch' in r['failure']
    terminal=tmp_path/(request['trial']+'_worker_terminal.json');before=terminal.read_bytes()
    with pytest.raises(FileExistsError):runtime.run_episode(request)
    assert terminal.read_bytes()==before


def test_installed_spawn_pool_retires_each_worker_after_one_task():
    import os
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),max_tasks_per_child=1) as executor:
        futures=[executor.submit(os.getpid) for _ in range(8)]
        ids=[f.result(timeout=30) for f in futures]
    assert len(set(ids))==8
