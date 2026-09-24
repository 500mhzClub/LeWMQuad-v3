"""Watch control flow: actual invocation, exclusive outputs and one verifier."""
import hashlib
import json
from pathlib import Path
import subprocess
import pytest
from scripts import await_go2_body_projected_tiled_completion_v1 as watch


def fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(watch, 'ROOT', tmp_path)
    for field in ('WATCH', 'RESULT', 'FAILURE'):
        monkeypatch.setattr(watch,field,tmp_path/getattr(watch,field).name)
    monkeypatch.setattr(watch.check,'OUTPUT',tmp_path/'completion.json')
    runtime=tmp_path/'runtime';runtime.mkdir()
    monkeypatch.setattr(watch.check.run,'OUTPUT',runtime)
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    invocation=dict(boot_id=boot,source_sha256={},owner={'pid':123},tool_session=456,
        previous_result_sha256='a'*64,previous_launch_sha256='b'*64,component_benchmark_sha256='c'*64)
    launch=dict(source_sha256={},previous_result_sha256='a'*64,previous_launch_sha256='b'*64,
        body_projection_component_benchmark_sha256='c'*64,baseline='TiledDensityProgressiveFloorController',
        candidate='BodyProjectedTiledController',frames=1428,native_execution=False,model_training=False)
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
    save(tmp_path/watch.INVOCATION,invocation)
    save(tmp_path/watch.PREPARATION,dict(source_sha256={}))
    save(runtime/'launch.json',launch);save(runtime/'result.json',dict(status='fixture complete'))
    monkeypatch.setattr(watch.sys,'argv',[watch.SOURCE,'--invocation-sha256','d'*64])
    monkeypatch.setattr(watch,'verify',lambda bindings:None)
    monkeypatch.setattr(watch,'verify_artifacts',lambda root,bindings:None)
    monkeypatch.setattr(watch,'discover_sources',lambda seeds,inherited:{})
    monkeypatch.setattr(watch,'digest',lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest())
    sleeps=[];commands=[]
    monkeypatch.setattr(watch.time,'sleep',lambda seconds:sleeps.append(seconds))
    live=iter([True,False])
    monkeypatch.setattr(watch.check.run,'owner_live',lambda owner:next(live))
    def verify_once(command,**kwargs):
        commands.append(command)
        save(watch.check.OUTPUT,dict(status='fixture verified'))
    monkeypatch.setattr(watch.subprocess,'run',verify_once)
    return invocation,launch,runtime,save,sleeps,commands


def test_waits_for_ended_owner_and_passes_actual_hashes_to_exactly_one_checker(monkeypatch,tmp_path):
    _,_,runtime,_,sleeps,commands=fixture(monkeypatch,tmp_path)
    watch.main()
    execution=tmp_path/watch.check.EXECUTION
    assert sleeps==[15]
    assert commands==[[watch.sys.executable,'-B',watch.check.SOURCE,'--result-sha256',
        watch.digest(runtime/'result.json'),'--execution-sha256',watch.digest(execution)]]
    result=json.loads(watch.RESULT.read_text())
    assert result['checker_invocations']==1 and result['replay_owner_ended'] is True
    assert result['completion_sha256']==watch.digest(watch.check.OUTPUT)
    assert not watch.FAILURE.exists()
    with pytest.raises(ValueError,match='exclusive'):watch.main()
    assert len(commands)==1


@pytest.mark.parametrize('fault',['launch_source','candidate','native','frame_count','terminal_failure','ended_before_launch','checker_failure'])
def test_terminal_failures_are_preserved_without_replacement(monkeypatch,tmp_path,fault):
    _,launch,runtime,save,_,commands=fixture(monkeypatch,tmp_path)
    if fault=='launch_source':launch['source_sha256']={'changed':'a'*64}
    elif fault=='candidate':launch['candidate']='different'
    elif fault=='native':launch['native_execution']=True
    elif fault=='frame_count':launch['frames']=1427
    elif fault=='terminal_failure':save(runtime/'failure.json',{'reason':'original failure'})
    elif fault=='checker_failure':
        def fail(command,**kwargs):
            commands.append(command);raise subprocess.CalledProcessError(1,command)
        monkeypatch.setattr(watch.subprocess,'run',fail)
    save(runtime/'launch.json',launch)
    if fault=='ended_before_launch':
        (runtime/'launch.json').unlink()
        monkeypatch.setattr(watch.check.run,'owner_live',lambda owner:False)
    with pytest.raises((ValueError,subprocess.CalledProcessError)):watch.main()
    assert len(commands)==(1 if fault=='checker_failure' else 0)
    failure=json.loads(watch.FAILURE.read_text())
    assert failure['automatic_retry'] is False and failure['evidence_preserved'] is True
    assert not watch.RESULT.exists()


def test_unbound_invocation_rejected_before_any_source_or_artifact_access(monkeypatch,tmp_path):
    fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(watch.sys,'argv',[watch.SOURCE,'--invocation-sha256','pending'])
    monkeypatch.setattr(watch,'verify',lambda *args:pytest.fail('unbound invocation reached sources'))
    with pytest.raises(ValueError,match='actual invocation'):watch.main()
    assert not watch.WATCH.exists()
