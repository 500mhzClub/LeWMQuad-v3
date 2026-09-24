"""Lifecycle headroom checks, failure latching and unchanged delegated decisions."""
from copy import deepcopy
import json

import pytest

from scripts import extended_return_budget_resource_guard_development as guard
from scripts import resource_guarded_extended_return_maze_development as pipeline


def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(guard,'validate_root',lambda root:root)
    state=dict(monotonic_s=0.,rss_bytes=guard.GIB,memory_available_bytes=70*guard.GIB,
        artifact_free_bytes=100*guard.GIB)
    calls=[]
    def snapshot(root):
        assert root==tmp_path;state['monotonic_s']+=1.;calls.append('snapshot');return dict(state)
    monkeypatch.setattr(guard,'snapshot',snapshot)
    return state,calls


def test_exact_initial_admission_and_declared_limits():
    report=guard.admission(dict(memory_available_bytes=64*guard.GIB,artifact_free_bytes=84*guard.GIB))
    assert report['native_scene_workers']==1 and report['maximum_sampled_worker_rss_bytes']==48*guard.GIB
    assert not report['operating_system_memory_limit_enforced'] and not report['between_sample_peak_bounded']
    for key,value in [('memory_available_bytes',64*guard.GIB-1),('artifact_free_bytes',84*guard.GIB-1)]:
        supplied=dict(memory_available_bytes=64*guard.GIB,artifact_free_bytes=84*guard.GIB);supplied[key]=value
        with pytest.raises(guard.ResourceLimitError):guard.admission(supplied)
    with pytest.raises(ValueError):guard.admission(dict(memory_available_bytes=True,artifact_free_bytes=84*guard.GIB))


def test_complete_8014_observation_resource_population_is_exclusive_and_bounded(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch);monitor=guard.ResourceGuard(tmp_path,'synthetic','collection')
    monitor.check('begin')
    for frame in range(8014):
        for stage in ('before_packet','after_packet','before_controller','after_controller'):
            monitor.check(stage,frame)
    monitor.check('completed');report=monitor.finish()
    assert report['samples']==32058 and report['phase_completed'] and report['sampled_limits_passed']
    assert not report['between_sample_peak_bounded'] and not report['cleanup_guaranteed']
    with (tmp_path/monitor.stream_name).open() as stream:
        rows=[json.loads(line) for line in stream]
    assert rows[-2]['frame']==8013 and rows[-1]['stage']=='completed'
    assert [r['sample'] for r in rows]==list(range(32058))
    with pytest.raises(ValueError):monitor.check('before_controller',8014)
    with pytest.raises(ValueError):guard.ResourceGuard(tmp_path,'synthetic','collection')


@pytest.mark.parametrize('field,value,reason',[
    ('memory_available_bytes',16*guard.GIB-1,'AVAILABLE_RAM_FLOOR'),
    ('rss_bytes',48*guard.GIB+1,'WORKER_RSS_CEILING'),
    ('artifact_free_bytes',55*guard.GIB,'DISK_RESERVE')])
def test_runtime_breach_latches_even_after_resources_recover(tmp_path,monkeypatch,field,value,reason):
    state,calls=setup(tmp_path,monkeypatch);monitor=guard.ResourceGuard(tmp_path,'synthetic','collection')
    monitor.check('begin');state[field]=value
    with pytest.raises(guard.ResourceLimitError,match=reason):monitor.check('before_controller',4)
    before=len(calls);state.update(memory_available_bytes=70*guard.GIB,rss_bytes=guard.GIB,artifact_free_bytes=100*guard.GIB)
    with pytest.raises(guard.ResourceLimitError,match='latched'):monitor.check('before_controller',5)
    assert len(calls)==before
    report=monitor.finish()
    assert not report['phase_completed'] and not report['sampled_limits_passed']
    assert reason in report['first_breach']['reasons']


def test_shared_filesystem_growth_is_not_attributed_to_audit_and_reserve_still_applies(tmp_path,monkeypatch):
    state,_=setup(tmp_path,monkeypatch);monitor=guard.ResourceGuard(tmp_path,'synthetic','audit')
    monitor.check('begin');state['artifact_free_bytes']-=8*guard.GIB+1
    monitor.check('before_controller',0)
    state['artifact_free_bytes']=40*guard.GIB-1
    with pytest.raises(guard.ResourceLimitError,match='DISK_RESERVE'):monitor.check('after_controller',0)
    monitor.finish()
    state['artifact_free_bytes']=56*guard.GIB
    monitor=guard.ResourceGuard(tmp_path,'synthetic','collection');monitor.check('begin')
    state['artifact_free_bytes']=48*guard.GIB
    monitor.check('completed');assert monitor.finish()['phase_completed']


@pytest.mark.parametrize('episode',['sealed','sealed_trial','../outside','a/b','sealed_test.json'])
def test_invalid_episode_is_rejected_before_any_resource_file(tmp_path,monkeypatch,episode):
    setup(tmp_path,monkeypatch)
    with pytest.raises(ValueError):guard.ResourceGuard(tmp_path,episode,'collection')
    assert not list(tmp_path.iterdir())


def test_invalid_clock_and_frame_cannot_admit_work(tmp_path,monkeypatch):
    state,_=setup(tmp_path,monkeypatch);monitor=guard.ResourceGuard(tmp_path,'synthetic','audit')
    monitor.check('begin')
    for frame in (True,-1,8014):
        with pytest.raises(ValueError):monitor.check('before_controller',frame)
    state['monotonic_s']=-2.
    with pytest.raises(ValueError,match='finite ordered'):monitor.check('before_controller',0)
    report=monitor.finish(ValueError('bad clock'))
    assert not report['phase_completed']


# Explicit private-bind targets for the synthetic lifecycle, not native code.
ResidualAnchoredContinuationController=None
RendererWitnessDualCameraMazeSession=None
audit_sensors=None


def synthetic_collect(*,output,episode_name,events):
    session=RendererWitnessDualCameraMazeSession();controller=ResidualAnchoredContinuationController()
    try:
        for frame in range(3):
            session.samples=range(750+50*frame)
            packet=session.sensor_packets();result=controller.observe(packet)
            events.append(('command',result))
        return {'actual_decisions':[v for k,v in events if k=='command']}
    finally:events.append(('persist','original cleanup'))


def synthetic_audit(*,input_root,episode_name,events):
    raw=audit_sensors()
    controller=ResidualAnchoredContinuationController()
    result=controller.observe(raw)
    return {'audited':result}


@pytest.mark.parametrize('fault',[None,'after_controller','sensor_audit'])
def test_wrappers_preserve_decisions_and_cleanup_but_stop_before_command_on_breach(tmp_path,monkeypatch,fault):
    state,_=setup(tmp_path,monkeypatch);events=[];model_calls=[]
    class Session:
        def sensor_packets(self):return {'frame':(len(self.samples)-750)//50,'pixels':[1,2,3]}
    class Controller:
        def observe(self,packet):
            model_calls.append(deepcopy(packet))
            if fault=='after_controller':state['rss_bytes']=49*guard.GIB
            return {'decision':'unchanged','packet':deepcopy(packet)}
    monkeypatch.setattr(pipeline.pipeline,'ExtendedReturnBudgetRendererSession',Session)
    monkeypatch.setattr(pipeline.pipeline,'ExtendedReturnBudgetChainedController',Controller)
    monkeypatch.setattr(pipeline.pipeline,'collect',synthetic_collect)
    monkeypatch.setattr(pipeline.pipeline,'audit',synthetic_audit)
    def sensors():
        if fault=='sensor_audit':state['memory_available_bytes']=15*guard.GIB
        return {'raw':'unchanged'}
    monkeypatch.setattr(pipeline.pipeline,'audit_sensors',sensors)
    if fault=='after_controller':
        with pytest.raises(guard.ResourceLimitError):pipeline.collect(output=tmp_path,episode_name='synthetic',events=events)
        assert events==[('persist','original cleanup')] and len(model_calls)==1
        report=json.loads((tmp_path/'synthetic_collection_resource_result.json').read_text())
        assert not report['phase_completed'] and report['first_breach']['stage']=='after_controller'
    elif fault=='sensor_audit':
        with pytest.raises(guard.ResourceLimitError):pipeline.audit(input_root=tmp_path,episode_name='synthetic',events=events)
        assert not model_calls
    else:
        result=pipeline.collect(output=tmp_path,episode_name='synthetic',events=events)
        assert result['actual_decisions']==[{'decision':'unchanged','packet':{'frame':f,'pixels':[1,2,3]}} for f in range(3)]
        assert events[-1]==('persist','original cleanup')
        result=pipeline.audit(input_root=tmp_path,episode_name='synthetic',events=events)
        assert result=={'audited':{'decision':'unchanged','packet':{'raw':'unchanged'}}}
        for leaf in pipeline.resource_artifacts('synthetic'):assert (tmp_path/leaf).is_file()
        assert pipeline.definition()['sampled_resource_guards_enabled']


def test_nonresource_failure_is_preserved_in_phase_receipt(tmp_path,monkeypatch):
    setup(tmp_path,monkeypatch)
    def fails(*args,**kwargs):raise ValueError('original sensor failure')
    # Include both globals so the original binding checks still run.
    assert 'ResidualAnchoredContinuationController' in fails.__globals__
    assert 'RendererWitnessDualCameraMazeSession' in fails.__globals__
    monkeypatch.setattr(pipeline.pipeline,'collect',fails)
    with pytest.raises(ValueError,match='original sensor failure'):
        pipeline.collect(output=tmp_path,episode_name='synthetic')
    report=json.loads((tmp_path/'synthetic_collection_resource_result.json').read_text())
    assert report['first_breach'] is None and report['sampled_limits_passed'] and not report['phase_completed']
    assert 'original sensor failure' in report['error']
