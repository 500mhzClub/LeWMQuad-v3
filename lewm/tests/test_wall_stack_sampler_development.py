import sys
import threading
import time
from types import SimpleNamespace
import pytest
from lewm import wall_stack_sampler_development as sampling


def test_owner_stack_is_sampled_without_installing_hooks_and_thread_is_joined():
    profile, trace = sys.getprofile(), sys.gettrace()
    with sampling.WallStackSampler() as sampler:
        assert sampler._sampled.wait(2.)
        assert sys.getprofile() is profile and sys.gettrace() is trace
    report = sampler.report()
    assert report['complete'] and report['sample_count'] >= 1
    assert any(row['function'] == 'test_owner_stack_is_sampled_without_installing_hooks_and_thread_is_joined'
        for sample in report['samples'] for row in sample['stack'])
    assert not sampler._thread.is_alive()
    assert all(0 <= row['offset_ns'] <= report['scope_wall_ns'] and row['capture_wall_ns'] >= 0
        for row in report['samples'])
    assert report['sampler_thread_cpu_ns'] >= 0
    report['samples'][0]['stack'][0]['function'] = 'changed external report'
    assert sampler.report()['samples'][0]['stack'][0]['function'] != 'changed external report'
    sampler.close()
    with pytest.raises(RuntimeError,match='one original'):sampler.__enter__()


@pytest.mark.parametrize('options', [dict(interval_s=True),dict(interval_s=float('nan')),
    dict(interval_s=.001),dict(interval_s=1.),dict(maximum_samples=True),
    dict(maximum_samples=0),dict(maximum_samples=10_001),dict(maximum_depth=False),
    dict(maximum_depth=0),dict(maximum_depth=129)])
def test_unbounded_or_nonexplicit_parameters_rejected(options):
    with pytest.raises(ValueError):sampling.WallStackSampler(**options)


def test_capacity_failure_retains_samples_but_cannot_claim_success():
    sampler = sampling.WallStackSampler(interval_s=.005,maximum_samples=1)
    with pytest.raises(RuntimeError,match='sample capacity'):
        with sampler:
            assert sampler._stop.wait(2.)
    assert not sampler._thread.is_alive()
    report = sampler.report()
    assert report['complete'] is False and report['sample_count'] == 1


def test_body_exception_is_preserved_if_sampler_also_fails(monkeypatch):
    def fail(*args):raise RuntimeError('injected sampling failure')
    monkeypatch.setattr(sampling,'capture_stack',fail)
    sampler=sampling.WallStackSampler()
    with pytest.raises(ValueError,match='original computation failure') as error:
        with sampler:
            assert sampler._sampled.wait(2.)
            raise ValueError('original computation failure')
    assert sampler.report()['error']=='injected sampling failure'
    assert 'Stack sampler also failed' in error.value.__notes__[0]
    assert not sampler._thread.is_alive()


def test_only_the_original_owner_can_close_the_scope():
    errors=[]
    with sampling.WallStackSampler() as sampler:
        def other():
            try:sampler.close()
            except RuntimeError as error:errors.append(str(error))
        thread=threading.Thread(target=other);thread.start();thread.join(timeout=2.)
        assert not thread.is_alive()
        assert errors==['original owner must close its sampling scope']
        with pytest.raises(RuntimeError,match='closed'):sampler.report()
    assert sampler.report()['complete']


@pytest.mark.parametrize('fault', ['missing','depth','protected'])
def test_unavailable_or_unrecordable_frames_fail_without_reading_files(monkeypatch,fault):
    frame=SimpleNamespace(f_code=SimpleNamespace(co_filename='ordinary.py',co_name='fixture'),f_lineno=1,f_back=None)
    if fault=='depth':frame.f_back=frame
    if fault=='protected':frame.f_code.co_filename='/synthetic/sealed_/fixture.py'
    monkeypatch.setattr(sampling.sys,'_current_frames',lambda:{} if fault=='missing' else {7:frame})
    with pytest.raises(RuntimeError):sampling.capture_stack(7,2)
