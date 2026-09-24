from types import SimpleNamespace
import pytest
from scripts import await_go2_all_phase_translation_bias_v1 as mod


@pytest.mark.parametrize('owner', [mod.WAIT_OWNER, mod.FIT_OWNER])
@pytest.mark.parametrize('fault', ['live', 'missing', 'zombie', 'pid_reuse', 'command'])
def test_original_owners_not_replacements(monkeypatch, owner, fault):
    monkeypatch.setattr(mod.Path, 'read_text', lambda *a, **k:mod.BOOT)
    def process(pid):
        assert pid==owner['pid']
        if fault=='missing': raise mod.psutil.NoSuchProcess(pid)
        return SimpleNamespace(create_time=lambda:0 if fault=='pid_reuse' else owner['created'],
            cmdline=lambda:['replacement'] if fault=='command' else owner['command'],
            status=lambda:mod.psutil.STATUS_ZOMBIE if fault=='zombie' else 'running')
    monkeypatch.setattr(mod.psutil, 'Process', process)
    if fault in ('pid_reuse', 'command'):
        with pytest.raises(ValueError, match='identity'): mod.owner_live(owner)
    else: assert mod.owner_live(owner) is (fault=='live')


def test_reboot_prevents_pid_lookup(monkeypatch):
    monkeypatch.setattr(mod.Path, 'read_text', lambda *a, **k:'another boot')
    monkeypatch.setattr(mod.psutil, 'Process', lambda _:pytest.fail('PID lookup after reboot'))
    with pytest.raises(ValueError, match='boot'): mod.owner_live(mod.WAIT_OWNER)


@pytest.mark.parametrize('root_index', [0, 1])
@pytest.mark.parametrize('failure', [True, False])
def test_failed_or_missing_terminal_never_admitted(monkeypatch, tmp_path, root_index, failure):
    roots = (tmp_path/'wait', tmp_path/'fits')
    for root in roots:
        root.mkdir(); (root/'result.json').write_text('{}')
    monkeypatch.setattr(mod.previous, 'OUTPUT', roots[0]); monkeypatch.setattr(mod.previous, 'FITS', roots[1])
    monkeypatch.setattr(mod, 'verify_artifacts', lambda *a:None)
    if failure: (roots[root_index]/'failure.json').write_text('{}')
    else: (roots[root_index]/'result.json').unlink()
    with pytest.raises(ValueError, match='failed|without complete'): mod.admit_completed_fits({}, {})


@pytest.mark.parametrize('fault', ['none', 'fit_identity', 'fit_owner', 'fit_count', 'wait_source', 'fit_source', 'benchmark'])
def test_original_completed_result_chain(monkeypatch, tmp_path, fault):
    roots = (tmp_path/'wait', tmp_path/'fits')
    for root in roots:
        root.mkdir(); (root/'result.json').write_text('{}')
    monkeypatch.setattr(mod.previous, 'OUTPUT', roots[0]); monkeypatch.setattr(mod.previous, 'FITS', roots[1])
    fit_sources={'fit.py':'fit source'}; wait_sources=fit_sources|{'wait.py':'wait source'}
    waited=dict(status='ALL_PHASE_MATCHED_FITS_WAIT_COMPLETE', source_sha256=wait_sources,
        artifact_sha256={'launch.json':mod.WAIT_LAUNCH_SHA}, fit_pid=mod.FIT_OWNER['pid'],
        fit_result_sha256='fit sha', benchmark_result_sha256=mod.BENCH_SHA, automatic_retry=False, native_execution=False)
    fitted=dict(status='ALL_PHASE_EIGHTEEN_FITS_COMPLETE', optimizer_updates=21600,
        source_sha256=fit_sources, artifact_sha256={'launch.json':mod.FIT_LAUNCH_SHA})
    if fault=='fit_identity': waited['fit_result_sha256']='another fit'
    if fault=='fit_owner': waited['fit_pid']+=1
    if fault=='fit_count': fitted['optimizer_updates']=1200
    if fault=='wait_source': waited['source_sha256']={}
    if fault=='fit_source': fitted['source_sha256']={}
    monkeypatch.setattr(mod, 'read_json', lambda root, _:waited if root==roots[0] else fitted)
    monkeypatch.setattr(mod, 'digest', lambda path:'wait sha' if path.parent==roots[0] else 'fit sha')
    checks=[]
    monkeypatch.setattr(mod, 'verify_artifacts', lambda root, ids:checks.append((root, ids)))
    monkeypatch.setattr(mod, 'verify', lambda sources:None)
    monkeypatch.setattr(mod.previous, 'admit_benchmark', lambda sources:'changed' if fault=='benchmark' else mod.BENCH_SHA)
    if fault=='none':
        result=mod.admit_completed_fits(fit_sources, wait_sources)
        assert result['fit_result_sha256']=='fit sha' and result['wait_result_sha256']=='wait sha'
        assert len(checks)==4
        assert checks[-1][1]['result.json']=='fit sha'
    else:
        with pytest.raises(ValueError, match='original'): mod.admit_completed_fits(fit_sources, wait_sources)
