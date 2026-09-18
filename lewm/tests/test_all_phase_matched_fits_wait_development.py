from types import SimpleNamespace
import pytest
from scripts import await_go2_all_phase_matched_fits_v1 as mod


def fake_process(*,created=None,command=None,status='running'):
    return SimpleNamespace(create_time=lambda:mod.OWNER_CREATED if created is None else created,
        cmdline=lambda:mod.OWNER_COMMAND if command is None else command,status=lambda:status)


@pytest.mark.parametrize('fault',['pid_reuse','different_command','missing','zombie','live'])
def test_original_owner_identity_and_actual_liveness(monkeypatch,fault):
    monkeypatch.setattr(mod.Path,'read_text',lambda *a,**k:mod.BOOT)
    def process(pid):
        assert pid==mod.OWNER_PID
        if fault=='missing':raise mod.psutil.NoSuchProcess(pid)
        return fake_process(created=0. if fault=='pid_reuse' else None,
            command=['replacement'] if fault=='different_command' else None,
            status=mod.psutil.STATUS_ZOMBIE if fault=='zombie' else 'running')
    monkeypatch.setattr(mod.psutil,'Process',process)
    if fault in ('pid_reuse','different_command'):
        with pytest.raises(ValueError,match='identity'):mod.owner_live()
    else:assert mod.owner_live() is (fault=='live')


def test_boot_change_rejects_before_process_lookup(monkeypatch):
    monkeypatch.setattr(mod.Path,'read_text',lambda *a,**k:'different_boot')
    monkeypatch.setattr(mod.psutil,'Process',lambda _:pytest.fail('looked up PID from different boot'))
    with pytest.raises(ValueError,match='boot'):mod.owner_live()


@pytest.mark.parametrize('failed',[True,False])
def test_failed_or_missing_benchmark_result_cannot_launch_fits(monkeypatch,tmp_path,failed):
    monkeypatch.setattr(mod,'BENCH',tmp_path)
    monkeypatch.setattr(mod,'verify_artifacts',lambda *a,**k:None)
    if failed:(tmp_path/'failure.json').write_text('{}')
    with pytest.raises(ValueError,match='failed|without a complete'):
        mod.admit_benchmark({})
