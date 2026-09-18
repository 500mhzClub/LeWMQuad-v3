import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts import external_body_projection_profile_v2_bindings_development as bindings
from scripts import run_go2_body_projection_external_profile_v2 as run
from lewm.tests import test_external_body_projection_profile_replay_development as fixture_source


@pytest.mark.parametrize('index',range(3))
def test_source_derivatives_contain_only_exact_declared_changes(index):
    manifest=json.loads(Path(bindings.DERIVATIVE).read_text())['files'][index]
    raw=Path(manifest['source']).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==manifest['source_sha256']
    expected=raw.decode()
    for before,after in manifest['replacements']:
        assert expected.count(before)==1
        expected=expected.replace(before,after)
    target=Path(manifest['target']).read_bytes()
    assert hashlib.sha256(target).hexdigest()==manifest['target_sha256']
    assert target.decode()==expected
    assert ast.dump(ast.parse(target))==ast.dump(ast.parse(expected))


@pytest.mark.parametrize('kind',['replay','admission'])
def test_original_globals_and_code_unchanged_with_private_output(tmp_path,kind):
    original=getattr(bindings,'original_'+kind)
    before=vars(original).copy()
    fork=getattr(bindings,'fork_'+kind)(tmp_path)
    names=('isolated_replay','replay') if kind=='replay' else ('capture_verification','admit_completed')
    for name in names:
        old=getattr(original,name);new=getattr(fork,name)
        assert new.__code__ is old.__code__
        assert new.__globals__ is not old.__globals__
        assert new.__globals__['OUTPUT']==tmp_path
        for dependency in names:assert new.__globals__[dependency] is getattr(fork,dependency)
    assert all(vars(original)[name] is value for name,value in before.items())
    assert original.OUTPUT != tmp_path


@pytest.fixture
def synthetic(monkeypatch,tmp_path):
    _,_,prior_rows,prior,_=fixture_source.fixture(monkeypatch,tmp_path)
    local=bindings.fork_replay(tmp_path)
    monkeypatch.setattr(run,'replay',local)
    report=local.replay(prior_rows,prior)
    rows=[json.loads(line) for line in (tmp_path/'comparison.jsonl').read_text().splitlines()]
    return json.loads(json.dumps(report)),rows,prior_rows,json.loads(json.dumps(prior))


def test_full_fresh_output_replay_and_json_report_preserve_every_identity(synthetic):
    run.validate_report(*synthetic)


@pytest.mark.parametrize('field',['observed_state_checks','model_state_sha256','raw_model_forecast_comparisons','real_time_qualified'])
def test_changed_replay_evidence_still_rejects(synthetic,field):
    report,rows,prior_rows,prior=deepcopy(synthetic)
    report[field]=True
    with pytest.raises(ValueError):run.validate_report(report,rows,prior_rows,prior)


def test_v2_command_adds_nonblocking_without_changing_target_or_filters():
    command=run.profiler_command(123)
    assert command[:3]==[str(run.TOOL),'record','--nonblocking']
    assert command[-2:]==['--pid','123']
    assert command[command.index('--output')+1]==str(bindings.OUTPUT/'profile.json')
    assert command[command.index('--rate')+1]=='100'
    assert '--idle' in command and '--gil' not in command and '--subprocesses' not in command


def test_failed_predecessor_live_owner_rejects_before_source_merge(monkeypatch,tmp_path):
    failure=tmp_path/'failed';failure.mkdir()
    audit=dict(status='TERMINAL_EXTERNAL_PROFILE_V1_TRACER_LOCK_FAILURE_PRESERVED',
        full_replay_completed=False,complete_profile_written=False,completion_checker_invoked=False,
        document_sha256={},artifact_sha256={})
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    record=dict(boot_id=bindings.BOOT,owner={'live':True})
    (tmp_path/'watch.json').write_text(json.dumps(record))
    for name in ('launch.json','child_execution.json','profiler_execution.json'):
        (failure/name).write_text(json.dumps(record))
    monkeypatch.setattr(bindings,'ROOT',tmp_path);monkeypatch.setattr(bindings,'FAILED',failure)
    monkeypatch.setattr(bindings,'AUDIT','audit.json');monkeypatch.setattr(bindings,'WATCH','watch.json')
    monkeypatch.setattr(bindings,'verify',lambda value:None)
    monkeypatch.setattr(bindings,'verify_artifacts',lambda *args:None)
    monkeypatch.setattr(bindings,'owner_live',lambda owner:owner['live'])
    monkeypatch.setattr(bindings,'merge_sources',lambda *args:pytest.fail('ended owners required first'))
    with pytest.raises(ValueError,match='must be ended'):bindings.failed_sources({})


def test_v2_admission_capture_never_targets_v1_attempt():
    assert bindings.admission.capture_verification.__globals__['OUTPUT']==bindings.OUTPUT
    assert bindings.admission.capture_verification.__globals__['OUTPUT']!=bindings.original_admission.OUTPUT
    assert run.OUTPUT==bindings.OUTPUT and run.OUTPUT!=bindings.FAILED
