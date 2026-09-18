from copy import deepcopy
import json
import sys
import pytest

from scripts import run_go2_independent_round_trip_population_v1 as run
from lewm.tests.test_independent_round_trip_final_admission_development import admissions


@pytest.fixture(scope='module')
def manifest():return run.admission.inputs.study.manifest()


@pytest.fixture(autouse=True)
def fixed_manifest(monkeypatch,manifest):
    monkeypatch.setattr(run.admission.inputs.study,'manifest',lambda:deepcopy(manifest))


def bundle():
    inp,queue=admissions()
    return dict(schema='independent_round_trip_completed_input_bundle.v1',source_sha256={'frozen.py':'a'*64},
        input_admission=inp,five_stage_queue_admission=queue,final_policy_review_completed=False,
        population_execution_permitted=False)


def make(monkeypatch):
    sources={'frozen.py':'a'*64,run.BUNDLE:'b'*64,run.admission.REVIEW:'c'*64}
    monkeypatch.setattr(run,'verify_artifacts',lambda *a:None)
    monkeypatch.setattr(run,'read_json',lambda *a:{k:{'identity':k} for k in run.TEMPLATE_KEYS})
    return run.make_launch(bundle(),sources,'c'*64,'b'*64),sources


def test_exact_monitored_launch_and_owned_bundle(monkeypatch):
    launch,sources=make(monkeypatch)
    assert len(launch['ordered_cases'])==32
    assert launch['population_entrypoint']==run.admission.overlap.ENTRYPOINT
    assert launch['runtime_verifier']==dict(source=run.admission.SOURCE,function='verify_population')
    assert launch['audit_cpu_monitor']==run.admission.overlap.monitored.FIXED
    assert launch['staged_runtime']==run.admission.overlap.driver.staged.FIXED
    sources['frozen.py']='d'*64
    assert launch['source_sha256']['frozen.py']=='a'*64
    assert launch['goal_achieved'] is False and launch['hardware_qualified'] is False


@pytest.mark.parametrize('fault',['input_approval','execution_approval','schema','sources','bundle_binding','review_binding'])
def test_unbound_or_self_approving_bundle_rejected(monkeypatch,fault):
    _,sources=make(monkeypatch);value=bundle()
    if fault=='input_approval':value['final_policy_review_completed']=True
    elif fault=='execution_approval':value['population_execution_permitted']=True
    elif fault=='schema':value['schema']='other'
    elif fault=='sources':sources['frozen.py']='d'*64
    elif fault=='bundle_binding':sources[run.BUNDLE]='d'*64
    else:sources[run.admission.REVIEW]='d'*64
    with pytest.raises(ValueError):run.make_launch(value,sources,'c'*64,'b'*64)


def prepare_main(monkeypatch,tmp_path):
    calls=[];out=tmp_path/'attempt'
    monkeypatch.setattr(run,'OUTPUT',out);monkeypatch.setattr(run,'ROOT',tmp_path)
    monkeypatch.setattr(run,'validate_root',lambda *a,**k:None)
    monkeypatch.setattr(run,'prepared_sources',lambda:{})
    monkeypatch.setattr(run,'resources',lambda:dict(hardware={},resource_admission={}))
    monkeypatch.setattr(run.admission.overlap.monitored,'require_environment',lambda:calls.append('environment'))
    monkeypatch.setattr(run.admission.queue,'owners_ended',lambda:calls.append('owners'))
    monkeypatch.setattr(run,'verify',lambda *a:None)
    monkeypatch.setattr(run,'discover_sources',lambda *a:{})
    monkeypatch.setattr(run.admission,'verify_population',lambda launch,**kw:calls.append(('final',kw['full'])))
    monkeypatch.setattr(run.admission.overlap,'verify_overlap',lambda launch,**kw:calls.append(('overlap',kw['full'])))
    monkeypatch.setattr(run.admission.runtime,'require_native_idle',lambda:calls.append('native_idle'))
    monkeypatch.setattr(run,'create_output',lambda p:(calls.append('create'),p.mkdir()))
    monkeypatch.setattr(run,'make_launch',lambda *a:dict(synthetic='launch'))
    path=tmp_path/run.BUNDLE;path.parent.mkdir();path.write_text('{}')
    return out,calls


def test_source_preflight_performs_no_input_or_native_work(monkeypatch,tmp_path):
    out,calls=prepare_main(monkeypatch,tmp_path)
    monkeypatch.setattr(run,'make_launch',lambda *a:pytest.fail('launch during source preflight'))
    monkeypatch.setattr(sys,'argv',[run.SOURCE,'--source-preflight-only'])
    run.main()
    assert calls==['environment'] and not out.exists()


def test_live_queue_stops_before_review_or_output(monkeypatch,tmp_path):
    out,calls=prepare_main(monkeypatch,tmp_path)
    def live():raise ValueError('diagnostic live')
    monkeypatch.setattr(run.admission.queue,'owners_ended',live)
    monkeypatch.setattr(sys,'argv',[run.SOURCE])
    with pytest.raises(ValueError,match='diagnostic live'):run.main()
    assert not out.exists()


def test_complete_preflight_authenticates_both_verifiers_without_output(monkeypatch,tmp_path):
    out,calls=prepare_main(monkeypatch,tmp_path)
    monkeypatch.setattr(sys,'argv',[run.SOURCE,'--preflight-only','--bundle-sha256','b'*64,'--review-sha256','c'*64])
    run.main()
    assert ('final',True) in calls and ('overlap',True) in calls
    assert 'native_idle' not in calls and not out.exists()


def test_dispatch_uses_actual_monitored_entry_only_after_both_verifiers(monkeypatch,tmp_path):
    out,calls=prepare_main(monkeypatch,tmp_path)
    def driver(output,sha,verifier):
        assert output==out and verifier is run.admission.verify_population
        assert calls.index(('final',True))<calls.index('create')
        assert calls.index(('overlap',True))<calls.index('create')
        assert run.digest(out/'launch.json')==sha
        result={'status':'synthetic_complete'};run.write_json(out/'result.json',result)
        calls.append('driver');return result
    monkeypatch.setattr(run.admission.overlap,'run_population',driver)
    monkeypatch.setattr(sys,'argv',[run.SOURCE,'--bundle-sha256','b'*64,'--review-sha256','c'*64])
    run.main()
    assert calls[-1]=='driver' and (out/'result.json').exists()
    with pytest.raises(ValueError,match='no retry or resume'):run.main()


@pytest.mark.parametrize('existing_failure',[False,True])
def test_driver_failure_preserved_without_retry(monkeypatch,tmp_path,existing_failure):
    out,calls=prepare_main(monkeypatch,tmp_path)
    original={'status':'original_failure','evidence':'retained'}
    def driver(*a):
        if existing_failure:run.write_json(out/'failure.json',original)
        raise ValueError('failed original run')
    monkeypatch.setattr(run.admission.overlap,'run_population',driver)
    monkeypatch.setattr(sys,'argv',[run.SOURCE,'--bundle-sha256','b'*64,'--review-sha256','c'*64])
    with pytest.raises(ValueError,match='failed original run'):run.main()
    saved=json.loads((out/'failure.json').read_text())
    if existing_failure:assert saved==original
    else:assert saved['automatic_retry'] is False and saved['evidence_preserved'] is True
    with pytest.raises(ValueError,match='no retry or resume'):run.main()


def test_bundle_preparation_refuses_live_queue_before_admission(monkeypatch):
    def live():raise ValueError('live original queue')
    monkeypatch.setattr(run.admission.queue,'owners_ended',live)
    monkeypatch.setattr(run.admission.inputs,'admit',lambda *a:pytest.fail('admission while live'))
    with pytest.raises(ValueError,match='live original queue'):
        run.prepare_input_bundle('a'*64,{},sources={})
