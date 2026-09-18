import json
import sys
import pytest
from scripts import await_go2_body_projection_external_profile_completion_v1 as watch


@pytest.mark.parametrize('mode',['complete','profile_failure','checker_failure','already_checked'])
def test_exactly_one_checker_after_ended_parent(monkeypatch,tmp_path,mode):
    for key in ('WATCH','RESULT','FAILURE'):monkeypatch.setattr(watch,key,tmp_path/(key+'.json'))
    monkeypatch.setattr(watch.check,'OUTPUT',tmp_path/'completion.json')
    root=tmp_path/'profile';root.mkdir();(root/'result.json').write_text('{}')
    monkeypatch.setattr(watch.run,'OUTPUT',root)
    launch={'owner':{'pid':123},'source_sha256':{'synthetic':'a'*64}}
    monkeypatch.setattr(watch.run,'authenticate_launch',lambda sha:launch)
    monkeypatch.setattr(watch,'discover_sources',lambda seed,inherited:inherited)
    monkeypatch.setattr(watch,'verify',lambda sources:None)
    monkeypatch.setattr(watch.run,'owner',lambda:{'pid':456})
    live=iter([True,False]);sleeps=[];commands=[]
    monkeypatch.setattr(watch.run,'owner_live',lambda owner:next(live))
    monkeypatch.setattr(watch.time,'sleep',lambda seconds:sleeps.append(seconds))
    monkeypatch.setattr(sys,'argv',['watch','--launch-sha256','b'*64])
    def execute(command,**kwargs):
        commands.append((command,kwargs))
        assert sleeps==[15]
        if mode=='checker_failure':raise RuntimeError('synthetic checker failure')
        watch.check.OUTPUT.write_text('{}')
    monkeypatch.setattr(watch.subprocess,'run',execute)
    if mode=='profile_failure':(root/'failure.json').write_text('{}')
    if mode=='already_checked':watch.check.OUTPUT.write_text('{}')
    if mode=='complete':watch.main()
    else:
        with pytest.raises((ValueError,RuntimeError)):watch.main()
    if mode in ('complete','checker_failure'):
        assert len(commands)==1
        assert commands[0][0]==[sys.executable,'-B',watch.run.CHECKER,'--result-sha256',
            watch.digest(root/'result.json'),'--launch-sha256','b'*64]
        assert commands[0][1]['check'] is True
    else:assert not commands
    if mode=='complete':assert json.loads(watch.RESULT.read_text())['checker_invocations']==1
    elif mode!='already_checked':assert json.loads(watch.FAILURE.read_text())['automatic_retry'] is False
    else:assert not watch.WATCH.exists()
