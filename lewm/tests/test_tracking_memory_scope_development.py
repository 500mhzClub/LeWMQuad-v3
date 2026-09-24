"""Fail-closed probe admission tests; never creates or changes a service unit."""
from pathlib import Path
import os

import pytest
from scripts import check_go2_tracking_memory_scope_development as mod


@pytest.mark.parametrize('fault',[None,'wrong_unit','wrong_user','legacy','unlimited','swap','group','pids','oom_exempt'])
def test_exact_own_kernel_scope_required_before_synthetic_allocation(monkeypatch,fault):
    unit=mod.UNITS['fit'];uid=os.getuid()
    relative=f'/user.slice/user-{uid}.slice/user@{uid}.service/app.slice/{unit}'
    group='0::'+relative
    values={'memory.max':str(mod.LIMIT),'memory.swap.max':'0','memory.oom.group':'1','pids.max':'16'}
    if fault=='wrong_unit':group=group.replace(unit,'unrelated-live-collector.scope')
    elif fault=='wrong_user':group=group.replace(f'user-{uid}.slice',f'user-{uid+1}.slice')
    elif fault=='legacy':group='5:memory:'+relative
    elif fault=='unlimited':values['memory.max']='max'
    elif fault=='swap':values['memory.swap.max']='max'
    elif fault=='group':values['memory.oom.group']='0'
    elif fault=='pids':values['pids.max']='max'
    reads=[]
    def read(path,*a,**k):
        name=str(path);reads.append(name)
        if name=='/proc/self/cgroup':return group+'\n'
        if name=='/proc/self/oom_score_adj':return '-1000' if fault=='oom_exempt' else '0'
        expected='/sys/fs/cgroup'+relative+'/'
        assert name.startswith(expected),'must not inspect another group'
        return values[name[len(expected):]]+'\n'
    monkeypatch.setattr(Path,'read_text',read)
    if fault:
        with pytest.raises(ValueError):mod.own_scope('fit')
    else:
        root,witness=mod.own_scope('fit')
        assert str(root)=='/sys/fs/cgroup'+relative and witness['controls']==values
    if fault in ('wrong_unit','wrong_user','legacy'):assert reads==['/proc/self/cgroup']
