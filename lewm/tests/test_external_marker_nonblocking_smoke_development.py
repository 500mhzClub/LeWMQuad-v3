import pytest
from scripts import run_go2_external_marker_nonblocking_smoke_v3 as smoke


def command():
    return [str(smoke.previous.TOOL),'record','--format','speedscope','--output',
        str(smoke.OUTPUT/'profile.json'),'--rate','100','--idle','--threads','--full-filenames','--pid','123']


def test_only_nonblocking_option_changes_original_profiler_invocation():
    original=command();actual=smoke.nonblocking_command(original)
    assert actual==original[:2]+['--nonblocking']+original[2:]
    assert original==command()


@pytest.mark.parametrize('fault',['output','rate','pid','flag'])
def test_changed_original_profiler_scope_rejected(fault):
    value=command()
    if fault=='output':value[value.index('--output')+1]='/tmp/other'
    elif fault=='rate':value[value.index('--rate')+1]='50'
    elif fault=='pid':value[-1]='0'
    else:value.insert(2,'--gil')
    with pytest.raises(ValueError):smoke.nonblocking_command(value)


def test_original_python_child_invocation_unchanged():
    value=['python','-B',smoke.previous.CHILD]
    assert smoke.nonblocking_command(value)==value
