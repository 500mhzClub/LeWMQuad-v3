"""One separate nonblocking-tool probe with the original synthetic marker child.

Preserve the V2 ownership, full marker checks and ended-child requirements.
Only the output, explicit sources, profiler option and result identity change.
"""
from types import FunctionType, SimpleNamespace
import subprocess

from scripts import run_go2_external_marker_owned_child_smoke_v2 as previous

SOURCE='scripts/run_go2_external_marker_nonblocking_smoke_v3.py'
TEST='lewm/tests/test_external_marker_nonblocking_smoke_development.py'
OUTPUT=previous.TOOL.parent/'marker_nonblocking_owned_child_smoke_v3'


def nonblocking_command(command):
    if command[0] != str(previous.TOOL):return list(command)
    expected=[str(previous.TOOL),'record','--format','speedscope','--output',
        str(OUTPUT/'profile.json'),'--rate','100','--idle','--threads','--full-filenames','--pid']
    if (list(command[:-1]) != expected or not command[-1].isdigit() or int(command[-1]) <= 0):
        raise ValueError('exact original owned-child sampling command required')
    return list(command[:2])+['--nonblocking']+list(command[2:])


def main():
    def popen(command,*args,**kwargs):
        return subprocess.Popen(nonblocking_command(command),*args,**kwargs)
    bindings=previous.__dict__.copy()
    bindings.update(OUTPUT=OUTPUT,SOURCES=previous.SOURCES+[SOURCE,TEST],
        subprocess=SimpleNamespace(Popen=popen,PIPE=subprocess.PIPE,TimeoutExpired=subprocess.TimeoutExpired))
    for name in ('write','main'):
        function=getattr(previous,name)
        bindings[name]=FunctionType(function.__code__,bindings,function.__name__,function.__defaults__)
    original_write=bindings['write']
    def write(name,value):
        if name=='result.json':
            value['status']='EXTERNAL_MARKER_NONBLOCKING_OWNED_CHILD_SMOKE_V3_COMPLETE'
            value['nonblocking_sampling']=True
            value['stack_snapshot_consistency_guaranteed']=False
            value['full_controller_replay_performed']=False
        original_write(name,value)
    bindings['write']=write
    bindings['main']()


if __name__=='__main__':main()
