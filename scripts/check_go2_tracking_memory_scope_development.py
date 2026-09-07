"""Tiny synthetic memory-controller probe, never a native experiment launcher.

Run only in its exact fresh systemd user unit with a 64MiB memory ceiling,
zero swap and group-OOM policy. The overflow mode intentionally terminates this
unit only. Does not write cgroups, move processes, initialize Genesis or retry.
"""
import argparse
import json
import os
from pathlib import Path

LIMIT=64*1024**2
UNITS={mode:f'lewm-tracking-memory-{mode}-20260907-v1.service' for mode in ('fit','overflow')}


def require(condition,message):
    if not condition:raise ValueError(message)


def own_scope(mode):
    require(mode in UNITS,'fixed synthetic memory probe mode')
    text=Path('/proc/self/cgroup').read_text().strip()
    require(text.startswith('0::/') and '\n' not in text,'one actual unified cgroup required')
    relative=Path(text[3:]);uid=os.getuid()
    require(relative.is_absolute() and relative.name==UNITS[mode]
        and relative.parts[:4]==('/','user.slice',f'user-{uid}.slice',f'user@{uid}.service'),
        'exact new per-user probe unit required, never the live collector scope')
    root=Path('/sys/fs/cgroup')/str(relative).lstrip('/')
    require(root.resolve()==root,'ordinary actual kernel scope path required')
    values={n:(root/n).read_text().strip() for n in ('memory.max','memory.swap.max','memory.oom.group','pids.max')}
    require(values=={'memory.max':str(LIMIT),'memory.swap.max':'0','memory.oom.group':'1','pids.max':'16'},
        'kernel must expose exact admitted memory, swap, OOM and task controls before allocating')
    require(int(Path('/proc/self/oom_score_adj').read_text())!=-1000,'probe cannot be OOM-exempt')
    return root,dict(unit=UNITS[mode],pid=os.getpid(),cgroup=str(relative),controls=values)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=UNITS,required=True)
    mode=p.parse_args().mode;root,witness=own_scope(mode)
    print(json.dumps(dict(status='MEMORY_SCOPE_ADMITTED',mode=mode,**witness)),flush=True)
    requested=8*1024**2 if mode=='fit' else 128*1024**2
    print(json.dumps(dict(status='SYNTHETIC_ALLOCATION_REQUEST',bytes=requested)),flush=True)
    payload=bytearray(requested)
    for i in range(0,len(payload),4096):payload[i]=1
    print(json.dumps(dict(status='SYNTHETIC_ALLOCATION_RETURNED',bytes=len(payload),
        current_bytes=int((root/'memory.current').read_text()),
        peak_bytes=int((root/'memory.peak').read_text()),native_execution=False)),flush=True)
    require(mode=='fit','overflow unexpectedly returned; do not claim containment verified')
    return 0


if __name__=='__main__':raise SystemExit(main())
