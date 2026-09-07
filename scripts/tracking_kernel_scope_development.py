"""Standard-library-only admission and command prefix for three fixed user units.

Read-only kernel checks; no unit starts, cgroup writes, imports of ML/native
libraries, data reads, or retries. Probe profiles do not authorize native work.
"""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_UNIT = 'lewm-independent-tracking-challenge-20260907-v1.service'
PROBE_UNITS = {mode: f'lewm-tracking-keeper-{mode}-20260907-v1.service'
    for mode in ('fit', 'overflow')}
PROFILES = {CHALLENGE_UNIT: (8 * 1024**3, 512, 48 * 60 * 60),
    **{unit: (64 * 1024**2, 16, 15) for unit in PROBE_UNITS.values()}}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def profile(unit):
    require(unit in PROFILES, 'one of three exact reviewed user units required')
    memory, tasks, runtime = PROFILES[unit]
    return dict(memory_bytes=memory, tasks=tasks, runtime_seconds=runtime)


def unified_group(text):
    require(text.startswith('0::/') and '\n' not in text.strip(), 'one unified cgroup required')
    name = text.strip()[3:]
    p = Path(name)
    require(p.is_absolute() and str(p) == name and '..' not in p.parts, 'canonical unified cgroup path')
    return p


def admit_scope(unit):
    p = profile(unit)
    group = unified_group(Path('/proc/self/cgroup').read_text())
    uid = os.getuid()
    require(group.name == unit and group.parts[:4] == (
        '/', 'user.slice', f'user-{uid}.slice', f'user@{uid}.service'),
        'exact new per-user challenge unit required; unbounded execution forbidden')
    root = Path('/sys/fs/cgroup') / str(group).lstrip('/')
    require(root.resolve() == root, 'ordinary kernel cgroup required')
    values = {n: (root / n).read_text().strip() for n in (
        'memory.max', 'memory.swap.max', 'memory.oom.group', 'pids.max')}
    require(values == {'memory.max': str(p['memory_bytes']), 'memory.swap.max': '0',
        'memory.oom.group': '1', 'pids.max': str(p['tasks'])}, 'exact effective kernel limits required')
    require(int(Path('/proc/self/oom_score_adj').read_text()) != -1000, 'workload cannot be OOM-exempt')
    return dict(unit=unit, pid=os.getpid(), cgroup=str(group), controls=values)


def service_prefix(unit):
    p = profile(unit)
    properties = dict(MemoryMax=p['memory_bytes'], MemorySwapMax=0, OOMPolicy='kill',
        TasksMax=p['tasks'], RuntimeMaxSec=p['runtime_seconds'], KillMode='control-group',
        TimeoutStopSec=30, Restart='no', MemoryAccounting='yes', TasksAccounting='yes')
    return ['/usr/bin/systemd-run', '--user', '--wait', '--pipe', '--collect',
        '--no-ask-password', '--expand-environment=no', '--service-type=exec', '--unit=' + unit,
        '--working-directory=' + str(ROOT)] + [f'--property={k}={v}' for k, v in properties.items()]


def require_fresh_unit(unit):
    profile(unit)
    result = subprocess.run(['/usr/bin/systemctl', '--user', 'show', unit,
        '--property=LoadState', '--value'], capture_output=True, text=True, timeout=10, check=False)
    require(result.returncode in (0, 1) and result.stdout.strip() == 'not-found',
        'exact challenge unit must be absent; no replacing or restarting a unit')
