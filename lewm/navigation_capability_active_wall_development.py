"""Active-job wall accounting for the capability programme (ruling of 28 September 2026).

Up to the orientation stop (commit 081465e4, 10:00:37 BST) the conservative
calendar window is kept as recorded; earlier accounting is not recomputed.
Afterwards only wall time while programme jobs run counts, and concurrent jobs
count once (union of intervals). The frozen run owner still enforces its own
calendar 160-hour closeout from `wall_budget_origin.json`.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import uuid

LEDGER = 'wall_active_ledger_2026-09-28.jsonl'
STOP_COMMIT_UNIX_S = 1790586037
CAP_HOURS = 160


def baseline_hours(root):
    origin = json.loads((Path(root)/'wall_budget_origin.json').read_text())['started_unix_s']
    return (STOP_COMMIT_UNIX_S-origin)/3600


def _append(root, row):
    with (Path(root)/LEDGER).open('a') as stream:
        stream.write(json.dumps(row, separators=(',', ':'))+'\n')


@contextmanager
def job(root, name):
    key = uuid.uuid4().hex
    _append(root, dict(job=name, id=key, event='start', unix_s=time.time()))
    status = 'ok'
    try:
        yield
    except BaseException as exc:
        status = 'error: '+repr(exc)[:200]
        raise
    finally:
        _append(root, dict(job=name, id=key, event='end', unix_s=time.time(), status=status))


def intervals(root, now=None):
    now = time.time() if now is None else now
    path = Path(root)/LEDGER
    starts, spans = {}, []
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['event'] == 'start':
                starts[row['id']] = row
            else:
                spans.append((starts.pop(row['id'])['unix_s'], row['unix_s'], row['job']))
    spans += [(r['unix_s'], now, r['job']+' (running)') for r in starts.values()]
    return sorted(spans)


def active_hours(root, now=None):
    total, end = 0., None
    for start, stop, _ in intervals(root, now):
        start = max(start, STOP_COMMIT_UNIX_S)
        if end is None or start > end:
            total += max(0., stop-start)
            end = stop
        elif stop > end:
            total += stop-end
            end = stop
    return total/3600


def hours_used(root, now=None):
    return baseline_hours(root)+active_hours(root, now)
