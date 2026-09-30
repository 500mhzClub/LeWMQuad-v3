"""Check two runs of the same mission for identical decisions (Andrew, 1 October 2026).

Used for the machine-load check: the same C3 mission run alone and alongside another run must
make identical decisions if behaviour does not depend on wall time. Compared:
- every planning record: frame, action, committed flag, route status and the selection's
  scored candidates and forecast values (the motion-correction forecast arrays);
- every 20-ms request: requested and applied command and dispatch reason;
- the physics trace (base pose and applied command at every 2-ms step), by hash;
- the evaluated outcome.
Wall-clock fields (profiles, wall_s, latencies) are ignored.

Usage: compare_go2_dev_runs_identity_development.py RUN_A RUN_B
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WALL_KEYS = ('wall', 'latency', 'profile', 'routing_s', 'added_routing_s', 'elapsed_wall', 'completed_wall')


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if not any(w in k for w in WALL_KEYS)}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def planning(run):
    rows = json.loads((run/'planning.json').read_text())
    return [strip(dict(frame=r.get('frame'), action=r.get('action'), committed=r.get('committed'), reason=r.get('reason'),
                       route=r.get('route_status'), selection=r.get('selection'), correction=r.get('motion_correction')))
            for r in rows]


def requests(run):
    return [(r['now_ns'], r['requested_command'], r.get('applied_command'), r['reason']) for r in json.loads((run/'requests.json').read_text())]


def trace_hash(run):
    with np.load(run/'native/physics_trace.npz', allow_pickle=False) as z:
        return hashlib.sha256(z['base_pose_world'].tobytes()+z['applied_command'].tobytes()+z['timestamp_s'].tobytes()).hexdigest()


def main(a, b):
    a, b = Path(a), Path(b)
    pa, pb = planning(a), planning(b)
    first = next((i for i, (x, y) in enumerate(zip(pa, pb)) if x != y), None)
    ra, rb = requests(a), requests(b)
    first_request = next((i for i, (x, y) in enumerate(zip(ra, rb)) if x != y), None)
    ea = json.loads((a/'episode_evaluation.json').read_text()) if (a/'episode_evaluation.json').exists() else {}
    eb = json.loads((b/'episode_evaluation.json').read_text()) if (b/'episode_evaluation.json').exists() else {}
    outcome = lambda e: {k: e.get(k) for k in ('round_trip_success', 'beacon_success', 'home_success', 'disallowed_contact_samples')}
    report = dict(
        planning_records=(len(pa), len(pb)), planning_identical=pa == pb, first_planning_difference=first,
        first_planning_difference_frames=None if first is None else (pa[first]['frame'], pb[first]['frame']),
        requests=(len(ra), len(rb)), requests_identical=ra == rb, first_request_difference=first_request,
        physics_trace_identical=trace_hash(a) == trace_hash(b), outcome_identical=outcome(ea) == outcome(eb),
        outcome=(outcome(ea), outcome(eb)))
    report['identical'] = all(report[k] for k in ('planning_identical', 'requests_identical', 'physics_trace_identical', 'outcome_identical'))
    print(json.dumps(report, indent=1))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run_a')
    p.add_argument('run_b')
    a = p.parse_args()
    main(a.run_a, a.run_b)
