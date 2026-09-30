"""Development cohort tables: success next to every recovery intervention (Andrew, 30 Sep 2026).

Recovery must not hide weak prediction, so every table reports, per mission and per
controller, how often each development fix intervened alongside success, contacts and wall
clearance:
- deadlock escapes (selection overrides by `deadlock`; by kind: in-place turn, reactive
  translation, reactive turn toward clearance, scripted back-up);
- stall reroutes (frontier exclusions by `stall`, from worker.log);
- latch timeouts (`latch` releases) and cooldown suppressions;
- terminal spin breaks (`terminal`, C2 reactive only);
- scripted back-ups (`backup`): episodes started, and aborted steps;
- pose corrections (Andrew, 30 Sep evening; not a fix, a property of the pose pipeline): changes
  of the published registered pose between consecutive 100-ms frames beyond the Go2's physical
  limits (more than POSE_JUMP_M or POSE_JUMP_RAD; the platform can move at most 3 cm and
  0.05 rad in 100 ms), plus frames published in the floor-transport re-anchoring mode.
Counts come from each run's planning log, worker log and published poses, so older cohorts are
covered too.

Usage: summarise_go2_dev_cohorts_development.py NAME [NAME ...] [--json OUT]
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
POSE_JUMP_M, POSE_JUMP_RAD = .05, .15


def pose_corrections(run):
    path = run/'poses.json'
    if not path.exists():
        return dict(pose_corrections=None, largest_pose_correction_m=None, floor_transport_frames=None)
    jumps, largest, transport, previous = 0, 0., 0, None
    for row in json.loads(path.read_text()):
        pose = row['registered_pose']
        transport += pose['mode'] == 'measured_visual_floor_transport'
        x, y = pose['position_initial_body_m'][:2]
        R = pose['rotation_initial_body_from_current_body']
        yaw = math.atan2(R[1][0], R[0][0])
        if previous is not None and row['frame'] == previous[0]+1:
            step = math.hypot(x-previous[1], y-previous[2])
            turn = abs(math.atan2(math.sin(yaw-previous[3]), math.cos(yaw-previous[3])))
            if step > POSE_JUMP_M or turn > POSE_JUMP_RAD:
                jumps += 1
                largest = max(largest, step)
        previous = (row['frame'], x, y, yaw)
    return dict(pose_corrections=jumps, largest_pose_correction_m=largest, floor_transport_frames=transport)


def interventions(run):
    rows = [r for r in json.loads((run/'planning.json').read_text()) if 'selection' in r] if (run/'planning.json').exists() else []
    kinds = Counter()
    latch = Counter()
    spin = 0
    backups = Counter()
    for r in rows:
        s = r['selection']
        if s.get('dev_backup'):
            backups['steps'] += 1
            backups['episodes'] += s['dev_backup']['step'] == 1
        backups['aborted'] += bool(s.get('dev_backup_aborted'))
        escape = s.get('dev_deadlock_escape')
        if escape:
            kinds[escape.get('kind', 'turn')] += 1
        event = (s.get('clearance_turn') or {}).get('event')
        if event == 'DEV_LATCH_TIMEOUT_RELEASED':
            latch['timeouts'] += 1
        elif event == 'DEV_LATCH_SUPPRESSED_COOLDOWN':
            latch['cooldown_suppressions'] += 1
        spin += bool(s.get('dev_terminal_spin_break'))
    log = run/'worker.log'
    events = [json.loads(line)['dev_stall'] for line in log.read_text(errors='replace').splitlines()
              if line.startswith('{"dev_stall"')] if log.exists() else []
    # A stall answered by a back-up is counted under back-ups; an undone exclusion is not a reroute.
    stalls = sum(1 for e in events if e.get('remedy') not in ('backup', 'exclusion_undone_no_other_route'))
    return dict(decisions=len(rows), deadlock_escapes=sum(kinds.values()), escape_kinds=dict(kinds), stall_reroutes=stalls,
                latch_timeouts=latch['timeouts'], latch_cooldown_suppressions=latch['cooldown_suppressions'], terminal_spin_breaks=spin,
                backups=backups['episodes'], backup_steps=backups['steps'], backups_aborted=backups['aborted'])


def mission_row(assignment, controller):
    run = BASE/'runs'/assignment
    row = dict(assignment=assignment, controller=controller)
    evaluation = run/'episode_evaluation.json'
    if evaluation.exists():
        ev = json.loads(evaluation.read_text())
        s = ev['safety']
        row.update(round_trip=ev['round_trip_success'], beacon=ev['beacon_success'], home=ev['home_success'],
                   contacts=ev['disallowed_contact_samples'], hard=s['hard']['confirmed_violation_samples'],
                   operating=s['operating']['confirmed_violation_samples'], min_clearance_m=s['hard']['minimum_separation_lower_m'],
                   source_error=ev['source_error'])
    else:
        row.update(round_trip=None, unread=True)
    return row | interventions(run) | pose_corrections(run)


def cohort_rows(name):
    config = json.loads((BASE/'dev_cohorts'/name/'config.json').read_text())
    return [mission_row(assignment, arm) | dict(cohort=name, fixes=config['fixes'])
            for arm, _set, _maze, _episode, assignment in config['plan'] if (BASE/'runs'/assignment).exists()]


def fmt(value, digits=3):
    return '-' if value is None else (f'{value:.{digits}f}' if isinstance(value, float) else str(value))


def tables(rows):
    lines = ['| Mission | Ctrl | Round trip | Deadlock escapes | Stall reroutes | Back-ups | Latch timeouts | Spin breaks | Pose corrections | Contacts | Hard | Min clearance (m) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        kinds = ', '.join(f'{k} {v}' for k, v in sorted(r['escape_kinds'].items()))
        lines.append(f"| {r['assignment']} | {r['controller']} | {fmt(r['round_trip'])} | {r['deadlock_escapes']}{f' ({kinds})' if kinds else ''} "
                     f"| {r['stall_reroutes']} | {r['backups']} | {r['latch_timeouts']} | {r['terminal_spin_breaks']} | {fmt(r['pose_corrections'])} | {fmt(r.get('contacts'))} | {fmt(r.get('hard'))} "
                     f"| {fmt(r.get('min_clearance_m'))} |")
    by = defaultdict(list)
    for r in rows:
        by[r['controller']].append(r)
    lines += ['', '| Ctrl | Missions read | Round trips | Missions with any recovery | Deadlock escapes | Stall reroutes | Back-ups | Latch timeouts | Spin breaks | Pose corrections | Contacts | Min clearance (m) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ctrl, rs in sorted(by.items()):
        read = [r for r in rs if r['round_trip'] is not None]
        helped = sum(1 for r in rs if r['deadlock_escapes'] or r['stall_reroutes'] or r['backups'] or r['latch_timeouts'] or r['terminal_spin_breaks'])
        clearances = [r['min_clearance_m'] for r in read if r.get('min_clearance_m') is not None]
        lines.append(f"| {ctrl} | {len(read)} | {sum(bool(r['round_trip']) for r in read)} | {helped} | {sum(r['deadlock_escapes'] for r in rs)} "
                     f"| {sum(r['stall_reroutes'] for r in rs)} | {sum(r['backups'] for r in rs)} | {sum(r['latch_timeouts'] for r in rs)} | {sum(r['terminal_spin_breaks'] for r in rs)} "
                     f"| {sum(r['pose_corrections'] or 0 for r in rs)} "
                     f"| {sum(r.get('contacts') or 0 for r in read)} | {fmt(min(clearances) if clearances else None)} |")
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('names', nargs='+')
    p.add_argument('--json')
    a = p.parse_args()
    rows = [r for name in a.names for r in cohort_rows(name)]
    print(tables(rows))
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
