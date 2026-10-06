"""PRELIMINARY: dynamics stage 1, open-loop friction characterisation (Andrew, 3 October 2026).

Rule (docs/go2_navigation_dynamics_perturbation_plan_2026-10-02.md, "Stage 1 mu selection, made concrete"; fixed before
any characterisation run):
- grid mu = 1.0 (nominal reference), 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15; three repeats each (spawn yaw 0, +0.3,
  -0.3 rad, distinct seeds);
- the robot drives one fixed tape of the harness's six primitives, each dispatched as the harness does (the primitive's
  first command held for a 400-ms commit, decisions every 400 ms with a 300-ms committed prefix), in a 12-m square room
  (walls 6 m away; the scene builder needs a bounded wall roster). Friction: floor and all 27 robot geometries
  (lewm/dev_dynamics_friction_development.py), so wall pairs stay nominal;
- fall: base height below 0.15 m, or a non-foot contact (a disallowed contact; with walls far away, a non-foot part
  touching the floor) lasting more than 0.5 s;
- stumble: any disallowed contact; or a base-height dip more than 5 cm below the nominal gait's minimum; or roll or pitch
  beyond the nominal gait's maximum plus 10 degrees;
- clearly above nominal: C1's open-loop 700-ms forecast error from commands alone (the deployed command-history model,
  fed exactly the runtime's inputs: the last four policy packets' applied-command records and the committed prefix),
  median over all decisions, at least twice nominal and above the nominal 95th percentile;
- choice: the lowest mu that is stable with one grid step of margin (the next lower level also stable) and clearly above
  nominal; reported with error by movement type, realised/commanded speed and yaw rate, and the stability margin.
  If no level qualifies, stage 1 stops for Andrew.

Usage: characterise_go2_friction_open_loop_development.py --out DIR [--workers N] [--levels ...] [--summarise-only]
"""
import argparse
import json
import math
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

GRID = (1.0, .8, .6, .5, .4, .3, .25, .2, .15)
REPEATS = ((0, 0.), (1, .3), (2, -.3))
TICK_S, TICKS_PER_FRAME, FRAMES_PER_COMMIT, DELAY_FRAMES = .02, 5, 4, 3
COMMIT_TICKS, FIRST_DISPATCH_TICK = TICKS_PER_FRAME*FRAMES_PER_COMMIT, TICKS_PER_FRAME*DELAY_FRAMES
HORIZON_STEPS = 7
FALL_HEIGHT_M, FALL_CONTACT_S, DIP_M, TILT_DEG = .15, .5, .05, 10.
TAPE = (['hold']*4+['forward']*6                       # rest start, then cruise
        + ['hold']*3+['left_arc']*5                    # rest start into a steady arc
        + ['hold']*3+['left_turn']*6                   # in-place turn
        + ['hold']*2+['right_arc']*5
        + ['hold']*2+['right_turn']*6
        + ['forward', 'left_arc', 'forward', 'right_turn', 'forward', 'left_turn', 'right_arc', 'forward']*2  # switches
        + ['forward']*5+['hold']*3)
LABEL = 'PRELIMINARY (open-loop characterisation; development mode)'


def arena_spec(owner, root, seed_offset, yaw):
    spec, _ = owner.episode_inputs(root, 0, 0)
    w0 = spec['geometry']['wall_boxes'][0]
    z, h = w0['centre_xyz'][2], w0['size_xyz'][2]
    walls = [w0 | dict(wall_id=name, centre_xyz=c, size_xyz=s, yaw_rad=0.)
             for name, c, s in (('arena_e', [6., 0., z], [.08, 12., h]), ('arena_w', [-6., 0., z], [.08, 12., h]),
                                ('arena_n', [0., 6., z], [12., .08, h]), ('arena_s', [0., -6., z], [12., .08, h]))]
    return spec | dict(procedural_seed=int(spec['procedural_seed'])+seed_offset,
                       geometry=spec['geometry'] | dict(wall_boxes=walls, spawn_se2_world=[-1., 0., yaw]))


def command_at(tick):
    from lewm.geometry_progress_pilot_development import candidate_commands
    if tick < FIRST_DISPATCH_TICK:
        return [0., 0., 0.]
    j = (tick-FIRST_DISPATCH_TICK)//COMMIT_TICKS
    return list(candidate_commands(TAPE[j] if j < len(TAPE) else 'hold')[0])


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def roll_pitch(q):
    x, y, z, w = q
    return math.degrees(math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))), math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x)))))


def run(job):
    mu, repeat, yaw, out = job
    from collections import deque
    from lewm.dev_dynamics_friction_development import friction_session
    from lewm.geometry_progress_pilot_development import ACTIONS
    from lewm.short_pulse_navigation_runtime_development import COMMAND_FIT, command_predictions, past_commands
    from lewm.terminal_translation_pulse_development import command_sequences
    from scripts.build_go2_dev_c3_feature_cache_development import category
    from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
    directory = Path(out)/f'mu{mu:.2f}_r{repeat}'
    (directory/'native').mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
    spec = arena_spec(owner, root, repeat, yaw)
    with np.load(COMMAND_FIT/'command_only.npz', allow_pickle=False) as arrays:
        model = {k: arrays[k].copy() for k in ('mean', 'scale', 'bias', 'coefficient')}
    owner.source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    session = friction_session(owner.make_session, mu)(spec, directory/'native')
    session.install_contact_identity()
    # As the owner does before settling: the locomotion checkpoint's PD gains (kp 20, kv 0.5), not the defaults.
    gains = owner.source.configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(),
                                         session.ctx.policy.env_cfg, 'checkpoint')
    session.settle_recorded()
    settled_samples = len(session.samples)
    session.phase = 2
    history, decisions, requested = deque(maxlen=4), [], []
    total_ticks = FIRST_DISPATCH_TICK+COMMIT_TICKS*len(TAPE)
    for tick in range(total_ticks):
        if tick % TICKS_PER_FRAME == 0:
            policy = session.sensor_packets()[0]
            history.append(policy)
            frame = tick//TICKS_PER_FRAME
            if frame % FRAMES_PER_COMMIT == 0 and len(history) == 4:
                j = frame//FRAMES_PER_COMMIT
                if j < len(TAPE):
                    now = policy['sensor_state']['decision_ns']
                    prefix = [command_at(tick+k*TICKS_PER_FRAME) for k in range(DELAY_FRAMES)]
                    forecast = command_predictions(model, past_commands(list(history), now), command_sequences(prefix, pulse=False))
                    decisions.append(dict(commit=j, action=TAPE[j], tick=tick, decision_ns=int(now), prefix=prefix,
                                          predicted=forecast[ACTIONS.index(TAPE[j]), HORIZON_STEPS-1, :2].tolist()))
        command = command_at(tick)
        requested.append(command)
        session.command_policy_step(command)
    samples = session.samples[settled_samples:]
    stamps = np.asarray([s['timestamp_s'] for s in session.samples])
    pose = np.asarray([s['base_pose_world'] for s in session.samples])
    rows = []
    for d in decisions:
        t = d['decision_ns']/1e9
        i, k = int(np.searchsorted(stamps, t-1e-6)), int(np.searchsorted(stamps, t+HORIZON_STEPS*.1-1e-6))
        if k >= len(stamps):
            continue
        yaw0 = yaw_of(pose[i, 3:])
        dxy = pose[k, :2]-pose[i, :2]
        true = [math.cos(yaw0)*dxy[0]+math.sin(yaw0)*dxy[1], -math.sin(yaw0)*dxy[0]+math.cos(yaw0)*dxy[1]]
        tape = [command_at(d['tick']+m*TICKS_PER_FRAME+2) for m in range(8)]
        past = [command_at(d['tick']+m*TICKS_PER_FRAME+2) for m in range(-15, 0)]
        realised_yaw = math.atan2(math.sin(yaw_of(pose[k, 3:])-yaw0), math.cos(yaw_of(pose[k, 3:])-yaw0))
        commanded = np.asarray(d['prefix']+[command_at(d['tick']+FIRST_DISPATCH_TICK)]*4)
        rows.append(dict(commit=d['commit'], action=d['action'], category=category(tape, past), predicted=d['predicted'], true=true,
                         error_m=float(np.hypot(*(np.asarray(d['predicted'])-true))),
                         commanded_xy_m=float(np.sum(commanded[:, 0])*.1), commanded_yaw_rad=float(np.sum(commanded[:, 2])*.1),
                         realised_xy_m=float(np.hypot(*true)), realised_yaw_rad=realised_yaw))
    z = np.asarray([s['base_pose_world'][2] for s in samples])
    rp = np.asarray([roll_pitch(s['base_pose_world'][3:]) for s in samples])
    events = session.contact_events[-len(samples):] if len(session.contact_events) >= len(samples) else session.contact_events
    bad = np.asarray([bool(e['disallowed_contacts']) for e in events])
    longest, current = 0, 0
    for b in bad:
        current = current+1 if b else 0
        longest = max(longest, current)
    dt = float(np.median(np.diff(stamps))) if len(stamps) > 1 else .002
    contact_bodies = sorted({str(c.get('link_name', c.get('link_a', '?')))+'|'+str(c.get('object_id', c.get('other', '?')))
                             for e in events for c in e['disallowed_contacts']})[:20]
    result = dict(label=LABEL, mu=mu, repeat=repeat, spawn_yaw_rad=yaw, friction=session.dynamics_friction, gains=gains, tape=TAPE,
                  decisions=rows, base_height_min_m=float(z.min()), base_height_median_m=float(np.median(z)),
                  roll_max_deg=float(np.abs(rp[:, 0]).max()), pitch_max_deg=float(np.abs(rp[:, 1]).max()),
                  disallowed_contact_samples=int(bad.sum()), longest_disallowed_contact_s=float(longest*dt), contact_bodies=contact_bodies,
                  final_pose=pose[-1].tolist(), wall_s=time.monotonic()-started)
    (directory/'result.json').write_text(json.dumps(result, indent=1)+'\n')
    return str(directory/'result.json')


def summarise(out):
    results = [json.loads(p.read_text()) for p in sorted(Path(out).glob('mu*_r*/result.json'))]
    by = {}
    for r in results:
        by.setdefault(r['mu'], []).append(r)
    nominal = by.get(1.0)
    if not nominal:
        raise SystemExit('nominal (mu = 1.0) runs required')
    nom_err = np.asarray([d['error_m'] for r in nominal for d in r['decisions']])
    nom_z = min(r['base_height_min_m'] for r in nominal)
    nom_roll, nom_pitch = max(r['roll_max_deg'] for r in nominal), max(r['pitch_max_deg'] for r in nominal)
    base = dict(median_mm=1000*float(np.median(nom_err)), p95_mm=1000*float(np.percentile(nom_err, 95)), z_min_m=nom_z,
                roll_max_deg=nom_roll, pitch_max_deg=nom_pitch)
    table = []
    for mu in sorted(by, reverse=True):
        rs = by[mu]
        err = np.asarray([d['error_m'] for r in rs for d in r['decisions']])
        fall = any(r['base_height_min_m'] < FALL_HEIGHT_M or r['longest_disallowed_contact_s'] > FALL_CONTACT_S for r in rs)
        stumble = any(r['disallowed_contact_samples'] > 0 or r['base_height_min_m'] < nom_z-DIP_M
                      or r['roll_max_deg'] > nom_roll+TILT_DEG or r['pitch_max_deg'] > nom_pitch+TILT_DEG for r in rs)
        cats = {}
        for c in ('hold', 'rest_start', 'turn', 'cruise', 'arc_steady', 'switch'):
            ds = [d for r in rs for d in r['decisions'] if d['category'] == c]
            if ds:
                moving = [d for d in ds if d['realised_xy_m'] >= .01]
                cats[c] = dict(n=len(ds), median_error_mm=1000*float(np.median([d['error_m'] for d in ds])),
                               median_ratio=float(np.median([np.hypot(*d['predicted'])/d['realised_xy_m'] for d in moving])) if moving else None)
        translate = [d for r in rs for d in r['decisions'] if d['commanded_xy_m'] > .02]
        turning = [d for r in rs for d in r['decisions'] if abs(d['commanded_yaw_rad']) > .05]
        table.append(dict(mu=mu, decisions=int(len(err)), median_mm=1000*float(np.median(err)), p95_mm=1000*float(np.percentile(err, 95)),
                          fall=fall, stumble=stumble, stable=not fall and not stumble,
                          z_min_m=min(r['base_height_min_m'] for r in rs), roll_max_deg=max(r['roll_max_deg'] for r in rs),
                          pitch_max_deg=max(r['pitch_max_deg'] for r in rs), disallowed_contact_samples=sum(r['disallowed_contact_samples'] for r in rs),
                          speed_ratio=float(np.median([d['realised_xy_m']/d['commanded_xy_m'] for d in translate])) if translate else None,
                          yaw_rate_ratio=float(np.median([d['realised_yaw_rad']/d['commanded_yaw_rad'] for d in turning])) if turning else None,
                          by_category=cats))
    levels = [row['mu'] for row in table]
    for row in table:
        row['clearly_above_nominal'] = bool(row['median_mm'] >= 2*base['median_mm'] and row['median_mm'] > base['p95_mm'])
        lower = [r for r in table if r['mu'] < row['mu']]
        row['next_lower_stable'] = bool(lower and max(lower, key=lambda r: r['mu'])['stable'])
        row['qualifies'] = bool(row['mu'] < 1. and row['stable'] and row['next_lower_stable'] and row['clearly_above_nominal'])
    chosen = min((r for r in table if r['qualifies']), key=lambda r: r['mu'], default=None)
    summary = dict(label=LABEL, nominal=base, levels=levels, table=table, chosen_mu=None if chosen is None else chosen['mu'],
                   reason=None if chosen is None else
                   (f"lowest mu stable with one grid step of margin (mu {max((r['mu'] for r in table if r['mu'] < chosen['mu']), default=None)} also stable) "
                    f"and C1's command-only 700-ms error clearly above nominal: median {chosen['median_mm']:.1f} mm vs nominal "
                    f"{base['median_mm']:.1f} mm (x{chosen['median_mm']/base['median_mm']:.1f}) and above nominal p95 {base['p95_mm']:.1f} mm"))
    (Path(out)/'summary.json').write_text(json.dumps(summary, indent=1)+'\n')
    lines = [f'**Stage 1 open-loop friction characterisation. {LABEL}**', '',
             f"Nominal (mu = 1.0): C1 command-only 700-ms error median {base['median_mm']:.1f} mm, p95 {base['p95_mm']:.1f} mm; "
             f"base height min {100*base['z_min_m']:.1f} cm; roll max {base['roll_max_deg']:.1f} deg, pitch max {base['pitch_max_deg']:.1f} deg.", '',
             '| mu | Decisions | C1 error median · p95 (mm) | Clearly above nominal | Fall | Stumble | Next lower stable | Qualifies | '
             'Speed ratio | Yaw-rate ratio | Base height min (cm) | Roll · pitch max (deg) | Non-foot contact samples |',
             '|---:|---:|---|---|---|---|---|---|---:|---:|---:|---|---:|']
    for r in table:
        f = lambda v: '-' if v is None else f'{v:.2f}'
        lines.append(f"| {r['mu']:.2f} | {r['decisions']} | {r['median_mm']:.1f} · {r['p95_mm']:.1f} | {r['clearly_above_nominal']} | {r['fall']} | {r['stumble']} | "
                     f"{r['next_lower_stable']} | {r['qualifies']} | {f(r['speed_ratio'])} | {f(r['yaw_rate_ratio'])} | {100*r['z_min_m']:.1f} | "
                     f"{r['roll_max_deg']:.1f} · {r['pitch_max_deg']:.1f} | {r['disallowed_contact_samples']} |")
    lines += ['', '**C1 command-only 700-ms error by movement type (median mm · median predicted/true).**', '',
              '| mu | '+' | '.join(('hold', 'rest_start', 'turn', 'cruise', 'arc_steady', 'switch'))+' |', '|---:|'+'---|'*6]
    for r in table:
        cells = []
        for c in ('hold', 'rest_start', 'turn', 'cruise', 'arc_steady', 'switch'):
            v = r['by_category'].get(c)
            cells.append('-' if v is None else f"{v['median_error_mm']:.1f} · {'-' if v['median_ratio'] is None else f'{v['median_ratio']:.2f}'} (n {v['n']})")
        lines.append(f"| {r['mu']:.2f} | "+' | '.join(cells)+' |')
    lines += ['', f"**Chosen mu: {summary['chosen_mu']}.** {summary['reason'] or 'No level qualifies: stage 1 stops for Andrew.'}"]
    (Path(out)/'summary.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main(out, workers, levels, summarise_only):
    if not summarise_only:
        Path(out).mkdir(parents=True, exist_ok=True)
        jobs = [(mu, repeat, yaw, out) for mu in levels for repeat, yaw in REPEATS]
        with get_context('spawn').Pool(workers) as pool:
            for path in pool.imap_unordered(run, jobs):
                print('done', path, flush=True)
    summarise(out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--levels', type=float, nargs='*', default=list(GRID))
    p.add_argument('--summarise-only', action='store_true')
    a = p.parse_args()
    main(a.out, a.workers, a.levels, a.summarise_only)
