"""Real-session test of the stage-2 patch hook (lewm/dev_dynamics_patches_development.py).

1. Placement on prelim_test_v1 mazes 30-49 (episode 0): runs of >= 2 interior route cells, 20-40 % coverage where the
   route allows, start and beacon cells never patched.
2. dev_tune maze 0, episode 0, with a test patch on the cell the robot faces at spawn (-1, 2), mu_p = 0.2:
   marked -> the floor quads in the patch are recoloured and appear in the policy RGB frame (saved for inspection);
   unmarked -> floor colours unchanged; both -> while walking forward onto the patch, every foot-floor contact's solver
   friction is mu_p over the patch and 1.0 off it (friction field). The speed effect is measured in the open-arena
   characterisation, not here (a single 1.3-m cell gives only about 2 s on the patch).
Run: PYTHONPATH=.:lewm_genesis:lewm_worlds python -m scripts.test_go2_dev_dynamics_patches_development --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np

from lewm.dev_dynamics_patches_development import on_patch, patch_session, place_patches
from scripts import run_go2_dev_mission_development as dev
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner


def placement_table(root):
    rows = []
    for maze in range(30, 50):
        spec, packet = dev.prelim_inputs(root, maze, 0)
        p = place_patches(spec, packet, seed=2026100300+maze)
        route = [tuple(c) for c in p['route']]
        cells = [tuple(c) for c in p['patch_cells']]
        assert route[0] not in cells and route[-1] not in cells
        idx = sorted(route.index(c) for c in cells)
        runs, run = [], [idx[0]] if idx else []
        for a, b in zip(idx, idx[1:]):
            if b == a+1:
                run.append(b)
            else:
                runs.append(run)
                run = [b]
        if run:
            runs.append(run)
        assert all(len(r) >= 2 for r in runs), (maze, runs)
        rows.append((maze, len(route), len(cells), round(p['coverage'], 2), [len(r) for r in runs]))
    for r in rows:
        print('maze', r[0], 'route cells', r[1], 'patch cells', r[2], 'coverage', r[3], 'runs', r[4])
    return rows


def floor_contacts(field):
    st = field.solver.collider._collider_state
    n = int(st.n_contacts.to_numpy()[0])
    ga, gb = st.contact_data.geom_a.to_numpy()[:n, 0], st.contact_data.geom_b.to_numpy()[:n, 0]
    fr = st.contact_data.friction.to_numpy()[:n, 0]
    feet = {int(g.idx): k for k, g in enumerate(field.feet)}
    out = []
    for a, b, x in zip(ga, gb, fr):
        if field.floor[0] in (a, b):
            other = int(b if a == field.floor[0] else a)
            if other in feet:
                out.append((feet[other], float(x)))
    return out


def walk(session, field, seconds):
    """Walk forward; return per-tick (time, x, y, feet on patch) and every foot-floor contact's solver friction."""
    xs, contacts = [], {True: [], False: []}
    for k in range(int(seconds/.02)):
        session.phase = 2
        if k % 5 == 0 and k:  # the start frame was the k = 0 packet
            session.sensor_packets()
        session.command_policy_step([.2, 0., 0.])
        s = session.samples[-1]
        on = field.last_on  # the on/off state the field applied during this step
        for foot, friction in floor_contacts(field):
            contacts[bool(on[foot])].append(friction)
        xs.append((s['timestamp_s'], *s['base_pose_world'][:2], int(on.sum())))
    return np.asarray(xs), contacts


def session_check(root, out, marked):
    spec, packet = owner.episode_inputs(root, 0, 0)
    placement = dict(route=[], patch_cells=[[-1, 2]], coverage=None, seed=None, pitch_m=1.3, rule='test patch in view at spawn')
    directory = out/('marked' if marked else 'unmarked')/'native'
    directory.mkdir(parents=True)
    owner.source.initialize_genesis(backend='cpu', seed=spec['procedural_seed'], logging_level='warning')
    session = patch_session(owner.make_session, placement, .2, marked)(spec, directory)
    record = json.loads((directory/'dynamics_patches.json').read_text())
    assert (record['marked_floor_quads'] > 50) == marked, record['marked_floor_quads']
    print('marked floor quads', record['marked_floor_quads'])
    session.install_contact_identity()
    owner.source.configure_gains(session.ctx.build.robot, session.ctx.runner._leg_dof_idx.tolist(), session.ctx.policy.env_cfg, 'checkpoint')
    session.settle_recorded()
    image = np.asarray(session.sensor_packets()[0]['image']['rgb'])
    np.save(out/f"{'marked' if marked else 'unmarked'}_start_image.npy", image)
    try:
        from PIL import Image
        Image.fromarray(image.astype(np.uint8)).save(out/f"{'marked' if marked else 'unmarked'}_start_image.png")
    except Exception as error:
        print('image save failed', error)
    xs, contacts = walk(session, session.dynamics_patches, 8.)
    on, off = np.asarray(contacts[True]), np.asarray(contacts[False])
    print('marked' if marked else 'unmarked', 'foot-floor contacts on patch', len(on), 'friction', np.unique(on.round(3)).tolist(),
          '| off patch', len(off), 'friction', np.unique(off.round(3)).tolist(), '| ticks by feet on patch', np.bincount(xs[:, 3].astype(int), minlength=5).tolist())
    assert len(on) > 50 and len(off) > 50
    assert np.allclose(on, .2, atol=1e-6) and np.allclose(off, 1., atol=1e-6), 'solver contact friction does not follow the field'
    return image


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
    placement_table(root)
    marked = session_check(root, out, True)
    unmarked = session_check(root, out, False)
    diff = np.abs(marked.astype(int)-unmarked.astype(int)).sum(axis=-1) > 30
    print('pixels changed by the marker', int(diff.sum()), 'of', diff.size)
    assert diff.sum() > 500, 'marker not visible in the start frame'
    print('ok')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    main(p.parse_args().out)
