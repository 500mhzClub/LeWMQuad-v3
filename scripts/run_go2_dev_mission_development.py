"""One development mission on the V4 harness with optional trap fixes and development models.

Development mode (30 Sep 2026). The frozen V4 owner runs with up to three functions swapped by
`bind`:
- the startup runtime mixin, composed with the requested fixes from
  `lewm/dev_harness_fixes_development.py`;
- the episode loader;
- the model loader.
No frozen file is edited.

Sets: `dev_tune` (0–9) and `validation` (10–29) via the owner's loader; `fresh_check` (0–9);
`round` (C3-v3 round layouts 0–21). Round layouts 22–31 are held for the final run and
need `--allow-final-round`.

`prelim_test` (capability layouts 30–89, episode 0) is the set Andrew declassified on 1 Oct
2026 (formerly sets/sealed_test_v1, now sets/prelim_test_v1; AGENTS.md exception). Every file
is verified against the capability registry by hash before use. The loader relabels the role
to prelim_test_v1, so the owner records the real file's hash from its new location. The owner
refuses C0 on maze IDs of 20 or more, so the dev entry runs a copy of the owner's `run` in
which only that check also admits the preliminary IDs. No sealed set is ever admitted.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import torch

from lewm import decision_headroom_json_v42_development as output
from lewm.dev_harness_fixes_development import (DEFAULT_RECOVERY, START_REACH_M, TRACK_RADIUS_M, UNKNOWN_REACH_M, PessimisticUnknownMixin,
                                               check_track_override, compose, degradation_mixin, fixes_for, margin_mixin)
from lewm.dev_readout_variants_development import ReadoutVariant, install
from lewm.eligible_floor_registration_development import bind
from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
from scripts import run_go2_c3v2_check_development as fresh
from scripts import run_go2_c3v3_round_development as round_sets
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner


class DevBudget(owner.Budget):
    """The owner's resource checks without its 160-hour programme window.

    Development mode (Andrew, 30 Sep) dropped formal budget stops. The owner's window counts
    wall time from a stored origin (reached 132 h on 1 Oct) and would stop every mission at
    160 h. Everything else is the owner's check unchanged: recovery and workspace filesystem
    reserves, and the per-device VRAM reserve. No decision reads the budget.
    """

    def check(self, force=False):
        self.origin = time.time()
        return super().check(force)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


PRELIM, PRELIM_MAZES = 'prelim_test_v1', range(30, 90)
C0_LIMIT = "        if maze>=20:raise ValueError('C0 validation limited to ten lowest IDs')\n"


def prelim_inputs(root, maze, episode):
    """Preliminary-test maze (declassified 1 Oct 2026), hash-verified against the capability registry."""
    if maze not in PRELIM_MAZES or episode != 0:
        raise ValueError('preliminary-test maze 30-89, episode 0 required')
    registry = json.loads((root/'registry.json').read_text())
    entry = next(e for e in registry['entries'] if e['maze_id'] == maze)
    assert entry['role'] == 'sealed_test'
    paths = {}
    for key, record in (('maze', entry['maze']), ('episode', entry['episodes'][0])):
        path = Path(record['path'].replace('/sets/sealed_test_v1/', f'/sets/{PRELIM}/'))
        assert path.parent.name == PRELIM and sha(path) == record['sha256'], path
        paths[key] = path
    packet = json.loads(paths['episode'].read_text())
    original = json.loads(paths['maze'].read_text())
    assert packet['role'] == original['data_role'] == 'sealed_test'
    packet = packet | dict(role=PRELIM, registered_role='sealed_test', declassified='Andrew, 1 Oct 2026')
    original = original | dict(data_role=PRELIM)
    spec = original | dict(procedural_seed=packet['simulation_seed'],
                           geometry=original['geometry'] | dict(spawn_se2_world=packet['home_se2_world']))
    return spec, packet


def owner_run_for_prelim():
    """The owner's run, with only its C0 maze limit also admitting the preliminary IDs."""
    source = inspect.getsource(owner.run)
    assert source.count(C0_LIMIT) == 1, 'owner C0 limit line not found exactly once'
    source = source.replace(C0_LIMIT, "        if maze>=20 and maze not in PRELIM_C0_MAZES:raise ValueError('C0 validation limited to ten lowest IDs')\n")
    namespace = dict(owner.__dict__, PRELIM_C0_MAZES=PRELIM_MAZES)
    exec(compile(source, owner.__file__, 'exec'), namespace)
    return namespace['run']


def loader_for(set_name, maze, allow_final_round):
    if set_name == 'dev_tune':
        assert maze in range(10)
        return owner.episode_inputs
    if set_name == 'validation':
        assert maze in range(10, 30)
        return owner.episode_inputs
    if set_name == 'fresh_check':
        return fresh.episode_inputs
    if set_name == 'round':
        if maze >= 22 and not allow_final_round:
            raise ValueError('round layouts 22-31 are held for the final run')
        return round_sets.episode_inputs
    if set_name == 'prelim_test':
        return prelim_inputs
    raise ValueError('development and preliminary-test sets only; a sealed set is never admitted')


def model_loader(c3_decoder, c4_weights):
    def load_model(arm, protocol, root):
        model = owner.load_model(arm, protocol, root)
        if arm == 'C3' and c3_decoder:
            state = torch.load(c3_decoder, map_location='cpu', weights_only=False)
            if 'model_state_dict' in state and 'readout' not in state:
                # A deployed DenseVisualMotionReadout checkpoint (for example C3-v3, 85ab19ec).
                model.readout.load_state_dict(state['model_state_dict'])
                model.readout.eval().requires_grad_(False)
                model.readout_identity = dict(model.readout_identity, arm='deployed_checkpoint', path=str(c3_decoder), sha256=sha(c3_decoder))
                return model
            variant = state.get('variant', 'base')
            config = state.get('readout_config', {})
            readout = ReadoutVariant(model.readout.cpu(), past_frames=variant == 'past_frames', history=variant == 'history', **config)
            readout.load_state_dict(state['readout'])
            if variant == 'base' and readout.config == dict(proj=32, hidden=128, depth=0):
                model.readout.project.load_state_dict(readout.project.state_dict())
                model.readout.decode.load_state_dict(torch.nn.Sequential(readout.flatten, readout.hidden, readout.act, readout.out).state_dict())
                model.readout.to(next(model.predictor.parameters()).device).eval().requires_grad_(False)
            else:
                install(model, readout)
            model.readout_identity = dict(model.readout_identity, arm='dev_decoder', variant=variant, readout_config=readout.config, path=str(c3_decoder), sha256=sha(c3_decoder))
        if arm == 'C4' and c4_weights:
            state = torch.load(c4_weights, map_location='cpu', weights_only=False)
            model.predictor.load_state_dict(state['c4'] if 'c4' in state else state['model_state_dict'])
            model.predictor.eval().requires_grad_(False)
        return model
    return load_model


MARGINS_FILE = 'docs/go2_navigation_calibrated_margins_2026-10-02.json'


def clearance_margin(arm, level, path=MARGINS_FILE):
    """The controller's committed calibrated margin (calibrated-margin experiment, 2 October)."""
    bounds = json.loads(Path(path).read_text())
    return dict(level=level, margin_m=bounds['controllers'][arm][level]['margin_m'], bounds_file=path, bounds_sha256=sha(path))


def main(arm, set_name, maze, episode, assignment, fixes, c3_decoder, c4_weights, allow_final_round, recovery=None, degrade=None, margin=None,
         pessimistic=False):
    protocol = json.loads(owner.PROTOCOL.read_text())
    root = Path(protocol['output_root'])
    # Hash the code at start: a later edit must not relabel a run that loaded the earlier file.
    code_sha = dict(fixes_module_sha256=sha('lewm/dev_harness_fixes_development.py'), entry_sha256=sha(__file__))
    margin_record = clearance_margin(arm, margin) if margin else None
    extra = (((PessimisticUnknownMixin,) if pessimistic else ())+((margin_mixin(margin_record['margin_m'], margin),) if margin else ())
             +((degradation_mixin(degrade),) if degrade else ()))
    pessimistic_record = dict(reach_m=UNKNOWN_REACH_M, start_reach_disc_m=START_REACH_M, track_radius_m=TRACK_RADIUS_M) if pessimistic else None
    runtime_mixin = compose(fixes, CompletedSupportRuntimeMixin, extra=extra)
    base_runtime = owner.source.DenseReactiveNavigationRuntime if arm == 'C2' else owner.source.DenseNavigationRuntime
    check_track_override(type('Checked', (runtime_mixin, base_runtime), {}))
    try:
        run = owner_run_for_prelim() if set_name == 'prelim_test' else owner.run
        bind(run, episode_inputs=loader_for(set_name, maze, allow_final_round), load_model=model_loader(c3_decoder, c4_weights),
             StartupRecoveryRuntimeMixin=runtime_mixin, Budget=DevBudget)(arm, maze, episode, assignment)
    finally:
        destination = root/'runs'/assignment
        if destination.exists():
            output.install(root)
            owner.save(destination/'dev_run.json', dict(mode='development', controller=arm, set=set_name, maze=maze, episode=episode,
                fixes=sorted(fixes), recovery=recovery, forecast_degradation=degrade, clearance_margin=margin_record, pessimistic_unknown=pessimistic_record, c3_decoder=c3_decoder, c3_decoder_sha256=sha(c3_decoder) if c3_decoder else None,
                c4_weights=c4_weights, c4_weights_sha256=sha(c4_weights) if c4_weights else None,
                owner_run=('copy of owner.run with only the C0 maze-ID limit relaxed for prelim_test IDs 30-89'
                           if set_name == 'prelim_test' else 'owner.run'), **code_sha))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--controller', required=True, choices=('C0', 'C1', 'C2', 'C3', 'C4'))
    p.add_argument('--set', required=True, choices=('dev_tune', 'validation', 'fresh_check', 'round', 'prelim_test'))
    p.add_argument('--maze', type=int, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--assignment', required=True)
    p.add_argument('--fixes', default='')
    p.add_argument('--recovery', choices=('on', 'off'), help='named fix set (default off since 2 Oct): off = harness fixes + pose record; on = also the five recovery fixes')
    p.add_argument('--degrade', help='forecast degradation for the sensitivity experiment: scale:S or noise:E_mm')
    p.add_argument('--margin', choices=('p95', 'p99'), help='calibrated clearance margin (committed conformal bound for this controller)')
    p.add_argument('--pessimistic-unknown', action='store_true', help='never-observed cells within reach count as occupied (stage 2)')
    p.add_argument('--c3-decoder')
    p.add_argument('--c4-weights')
    p.add_argument('--allow-final-round', action='store_true')
    a = p.parse_args()
    fixes = [f for f in a.fixes.split(',') if f]
    if a.recovery and fixes:
        p.error('--recovery and --fixes are exclusive')
    recovery = a.recovery or (None if fixes else DEFAULT_RECOVERY)
    if recovery:
        fixes = fixes_for(recovery)
    main(a.controller, a.set, a.maze, a.episode, a.assignment, fixes, a.c3_decoder, a.c4_weights, a.allow_final_round, recovery, a.degrade, a.margin, a.pessimistic_unknown)
