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
need `--allow-final-round`. The sealed set is never admitted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from lewm import decision_headroom_json_v42_development as output
from lewm.dev_harness_fixes_development import check_track_override, compose, fixes_for
from lewm.dev_readout_variants_development import ReadoutVariant, install
from lewm.eligible_floor_registration_development import bind
from lewm.navigation_capability_completed_support_development import CompletedSupportRuntimeMixin
from scripts import run_go2_c3v2_check_development as fresh
from scripts import run_go2_c3v3_round_development as round_sets
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    raise ValueError('development sets only; the sealed set is never admitted')


def model_loader(c3_decoder, c4_weights):
    def load_model(arm, protocol, root):
        model = owner.load_model(arm, protocol, root)
        if arm == 'C3' and c3_decoder:
            state = torch.load(c3_decoder, map_location='cpu', weights_only=False)
            variant = state.get('variant', 'base')
            readout = ReadoutVariant(model.readout.cpu(), past_frames=variant == 'past_frames', history=variant == 'history')
            readout.load_state_dict(state['readout'])
            if variant == 'base':
                model.readout.project.load_state_dict(readout.project.state_dict())
                model.readout.decode.load_state_dict(torch.nn.Sequential(readout.flatten, readout.hidden, readout.act, readout.out).state_dict())
                model.readout.to(next(model.predictor.parameters()).device).eval().requires_grad_(False)
            else:
                install(model, readout)
            model.readout_identity = dict(model.readout_identity, arm='dev_decoder', variant=variant, path=str(c3_decoder), sha256=sha(c3_decoder))
        if arm == 'C4' and c4_weights:
            state = torch.load(c4_weights, map_location='cpu', weights_only=False)
            model.predictor.load_state_dict(state['c4'] if 'c4' in state else state['model_state_dict'])
            model.predictor.eval().requires_grad_(False)
        return model
    return load_model


def main(arm, set_name, maze, episode, assignment, fixes, c3_decoder, c4_weights, allow_final_round, recovery=None):
    protocol = json.loads(owner.PROTOCOL.read_text())
    root = Path(protocol['output_root'])
    # Hash the code at start: a later edit must not relabel a run that loaded the earlier file.
    code_sha = dict(fixes_module_sha256=sha('lewm/dev_harness_fixes_development.py'), entry_sha256=sha(__file__))
    runtime_mixin = compose(fixes, CompletedSupportRuntimeMixin)
    base_runtime = owner.source.DenseReactiveNavigationRuntime if arm == 'C2' else owner.source.DenseNavigationRuntime
    check_track_override(type('Checked', (runtime_mixin, base_runtime), {}))
    try:
        bind(owner.run, episode_inputs=loader_for(set_name, maze, allow_final_round), load_model=model_loader(c3_decoder, c4_weights),
             StartupRecoveryRuntimeMixin=runtime_mixin)(arm, maze, episode, assignment)
    finally:
        destination = root/'runs'/assignment
        if destination.exists():
            output.install(root)
            owner.save(destination/'dev_run.json', dict(mode='development', controller=arm, set=set_name, maze=maze, episode=episode,
                fixes=sorted(fixes), recovery=recovery, c3_decoder=c3_decoder, c3_decoder_sha256=sha(c3_decoder) if c3_decoder else None,
                c4_weights=c4_weights, c4_weights_sha256=sha(c4_weights) if c4_weights else None, **code_sha))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--controller', required=True, choices=('C0', 'C1', 'C2', 'C3', 'C4'))
    p.add_argument('--set', required=True, choices=('dev_tune', 'validation', 'fresh_check', 'round'))
    p.add_argument('--maze', type=int, required=True)
    p.add_argument('--episode', type=int, default=0)
    p.add_argument('--assignment', required=True)
    p.add_argument('--fixes', default='')
    p.add_argument('--recovery', choices=('on', 'off'), help='named fix set: on = all development fixes, off = pose record only')
    p.add_argument('--c3-decoder')
    p.add_argument('--c4-weights')
    p.add_argument('--allow-final-round', action='store_true')
    a = p.parse_args()
    fixes = [f for f in a.fixes.split(',') if f]
    if a.recovery:
        if fixes:
            p.error('--recovery and --fixes are exclusive')
        fixes = fixes_for(a.recovery)
    main(a.controller, a.set, a.maze, a.episode, a.assignment, fixes, a.c3_decoder, a.c4_weights, a.allow_final_round, a.recovery)
