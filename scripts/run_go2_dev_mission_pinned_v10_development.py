"""Mission entry for pinned launches, v10 (Andrew, 5 October 2026).

A copy of scripts/run_go2_dev_mission_pinned_v9_development.py (unchanged), except that --controller C1A uses C1A v3
(lewm/dev_c1_adaptive_travel_v3_development.py), which keeps translation and rotation ratios per movement type.
Everything else, including patches3 and the stage-2 sets, is as in v9.
"""
import importlib
import json
import os
from pathlib import Path
import runpy
import sys

from scripts.run_go2_dev_mission_pinned_development import ENTRY, REPO, verify

STAGE2_ENTRY = 'scripts/run_go2_dev_mission_stage2_development.py'

HARNESSES = {'reserve_exit_v1': 'lewm.dev_harness_reserve_exit_development',
             'reserve_exit_v1_1': 'lewm.dev_harness_reserve_exit_v1_1_development',
             'reserve_exit_v2': 'lewm.dev_harness_reserve_exit_v2_development'}


def parse_dynamics(text):
    if not text:
        return None
    kind, _, rest = text.partition(':')
    if kind == 'friction':
        return dict(perturbation='uniform_floor_friction', mu=float(rest))
    if kind in ('patches', 'patches2', 'patches3'):
        mu, _, mark = rest.partition(':')
        if mark not in ('marked', 'unmarked'):
            raise SystemExit(f'patches need :marked or :unmarked: {text!r}')
        return dict(perturbation='low_friction_patches', mu=float(mu), marked=mark == 'marked',
                    placement_version='v1' if kind == 'patches' else 'v2',
                    patches_version={'patches': 'v1', 'patches2': 'v2', 'patches3': 'v3'}[kind])
    raise SystemExit(f'unknown dynamics perturbation: {text!r}')


def argument(name, default=None):
    return sys.argv[sys.argv.index(name)+1] if name in sys.argv else default


def main():
    pin = json.loads(os.environ['LEWM_LAUNCH_PIN'])
    verify(pin)
    harness = os.environ.get('LEWM_HARNESS')
    if harness not in HARNESSES:
        raise SystemExit(f'unknown harness version: {harness!r}')
    dynamics = parse_dynamics(os.environ.get('LEWM_DYNAMICS'))
    arm = sys.argv[sys.argv.index('--controller')+1]
    variant = arm
    extra_mixins = ()
    if arm == 'C1A':
        from lewm.dev_c1_adaptive_travel_v3_development import AdaptiveTravelMixin
        arm = 'C1'
        sys.argv[sys.argv.index('--controller')+1] = 'C1'
        extra_mixins = (AdaptiveTravelMixin,)
    from lewm import dev_harness_fixes_development as fixes
    version = importlib.import_module(HARNESSES[harness])
    from scripts import run_go2_navigation_capability_completed_support_v4_development as owner
    if version.HARNESS != harness:
        raise SystemExit(f'harness module is {version.HARNESS}, launch asked for {harness}')
    original = fixes.compose

    def compose(fix_names, base, extra=()):
        return original(fix_names, base, extra=extra_mixins+tuple(version.mixins_for(arm))+tuple(extra))
    fixes.compose = compose  # this process only; the entry imports it by name
    if dynamics and dynamics['perturbation'] == 'uniform_floor_friction':
        from lewm.dev_dynamics_friction_development import friction_session
        owner.make_session = friction_session(owner.make_session, dynamics['mu'])  # read when the entry binds owner.run
    elif dynamics and dynamics['patches_version'] == 'v3':
        from lewm.dev_dynamics_patches_v3_development import patch_session, place_patches, placement_seed
        from scripts import run_go2_dev_mission_stage2_development as dev  # the stage-2 loader wraps the unchanged one
        set_name, maze, episode = argument('--set'), int(argument('--maze')), int(argument('--episode', 0))
        root = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
        spec, packet = dev.loader_for(set_name, maze, '--allow-final-round' in sys.argv)(root, maze, episode)
        dynamics['placement'] = place_patches(spec, packet, seed=placement_seed(set_name, maze, episode))
        owner.make_session = patch_session(owner.make_session, dynamics['placement'], dynamics['mu'], dynamics['marked'])
    elif dynamics and dynamics['placement_version'] == 'v2':
        from lewm.dev_dynamics_patches_v2_development import patch_session, place_patches, placement_seed
        from scripts import run_go2_dev_mission_stage2_development as dev  # the stage-2 loader wraps the unchanged one
        set_name, maze, episode = argument('--set'), int(argument('--maze')), int(argument('--episode', 0))
        root = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
        spec, packet = dev.loader_for(set_name, maze, '--allow-final-round' in sys.argv)(root, maze, episode)
        dynamics['placement'] = place_patches(spec, packet, seed=placement_seed(set_name, maze, episode))
        owner.make_session = patch_session(owner.make_session, dynamics['placement'], dynamics['mu'], dynamics['marked'])
    elif dynamics:
        from lewm.dev_dynamics_patches_development import patch_session, place_patches, placement_seed
        from scripts import run_go2_dev_mission_stage2_development as dev  # the stage-2 loader wraps the unchanged one
        set_name, maze, episode = argument('--set'), int(argument('--maze')), int(argument('--episode', 0))
        root = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
        spec, packet = dev.loader_for(set_name, maze, '--allow-final-round' in sys.argv)(root, maze, episode)
        dynamics['placement'] = place_patches(spec, packet, seed=placement_seed(set_name, maze, episode))
        owner.make_session = patch_session(owner.make_session, dynamics['placement'], dynamics['mu'], dynamics['marked'])
    assignment = sys.argv[sys.argv.index('--assignment')+1]
    sys.argv = [str(REPO/STAGE2_ENTRY)]+sys.argv[1:]
    try:
        runpy.run_path(str(REPO/STAGE2_ENTRY), run_name='__main__')
    finally:
        try:  # never mask the mission's own outcome
            run = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])/'runs'/assignment
            if run.exists() and not (run/'launch_pin.json').exists():
                owner.save(run/'launch_pin.json', dict(pin, harness=harness, dynamics=dynamics, controller_variant=variant,
                                                       mixins=[m.__name__ for m in extra_mixins+tuple(version.mixins_for(arm))],
                                                       entry=ENTRY, pinned_entry='scripts/run_go2_dev_mission_pinned_v10_development.py', stage2_entry=STAGE2_ENTRY))
        except Exception as error:
            print(f'launch pin record failed: {error!r}', flush=True)


if __name__ == '__main__':
    main()
