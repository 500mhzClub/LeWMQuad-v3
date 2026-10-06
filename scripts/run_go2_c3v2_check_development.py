"""One fresh-check mission on the frozen V4 harness (pre-declared C3-v2 check, 29 Sep 2026, §6).

The frozen run owner executes unchanged. Only two functions are rebound: the episode loader
(registered fresh-check episodes) and the prediction-slot model loader (C3-v2 readout weights,
C4-v2 weights; C1 unchanged). Harness bindings, sensors, selector and readers are untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from lewm import decision_headroom_json_v42_development as output
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

VERSIONS = {'C3': ('c3v2_readout_fit_v1/readout_v2_final.pt', 'c3v2_readout_fit_v1/result.json', 'C3-v2'),
            'C4': ('c4v2_fit_v1/direct_v2_final.pt', 'c4v2_fit_v1/result.json', 'C4-v2')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def episode_inputs(root, maze, episode):
    if maze not in range(10) or episode != 0:
        raise ValueError('registered fresh-check episode required')
    registry = json.loads((root/'c3v2_sets_v1/registry.json').read_text())
    entry = next(e for e in registry['entries'] if e['maze_id'] == maze)
    assert entry['role'] == 'fresh_check'
    for record in (entry['maze'], entry['episodes'][0]):
        assert sha(record['path']) == record['sha256']
    packet = json.loads(Path(entry['episodes'][0]['path']).read_text())
    original = json.loads(Path(entry['maze']['path']).read_text())
    assert packet['role'] == original['data_role'] == 'fresh_check'
    spec = original | dict(procedural_seed=packet['simulation_seed'],
        geometry=original['geometry'] | dict(spawn_se2_world=packet['home_se2_world']))
    return spec, packet


def model_identity(root, arm):
    if arm == 'C1':
        return dict(controller='C1', version='C1 (unchanged)')
    checkpoint, result, version = VERSIONS[arm]
    completed = json.loads((root/result).read_text())
    assert completed['status'] == 'COMPLETE' and sha(root/checkpoint) == completed['checkpoint_sha256']
    return dict(controller=arm, version=version, checkpoint=str(root/checkpoint), sha256=completed['checkpoint_sha256'])


def load_model(arm, protocol, root):
    model = owner.load_model(arm, protocol, root)
    if arm == 'C1':
        return model
    identity = model_identity(root, arm)
    state = torch.load(identity['checkpoint'], map_location='cpu', weights_only=False)['model_state_dict']
    target = model.readout if arm == 'C3' else model.predictor
    target.load_state_dict(state)
    target.eval().requires_grad_(False)
    if arm == 'C3':
        model.readout_identity = dict(model.readout_identity, arm='maze_view_maze_data_c3v2', path=identity['checkpoint'], sha256=identity['sha256'])
    return model


def main(arm, maze, assignment):
    if arm not in ('C1', 'C3', 'C4'):
        raise ValueError('fresh check runs C1, C3-v2 and C4-v2 only')
    protocol = json.loads(owner.PROTOCOL.read_text())
    root = Path(protocol['output_root'])
    acceptance = json.loads((root/'c3v2_acceptance_v1/result.json').read_text())
    if not acceptance['passed']:
        raise RuntimeError('C3-v2 did not pass offline acceptance; no closed-loop check is authorised')
    identity = model_identity(root, arm)
    try:
        bind(owner.run, episode_inputs=episode_inputs, load_model=load_model)(arm, maze, 0, assignment)
    finally:
        destination = root/'runs'/assignment
        if destination.exists():
            output.install(root)
            owner.save(destination/'model_version.json', identity | dict(
                entry_script_sha256=sha(__file__), acceptance_sha256=sha(root/'c3v2_acceptance_v1/result.json'),
                harness_unchanged=True, rebound=['episode_inputs', 'load_model']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--controller', required=True)
    p.add_argument('--maze', type=int, required=True)
    p.add_argument('--assignment', required=True)
    a = p.parse_args()
    main(a.controller, a.maze, a.assignment)
