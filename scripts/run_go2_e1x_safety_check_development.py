"""One E1 exploratory-arm safety-check mission (declared 30 Sep 2026, commit a435cbcc).

C1, C3-v3 and C4-v3 on the 10 unused C3-v3 round safety-check mazes (layouts 22-31), on the
frozen V4 harness. It reuses the round's episode and model loaders unchanged; only the round's
acceptance gate is replaced by this declaration. No sealed-set access.
"""
import argparse
import hashlib
import json
from pathlib import Path

from lewm import decision_headroom_json_v42_development as output
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_c3v3_round_development as round_entry
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

DECLARATION = 'docs/go2_navigation_e1_exploratory_arm_declaration_2026-09-30.md'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(arm, maze, assignment):
    if arm not in ('C1', 'C3', 'C4') or round_entry.role_for(maze) != 'safety_check':
        raise ValueError('the exploratory safety check runs C1, C3-v3 and C4-v3 on safety-check mazes 22-31 only')
    protocol = json.loads(owner.PROTOCOL.read_text())
    root = Path(protocol['output_root'])
    identity = round_entry.model_identity(root, arm) | dict(role='safety_check', purpose='E1 exploratory-arm safety gate')
    try:
        bind(owner.run, episode_inputs=round_entry.episode_inputs, load_model=round_entry.load_model)(arm, maze, 0, assignment)
    finally:
        destination = root/'runs'/assignment
        if destination.exists():
            output.install(root)
            owner.save(destination/'model_version.json', identity | dict(
                entry_script_sha256=sha(__file__), declaration_sha256=sha(DECLARATION), declaration_commit='a435cbcc',
                harness_unchanged=True, rebound=['episode_inputs', 'load_model']))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--controller', required=True)
    p.add_argument('--maze', type=int, required=True)
    p.add_argument('--assignment', required=True)
    a = p.parse_args()
    main(a.controller, a.maze, a.assignment)
