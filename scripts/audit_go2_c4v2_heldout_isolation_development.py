"""Held-out isolation audit for C4-v2 (Andrew's ruling of 29 Sep 2026, item 2).

Confirms that the held-out rest/turn recordings played no part in C4-v2's training or
model selection, and maps its 14,340 encoded frames to the 11,182 training contexts. Reads
the prepared data and the fit's own records only; runs no model.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from lewm import decision_headroom_json_v42_development as output
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

BASE = Path(json.loads(owner.PROTOCOL.read_text())['output_root'])
DATA = BASE/'c3v2_data_v1'
FIT = BASE/'c4v2_fit_v1'
TRAINER = Path('scripts/train_go2_c4v2_development.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category(row):
    return 'v1_'+row['group'] if row['source'] == 'c4_v1_population' else row['source']


def main():
    output.install(BASE)
    train = json.loads((DATA/'train_samples.json').read_text())
    heldout = json.loads((DATA/'heldout_samples.json').read_text())
    paths = json.loads((DATA/'frame_paths.json').read_text())
    v1_paths = json.loads((BASE/'c4_preparation/frame_paths.json').read_text())
    plan = json.loads((FIT/'plan.json').read_text())
    recordings = json.loads((BASE/'c3v2_rest_turn_recordings_v1/plan.json').read_text())['splits']
    frames, contexts = defaultdict(set), Counter()
    for row in train+heldout:
        frames[category(row)].update(row['frame_indices'])
        contexts[category(row)] += 1
    train_frames = set().union(*(set(r['frame_indices']) for r in train))
    heldout_frames = set().union(*(set(r['frame_indices']) for r in heldout))
    heldout_dirs = {r['directory'] for r in heldout}
    train_dirs = {r['directory'] for r in train}
    source = TRAINER.read_text()
    checks = dict(
        trainer_matches_fit_plan=plan['source_sha256'][str(TRAINER.resolve())] == sha(TRAINER),
        fit_plan_heldout_used=plan['heldout_used'],
        trainer_reads_train_samples_only=("'train_samples.json'" in source and 'heldout_samples' not in source),
        checkpoint_selection=plan['checkpoint_selection'],
        heldout_cases_are_heldout_split=all(recordings[str(r['case'])] == 'heldout' for r in heldout),
        train_new_cases_are_fit_split=all(recordings[str(r['case'])] == 'fit' for r in train if r['source'].startswith('c3v2_rest_turn')),
        frames_shared_by_train_and_heldout=len(train_frames & heldout_frames),
        recording_directories_shared_by_train_and_heldout=len(train_dirs & heldout_dirs),
        train_frames_inside_heldout_directories=sum(str(Path(paths[i]).parent) in heldout_dirs for i in train_frames),
        target_normalisation='fixed from the pre-registered C3 head binding (target_mean, target_scale); no statistic of the fit data',
    )
    passed = (checks['trainer_matches_fit_plan'] and checks['fit_plan_heldout_used'] is False and checks['trainer_reads_train_samples_only']
              and checks['checkpoint_selection'] == 'fixed final' and checks['heldout_cases_are_heldout_split']
              and checks['train_new_cases_are_fit_split'] and not checks['frames_shared_by_train_and_heldout']
              and not checks['recording_directories_shared_by_train_and_heldout'] and not checks['train_frames_inside_heldout_directories'])
    mapping = dict(
        encoded_frames=len(paths),
        c4_v1_frames_reused_as_prefix=len(v1_paths) if paths[:len(v1_paths)] == v1_paths else None,
        by_source={k: dict(contexts=contexts[k], frame_references=3*contexts[k], unique_frames=len(frames[k])) for k in sorted(frames)},
        training_contexts=len(train), training_frame_references=3*len(train), training_unique_frames=len(train_frames),
        heldout_unique_frames_encoded_but_never_indexed_by_training=len(heldout_frames),
        unreferenced_frames=len(set(range(len(paths)))-train_frames-heldout_frames),
        rule='Each context uses three causal frames (t-10, t-5, t). Contexts on one recording sit on consecutive frames, so one frame serves up to three contexts.',
    )
    result = dict(schema='c4v2_heldout_isolation_audit.v1', passed=passed, checks=checks, frame_mapping=mapping,
                  inputs_sha256={n: sha(DATA/n) for n in ('train_samples.json', 'heldout_samples.json', 'frame_paths.json')} | {'c4v2_plan.json': sha(FIT/'plan.json')},
                  auditor_sha256=sha(__file__))
    owner.save(DATA/'c4v2_heldout_isolation_audit.json', result)
    print(json.dumps(dict(passed=passed, checks=checks, frame_mapping=mapping)))


if __name__ == '__main__':
    main()
