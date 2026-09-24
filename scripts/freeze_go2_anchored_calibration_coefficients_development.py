"""Freeze the split-calibration coefficients at full precision for the transfer successor.

Refits deterministically from the retained branch assay (train role only, clusters 00/01)
and writes float64 values plus exact hex representations, so the successor loads saved
coefficients rather than transcribing rounded figures from a report.

The reference bank is the dimension that VARIES WITHIN an identical-history group, which
the evaluator keys by (cluster, prefix_action). That dimension is pulse_action:
(forward, left_arc, right_arc). hold/left_turn/right_turn are the committed PREFIXES that
define the groups, not the candidate bank.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

PULSE = ('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/'
         'navigation_development_artifacts_v1/go2_short_pulse_learning_v1_attempt_001')
STEAM = '/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1'
OUT = Path('/home/andrewknowles/Workspace/LeWMQuad-v3/.generated/'
           'navigation_development_artifacts_v1/go2_anchored_navigation_transfer_v1_attempt_001')
BANK = ('forward', 'left_arc', 'right_arc')
PREFIXES = ('hold', 'left_turn', 'right_turn')


def decompose(array, groups):
    common = np.zeros_like(array)
    deviation = np.zeros_like(array)
    for members in groups.values():
        mean = array[members].mean(0)
        for i in members:
            common[i] = mean
            deviation[i] = array[i] - mean
    return common, deviation


def main():
    rows = json.loads(open(f'{PULSE}/windows.json').read())
    sel = sorted([r for r in rows if r['observation_frame'] == 13],
                 key=lambda r: (r['data_role'], r['cluster'], r['prefix_action'], r['pulse_action']))
    assert {r['pulse_action'] for r in sel} == set(BANK), 'branch dimension is not the declared bank'
    assert {r['prefix_action'] for r in sel} == set(PREFIXES), 'prefix set changed'

    with np.load(f'{STEAM}/go2_visual_target_jepa_v1_attempt_001/branch_evaluation/encoded_targets.npz') as a:
        target = a['visual_jepa_targets'].astype(np.float64)
    with np.load(f'{STEAM}/go2_anchored_visual_dynamics_v1_attempt_001/branch_evaluation/causal_predictions.npz') as a:
        arms = {k: a[k].astype(np.float64) for k in a.files}
    anchor = arms['persistence']

    train = [i for i, r in enumerate(sel) if r['data_role'] == 'train']
    groups = defaultdict(list)
    for i in train:
        groups[sel[i]['cluster'], sel[i]['prefix_action']].append(i)
    groups = dict(groups)
    assert len(groups) == 6 and all(len(v) == 3 for v in groups.values())

    record = {}
    for arm in ('action', 'no_future_action'):
        p = arms[arm] - anchor
        d = target - anchor
        pc, pd = decompose(p, groups)
        dc, dd = decompose(d, groups)
        denominator = float((pd[train] * pd[train]).sum())
        common = float((pc[train] * dc[train]).sum() / (pc[train] * pc[train]).sum())
        branch = float((pd[train] * dd[train]).sum() / denominator) if denominator > 0 else 0.0
        record[arm] = dict(alpha_common=common, alpha_branch=branch,
                           alpha_common_hex=common.hex(), alpha_branch_hex=branch.hex(),
                           branch_denominator=denominator)
        print(f'{arm:18s} alpha_common={common!r}  alpha_branch={branch!r}')

    OUT.mkdir(parents=True, exist_ok=True)
    payload = dict(
        schema='anchored_split_calibration_coefficients.v1',
        fitted_on='branch assay train role only (clusters 00/01), 18 cases in 6 groups',
        pooled_over_horizons='all 8 horizons, 100-800 ms',
        reference_bank=BANK,
        committed_prefixes=PREFIXES,
        bank_note=('The bank is the within-group varying dimension (pulse_action). '
                   'hold/left_turn/right_turn are committed prefixes defining the groups.'),
        equal_weight_per_candidate=True,
        coefficients=record,
        transfer_fitted_diagnostic_excluded=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    path = OUT / 'frozen_coefficients.json'
    path.write_text(json.dumps(payload, indent=1))
    print(f'\nwrote {path}')
    print(f'sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}')


if __name__ == '__main__':
    main()
