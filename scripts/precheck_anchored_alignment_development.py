"""Separate magnitude miscalibration from directional error in the fitted scalars.

A least-squares coefficient conflates the two: alpha* = (||d||/||p||) * cos(p,d).
Reporting the norm ratio and the cosine separately says whether the innovation is
merely mis-scaled or also mis-directed. Retained data only; no fitting, no disk cost.
"""
from collections import defaultdict
import json
import numpy as np

PULSE = ('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/'
         'navigation_development_artifacts_v1/go2_short_pulse_learning_v1_attempt_001')
STEAM = '/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1'


def load():
    rows = json.loads(open(f'{PULSE}/windows.json').read())
    sel = sorted([r for r in rows if r['observation_frame'] == 13],
                 key=lambda r: (r['data_role'], r['cluster'], r['prefix_action'], r['pulse_action']))
    with np.load(f'{STEAM}/go2_visual_target_jepa_v1_attempt_001/branch_evaluation/encoded_targets.npz') as a:
        target = a['visual_jepa_targets'].astype(np.float64)
    with np.load(f'{STEAM}/go2_anchored_visual_dynamics_v1_attempt_001/branch_evaluation/causal_predictions.npz') as a:
        arms = {k: a[k].astype(np.float64) for k in a.files}
    return sel, target, arms


def decompose(array, groups):
    """Split into the within-group common part and the branch deviation."""
    common = np.zeros_like(array)
    deviation = np.zeros_like(array)
    for members in groups.values():
        mean = array[members].mean(0)
        for i in members:
            common[i] = mean
            deviation[i] = array[i] - mean
    return common, deviation


def stats(p, d, index):
    pn = float(np.sqrt(np.square(p[index]).sum()))
    dn = float(np.sqrt(np.square(d[index]).sum()))
    if pn == 0 or dn == 0:
        return dict(norm_ratio=None, cosine=None, alpha=None)
    cos = float((p[index] * d[index]).sum() / (pn * dn))
    return dict(norm_ratio=pn / dn, cosine=cos,
                alpha=float((p[index] * d[index]).sum() / np.square(p[index]).sum()))


def main():
    sel, target, arms = load()
    anchor = arms['persistence']
    roles = {r: [i for i, s in enumerate(sel) if s['data_role'] == r]
             for r in ('train', 'geometry_transfer')}
    groups = {}
    for role, index in roles.items():
        g = defaultdict(list)
        for i in index:
            g[sel[i]['cluster'], sel[i]['prefix_action']].append(i)
        groups[role] = dict(g)

    print('alpha = (||d||/||p||) * cos(p,d).  norm_ratio = ||p||/||d||: >1 over-predicted.')
    for arm in ('action', 'no_future_action'):
        p = arms[arm] - anchor
        d = target - anchor
        print(f'\n=== {arm} ===')
        print(f"{'role':>18} {'component':>10} {'norm_ratio':>11} {'cosine':>8} {'alpha':>8}")
        for role, index in roles.items():
            pc, pd = decompose(p, groups[role])
            dc, dd = decompose(d, groups[role])
            for label, pp, dd_ in (('full', p, d), ('common', pc, dc), ('branch', pd, dd)):
                s = stats(pp, dd_, index)
                if s['cosine'] is None:
                    print(f'{role:>18} {label:>10} {"n/a":>11} {"n/a":>8} {"n/a":>8}')
                else:
                    print(f"{role:>18} {label:>10} {s['norm_ratio']:11.3f} "
                          f"{s['cosine']:8.3f} {s['alpha']:8.4f}")


if __name__ == '__main__':
    main()
