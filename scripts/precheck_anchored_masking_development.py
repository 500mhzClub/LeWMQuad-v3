"""Is branch information present but masked by the common bias?"""
from collections import defaultdict
import json
import numpy as np
PULSE='/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_short_pulse_learning_v1_attempt_001'
STEAM='/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1'
def mse(v): return float(np.mean(np.square(np.asarray(v,dtype=np.float64))))
rows=json.loads(open(f'{PULSE}/windows.json').read())
sel=sorted([r for r in rows if r['observation_frame']==13],
           key=lambda r:(r['data_role'],r['cluster'],r['prefix_action'],r['pulse_action']))
with np.load(f'{STEAM}/go2_visual_target_jepa_v1_attempt_001/branch_evaluation/encoded_targets.npz') as a:
    target=a['visual_jepa_targets'].copy()
with np.load(f'{STEAM}/go2_anchored_visual_dynamics_v1_attempt_001/branch_evaluation/causal_predictions.npz') as a:
    P={k:a[k].copy() for k in a.files}
H=7
for role in ('train','geometry_transfer'):
    idx=[i for i,r in enumerate(sel) if r['data_role']==role]
    groups=defaultdict(list)
    for i in idx: groups[sel[i]['cluster'],sel[i]['prefix_action']].append(i)
    print(f'\n=== {role} @ 800 ms ===')
    print(f"{'group':>22} {'predspread':>11} {'commonerr':>11} {'ratio':>8} | {'raw':>4} {'centred':>8} {'S_branch':>10}")
    tot_raw=tot_cen=0
    for key,gi in sorted(groups.items()):
        tt=target[gi,H].astype(np.float64); pp=P['action'][gi,H].astype(np.float64)
        spread=mse(pp-pp.mean(0)); common=mse(pp.mean(0)-tt.mean(0))
        def wins(p,t):
            m=((p[:,None]-t[None])**2).mean(-1)
            return sum(int(m[j,j]<min(m[j,k] for k in range(3) if k!=j)) for j in range(3))
        w_raw=wins(pp,tt); w_cen=wins(pp-pp.mean(0),tt-tt.mean(0))
        tot_raw+=w_raw; tot_cen+=w_cen
        print(f"{key[0]+'/'+key[1]:>22} {spread:11.6f} {common:11.6f} {spread/common:8.4f} | "
              f"{w_raw:>4} {w_cen:>8} {mse(tt-tt.mean(0)):10.6f}")
    print(f"{'TOTAL /18':>22} {'':>11} {'':>11} {'':>8} | {tot_raw:>4} {tot_cen:>8}")
