"""Per-group branch-information budget and retrieval, fixed branch assay."""
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
with np.load(f'{STEAM}/go2_visual_target_jepa_v1_attempt_001/branch_evaluation/causal_predictions.npz') as a:
    P['original_visual_jepa']=a['visual_jepa_predictions'].copy()

H=7  # 800 ms
for role in ('train','geometry_transfer'):
    idx=[i for i,r in enumerate(sel) if r['data_role']==role]
    groups=defaultdict(list)
    for i in idx: groups[sel[i]['cluster'],sel[i]['prefix_action']].append(i)
    print(f'\n=== {role} @ 800 ms ===')
    hdr=f"{'group':>22} {'S_branch':>10} " + ' '.join(f'{a[:9]:>9}' for a in P)
    print(hdr)
    totals=defaultdict(int)
    for key,gi in sorted(groups.items()):
        # anchors identical within group (shared history)
        assert np.allclose(P['persistence'][gi,H],P['persistence'][gi[0],H]),key
        tt=target[gi,H].astype(np.float64); sb=mse(tt-tt.mean(0))
        line=f"{key[0]+'/'+key[1]:>22} {sb:10.6f} "
        for arm in P:
            pp=P[arm][gi,H].astype(np.float64)
            m=((pp[:,None]-tt[None])**2).mean(-1)
            w=sum(int(m[j,j]<min(m[j,k] for k in range(3) if k!=j)) for j in range(3))
            totals[arm]+=w; line+=f'{w:>9}'
        print(line)
    print(f"{'TOTAL wins /18':>22} {'':>10} "+' '.join(f'{totals[a]:>9}' for a in P))
    print(f"{'chance = 6/18':>22}")
