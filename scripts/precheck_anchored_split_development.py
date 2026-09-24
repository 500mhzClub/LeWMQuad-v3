"""Split calibration: separate scalars for common and branch-deviation innovation."""
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
    target=a['visual_jepa_targets'].astype(np.float64)
with np.load(f'{STEAM}/go2_anchored_visual_dynamics_v1_attempt_001/branch_evaluation/causal_predictions.npz') as a:
    P={k:a[k].astype(np.float64) for k in a.files}
anchor=P['persistence']
ROLE={r:[i for i,s in enumerate(sel) if s['data_role']==r] for r in ('train','geometry_transfer')}
GRP={}
for r,idx in ROLE.items():
    g=defaultdict(list)
    for i in idx: g[sel[i]['cluster'],sel[i]['prefix_action']].append(i)
    GRP[r]=dict(g)

def decompose(arr,role):
    """Split innovation into within-group common and branch-deviation parts."""
    com=np.zeros_like(arr);dev=np.zeros_like(arr)
    for gi in GRP[role].values():
        m=arr[gi].mean(0)
        for i in gi: com[i]=m;dev[i]=arr[i]-m
    return com,dev

for arm in ('action','no_future_action'):
    p=P[arm]-anchor;d=target-anchor
    pc,pd=decompose(p,'train');dc,dd=decompose(d,'train')
    tr=ROLE['train']
    ac=float((pc[tr]*dc[tr]).sum()/ (pc[tr]*pc[tr]).sum())
    den=float((pd[tr]*pd[tr]).sum())
    ab=float((pd[tr]*dd[tr]).sum()/den) if den>0 else 0.0
    print(f'--- {arm}: alpha_common={ac:.4f}  alpha_branch={ab:.4f} (fit on train role) ---')
    print(f"  {'role':>18} {'variant':>16} {'total800':>10} {'common800':>10} {'centred800':>11} {'retr':>5}")
    for role in ('train','geometry_transfer'):
        pc2,pd2=decompose(p,role)
        variants={'alpha=1':P[arm],'split':anchor+ac*pc2+ab*pd2,'persistence':anchor}
        for label,pred in variants.items():
            idx=ROLE[role];H=7
            tot=mse(pred[idx][:,H]-target[idx][:,H])
            cenp=[];cent=[];com=[];w=0
            for gi in GRP[role].values():
                tt=target[gi][:,H];pp=pred[gi][:,H]
                cenp.extend(pp-pp.mean(0));cent.extend(tt-tt.mean(0));com.append(mse(pp.mean(0)-tt.mean(0)))
                m=((pp[:,None]-tt[None])**2).mean(-1)
                w+=sum(int(m[j,j]<min(m[j,k] for k in range(3) if k!=j)) for j in range(3))
            print(f"  {role:>18} {label:>16} {tot:10.6f} {float(np.mean(com)):10.6f} "
                  f"{mse(np.asarray(cenp)-np.asarray(cent)):11.6f} {w:>5}")
    print()
