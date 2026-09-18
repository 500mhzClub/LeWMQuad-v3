"""Per-group held-out centred error under split calibration + leave-one-group-out stability."""
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
anchor=P['persistence'];p=P['action']-anchor;d=target-anchor
ROLE={r:[i for i,s in enumerate(sel) if s['data_role']==r] for r in ('train','geometry_transfer')}
GRP={}
for r,idx in ROLE.items():
    g=defaultdict(list)
    for i in idx: g[sel[i]['cluster'],sel[i]['prefix_action']].append(i)
    GRP[r]=dict(g)
def dec(arr,groups):
    com=np.zeros_like(arr);dev=np.zeros_like(arr)
    for gi in groups.values():
        m=arr[gi].mean(0)
        for i in gi: com[i]=m;dev[i]=arr[i]-m
    return com,dev
def fit(group_keys):
    gs={k:GRP['train'][k] for k in group_keys}
    pc,pd=dec(p,gs);dc,dd=dec(d,gs);ii=[i for v in gs.values() for i in v]
    ac=float((pc[ii]*dc[ii]).sum()/(pc[ii]*pc[ii]).sum())
    ab=float((pd[ii]*dd[ii]).sum()/(pd[ii]*pd[ii]).sum())
    return ac,ab
ac,ab=fit(list(GRP['train']))
pc2,pd2=dec(p,GRP['geometry_transfer'])
pred=anchor+ac*pc2+ab*pd2
H=7
print(f'alpha_common={ac:.4f} alpha_branch={ab:.4f}\n')
print('Held-out geometry_transfer, centred (branch) error at 800 ms, per group:')
print(f"{'group':>22} {'S_branch':>10} {'split_cen':>10} {'captured':>9} {'retr':>5}")
tot_s=[];tot_p=[]
for key,gi in sorted(GRP['geometry_transfer'].items()):
    tt=target[gi][:,H];pp=pred[gi][:,H]
    sb=mse(tt-tt.mean(0));cen=mse((pp-pp.mean(0))-(tt-tt.mean(0)))
    m=((pp[:,None]-tt[None])**2).mean(-1)
    w=sum(int(m[j,j]<min(m[j,k] for k in range(3) if k!=j)) for j in range(3))
    tot_s.append(cen);tot_p.append(sb)
    print(f"{key[0]+'/'+key[1]:>22} {sb:10.6f} {cen:10.6f} {1-cen/sb if sb else 0:9.1%} {w:>5}")
print(f"{'AGGREGATE':>22} {np.mean(tot_p):10.6f} {np.mean(tot_s):10.6f} {1-np.mean(tot_s)/np.mean(tot_p):9.1%}")
print('\nLeave-one-group-out stability of the fitted scalars (train role):')
keys=list(GRP['train'])
for k in keys:
    a1,b1=fit([x for x in keys if x!=k])
    print(f"  drop {k[0]+'/'+k[1]:>22}  alpha_common={a1:.4f}  alpha_branch={b1:.4f}")
