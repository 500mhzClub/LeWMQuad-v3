"""Pre-check 1: scalar innovation shrinkage, fitted train-role only, frozen for transfer."""
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

def metrics(pred,role):
    idx=ROLE[role]; t=target[idx]; out={}
    out['total']=mse(pred[idx]-t)
    cen_p=[];cen_t=[];com=[]
    raw_w=cen_w=0; margins=[]
    for key,gi in GRP[role].items():
        tt=target[gi];pp=pred[gi]
        cen_p.extend(pp-pp.mean(0));cen_t.extend(tt-tt.mean(0));com.append(mse(pp.mean(0)-tt.mean(0)))
        m=((pp[:,None,7]-tt[None,:,7])**2).mean(-1)
        for j in range(3):
            other=min(m[j,k] for k in range(3) if k!=j)
            raw_w+=int(m[j,j]<other); margins.append(other-m[j,j])
        cp=pp-pp.mean(0);ct=tt-tt.mean(0)
        mc=((cp[:,None,7]-ct[None,:,7])**2).mean(-1)
        cen_w+=sum(int(mc[j,j]<min(mc[j,k] for k in range(3) if k!=j)) for j in range(3))
    out['centred']=mse(np.asarray(cen_p)-np.asarray(cen_t));out['common']=float(np.mean(com))
    out['retrieval_raw']=raw_w;out['retrieval_centred']=cen_w
    out['mean_margin_800ms']=float(np.mean(margins))
    return out

print('Fit population: branch-assay TRAIN role (clusters 00/01, 18 cases, 6 groups).')
print('The broad 4,694-context fitting distribution is NOT retained; see note.\n')
report={}
for arm in ('action','no_future_action'):
    p=P[arm]-anchor; d=target-anchor
    tr=ROLE['train']
    num=float((p[tr]*d[tr]).sum()); den=float((p[tr]*p[tr]).sum())
    a_unc=num/den; a_use=min(1.0,max(0.0,a_unc))
    print(f'--- {arm}: alpha_unconstrained={a_unc:.4f}  alpha_applied(clamped 0..1)={a_use:.4f} ---')
    report[arm]={'alpha_unconstrained':a_unc,'alpha_applied':a_use,'roles':{}}
    print(f"  {'role':>18} {'variant':>12} {'total':>10} {'common':>10} {'centred':>10} {'retr_raw':>9} {'retr_cen':>9}")
    for role in ('train','geometry_transfer'):
        for label,pred in (('alpha=1 (orig)',P[arm]),(f'alpha={a_use:.3f}',anchor+a_use*p),('persistence',anchor)):
            m=metrics(pred,role)
            if label=='persistence' and arm!='action': continue
            report[arm]['roles'].setdefault(role,{})[label]=m
            print(f"  {role:>18} {label:>12} {m['total']:10.6f} {m['common']:10.6f} {m['centred']:10.6f} "
                  f"{m['retrieval_raw']:>9} {m['retrieval_centred']:>9}")
    print()
json.dump(report,open('/tmp/precheck/shrinkage.json','w'),indent=1)
