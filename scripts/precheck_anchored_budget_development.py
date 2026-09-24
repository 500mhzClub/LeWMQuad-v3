"""Pre-check 2: empirical action-signal budget on the fixed branch assay."""
from collections import defaultdict
import json
import numpy as np

PULSE='/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1/go2_short_pulse_learning_v1_attempt_001'
STEAM='/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1'
ANCH=f'{STEAM}/go2_anchored_visual_dynamics_v1_attempt_001'
REP=f'{STEAM}/go2_visual_target_jepa_v1_attempt_001'

def mse(v): return float(np.mean(np.square(np.asarray(v,dtype=np.float64))))

rows=json.loads(open(f'{PULSE}/windows.json').read())
sel=[r for r in rows if r['observation_frame']==13]
assert len(sel)==36 and all(r['available'] for r in sel)
sel=sorted(sel,key=lambda r:(r['data_role'],r['cluster'],r['prefix_action'],r['pulse_action']))

with np.load(f'{REP}/branch_evaluation/encoded_targets.npz',allow_pickle=False) as a:
    target=a['visual_jepa_targets'].copy()
with np.load(f'{ANCH}/branch_evaluation/causal_predictions.npz',allow_pickle=False) as a:
    anchor=a['persistence'].copy(); action=a['action'].copy(); noact=a['no_future_action'].copy()
print('shapes',target.shape,anchor.shape)

# Validation: reproduce published evaluator numbers exactly.
pub=json.load(open('/home/andrewknowles/Workspace/LeWMQuad-v3/docs/go2_anchored_visual_dynamics_result_2026-09-17.json'))['scores']
out={}
for role in ('train','geometry_transfer'):
    idx=[i for i,r in enumerate(sel) if r['data_role']==role]
    groups=defaultdict(list)
    for i in idx: groups[sel[i]['cluster'],sel[i]['prefix_action']].append(i)
    assert len(groups)==6 and all(len(v)==3 for v in groups.values())
    role_out={'groups':len(groups),'cases':len(idx),'horizons':{}}
    for h in range(8):
        t=target[idx,h].astype(np.float64); a0=anchor[idx,h].astype(np.float64)
        ct=[]; per_group=[]
        Sc_parts=[]
        for key,gi in groups.items():
            tt=target[gi,h].astype(np.float64); aa=anchor[gi,h].astype(np.float64)
            gmean=tt.mean(0)
            ct.extend(tt-gmean)
            sb_g=mse(tt-gmean)                      # within-group branch variance
            sc_g=mse(gmean-aa[0])                   # common change from current state
            per_group.append(dict(cluster=key[0],prefix_action=key[1],
                S_branch=sb_g,S_common=sc_g,ratio=(sb_g/sc_g if sc_g else None)))
            Sc_parts.append(sc_g)
        S_branch=mse(ct); S_common=float(np.mean(Sc_parts))
        persist=mse(a0-t)
        # cross-check against evaluator
        p=pub['persistence'][role]['curve'][h]
        assert abs(persist-p['prediction_mse'])<=1e-6*abs(p['prediction_mse'])+1e-15, (persist,p['prediction_mse'])
        assert abs(S_branch-p['action_independent_effect_mse'])<=1e-6*abs(p['action_independent_effect_mse'])+1e-15
        role_out['horizons'][p['horizon_ms']]=dict(S_branch=S_branch,S_common=S_common,
            persistence_mse=persist,branch_share_of_persistence=S_branch/persist if persist else None,
            branch_over_common=S_branch/S_common if S_common else None,per_group=per_group)
    out[role]=role_out
print('VALIDATED: reproduces evaluator persistence_mse and action_independent_effect_mse to 1e-6 relative')
json.dump(out,open('/tmp/precheck/budget.json','w'),indent=1)

for role in out:
    print(f'\n=== {role} — per-group at 800 ms ===')
    hs=out[role]['horizons']['800' if '800' in out[role]['horizons'] else 800]
    print(f"{'cluster':>9} {'prefix':>8} {'S_branch':>10} {'S_common':>10} {'ratio':>7}")
    for g in hs['per_group']:
        print(f"{str(g['cluster']):>9} {str(g['prefix_action']):>8} {g['S_branch']:10.6f} {g['S_common']:10.6f} {g['ratio']:7.2f}")
    print(f"  aggregate S_branch={hs['S_branch']:.6f} S_common={hs['S_common']:.6f} "
          f"branch/persistence={hs['branch_share_of_persistence']:.3f}")
