"""Exploratory decision and bias diagnostics from saved motion predictions."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

SOURCE=Path('docs/go2_dense_visual_motion_readout_branch_result_2026-09-17.json')
OUTPUT=Path('docs/go2_dense_motion_interface_diagnostic_2026-09-17.json')


def wrap(a): return math.atan2(math.sin(a),math.cos(a))


def main():
    r=json.loads(SOURCE.read_text());assert r['status']=='COMPLETE'
    lookup={t:i for i,t in enumerate(r['trials'])};summaries={};decisions=[];bias={}
    for role in ('train','geometry_transfer'):
        totals={a:dict(point=[],scan=[]) for a in r['predictions']}
        trials={t for g in r['groups'] if g['role']==role for t in g['trials']}
        indices=[i for i,t in enumerate(r['trials']) if t in trials]
        bias[role]={}
        for arm in ('action','no_future_action','observed_future'):
            error=[]
            for i in indices:
                e=[v-b-y for v,b,y in zip(r['predictions'][arm][i],r['predictions']['persistence'][i],r['physical_targets'][i])]
                e[2]=wrap(e[2]);error.append(e)
            bias[role][arm]=dict(xy_rmse_mm=math.sqrt(mean(e[0]**2+e[1]**2 for e in error))*1000,
                yaw_rmse_deg=math.sqrt(mean(e[2]**2 for e in error))*180/math.pi)
        for g in r['groups']:
            if g['role']!=role:continue
            idx=[lookup[t] for t in g['trials']];truth=[r['physical_targets'][i] for i in idx]
            for kind,angles in (('point',[-math.pi/4,0.,math.pi/4]),('scan',[-math.pi/6,math.pi/6])):
                for angle in angles:
                    def cost(v):
                        return math.hypot(.25*math.cos(angle)-v[0],.25*math.sin(angle)-v[1]) if kind=='point' else abs(wrap(angle-v[2]))
                    actual=[cost(v) for v in truth]
                    for arm,values in r['predictions'].items():
                        scores=[cost(values[i]) for i in idx]
                        chosen=[i for i,x in enumerate(scores) if x==min(scores)]
                        regret=(mean(actual[i] for i in chosen)-min(actual))*(1000 if kind=='point' else 180/math.pi)
                        totals[arm][kind].append(regret)
                        decisions.append(dict(role=role,cluster=g['cluster'],prefix=g['prefix_action'],kind=kind,
                            angle_rad=angle,arm=arm,chosen_local_indices=chosen,expected_regret=regret))
        summaries[role]={a:dict(point_regret_mm=mean(v['point']),scan_regret_deg=mean(v['scan'])) for a,v in totals.items()}
    grouped={}
    for g in r['groups']:
        for j,t in enumerate(g['trials']):grouped.setdefault((g['prefix_action'],j),[]).append(r['physical_targets'][lookup[t]])
    assert all(len(v)==4 for v in grouped.values())
    spread=max(max(v[k] for v in vs)-min(v[k] for v in vs) for vs in grouped.values() for k in range(3))
    result=dict(status='COMPLETE',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        summaries=summaries,decisions=decisions,subtract_persistence_readout_diagnostic=bias,
        maximum_corresponding_motion_label_spread_across_four_geometries=spread,
        task_definition='point goals 0.25 m away at -45/0/+45 degrees; scan goals -30/+30 degrees; uniform exact ties',
        post_hoc_exploratory=True,calibration_adopted=False,new_model_execution=False,new_navigation=False,
        limitations=['tiny pulse branches; three candidates; dependent goals and histories',
            'static 500-ms endpoint cost, not the current delayed eight-horizon controller',
            'corresponding physical outcomes repeat across geometries; no geometry-dependent motion advantage test'])
    with OUTPUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(summaries,indent=2));print('physical_label_spread',spread)


if __name__=='__main__':main()
