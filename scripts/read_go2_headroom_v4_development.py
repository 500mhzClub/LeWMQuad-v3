"""Frozen V4 analysis: each quantity has its own mask, clustered by layout."""
import json
from pathlib import Path
import math
import numpy as np
from scipy.stats import t
from scripts.run_go2_decision_headroom_branches_development import save

ROWS=('R2','R5c','R4/old_data','R4/maze_data')

def ratio(items,numerator,denominator):
    n=sum(w*numerator(r) for w,r in items);d=sum(w*denominator(r) for w,r in items)
    return dict(value=n/d if d else None,numerator=n,denominator=d)


def layout_interval(values,confidence):
    finite=[v for v in values if v is not None];n=len(finite)
    if n<2:return dict(n=n,mean=None if not n else finite[0],interval=None,classification='inconclusive')
    mean=float(np.mean(finite));half=float(t.ppf((1+confidence)/2,n-1)*np.std(finite,ddof=1)/math.sqrt(n))
    return dict(n=n,mean=mean,interval=[mean-half,mean+half],confidence=confidence,layout_values=values,
                exploratory=True,all_four_layouts_present=n==4)


def report(root,*,budget=None):
    records=[];coverage=[]
    for case in range(24):
        if budget:budget.check('analysis_read')
        source=root/f'source_{case:02d}';snapshot=source/'snapshots.json'
        states=json.loads(snapshot.read_text()) if snapshot.exists() else []
        collected=0
        for s in states:
            path=source/f'state_{s["frame"]:04d}'/'audit_v4.json'
            if path.exists():
                row=json.loads(path.read_text());row['case']=case;row['layout']=case//3;records.append(row);collected+=1
        coverage.append(dict(case=case,planned_maximum=24,sampled=len(states),retained_audit_records=collected,missing_or_unresolved=len(states)-collected,shortfall=24-len(states)))
    # State weights are inverse inclusion probabilities. Each source cell is
    # normalized independently, then equally weighted within its layout.
    cells={};primary_names=['G','H_scorer','H_motion','D_old','D_maze']
    for case in range(24):
        if budget:budget.check('analysis_cell')
        population=[r for r in records if r['case']==case]
        for mode in ('all','view_seeking','positional_route','positional_terminal'):
            for phase in ('all','outbound_exploration','goal_approach_settle','return'):
                state_rows=[(r['sampling']['weight'],r) for r in population if (mode=='all' or r['objective']['mode']==mode) and (phase=='all' or r['objective']['phase']==phase)]
                summaries={}
                for motion in ROWS:
                    for criterion in ('operating','hard'):
                        applicable=[(w,r['filter_audit'][motion][criterion]) for w,r in state_rows if motion in r['filter_audit']]
                        candidates=[(w,c) for w,r in applicable for c in r['candidates']]
                        known=[(w,c) for w,c in candidates if c['resolved']]
                        safe=lambda c:c['safety']=='safe';unsafe=lambda c:c['safety']=='unsafe'
                        prefix=f'filter/{motion}/{criterion}'
                        summaries[prefix+'/excluded_safe']=ratio(known,lambda c:c['excluded_but_safe'],safe)
                        summaries[prefix+'/admitted_unsafe']=ratio(known,lambda c:c['admitted_but_unsafe'],lambda c:c['eligible'])
                        summaries[prefix+'/unsafe_admission_given_unsafe']=ratio(known,lambda c:c['admitted_but_unsafe'],unsafe)
                        summaries[prefix+'/safe_among_excluded']=ratio(known,lambda c:c['excluded_but_safe'],lambda c:not c['eligible'])
                        summaries[prefix+'/all_excluded_despite_safe']=ratio([(w,r) for w,r in applicable if r['all_movement_excluded_despite_safe'] is not None],lambda r:r['all_movement_excluded_despite_safe'],lambda r:1)
                        summaries[prefix+'/coverage']=dict(value=sum(w for w,c in known)/(5*sum(w for w,r in state_rows)) if state_rows else None,known_candidates=len(known),attempted_candidates=5*len(state_rows),unresolved=5*len(state_rows)-len(known))
                        reason_counts={}
                        for w,c in known:
                            if c['excluded_but_safe']:reason_counts[c['binding_rule']]=reason_counts.get(c['binding_rule'],0)+w
                        summaries[prefix+'/binding_rule_counts']=reason_counts
                all_row_names=('R0','R1','R2','R5c','R5r','R4/old_data','R4/maze_data','R3/old_data','R3/maze_data','R4s/old_data','R4s/maze_data','R2b/old_data','R2b/maze_data')
                for name in all_row_names:
                    outcomes=[(w,r['rows'][name]) for w,r in state_rows if name in r['rows'] and 'physical_status' in r['rows'][name]]
                    for criterion in ('hard','operating'):
                        known=[(w,r) for w,r in outcomes if r['physical_status'][criterion] in ('safe','unsafe')]
                        summaries[f'outcome/{name}/{criterion}']=ratio(known,lambda r:r['physical_status'][criterion]=='unsafe',lambda r:1)
                        summaries[f'outcome/{name}/{criterion}']['unresolved_states']=len(state_rows)-len(known)
                    summaries[f'outcome/{name}/contact']=ratio(outcomes,lambda r:r['physical_status']['contact'],lambda r:1)
                for head in ('old_data','maze_data'):
                    paired=[(w,(r['rows'].get(f'R4/{head}',{}).get('physical_status'),r['rows'].get('R5c',{}).get('physical_status'))) for w,r in state_rows]
                    paired=[(w,(a,b)) for w,(a,b) in paired if a and b and a['hard'] in ('safe','unsafe') and b['hard'] in ('safe','unsafe')]
                    summaries['paired_harm/'+head]=ratio(paired,lambda v:int(v[0]['hard']=='unsafe')-int(v[1]['hard']=='unsafe'),lambda v:1)
                def regrets(row,names):
                    if not row['objective']['reference_regret_applicable']:return None
                    values=[row['rows'].get(n,{}).get('regret_s') for n in names]
                    return values if all(v is not None for v in values) else None
                contrasts={'G':('R5c',),'H_scorer':('R2',),'H_motion':('R5c','R2'),'D_old':('R4/old_data','R5c'),'D_maze':('R4/maze_data','R5c'),'A_action_old':('R4s/old_data','R4/old_data'),'A_action_maze':('R4s/maze_data','R4/maze_data'),'shrinkage_old':('R4/old_data','R2b/old_data'),'shrinkage_maze':('R4/maze_data','R2b/maze_data')}
                for name,names in contrasts.items():
                    supported=[(w,regrets(r,names)) for w,r in state_rows if regrets(r,names) is not None]
                    summaries[name]=ratio(supported,lambda v:v[0] if len(v)==1 else v[0]-v[1],lambda v:1)
                    summaries[name]['positional_states']=sum(r['objective']['reference_regret_applicable'] for w,r in state_rows)
                    summaries[name]['supported_states']=len(supported)
                summaries['R5r_offbank_signed_gap']=ratio([(w,r['rows']['R5r']['signed_bank_gap_s']) for w,r in state_rows if r['rows'].get('R5r',{}).get('signed_bank_gap_s') is not None],lambda v:v,lambda v:1)
                for head in ('old_data','maze_data'):
                    names=('R2',f'R3/{head}',f'R4/{head}','R5c')
                    supported=[(w,regrets(r,names)) for w,r in state_rows if regrets(r,names) is not None]
                    chain={name:ratio(supported,function,lambda v:1) for name,function in {
                        'G':lambda v:v[3],'H_scorer':lambda v:v[0],'H_motion':lambda v:v[3]-v[0],
                        'D':lambda v:v[2]-v[3],'L_readout':lambda v:v[1]-v[0],'L_forecast':lambda v:v[2]-v[1]}.items()}
                    summaries['chain/'+head]=dict(components=chain,states=len(supported),same_mask_and_weights=True,training_render_provenance='render_unverified; decomposition is not causal attribution')
                cells[f'{case}/{mode}/{phase}']=summaries
    clustered={};confidence=1-.05/17
    quantities=primary_names+[f'filter/{m}/operating/{q}' for m in ROWS for q in ('excluded_safe','admitted_unsafe','all_excluded_despite_safe')]
    secondary_quantities=[f'outcome/{name}/hard' for name in ('R5c','R4/old_data','R4/maze_data')]+['paired_harm/old_data','paired_harm/maze_data']
    for exposure,indices in [('exposed',range(4)),('runtime_unexamined_at_freeze',range(4,8))]:
        for mode in ('all','view_seeking','positional_route','positional_terminal'):
            for phase in ('all','outbound_exploration','goal_approach_settle','return'):
                for quantity in quantities+secondary_quantities:
                    values=[]
                    for layout in indices:
                        cells3=[cells[f'{3*layout+j}/{mode}/{phase}'][quantity].get('value') for j in range(3)]
                        values.append(float(np.mean(cells3)) if all(v is not None for v in cells3) else None)
                    clustered[f'{exposure}/{mode}/{phase}/{quantity}']=layout_interval(values,confidence if mode=='all' and phase=='all' and quantity in quantities else .95)
    if budget:budget.admit_write(16*1024**2)
    save(root/'analysis_v4.json',dict(schema='headroom_v4_analysis.v1',cell_coverage=coverage,cells=cells,layout_clustered=clustered,
        primary_family_size=17,training_render_provenance='render_unverified',source_navigation_is_descriptive_only=True,
        no_unresolved_item_blocks_other_quantities=True,decision_memo_requires_joint_harm_coverage_and_precision_review=True))
    (root/'analysis_v4.md').write_text('# V4 fixed audit closure\n\nAll assigned cells, missing quantities, weighted denominators, filter binding rules, paired regret populations and per-layout intervals are in analysis_v4.json. No source-run navigation outcome is a cohort estimate. No automated recommendation is executed. Prepare the one-step decision memo from these fixed results and then stop.\n')
