"""Fixed secondary local-intent decisions on fully observed action branches."""
import numpy as np

INTENTS=(('forward',(.8,0.)),('left',(0.,.8)),('right',(0.,-.8)))


def decision_rows(prediction,targets,metadata):
    prediction=np.asarray(prediction,dtype=np.float64)
    if prediction.shape!=(len(metadata),8,5) or not np.isfinite(prediction).all(): raise ValueError('fixed finite outcome predictions required')
    motion=np.asarray(targets['motion']); mv=np.asarray(targets['motion_valid'],dtype=bool)
    contact=np.asarray(targets['contact']); cv=np.asarray(targets['contact_valid'],dtype=bool)
    rows=[]; rejected=[]
    for layout in sorted({m['layout_id'] for m in metadata}):
        indices=sorted([i for i,m in enumerate(metadata) if m['layout_id']==layout],key=lambda i:metadata[i]['action_index'])
        if [metadata[i]['action_index'] for i in indices]!=list(range(5)): raise ValueError('all five sibling branches required')
        if not all(cv[i,-1] and (contact[i,-1]==1 or mv[i,-1]) for i in indices):
            rejected.append(layout); continue
        probabilities=1/(1+np.exp(-np.clip(prediction[indices,-1,4],-60,60)))
        for intent,goal in INTENTS:
            goal=np.array(goal)
            estimated=10*probabilities+np.linalg.norm(prediction[indices,-1,:2]-goal,axis=-1)
            costs=np.array([10. if contact[i,-1]==1 else np.linalg.norm(motion[i,-1,:2]-goal) for i in indices])
            chosen=int(np.argmin(estimated)); oracle=int(np.argmin(costs)); selected=indices[chosen]
            rows.append({'layout_id':layout,'intent':intent,'chosen_action_index':chosen,'oracle_action_index':oracle,
                'predicted_candidate_costs':estimated.tolist(),'realized_candidate_costs':costs.tolist(),
                'contact':bool(contact[selected,-1]),'stop':chosen==0,'oracle_action_agreement':chosen==oracle,
                'realized_cost':float(costs[chosen]),'oracle_cost':float(costs[oracle]),'all_stop_cost':float(costs[0]),
                'regret':float(costs[chosen]-costs[oracle])})
    metrics=('contact','stop','oracle_action_agreement','realized_cost','oracle_cost','all_stop_cost','regret')
    per_layout=[{'layout_id':layout,**{k:float(np.mean([r[k] for r in rows if r['layout_id']==layout])) for k in metrics}}
        for layout in sorted({r['layout_id'] for r in rows})]
    macro={k:float(np.mean([r[k] for r in per_layout])) for k in metrics} if per_layout else None
    return {'rows':rows,'layouts':per_layout,'layout_macro':macro,'excluded_incomplete_layouts':rejected}
