"""Exploratory action-history support accounting, never policy input or fitting."""
import numpy as np

from lewm.counterfactual_maze_development import ACTIONS
from lewm.simulated_body_observation_development import validate_policy_packet


def command_context(packet):
    validate_policy_packet(packet)
    row=packet['sensor_state']['control']['applied_command']
    values=np.asarray(row['values']); valid=np.asarray(row['valid']); ns=np.asarray(row['measured_ns'])
    if not valid[-5:].all() or not np.array_equal(np.diff(ns[-5:]),np.full(4,100_000_000)):
        raise ValueError('five actual consecutive past commands required')
    matches=[i for i,(_,command) in enumerate(ACTIONS) if np.allclose(values[-1],command,atol=1e-7,rtol=0)]
    if len(matches)>1: raise ValueError('ambiguous command bank')
    return {'last_action_index':matches[0] if matches else None,
        'last_action_name':ACTIONS[matches[0]][0] if matches else 'nonbank',
        'five_applied_ticks_constant':bool(np.allclose(values[-5:],values[-1],atol=1e-7,rtol=0)),
        'last_applied_command':values[-1].tolist()}


def support_table(rows):
    result=[]
    for stage in ('initial','later'):
        for prior in range(6):
            for future in range(5):
                selected=[r for r in rows if r['stage']==stage and r['prior_index']==prior and r['future_index']==future]
                result.append({'stage':stage,'prior_action_index':prior if prior<5 else None,
                    'prior_action_name':ACTIONS[prior][0] if prior<5 else 'nonbank',
                    'future_action_index':future,'future_action_name':ACTIONS[future][0],
                    'windows':len(selected),'layouts':len({r['layout_id'] for r in selected}),
                    'constant_past_five_tick_windows':sum(r['constant_past'] for r in selected)})
    return result


def stratified_errors(rows):
    output=[]
    for method in sorted({r['method'] for r in rows}):
        for group in ('initial','repeat_previous_selection','switch_previous_selection'):
            selected=[r for r in rows if r['method']==method and r['group']==group]
            layouts=[]
            for layout in sorted({r['layout_id'] for r in selected}):
                values=[r for r in selected if r['layout_id']==layout]
                entry={'layout_id':layout,'choices':len(values),'known_contact':sum(r['contact'] is not None for r in values),
                    'contact_positive':sum(r['contact'] is True for r in values),
                    'absent_training_later_pair':sum(not r['training_later_pair_windows'] for r in values)}
                for key in ('position_error_m','yaw_error_rad','contact_brier'):
                    observed=[r[key] for r in values if r[key] is not None]
                    entry[key]={'count':len(observed),'mean':float(np.mean(observed)) if observed else None}
                layouts.append(entry)
            macro={}
            for key in ('position_error_m','yaw_error_rad','contact_brier'):
                values=[r[key]['mean'] for r in layouts if r[key]['mean'] is not None]
                macro[key]={'contributing_layouts':len(values),'mean':float(np.mean(values)) if values else None}
            output.append({'method':method,'group':group,'choices':len(selected),'layouts':layouts,'layout_macro':macro})
    return output
