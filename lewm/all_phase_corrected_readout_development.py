"""Descriptive paired prediction accounting; no model selection or fitting."""
from collections import defaultdict
import numpy as np
from lewm.training_translation_bias_development import validate_arrays

METRICS=('position_error_m','yaw_error_rad','contact_brier')


def verify_corrected_arrays(before,after,heads):
    if set(before)!=set(after) or set(heads)!=(set(before)-{'indices','prediction_valid','target_offsets_ns'}):
        raise ValueError('all and only trained heads and original prediction metadata required')
    for name in ('indices','prediction_valid','target_offsets_ns'):
        if not np.array_equal(before[name],after[name]):raise ValueError('original prediction indices and clocks changed')
    for head in heads:
        validate_arrays(before,head);validate_arrays(after,head)
        if not np.array_equal(before[head][...,2:],after[head][...,2:]):
            raise ValueError('original yaw/contact values changed')


def difference(candidate,reference):
    denominators=('motion_targets','contact_targets','contact_positives')
    if any(candidate[k]!=reference[k] for k in denominators):
        raise ValueError('paired target denominators must remain identical')
    return {k:None if candidate[k] is None or reference[k] is None else candidate[k]-reference[k]
        for k in METRICS}


def summarize_primary(rows,roster,seeds):
    assignments={r['name']:r for r in roster};primary=[r for r in rows if r['primary_head']]
    groups=defaultdict(list);index={}
    for row in primary:
        if row['model'] not in assignments:raise ValueError('only preassigned models required')
        assignment=assignments[row['model']]
        if any(row[k]!=assignment[k] for k in ('seed','variant','condition')):
            raise ValueError('unchanged original model treatment required')
        if row['head']!=('direct_outcomes' if row['condition']=='direct' else 'rollout_outcomes'):
            raise ValueError('fixed primary outcome head required')
        if row['role'] not in ('train','geometry_transfer') or row['source'] not in ('family','switch'):
            raise ValueError('unchanged role and source required')
        key=tuple(row[k] for k in ('model','role','source','scope'))
        if key in index:raise ValueError('each primary model/source/role/stratum exactly once required')
        difference(row['after'],row['before']);index[key]=row
        groups[tuple(row[k] for k in ('variant','condition','role','source','scope'))].append(row)
    expected={(r['name'],role,source) for r in roster for role in ('train','geometry_transfer') for source in ('family','switch')}
    if {(k[0],k[1],k[2]) for k in index}!=expected:
        raise ValueError('all fixed models, both roles and both sources required')
    scopes={(k[1],k[2]):set() for k in index}
    for _,role,source,scope in index:scopes[role,source].add(scope)
    if set(index)!={(r['name'],role,source,scope) for r in roster
            for (role,source),values in scopes.items() for scope in values}:
        raise ValueError('complete shared strata across every primary model required')
    grouped=[]
    for key,members in sorted(groups.items()):
        if len(members)!=len(seeds) or {r['seed'] for r in members}!=set(seeds):
            raise ValueError('all original optimization seeds required for every group')
        values={}
        for phase in ('before','after'):
            values[phase]={}
            for metric in METRICS:
                numbers=[r[phase][metric] for r in members]
                values[phase][metric]=None if any(v is None for v in numbers) else dict(
                    mean=float(np.mean(numbers)),optimization_seed_sd=float(np.std(numbers,ddof=1)))
        grouped.append(dict(zip(('variant','condition','role','source','scope'),key))|dict(
            seeds=list(seeds),metrics=values,independent_maze_confidence_interval=False))
    pairs=[]
    for row in primary:
        comparison=None
        if row['variant']=='no_rgb':
            reference=f"seed_{row['seed']}_full_{row['condition']}";comparison='no_rgb_minus_full_same_condition'
        elif row['condition']!='jepa':
            reference=f"seed_{row['seed']}_full_jepa";comparison='training_control_minus_full_jepa'
        if comparison is None:continue
        other=index[reference,row['role'],row['source'],row['scope']]
        pairs.append(dict(comparison=comparison,seed=row['seed'],candidate_model=row['model'],reference_model=reference,
            role=row['role'],source=row['source'],scope=row['scope'],candidate=row['after'],reference=other['after'],
            candidate_minus_reference=difference(row['after'],other['after']),independent_maze_inference=False))
    return dict(three_seed_primary_descriptive=grouped,corrected_primary_matched_pairs=pairs,
        target_weighting='original_motion_or_contact_target_counts',
        optimization_seeds_are_not_independent_mazes=True,checkpoint_selection_performed=False,
        native_assignments_changed=False,navigation_qualified=False)
