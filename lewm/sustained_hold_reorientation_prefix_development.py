"""Strict comparison of complete original and prospective raw-controller decisions."""
from copy import deepcopy
import hashlib
import json


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def compare_step(old,new,actual_command,*,frame,expected_selection_sha256):
    if (type(frame) is not int or not 0<=frame<=406 or old['tick']!=frame or new['tick']!=frame
            or old['controller']!='hold_reorientation_controller_v1'
            or new['controller']!='sustained_hold_reorientation_controller_v1'
            or new['sustained_hold_reorientation_enabled'] is not True
            or old['requested_command']!=actual_command
            or any(d['terminal'] is not None or d['failure'] is not None for d in (old,new))):
        raise ValueError('exact current nonterminal original and candidate decisions required')
    a,b=old['new_selection'],new['new_selection']
    if identity(b)!=expected_selection_sha256:
        raise ValueError('complete candidate selection must match frozen saved-input expectation')
    restored=deepcopy(b);receipt=None
    if restored is not None:receipt=restored.pop('sustained_hold_reorientation',None)
    changed=new['requested_command']!=actual_command
    if receipt is not None:
        if (receipt['frame']!=frame or receipt['measured_ns']!=1_500_000_000+frame*100_000_000
                or b['requested_command']!=new['requested_command'] or b['action']!=new['selected_action']):
            raise ValueError('same-frame sustained intervention and actual request required')
    if changed:
        if (receipt is None or receipt['starting'] is not False or frame!=406
                or a['action']!='hold' or b['action']!='left_turn'):
            raise ValueError('only the declared first continued left turn may change a request')
        for key in ('action','action_index','requested_command'):restored[key]=deepcopy(a[key])
    if restored!=a:raise ValueError('complete original selection evidence must remain exact')
    normalized=deepcopy(new);normalized.pop('sustained_hold_reorientation_enabled')
    normalized['controller']=old['controller'];normalized['new_selection']=restored
    if changed:
        for key in ('requested_command','selected_action'):normalized[key]=deepcopy(old[key])
    if normalized!=old:raise ValueError('complete observed mission and executed-state decision must remain exact')
    return dict(requested_command_changed=changed,normalized_complete_decision_exact=True,
        complete_original_selection_evidence_exact=True,candidate_matches_saved_selection_expectation=True,
        raw_model_forecasts_compared=bool(a and 'prediction' in a))
