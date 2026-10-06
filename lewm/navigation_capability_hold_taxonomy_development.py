"""Read-only v0 hold accounting, including two retained final-selection rules."""
from scripts.analyse_go2_stage_a_holds_readonly_development import classify as inherited


def classify(row):
    result=inherited(row);selection=row['selection']
    result['inherited_stage_a_category']=result['category']
    coverage=selection.get('translation_footprint_coverage',{})
    arrival=selection.get('predictive_arrival_hold',{})
    if (selection['action']=='hold' and coverage.get('rejected')
            and coverage.get('selected_action')=='hold'
            and coverage.get('previous_action') in ('forward','left_arc','right_arc')):
        result.update(category='explicit_override',override_reason='TRANSLATION_FOOTPRINT_COVERAGE',
            binding_rule='predicted footprint extends beyond observed floor coverage',
            binding_depends_on_motion=True,observation_only_binding=False,
            reason='Retained coverage rule changed the selected translation to hold')
    elif (selection['action']=='hold' and arrival.get('changed')
            and arrival.get('selected_action')=='hold' and arrival.get('eligible')):
        result.update(category='explicit_override',override_reason='PREDICTIVE_ARRIVAL_HOLD',
            binding_rule='existing hold forecast predicts quiet arrival',
            binding_depends_on_motion=True,observation_only_binding=False,
            intended_arrival_settling=True,
            reason='Retained terminal rule explicitly selected hold; not itself a failure')
    return result


def override_label(row):
    return {'TRANSLATION_FOOTPRINT_COVERAGE':'observation/view restriction (motion-dependent footprint)',
        'PREDICTIVE_ARRIVAL_HOLD':'planned arrival settling'}.get(row.get('override_reason'),'blocked recovery/override')
