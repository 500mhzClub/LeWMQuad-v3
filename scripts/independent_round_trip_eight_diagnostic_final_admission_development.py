"""Join authenticated inputs, all eight diagnostics and a bound policy review.

This verifier supports the existing fixed 32-case definition only. It never
creates a review or launches a scene. A review requesting a policy/budget
change requires a separately implemented and checked successor definition.
"""
from copy import deepcopy
import json

from scripts import independent_round_trip_population_inputs_development as inputs
from scripts import independent_round_trip_later_diagnostics_evidence_development as queue
from scripts import independent_round_trip_final_admission_development as original
from scripts import independent_round_trip_audit_overlap_admission_development as overlap
from scripts import independent_round_trip_population_runtime_development as runtime
from lewm.independent_reactive_floor_transport_study_development import merge_sources
from lewm.novel_maze_round_trip_contract_development import NAVIGATION_TICKS
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, verify
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.navigation_artifact_root_development import verify_artifacts
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

SOURCE='scripts/independent_round_trip_eight_diagnostic_final_admission_development.py'
TEST='lewm/tests/test_independent_round_trip_eight_diagnostic_final_admission_development.py'
PROTOCOL='docs/go2_independent_round_trip_eight_diagnostic_final_admission_v1_2026-09-11.md'
REVIEW='docs/go2_independent_round_trip_eight_diagnostic_final_policy_review_v1.json'
REASONS=('policy_and_models','navigation_budget','sensing_failures',
         'causal_comparison_scope','timing_and_hardware_scope','later_diagnostic_outcomes')


def prepared_sources():
    sources=merge_sources(inputs.prepared_sources(),queue.prepared_sources())
    sources=merge_sources(sources,overlap.prepared_sources())
    sources=discover_sources((SOURCE,TEST,PROTOCOL),sources)
    verify(sources)
    return sources


def require_links(input_admission,queue_admission):
    return original.require_links(input_admission,queue_admission['original_five_stage_admission'])


def reconstruct_full_queue(admission,sources):
    five=original.reconstruct_full_queue(admission['original_five_stage_admission'],sources)
    return queue.admit(five,admission['later_waiter_result_sha256'],sources=sources)


def review_evidence(input_admission,queue_admission):
    """Extract after verify_population has reauthenticated the complete queue."""
    require_links(input_admission,queue_admission)
    later=queue_admission['later_diagnostics']
    if (queue_admission['all_eight_diagnostics_authenticated'] is not True
            or [r['stage'] for r in later]!=[s.name for s in queue.STAGES]):
        raise ValueError('all eight authenticated diagnostic outcomes required')
    return dict(original_five_diagnostic_evidence=original.review_evidence(input_admission,
            queue_admission['original_five_stage_admission']),later_diagnostics=deepcopy(later),
        diagnostic_count=8,final_policy_review_completed=False,
        population_definition_selected=False,population_execution_permitted=False)


def require_review(review,input_admission,queue_admission,evidence):
    expected=dict(schema='independent_round_trip_eight_diagnostic_final_policy_review.v1',
        status='COMPLETE_EIGHT_DIAGNOSTIC_POLICY_REVIEW',
        decision='execute_original_fixed_32_case_definition',
        study_manifest=inputs.study.manifest(),navigation_ticks=NAVIGATION_TICKS,
        input_admission_sha256=fingerprint(input_admission),
        eight_stage_queue_admission_sha256=fingerprint(queue_admission),
        development_evidence=evidence,all_scientific_failures_retained=True,
        policy_or_budget_changes_requested=False,independent_layout_sensor_data_consumed=False,
        prior_navigation_qualification_claimed=False,prior_real_time_qualification_claimed=False,
        prior_hardware_qualification_claimed=False)
    if set(review)!=set(expected)|{'rationale'}:
        raise ValueError('complete explicit fixed-policy review schema required')
    if fingerprint({k:review[k] for k in expected})!=fingerprint(expected):
        raise ValueError('review must match complete actual outcomes, inputs, policy and budget')
    reasons=review['rationale']
    if (type(reasons) is not dict or set(reasons)!=set(REASONS)
            or any(type(v) is not str or not v.strip() for v in reasons.values())):
        raise ValueError('explicit policy, budget, sensing, comparison and timing reasoning required')
    # This authenticates the written decision and its evidence; it cannot
    # mechanically certify the quality of its scientific reasoning.


def verify_population(launch,*,full=False):
    if type(full) is not bool:raise ValueError('explicit full-input verification boolean required')
    # Live work is rejected before large input rehashes or reading a review.
    queue.owners_ended()
    sources=launch['source_sha256'];verify(sources)
    required=prepared_sources()
    if any(sources.get(n)!=h for n,h in required.items()):
        raise ValueError('complete frozen joined admission and monitored runtime sources required')
    runtime.require_verifier(verify_population,launch)
    binding=launch['policy_review']
    if (set(binding)!= {'path','sha256'} or binding['path']!=REVIEW
            or sources.get(REVIEW)!=binding['sha256']):
        raise ValueError('exact source-bound ordinary final review document required')
    verify({REVIEW:binding['sha256']})
    input_admission=launch['input_admission'];queue_admission=launch['eight_stage_queue_admission']
    require_links(input_admission,queue_admission)
    inputs.verify_bound(input_admission,sources)
    queue.verify_bound(queue_admission,sources)
    if full:
        if (fingerprint(inputs.admit(input_admission['adapter_batch_result_sha256'],sources))
                !=fingerprint(input_admission)):
            raise ValueError('full original population input admission changed')
        if fingerprint(reconstruct_full_queue(queue_admission,sources))!=fingerprint(queue_admission):
            raise ValueError('full original eight-stage queue admission changed')
    evidence=review_evidence(input_admission,queue_admission)
    review=json.loads((ROOT/REVIEW).read_text())
    require_review(review,input_admission,queue_admission,evidence)
    expected=dict(final_policy_review_completed=True,complete_input_admission_performed=True,
        native_queue_completion_verified=True,all_eight_diagnostics_reviewed=True,ordered_cases=inputs.study.manifest()['ordered_cases'],
        runtime=runtime.FIXED_RUNTIME,staged_runtime=overlap.driver.staged.FIXED,
        audit_cpu_monitor=overlap.monitored.FIXED,overlap_evidence=overlap.EVIDENCE,
        population_entrypoint=overlap.ENTRYPOINT,
        overlap_verifier=dict(source=overlap.SOURCE,function='verify_overlap'))
    if fingerprint({k:launch[k] for k in expected})!=fingerprint(expected):
        raise ValueError('reviewed fixed cases and mandatory monitored driver required')
    # The actual driver's independent overlap callback authenticates its
    # bounded CPU proof outputs. These fields do not replace that callback.
    queue.owners_ended();verify(sources);verify({REVIEW:binding['sha256']})
