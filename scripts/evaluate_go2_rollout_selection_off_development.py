"""Verify discarded rollout values and physical outcomes of the off control."""
import json
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from lewm.rollout_selection_off_development import select_current_clearance
from scripts import evaluate_go2_multiseed_navigation_development as previous
from scripts import run_go2_rollout_selection_off_development as experiment

base=previous.previous
FITS=base.FITS


def verify_treatment(root,condition):
    # Reuse checks on the computed model/correction arrays, then explicitly
    # distinguish that computation from their absence in command selection.
    evidence=bind(base.verify_treatment,FITS=FITS)(root,condition)
    plans=[p for p in base.read(root,'planning.json') if 'selection' in p]
    blocked=0
    for p in plans:
        s=p['selection']
        expected=select_current_clearance(s['waypoint_body_xy_m'],
            scan_error=s['scan_heading_error_rad'],pulse=s['terminal_translation_pulse']['enabled'],
            clearance_m=s['current_stored_clearance_m'])
        if any(s.get(k)!=v for k,v in expected.items()):
            raise ValueError('selection differs from instantaneous current-clearance control')
        if any(k in s for k in ('memory_forecast_candidates','predictive_arrival_hold',
                'arrival_entry_terminal_priority','clearance_turn','instantaneous_ranking')):
            raise ValueError('prediction-dependent selection evidence unexpectedly retained')
        if s['action']!=p['action']:raise ValueError('executed planning action differs')
        blocked+=not s['current_nominal_disk_clear']
    return evidence|dict(predictive_outcomes_used=False,
        actual_xy_source='learned_computed_but_not_used_for_selection',
        actual_yaw_source='learned_computed_but_not_used_for_selection',
        actual_rollout_selection_off_verified=bool(plans),
        current_clearance_blocked_plans=blocked,model_output_validity_still_checked=True,
        geometric_view_planning_retained=True,actual_dispatch_projection_retained=True)


def evaluate(index,arm):
    study=SimpleNamespace(**(vars(previous.study)|dict(ROOT=experiment.ROOT)))
    provider=SimpleNamespace(**(vars(base)|dict(verify_treatment=verify_treatment)))
    return bind(previous.evaluate,study=study,previous=provider)(index,arm)


if __name__=='__main__':
    study=SimpleNamespace(**(vars(previous.study)|dict(ARMS=experiment.ARMS)))
    bind(previous.main,study=study,evaluate=evaluate)()
