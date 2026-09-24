"""Same progressing action task with an explicitly new ordered5mm camera scene."""
from dataclasses import replace
import hashlib
import json
from lewm.geometry_progress_pilot_development import (TRIALS, ACTIONS, APPEARANCES, GEOMETRIES,
    WARMUP_TICKS, HORIZON_TICKS, assignments, schedule, decision, candidate_commands,
    timed_candidate, progress_outcome, panel_informativeness,
    specification as previous_specification, pack as previous_pack)


def specification(trial):
    return previous_specification(trial)|dict(scene_id='geometry-progress-near-field-v1-'+trial,
        family='GEOMETRY_PROGRESS_ORDERED_NEAR_FIELD_PILOT',render_near_m=.005,
        visual_surface_contract='floor_first_ordered_union_walls')


def pack(spec):
    if spec != specification(spec['trial']):
        raise ValueError('exact new ordered near-field pilot specification required')
    base=previous_pack(previous_specification(spec['trial']))
    return replace(base,scene_id=spec['scene_id'],family=spec['family'],
        camera=replace(base.camera,near_m=.005),
        manifest_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest())


def measurement_gate(reports):
    if len(reports)!=24 or [r['trial'] for r in reports]!=list(TRIALS):
        raise ValueError('complete ordered24episode cohort required')
    panel=panel_informativeness(reports)
    hard={r['trial']:r['hard_measurement_failed_frames'] for r in reports if r['hard_measurement_failed_frames']}
    complete=all(r['setup_admitted'] and r['targets'] is not None and r['frames']>0
        and (r['outcome']['complete_horizon'] or r['outcome']['physical_stop']=='DISALLOWED_CONTACT')
        and r['outcome']['acquisition_stop'] is None for r in reports)
    return dict(native_action_design=panel,all_candidate_acquisitions_complete=complete,
        hard_measurement_failed_cases=hard,
        strict_depth_failed_cases=[r['trial'] for r in reports if not r['strict_physical_visibility_pass']],
        prediction_design_and_measurement_gate_pass=bool(complete and not hard and panel['informative_for_next_dataset']),
        depth_navigation_qualified=False,rgb_benefit_established=False,model_trained=False,
        independent_maze_evaluation_layouts=0,navigation_qualified=False,goal_achieved=False)
