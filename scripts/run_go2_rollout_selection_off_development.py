"""Two fixed exposed-maze missions with learned-rollout selection disabled."""
import hashlib
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.rollout_selection_off_development import RolloutSelectionOffMixin
from scripts import run_go2_shadow_stopping_projection_development as runner

reference=runner.reference
ARMS=runner.ARMS
ROOT='go2_rollout_selection_off_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


class RolloutSelectionOffRuntime(RolloutSelectionOffMixin,reference.SignedLearnedRuntime):
    pass


def mark(name,value):
    if name=='launch.json':
        value=value|dict(experiment='rollout_selection_off_v1',
            comparison='learned_rollout_selection_vs_instantaneous_current_clearance_feedback',
            reference_root_name=reference.ROOT.format(index=INDEX,arm=value['study_arm']),
            intermediate_reference_root_name=f'go2_instantaneous_waypoint_score_{value["study_arm"]}_noise_2mm_native_layout{INDEX:02d}_4800_v1_attempt_001',
            planned_conditions=list(ARMS),planned_native_assignments=2,
            fixed_sequential_assignments=list(runner.ASSIGNMENTS),
            new_independent_development_layout=False,exposed_development_layout=True,
            repeated_exposed_development_maze=True,
            layout_novelty_scope='repeated_shared_recovery_development_layouts',
            actual_runtime_class='RolloutSelectionOffRuntime',raw_tracker_changed=False,
            main_action_score_uses_predicted_outcomes=False,
            learned_candidate_rollouts_used_for_selection=False,
            predictive_clearance_recovery_arrival_and_stopping_retained=False,
            model_forecasts_computed_for_workload_control=True,
            total_computation_identical=False,model_output_validity_still_checked=True,
            geometric_view_planning_retained=True,actual_dispatch_projection_retained=True,
            full_learned_rollout_selection_disabled=True,
            turn_recovery_releases_for_full_reserve_progress=False,
            turn_prediction_error_reserve_m=0.,stepwise_turn_reserve_recovery=False,
            predictive_terminal_arrival_hold=False,waypoint_predicted_alignment_progress=False,
            translation_prediction_error_reserve_m=0.,
            reserve_recovery_requires_no_further_encroachment=False,
            terminal_priority_requires_predicted_arrival=False,
            hold_relative_clearance_recovery=False,hold_relative_heading_recovery=False,
            downstream_planner_and_recovery_unchanged=False,predictive_planning_in_both_arms=False,
            planned_stopping_projection=False,perception_and_recovery_shared_across_conditions=False,
            perception_and_measured_view_recovery_shared=True,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'lewm/rollout_selection_off_development.py',
                    'lewm/instantaneous_waypoint_score_development.py',runner.__file__)})
    RAW_WRITE(name,value)


def annotate(name,value):
    bind(reference.annotate,ARM=ARM,RAW_WRITE=bind(mark,INDEX=INDEX,RAW_WRITE=RAW_WRITE))(name,value)


def main():
    bind(runner.main,ROOT=ROOT,annotate=annotate,ShadowStoppingRuntime=RolloutSelectionOffRuntime)()


if __name__=='__main__':main()
