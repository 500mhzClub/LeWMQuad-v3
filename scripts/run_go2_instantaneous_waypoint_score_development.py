"""Same frozen controller with instantaneous main action utilities."""
import hashlib
from pathlib import Path
from lewm.eligible_floor_registration_development import bind
from lewm.instantaneous_waypoint_score_development import InstantaneousWaypointScoreMixin
from scripts import run_go2_shadow_stopping_projection_development as runner

reference = runner.reference
ARMS = runner.ARMS
ROOT = 'go2_instantaneous_waypoint_score_{arm}_noise_2mm_native_layout{index:02d}_4800_v1_attempt_001'


class InstantaneousScoreRuntime(InstantaneousWaypointScoreMixin,reference.SignedLearnedRuntime):
    pass


def mark(name,value):
    if name=='launch.json':
        value = value | dict(experiment='instantaneous_waypoint_score_v1',
            comparison='forecast_main_utilities_vs_instantaneous_cost_directional_derivative',
            reference_root_name=reference.ROOT.format(index=INDEX,arm=value['study_arm']),
            planned_conditions=list(ARMS),planned_native_assignments=2,
            fixed_sequential_assignments=list(runner.ASSIGNMENTS),
            new_independent_development_layout=False,exposed_development_layout=True,
            repeated_exposed_development_maze=True,
            layout_novelty_scope='repeated_shared_recovery_development_layouts',
            actual_runtime_class='InstantaneousScoreRuntime',raw_tracker_changed=False,
            main_action_score_uses_predicted_outcomes=False,
            predictive_clearance_recovery_arrival_and_stopping_retained=True,
            full_online_rollout_ablation=False,
            extra_sources=value['extra_sources']|{p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in (__file__,'lewm/instantaneous_waypoint_score_development.py',runner.__file__)})
    RAW_WRITE(name,value)


def annotate(name,value):
    bind(reference.annotate,ARM=ARM,RAW_WRITE=bind(mark,INDEX=INDEX,RAW_WRITE=RAW_WRITE))(name,value)


def main():
    bind(runner.main,ROOT=ROOT,annotate=annotate,ShadowStoppingRuntime=InstantaneousScoreRuntime)()


if __name__=='__main__':main()
