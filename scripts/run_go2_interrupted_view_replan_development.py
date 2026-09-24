"""One exposed mission that remembers visual interruption of a viewing pose."""
import hashlib
import json
from pathlib import Path
import sys

from lewm.interrupted_view_replan_development import InterruptedViewReplanMixin
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_cached_fine_connectivity_development as previous

BASE=previous.BASE
ROOT='go2_interrupted_view_replan_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001'
REFERENCE=BASE/previous.ROOT
PLAN=Path('docs/go2_interrupted_view_replan_plan_2026-09-17.json')


class InterruptedViewRuntime(InterruptedViewReplanMixin,previous.CachedFineConnectivityRuntime):
    pass


def source_hashes():
    return previous.source_hashes() | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__,'lewm/interrupted_view_replan_development.py')}


def write(name,value):
    if name=='launch.json':
        value=value | dict(experiment='interrupted_view_replan_v1',
            comparison_condition='exclude_attempted_viewpoint_after_visual_recovery_interrupts_it',
            reference_root_name=previous.ROOT,reference_root_path=str(REFERENCE),
            interrupted_view_replanning_enabled=True,
            viewpoint_failure_does_not_mark_floor_occupied=True,
            feature_and_tracking_thresholds_unchanged=True,
            cached_fine_connectivity_enabled=True,current_position_view_preferred=True,
            route_semantics_intended_unchanged=False,fine_connectivity_semantics_unchanged=True,
            hidden_wall_geometry_used_for_control=False)
    bind(previous.collection.study.source.write,OUTPUT=BASE/ROOT)(name,value)


def main():
    if sys.argv[1:]==['--prepare']:
        frozen=json.loads(previous.PLAN.read_text())
        if previous.source_hashes()!=frozen['source_sha256']:
            raise ValueError('preserve completed cache-trial reference')
        probe=json.loads((REFERENCE/'interrupted_view_saved_activation_v1.json').read_text())
        assert probe['frame']==344 and probe['alternative_viewpoint'] is not None
        plan=frozen | dict(schema='interrupted_view_replan_plan.v1',reference_root=str(REFERENCE),
            source_sha256=source_hashes(),
            intervention='record attempted viewing position as unsuccessful when measured visual recovery interrupts it, then choose another viewpoint',
            constraints=['same model, six actions, sensors, CPU group and 4800-tick budget',
                'same cached routing, current-position view preference and coverage filter',
                'same tracking acceptance and visual-recovery thresholds',
                'same clearance, reserve, stopping and measured dispatch checks',
                'keep recovery heading while it is active',
                'only retire a previously requested nearby camera view, once per recovery trigger',
                'exclude viewing position only; retain floor and unknown target status',
                'native geometry excluded from control'],
            reference_diagnosis='ten recovery interruptions of one frontier view preceded exact-reproduced feature-match loss at frame 678',
            focused_tests_passed=6,
            secondary_outcomes=['interrupted-view exclusions and alternate requests',
                'actual patch observations','tracking','planning timing','physical clearance'])
        with PLAN.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
        print('PREPARED one exposed interrupted-view replan mission',flush=True)
        return
    bind(previous.main,ROOT=ROOT,REFERENCE=REFERENCE,PLAN=PLAN,
        CachedFineConnectivityRuntime=InterruptedViewRuntime,source_hashes=source_hashes,write=write)()


if __name__=='__main__':main()
