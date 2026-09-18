"""One exposed native mission with exact fine-connectivity result reuse."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from lewm.cached_fine_connectivity_development import CachedFineConnectivityMixin
from lewm.cached_fine_goal_route_development import CachedFineGoalRecoveryRuntime
from lewm.eligible_floor_registration_development import bind
from scripts.navigation_artifact_root_development import validate_root
from scripts import run_go2_current_position_coverage_view_development as previous
from scripts import evaluate_go2_short_pulse_navigation_development as evaluation

BASE=previous.BASE
ROOT='go2_cached_fine_connectivity_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001'
REFERENCE=BASE/previous.ROOT
PLAN=Path('docs/go2_cached_fine_connectivity_plan_2026-09-17.json')
collection=previous.collection


class CachedFineConnectivityRuntime(previous.CurrentPositionCoverageRuntime,
        CachedFineConnectivityMixin,CachedFineGoalRecoveryRuntime):
    pass


def source_hashes():
    return previous.source_hashes() | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__,'lewm/cached_fine_connectivity_development.py')}


def write(name,value):
    if name=='launch.json':
        value=value | dict(experiment='cached_fine_connectivity_v1',
            comparison_condition='identical_observed_graph_search_cache',
            reference_root_name=previous.ROOT,reference_root_path=str(REFERENCE),
            output_base=str(BASE),model_input_base=str(collection.study.BASE),
            cached_fine_connectivity_enabled=True,current_position_view_preferred=True,
            route_semantics_intended_unchanged=True,deadline_and_extra_delay_unchanged=True,
            current_pose_connectors_rechecked=True,hidden_wall_geometry_used_for_control=False)
    bind(collection.study.source.write,OUTPUT=BASE/ROOT)(name,value)


def main():
    mro=CachedFineConnectivityRuntime.__mro__
    assert mro.index(CachedFineConnectivityMixin)+1==mro.index(CachedFineGoalRecoveryRuntime)
    if sys.argv[1:]==['--prepare']:
        frozen=json.loads(previous.PLAN.read_text())
        if previous.source_hashes()!=frozen['source_sha256']:
            raise ValueError('preserve completed current-position reference')
        evidence=json.loads((REFERENCE/'cached_fine_connectivity_comparison_v1.json').read_text())
        assert evidence['all_outputs_match_except_elapsed_time'] and evidence['compared_calls']==120
        plan=frozen | dict(schema='cached_fine_connectivity_plan.v1',
            reference_root=str(REFERENCE),source_sha256=source_hashes(),
            intervention='cache exact fine-grid A-star result and shortlist nearby floor cells before original exact connector ordering',
            constraints=['same model, six candidates, sensors, CPU group and 4800-tick budget',
                'same 300-ms deadline and additional 20-ms planning delay',
                'same current-position viewing and translation coverage guard',
                'cache keys include exact floor, coarse and fine obstacles, radius, seed and target',
                'continuous start and goal connectors rechecked at exact current coordinates',
                'same graph edges, costs and tie-breaking','all dispatch and clearance checks remain'],
            reference_diagnosis='all 710 plans after frame 1600 late; fine-goal search repeatedly failed on unchanged graph',
            focused_tests_passed=4,saved_comparison_calls=120,
            secondary_outcomes=['planning deadlines','route computation time','queue overflow',
                'coverage-view resolutions','physical clearance and contacts'])
        with PLAN.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
        print('PREPARED one exposed cached-connectivity mission',flush=True)
        return
    if sys.argv[1:]==['--evaluate']:
        selected=SimpleNamespace(BASE=BASE,ROOT=ROOT,PLAN=PLAN,ASSIGNMENTS=((1,'supervised_rollout'),))
        return bind(evaluation.evaluate,study=selected,
            xy=bind(evaluation.xy,validate_root=bind(validate_root,BASE=BASE)))(1)
    source=SimpleNamespace(**(vars(collection.study.source)|dict(write=write,
        main=bind(collection.study.source.main,validate_root=bind(validate_root,BASE=BASE)))))
    study=SimpleNamespace(**(vars(collection.study)|dict(source=source,BASE=BASE)))
    bind(collection.main,ROOT=ROOT,REFERENCE=REFERENCE,PLAN=PLAN,
        ViewArcRecoveryRuntime=CachedFineConnectivityRuntime,source_hashes=source_hashes,study=study)()


if __name__=='__main__':main()
