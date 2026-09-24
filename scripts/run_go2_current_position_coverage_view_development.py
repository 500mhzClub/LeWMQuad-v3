"""One exposed mission preferring a measured current-position coverage view."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from lewm.current_position_coverage_view_development import CurrentPositionCoverageMixin
from lewm.eligible_floor_registration_development import bind
from scripts.navigation_artifact_root_development import validate_root
from scripts import run_go2_coverage_translation_view_development as previous

collection=previous.collection
BASE=Path('/mnt/steam_drive/LeWMQuad-v3/navigation_development_artifacts_v1')
ROOT='go2_current_position_coverage_view_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001'
REFERENCE=collection.study.BASE/previous.ROOT
PLAN=Path('docs/go2_current_position_coverage_view_plan_2026-09-17.json')


class CurrentPositionCoverageRuntime(CurrentPositionCoverageMixin,previous.CoverageViewRuntime):
    pass


def source_hashes():
    return previous.source_hashes() | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__,'lewm/current_position_coverage_view_development.py')}


def write(name,value):
    if name=='launch.json':
        value=value | dict(experiment='current_position_coverage_view_v1',
            comparison_condition='prefer_measured_current_position_coverage_view',
            reference_root_name=previous.ROOT,reference_root_path=str(REFERENCE),
            output_base=str(BASE),model_input_base=str(collection.study.BASE),
            translation_coverage_view_enabled=True,coverage_radius_m=.48,
            current_position_view_preferred=True,early_preferred_heading_release_enabled=False,
            measured_view_arc_fallback_enabled=True,hidden_wall_geometry_used_for_control=False)
    bind(collection.study.source.write,OUTPUT=BASE/ROOT)(name,value)


def main():
    if '--prepare' in sys.argv:
        if sys.argv[1:]!=['--prepare']:raise ValueError('prepare separately')
        frozen=json.loads(previous.PLAN.read_text())
        if previous.source_hashes()!=frozen['source_sha256']:
            raise ValueError('preserve completed coverage-view reference')
        plan=frozen | dict(schema='current_position_coverage_view_plan.v1',
            reference_root=str(REFERENCE),output_base=str(BASE),source_sha256=source_hashes(),
            intervention='prefer turning at measured current position when calibrated projection supports the requested patch',
            constraints=['same model, six candidates, sensors, timing and 4800-tick budget',
                'same coverage translation filter and no-early-release runtime',
                'all existing clearance, reserve, stopping and dispatch checks remain',
                'current footprint floor is not declared observed',
                'projection is a hypothesis; actual mapped floor or obstacle resolves request',
                'fresh aligned but unknown patch triggers existing alternate-view search',
                'weak visual-support recovery retains priority','native geometry excluded from control'],
            reference_diagnosis='1174 translation rejections for one patch; all 74 early replayed states supported current-position directed projection',
            focused_tests_passed=9,
            secondary_outcomes=['current-position view requests and actual resolutions',
                'coverage rejection and unresolved views','tracking','planning timing'])
        with PLAN.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
        print('PREPARED one exposed current-position coverage-view mission',flush=True)
        return
    source=SimpleNamespace(**(vars(collection.study.source)|dict(write=write,
        main=bind(collection.study.source.main,validate_root=bind(validate_root,BASE=BASE)))))
    # Only output/reference resolution changes. load_model retains its original
    # module globals and original model-input base; its plan binding is reused.
    study=SimpleNamespace(**(vars(collection.study)|dict(source=source,BASE=BASE)))
    bind(collection.main,ROOT=ROOT,REFERENCE=REFERENCE,PLAN=PLAN,
        ViewArcRecoveryRuntime=CurrentPositionCoverageRuntime,source_hashes=source_hashes,study=study)()


if __name__=='__main__':main()
