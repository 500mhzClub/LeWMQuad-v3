"""One exposed-maze mission with translation coverage and measured viewpoints."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from lewm.coverage_translation_view_development import CoverageTranslationViewMixin
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_view_arc_no_early_release_development as previous

collection = previous.previous
ROOT = 'go2_coverage_translation_view_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001'
PLAN = Path('docs/go2_coverage_translation_view_plan_2026-09-17.json')


class CoverageViewRuntime(CoverageTranslationViewMixin,previous.NoEarlyReleaseRuntime):
    pass


def source_hashes():
    return previous.source_hashes() | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__,'lewm/coverage_translation_view_development.py')}


def write(name,value):
    if name=='launch.json':
        value=value | dict(experiment='coverage_translation_view_v1',
            comparison_condition='translation_footprint_extension_with_measured_view',
            reference_root_name=previous.ROOT,
            early_preferred_heading_release_enabled=False,early_heading_release_unchanged=True,
            measured_view_arc_fallback_enabled=True,translation_coverage_view_enabled=True,
            coverage_radius_m=.48,coverage_baseline='current footprint and predicted hold sweep',
            coverage_applies_to_translating_selected_actions_only=True,
            pure_turn_clearance_rules_unchanged=True,
            hidden_wall_geometry_used_for_control=False)
    bind(collection.study.source.write,OUTPUT=collection.study.BASE/ROOT)(name,value)


def main():
    if '--prepare' in sys.argv:
        if sys.argv[1:]!=['--prepare']:raise ValueError('prepare separately')
        frozen=json.loads(previous.PLAN.read_text())
        if previous.source_hashes()!=frozen['source_sha256']:
            raise ValueError('preserve completed exposed reference')
        plan=frozen | dict(schema='coverage_translation_view_plan.v1',reference_root=previous.ROOT,
            source_sha256=source_hashes(),
            intervention='reject extra unknown footprint extension for translations and request an existing measured camera viewpoint',
            constraints=['same model, six candidates, sensors, timing and 4800-tick budget',
                'same no-early-release runtime and measured-view arc fallback',
                '0.48-m footprint uses existing nominal plus reserve radii',
                'current and hold-sweep unknown is not declared free',
                'pure turns retain existing eligibility',
                'view resolves only through actual floor or obstacle observation',
                'existing viewpoint search and alternate-view behavior reused',
                'weak visual-support recovery retains priority',
                'native wall geometry excluded from control'],
            secondary_outcomes=['coverage-rejected translations','viewpoint requests and actual observations',
                'unresolved or unreachable views','tracking','physical clearance diagnosis','planning timing'],
            reference_diagnosis='retained opposite wall face overstated clearance before reverse-face observation',
            focused_tests_passed=6,coverage_projection_is_visibility_hypothesis_only=True,
            primary_outcome='physically verified goal-and-home round trip with no disallowed contact')
        with PLAN.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
        print('PREPARED one exposed-maze coverage-view mission',flush=True)
        return
    source=SimpleNamespace(**(vars(collection.study.source)|dict(write=write)))
    study=SimpleNamespace(**(vars(collection.study)|dict(source=source)))
    bind(collection.main,ROOT=ROOT,REFERENCE=previous.ROOT,PLAN=PLAN,
        ViewArcRecoveryRuntime=CoverageViewRuntime,source_hashes=source_hashes,study=study)()


if __name__=='__main__':main()
