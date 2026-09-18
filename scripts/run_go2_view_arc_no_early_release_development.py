"""Isolate early heading release in the exposed supervised turn-cycle trial."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.full_reserve_heading_release_development import FullReserveHeadingReleaseRuntime
from lewm.no_early_heading_release_development import NoEarlyHeadingReleaseMixin
from scripts import run_go2_measured_view_arc_recovery_development as previous

ROOT = 'go2_view_arc_no_early_release_supervised_rollout_noise_2mm_native_layout01_4800_v1_attempt_001'
PLAN = Path('docs/go2_view_arc_no_early_release_plan_2026-09-17.json')


class NoEarlyReleaseRuntime(previous.ViewArcRecoveryRuntime,
        NoEarlyHeadingReleaseMixin, FullReserveHeadingReleaseRuntime):
    pass


def source_hashes():
    return previous.source_hashes() | {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__, 'lewm/no_early_heading_release_development.py')}


def write(name, value):
    if name == 'launch.json':
        value = value | dict(experiment='view_arc_no_early_release_v1',
            comparison_condition='suppress_early_preferred_heading_release',
            reference_root_name=previous.ROOT,
            early_preferred_heading_release_enabled=False,
            early_heading_release_unchanged=False,
            measured_view_arc_fallback_enabled=True,
            original_clearance_and_stopping_checks_retained=True,
            translation_progress_release_retained=True,
            only_intervention='disable early preferred-heading reversal of a latched recovery turn')
    bind(previous.study.source.write, OUTPUT=previous.study.BASE/ROOT)(name, value)


def main():
    mro = NoEarlyReleaseRuntime.__mro__
    assert mro.index(NoEarlyHeadingReleaseMixin)+1 == mro.index(FullReserveHeadingReleaseRuntime)
    if '--prepare' in sys.argv:
        if sys.argv[1:] != ['--prepare']:
            raise ValueError('prepare separately')
        frozen = json.loads(previous.PLAN.read_text())
        if previous.source_hashes() != frozen['source_sha256']:
            raise ValueError('preserve completed exposed reference')
        evaluation = json.loads((previous.study.BASE/previous.ROOT/'short_pulse_navigation_evaluation_v1.json').read_text())
        assert evaluation['terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
        plan = frozen | dict(schema='view_arc_no_early_release_plan.v1',
            reference_root=previous.ROOT, source_sha256=source_hashes(),
            intervention='suppress early preferred-heading reversal of a latched recovery turn',
            constraints=['same model, six candidates, timing, sensors and mission budget',
                'retain measured-view arc fallback in both reference and intervention',
                'retain measured completion, translation release and clearance direction switching',
                'retain all nominal/reserve, stopping and dispatch checks'],
            secondary_outcomes=['suppressed release opportunities and final selected actions',
                'actual dispatched turning and direction reversals', 'clearance turn completion',
                'view-arc fallback activation', 'tracking loss', 'planning timing'],
            reference_diagnosis='88 early releases, 83 followed by relatching within two seconds during frames 800-3796',
            matched_asynchronous_trajectory_guaranteed=False)
        with PLAN.open('x') as f:
            json.dump(plan, f, indent=2); f.write('\n')
        print('PREPARED one exposed-maze early-release ablation', flush=True)
        return
    source = SimpleNamespace(**(vars(previous.study.source) | dict(write=write)))
    study = SimpleNamespace(**(vars(previous.study) | dict(source=source)))
    bind(previous.main, ROOT=ROOT, REFERENCE=previous.ROOT, PLAN=PLAN,
        ViewArcRecoveryRuntime=NoEarlyReleaseRuntime, source_hashes=source_hashes,
        study=study)()


if __name__ == '__main__':
    main()
