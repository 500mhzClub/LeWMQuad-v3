"""Two fixed JEPA missions changing only the weak visual-support threshold."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from lewm.earlier_visual_recovery_development import EarlierVisualRecoveryMixin
from scripts import run_go2_view_replan_repeatability_development as previous

BASE=previous.BASE
PLAN=Path('docs/go2_earlier_visual_recovery_plan_2026-09-17.json')
ARMS=('jepa','jepa')
REFERENCE=BASE/previous.root_name(4)


class EarlierVisualRecoveryRuntime(EarlierVisualRecoveryMixin,previous.previous.InterruptedViewRuntime):
    pass


def root_name(number):
    return f'go2_earlier_visual_recovery_{number:02d}_jepa_noise_2mm_native_layout01_4800_v1_attempt_001'


def source_hashes():
    return previous.source_hashes() | {p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
        for p in (__file__,'lewm/earlier_visual_recovery_development.py')}


def write(name,value):
    if name=='launch.json':
        value=value | dict(experiment='earlier_visual_recovery_v1',
            planned_native_assignments=2,weak_visual_support_threshold=72,
            aligned_view_release_support_threshold=72,strong_view_support_threshold=96,
            pose_acceptance_thresholds_unchanged=True,
            intervention='raise weak-view onset and aligned-view release floor from 48 to 72 features',
            reference_batch_completed=True)
    bind(previous.study.source.write,OUTPUT=OUTPUT)(name,value)


def prepare():
    frozen=json.loads(previous.PLAN.read_text())
    if previous.source_hashes()!=frozen['source_sha256']:
        raise ValueError('preserve completed batch sources')
    for number in range(1,5):
        if not (BASE/previous.root_name(number)/'view_replan_repeatability_readout_v1.json').exists():
            raise ValueError('complete the previous fixed batch first')
    if any((BASE/root_name(n)).exists() for n in (1,2)):
        raise ValueError('preserve both assigned attempts')
    probe=json.loads((REFERENCE/'earlier_visual_recovery_saved_probe_v2.json').read_text())
    assert probe['matched_original_planning_recovery_states']==116
    assert probe['same_reference_warning_gain_ms']==500
    plan=frozen | dict(schema='earlier_visual_recovery_plan.v1',
        assignments=[[1,a] for a in ARMS],root_names=[root_name(n) for n in (1,2)],
        source_sha256=source_hashes(),reference_root=str(REFERENCE),planned_native_assignments=2,
        intervention='weak-feature onset and aligned-view release floor 48 to 72; strong-reference threshold remains 96',
        saved_same_reference_warning_gain_ms=500,focused_tests_passed=4,
        constraints=['same frozen JEPA model and six actions','same pose acceptance, sensors and CPU group',
            'same .20-m measured-reference locality and no age expiry',
            'same predictive clearance, coverage and dispatch guards',
            'same 4800-tick budget and deadlines','no tuning between the two runs'],
        limitations=['one exposed maze, one frozen JEPA training seed, two runs',
            'feature counts are an uncalibrated development heuristic',
            'more frequent or longer recovery may stall exploration',
            'earlier warning on saved sensors is not proof of prevented failure'])
    previous.save(PLAN,plan)
    print('PREPARED two fixed JEPA missions at weak-feature threshold 72',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--evaluate',action='store_true');parser.add_argument('--assignment',type=int,choices=(1,2))
    args=parser.parse_args()
    if args.prepare:
        if args.assignment is not None or args.evaluate:raise ValueError('prepare separately')
        return prepare()
    if args.assignment is None:raise ValueError('fixed assignment required')
    options=dict(BASE=BASE,PLAN=PLAN,ARMS=ARMS,root_name=root_name)
    if args.evaluate:return bind(previous.evaluate,**options)(args.assignment)
    source=SimpleNamespace(**(vars(previous.study.source)|dict(write=write)))
    study=SimpleNamespace(**(vars(previous.study)|dict(source=source)))
    runtime=SimpleNamespace(ROOT=REFERENCE.name,InterruptedViewRuntime=EarlierVisualRecoveryRuntime)
    return bind(previous.run,**options,REFERENCE=REFERENCE,source_hashes=source_hashes,
        previous=runtime,study=study)(args.assignment)


if __name__=='__main__':main()
