"""Fixed JEPA-seed-1401 follow-up of the second original tracking overflow."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_cached_floor_moments_followup_development as first
from scripts import evaluate_go2_multiseed_navigation_development as evaluation

ARM = 'seed_2026091401_full_jepa'


def mark(name, value):
    if name == 'launch.json':
        value = value | dict(extra_sources=value['extra_sources'] | {
            __file__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    bind(first.mark, ARM=ARM, RAW_WRITE=RAW_WRITE)(name, value)


def annotate(name, value):
    bind(first.study.annotate, ARM=ARM, RAW_WRITE=bind(mark, RAW_WRITE=RAW_WRITE))(name, value)


def main():
    if sys.argv[1:] == ['--evaluate']:
        study = SimpleNamespace(**(vars(evaluation.study) | dict(ROOT=first.ROOT)))
        bind(evaluation.evaluate, study=study)(1, ARM)
        return
    if sys.argv[1:] != ['--layout-index', '1', '--arm', ARM]:
        raise ValueError('fixed JEPA-seed-1401 layout-1 follow-up or --evaluate required')
    bind(first.study.main, ROOT=first.ROOT, annotate=annotate,
        initialize_pose=first.initialize_pose)()


if __name__ == '__main__': main()
