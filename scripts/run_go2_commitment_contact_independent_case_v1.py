"""Run one separate, fixed-model contact-horizon case using the existing runner."""
import argparse

from scripts import run_go2_stop_conditioned_independent_case_v1 as original
from scripts import commitment_contact_independent_pipeline_development as pipeline

MODELS = ('seed_2026091001_full_jepa', 'seed_2026091001_full_supervised_rollout')
SOURCE = 'scripts/run_go2_commitment_contact_independent_case_v1.py'
PROTOCOL = 'docs/go2_commitment_contact_scoring_experiment_2026-09-13.md'
SOURCES = original.SOURCES + (SOURCE, PROTOCOL,
    'scripts/commitment_contact_independent_pipeline_development.py',
    'lewm/commitment_contact_controller_development.py',
    'lewm/tests/test_commitment_contact_controller_development.py',
    'lewm/tests/test_commitment_contact_pipeline_development.py')


def assignment(layout, mode, model_name):
    pipeline.require_mode(mode)
    if layout != 0 or model_name not in MODELS:
        raise ValueError('layout 0 and one of the two fixed contact-horizon model assignments required')
    return original.assignment(layout, 'frozen_reference', model_name) | dict(mode=mode)


main = pipeline.bind(original.main, pipeline=pipeline, assignment=assignment,
    SOURCES=SOURCES, PROTOCOL=PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout', type=int, choices=(0,), required=True)
    parser.add_argument('--model', choices=MODELS, required=True)
    args = parser.parse_args()
    main(args.layout, pipeline.MODE, args.model)
