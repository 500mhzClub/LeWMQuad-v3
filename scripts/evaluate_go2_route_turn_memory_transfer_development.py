"""Use the existing physical evaluator with fresh-layout result metadata."""
import argparse
from types import SimpleNamespace

from lewm.eligible_floor_registration_development import bind
from scripts import run_go2_route_turn_memory_transfer_development as run


def save(path, value):
    # The shared readout originated in an exposed-layout study. Correct its
    # fixed descriptive flag without changing physical or forecast evaluation.
    if path.name == 'frozen_readout_navigation_readout_v1.json':
        value = value | dict(exposed_layout=False,
            new_independent_development_layout=True, final_evaluation=False,
            prospective_plan=str(run.PLAN))
    run.previous.previous.save(path, value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assignment', type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    bind(run.previous.evaluate, PLAN=run.PLAN, ARMS=run.ARMS,
        root_name=run.root_name, previous=SimpleNamespace(save=save))(args.assignment)
