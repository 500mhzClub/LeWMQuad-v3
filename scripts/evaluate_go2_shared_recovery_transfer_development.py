"""Evaluate each fixed fresh-maze assignment, preserving failures."""
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_multiseed_navigation_development as previous
from scripts import run_go2_shared_recovery_transfer_development as transfer


def evaluate(index, arm):
    study = SimpleNamespace(**(vars(previous.study) | dict(ROOT=transfer.ROOT)))
    return bind(previous.evaluate, study=study)(index, arm)


if __name__ == '__main__':
    study = SimpleNamespace(**(vars(previous.study) | dict(ARMS=transfer.ARMS)))
    bind(previous.main, study=study, evaluate=evaluate)()
