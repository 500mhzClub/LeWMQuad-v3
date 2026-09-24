"""Physical and actual-selection checks for the stronger non-predictive control."""
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from lewm.current_reserve_terminal_feedback_development import select_reserved_terminal
from scripts import evaluate_go2_rollout_selection_off_development as previous
from scripts import run_go2_current_reserve_terminal_feedback_development as experiment


def evaluate(index,arm):
    checked=bind(previous.verify_treatment,select_current_clearance=select_reserved_terminal)
    provider=SimpleNamespace(**(vars(previous.base)|dict(verify_treatment=checked)))
    study=SimpleNamespace(**(vars(previous.previous.study)|dict(ROOT=experiment.ROOT)))
    return bind(previous.previous.evaluate,study=study,previous=provider)(index,arm)


if __name__=='__main__':
    study=SimpleNamespace(**(vars(previous.previous.study)|dict(ARMS=experiment.ARMS)))
    bind(previous.previous.main,study=study,evaluate=evaluate)()
