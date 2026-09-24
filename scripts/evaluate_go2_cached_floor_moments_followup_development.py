"""Evaluate the tracking follow-up with the original physical and model checks."""
from types import SimpleNamespace
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_multiseed_navigation_development as original
from scripts import run_go2_cached_floor_moments_followup_development as followup


if __name__ == '__main__':
    study = SimpleNamespace(**(vars(original.study) | dict(ROOT=followup.ROOT)))
    bind(original.evaluate, study=study)(1, followup.ARM)
