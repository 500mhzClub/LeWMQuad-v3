"""Evaluate physical navigation and degraded turns in the tracking follow-up."""
from lewm.eligible_floor_registration_development import bind
from scripts import evaluate_go2_auxiliary_turn_recovery_development as previous
from scripts import run_go2_combined_tracking_floor_recovery_development as recovery


if __name__ == '__main__':
    bind(previous.main, recovery=recovery)()
