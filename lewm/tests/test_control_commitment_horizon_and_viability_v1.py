import numpy as np

from lewm.safety import control_commitment_horizon_and_viability_v1 as V


def test_fixture_is_deterministic():
    first = V.fixture_payload()
    assert first == V.fixture_payload()
    assert first["pass"]


def test_prefix_integrator_uses_only_selected_ticks():
    commands = [[[0.2, 0.0, 0.0]] * 5]
    one = V.integrate_prefix(commands, [1.0, 0.0, 0.0, 1.0], 1)
    five = V.integrate_prefix(commands, [1.0, 0.0, 0.0, 1.0], 5)
    assert np.isclose(one[0], 0.02)
    assert np.isclose(five[0], 0.1)


def test_contact_before_divergence_is_not_candidate_bank_failure():
    rows = [
        {"committed_contact": True, "first_contact_step": 2, "realised_progress_m": 0.0},
        {"committed_contact": True, "first_contact_step": 3, "realised_progress_m": 0.0},
    ]
    assert V.availability_class(rows, boundary_contact=False, divergence_step=3) == (
        "CONTACT_PRECEDES_CANDIDATE_DIVERGENCE"
    )


def test_shorter_trace_is_not_called_deployable_when_interface_unresolved():
    assert V.viability_class(
        boundary_contact=False,
        safe_counts={1: 2, 2: 1, 3: 0, 4: 0, 5: 0},
        first_contact_steps=[120, 130],
        divergence_step=2,
        shorter_horizon_technically_available=False,
    ) == "UNRESOLVED"
