from lewm.control import lateral_controller_failure_attribution_v2 as subject


def test_fixture():
    assert subject.fixture_payload()["pass"]


def test_reward_interpretability():
    assert subject.tracking_reward(0.2) > 0.85
    for reward in (0.90, 0.75, 0.50):
        error = subject.error_for_reward(reward)
        assert abs(subject.tracking_reward(error) - reward) < 1e-12


def test_path_c_has_precedence_for_proven_reward_defect():
    assert subject.choose_successor_path(
        {
            "v1_requalification_pass": False,
            "plant_or_gait_authority_absent": False,
            "concrete_reward_or_binding_defect": True,
            "bindings_correct": True,
            "policy_command_sensitive": True,
            "v1_failure_classification": "LIKELY_UNDERTRAINED",
        }
    ) == "PATH_C_CORRECTED_SUCCESSOR_TRAINING"
