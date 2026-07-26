from __future__ import annotations

import builtins
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/"
    "go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement.py"
)
V10_CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_rgb_jepa_encoder_pretraining_v1.py"
)
V10_TEST_PATH = (
    ROOT
    / "lewm/tests/test_go2_rgb_jepa_encoder_pretraining_v1_contract.py"
)
RUNNER_PATH = (
    ROOT
    / "scripts/"
    "run_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement.py"
)
LAUNCHER_PATH = (
    ROOT
    / "scripts/"
    "launch_go2_rgb_action_conditioned_next_target_retrieval_jepa_"
    "v10r_integrity_replacement.py"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


contract = _load("_test_retrieval_jepa_v10r_contract", CONTRACT_PATH)
v10_contract = _load(
    "_test_retrieval_jepa_v10r_frozen_v10_contract",
    V10_CONTRACT_PATH,
)
v10_fixtures = _load(
    "_test_retrieval_jepa_v10r_frozen_v10_fixtures",
    V10_TEST_PATH,
)


FLOAT32_EQUAL_LOGIT_REFERENCE = 2.1972246170043945
FLOAT32_THREE_ULP_REDUCED_MEAN = 2.1972239017486572


def _update0_with_action_nll(
    observed: float,
) -> tuple[dict, dict]:
    metrics = v10_fixtures._update100_metrics()
    update0 = v10_fixtures._update0_metrics()
    for value in (metrics, update0):
        value["factorized_retrieval"][
            "action_equal_logit_reference"
        ] = FLOAT32_EQUAL_LOGIT_REFERENCE
    retrieval = update0["factorized_retrieval"]
    retrieval["action_retrieval_nll"] = observed
    for row in retrieval[
        "per_executed_action_action_retrieval"
    ].values():
        row["mean_nll"] = observed
    return metrics, update0


def test_v10r_science_contract_is_deep_equal_to_frozen_v10() -> None:
    v10 = v10_contract.science_contract()
    v10r = contract.science_contract()
    assert contract.frozen_v10_science_contract() == v10
    assert contract.normalize_v10r_operational_identity(v10r) == v10
    assert contract.canonical_json_sha256(v10) == (
        "3dfcc168a570c9e69be57d03b0281fb66ce08b7acc33f1929a0b9e54237f158c"
    )
    assert v10r["phase_a"] == v10["phase_a"]
    assert v10r["phase_b"] == v10["phase_b"]
    assert v10r["cumulative_caps"] == v10["cumulative_caps"]
    assert v10r["schema"].startswith(contract.SCHEMA_PREFIX)
    lifecycle = v10r["lifecycle"]
    assert lifecycle["output_root"] == contract.OUTPUT_ROOT_RELATIVE_PATH
    assert lifecycle["output_root"] != contract.V10_OUTPUT_ROOT_RELATIVE_PATH
    failure = lifecycle["operational_failure"]
    assert failure["failure_status"] == contract.OPERATIONAL_FAILURE_STATUS
    assert failure["reservation_publication_failure_status"] == (
        contract.RESERVATION_PUBLICATION_FAILURE_STATUS
    )
    assert contract.V10R_OPERATIONAL_IDENTITY_LEAVES == (
        "/schema",
        "/lifecycle/output_root",
        "/lifecycle/operational_failure/failure_status",
        (
            "/lifecycle/operational_failure/"
            "reservation_publication_failure_status"
        ),
    )
    assert (
        "final_single_frame_v5_family_closure_exact"
        not in contract.SCIENTIFIC_REVIEW_CHECKS
    )
    for field in (
        (
            "v10r_contract_normalizes_to_frozen_v10_at_only_four_"
            "operational_identity_leaves_exact"
        ),
        "sole_eight_float32_epsilon_integrity_adapter_exact",
        "one_v10r_only_limited_supersession_exact",
        "no_further_integrity_replacement_authorized",
    ):
        assert contract.SCIENTIFIC_REVIEW_CHECKS[field] is True


def test_v10r_preregistration_is_exactly_bound() -> None:
    assert contract.preregistration_binding() == {
        "path": contract.PREREGISTRATION_RELATIVE_PATH,
        "commit": "bdf30305645efbcde56c7e52711e2ded7bf728fb",
        "file_sha256": (
            "38e3f4d9378d4974f77b4a10b069a704b6722caea31bd97f237f0eac00f2308a"
        ),
        "byte_count": 16_613,
    }
    assert contract.PREREGISTRATION_REVIEW_RELATIVE_PATH in (
        contract.SOURCE_REVIEW_ADDITIONAL_PATHS
    )
    assert contract.PREREGISTRATION_REVIEW_COMMIT == (
        "5d532e814c73c7c8238a59cf853e9cef4975c541"
    )
    assert contract.PREREGISTRATION_REVIEW_FILE_SHA256 == (
        "606138757d9292ef3c8a75f16c1e8abb34da5fa11d84db60362d126b68cf2acf"
    )
    assert contract.PREREGISTRATION_REVIEW_CONTENT_SHA256 == (
        "8f1316b203734fdf844cc04819e8b370510258da3d6680381667228f22037763"
    )
    assert contract.PREREGISTRATION_REVIEW_BYTE_COUNT == 13_235
    assert contract.preregistration_review_binding() == {
        "path": contract.PREREGISTRATION_REVIEW_RELATIVE_PATH,
        "commit": "5d532e814c73c7c8238a59cf853e9cef4975c541",
        "file_sha256": (
            "606138757d9292ef3c8a75f16c1e8abb34da5fa11d84db60362d126b68cf2acf"
        ),
        "content_sha256": (
            "8f1316b203734fdf844cc04819e8b370510258da3d6680381667228f22037763"
        ),
        "byte_count": 13_235,
    }
    assert contract.prior_terminal_audit_binding() == {
        "path": contract.V10_TERMINAL_AUDIT_RELATIVE_PATH,
        "commit": "b590e50af272ae046618819eed4b88f1cd7a0cab",
        "file_sha256": (
            "e33030f59c1d36aecf61d98750213daa89d7aeb8ee0daf83ff92812ca31ce4e5"
        ),
        "content_sha256": (
            "9ab2aec125e2d8ced8f35da7dab6c2d2794035d33c3888c28f8584b6e7070eb4"
        ),
        "byte_count": 5_999,
    }


def test_current_source_bindings_appends_exact_preregistration_review() -> None:
    observed = contract.current_source_bindings(ROOT)
    assert observed[contract.PREREGISTRATION_REVIEW_RELATIVE_PATH] == (
        contract.PREREGISTRATION_REVIEW_FILE_SHA256
    )
    assert observed[contract.V10_TERMINAL_AUDIT_RELATIVE_PATH] == (
        contract.PRIOR_TERMINAL_AUDIT_FILE_SHA256
    )
    assert observed[contract.PREREGISTRATION_RELATIVE_PATH] == (
        contract.PREREGISTRATION_FILE_SHA256
    )

    real_reader = contract._BASE_V10_READ_REGULAR_SOURCE

    def tampered_reader(path: Path) -> bytes:
        raw = real_reader(path)
        if path == ROOT / contract.PREREGISTRATION_REVIEW_RELATIVE_PATH:
            return raw + b" "
        return raw

    with (
        mock.patch.object(
            contract,
            "_BASE_V10_READ_REGULAR_SOURCE",
            side_effect=tampered_reader,
        ),
        pytest.raises(
            PermissionError,
            match="preregistration independent review changed",
        ),
    ):
        contract.current_source_bindings(ROOT)


def test_three_ulp_cross_reduction_difference_is_accepted() -> None:
    metrics, update0 = _update0_with_action_nll(
        FLOAT32_THREE_ULP_REDUCED_MEAN
    )
    update0_before = deepcopy(update0)
    difference = abs(
        FLOAT32_EQUAL_LOGIT_REFERENCE
        - FLOAT32_THREE_ULP_REDUCED_MEAN
    )
    assert difference > 1e-7
    assert difference <= contract.UPDATE_ZERO_ACTION_NLL_ABS_TOLERANCE

    result = contract.evaluate_phase_a_continuation(
        100,
        metrics,
        update0,
        v10_fixtures._integrity(),
    )
    assert result["passed"] is True
    assert update0 == update0_before


def test_material_action_nll_mismatch_is_rejected() -> None:
    metrics, update0 = _update0_with_action_nll(
        FLOAT32_EQUAL_LOGIT_REFERENCE - 1e-3
    )
    with pytest.raises(
        ValueError,
        match="update-zero action symmetry and chance receipt changed",
    ):
        contract.evaluate_phase_a_continuation(
            100,
            metrics,
            update0,
            v10_fixtures._integrity(),
        )


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_action_nll_difference_immediately_outside_tolerance_is_rejected(
    direction: float,
) -> None:
    boundary = (
        FLOAT32_EQUAL_LOGIT_REFERENCE
        + direction * contract.UPDATE_ZERO_ACTION_NLL_ABS_TOLERANCE
    )
    observed = math.nextafter(
        boundary,
        -math.inf if direction < 0.0 else math.inf,
    )
    assert abs(observed - FLOAT32_EQUAL_LOGIT_REFERENCE) > (
        contract.UPDATE_ZERO_ACTION_NLL_ABS_TOLERANCE
    )
    metrics, update0 = _update0_with_action_nll(observed)
    with pytest.raises(
        ValueError,
        match="update-zero action symmetry and chance receipt changed",
    ):
        contract.evaluate_phase_a_continuation(
            100,
            metrics,
            update0,
            v10_fixtures._integrity(),
        )


def test_exact_action_ratio_and_margin_checks_are_unchanged() -> None:
    metrics, update0 = _update0_with_action_nll(
        FLOAT32_THREE_ULP_REDUCED_MEAN
    )
    changed_ratio = deepcopy(update0)
    changed_ratio["factorized_retrieval"][
        "executed_to_cyclic_ratio"
    ] = 1.0 - contract.FLOAT32_EPSILON
    with pytest.raises(
        ValueError,
        match="update-zero action symmetry and chance receipt changed",
    ):
        contract.evaluate_phase_a_continuation(
            100,
            metrics,
            changed_ratio,
            v10_fixtures._integrity(),
        )

    changed_margin = deepcopy(update0)
    first_family = contract.SCENE_FAMILIES[0]
    changed_margin["factorized_retrieval"]["per_family"][first_family][
        "cyclic_wrong_minus_executed_energy"
    ] = contract.FLOAT32_EPSILON
    changed_margin["factorized_retrieval"][
        "cyclic_positive_family_margin_count"
    ] = 1
    with pytest.raises(
        ValueError,
        match="update-zero action symmetry and chance receipt changed",
    ):
        contract.evaluate_phase_a_continuation(
            100,
            metrics,
            changed_margin,
            v10_fixtures._integrity(),
        )

    changed_bitwise = deepcopy(update0)
    changed_bitwise["all_action_predictions_bitwise_equal"] = False
    result = contract.evaluate_phase_a_continuation(
        100,
        metrics,
        changed_bitwise,
        v10_fixtures._integrity(),
    )
    assert result["passed"] is False
    assert result["conjuncts"][
        "update_zero_all_action_predictions_bitwise_equal"
    ] is False


def test_v10r_identity_is_fresh_one_shot_and_v10_root_is_prohibited() -> None:
    assert contract.SCHEMA_PREFIX.endswith("v10r_integrity_replacement")
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "/rgb_action_conditioned_next_target_retrieval_jepa_probe_"
        "v10r_integrity_replacement"
    )
    assert contract.OUTPUT_ROOT_RELATIVE_PATH != (
        contract.V10_OUTPUT_ROOT_RELATIVE_PATH
    )
    assert contract.PROHIBITED_RUNTIME_OUTPUT_ROOTS == (
        contract.V10_OUTPUT_ROOT_RELATIVE_PATH,
    )
    assert contract.EXECUTION_AUTHORITY["attempt_index"] == 1
    assert contract.EXECUTION_AUTHORITY["maximum_attempts"] == 1
    assert contract.EXECUTION_AUTHORITY["generated_mutation_scope"] == (
        contract.OUTPUT_ROOT_RELATIVE_PATH
    )
    assert contract.INTEGRITY_REPLACEMENT_DELTA[
        "retry_resume_second_seed_or_schedule_extension_authorized"
    ] is False
    for schema in (
        contract.SOURCE_MANIFEST_SCHEMA,
        contract.REVIEW_SCHEMA,
        contract.AUTHORIZATION_SCHEMA,
        contract.RESERVATION_SCHEMA,
        contract.PHASE_A_METRICS_SCHEMA,
        contract.PHASE_A_ARTIFACT_SCHEMA,
        contract.PHASE_B_METRICS_SCHEMA,
        contract.ACCESS_SCHEMA,
        contract.RESULT_SCHEMA,
        contract.COMPLETION_SCHEMA,
        contract.FAILURE_SCHEMA,
    ):
        assert schema.startswith(contract.SCHEMA_PREFIX)


def test_contract_runner_and_launcher_import_are_source_only() -> None:
    real_import = builtins.__import__

    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".", 1)[0] in {
            "torch", "numpy", "PIL", "cv2", "jax", "tensorflow",
        }:
            raise AssertionError(f"source-only import loaded {name}")
        return real_import(name, globals, locals, fromlist, level)

    with mock.patch("builtins.__import__", side_effect=guarded):
        runner = _load("_test_retrieval_jepa_v10r_runner", RUNNER_PATH)
        launcher = _load(
            "_test_retrieval_jepa_v10r_launcher",
            LAUNCHER_PATH,
        )

    assert runner.contract.OUTPUT_ROOT_RELATIVE_PATH == (
        contract.OUTPUT_ROOT_RELATIVE_PATH
    )
    assert launcher.contract.OUTPUT_ROOT_RELATIVE_PATH == (
        contract.OUTPUT_ROOT_RELATIVE_PATH
    )
    assert runner._V10.contract is runner.contract
    assert launcher._V10.contract is launcher.contract
    assert launcher._BASE.contract is launcher.contract
    assert runner.PREFLIGHT_ENVIRONMENT_KEY == (
        launcher.PREFLIGHT_ENVIRONMENT_KEY
    )
