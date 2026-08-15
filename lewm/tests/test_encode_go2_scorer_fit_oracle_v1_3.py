"""Focused synthetic tests for the oracle-v1.3 target-encoding consumer."""
from __future__ import annotations

import copy
import hashlib
import inspect
from pathlib import Path

import pytest

from lewm.oracle import go2_scorer_fit_oracle_v1_3_contract as CONTRACT
from scripts import encode_go2_scorer_fit_oracle_v1_3 as ENCODER


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _special_fit_states() -> dict[tuple[str, str], list[object]]:
    result: dict[tuple[str, str], list[object]] = {}
    seen: set[str] = set()
    for failure in CONTRACT.FAILED_BRANCH_IDENTITIES:
        if failure.split_role != "fit" or failure.state_id in seen:
            continue
        family = next(value for value in ENCODER.EXPECTED_FAMILIES
                      if failure.state_id.startswith(f"scorer_fit-{value}-"))
        remainder = failure.state_id.removeprefix(f"scorer_fit-{family}-")
        stratum = next(value for value in ENCODER.EXPECTED_STRATA
                       if remainder.startswith(f"{value}-"))
        result.setdefault((family, stratum), []).append(failure)
        seen.add(failure.state_id)
    return result


def synthetic_training_view() -> dict:
    failures_by_state: dict[str, dict[int, object]] = {}
    for failure in CONTRACT.FAILED_BRANCH_IDENTITIES:
        if failure.split_role == "fit":
            failures_by_state.setdefault(failure.state_id, {})[
                failure.candidate_index] = failure
    special = _special_fit_states()
    rows = []
    for family in ENCODER.EXPECTED_FAMILIES:
        for stratum in ENCODER.EXPECTED_STRATA:
            cell_special = special.get((family, stratum), [])
            for state_number in range(4):
                if state_number < len(cell_special):
                    witness = cell_special[state_number]
                    state_id = witness.state_id
                    state_digest = witness.state_identity_digest
                    scene_id = witness.scene_id
                else:
                    state_id = f"synthetic-fit-{family}-{stratum}-{state_number}"
                    state_digest = _digest(f"state:{state_id}")
                    scene_id = f"synthetic-scene:{state_id}"
                for candidate in range(12):
                    failure = failures_by_state.get(state_id, {}).get(candidate)
                    branch_digest = (failure.branch_identity_digest if failure
                                     else _digest(
                                         f"branch:{state_id}:{candidate}"))
                    safety = float(candidate in {0, 6})
                    completion = float(candidate == 11)
                    progress = float(candidate) / 10.0
                    rows.append({
                        "role": "fit",
                        "source_kind": ("V13_REPLAY_OVERLAY" if failure
                                        else "V2_VALID_ADOPTION"),
                        "state_id": state_id,
                        "state_identity_digest": state_digest,
                        "scene_id": scene_id,
                        "family": family,
                        "stratum": stratum,
                        "candidate_index": candidate,
                        "branch_identity_digest": branch_digest,
                        "training_view_row_digest": _digest(
                            f"view-row:{state_id}:{candidate}"),
                        "progress": progress,
                        "safety": safety,
                        "completion": completion,
                        "utility": progress - 2.0 * safety
                                   + 0.5 * completion,
                        "action_blocks": [[0.0] * 10 for _ in range(4)],
                        "goal_binding_input": [0.0, 1.0, 2.0],
                    })
            state_id = f"synthetic-calibration-{family}-{stratum}"
            state_digest = _digest(f"state:{state_id}")
            scene_id = f"synthetic-scene:{state_id}"
            for candidate in range(12):
                safety = float(candidate in {0, 6})
                completion = float(candidate == 11)
                progress = float(candidate) / 10.0
                rows.append({
                    "role": "calibration",
                    "source_kind": "V13_FRESH_CALIBRATION",
                    "state_id": state_id,
                    "state_identity_digest": state_digest,
                    "scene_id": scene_id,
                    "family": family,
                    "stratum": stratum,
                    "candidate_index": candidate,
                    "branch_identity_digest": _digest(
                        f"branch:{state_id}:{candidate}"),
                    "training_view_row_digest": _digest(
                        f"view-row:{state_id}:{candidate}"),
                    "progress": progress,
                    "safety": safety,
                    "completion": completion,
                    "utility": progress - 2.0 * safety + 0.5 * completion,
                    "action_blocks": [[0.0] * 10 for _ in range(4)],
                    "goal_binding_input": [0.0, 1.0, 2.0],
                })
    disposition = {
        "state_count": 24,
        "branch_count": 288,
        "status": "DEVELOPMENT_ONLY",
        "qualification_eligible": False,
        "discarded": False,
        "state_identity_digests": [
            row.state_identity_digest for row in CONTRACT.OLD_CALIBRATION_STATES
        ],
        "scene_ids": [row.scene_id for row in CONTRACT.OLD_CALIBRATION_STATES],
    }
    disposition["disposition_digest"] = ENCODER.canonical_digest(disposition)
    return {
        "schema": ENCODER.WORKFLOW.TRAINING_VIEW_SCHEMA,
        "status": ENCODER.WORKFLOW.STATUS,
        "complete": True,
        "missing_label_count": 0,
        "training_view_digest": "1" * 64,
        "oracle_v1_3_digest": ENCODER.ORACLE.oracle_digest(),
        "scorer_fit_oracle_v1_3_contract_digest": CONTRACT.contract_digest(),
        "authority_digest": "2" * 64,
        "v2_corpus_digest": CONTRACT.FROZEN_CORPUS_DIGEST,
        "equivalence_receipt_digest": "3" * 64,
        "replay_overlay_manifest_digest": "4" * 64,
        "fresh_calibration_state_manifest_digest": "5" * 64,
        "fresh_calibration_corpus_digest": "6" * 64,
        "fit_state_count": 96,
        "fit_branch_count": 1_152,
        "calibration_state_count": 24,
        "calibration_branch_count": 288,
        "row_count": 1_440,
        "source_kind_counts": dict(ENCODER.EXPECTED_SOURCE_KINDS),
        "historical_calibration_disposition": disposition,
        "rows": rows,
    }


def test_exact_training_view_composition_and_replay_overlay_identities_pass():
    view = ENCODER.validate_training_view_structure(synthetic_training_view())
    assert len(view["rows"]) == 1_440
    assert sum(row["role"] == "fit" for row in view["rows"]) == 1_152
    assert {
        row["branch_identity_digest"] for row in view["rows"]
        if row["source_kind"] == "V13_REPLAY_OVERLAY"
    } == {
        row.branch_identity_digest for row in CONTRACT.FAILED_BRANCH_IDENTITIES
        if row.split_role == "fit"
    }


@pytest.mark.parametrize("mutation", ["nonfinite", "old_calibration", "overlap"])
def test_training_view_rejects_missing_label_and_calibration_leakage(mutation):
    view = copy.deepcopy(synthetic_training_view())
    if mutation == "nonfinite":
        view["rows"][0]["progress"] = float("nan")
    elif mutation == "old_calibration":
        selected = view["rows"][-12:]
        old = CONTRACT.OLD_CALIBRATION_STATES[0]
        for row in selected:
            row["state_identity_digest"] = old.state_identity_digest
            row["scene_id"] = old.scene_id
    else:
        fit_scene = next(row["scene_id"] for row in view["rows"]
                         if row["role"] == "fit")
        calibration_id = next(row["state_id"] for row in view["rows"]
                              if row["role"] == "calibration")
        for row in view["rows"]:
            if row["state_id"] == calibration_id:
                row["scene_id"] = fit_scene
    with pytest.raises(ENCODER.V13EncodingError):
        ENCODER.validate_training_view_structure(view)


def test_frame_bytes_are_checked_before_heavy_encoder_import(tmp_path: Path):
    view = synthetic_training_view()
    frame_root = tmp_path / "fixture_frames"
    frame_root.mkdir()
    context = []
    horizon = []
    for kind, ordinals in (("context", range(3)), ("horizon", range(1, 5))):
        for ordinal in ordinals:
            path = frame_root / f"{kind}-{ordinal}.png"
            path.write_bytes(f"synthetic-{kind}-{ordinal}".encode("ascii"))
            record = {
                "path": path.name,
                "sha256": ENCODER.file_sha256(path),
                "byte_count": path.stat().st_size,
                "shape": [224, 224, 3],
                "dtype": "uint8",
                ("slot" if kind == "context" else "horizon"): ordinal,
            }
            (context if kind == "context" else horizon).append(record)
    for row in view["rows"]:
        row["frame_root"] = "fixture_frames"
        row["context_frames"] = copy.deepcopy(context)
        row["horizon_frames"] = copy.deepcopy(horizon)
    paths = ENCODER.validate_frame_inputs(view, root=tmp_path)
    assert len(paths) == 1_440
    source = inspect.getsource(ENCODER.encode_training_view)
    assert source.index("validate_frame_inputs") < source.index("import torch")
    assert source.index("import torch") < source.index("arm_factory()")


def test_encoder_has_no_predictor_or_final_benchmark_route():
    source = Path(ENCODER.__file__).read_text()
    assert "predictor_checkpoint" not in "\n".join(
        line for line in source.splitlines() if line.lstrip().startswith(
            ("from ", "import ")))
    assert "final_eval" not in source
    assert ENCODER.latent_index_path() == ENCODER.ROOT / CONTRACT.LATENT_INDEX_PATH
    assert ENCODER.encoding_receipt_path() == (
        ENCODER.ROOT / CONTRACT.ENCODING_RECEIPT_PATH
    )
