from __future__ import annotations

import ast
import copy
import json
import math
import os
from pathlib import Path
import subprocess

import pytest

from scripts import evaluate_non_greedy_local_subgoal_jepa_planning_v1 as E


_REAL_SOURCE_FREEZE_OBSERVER = E._observe_source_freeze


def _synthetic_source_freeze() -> dict[str, object]:
    return {
        "head_commit": "a" * 40,
        "worktree_clean": True,
        "sources_exactly_equal_head": True,
        "sources": {
            "independent_reducer": {
                "path": E.REDUCER_SOURCE_PATH,
                "bytes": 101,
                "sha256": "b" * 64,
            },
            "metrics_module": {
                "path": E.METRICS_SOURCE_PATH,
                "bytes": 202,
                "sha256": "c" * 64,
            },
        },
    }


@pytest.fixture(autouse=True)
def _freeze_exact_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        E,
        "_observe_source_freeze",
        lambda _module: _synthetic_source_freeze(),
    )


class SyntheticMetrics:
    __name__ = "synthetic_non_greedy_metrics"

    @staticmethod
    def score_row_authority() -> dict[str, object]:
        fields = [
            "stage_id",
            "state_id",
            "family",
            "split_role",
            "model_id",
            "source_id",
            "candidate_index",
            "score",
            "geodesic_progress_m",
            "euclidean_progress_m",
            "remaining_geodesic_m",
            "heading_error_to_next_shortest_segment_rad",
            "oracle_admissible",
            "immediate_contact",
            "committed_prefix_contact",
            "successor_viable",
            "stuck",
            "dead_end",
            "completed",
        ]
        return {
            "schema": E.AUTHORITY_SCHEMA,
            "stages": {
                "stage_a": {
                    "stage_id": "STAGE_A",
                    "file_name": E.STAGE_A_FILE,
                    "required_fields": fields,
                    "finite_numeric_fields": [
                        "score",
                        "geodesic_progress_m",
                        "euclidean_progress_m",
                        "remaining_geodesic_m",
                        "heading_error_to_next_shortest_segment_rad",
                    ],
                    "state_ids": [f"state-{index}" for index in range(16)],
                    "model_source_pairs": [
                        {"model_id": model_id, "source_id": source_id}
                        for model_id, source_id in E.EXPECTED_MODEL_SOURCE_PAIRS[
                            "stage_a"
                        ]
                    ],
                    "candidate_count": 12,
                },
                "conditional_stage_b": {
                    "stage_id": "CONDITIONAL_STAGE_B",
                    "file_name": E.STAGE_B_FILE,
                    "required_fields": fields,
                    "finite_numeric_fields": [
                        "score",
                        "geodesic_progress_m",
                        "euclidean_progress_m",
                        "remaining_geodesic_m",
                        "heading_error_to_next_shortest_segment_rad",
                    ],
                    "state_ids": [f"state-{index}" for index in range(16)],
                    "model_source_pairs": [
                        {"model_id": model_id, "source_id": source_id}
                        for model_id, source_id in E.EXPECTED_MODEL_SOURCE_PAIRS[
                            "conditional_stage_b"
                        ]
                    ],
                    "candidate_count": 12,
                },
            },
        }

    @staticmethod
    def recompute_metrics_and_gates(
        stage_a_rows: tuple[dict[str, object], ...],
        conditional_stage_b_rows: tuple[dict[str, object], ...] | None,
    ) -> dict[str, object]:
        a_mean = sum(float(row["score"]) for row in stage_a_rows) / len(stage_a_rows)
        b_mean = (
            None
            if conditional_stage_b_rows is None
            else sum(float(row["score"]) for row in conditional_stage_b_rows)
            / len(conditional_stage_b_rows)
        )
        return {
            "schema": "synthetic.metrics.v1",
            "stage_a": {"mean_score": a_mean, "rows": len(stage_a_rows)},
            "conditional_stage_b": (
                None
                if conditional_stage_b_rows is None
                else {"mean_score": b_mean, "rows": len(conditional_stage_b_rows)}
            ),
            "gates": {
                "stage_a_pass": a_mean < 20.0,
                "conditional_stage_b_pass": (
                    None if b_mean is None else b_mean < 30.0
                ),
            },
        }


def _rows(
    stage: str,
    model_source_pairs: tuple[tuple[str, str], ...],
    *,
    score_offset: float = 0.0,
) -> list[dict[str, object]]:
    return [
        {
            "stage_id": stage,
            "state_id": state_id,
            "family": "left-detour" if state_index % 2 == 0 else "right-detour",
            "split_role": "DEVELOPMENT_HELDOUT",
            "model_id": model_id,
            "source_id": source,
            "candidate_index": candidate_index,
            "score": score_offset + candidate_index + state_index / 10.0,
            "geodesic_progress_m": candidate_index / 10.0,
            "euclidean_progress_m": candidate_index / 20.0,
            "remaining_geodesic_m": 10.0 - candidate_index / 10.0,
            "heading_error_to_next_shortest_segment_rad": candidate_index / 100.0,
            "oracle_admissible": candidate_index % 2 == 0,
            "immediate_contact": candidate_index % 3 == 0,
            "committed_prefix_contact": candidate_index % 5 == 0,
            "successor_viable": candidate_index % 4 != 0,
            "stuck": candidate_index == 11,
            "dead_end": candidate_index in {5, 11},
            "completed": candidate_index in {2, 8},
        }
        for state_index, state_id in enumerate(
            f"state-{index}" for index in range(16)
        )
        for model_id, source in model_source_pairs
        for candidate_index in range(12)
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(E.canonical_document_bytes(row) for row in rows))


def _prepare(
    root: Path,
    *,
    include_stage_b: bool,
) -> tuple[list[dict[str, object]], list[dict[str, object]] | None]:
    root.mkdir()
    stage_a = _rows("STAGE_A", E.EXPECTED_MODEL_SOURCE_PAIRS["stage_a"])
    stage_b = (
        _rows(
            "CONDITIONAL_STAGE_B",
            E.EXPECTED_MODEL_SOURCE_PAIRS["conditional_stage_b"],
            score_offset=10.0,
        )
        if include_stage_b
        else None
    )
    _write_jsonl(root / E.STAGE_A_FILE, stage_a)
    if stage_b is not None:
        _write_jsonl(root / E.STAGE_B_FILE, stage_b)
    metrics = SyntheticMetrics.recompute_metrics_and_gates(
        tuple(stage_a), None if stage_b is None else tuple(stage_b)
    )
    (root / E.METRICS_FILE).write_bytes(E.canonical_document_bytes(metrics))
    return stage_a, stage_b


def test_absent_conditional_stage_recomputes_and_emits_canonical_receipt(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    _prepare(output, include_stage_b=False)
    receipt = E.verify_and_emit(output, metrics_module=SyntheticMetrics)
    raw = (output / E.REGENERATION_RECEIPT_FILE).read_bytes()
    assert raw == E.canonical_document_bytes(receipt)
    assert receipt["pass"] is True
    assert receipt["conditional_stage_b_present"] is False
    assert receipt["conditional_stage_b_validation"] is None
    assert receipt["stage_a_validation"]["rows"] == 1152
    assert receipt["stage_a_validation"]["candidates_per_state_model_source"] == 12
    assert receipt["source_freeze"] == _synthetic_source_freeze()
    assert set(receipt["scientific_execution_counters"].values()) == {0}
    core = dict(receipt)
    observed = core.pop("content_digest")
    assert observed == E.canonical_digest(core)
    assert E.emit_regeneration_receipt(output, receipt) == raw


def test_present_conditional_stage_is_exactly_reduced(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _prepare(output, include_stage_b=True)
    receipt = E.build_regeneration_receipt(output, metrics_module=SyntheticMetrics)
    assert receipt["conditional_stage_b_present"] is True
    assert receipt["conditional_stage_b_validation"]["rows"] == 384
    assert receipt["inputs"]["conditional_stage_b_scores"]["rows"] == 384


def test_existing_receipt_requires_exact_source_bound_rebuild(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _prepare(output, include_stage_b=True)
    expected = E.verify_and_emit(output, metrics_module=SyntheticMetrics)
    assert E.validate_existing_regeneration_receipt(
        output,
        metrics_module=SyntheticMetrics,
    ) == expected
    tampered = dict(expected)
    tampered["source_freeze"] = {
        **expected["source_freeze"],
        "head_commit": "d" * 40,
    }
    tampered_core = dict(tampered)
    tampered_core.pop("content_digest")
    tampered["content_digest"] = E.canonical_digest(tampered_core)
    (output / E.REGENERATION_RECEIPT_FILE).write_bytes(
        E.canonical_document_bytes(tampered)
    )
    with pytest.raises(E.RegenerationError, match="differs from exact rebuild"):
        E.validate_existing_regeneration_receipt(
            output,
            metrics_module=SyntheticMetrics,
        )


def test_candidate_facts_must_match_across_conditional_stage(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _, stage_b = _prepare(output, include_stage_b=True)
    assert stage_b is not None
    stage_b[0]["geodesic_progress_m"] = 99.0
    _write_jsonl(output / E.STAGE_B_FILE, stage_b)
    with pytest.raises(E.RegenerationError, match="candidate payload differs"):
        E.build_regeneration_receipt(output, metrics_module=SyntheticMetrics)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rows: rows.pop(), "cardinality drift"),
        (
            lambda rows: rows.__setitem__(1, copy.deepcopy(rows[0])),
            "duplicate row identity",
        ),
        (
            lambda rows: rows[0].__setitem__("candidate_index", True),
            "candidate_index drift",
        ),
        (
            lambda rows: rows[0].__setitem__("source_id", "unknown"),
            "identity coverage drift",
        ),
    ],
)
def test_row_identity_and_cardinality_tamper_rejected(
    tmp_path: Path, mutation, match: str
) -> None:
    output = tmp_path / "output"
    stage_a, _ = _prepare(output, include_stage_b=False)
    mutation(stage_a)
    _write_jsonl(output / E.STAGE_A_FILE, stage_a)
    with pytest.raises(E.RegenerationError, match=match):
        E.build_regeneration_receipt(output, metrics_module=SyntheticMetrics)


def test_exact_row_schema_and_finite_score_are_required(tmp_path: Path) -> None:
    output = tmp_path / "output"
    stage_a, _ = _prepare(output, include_stage_b=False)
    stage_a[0]["extra"] = "forbidden"
    _write_jsonl(output / E.STAGE_A_FILE, stage_a)
    with pytest.raises(E.RegenerationError, match="row schema drift"):
        E.build_regeneration_receipt(output, metrics_module=SyntheticMetrics)

    del stage_a[0]["extra"]
    raw = b"".join(E.canonical_document_bytes(row) for row in stage_a)
    raw = raw.replace(b'"score":0.0', b'"score":1e999', 1)
    (output / E.STAGE_A_FILE).write_bytes(raw)
    with pytest.raises(E.RegenerationError, match="non-finite"):
        E.build_regeneration_receipt(output, metrics_module=SyntheticMetrics)


def test_noncanonical_and_duplicate_key_jsonl_are_rejected(tmp_path: Path) -> None:
    output = tmp_path / "output"
    stage_a, _ = _prepare(output, include_stage_b=False)
    raw = (output / E.STAGE_A_FILE).read_bytes()
    (output / E.STAGE_A_FILE).write_bytes(b" " + raw)
    with pytest.raises(E.RegenerationError, match="not canonical JSON"):
        E.build_regeneration_receipt(output, metrics_module=SyntheticMetrics)

    first = E.canonical_json_bytes(stage_a[0])
    duplicate = first[:-1] + b',"score":0.0}'
    rest = b"".join(E.canonical_document_bytes(row) for row in stage_a[1:])
    (output / E.STAGE_A_FILE).write_bytes(duplicate + b"\n" + rest)
    with pytest.raises(E.RegenerationError, match="duplicate JSON object key"):
        E.build_regeneration_receipt(output, metrics_module=SyntheticMetrics)


def test_supplied_metrics_must_equal_exact_recomputation(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _prepare(output, include_stage_b=True)
    supplied = json.loads((output / E.METRICS_FILE).read_bytes())
    supplied["gates"]["stage_a_pass"] = False
    (output / E.METRICS_FILE).write_bytes(E.canonical_document_bytes(supplied))
    with pytest.raises(E.RegenerationError, match="differs from exact recomputation"):
        E.build_regeneration_receipt(output, metrics_module=SyntheticMetrics)


def test_cli_requires_verify_and_prints_exact_receipt(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "output"
    _prepare(output, include_stage_b=False)
    with pytest.raises(SystemExit):
        E.main(["--output-root", str(output)], metrics_module=SyntheticMetrics)
    assert E.main(
        ["--output-root", str(output), "--verify"],
        metrics_module=SyntheticMetrics,
    ) == 0
    stdout = capsys.readouterr().out.encode("ascii")
    assert stdout == (output / E.REGENERATION_RECEIPT_FILE).read_bytes()


def test_reducer_source_has_no_model_tensor_training_or_inference_import() -> None:
    source = Path(E.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"torch", "numpy", "scipy", "genesis"})
    assert "model_initializations" in source
    assert "training_steps" in source
    assert "inference_calls" in source
    assert math.isfinite(1.0)


def test_system_python_imports_frozen_metrics_without_model_execution() -> None:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            (
                "from scripts import "
                "evaluate_non_greedy_local_subgoal_jepa_planning_v1 as e; "
                "m=e._load_metrics_module(); "
                "assert callable(m.score_row_authority); "
                "assert callable(m.recompute_metrics_and_gates)"
            ),
        ],
        cwd=E.ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
